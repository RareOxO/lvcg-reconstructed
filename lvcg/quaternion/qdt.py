"""V2, QDT-LVCG: quaternion dynamics injected into the pretrained beat tokens.

    beat n -> BeatEncoder (frozen)                  -> z_n^VCG  [B, N, 256]
    beat n -> Q_n -> DynamicEncoder (per beat)      -> z_n^Q    [B, N, E]
    z'_n = TokenFusion(z_n^VCG, z_n^Q)                          [B, N, 256]
    z'_n -> the pretrained StateGRU and embeddings  -> the V0 head

Plan section 4 forbids replacing the pretrained token with a random projection, so both
fusions start as the identity on z^VCG: ``concat`` initialises W = [I, 0] and ``gated``
initialises the value projection at zero. At step 0 the model is therefore V0 exactly,
and everything it learns beyond V0 comes from the quaternion branch.

One consequence of that zero block is worth stating plainly: at step 0 **the quaternion
encoder receives no gradient at all**, because the path from z^Q to the output is
multiplied by zero. What does receive gradient is the fusion itself (its gradient is
z^Q, which is not zero), so after the first update the branch starts to learn. The plan
also allows a near-zero gate, and ``init_std`` provides it: a small random block lets
gradient reach the branch immediately, at the price of V2(t=0) only approximating V0.

**A structural limit of the released architecture.** In ``forward_inference`` only beat
token 1 reaches the output: ``emb_struct`` is that token and the GRU rollout is unrolled
from it alone, so the other beats' tokens never enter the embedding. Fusing every token
is therefore faithful to the plan but can only act through token 1 -- a fact this module
does not hide, and ``tests/test_quaternion_v2.py`` demonstrates it. Reading the other
tokens would require changing the pretrained temporal path, which V2 may not do.

The rollout here takes each record's own beat count. The release takes the batch's
largest, which makes a record's embedding depend on the batch it landed in;
``scripts/train_qdt.py`` reports both so the difference is visible rather than assumed.
"""

import torch
import torch.nn as nn

from .features import BeatQuaternionFeatures
from .qdf import DynamicEncoder


class TokenFusion(nn.Module):
    """Fuse a beat's VCG token z^VCG [.., D] with its quaternion token z^Q [.., E] -> [.., D].

        concat  z = W [z_VCG ; z_Q] + b,          W = [I, 0] at initialisation
        gated   z = z_VCG + sigmoid(W_g [..]) * W_q z_Q,   W_q = 0 at initialisation

    Both are the identity on z_VCG when training starts, so V2(t=0) = V0. The block that
    multiplies z_Q is zero, so the branch itself sees no gradient on the first step while
    the fusion does; ``init_std > 0`` replaces that block with a small random one, which
    trades exact V0 equivalence for gradient reaching the branch immediately.
    """

    def __init__(self, token_dim, quaternion_dim, mode="gated", init_std=0.0):
        super().__init__()
        if mode not in ("concat", "gated"):
            raise ValueError("fusion must be concat or gated")
        self.mode = mode
        self.init_std = float(init_std)
        joint = token_dim + quaternion_dim
        if mode == "concat":
            self.projection = nn.Linear(joint, token_dim)
            with torch.no_grad():
                self.projection.weight.zero_()
                if self.init_std > 0:
                    self.projection.weight[:, token_dim:].normal_(0.0, self.init_std)
                self.projection.weight[:, :token_dim].copy_(torch.eye(token_dim))
                self.projection.bias.zero_()
        else:
            self.gate = nn.Linear(joint, token_dim)
            self.value = nn.Linear(quaternion_dim, token_dim)
            with torch.no_grad():
                self.value.weight.normal_(0.0, self.init_std) if self.init_std > 0 else self.value.weight.zero_()
                self.value.bias.zero_()

    def forward(self, z_vcg, z_q):
        joint = torch.cat((z_vcg, z_q), dim=-1)
        if self.mode == "concat":
            return self.projection(joint)
        return z_vcg + torch.sigmoid(self.gate(joint)) * self.value(z_q)


def rollout_hidden(state_generator, state_0, steps):
    """The pretrained GRU's last hidden state, with each record taking its own step count.

    ``state_generator`` is the frozen ``StateGRU``; ``steps`` [B] is the release's
    ``num_gen_steps`` per record, i.e. (beats - 1). The loop reproduces ``StateGRU``
    exactly -- the same GRU call, the same output projection -- and a record whose step
    count is 1 or less keeps the zero hidden state the release gives it.
    """
    batch = state_0.shape[0]
    hidden_dim = state_generator.hidden_dim
    out = state_0.new_zeros(batch, hidden_dim)
    steps = steps.to(state_0.device)
    h, state_t = None, state_0
    for t in range(1, int(steps.max().item()) if steps.numel() else 1):
        gru_out, h = state_generator.gru(state_t.unsqueeze(1), h)
        state_t = state_generator.out_proj(gru_out.squeeze(1))
        finished = steps == (t + 1)
        if finished.any():
            out = torch.where(finished.unsqueeze(-1), h[-1], out)
    return out


class QDTProbe(nn.Module):
    """The trainable part of V2: beat quaternion tokens, fusion, and the linear head.

    The frozen pieces it needs from the pretrained model -- the StateGRU and the two
    LayerNorms -- are passed in and kept with ``requires_grad=False``; the beat encoder
    and the rhythm embedding are applied earlier, when features are cached.
    """

    def __init__(
        self,
        state_generator,
        norm_struct,
        norm_dynamic,
        num_classes=5,
        features=("q", "theta", "omega", "magnitude"),
        fs=100,
        token_dim=256,
        rhythm_dim=128,
        quaternion_dim=128,
        hidden=64,
        kernel=7,
        dropout=0.1,
        fusion="gated",
        fusion_init_std=0.0,
        min_magnitude_fraction=0.02,
        sign_continuity=True,
        masked_pooling=True,
        scale=1.0,
    ):
        super().__init__()
        self.state_generator = state_generator
        self.norm_struct = norm_struct
        self.norm_dynamic = norm_dynamic
        for module in (self.state_generator, self.norm_struct, self.norm_dynamic):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        # cuDNN refuses to back-propagate through an RNN that is in evaluation mode, and
        # the gradient of the fusion runs through this GRU. The GRU is therefore kept in
        # training mode (see ``train``) with its dropout switched off, which makes the two
        # modes numerically identical: an nn.GRU has no other mode-dependent behaviour.
        for module in self.state_generator.modules():
            if isinstance(module, nn.RNNBase):
                module.dropout = 0.0
            elif isinstance(module, nn.Dropout):
                module.p = 0.0

        self.dynamics = BeatQuaternionFeatures(features, fs, min_magnitude_fraction, sign_continuity)
        self.encoder = DynamicEncoder(self.dynamics.channels, quaternion_dim, hidden, kernel, dropout)
        self.fusion = TokenFusion(token_dim, quaternion_dim, fusion, fusion_init_std)
        self.head = nn.Linear(token_dim + state_generator.hidden_dim + rhythm_dim, num_classes)
        self.masked_pooling = bool(masked_pooling)
        self.quaternion_dim = quaternion_dim
        self.register_buffer("scale", torch.tensor(float(scale)))

    def train(self, mode=True):
        super().train(mode)
        # The pretrained pieces are frozen, and with dropout off their mode changes
        # nothing numerically -- but cuDNN needs the GRU in training mode to give a
        # backward pass, so it follows the probe's mode while the norms stay in eval.
        self.state_generator.train(mode)
        self.norm_struct.eval()
        self.norm_dynamic.eval()
        return self

    def beat_tokens(self, beats, rr_intervals, beat_mask, reference):
        """Per-beat quaternion tokens z^Q [B, N, E]."""
        B, N = beats.shape[:2]
        features, mask = self.dynamics(beats, rr_intervals, beat_mask, reference)
        flat = features.reshape(B * N, features.shape[2], features.shape[3])
        pooled = self.encoder(flat, mask.reshape(B * N, -1) if self.masked_pooling else None)
        return pooled.reshape(B, N, -1)

    def forward(self, beats, rr_intervals, beat_mask, reference, states_real, steps, emb_rhythm):
        """Cached frozen tensors in, logits [B, 5] out.

        beats [B, N, 3, P], rr_intervals [B, N], beat_mask [B, N], reference [B],
        states_real [B, N, 256] (frozen BeatEncoder output), steps [B] (beats - 1),
        emb_rhythm [B, 128] (frozen, already normalised).
        """
        if float(self.scale) == 0.0:
            # V0 exactly: the fusion is bypassed, not merely initialised at the identity,
            # so a trained bias inside it cannot drift away from the pretrained tokens.
            fused = states_real
        else:
            z_q = self.beat_tokens(beats, rr_intervals, beat_mask, reference) * self.scale
            fused = self.fusion(states_real, z_q)
        anchor = fused[:, 1] if fused.shape[1] > 1 else fused[:, 0]
        emb_struct = self.norm_struct(anchor)
        emb_dynamic = self.norm_dynamic(rollout_hidden(self.state_generator, anchor, steps))
        return self.head(torch.cat((emb_struct, emb_dynamic, emb_rhythm), dim=-1))

    def parameter_counts(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        return {
            "trainable_total": sum(p.numel() for p in trainable),
            "quaternion_branch": sum(p.numel() for p in self.encoder.parameters()),
            "fusion": sum(p.numel() for p in self.fusion.parameters()),
            "head": sum(p.numel() for p in self.head.parameters()),
        }
