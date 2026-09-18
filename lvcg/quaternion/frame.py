"""Route C: separating what the observation frame changes from what it does not.

The latent VCG's absolute orientation carries cardiac information *and* depends on the
frame the electrodes happened to define. V5 and V6 attacked that by learning a rotation
and undoing it -- a free pose head, or the lead matrix turned -- which throws the
orientation away wholesale and, as both reports showed, buys about +0.5 pp and no
invariance. Route C is deliberately not that.

Here the frame is **computed from the record, not learned**, and the representation is
**split by how it transforms**:

    R_frame = [ x , y , z ]      x = the record's dominant cardiac axis
                                 z = the normal of the loop it sweeps
                                 y = z x x            (right-handed, orthonormalised)

    invariant   V_int = R_frame^T V        the trajectory seen in its own frame
    equivariant q_frame                    where that frame sits in the world

Under a global rotation q of the record, the two behave in exactly opposite ways and
both behaviours are provable, not hoped for:

    V_int  -> V_int                        unchanged, because R_frame -> R R_frame
    q_frame -> q (x) q_frame               predictable, the plan's "varies predictably"

So the absolute orientation is **kept, not deleted** -- it is simply routed into its own
embedding, where a classifier can use it when the frame is trustworthy and where a
rotation sweep can show exactly which part of the model it perturbs. An
``invariant``-only model is exactly frame-invariant by construction, which is a stronger
statement than a robustness curve that merely looks flat.

The frame is built from the whole record (aggregated over its valid beats) rather than
per beat, so it is stable; a record whose loop is degenerate -- no plane to speak of --
falls back to a deterministic axis instead of producing a noise frame.
"""

import torch
import torch.nn as nn

from .interloop import segment_axis
from .qdf import DynamicEncoder
from .utils import _safe_norm, quaternion_angle, rotation_matrix_to_quaternion

PARTS = ("invariant", "equivariant")
DEGENERATE = 1e-4  # a loop whose area vector is shorter than this has no reliable plane


def intrinsic_frame(beats, beat_mask):
    """The record's own right-handed frame: rotation matrices [B, 3, 3] and quaternions [B, 4].

    x is the magnitude-weighted dominant direction of the whole beat, z the normal of the
    loop it sweeps (twice its vector area), y their cross product. Both are equivariant
    under a global rotation, so the frame turns with the record -- which is what makes the
    coordinates taken in it invariant.
    """
    weight = beat_mask.to(beats.dtype)
    everything = torch.ones(beats.shape[-1], dtype=torch.bool, device=beats.device)
    axis = segment_axis(beats, everything)                      # [B, N, 3]
    trajectory = beats.transpose(-1, -2)
    area = torch.cross(trajectory[..., :-1, :], trajectory[..., 1:, :], dim=-1).sum(-2)

    # Aggregate over the record's valid beats, keeping each beat's sign convention.
    axis = (axis * weight.unsqueeze(-1)).sum(1)
    normal = (area * weight.unsqueeze(-1)).sum(1)

    fallback = torch.tensor([0.0, 0.0, 1.0], device=beats.device, dtype=beats.dtype)
    degenerate = _safe_norm(normal, keepdim=True) < DEGENERATE
    normal = torch.where(degenerate, fallback.expand_as(normal), normal)
    z = normal / _safe_norm(normal, keepdim=True)

    x = axis - (axis * z).sum(-1, keepdim=True) * z
    alternative = torch.cross(z, fallback.expand_as(z), dim=-1)
    x = torch.where(_safe_norm(x, keepdim=True) < DEGENERATE, alternative, x)
    x = x / _safe_norm(x, keepdim=True)
    y = torch.cross(z, x, dim=-1)

    rotation = torch.stack((x, y, z), dim=-1)  # columns are the axes
    return rotation, rotation_matrix_to_quaternion(rotation)


def to_frame(beats, rotation):
    """Express a trajectory in the given frame: R^T V, invariant under a global rotation."""
    return torch.einsum("bji,bnjp->bnip", rotation, beats)


class FrameSplitEncoder(nn.Module):
    """Two embeddings: the frame-invariant shape and the frame itself.

    ``invariant`` encodes magnitude and direction of the trajectory *in its own frame*,
    which is V3's strongest channel set applied to coordinates that a global rotation
    cannot touch. ``equivariant`` encodes the frame's quaternion, its angle and its axes,
    so the absolute orientation stays available instead of being discarded.
    """

    def __init__(self, parts=PARTS, embedding_dim=128, hidden=64, kernel=7, dropout=0.1):
        super().__init__()
        unknown = [p for p in parts if p not in PARTS]
        if unknown or not parts:
            raise ValueError(f"Unknown or empty parts {parts!r}; expected from {PARTS}")
        self.parts = tuple(parts)
        self.embedding_dim = embedding_dim
        if "invariant" in self.parts:
            self.invariant = DynamicEncoder(4, embedding_dim, hidden, kernel, dropout)
        if "equivariant" in self.parts:
            self.equivariant = nn.Sequential(
                nn.Linear(4 + 1 + 9, hidden), nn.GELU(), nn.Linear(hidden, embedding_dim),
                nn.LayerNorm(embedding_dim))
        self.width = embedding_dim * len(self.parts)

    def invariant_embedding(self, beats, beat_mask, rotation):
        """Magnitude and unit direction in the record's own frame, pooled over its beats."""
        local = to_frame(beats, rotation)                        # [B, N, 3, P]
        magnitude = _safe_norm(local, dim=2, keepdim=True)
        features = torch.cat((magnitude, local / magnitude), dim=2)  # 1 + 3 channels
        B, N = beats.shape[:2]
        pooled = self.invariant(features.reshape(B * N, 4, -1)).reshape(B, N, -1)
        weight = beat_mask.to(pooled.dtype).unsqueeze(-1)
        return (pooled * weight).sum(1) / weight.sum(1).clamp_min(1.0)

    def equivariant_embedding(self, quaternion, rotation):
        summary = torch.cat((quaternion, quaternion_angle(quaternion).unsqueeze(-1),
                             rotation.flatten(-2)), dim=-1)
        return self.equivariant(summary)

    def forward(self, beats, beat_mask):
        rotation, quaternion = intrinsic_frame(beats, beat_mask)
        parts = []
        for name in self.parts:
            if name == "invariant":
                parts.append(self.invariant_embedding(beats, beat_mask, rotation))
            else:
                parts.append(self.equivariant_embedding(quaternion, rotation))
        return torch.cat(parts, dim=-1)


class FrameProbe(nn.Module):
    """Route C's model: the frame-split representation beside the frozen LVCG embedding.

    ``parts`` selects which halves the head reads, so the plan's three questions are
    three runs: both (keep orientation, structured), invariant only (frame-stable
    diagnosis) and equivariant only (is the frame itself diagnostic?).
    ``scale = 0`` reproduces V0.
    """

    STRUCTS = ("mean", "anchor")

    def __init__(self, state_generator, norm_struct, norm_dynamic, num_classes=5,
                 token_dim=256, rhythm_dim=128, struct="mean", scale=1.0, **encoder_kwargs):
        super().__init__()
        if struct not in self.STRUCTS:
            raise ValueError(f"Unknown struct {struct!r}; expected one of {self.STRUCTS}")
        self.state_generator = state_generator
        self.norm_struct = norm_struct
        self.norm_dynamic = norm_dynamic
        for module in (state_generator, norm_struct, norm_dynamic):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        for module in self.state_generator.modules():
            if isinstance(module, nn.RNNBase):
                module.dropout = 0.0
            elif isinstance(module, nn.Dropout):
                module.p = 0.0

        self.struct = struct
        self.frame = FrameSplitEncoder(**encoder_kwargs)
        self.head = nn.Linear(
            token_dim + state_generator.hidden_dim + rhythm_dim + self.frame.width, num_classes)
        self.register_buffer("scale", torch.tensor(float(scale)))

    def train(self, mode=True):
        super().train(mode)
        self.norm_struct.eval()
        self.norm_dynamic.eval()
        self.state_generator.train(mode)
        return self

    def embeddings(self, beats, tokens, beat_mask, steps, rhythm):
        from .qdt import rollout_hidden

        anchor = tokens[:, 1] if tokens.shape[1] > 1 else tokens[:, 0]
        if self.struct == "mean":
            weight = beat_mask.to(tokens.dtype).unsqueeze(-1)
            structure = (tokens * weight).sum(1) / weight.sum(1).clamp_min(1.0)
        else:
            structure = anchor
        base = torch.cat((self.norm_struct(structure),
                          self.norm_dynamic(rollout_hidden(self.state_generator, anchor, steps)),
                          rhythm), dim=-1)
        split = (base.new_zeros(base.shape[0], self.frame.width)
                 if float(self.scale) == 0.0 else self.frame(beats, beat_mask) * self.scale)
        return base, split

    def forward(self, beats, tokens, beat_mask, steps, rhythm):
        base, split = self.embeddings(beats, tokens, beat_mask, steps, rhythm)
        return self.head(torch.cat((base, split), dim=-1))

    def parameter_counts(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        return {
            "trainable_total": sum(p.numel() for p in trainable),
            "frame_branch": sum(p.numel() for p in self.frame.parameters()),
            "head": sum(p.numel() for p in self.head.parameters()),
        }
