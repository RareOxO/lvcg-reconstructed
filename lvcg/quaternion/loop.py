"""Route A: the rotational loop geometry of a latent VCG beat.

A beat is not three channels that happen to vary; it is one closed 3D trajectory. The
research question is whether the *ordered composition* of its rotations carries
information the pretrained LVCG does not already encode -- not whether four quaternion
components fed to a convolution do, which V1 already answered in the negative.

The difference is composition. For unit directions u_t of one beat,

    q_t          = shortest arc u_t -> u_{t+1}                one step
    q_{j<-i}     = q_{j-1} (x) ... (x) q_i                    an ordered window
    W_s[t]       = inv(C[t-s]) (x) C[t],   C = prefix product of q

so a window of length s is read as a single rotation: how far, about which axis, the
cardiac vector has turned over that stretch. ``multi_scale_rotations`` returns those
windows at several scales (short, medium, whole beat), and ``RotationalLoopEncoder``
turns them into one embedding per beat, pooled over the beat's valid steps and then over
the record's valid beats.

The prefix product uses a doubling scan, so the whole composition costs about seven
quaternion multiplications rather than one per step, and it is exact -- quaternion
multiplication is associative, and the scan never reorders the factors.

**Two properties worth stating.** Steps whose direction is unreliable receive the
identity quaternion before composition, so a masked stretch leaves a window unchanged
rather than corrupting it. And a window's rotation is invariant to where the beat starts
in space but *not* to the order of its steps -- which is exactly what route A's signature
test perturbs.
"""

import torch
import torch.nn as nn

from .features import _IDENTITY
from .qdf import DynamicEncoder
from .utils import (
    _safe_norm,
    enforce_sign_continuity,
    quaternion_angle,
    quaternion_conjugate,
    quaternion_multiply,
    valid_rotation_mask,
    vectors_to_quaternion,
)

DEFAULT_SCALES = (8, 32, 0)  # 0 means "the whole beat so far"


def beat_directions(beats, beat_mask, reference, min_fraction=0.02):
    """beats [B, N, 3, P] -> unit directions [B, N, P, 3] and a step mask [B, N, P-1].

    ``reference`` [B] is the record's 99th-percentile VCG magnitude, so a quiet beat is
    judged against the record rather than against itself.
    """
    trajectory = beats.transpose(-1, -2)  # [B, N, P, 3]
    if reference.dim() == 1:
        reference = reference[:, None, None]
    mask = valid_rotation_mask(trajectory, 1, min_fraction, reference=reference)
    mask = mask & beat_mask.to(torch.bool).unsqueeze(-1)
    return trajectory / _safe_norm(trajectory, keepdim=True), mask


def step_rotations(directions, mask, sign_continuity=True):
    """Shortest-arc rotation of every step: [..., T, 3] -> [..., T-1, 4].

    Unreliable steps become the identity, so they compose without effect.
    """
    quaternions = torch.where(
        mask.unsqueeze(-1),
        vectors_to_quaternion(directions[..., :-1, :], directions[..., 1:, :]),
        directions.new_tensor(_IDENTITY),
    )
    return enforce_sign_continuity(quaternions) if sign_continuity else quaternions


def prefix_composition(quaternions):
    """Ordered prefix products C[t] = q_t (x) ... (x) q_0, by doubling scan.

    The later rotation stands on the left, which is the plan's convention and the one
    rotation composition needs: applying C[t] to the first direction gives direction
    t + 1. Associativity makes the scan exact and the factors are never reordered.
    """
    result = quaternions
    length = result.shape[-2]
    shift = 1
    while shift < length:
        earlier = result[..., : length - shift, :]
        later = result[..., shift:, :]
        result = torch.cat((result[..., :shift, :], quaternion_multiply(later, earlier)), dim=-2)
        shift *= 2
    return result


def window_rotations(prefix, scale):
    """The rotation composed over the last ``scale`` steps ending at each t.

    ``scale = 0`` means "everything so far", i.e. the prefix itself. For t < scale the
    window is also the prefix, because nothing earlier exists.
    """
    if scale <= 0:
        return prefix
    head = prefix[..., :scale, :]
    earlier = quaternion_conjugate(prefix[..., : prefix.shape[-2] - scale, :])
    return torch.cat((head, quaternion_multiply(prefix[..., scale:, :], earlier)), dim=-2)


def multi_scale_rotations(directions, mask, scales=DEFAULT_SCALES, sign_continuity=True):
    """Per-step features of the ordered composition at several scales.

    Returns [..., T-1, 5 * len(scales)]: for every scale the composed quaternion (4) and
    its rotation angle (1).
    """
    steps = step_rotations(directions, mask, sign_continuity)
    prefix = prefix_composition(steps)
    parts = []
    for scale in scales:
        composed = steps if scale == 1 else window_rotations(prefix, scale)
        parts.append(torch.cat((composed, quaternion_angle(composed).unsqueeze(-1)), dim=-1))
    return torch.cat(parts, dim=-1)


def perturb_rotation_order(beats, beat_mask, reference, mode="shuffle", block=8,
                           generator=None, min_fraction=0.02):
    """Route A's signature test: reorder a beat's rotations, keeping every step's size.

    The per-step turning angles are preserved **exactly**, not approximately. A rotation
    only moves a vector by its own angle when its axis is perpendicular to that vector,
    so replaying an original quaternion on a different direction would change the step
    size. Each permuted step is therefore rebuilt from its original angle and its axis
    projected onto the plane perpendicular to the current direction; the multiset of step
    angles is then identical by construction, and only their order differs. The original
    magnitude profile is put back afterwards.

    ``mode``: ``shuffle`` (all steps), ``block`` (blocks of ``block`` steps, shuffled
    among themselves) or ``reverse`` (composition order reversed).
    """
    directions, mask = beat_directions(beats, beat_mask, reference, min_fraction)
    steps = step_rotations(directions, mask)
    B, N, T, _ = steps.shape
    device = steps.device

    if mode == "reverse":
        order = torch.arange(T - 1, -1, -1, device=device).expand(B, N, T)
    elif mode == "shuffle":
        order = torch.argsort(torch.rand(B, N, T, generator=generator, device=device), dim=-1)
    elif mode == "block":
        # Whole blocks are shuffled among themselves; a trailing partial block stays where
        # it is, so the indices remain a permutation and no step is dropped or repeated.
        blocks = T // block
        full = blocks * block
        block_order = torch.argsort(torch.rand(B, N, max(blocks, 1), generator=generator, device=device), dim=-1)
        within = torch.arange(block, device=device)
        shuffled = (block_order.unsqueeze(-1) * block + within).reshape(B, N, -1)[..., :full]
        tail = torch.arange(full, T, device=device).expand(B, N, T - full)
        order = torch.cat((shuffled, tail), dim=-1)
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected shuffle, block or reverse")

    reordered = steps.gather(2, order.unsqueeze(-1).expand(B, N, T, 4))
    angles = quaternion_angle(reordered)                      # [B, N, T], preserved exactly
    axes = reordered[..., 1:] / _safe_norm(reordered[..., 1:], keepdim=True)

    current = directions[..., 0, :]
    rebuilt = [current]
    fallback = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=beats.dtype)
    for t in range(T):
        axis = axes[..., t, :]
        # Only the component perpendicular to the current direction turns it, so project;
        # an axis parallel to it would not move the vector at all.
        perpendicular = axis - (axis * current).sum(-1, keepdim=True) * current
        norm = _safe_norm(perpendicular, keepdim=True)
        alternative = torch.cross(current, fallback.expand_as(current), dim=-1)
        perpendicular = torch.where(norm > 1e-3, perpendicular / norm,
                                    alternative / _safe_norm(alternative, keepdim=True))
        half = 0.5 * angles[..., t:t + 1]
        quaternion = torch.cat((torch.cos(half), torch.sin(half) * perpendicular), dim=-1)
        current = _rotate(current, quaternion)
        current = current / _safe_norm(current, keepdim=True)
        rebuilt.append(current)

    new_directions = torch.stack(rebuilt, dim=-2)
    magnitude = _safe_norm(beats.transpose(-1, -2), keepdim=True)
    return (new_directions * magnitude).transpose(-1, -2).contiguous()


def _rotate(vectors, quaternions):
    """Apply q to v: q (x) [0, v] (x) q*, kept here because the shapes are beat-wise."""
    padded = torch.cat((torch.zeros_like(vectors[..., :1]), vectors), dim=-1)
    turned = quaternion_multiply(quaternion_multiply(quaternions, padded), quaternion_conjugate(quaternions))
    return turned[..., 1:]


class RotationalLoopEncoder(nn.Module):
    """Multi-scale ordered rotations of every beat -> one embedding per record.

    ``local_only`` reduces the model to V1's question -- the single-step quaternion with
    no composition -- which is the control that tells composition apart from "quaternion
    features in general".
    """

    def __init__(self, scales=DEFAULT_SCALES, embedding_dim=128, hidden=64, kernel=7,
                 dropout=0.1, min_magnitude_fraction=0.02, sign_continuity=True,
                 local_only=False, include_magnitude=False):
        super().__init__()
        self.scales = (1,) if local_only else tuple(scales)
        self.local_only = bool(local_only)
        self.include_magnitude = bool(include_magnitude)
        self.min_fraction = float(min_magnitude_fraction)
        self.sign_continuity = bool(sign_continuity)
        channels = 5 * len(self.scales) + 1 + int(self.include_magnitude)
        self.channels = channels
        self.encoder = DynamicEncoder(channels, embedding_dim, hidden, kernel, dropout)
        self.embedding_dim = embedding_dim

    def features(self, beats, beat_mask, reference):
        """[B, N, C, P-1] and the step mask [B, N, P-1]."""
        directions, mask = beat_directions(beats, beat_mask, reference, self.min_fraction)
        parts = [multi_scale_rotations(directions, mask, self.scales, self.sign_continuity)]
        if self.include_magnitude:
            magnitude = _safe_norm(beats.transpose(-1, -2), keepdim=True)[..., :-1, :]
            parts.append(magnitude)
        parts.append(mask.unsqueeze(-1).to(beats.dtype))
        return torch.cat(parts, dim=-1).transpose(-1, -2), mask

    def forward(self, beats, beat_mask, reference):
        features, mask = self.features(beats, beat_mask, reference)
        B, N = beats.shape[:2]
        pooled = self.encoder(features.reshape(B * N, self.channels, -1), mask.reshape(B * N, -1))
        pooled = pooled.reshape(B, N, -1)
        # Mean over the record's valid beats, which is the plan's pooling for structure.
        weight = beat_mask.to(pooled.dtype).unsqueeze(-1)
        return (pooled * weight).sum(1) / weight.sum(1).clamp_min(1.0)


class LoopProbe(nn.Module):
    """Route A's model: the rotational loop branch beside the frozen LVCG embedding.

        e_base  = [ norm_struct(struct) ; norm_dynamic(GRU rollout) ; rhythm ]   frozen
        e_loop  = RotationalLoopEncoder(beats)                                    trained
        logits  = Linear([ e_base ; scale * e_loop ])

    ``struct`` follows the research plan's locked definition -- the mean of the record's
    valid beat tokens -- with ``struct="anchor"`` keeping the released behaviour (beat
    token 1 alone), because every earlier stage's V0 used that and the two give different
    baselines. The GRU rollout starts from the anchor token either way, since only the
    structural representation is at issue.

    ``scale = 0`` reproduces the corresponding V0 exactly.
    """

    STRUCTS = ("mean", "anchor")

    def __init__(self, state_generator, norm_struct, norm_dynamic, num_classes=5,
                 token_dim=256, rhythm_dim=128, struct="mean", scale=1.0,
                 beat_encoder=None, **encoder_kwargs):
        super().__init__()
        if struct not in self.STRUCTS:
            raise ValueError(f"Unknown struct {struct!r}; expected one of {self.STRUCTS}")
        # Training reads cached tokens; the signature test perturbs the trajectory and
        # must recompute them, or the perturbation would never reach the frozen path.
        self.beat_encoder = beat_encoder
        if beat_encoder is not None:
            beat_encoder.eval()
            for parameter in beat_encoder.parameters():
                parameter.requires_grad_(False)
        self.state_generator = state_generator
        self.norm_struct = norm_struct
        self.norm_dynamic = norm_dynamic
        for module in (state_generator, norm_struct, norm_dynamic):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        for module in self.state_generator.modules():
            if isinstance(module, nn.RNNBase):
                module.dropout = 0.0  # cuDNN needs training mode for the backward pass
            elif isinstance(module, nn.Dropout):
                module.p = 0.0

        self.struct = struct
        self.loop = RotationalLoopEncoder(**encoder_kwargs)
        self.head = nn.Linear(
            token_dim + state_generator.hidden_dim + rhythm_dim + self.loop.embedding_dim, num_classes)
        self.register_buffer("scale", torch.tensor(float(scale)))

    def train(self, mode=True):
        super().train(mode)
        self.norm_struct.eval()
        self.norm_dynamic.eval()
        if self.beat_encoder is not None:
            self.beat_encoder.eval()
        self.state_generator.train(mode)
        return self

    def base_embedding(self, tokens, beat_mask, steps, rhythm):
        from .qdt import rollout_hidden

        anchor = tokens[:, 1] if tokens.shape[1] > 1 else tokens[:, 0]
        if self.struct == "mean":
            weight = beat_mask.to(tokens.dtype).unsqueeze(-1)
            structure = (tokens * weight).sum(1) / weight.sum(1).clamp_min(1.0)
        else:
            structure = anchor
        dynamic = rollout_hidden(self.state_generator, anchor, steps)
        return torch.cat((self.norm_struct(structure), self.norm_dynamic(dynamic), rhythm), dim=-1)

    def embeddings(self, beats, tokens, beat_mask, reference, steps, rhythm, recompute=False):
        """(e_base [B, 640], e_loop [B, E]). ``recompute`` re-encodes the beats given."""
        if recompute:
            if self.beat_encoder is None:
                raise ValueError("recompute needs the frozen beat encoder")
            tokens = self.beat_encoder(beats)
        base = self.base_embedding(tokens, beat_mask, steps, rhythm)
        loop = (base.new_zeros(base.shape[0], self.loop.embedding_dim)
                if float(self.scale) == 0.0 else self.loop(beats, beat_mask, reference) * self.scale)
        return base, loop

    def forward(self, beats, tokens, beat_mask, reference, steps, rhythm, recompute=False):
        base, loop = self.embeddings(beats, tokens, beat_mask, reference, steps, rhythm, recompute)
        return self.head(torch.cat((base, loop), dim=-1))

    def parameter_counts(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        return {
            "trainable_total": sum(p.numel() for p in trainable),
            "loop_branch": sum(p.numel() for p in self.loop.parameters()),
            "head": sum(p.numel() for p in self.head.parameters()),
        }
