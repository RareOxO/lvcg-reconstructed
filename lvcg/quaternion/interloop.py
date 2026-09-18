"""Route B: the 3D rotational relationship between depolarisation and repolarisation.

A beat carries two physiologically distinct 3D processes: the QRS loop (ventricular
depolarisation) and the T loop (repolarisation). The pretrained LVCG never asks how one
is oriented with respect to the other. Route B encodes that relation with quaternions,
from the coarsest form to the finest:

    axis        q(QRS axis -> T axis) and its angle -- the spatial QRS-T angle
    plane       the same for the two loop normals: how the planes of the loops differ
    trajectory  a per-step relation between the two resampled loops

The angle of the axis relation is the textbook **spatial QRS-T angle**, so route B has a
mechanistic target that exists independently of the labels: a representation that encodes
the inter-loop geometry should make that angle easier to decode linearly than V0 does.
``spatial_qrst_angle`` computes it for exactly that test.

**Delineation, stated plainly (plan B1).** This repository has no QRS onset/offset or
T-wave delineation -- the segmentation gives R peaks and nothing else. The windows here
are therefore *R-peak relative fractions of the beat*, which is the plan's fallback and
is explicitly a Tier 1 exploratory setting, not physiologically precise segmentation.
Two consequences to keep in mind when reading any result:

* A beat patch starts at its own R peak, so the window called "QRS" covers R to the J
  point -- the upstroke before R belongs to the previous patch and is not included.
* The fractions are of the R-R interval, so they follow heart rate but not the actual
  QT dynamics.

Any conclusion that needs true onsets and offsets has to wait for a real delineator.
"""

import torch
import torch.nn as nn

from .loop import beat_directions
from .qdf import DynamicEncoder
from .utils import _safe_norm, quaternion_angle, vectors_to_quaternion

DEFAULT_WINDOWS = {"qrs": (0.0, 0.12), "t": (0.15, 0.55)}
LEVELS = ("axis", "plane", "trajectory")


def segment_masks(length, windows=None, device=None):
    """Boolean masks over a beat patch for each named window: {name: [P]}."""
    windows = {**DEFAULT_WINDOWS, **(windows or {})}
    position = torch.arange(length, device=device, dtype=torch.float32) / max(length - 1, 1)
    return {name: (position >= low) & (position <= high) for name, (low, high) in windows.items()}


def segment_axis(beats, mask):
    """The dominant 3D direction of a windowed trajectory: [B, N, 3].

    The magnitude-weighted principal axis of the segment -- the same quantity clinical
    vectorcardiography calls the QRS or T axis. Its sign is fixed by the segment's mean
    vector, because an eigenvector is defined only up to a sign.
    """
    trajectory = beats.transpose(-1, -2)  # [B, N, P, 3]
    weight = mask.to(trajectory.dtype).view(1, 1, -1, 1)
    weighted = trajectory * weight
    covariance = torch.einsum("bnpi,bnpj->bnij", weighted, trajectory * weight)
    _, vectors = torch.linalg.eigh(covariance.double())
    axis = vectors[..., -1].to(trajectory.dtype)  # largest eigenvalue last
    mean_vector = weighted.sum(-2)
    sign = torch.sign((axis * mean_vector).sum(-1, keepdim=True))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return axis * sign


def segment_normal(beats, mask):
    """The normal of the plane the windowed loop sweeps: [B, N, 3].

    Twice the vector area of the loop, sum_t P_t x P_{t+1}, which is zero only for a
    trajectory that never turns out of a line.
    """
    trajectory = beats.transpose(-1, -2)
    current, following = trajectory[..., :-1, :], trajectory[..., 1:, :]
    pairs = (mask[:-1] & mask[1:]).to(trajectory.dtype).view(1, 1, -1, 1)
    area = torch.cross(current, following, dim=-1) * pairs
    return area.sum(-2)


def relation_quaternion(source, target, eps=1e-8):
    """Shortest-arc rotation taking one direction to the other, with its angle."""
    unit_source = source / _safe_norm(source, keepdim=True)
    unit_target = target / _safe_norm(target, keepdim=True)
    quaternion = vectors_to_quaternion(unit_source, unit_target, eps=eps)
    return quaternion, quaternion_angle(quaternion)


def spatial_qrst_angle(beats, beat_mask, windows=None, degrees=True):
    """The spatial QRS-T angle per record: [B], averaged over the record's valid beats.

    Route B's mechanistic target. It is computed from the geometry alone -- no labels,
    no training -- so how well a representation decodes it is a property of the
    representation.
    """
    masks = segment_masks(beats.shape[-1], windows, beats.device)
    _, angle = relation_quaternion(segment_axis(beats, masks["qrs"]), segment_axis(beats, masks["t"]))
    if degrees:
        angle = angle * 180.0 / torch.pi
    weight = beat_mask.to(angle.dtype)
    return (angle * weight).sum(-1) / weight.sum(-1).clamp_min(1.0)


class InterLoopEncoder(nn.Module):
    """QRS and T loops -> one embedding per record, at the requested level of detail.

    ``axis`` and ``plane`` produce a handful of numbers per beat and are encoded by a
    small MLP; ``trajectory`` adds a per-step relation between the two loops, resampled
    to a common length, and is encoded by the same convolutional encoder the other
    stages use. Each level contains the previous one, so the comparison is nested.
    """

    def __init__(self, level="axis", embedding_dim=128, hidden=64, kernel=7, dropout=0.1,
                 windows=None, steps=32, min_magnitude_fraction=0.02):
        super().__init__()
        if level not in LEVELS:
            raise ValueError(f"Unknown level {level!r}; expected one of {LEVELS}")
        self.level = level
        self.windows = {**DEFAULT_WINDOWS, **(windows or {})}
        self.steps = int(steps)
        self.min_fraction = float(min_magnitude_fraction)
        self.embedding_dim = embedding_dim

        # axis: q(4) + angle(1) + the two axes(6); plane adds q(4) + angle(1) + normals(6).
        self.summary_channels = 11 if level == "axis" else 22
        self.summary = nn.Sequential(
            nn.Linear(self.summary_channels, hidden), nn.GELU(), nn.Linear(hidden, embedding_dim))
        self.trajectory_encoder = (
            DynamicEncoder(6, embedding_dim, hidden, kernel, dropout) if level == "trajectory" else None)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def _resampled_directions(self, beats, mask):
        """The windowed loop resampled to ``steps`` unit directions: [B, N, steps, 3]."""
        B, N, _, P = beats.shape
        index = mask.nonzero().flatten()
        if index.numel() < 2:
            return beats.new_zeros(B, N, self.steps, 3)
        segment = beats[..., index]  # [B, N, 3, L]
        resampled = torch.nn.functional.interpolate(
            segment.reshape(B * N, 3, -1), size=self.steps, mode="linear", align_corners=True)
        directions = resampled.reshape(B, N, 3, self.steps).transpose(-1, -2)
        return directions / _safe_norm(directions, keepdim=True)

    def features(self, beats, beat_mask):
        """Per-beat summary features [B, N, C] and, for ``trajectory``, the step relation."""
        masks = segment_masks(beats.shape[-1], self.windows, beats.device)
        qrs_axis, t_axis = segment_axis(beats, masks["qrs"]), segment_axis(beats, masks["t"])
        quaternion, angle = relation_quaternion(qrs_axis, t_axis)
        parts = [quaternion, angle.unsqueeze(-1), qrs_axis, t_axis]

        if self.level != "axis":
            qrs_normal, t_normal = segment_normal(beats, masks["qrs"]), segment_normal(beats, masks["t"])
            plane_quaternion, plane_angle = relation_quaternion(qrs_normal, t_normal)
            unit = lambda v: v / _safe_norm(v, keepdim=True)  # noqa: E731
            parts += [plane_quaternion, plane_angle.unsqueeze(-1), unit(qrs_normal), unit(t_normal)]

        summary = torch.cat(parts, dim=-1)
        steps = None
        if self.level == "trajectory":
            qrs = self._resampled_directions(beats, masks["qrs"])
            t_wave = self._resampled_directions(beats, masks["t"])
            step_quaternion, step_angle = relation_quaternion(qrs, t_wave)
            steps = torch.cat((step_quaternion, step_angle.unsqueeze(-1),
                               (qrs * t_wave).sum(-1, keepdim=True)), dim=-1)
        return summary, steps

    def forward(self, beats, beat_mask):
        summary, steps = self.features(beats, beat_mask)
        embedding = self.summary(summary)
        if steps is not None:
            B, N = beats.shape[:2]
            flat = steps.reshape(B * N, self.steps, 6).transpose(1, 2)
            embedding = embedding + self.trajectory_encoder(flat).reshape(B, N, -1)
        weight = beat_mask.to(embedding.dtype).unsqueeze(-1)
        pooled = (embedding * weight).sum(1) / weight.sum(1).clamp_min(1.0)
        return self.norm(self.dropout(pooled))


class QRSTProbe(nn.Module):
    """Route B's model: the inter-loop relation beside the frozen LVCG embedding.

    Shares route A's structural conventions -- ``struct="mean"`` is the research plan's
    pooling, ``"anchor"`` the released behaviour -- and ``scale = 0`` reproduces V0.
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
        self.interloop = InterLoopEncoder(**encoder_kwargs)
        self.head = nn.Linear(
            token_dim + state_generator.hidden_dim + rhythm_dim + self.interloop.embedding_dim,
            num_classes)
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
        relation = (base.new_zeros(base.shape[0], self.interloop.embedding_dim)
                    if float(self.scale) == 0.0 else self.interloop(beats, beat_mask) * self.scale)
        return base, relation

    def forward(self, beats, tokens, beat_mask, steps, rhythm):
        base, relation = self.embeddings(beats, tokens, beat_mask, steps, rhythm)
        return self.head(torch.cat((base, relation), dim=-1))

    def parameter_counts(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        return {
            "trainable_total": sum(p.numel() for p in trainable),
            "interloop_branch": sum(p.numel() for p in self.interloop.parameters()),
            "head": sum(p.numel() for p in self.head.parameters()),
        }
