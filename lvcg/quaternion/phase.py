"""V4, Phase-Q LVCG: the same dynamics read separately over QRS, T and the whole beat.

V1 and V3 measured *which* geometric quantity carries diagnostic information. V4 asks
*when* within the cardiac cycle it is carried: depolarisation (QRS) or repolarisation
(T). The feature set is a free choice, so the question can be asked of the quaternion
channels the plan specifies and of the ``mo`` set V3 found strongest -- if phase
resolution only helps one of them, that is itself a result.

    P (VCG) -> features(variant) -> DynamicEncoder conv stack -> h [B, E, T']
    e_phase = concat( pool(h, mask_QRS), pool(h, mask_T), pool(h, mask_whole) )
    logits  = Linear([ e_base ; scale * e_phase ])

One encoder is shared and its feature map pooled once per phase, so a phase costs only
its share of the head (128 x 5 weights). Capacity therefore stays within a fraction of a
percent of V1's, and what changes between variants is the pooling window, not the model.

With ``phases=("whole",)`` the model is V3's variant of the same name, tensor for
tensor, which is the equivalence check this stage inherits.

**What a phase does and does not separate.** The windows separate where the encoder
pools, not what its convolutions see: the stack spans about 43 samples, and at a normal
heart rate the T window opens only a few samples after QRS closes. Measured on random
records, a change confined to QRS moves the QRS embedding only about two to three times
as much as it moves the T one. The phases are therefore differently weighted views of
the same trajectory rather than independent measurements, and any "the information is in
QRS" conclusion has to be read with that leakage in mind.

**Windows.** The repository's segmentation gives R-peak positions and nothing finer, so
the windows are R-peak relative, as the plan allows: QRS is a fixed span in milliseconds
around R, and the T window is a fraction of the R-R interval that follows R, which keeps
it where repolarisation actually falls at any heart rate. Both are configurable, and
every window is intersected with the reliability mask, so a phase never pools a
transition whose direction was too small to trust.
"""

import torch
import torch.nn as nn

from .features import QuaternionDynamicFeatures
from .mrq import VARIANTS
from .qdf import DynamicEncoder

PHASES = ("qrs", "t", "whole")
# QRS: milliseconds either side of R. T: fraction of the following R-R interval.
DEFAULT_WINDOWS = {"qrs_ms": (-40.0, 60.0), "t_fraction": (0.15, 0.55)}


def phase_masks(peaks, peak_mask, steps, fs, windows=None, dtype=torch.bool):
    """Per-transition masks for each phase: {name: [B, steps]}.

    ``peaks`` [B, K] are R-peak sample positions with ``peak_mask`` [B, K] marking the
    real ones; ``steps`` is the number of transitions (T - 1). A sample belongs to QRS
    when it lies in the millisecond window of any R peak, and to T when it lies in the
    fraction window of the interval that starts at a peak. The two windows may overlap
    at extreme heart rates; nothing here forces them apart, and the overlap is reported
    rather than hidden.
    """
    windows = {**DEFAULT_WINDOWS, **(windows or {})}
    device = peaks.device
    batch, _ = peaks.shape
    time = torch.arange(steps, device=device).view(1, 1, steps)
    peak_positions = peaks.unsqueeze(-1)
    valid = peak_mask.unsqueeze(-1)

    low_ms, high_ms = windows["qrs_ms"]
    low = int(round(low_ms * fs / 1000.0))
    high = int(round(high_ms * fs / 1000.0))
    offset = time - peak_positions
    qrs = valid & (offset >= low) & (offset <= high)

    # The interval that follows each peak; the last peak borrows the record's median.
    following = torch.where(peak_mask[:, 1:], peaks[:, 1:] - peaks[:, :-1], torch.zeros_like(peaks[:, 1:]))
    counts = peak_mask[:, 1:].sum(dim=1, keepdim=True).clamp_min(1)
    typical = (following.sum(dim=1, keepdim=True) / counts).clamp_min(1.0)
    rr = torch.cat((following.to(typical.dtype), typical), dim=1).clamp_min(1.0)
    rr = torch.where(rr > 1.0, rr, typical.expand_as(rr)).unsqueeze(-1)

    start_fraction, end_fraction = windows["t_fraction"]
    after = (time - peak_positions).to(rr.dtype)
    t_wave = valid & (after >= start_fraction * rr) & (after <= end_fraction * rr)

    return {
        "qrs": qrs.any(dim=1).to(dtype),
        "t": t_wave.any(dim=1).to(dtype),
        "whole": torch.ones(batch, steps, device=device, dtype=dtype),
    }


class PhaseProbe(nn.Module):
    """V4: one feature encoder, pooled once per cardiac phase, beside the frozen embedding.

    Args:
        variant: which channels to read, from ``lvcg.quaternion.mrq.VARIANTS``
            (``mq`` for the plan's quaternion set, ``mo`` for V3's strongest set).
        phases: any of ``qrs``, ``t``, ``whole``, in the order the head reads them.
        scale: 0 reproduces V0 exactly.
    """

    def __init__(
        self,
        variant="mq",
        phases=("qrs", "t", "whole"),
        base_dim=640,
        num_classes=5,
        fs=100,
        embedding_dim=128,
        hidden=64,
        kernel=7,
        dropout=0.1,
        min_magnitude_fraction=0.02,
        sign_continuity=True,
        windows=None,
        scale=1.0,
    ):
        super().__init__()
        unknown = [p for p in phases if p not in PHASES]
        if unknown or not phases:
            raise ValueError(f"Unknown or empty phases {phases!r}; expected from {PHASES}")
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; expected one of {list(VARIANTS)}")
        self.variant = variant
        self.phases = tuple(phases)
        self.fs = int(fs)
        self.windows = {**DEFAULT_WINDOWS, **(windows or {})}

        self.dynamics = QuaternionDynamicFeatures(
            VARIANTS[variant], 1.0 / float(fs), min_magnitude_fraction, sign_continuity
        )
        self.encoder = DynamicEncoder(self.dynamics.channels, embedding_dim, hidden, kernel, dropout)
        self.head = nn.Linear(base_dim + embedding_dim * len(self.phases), num_classes)
        self.embedding_dim = embedding_dim
        self.register_buffer("scale", torch.tensor(float(scale)))

    def phase_embeddings(self, vcg, peaks, peak_mask):
        """One pooled embedding per phase, each over the reliable transitions inside it."""
        features, reliable = self.dynamics(vcg)
        masks = phase_masks(peaks, peak_mask, reliable.shape[-1], self.fs, self.windows)
        h = self.encoder.encode(features)
        return [self.encoder.pool(h, masks[name] & reliable) for name in self.phases]

    def forward(self, base_embedding, vcg, peaks, peak_mask):
        if float(self.scale) == 0.0:
            zeros = base_embedding.new_zeros(base_embedding.shape[0], self.embedding_dim * len(self.phases))
            return self.head(torch.cat((base_embedding, zeros), dim=-1))
        e_phase = torch.cat(self.phase_embeddings(vcg, peaks, peak_mask), dim=-1) * self.scale
        return self.head(torch.cat((base_embedding, e_phase), dim=-1))

    def phase_coverage(self, peaks, peak_mask, steps):
        """Share of transitions each phase covers, and the QRS/T overlap -- a sanity report."""
        masks = phase_masks(peaks, peak_mask, steps, self.fs, self.windows)
        return {
            **{name: float(masks[name].float().mean()) for name in PHASES},
            "qrs_and_t": float((masks["qrs"] & masks["t"]).float().mean()),
        }

    def parameter_counts(self):
        branch = sum(p.numel() for p in self.encoder.parameters())
        branch += self.embedding_dim * len(self.phases) * self.head.out_features
        base = self.head.in_features - self.embedding_dim * len(self.phases)
        return {
            "trainable_total": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "phase_branch": branch,
            "v0_head": base * self.head.out_features + self.head.out_features,
        }
