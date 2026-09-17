"""V3, MRQ-LVCG: the cardiac vector split into magnitude and rotation.

    P_t = r_t u_t,    r_t = ||P_t||,    q_t = Rot(u_t -> u_{t+1})

Each factor gets its own encoder beside the frozen LVCG embedding, and the head sees
whichever components are switched on:

    logits = Linear([ e_base? ; e_magnitude? ; e_rotation? ; e_direction? ])

That makes the ablation matrix of plan section 6.2 a matter of listing components --
magnitude only, rotation only, magnitude+rotation, VCG+rotation, VCG+magnitude, all --
with the encoder, the pooling and the head identical in every case. With ``vcg`` alone
the model is V0's linear probe, which is this stage's V0-equivalence check.

The attachment point is V1's: parallel branches read by the head. V2 showed that routing
the same information through a beat token instead costs most of the effect, because the
released architecture lets only beat token 1 reach the output.

``direction`` is an addition to the plan, not one of its components. V1 and V2 both found
the parameter-matched real-valued control ahead of the quaternion features, and the
explanation on offer was that q_t keeps only how the vector turns and discards where it
points. ``direction`` is u_t itself, so a rotation branch that gains from adding it
confirms that explanation, and one that does not, refutes it.
"""

import torch
import torch.nn as nn

from .features import QuaternionDynamicFeatures
from .qdf import DynamicEncoder

# The plan's factorisation, plus the two diagnostics the earlier stages asked for.
COMPONENTS = {
    "magnitude": ("magnitude", "linear_velocity"),
    "rotation": ("q", "theta", "omega"),
    "direction": ("direction",),
    "control": ("position", "next_position", "delta"),
}


class MRQProbe(nn.Module):
    """Frozen LVCG embedding plus one encoder per enabled component.

    Args:
        components: names from ``COMPONENTS``, in the order the head reads them.
            ``vcg`` stands for the frozen embedding itself and carries no parameters.
        base_dim: width of the frozen embedding (640).
        fs: sampling rate of the cached VCG, so dt = 1 / fs.
    """

    def __init__(
        self,
        components=("vcg", "magnitude", "rotation"),
        base_dim=640,
        num_classes=5,
        fs=100,
        embedding_dim=128,
        hidden=64,
        kernel=7,
        dropout=0.1,
        min_magnitude_fraction=0.02,
        sign_continuity=True,
        masked_pooling=True,
    ):
        super().__init__()
        unknown = [c for c in components if c != "vcg" and c not in COMPONENTS]
        if unknown or not components:
            raise ValueError(f"Unknown or empty components {components!r}")
        self.components = tuple(components)
        self.use_base = "vcg" in self.components

        self.dynamics = nn.ModuleDict()
        self.encoders = nn.ModuleDict()
        width = base_dim if self.use_base else 0
        for name in self.components:
            if name == "vcg":
                continue
            dynamics = QuaternionDynamicFeatures(
                COMPONENTS[name], 1.0 / float(fs), min_magnitude_fraction, sign_continuity
            )
            self.dynamics[name] = dynamics
            self.encoders[name] = DynamicEncoder(dynamics.channels, embedding_dim, hidden, kernel, dropout)
            width += embedding_dim
        self.head = nn.Linear(width, num_classes)
        self.masked_pooling = bool(masked_pooling)
        self.embedding_dim = embedding_dim

    def embeddings(self, base_embedding, vcg):
        """The parts the head concatenates, in order."""
        parts = [base_embedding] if self.use_base else []
        for name in self.components:
            if name == "vcg":
                continue
            features, mask = self.dynamics[name](vcg)
            parts.append(self.encoders[name](features, mask if self.masked_pooling else None))
        return parts

    def forward(self, base_embedding, vcg):
        return self.head(torch.cat(self.embeddings(base_embedding, vcg), dim=-1))

    def parameter_counts(self):
        counts = {"trainable_total": sum(p.numel() for p in self.parameters() if p.requires_grad)}
        for name, encoder in self.encoders.items():
            counts[f"branch_{name}"] = sum(p.numel() for p in encoder.parameters())
        counts["head"] = sum(p.numel() for p in self.head.parameters())
        return counts
