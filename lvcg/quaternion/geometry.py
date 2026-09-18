"""V6, QA-LVCG: a constrained SO(3) correction of the lead geometry itself.

The lead direction matrix A0 is a textbook constant: twelve unit vectors that say where
each electrode looks. V6 asks whether a *rotation* of that geometry fits the data better,
while keeping its physical structure -- no free 12x3 matrix, only

    A_Q = A_0 R(q_A),    q_A(t=0) = [1, 0, 0, 0]

so the model starts at the published geometry and can only turn it. ``global`` learns one
correction for the whole dataset (a systematic convention or electrode-placement bias);
``conditioned`` predicts one per record (individual anatomy), bounded and reported.

**An identity worth stating.** Recovering the VCG from the rotated geometry is the same
as rotating the VCG recovered from the original geometry:

    pinv(A_0 R) e = R^T pinv(A_0) e

because the regularised pseudo-inverse commutes with an orthogonal factor. The tests
check it against the released ``VCGPseudoInverse`` and find agreement to 5e-13 in float64.
Two consequences, both stated rather than hidden:

* V6 can rotate the cached beat patches instead of re-running the backbone, exactly as
  V5 does -- and it inherits V5's proof that rotation commutes with beat segmentation.
* ``conditioned`` mode is the same family of functions as V5's pose canonicalisation.
  What is genuinely new in V6 is the ``global`` mode and the geometric reading of the
  correction: an angle on the lead matrix, reported in degrees, with a geodesic penalty
  available to keep it near the published geometry.
"""

import torch
import torch.nn as nn

from .canon import CanonProbe, PoseNet, axis_angle_to_quaternion
from .utils import quaternion_angle, quaternion_to_rotation_matrix

TINY = 1e-12


def rotated_directions(directions, quaternion):
    """A_Q = A_0 R(q_A). ``directions`` [L, 3], ``quaternion`` [4] or [B, 4] -> [.., L, 3]."""
    rotation = quaternion_to_rotation_matrix(quaternion)
    if rotation.dim() == 2:
        return directions @ rotation
    return torch.einsum("lj,bjk->blk", directions, rotation)


class GlobalRotation(nn.Module):
    """One rotation shared by every record, stored as an axis-angle vector at zero.

    Zero initialisation means ``A_Q = A_0`` exactly when training starts, which is the
    plan's requirement; ``max_degrees`` bounds how far the geometry may turn.
    """

    def __init__(self, max_degrees=None):
        super().__init__()
        self.vector = nn.Parameter(torch.zeros(3))
        self.max_degrees = max_degrees

    def quaternion(self):
        vector = self.vector
        if self.max_degrees:
            limit = float(self.max_degrees) * torch.pi / 180.0
            norm = torch.sqrt(vector.square().sum() + TINY)
            vector = vector * (limit * torch.tanh(norm / limit) / norm)
        return axis_angle_to_quaternion(vector)

    def forward(self, beats, beat_mask=None):
        return self.quaternion().expand(beats.shape[0], 4)


class QAProbe(CanonProbe):
    """V6: the lead geometry turned by q_A, in front of the frozen pretrained path.

    ``mode``:
        ``fixed``        the published A_0 -- this stage's V0 reference
        ``global``       one learned rotation for the whole dataset
        ``conditioned``  one per record, from the same pose network V5 uses
    """

    MODES = ("fixed", "global", "conditioned")

    def __init__(self, *args, mode="global", max_degrees=None, pose_dim=64, pose_hidden=32, **kwargs):
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {self.MODES}")
        pose_module = None
        if mode == "global":
            pose_module = GlobalRotation(max_degrees)
        elif mode == "conditioned":
            pose_module = PoseNet(pose_dim, pose_hidden, max_degrees=max_degrees)
        super().__init__(*args, canonicalize=mode != "fixed", pose_module=pose_module, **kwargs)
        self.mode = mode
        self.max_degrees = max_degrees

    def geometry(self, directions, beats=None, beat_mask=None):
        """A_Q for reporting: [L, 3] in ``global`` mode, [B, L, 3] in ``conditioned``."""
        if self.mode == "fixed":
            return directions
        if self.mode == "global":
            return rotated_directions(directions, self.pose.quaternion())
        return rotated_directions(directions, self.pose(beats, beat_mask))

    def geodesic_penalty(self, beats, beat_mask=None):
        """Mean squared correction angle in radians -- the plan's geodesic regularisation."""
        if self.mode == "fixed":
            return torch.zeros((), device=beats.device)
        return quaternion_angle(self.pose(beats, beat_mask)).square().mean()

    def parameter_counts(self):
        counts = super().parameter_counts()
        counts["geometry"] = counts.pop("pose")
        return counts
