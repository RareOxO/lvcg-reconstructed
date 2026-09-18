"""Quaternion additions to the pretrained LVCG backbone (plan V1-V8).

Nothing here changes the pretrained model: each version adds a branch whose
contribution can be switched off, and with it off the model must reproduce V0.
"""

from .attention import LabelAttentionProbe
from .canon import CanonProbe, PoseNet, random_rotations, rotate_beats
from .interloop import InterLoopEncoder, QRSTProbe, spatial_qrst_angle
from .loop import LoopProbe, RotationalLoopEncoder, multi_scale_rotations, perturb_rotation_order
from .frame import FrameProbe, FrameSplitEncoder, intrinsic_frame, to_frame
from .sparse import LEAD_SETS, frame_agreement, geometry_fidelity, recover_vcg
from .geometry import GlobalRotation, QAProbe, rotated_directions
from .features import (
    FEATURE_CHANNELS,
    BeatQuaternionFeatures,
    QuaternionDynamicFeatures,
    transition_features,
)
from .qdf import DynamicEncoder, QDFProbe
from .mrq import VARIANTS, build_probe
from .phase import PHASES, PhaseProbe, phase_masks
from .qdt import QDTProbe, TokenFusion

__all__ = [
    "LabelAttentionProbe",
    "CanonProbe",
    "LEAD_SETS",
    "frame_agreement",
    "geometry_fidelity",
    "recover_vcg",
    "FrameProbe",
    "FrameSplitEncoder",
    "intrinsic_frame",
    "to_frame",
    "GlobalRotation",
    "InterLoopEncoder",
    "QRSTProbe",
    "spatial_qrst_angle",
    "LoopProbe",
    "RotationalLoopEncoder",
    "multi_scale_rotations",
    "perturb_rotation_order",
    "QAProbe",
    "rotated_directions",
    "PoseNet",
    "random_rotations",
    "rotate_beats",
    "FEATURE_CHANNELS",
    "BeatQuaternionFeatures",
    "QuaternionDynamicFeatures",
    "transition_features",
    "DynamicEncoder",
    "VARIANTS",
    "build_probe",
    "PHASES",
    "PhaseProbe",
    "phase_masks",
    "QDFProbe",
    "QDTProbe",
    "TokenFusion",
]
