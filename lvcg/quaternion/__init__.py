"""Quaternion additions to the pretrained LVCG backbone (plan V1-V8).

Nothing here changes the pretrained model: each version adds a branch whose
contribution can be switched off, and with it off the model must reproduce V0.
"""

from .canon import CanonProbe, PoseNet, random_rotations, rotate_beats
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
    "CanonProbe",
    "GlobalRotation",
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
