"""Quaternion additions to the pretrained LVCG backbone (plan V1-V8).

Nothing here changes the pretrained model: each version adds a branch whose
contribution can be switched off, and with it off the model must reproduce V0.
"""

from .features import FEATURE_CHANNELS, QuaternionDynamicFeatures, transition_features
from .qdf import DynamicEncoder, QDFProbe

__all__ = [
    "FEATURE_CHANNELS",
    "QuaternionDynamicFeatures",
    "transition_features",
    "DynamicEncoder",
    "QDFProbe",
]
