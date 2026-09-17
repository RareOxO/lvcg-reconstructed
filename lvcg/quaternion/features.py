"""Per-transition features of a cardiac-vector trajectory (plan section 3.2, 3.3).

The latent VCG is a path P_t in R^3. Between neighbouring samples the plan defines the
shortest-arc rotation q_t = Rot(u_t -> u_{t+1}) and derives theta and omega from it;
this module turns a VCG tensor into the channels a branch consumes, together with the
validity mask that says where a direction was reliable.

The same builder serves the parameter-matched real-valued control of plan section 6.1:
instead of q / theta / omega it emits the raw pair and its difference, so the control
sees the same trajectory through real-valued features only. A feature set is one or the
other, never a mix.

Ported from the q_wyt project (V1-V4 there), which is where these choices were tested.
"""

import torch
import torch.nn as nn

from .utils import (
    _safe_norm,
    enforce_sign_continuity,
    quaternion_angle,
    valid_rotation_mask,
    vectors_to_quaternion,
)

FEATURE_CHANNELS = {
    "q": 4,
    "theta": 1,
    "omega": 1,
    "magnitude": 1,
    "linear_velocity": 1,
    "position": 3,
    "next_position": 3,
    "delta": 3,
}
QUATERNION_FEATURES = ("q", "theta", "omega", "magnitude", "linear_velocity")
CONTROL_FEATURES = ("position", "next_position", "delta")

_IDENTITY = (1.0, 0.0, 0.0, 0.0)


def check_features(features):
    unknown = [f for f in features if f not in FEATURE_CHANNELS]
    if unknown or not features:
        raise ValueError(f"Unknown or empty feature set {features!r}")
    if any(f in CONTROL_FEATURES for f in features) and any(
        f in QUATERNION_FEATURES for f in features
    ):
        raise ValueError("A feature set is either quaternion or the real control, not both")
    return tuple(features)


def transition_features(p, mask, features, dt, sign_continuity=True, with_mask=True):
    """Per-transition features of a trajectory p [..., T, 3] -> [..., T - 1, C].

    ``mask`` [..., T - 1] marks transitions with a reliable direction; ``dt`` is a float
    or a tensor broadcastable to [..., 1, 1].

    A transition whose endpoints are too small to carry a direction gets the identity
    rotation -- no rotation, theta = omega = 0 -- instead of the arbitrary one noise
    would produce; this happens before sign continuity is enforced, so the filled
    stretch joins the sequence without a sign jump. The mask is appended as a final
    channel for every feature set, control included, so the encoder can tell "no
    rotation" from "rotation undefined" and both branches see the same information
    about where the signal was too small.
    """
    current, following = p[..., :-1, :], p[..., 1:, :]
    parts = {}
    if any(f in ("q", "theta", "omega") for f in features):
        q = torch.where(
            mask.unsqueeze(-1),
            vectors_to_quaternion(current, following),
            p.new_tensor(_IDENTITY),
        )
        if sign_continuity:
            q = enforce_sign_continuity(q)
        theta = quaternion_angle(q).unsqueeze(-1)
        parts.update(q=q, theta=theta, omega=theta / dt)
    if "magnitude" in features:
        parts["magnitude"] = _safe_norm(current, keepdim=True)
    if "linear_velocity" in features:
        parts["linear_velocity"] = _safe_norm(following - current, keepdim=True) / dt
    parts.update(position=current, next_position=following, delta=following - current)
    extra = [mask.unsqueeze(-1).to(p.dtype)] if with_mask else []
    return torch.cat([parts[f] for f in features] + extra, -1)


class QuaternionDynamicFeatures(nn.Module):
    """VCG [B, 3, T] -> per-transition features [B, C, T - 1] and a validity mask [B, T - 1].

    Parameter-free. For adjacent cardiac vectors P_t, P_{t+1} (one sample apart, so
    dt = 1 / fs):

        q_t      shortest-arc rotation u_t -> u_{t+1}    4 channels
        theta_t  2 atan2(||q_xyz||, |q_w| + eps)         1
        omega_t  theta_t / dt, rad/s                     1
        magnitude         ||P_t||                        1
        linear_velocity   ||P_{t+1} - P_t|| / dt         1

    The real-valued control replaces these with ``position``, ``next_position`` and
    ``delta`` (3 channels each).
    """

    def __init__(self, features, dt, min_fraction=0.02, sign_continuity=True, with_mask=True):
        super().__init__()
        self.features = check_features(features)
        self.dt = float(dt)
        self.min_fraction = float(min_fraction)
        self.sign_continuity = bool(sign_continuity)
        self.with_mask = bool(with_mask)
        self.channels = sum(FEATURE_CHANNELS[f] for f in self.features) + int(self.with_mask)

    def forward(self, vcg):
        p = vcg.transpose(1, 2)  # [B, T, 3]
        mask = valid_rotation_mask(p, lag=1, min_fraction=self.min_fraction)
        x = transition_features(
            p, mask, self.features, self.dt, self.sign_continuity, self.with_mask
        )
        return x.transpose(1, 2), mask
