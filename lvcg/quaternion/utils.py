"""Quaternion utilities shared by every Quaternion-LVCG variant (plan section 3.1).

Ported unchanged from the q_wyt project, where the same functions are unit tested.

Conventions, fixed once for V1-V8:

* A quaternion is a tensor whose last dimension is 4, ordered [w, x, y, z]; any leading
  batch shape is accepted and kept.
* Every quaternion that stands for a rotation is unit length. q and -q are the same
  rotation (the double cover), so the distance, the angle and sign continuity below
  are all written to be indifferent to that sign.
* Nothing here uses Euler angles.

Norms go through ``_safe_norm``, which keeps a tiny positive floor inside the square
root. The value it changes is of order 1e-6, but the gradient of an ordinary norm at an
exactly zero vector is 0/0, and the later variants (canonicalisation, adaptive lead
geometry) put learnable parameters upstream of these functions.
"""

import torch

TINY = 1e-12


def _safe_norm(x, dim=-1, keepdim=False):
    return torch.sqrt(x.square().sum(dim=dim, keepdim=keepdim) + TINY)


def normalize_quaternion(q):
    return q / _safe_norm(q, keepdim=True)


def quaternion_conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quaternion_inverse(q):
    return quaternion_conjugate(q) / (q.square().sum(-1, keepdim=True) + TINY)


def quaternion_multiply(q1, q2):
    """Hamilton product q1 * q2."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quaternion_to_rotation_matrix(q):
    """[..., 4] -> [..., 3, 3]. The input is normalised first."""
    w, x, y, z = normalize_quaternion(q).unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def rotate_vector_by_quaternion(v, q):
    """v' = q [0, v] q*, written as v + 2w (u x v) + 2 u x (u x v) for unit q = [w, u]."""
    q = normalize_quaternion(q)
    w, u = q[..., :1], q[..., 1:]
    t = 2 * torch.linalg.cross(u, v, dim=-1)
    return v + w * t + torch.linalg.cross(u, t, dim=-1)


def quaternion_angle(q, eps=1e-8):
    """Rotation angle in [0, pi]: theta = 2 atan2(||q_xyz||, |q_w| + eps).

    |q_w| makes the angle identical for q and -q.
    """
    return 2 * torch.atan2(_safe_norm(q[..., 1:]), q[..., 0].abs() + eps)


def quaternion_geodesic_distance(q1, q2):
    """Angle of the relative rotation q1^-1 q2, in [0, pi]; distance(q, -q) = 0."""
    relative = quaternion_multiply(
        quaternion_conjugate(normalize_quaternion(q1)), normalize_quaternion(q2)
    )
    return quaternion_angle(relative, eps=0.0)


def vectors_to_quaternion(v1, v2, eps=1e-8, antiparallel_tolerance=1e-3):
    """Shortest-arc unit quaternion rotating direction v1 onto direction v2.

        u1 = v1 / (||v1|| + eps),  u2 = v2 / (||v2|| + eps)
        q  = normalize([1 + u1.u2, u1 x u2])

    The scalar part 1 + u1.u2 is never negative, so q already has w >= 0. When the
    directions are antiparallel both parts vanish; the rotation is then pi about any
    axis perpendicular to u1, and the axis is chosen deterministically from the
    coordinate axis least aligned with u1. Near-zero input vectors are not an error here
    -- they produce a finite quaternion -- but the direction they carry is meaningless,
    which is what ``valid_rotation_mask`` is for.
    """
    v1, v2 = torch.broadcast_tensors(v1, v2)
    u1 = v1 / (_safe_norm(v1, keepdim=True) + eps)
    u2 = v2 / (_safe_norm(v2, keepdim=True) + eps)
    dot = (u1 * u2).sum(-1, keepdim=True)
    raw = torch.cat((1 + dot, torch.linalg.cross(u1, u2, dim=-1)), dim=-1)
    index = u1.abs().argmin(-1, keepdim=True)
    basis = torch.zeros_like(u1).scatter(-1, index, 1.0)
    axis = torch.linalg.cross(u1, basis, dim=-1)
    fallback = torch.cat((torch.zeros_like(dot), axis / _safe_norm(axis, keepdim=True)), dim=-1)
    # ||raw|| = 2 cos(theta / 2), so the tolerance means "within ~0.06 degrees of pi". It
    # is tested on a plain norm: the safe norm's floor is 1e-6, and a floor at or above
    # the tolerance would make this branch unreachable.
    antiparallel = torch.linalg.vector_norm(raw.detach(), dim=-1, keepdim=True) < (
        antiparallel_tolerance
    )
    return torch.where(antiparallel, fallback, normalize_quaternion(raw))


def valid_rotation_mask(p, lag=1, min_fraction=0.02, reference=None):
    """True where both endpoints of a transition carry a reliable direction.

    p is [..., T, 3]. A vector counts as reliable when its magnitude reaches
    ``min_fraction`` of the record's own 99th-percentile magnitude: a relative rule, so
    it is indifferent to the amplitude scale of the input, and a percentile rather than
    the maximum, so one spike cannot raise it. Returns [..., T - lag].

    The default was chosen on 256 training-fold records (no labels involved), from the
    rotation per 10 ms the rule removes. At 0.02 it masks 3% of transitions, whose
    median rotation is 34 degrees against 10 for the rest -- noise, not physiology. At
    0.05 it already masks a quarter of every record (median 15 degrees), which reaches
    into genuine low-amplitude P and ST segments.

    ``reference`` overrides the percentile, broadcast against [..., T]. Beat patches need
    it: a patch's own percentile would call a quiet boundary interval reliable, so they
    take the whole record's instead.
    """
    magnitude = _safe_norm(p)
    if reference is None:
        reference = torch.quantile(magnitude.detach(), 0.99, dim=-1, keepdim=True)
    reliable = magnitude >= min_fraction * reference
    return reliable[..., :-lag] & reliable[..., lag:]


def enforce_sign_continuity(q):
    """Flip signs along the time axis (-2) so consecutive quaternions have dot >= 0.

    q_t and -q_t are the same rotation, but an unconstrained sign makes a temporal
    sequence jump. With c_0 = 1 and c_t = c_{t-1} sign(q_t . q_{t-1}), every
    consecutive pair of c_t q_t has a non-negative dot product.
    """
    dots = (q[..., 1:, :] * q[..., :-1, :]).sum(-1)
    signs = torch.where(dots < 0, -1.0, 1.0).to(q.dtype)
    flips = torch.cat((torch.ones_like(signs[..., :1]), torch.cumprod(signs, dim=-1)), dim=-1)
    return q * flips.detach().unsqueeze(-1)


def quaternion_sequence(p, lag=1, sign_continuity=True, eps=1e-8):
    """q_t = Rot(u_t -> u_{t+lag}) for a trajectory p [..., T, 3] -> [..., T - lag, 4]."""
    if lag < 1 or lag >= p.shape[-2]:
        raise ValueError("lag must be at least 1 and shorter than the sequence")
    q = vectors_to_quaternion(p[..., :-lag, :], p[..., lag:, :], eps=eps)
    return enforce_sign_continuity(q) if sign_continuity else q


def angular_velocity_from_quaternion(q, dt):
    """Angular speed omega = theta / dt, in radians per second when dt is in seconds."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    return quaternion_angle(q) / dt
