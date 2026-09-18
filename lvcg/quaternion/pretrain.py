"""Route E: rotational geometry as a pretraining objective, not a downstream branch.

A/B/C/D all add something *beside* a frozen model. Route E asks the different question:
should the representation learner itself be told to get the rotation right? The release's
beat loss is a mean squared error on the VCG patch, and squared error is dominated by
amplitude -- a beat can be reconstructed with the right size and the wrong turning and
still score well. This module adds the term that squared error does not see:

    L_rot = mean over scales s, over valid steps of  1 - |<W_hat_s(t) , W_s(t)>|

where W_s(t) is the rotation composed over the last s steps of the true VCG patch and
W_hat_s(t) the same for the reconstructed one. The quantity is the geodesic distance on
the rotation group, written so that q and -q -- the same rotation -- cost nothing.

**Why composed windows rather than single steps.** Between two adjacent samples the
cardiac vector turns by a hundredth of a radian, so a single-step quaternion is almost
the identity and ``|<q_hat, q>|`` stays near 1 whatever the *axis* does: a
time-reversed reconstruction, or one turned rigidly away from the target, both cost
about 6e-5. Composing over eight or thirty-two steps makes the accumulated angle large
enough that the axis matters, and those two cases then cost what they should. The scale
1 term is kept as well, so the local rate of turning still has a gradient.

Three properties make it safe to add to an existing objective:

* **It is scale free.** Multiplying a trajectory by any positive constant leaves every
  step rotation unchanged, so the term cannot be minimised by shrinking the output, and
  it does not compete with the reconstruction term for amplitude.
* **It is rotation covariant, not invariant.** Turning the prediction relative to the
  target still costs, because the *relative* rotations between consecutive samples change
  unless the whole trajectory is turned rigidly -- and a rigid turn of the prediction is
  exactly what the reconstruction term already penalises.
* **It is inert at weight zero.** ``scripts/train.py`` adds the term only when its weight
  is non-zero, so an unchanged configuration reproduces the released objective exactly,
  which ``tests/test_route_e_pretrain.py`` checks.

**Gate status (plan section 7).** The plan admits route E only once A/B show that
rotational geometry holds information V0 does not expose. A's composition beat its local
control by +0.05 pp and B's inter-loop relation by +0.11 pp, so that gate is **not met**;
route C was the positive result, and it concerns the frame of the representation rather
than the learning objective. What follows is therefore the plan's *feasibility* step --
is the loss learnable, numerically sound, and harmless to the original reconstruction --
and not a licence to spend a full MIMIC pretraining run.
"""

import torch

from .features import _IDENTITY
from .loop import beat_directions, prefix_composition, window_rotations
from .utils import _safe_norm, vectors_to_quaternion

TINY = 1e-12


def step_quaternions(trajectory, mask):
    """Shortest-arc rotations along a trajectory [..., 3, P] -> [..., P-1, 4].

    Steps outside ``mask`` become the identity so they contribute nothing to a distance.
    """
    points = trajectory.transpose(-1, -2)
    directions = points / _safe_norm(points, keepdim=True)
    quaternions = vectors_to_quaternion(directions[..., :-1, :], directions[..., 1:, :])
    return torch.where(mask.unsqueeze(-1), quaternions, points.new_tensor(_IDENTITY))


DEFAULT_SCALES = (1, 8, 32)


def rotational_consistency_loss(predicted, target, beat_mask, reference=None,
                                min_fraction=0.02, scales=DEFAULT_SCALES, eps=1e-8):
    """Geodesic disagreement between the two trajectories' composed rotations.

    Args:
        predicted, target: [B, N, 3, P] reconstructed and geometry-recovered VCG patches.
        beat_mask: [B, N] valid beats.
        reference: [B] the record's 99th-percentile magnitude; taken from the target when
            omitted, so the reliability rule matches the rest of the project.
        scales: window lengths to compose over; 1 is the single step.

    Returns a scalar in [0, 1]: 0 when the reconstruction turns exactly as the target
    does at every scale, 1 when the rotations are orthogonal. Steps whose target
    direction is unreliable are excluded, and a record with no reliable step contributes
    nothing rather than a NaN.
    """
    if reference is None:
        magnitude = _safe_norm(target.transpose(-1, -2))
        reference = torch.quantile(magnitude.detach().flatten(1), 0.99, dim=-1)
    _, mask = beat_directions(target, beat_mask, reference, min_fraction)

    steps_target = step_quaternions(target, mask)
    steps_predicted = step_quaternions(predicted, mask)
    prefix_target = prefix_composition(steps_target)
    prefix_predicted = prefix_composition(steps_predicted)

    weight = mask.to(steps_target.dtype)
    total = weight.sum().clamp_min(1.0)
    loss = steps_target.new_zeros(())
    for scale in scales:
        if scale == 1:
            a, b = steps_predicted, steps_target
        else:
            a = window_rotations(prefix_predicted, scale)
            b = window_rotations(prefix_target, scale)
        # |<W_hat, W>| makes the double cover free: W and -W are the same rotation.
        agreement = (a * b).sum(-1).abs().clamp(max=1.0)
        loss = loss + ((1 - agreement) * weight).sum() / total
    return loss / len(scales)


def rotation_angle_error(predicted, target, beat_mask, reference=None, min_fraction=0.02):
    """The same disagreement in degrees, for reporting rather than for the gradient."""
    with torch.no_grad():
        loss = rotational_consistency_loss(predicted, target, beat_mask, reference, min_fraction)
        # 1 - |cos(theta/2)| -> theta, the rotation between the two step rotations.
        return 2 * torch.arccos((1 - loss).clamp(-1.0, 1.0)) * 180.0 / torch.pi
