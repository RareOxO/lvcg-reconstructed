"""V5 (QC-LVCG): pose estimation, identity initialisation and the rotation machinery."""

import pytest
import torch

from lvcg.data.beat_segmentation import BeatSegmenter
from lvcg.models.lvcg import StateGRU
from lvcg.models.blocks.beat_modules import BeatEncoder
from lvcg.quaternion.canon import (
    CanonProbe,
    PoseNet,
    axis_angle_to_quaternion,
    random_rotations,
    rotate_beats,
)
from lvcg.quaternion.utils import quaternion_angle, quaternion_to_rotation_matrix

TOKEN_DIM, BEATS, PATCH = 256, 6, 128
IDENTITY = torch.tensor([1.0, 0.0, 0.0, 0.0])


def _frozen_parts():
    torch.manual_seed(0)
    return dict(
        beat_encoder=BeatEncoder(beat_len=PATCH, state_dim=TOKEN_DIM),
        state_generator=StateGRU(state_dim=TOKEN_DIM, hidden_dim=TOKEN_DIM, num_layers=2, dropout=0.1),
        norm_struct=torch.nn.LayerNorm(TOKEN_DIM),
        norm_dynamic=torch.nn.LayerNorm(TOKEN_DIM),
    )


def _batch(batch=3, seed=1):
    generator = torch.Generator().manual_seed(seed)
    beats = torch.randn(batch, BEATS, 3, PATCH, generator=generator)
    rr = torch.full((batch, BEATS), 85.0)
    mask = torch.ones(batch, BEATS)
    steps = torch.full((batch,), BEATS - 1, dtype=torch.long)
    rhythm = torch.randn(batch, 128, generator=generator)
    return beats, rr, mask, steps, rhythm


# --- the rotation machinery -------------------------------------------------------


def test_zero_axis_angle_is_exactly_the_identity():
    assert torch.equal(axis_angle_to_quaternion(torch.zeros(4, 3))[:, 0], torch.ones(4))
    assert axis_angle_to_quaternion(torch.zeros(4, 3))[:, 1:].abs().max() < 1e-6


def test_axis_angle_turns_by_the_requested_angle_about_the_requested_axis():
    axis = torch.tensor([[0.0, 0.0, 1.0]])
    angle = torch.tensor([[torch.pi / 3]])
    q = axis_angle_to_quaternion(axis * angle)
    assert torch.allclose(quaternion_angle(q), angle.squeeze(-1), atol=1e-6)
    turned = quaternion_to_rotation_matrix(q)[0] @ torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(turned, torch.tensor([0.5, float(torch.sin(angle)), 0.0]), atol=1e-6)


def test_random_rotations_are_rotations_with_the_requested_angle():
    matrices = random_rotations(16, 30, torch.Generator().manual_seed(0))
    identity = torch.eye(3).expand(16, 3, 3)
    assert torch.allclose(matrices.transpose(-1, -2) @ matrices, identity, atol=1e-5)
    assert torch.allclose(torch.linalg.det(matrices), torch.ones(16), atol=1e-5)
    # The rotation angle of R is arccos((trace - 1) / 2).
    angle = torch.arccos(((matrices.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))
    assert torch.allclose(angle * 180 / torch.pi, torch.full((16,), 30.0), atol=1e-3)


def test_rotation_commutes_with_beat_segmentation():
    """Why V5 may rotate cached patches instead of re-running segmentation."""
    torch.manual_seed(0)
    segmenter = BeatSegmenter(beat_len=64, fs=100, max_beats=10)
    vcg = torch.randn(2, 3, 1000)
    ecg = torch.randn(2, 12, 1000)
    ecg[:, 1, 40::85] += 6.0
    rotation = random_rotations(1, 40, torch.Generator().manual_seed(3))[0]

    rotated_first, _, _ = segmenter(torch.einsum("ij,bjt->bit", rotation, vcg), ecg)
    segmented_first, _, _ = segmenter(vcg, ecg)
    assert torch.allclose(rotated_first, rotate_beats(segmented_first, rotation), atol=1e-5)


def test_rotated_beats_are_contiguous_for_the_pretrained_encoder():
    beats = torch.randn(2, BEATS, 3, PATCH)
    assert rotate_beats(beats, random_rotations(2, 20, torch.Generator().manual_seed(0))).is_contiguous()


# --- the pose head ----------------------------------------------------------------


def test_pose_starts_at_the_identity():
    """Plan's hard requirement: V5 must not perturb the input distribution at step 0."""
    pose = PoseNet().eval()
    beats = torch.randn(4, BEATS, 3, PATCH)
    q = pose(beats)
    assert torch.allclose(q, IDENTITY.expand(4, 4), atol=1e-6)
    assert float(quaternion_angle(q).max()) < 1e-4


def test_max_degrees_bounds_the_correction():
    pose = PoseNet(max_degrees=10).eval()
    torch.nn.init.normal_(pose.head.weight, std=5.0)  # a head that wants a large rotation
    torch.nn.init.normal_(pose.head.bias, std=5.0)
    angles = quaternion_angle(pose(torch.randn(8, BEATS, 3, PATCH))) * 180 / torch.pi
    assert float(angles.max()) <= 10.0 + 1e-3, float(angles.max())


def test_the_pose_weighs_padding_beats_far_less_than_real_ones():
    """The mask restricts pooling, not the convolutions' receptive field.

    Padding beats are zeros in the cache, so the leak is small in practice; this checks
    that a change confined to padding moves the pose much less than the same change
    inside the valid beats.
    """
    torch.manual_seed(0)
    pose = PoseNet().eval()
    torch.nn.init.normal_(pose.head.weight, std=1.0)
    beats = torch.zeros(2, BEATS, 3, PATCH)
    beats[:, :-2] = torch.randn(2, BEATS - 2, 3, PATCH)
    mask = torch.ones(2, BEATS)
    mask[:, -2:] = 0
    reference = quaternion_angle(pose(beats, mask))

    in_padding = beats.clone()
    in_padding[:, -2:] = torch.randn(2, 2, 3, PATCH)
    in_valid = beats.clone()
    in_valid[:, 0] = torch.randn(2, 3, PATCH)
    moved_padding = (quaternion_angle(pose(in_padding, mask)) - reference).abs().mean()
    moved_valid = (quaternion_angle(pose(in_valid, mask)) - reference).abs().mean()
    assert moved_valid > 3 * moved_padding, (float(moved_valid), float(moved_padding))


# --- the probe --------------------------------------------------------------------


def test_identity_pose_reproduces_the_frozen_path_bitwise():
    parts = _frozen_parts()
    canon = CanonProbe(**parts, canonicalize=True, max_degrees=30).eval()
    plain = CanonProbe(**parts, canonicalize=False).eval()
    plain.head.load_state_dict(canon.head.state_dict())
    batch = _batch()
    assert torch.equal(canon(*batch), plain(*batch))


def test_a_fixed_pose_undoes_exactly_that_rotation():
    """With the pose head pinned to r, a record turned by R(r) must land back where it was."""
    parts = _frozen_parts()
    canon = CanonProbe(**parts, canonicalize=True).eval()
    plain = CanonProbe(**parts, canonicalize=False).eval()
    plain.head.load_state_dict(canon.head.state_dict())

    vector = torch.tensor([0.3, -0.2, 0.5])
    with torch.no_grad():
        canon.pose.head.bias.copy_(vector)  # weights stay zero: the pose is constant
    rotation = quaternion_to_rotation_matrix(axis_angle_to_quaternion(vector))

    beats, rr, mask, steps, rhythm = _batch()
    turned = rotate_beats(beats, rotation)
    assert torch.allclose(canon(turned, rr, mask, steps, rhythm),
                          plain(beats, rr, mask, steps, rhythm), atol=1e-4)


def test_rotation_changes_the_frozen_model_that_has_no_pose_head():
    """The nuisance V5 targets is real: the pretrained path is not rotation invariant."""
    plain = CanonProbe(**_frozen_parts(), canonicalize=False).eval()
    beats, rr, mask, steps, rhythm = _batch()
    turned = rotate_beats(beats, random_rotations(1, 60, torch.Generator().manual_seed(0))[0])
    assert not torch.allclose(plain(beats, rr, mask, steps, rhythm),
                              plain(turned, rr, mask, steps, rhythm), atol=1e-3)


def test_only_the_pose_and_the_head_are_trainable():
    model = CanonProbe(**_frozen_parts(), canonicalize=True)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert all(name.startswith(("pose", "head")) for name in trainable), sorted(trainable)[:5]
    counts = model.parameter_counts()
    assert counts["trainable_total"] == counts["pose"] + counts["head"]


def test_gradients_reach_the_pose_head_through_the_frozen_model():
    model = CanonProbe(**_frozen_parts(), canonicalize=True).train()
    model(*_batch()).sum().backward()
    assert model.pose.head.weight.grad.abs().max() > 0
    assert model.beat_encoder.parameters().__next__().grad is None
    assert model.state_generator.gru.weight_ih_l0.grad is None


def test_frozen_gru_follows_the_probe_mode_with_dropout_off():
    model = CanonProbe(**_frozen_parts(), canonicalize=True)
    assert model.state_generator.gru.dropout == 0.0
    model.train()
    assert model.state_generator.training and not model.beat_encoder.training


def test_pose_angles_are_zero_without_a_pose_head():
    model = CanonProbe(**_frozen_parts(), canonicalize=False).eval()
    assert float(model.pose_angles(_batch()[0]).abs().max()) == 0.0


@pytest.mark.parametrize("degrees", [0, 5, 30, 90, 150])
def test_the_robustness_sweep_stays_finite_at_every_angle(degrees):
    model = CanonProbe(**_frozen_parts(), canonicalize=True, max_degrees=30).eval()
    beats, rr, mask, steps, rhythm = _batch()
    rotation = random_rotations(beats.shape[0], degrees, torch.Generator().manual_seed(degrees))
    assert torch.isfinite(model(rotate_beats(beats, rotation), rr, mask, steps, rhythm)).all()
