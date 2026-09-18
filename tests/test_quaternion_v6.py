"""V6 (QA-LVCG): the constrained lead-geometry rotation and its equivalences."""

import pytest
import torch

from lvcg.data.angle import get_lead_directions
from lvcg.models.lvcg import StateGRU
from lvcg.models.blocks.beat_modules import BeatEncoder
from lvcg.models.vcg import VCGPseudoInverse
from lvcg.quaternion.canon import axis_angle_to_quaternion, random_rotations, rotate_beats
from lvcg.quaternion.geometry import GlobalRotation, QAProbe, rotated_directions
from lvcg.quaternion.utils import quaternion_angle, quaternion_to_rotation_matrix

TOKEN_DIM, BEATS, PATCH = 256, 6, 128
LIFT_EPS = 0.1  # the released VCGPseudoInverse's regularisation


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
    return (torch.randn(batch, BEATS, 3, PATCH, generator=generator),
            torch.full((batch, BEATS), 85.0),
            torch.ones(batch, BEATS),
            torch.full((batch,), BEATS - 1, dtype=torch.long),
            torch.randn(batch, 128, generator=generator))


# --- the geometry -----------------------------------------------------------------


def test_rotating_the_geometry_equals_rotating_the_recovered_vcg():
    """pinv(A_0 R) e = R^T pinv(A_0) e -- why V6 may reuse the cached beats.

    Checked against the released pseudo-inverse, with its own regularisation, in float64.
    """
    lift = VCGPseudoInverse(eps=LIFT_EPS)
    directions = get_lead_directions("mimic", as_tensor=True).double()
    ecg = torch.randn(2, 12, 64, dtype=torch.float64)
    rotation = quaternion_to_rotation_matrix(
        axis_angle_to_quaternion(torch.tensor([0.3, -0.2, 0.5], dtype=torch.float64)))

    from_geometry = lift(ecg, (directions @ rotation).unsqueeze(0).expand(2, -1, -1))
    from_rotation = torch.einsum("ij,bjt->bit", rotation.T, lift(ecg, directions.unsqueeze(0).expand(2, -1, -1)))
    assert torch.allclose(from_geometry, from_rotation, atol=1e-10), \
        float((from_geometry - from_rotation).abs().max())


def test_rotated_directions_keep_the_lead_matrix_physical():
    """Unit directions stay unit and their mutual angles are unchanged: A_Q is A_0 turned."""
    directions = get_lead_directions("mimic", as_tensor=True).double()
    rotation = axis_angle_to_quaternion(torch.tensor([0.2, 0.4, -0.1], dtype=torch.float64))
    rotated = rotated_directions(directions, rotation)
    assert torch.allclose(rotated.norm(dim=-1), directions.norm(dim=-1), atol=1e-12)
    assert torch.allclose(rotated @ rotated.T, directions @ directions.T, atol=1e-12)


def test_the_identity_quaternion_leaves_the_geometry_untouched():
    directions = get_lead_directions("mimic", as_tensor=True)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert torch.allclose(rotated_directions(directions, identity), directions, atol=1e-6)


# --- the global rotation ----------------------------------------------------------


def test_global_rotation_starts_at_the_identity_and_is_shared():
    rotation = GlobalRotation()
    assert torch.allclose(rotation.quaternion(), torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6)
    beats = torch.randn(5, BEATS, 3, PATCH)
    quaternions = rotation(beats)
    assert quaternions.shape == (5, 4)
    assert torch.allclose(quaternions, quaternions[:1].expand(5, 4)), "one rotation for every record"


def test_global_rotation_respects_its_bound():
    rotation = GlobalRotation(max_degrees=10)
    with torch.no_grad():
        rotation.vector.copy_(torch.tensor([3.0, -2.0, 1.0]))  # far beyond the bound
    assert float(quaternion_angle(rotation.quaternion()) * 180 / torch.pi) <= 10.0 + 1e-3


def test_global_rotation_has_three_parameters():
    assert sum(p.numel() for p in GlobalRotation().parameters()) == 3


# --- the probe --------------------------------------------------------------------


@pytest.mark.parametrize("mode", list(QAProbe.MODES))
def test_every_mode_runs_and_starts_at_the_published_geometry(mode):
    model = QAProbe(**_frozen_parts(), mode=mode, max_degrees=10).eval()
    directions = get_lead_directions("mimic", as_tensor=True)
    beats, rr, mask, steps, rhythm = _batch()
    logits = model(beats, rr, mask, steps, rhythm)
    assert logits.shape == (3, 5) and torch.isfinite(logits).all()
    geometry = model.geometry(directions, beats, mask)
    assert torch.allclose(geometry, directions.expand_as(geometry), atol=1e-6), "A_Q(t=0) = A_0"


def test_identity_geometry_reproduces_the_frozen_path_bitwise():
    parts = _frozen_parts()
    fixed = QAProbe(**parts, mode="fixed").eval()
    for mode in ("global", "conditioned"):
        adaptive = QAProbe(**parts, mode=mode, max_degrees=10).eval()
        adaptive.head.load_state_dict(fixed.head.state_dict())
        assert torch.equal(adaptive(*_batch()), fixed(*_batch())), mode


def test_a_known_global_rotation_matches_rotating_the_beats():
    """The geometry correction and a rotation of the trajectory are the same operation."""
    parts = _frozen_parts()
    adaptive = QAProbe(**parts, mode="global").eval()
    fixed = QAProbe(**parts, mode="fixed").eval()
    fixed.head.load_state_dict(adaptive.head.state_dict())

    vector = torch.tensor([0.1, -0.25, 0.3])
    with torch.no_grad():
        adaptive.pose.vector.copy_(vector)
    rotation = quaternion_to_rotation_matrix(axis_angle_to_quaternion(vector))

    beats, rr, mask, steps, rhythm = _batch()
    turned = rotate_beats(beats, rotation.T)  # what A_Q = A_0 R does to the recovered VCG
    assert torch.allclose(adaptive(beats, rr, mask, steps, rhythm),
                          fixed(turned, rr, mask, steps, rhythm), atol=1e-5)


def test_conditioned_mode_gives_each_record_its_own_correction():
    model = QAProbe(**_frozen_parts(), mode="conditioned", max_degrees=30).eval()
    # A small standard deviation keeps the corrections inside the bound; larger weights
    # saturate the tanh and every record gets the maximum, which is what V5 observed.
    torch.nn.init.normal_(model.pose.head.weight, std=0.01)
    angles = model.pose_angles(*_batch(batch=6)[:1], _batch(batch=6)[2])
    assert angles.shape == (6,) and float(angles.std()) > 0.1, float(angles.std())


def test_global_mode_reports_one_angle_for_every_record():
    model = QAProbe(**_frozen_parts(), mode="global", max_degrees=30).eval()
    with torch.no_grad():
        model.pose.vector.copy_(torch.tensor([0.1, 0.1, 0.1]))
    angles = model.pose_angles(_batch(batch=4)[0])
    assert angles.shape == (4,) and float(angles.std()) < 1e-6


def test_geodesic_penalty_is_the_squared_correction_angle():
    model = QAProbe(**_frozen_parts(), mode="global").eval()
    vector = torch.tensor([0.0, 0.0, 0.4])
    with torch.no_grad():
        model.pose.vector.copy_(vector)
    beats = _batch()[0]
    assert torch.allclose(model.geodesic_penalty(beats), torch.tensor(0.4 ** 2), atol=1e-5)
    assert float(QAProbe(**_frozen_parts(), mode="fixed").geodesic_penalty(beats)) == 0.0


def test_bounds_are_honoured_in_conditioned_mode():
    for bound in (5, 10, 30):
        model = QAProbe(**_frozen_parts(), mode="conditioned", max_degrees=bound).eval()
        torch.nn.init.normal_(model.pose.head.weight, std=5.0)
        torch.nn.init.normal_(model.pose.head.bias, std=5.0)
        angles = model.pose_angles(*_batch()[:1], _batch()[2])
        assert float(angles.max()) <= bound + 1e-3, (bound, float(angles.max()))


def test_only_the_geometry_and_the_head_are_trainable():
    model = QAProbe(**_frozen_parts(), mode="global")
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert all(name.startswith(("pose", "head")) for name in trainable), sorted(trainable)[:5]
    counts = model.parameter_counts()
    assert counts["geometry"] == 3 and counts["trainable_total"] == 3 + counts["head"]


def test_gradients_reach_the_geometry_through_the_frozen_model():
    for mode in ("global", "conditioned"):
        model = QAProbe(**_frozen_parts(), mode=mode, max_degrees=10).train()
        model(*_batch()).sum().backward()
        parameter = model.pose.vector if mode == "global" else model.pose.head.weight
        assert parameter.grad.abs().max() > 0, mode
        assert model.state_generator.gru.weight_ih_l0.grad is None


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        QAProbe(**_frozen_parts(), mode="free_matrix")
