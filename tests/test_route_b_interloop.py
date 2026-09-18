"""Route B: QRS/T segment geometry, their quaternion relation and the mechanistic target."""

import math

import pytest
import torch

from lvcg.models.lvcg import StateGRU
from lvcg.quaternion.interloop import (
    DEFAULT_WINDOWS,
    LEVELS,
    InterLoopEncoder,
    QRSTProbe,
    relation_quaternion,
    segment_axis,
    segment_masks,
    segment_normal,
    spatial_qrst_angle,
)
from lvcg.quaternion.utils import quaternion_angle, rotate_vector_by_quaternion

TOKEN_DIM, BEATS, PATCH = 256, 4, 128


def _frozen_parts():
    torch.manual_seed(0)
    return dict(
        state_generator=StateGRU(state_dim=TOKEN_DIM, hidden_dim=TOKEN_DIM, num_layers=2, dropout=0.1),
        norm_struct=torch.nn.LayerNorm(TOKEN_DIM),
        norm_dynamic=torch.nn.LayerNorm(TOKEN_DIM),
    )


def _planted_beats(qrs_axis, t_axis, batch=2, beats=BEATS, noise=0.0):
    """A synthetic record whose QRS and T windows point along known axes."""
    masks = segment_masks(PATCH)
    trajectory = torch.zeros(batch, beats, 3, PATCH)
    time = torch.linspace(0, math.pi, PATCH)
    for name, axis in (("qrs", qrs_axis), ("t", t_axis)):
        window = masks[name]
        shape = torch.sin(time[window] / time[window].max() * math.pi)
        trajectory[..., window] = axis.view(1, 1, 3, 1) * shape
    if noise:
        trajectory = trajectory + noise * torch.randn_like(trajectory)
    return trajectory, torch.ones(batch, beats)


# --- the windows ------------------------------------------------------------------


def test_windows_are_fractions_of_the_beat_and_do_not_overlap():
    masks = segment_masks(PATCH)
    qrs, t_wave = masks["qrs"], masks["t"]
    assert qrs.sum() == pytest.approx(PATCH * 0.12, abs=2)
    assert t_wave.sum() == pytest.approx(PATCH * 0.40, abs=2)
    assert not (qrs & t_wave).any()
    assert qrs[0], "a patch starts at its own R peak, so QRS opens the beat"


def test_windows_are_configurable():
    masks = segment_masks(PATCH, {"t": (0.5, 0.9)})
    assert masks["t"][int(0.5 * PATCH)] and masks["t"][int(0.85 * PATCH)]
    assert not masks["t"][int(0.3 * PATCH)]


# --- the segment geometry ---------------------------------------------------------


def test_segment_axis_recovers_a_planted_direction():
    axis = torch.tensor([0.0, 0.6, 0.8])
    beats, mask = _planted_beats(axis, torch.tensor([1.0, 0.0, 0.0]))
    found = segment_axis(beats, segment_masks(PATCH)["qrs"])
    assert torch.allclose(found[0, 0], axis, atol=1e-3), found[0, 0]


def test_segment_axis_ignores_amplitude_but_follows_direction():
    axis = torch.tensor([1.0, 0.0, 0.0])
    beats, _ = _planted_beats(axis, torch.tensor([0.0, 1.0, 0.0]))
    mask = segment_masks(PATCH)["qrs"]
    assert torch.allclose(segment_axis(beats, mask), segment_axis(beats * 5.0, mask), atol=1e-4)


def test_segment_axis_sign_follows_the_mean_vector():
    """An eigenvector is defined up to a sign; the segment's own mean fixes it."""
    axis = torch.tensor([0.0, 0.0, 1.0])
    beats, _ = _planted_beats(axis, torch.tensor([1.0, 0.0, 0.0]))
    found = segment_axis(beats, segment_masks(PATCH)["qrs"])
    assert float((found * axis).sum(-1).min()) > 0


def test_segment_normal_is_zero_for_a_straight_loop_and_not_for_a_turning_one():
    straight, _ = _planted_beats(torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]))
    mask = segment_masks(PATCH)["qrs"]
    assert float(segment_normal(straight, mask).norm(dim=-1).max()) < 1e-4

    circle = torch.zeros(1, 1, 3, PATCH)
    angle = torch.linspace(0, 2 * math.pi, PATCH)
    circle[0, 0, 0], circle[0, 0, 1] = torch.cos(angle), torch.sin(angle)
    normal = segment_normal(circle, mask)[0, 0]
    assert float(normal.norm()) > 1e-3
    assert torch.allclose(normal / normal.norm(), torch.tensor([0.0, 0.0, 1.0]), atol=1e-3)


# --- the relation and the mechanistic target --------------------------------------


def test_relation_quaternion_takes_the_first_axis_to_the_second():
    source = torch.tensor([[1.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, 1.0, 0.0]])
    quaternion, angle = relation_quaternion(source, target)
    assert torch.allclose(rotate_vector_by_quaternion(source, quaternion), target, atol=1e-6)
    assert float(angle) == pytest.approx(math.pi / 2, abs=1e-5)


@pytest.mark.parametrize("degrees", [0.0, 30.0, 90.0, 150.0])
def test_spatial_qrst_angle_matches_a_planted_angle(degrees):
    """The mechanistic target must be the angle between the QRS and T axes."""
    radians = math.radians(degrees)
    qrs = torch.tensor([1.0, 0.0, 0.0])
    t_wave = torch.tensor([math.cos(radians), math.sin(radians), 0.0])
    beats, mask = _planted_beats(qrs, t_wave)
    measured = spatial_qrst_angle(beats, mask)
    assert float(measured.mean()) == pytest.approx(degrees, abs=0.5), float(measured.mean())


def test_spatial_qrst_angle_is_invariant_to_a_global_rotation():
    """A rotation of the whole record turns both loops alike, so their angle is unchanged."""
    beats, mask = _planted_beats(torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 0.8, 0.6]))
    angle = torch.tensor(0.7)
    rotation = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0.0],
                             [torch.sin(angle), torch.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    turned = torch.einsum("ij,bnjp->bnip", rotation, beats)
    assert torch.allclose(spatial_qrst_angle(beats, mask), spatial_qrst_angle(turned, mask), atol=0.2)


def test_spatial_qrst_angle_averages_over_valid_beats_only():
    beats, mask = _planted_beats(torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]))
    padded = beats.clone()
    partial = mask.clone()
    partial[:, -1] = 0
    padded[:, -1] = torch.randn_like(padded[:, -1])
    assert torch.allclose(spatial_qrst_angle(beats[:, :-1], partial[:, :-1]),
                          spatial_qrst_angle(padded, partial), atol=1e-3)


# --- the encoder and the probe ----------------------------------------------------


@pytest.mark.parametrize("level", list(LEVELS))
def test_every_level_runs_and_is_finite(level):
    encoder = InterLoopEncoder(level=level).eval()
    beats, mask = _planted_beats(torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]), noise=0.1)
    embedding = encoder(beats, mask)
    assert embedding.shape == (2, 128) and torch.isfinite(embedding).all()


def test_levels_are_nested_in_their_summary_channels():
    assert InterLoopEncoder(level="axis").summary_channels == 11
    assert InterLoopEncoder(level="plane").summary_channels == 22
    assert InterLoopEncoder(level="trajectory").trajectory_encoder is not None
    assert InterLoopEncoder(level="axis").trajectory_encoder is None


def test_the_encoder_reacts_to_the_inter_loop_angle():
    encoder = InterLoopEncoder(level="axis").eval()
    aligned, mask = _planted_beats(torch.tensor([1.0, 0.0, 0.0]), torch.tensor([1.0, 0.0, 0.0]))
    opposed, _ = _planted_beats(torch.tensor([1.0, 0.0, 0.0]), torch.tensor([-1.0, 0.0, 0.0]))
    assert not torch.allclose(encoder(aligned, mask), encoder(opposed, mask), atol=1e-3)


def test_encoder_pools_over_valid_beats_only():
    encoder = InterLoopEncoder(level="plane").eval()
    beats, mask = _planted_beats(torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]), noise=0.05)
    padded = beats.clone()
    partial = mask.clone()
    partial[:, -1] = 0
    padded[:, -1] = torch.randn_like(padded[:, -1]) * 3
    assert torch.allclose(encoder(beats[:, :-1], partial[:, :-1]), encoder(padded, partial), atol=1e-4)


def test_unknown_level_or_structure_is_refused():
    with pytest.raises(ValueError):
        InterLoopEncoder(level="volume")
    with pytest.raises(ValueError):
        QRSTProbe(**_frozen_parts(), struct="first")


@pytest.mark.parametrize("struct", list(QRSTProbe.STRUCTS))
def test_probe_runs_in_both_structural_conventions(struct):
    model = QRSTProbe(**_frozen_parts(), struct=struct).eval()
    beats, mask = _planted_beats(torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]))
    logits = model(beats, torch.randn(2, BEATS, TOKEN_DIM), mask,
                   torch.full((2,), BEATS - 1, dtype=torch.long), torch.randn(2, 128))
    assert logits.shape == (2, 5) and torch.isfinite(logits).all()


def test_scale_zero_ignores_the_branch():
    model = QRSTProbe(**_frozen_parts(), scale=0.0).eval()
    beats, mask = _planted_beats(torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]))
    other, _ = _planted_beats(torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0, 1.0, 0.0]))
    tokens = torch.randn(2, BEATS, TOKEN_DIM)
    steps = torch.full((2,), BEATS - 1, dtype=torch.long)
    rhythm = torch.randn(2, 128)
    assert torch.equal(model(beats, tokens, mask, steps, rhythm),
                       model(other, tokens, mask, steps, rhythm))


def test_only_the_branch_and_the_head_train():
    model = QRSTProbe(**_frozen_parts(), level="trajectory")
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert all(name.startswith(("interloop", "head")) for name in trainable), sorted(trainable)[:5]
    beats, mask = _planted_beats(torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]), noise=0.1)
    model.train()
    model(beats, torch.randn(2, BEATS, TOKEN_DIM), mask,
          torch.full((2,), BEATS - 1, dtype=torch.long), torch.randn(2, 128)).sum().backward()
    assert model.interloop.summary[0].weight.grad.abs().max() > 0
    assert model.state_generator.gru.weight_ih_l0.grad is None


def test_default_windows_are_the_documented_ones():
    assert DEFAULT_WINDOWS == {"qrs": (0.0, 0.12), "t": (0.15, 0.55)}


def test_the_training_loop_drives_the_probe(tmp_path):
    """The script's own loop, exercised end to end on cached-shaped tensors."""
    import importlib.util
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "train_qrst", os.path.join(root, "scripts", "train_qrst.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    generator = torch.Generator().manual_seed(0)

    def make(n):
        labels = (torch.rand(n, 5, generator=generator) > 0.5).float()
        # Label 0 lives in the QRS-T angle: aligned loops for one class, opposed for the other.
        beats = torch.zeros(n, BEATS, 3, PATCH)
        masks = segment_masks(PATCH)
        shape = torch.sin(torch.linspace(0, math.pi, PATCH))
        beats[..., 0, masks["qrs"]] = shape[masks["qrs"]]
        sign = (1 - 2 * labels[:, 0]).view(n, 1, 1)
        beats[..., 0, masks["t"]] = shape[masks["t"]].view(1, 1, -1) * sign
        beats = beats + 0.02 * torch.randn(beats.shape, generator=generator)
        mask = torch.ones(n, BEATS)
        # The cache layout train_qrst reads: beats, rr, mask, reference, tokens, steps, rhythm, labels.
        return (beats, torch.full((n, BEATS), 85.0), mask, torch.ones(n),
                torch.zeros(n, BEATS, TOKEN_DIM), torch.full((n,), BEATS - 1, dtype=torch.long),
                torch.zeros(n, 128), labels)

    data = {"train": make(256), "val": make(64), "test": make(64)}
    model = QRSTProbe(**_frozen_parts(), level="axis", embedding_dim=32, hidden=16)
    module.train(model, data, torch.device("cpu"),
                 {"batch_size": 32, "max_epochs": 8, "patience": 8}, seed=0)
    scores = module.evaluate(model, data["test"], torch.device("cpu"), 64)
    assert scores["per_label_auroc"][0] > 0.85, scores["per_label_auroc"]
