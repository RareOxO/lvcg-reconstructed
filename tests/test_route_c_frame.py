"""Route C: the intrinsic frame, the equivariant/invariant split and its transformation laws."""

import importlib.util
import math
import os

import pytest
import torch

from lvcg.models.lvcg import StateGRU
from lvcg.quaternion.canon import random_rotations, rotate_beats
from lvcg.quaternion.frame import (
    PARTS,
    FrameProbe,
    FrameSplitEncoder,
    intrinsic_frame,
    to_frame,
)
from lvcg.quaternion.utils import (
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    vectors_to_quaternion,
)

TOKEN_DIM, BEATS, PATCH = 256, 4, 64
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _frozen_parts():
    torch.manual_seed(0)
    return dict(
        state_generator=StateGRU(state_dim=TOKEN_DIM, hidden_dim=TOKEN_DIM, num_layers=2, dropout=0.1),
        norm_struct=torch.nn.LayerNorm(TOKEN_DIM),
        norm_dynamic=torch.nn.LayerNorm(TOKEN_DIM),
    )


def _record(batch=3, seed=0):
    """A record whose beats sweep a real loop, so the frame is well defined."""
    generator = torch.Generator().manual_seed(seed)
    angle = torch.linspace(0, 2 * math.pi, PATCH)
    loop = torch.stack((2.0 * torch.cos(angle), torch.sin(angle), 0.2 * torch.sin(2 * angle)))
    beats = loop.expand(batch, BEATS, 3, PATCH).clone()
    beats = beats + 0.05 * torch.randn(beats.shape, generator=generator)
    return beats, torch.ones(batch, BEATS)


def _probe_inputs(batch=3):
    beats, mask = _record(batch)
    return (beats, torch.randn(batch, BEATS, TOKEN_DIM), mask,
            torch.full((batch,), BEATS - 1, dtype=torch.long), torch.randn(batch, 128))


# --- the matrix-to-quaternion helper ----------------------------------------------


def test_rotation_matrix_to_quaternion_round_trips():
    torch.manual_seed(0)
    a = torch.randn(200, 3, dtype=torch.float64)
    b = torch.randn(200, 3, dtype=torch.float64)
    q = vectors_to_quaternion(a / a.norm(dim=-1, keepdim=True), b / b.norm(dim=-1, keepdim=True))
    R = quaternion_to_rotation_matrix(q)
    assert torch.allclose(quaternion_to_rotation_matrix(rotation_matrix_to_quaternion(R)), R, atol=1e-9)


def test_rotation_matrix_to_quaternion_handles_the_hard_cases():
    """A half turn is where the naive formula divides by zero."""
    for axis in (torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])):
        q = torch.cat((torch.zeros(1), axis)).double()
        R = quaternion_to_rotation_matrix(q)
        recovered = rotation_matrix_to_quaternion(R)
        assert torch.isfinite(recovered).all()
        assert torch.allclose(quaternion_to_rotation_matrix(recovered), R, atol=1e-9)
    identity = rotation_matrix_to_quaternion(torch.eye(3, dtype=torch.float64))
    assert torch.allclose(identity, torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64), atol=1e-9)


def test_the_returned_quaternion_has_a_non_negative_real_part():
    torch.manual_seed(1)
    a, b = torch.randn(50, 3), torch.randn(50, 3)
    R = quaternion_to_rotation_matrix(vectors_to_quaternion(a, b))
    assert (rotation_matrix_to_quaternion(R)[..., 0] >= 0).all()


# --- the intrinsic frame ----------------------------------------------------------


def test_the_frame_is_an_orthonormal_right_handed_basis():
    beats, mask = _record()
    rotation, quaternion = intrinsic_frame(beats, mask)
    identity = torch.eye(3).expand(len(beats), 3, 3)
    assert torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-4)
    assert torch.allclose(torch.linalg.det(rotation), torch.ones(len(beats)), atol=1e-4)
    assert torch.allclose(quaternion_to_rotation_matrix(quaternion), rotation, atol=1e-4)


def test_the_frame_is_equivariant():
    """R_frame -> G R_frame, and q_frame -> q_G (x) q_frame: the plan's predictable part."""
    beats, mask = _record()
    turn = random_rotations(1, 50, torch.Generator().manual_seed(0))[0]
    rotation, quaternion = intrinsic_frame(beats, mask)
    turned_rotation, turned_quaternion = intrinsic_frame(rotate_beats(beats, turn), mask)

    assert torch.allclose(turned_rotation, turn @ rotation, atol=1e-3)
    expected = quaternion_multiply(rotation_matrix_to_quaternion(turn).expand(len(beats), 4), quaternion)
    # q and -q are the same rotation, so compare the rotations they define.
    assert torch.allclose(quaternion_to_rotation_matrix(turned_quaternion),
                          quaternion_to_rotation_matrix(expected), atol=1e-3)


def test_coordinates_in_the_frame_are_invariant():
    beats, mask = _record()
    turn = random_rotations(1, 75, torch.Generator().manual_seed(1))[0]
    rotation, _ = intrinsic_frame(beats, mask)
    turned = rotate_beats(beats, turn)
    turned_rotation, _ = intrinsic_frame(turned, mask)
    assert torch.allclose(to_frame(beats, rotation), to_frame(turned, turned_rotation), atol=1e-4)


def test_a_degenerate_loop_falls_back_instead_of_producing_noise():
    """A trajectory with no plane still has to give a finite, deterministic frame."""
    line = torch.zeros(2, BEATS, 3, PATCH)
    line[..., 0, :] = torch.linspace(-1, 1, PATCH)  # a straight line: zero area vector
    mask = torch.ones(2, BEATS)
    rotation, quaternion = intrinsic_frame(line, mask)
    assert torch.isfinite(rotation).all() and torch.isfinite(quaternion).all()
    assert torch.allclose(rotation.transpose(-1, -2) @ rotation, torch.eye(3).expand(2, 3, 3), atol=1e-4)
    again, _ = intrinsic_frame(line, mask)
    assert torch.equal(rotation, again), "the fallback must be deterministic"


def test_the_frame_uses_valid_beats_only():
    beats, mask = _record()
    padded = beats.clone()
    partial = mask.clone()
    partial[:, -1] = 0
    padded[:, -1] = torch.randn_like(padded[:, -1]) * 10
    first, _ = intrinsic_frame(beats[:, :-1], partial[:, :-1])
    second, _ = intrinsic_frame(padded, partial)
    assert torch.allclose(first, second, atol=1e-4)


# --- the split representation -----------------------------------------------------


@pytest.mark.parametrize("parts", [("invariant",), ("equivariant",), ("invariant", "equivariant")])
def test_every_part_combination_runs(parts):
    encoder = FrameSplitEncoder(parts=parts).eval()
    beats, mask = _record()
    embedding = encoder(beats, mask)
    assert embedding.shape == (3, 128 * len(parts)) and torch.isfinite(embedding).all()


def test_the_invariant_part_does_not_move_under_a_global_rotation():
    """Exactly frame-invariant by construction, not merely robust."""
    encoder = FrameSplitEncoder(parts=("invariant",)).eval()
    beats, mask = _record()
    turned = rotate_beats(beats, random_rotations(1, 90, torch.Generator().manual_seed(2))[0])
    assert torch.allclose(encoder(beats, mask), encoder(turned, mask), atol=1e-4)


def test_the_equivariant_part_does_move():
    """Otherwise the orientation would have been discarded, which the plan forbids."""
    encoder = FrameSplitEncoder(parts=("equivariant",)).eval()
    beats, mask = _record()
    turned = rotate_beats(beats, random_rotations(1, 90, torch.Generator().manual_seed(3))[0])
    assert not torch.allclose(encoder(beats, mask), encoder(turned, mask), atol=1e-3)


def test_unknown_or_empty_parts_are_refused():
    with pytest.raises(ValueError):
        FrameSplitEncoder(parts=("invariant", "covariant"))
    with pytest.raises(ValueError):
        FrameSplitEncoder(parts=())


# --- the probe --------------------------------------------------------------------


def test_an_invariant_only_probe_is_rotation_invariant_end_to_end():
    model = FrameProbe(**_frozen_parts(), parts=("invariant",)).eval()
    beats, tokens, mask, steps, rhythm = _probe_inputs()
    turned = rotate_beats(beats, random_rotations(1, 60, torch.Generator().manual_seed(4))[0])
    assert torch.allclose(model(beats, tokens, mask, steps, rhythm),
                          model(turned, tokens, mask, steps, rhythm), atol=1e-4)


def test_a_probe_with_the_frame_reacts_to_rotation():
    model = FrameProbe(**_frozen_parts(), parts=("invariant", "equivariant")).eval()
    beats, tokens, mask, steps, rhythm = _probe_inputs()
    turned = rotate_beats(beats, random_rotations(1, 60, torch.Generator().manual_seed(5))[0])
    assert not torch.allclose(model(beats, tokens, mask, steps, rhythm),
                              model(turned, tokens, mask, steps, rhythm), atol=1e-3)


def test_scale_zero_reproduces_v0():
    model = FrameProbe(**_frozen_parts(), scale=0.0).eval()
    beats, tokens, mask, steps, rhythm = _probe_inputs()
    other, _ = _record(seed=9)
    assert torch.equal(model(beats, tokens, mask, steps, rhythm),
                       model(other, tokens, mask, steps, rhythm))


@pytest.mark.parametrize("struct", list(FrameProbe.STRUCTS))
def test_both_structural_conventions_run(struct):
    model = FrameProbe(**_frozen_parts(), struct=struct).eval()
    logits = model(*_probe_inputs())
    assert logits.shape == (3, 5) and torch.isfinite(logits).all()


def test_only_the_branch_and_the_head_train():
    model = FrameProbe(**_frozen_parts(), parts=("invariant", "equivariant"))
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert all(name.startswith(("frame", "head")) for name in trainable), sorted(trainable)[:5]
    model.train()
    model(*_probe_inputs()).sum().backward()
    assert model.frame.invariant.net[1].weight.grad.abs().max() > 0
    assert model.frame.equivariant[0].weight.grad.abs().max() > 0
    assert model.state_generator.gru.weight_ih_l0.grad is None


def test_the_script_reports_both_transformation_properties():
    spec = importlib.util.spec_from_file_location(
        "train_frame", os.path.join(REPO_ROOT, "scripts", "train_frame.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    beats, mask = _record(batch=8)
    tensors = (beats, torch.full((8, BEATS), 85.0), mask, torch.ones(8),
               torch.zeros(8, BEATS, TOKEN_DIM), torch.full((8,), BEATS - 1, dtype=torch.long),
               torch.zeros(8, 128), torch.zeros(8, 5))
    report = module.frame_properties(tensors, torch.device("cpu"))
    assert report["frame_equivariance_error"] < 1e-2
    assert report["frame_invariance_error"] < 1e-2


def test_the_training_loop_learns_a_frame_invariant_signal():
    spec = importlib.util.spec_from_file_location(
        "train_frame", os.path.join(REPO_ROOT, "scripts", "train_frame.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    generator = torch.Generator().manual_seed(0)

    def make(n):
        labels = (torch.rand(n, 5, generator=generator) > 0.5).float()
        angle = torch.linspace(0, 2 * math.pi, PATCH)
        # Label 0 changes the loop's shape in its own frame: a circle or a flat ellipse.
        width = 1.0 + 3.0 * labels[:, 0]
        beats = torch.stack((width.view(n, 1, 1) * torch.cos(angle).view(1, 1, -1).expand(n, BEATS, PATCH),
                             torch.sin(angle).view(1, 1, -1).expand(n, BEATS, PATCH),
                             0.2 * torch.sin(2 * angle).view(1, 1, -1).expand(n, BEATS, PATCH)), dim=2)
        beats = beats + 0.05 * torch.randn(beats.shape, generator=generator)
        # Every record sits in its own random frame, so only the invariant part can help.
        turns = random_rotations(n, None, generator)
        beats = rotate_beats(beats, turns)
        return (beats, torch.full((n, BEATS), 85.0), torch.ones(n, BEATS), torch.ones(n),
                torch.zeros(n, BEATS, TOKEN_DIM), torch.full((n,), BEATS - 1, dtype=torch.long),
                torch.zeros(n, 128), labels)

    data = {"train": make(256), "val": make(64), "test": make(64)}
    model = FrameProbe(**_frozen_parts(), parts=("invariant",), embedding_dim=32, hidden=16)
    module.train(model, data, torch.device("cpu"),
                 {"batch_size": 32, "max_epochs": 10, "patience": 10}, seed=0)
    scores = module.evaluate(model, data["test"], torch.device("cpu"), 64)
    assert scores["per_label_auroc"][0] > 0.85, scores["per_label_auroc"]


def test_parts_are_the_documented_ones():
    assert PARTS == ("invariant", "equivariant")
