"""V1 (QDF-LVCG): the quaternion utilities and the branch, per plan appendix B."""

import importlib.util
import os

import numpy as np
import pytest
import torch

from lvcg.quaternion.features import QuaternionDynamicFeatures, check_features
from lvcg.quaternion.qdf import QDFProbe
from lvcg.quaternion.utils import (
    enforce_sign_continuity,
    quaternion_angle,
    quaternion_geodesic_distance,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    rotate_vector_by_quaternion,
    valid_rotation_mask,
    vectors_to_quaternion,
)

IDENTITY = torch.tensor([1.0, 0.0, 0.0, 0.0])
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _random_unit(shape, seed=0):
    generator = torch.Generator().manual_seed(seed)
    v = torch.randn(*shape, 3, generator=generator, dtype=torch.float64)
    return v / v.norm(dim=-1, keepdim=True)


# --- appendix B.1-B.8: the quaternion utilities -----------------------------------


def test_identity_quaternion_leaves_a_vector_alone():
    v = _random_unit((16,))
    assert torch.allclose(rotate_vector_by_quaternion(v, IDENTITY.double().expand(16, 4)), v)


def test_rotation_preserves_norm():
    v = torch.randn(32, 3, dtype=torch.float64) * 5
    q = vectors_to_quaternion(_random_unit((32,), 1), _random_unit((32,), 2))
    assert torch.allclose(rotate_vector_by_quaternion(v, q).norm(dim=-1), v.norm(dim=-1))


def test_rotation_matrix_is_orthogonal_with_unit_determinant():
    q = vectors_to_quaternion(_random_unit((24,), 3), _random_unit((24,), 4))
    R = quaternion_to_rotation_matrix(q)
    identity = torch.eye(3, dtype=torch.float64).expand(24, 3, 3)
    # The tolerance is 1e-9 because normalisation keeps a small positive floor
    # inside the square root, which perturbs R by about 1e-12.
    assert torch.allclose(R.transpose(-1, -2) @ R, identity, atol=1e-9)
    assert torch.allclose(torch.linalg.det(R), torch.ones(24, dtype=torch.float64))


def test_q_and_minus_q_are_the_same_rotation():
    q = vectors_to_quaternion(_random_unit((10,), 5), _random_unit((10,), 6))
    assert torch.allclose(quaternion_to_rotation_matrix(q), quaternion_to_rotation_matrix(-q))
    assert torch.allclose(quaternion_geodesic_distance(q, -q), torch.zeros(10, dtype=torch.float64), atol=1e-5)
    assert torch.allclose(quaternion_angle(q), quaternion_angle(-q))


def test_vectors_to_quaternion_rotates_the_first_onto_the_second():
    u, w = _random_unit((64,), 7), _random_unit((64,), 8)
    assert torch.allclose(rotate_vector_by_quaternion(u, vectors_to_quaternion(u, w)), w, atol=1e-10)


def test_parallel_vectors_give_the_identity_rotation():
    u = _random_unit((8,), 9)
    q = vectors_to_quaternion(u, u * 3.0)  # same direction, different magnitude
    assert torch.allclose(q.abs(), IDENTITY.double().expand(8, 4), atol=1e-8)
    # The safe norm's floor leaves an angle of about 2e-6 rather than exactly 0.
    assert torch.allclose(quaternion_angle(q), torch.zeros(8, dtype=torch.float64), atol=1e-5)


def test_antiparallel_vectors_use_a_deterministic_finite_fallback():
    u = _random_unit((8,), 10)
    q = vectors_to_quaternion(u, -u)
    assert torch.isfinite(q).all()
    assert torch.allclose(q, vectors_to_quaternion(u, -u)), "fallback must be deterministic"
    # A half turn, and its axis is perpendicular to the vector it turns.
    assert torch.allclose(quaternion_angle(q), torch.full((8,), float(np.pi), dtype=torch.float64), atol=1e-5)
    assert torch.allclose((q[..., 1:] * u).sum(-1), torch.zeros(8, dtype=torch.float64), atol=1e-8)


def test_near_zero_vectors_are_masked_and_stay_finite():
    p = torch.randn(2, 50, 3, dtype=torch.float64)
    p[:, 10:20] *= 1e-9  # a quiet stretch
    mask = valid_rotation_mask(p, lag=1, min_fraction=0.02)
    assert not mask[:, 10:19].any() and mask.sum() > 0
    features, returned = QuaternionDynamicFeatures(("q", "theta", "omega", "magnitude"), dt=0.01)(
        p.transpose(1, 2).float()
    )
    assert torch.isfinite(features).all()
    assert returned.shape == (2, 49)


def test_sign_continuity_removes_jumps_without_changing_the_rotation():
    q = vectors_to_quaternion(_random_unit((3, 40), 11)[:, :-1], _random_unit((3, 40), 11)[:, 1:])
    flipped = q.clone()
    flipped[:, ::2] *= -1
    continuous = enforce_sign_continuity(flipped)
    assert ((continuous[:, 1:] * continuous[:, :-1]).sum(-1) >= -1e-12).all()
    assert torch.allclose(quaternion_to_rotation_matrix(continuous), quaternion_to_rotation_matrix(q))


def test_quaternion_multiply_matches_matrix_composition():
    q1 = vectors_to_quaternion(_random_unit((12,), 12), _random_unit((12,), 13))
    q2 = vectors_to_quaternion(_random_unit((12,), 14), _random_unit((12,), 15))
    composed = quaternion_to_rotation_matrix(quaternion_multiply(q1, q2))
    assert torch.allclose(composed, quaternion_to_rotation_matrix(q1) @ quaternion_to_rotation_matrix(q2), atol=1e-12)


# --- the V1 branch ----------------------------------------------------------------


def test_feature_sets_are_quaternion_or_control_but_not_both():
    check_features(("q", "theta", "omega", "magnitude"))
    check_features(("position", "next_position", "delta"))
    with pytest.raises(ValueError):
        check_features(("q", "delta"))
    with pytest.raises(ValueError):
        check_features(())


def test_forward_is_finite_and_has_the_ptbxl_shape():
    model = QDFProbe().eval()
    logits = model(torch.randn(4, 640), torch.randn(4, 3, 1000))
    assert logits.shape == (4, 5) and torch.isfinite(logits).all()


def test_scale_zero_reproduces_v0(monkeypatch):
    """Plan 7.1 Level 0: with the quaternion contribution off, the model is V0's probe."""
    model = QDFProbe(scale=0.0).eval()
    base, vcg = torch.randn(8, 640), torch.randn(8, 3, 1000)
    v0 = model.head(torch.cat((base, torch.zeros(8, 128)), dim=-1))
    assert torch.equal(model(base, vcg), v0)
    # And the branch cannot receive gradients through the head.
    model(base, vcg).sum().backward()
    assert model.encoder.net[1].weight.grad.abs().max() == 0


def test_gradients_reach_the_branch_when_it_is_on():
    model = QDFProbe(scale=1.0)
    model(torch.randn(8, 640), torch.randn(8, 3, 1000)).sum().backward()
    assert model.encoder.net[1].weight.grad.abs().max() > 0


def test_masked_pooling_ignores_the_quiet_stretch():
    model = QDFProbe(masked_pooling=True).eval()
    vcg = torch.randn(2, 3, 1000)
    quiet = vcg.clone()
    quiet[:, :, 500:] *= 1e-9  # masked out, so the embedding must not follow it
    loud = vcg.clone()
    loud[:, :, 500:] *= 1e-9
    loud[:, 0, 500:] += 1e-9
    assert torch.allclose(model.quaternion_embedding(quiet), model.quaternion_embedding(loud), atol=1e-5)


def test_control_branch_is_parameter_matched_within_a_percent():
    quaternion = QDFProbe(features=("q", "theta", "omega", "magnitude"))
    control = QDFProbe(features=("position", "next_position", "delta"))
    q_params = quaternion.parameter_counts()["trainable_total"]
    c_params = control.parameter_counts()["trainable_total"]
    assert abs(q_params - c_params) / q_params < 0.01, (q_params, c_params)


def test_training_loop_learns_a_signal_that_lives_only_in_the_rotation():
    """The branch must fit a label carried by the VCG's rotation and nothing else.

    Only label 0 is learnable here -- the base embedding is all zeros and the other four
    labels are noise -- so the check is on that label's AUROC, not the macro average.
    """
    spec = importlib.util.spec_from_file_location(
        "train_qdf", os.path.join(REPO_ROOT, "scripts", "train_qdf.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    generator = torch.Generator().manual_seed(0)
    def make(n):
        labels = (torch.rand(n, 5, generator=generator) > 0.5).float()
        vcg = torch.randn(n, 3, 200, generator=generator) * 0.2
        # Class 0 rotates the cardiac vector around z, the others do not.
        angle = torch.linspace(0, 6.0, 200) * labels[:, :1]
        vcg[:, 0] += torch.cos(angle)
        vcg[:, 1] += torch.sin(angle)
        return torch.zeros(n, 8), vcg, labels

    data = {"train": make(512), "val": make(128), "test": make(128)}
    model = QDFProbe(base_dim=8, fs=100, embedding_dim=16, hidden=8)
    module.train(model, data, torch.device("cpu"),
                 {"batch_size": 32, "max_epochs": 12, "patience": 12}, seed=0)
    scores = module.evaluate(model, data["val"], torch.device("cpu"), 64)
    assert scores["per_label_auroc"][0] > 0.85, scores["per_label_auroc"]
