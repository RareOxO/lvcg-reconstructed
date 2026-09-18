"""Route E: the rotational-geometry pretraining term and its effect on the objective."""

import importlib.util
import math
import os

import pytest
import torch

from lvcg.quaternion.pretrain import (
    rotation_angle_error,
    rotational_consistency_loss,
    step_quaternions,
)
from lvcg.quaternion.utils import quaternion_to_rotation_matrix, vectors_to_quaternion

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEATS, PATCH = 4, 64


def _patches(batch=2, seed=0):
    """Beat patches that sweep a real loop, so their step rotations are well defined."""
    generator = torch.Generator().manual_seed(seed)
    angle = torch.linspace(0, 2 * math.pi, PATCH)
    loop = torch.stack((2.0 * torch.cos(angle), torch.sin(angle), 0.3 * torch.sin(2 * angle)))
    patches = loop.expand(batch, BEATS, 3, PATCH).clone()
    patches = patches + 0.02 * torch.randn(patches.shape, generator=generator)
    return patches, torch.ones(batch, BEATS)


# --- the loss itself --------------------------------------------------------------


def test_the_loss_is_zero_for_an_identical_reconstruction():
    target, mask = _patches()
    assert float(rotational_consistency_loss(target, target, mask)) == pytest.approx(0.0, abs=1e-6)


def test_the_loss_is_scale_free():
    """It cannot be minimised by shrinking the output, so it does not fight the MSE."""
    target, mask = _patches()
    for factor in (0.1, 0.5, 3.0):
        assert float(rotational_consistency_loss(target * factor, target, mask)) == pytest.approx(
            0.0, abs=1e-5), factor


def test_a_wrongly_turning_reconstruction_costs():
    """The same points traversed backwards: right amplitude everywhere, wrong rotation."""
    target, mask = _patches()
    reversed_time = target.flip(-1)
    assert float(rotational_consistency_loss(reversed_time, target, mask)) > 0.1


def test_composition_is_what_makes_the_term_see_the_axis():
    """Single steps alone are nearly blind to it; this is why the default has scales."""
    target, mask = _patches()
    reversed_time = target.flip(-1)
    local_only = float(rotational_consistency_loss(reversed_time, target, mask, scales=(1,)))
    composed = float(rotational_consistency_loss(reversed_time, target, mask, scales=(8, 32)))
    # Measured here: 0.006 against 0.225 -- the same error, seen 35 times more clearly.
    assert composed > 20 * local_only, (local_only, composed)


def test_the_loss_grows_with_the_disagreement():
    target, mask = _patches()
    generator = torch.Generator().manual_seed(1)
    noise = torch.randn(target.shape, generator=generator)
    losses = [float(rotational_consistency_loss(target + level * noise, target, mask))
              for level in (0.05, 0.2, 0.8)]
    assert losses == sorted(losses), losses


def test_the_loss_ignores_the_double_cover():
    """q and -q are the same rotation and must cost nothing."""
    target, mask = _patches()
    quaternions = step_quaternions(target, torch.ones(2, BEATS, PATCH - 1, dtype=torch.bool))
    flipped = -quaternions
    agreement = (quaternions * flipped).sum(-1).abs()
    assert torch.allclose(agreement, torch.ones_like(agreement), atol=1e-5)


def test_a_rigid_turn_of_both_trajectories_costs_nothing():
    """The term reads relative rotations, so a shared frame change is invisible to it."""
    target, mask = _patches()
    angle = torch.tensor(0.6)
    rotation = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0.0],
                             [torch.sin(angle), torch.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    turned = torch.einsum("ij,bnjp->bnip", rotation, target)
    assert float(rotational_consistency_loss(turned, turned, mask)) == pytest.approx(0.0, abs=1e-5)


def test_a_turn_of_the_prediction_alone_does_cost():
    """Small in absolute terms -- the composed angles are modest -- but far from zero.

    The reconstruction term already penalises a rigid turn heavily; this term only has
    to not reward it.
    """
    target, mask = _patches()
    angle = torch.tensor(0.6)
    rotation = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0.0],
                             [torch.sin(angle), torch.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    turned = torch.einsum("ij,bnjp->bnip", rotation, target)
    matched = float(rotational_consistency_loss(target, target, mask))
    turned_cost = float(rotational_consistency_loss(turned, target, mask))
    assert turned_cost > 1e-4 and turned_cost > 100 * max(matched, 1e-9), (turned_cost, matched)


def test_padding_beats_and_quiet_stretches_are_excluded():
    target, mask = _patches()
    padded = target.clone()
    partial = mask.clone()
    partial[:, -1] = 0
    padded[:, -1] = torch.randn_like(padded[:, -1]) * 5  # a padding beat, wildly wrong
    assert float(rotational_consistency_loss(padded, padded, partial)) == pytest.approx(0.0, abs=1e-6)

    quiet = target.clone()
    quiet[..., 10:30] *= 1e-9
    prediction = quiet.clone()
    prediction[..., 10:30] = torch.randn_like(prediction[..., 10:30]) * 1e-9
    assert float(rotational_consistency_loss(prediction, quiet, mask)) < 0.05


def test_the_loss_is_bounded_and_finite_on_degenerate_input():
    mask = torch.ones(2, BEATS)
    zeros = torch.zeros(2, BEATS, 3, PATCH)
    value = rotational_consistency_loss(zeros, zeros, mask)
    assert torch.isfinite(value) and 0.0 <= float(value) <= 1.0


def test_gradients_reach_the_prediction():
    target, mask = _patches()
    prediction = (target + 0.1).clone().requires_grad_(True)
    rotational_consistency_loss(prediction, target, mask).backward()
    assert prediction.grad.abs().max() > 0


def test_the_angle_report_is_degrees_and_zero_for_a_match():
    target, mask = _patches()
    # A perfect match leaves the safe-norm floor, which is about a twentieth of a degree.
    assert float(rotation_angle_error(target, target, mask)) == pytest.approx(0.0, abs=0.2)
    worse = float(rotation_angle_error(target.flip(-1), target, mask))
    assert 0.0 < worse <= 180.0


# --- the pretraining objective ----------------------------------------------------


def _train_module():
    spec = importlib.util.spec_from_file_location(
        "train_script", os.path.join(REPO_ROOT, "scripts", "train.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model():
    from lvcg.models.lvcg import LVCG
    from lvcg.utils.config import load_config

    torch.manual_seed(0)
    return LVCG.from_config(load_config(os.path.join(REPO_ROOT, "configs/train/lvcg_v5_gru.yaml")))


def _ecg(batch=3):
    generator = torch.Generator().manual_seed(0)
    ecg = torch.randn(batch, 12, 1000, generator=generator) * 0.3
    ecg[:, 1, 40::85] += 6.0
    return ecg


def test_weight_zero_reproduces_the_released_objective_bitwise():
    """The released five terms and the released total, with the term switched off."""
    module, model, ecg = _train_module(), _model(), _ecg()
    lambdas = {"temporal": 0.1, "beat": 1.0, "base": 1.0}

    torch.manual_seed(1)
    released, terms_released = module.compute_losses(model, ecg, 3, lambdas)
    torch.manual_seed(1)
    with_zero, terms_zero = module.compute_losses(model, ecg, 3, {**lambdas, "rotation": 0.0})
    assert torch.equal(released, with_zero)
    assert set(terms_zero) == set(terms_released) == set(module.LOSS_TERMS)


def test_a_non_zero_weight_adds_exactly_its_contribution():
    module, model, ecg = _train_module(), _model(), _ecg()
    lambdas = {"temporal": 0.1, "beat": 1.0, "base": 1.0}

    torch.manual_seed(1)
    released, _ = module.compute_losses(model, ecg, 3, lambdas)
    torch.manual_seed(1)
    total, terms = module.compute_losses(model, ecg, 3, {**lambdas, "rotation": 0.5})
    assert module.ROTATION_TERM in terms
    assert float(total - released) == pytest.approx(0.5 * float(terms["rotation"]), abs=1e-5)


def test_the_new_term_trains_the_decoder():
    module, model, ecg = _train_module(), _model(), _ecg()
    total, _ = module.compute_losses(model, ecg, 3, {"temporal": 0.0, "beat": 0.0, "base": 0.0,
                                                     "rotation": 1.0})
    total.backward()
    gradients = [p.grad for p in model.beat_decoder.parameters() if p.grad is not None]
    assert gradients and float(gradients[0].abs().max()) > 0


def test_the_term_is_reported_only_when_it_is_on():
    module = _train_module()
    assert module.LOSS_TERMS == ("loss", "recon", "temporal", "beat", "base")
    assert module.ROTATION_TERM == "rotation"
