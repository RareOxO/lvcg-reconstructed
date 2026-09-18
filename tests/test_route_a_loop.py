"""Route A: ordered rotation composition, its multi-scale windows and the signature test."""

import pytest
import torch

from lvcg.models.lvcg import StateGRU
from lvcg.models.blocks.beat_modules import BeatEncoder
from lvcg.quaternion.loop import (
    LoopProbe,
    RotationalLoopEncoder,
    beat_directions,
    multi_scale_rotations,
    perturb_rotation_order,
    prefix_composition,
    step_rotations,
    window_rotations,
)
from lvcg.quaternion.utils import (
    quaternion_angle,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    rotate_vector_by_quaternion,
)

TOKEN_DIM, BEATS, PATCH = 256, 4, 64


def _beats(batch=2, seed=0, scale=1.0):
    generator = torch.Generator().manual_seed(seed)
    beats = torch.randn(batch, BEATS, 3, PATCH, generator=generator) * scale
    mask = torch.ones(batch, BEATS)
    reference = beats.transpose(-1, -2).norm(dim=-1).reshape(batch, -1).quantile(0.99, dim=-1)
    return beats, mask, reference


def _frozen_parts():
    torch.manual_seed(0)
    return dict(
        state_generator=StateGRU(state_dim=TOKEN_DIM, hidden_dim=TOKEN_DIM, num_layers=2, dropout=0.1),
        norm_struct=torch.nn.LayerNorm(TOKEN_DIM),
        norm_dynamic=torch.nn.LayerNorm(TOKEN_DIM),
        beat_encoder=BeatEncoder(beat_len=PATCH, state_dim=TOKEN_DIM),
    )


# --- the composition --------------------------------------------------------------


def test_prefix_composition_matches_a_sequential_product():
    """The doubling scan must give exactly the ordered product, not a reordered one."""
    torch.manual_seed(0)
    q = torch.randn(3, 17, 4, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    prefix = prefix_composition(q)

    # The plan's convention: the later rotation stands on the left.
    expected = q[:, 0]
    assert torch.allclose(prefix[:, 0], expected, atol=1e-12)
    for t in range(1, q.shape[1]):
        expected = quaternion_multiply(q[:, t], expected)
        assert torch.allclose(prefix[:, t], expected, atol=1e-10), t


def test_composition_is_not_commutative_so_order_matters():
    torch.manual_seed(1)
    q = torch.randn(1, 6, 4, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    forward = prefix_composition(q)[:, -1]
    backward = prefix_composition(q.flip(1))[:, -1]
    assert not torch.allclose(forward, backward, atol=1e-6), "a reordering must change the product"


def test_window_rotation_is_the_product_of_that_window():
    torch.manual_seed(2)
    q = torch.randn(2, 20, 4, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    prefix = prefix_composition(q)
    scale = 5
    windows = window_rotations(prefix, scale)
    for t in range(scale, q.shape[1]):
        expected = q[:, t - scale + 1]
        for k in range(t - scale + 2, t + 1):
            expected = quaternion_multiply(q[:, k], expected)
        assert torch.allclose(windows[:, t].abs(), expected.abs(), atol=1e-8), t


def test_whole_beat_scale_is_the_prefix_itself():
    torch.manual_seed(3)
    q = torch.randn(1, 9, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    prefix = prefix_composition(q)
    assert torch.equal(window_rotations(prefix, 0), prefix)


def test_composed_rotation_takes_the_first_direction_to_the_last():
    """The point of composition: one rotation summarising the whole stretch."""
    beats, mask, reference = _beats()
    directions, step_mask = beat_directions(beats, mask, reference)
    steps = step_rotations(directions, step_mask)
    total = prefix_composition(steps)[:, :, -1]  # [B, N, 4]
    turned = rotate_vector_by_quaternion(directions[:, :, 0], total)
    assert torch.allclose(turned, directions[:, :, -1], atol=1e-4)


def test_masked_steps_compose_as_the_identity():
    beats, mask, reference = _beats()
    quiet = beats.clone()
    quiet[:, :, :, 20:40] *= 1e-9  # a stretch with no reliable direction
    directions, step_mask = beat_directions(quiet, mask, reference)
    assert not step_mask[:, :, 20:38].any()
    steps = step_rotations(directions, step_mask)
    assert torch.allclose(quaternion_angle(steps[:, :, 20:38]),
                          torch.zeros_like(quaternion_angle(steps[:, :, 20:38])), atol=1e-5)


def test_multi_scale_features_have_five_channels_per_scale():
    beats, mask, reference = _beats()
    directions, step_mask = beat_directions(beats, mask, reference)
    features = multi_scale_rotations(directions, step_mask, scales=(4, 16, 0))
    assert features.shape == (2, BEATS, PATCH - 1, 15)
    assert torch.isfinite(features).all()


# --- the signature perturbation ---------------------------------------------------


@pytest.mark.parametrize("mode", ["shuffle", "block", "reverse"])
def test_perturbation_keeps_every_step_rotation_size(mode):
    """Only the order may change: the multiset of step rotations must be preserved."""
    beats, mask, reference = _beats(seed=4)
    generator = torch.Generator().manual_seed(0)
    perturbed = perturb_rotation_order(beats, mask, reference, mode, generator=generator)

    def angles(trajectory):
        directions, step_mask = beat_directions(trajectory, mask, reference)
        return quaternion_angle(step_rotations(directions, step_mask)).sort(dim=-1).values

    before, after = angles(beats), angles(perturbed)
    assert torch.allclose(before, after, atol=1e-3), float((before - after).abs().max())


def test_perturbation_keeps_the_magnitude_profile():
    beats, mask, reference = _beats(seed=5)
    perturbed = perturb_rotation_order(beats, mask, reference, "shuffle",
                                       generator=torch.Generator().manual_seed(0))
    assert torch.allclose(beats.transpose(-1, -2).norm(dim=-1),
                          perturbed.transpose(-1, -2).norm(dim=-1), atol=1e-4)


def test_perturbation_actually_changes_the_trajectory():
    beats, mask, reference = _beats(seed=6)
    for mode in ("shuffle", "block", "reverse"):
        perturbed = perturb_rotation_order(beats, mask, reference, mode,
                                           generator=torch.Generator().manual_seed(0))
        assert not torch.allclose(beats, perturbed, atol=1e-3), mode


def test_perturbation_is_deterministic_given_a_generator():
    beats, mask, reference = _beats(seed=7)
    first = perturb_rotation_order(beats, mask, reference, "shuffle",
                                   generator=torch.Generator().manual_seed(3))
    second = perturb_rotation_order(beats, mask, reference, "shuffle",
                                    generator=torch.Generator().manual_seed(3))
    assert torch.equal(first, second)


def test_unknown_perturbation_is_refused():
    beats, mask, reference = _beats()
    with pytest.raises(ValueError):
        perturb_rotation_order(beats, mask, reference, "jitter")


# --- the encoder and the probe ----------------------------------------------------


def test_encoder_pools_over_valid_beats_only():
    encoder = RotationalLoopEncoder(scales=(4, 0)).eval()
    beats, mask, reference = _beats(seed=8)
    padded = beats.clone()
    partial = mask.clone()
    partial[:, -1] = 0
    padded[:, -1] = torch.randn_like(padded[:, -1]) * 5
    assert torch.allclose(encoder(beats[:, :-1], partial[:, :-1], reference),
                          encoder(padded, partial, reference), atol=1e-4)


def test_local_only_removes_the_composition():
    local = RotationalLoopEncoder(local_only=True)
    multi = RotationalLoopEncoder(scales=(8, 32, 0))
    assert local.scales == (1,) and local.channels == 6
    assert multi.channels == 16, "three scales of (quaternion, angle) plus the mask"


@pytest.mark.parametrize("struct", list(LoopProbe.STRUCTS))
def test_probe_runs_in_both_structural_conventions(struct):
    model = LoopProbe(**_frozen_parts(), struct=struct).eval()
    beats, mask, reference = _beats()
    tokens = torch.randn(2, BEATS, TOKEN_DIM)
    steps = torch.full((2,), BEATS - 1, dtype=torch.long)
    logits = model(beats, tokens, mask, reference, steps, torch.randn(2, 128))
    assert logits.shape == (2, 5) and torch.isfinite(logits).all()


def test_mean_and_anchor_structures_differ():
    beats, mask, reference = _beats()
    tokens = torch.randn(2, BEATS, TOKEN_DIM)
    steps = torch.full((2,), BEATS - 1, dtype=torch.long)
    parts = _frozen_parts()
    mean = LoopProbe(**parts, struct="mean").eval()
    anchor = LoopProbe(**parts, struct="anchor").eval()
    anchor.load_state_dict(mean.state_dict())
    arguments = (beats, tokens, mask, reference, steps, torch.randn(2, 128))
    assert not torch.allclose(mean(*arguments), anchor(*arguments), atol=1e-4)


def test_scale_zero_ignores_the_loop_branch():
    model = LoopProbe(**_frozen_parts(), scale=0.0).eval()
    beats, mask, reference = _beats()
    tokens = torch.randn(2, BEATS, TOKEN_DIM)
    steps = torch.full((2,), BEATS - 1, dtype=torch.long)
    rhythm = torch.randn(2, 128)
    other, _, _ = _beats(seed=9)
    assert torch.equal(model(beats, tokens, mask, reference, steps, rhythm),
                       model(other, tokens, mask, reference, steps, rhythm))


def test_recomputing_tokens_matches_the_cached_ones():
    parts = _frozen_parts()
    model = LoopProbe(**parts, struct="mean").eval()
    beats, mask, reference = _beats()
    tokens = parts["beat_encoder"](beats).detach()
    steps = torch.full((2,), BEATS - 1, dtype=torch.long)
    rhythm = torch.randn(2, 128)
    cached = model(beats, tokens, mask, reference, steps, rhythm)
    recomputed = model(beats, tokens, mask, reference, steps, rhythm, recompute=True)
    assert torch.allclose(cached, recomputed, atol=1e-5)


def test_perturbation_reaches_the_frozen_path_when_tokens_are_recomputed():
    """Otherwise the signature test would compare the branch against a constant."""
    parts = _frozen_parts()
    model = LoopProbe(**parts, struct="mean").eval()
    beats, mask, reference = _beats()
    tokens = parts["beat_encoder"](beats).detach()
    steps = torch.full((2,), BEATS - 1, dtype=torch.long)
    rhythm = torch.randn(2, 128)
    perturbed = perturb_rotation_order(beats, mask, reference, "shuffle",
                                       generator=torch.Generator().manual_seed(0))
    base_before, loop_before = model.embeddings(beats, tokens, mask, reference, steps, rhythm, recompute=True)
    base_after, loop_after = model.embeddings(perturbed, tokens, mask, reference, steps, rhythm, recompute=True)
    assert float((base_after - base_before).norm()) > 0, "the frozen path must see the perturbation"
    assert float((loop_after - loop_before).norm()) > 0


def test_only_the_branch_and_the_head_train():
    model = LoopProbe(**_frozen_parts())
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert all(name.startswith(("loop", "head")) for name in trainable), sorted(trainable)[:5]
    beats, mask, reference = _beats()
    model.train()
    model(beats, torch.randn(2, BEATS, TOKEN_DIM), mask, reference,
          torch.full((2,), BEATS - 1, dtype=torch.long), torch.randn(2, 128)).sum().backward()
    assert model.loop.encoder.net[1].weight.grad.abs().max() > 0
    assert model.state_generator.gru.weight_ih_l0.grad is None


def test_unknown_structure_is_refused():
    with pytest.raises(ValueError):
        LoopProbe(**_frozen_parts(), struct="first")
