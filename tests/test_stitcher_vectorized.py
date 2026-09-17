"""VectorizedBeatStitcher against the author's BeatStitcher: outputs, gradients, and the switch."""

import argparse

import pytest
import torch

from lvcg.models.blocks.beat_modules import BeatStitcher
from lvcg.models.blocks.stitcher_vectorized import VectorizedBeatStitcher
from lvcg.models.lvcg import LVCG, base_beat_loss, beat_level_loss, temporal_loss
from lvcg.models.utils.loss import masked_reconstruction_loss
from lvcg.utils.config import Config, add_cli_overrides, apply_overrides

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _both(beats, rr, mask, target_len=1000, device="cpu"):
    beats = beats.to(device)
    a_in = beats.clone().requires_grad_(True)
    b_in = beats.clone().requires_grad_(True)
    a = BeatStitcher(beats.shape[-1], target_len)(a_in, rr.to(device), mask.to(device))
    b = VectorizedBeatStitcher(beats.shape[-1], target_len)(b_in, rr.to(device), mask.to(device))
    upstream = torch.randn(a.shape, generator=torch.Generator().manual_seed(9), dtype=a.dtype).to(device)
    (a * upstream).sum().backward()
    (b * upstream).sum().backward()
    return a, b, a_in.grad, b_in.grad


def _adversarial_batches(target_len, dtype):
    """Zero, negative and one-sample lengths, invalid beats, truncation, an empty record."""
    generator = torch.Generator().manual_seed(target_len)
    for _ in range(40):
        count = int(torch.randint(1, 22, (), generator=generator))
        beats = torch.randn(4, count, 3, 128, generator=generator, dtype=dtype)
        rr = torch.randint(-2, 160, (4, count), generator=generator).to(dtype)
        tiny = torch.rand(4, count, generator=generator) < 0.25
        rr[tiny] = torch.randint(0, 11, (int(tiny.sum()),), generator=generator).to(dtype)
        mask = torch.rand(4, count, generator=generator) > 0.2
        mask[0] = False
        yield beats, rr, mask


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("target_len", [64, 200, 1000])
def test_matches_the_loop_exactly_in_float64(device, target_len):
    """In double precision the two agree to rounding of 1e-14: the algebra is the same."""
    for beats, rr, mask in _adversarial_batches(target_len, torch.float64):
        a, b, grad_a, grad_b = _both(beats, rr, mask, target_len, device)
        torch.testing.assert_close(b, a, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(grad_b, grad_a, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_float32_differences_are_rounding_sized(device):
    """Float32 differences are rounding: the GPU interpolation kernel rounds its source
    coordinate its own way (outputs within 1e-5), and a beat sample's gradient sums many
    outputs in a different order (within 5e-5). The float64 test is the exact check."""
    for beats, rr, mask in _adversarial_batches(1000, torch.float32):
        a, b, grad_a, grad_b = _both(beats, rr, mask, 1000, device)
        torch.testing.assert_close(b, a, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(grad_b, grad_a, atol=5e-5, rtol=1e-3)


def test_one_sample_fades_follow_linspace():
    """linspace(0, 1, 1) is [0] and linspace(1, 0, 1) is [1]; the loop inherits both."""
    beats = torch.ones(1, 3, 3, 128)
    rr = torch.tensor([[5.0, 5.0, 5.0]])  # fade length max(1, int(0.5)) = 1
    mask = torch.ones(1, 3, dtype=torch.bool)
    a, b, _, _ = _both(beats, rr, mask, target_len=15)
    torch.testing.assert_close(b, a)
    # Beat 1 (samples 5..9): its first sample is zeroed, its last is left at 1.
    assert a[0, 0, 5] == 0 and a[0, 0, 9] == 1


def test_a_skipped_beat_still_counts_for_the_fade_rules():
    beats = torch.randn(1, 3, 3, 128)
    rr = torch.tensor([[40.0, 0.0, 40.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    a, b, _, _ = _both(beats, rr, mask, target_len=100)
    torch.testing.assert_close(b, a, atol=2e-6, rtol=1e-5)
    assert a[0, :, 80:].abs().max() == 0, "samples no beat reaches stay zero"


def test_the_switch_and_checkpoint_compatibility():
    base = LVCG(time_len=1000, lead_order="mimic", fs=100)
    fast = LVCG(time_len=1000, lead_order="mimic", fs=100, vectorized_stitcher=True)
    assert type(base.stitcher) is BeatStitcher
    assert type(fast.stitcher) is VectorizedBeatStitcher
    assert base.state_dict().keys() == fast.state_dict().keys()
    fast.load_state_dict(base.state_dict(), strict=True)


@pytest.mark.parametrize("value, expected", [(True, True), (False, False), ("true", True), ("false", False)])
def test_config_and_cli_select_the_stitcher(value, expected):
    cfg = Config(raw={"model": {"vectorized_stitcher": value}, "data": {"time_len": 1000, "fs": 100}})
    assert isinstance(LVCG.from_config(cfg).stitcher, VectorizedBeatStitcher) is expected
    parser = add_cli_overrides(argparse.ArgumentParser())
    args = parser.parse_args(["--config", "x.yaml", "--model.vectorized_stitcher", "true"])
    raw = apply_overrides(Config(raw={"model": {}}), args).raw
    assert raw["model"]["vectorized_stitcher"] == "true"
    default = apply_overrides(Config(raw={"model": {}}), parser.parse_args(["--config", "x.yaml"])).raw
    assert "vectorized_stitcher" not in default["model"]


@pytest.mark.parametrize("device", DEVICES)
def test_pretraining_losses_and_gradients_match(device):
    """The whole forward_train and the author's objective, with either stitcher."""
    # Double precision, so any difference beyond 1e-10 is a real divergence, not rounding.
    torch.manual_seed(0)
    base = LVCG(time_len=1000, lead_order="mimic", fs=100).to(device).double().train()
    fast = LVCG(time_len=1000, lead_order="mimic", fs=100, vectorized_stitcher=True)
    fast = fast.to(device).double().train()
    fast.load_state_dict(base.state_dict())
    ecg = torch.randn(6, 12, 1000, generator=torch.Generator().manual_seed(1), dtype=torch.float64) * 0.3
    ecg[:, 1, 40::85] += 6.0
    ecg = ecg.to(device)
    visible = torch.tensor([[0, 1, 6], [2, 7, 9], [1, 4, 11], [3, 5, 8], [0, 6, 10], [1, 2, 3]], device=device)
    mask = torch.ones(6, 12, dtype=torch.bool, device=device)
    mask[torch.arange(6, device=device)[:, None], visible] = False

    def run(model):
        # Dropout draws must be identical for both runs.
        torch.manual_seed(123)
        if device == "cuda":
            torch.cuda.manual_seed_all(123)
        out = model.forward_train(ecg, visible)
        loss = (
            masked_reconstruction_loss(out["recon"], ecg, mask)
            + 0.1 * temporal_loss(out["states_pred"], out["states_real"], out["beat_mask"])
            + beat_level_loss(out["V_hat_beats"], out["V_beats"], out["beat_mask_full"])
            + base_beat_loss(out["V_base_hat"], out["V_base"])
        )
        model.zero_grad()
        loss.backward()
        return out["recon"].detach(), loss.detach(), {n: p.grad for n, p in model.named_parameters()}

    recon_a, loss_a, grads_a = run(base)
    recon_b, loss_b, grads_b = run(fast)
    torch.testing.assert_close(recon_b, recon_a, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(loss_b, loss_a, atol=1e-10, rtol=1e-10)
    for name, grad in grads_a.items():
        if grad is None:
            assert grads_b[name] is None, name
            continue
        torch.testing.assert_close(grads_b[name], grad, atol=1e-10, rtol=1e-8, msg=name)
