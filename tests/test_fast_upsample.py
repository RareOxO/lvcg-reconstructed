"""The fast_upsample switch: equal to F.interpolate, in the decoder and in a training step."""

import argparse

import pytest
import torch
import torch.nn.functional as F

from lvcg.models.blocks.beat_modules import BeatDecoder
from lvcg.models.blocks.fast_upsample import upsample_linear_x2
from lvcg.models.lvcg import LVCG, base_beat_loss, beat_level_loss, temporal_loss
from lvcg.models.utils.loss import masked_reconstruction_loss
from lvcg.utils.config import Config, add_cli_overrides, apply_overrides

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _both(x):
    a_in, b_in = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    a = F.interpolate(a_in, scale_factor=2, mode="linear", align_corners=False)
    b = upsample_linear_x2(b_in)
    upstream = torch.randn(a.shape, generator=torch.Generator().manual_seed(3), dtype=x.dtype).to(x.device)
    (a * upstream).sum().backward()
    (b * upstream).sum().backward()
    return a, b, a_in.grad, b_in.grad


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("length", [1, 2, 3, 4, 16, 127])
def test_the_operation_matches_interpolate_exactly_in_float64(device, length):
    x = torch.randn(5, 7, length, generator=torch.Generator().manual_seed(length), dtype=torch.float64).to(device)
    a, b, grad_a, grad_b = _both(x)
    assert b.shape == a.shape == (5, 7, 2 * length)
    torch.testing.assert_close(b, a, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(grad_b, grad_a, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_float32_differences_are_rounding(device):
    x = torch.randn(64, 32, 64, generator=torch.Generator().manual_seed(0)).to(device)
    a, b, grad_a, grad_b = _both(x)
    torch.testing.assert_close(b, a, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(grad_b, grad_a, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_the_decoder_matches_in_values_and_gradients(device):
    torch.manual_seed(0)
    slow = BeatDecoder(state_dim=256, beat_len=128, initial_channels=128,
                       hidden_channels=[128, 64, 64, 32, 32]).to(device).double().eval()
    fast = BeatDecoder(state_dim=256, beat_len=128, initial_channels=128,
                       hidden_channels=[128, 64, 64, 32, 32], fast_upsample=True).to(device).double().eval()
    fast.load_state_dict(slow.state_dict())
    z = torch.randn(3, 5, 256, generator=torch.Generator().manual_seed(1), dtype=torch.float64).to(device)
    out_slow, out_fast = slow(z), fast(z)
    torch.testing.assert_close(out_fast, out_slow, atol=1e-12, rtol=1e-12)
    out_slow.square().sum().backward()
    out_fast.square().sum().backward()
    for (name, a), b in zip(slow.named_parameters(), fast.parameters()):
        torch.testing.assert_close(b.grad, a.grad, atol=1e-10, rtol=1e-10, msg=name)


@pytest.mark.parametrize("device", DEVICES)
def test_a_pretraining_step_matches_with_both_speed_switches(device):
    torch.manual_seed(0)
    base = LVCG(time_len=1000, lead_order="mimic", fs=100).to(device).double().train()
    fast = LVCG(time_len=1000, lead_order="mimic", fs=100, vectorized_stitcher=True,
                fast_upsample=True).to(device).double().train()
    fast.load_state_dict(base.state_dict())
    ecg = torch.randn(4, 12, 1000, generator=torch.Generator().manual_seed(1), dtype=torch.float64) * 0.3
    ecg[:, 1, 40::85] += 6.0
    ecg = ecg.to(device)
    visible = torch.tensor([[0, 1, 6], [2, 7, 9], [1, 4, 11], [3, 5, 8]], device=device)
    mask = torch.ones(4, 12, dtype=torch.bool, device=device)
    mask[torch.arange(4, device=device)[:, None], visible] = False

    def run(model):
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
        return loss.detach(), {n: p.grad for n, p in model.named_parameters()}

    loss_a, grads_a = run(base)
    loss_b, grads_b = run(fast)
    torch.testing.assert_close(loss_b, loss_a, atol=1e-10, rtol=1e-10)
    for name, grad in grads_a.items():
        if grad is None:
            assert grads_b[name] is None, name
            continue
        torch.testing.assert_close(grads_b[name], grad, atol=1e-10, rtol=1e-8, msg=name)


def test_the_switch_config_cli_and_checkpoints():
    base = LVCG(time_len=1000, lead_order="mimic", fs=100)
    fast = LVCG.from_config(Config(raw={"model": {"fast_upsample": "true"}, "data": {"time_len": 1000, "fs": 100}}))
    assert not any(block.fast_upsample for block in base.beat_decoder.blocks)
    assert all(block.fast_upsample for block in fast.beat_decoder.blocks)
    fast.load_state_dict(base.state_dict(), strict=True)
    parser = add_cli_overrides(argparse.ArgumentParser())
    args = parser.parse_args(["--config", "x.yaml", "--model.fast_upsample", "true"])
    assert apply_overrides(Config(raw={"model": {}}), args).raw["model"]["fast_upsample"] == "true"
