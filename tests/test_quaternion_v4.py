"""V4 (Phase-Q LVCG): R-peak relative phase windows and the per-phase pooling."""

import math

import pytest
import torch

from lvcg.quaternion.mrq import build_probe
from lvcg.quaternion.phase import DEFAULT_WINDOWS, PHASES, PhaseProbe, phase_masks

FS, STEPS = 100, 999


def _peaks(positions, total=8):
    peaks = torch.zeros(1, total, dtype=torch.long)
    mask = torch.zeros(1, total, dtype=torch.bool)
    peaks[0, : len(positions)] = torch.tensor(positions)
    mask[0, : len(positions)] = True
    return peaks, mask


def _batch(n=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(n, 640, generator=generator)
    vcg = torch.randn(n, 3, 1000, generator=generator)
    peaks, mask = _peaks([100, 185, 270, 355, 440, 525, 610, 695])
    return base, vcg, peaks.expand(n, -1).contiguous(), mask.expand(n, -1).contiguous()


# --- the windows ------------------------------------------------------------------


def test_qrs_window_is_the_configured_span_around_every_r_peak():
    peaks, mask = _peaks([200, 500])
    masks = phase_masks(peaks, mask, STEPS, FS)
    qrs = masks["qrs"][0]
    # -40 ms .. +60 ms at 100 Hz is samples -4 .. +6 inclusive: 11 samples per peak.
    assert qrs[196:207].all() and not qrs[195] and not qrs[207]
    assert qrs[496:507].all() and qrs.sum() == 22


def test_t_window_follows_the_heart_rate():
    """The T window is a fraction of the interval, so it moves with the R-R length."""
    fast, mask = _peaks([100, 160, 220])   # 0.6 s intervals
    slow, _ = _peaks([100, 250, 400])      # 1.5 s intervals
    start, end = DEFAULT_WINDOWS["t_fraction"]
    fast_mask = phase_masks(fast, mask, STEPS, FS)["t"][0]
    slow_mask = phase_masks(slow, mask, STEPS, FS)["t"][0]
    # The bounds are fractions of the interval, so the first covered sample is the
    # first integer at or beyond start * rr.
    assert fast_mask[100 + math.ceil(start * 60):100 + math.floor(end * 60) + 1].all()
    assert slow_mask[100 + math.ceil(start * 150):100 + math.floor(end * 150) + 1].all()
    assert slow_mask.sum() > fast_mask.sum(), "a slower heart gives a longer T window"


def test_padding_peaks_are_ignored():
    peaks, mask = _peaks([200])
    padded = peaks.clone()
    padded[0, 3] = 700  # a position that is not a real peak
    assert torch.equal(phase_masks(peaks, mask, STEPS, FS)["qrs"],
                       phase_masks(padded, mask, STEPS, FS)["qrs"])


def test_whole_covers_everything_and_the_phases_do_not():
    peaks, mask = _peaks([100, 185, 270, 355])
    masks = phase_masks(peaks, mask, STEPS, FS)
    assert masks["whole"].all()
    assert 0 < masks["qrs"].float().mean() < 0.2
    assert 0 < masks["t"].float().mean() < 0.5


def test_windows_are_configurable():
    peaks, mask = _peaks([300])
    wide = phase_masks(peaks, mask, STEPS, FS, {"qrs_ms": (-100.0, 100.0)})["qrs"]
    assert wide.sum() == 21 and wide[200:221].all()


# --- the probe --------------------------------------------------------------------


@pytest.mark.parametrize("phases", [("whole",), ("qrs",), ("t",), ("qrs", "t"), ("qrs", "t", "whole")])
def test_every_phase_ablation_runs_and_stays_finite(phases):
    logits = PhaseProbe(phases=phases).eval()(*_batch())
    assert logits.shape == (2, 5) and torch.isfinite(logits).all()


def test_whole_only_reproduces_v3_tensor_for_tensor():
    """Plan equivalence: with one whole-beat window, V4 is V3's variant of the same name."""
    base, vcg, peaks, mask = _batch()
    for variant in ("mq", "mo"):
        torch.manual_seed(0)
        v4 = PhaseProbe(variant=variant, phases=("whole",)).eval()
        torch.manual_seed(0)
        v3 = build_probe(variant).eval()
        assert torch.equal(v4(base, vcg, peaks, mask), v3(base, vcg)), variant


def test_scale_zero_reproduces_v0():
    model = PhaseProbe(phases=("qrs", "t", "whole"), scale=0.0).eval()
    base, vcg, peaks, mask = _batch()
    zeros = torch.zeros(base.shape[0], 128 * 3)
    assert torch.equal(model(base, vcg, peaks, mask),
                       model.head(torch.cat((base, zeros), dim=-1)))


def test_a_change_inside_qrs_moves_the_qrs_embedding_far_more_than_the_t_one():
    """Phases separate where the encoder pools, not what its receptive field covers.

    A change inside the QRS window still reaches the T embedding: the convolution stack
    spans about 43 samples, and at a normal heart rate the T window opens only a few
    samples after QRS closes. Measured across initialisations, QRS moves 2.3 to 3.2 times
    as much as T. What must hold is that the phase the change belongs to moves clearly
    more, not that the two are independent.
    """
    torch.manual_seed(0)
    model = PhaseProbe(phases=("qrs", "t"), variant="mo").eval()
    base, vcg, peaks, mask = _batch()
    before_qrs, before_t = model.phase_embeddings(vcg, peaks, mask)

    changed = vcg.clone()
    changed[:, :, 96:107] *= 5.0  # inside the first QRS window only
    after_qrs, after_t = model.phase_embeddings(changed, peaks, mask)
    moved_qrs = (after_qrs - before_qrs).norm()
    moved_t = (after_t - before_t).norm()
    assert moved_qrs > 2 * moved_t, (float(moved_qrs), float(moved_t))


def test_phase_coverage_reports_the_overlap():
    model = PhaseProbe()
    peaks, mask = _peaks([100, 185, 270, 355])
    coverage = model.phase_coverage(peaks, mask, STEPS)
    assert coverage["whole"] == 1.0
    assert coverage["qrs_and_t"] <= min(coverage["qrs"], coverage["t"])


def test_more_phases_cost_only_their_share_of_the_head():
    one = PhaseProbe(phases=("whole",)).parameter_counts()["trainable_total"]
    three = PhaseProbe(phases=("qrs", "t", "whole")).parameter_counts()["trainable_total"]
    assert three - one == 2 * 128 * 5, "one shared encoder, one head slice per phase"


def test_gradients_reach_the_shared_encoder_from_every_phase():
    for phases in (("qrs",), ("t",), ("qrs", "t", "whole")):
        model = PhaseProbe(phases=phases)
        model(*_batch()).sum().backward()
        assert model.encoder.net[1].weight.grad.abs().max() > 0, phases


def test_unknown_phase_or_variant_is_refused():
    with pytest.raises(ValueError):
        PhaseProbe(phases=("pwave",))
    with pytest.raises(ValueError):
        PhaseProbe(phases=())
    with pytest.raises(ValueError):
        PhaseProbe(variant="curvature")


def test_a_record_without_peaks_stays_finite():
    """Detection can fail; an empty phase pools to zero rather than to NaN."""
    model = PhaseProbe(phases=("qrs", "t")).eval()
    base, vcg, peaks, mask = _batch()
    logits = model(base, vcg, torch.zeros_like(peaks), torch.zeros_like(mask))
    assert torch.isfinite(logits).all()


def test_phase_names_are_the_documented_ones():
    assert PHASES == ("qrs", "t", "whole")
