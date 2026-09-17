"""The reconstructed lvcg/data package.

The release shipped without it, so these tests hold the reconstruction to the two
things that define it: the paper (Table 7 geometry, Appendix A.4 segmentation) and the
contract the author's own model code expects -- above all that the author's
BeatStitcher can put the segmented beats back together.
"""

import numpy as np
import pytest
import torch

from lvcg.data import BeatSegmenter, get_lead_directions, reorder_leads
from lvcg.data.angle import (
    LEAD_DIRECTIONS_PTBXL,
    LEAD_NAMES,
    compute_lead_directions,
    compute_lead_directions_np,
    directions_to_angles,
    get_lead_angles,
)
from lvcg.models import LVCG
from lvcg.models.blocks.beat_modules import BeatStitcher
from lvcg.models.vcg import VCGPseudoInverse

RATE, LENGTH = 100, 1000


# --- Lead geometry ---


def test_table7_directions_are_unit_and_span_three_dimensions():
    assert LEAD_DIRECTIONS_PTBXL.shape == (12, 3)
    np.testing.assert_allclose(np.linalg.norm(LEAD_DIRECTIONS_PTBXL, axis=1), 1.0, atol=1e-5)
    assert np.linalg.matrix_rank(LEAD_DIRECTIONS_PTBXL) == 3


def test_table7_matches_the_stated_hexaxial_and_precordial_angles():
    """Limb leads at Einthoven angles; precordials on a ring tilted 10 degrees inferior."""
    limb = [0, 60, 120, -150, -30, 90]
    precordial = [110, 70, 40, 10, -20, -50]
    tilt = np.radians(10)
    expected = [(np.cos(np.radians(a)), np.sin(np.radians(a)), 0.0) for a in limb] + [
        (np.cos(tilt) * np.cos(np.radians(b)), np.sin(tilt), np.cos(tilt) * np.sin(np.radians(b)))
        for b in precordial
    ]
    np.testing.assert_allclose(LEAD_DIRECTIONS_PTBXL, expected, atol=1e-5)


def test_angles_round_trip_through_both_implementations():
    # Table 7 is published to five decimals, so its rows are unit only to ~5e-6; the
    # round trip returns exact unit vectors and is compared at that precision.
    theta, phi = directions_to_angles(LEAD_DIRECTIONS_PTBXL)
    np.testing.assert_allclose(
        compute_lead_directions_np(theta, phi), LEAD_DIRECTIONS_PTBXL, atol=1e-5
    )
    tensor = compute_lead_directions(torch.as_tensor(theta), torch.as_tensor(phi))
    np.testing.assert_allclose(tensor.numpy(), LEAD_DIRECTIONS_PTBXL, atol=1e-5)
    assert get_lead_angles("ptbxl").shape == (12, 2)


def test_mimic_order_is_the_native_mimic_header_order():
    """MIMIC-IV-ECG headers list aVF before aVL; the release's probing loader agrees."""
    from lvcg.data import LEAD_ORDERS

    assert LEAD_ORDERS["mimic"][3:6] == ("aVR", "aVF", "aVL")
    assert LEAD_ORDERS["ptbxl"][3:6] == ("aVR", "aVL", "aVF")
    mimic = get_lead_directions("mimic")
    np.testing.assert_array_equal(mimic[4], LEAD_DIRECTIONS_PTBXL[5])  # aVF
    np.testing.assert_array_equal(mimic[5], LEAD_DIRECTIONS_PTBXL[4])  # aVL
    # The probing loader's swap of indices 4 and 5 turns PTB-XL order into this one.
    swapped = torch.arange(12)[[0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11]]
    torch.testing.assert_close(reorder_leads(torch.arange(12)[None, :, None], "ptbxl", "mimic")[0, :, 0], swapped)


def test_directions_are_looked_up_by_lead_name():
    shuffled = ["V6", "I", "aVF", "II"]
    rows = get_lead_directions(shuffled)
    for row, name in zip(rows, shuffled):
        np.testing.assert_array_equal(row, LEAD_DIRECTIONS_PTBXL[LEAD_NAMES.index(name)])
    with pytest.raises(ValueError):
        get_lead_directions(["I", "not-a-lead"])


def test_reorder_leads_round_trips():
    ecg = torch.randn(2, 12, 50)
    target = list(reversed(LEAD_NAMES))
    there = reorder_leads(ecg, "ptbxl", target)
    torch.testing.assert_close(reorder_leads(there, target, "ptbxl"), ecg)


def test_the_lift_recovers_a_planted_cardiac_vector():
    """E = A v, so the pseudo-inverse of all twelve rows gives v back."""
    directions = get_lead_directions("ptbxl", as_tensor=True)
    vcg = torch.randn(2, 3, 200, dtype=torch.float64)
    ecg = torch.einsum("lc,bct->blt", directions.double(), vcg)
    lift = VCGPseudoInverse(eps=1e-9)
    recovered = lift(ecg, directions.double().expand(2, -1, -1))
    torch.testing.assert_close(recovered, vcg, atol=1e-6, rtol=0)


# --- Beat segmentation ---


def _reference_lead(peaks, height=5.0, length=LENGTH):
    lead = np.zeros(length)
    for peak in peaks:
        lead[peak - 1 : peak + 2] = [height / 2, height, height / 2]
    return lead


def _batch(leads, vcg=None):
    leads = np.stack(leads)
    ecg = torch.zeros(len(leads), 12, leads.shape[1])
    ecg[:, 1] = torch.as_tensor(leads, dtype=torch.float32)
    if vcg is None:
        vcg = torch.randn(len(leads), 3, leads.shape[1])
    return vcg, ecg


def test_detector_finds_planted_peaks():
    planted = np.arange(80, 1000, 80)
    found = BeatSegmenter(fs=RATE)._detect_r_peaks(_reference_lead(planted))
    np.testing.assert_array_equal(found, planted)


def test_intervals_partition_the_record_starting_at_zero():
    """The author's stitcher lays valid beats end to end from sample 0."""
    planted = np.arange(95, 1000, 90)
    vcg, ecg = _batch([_reference_lead(planted)])
    beats, rr, mask = BeatSegmenter(fs=RATE)(vcg, ecg, rr_lead_idx=1)
    assert mask.dtype == torch.bool and rr.dtype == vcg.dtype
    assert rr[mask].sum().item() == LENGTH
    assert rr[0, 0].item() == planted[0], "beat 0 is the partial interval [0, r_0)"
    assert rr[0, 1].item() == planted[1] - planted[0], "beat 1 is the first complete beat"
    assert beats.shape == (1, len(planted) + 1, 3, 128)


def test_resampling_keeps_interval_endpoints_and_is_linear():
    planted = np.array([100, 300, 500, 700, 900])
    ramp = torch.arange(LENGTH, dtype=torch.float32).expand(1, 3, -1).clone()
    vcg, ecg = _batch([_reference_lead(planted)], ramp)
    beats, rr, mask = BeatSegmenter(beat_len=128, fs=RATE)(vcg, ecg)
    first_complete = beats[0, 1, 0]
    assert first_complete[0].item() == pytest.approx(100.0)
    assert first_complete[-1].item() == pytest.approx(299.0)
    steps = first_complete[1:] - first_complete[:-1]
    torch.testing.assert_close(steps, steps.mean().expand_as(steps), atol=1e-4, rtol=0)


def test_author_stitcher_reassembles_the_segmented_signal():
    """The contract test: segment a smooth VCG, stitch it back with the author's code.

    Everywhere but the beat edges the round trip is near exact. At the edges the
    author's BeatStitcher itself outputs zero: its cross-fade windows are linspace(0, 1)
    and linspace(1, 0), which reach exactly 0 at the first and last sample of every
    inner beat, and it places beats end to end without overlap, so nothing else covers
    those samples. That is the release's behaviour, pinned here so it is not mistaken
    for a segmentation error.
    """
    planted = np.arange(90, 1000, 85)
    time = torch.arange(LENGTH, dtype=torch.float32)
    smooth = torch.stack([torch.sin(time / 37), torch.cos(time / 53), torch.sin(time / 71)])[None]
    vcg, ecg = _batch([_reference_lead(planted)], smooth)
    beats, rr, mask = BeatSegmenter(beat_len=128, fs=RATE)(vcg, ecg)
    stitched = BeatStitcher(beat_len=128, target_len=LENGTH)(beats, rr, mask)
    edges = sorted({int(p) for p in planted} | {int(p) - 1 for p in planted})
    inner = torch.ones(LENGTH, dtype=torch.bool)
    inner[edges] = False
    assert (stitched - smooth)[..., inner].abs().max().item() < 0.01
    assert stitched[..., edges].abs().max().item() == 0.0


def test_cap_on_beats_preserves_coverage():
    planted = np.arange(40, 1000, 40)  # 24 peaks -> 25 intervals
    vcg, ecg = _batch([_reference_lead(planted)])
    beats, rr, mask = BeatSegmenter(fs=RATE, max_beats=20)(vcg, ecg)
    assert mask.sum().item() == 20
    assert rr[mask].sum().item() == LENGTH


def test_a_record_without_peaks_is_one_interval():
    vcg, ecg = _batch([np.zeros(LENGTH)])
    beats, rr, mask = BeatSegmenter(fs=RATE)(vcg, ecg)
    assert mask.tolist() == [[True]] and rr[0, 0].item() == LENGTH


def test_too_few_peaks_retries_at_the_lower_height():
    """Three planted beats only clear the fallback threshold."""
    lead = _reference_lead([200, 500, 800], height=1.0) + 0.02 * np.sin(np.arange(LENGTH))
    segmenter = BeatSegmenter(fs=RATE, height=50.0, fallback_height=1.0)
    assert len(segmenter._detect_r_peaks(lead)) == 3


def test_peaks_closer_than_the_minimum_rr_are_merged():
    boundaries = BeatSegmenter(fs=RATE)._get_beat_boundaries(np.array([100, 110, 400]), LENGTH)
    assert boundaries == [(0, 100), (100, 400), (400, LENGTH)]


def test_padding_is_masked_and_zero_in_a_mixed_batch():
    vcg, ecg = _batch([_reference_lead([300, 600]), _reference_lead(np.arange(80, 1000, 80))])
    beats, rr, mask = BeatSegmenter(fs=RATE)(vcg, ecg)
    assert mask[0].sum().item() == 3 and mask[1].sum().item() == 13
    assert beats[0, 3:].abs().max().item() == 0.0 and rr[0, 3:].abs().max().item() == 0.0


def test_gradients_reach_the_vcg():
    vcg, ecg = _batch([_reference_lead(np.arange(80, 1000, 80))])
    vcg.requires_grad_(True)
    beats, _, _ = BeatSegmenter(fs=RATE)(vcg, ecg)
    beats.sum().backward()
    assert vcg.grad is not None and vcg.grad.abs().sum().item() > 0


# --- The vendored model now imports and runs ---


def test_released_lvcg_runs_both_passes():
    torch.manual_seed(0)
    model = LVCG(time_len=LENGTH, lead_order="mimic", fs=RATE)
    lead = _reference_lead(np.arange(80, 1000, 80))
    ecg = torch.randn(2, 12, LENGTH) * 0.1
    ecg[:, 1] += torch.as_tensor(lead, dtype=torch.float32)
    embedding = model.forward_inference(ecg)
    assert embedding.shape == (2, model.out_features) and torch.isfinite(embedding).all()
    visible = torch.tensor([[0, 1, 6], [2, 7, 9]])
    out = model.forward_train(ecg, visible)
    assert out["recon"].shape == ecg.shape and torch.isfinite(out["recon"]).all()
