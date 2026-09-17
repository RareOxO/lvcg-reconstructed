"""The reconstructed MIMIC reader and data pipeline that scripts/train.py imports."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import wfdb

from lvcg.data import LEAD_ORDERS, MimicRawDataset, make_dataloaders, split_indices
from lvcg.data.mimic import (
    load_mimic_manifest,
    load_wfdb_record,
    normalize_ecg,
    preprocess_mimic_record,
)
from lvcg.data.pipeline import ECGDataset

MIMIC_HEADER_ORDER = ["I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]


def _write_record(directory, name, signal, fs=500, names=MIMIC_HEADER_ORDER):
    wfdb.wrsamp(
        name,
        fs=fs,
        units=["mV"] * len(names),
        sig_name=list(names),
        p_signal=signal,
        fmt=["16"] * len(names),
        adc_gain=[200.0] * len(names),
        baseline=[0] * len(names),
        write_dir=str(directory),
    )
    return str(Path(directory) / name)


def _synthetic(fs=500, seconds=10, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(fs * seconds) / fs
    # Each lead a distinct 5 Hz amplitude, so a reordering is visible after preprocessing.
    leads = [(1 + k) * 0.1 * np.sin(2 * np.pi * 5 * t) + 0.01 * rng.standard_normal(t.size) for k in range(12)]
    return np.stack(leads, axis=1)  # [T, L]


@pytest.fixture
def manifest(tmp_path):
    rows = []
    for k in range(10):
        stem = _write_record(tmp_path, f"rec{k}", _synthetic(seed=k))
        rows.append({"id": f"mimic_{k}", "ecg_path": stem, "messages": []})
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_wfdb_record_round_trip(tmp_path):
    signal = _synthetic()
    stem = _write_record(tmp_path, "one", signal)
    data, fs, names = load_wfdb_record(stem)
    assert data.shape == (5000, 12) and data.dtype == np.float32 and fs == 500
    assert names == MIMIC_HEADER_ORDER
    np.testing.assert_allclose(data, signal, atol=5e-3)
    with pytest.raises(FileNotFoundError):
        load_wfdb_record(str(tmp_path / "missing"))


def test_preprocessing_shape_rate_and_normalisation():
    out = preprocess_mimic_record(_synthetic(), 500, MIMIC_HEADER_ORDER)
    assert out.shape == (12, 1000) and out.dtype == np.float32 and np.isfinite(out).all()
    np.testing.assert_allclose(out.mean(-1), 0, atol=1e-5)
    np.testing.assert_allclose(out.std(-1), 1, atol=1e-4)


def test_leads_are_reordered_by_name():
    """A header in PTB-XL order must come out in the model's MIMIC order."""
    signal = _synthetic()
    ptbxl_names = list(LEAD_ORDERS["ptbxl"])
    ptbxl_signal = signal[:, [MIMIC_HEADER_ORDER.index(n) for n in ptbxl_names]]
    from_mimic = preprocess_mimic_record(signal, 500, MIMIC_HEADER_ORDER, scale_method="none")
    from_ptbxl = preprocess_mimic_record(ptbxl_signal, 500, ptbxl_names, scale_method="none")
    np.testing.assert_allclose(from_mimic, from_ptbxl, atol=1e-6)
    with pytest.raises(ValueError, match="lacks leads"):
        preprocess_mimic_record(signal[:, :11], 500, MIMIC_HEADER_ORDER[:11])


def test_band_pass_keeps_the_band_and_removes_what_is_outside():
    fs, t = 500, np.arange(5000) / 500
    inside = np.sin(2 * np.pi * 10 * t)
    outside = 2.0 + np.sin(2 * np.pi * 0.1 * t) + np.sin(2 * np.pi * 45 * t)  # DC, drift, above 40 Hz
    signal = np.stack([inside + outside] * 12, axis=1)
    out = preprocess_mimic_record(signal, fs, MIMIC_HEADER_ORDER, scale_method="none")[0, 100:900]
    reference = preprocess_mimic_record(
        np.stack([inside] * 12, axis=1), fs, MIMIC_HEADER_ORDER, bandpass=None, scale_method="none"
    )[0, 100:900]
    assert np.abs(out - reference).max() < 0.1, "10 Hz passes, DC/drift/45 Hz are removed"


def test_missing_samples_do_not_spread():
    signal = _synthetic()
    signal[1000:1010, 3] = np.nan
    out = preprocess_mimic_record(signal, 500, MIMIC_HEADER_ORDER)
    assert np.isfinite(out).all()


def test_cut_and_pad_to_time_len():
    long = preprocess_mimic_record(_synthetic(seconds=12), 500, MIMIC_HEADER_ORDER)
    short = preprocess_mimic_record(_synthetic(seconds=7), 500, MIMIC_HEADER_ORDER, scale_method="none")
    assert long.shape == (12, 1000) and short.shape == (12, 1000)
    assert np.abs(short[:, 700:]).max() == 0


def test_normalize_methods():
    data = np.random.default_rng(0).normal(3, 5, (12, 400)).astype(np.float32)
    minmax = normalize_ecg(data, "minmax")
    assert minmax.min() >= 0 and minmax.max() <= 1 + 1e-6
    np.testing.assert_array_equal(normalize_ecg(data, "none"), data)
    with pytest.raises(ValueError):
        normalize_ecg(data, "bogus")


def test_split_is_disjoint_complete_and_seeded():
    train, val = split_indices(1000, 0.1, seed=7)
    assert len(val) == 100 and not set(train) & set(val) and sorted(train + val) == list(range(1000))
    assert (train, val) == split_indices(1000, 0.1, seed=7)
    assert val != split_indices(1000, 0.1, seed=8)[1]
    assert len(split_indices(5, 0.1)[1]) == 1


def test_dataset_skips_an_unreadable_record(manifest, tmp_path):
    rows = load_mimic_manifest(manifest)
    Path(rows[3]["ecg_path"] + ".dat").write_bytes(b"")  # corrupt one record
    dataset = MimicRawDataset(str(manifest))
    with pytest.warns(UserWarning, match="Skipping unreadable record"):
        item = dataset[3]
    assert item["id"] == "mimic_4" and item["ecg"].shape == (12, 1000)


def test_make_dataloaders_matches_what_train_py_consumes(manifest):
    cfg = {
        "data": {"dataset_type": "mimic_raw", "meta_root": str(manifest), "time_len": 1000,
                 "fs": 100, "scale_method": "zscore", "num_workers": 0},
        "train": {"batch_size": 4},
        "run": {"mini_val_ratio": 0.2},
        "model": {"lead_order": "mimic"},
    }
    train_loader, val_loader = make_dataloaders(cfg)
    assert len(train_loader.dataset) == 8 and len(val_loader.dataset) == 2
    batch = next(iter(train_loader))
    assert batch["ecg"].shape == (4, 12, 1000) and batch["ecg"].dtype == torch.float32
    assert len(train_loader) == 2, "drop_last keeps every training step a full batch"


def test_npy_dataset(tmp_path):
    np.save(tmp_path / "stack.npy", np.random.default_rng(0).normal(size=(5, 12, 1200)).astype(np.float32))
    dataset = ECGDataset(str(tmp_path), time_len=1000)
    assert len(dataset) == 5 and dataset[2]["ecg"].shape == (12, 1000)


def test_training_step_on_the_pipeline_output(manifest):
    """One step of exactly what scripts/train.py does, on the reconstructed loader."""
    from lvcg.models.lvcg import LVCG, base_beat_loss, beat_level_loss, temporal_loss
    from lvcg.models.utils.loss import masked_reconstruction_loss, random_lead_mask

    torch.manual_seed(0)
    cfg = {"data": {"meta_root": str(manifest), "num_workers": 0}, "train": {"batch_size": 4},
           "run": {"mini_val_ratio": 0.2}}
    ecg = next(iter(make_dataloaders(cfg)[0]))["ecg"]
    # Planted sharp beats on lead II so every record segments into several beats.
    ecg[:, 1, 50::80] += 8.0
    model = LVCG(time_len=1000, lead_order="mimic", fs=100)
    visible, mask = random_lead_mask(ecg.shape[0], num_visible=3)
    out = model.forward_train(ecg, visible)
    loss = (
        masked_reconstruction_loss(out["recon"], ecg, mask)
        + 0.1 * temporal_loss(out["states_pred"], out["states_real"], out["beat_mask"])
        + beat_level_loss(out["V_hat_beats"], out["V_beats"], out["beat_mask_full"])
        + base_beat_loss(out["V_base_hat"], out["V_base"])
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
