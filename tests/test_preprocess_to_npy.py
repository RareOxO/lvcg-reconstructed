"""scripts/preprocess_to_npy.py: the preprocessed stack feeds training exactly the raw WFDB inputs."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lvcg.data import MimicRawDataset
from lvcg.data.pipeline import ECGDataset, make_dataloader

from test_data_pipeline import _synthetic, _write_record

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def manifest_with_a_bad_record(tmp_path):
    rows = []
    for k in range(9):
        stem = _write_record(tmp_path, f"rec{k}", _synthetic(seed=k))
        rows.append({"id": f"mimic_{k}", "ecg_path": stem})
    rows.insert(4, {"id": "missing", "ecg_path": str(tmp_path / "does_not_exist")})
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def _preprocess(manifest, out):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "preprocess_to_npy.py"), "--manifest", str(manifest),
         "--out", str(out), "--workers", "2", "--chunk", "3", "--verify", "4"],
        check=True, env=env, capture_output=True, text=True,
    )
    return out / "signals.npy"


def test_stack_matches_the_raw_pipeline_record_for_record(manifest_with_a_bad_record, tmp_path):
    signals = _preprocess(manifest_with_a_bad_record, tmp_path / "npy")
    meta = json.loads((tmp_path / "npy" / "meta.json").read_text())
    failed = json.loads((tmp_path / "npy" / "signals.failed.json").read_text())
    assert meta["records"] == 10 and meta["failed"] == 1 and [f["row"] for f in failed] == [4]
    assert np.load(signals, mmap_mode="r").shape == (10, 12, 1000)

    raw = MimicRawDataset(str(manifest_with_a_bad_record))
    stacked = ECGDataset(str(signals), time_len=1000)
    assert len(stacked) == len(raw) == 10
    with pytest.warns(UserWarning):
        raw[4]
    for index in range(10):
        # The unreadable row 4 becomes row 5 in both.
        assert np.array_equal(stacked[index]["ecg"].numpy(), raw[index]["ecg"].numpy()), index


def test_training_split_is_the_same(manifest_with_a_bad_record, tmp_path):
    signals = _preprocess(manifest_with_a_bad_record, tmp_path / "npy")
    base = {"train": {"batch_size": 2}, "run": {"mini_val_ratio": 0.2}, "model": {"lead_order": "mimic"}}
    for split in ("train", "val"):
        raw = make_dataloader({**base, "data": {"meta_root": str(manifest_with_a_bad_record), "num_workers": 0}}, split)
        npy = make_dataloader({**base, "data": {"dataset_type": "npy", "meta_root": str(signals), "num_workers": 0}}, split)
        assert len(raw.dataset) == len(npy.dataset)
        for index in range(len(raw.dataset)):
            assert np.array_equal(raw.dataset[index]["ecg"].numpy(), npy.dataset[index]["ecg"].numpy())


def test_refuses_to_overwrite(manifest_with_a_bad_record, tmp_path):
    _preprocess(manifest_with_a_bad_record, tmp_path / "npy")
    with pytest.raises(subprocess.CalledProcessError):
        _preprocess(manifest_with_a_bad_record, tmp_path / "npy")
