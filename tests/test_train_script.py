"""scripts/train.py: the log it writes, and that resuming continues rather than restarts."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import wfdb

from lvcg.utils.config import Config

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_script", ROOT / "scripts" / "train.py")
train_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_script)

LEADS = ["I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]


@pytest.fixture
def manifest(tmp_path):
    rows = []
    for k in range(10):
        rng = np.random.default_rng(k)
        signal = 0.05 * rng.standard_normal((5000, 12))
        signal[200::400, 1] += 2.0  # beats on lead II
        wfdb.wrsamp(f"rec{k}", fs=500, units=["mV"] * 12, sig_name=LEADS, p_signal=signal,
                    fmt=["16"] * 12, adc_gain=[200.0] * 12, baseline=[0] * 12, write_dir=str(tmp_path))
        rows.append({"id": f"mimic_{k}", "ecg_path": str(tmp_path / f"rec{k}")})
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _config(tmp_path, manifest, run, max_steps=6, **train):
    return Config(raw={
        "run": {"m": run, "s": "s1", "k": "k1", "mini_val_ratio": 0.2,
                "checkpoint_root": str(tmp_path / "checkpoints"), "log_root": str(tmp_path / "logs")},
        "data": {"dataset_type": "mimic_raw", "meta_root": str(manifest), "time_len": 1000,
                 "fs": 100, "scale_method": "zscore", "num_workers": 0},
        "model": {"type": "lvcg", "lead_order": "mimic", "beat_len": 128, "state_dim": 256,
                  "max_beats": 20, "vectorized_stitcher": True},
        "train": {"device": "cpu", "batch_size": 2, "lr": 5e-4, "max_steps": max_steps, "seed": 7,
                  "log_interval": 2, "eval_interval": 3, "eval_batches": 1, "resume_interval": 3,
                  "save_interval": 1000, **train},
    })


def _events(tmp_path, run):
    path = tmp_path / "logs" / f"{run}s1k1" / "train_log.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_the_log_records_every_loss_term_validation_and_saves(tmp_path, manifest):
    train_script.train(_config(tmp_path, manifest, "log"))
    events = _events(tmp_path, "log")
    kinds = [e["type"] for e in events]
    assert kinds[0] == "start" and "val" in kinds and "checkpoint" in kinds
    trained = [e for e in events if e["type"] == "train"]
    assert [e["step"] for e in trained] == [2, 4, 6]
    for event in trained:
        for key in ("loss", "recon", "temporal", "beat", "base", "grad_norm", "lr", "steps_per_sec"):
            assert event[key] is not None and np.isfinite(event[key]), key
    val = [e for e in events if e["type"] == "val"]
    assert [e["step"] for e in val] == [3, 6] and all(e["records"] == 2 for e in val)
    info = json.loads((tmp_path / "logs" / "logs1k1" / "run_info.json").read_text())
    assert info["config"]["train"]["batch_size"] == 2 and info["steps_per_epoch"] == 4
    names = {p.name for p in (tmp_path / "checkpoints" / "logs1k1").iterdir()}
    assert {"final.pt", "last.pt"} <= names and not any(n.endswith(".tmp") for n in names)


def test_resuming_continues_exactly(tmp_path, manifest):
    """3 steps + resume to 6 must equal 6 straight steps: data order, masks, dropout, optimiser.

    Step 3 is mid-epoch (4 steps per epoch), and an evaluation runs at step 3 in both.
    """
    train_script.train(_config(tmp_path, manifest, "straight"))
    train_script.train(_config(tmp_path, manifest, "split", max_steps=3))
    train_script.train(_config(tmp_path, manifest, "split", max_steps=6), resume="auto")
    straight = torch.load(tmp_path / "checkpoints" / "straights1k1" / "final.pt", weights_only=False)
    split = torch.load(tmp_path / "checkpoints" / "splits1k1" / "final.pt", weights_only=False)
    assert split["global_step"] == straight["global_step"] == 6
    for name, tensor in straight["model_state_dict"].items():
        torch.testing.assert_close(split["model_state_dict"][name], tensor, msg=name)
    straight_losses = [e["loss"] for e in _events(tmp_path, "straight") if e["type"] == "train"]
    split_losses = [e["loss"] for e in _events(tmp_path, "split") if e["type"] == "train"]
    assert straight_losses[-1] == pytest.approx(split_losses[-1], rel=1e-6)
    assert [e["type"] for e in _events(tmp_path, "split")].count("resume") == 1


def test_a_fresh_run_never_overwrites_checkpoints(tmp_path, manifest):
    train_script.train(_config(tmp_path, manifest, "once", max_steps=3))
    with pytest.raises(SystemExit, match="already holds"):
        train_script.train(_config(tmp_path, manifest, "once", max_steps=3))


def test_resume_refuses_a_changed_config(tmp_path, manifest):
    train_script.train(_config(tmp_path, manifest, "changed", max_steps=3))
    with pytest.raises(SystemExit, match="batch_size"):
        train_script.train(_config(tmp_path, manifest, "changed", batch_size=4), resume="auto")
    with pytest.raises(SystemExit, match="Nothing to resume"):
        train_script.train(_config(tmp_path, manifest, "missing"), resume="auto")


def test_resume_may_turn_the_speed_switches_on(tmp_path, manifest):
    """Same numbers, so switching them on resume is allowed; the architecture is still checked."""
    train_script.train(_config(tmp_path, manifest, "switch", max_steps=3))
    cfg = _config(tmp_path, manifest, "switch", max_steps=6)
    cfg.raw["model"]["fast_upsample"] = True
    cfg.raw["model"]["vectorized_stitcher"] = False
    train_script.train(cfg, resume="auto")
    final = torch.load(tmp_path / "checkpoints" / "switchs1k1" / "final.pt", weights_only=False)
    assert final["global_step"] == 6
    cfg.raw["model"]["state_dim"] = 128
    cfg.raw["train"]["max_steps"] = 9
    with pytest.raises(SystemExit, match="model section"):
        train_script.train(cfg, resume="auto")
