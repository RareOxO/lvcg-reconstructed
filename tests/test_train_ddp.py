"""scripts/train.py under DistributedDataParallel, on CPU processes with the gloo backend."""

import importlib.util
import json
import os
import socket
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
import wfdb

from lvcg.models.lvcg import LVCG
from lvcg.utils.config import Config

ROOT = Path(__file__).resolve().parents[1]
LEADS = ["I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]


def _train_script():
    spec = importlib.util.spec_from_file_location("train_script", ROOT / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def manifest(tmp_path):
    rows = []
    for k in range(12):
        rng = np.random.default_rng(k)
        signal = 0.05 * rng.standard_normal((5000, 12))
        signal[200::400, 1] += 2.0
        wfdb.wrsamp(f"rec{k}", fs=500, units=["mV"] * 12, sig_name=LEADS, p_signal=signal,
                    fmt=["16"] * 12, adc_gain=[200.0] * 12, baseline=[0] * 12, write_dir=str(tmp_path))
        rows.append({"id": f"mimic_{k}", "ecg_path": str(tmp_path / f"rec{k}")})
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _config(tmp_path, manifest, run, max_steps):
    return {
        "run": {"m": run, "s": "s1", "k": "k1", "mini_val_ratio": 0.2,
                "checkpoint_root": str(tmp_path / "checkpoints"), "log_root": str(tmp_path / "logs")},
        "data": {"dataset_type": "mimic_raw", "meta_root": str(manifest), "time_len": 1000,
                 "fs": 100, "scale_method": "zscore", "num_workers": 0},
        "model": {"type": "lvcg", "lead_order": "mimic", "vectorized_stitcher": True, "fast_upsample": True},
        "train": {"device": "cpu", "batch_size": 4, "lr": 5e-4, "max_steps": max_steps, "seed": 7,
                  "log_interval": 2, "eval_interval": 3, "eval_batches": 1, "resume_interval": 3,
                  "save_interval": 1000, "sync_batchnorm": False},
    }


def _worker(rank, world, port, raw, resume):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    _train_script().train(Config(raw=raw), resume=resume)


def _run(raw, world=2, resume=None):
    mp.spawn(_worker, args=(world, _free_port(), raw, resume), nprocs=world, join=True)


def test_epoch_sampler_slices_are_disjoint_complete_and_resumable():
    train_script = _train_script()
    size, world, seed = 23, 3, 5
    slices = []
    for rank in range(world):
        sampler = train_script.EpochSampler(size, seed, rank, world)
        sampler.set_position(epoch=2, skip=0)
        slices.append(list(sampler))
    assert len({len(s) for s in slices}) == 1 and len(slices[0]) == size // world
    joined = [i for s in slices for i in s]
    assert len(set(joined)) == len(joined), "GPUs never read the same record"
    permutation = torch.randperm(size, generator=torch.Generator().manual_seed(seed + 2)).tolist()
    assert set(joined) == set(permutation[: size // world * world])
    sampler = train_script.EpochSampler(size, seed, 1, world)
    sampler.set_position(epoch=2, skip=6)
    assert list(sampler) == permutation[6:][: (size - 6) // world * world][1::world]


def test_ddp_training_logs_once_and_saves_a_single_gpu_checkpoint(tmp_path, manifest):
    _run(_config(tmp_path, manifest, "ddp", max_steps=4))
    events = [json.loads(line) for line in (tmp_path / "logs" / "ddps1k1" / "train_log.jsonl").read_text().splitlines()]
    assert [e["type"] for e in events].count("start") == 1
    trained = [e for e in events if e["type"] == "train"]
    assert [e["step"] for e in trained] == [2, 4] and all(np.isfinite(e["loss"]) for e in trained)
    info = json.loads((tmp_path / "logs" / "ddps1k1" / "run_info.json").read_text())
    assert info["processes"] == 2 and info["batch_per_gpu"] == 2 and info["steps_per_epoch"] == 2
    state = torch.load(tmp_path / "checkpoints" / "ddps1k1" / "final.pt", weights_only=False)
    assert len(state["rng_per_process"]) == 2 and state["processes"] == 2
    LVCG(time_len=1000, lead_order="mimic", fs=100).load_state_dict(state["model_state_dict"], strict=True)


def test_ddp_resuming_continues_exactly(tmp_path, manifest):
    """Two processes: 3 steps + resume to 6 equals 6 straight, across an epoch boundary."""
    _run(_config(tmp_path, manifest, "straight", max_steps=6))
    _run(_config(tmp_path, manifest, "split", max_steps=3))
    _run(_config(tmp_path, manifest, "split", max_steps=6), resume="auto")
    straight = torch.load(tmp_path / "checkpoints" / "straights1k1" / "final.pt", weights_only=False)
    split = torch.load(tmp_path / "checkpoints" / "splits1k1" / "final.pt", weights_only=False)
    assert straight["global_step"] == split["global_step"] == 6
    for name, tensor in straight["model_state_dict"].items():
        torch.testing.assert_close(split["model_state_dict"][name], tensor, msg=name)


def test_the_total_batch_must_divide_across_gpus(tmp_path, manifest):
    raw = _config(tmp_path, manifest, "odd", max_steps=2)
    raw["train"]["batch_size"] = 5
    with pytest.raises(Exception):
        _run(raw)
