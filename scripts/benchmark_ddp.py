"""Multi-GPU pretraining throughput: GPU count x batch per GPU x BatchNorm synchronisation.

One command runs the whole grid on the machine it is started on:

    python scripts/benchmark_ddp.py --manifest ~/data/mimic_manifest.jsonl

GPU counts default to 1, 2, 4, ... up to every visible GPU. For each count it times one
full DistributedDataParallel pretraining step -- the release's losses, backward with
gradient synchronisation, clipping, AdamW -- exactly as ``scripts/train.py`` takes it, on
real records held on each GPU, so data loading is left out:

* the paper's total batch of 64 split across the GPUs, with BatchNorm synchronised
  (what ``train.py`` does by default) and without;
* every batch per GPU in ``--per-gpu-batches``, without synchronisation, to show the
  throughput ceiling.

The table reports samples per second over all GPUs, the speed-up against one GPU at
batch 64, and the scaling efficiency: throughput against the GPU count times one GPU at
the same batch per GPU. Rows whose total batch is 64 keep the paper's optimisation;
larger totals process more samples per second but change the optimisation, so they
measure speed, not time to the same model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import time
from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from lvcg.models.lvcg import LVCG

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_BATCH = 64
_spec = importlib.util.spec_from_file_location("train_script", os.path.join(HERE, "train.py"))
train_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_script)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _records(manifest: str | None, count: int) -> torch.Tensor:
    if manifest is None:
        generator = torch.Generator().manual_seed(0)
        ecg = torch.randn(count, 12, 1000, generator=generator) * 0.3
        ecg[:, 1, 40::85] += 6.0
        return ecg
    from lvcg.data import MimicRawDataset

    dataset = MimicRawDataset(manifest)
    return torch.stack([dataset[i % len(dataset)]["ecg"] for i in range(count)])


def _configs(world: int, per_gpu_batches: List[int], counts: List[int]) -> List[Dict]:
    """(batch per GPU, sync) pairs to time at this GPU count."""
    configs = {(b, False) for b in per_gpu_batches}
    if PAPER_BATCH % world == 0:
        configs.add((PAPER_BATCH // world, False))
        if world > 1:
            configs.add((PAPER_BATCH // world, True))
    if world == 1:
        # One GPU at every batch per GPU used anywhere, for the scaling efficiency.
        configs |= {(PAPER_BATCH // w, False) for w in counts if PAPER_BATCH % w == 0}
    return [{"batch": b, "sync": s} for b, s in sorted(configs)]


def _worker(rank, world, port, backend, configs, pool, steps, switches, queue):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    device = torch.device("cuda", rank % torch.cuda.device_count())
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group(backend, rank=rank, world_size=world)
    lambdas = {"temporal": 0.1, "beat": 1.0, "base": 1.0}
    for config in configs:
        batch, sync = config["batch"], config["sync"]
        torch.manual_seed(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = LVCG(time_len=1000, lead_order="mimic", fs=100, **switches).to(device).train()
        if sync and world > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        step_model = (
            DistributedDataParallel(train_script.TrainStep(model), device_ids=[device.index],
                                    find_unused_parameters=True)
            if world > 1 else model
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
        ecg = pool[(rank * batch) % len(pool):][:batch].to(device)
        if len(ecg) < batch:
            ecg = pool[:batch].to(device)
        status, times = "ok", []
        try:
            for _ in range(steps):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                optimizer.zero_grad()
                loss, _ = train_script.compute_losses(step_model, ecg, 3, lambdas)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                torch.cuda.synchronize(device)
                times.append(time.perf_counter() - start)
        except torch.OutOfMemoryError:
            status = "oom"
        if rank == 0:
            step = float(np.median(times[3:])) if status == "ok" and len(times) > 3 else None
            queue.put({
                "gpus": world, "per_gpu": batch, "total": batch * world, "sync": sync,
                "status": status, "step": step,
                "samples_per_sec": batch * world / step if step else None,
                "peak_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            })
        del step_model, model, optimizer, ecg
        torch.cuda.empty_cache()
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()


def _table(rows: List[Dict]) -> str:
    base = {(r["per_gpu"]): r["samples_per_sec"] for r in rows
            if r["gpus"] == 1 and not r["sync"] and r["status"] == "ok"}
    reference = base.get(PAPER_BATCH)
    header = (f"{'GPUs':>4} {'per GPU':>7} {'total':>6} {'SyncBN':>6} {'s/step':>8} "
              f"{'samples/s':>10} {'vs 1x64':>8} {'scaling':>8} {'peak GiB':>9}")
    lines = [header, "-" * len(header)]
    for r in sorted(rows, key=lambda r: (r["gpus"], r["per_gpu"], r["sync"])):
        mark = "  <- total 64 (paper)" if r["total"] == PAPER_BATCH else ""
        if r["status"] != "ok":
            lines.append(f"{r['gpus']:>4} {r['per_gpu']:>7} {r['total']:>6} {str(r['sync']):>6} "
                         f"{'out of memory':>28}{mark}")
            continue
        vs = f"{r['samples_per_sec'] / reference:.2f}x" if reference else "-"
        single = base.get(r["per_gpu"])
        scaling = f"{r['samples_per_sec'] / (r['gpus'] * single):.0%}" if single else "-"
        lines.append(f"{r['gpus']:>4} {r['per_gpu']:>7} {r['total']:>6} {str(r['sync']):>6} "
                     f"{r['step']:>8.3f} {r['samples_per_sec']:>10.0f} {vs:>8} {scaling:>8} "
                     f"{r['peak_gib']:>9.2f}{mark}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest")
    parser.add_argument("--gpu-counts", type=int, nargs="+",
                        help="default: 1, 2, 4, ... and every visible GPU")
    parser.add_argument("--per-gpu-batches", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--vectorized-stitcher", choices=("true", "false"), default="true")
    parser.add_argument("--fast-upsample", choices=("true", "false"), default="true")
    parser.add_argument("--backend", default="nccl", help="nccl on real multi-GPU machines")
    parser.add_argument("--out", default="benchmark_ddp.jsonl")
    args = parser.parse_args()

    available = torch.cuda.device_count()
    if available == 0:
        raise SystemExit("No CUDA device visible")
    counts = args.gpu_counts or sorted({c for c in (1, 2, 4, 8, 16) if c <= available} | {available})
    print(f"Visible GPUs: {available} x {torch.cuda.get_device_name(0)}; timing GPU counts {counts}")
    largest = max(max(args.per_gpu_batches), PAPER_BATCH)
    pool = _records(args.manifest, largest * max(counts)).share_memory_()
    switches = {"vectorized_stitcher": args.vectorized_stitcher == "true",
                "fast_upsample": args.fast_upsample == "true"}

    rows: List[Dict] = []
    context = mp.get_context("spawn")
    for world in counts:
        configs = _configs(world, args.per_gpu_batches, counts)
        queue = context.SimpleQueue()
        mp.spawn(_worker, args=(world, _free_port(), args.backend, configs, pool, args.steps, switches, queue),
                 nprocs=world, join=True)
        while not queue.empty():
            row = queue.get()
            rows.append(row)
            print(f"  {world} GPU(s), {row['per_gpu']}/GPU, sync={row['sync']}: "
                  + (f"{row['samples_per_sec']:.0f} samples/s" if row["status"] == "ok" else row["status"]),
                  flush=True)

    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(r) + "\n" for r in rows))
    print()
    print(_table(rows))
    print(f"\nRows saved to {args.out}")


if __name__ == "__main__":
    main()
