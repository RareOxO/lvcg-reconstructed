"""Pretraining throughput at several batch sizes, on the GPU this runs on.

Times one full step exactly as ``scripts/train.py`` takes it -- the release's losses,
backward, gradient clipping, AdamW -- on a batch of real records held on the GPU, so
data loading is left out and the number is the model's own throughput. Run it on the
machine you will train on: the answer depends on the GPU.

    python scripts/benchmark_batch.py --manifest ~/data/mimic_manifest.jsonl
    python scripts/benchmark_batch.py --manifest ~/data/mimic_manifest.jsonl --batches 64 128 256 512 1024

Without ``--manifest`` it uses synthetic records with beats planted on lead II.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import time

import numpy as np
import torch

from lvcg.models.lvcg import LVCG

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("train_script", os.path.join(HERE, "train.py"))
train_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_script)


def records(manifest: str | None, count: int) -> torch.Tensor:
    if manifest is None:
        generator = torch.Generator().manual_seed(0)
        ecg = torch.randn(count, 12, 1000, generator=generator) * 0.3
        ecg[:, 1, 40::85] += 6.0
        return ecg
    from lvcg.data import MimicRawDataset

    dataset = MimicRawDataset(manifest)
    return torch.stack([dataset[i % len(dataset)]["ecg"] for i in range(count)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest")
    parser.add_argument("--batches", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--vectorized-stitcher", choices=("true", "false"), default="true")
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}, "
          f"{torch.cuda.get_device_properties(device).total_memory / 2**30:.1f} GiB")
    pool = records(args.manifest, max(args.batches)).to(device)
    lambdas = {"temporal": 0.1, "beat": 1.0, "base": 1.0}
    print(f"{'batch':>6} {'s/step':>8} {'samples/s':>10} {'vs 64':>6} {'peak GiB':>9}")
    reference = None
    for batch in sorted(args.batches):
        torch.manual_seed(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = LVCG(time_len=1000, lead_order="mimic", fs=100,
                     vectorized_stitcher=args.vectorized_stitcher == "true").to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
        ecg = pool[:batch]
        times = []
        try:
            for _ in range(args.steps):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                optimizer.zero_grad()
                loss, _ = train_script.compute_losses(model, ecg, 3, lambdas)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                torch.cuda.synchronize(device)
                times.append(time.perf_counter() - start)
        except torch.OutOfMemoryError:
            print(f"{batch:>6}  out of memory")
            break
        step = float(np.median(times[2:]))
        rate = batch / step
        if batch == 64:
            reference = rate
        relative = f"{rate / reference:.2f}x" if reference else "-"
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        total = torch.cuda.get_device_properties(device).total_memory / 2**30
        note = "  (exceeds GPU memory: not valid)" if peak > total else ""
        print(f"{batch:>6} {step:>8.3f} {rate:>10.0f} {relative:>6} {peak:>9.2f}{note}")
        del model, optimizer


if __name__ == "__main__":
    main()
