"""Preprocess every record of a MIMIC manifest once, into a single memory-mapped ``.npy``.

Reading WFDB, resampling to 100 Hz and band-pass filtering take about 10 ms per record
on a CPU, which caps training at roughly 1,400 records per second however many GPUs are
used. This script does that work once, in parallel, and writes

    <out>/signals.npy            float32 [N, 12, time_len], manifest order
    <out>/signals.failed.json    manifest rows that could not be read (their rows are 0)
    <out>/meta.json              settings, counts, source manifest and its SHA-256

Training then reads the file with ``data.dataset_type: npy`` and
``data.meta_root: <out>/signals.npy``. Everything ``preprocess_mimic_record`` does is
stored except the last step, the per-lead z-score, which ``ECGDataset`` applies when a
record is read -- so the inputs the model sees are identical to reading the WFDB files
on the fly, and so is the train/validation split, because the rows keep the manifest's
order. A random sample of stored rows is re-derived from the WFDB files at the end and
must match exactly.

    python scripts/preprocess_to_npy.py --manifest ~/data/mimic_manifest.jsonl
"""

from __future__ import annotations

import os

# One thread per worker process: the parallelism is across processes.
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
import hashlib
import json
import multiprocessing as mp
import shutil
import time
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from lvcg.data.mimic import DEFAULT_BANDPASS, load_mimic_manifest, load_wfdb_record, preprocess_mimic_record
from lvcg.utils.config import load_config

DEFAULT_OUT = "/home/featurize/mimic_npy"
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "train", "lvcg_v5_gru.yaml")

_state: Dict[str, object] = {}


def _settings(config_path: str) -> Dict[str, object]:
    cfg = load_config(config_path)
    bandpass = cfg.data.get("bandpass", DEFAULT_BANDPASS)
    return {
        "fs": int(cfg.data.get("fs", 100)),
        "time_len": int(cfg.data.get("time_len", 1000)),
        "bandpass": list(bandpass) if bandpass is not None else None,
        "lead_order": cfg.model.get("lead_order", "mimic"),
    }


def _stored_record(path: str, settings: Dict[str, object]) -> np.ndarray:
    data, fs, names = load_wfdb_record(path)
    bandpass = tuple(settings["bandpass"]) if settings["bandpass"] is not None else None
    return preprocess_mimic_record(
        data, fs, names, settings["fs"], settings["time_len"], bandpass,
        scale_method="none", lead_order=settings["lead_order"],
    ).astype(np.float32)


def _init_worker(partial: str, paths: List[str], settings: Dict[str, object]) -> None:
    _state["signals"] = np.load(partial, mmap_mode="r+")
    _state["paths"] = paths
    _state["settings"] = settings


def _process(chunk: Tuple[int, int]) -> Tuple[int, List[Tuple[int, str]]]:
    start, stop = chunk
    signals, paths, settings = _state["signals"], _state["paths"], _state["settings"]
    failures = []
    for row in range(start, stop):
        try:
            record = _stored_record(paths[row], settings)
            if not np.isfinite(record).all():
                raise ValueError("non-finite values after preprocessing")
            signals[row] = record
        except (OSError, ValueError) as error:
            failures.append((row, f"{paths[row]}: {error}"))
    signals.flush()
    return stop - start, failures


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"output directory (default {DEFAULT_OUT})")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="read data.fs / time_len / bandpass and model.lead_order")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--verify", type=int, default=64, help="rows re-derived from WFDB at the end")
    args = parser.parse_args()

    settings = _settings(args.config)
    rows = load_mimic_manifest(args.manifest)
    paths = [row["ecg_path"] for row in rows]
    count = len(paths)
    os.makedirs(args.out, exist_ok=True)
    final = os.path.join(args.out, "signals.npy")
    partial = final + ".partial"
    if os.path.exists(final):
        raise SystemExit(f"{final} already exists; remove it or choose another --out")

    shape = (count, 12, settings["time_len"])
    needed = int(np.prod(shape)) * 4
    free = shutil.disk_usage(args.out).free
    print(f"{count} records -> {final}, {needed / 2**30:.1f} GiB ({free / 2**30:.0f} GiB free), "
          f"{args.workers} workers, settings {settings}", flush=True)
    if needed > free:
        raise SystemExit("Not enough free disk space")

    signals = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=shape)
    del signals
    started = time.perf_counter()
    chunks = [(s, min(s + args.chunk, count)) for s in range(0, count, args.chunk)]
    failures: List[Tuple[int, str]] = []
    context = mp.get_context("fork")
    with context.Pool(args.workers, initializer=_init_worker, initargs=(partial, paths, settings)) as pool:
        with tqdm(total=count, unit="rec", desc="Preprocess") as bar:
            for done, failed in pool.imap_unordered(_process, chunks):
                failures.extend(failed)
                bar.update(done)
    seconds = time.perf_counter() - started

    stored = np.load(partial, mmap_mode="r")
    failed_rows = {row for row, _ in failures}
    rng = np.random.default_rng(0)
    candidates = [r for r in rng.choice(count, size=min(args.verify * 2, count), replace=False) if r not in failed_rows]
    for row in candidates[: args.verify]:
        if not np.array_equal(stored[row], _stored_record(paths[row], settings)):
            raise SystemExit(f"Verification failed at row {row}: stored values differ from WFDB")
    del stored

    os.replace(partial, final)
    with open(os.path.join(args.out, "signals.failed.json"), "w", encoding="utf-8") as handle:
        json.dump(sorted(({"row": r, "error": e} for r, e in failures), key=lambda x: x["row"]), handle, indent=1)
    meta = {
        "records": count,
        "failed": len(failures),
        "shape": list(shape),
        "dtype": "float32",
        **settings,
        "filter": "4th-order Butterworth band-pass, sosfiltfilt; resample_poly; NaN -> 0",
        "normalisation": "none stored; ECGDataset applies the per-lead z-score on read",
        "row_order": "manifest order",
        "source_manifest": os.path.abspath(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "verified_rows": len(candidates[: args.verify]),
        "seconds": round(seconds, 1),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(f"Done in {seconds / 60:.1f} min: {count - len(failures)} records written, "
          f"{len(failures)} failed, {meta['verified_rows']} rows verified against WFDB.")
    print(f"Train with:  --data.meta_root {final}  and  data.dataset_type: npy")


if __name__ == "__main__":
    main()
