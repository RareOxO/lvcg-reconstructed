"""Data loading pipeline for ECG datasets.

Provides:
- ECGDataset: pre-processed ECG ``.npy`` files
- MimicRawDataset: raw MIMIC-IV WFDB records read through a JSONL manifest
- split_indices / make_dataloader / make_dataloaders: the train/validation loaders that
  ``scripts/train.py`` builds with ``make_dataloaders(cfg.raw)``

Reconstructed; the class and function names, what calls what, and the config keys
(``data.dataset_type``, ``data.meta_root``, ``data.time_len``, ``data.fs``,
``data.scale_method``, ``run.mini_val_ratio``, ``train.batch_size``) are the release's.
Choices it does not record, stated here:

* The validation split is a seeded random split over manifest rows, which is what a
  ``split_indices(total_size, ...)`` signature implies; it is not a patient-level split.
  For self-supervised pretraining that only affects how optimistic the reconstruction
  loss on the validation rows is.
* A record that cannot be read is replaced by the next readable one, deterministically,
  instead of stopping a run that may be days long. Each skip raises a warning naming the
  record, so it shows in the training log whichever worker process read it.
* Loader defaults: 4 workers, pinned memory, shuffled and ``drop_last`` for training so
  every step has a full batch, unshuffled for validation.
"""

import json
import os
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .mimic import DEFAULT_BANDPASS, load_mimic_manifest, load_wfdb_record, normalize_ecg, preprocess_mimic_record

MAX_SKIPPED = 32


@dataclass
class _Preprocessing:
    time_len: int = 1000
    fs: int = 100
    scale_method: str = "zscore"
    bandpass: Optional[Tuple[float, float]] = DEFAULT_BANDPASS
    lead_order: str = "mimic"


class ECGDataset(Dataset):
    """Dataset for pre-processed ECG .npy files.

    ``root`` is a directory of ``.npy`` files or a single ``.npy``; each file holds one
    record [L, T] or a stack [N, L, T] already at the model's sampling rate and lead
    order. Records are cut or zero-padded to ``time_len`` and normalised on access.

    A stack written by ``scripts/preprocess_to_npy.py`` has a sibling
    ``<name>.failed.json`` listing the rows whose WFDB record could not be read. Those
    rows are replaced by the next readable one, exactly as ``MimicRawDataset`` does, so
    the two datasets yield the same records in the same order. Each process keeps its
    memory maps open rather than reopening a file per record.
    """

    def __init__(self, root: str, time_len: int = 1000, scale_method: str = "zscore",
                 indices: Optional[Sequence[int]] = None):
        self.time_len = time_len
        self.scale_method = scale_method
        files = [root] if os.path.isfile(root) else sorted(
            os.path.join(root, name) for name in os.listdir(root) if name.endswith(".npy")
        )
        if not files:
            raise FileNotFoundError(f"No .npy files at {root}")
        self.items: List[Tuple[str, Optional[int]]] = []
        failed = set()
        for path in files:
            array = np.load(path, mmap_mode="r")
            if array.ndim == 2:
                self.items.append((path, None))
            elif array.ndim == 3:
                sidecar = os.path.splitext(path)[0] + ".failed.json"
                if os.path.exists(sidecar):
                    with open(sidecar, "r", encoding="utf-8") as handle:
                        failed |= {(path, entry["row"]) for entry in json.load(handle)}
                self.items.extend((path, row) for row in range(array.shape[0]))
            else:
                raise ValueError(f"{path} must be [L, T] or [N, L, T], got {array.shape}")
        if indices is not None:
            self.items = [self.items[i] for i in indices]
        self.unreadable = [item in failed for item in self.items]
        if self.items and all(self.unreadable):
            raise ValueError(f"Every selected record in {root} is marked unreadable")
        self._arrays: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.items)

    def __getstate__(self):
        # Memory maps are reopened in each DataLoader worker, never pickled.
        return {**self.__dict__, "_arrays": {}}

    def _load_file(self, index: int) -> np.ndarray:
        position = index
        while self.unreadable[position]:
            position = (position + 1) % len(self.items)
        path, row = self.items[position]
        array = self._arrays.get(path)
        if array is None:
            array = self._arrays[path] = np.load(path, mmap_mode="r")
        signal = np.asarray(array if row is None else array[row], dtype=np.float32)
        if signal.shape[-1] >= self.time_len:
            signal = signal[:, : self.time_len]
        else:
            signal = np.pad(signal, ((0, 0), (0, self.time_len - signal.shape[-1])))
        return normalize_ecg(signal, self.scale_method)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return {"ecg": torch.from_numpy(self._load_file(index)), "id": f"npy_{index}"}


class MimicRawDataset(Dataset):
    """Dataset for raw MIMIC-IV WFDB files using a manifest.

    Supports train/validation subsets through ``indices`` into the manifest rows.
    """

    def __init__(
        self,
        manifest_path: str,
        fs_out: int = 100,
        time_len: int = 1000,
        scale_method: str = "zscore",
        bandpass: Optional[Tuple[float, float]] = DEFAULT_BANDPASS,
        indices: Optional[Sequence[int]] = None,
        lead_order: str = "mimic",
    ):
        """
        Args:
            manifest_path: Path to JSONL manifest file
            fs_out: Output sampling rate (Hz)
            time_len: Output length in samples
            scale_method: Per-lead normalisation, see ``normalize_ecg``
            bandpass: (low, high) Hz, or None to skip filtering
            indices: Manifest rows in this subset; all rows when None
            lead_order: Named lead order the model expects
        """
        rows = load_mimic_manifest(manifest_path)
        self.rows = rows if indices is None else [rows[i] for i in indices]
        if not self.rows:
            raise ValueError(f"No records selected from {manifest_path}")
        self.settings = _Preprocessing(time_len, fs_out, scale_method, bandpass, lead_order)

    def __len__(self) -> int:
        return len(self.rows)

    def _read(self, index: int) -> np.ndarray:
        data, fs, names = load_wfdb_record(self.rows[index]["ecg_path"])
        s = self.settings
        signal = preprocess_mimic_record(
            data, fs, names, s.fs, s.time_len, s.bandpass, s.scale_method, s.lead_order
        )
        if not np.isfinite(signal).all():
            raise ValueError("non-finite signal after preprocessing")
        return signal

    def __getitem__(self, index: int) -> Dict[str, object]:
        for offset in range(MAX_SKIPPED):
            position = (index + offset) % len(self.rows)
            try:
                signal = self._read(position)
            except (OSError, ValueError) as error:
                warnings.warn(f"Skipping unreadable record {self.rows[position].get('ecg_path')}: {error}")
                last = error
                continue
            return {"ecg": torch.from_numpy(signal), "id": self.rows[position].get("id", str(position))}
        raise RuntimeError(f"{MAX_SKIPPED} consecutive unreadable records from row {index}: {last}")


def _collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Collate function for ECG batches."""
    return {
        "ecg": torch.stack([item["ecg"] for item in batch]),
        "id": [item["id"] for item in batch],
    }


def split_indices(total_size: int, val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[int], List[int]]:
    """Split indices into train and validation sets.

    Args:
        total_size: Number of records.
        val_ratio: Fraction held out for validation; at least one record when > 0.
        seed: Seed of the permutation, so the split is identical across runs.
    """
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must lie in [0, 1)")
    order = list(range(total_size))
    random.Random(seed).shuffle(order)
    count = 0 if val_ratio == 0 else max(1, int(round(total_size * val_ratio)))
    return sorted(order[count:]), sorted(order[:count])


def make_dataloader(cfg: Dict, split: str = "train") -> DataLoader:
    """Construct DataLoader for the given split.

    Config keys (data section):
        dataset_type: ``mimic_raw`` (WFDB through a manifest) or ``npy``
        meta_root: manifest path (mimic_raw) or .npy file/directory (npy)
        time_len, fs, scale_method: output length, rate and normalisation
        bandpass (optional): [low, high] Hz, default [0.67, 40]
        num_workers (optional), seed (optional)
    Also read: ``train.batch_size`` and ``run.mini_val_ratio``.
    """
    if split not in ("train", "val"):
        raise ValueError("split must be train or val")
    data = cfg.get("data", {})
    batch_size = int(cfg.get("train", {}).get("batch_size", 64))
    val_ratio = float(cfg.get("run", {}).get("mini_val_ratio", 0.1))
    seed = int(data.get("seed", 42))
    dataset_type = data.get("dataset_type", "mimic_raw")
    meta_root = data["meta_root"]
    time_len = int(data.get("time_len", 1000))
    scale_method = data.get("scale_method", "zscore")

    if dataset_type == "mimic_raw":
        total = len(load_mimic_manifest(meta_root))
    elif dataset_type == "npy":
        total = len(ECGDataset(meta_root, time_len, scale_method))
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    train_idx, val_idx = split_indices(total, val_ratio, seed)
    indices = train_idx if split == "train" else val_idx

    if dataset_type == "mimic_raw":
        bandpass = data.get("bandpass", DEFAULT_BANDPASS)
        dataset = MimicRawDataset(
            meta_root,
            fs_out=int(data.get("fs", 100)),
            time_len=time_len,
            scale_method=scale_method,
            bandpass=tuple(bandpass) if bandpass is not None else None,
            indices=indices,
            lead_order=cfg.get("model", {}).get("lead_order", "mimic"),
        )
    else:
        dataset = ECGDataset(meta_root, time_len, scale_method, indices=indices)

    workers = int(data.get("num_workers", 4))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        drop_last=split == "train",
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=_collate,
    )


def make_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader]:
    """Create both train and validation DataLoaders.

    Args:
        cfg: Full config dict (``Config.raw``), sections data / train / run / model.
    """
    return make_dataloader(cfg, "train"), make_dataloader(cfg, "val")
