"""
LVCG pretraining on MIMIC-IV ECG (self-supervised reconstruction).

The objective and the optimisation are the release's. Added around them:

* Logging. ``logs/<run_id>/train_log.jsonl`` gets one JSON object per event:
  ``train`` every ``train.log_interval`` steps (all five loss terms averaged over the
  interval, learning rate, gradient norm before clipping, speed, elapsed time), ``val``
  every ``train.eval_interval`` steps, ``checkpoint`` for every save, and ``start`` /
  ``resume`` markers. ``run_info.json`` records the config, environment and data sizes.
  The progress bar shows all five terms too; the release showed three.
* Validation. The release built a validation loader and never used it. Every
  ``eval_interval`` steps the same first ``train.eval_batches`` validation batches are
  scored with the same visible-lead masks each time, so the numbers are comparable
  across the run; ``eval_interval: 0`` turns it off.
* Resuming. Every ``train.resume_interval`` steps ``last.pt`` is written atomically with
  the model, optimiser, step and every random number generator state, and
  ``--resume`` continues from it. The training order is a seeded shuffle per epoch
  (``train.seed``), where the release's was unseeded, so a resumed run starts at the
  exact batch it stopped at and draws the same lead masks and dropout it would have
  drawn: resuming is a continuation, not a restart.
* Overwrite protection. A fresh run refuses to start in a checkpoint directory that
  already holds checkpoints; resume it or choose another run id.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import time
from typing import Dict, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from lvcg.data.pipeline import make_dataloaders
from lvcg.models import build_model
from lvcg.models.lvcg import base_beat_loss, beat_level_loss, temporal_loss
from lvcg.models.utils.loss import masked_reconstruction_loss, random_lead_mask
from lvcg.utils.config import Config, add_cli_overrides, apply_overrides, load_config
from lvcg.utils.run_id import ensure_run_dirs

LOSS_TERMS = ("loss", "recon", "temporal", "beat", "base")


class EpochSampler(Sampler):
    """A seeded permutation per epoch that can start part-way through.

    ``set_position(epoch, skip)`` chooses the epoch and how many leading samples of its
    permutation to skip; the DataLoader asks for a fresh iterator at the start of every
    epoch, and after the first one ``skip`` returns to zero.
    """

    def __init__(self, size: int, seed: int):
        self.size = size
        self.seed = seed
        self.epoch = 0
        self.skip = 0

    def set_position(self, epoch: int, skip: int) -> None:
        self.epoch, self.skip = epoch, skip

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.size, generator=generator)[self.skip :]
        self.skip = 0
        return iter(order.tolist())

    def __len__(self) -> int:
        return self.size - self.skip


def _seeded_train_loader(loader: DataLoader, seed: int) -> DataLoader:
    sampler = EpochSampler(len(loader.dataset), seed)
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=sampler,
        drop_last=loader.drop_last,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        persistent_workers=loader.num_workers > 0,
        collate_fn=loader.collate_fn,
        # Its own generator: starting a new pass otherwise draws the workers' base seed
        # from the global generator, one extra draw that would shift every lead mask
        # and dropout pattern after a resume.
        generator=torch.Generator().manual_seed(seed),
    )


def _rng_state(device: torch.device) -> Dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _set_rng_state(state: Dict[str, object], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def _atomic_save(payload: Dict[str, object], path: str) -> None:
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _git_commit() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def compute_losses(model, ecg, num_visible, lambdas):
    """The release's objective, unchanged. Returns (total, {term: detached value})."""
    B = ecg.shape[0]
    visible_indices, mask = random_lead_mask(B, num_visible=num_visible, device=ecg.device)
    outputs = model.forward_train(ecg, visible_indices)

    loss_recon = masked_reconstruction_loss(outputs["recon"], ecg, mask)
    loss_temporal = temporal_loss(
        outputs["states_pred"],
        outputs["states_real"],
        outputs["beat_mask"],
    )
    loss_base = base_beat_loss(outputs["V_base_hat"], outputs["V_base"])
    # V_beats is full length (N) in the GRU path but core-only (N-2) in the
    # TTT path, while V_hat_beats is always length N; align before comparing.
    V_hat_beats = outputs["V_hat_beats"]
    V_beats = outputs["V_beats"]
    if V_hat_beats.shape[1] == V_beats.shape[1]:
        loss_beat = beat_level_loss(V_hat_beats, V_beats, outputs["beat_mask_full"])
    else:
        loss_beat = beat_level_loss(
            V_hat_beats[:, 1:-1],
            V_beats,
            outputs["beat_mask_full"][:, 1:-1],
        )

    loss = (
        loss_recon
        + lambdas["temporal"] * loss_temporal
        + lambdas["beat"] * loss_beat
        + lambdas["base"] * loss_base
    )
    terms = {
        "loss": loss.detach(),
        "recon": loss_recon.detach(),
        "temporal": loss_temporal.detach(),
        "beat": loss_beat.detach(),
        "base": loss_base.detach(),
    }
    return loss, terms


@torch.no_grad()
def evaluate(model, loader, device, num_visible, lambdas, batches, seed):
    """Mean loss terms on the first ``batches`` validation batches, with fixed masks.

    The random state is forked, so evaluating never changes what training draws next.
    """
    was_training = model.training
    model.eval()
    sums = {term: 0.0 for term in LOSS_TERMS}
    count = 0
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for index, batch in enumerate(loader):
            if index >= batches:
                break
            ecg = batch["ecg"].to(device, non_blocking=True)
            _, terms = compute_losses(model, ecg, num_visible, lambdas)
            for term in LOSS_TERMS:
                sums[term] += terms[term].item() * ecg.shape[0]
            count += ecg.shape[0]
    model.train(was_training)
    return {term: value / max(count, 1) for term, value in sums.items()}, count


def train(cfg: Config, resume: Optional[str] = None) -> str:
    requested = cfg.train.get("device", "auto")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    print(f"Using device: {device}")

    run_id = f"{cfg.run.get('m', 'm5')}{cfg.run.get('s', 's1')}{cfg.run.get('k', 'k1')}"
    ckpt_dir = ensure_run_dirs(cfg.run.get("checkpoint_root", "./checkpoints"), run_id)
    log_dir = ensure_run_dirs(cfg.run.get("log_root", "./logs"), run_id)
    print(f"Run ID: {run_id}")
    print(f"Checkpoint dir: {ckpt_dir}")
    print(f"Log dir: {log_dir}")

    last_path = os.path.join(ckpt_dir, "last.pt")
    resume_path = last_path if resume == "auto" else resume
    existing = sorted(n for n in os.listdir(ckpt_dir) if n.endswith(".pt"))
    if resume_path is None and existing:
        raise SystemExit(
            f"{ckpt_dir} already holds {existing}. Pass --resume to continue that run, "
            "or choose a new run id (e.g. --run.m) to start another."
        )

    train_cfg = cfg.train
    seed = int(train_cfg.get("seed", 42))
    if resume_path is None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    model = build_model(cfg).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {parameters:,}")

    train_loader, val_loader = make_dataloaders(cfg.raw)
    train_loader = _seeded_train_loader(train_loader, seed)
    sampler: EpochSampler = train_loader.sampler
    print(
        f"Train samples: {len(train_loader.dataset)}, "
        f"Val samples: {len(val_loader.dataset)}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 5e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    lambdas = {
        "temporal": float(train_cfg.get("lambda_temporal", 0.1)),
        "beat": float(train_cfg.get("lambda_beat", 1.0)),
        "base": float(train_cfg.get("lambda_base", 1.0)),
    }
    num_visible = int(train_cfg.get("num_visible", 3))

    max_steps = int(train_cfg.get("max_steps", 500000))
    log_interval = int(train_cfg.get("log_interval", 100))
    save_interval = int(train_cfg.get("save_interval", 50000))
    eval_interval = int(train_cfg.get("eval_interval", 1000))
    eval_batches = int(train_cfg.get("eval_batches", 50))
    resume_interval = int(train_cfg.get("resume_interval", 2000))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    steps_per_epoch = len(train_loader.dataset) // train_loader.batch_size
    if steps_per_epoch == 0:
        raise SystemExit("Fewer training records than one batch")

    global_step = 0
    log_path = os.path.join(log_dir, "train_log.jsonl")
    if resume_path is not None:
        if not os.path.exists(resume_path):
            raise SystemExit(f"Nothing to resume: {resume_path} does not exist")
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        saved = state.get("config", {})
        for section, key in (("train", "batch_size"), ("train", "lr"), ("train", "seed"), ("data", "meta_root")):
            if saved.get(section, {}).get(key) != cfg.raw.get(section, {}).get(key):
                raise SystemExit(
                    f"Refusing to resume: {section}.{key} was {saved.get(section, {}).get(key)!r}, "
                    f"now {cfg.raw.get(section, {}).get(key)!r}"
                )
        if saved.get("model") != cfg.raw.get("model"):
            raise SystemExit("Refusing to resume: the model section of the config changed")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = int(state["global_step"])
        _set_rng_state(state["rng"], device)
        print(f"Resumed from {resume_path} at step {global_step}")

    def log(event: Dict[str, object]) -> None:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

    info = {
        "run_id": run_id,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "resumed_from_step": global_step if resume_path else None,
        "config": cfg.raw,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "git_commit": _git_commit(),
        "parameters": parameters,
        "train_records": len(train_loader.dataset),
        "val_records": len(val_loader.dataset),
        "steps_per_epoch": steps_per_epoch,
    }
    info_name = "run_info.json" if resume_path is None else f"run_info_resume_{global_step}.json"
    with open(os.path.join(log_dir, info_name), "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2, default=str)
    log({"type": "resume" if resume_path else "start", "step": global_step, "time": info["started"]})

    def checkpoint_payload() -> Dict[str, object]:
        return {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "rng": _rng_state(device),
            "config": cfg.raw,
        }

    model.train()
    pbar = tqdm(total=max_steps, initial=global_step, desc="Training")
    sums = {term: torch.zeros((), device=device) for term in LOSS_TERMS}
    grad_norm_sum = torch.zeros((), device=device)
    interval_steps, interval_start, run_start = 0, time.perf_counter(), time.perf_counter()

    while global_step < max_steps:
        sampler.set_position(
            global_step // steps_per_epoch,
            (global_step % steps_per_epoch) * train_loader.batch_size,
        )
        for batch in train_loader:
            if global_step >= max_steps:
                break

            ecg = batch["ecg"].to(device, non_blocking=True)
            optimizer.zero_grad()
            loss, terms = compute_losses(model, ecg, num_visible, lambdas)
            loss.backward()

            if grad_clip > 0:
                grad_norm_sum += nn.utils.clip_grad_norm_(model.parameters(), grad_clip).detach()

            optimizer.step()
            global_step += 1
            pbar.update(1)
            for term in LOSS_TERMS:
                sums[term] += terms[term]
            interval_steps += 1

            if global_step % log_interval == 0:
                elapsed = time.perf_counter() - interval_start
                means = {term: (sums[term] / interval_steps).item() for term in LOSS_TERMS}
                log(
                    {
                        "type": "train",
                        "step": global_step,
                        "epoch": global_step / steps_per_epoch,
                        **{term: round(value, 6) for term, value in means.items()},
                        "grad_norm": round((grad_norm_sum / interval_steps).item(), 6) if grad_clip > 0 else None,
                        "lr": optimizer.param_groups[0]["lr"],
                        "steps_per_sec": round(interval_steps / elapsed, 3),
                        "elapsed_hours": round((time.perf_counter() - run_start) / 3600, 4),
                    }
                )
                pbar.set_postfix({term: f"{means[term]:.4f}" for term in LOSS_TERMS})
                sums = {term: torch.zeros((), device=device) for term in LOSS_TERMS}
                grad_norm_sum = torch.zeros((), device=device)
                interval_steps, interval_start = 0, time.perf_counter()

            if eval_interval > 0 and global_step % eval_interval == 0:
                values, count = evaluate(model, val_loader, device, num_visible, lambdas, eval_batches, seed)
                log({"type": "val", "step": global_step, "records": count,
                     **{term: round(value, 6) for term, value in values.items()}})

            if resume_interval > 0 and global_step % resume_interval == 0:
                _atomic_save(checkpoint_payload(), last_path)
                log({"type": "checkpoint", "step": global_step, "path": last_path})

            if global_step % save_interval == 0:
                ckpt_path = os.path.join(ckpt_dir, f"step_{global_step}.pt")
                _atomic_save(checkpoint_payload(), ckpt_path)
                log({"type": "checkpoint", "step": global_step, "path": ckpt_path})
                print(f"\nSaved checkpoint: {ckpt_path}")

    pbar.close()

    final_path = os.path.join(ckpt_dir, "final.pt")
    _atomic_save(checkpoint_payload(), final_path)
    _atomic_save(checkpoint_payload(), last_path)
    log({"type": "checkpoint", "step": global_step, "path": final_path})
    print(f"Saved checkpoint: {final_path}")
    return ckpt_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="LVCG pretraining")
    parser = add_cli_overrides(parser)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Continue a run: bare --resume uses checkpoints/<run_id>/last.pt, or give a path.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
