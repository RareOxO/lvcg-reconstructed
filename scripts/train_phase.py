"""V4 (Phase-Q LVCG), Tier 1: the same dynamics pooled over QRS, T and the whole beat.

V1 and V3 asked which geometric quantity carries diagnostic information; V4 asks when in
the cardiac cycle it is carried. The architecture is V1's, the backbone is frozen, and a
variant differs only in the pooling window:

    --phases whole            V3's variant, tensor for tensor (the equivalence check)
    --phases qrs              depolarisation only
    --phases t                repolarisation only
    --phases qrs t            both, read separately
    --phases qrs t whole      the plan's full split

``--variant`` chooses the channels: ``mq`` is the plan's quaternion set, ``mo`` the
strongest set V3 found, ``cdf`` the Cartesian reference. Asking the phase question of
more than one representation is the point: a gain that appears only for one of them says
something different from a gain that appears for all.

The frozen pass caches e_base, the latent VCG and the R-peak positions, so every variant
trains in about a minute. Selection is on validation macro AUROC; the test fold is read
once, at the end.

    python scripts/train_phase.py --config configs/eval/phase_v4.yaml --phases qrs t whole
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from typing import Dict

import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lvcg.quaternion.phase import PHASES, PhaseProbe  # noqa: E402
from probing.datasets import create_provider  # noqa: E402
from probing.encoders.lvcg_encoder import LVCGEncoder  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "train_qdf", os.path.join(REPO_ROOT, "scripts", "train_qdf.py"))
train_qdf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_qdf)
LABELS, metrics = train_qdf.LABELS, train_qdf.metrics

MAX_PEAKS = 32


@torch.no_grad()
def frozen_tensors(encoder: LVCGEncoder, loader, device):
    """(e_base [N,640], vcg [N,3,1000], peaks [N,32], peak_mask [N,32], labels [N,5])."""
    from tqdm import tqdm

    backbone = encoder.backbone
    encoder.eval()
    parts = {name: [] for name in ("base", "vcg", "peaks", "peak_mask", "labels")}
    for batch in tqdm(loader, desc="    frozen pass", leave=False):
        ecg = encoder._resample(batch["ecg"].to(device), 500)
        directions = backbone.all_lead_directions.unsqueeze(0).expand(ecg.shape[0], -1, -1)
        peaks, peak_mask = backbone.beat_segmenter.r_peaks(ecg, backbone.rr_lead_idx, MAX_PEAKS)
        parts["base"].append(backbone.ext_ecg_emb(ecg).float().cpu())
        parts["vcg"].append(backbone.vcg_inverse(ecg, directions).float().cpu())
        parts["peaks"].append(peaks.cpu())
        parts["peak_mask"].append(peak_mask.cpu())
        parts["labels"].append(batch["label"].float())
    return tuple(torch.cat(parts[name]) for name in ("base", "vcg", "peaks", "peak_mask", "labels"))


def cached_tensors(encoder, loaders, device, cache_dir, checkpoint, ratio):
    key = hashlib.sha256(f"{os.path.abspath(checkpoint)}|{ratio}|v4".encode()).hexdigest()[:16]
    out = {}
    for split, loader in loaders.items():
        path = os.path.join(cache_dir, f"{split}_{key}.pt") if cache_dir else None
        if path and os.path.exists(path):
            out[split] = torch.load(path, weights_only=True)
            print(f"  {split}: cached {tuple(out[split][0].shape)}")
            continue
        started = time.perf_counter()
        out[split] = frozen_tensors(encoder, loader, device)
        print(f"  {split}: {tuple(out[split][0].shape)} in {time.perf_counter() - started:.1f}s")
        if path:
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(out[split], path)
    return out


def _forward(model, tensors, index, device):
    base, vcg, peaks, peak_mask, _ = tensors
    return model(base[index].to(device), vcg[index].to(device),
                 peaks[index].to(device), peak_mask[index].to(device))


@torch.no_grad()
def evaluate(model, tensors, device, batch_size) -> Dict[str, object]:
    model.eval()
    labels = tensors[-1]
    probabilities = []
    for start in range(0, len(labels), batch_size):
        index = torch.arange(start, min(start + batch_size, len(labels)))
        probabilities.append(torch.sigmoid(_forward(model, tensors, index, device)).cpu())
    return metrics(labels.numpy(), torch.cat(probabilities).numpy())


def train(model, data, device, probe_cfg, seed):
    batch_size = int(probe_cfg.get("batch_size", 256))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(probe_cfg.get("lr", 1e-3)), weight_decay=float(probe_cfg.get("weight_decay", 1e-4)))
    criterion = torch.nn.BCEWithLogitsLoss()
    labels = data["train"][-1]
    generator = torch.Generator().manual_seed(seed)
    best = {"val_macro_auroc": -1.0, "epoch": -1, "state": None}
    curve, patience = [], int(probe_cfg.get("patience", 5))

    for epoch in range(int(probe_cfg.get("max_epochs", 50))):
        model.train()
        order = torch.randperm(len(labels), generator=generator)
        total = 0.0
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            if len(index) < 2:
                continue
            optimizer.zero_grad()
            loss = criterion(_forward(model, data["train"], index, device), labels[index].to(device))
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        validation = evaluate(model, data["val"], device, batch_size)
        curve.append({"epoch": epoch, "train_loss": total / len(labels),
                      "val_macro_auroc": validation["macro_auroc"]})
        print(f"    epoch {epoch:>2} loss {curve[-1]['train_loss']:.4f} "
              f"val macro AUROC {validation['macro_auroc']:.4f}"
              + ("  <- best" if validation["macro_auroc"] > best["val_macro_auroc"] else ""))
        if validation["macro_auroc"] > best["val_macro_auroc"]:
            best = {"val_macro_auroc": validation["macro_auroc"], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= patience:
            print(f"    no improvement for {patience} epochs, stopping")
            break
    model.load_state_dict(best["state"])
    return best, curve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "eval", "phase_v4.yaml"))
    parser.add_argument("--phases", nargs="+", default=["qrs", "t", "whole"],
                        help=f"any of {list(PHASES)}; also accepts one comma separated argument")
    parser.add_argument("--variant", default="mq", help="channels: mq (plan), mo (V3's best), cdf, o, q, ...")
    parser.add_argument("--model", choices=("phase", "v0"), default="phase",
                        help="v0 switches the branch off, for the equivalence check")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--results", default="probing/results/phase_v4.csv")
    parser.add_argument("--cache", default="probing/results/feature_cache")
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    checkpoint = args.checkpoint or cfg["pretrained"]["checkpoint"]
    probe_cfg, quaternion_cfg = cfg.get("probe", {}), dict(cfg.get("quaternion", {}))
    windows = dict(cfg.get("phase_windows", {}))
    device = torch.device(args.device)
    train_qdf.set_seed(args.seed)

    phases = tuple(p for argument in args.phases for p in re.split(r"[\s,+]+", argument.strip()) if p)
    print(f"Variant {args.variant} | phases {' '.join(phases)} | seed {args.seed} "
          f"| ratio {args.ratio:.0%} | windows {windows or 'default'}")

    provider_cfg = {
        "type": "benchmark", "dataset_name": "ptbxl_super_class",
        "raw_root": cfg["data"]["raw_root"],
        "splits_root": cfg["data"].get("splits_root", "probing/data_splits"),
        "norm_method": cfg["data"].get("norm_method", "zscore"),
    }
    bundle = create_provider("ptbxl_super_class", provider_cfg).build(
        label_ratio=args.ratio, batch_size=int(probe_cfg.get("batch_size", 256)),
        num_workers=int(probe_cfg.get("num_workers", 4)))
    label_names = list(bundle.label_names) if bundle.label_names else list(LABELS)
    print(f"Train {len(bundle.train_loader.dataset)} | Val {len(bundle.val_loader.dataset)} | "
          f"Test {len(bundle.test_loader.dataset)} | classes {bundle.num_classes} {label_names}")

    encoder = LVCGEncoder(checkpoint).to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    data = cached_tensors(
        encoder, {"train": bundle.train_loader, "val": bundle.val_loader, "test": bundle.test_loader},
        device, args.cache, checkpoint, args.ratio)

    model = PhaseProbe(
        variant=args.variant, phases=phases,
        base_dim=data["train"][0].shape[1], num_classes=bundle.num_classes,
        fs=int(quaternion_cfg.pop("fs", 100)),
        windows=windows or None,
        scale=0.0 if args.model == "v0" else 1.0,
        **{k: v for k, v in quaternion_cfg.items() if k in {
            "embedding_dim", "hidden", "kernel", "dropout",
            "min_magnitude_fraction", "sign_continuity"}},
    ).to(device)
    counts = model.parameter_counts()
    coverage = model.phase_coverage(data["train"][2][:512].to(device), data["train"][3][:512].to(device),
                                    data["train"][1].shape[-1] - 1)
    print(f"Trainable {counts['trainable_total']:,} (branch {counts['phase_branch']:,})")
    print("Phase coverage of the record: " + ", ".join(f"{k} {v:.1%}" for k, v in coverage.items()))

    started = time.perf_counter()
    best, curve = train(model, data, device, probe_cfg, args.seed)
    test = evaluate(model, data["test"], device, int(probe_cfg.get("batch_size", 256)))
    seconds = time.perf_counter() - started

    print(f"\n  best epoch {best['epoch']} | val macro AUROC {best['val_macro_auroc']:.4f}")
    print(f"  TEST macro AUROC {test['macro_auroc']:.4f} | micro {test['micro_auroc']:.4f} "
          f"| macro F1 {test['macro_f1']:.4f} | micro F1 {test['micro_f1']:.4f}")
    print("  per label: " + ", ".join(
        f"{name} {auroc:.4f}" for name, auroc in zip(label_names, test["per_label_auroc"])))

    row = {
        "model": args.model, "variant": args.variant, "phases": "+".join(phases), "tag": args.tag,
        "label_ratio": args.ratio, "seed": args.seed,
        "test_macro_auroc": test["macro_auroc"], "test_micro_auroc": test["micro_auroc"],
        "test_macro_f1": test["macro_f1"], "test_micro_f1": test["micro_f1"],
        "val_macro_auroc": best["val_macro_auroc"], "best_epoch": best["epoch"],
        "trainable_params": counts["trainable_total"],
        "qrs_coverage": round(coverage["qrs"], 4), "t_coverage": round(coverage["t"], 4),
        "qrs_t_overlap": round(coverage["qrs_and_t"], 4),
        "checkpoint": checkpoint, "seconds": round(seconds, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for name, auroc, f1 in zip(label_names, test["per_label_auroc"], test["per_label_f1"]):
        row[f"auroc_{name}"], row[f"f1_{name}"] = auroc, f1

    results_path = args.results if os.path.isabs(args.results) else os.path.join(REPO_ROOT, args.results)
    train_qdf.append_row(results_path, row)
    curve_path = results_path.replace(
        ".csv", f"_curve_{args.variant}-{'-'.join(phases)}{args.tag}_s{args.seed}.json")
    with open(curve_path, "w", encoding="utf-8") as handle:
        json.dump({"row": row, "curve": curve}, handle, indent=1)
    print(f"\nAppended to {results_path}; curve in {os.path.basename(curve_path)}")


if __name__ == "__main__":
    main()
