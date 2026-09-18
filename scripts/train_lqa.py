"""V8 (LQA-LVCG), Tier 1: one learnable query per superclass over the dynamics sequence.

Plan section 4: five label queries read the quaternion sequence, each label pooling it
where it wants, with the pretrained embedding frozen and one small head per label rather
than five large MLPs.

    --mode label    the plan: one query per superclass
    --mode shared   one query for every label -- attention without label conditioning
    --mode mean     uniform masked averaging, i.e. V3's variant, tensor for tensor

The three modes separate "attention helps" from "label-specific attention helps", which
is the only way the plan's claim can fail cleanly. ``--variant`` chooses the channels;
the default is ``mo``, V3's strongest set, because V8 asks *where* each label looks
rather than *what* it reads.

The script also reports what the attention actually did: the share of each label's
attention mass falling inside the QRS and T windows of V4. That is the interpretability
V8 exists for, and it is reported whether or not the accuracy moves.

    python scripts/train_lqa.py --config configs/eval/lqa_v8.yaml --mode label --seed 42
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lvcg.quaternion.attention import MODES, LabelAttentionProbe  # noqa: E402
from lvcg.quaternion.phase import phase_masks  # noqa: E402
from lvcg.quaternion.qdf import DynamicEncoder  # noqa: E402
from probing.datasets import create_provider  # noqa: E402
from probing.encoders.lvcg_encoder import LVCGEncoder  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO_ROOT, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_qdf, train_phase = _load("train_qdf"), _load("train_phase")
LABELS = train_qdf.LABELS


@torch.no_grad()
def attention_report(model, tensors, device, fs, windows, label_names, limit=1024, batch_size=256):
    """Share of each label's attention mass inside the QRS and T windows."""
    if model.mode == "mean":
        return {}
    model.eval()
    base, vcg, peaks, peak_mask, _ = tensors
    totals = {}
    count = 0
    for start in range(0, min(len(vcg), limit), batch_size):
        index = slice(start, min(start + batch_size, len(vcg)))
        weights, _, _ = model.attention(vcg[index].to(device))
        masks = phase_masks(peaks[index].to(device), peak_mask[index].to(device),
                            vcg.shape[-1] - 1, fs, windows)
        for name, mask in masks.items():
            pooled = mask.unsqueeze(1).float()
            for _ in range(DynamicEncoder.STRIDES):
                pooled = F.max_pool1d(pooled, kernel_size=2, stride=2, ceil_mode=True)
            pooled = pooled.squeeze(1)[:, : weights.shape[-1]] > 0
            share = (weights * pooled.unsqueeze(1)).sum(-1)  # [B, K]
            totals[name] = totals.get(name, 0.0) + share.sum(0).cpu()
        count += weights.shape[0]
    report = {}
    for name, value in totals.items():
        if name == "whole":
            continue
        share = value / max(count, 1)
        for label, fraction in zip(label_names, share.tolist() if share.numel() > 1
                                   else share.tolist() * len(label_names)):
            report[f"attention_{name}_{label}"] = round(float(fraction), 4)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "eval", "lqa_v8.yaml"))
    parser.add_argument("--mode", choices=MODES, default="label")
    parser.add_argument("--variant", default="mo", help="channels: mo (V3's best), mq (plan), cdf, ...")
    parser.add_argument("--model", choices=("lqa", "v0"), default="lqa",
                        help="v0 switches the branch off, for the equivalence check")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--results", default="probing/results/lqa_v8.csv")
    parser.add_argument("--cache", default="probing/results/feature_cache")
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    checkpoint = args.checkpoint or cfg["pretrained"]["checkpoint"]
    probe_cfg, quaternion_cfg = cfg.get("probe", {}), dict(cfg.get("quaternion", {}))
    windows = dict(cfg.get("phase_windows", {})) or None
    device = torch.device(args.device)
    train_qdf.set_seed(args.seed)

    print(f"Mode {args.mode} | variant {args.variant} | seed {args.seed} | ratio {args.ratio:.0%}")

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
    # V4's cache: embedding, VCG, R peaks, peak mask, labels -- the peaks are what the
    # attention report needs.
    cached = train_phase.cached_tensors(
        encoder, {"train": bundle.train_loader, "val": bundle.val_loader, "test": bundle.test_loader},
        device, args.cache, checkpoint, args.ratio)
    data = {split: (tensors[0], tensors[1], tensors[4]) for split, tensors in cached.items()}

    fs = int(quaternion_cfg.pop("fs", 100))
    model = LabelAttentionProbe(
        variant=args.variant, mode=args.mode,
        base_dim=data["train"][0].shape[1], num_classes=bundle.num_classes, fs=fs,
        scale=0.0 if args.model == "v0" else 1.0,
        **{k: v for k, v in quaternion_cfg.items() if k in {
            "embedding_dim", "hidden", "kernel", "dropout",
            "min_magnitude_fraction", "sign_continuity"}},
    ).to(device)
    counts = model.parameter_counts()
    print(f"Trainable {counts['trainable_total']:,} "
          f"(branch {counts['branch']:,}, queries {counts['queries']:,})")

    started = time.perf_counter()
    best, curve = train_qdf.train(model, data, device, probe_cfg, args.seed)
    test = train_qdf.evaluate(model, data["test"], device, int(probe_cfg.get("batch_size", 256)))
    attention = attention_report(model, cached["test"], device, fs, windows, label_names)
    seconds = time.perf_counter() - started

    print(f"\n  best epoch {best['epoch']} | val macro AUROC {best['val_macro_auroc']:.4f}")
    print(f"  TEST macro AUROC {test['macro_auroc']:.4f} | micro {test['micro_auroc']:.4f} "
          f"| macro F1 {test['macro_f1']:.4f}")
    print("  per label: " + ", ".join(
        f"{name} {auroc:.4f}" for name, auroc in zip(label_names, test["per_label_auroc"])))
    if attention:
        print("  attention mass inside QRS / T:")
        for name in label_names:
            print(f"    {name:>5}: QRS {attention.get(f'attention_qrs_{name}', 0):.1%}, "
                  f"T {attention.get(f'attention_t_{name}', 0):.1%}")

    row = {
        "model": args.model, "mode": args.mode, "variant": args.variant, "tag": args.tag,
        "label_ratio": args.ratio, "seed": args.seed,
        "test_macro_auroc": test["macro_auroc"], "test_micro_auroc": test["micro_auroc"],
        "test_macro_f1": test["macro_f1"], "test_micro_f1": test["micro_f1"],
        "val_macro_auroc": best["val_macro_auroc"], "best_epoch": best["epoch"],
        "trainable_params": counts["trainable_total"],
        "checkpoint": checkpoint, "seconds": round(seconds, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for name, auroc, f1 in zip(label_names, test["per_label_auroc"], test["per_label_f1"]):
        row[f"auroc_{name}"], row[f"f1_{name}"] = auroc, f1
    for name in label_names:  # a fixed schema: zeros when the mode has no attention
        row[f"attention_qrs_{name}"] = attention.get(f"attention_qrs_{name}", 0.0)
        row[f"attention_t_{name}"] = attention.get(f"attention_t_{name}", 0.0)

    results_path = args.results if os.path.isabs(args.results) else os.path.join(REPO_ROOT, args.results)
    train_qdf.append_row(results_path, row)
    curve_path = results_path.replace(".csv", f"_curve_{args.mode}-{args.variant}{args.tag}_s{args.seed}.json")
    with open(curve_path, "w", encoding="utf-8") as handle:
        json.dump({"row": row, "curve": curve}, handle, indent=1)
    print(f"\nAppended to {results_path}; curve in {os.path.basename(curve_path)}")


if __name__ == "__main__":
    main()
