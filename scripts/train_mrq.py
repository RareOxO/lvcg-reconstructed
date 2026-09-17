"""V3 (MRQ-LVCG), Tier 1: magnitude and rotation as separate branches on the frozen LVCG.

Plan section 4/6.2: the cardiac vector is split into P_t = r_t u_t, and the ablation
matrix is a matter of listing which components the head reads:

    --components vcg magnitude rotation      the full model
    --components vcg rotation                VCG + rotation
    --components vcg magnitude               VCG + magnitude
    --components magnitude rotation          no pretrained embedding
    --components rotation                    rotation only
    --components magnitude                   magnitude only
    --components vcg                         V0: the pretrained embedding alone
    --components vcg control                 the parameter-matched real-valued control
    --components vcg rotation direction      the diagnostic V1 and V2 asked for

Everything else is V1's: the backbone runs once, frozen, and this script reuses the very
same feature cache, so a variant trains in under a minute. Selection is on validation
macro AUROC; the test fold is read once, at the end.

    python scripts/train_mrq.py --config configs/eval/mrq_v3.yaml --components vcg magnitude rotation
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time

import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lvcg.quaternion.mrq import COMPONENTS, MRQProbe  # noqa: E402
from probing.datasets import create_provider  # noqa: E402
from probing.encoders.lvcg_encoder import LVCGEncoder  # noqa: E402

# V1's script owns the frozen pass, its cache, the training loop and the metrics; V3
# reuses all four unchanged, so the two stages are directly comparable and the cached
# embeddings are shared rather than recomputed.
_spec = importlib.util.spec_from_file_location(
    "train_qdf", os.path.join(REPO_ROOT, "scripts", "train_qdf.py"))
train_qdf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_qdf)
LABELS = train_qdf.LABELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "eval", "mrq_v3.yaml"))
    parser.add_argument("--components", nargs="+", default=["vcg", "magnitude", "rotation"],
                        help=f"vcg plus any of {list(COMPONENTS)}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--results", default="probing/results/mrq_v3.csv")
    parser.add_argument("--cache", default="probing/results/feature_cache")
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    checkpoint = args.checkpoint or cfg["pretrained"]["checkpoint"]
    probe_cfg, quaternion_cfg = cfg.get("probe", {}), dict(cfg.get("quaternion", {}))
    device = torch.device(args.device)
    train_qdf.set_seed(args.seed)

    components = tuple(args.components)
    print(f"Components {' + '.join(components)} | seed {args.seed} | ratio {args.ratio:.0%}")

    provider_cfg = {
        "type": "benchmark", "dataset_name": "ptbxl_super_class",
        "raw_root": cfg["data"]["raw_root"],
        "splits_root": cfg["data"].get("splits_root", "probing/data_splits"),
        "norm_method": cfg["data"].get("norm_method", "zscore"),
    }
    bundle = create_provider("ptbxl_super_class", provider_cfg).build(
        label_ratio=args.ratio, batch_size=int(probe_cfg.get("batch_size", 256)),
        num_workers=int(probe_cfg.get("num_workers", 4)),
    )
    label_names = list(bundle.label_names) if bundle.label_names else list(LABELS)
    print(f"Train {len(bundle.train_loader.dataset)} | Val {len(bundle.val_loader.dataset)} | "
          f"Test {len(bundle.test_loader.dataset)} | classes {bundle.num_classes} {label_names}")

    encoder = LVCGEncoder(checkpoint).to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    data = train_qdf.cached_features(
        encoder, {"train": bundle.train_loader, "val": bundle.val_loader, "test": bundle.test_loader},
        device, args.cache, checkpoint, args.ratio,
    )

    model = MRQProbe(
        components=components,
        base_dim=data["train"][0].shape[1],
        num_classes=bundle.num_classes,
        fs=int(quaternion_cfg.pop("fs", 100)),
        **{k: v for k, v in quaternion_cfg.items() if k in {
            "embedding_dim", "hidden", "kernel", "dropout",
            "min_magnitude_fraction", "sign_continuity", "masked_pooling"}},
    ).to(device)
    counts = model.parameter_counts()
    print("Trainable {:,} ({})".format(
        counts["trainable_total"],
        ", ".join(f"{k.replace('branch_', '')} {v:,}" for k, v in counts.items() if k != "trainable_total")))

    started = time.perf_counter()
    best, curve = train_qdf.train(model, data, device, probe_cfg, args.seed)
    test = train_qdf.evaluate(model, data["test"], device, int(probe_cfg.get("batch_size", 256)))
    seconds = time.perf_counter() - started

    print(f"\n  best epoch {best['epoch']} | val macro AUROC {best['val_macro_auroc']:.4f}")
    print(f"  TEST macro AUROC {test['macro_auroc']:.4f} | micro {test['micro_auroc']:.4f} "
          f"| macro F1 {test['macro_f1']:.4f} | micro F1 {test['micro_f1']:.4f}")
    print("  per label: " + ", ".join(
        f"{name} {auroc:.4f}" for name, auroc in zip(label_names, test["per_label_auroc"])))

    row = {
        "model": "v0" if components == ("vcg",) else "mrq",
        "components": "+".join(components), "tag": args.tag,
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

    results_path = args.results if os.path.isabs(args.results) else os.path.join(REPO_ROOT, args.results)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    exists = os.path.exists(results_path)
    with open(results_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    curve_path = results_path.replace(
        ".csv", f"_curve_{'-'.join(components)}{args.tag}_s{args.seed}.json")
    with open(curve_path, "w", encoding="utf-8") as handle:
        json.dump({"row": row, "curve": curve}, handle, indent=1)
    print(f"\nAppended to {results_path}; curve in {os.path.basename(curve_path)}")


if __name__ == "__main__":
    main()
