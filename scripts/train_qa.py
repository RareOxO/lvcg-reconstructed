"""V6 (QA-LVCG), Tier 1: a constrained SO(3) correction of the lead geometry.

Plan section 4: keep the physical structure of the lead matrix and let the model only
turn it, ``A_Q = A_0 R(q_A)`` with ``q_A`` identity-initialised. The plan's order is
global first, then sample-conditioned with a bounded correction angle.

* ``--mode fixed``        the published geometry: this stage's V0 reference
* ``--mode global``       one rotation for the whole dataset
* ``--mode conditioned``  one per record, bounded by ``--max-degrees``

Recovering the VCG from ``A_Q`` equals rotating the VCG recovered from ``A_0`` (proved in
``tests/test_quaternion_v6.py`` against the released pseudo-inverse), and rotation
commutes with beat segmentation, so this script rotates V2's cached beat patches instead
of running the backbone again. The training loop, the evaluation and the rotation sweep
are V5's, so the two stages are directly comparable.

    python scripts/train_qa.py --config configs/eval/qa_v6.yaml --mode global --seed 42
"""

from __future__ import annotations

import argparse
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

from lvcg.data.angle import get_lead_directions  # noqa: E402
from lvcg.quaternion.geometry import QAProbe  # noqa: E402
from probing.datasets import create_provider  # noqa: E402
from probing.encoders.lvcg_encoder import LVCGEncoder  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO_ROOT, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_qdf, train_qdt, train_canon = _load("train_qdf"), _load("train_qdt"), _load("train_canon")
LABELS = train_qdf.LABELS


def geometry_report(model, directions, tensors, device, limit=512):
    """How far the correction turned the lead matrix, in degrees and in lead direction."""
    if model.mode == "fixed":
        return {"geometry_angle_deg": 0.0, "lead_shift_deg": 0.0}
    with torch.no_grad():
        beats = tensors[0][:limit].to(device)
        mask = tensors[2][:limit].to(device)
        rotated = model.geometry(directions.to(device), beats, mask)
        if rotated.dim() == 3:
            rotated = rotated.mean(dim=0)
        cosine = (rotated * directions.to(device)).sum(-1).clamp(-1.0, 1.0)
        shift = torch.arccos(cosine).mean() * 180.0 / torch.pi
        angle = model.pose_angles(beats, mask).mean()
    return {"geometry_angle_deg": float(angle), "lead_shift_deg": float(shift)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "eval", "qa_v6.yaml"))
    parser.add_argument("--mode", choices=QAProbe.MODES, default="global")
    parser.add_argument("--max-degrees", type=float, help="bound on the correction; 0 leaves it free")
    parser.add_argument("--geodesic-weight", type=float,
                        help="penalty on the squared correction angle (plan: geodesic regularisation)")
    parser.add_argument("--augment", action="store_true", help="random training rotations")
    parser.add_argument("--eval-rotations", type=int, nargs="*", default=list(train_canon.DEFAULT_ANGLES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--results", default="probing/results/qa_v6.csv")
    parser.add_argument("--cache", default="probing/results/feature_cache")
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    checkpoint = args.checkpoint or cfg["pretrained"]["checkpoint"]
    probe_cfg, geometry_cfg = cfg.get("probe", {}), dict(cfg.get("geometry", {}))
    max_degrees = args.max_degrees if args.max_degrees is not None else geometry_cfg.get("max_degrees")
    max_degrees = max_degrees or None  # 0 means "unbounded", which the plan asks to test
    geodesic_weight = (args.geodesic_weight if args.geodesic_weight is not None
                       else float(geometry_cfg.get("geodesic_weight", 0.0)))
    device = torch.device(args.device)
    train_qdf.set_seed(args.seed)

    print(f"Mode {args.mode} | max_degrees {max_degrees or 'unbounded'} | "
          f"geodesic_weight {geodesic_weight} | augment {args.augment} | "
          f"seed {args.seed} | ratio {args.ratio:.0%}")

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
    data = train_qdt.cached_tensors(
        encoder, {"train": bundle.train_loader, "val": bundle.val_loader, "test": bundle.test_loader},
        device, args.cache, checkpoint, args.ratio)

    backbone = encoder.backbone
    model = QAProbe(
        beat_encoder=backbone.beat_encoder,
        state_generator=backbone.state_generator,
        norm_struct=backbone.norm_struct,
        norm_dynamic=backbone.norm_dynamic,
        num_classes=bundle.num_classes,
        mode=args.mode,
        max_degrees=max_degrees,
        **{k: v for k, v in geometry_cfg.items() if k in {"pose_dim", "pose_hidden"}},
    ).to(device)
    counts = model.parameter_counts()
    print(f"Trainable {counts['trainable_total']:,} "
          f"(geometry {counts['geometry']:,}, head {counts['head']:,})")

    started = time.perf_counter()
    best, curve = train_canon.train(model, data, device, probe_cfg, args.seed,
                                    augment=args.augment, identity_weight=geodesic_weight)
    batch_size = int(probe_cfg.get("batch_size", 256))
    test = train_canon.evaluate(model, data["test"], device, batch_size)
    directions = get_lead_directions(cfg.get("lead_order", "mimic"), as_tensor=True)
    report = geometry_report(model, directions, data["test"], device)
    poses = train_canon.pose_report(model, data["test"], device, batch_size)
    sweep = {d: train_canon.evaluate(model, data["test"], device, batch_size, d, seed=1000 + d)
             for d in args.eval_rotations}
    seconds = time.perf_counter() - started

    print(f"\n  best epoch {best['epoch']} | val macro AUROC {best['val_macro_auroc']:.4f}")
    print(f"  TEST macro AUROC {test['macro_auroc']:.4f} | micro {test['micro_auroc']:.4f} "
          f"| macro F1 {test['macro_f1']:.4f}")
    print("  per label: " + ", ".join(
        f"{name} {auroc:.4f}" for name, auroc in zip(label_names, test["per_label_auroc"])))
    if model.mode != "fixed":
        print(f"  geometry: correction {report['geometry_angle_deg']:.1f} deg, "
              f"mean lead direction moved {report['lead_shift_deg']:.1f} deg "
              f"(per record median {poses['pose_median_deg']:.1f}, max {poses['pose_max_deg']:.1f})")
    if sweep:
        reference = sweep.get(0, test)["macro_auroc"]
        print("  rotation sweep: " + ", ".join(
            f"{d}deg {s['macro_auroc'] * 100:.2f} ({(s['macro_auroc'] - reference) * 100:+.2f})"
            for d, s in sweep.items()))

    row = {
        "model": "v0" if args.mode == "fixed" else "qa", "mode": args.mode,
        "augment": int(args.augment), "tag": args.tag,
        "max_degrees": max_degrees or 0, "geodesic_weight": geodesic_weight,
        "label_ratio": args.ratio, "seed": args.seed,
        "test_macro_auroc": test["macro_auroc"], "test_micro_auroc": test["micro_auroc"],
        "test_macro_f1": test["macro_f1"], "test_micro_f1": test["micro_f1"],
        "val_macro_auroc": best["val_macro_auroc"], "best_epoch": best["epoch"],
        "trainable_params": counts["trainable_total"],
        **report, **poses,
        **{f"rot{d}_macro_auroc": s["macro_auroc"] for d, s in sweep.items()},
        "checkpoint": checkpoint, "seconds": round(seconds, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for name, auroc, f1 in zip(label_names, test["per_label_auroc"], test["per_label_f1"]):
        row[f"auroc_{name}"], row[f"f1_{name}"] = auroc, f1

    results_path = args.results if os.path.isabs(args.results) else os.path.join(REPO_ROOT, args.results)
    train_qdf.append_row(results_path, row)
    curve_path = results_path.replace(".csv", f"_curve_{args.mode}{args.tag}_s{args.seed}.json")
    with open(curve_path, "w", encoding="utf-8") as handle:
        json.dump({"row": row, "curve": curve,
                   "sweep": {str(d): s["macro_auroc"] for d, s in sweep.items()}}, handle, indent=1)
    print(f"\nAppended to {results_path}; curve in {os.path.basename(curve_path)}")


if __name__ == "__main__":
    main()
