"""V5 (QC-LVCG), Tier 1: canonicalise the latent VCG's pose in front of the frozen model.

Plan section 4: a per-record rotation is predicted and undone before the pretrained path
sees the trajectory. The pose head is identity-initialised, so training starts exactly at
V0, and the frozen beat encoder, GRU and norms are replayed on the canonicalised beats.

Three models share the command, and the third is what makes the second interpretable:

* ``--model canon``    pose estimated and undone
* ``--model v0``       no pose head: the frozen path untouched (the locked reference)
* ``--model augment``  no pose head, but trained on the same randomly rotated batches --
  the control the plan demands, because augmentation alone also buys robustness

``--augment`` turns random training rotations on for any model (it is implied by
``--model augment``), and ``--eval-rotations`` runs the robustness sweep of plan section
6.3, reporting test macro AUROC at each angle.

The rotation commutes with beat segmentation, so this script reuses V2's cached beat
patches rather than running the backbone again.

    python scripts/train_canon.py --config configs/eval/canon_v5.yaml --model canon --seed 42
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from typing import Dict

import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lvcg.quaternion.canon import CanonProbe, random_rotations, rotate_beats  # noqa: E402
from probing.datasets import create_provider  # noqa: E402
from probing.encoders.lvcg_encoder import LVCGEncoder  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO_ROOT, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_qdf, train_qdt = _load("train_qdf"), _load("train_qdt")
LABELS, metrics = train_qdf.LABELS, train_qdf.metrics
DEFAULT_ANGLES = (0, 5, 10, 30, 60, 90, 150)


def _forward(model, tensors, index, device, rotation=None):
    beats, rr, mask, _reference, _tokens, steps, rhythm, _labels = tensors
    batch = beats[index].to(device)
    if rotation is not None:
        batch = rotate_beats(batch, rotation.to(device))
    return model(batch, rr[index].to(device), mask[index].to(device),
                 steps[index].to(device), rhythm[index].to(device))


@torch.no_grad()
def evaluate(model, tensors, device, batch_size, degrees=None, seed=0) -> Dict[str, object]:
    """Test metrics, optionally with every record turned by a fixed angle about a random axis."""
    model.eval()
    labels = tensors[-1]
    generator = torch.Generator().manual_seed(seed)
    probabilities = []
    for start in range(0, len(labels), batch_size):
        index = torch.arange(start, min(start + batch_size, len(labels)))
        rotation = None if degrees in (None, 0) else random_rotations(len(index), degrees, generator)
        probabilities.append(torch.sigmoid(_forward(model, tensors, index, device, rotation)).cpu())
    return metrics(labels.numpy(), torch.cat(probabilities).numpy())


def train(model, data, device, probe_cfg, seed, augment=False, identity_weight=0.0):
    """Train the pose head and the linear head; keep the best validation macro AUROC."""
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
        total, angles = 0.0, []
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            if len(index) < 2:
                continue
            rotation = random_rotations(len(index), None, generator) if augment else None
            optimizer.zero_grad()
            logits = _forward(model, data["train"], index, device, rotation)
            loss = criterion(logits, labels[index].to(device))
            if identity_weight and model.canonicalize:
                beats = data["train"][0][index].to(device)
                if rotation is not None:
                    beats = rotate_beats(beats, rotation.to(device))
                degrees = model.pose_angles(beats, data["train"][2][index].to(device))
                loss = loss + identity_weight * (degrees / 180.0).square().mean()
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


@torch.no_grad()
def pose_report(model, tensors, device, batch_size=256, limit=2048):
    """What the pose head actually does: the distribution of its correction angles."""
    if not model.canonicalize:
        return {}
    model.eval()
    beats, _rr, mask = tensors[0], tensors[1], tensors[2]
    angles = []
    for start in range(0, min(len(beats), limit), batch_size):
        index = slice(start, min(start + batch_size, len(beats)))
        angles.append(model.pose_angles(beats[index].to(device), mask[index].to(device)).cpu())
    angles = torch.cat(angles)
    quantiles = torch.quantile(angles, torch.tensor([0.5, 0.9, 1.0]))
    return {"pose_median_deg": float(quantiles[0]), "pose_p90_deg": float(quantiles[1]),
            "pose_max_deg": float(quantiles[2])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "eval", "canon_v5.yaml"))
    parser.add_argument("--model", choices=("canon", "v0", "augment"), default="canon")
    parser.add_argument("--augment", action="store_true", help="random training rotations")
    parser.add_argument("--max-degrees", type=float, help="bound on the pose correction")
    parser.add_argument("--identity-weight", type=float, help="penalty on the correction angle")
    parser.add_argument("--eval-rotations", type=int, nargs="*", default=list(DEFAULT_ANGLES),
                        help="angles of the robustness sweep; empty disables it")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--results", default="probing/results/canon_v5.csv")
    parser.add_argument("--cache", default="probing/results/feature_cache")
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    checkpoint = args.checkpoint or cfg["pretrained"]["checkpoint"]
    probe_cfg, canon_cfg = cfg.get("probe", {}), dict(cfg.get("canonicalization", {}))
    max_degrees = args.max_degrees if args.max_degrees is not None else canon_cfg.get("max_degrees")
    identity_weight = (args.identity_weight if args.identity_weight is not None
                       else float(canon_cfg.get("identity_weight", 0.0)))
    augment = args.augment or args.model == "augment"
    device = torch.device(args.device)
    train_qdf.set_seed(args.seed)

    print(f"Model {args.model} | augment {augment} | max_degrees {max_degrees or 'unbounded'} "
          f"| identity_weight {identity_weight} | seed {args.seed} | ratio {args.ratio:.0%}")

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
    # V2's cache: beat patches, R-R, mask, reference, tokens, steps, rhythm, labels.
    data = train_qdt.cached_tensors(
        encoder, {"train": bundle.train_loader, "val": bundle.val_loader, "test": bundle.test_loader},
        device, args.cache, checkpoint, args.ratio)

    backbone = encoder.backbone
    model = CanonProbe(
        beat_encoder=backbone.beat_encoder,
        state_generator=backbone.state_generator,
        norm_struct=backbone.norm_struct,
        norm_dynamic=backbone.norm_dynamic,
        num_classes=bundle.num_classes,
        canonicalize=args.model == "canon",
        max_degrees=max_degrees,
        **{k: v for k, v in canon_cfg.items() if k in {"pose_dim", "pose_hidden"}},
    ).to(device)
    counts = model.parameter_counts()
    print(f"Trainable {counts['trainable_total']:,} (pose {counts['pose']:,}, head {counts['head']:,})")

    started = time.perf_counter()
    best, curve = train(model, data, device, probe_cfg, args.seed, augment, identity_weight)
    batch_size = int(probe_cfg.get("batch_size", 256))
    test = evaluate(model, data["test"], device, batch_size)
    poses = pose_report(model, data["test"], device, batch_size)
    sweep = {}
    for degrees in args.eval_rotations:
        sweep[degrees] = evaluate(model, data["test"], device, batch_size, degrees, seed=1000 + degrees)
    seconds = time.perf_counter() - started

    print(f"\n  best epoch {best['epoch']} | val macro AUROC {best['val_macro_auroc']:.4f}")
    print(f"  TEST macro AUROC {test['macro_auroc']:.4f} | micro {test['micro_auroc']:.4f} "
          f"| macro F1 {test['macro_f1']:.4f}")
    print("  per label: " + ", ".join(
        f"{name} {auroc:.4f}" for name, auroc in zip(label_names, test["per_label_auroc"])))
    if poses:
        print(f"  pose correction: median {poses['pose_median_deg']:.1f} deg, "
              f"p90 {poses['pose_p90_deg']:.1f}, max {poses['pose_max_deg']:.1f}")
    if sweep:
        reference = sweep.get(0, test)["macro_auroc"]
        print("  rotation sweep: " + ", ".join(
            f"{d}deg {s['macro_auroc'] * 100:.2f} ({(s['macro_auroc'] - reference) * 100:+.2f})"
            for d, s in sweep.items()))

    row = {
        "model": args.model, "augment": int(augment), "tag": args.tag,
        "max_degrees": max_degrees or 0, "identity_weight": identity_weight,
        "label_ratio": args.ratio, "seed": args.seed,
        "test_macro_auroc": test["macro_auroc"], "test_micro_auroc": test["micro_auroc"],
        "test_macro_f1": test["macro_f1"], "test_micro_f1": test["micro_f1"],
        "val_macro_auroc": best["val_macro_auroc"], "best_epoch": best["epoch"],
        "trainable_params": counts["trainable_total"],
        **{f"pose_{k}": v for k, v in poses.items()},
        **{f"rot{d}_macro_auroc": s["macro_auroc"] for d, s in sweep.items()},
        "checkpoint": checkpoint, "seconds": round(seconds, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for name, auroc, f1 in zip(label_names, test["per_label_auroc"], test["per_label_f1"]):
        row[f"auroc_{name}"], row[f"f1_{name}"] = auroc, f1

    results_path = args.results if os.path.isabs(args.results) else os.path.join(REPO_ROOT, args.results)
    train_qdf.append_row(results_path, row)
    curve_path = results_path.replace(".csv", f"_curve_{args.model}{args.tag}_s{args.seed}.json")
    with open(curve_path, "w", encoding="utf-8") as handle:
        json.dump({"row": row, "curve": curve,
                   "sweep": {str(d): s["macro_auroc"] for d, s in sweep.items()}}, handle, indent=1)
    print(f"\nAppended to {results_path}; curve in {os.path.basename(curve_path)}")


if __name__ == "__main__":
    main()
