"""Route B (QRS-T inter-loop relation), Tier 1: how depolarisation relates to repolarisation.

Plan section 4. Three nested levels of the relation, each read beside the frozen LVCG
embedding:

    --level axis        the rotation from the QRS axis to the T axis, and its angle
                        (the spatial QRS-T angle)
    --level plane       adds the rotation between the two loop normals
    --level trajectory  adds a per-step relation between the two resampled loops
    --variant v0        the branch switched off: this stage's baseline

Besides macro AUROC the script reports every class, because the plan forbids narrating
only the class that improved most, and it runs the **mechanistic test**: the spatial
QRS-T angle is computed from the geometry alone, then decoded linearly from the frozen
embedding and from the proposed one. A representation that really encodes the inter-loop
geometry should make that angle easier to read off.

**Delineation (plan B1).** This repository has no QRS/T delineation; the windows are
R-peak relative fractions of the beat, which the plan allows only as a Tier 1
exploratory setting. Every result here carries that caveat, and the windows are printed
and stored with each row.

    python scripts/train_qrst.py --config configs/eval/qrst_b.yaml --level axis --seed 42
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

from lvcg.quaternion.interloop import LEVELS, QRSTProbe, spatial_qrst_angle  # noqa: E402
from probing.datasets import create_provider  # noqa: E402
from probing.encoders.lvcg_encoder import LVCGEncoder  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO_ROOT, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_qdf, train_qdt = _load("train_qdf"), _load("train_qdt")
LABELS, metrics = train_qdf.LABELS, train_qdf.metrics


def _inputs(tensors, index, device):
    beats, _rr, mask, _reference, tokens, steps, rhythm, _labels = tensors
    return (beats[index].to(device), tokens[index].to(device), mask[index].to(device),
            steps[index].to(device), rhythm[index].to(device))


def _forward(model, tensors, index, device):
    return model(*_inputs(tensors, index, device))


@torch.no_grad()
def evaluate(model, tensors, device, batch_size):
    model.eval()
    labels = tensors[-1]
    probabilities = []
    for start in range(0, len(labels), batch_size):
        index = torch.arange(start, min(start + batch_size, len(labels)))
        probabilities.append(torch.sigmoid(_forward(model, tensors, index, device)).cpu())
    return metrics(labels.numpy(), torch.cat(probabilities).numpy())


@torch.no_grad()
def collect(model, tensors, device, batch_size, windows):
    """(e_base, e_relation, spatial QRS-T angle) for every record, all detached."""
    model.eval()
    bases, relations, angles = [], [], []
    for start in range(0, len(tensors[-1]), batch_size):
        index = torch.arange(start, min(start + batch_size, len(tensors[-1])))
        base, relation = model.embeddings(*_inputs(tensors, index, device))
        bases.append(base.cpu())
        relations.append(relation.cpu())
        angles.append(spatial_qrst_angle(tensors[0][index].to(device), tensors[2][index].to(device),
                                         windows).cpu())
    return torch.cat(bases), torch.cat(relations), torch.cat(angles)


def train(model, data, device, probe_cfg, seed):
    """Route B's own loop: its probe takes different arguments from route A's."""
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


def linear_decoding(train_features, train_target, test_features, test_target, ridge=1.0):
    """R^2 of a ridge fit -- how linearly readable the target is from the features."""
    x = torch.cat((train_features, torch.ones(len(train_features), 1)), dim=1).double()
    y = train_target.double().unsqueeze(1)
    gram = x.T @ x + ridge * torch.eye(x.shape[1], dtype=x.dtype)
    weights = torch.linalg.solve(gram, x.T @ y)
    test = torch.cat((test_features, torch.ones(len(test_features), 1)), dim=1).double()
    prediction = (test @ weights).squeeze(1)
    target = test_target.double()
    residual = (target - prediction).square().sum()
    total = (target - target.mean()).square().sum()
    return float(1 - residual / total.clamp_min(1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "eval", "qrst_b.yaml"))
    parser.add_argument("--level", choices=LEVELS, default="axis")
    parser.add_argument("--variant", choices=("relation", "v0"), default="relation")
    parser.add_argument("--struct", choices=QRSTProbe.STRUCTS, default="mean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--results", default="probing/results/qrst_b.csv")
    parser.add_argument("--cache", default="probing/results/feature_cache")
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    checkpoint = args.checkpoint or cfg["pretrained"]["checkpoint"]
    probe_cfg, interloop_cfg = cfg.get("probe", {}), dict(cfg.get("interloop", {}))
    windows = interloop_cfg.pop("windows", None)
    device = torch.device(args.device)
    train_qdf.set_seed(args.seed)

    print(f"Level {args.level} | variant {args.variant} | struct {args.struct} | "
          f"seed {args.seed} | ratio {args.ratio:.0%}")
    print(f"Windows (R-peak relative fractions of the beat, exploratory): "
          f"{windows or 'QRS 0.00-0.12, T 0.15-0.55'}")

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
    model = QRSTProbe(
        state_generator=backbone.state_generator,
        norm_struct=backbone.norm_struct,
        norm_dynamic=backbone.norm_dynamic,
        num_classes=bundle.num_classes,
        struct=args.struct,
        scale=0.0 if args.variant == "v0" else 1.0,
        level=args.level,
        windows=windows,
        **{k: v for k, v in interloop_cfg.items() if k in {
            "embedding_dim", "hidden", "kernel", "dropout", "steps", "min_magnitude_fraction"}},
    ).to(device)
    counts = model.parameter_counts()
    print(f"Trainable {counts['trainable_total']:,} "
          f"(inter-loop branch {counts['interloop_branch']:,}, head {counts['head']:,})")

    started = time.perf_counter()
    batch_size = int(probe_cfg.get("batch_size", 256))
    best, curve = train(model, data, device, probe_cfg, args.seed)
    test = evaluate(model, data["test"], device, batch_size)

    # The mechanistic test: how linearly readable is the spatial QRS-T angle?
    train_base, train_relation, train_angle = collect(model, data["train"], device, batch_size, windows)
    test_base, test_relation, test_angle = collect(model, data["test"], device, batch_size, windows)
    decoding = {
        "decode_r2_base": linear_decoding(train_base, train_angle, test_base, test_angle),
        "decode_r2_full": linear_decoding(
            torch.cat((train_base, train_relation), 1), train_angle,
            torch.cat((test_base, test_relation), 1), test_angle),
        "qrst_angle_mean": float(test_angle.mean()),
        "qrst_angle_std": float(test_angle.std()),
    }
    seconds = time.perf_counter() - started

    print(f"\n  best epoch {best['epoch']} | val macro AUROC {best['val_macro_auroc']:.4f}")
    print(f"  TEST macro AUROC {test['macro_auroc']:.4f} | micro {test['micro_auroc']:.4f} "
          f"| macro F1 {test['macro_f1']:.4f}")
    print("  per label: " + ", ".join(
        f"{name} {auroc:.4f}" for name, auroc in zip(label_names, test["per_label_auroc"])))
    print(f"  spatial QRS-T angle: mean {decoding['qrst_angle_mean']:.1f} deg, "
          f"sd {decoding['qrst_angle_std']:.1f}")
    print(f"  linear decoding of that angle: R2 {decoding['decode_r2_base']:.3f} from e_base, "
          f"{decoding['decode_r2_full']:.3f} with the branch "
          f"({decoding['decode_r2_full'] - decoding['decode_r2_base']:+.3f})")

    row = {
        "model": "v0" if args.variant == "v0" else "qrst", "level": args.level,
        "struct": args.struct, "tag": args.tag,
        "windows": json.dumps(windows) if windows else "default",
        "label_ratio": args.ratio, "seed": args.seed,
        "test_macro_auroc": test["macro_auroc"], "test_micro_auroc": test["micro_auroc"],
        "test_macro_f1": test["macro_f1"], "test_micro_f1": test["micro_f1"],
        "val_macro_auroc": best["val_macro_auroc"], "best_epoch": best["epoch"],
        "trainable_params": counts["trainable_total"],
        **{key: round(value, 4) for key, value in decoding.items()},
        "checkpoint": checkpoint, "seconds": round(seconds, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for name, auroc, f1 in zip(label_names, test["per_label_auroc"], test["per_label_f1"]):
        row[f"auroc_{name}"], row[f"f1_{name}"] = auroc, f1

    results_path = args.results if os.path.isabs(args.results) else os.path.join(REPO_ROOT, args.results)
    train_qdf.append_row(results_path, row)
    curve_path = results_path.replace(
        ".csv", f"_curve_{args.variant}-{args.level}{args.tag}_s{args.seed}.json")
    with open(curve_path, "w", encoding="utf-8") as handle:
        json.dump({"row": row, "curve": curve}, handle, indent=1)
    print(f"\nAppended to {results_path}; curve in {os.path.basename(curve_path)}")


if __name__ == "__main__":
    main()
