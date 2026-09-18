"""V1 (QDF-LVCG), Tier 1: a quaternion branch on the frozen pretrained LVCG.

Plan section 4/5: the pretrained backbone is the locked V0 reference and is never
updated. This script runs it once over PTB-XL, caches the 640-d embedding and the
latent VCG of every record, and then trains only the quaternion branch and the linear
head on those cached tensors:

    logits = Linear([ e_base ; scale * DynamicEncoder(features(VCG)) ])

Because the backbone is frozen, caching is exact and training a variant takes seconds
rather than minutes. Three model kinds share one command:

* ``--model qdf``   quaternion features q / theta / omega / magnitude (plan 3.2)
* ``--model control`` the parameter-matched real-valued control (plan 6.1): the same
  encoder over the raw vector pair and its difference
* ``--model v0``    scale = 0, i.e. the linear probe on e_base alone -- the reference,
  and the V0-equivalence check of plan 7.1

Checkpoint selection and every hyperparameter decision use validation macro AUROC only;
the test fold is read once, at the end (plan section 5).

    python scripts/train_qdf.py --config configs/eval/qdf_v1.yaml --model qdf --seed 42

Results are appended to ``probing/results/qdf_v1.csv``, one row per run, with per-label
AUROC and F1, and the per-epoch curve is written next to it as JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score, roc_auc_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lvcg.quaternion.qdf import QDFProbe  # noqa: E402
from probing.encoders.lvcg_encoder import LVCGEncoder  # noqa: E402
from probing.datasets import create_provider  # noqa: E402

FEATURE_SETS = {
    "qdf": ("q", "theta", "omega", "magnitude"),
    # V3's plan renames this: the quaternion channels are real numbers too, so the
    # contrast being measured is Cartesian versus decomposed, not "quaternion vs real".
    "control": ("position", "next_position", "delta"),
    "cdf": ("position", "next_position", "delta"),
}
LABELS = ("NORM", "MI", "STTC", "CD", "HYP")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def frozen_features(encoder: LVCGEncoder, loader, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the frozen backbone once: (e_base [N, 640], vcg [N, 3, 1000], labels [N, 5])."""
    from tqdm import tqdm

    encoder.eval()
    embeddings, vcgs, labels = [], [], []
    backbone = encoder.backbone
    for batch in tqdm(loader, desc="    frozen pass", leave=False):
        ecg = encoder._resample(batch["ecg"].to(device), 500)
        directions = backbone.all_lead_directions.unsqueeze(0).expand(ecg.shape[0], -1, -1)
        embeddings.append(backbone.ext_ecg_emb(ecg).float().cpu())
        vcgs.append(backbone.vcg_inverse(ecg, directions).float().cpu())
        labels.append(batch["label"].float())
    return torch.cat(embeddings), torch.cat(vcgs), torch.cat(labels)


def cached_features(encoder, loaders, device, cache_dir, checkpoint, ratio):
    """``frozen_features`` per split, reused across runs through an on-disk cache."""
    key = hashlib.sha256(f"{os.path.abspath(checkpoint)}|{ratio}".encode()).hexdigest()[:16]
    out = {}
    for split, loader in loaders.items():
        path = os.path.join(cache_dir, f"{split}_{key}.pt") if cache_dir else None
        if path and os.path.exists(path):
            out[split] = torch.load(path, weights_only=True)
            print(f"  {split}: cached {tuple(out[split][0].shape)}")
            continue
        started = time.perf_counter()
        out[split] = frozen_features(encoder, loader, device)
        print(f"  {split}: {tuple(out[split][0].shape)} in {time.perf_counter() - started:.1f}s")
        if path:
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(out[split], path)
    return out


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, object]:
    predictions = (probabilities >= 0.5).astype(int)
    per_label_auroc, per_label_f1 = [], []
    for index in range(labels.shape[1]):
        column = labels[:, index]
        per_label_auroc.append(
            float(roc_auc_score(column, probabilities[:, index])) if column.min() != column.max() else float("nan")
        )
        per_label_f1.append(float(f1_score(column, predictions[:, index], zero_division=0)))
    return {
        "macro_auroc": float(roc_auc_score(labels, probabilities, average="macro")),
        "micro_auroc": float(roc_auc_score(labels, probabilities, average="micro")),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        "per_label_auroc": per_label_auroc,
        "per_label_f1": per_label_f1,
    }


@torch.no_grad()
def evaluate(model, features, device, batch_size) -> Dict[str, object]:
    model.eval()
    embeddings, vcg, labels = features
    probabilities = []
    for start in range(0, len(labels), batch_size):
        stop = start + batch_size
        logits = model(embeddings[start:stop].to(device), vcg[start:stop].to(device))
        probabilities.append(torch.sigmoid(logits).cpu())
    return metrics(labels.numpy(), torch.cat(probabilities).numpy())


def train(model, data, device, probe_cfg, seed) -> Tuple[Dict, List[Dict]]:
    """Train the branch and head; keep the epoch with the best validation macro AUROC."""
    batch_size = int(probe_cfg.get("batch_size", 256))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(probe_cfg.get("lr", 1e-3)),
        weight_decay=float(probe_cfg.get("weight_decay", 1e-4)),
    )
    criterion = nn.BCEWithLogitsLoss()
    embeddings, vcg, labels = data["train"]
    generator = torch.Generator().manual_seed(seed)

    best = {"val_macro_auroc": -1.0, "epoch": -1, "state": None}
    curve = []
    patience = int(probe_cfg.get("patience", 5))
    for epoch in range(int(probe_cfg.get("max_epochs", 50))):
        model.train()
        order = torch.randperm(len(labels), generator=generator)
        total = 0.0
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            if len(index) < 2:  # BatchNorm needs more than one record
                continue
            optimizer.zero_grad()
            logits = model(embeddings[index].to(device), vcg[index].to(device))
            loss = criterion(logits, labels[index].to(device))
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


def append_row(path: str, row: dict) -> None:
    """Append one result, refusing to write into a CSV whose columns are different.

    Stages share a results directory, and a file written by an earlier version of a
    script has different columns; appending to it silently shifts every field. This
    stops instead and says what to do.
    """
    import csv as _csv

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as handle:
            header = next(_csv.reader(handle), [])
        if header and header != list(row):
            missing = [name for name in row if name not in header]
            extra = [name for name in header if name not in row]
            parts = []
            if missing:
                parts.append(f"this run adds {missing}")
            if extra:
                parts.append(f"the file has {extra}")
            raise SystemExit(
                f"{path} has different columns: "
                f"{'; '.join(parts) or 'the same names in another order'}.\n"
                "It was written by another variant of this experiment. Move it aside, "
                "e.g.\n  mkdir -p probing/results/superseded && "
                f"mv {path} probing/results/superseded/"
            )
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "eval", "qdf_v1.yaml"))
    parser.add_argument("--model", choices=("qdf", "control", "cdf", "v0"), default="qdf")
    parser.add_argument("--features", nargs="+", help="override the feature set (plan 6.2 ablations)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=float, default=1.0, help="label ratio")
    parser.add_argument("--checkpoint", help="override the pretrained checkpoint")
    parser.add_argument("--results", default="probing/results/qdf_v1.csv")
    parser.add_argument("--cache", default="probing/results/feature_cache")
    parser.add_argument("--tag", default="", help="free-text label for this run")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    checkpoint = args.checkpoint or cfg["pretrained"]["checkpoint"]
    probe_cfg = cfg.get("probe", {})
    quaternion_cfg = dict(cfg.get("quaternion", {}))
    device = torch.device(args.device)
    set_seed(args.seed)

    features = tuple(args.features) if args.features else FEATURE_SETS.get(args.model, FEATURE_SETS["qdf"])
    print(f"Model {args.model} | features {features} | seed {args.seed} | ratio {args.ratio:.0%}")

    provider_cfg = {
        "type": "benchmark",
        "dataset_name": "ptbxl_super_class",
        "raw_root": cfg["data"]["raw_root"],
        "splits_root": cfg["data"].get("splits_root", "probing/data_splits"),
        "norm_method": cfg["data"].get("norm_method", "zscore"),
    }
    bundle = create_provider("ptbxl_super_class", provider_cfg).build(
        label_ratio=args.ratio,
        batch_size=int(probe_cfg.get("batch_size", 256)),
        num_workers=int(probe_cfg.get("num_workers", 4)),
    )
    train_loader, val_loader, test_loader = bundle.train_loader, bundle.val_loader, bundle.test_loader
    num_classes = bundle.num_classes
    label_names = list(bundle.label_names) if bundle.label_names else list(LABELS)
    print(f"Train {len(train_loader.dataset)} | Val {len(val_loader.dataset)} | "
          f"Test {len(test_loader.dataset)} | classes {num_classes} {label_names}")

    encoder = LVCGEncoder(checkpoint).to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    print(f"Frozen backbone from {checkpoint}: "
          f"{sum(p.numel() for p in encoder.parameters()):,} parameters, none trainable")
    data = cached_features(
        encoder, {"train": train_loader, "val": val_loader, "test": test_loader},
        device, args.cache, checkpoint, args.ratio,
    )

    model = QDFProbe(
        base_dim=data["train"][0].shape[1],
        num_classes=num_classes,
        features=features,
        fs=int(quaternion_cfg.pop("fs", 100)),
        scale=0.0 if args.model == "v0" else float(quaternion_cfg.pop("scale", 1.0)),
        **{k: v for k, v in quaternion_cfg.items() if k in {
            "embedding_dim", "hidden", "kernel", "dropout",
            "min_magnitude_fraction", "sign_continuity", "masked_pooling"}},
    ).to(device)
    counts = model.parameter_counts()
    print(f"Trainable {counts['trainable_total']:,} "
          f"(quaternion branch {counts['quaternion_branch']:,}, V0 head {counts['v0_head']:,})")

    started = time.perf_counter()
    best, curve = train(model, data, device, probe_cfg, args.seed)
    test = evaluate(model, data["test"], device, int(probe_cfg.get("batch_size", 256)))
    seconds = time.perf_counter() - started

    print(f"\n  best epoch {best['epoch']} | val macro AUROC {best['val_macro_auroc']:.4f}")
    print(f"  TEST macro AUROC {test['macro_auroc']:.4f} | micro {test['micro_auroc']:.4f} "
          f"| macro F1 {test['macro_f1']:.4f} | micro F1 {test['micro_f1']:.4f}")
    print("  per label: " + ", ".join(
        f"{name} {auroc:.4f}" for name, auroc in zip(label_names or LABELS, test["per_label_auroc"])))

    row = {
        "model": args.model, "tag": args.tag, "features": "+".join(features),
        "label_ratio": args.ratio, "seed": args.seed,
        "test_macro_auroc": test["macro_auroc"], "test_micro_auroc": test["micro_auroc"],
        "test_macro_f1": test["macro_f1"], "test_micro_f1": test["micro_f1"],
        "val_macro_auroc": best["val_macro_auroc"], "best_epoch": best["epoch"],
        "trainable_params": counts["trainable_total"],
        "quaternion_params": counts["quaternion_branch"],
        "checkpoint": checkpoint, "seconds": round(seconds, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for name, auroc, f1 in zip(label_names, test["per_label_auroc"], test["per_label_f1"]):
        row[f"auroc_{name}"] = auroc
        row[f"f1_{name}"] = f1

    results_path = args.results if os.path.isabs(args.results) else os.path.join(REPO_ROOT, args.results)
    append_row(results_path, row)
    curve_path = results_path.replace(".csv", f"_curve_{args.model}{args.tag}_s{args.seed}.json")
    with open(curve_path, "w", encoding="utf-8") as handle:
        json.dump({"row": row, "curve": curve}, handle, indent=1)
    print(f"\nAppended to {results_path}; curve in {os.path.basename(curve_path)}")


if __name__ == "__main__":
    main()
