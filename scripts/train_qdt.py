"""V2 (QDT-LVCG), Tier 1: quaternion beat tokens fused into the frozen pretrained path.

Plan section 4: the pretrained LVCG stays locked. Unlike V1, V2 acts inside the model,
so the frozen pass here caches what sits *before* the fusion point and the training loop
replays the pretrained temporal path on top of it:

    cached (frozen)   beat patches [N, 20, 3, 128], R-R intervals, beat mask,
                      BeatEncoder tokens [N, 20, 256], rhythm embedding [N, 128],
                      the record's 99th-percentile VCG magnitude, its beat count
    trained           per-beat quaternion tokens, the fusion, the linear head
    frozen, replayed  StateGRU rollout, norm_struct, norm_dynamic

Models, all sharing one command:

* ``--model qdt``     quaternion beat tokens (plan 3.2, computed inside each beat)
* ``--model control`` the parameter-matched real-valued control (plan 6.1)
* ``--model v0``      scale = 0, the fusion bypassed: the pretrained path untouched

Two deliberate deviations from ``scripts/train_qdf.py``, both reported in the output:

1. The GRU rollout uses each record's own beat count. The release uses the batch's
   largest, so a record's embedding depends on the batch it lands in. The V0 row below
   is therefore computed in this script, not taken from V1's V0.
2. ``max_beats`` pads every record to 20 beats, the model's own limit.

    python scripts/train_qdt.py --config configs/eval/qdt_v2.yaml --model qdt --seed 42
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from typing import Dict, Tuple

import numpy as np
import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lvcg.quaternion.qdt import QDTProbe  # noqa: E402
from probing.datasets import create_provider  # noqa: E402
from probing.encoders.lvcg_encoder import LVCGEncoder  # noqa: E402

# V1's script owns the metric and seeding helpers; V2 reuses them unchanged so the two
# stages are scored identically.
_spec = importlib.util.spec_from_file_location(
    "train_qdf", os.path.join(REPO_ROOT, "scripts", "train_qdf.py"))
train_qdf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_qdf)
FEATURE_SETS, LABELS = train_qdf.FEATURE_SETS, train_qdf.LABELS
metrics, set_seed = train_qdf.metrics, train_qdf.set_seed

MAX_BEATS = 20
FIELDS = ("beats", "rr", "mask", "reference", "tokens", "steps", "rhythm", "labels")


@torch.no_grad()
def frozen_tensors(encoder: LVCGEncoder, loader, device) -> Tuple[torch.Tensor, ...]:
    """Everything the pretrained model computes before the fusion point, per record."""
    from tqdm import tqdm

    backbone = encoder.backbone
    encoder.eval()
    out = {name: [] for name in FIELDS}
    for batch in tqdm(loader, desc="    frozen pass", leave=False):
        ecg = encoder._resample(batch["ecg"].to(device), 500)
        directions = backbone.all_lead_directions.unsqueeze(0).expand(ecg.shape[0], -1, -1)
        vcg = backbone.vcg_inverse(ecg, directions)
        beats, rr, mask = backbone.beat_segmenter(vcg, ecg, rr_lead_idx=backbone.rr_lead_idx)
        tokens = backbone.beat_encoder(beats)
        rhythm = backbone.norm_rhythm(backbone.global_rr_embedding(rr, mask))

        found = beats.shape[1]
        pad = MAX_BEATS - min(found, MAX_BEATS)
        take = slice(0, min(found, MAX_BEATS))
        out["beats"].append(torch.nn.functional.pad(beats[:, take], (0, 0, 0, 0, 0, pad)).cpu())
        out["rr"].append(torch.nn.functional.pad(rr[:, take], (0, pad)).cpu())
        out["mask"].append(torch.nn.functional.pad(mask[:, take].float(), (0, pad)).cpu())
        out["tokens"].append(torch.nn.functional.pad(tokens[:, take], (0, 0, 0, pad)).cpu())
        out["reference"].append(torch.quantile(vcg.transpose(1, 2).norm(dim=-1), 0.99, dim=-1).cpu())
        # The release's num_gen_steps, but per record: its own beat count minus one.
        out["steps"].append((mask[:, take].sum(dim=1).clamp(max=MAX_BEATS) - 1).long().cpu())
        out["rhythm"].append(rhythm.float().cpu())
        out["labels"].append(batch["label"].float())
    return tuple(torch.cat(out[name]) for name in FIELDS)


def cached_tensors(encoder, loaders, device, cache_dir, checkpoint, ratio):
    key = hashlib.sha256(f"{os.path.abspath(checkpoint)}|{ratio}|v2".encode()).hexdigest()[:16]
    out = {}
    for split, loader in loaders.items():
        path = os.path.join(cache_dir, f"{split}_{key}.pt") if cache_dir else None
        if path and os.path.exists(path):
            out[split] = torch.load(path, weights_only=True)
            print(f"  {split}: cached beats {tuple(out[split][0].shape)}")
            continue
        started = time.perf_counter()
        out[split] = frozen_tensors(encoder, loader, device)
        print(f"  {split}: beats {tuple(out[split][0].shape)} in {time.perf_counter() - started:.1f}s")
        if path:
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(out[split], path)
    return out


def _forward(model, tensors, index, device):
    beats, rr, mask, reference, tokens, steps, rhythm, _ = tensors
    return model(
        beats[index].to(device), rr[index].to(device), mask[index].to(device),
        reference[index].to(device), tokens[index].to(device),
        steps[index].to(device), rhythm[index].to(device),
    )


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
    """Train the branch, the fusion and the head; keep the best validation macro AUROC."""
    batch_size = int(probe_cfg.get("batch_size", 256))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(probe_cfg.get("lr", 1e-3)),
        weight_decay=float(probe_cfg.get("weight_decay", 1e-4)),
    )
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
    parser.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "eval", "qdt_v2.yaml"))
    parser.add_argument("--model", choices=("qdt", "control", "v0"), default="qdt")
    parser.add_argument("--features", nargs="+", help="override the feature set (plan 6.2)")
    parser.add_argument("--fusion", choices=("gated", "concat"), help="override the fusion mode")
    parser.add_argument("--fusion-init-std", type=float, default=None,
                        help="near-zero fusion init; 0 keeps exact V0 equivalence but leaves the "
                             "branch without gradient on the first step")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--results", default="probing/results/qdt_v2.csv")
    parser.add_argument("--cache", default="probing/results/feature_cache")
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    checkpoint = args.checkpoint or cfg["pretrained"]["checkpoint"]
    probe_cfg, quaternion_cfg = cfg.get("probe", {}), dict(cfg.get("quaternion", {}))
    fusion = args.fusion or quaternion_cfg.pop("fusion", "gated")
    quaternion_cfg.pop("fusion", None)
    init_std = float(args.fusion_init_std if args.fusion_init_std is not None
                     else quaternion_cfg.pop("fusion_init_std", 0.0))
    quaternion_cfg.pop("fusion_init_std", None)
    device = torch.device(args.device)
    set_seed(args.seed)

    features = tuple(args.features) if args.features else FEATURE_SETS.get(
        "qdf" if args.model != "control" else "control")
    print(f"Model {args.model} | features {features} | fusion {fusion} (init std {init_std}) | "
          f"seed {args.seed} | ratio {args.ratio:.0%}")

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
    data = cached_tensors(
        encoder, {"train": bundle.train_loader, "val": bundle.val_loader, "test": bundle.test_loader},
        device, args.cache, checkpoint, args.ratio,
    )
    beats_per_record = data["train"][5].float() + 1
    print(f"Beats per record: mean {beats_per_record.mean():.1f}, "
          f"min {int(beats_per_record.min())}, max {int(beats_per_record.max())} (padded to {MAX_BEATS})")

    backbone = encoder.backbone
    model = QDTProbe(
        state_generator=backbone.state_generator,
        norm_struct=backbone.norm_struct,
        norm_dynamic=backbone.norm_dynamic,
        num_classes=bundle.num_classes,
        features=features,
        fs=int(quaternion_cfg.pop("fs", 100)),
        fusion=fusion,
        fusion_init_std=init_std,
        scale=0.0 if args.model == "v0" else float(quaternion_cfg.pop("scale", 1.0)),
        **{k: v for k, v in quaternion_cfg.items() if k in {
            "quaternion_dim", "hidden", "kernel", "dropout",
            "min_magnitude_fraction", "sign_continuity", "masked_pooling"}},
    ).to(device)
    counts = model.parameter_counts()
    print(f"Trainable {counts['trainable_total']:,} (branch {counts['quaternion_branch']:,}, "
          f"fusion {counts['fusion']:,}, head {counts['head']:,})")

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
        "model": args.model, "tag": args.tag, "features": "+".join(features),
        "fusion": fusion, "fusion_init_std": init_std,
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
    train_qdf.append_row(results_path, row)
    curve_path = results_path.replace(".csv", f"_curve_{args.model}{args.tag}_s{args.seed}.json")
    with open(curve_path, "w", encoding="utf-8") as handle:
        json.dump({"row": row, "curve": curve}, handle, indent=1)
    print(f"\nAppended to {results_path}; curve in {os.path.basename(curve_path)}")


if __name__ == "__main__":
    main()
