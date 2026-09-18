"""Route C (frame handling), Tier 1: what the observation frame changes, and what it does not.

Plan section 5. The frame is computed from the record -- not learned, which is what
separates route C from V5's pose head and V6's lead-matrix rotation -- and the
representation is split by transformation behaviour:

    --parts invariant                the trajectory in its own frame: provably unchanged
                                     by a global rotation
    --parts equivariant              the frame itself: absolute orientation, kept rather
                                     than discarded
    --parts invariant equivariant    both, which is the plan's "keep the useful
                                     orientation while structuring the frame handling"
    --variant v0                     the branch off: this stage's baseline

Both numbers the plan demands are reported: clean macro AUROC **and** the rotation sweep.
A robustness gain paid for with a clear clean loss is explicitly not full support, so the
two are printed side by side, and the script also verifies the two transformation
properties numerically on the test fold rather than asserting them in prose.

    python scripts/train_frame.py --config configs/eval/frame_c.yaml --parts invariant --seed 42
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

from lvcg.quaternion.canon import random_rotations, rotate_beats  # noqa: E402
from lvcg.quaternion.frame import PARTS, FrameProbe, intrinsic_frame, to_frame  # noqa: E402
from lvcg.quaternion.utils import quaternion_multiply  # noqa: E402
from probing.datasets import create_provider  # noqa: E402
from probing.encoders.lvcg_encoder import LVCGEncoder  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO_ROOT, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_qdf, train_qdt, train_canon = _load("train_qdf"), _load("train_qdt"), _load("train_canon")
LABELS, metrics = train_qdf.LABELS, train_qdf.metrics


def _inputs(tensors, index, device, rotation=None):
    beats, _rr, mask, _reference, tokens, steps, rhythm, _labels = tensors
    beats = beats[index].to(device)
    if rotation is not None:
        beats = rotate_beats(beats, rotation.to(device))
    return (beats, tokens[index].to(device), mask[index].to(device),
            steps[index].to(device), rhythm[index].to(device))


def _forward(model, tensors, index, device, rotation=None):
    return model(*_inputs(tensors, index, device, rotation))


@torch.no_grad()
def evaluate(model, tensors, device, batch_size, degrees=None, seed=0):
    model.eval()
    labels = tensors[-1]
    generator = torch.Generator().manual_seed(seed)
    probabilities = []
    for start in range(0, len(labels), batch_size):
        index = torch.arange(start, min(start + batch_size, len(labels)))
        rotation = None if degrees in (None, 0) else random_rotations(len(index), degrees, generator)
        probabilities.append(torch.sigmoid(_forward(model, tensors, index, device, rotation)).cpu())
    return metrics(labels.numpy(), torch.cat(probabilities).numpy())


@torch.no_grad()
def frame_properties(tensors, device, degrees=60, limit=512, seed=11):
    """Check the two transformation claims on real records instead of asserting them."""
    beats = tensors[0][:limit].to(device)
    mask = tensors[2][:limit].to(device)
    rotation = random_rotations(len(beats), degrees, torch.Generator().manual_seed(seed)).to(device)
    turned = rotate_beats(beats, rotation)

    frame, quaternion = intrinsic_frame(beats, mask)
    turned_frame, turned_quaternion = intrinsic_frame(turned, mask)
    equivariance = float((turned_frame - rotation @ frame).abs().max())
    local, turned_local = to_frame(beats, frame), to_frame(turned, turned_frame)
    invariance = float((turned_local - local).abs().max() / local.abs().max().clamp_min(1e-9))
    return {"frame_equivariance_error": round(equivariance, 6),
            "frame_invariance_error": round(invariance, 6)}


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
    parser.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "eval", "frame_c.yaml"))
    parser.add_argument("--parts", nargs="+", default=list(PARTS),
                        help=f"any of {list(PARTS)}; also accepts one comma separated argument")
    parser.add_argument("--variant", choices=("frame", "v0"), default="frame")
    parser.add_argument("--struct", choices=FrameProbe.STRUCTS, default="mean")
    parser.add_argument("--eval-rotations", type=int, nargs="*", default=list(train_canon.DEFAULT_ANGLES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--results", default="probing/results/frame_c.csv")
    parser.add_argument("--cache", default="probing/results/feature_cache")
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    checkpoint = args.checkpoint or cfg["pretrained"]["checkpoint"]
    probe_cfg, frame_cfg = cfg.get("probe", {}), dict(cfg.get("frame", {}))
    device = torch.device(args.device)
    train_qdf.set_seed(args.seed)

    import re
    parts = tuple(p for argument in args.parts for p in re.split(r"[\s,+]+", argument.strip()) if p)
    print(f"Parts {' + '.join(parts)} | variant {args.variant} | struct {args.struct} | "
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
    model = FrameProbe(
        state_generator=backbone.state_generator,
        norm_struct=backbone.norm_struct,
        norm_dynamic=backbone.norm_dynamic,
        num_classes=bundle.num_classes,
        struct=args.struct,
        scale=0.0 if args.variant == "v0" else 1.0,
        parts=parts,
        **{k: v for k, v in frame_cfg.items() if k in {"embedding_dim", "hidden", "kernel", "dropout"}},
    ).to(device)
    counts = model.parameter_counts()
    print(f"Trainable {counts['trainable_total']:,} "
          f"(frame branch {counts['frame_branch']:,}, head {counts['head']:,})")

    started = time.perf_counter()
    batch_size = int(probe_cfg.get("batch_size", 256))
    best, curve = train(model, data, device, probe_cfg, args.seed)
    test = evaluate(model, data["test"], device, batch_size)
    properties = frame_properties(data["test"], device)
    sweep = {d: evaluate(model, data["test"], device, batch_size, d, seed=1000 + d)["macro_auroc"]
             for d in args.eval_rotations}
    seconds = time.perf_counter() - started

    print(f"\n  best epoch {best['epoch']} | val macro AUROC {best['val_macro_auroc']:.4f}")
    print(f"  TEST macro AUROC {test['macro_auroc']:.4f} | micro {test['micro_auroc']:.4f} "
          f"| macro F1 {test['macro_f1']:.4f}")
    print("  per label: " + ", ".join(
        f"{name} {auroc:.4f}" for name, auroc in zip(label_names, test["per_label_auroc"])))
    print(f"  frame properties on real records: equivariance error "
          f"{properties['frame_equivariance_error']:.2e}, "
          f"invariance error {properties['frame_invariance_error']:.2e}")
    if sweep:
        reference = sweep.get(0, test["macro_auroc"])
        print("  rotation sweep: " + ", ".join(
            f"{d}deg {v * 100:.2f} ({(v - reference) * 100:+.2f})" for d, v in sweep.items()))

    row = {
        "model": "v0" if args.variant == "v0" else "frame", "parts": "+".join(parts),
        "struct": args.struct, "tag": args.tag,
        "label_ratio": args.ratio, "seed": args.seed,
        "test_macro_auroc": test["macro_auroc"], "test_micro_auroc": test["micro_auroc"],
        "test_macro_f1": test["macro_f1"], "test_micro_f1": test["micro_f1"],
        "val_macro_auroc": best["val_macro_auroc"], "best_epoch": best["epoch"],
        "trainable_params": counts["trainable_total"], **properties,
        **{f"rot{d}_macro_auroc": v for d, v in sweep.items()},
        "checkpoint": checkpoint, "seconds": round(seconds, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for name, auroc, f1 in zip(label_names, test["per_label_auroc"], test["per_label_f1"]):
        row[f"auroc_{name}"], row[f"f1_{name}"] = auroc, f1

    results_path = args.results if os.path.isabs(args.results) else os.path.join(REPO_ROOT, args.results)
    train_qdf.append_row(results_path, row)
    curve_path = results_path.replace(
        ".csv", f"_curve_{args.variant}-{'-'.join(parts)}{args.tag}_s{args.seed}.json")
    with open(curve_path, "w", encoding="utf-8") as handle:
        json.dump({"row": row, "curve": curve, "sweep": {str(d): v for d, v in sweep.items()}},
                  handle, indent=1)
    print(f"\nAppended to {results_path}; curve in {os.path.basename(curve_path)}")


if __name__ == "__main__":
    main()
