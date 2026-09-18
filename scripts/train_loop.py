"""Route A (rotational loop geometry), Tier 1: ordered rotation composition on frozen LVCG.

A1 asks whether the *composition* of a beat's rotations -- not four quaternion components
fed to a convolution, which V1 already answered -- adds to the pretrained embedding:

    --variant loop    multi-scale ordered composition (short / medium / whole beat)
    --variant local   single-step quaternions only: composition removed, everything else equal
    --variant v0      the branch switched off: this stage's baseline

A2 is the signature test. After training, the test trajectories are perturbed so that
*the size of every step rotation is preserved and only its order changes* -- shuffled,
shuffled in blocks, or reversed -- and the script reports, for the same model:

* how much test macro AUROC falls, and
* how far the embedding moves, separately for the frozen part and the loop branch.

The frozen beat encoder is re-run on the perturbed trajectory, so the perturbation
reaches the pretrained path too; otherwise the comparison would be vacuous.

The research plan locks the structural representation to the mean of the beat tokens,
which is not what the released model does (it uses beat token 1). ``--struct`` selects
either, and both baselines are reported, because they are different V0s.

    python scripts/train_loop.py --config configs/eval/loop_a.yaml --variant loop --seed 42
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

from lvcg.quaternion.loop import LoopProbe, perturb_rotation_order  # noqa: E402
from probing.datasets import create_provider  # noqa: E402
from probing.encoders.lvcg_encoder import LVCGEncoder  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO_ROOT, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_qdf, train_qdt = _load("train_qdf"), _load("train_qdt")
LABELS, metrics = train_qdf.LABELS, train_qdf.metrics
PERTURBATIONS = ("shuffle", "block", "reverse")


def _inputs(tensors, index, device, perturb=None, generator=None):
    """V2's cache -> the probe's arguments, optionally with the rotations reordered."""
    beats, rr, mask, reference, tokens, steps, rhythm, _labels = tensors
    beats = beats[index].to(device)
    mask = mask[index].to(device)
    reference = reference[index].to(device)
    if perturb:
        beats = perturb_rotation_order(beats, mask, reference, perturb, generator=generator)
    return (beats, tokens[index].to(device), mask, reference,
            steps[index].to(device), rhythm[index].to(device))


def _forward(model, tensors, index, device, perturb=None, generator=None):
    return model(*_inputs(tensors, index, device, perturb, generator), recompute=perturb is not None)


@torch.no_grad()
def evaluate(model, tensors, device, batch_size, perturb=None, seed=0) -> Dict[str, object]:
    model.eval()
    labels = tensors[-1]
    generator = torch.Generator(device=device).manual_seed(seed)
    probabilities = []
    for start in range(0, len(labels), batch_size):
        index = torch.arange(start, min(start + batch_size, len(labels)))
        probabilities.append(torch.sigmoid(_forward(model, tensors, index, device, perturb, generator)).cpu())
    return metrics(labels.numpy(), torch.cat(probabilities).numpy())


@torch.no_grad()
def embedding_shift(model, tensors, device, perturb, batch_size=256, limit=1024, seed=0):
    """Relative movement of each embedding part when only the rotation order changes."""
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    moved = {"base": 0.0, "loop": 0.0}
    norms = {"base": 0.0, "loop": 0.0}
    for start in range(0, min(len(tensors[-1]), limit), batch_size):
        index = torch.arange(start, min(start + batch_size, len(tensors[-1])))
        real = model.embeddings(*_inputs(tensors, index, device), recompute=True)
        fake = model.embeddings(*_inputs(tensors, index, device, perturb, generator), recompute=True)
        for name, before, after in zip(("base", "loop"), real, fake):
            moved[name] += float((after - before).norm())
            norms[name] += float(before.norm())
    return {f"{perturb}_shift_{name}": round(moved[name] / max(norms[name], 1e-9), 4) for name in moved}


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
    parser.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "eval", "loop_a.yaml"))
    parser.add_argument("--variant", choices=("loop", "local", "v0"), default="loop")
    parser.add_argument("--struct", choices=LoopProbe.STRUCTS, default="mean",
                        help="mean: the research plan's pooling; anchor: the released behaviour")
    parser.add_argument("--scales", type=int, nargs="+", help="composition scales; 0 = whole beat")
    parser.add_argument("--include-magnitude", action="store_true")
    parser.add_argument("--perturbations", nargs="*", default=list(PERTURBATIONS),
                        help="signature test; empty disables it")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--results", default="probing/results/loop_a.csv")
    parser.add_argument("--cache", default="probing/results/feature_cache")
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    checkpoint = args.checkpoint or cfg["pretrained"]["checkpoint"]
    probe_cfg, loop_cfg = cfg.get("probe", {}), dict(cfg.get("loop", {}))
    scales = tuple(args.scales) if args.scales else tuple(loop_cfg.pop("scales", (8, 32, 0)))
    loop_cfg.pop("scales", None)
    device = torch.device(args.device)
    train_qdf.set_seed(args.seed)

    print(f"Variant {args.variant} | struct {args.struct} | scales {scales} | "
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
    model = LoopProbe(
        state_generator=backbone.state_generator,
        norm_struct=backbone.norm_struct,
        norm_dynamic=backbone.norm_dynamic,
        beat_encoder=backbone.beat_encoder,
        num_classes=bundle.num_classes,
        struct=args.struct,
        scale=0.0 if args.variant == "v0" else 1.0,
        scales=scales,
        local_only=args.variant == "local",
        include_magnitude=args.include_magnitude,
        **{k: v for k, v in loop_cfg.items() if k in {
            "embedding_dim", "hidden", "kernel", "dropout",
            "min_magnitude_fraction", "sign_continuity"}},
    ).to(device)
    counts = model.parameter_counts()
    print(f"Trainable {counts['trainable_total']:,} "
          f"(loop branch {counts['loop_branch']:,}, head {counts['head']:,}); "
          f"{model.loop.channels} input channels")

    started = time.perf_counter()
    best, curve = train(model, data, device, probe_cfg, args.seed)
    batch_size = int(probe_cfg.get("batch_size", 256))
    test = evaluate(model, data["test"], device, batch_size)
    signature = {}
    for perturb in args.perturbations:
        scores = evaluate(model, data["test"], device, batch_size, perturb, seed=7)
        signature[perturb] = scores["macro_auroc"]
        signature.update(embedding_shift(model, data["test"], device, perturb, batch_size, seed=7))
    seconds = time.perf_counter() - started

    print(f"\n  best epoch {best['epoch']} | val macro AUROC {best['val_macro_auroc']:.4f}")
    print(f"  TEST macro AUROC {test['macro_auroc']:.4f} | micro {test['micro_auroc']:.4f} "
          f"| macro F1 {test['macro_f1']:.4f}")
    print("  per label: " + ", ".join(
        f"{name} {auroc:.4f}" for name, auroc in zip(label_names, test["per_label_auroc"])))
    if args.perturbations:
        print("  signature test (rotation order changed, step sizes kept):")
        for perturb in args.perturbations:
            drop = (signature[perturb] - test["macro_auroc"]) * 100
            print(f"    {perturb:>7}: AUROC {signature[perturb] * 100:.2f} ({drop:+.2f}), "
                  f"embedding moved base {signature[f'{perturb}_shift_base']:.3f}, "
                  f"loop {signature[f'{perturb}_shift_loop']:.3f}")

    row = {
        "model": "v0" if args.variant == "v0" else "loop", "variant": args.variant,
        "struct": args.struct, "scales": "+".join(str(s) for s in scales),
        "magnitude": int(args.include_magnitude), "tag": args.tag,
        "label_ratio": args.ratio, "seed": args.seed,
        "test_macro_auroc": test["macro_auroc"], "test_micro_auroc": test["micro_auroc"],
        "test_macro_f1": test["macro_f1"], "test_micro_f1": test["micro_f1"],
        "val_macro_auroc": best["val_macro_auroc"], "best_epoch": best["epoch"],
        "trainable_params": counts["trainable_total"],
        **{f"rot_{key}": value for key, value in signature.items()},
        "checkpoint": checkpoint, "seconds": round(seconds, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for name, auroc, f1 in zip(label_names, test["per_label_auroc"], test["per_label_f1"]):
        row[f"auroc_{name}"], row[f"f1_{name}"] = auroc, f1

    results_path = args.results if os.path.isabs(args.results) else os.path.join(REPO_ROOT, args.results)
    train_qdf.append_row(results_path, row)
    curve_path = results_path.replace(
        ".csv", f"_curve_{args.variant}-{args.struct}{args.tag}_s{args.seed}.json")
    with open(curve_path, "w", encoding="utf-8") as handle:
        json.dump({"row": row, "curve": curve}, handle, indent=1)
    print(f"\nAppended to {results_path}; curve in {os.path.basename(curve_path)}")


if __name__ == "__main__":
    main()
