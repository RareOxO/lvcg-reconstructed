"""Route D (sparse leads), Tier 1: does the geometry still help when leads are removed?

Plan section 6. The visible-lead protocol is the release's own: only the VCG recovery is
restricted, the R peaks still come from lead II of the full recording, and the
regularised pseudo-inverse keeps its eps. The sweep runs 12 / 8 / 6 / 3 / 2 / 1 leads
(named configurations, see ``lvcg.quaternion.sparse.LEAD_SETS``) and reports, at every
count:

* absolute macro AUROC for the baseline and the proposed branch,
* the increment over V0, so the curve answers "does the advantage grow as leads go away",
* how much of the 12-lead geometry survived the recovery (direction cosine, magnitude
  ratio, planarity) and how far the intrinsic frame has drifted.

The plan is explicit that an improvement seen only at full leads must not be dressed up
as spatial robustness, so both ends of the curve are always printed together.

    python scripts/train_sparse.py --config configs/eval/sparse_d.yaml --leads 12 6 3 2 1 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
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

from lvcg.quaternion.frame import PARTS, FrameProbe  # noqa: E402
from lvcg.quaternion.sparse import LEAD_SETS, frame_agreement, geometry_fidelity, recover_vcg, visible_leads  # noqa: E402
from probing.datasets import create_provider  # noqa: E402
from probing.encoders.lvcg_encoder import LVCGEncoder  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO_ROOT, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_qdf, train_frame = _load("train_qdf"), _load("train_frame")
LABELS = train_qdf.LABELS
MAX_BEATS = 20


@torch.no_grad()
def frozen_tensors(encoder, loader, device, leads):
    """The frozen model's view of a record when only ``leads`` are visible."""
    from tqdm import tqdm

    backbone = encoder.backbone
    encoder.eval()
    names = ("beats", "rr", "mask", "reference", "tokens", "steps", "rhythm", "labels")
    out = {name: [] for name in names}
    fidelity = {"direction_cosine": [], "magnitude_ratio": [], "planarity": [], "frame_drift": []}

    for batch in tqdm(loader, desc="    frozen pass", leave=False):
        ecg = encoder._resample(batch["ecg"].to(device), 500)
        full_directions = backbone.all_lead_directions.unsqueeze(0).expand(ecg.shape[0], -1, -1)
        full_vcg = backbone.vcg_inverse(ecg, full_directions)
        vcg = recover_vcg(backbone, ecg, leads)
        # R peaks come from the whole recording, exactly as forward_inference does.
        beats, rr, mask = backbone.beat_segmenter(vcg, ecg, rr_lead_idx=backbone.rr_lead_idx)
        full_beats, _, _ = backbone.beat_segmenter(full_vcg, ecg, rr_lead_idx=backbone.rr_lead_idx)
        tokens = backbone.beat_encoder(beats)
        rhythm = backbone.norm_rhythm(backbone.global_rr_embedding(rr, mask))

        found = beats.shape[1]
        pad = MAX_BEATS - min(found, MAX_BEATS)
        take = slice(0, min(found, MAX_BEATS))
        pad3 = torch.nn.functional.pad
        out["beats"].append(pad3(beats[:, take], (0, 0, 0, 0, 0, pad)).cpu())
        out["rr"].append(pad3(rr[:, take], (0, pad)).cpu())
        out["mask"].append(pad3(mask[:, take].float(), (0, pad)).cpu())
        out["tokens"].append(pad3(tokens[:, take], (0, 0, 0, pad)).cpu())
        out["reference"].append(torch.quantile(vcg.transpose(1, 2).norm(dim=-1), 0.99, dim=-1).cpu())
        out["steps"].append((mask[:, take].sum(dim=1).clamp(max=MAX_BEATS) - 1).long().cpu())
        out["rhythm"].append(rhythm.float().cpu())
        out["labels"].append(batch["label"].float())

        scores = geometry_fidelity(full_vcg, vcg)
        for key, value in scores.items():
            fidelity[key].append(value.cpu())
        fidelity["frame_drift"].append(
            frame_agreement(pad3(full_beats[:, take], (0, 0, 0, 0, 0, pad)),
                            pad3(beats[:, take], (0, 0, 0, 0, 0, pad)),
                            pad3(mask[:, take].float(), (0, pad))).cpu())

    tensors = tuple(torch.cat(out[name]) for name in names)
    summary = {key: float(torch.cat(values).mean()) for key, values in fidelity.items()}
    return tensors, summary


def cached_tensors(encoder, loaders, device, cache_dir, checkpoint, ratio, leads):
    key = hashlib.sha256(
        f"{os.path.abspath(checkpoint)}|{ratio}|leads{'-'.join(map(str, leads))}".encode()).hexdigest()[:16]
    data, fidelity = {}, {}
    for split, loader in loaders.items():
        path = os.path.join(cache_dir, f"{split}_{key}.pt") if cache_dir else None
        if path and os.path.exists(path):
            data[split], fidelity[split] = torch.load(path, weights_only=False)
            print(f"  {split}: cached {tuple(data[split][0].shape)}")
            continue
        started = time.perf_counter()
        data[split], fidelity[split] = frozen_tensors(encoder, loader, device, leads)
        print(f"  {split}: {tuple(data[split][0].shape)} in {time.perf_counter() - started:.1f}s")
        if path:
            os.makedirs(cache_dir, exist_ok=True)
            torch.save((data[split], fidelity[split]), path)
    return data, fidelity["test"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "eval", "sparse_d.yaml"))
    parser.add_argument("--leads", type=int, nargs="+", default=[12, 6, 3, 2, 1],
                        help=f"lead counts to sweep, from {sorted(LEAD_SETS)}")
    parser.add_argument("--parts", nargs="+", default=["invariant"],
                        help=f"branch parts, any of {list(PARTS)}; route C found 'invariant' strongest")
    parser.add_argument("--struct", choices=FrameProbe.STRUCTS, default="mean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--results", default="probing/results/sparse_d.csv")
    parser.add_argument("--cache", default="probing/results/feature_cache")
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    checkpoint = args.checkpoint or cfg["pretrained"]["checkpoint"]
    probe_cfg, frame_cfg = cfg.get("probe", {}), dict(cfg.get("frame", {}))
    device = torch.device(args.device)

    import re
    parts = tuple(p for argument in args.parts for p in re.split(r"[\s,+]+", argument.strip()) if p)
    print(f"Lead sweep {args.leads} | branch {' + '.join(parts)} | struct {args.struct} | "
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
    loaders = {"train": bundle.train_loader, "val": bundle.val_loader, "test": bundle.test_loader}
    print(f"Train {len(bundle.train_loader.dataset)} | Val {len(bundle.val_loader.dataset)} | "
          f"Test {len(bundle.test_loader.dataset)} | classes {bundle.num_classes} {label_names}")

    encoder = LVCGEncoder(checkpoint).to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    backbone = encoder.backbone
    batch_size = int(probe_cfg.get("batch_size", 256))
    results_path = args.results if os.path.isabs(args.results) else os.path.join(REPO_ROOT, args.results)

    curve = []
    for count in args.leads:
        leads = visible_leads(count)
        print(f"\n=== {count} lead(s): indices {leads} ===")
        train_qdf.set_seed(args.seed)
        data, fidelity = cached_tensors(encoder, loaders, device, args.cache, checkpoint, args.ratio, leads)
        print(f"  geometry kept: direction cosine {fidelity['direction_cosine']:.3f}, "
              f"magnitude ratio {fidelity['magnitude_ratio']:.3f}, "
              f"planarity {fidelity['planarity']:.3f}, frame drift {fidelity['frame_drift']:.1f} deg")

        scores = {}
        for variant in ("v0", "frame"):
            train_qdf.set_seed(args.seed)
            model = FrameProbe(
                state_generator=backbone.state_generator,
                norm_struct=backbone.norm_struct,
                norm_dynamic=backbone.norm_dynamic,
                num_classes=bundle.num_classes,
                struct=args.struct,
                scale=0.0 if variant == "v0" else 1.0,
                parts=parts,
                **{k: v for k, v in frame_cfg.items() if k in {
                    "embedding_dim", "hidden", "kernel", "dropout"}},
            ).to(device)
            best, epochs = train_frame.train(model, data, device, probe_cfg, args.seed)
            test = train_frame.evaluate(model, data["test"], device, batch_size)
            scores[variant] = test
            print(f"  {variant:>5}: TEST macro AUROC {test['macro_auroc']:.4f} "
                  f"(val {best['val_macro_auroc']:.4f}, epoch {best['epoch']})")

            row = {
                "model": variant, "leads": count, "lead_indices": "-".join(map(str, leads)),
                "parts": "+".join(parts), "struct": args.struct, "tag": args.tag,
                "label_ratio": args.ratio, "seed": args.seed,
                "test_macro_auroc": test["macro_auroc"], "test_micro_auroc": test["micro_auroc"],
                "test_macro_f1": test["macro_f1"], "test_micro_f1": test["micro_f1"],
                "val_macro_auroc": best["val_macro_auroc"], "best_epoch": best["epoch"],
                "trainable_params": model.parameter_counts()["trainable_total"],
                **{f"geometry_{k}": round(v, 4) for k, v in fidelity.items()},
                "checkpoint": checkpoint, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for name, auroc, f1 in zip(label_names, test["per_label_auroc"], test["per_label_f1"]):
                row[f"auroc_{name}"], row[f"f1_{name}"] = auroc, f1
            train_qdf.append_row(results_path, row)

        delta = (scores["frame"]["macro_auroc"] - scores["v0"]["macro_auroc"]) * 100
        curve.append({"leads": count, "v0": scores["v0"]["macro_auroc"],
                      "frame": scores["frame"]["macro_auroc"], "delta": delta, **fidelity})
        print(f"  delta vs V0: {delta:+.2f} pp")

    print("\n  leads    V0   branch   delta   dir cos   planarity   frame drift")
    for point in curve:
        print(f"  {point['leads']:>5} {point['v0'] * 100:>6.2f} {point['frame'] * 100:>7.2f} "
              f"{point['delta']:>+7.2f} {point['direction_cosine']:>9.3f} "
              f"{point['planarity']:>11.3f} {point['frame_drift']:>12.1f}")
    if len(curve) > 1:
        slope = (curve[0]["v0"] - curve[-1]["v0"]) * 100 / max(curve[0]["leads"] - curve[-1]["leads"], 1)
        branch_slope = (curve[0]["frame"] - curve[-1]["frame"]) * 100 / max(
            curve[0]["leads"] - curve[-1]["leads"], 1)
        print(f"\n  decline per lead removed: V0 {slope:.3f} pp, branch {branch_slope:.3f} pp")

    curve_path = results_path.replace(".csv", f"_curve_{'-'.join(parts)}{args.tag}_s{args.seed}.json")
    with open(curve_path, "w", encoding="utf-8") as handle:
        json.dump(curve, handle, indent=1)
    print(f"\nAppended to {results_path}; curve in {os.path.basename(curve_path)}")


if __name__ == "__main__":
    main()
