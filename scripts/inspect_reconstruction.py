"""Masked-lead reconstruction on held-out MIMIC records: figures and per-lead numbers.

The pretraining loss is dominated by the VCG beat term, so a falling total says little
about whether masked leads are actually reconstructed. This script takes a pretrained
checkpoint, hides all but a few leads of validation records the model never trained on
(the same split ``scripts/train.py`` holds out), and reports what comes back:

* ``reconstruction_<id>.png`` -- ground truth and reconstruction for all 12 leads, with
  the visible leads marked, to read P wave, QRS morphology, T wave, R-peak timing and
  polarity by eye.
* Per-lead numbers over many records, printed and written to ``metrics.csv``: R^2,
  correlation, and the ratio of reconstructed to true standard deviation.
* Two baselines a trivial solution would match: predicting zeros (the mean of every
  z-scored lead) and copying the *best* visible lead for that masked lead, chosen per
  lead on the data itself, which is deliberately generous. A model that only smooths or
  copies will not beat them, and its standard-deviation ratio will sit well below 1.

    python scripts/inspect_reconstruction.py --checkpoint checkpoints/m5fasts1k1/final.pt \
        --meta-root ~/data/mimic_manifest.jsonl --visible II,V2,V5

Leads are named in the model's own order (``model.lead_order``, ``mimic`` by default:
I, II, III, aVR, aVF, aVL, V1-V6).
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import torch

from lvcg.data.angle import LEAD_ORDERS
from lvcg.data.pipeline import make_dataloader
from lvcg.models.lvcg import LVCG
from lvcg.utils.config import load_config

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "..", "configs", "train", "lvcg_v5_gru.yaml")


def _visible_indices(names: str, lead_names) -> List[int]:
    lookup = {name.lower(): index for index, name in enumerate(lead_names)}
    chosen = []
    for name in names.split(","):
        key = name.strip().lower()
        if key not in lookup:
            raise SystemExit(f"Unknown lead {name!r}; expected from {list(lead_names)}")
        chosen.append(lookup[key])
    return chosen


def _figure(truth, recon, lead_names, visible, path, fs, seconds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = min(int(seconds * fs), truth.shape[-1])
    time = np.arange(samples) / fs
    fig, axes = plt.subplots(6, 2, figsize=(14, 12), sharex=True)
    for lead, ax in enumerate(axes.T.reshape(-1)):
        seen = lead in visible
        ax.plot(time, truth[lead, :samples], color="black", linewidth=0.9, label="truth")
        ax.plot(time, recon[lead, :samples], color="crimson", linewidth=0.9,
                alpha=0.85, label="reconstruction")
        ax.set_ylabel(f"{lead_names[lead]}{' (visible)' if seen else ''}",
                      fontsize=9, color="tab:blue" if seen else "black")
        if seen:
            ax.set_facecolor("#f2f6ff")
        ax.tick_params(labelsize=8)
    axes[-1, 0].set_xlabel("seconds")
    axes[-1, 1].set_xlabel("seconds")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle(os.path.basename(path).replace(".png", ""), fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _accumulate(stats: Dict[str, np.ndarray], truth, recon, visible, num_leads):
    """Sum of squares per lead for the model and the two baselines."""
    error = ((recon - truth) ** 2).mean(axis=-1)            # [B, L]
    variance = truth.var(axis=-1)                           # [B, L]
    zeros = (truth ** 2).mean(axis=-1)                      # predicting 0
    copy = np.stack([((truth[:, [v]] - truth) ** 2).mean(axis=-1) for v in visible])
    stats["error"] += error.sum(axis=0)
    stats["variance"] += variance.sum(axis=0)
    stats["zeros"] += zeros.sum(axis=0)
    stats["copy"] += copy.min(axis=0).sum(axis=0)           # best visible lead, per record
    centred_truth = truth - truth.mean(-1, keepdims=True)
    centred_recon = recon - recon.mean(-1, keepdims=True)
    covariance = (centred_truth * centred_recon).mean(axis=-1)
    spread = truth.std(axis=-1) * recon.std(axis=-1)
    stats["corr"] += np.where(spread > 1e-12, covariance / np.maximum(spread, 1e-12), 0.0).sum(axis=0)
    stats["std_truth"] += truth.std(axis=-1).sum(axis=0)
    stats["std_recon"] += recon.std(axis=-1).sum(axis=0)
    stats["count"] += truth.shape[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--meta-root", help="override data.meta_root")
    parser.add_argument("--dataset-type", choices=("mimic_raw", "npy"), help="override data.dataset_type")
    parser.add_argument("--visible", default="II,V2,V5", help="visible lead names, comma separated")
    parser.add_argument("--records", type=int, default=6, help="records to draw")
    parser.add_argument("--metric-batches", type=int, default=20, help="validation batches for the numbers")
    parser.add_argument("--seconds", type=float, default=5.0, help="seconds to draw")
    parser.add_argument("--out", default="reports/reconstruction")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.meta_root:
        cfg.raw.setdefault("data", {})["meta_root"] = args.meta_root
    if args.dataset_type:
        cfg.raw.setdefault("data", {})["dataset_type"] = args.dataset_type
    lead_names = LEAD_ORDERS[cfg.model.get("lead_order", "mimic")]
    visible = _visible_indices(args.visible, lead_names)
    num_leads = len(lead_names)
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    model = LVCG.from_config(cfg).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    missing, unexpected = model.load_state_dict(state, strict=False)
    step = checkpoint.get("global_step", "?")
    print(f"Checkpoint step {step}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    if missing or unexpected:
        print(f"  missing: {list(missing)[:5]}\n  unexpected: {list(unexpected)[:5]}")

    loader = make_dataloader(cfg.raw, "val")
    print(f"Validation records: {len(loader.dataset)}; visible leads "
          f"{[lead_names[v] for v in visible]}; masked "
          f"{[lead_names[l] for l in range(num_leads) if l not in visible]}")

    stats = {key: np.zeros(num_leads) for key in
             ("error", "variance", "zeros", "copy", "corr", "std_truth", "std_recon")}
    stats["count"] = 0
    drawn = 0
    fs = int(cfg.data.get("fs", 100))
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= args.metric_batches:
                break
            ecg = batch["ecg"].to(device)
            indices = torch.tensor(visible, device=device).expand(ecg.shape[0], len(visible))
            recon = model.forward_train(ecg, indices)["recon"]
            truth_np = ecg.cpu().numpy()
            recon_np = recon.float().cpu().numpy()
            _accumulate(stats, truth_np, recon_np, visible, num_leads)
            while drawn < args.records and drawn < truth_np.shape[0] * (index + 1):
                position = drawn % truth_np.shape[0]
                name = str(batch["id"][position]).replace("/", "_")
                path = os.path.join(args.out, f"reconstruction_{drawn:02d}_{name}.png")
                _figure(truth_np[position], recon_np[position], lead_names, visible, path, fs, args.seconds)
                drawn += 1
                if position + 1 >= truth_np.shape[0]:
                    break

    count = max(stats["count"], 1)
    r2 = 1 - stats["error"] / np.maximum(stats["variance"], 1e-12)
    r2_zeros = 1 - stats["zeros"] / np.maximum(stats["variance"], 1e-12)
    r2_copy = 1 - stats["copy"] / np.maximum(stats["variance"], 1e-12)
    correlation = stats["corr"] / count
    std_ratio = stats["std_recon"] / np.maximum(stats["std_truth"], 1e-12)

    header = f"{'lead':>5} {'MSE':>8} {'R2':>7} {'R2 zeros':>9} {'R2 copy':>8} {'corr':>6} {'std ratio':>10}"
    print(f"\n{stats['count']} records, {args.metric_batches} batches\n{header}\n" + "-" * len(header))
    rows = []
    for lead in range(num_leads):
        tag = f"{lead_names[lead]}{'*' if lead in visible else ''}"
        row = dict(lead=lead_names[lead], visible=lead in visible,
                   mse=stats["error"][lead] / count, r2=r2[lead], r2_zeros=r2_zeros[lead],
                   r2_copy=r2_copy[lead], corr=correlation[lead], std_ratio=std_ratio[lead])
        rows.append(row)
        print(f"{tag:>5} {row['mse']:>8.4f} {row['r2']:>7.3f} {row['r2_zeros']:>9.3f} "
              f"{row['r2_copy']:>8.3f} {row['corr']:>6.3f} {row['std_ratio']:>10.3f}")
    masked = [r for r in rows if not r["visible"]]
    print(f"\nMasked leads: mean R2 {np.mean([r['r2'] for r in masked]):.3f}, "
          f"zeros {np.mean([r['r2_zeros'] for r in masked]):.3f}, "
          f"best-visible copy {np.mean([r['r2_copy'] for r in masked]):.3f}, "
          f"std ratio {np.mean([r['std_ratio'] for r in masked]):.3f}")
    print("A reconstruction worth trusting beats both baselines and keeps the standard "
          "deviation ratio near 1; a smoothed or copied output does not.")

    csv_path = os.path.join(args.out, "metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write("lead,visible,mse,r2,r2_zeros,r2_copy,corr,std_ratio\n")
        for row in rows:
            handle.write(f"{row['lead']},{int(row['visible'])},{row['mse']:.6f},{row['r2']:.6f},"
                         f"{row['r2_zeros']:.6f},{row['r2_copy']:.6f},{row['corr']:.6f},"
                         f"{row['std_ratio']:.6f}\n")
    print(f"\n{drawn} figures and {csv_path} written to {args.out}")


if __name__ == "__main__":
    main()
