"""One table for the two checks after pretraining: reconstruction and linear probing.

Reads what the other two scripts wrote and prints them side by side:

* ``reports/reconstruction/metrics.csv`` from ``scripts/inspect_reconstruction.py`` --
  per-lead R^2 against the two baselines, summarised over visible and masked leads.
* every probing CSV under ``probing/results/`` -- test AUROC per dataset, label ratio and
  seed, next to the numbers in Table 1 of the paper. Rows a failed run left behind
  (AUROC -1) are dropped and counted.

    python scripts/summarize_eval.py
    python scripts/summarize_eval.py --probing probing/results/probing_m5fast.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict
from statistics import mean, pstdev

# Table 1 of the paper (LVCG row), test AUROC x100 at 1% / 10% / 100% labels.
PAPER = {
    "ptbxl_super_class": {0.01: 75.33, 0.1: 79.03, 1.0: 80.13},
    "ptbxl_sub_class": {0.01: 70.61, 0.1: 74.62, 1.0: 79.19},
    "ptbxl_form": {0.01: 52.28, 0.1: 59.12, 1.0: 71.24},
    "ptbxl_rhythm": {0.01: 72.03, 0.1: 79.87, 1.0: 83.94},
    "icbeb": {0.01: 71.09, 0.1: 79.44, 1.0: 84.15},
    "chapman": {0.01: 62.47, 0.1: 75.17, 1.0: 84.14},
}


def _reconstruction(path: str) -> None:
    if not os.path.exists(path):
        print(f"Reconstruction: {path} not found; run scripts/inspect_reconstruction.py\n")
        return
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    groups = {"visible": [r for r in rows if r["visible"] == "1"],
              "masked": [r for r in rows if r["visible"] == "0"]}
    print(f"Masked-lead reconstruction ({path})")
    print(f"  {'leads':>8} {'n':>3} {'R2':>7} {'R2 zeros':>9} {'R2 copy':>8} {'corr':>6} {'std ratio':>10}")
    for name, group in groups.items():
        if not group:
            continue
        pick = lambda key: mean(float(r[key]) for r in group)  # noqa: E731
        print(f"  {name:>8} {len(group):>3} {pick('r2'):>7.3f} {pick('r2_zeros'):>9.3f} "
              f"{pick('r2_copy'):>8.3f} {pick('corr'):>6.3f} {pick('std_ratio'):>10.3f}")
    masked = groups["masked"]
    if masked:
        worst = sorted(masked, key=lambda r: float(r["r2"]))[:3]
        print("  weakest masked leads: " + ", ".join(f"{r['lead']} R2 {float(r['r2']):.3f}" for r in worst))
    print()


def _probing(paths) -> None:
    rows, dropped = [], 0
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if float(row.get("auroc", -1)) < 0:
                    dropped += 1
                    continue
                rows.append(row)
    if not rows:
        print("Probing: no completed runs found")
        return
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["dataset"], float(row["label_ratio"]))].append(row)

    print(f"Linear probing, frozen backbone ({len(rows)} runs"
          + (f", {dropped} failed rows ignored)" if dropped else ")"))
    header = (f"  {'model':>6} {'dataset':>18} {'labels':>7} {'seeds':>5} {'test AUROC':>11} "
              f"{'val AUROC':>10} {'F1':>6} {'paper':>7} {'diff':>7}")
    print(header + "\n  " + "-" * (len(header) - 2))
    for key in sorted(grouped, key=lambda k: (k[0], k[1], k[2])):
        model, dataset, ratio = key
        group = grouped[key]
        auroc = [float(r["auroc"]) * 100 for r in group]
        spread = f" +-{pstdev(auroc):.2f}" if len(auroc) > 1 else ""
        reference = PAPER.get(dataset, {}).get(ratio)
        paper = f"{reference:.2f}" if reference else "-"
        diff = f"{mean(auroc) - reference:+.2f}" if reference else "-"
        print(f"  {model:>6} {dataset:>18} {ratio:>7.0%} {len(group):>5} "
              f"{mean(auroc):>6.2f}{spread:>5} {mean(float(r['val_auroc']) for r in group) * 100:>10.2f} "
              f"{mean(float(r['f1']) for r in group):>6.3f} {paper:>7} {diff:>7}")
    print("\n  paper: Table 1, LVCG row (test AUROC x100). diff = ours - paper.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reconstruction", default="reports/reconstruction/metrics.csv")
    parser.add_argument("--probing", nargs="*", default=None,
                        help="probing CSVs (default: every CSV in probing/results/)")
    args = parser.parse_args()

    _reconstruction(args.reconstruction)
    _probing(args.probing if args.probing is not None else sorted(glob.glob("probing/results/*.csv")))


if __name__ == "__main__":
    main()
