"""One table per Quaternion stage: every run grouped by variant, with the delta against V0.

Reads the CSVs the stage scripts append to (``probing/results/qdf_v1.csv``,
``qdt_v2.csv``, ``mrq_v3.csv``, ...) and prints, for each variant, the mean and spread of
test macro AUROC over seeds and its distance from that stage's own V0.

Each stage is compared only against the V0 row from the same CSV, because the stages
differ in how they replay the pretrained path: V1 takes the released embedding, V2
replays the GRU with each record's own beat count. Mixing them would compare two
different V0 definitions.

    python scripts/summarize_quaternion.py
    python scripts/summarize_quaternion.py --results probing/results/qdt_v2.csv --metric val_macro_auroc
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict
from statistics import mean, pstdev

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ("NORM", "MI", "STTC", "CD", "HYP")
# Long feature strings would push the table out of a terminal; these are the sets in use.
SHORT_FEATURES = {
    "q+theta+omega+magnitude": "q/th/om/mag",
    "q+theta+omega": "q/th/om",
    "position+next_position+delta": "xyz-pair",
}


def _variant(row) -> str:
    """The run's identity: model, features or components, fusion and tag, minus the seed."""
    parts = [row["model"]]
    # V3 names its runs by variant (m / o / q / mq / oq / mo / moq / cdf); V1 and V2 by
    # the feature set. An older V3 CSV used a "components" column.
    if row.get("variant"):
        parts = [row["variant"]]
    elif row.get("components"):
        parts = [row["components"].replace("+", " + ")]
    features = row.get("features", "")
    if row["model"] != "v0" and features:
        parts.append(SHORT_FEATURES.get(features, features.replace("+", "/")))
    if row.get("fusion"):
        parts.append(row["fusion"])
    if float(row.get("fusion_init_std") or 0):
        parts.append(f"init{row['fusion_init_std']}")
    if row.get("tag"):
        parts.append(row["tag"].lstrip("_"))
    if float(row.get("label_ratio", 1.0)) != 1.0:
        parts.append(f"{float(row['label_ratio']):.0%}")
    return " ".join(parts)


def summarize(path: str, metric: str) -> None:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get(metric)]
    if not rows:
        print(f"{os.path.basename(path)}: no runs\n")
        return

    groups = defaultdict(list)
    for row in rows:
        groups[_variant(row)].append(row)
    v0 = {int(r["seed"]): float(r[metric]) for r in rows if r["model"] == "v0"}
    is_v0 = lambda name: name.startswith("v0") or name == "vcg"  # noqa: E731
    reference = mean(v0.values()) if v0 else None

    print(f"{os.path.basename(path)} — {len(rows)} runs, metric {metric}")
    width = max(12, min(30, max(len(n) for n in groups)))
    header = (f"  {'variant':>{width}} {'seeds':>5} {'AUROC':>7} {'+-':>5} {'vs V0':>7} "
              f"{'paired':>7} {'epoch':>6} {'params':>9}")
    print(header + "\n  " + "-" * (len(header) - 2))
    for name in sorted(groups, key=lambda n: (not is_v0(n), n)):
        group = groups[name]
        scores = [float(r[metric]) * 100 for r in group]
        spread = f"{pstdev(scores):.2f}" if len(scores) > 1 else "-"
        delta = f"{mean(scores) - reference * 100:+.2f}" if reference else "-"
        # Paired: only over the seeds where a V0 run exists, which removes seed variance.
        paired = [float(r[metric]) * 100 - v0[int(r["seed"])] * 100
                  for r in group if int(r["seed"]) in v0]
        paired_text = f"{mean(paired):+.2f}" if paired and not is_v0(name) else "-"
        epochs = mean(float(r["best_epoch"]) for r in group)
        params = int(float(group[0]["trainable_params"]))
        print(f"  {name[:width]:>{width}} {len(group):>5} {mean(scores):>7.2f} {spread:>5} {delta:>7} "
              f"{paired_text:>7} {epochs:>6.1f} {params:>9,}")

    best = max((g for n, g in groups.items() if not is_v0(n)), default=None,
               key=lambda g: mean(float(r[metric]) for r in g))
    if best:
        print("\n  per-label test AUROC (seed of the first run of each variant):")
        print(f"    {'variant':>{width}} " + " ".join(f"{label:>7}" for label in LABELS))
        for name in sorted(groups, key=lambda n: (not is_v0(n), n)):
            row = groups[name][0]
            values = [row.get(f"auroc_{label}") for label in LABELS]
            if all(values):
                print(f"    {name[:width]:>{width}} " + " ".join(f"{float(v):>7.4f}" for v in values))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", nargs="*", help="CSVs (default: every stage under probing/results)")
    parser.add_argument("--metric", default="test_macro_auroc",
                        help="test_macro_auroc (default), val_macro_auroc, test_macro_f1, ...")
    args = parser.parse_args()

    # One CSV per stage: qdf_v1, qdt_v2, mrq_v3, ... ordered by the stage number.
    paths = args.results or sorted(
        glob.glob(os.path.join(REPO_ROOT, "probing", "results", "*_v[0-9]*.csv")),
        key=lambda path: os.path.basename(path).rsplit("_v", 1)[-1],
    )
    if not paths:
        raise SystemExit("No stage CSVs found; run scripts/train_qdf.py or train_qdt.py first")
    for path in paths:
        summarize(path if os.path.isabs(path) else os.path.join(REPO_ROOT, path), args.metric)


if __name__ == "__main__":
    main()
