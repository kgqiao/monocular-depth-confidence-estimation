"""
label_count_table.py — count labeled pixels per BEVCar condition.

Reads the nuScenes-trainval cache (no GPU, no dataroot needed) and prints
a table with one row per BEVCar condition (day / rain / night / total):

  condition | scenes | images | labeled_cells | wrong_cells | wrong_pct

Usage (from confhead/):
    uv run python -m evals.label_count_table \
        --cache-root /scratch1/tawde/confhead-cache/nuscenes-trainval
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from confhead.cache.store import Cache
from confhead.labels import LabelConfig, lidar_labels
from evals.bevcar_splits import BEVCAR_CONDITION


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", required=True,
                   help="path to nuscenes-trainval cache")
    p.add_argument("--tau", type=float, default=None,
                   help="AbsRel threshold (default: LabelConfig default = 0.10)")
    return p.parse_args()


def main():
    args = parse_args()
    label_cfg = LabelConfig()
    if args.tau is not None:
        label_cfg.tau = args.tau
    tau = label_cfg.tau

    cache = Cache(root=args.cache_root, dataroot=None)
    rows = cache.rows(tag="val")
    print(f"Val entries: {len(rows)}  |  tau = {tau}\n")

    stats = defaultdict(lambda: {
        "scenes": set(),
        "images": 0,
        "labeled": 0,
        "wrong": 0,
    })

    for row in rows:
        entry  = cache.entry(row)
        arrays = entry.load(["lidar"])
        abs_rel, target_grid, mask = lidar_labels(entry.meta, arrays, label_cfg)

        scene = row["scene"]
        cond  = BEVCAR_CONDITION.get(scene, "unknown")

        n_labeled = int(mask.sum())
        n_wrong   = int(target_grid[mask].sum())

        for key in (cond, "total"):
            stats[key]["scenes"].add(scene)
            stats[key]["images"]  += 1
            stats[key]["labeled"] += n_labeled
            stats[key]["wrong"]   += n_wrong

    header = f"{'condition':<10} {'scenes':>7} {'images':>7} {'labeled_cells':>14} {'wrong_cells':>12} {'wrong_%':>8}"
    print(header)
    print("-" * len(header))

    for cond in ("day", "rain", "night", "unknown", "total"):
        if cond not in stats:
            continue
        s = stats[cond]
        n_scenes  = len(s["scenes"])
        n_images  = s["images"]
        n_labeled = s["labeled"]
        n_wrong   = s["wrong"]
        pct       = 100.0 * n_wrong / n_labeled if n_labeled > 0 else float("nan")
        print(f"{cond:<10} {n_scenes:>7} {n_images:>7} {n_labeled:>14,} {n_wrong:>12,} {pct:>7.2f}%")


if __name__ == "__main__":
    main()