"""
Flip-augmentation confidence baseline — CARC evaluation script.

Runs the depth model on each val image and its horizontal mirror.
Disagreement between the two disparity maps is used as uncertainty
(no trained confidence head required).

Usage (from confhead/):
    uv run python -m examples.run_flip_baseline \
        --depth-ckpt /path/to/depth_anything_v2_vitb.pth \
        --cache-root /path/to/cache \
        --dataroot   /path/to/nuscenes \
        --out-dir    results/flip_baseline

Outputs:
    <out-dir>/flip_baseline.csv       one row per val image
    <out-dir>/flip_baseline_cells.npz per-cell arrays (p_error, abs_rel, target, scene)
    per-BEVCar-condition AUC, ECE, AUSE, AURG printed to stdout
"""

from __future__ import annotations

import argparse
import csv
import pathlib
from collections import defaultdict

import numpy as np

from confhead.cache.store import Cache
from confhead.depth import FrozenDepthAnything
from confhead.labels import LabelConfig, lidar_labels
from evals.bevcar_splits import BEVCAR_CONDITION
from evals.flip_confidence import flip_confidence
from evals.metrics import (
    calculate_rejection_auc,
    calculate_ause_and_aurg,
    compute_ece,
    compute_sparsification_curves,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--depth-ckpt", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--dataroot",   required=True)
    p.add_argument("--device",     default="auto")
    p.add_argument("--out-dir",    default="results/flip_baseline",
                   help="directory to write CSV and cells.npz")
    return p.parse_args()


def _per_image_metrics(abs_rel_flat, conf_flat, tau):
    """Compute per-image scalar metrics; return NaN if too few cells."""
    is_correct = (abs_rel_flat < tau).astype(int)
    uncertainty = 1.0 - conf_flat
    n = len(abs_rel_flat)

    if n < 2 or is_correct.sum() == 0 or is_correct.sum() == n:
        # AUROC undefined when only one class present
        auroc = float("nan")
    else:
        try:
            auroc = calculate_rejection_auc(is_correct, conf_flat)
        except Exception:
            auroc = float("nan")

    ece  = compute_ece(is_correct, conf_flat) if n >= 1 else float("nan")

    if n >= 2:
        fracs, mc, oc, rc = compute_sparsification_curves(abs_rel_flat, uncertainty)
        ause, aurg = calculate_ause_and_aurg(fracs, mc, oc, rc)
    else:
        ause = aurg = float("nan")

    brier = float(np.mean((conf_flat - is_correct.astype(np.float32)) ** 2)) if n >= 1 else float("nan")

    return auroc, ece, ause, aurg, brier


def main():
    args = parse_args()

    print("Loading depth model...")
    depth_model = FrozenDepthAnything(
        checkpoint=args.depth_ckpt,
        encoder="vitb",
        device=args.device,
    )

    print("Loading cache...")
    cache = Cache(root=args.cache_root, dataroot=args.dataroot)

    label_cfg = LabelConfig()   # tau = 0.10
    tau = label_cfg.tau
