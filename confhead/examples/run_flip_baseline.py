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
from confhead.depth import FrozenDepthAnything, preprocess
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

    rows = cache.rows(tag="val")
    print(f"Val entries: {len(rows)}  |  tau = {tau}")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "flip_baseline.csv"
    npz_path = out_dir / "flip_baseline_cells.npz"

    import torch
    device = next(depth_model.parameters()).device

    all_p_error: list[float] = []
    all_abs_rel: list[float] = []
    all_target:  list[int]   = []
    all_scene:   list[str]   = []
    csv_rows: list[dict] = []

    for i, row in enumerate(rows):
        entry  = cache.entry(row)
        arrays = entry.load(["lidar"])
        abs_rel_grid, target_grid, mask = lidar_labels(entry.meta, arrays, label_cfg)

        if mask.sum() == 0:
            continue

        # image → flip confidence
        img_path  = pathlib.Path(args.dataroot) / entry.meta["image"]["path"]
        img_np    = preprocess(img_path, short_side=518)               # (3, H, W)
        img_t     = torch.from_numpy(img_np).unsqueeze(0).to(device)   # (1,3,H,W)
        conf_full = flip_confidence(depth_model, img_t)                # (H, W) float32

        # pool image-space confidence down to BEV grid (7× stride)
        mw, mh = entry.meta["model_size"]   # 924, 518
        gw, gh = entry.meta["grid_size"]    # 132, 74
        stride  = mh // gh                  # 7
        conf_grid = conf_full.reshape(gh, stride, gw, stride).mean(axis=(1, 3))

        abs_rel_flat = abs_rel_grid[mask]
        conf_flat    = conf_grid[mask]
        target_flat  = target_grid[mask]

        auroc, ece, ause, aurg, brier = _per_image_metrics(abs_rel_flat, conf_flat, tau)

        scene = row["scene"]
        cond  = BEVCAR_CONDITION.get(scene, "unknown")

        csv_rows.append({
            "scene":     scene,
            "condition": cond,
            "token":     entry.meta.get("sample_token", ""),
            "n_cells":   int(mask.sum()),
            "auroc":     auroc,
            "ece":       ece,
            "ause":      ause,
            "aurg":      aurg,
            "brier":     brier,
        })

        all_p_error.extend((1.0 - conf_flat).tolist())
        all_abs_rel.extend(abs_rel_flat.tolist())
        all_target.extend(target_flat.astype(int).tolist())
        all_scene.extend([scene] * int(mask.sum()))

        if (i + 1) % 500 == 0 or i == 0:
            print(f"  [{i+1}/{len(rows)}] {scene} cond={cond} cells={mask.sum()}")

    # write CSV
    fieldnames = ["scene", "condition", "token", "n_cells", "auroc", "ece", "ause", "aurg", "brier"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nWrote {csv_path}")

    # write cells.npz
    np.savez(
        npz_path,
        p_error=np.array(all_p_error, dtype=np.float32),
        abs_rel=np.array(all_abs_rel, dtype=np.float32),
        target =np.array(all_target,  dtype=np.int8),
        scene  =np.array(all_scene,   dtype=object),
    )
    print(f"Wrote {npz_path}")

    # per-condition summary
    print("\n=== per-BEVCar-condition summary ===")
    cond_rows: dict = defaultdict(list)
    for r in csv_rows:
        cond_rows[r["condition"]].append(r)
        cond_rows["total"].append(r)

    def _nanmean(vals):
        v = [x for x in vals if not (isinstance(x, float) and np.isnan(x))]
        return float(np.mean(v)) if v else float("nan")

    hdr = f"{'condition':<10} {'n_img':>6} {'auroc':>7} {'ece':>7} {'ause':>7} {'aurg':>7} {'brier':>7}"
    print(hdr)
    print("-" * len(hdr))
    for cond in ("day", "rain", "night", "unknown", "total"):
        if cond not in cond_rows:
            continue
        rr = cond_rows[cond]
        print(
            f"{cond:<10} {len(rr):>6}"
            f" {_nanmean([r['auroc'] for r in rr]):>7.4f}"
            f" {_nanmean([r['ece']   for r in rr]):>7.4f}"
            f" {_nanmean([r['ause']  for r in rr]):>7.4f}"
            f" {_nanmean([r['aurg']  for r in rr]):>7.4f}"
            f" {_nanmean([r['brier'] for r in rr]):>7.4f}"
        )


if __name__ == "__main__":
    main()
