"""
Flip-augmentation confidence baseline — CARC evaluation script.

Runs the depth model on each val image and its horizontal mirror.
Disagreement between the two disparity maps is used as uncertainty
(no trained confidence head required).

Usage (from confhead/):
    uv run python -m examples.run_flip_baseline \
        --depth-ckpt /path/to/depth_anything_v2_vitb.pth \
        --cache-root /path/to/cache \
        --dataroot   /path/to/nuscenes

Outputs per-BEVCar-condition AUC, ECE, AUSE, AURG.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from confhead.cache.store import Cache
from confhead.depth import FrozenDepthAnything
from confhead.labels import LabelConfig, lidar_labels
from evals.bevcar_splits import BEVCAR_CONDITION
from evals.flip_confidence import flip_confidence
from evals.metrics import calculate_rejection_auc, compute_ece, compute_sparsification_curves


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--depth-ckpt", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--dataroot",   required=True)
    p.add_argument("--device",     default="auto")
    return p.parse_args()


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
    results = defaultdict(lambda: {"errors": [], "confs": []})

    rows = cache.rows(tag="val")
    print(f"Running flip baseline on {len(rows)} val entries...\n")

    skipped = 0
    for i, row in enumerate(rows):
        entry  = cache.entry(row)
        arrays = entry.load(["lidar"])
        abs_rel, _target, mask = lidar_labels(entry.meta, arrays, label_cfg)

        if mask.sum() == 0:
            skipped += 1
            continue

        # run flip confidence — preprocess image and call the model
        import torch
        from confhead.depth import preprocess
        x = torch.from_numpy(preprocess(entry.image_path)).unsqueeze(0).to(depth_model.device)
        conf_full = flip_confidence(depth_model, x)   # (H_model, W_model)

        # resize confidence to label grid
        import torch.nn.functional as F
        gh, gw = abs_rel.shape
        conf_grid = F.interpolate(
            torch.from_numpy(conf_full).unsqueeze(0).unsqueeze(0),
            size=(gh, gw), mode="bilinear", align_corners=False,
        ).squeeze().numpy()

        abs_rel_flat = abs_rel[mask].astype(np.float32)
        conf_flat    = conf_grid[mask].astype(np.float32)

        scene     = row["scene"]
        bev_group = BEVCAR_CONDITION.get(scene, "unknown")

        results[bev_group]["errors"].append(abs_rel_flat)
        results[bev_group]["confs"].append(conf_flat)
        results["all"]["errors"].append(abs_rel_flat)
        results["all"]["confs"].append(conf_flat)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(rows)} entries...")

    if skipped:
        print(f"  (skipped {skipped} entries with no LiDAR labels)")

    # --- metrics ---
    tau = label_cfg.tau
    print("\n" + "=" * 55)
    print(f"  Flip-Augmentation Baseline  |  tau = {tau:.2f}")
    print("=" * 55)

    for group in ("all", "day", "rain", "night"):
        if group not in results or not results[group]["errors"]:
            print(f"\n[{group}]  no data")
            continue

        errors = np.concatenate(results[group]["errors"])
        confs  = np.concatenate(results[group]["confs"])
        uncertainty = 1.0 - confs
        is_correct  = (errors < tau).astype(int)

        auc  = calculate_rejection_auc(is_correct, confs)
        ece  = compute_ece(is_correct, confs)
        fracs, method_curve, oracle_curve, random_curve = compute_sparsification_curves(errors, uncertainty)
        ause = float(np.trapz(method_curve - oracle_curve, fracs))
        aurg = float(np.trapz(random_curve  - method_curve, fracs))

        if group == "all":
            n_scenes = len({r["scene"] for r in rows})
        else:
            n_scenes = len({r["scene"] for r in rows if BEVCAR_CONDITION.get(r["scene"]) == group})

        print(f"\n[{group}]  scenes={n_scenes}  cells={len(errors):,}")
        print(f"  AUC-ROC : {auc:.4f}")
        print(f"  ECE     : {ece:.4f}")
        print(f"  AUSE    : {ause:.4f}")
        print(f"  AURG    : {aurg:.4f}")

    print("\n" + "=" * 55)


if __name__ == "__main__":
    main()