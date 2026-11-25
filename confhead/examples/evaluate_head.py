"""
Score one trained confidence head on the val split — CARC evaluation script.

Loads a trained head and rebuilds its inputs from the cache, so Depth Anything
doesn't run again. For every val image, it scores the head's p_error on the
labeled cells (AUSE, AURG, AUROC, ECE, Brier) and writes one CSV row.

Usage (from confhead/):
    uv run python -m examples.evaluate_head \
        --checkpoint /scratch1/naday/confhead-runs/run-v1-radar/best.pt \
        --out-dir    results/run-v1-radar 

Outputs:
    <out-dir>/<run>.csv     one row per val image, written as it goes
                            (default <out-dir>: results/<run>)
    /scratch1/$USER/confhead-eval/<run>/<run>_cells.npz
                            per-cell arrays (p_error, abs_rel, target, scene);
                            on scratch because it's too large for git
    per-BEVCar-condition AUROC, ECE, AUSE, AURG, Brier printed to stdout

<run> is the checkpoint's folder name, e.g. run-v1-radar. The CSV columns match
results/flip_baseline/flip_baseline.csv, so examples/build_results_table.py
picks it up.
"""

from __future__ import annotations

import os
import argparse
import csv
import pathlib
from collections import defaultdict

import numpy as np

from torch.utils.data import DataLoader, Subset

from confhead import Cache, ConfidenceDataset, load_head
from evals.bevcar_splits import BEVCAR_CONDITION
from evals.metrics import (
    calculate_rejection_auc,
    calculate_ause_and_aurg,
    compute_ece,
    compute_sparsification_curves,
    compute_brier_score,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--limit",      type = int, default = None)
    p.add_argument("--out-dir",    default= None,
                   help="directory to write CSV and cells.npz (default: results/<run>)")
    return p.parse_args()


def _per_image_metrics(abs_rel_flat, is_correct, conf_flat):
    """Compute per-image scalar metrics; return NaN if too few cells."""
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

    brier = compute_brier_score(is_correct, conf_flat)

    return auroc, ece, ause, aurg, brier


def main():
    args = parse_args()

    run = pathlib.Path(args.checkpoint).parent.name
    out_dir = pathlib.Path(args.out_dir or f"results/{run}")
    csv_path = out_dir / f"{run}.csv"
    npz_path = pathlib.Path(f"/scratch1/{os.environ.get('USER', 'local')}/confhead-eval/{run}") / f"{run}_cells.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading head...")
    head, inputs, info = load_head(args.checkpoint, device)
    print(f"  {run}: {len(info['channels'])} channels, tau {inputs.labels.tau}, epoch {info['epoch']}")

    print("Loading cache...")
    cache = Cache(info["cache"]["root"])
    data = ConfidenceDataset(cache, inputs, tag="val")
    if data.channel_names != info["channels"]:
        raise ValueError(f"{cache.root} gives other channels than {run} was trained on")
    if cache.header["fingerprint"] != info["cache"]["fingerprint"]:
        print("  warning: this cache is another build than the training cache")

    rows = data.rows
    if args.limit:
        data = Subset(data, range(args.limit))
    loader = DataLoader(data, batch_size=8, num_workers=4)
    print(f"Val images: {len(data)}")

    all_p_error: list[np.ndarray] = []
    all_abs_rel: list[np.ndarray] = []
    all_target:  list[np.ndarray] = []
    all_scene:   list[np.ndarray] = []
    csv_rows: list[dict] = []

    fieldnames = ["key", "scene", "camera", "condition", "n_labeled", "n_wrong",
                  "ause", "aurg", "auroc", "ece", "brier"]
    counter = 0      # which val image we're on, so rows[counter] is its cache row
    skipped = 0      # images with no lidar labels

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        with torch.no_grad():
            for batch in loader:
                p_error = torch.sigmoid(head(batch["x"].to(device))).cpu().numpy()[:, 0]
                for j, key in enumerate(batch["key"]):
                    row = rows[counter]
                    counter += 1
                    assert row["key"] == key, f"row {row['key']} != image {key}"

                    mask = batch["mask"][j, 0].numpy() > 0
                    if mask.sum() == 0:
                        skipped += 1
                        continue

                    abs_rel_flat = batch["abs_rel"][j, 0].numpy()[mask]
                    target_flat = batch["target"][j, 0].numpy()[mask].astype(int)
                    p_flat = p_error[j][mask]
                    is_correct = 1 - target_flat
                    conf_flat = 1.0 - p_flat

                    auroc, ece, ause, aurg, brier = _per_image_metrics(abs_rel_flat, is_correct, conf_flat)
                    scene = row["scene"]
                    cond = BEVCAR_CONDITION.get(scene, "unknown")

                    r = {"key": key, "scene": scene, "camera": row["channel"], "condition": cond,
                         "n_labeled": int(mask.sum()), "n_wrong": int(target_flat.sum()),
                         "ause": ause, "aurg": aurg, "auroc": auroc, "ece": ece, "brier": brier}
                    writer.writerow(r)
                    f.flush()
                    csv_rows.append(r)

                    all_p_error.append(p_flat.astype(np.float32))
                    all_abs_rel.append(abs_rel_flat.astype(np.float32))
                    all_target.append(target_flat.astype(np.int8))
                    all_scene.append(np.full(len(p_flat), scene))

                # progress: counts only, so it works even if the first images have no labels.
                if counter % 800 < len(batch["key"]):
                    print(f"  [{counter}/{len(data)}] scored {len(csv_rows)}, skipped {skipped}")

    if not csv_rows:
        print("No images with labels; nothing to save.")
        return
    
    print(f"\nWrote {csv_path}: {len(csv_rows)} images scored, {skipped} skipped (no labels)")

    p = np.concatenate(all_p_error)
    a = np.concatenate(all_abs_rel)
    t = np.concatenate(all_target)
    s = np.concatenate(all_scene)

    np.savez(npz_path, p_error = p, abs_rel = a, target = t, scene = s)
    print(f"Wrote {npz_path}")


    # per-condition summary
    print(f"\n=== {run}: mean of per-image metrics, by BEVCar condition ===")
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


    # pooled over all labeled cells of a condition (the deep-review checklist's choice)
    cell_cond = np.array([BEVCAR_CONDITION.get(x, "unknown") for x in s])

    print(f"\n=== {run}: pooled over all labeled cells, by BEVCar condition ===")
    for cond in ("day", "rain", "night"):
        m = cell_cond == cond
        if m.sum() == 0:
            continue
        is_correct = 1 - t[m].astype(int)
        conf = 1.0 - p[m]
        print(f"{cond:<10} cells={m.sum():>9} auroc={calculate_rejection_auc(is_correct, conf):.4f}"
              f" ece={compute_ece(is_correct, conf):.4f}")


if __name__ == "__main__":
    main()
