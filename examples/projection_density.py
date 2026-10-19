"""Measure how many pixels of each keyframe camera image get a lidar or a radar return.

    uv run examples/projection_density.py

Writes one row per image to outputs/projection_density.csv and the summary
to results/projection-density.md.
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from confhead.items import load_points
from confhead.nuscenes import CAMERA_CHANNELS, DATAROOT_ENV, NuScenes

REPO = Path(__file__).resolve().parents[1]
MIN_DEPTH, MAX_DEPTH = 1.0, 80.0


def pixels(points: np.ndarray, width: int) -> int:
    return len(np.unique(np.round(points[:, 1]).astype(np.int64) * width + np.round(points[:, 0]).astype(np.int64)))


def table(rows: list[dict], key: str, order) -> list[str]:
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    groups["all"] = rows
    lines = [f"| {key.capitalize()} | Images | Lidar pixels | Lidar fraction | Radar points | Radar pixels | Radar fraction |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in [k for k in order if k in groups] + ["all"]:
        g = groups[name]
        m = {f: float(np.mean([r[f] for r in g])) for f in g[0] if f.startswith(("lidar", "radar"))}
        lines.append(f"| {name} | {len(g)} | {m['lidar_pixels']:.0f} | {m['lidar_fraction']:.3%} "
                     f"| {m['radar_points']:.1f} | {m['radar_pixels']:.1f} | {m['radar_fraction']:.4%} |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataroot", type=Path,
                        default=os.environ.get(DATAROOT_ENV, REPO / "datasets/nuscenes/v1.0-mini"))
    parser.add_argument("--csv", type=Path, default=REPO / "outputs/projection_density.csv")
    parser.add_argument("--summary", type=Path, default=REPO / "results/projection-density.md")
    args = parser.parse_args()

    nusc = NuScenes(args.dataroot)
    rows = []
    for item in nusc.items():
        lidar = load_points(item.lidar, item.view, MIN_DEPTH, MAX_DEPTH)
        radar = load_points(item.radar, item.view, MIN_DEPTH, MAX_DEPTH)
        area = item.view.width * item.view.height
        rows.append({
            "scene": item.scene, "sample_token": item.sample_token, "channel": item.channel,
            "condition": item.condition,
            "lidar_points": len(lidar), "lidar_pixels": pixels(lidar, item.view.width),
            "lidar_fraction": pixels(lidar, item.view.width) / area,
            "radar_points": len(radar), "radar_pixels": pixels(radar, item.view.width),
            "radar_fraction": pixels(radar, item.view.width) / area,
        })

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_condition = defaultdict(set)
    for r in rows:
        by_condition[r["condition"]].add(r["scene"])
    scenes = ", ".join(f"{c} {len(s)}" for c, s in sorted(by_condition.items()))
    keyframes = len({r["sample_token"] for r in rows})
    no_radar = sum(r["radar_points"] == 0 for r in rows)
    lines = [
        "# Lidar and radar density in the camera images",
        "",
        f"Written by `examples/projection_density.py`. nuScenes {nusc.version}: {len(nusc.scenes)} scenes ({scenes}), "
        f"{keyframes} keyframes, {len(rows)} camera images of 1600 x 900.",
        "",
        "Each image gets the keyframe lidar sweep and the keyframe returns of all five radars "
        "(the devkit's default filter: `invalid_state` 0, `ambig_state` 3, `dyn_prop` 0 to 6), with no earlier sweeps. "
        f"Points count from {MIN_DEPTH:g} m to {MAX_DEPTH:g} m in depth along the optical axis. "
        "A pixel counts once, however many points land on it. All values are means per image.",
        "",
        "## Per camera",
        "",
        *table(rows, "channel", CAMERA_CHANNELS),
        "",
        "## Per condition",
        "",
        *table(rows, "condition", ["day", "night"]),
        "",
        f"{no_radar} of {len(rows)} images get no radar point.",
        "",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines))
    print(f"wrote {len(rows)} rows to {args.csv} and the summary to {args.summary}")


if __name__ == "__main__":
    main()
