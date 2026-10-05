"""Lidar and radar pixel density over every keyframe camera image."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .nuscenes import CAMERA_CHANNELS, NuScenes
from .project import MAX_DEPTH, MIN_DEPTH, project_into

FIELDS = [
    "scene", "sample_token", "channel", "condition",
    "lidar_points", "lidar_pixels", "lidar_fraction",
    "radar_points", "radar_pixels", "radar_fraction",
]


def density_rows(nusc: NuScenes) -> list[dict]:
    rows = []
    for scene in nusc.scenes:
        condition = nusc.condition(scene)
        for sample in nusc.samples_of_scene[scene["token"]]:
            frame = nusc.frame(sample["token"])
            lidar_pts = nusc.read_lidar(frame)[:, :3]
            radar_pts = nusc.read_all_radar(frame)
            for channel in CAMERA_CHANNELS:
                cam = nusc.camera(frame, channel)
                lidar = project_into(cam, lidar_pts)
                radar = project_into(cam, radar_pts)
                rows.append({
                    "scene": scene["name"], "sample_token": sample["token"],
                    "channel": channel, "condition": condition,
                    "lidar_points": lidar.n_points, "lidar_pixels": lidar.n_pixels,
                    "lidar_fraction": lidar.fraction,
                    "radar_points": radar.n_points, "radar_pixels": radar.n_pixels,
                    "radar_fraction": radar.fraction,
                })
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _table(rows: list[dict], key: str, order: list[str]) -> list[str]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    groups["all"] = rows
    lines = [
        f"| {key.capitalize()} | Images | Lidar pixels | Lidar fraction | Radar points | Radar pixels | Radar fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in [k for k in order if k in groups] + ["all"]:
        g = groups[name]
        mean = lambda f: float(np.mean([r[f] for r in g]))  # noqa: E731
        lines.append(
            f"| {name} | {len(g)} | {mean('lidar_pixels'):.0f} | {mean('lidar_fraction'):.3%} "
            f"| {mean('radar_points'):.1f} | {mean('radar_pixels'):.1f} | {mean('radar_fraction'):.4%} |"
        )
    return lines


def write_summary(rows: list[dict], nusc: NuScenes, path: Path) -> None:
    n_samples = len({r["sample_token"] for r in rows})
    by_condition = defaultdict(set)
    for r in rows:
        by_condition[r["condition"]].add(r["scene"])
    scenes = ", ".join(f"{c} {len(s)}" for c, s in sorted(by_condition.items()))
    no_radar = sum(r["radar_points"] == 0 for r in rows)
    lines = [
        "# Lidar and radar density in the camera images",
        "",
        f"Written by `confhead report`. nuScenes {nusc.version}: {len(nusc.scenes)} scenes ({scenes}), "
        f"{n_samples} keyframes, {len(rows)} camera images of 1600 x 900.",
        "",
        f"Each image gets the keyframe lidar sweep and the keyframe returns of all five radars "
        f"(devkit filter: `invalid_state` 0, `ambig_state` 3), with no earlier sweeps. "
        f"Points count from {MIN_DEPTH:g} m to {MAX_DEPTH:g} m in depth along the optical axis. "
        f"A pixel counts once, however many points land on it. All values are means per image.",
        "",
        "## Per camera",
        "",
        *_table(rows, "channel", CAMERA_CHANNELS),
        "",
        "## Per condition",
        "",
        *_table(rows, "condition", ["day", "night"]),
        "",
        f"{no_radar} of {len(rows)} images get no radar point.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
