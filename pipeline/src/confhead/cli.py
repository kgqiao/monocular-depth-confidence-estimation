"""Command line entry point: `confhead <command>`."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from .nuscenes import CAMERA_CHANNELS, NuScenes, resolve_dataroot
from .overlay import draw_overlay
from .project import project_lidar, project_radar
from .report import density_rows, write_csv, write_summary

PIPELINE_DIR = Path(__file__).resolve().parents[2]


def load(args) -> NuScenes:
    return NuScenes(resolve_dataroot(args.dataroot), args.version)


def cmd_scenes(args) -> None:
    nusc = load(args)
    total = 0
    for scene in nusc.scenes:
        n = len(nusc.samples_of_scene[scene["token"]])
        total += n
        print(f"{scene['name']}  {nusc.condition(scene):5}  {n:3d} keyframes  {scene['description']}")
    print(f"{len(nusc.scenes)} scenes, {total} keyframes")


def cmd_overlays(args) -> None:
    nusc = load(args)
    out = Path(args.out) / "overlays"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for scene in nusc.scenes:
        frame = nusc.frame(scene["first_sample_token"])
        for channel in CAMERA_CHANNELS:
            cam = nusc.camera(frame, channel)
            lidar = project_lidar(nusc, frame, cam)
            radar = project_radar(nusc, frame, cam)
            title = (f"{scene['name']} ({nusc.condition(scene)})  {channel}  "
                     f"lidar {lidar.n_pixels} px ({lidar.fraction:.2%})  "
                     f"radar {radar.n_pixels} px from {radar.n_points} points")
            img = draw_overlay(Image.open(cam.image_path), lidar, radar, title)
            img.save(out / f"{scene['name']}_{channel}.jpg", quality=90)
            n += 1
    print(f"wrote {n} overlays to {out}")


def cmd_report(args) -> None:
    nusc = load(args)
    rows = density_rows(nusc)
    csv_path = Path(args.out) / "projection_density.csv"
    write_csv(rows, csv_path)
    write_summary(rows, nusc, Path(args.summary))
    print(f"wrote {len(rows)} rows to {csv_path}")
    print(f"wrote the summary to {args.summary}")


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataroot", help="nuScenes folder; default $NUSCENES_DATAROOT")
    common.add_argument("--version", default="v1.0-mini")

    parser = argparse.ArgumentParser(prog="confhead", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scenes", parents=[common], help="list the scenes on disk")
    p.set_defaults(run=cmd_scenes)

    p = sub.add_parser("overlays", parents=[common],
                       help="draw lidar and radar on the first keyframe of each scene")
    p.add_argument("--out", default=PIPELINE_DIR / "outputs")
    p.set_defaults(run=cmd_overlays)

    p = sub.add_parser("report", parents=[common],
                       help="lidar and radar pixel density over every keyframe camera image")
    p.add_argument("--out", default=PIPELINE_DIR / "outputs")
    p.add_argument("--summary", default=PIPELINE_DIR / "results" / "projection-density.md")
    p.set_defaults(run=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.run(args)
