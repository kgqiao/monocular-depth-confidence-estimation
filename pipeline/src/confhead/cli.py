"""Command line entry point: `confhead <command>`."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image

from .nuscenes import CAMERA_CHANNELS, NuScenes, resolve_dataroot
from .overlay import draw_overlay
from .project import project_lidar, project_radar
from .report import density_rows, write_csv, write_summary

DEFAULT_VERSION = "v1.0-mini"


def load(args) -> NuScenes:
    nusc = NuScenes(resolve_dataroot(args.dataroot), args.version)
    if nusc.skipped_scenes:
        names = ", ".join(s["name"] for s in nusc.skipped_scenes)
        print(f"warning: skipped {len(nusc.skipped_scenes)} scenes with missing keyframe files: {names}",
              file=sys.stderr)
    return nusc


def default_summary(version: str) -> Path:
    name = "projection-density.md" if version == DEFAULT_VERSION else f"projection-density-{version}.md"
    return Path("results") / name


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
    summary = Path(args.summary) if args.summary else default_summary(nusc.version)
    if nusc.skipped_scenes and not args.summary:
        raise SystemExit(f"Scenes are missing, so the report would not cover the whole split. "
                         f"Pass --summary to write it anyway; {summary} is left as it is.")
    rows = density_rows(nusc)
    csv_path = Path(args.out) / "projection_density.csv"
    write_csv(rows, csv_path)
    write_summary(rows, nusc, summary)
    print(f"wrote {len(rows)} rows to {csv_path}")
    print(f"wrote the summary to {summary}")


def cmd_depth_check(args) -> None:
    from .depth import DepthPredictor, resolve_checkpoint

    checkpoint = resolve_checkpoint(args.checkpoint)
    nusc = load(args)
    frame = nusc.frame(nusc.scenes[0]["first_sample_token"])
    cam = nusc.camera(frame, "CAM_FRONT")

    t0 = time.perf_counter()
    predictor = DepthPredictor(checkpoint)
    t1 = time.perf_counter()
    pred = predictor.predict(cam.image_path)
    t2 = time.perf_counter()
    pred = predictor.predict(cam.image_path)
    t3 = time.perf_counter()

    print(f"image   {cam.image_path.name} ({cam.width} x {cam.height})")
    print(f"device  {predictor.device}")
    print(f"output  {pred.shape} {pred.dtype}, relative inverse depth {pred.min():.3f} to {pred.max():.3f}")
    print(f"time    load {t1 - t0:.1f} s, first pass {t2 - t1:.2f} s, second pass {t3 - t2:.2f} s")


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataroot", help="nuScenes folder; default $NUSCENES_DATAROOT")
    common.add_argument("--version", default=DEFAULT_VERSION)

    parser = argparse.ArgumentParser(prog="confhead", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scenes", parents=[common], help="list the scenes on disk")
    p.set_defaults(run=cmd_scenes)

    p = sub.add_parser("overlays", parents=[common],
                       help="draw lidar and radar on the first keyframe of each scene")
    p.add_argument("--out", default="outputs")
    p.set_defaults(run=cmd_overlays)

    p = sub.add_parser("report", parents=[common],
                       help="lidar and radar pixel density over every keyframe camera image")
    p.add_argument("--out", default="outputs")
    p.add_argument("--summary", help="default results/projection-density.md, with -<version> for other splits")
    p.set_defaults(run=cmd_report)

    p = sub.add_parser("depth-check", parents=[common],
                       help="run Depth Anything V2 once on one CAM_FRONT keyframe")
    p.add_argument("--checkpoint", help="ViT-B checkpoint; default $DEPTH_ANYTHING_CHECKPOINT")
    p.set_defaults(run=cmd_depth_check)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.run(args)
