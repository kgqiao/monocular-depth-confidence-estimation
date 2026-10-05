"""Command line entry point: `confhead <command>`."""

from __future__ import annotations

import argparse

from .nuscenes import NuScenes, resolve_dataroot


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


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataroot", help="nuScenes folder; default $NUSCENES_DATAROOT")
    common.add_argument("--version", default="v1.0-mini")

    parser = argparse.ArgumentParser(prog="confhead", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scenes", parents=[common], help="list the scenes on disk")
    p.set_defaults(run=cmd_scenes)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.run(args)
