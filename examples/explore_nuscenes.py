"""List the nuScenes scenes and draw the lidar and radar of each scene's first keyframe on its six cameras.

    uv run examples/explore_nuscenes.py
    uv run examples/explore_nuscenes.py --scenes scene-1077 scene-0553 --radar-sweeps 5
"""

import argparse
import os
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from confhead.items import load_points
from confhead.nuscenes import DATAROOT_ENV, NuScenes
from confhead.plot import draw_points

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataroot", type=Path,
                        default=os.environ.get(DATAROOT_ENV, REPO / "datasets/nuscenes/v1.0-mini"))
    parser.add_argument("--scenes", nargs="*", help="scene names; default every scene")
    parser.add_argument("--radar-sweeps", type=int, default=1)
    parser.add_argument("--out", type=Path, default=REPO / "outputs/overlays")
    args = parser.parse_args()

    nusc = NuScenes(args.dataroot)
    for s in nusc.scenes:
        print(f"{s.name}  {s.condition:5}  {s.weather:5}  {len(s.samples):3d} keyframes  {s.description}")
    for s in nusc.skipped_scenes:
        print(f"{s.name}  skipped: keyframe files are missing")

    args.out.mkdir(parents=True, exist_ok=True)
    for scene in nusc.scenes:
        if args.scenes and scene.name not in args.scenes:
            continue
        fig = Figure(figsize=(18, 7), layout="constrained")
        fig.suptitle(f"{scene.name} ({scene.condition}, {scene.weather}), first keyframe")
        for ax, item in zip(fig.subplots(2, 3).flat, nusc.items([scene.name], max_samples_per_scene=1,
                                                                radar_sweeps=args.radar_sweeps)):
            lidar = load_points(item.lidar, item.view)
            radar = load_points(item.radar, item.view)
            image = np.asarray(Image.open(item.view.image_path).convert("RGB"))
            dots = draw_points(ax, image, lidar, radar)
            ax.set_title(f"{item.channel}: {len(lidar)} lidar, {len(radar)} radar points", fontsize=10)
        fig.colorbar(dots, ax=fig.axes, shrink=0.6, label="lidar depth (m)")
        path = args.out / f"{scene.name}.jpg"
        fig.savefig(path, dpi=80)
        print(path)


if __name__ == "__main__":
    main()
