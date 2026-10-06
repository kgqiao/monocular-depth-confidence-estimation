"""Draw projected lidar and radar points on a camera image."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .project import Projection

# Viridis at five stops; near is yellow, far is purple.
_STOPS = np.array([
    [253, 231, 37], [94, 201, 98], [33, 145, 140], [59, 82, 139], [68, 1, 84],
], dtype=np.float64)

RADAR_COLOR = (255, 40, 200)
COLOR_NEAR, COLOR_FAR = 3.0, 50.0


def depth_colors(z: np.ndarray) -> np.ndarray:
    """(N,) depth -> (N,3) uint8, on a log scale from COLOR_NEAR to COLOR_FAR."""
    t = (np.log(z) - np.log(COLOR_NEAR)) / (np.log(COLOR_FAR) - np.log(COLOR_NEAR))
    x = np.clip(t, 0, 1) * (len(_STOPS) - 1)
    i = np.minimum(x.astype(int), len(_STOPS) - 2)
    f = (x - i)[:, None]
    return ((1 - f) * _STOPS[i] + f * _STOPS[i + 1]).astype(np.uint8)


def draw_overlay(image: Image.Image, lidar: Projection, radar: Projection, title: str) -> Image.Image:
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)

    order = np.argsort(-lidar.z)
    for (u, v), c in zip(np.stack([lidar.u, lidar.v], 1)[order], depth_colors(lidar.z[order])):
        draw.ellipse([u - 2, v - 2, u + 2, v + 2], fill=tuple(int(x) for x in c))

    for u, v, z in zip(radar.u, radar.v, radar.z):
        draw.ellipse([u - 7, v - 7, u + 7, v + 7], outline=RADAR_COLOR, width=3)
        draw.text((u + 9, v - 7), f"{z:.0f}m", fill=RADAR_COLOR)

    draw.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
    draw.text((6, 5), title, fill=(255, 255, 255))
    return img
