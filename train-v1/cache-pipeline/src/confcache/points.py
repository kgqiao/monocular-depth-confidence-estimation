"""Read point files and project them into the camera image.

A projected point is one row of (u, v, z, dt, sweep): image x and y in pixels
of the original image, depth along the optical axis in meters, the time of its
sweep relative to the image in seconds, and the sweep number (0 for the
keyframe sweep of its sensor). The cache adds a sixth column, the model
disparity at (u, v). Every point inside the image and the depth range
is kept; rasterizing to a grid is the reader's choice.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from confhead.nuscenes import apply, parse_radar_pcd

from .sources.base import Item, PointFile

POINT_COLUMNS = ("u", "v", "z", "dt", "sweep", "disparity")


def read_nuscenes_lidar(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :3].astype(np.float64)


def read_nuscenes_radar(path: Path) -> np.ndarray:
    """The devkit's default filter: invalid_state 0, ambig_state 3, dyn_prop 0 to 6."""
    rec = parse_radar_pcd(path.read_bytes(), path.name)
    rec = rec[(rec["invalid_state"] == 0) & (rec["ambig_state"] == 3) & (rec["dyn_prop"] < 7)]
    return np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float64)


READERS: dict[str, Callable[[Path], np.ndarray]] = {
    "nuscenes-lidar": read_nuscenes_lidar,
    "nuscenes-radar": read_nuscenes_radar,
}


def project(points_cam: np.ndarray, K: np.ndarray, width: int, height: int,
            min_depth: float, max_depth: float) -> tuple[np.ndarray, np.ndarray]:
    """(N, 3) camera-frame points -> (M, 3) rows of (u, v, z) and the (N,) mask of kept points.

    A point is kept when its depth is in (min_depth, max_depth) and its nearest
    pixel center is inside the image.
    """
    z = points_cam[:, 2]
    keep = (z > min_depth) & (z < max_depth)
    safe_z = np.where(keep, z, 1.0)
    # float32, as the cache stores them, so that the stored (u, v) round to a pixel inside the image.
    u = (K[0, 0] * points_cam[:, 0] / safe_z + K[0, 2]).astype(np.float32)
    v = (K[1, 1] * points_cam[:, 1] / safe_z + K[1, 2]).astype(np.float32)
    col, row = np.round(u), np.round(v)
    keep &= (col >= 0) & (col < width) & (row >= 0) & (row < height)
    return np.stack([u[keep], v[keep], z[keep]], axis=1), keep


def load_points(files: tuple[PointFile, ...], item: Item, min_depth: float, max_depth: float) -> np.ndarray:
    """Every sweep of `files` projected into the item's image: (N, 5) float32 rows of (u, v, z, dt, sweep)."""
    parts = []
    for pf in files:
        pts = apply(pf.sensor_to_cam, READERS[pf.format](pf.path))
        uvz, _ = project(pts, item.view.K, item.view.width, item.view.height, min_depth, max_depth)
        parts.append(np.concatenate([uvz, np.full((len(uvz), 1), pf.dt), np.full((len(uvz), 1), pf.sweep)], axis=1))
    if not parts:
        return np.empty((0, 5), dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)
