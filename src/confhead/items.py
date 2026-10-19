"""One camera image of a keyframe, with everything needed to load it.

An `Item` carries the image, its calibration and pose, the neighbor images in
time, and the lidar and radar sweeps with their transforms into the camera. A
worker process loads an item with no access to the dataset tables.

A projected point is one row of (u, v, z, dt, sweep): image x and y in pixels
of the original image, depth along the optical axis in meters, the time of
its sweep relative to the image in seconds, and the sweep number (0 for the
keyframe sweep of its sensor). The cache adds a sixth column, the model
disparity at (u, v).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import Fit, apply, fit_scale_shift, project

POINT_COLUMNS = ("u", "v", "z", "dt", "sweep", "disparity")


@dataclass(frozen=True)
class View:
    """One camera image with its calibration and pose."""

    token: str
    image_path: Path
    width: int
    height: int
    K: np.ndarray  # (3, 3)
    cam_to_global: np.ndarray  # (4, 4) at the image timestamp
    timestamp: int  # microseconds

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


@dataclass(frozen=True)
class PointFile:
    """One lidar or radar sweep, and how to bring its points into the camera."""

    path: Path
    read: Callable[[Path], np.ndarray]  # path -> (N, 3) points in the sensor frame
    sensor_to_cam: np.ndarray  # (4, 4)
    dt: float  # seconds from the camera image to the sweep, negative for earlier sweeps
    sweep: int  # 0 for the keyframe sweep of its sensor, 1 for the sweep before it, and so on


@dataclass(frozen=True)
class Item:
    key: str  # unique and safe as a file name
    scene: str
    condition: str  # day or night
    weather: str  # clear or rain
    channel: str
    sample_token: str
    index: int  # keyframe position in its scene, from 0
    view: View
    neighbors: tuple[tuple[int, View], ...]  # (offset, view) for every offset that exists
    lidar: tuple[PointFile, ...]
    radar: tuple[PointFile, ...]


def load_points(files: tuple[PointFile, ...], view: View, min_depth: float = 1.0,
                max_depth: float = 80.0) -> np.ndarray:
    """Every sweep of `files` projected into the image of `view`: (N, 5) float32 rows of (u, v, z, dt, sweep)."""
    parts = []
    for pf in files:
        uvz, _ = project(apply(pf.sensor_to_cam, pf.read(pf.path)), view.K, view.width, view.height,
                         min_depth, max_depth)
        parts.append(np.concatenate([uvz, np.full((len(uvz), 2), (pf.dt, pf.sweep), dtype=np.float32)], axis=1))
    if not parts:
        return np.empty((0, 5), dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


def fit_to_points(points: np.ndarray, min_points: int) -> Fit | None:
    """Fit the disparity column of points (u, v, z, dt, sweep, disparity) to 1 / z."""
    if len(points) == 0:
        return None
    return fit_scale_shift(points[:, 5], 1.0 / points[:, 2], min_points)
