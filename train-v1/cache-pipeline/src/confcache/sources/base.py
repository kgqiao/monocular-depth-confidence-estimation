"""What the cache build needs from a dataset.

A source turns a dataset into a list of `Item`s, one per camera image of a
keyframe. An item carries everything the build reads: the image, its
calibration and pose, the neighbor images in time, and the point files with
their transforms into the camera. So a worker process loads an item with no
access to the dataset tables.

Camera frame: x right, y down, z forward. Every 4x4 matrix maps a column
vector from the frame named first to the frame named second.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


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


@dataclass(frozen=True)
class PointFile:
    """One lidar or radar sweep, and how to bring its points into the camera."""

    path: Path
    format: str  # a key of confcache.points.READERS
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
    lidar: tuple[PointFile, ...]  # keyframe sweep first
    radar: tuple[PointFile, ...]


class Source(Protocol):
    name: str
    skipped_scenes: list[str]

    def items(self) -> list[Item]:
        """Every item to cache, in scene order then keyframe order then camera order."""
        ...

    def scene_names(self) -> list[str]:
        ...
