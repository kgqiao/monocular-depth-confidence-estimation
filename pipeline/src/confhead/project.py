"""Project lidar and radar points into a camera image.

The result is a sparse depth map at full image resolution: z along the
camera's optical axis in meters, 0 where no point lands, and a mask of the
pixels that hold a depth. Where several points land on one pixel, the
nearest wins. Pixel (row, col) covers image coordinates within half a pixel
of its center at (col, row).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nuscenes import Camera, Frame, NuScenes, apply

MIN_DEPTH = 1.0
MAX_DEPTH = 80.0


@dataclass
class Projection:
    depth: np.ndarray  # (H, W) float32, meters, 0 where mask is False
    mask: np.ndarray  # (H, W) bool
    u: np.ndarray  # (N,) image x of every point inside the image
    v: np.ndarray  # (N,) image y
    z: np.ndarray  # (N,) depth

    @property
    def n_points(self) -> int:
        return len(self.z)

    @property
    def n_pixels(self) -> int:
        return int(self.mask.sum())

    @property
    def fraction(self) -> float:
        return self.n_pixels / self.mask.size


def project_points(pts_ref: np.ndarray, K: np.ndarray, ref_to_cam: np.ndarray,
                   width: int, height: int) -> Projection:
    """(N,3) points in the reference frame -> depth map and mask."""
    cam = apply(ref_to_cam, np.asarray(pts_ref, dtype=np.float64)[:, :3])
    cam = cam[(cam[:, 2] > MIN_DEPTH) & (cam[:, 2] < MAX_DEPTH)]
    z = cam[:, 2]
    u = K[0, 0] * cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    col = np.round(u).astype(np.int64)
    row = np.round(v).astype(np.int64)
    inside = (col >= 0) & (col < width) & (row >= 0) & (row < height)
    u, v, z, col, row = u[inside], v[inside], z[inside], col[inside], row[inside]

    nearest = np.full(height * width, np.inf)
    np.minimum.at(nearest, row * width + col, z)
    mask = np.isfinite(nearest)
    depth = np.where(mask, nearest, 0.0).astype(np.float32)
    return Projection(
        depth=depth.reshape(height, width),
        mask=mask.reshape(height, width),
        u=u.astype(np.float32), v=v.astype(np.float32), z=z.astype(np.float32),
    )


def project_into(cam: Camera, pts_ref: np.ndarray) -> Projection:
    return project_points(pts_ref, cam.K, cam.ref_to_cam, cam.width, cam.height)


def project_lidar(nusc: NuScenes, frame: Frame, cam: Camera) -> Projection:
    """The keyframe lidar sweep."""
    return project_into(cam, nusc.read_lidar(frame)[:, :3])


def project_radar(nusc: NuScenes, frame: Frame, cam: Camera) -> Projection:
    """The keyframe returns of all five radars."""
    return project_into(cam, nusc.read_all_radar(frame))
