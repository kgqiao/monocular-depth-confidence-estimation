"""Frames, projection, pixel grids, sampling, and the scale-and-shift fit of relative disparity.

Frames: every 4x4 matrix maps a column vector from the frame named first to the
frame named second, e.g. `sensor_to_cam`. The camera frame is x right, y down,
z forward. Quaternions are [w, x, y, z].

Pixels: integer coordinates are pixel centers. An image of width W covers x
from -0.5 to W - 0.5. A map of any size covers the same field of view, so a
point at original x = u sits at (u + 0.5) * w / W - 0.5 in a map of width w.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pose_to_mat(record: dict) -> np.ndarray:
    """A record with `rotation` (quaternion) and `translation` -> 4x4 homogeneous."""
    m = np.eye(4)
    m[:3, :3] = quat_to_mat(record["rotation"])
    m[:3, 3] = record["translation"]
    return m


def apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 to (N, 3) points."""
    return pts @ m[:3, :3].T + m[:3, 3]


def project(points_cam: np.ndarray, K: np.ndarray, width: int, height: int,
            min_depth: float, max_depth: float) -> tuple[np.ndarray, np.ndarray]:
    """(N, 3) camera-frame points -> (M, 3) float32 rows of (u, v, z) and the (N,) mask of kept points.

    A point is kept when its depth is in (min_depth, max_depth) and its nearest
    pixel center is inside the image.
    """
    z = points_cam[:, 2]
    keep = (z > min_depth) & (z < max_depth)
    safe_z = np.where(keep, z, 1.0)
    # float32 before the check, so that the stored (u, v) round to a pixel inside the image.
    u = (K[0, 0] * points_cam[:, 0] / safe_z + K[0, 2]).astype(np.float32)
    v = (K[1, 1] * points_cam[:, 1] / safe_z + K[1, 2]).astype(np.float32)
    col, row = np.round(u), np.round(v)
    keep &= (col >= 0) & (col < width) & (row >= 0) & (row < height)
    return np.stack([u[keep], v[keep], z[keep].astype(np.float32)], axis=1), keep


def rescale_coords(u: np.ndarray, v: np.ndarray, src: tuple[int, int], dst: tuple[int, int]):
    """Pixel coordinates in an image of size src (w, h) -> in a map of size dst (w, h)."""
    return (u + 0.5) * dst[0] / src[0] - 0.5, (v + 0.5) * dst[1] / src[1] - 0.5


def grid_centers(grid: tuple[int, int], image: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Centers of the cells of a grid (w, h), in original image pixels: two (h, w) arrays."""
    gx, gy = np.meshgrid(np.arange(grid[0]), np.arange(grid[1]))
    return rescale_coords(gx.astype(np.float64), gy.astype(np.float64), grid, image)


def sample_bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Values of a (h, w) map at map coordinates (x, y), clamped at the edges."""
    h, w = img.shape
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    x0 = np.minimum(np.floor(x).astype(np.int64), w - 2) if w > 1 else np.zeros_like(x, dtype=np.int64)
    y0 = np.minimum(np.floor(y).astype(np.int64), h - 2) if h > 1 else np.zeros_like(y, dtype=np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    img = img.astype(np.float64)
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bottom = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def rasterize(points: np.ndarray, values: np.ndarray, image_size: tuple[int, int],
              grid_size: tuple[int, int], radius: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Scatter one value per point onto a grid; the nearest point (smallest z) wins a cell.

    points: (N, >=3) rows starting with u, v, z in original image pixels.
    radius: also fill the cells within this Chebyshev distance of the point's cell.
    Returns the (h, w) float32 values, 0 where empty, and the (h, w) bool mask.
    """
    gw, gh = grid_size
    out = np.zeros((gh, gw), dtype=np.float32)
    mask = np.zeros((gh, gw), dtype=bool)
    if len(points) == 0:
        return out, mask
    x, y = rescale_coords(points[:, 0].astype(np.float64), points[:, 1].astype(np.float64), image_size, grid_size)
    col = np.clip(np.round(x).astype(np.int64), 0, gw - 1)
    row = np.clip(np.round(y).astype(np.int64), 0, gh - 1)
    z = points[:, 2].astype(np.float64)
    offsets = [(dr, dc) for dr in range(-radius, radius + 1) for dc in range(-radius, radius + 1)]
    nearest = np.full(gh * gw, np.inf)
    for dr, dc in offsets:
        r, c = row + dr, col + dc
        ok = (r >= 0) & (r < gh) & (c >= 0) & (c < gw)
        np.minimum.at(nearest, r[ok] * gw + c[ok], z[ok])
    # A second pass picks, per cell, a point at the winning depth.
    winner = np.full(gh * gw, -1, dtype=np.int64)
    for dr, dc in offsets:
        r, c = row + dr, col + dc
        ok = (r >= 0) & (r < gh) & (c >= 0) & (c < gw)
        idx = np.nonzero(ok)[0]
        cells = r[ok] * gw + c[ok]
        hit = z[idx] == nearest[cells]
        winner[cells[hit]] = idx[hit]
    filled = winner >= 0
    out.reshape(-1)[filled] = np.asarray(values, dtype=np.float32)[winner[filled]]
    mask.reshape(-1)[filled] = True
    return out, mask


@dataclass(frozen=True)
class Fit:
    """inverse depth = scale * disparity + shift, fitted by least squares."""

    scale: float
    shift: float
    n: int
    rmse: float  # in inverse meters

    def inverse_depth(self, disparity, min_depth: float, max_depth: float):
        return np.clip(self.scale * disparity + self.shift, 1.0 / max_depth, 1.0 / min_depth)

    def depth(self, disparity, min_depth: float, max_depth: float):
        return 1.0 / self.inverse_depth(disparity, min_depth, max_depth)

    def to_dict(self) -> dict:
        return {"scale": self.scale, "shift": self.shift, "n": self.n, "rmse": self.rmse}

    @staticmethod
    def from_dict(d: dict | None) -> Fit | None:
        return None if d is None else Fit(d["scale"], d["shift"], d["n"], d["rmse"])


def fit_scale_shift(x: np.ndarray, y: np.ndarray, min_points: int) -> Fit | None:
    """Least squares y = scale * x + shift. None with fewer than min_points
    points, a degenerate x, or a scale that is not positive."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < max(min_points, 2):
        return None
    xm, ym = x.mean(), y.mean()
    var = ((x - xm) ** 2).sum()
    if var <= 1e-12 * max(1.0, (x ** 2).sum()):
        return None
    scale = ((x - xm) * (y - ym)).sum() / var
    if not scale > 0:
        return None
    shift = ym - scale * xm
    rmse = float(np.sqrt(np.mean((scale * x + shift - y) ** 2)))
    return Fit(float(scale), float(shift), len(x), rmse)
