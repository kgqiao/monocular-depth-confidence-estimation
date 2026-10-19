"""Pixel grids, sampling, and the scale-and-shift fit of the relative disparity.

Pixel convention: integer coordinates are pixel centers, as in the
projection. An image of width W covers x from -0.5 to W - 0.5. A map of any
size covers the same field of view, so a point at original x = u sits at
(u + 0.5) * w / W - 0.5 in a map of width w.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
    return Fit(float(scale), float(shift), int(len(x)), rmse)


def fit_to_points(points: np.ndarray, min_points: int) -> Fit | None:
    """Fit the disparity column of cached points (u, v, z, dt, sweep, disparity) to 1 / z."""
    if len(points) == 0:
        return None
    return fit_scale_shift(points[:, 5], 1.0 / points[:, 2], min_points)
