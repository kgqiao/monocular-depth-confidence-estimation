"""Channels and labels that are cheap to compute from an entry, at load time.

They stay out of the cache so that the threshold, the alignment, and the
radar radius change without a new build.

Alignment: the model gives disparity up to a scale and a shift. A metric
depth needs a fit: "lidar" fits to the keyframe lidar sweep (the standard
evaluation protocol, and the default for labels), "radar" fits to the radar
points (available at deployment, but often fails on few points).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Fit, rescale_coords


def fit_of(meta: dict, align: str) -> Fit | None:
    return Fit.from_dict(meta[f"fit_{align}"])


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
    nearest = np.full(gh * gw, np.inf)
    winner = np.full(gh * gw, -1, dtype=np.int64)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            r, c = row + dr, col + dc
            ok = (r >= 0) & (r < gh) & (c >= 0) & (c < gw)
            cells = r[ok] * gw + c[ok]
            np.minimum.at(nearest, cells, z[ok])
    # A second pass picks, per cell, a point at the winning depth.
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
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
class LabelConfig:
    align: str = "lidar"
    tau: float = 0.10  # AbsRel above which a depth counts as wrong
    keyframe_sweep_only: bool = True
    min_depth: float = 1.0
    max_depth: float = 80.0


@dataclass(frozen=True)
class RadarConfig:
    align: str = "radar"
    radius: int = 1  # grid cells around each radar point that count as covered
    max_sweeps: int = 0  # 0: every cached sweep
    relative: bool = True  # residual as |pred - z| / z, else in meters
    min_depth: float = 1.0
    max_depth: float = 80.0


def point_abs_rel(points: np.ndarray, fit: Fit, min_depth: float, max_depth: float) -> np.ndarray:
    """|predicted depth - z| / z at each cached point."""
    pred = fit.depth(points[:, 5].astype(np.float64), min_depth, max_depth)
    z = points[:, 2].astype(np.float64)
    return np.abs(pred - z) / z


def lidar_labels(meta: dict, arrays: dict[str, np.ndarray], cfg: LabelConfig = LabelConfig()):
    """Per-cell AbsRel of the model against lidar, the label P(error > tau), and the label mask.

    Every output is (h, w): abs_rel float32, target float32 in {0, 1}, mask bool.
    A missing fit gives an all-false mask, so the image adds nothing to the loss.
    """
    grid = tuple(meta["grid_size"])
    image = (meta["image"]["width"], meta["image"]["height"])
    pts = arrays["lidar"]
    if cfg.keyframe_sweep_only:
        pts = pts[pts[:, 4] == 0]
    fit = fit_of(meta, cfg.align)
    if fit is None or len(pts) == 0:
        empty = np.zeros(grid[::-1], dtype=np.float32)
        return empty, empty.copy(), empty.astype(bool)
    abs_rel, mask = rasterize(pts, point_abs_rel(pts, fit, cfg.min_depth, cfg.max_depth), image, grid)
    target = (abs_rel > cfg.tau).astype(np.float32) * mask
    return abs_rel, target, mask


def radar_channels(meta: dict, arrays: dict[str, np.ndarray], cfg: RadarConfig = RadarConfig()):
    """The radar residual and the radar validity mask, both (h, w) float32."""
    grid = tuple(meta["grid_size"])
    image = (meta["image"]["width"], meta["image"]["height"])
    pts = arrays["radar"]
    if cfg.max_sweeps:
        pts = pts[pts[:, 4] < cfg.max_sweeps]
    fit = fit_of(meta, cfg.align)
    if fit is None or len(pts) == 0:
        empty = np.zeros(grid[::-1], dtype=np.float32)
        return empty, empty.copy()
    pred = fit.depth(pts[:, 5].astype(np.float64), cfg.min_depth, cfg.max_depth)
    z = pts[:, 2].astype(np.float64)
    residual = np.abs(pred - z) / z if cfg.relative else np.abs(pred - z)
    values, mask = rasterize(pts, residual, image, grid, radius=cfg.radius)
    return values, mask.astype(np.float32)


def normalized_disparity(disparity: np.ndarray) -> np.ndarray:
    """Disparity over its image mean, so every image has the same scale."""
    mean = float(disparity.mean())
    return (disparity / mean if mean > 0 else disparity).astype(np.float32)
