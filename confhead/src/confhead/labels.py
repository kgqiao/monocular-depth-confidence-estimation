"""Training labels: where the model's depth is wrong against lidar.

The model gives disparity up to a scale and a shift, so a metric depth needs a
fit. "lidar" fits to the keyframe lidar sweep (the standard evaluation of
relative depth, and the default); "radar" fits to the radar points, which a
deployed car has.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Fit, rasterize


@dataclass(frozen=True)
class LabelConfig:
    align: str = "lidar"
    tau: float = 0.10  # AbsRel above which a depth counts as wrong
    min_depth: float = 1.0
    max_depth: float = 80.0


def fit_of(meta: dict, align: str) -> Fit | None:
    """The scale-and-shift fit in an entry's meta, "lidar" or "radar"; None when it failed."""
    return Fit.from_dict(meta[f"fit_{align}"])


def point_abs_rel(points: np.ndarray, fit: Fit, min_depth: float, max_depth: float) -> np.ndarray:
    """|predicted depth - z| / z at each point (u, v, z, dt, sweep, disparity)."""
    pred = fit.depth(points[:, 5].astype(np.float64), min_depth, max_depth)
    z = points[:, 2].astype(np.float64)
    return np.abs(pred - z) / z


def lidar_labels(meta: dict, arrays: dict[str, np.ndarray], cfg: LabelConfig = LabelConfig()):
    """Per-cell AbsRel against lidar, the label 1[AbsRel > tau], and the label mask.

    Every output is (h, w): abs_rel float32, target float32 in {0, 1}, mask bool.
    A missing fit gives an all-false mask, so the image adds nothing to the loss.
    """
    grid = tuple(meta["grid_size"])
    image = (meta["image"]["width"], meta["image"]["height"])
    pts = arrays["lidar"]
    fit = fit_of(meta, cfg.align)
    if fit is None or len(pts) == 0:
        empty = np.zeros(grid[::-1], dtype=np.float32)
        return empty, empty.copy(), empty.astype(bool)
    abs_rel, mask = rasterize(pts, point_abs_rel(pts, fit, cfg.min_depth, cfg.max_depth), image, grid)
    target = (abs_rel > cfg.tau).astype(np.float32) * mask
    return abs_rel, target, mask
