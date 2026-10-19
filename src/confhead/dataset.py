"""The training view of a cache: one confidence-head input and label per entry.

    x       (C + 4, h, w)  float32  features, depth, temporal spread, radar residual, radar validity
    target  (1, h, w)      float32  1 where the model's AbsRel against lidar exceeds tau
    mask    (1, h, w)      float32  1 where a lidar label exists
    abs_rel (1, h, w)      float32  the error itself, for evaluation

The radar channels and the labels are computed here, at load time, so that
their settings change without a new cache build.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache.store import Cache
from .geometry import rasterize
from .labels import LabelConfig, fit_of, lidar_labels

ARRAYS = ["features", "disparity_grid", "temporal_relstd", "lidar", "radar"]


@dataclass(frozen=True)
class RadarConfig:
    align: str = "radar"  # the radar fit, because a deployed car has radar but no lidar
    radius: int = 1  # grid cells around each radar point that count as covered
    max_sweeps: int = 0  # 0: every cached sweep
    min_depth: float = 1.0
    max_depth: float = 80.0


def radar_channels(meta: dict, arrays: dict[str, np.ndarray], cfg: RadarConfig = RadarConfig()):
    """The radar residual |pred - z| / z and the radar validity mask, both (h, w) float32."""
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
    values, mask = rasterize(pts, np.abs(pred - z) / z, image, grid, radius=cfg.radius)
    return values, mask.astype(np.float32)


def normalized_disparity(disparity: np.ndarray) -> np.ndarray:
    """Disparity over its image mean, so every image has the same scale."""
    mean = float(disparity.mean())
    return (disparity / mean if mean > 0 else disparity).astype(np.float32)


@dataclass(frozen=True)
class HeadInputs:
    """Which channel groups go into x. Turn groups off for the ablations."""

    features: bool = True
    depth: bool = True
    temporal: bool = True
    radar: bool = True
    labels: LabelConfig = field(default_factory=LabelConfig)
    radar_cfg: RadarConfig = field(default_factory=RadarConfig)

    def channel_names(self, n_features: int) -> list[str]:
        names = [f"feature_{i}" for i in range(n_features)] if self.features else []
        names += ["depth"] if self.depth else []
        names += ["temporal_relstd"] if self.temporal else []
        names += ["radar_residual", "radar_valid"] if self.radar else []
        return names


class ConfidenceDataset(Dataset):
    """`ConfidenceDataset(Cache(root), tag="train")`. Filters are those of `Cache.rows`."""

    def __init__(self, cache: Cache, inputs: HeadInputs = HeadInputs(), **filters):
        self.cache = cache
        self.inputs = inputs
        self.rows = cache.rows(**filters)
        if not self.rows:
            raise ValueError(f"no entries in {cache.root} match {filters}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | str]:
        entry = self.cache.entry(self.rows[i])
        a = entry.load(ARRAYS)
        meta = entry.meta
        channels = []
        if self.inputs.features:
            channels.append(a["features"].astype(np.float32))
        if self.inputs.depth:
            channels.append(normalized_disparity(a["disparity_grid"])[None])
        if self.inputs.temporal:
            channels.append(a["temporal_relstd"].astype(np.float32)[None])
        if self.inputs.radar:
            residual, valid = radar_channels(meta, a, self.inputs.radar_cfg)
            channels.extend([residual[None], valid[None]])
        abs_rel, target, mask = lidar_labels(meta, a, self.inputs.labels)
        return {
            "key": entry.key,
            "x": torch.from_numpy(np.concatenate(channels)),
            "target": torch.from_numpy(target[None]),
            "mask": torch.from_numpy(mask[None].astype(np.float32)),
            "abs_rel": torch.from_numpy(abs_rel[None]),
        }
