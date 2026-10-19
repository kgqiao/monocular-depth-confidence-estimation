"""The training view of a cache: one confidence-head input and label per entry.

    x       (C + 4, h, w)  float32  features, depth, temporal spread, radar residual, radar validity
    target  (1, h, w)      float32  1 where the model's AbsRel against lidar exceeds tau
    mask    (1, h, w)      float32  1 where a lidar label exists
    abs_rel (1, h, w)      float32  the error itself, for evaluation
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset

from .derive import LabelConfig, RadarConfig, lidar_labels, normalized_disparity, radar_channels
from .store import Cache

ARRAYS = ["features", "disparity_grid", "temporal_relstd", "lidar", "radar"]


@dataclass
class HeadInputs:
    features: bool = True
    depth: bool = True
    temporal: bool = True
    radar: bool = True
    labels: LabelConfig = field(default_factory=LabelConfig)
    radar_cfg: RadarConfig = field(default_factory=RadarConfig)


class ConfidenceDataset(Dataset):
    """`ConfidenceDataset(Cache(root), tag="train")`. Turn channel groups off for ablations."""

    def __init__(self, cache: Cache, inputs: HeadInputs = HeadInputs(), **filters):
        self.cache = cache
        self.inputs = inputs
        self.rows = cache.rows(**filters)
        if not self.rows:
            raise ValueError(f"no entries in {cache.root} match {filters}")

    def __len__(self) -> int:
        return len(self.rows)

    def channel_names(self, n_features: int) -> list[str]:
        names = [f"feature_{i}" for i in range(n_features)] if self.inputs.features else []
        names += ["depth"] if self.inputs.depth else []
        names += ["temporal_relstd"] if self.inputs.temporal else []
        names += ["radar_residual", "radar_valid"] if self.inputs.radar else []
        return names

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
