"""SeeingThroughFog (STF / DENSE) as a cache source. Not built yet.

To build it, return `Item`s as `NuScenesSource` does. Open points:
- STF ships discrete samples. If it has no image sequences, set
  `temporal.enabled = false`, and the temporal channels hold zeros.
- Its lidar and radar formats need readers in `confcache.points.READERS`.
"""

from __future__ import annotations

from ..config import PointsConfig, SourceConfig, TemporalConfig
from .base import Item


class STFSource:
    name = "stf"

    def __init__(self, source: SourceConfig, points: PointsConfig, temporal: TemporalConfig):
        raise SystemExit("source.kind = \"stf\" is not built yet; see confcache/sources/stf.py.")

    def scene_names(self) -> list[str]:
        raise NotImplementedError

    def items(self) -> list[Item]:
        raise NotImplementedError
