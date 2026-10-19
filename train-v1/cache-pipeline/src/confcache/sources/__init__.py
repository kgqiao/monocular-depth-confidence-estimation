from ..config import Config
from .base import Item, PointFile, Source, View


def make_source(cfg: Config) -> Source:
    if cfg.source.kind == "nuscenes":
        from .nuscenes import NuScenesSource
        return NuScenesSource(cfg.source, cfg.points, cfg.temporal)
    if cfg.source.kind == "stf":
        from .stf import STFSource
        return STFSource(cfg.source, cfg.points, cfg.temporal)
    raise ValueError(f"unknown source.kind {cfg.source.kind!r}")


__all__ = ["Item", "PointFile", "Source", "View", "make_source"]
