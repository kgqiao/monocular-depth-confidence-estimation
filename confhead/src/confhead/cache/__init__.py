"""The cache of frozen Depth Anything V2 outputs that the confidence head trains on.

    cfg = load_config("configs/nuscenes-mini.toml")
    build(cfg)
    cache = Cache(cfg.output.root)
"""

from .build import BuildReport, build, select_items
from .config import CacheConfig, load_config, to_toml
from .report import format_plan, format_verify, plan, verify
from .store import Cache, CacheMismatch, Entry, build_index

__all__ = [
    "BuildReport", "Cache", "CacheConfig", "CacheMismatch", "Entry", "build", "build_index",
    "format_plan", "format_verify", "load_config", "plan", "select_items", "to_toml", "verify",
]
