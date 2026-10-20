"""The cache of frozen Depth Anything V2 outputs that the confidence head trains on.

    cfg = load_config("configs/nuscenes-mini.toml")
    build(cfg)
    cache = Cache(cfg.output.root)
"""

from .backup import BackupConfig, download, load_backup_config, upload
from .build import BuildReport, build, select_items
from .config import CacheConfig, load_config, to_toml
from .report import format_info, format_plan, format_verify, info, plan, verify
from .store import Cache, CacheMismatch, Entry, build_index

__all__ = [
    "BackupConfig", "BuildReport", "Cache", "CacheConfig", "CacheMismatch", "Entry", "build", "build_index",
    "download", "format_info", "format_plan", "format_verify", "info", "load_backup_config", "load_config", "plan",
    "select_items", "to_toml", "upload", "verify",
]
