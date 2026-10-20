"""Look at a cache before and after the build: the plan and the checks."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..depth import input_size, tap_channels
from ..items import load_points
from ..labels import LabelConfig, lidar_labels
from .build import select_items
from .config import CacheConfig
from .store import Cache, done_keys

POINT_ROW_BYTES = 6 * 4


def _mb(n: float) -> str:
    return f"{n / 1e6:,.1f} MB" if n < 1e9 else f"{n / 1e9:,.2f} GB"


def _names(names: list[str], most: int = 12) -> str:
    if len(names) <= most:
        return ", ".join(names)
    return f"{', '.join(names[:3])}, ..., {names[-1]}"


# ----- plan -----------------------------------------------------------------


def plan(cfg: CacheConfig, shard: tuple[int, int] = (0, 1)) -> dict[str, Any]:
    """What a build would do, and how much disk it would take, without the model."""
    items, skipped = select_items(cfg, shard)
    if not items:
        raise ValueError("No items match source.scenes and source.cameras.")
    root = Path(cfg.output.root)
    fingerprint = cfg.fingerprint()
    done = len(done_keys(root, ((it.scene, it.key) for it in items), fingerprint))

    first = items[0].view
    gw, gh = cfg.grid_size(first.width, first.height)
    mw, mh = input_size(first.width, first.height, cfg.model.short_side)
    channels = tap_channels(cfg.model.encoder, cfg.model.tap)
    dense = {
        "features": channels * gh * gw * 2,
        "disparity_grid": 4 * gh * gw,
        "temporal_relstd": 2 * gh * gw,
        "temporal_count": gh * gw,
    }

    sample = items[:: max(1, len(items) // 12)][:12]
    lidar_pts = np.mean([len(load_points(it.lidar, it.view, cfg.points.min_depth, cfg.points.max_depth)) for it in sample])
    radar_pts = np.mean([len(load_points(it.radar, it.view, cfg.points.min_depth, cfg.points.max_depth)) for it in sample])
    sparse = (lidar_pts + radar_pts) * POINT_ROW_BYTES
    per_entry = sum(dense.values()) + sparse

    want = len(cfg.temporal.offsets) if cfg.temporal.enabled else 0
    neighbor_counts = Counter(len(it.neighbors) for it in items)
    return {
        "root": str(root),
        "fingerprint": fingerprint,
        "scenes": sorted({it.scene for it in items}),
        "skipped_scenes": skipped,
        "items": len(items),
        "done": done,
        "images_per_item": 1 + want,
        "model_size": (mw, mh),
        "grid_size": (gw, gh),
        "channels": channels,
        "items_by_neighbors": dict(sorted(neighbor_counts.items())),
        "lidar_points": float(lidar_pts),
        "radar_points": float(radar_pts),
        "bytes_dense": dense,
        "bytes_per_entry": float(per_entry),
        "bytes_total": float(per_entry * len(items)),
        "bytes_todo": float(per_entry * (len(items) - done)),
    }


def format_plan(p: dict[str, Any], cfg: CacheConfig) -> str:
    lines = [
        f"cache        {p['root']}  (fingerprint {p['fingerprint']})",
        f"scenes       {len(p['scenes'])}: {_names(p['scenes'])}",
    ]
    if cfg.splits:
        lines.append("splits       " + ", ".join(f"{tag} {len(names)} scenes" for tag, names in cfg.splits.items()))
    if p["skipped_scenes"]:
        lines.append(f"skipped      {', '.join(p['skipped_scenes'])} (keyframe files missing)")
    lines += [
        f"items        {p['items']} camera images, {p['done']} already built",
        f"model        {cfg.model.encoder}, input {p['model_size'][0]} x {p['model_size'][1]}, "
        f"{p['images_per_item']} passes per item",
        f"grid         {p['grid_size'][0]} x {p['grid_size'][1]}, tap {cfg.model.tap} "
        f"({p['channels']} channels)",
        "neighbors    items by neighbor count: "
        + ", ".join(f"{k}: {v}" for k, v in p["items_by_neighbors"].items()),
        f"points       about {p['lidar_points']:.0f} lidar and {p['radar_points']:.0f} radar per image "
        f"(1 and {cfg.points.radar_sweeps} sweeps)",
        "size         " + ", ".join(f"{k} {_mb(v)}" for k, v in p["bytes_dense"].items()),
        f"             {_mb(p['bytes_per_entry'])} per entry, {_mb(p['bytes_total'])} in all, "
        f"{_mb(p['bytes_todo'])} still to write",
    ]
    return "\n".join(lines)


# ----- verify -----------------------------------------------------------------


def verify(root: str | Path, labels: LabelConfig = LabelConfig()) -> dict[str, Any]:
    """Open every indexed entry, check it against its meta, and collect statistics."""
    cache = Cache(root)
    settings = cache.settings
    channels = tap_channels(settings["model"]["encoder"], settings["model"]["tap"])
    min_d, max_d = settings["points"]["min_depth"], settings["points"]["max_depth"]
    problems: list[str] = []
    stats: dict[str, list[float]] = {k: [] for k in (
        "lidar_points", "radar_points", "label_cells", "positive", "seen2", "relstd_median",
        "fit_lidar_rmse", "bytes")}
    fits = Counter()
    status = Counter()
    by_condition: dict[str, list[float]] = {}

    for row in cache.rows():
        entry = cache.entry(row)
        meta = entry.meta
        key = entry.key
        try:
            a = entry.load()
        except Exception as e:
            problems.append(f"{key}: cannot load: {e}")
            continue
        gw, gh = meta["grid_size"]
        W, H = meta["image"]["width"], meta["image"]["height"]

        def check(ok: bool, what: str) -> None:
            if not ok:
                problems.append(f"{key}: {what}")

        check(meta["fingerprint"] == cache.header["fingerprint"], "fingerprint differs from cache.json")
        check(a["features"].shape == (channels, gh, gw), f"features shape {a['features'].shape}")
        check(np.isfinite(a["features"]).all(), "features hold NaN or inf")
        check(a["disparity_grid"].shape == (gh, gw), "disparity_grid shape")
        check(np.isfinite(a["disparity_grid"]).all() and (a["disparity_grid"] >= 0).all(),
              "disparity_grid is not finite and non-negative")
        rel = a["temporal_relstd"].astype(np.float32)
        check(rel.shape == (gh, gw) and np.isfinite(rel).all() and (rel >= 0).all(),
              "temporal_relstd is not finite and non-negative")
        for name in ("lidar", "radar"):
            p = a[name]
            check(p.ndim == 2 and p.shape[1] == 6, f"{name} shape {p.shape}")
            if len(p):
                check(bool(((p[:, 2] > min_d) & (p[:, 2] < max_d)).all()), f"{name} depth out of range")
                check(bool(((np.round(p[:, 0]) >= 0) & (np.round(p[:, 0]) < W)
                            & (np.round(p[:, 1]) >= 0) & (np.round(p[:, 1]) < H)).all()),
                      f"{name} point outside the image")
            check(len(p) == meta[name]["points"], f"{name} count differs from meta")

        abs_rel, target, mask = lidar_labels(meta, a, labels)
        n_label = int(mask.sum())
        stats["lidar_points"].append(len(a["lidar"]))
        stats["radar_points"].append(len(a["radar"]))
        stats["label_cells"].append(n_label / mask.size)
        if n_label:
            stats["positive"].append(float(target[mask].mean()))
            by_condition.setdefault(meta["condition"], []).append(float(np.median(abs_rel[mask])))
        stats["seen2"].append(float((a["temporal_count"] >= 2).mean()))
        stats["relstd_median"].append(float(np.median(rel)))
        stats["bytes"].append(entry.path.stat().st_size + entry.path.with_suffix(".json").stat().st_size)
        if meta["fit_lidar"]:
            stats["fit_lidar_rmse"].append(meta["fit_lidar"]["rmse"])
        fits["lidar"] += meta["fit_lidar"] is not None
        fits["radar"] += meta["fit_radar"] is not None
        status[meta["temporal"].get("status", "off")] += 1

    n = len(cache)
    return {
        "root": str(cache.root),
        "entries": n,
        "stale_entries": cache.index.get("stale_entries", 0),
        "problems": problems,
        "tau": labels.tau,
        "means": {k: float(np.mean(v)) if v else float("nan") for k, v in stats.items()},
        "bytes_total": float(np.sum(stats["bytes"])),
        "fit_rate": {k: v / n for k, v in fits.items()} if n else {},
        "temporal_status": dict(status),
        "median_abs_rel_by_condition": {k: float(np.median(v)) for k, v in sorted(by_condition.items())},
        "tags": dict(Counter(t for r in cache.rows() for t in r["tags"])),
    }


def format_verify(v: dict[str, Any]) -> str:
    m = v["means"]
    lines = [
        f"cache        {v['root']}",
        f"entries      {v['entries']} ({v['stale_entries']} stale files left out), {_mb(v['bytes_total'])} on disk",
        "tags         " + (", ".join(f"{k}: {c}" for k, c in v["tags"].items()) or "none"),
        f"points       {m['lidar_points']:.0f} lidar and {m['radar_points']:.0f} radar per entry",
        f"fits         lidar {v['fit_rate'].get('lidar', 0):.1%}, radar {v['fit_rate'].get('radar', 0):.1%} "
        f"of entries; mean lidar fit RMSE {m['fit_lidar_rmse']:.4f} 1/m",
        f"labels       {m['label_cells']:.2%} of grid cells hold a lidar label; "
        f"{m['positive']:.1%} of them have AbsRel > {v['tau']}",
        "AbsRel       median over entries of the median per entry: "
        + ", ".join(f"{k} {x:.3f}" for k, x in v["median_abs_rel_by_condition"].items()),
        "temporal     " + ", ".join(f"{k}: {c}" for k, c in v["temporal_status"].items())
        + f"; {m['seen2']:.1%} of cells seen by 2+ frames, median relstd {m['relstd_median']:.4f}",
    ]
    if v["problems"]:
        lines.append(f"problems     {len(v['problems'])}:")
        lines += [f"  {p}" for p in v["problems"][:50]]
    else:
        lines.append("problems     none")
    return "\n".join(lines)
