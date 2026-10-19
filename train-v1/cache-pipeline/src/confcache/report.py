"""Look at a cache before and after the build: the plan, the checks, and previews."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .build import grid_size, select_items
from .config import Config
from .derive import LabelConfig, RadarConfig, fit_of, lidar_labels, radar_channels
from .model import model_size, tap_channels
from .points import load_points
from .store import Cache, entry_done

POINT_ROW_BYTES = 6 * 4


def _mb(n: float) -> str:
    return f"{n / 1e6:,.1f} MB" if n < 1e9 else f"{n / 1e9:,.2f} GB"


# ----- plan -----------------------------------------------------------------


def plan(cfg: Config, shard: tuple[int, int] = (0, 1)) -> dict[str, Any]:
    """What a build would do, and how much disk it would take, without the model."""
    items, skipped = select_items(cfg, shard)
    if not items:
        raise SystemExit("No items match source.scenes and source.cameras.")
    root = Path(cfg.output.root)
    fingerprint = cfg.fingerprint()
    done = sum(entry_done(root, it.scene, it.key, fingerprint) for it in items)

    first = items[0].view
    gw, gh = grid_size(first.width, first.height, cfg)
    mw, mh = model_size(first.width, first.height, cfg.model.short_side)
    channels = tap_channels(cfg.model.encoder, cfg.model.tap)
    feature_bytes = 2 if cfg.output.feature_dtype == "float16" else 4
    dense = {
        "features": channels * gh * gw * feature_bytes,
        "disparity_grid": 4 * gh * gw,
        "temporal_relstd": 2 * gh * gw,
        "temporal_count": gh * gw,
    }
    if cfg.output.store_full_disparity:
        dense["disparity"] = 2 * mh * mw
    if cfg.temporal.keep_neighbors:
        dense["temporal_aligned"] = 2 * len(cfg.temporal.offsets) * gh * gw

    sample = items[:: max(1, len(items) // 12)][:12]
    lidar_pts = np.mean([len(load_points(it.lidar, it, cfg.points.min_depth, cfg.points.max_depth)) for it in sample])
    radar_pts = np.mean([len(load_points(it.radar, it, cfg.points.min_depth, cfg.points.max_depth)) for it in sample])
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


def format_plan(p: dict[str, Any], cfg: Config) -> str:
    lines = [
        f"cache        {p['root']}  (fingerprint {p['fingerprint']})",
        f"scenes       {len(p['scenes'])}: {', '.join(p['scenes'])}",
    ]
    if p["skipped_scenes"]:
        lines.append(f"skipped      {', '.join(p['skipped_scenes'])} (keyframe files missing)")
    lines += [
        f"items        {p['items']} camera images, {p['done']} already built",
        f"model        {cfg.model.encoder}, input {p['model_size'][0]} x {p['model_size'][1]}, "
        f"{p['images_per_item']} passes per item",
        f"grid         {p['grid_size'][0]} x {p['grid_size'][1]}, tap {cfg.model.tap} "
        f"({p['channels']} channels)",
        f"neighbors    items by neighbor count: "
        + ", ".join(f"{k}: {v}" for k, v in p["items_by_neighbors"].items()),
        f"points       about {p['lidar_points']:.0f} lidar and {p['radar_points']:.0f} radar per image "
        f"({cfg.points.lidar_sweeps} and {cfg.points.radar_sweeps} sweeps)",
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
        except Exception as e:  # noqa: BLE001
            problems.append(f"{key}: cannot load: {e}")
            continue
        gw, gh = meta["grid_size"]
        mw, mh = meta["model_size"]
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
        if "disparity" in a:
            check(a["disparity"].shape == (mh, mw), "disparity shape")
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
        radar_channels(meta, a, RadarConfig())
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
        f"tags         " + (", ".join(f"{k}: {c}" for k, c in v["tags"].items()) or "none"),
        f"points       {m['lidar_points']:.0f} lidar and {m['radar_points']:.0f} radar per entry",
        f"fits         lidar {v['fit_rate'].get('lidar', 0):.1%}, radar {v['fit_rate'].get('radar', 0):.1%} "
        f"of entries; mean lidar fit RMSE {m['fit_lidar_rmse']:.4f} 1/m",
        f"labels       {m['label_cells']:.2%} of grid cells hold a lidar label; "
        f"{m['positive']:.1%} of them have AbsRel > {v['tau']}",
        "AbsRel       median over entries of the median per entry: "
        + ", ".join(f"{k} {x:.3f}" for k, x in v["median_abs_rel_by_condition"].items()),
        f"temporal     " + ", ".join(f"{k}: {c}" for k, c in v["temporal_status"].items())
        + f"; {m['seen2']:.1%} of cells seen by 2+ frames, median relstd {m['relstd_median']:.4f}",
    ]
    if v["problems"]:
        lines.append(f"problems     {len(v['problems'])}:")
        lines += [f"  {p}" for p in v["problems"][:50]]
    else:
        lines.append("problems     none")
    return "\n".join(lines)


# ----- preview ----------------------------------------------------------------


def _pca_rgb(features: np.ndarray) -> np.ndarray:
    """(C, h, w) features -> (h, w, 3) in [0, 1], the first three principal components."""
    c, h, w = features.shape
    f = features.reshape(c, -1).T.astype(np.float64)
    f -= f.mean(0)
    _, _, vt = np.linalg.svd(f[:: max(1, len(f) // 20000)], full_matrices=False)
    proj = f @ vt[:3].T
    lo, hi = np.percentile(proj, [2, 98], axis=0)
    return np.clip((proj - lo) / np.where(hi > lo, hi - lo, 1), 0, 1).reshape(h, w, 3)


def preview(cache: Cache, key: str, dataroot: str | None, out_dir: Path,
            labels: LabelConfig = LabelConfig()) -> Path:
    """One entry as a 3 x 2 figure. Each grid cell shows as one block, on the image it covers."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.colors import LogNorm
    from matplotlib.figure import Figure

    entry = cache.entry(key)
    meta, a = entry.meta, entry.load()
    gw, gh = meta["grid_size"]
    w, h = meta["image"]["width"], meta["image"]["height"]
    extent = (-0.5, w - 0.5, h - 0.5, -0.5)
    root = Path(dataroot or cache.settings["source"]["dataroot"])
    image_path = root / meta["image"]["path"]
    rgb = np.asarray(Image.open(image_path).convert("RGB")) if image_path.is_file() else np.full((h, w, 3), 40, np.uint8)
    gray = rgb.mean(-1) / 255

    fit = fit_of(meta, labels.align)
    depth = fit.depth(a["disparity_grid"].astype(np.float64), labels.min_depth, labels.max_depth) if fit else None
    abs_rel, _, mask = lidar_labels(meta, a, labels)
    residual, valid = radar_channels(meta, a, RadarConfig())
    relstd = a["temporal_relstd"].astype(np.float32)
    count = a["temporal_count"]
    lidar = a["lidar"][a["lidar"][:, 4] == 0]

    fig = Figure(figsize=(12, 10.9), layout="constrained")
    fig.suptitle(f"{key} ({meta['condition']}, {meta['weather']})")
    axes = fig.subplots(3, 2)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    def show(ax, values, title, cmap, label, under_image=False, **kw):
        if under_image:
            ax.imshow(gray, cmap="gray", vmin=-0.6, vmax=1.6, extent=extent)
        im = ax.imshow(values, cmap=cmap, extent=extent, interpolation="nearest", **kw)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.9, label=label)

    ax = axes[0, 0]
    ax.imshow(rgb, extent=extent)
    ax.scatter(lidar[:, 0], lidar[:, 1], c=lidar[:, 2], s=1.5, cmap="turbo_r", norm=LogNorm(1, 80), linewidths=0)
    ax.set_title("image and keyframe lidar", fontsize=10)
    sm = ax.collections[0]
    fig.colorbar(sm, ax=ax, shrink=0.9, label="lidar depth (m)")

    if depth is None:
        show(axes[0, 1], a["disparity_grid"], f"predicted disparity on the {gw} x {gh} grid, no {labels.align} fit",
             "turbo", "disparity (relative)")
    else:
        show(axes[0, 1], depth, f"predicted depth on the {gw} x {gh} grid, {labels.align} fit",
             "turbo_r", "depth (m)", norm=LogNorm(labels.min_depth, labels.max_depth))
    temporal = meta["temporal"]
    title = (f"temporal spread over {len(temporal['neighbors'])} neighbor frames" if temporal["status"] == "ok"
             else f"temporal spread: {temporal['status']}")
    show(axes[1, 0], np.ma.masked_where(count < 2, relstd), title,
         "magma", "std / mean of inverse depth", under_image=True, vmin=0, vmax=0.1)
    ax = axes[1, 1]
    ax.imshow(_pca_rgb(a["features"].astype(np.float32)), extent=extent, interpolation="nearest")
    ax.set_title(f"features of {meta['tap']}, first 3 principal components", fontsize=10)
    show(axes[2, 0], np.ma.masked_where(~mask, abs_rel),
         f"label: lidar AbsRel, {mask.mean():.1%} of cells, {(abs_rel[mask] > labels.tau).mean() if mask.any() else 0:.0%} above tau {labels.tau:g}",
         "RdYlGn_r", "AbsRel", under_image=True, vmin=0, vmax=3 * labels.tau)
    on = valid > 0
    show(axes[2, 1], np.ma.masked_where(~on, residual), f"input: radar residual, {int(on.sum())} cells",
         "RdYlGn_r", "|pred - z| / z", under_image=True, vmin=0, vmax=3 * labels.tau)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{key}.png"
    fig.savefig(out, dpi=110)
    return out


def write_summary(v: dict[str, Any], cfg_text: str, path: Path) -> None:
    """The verify result as a small markdown file for the team."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Cache summary\n\nWritten by `confcache verify --summary`.\n\n```\n"
                    + format_verify(v) + "\n```\n\nSettings:\n\n```toml\n" + cfg_text + "```\n")
