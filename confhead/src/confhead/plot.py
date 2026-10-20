"""Figures of the data and the cache. Each function returns a matplotlib Figure; the caller saves it."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from PIL import Image

from .cache.store import Entry
from .dataset import RadarConfig, radar_channels
from .labels import LabelConfig, fit_of, lidar_labels

RADAR_COLOR = "#ff28c8"


def draw_points(ax: Axes, image: np.ndarray, lidar: np.ndarray | None = None, radar: np.ndarray | None = None,
                min_depth: float = 1.0, max_depth: float = 80.0):
    """The image with lidar dots colored by depth and radar circles labeled in meters.

    lidar, radar: rows starting with (u, v, z) in image pixels, as from `confhead.items.load_points`.
    Returns the lidar scatter, for a colorbar, or None without lidar.
    """
    h, w = image.shape[:2]
    ax.imshow(image, extent=(-0.5, w - 0.5, h - 0.5, -0.5))
    dots = None
    if lidar is not None and len(lidar):
        far_first = np.argsort(-lidar[:, 2])
        dots = ax.scatter(lidar[far_first, 0], lidar[far_first, 1], c=lidar[far_first, 2], s=1.5, cmap="turbo_r",
                          norm=LogNorm(min_depth, max_depth), linewidths=0)
    if radar is not None and len(radar):
        ax.scatter(radar[:, 0], radar[:, 1], s=60, facecolors="none", edgecolors=RADAR_COLOR, linewidths=1.5)
        for u, v, z in radar[:, :3]:
            ax.annotate(f"{z:.0f} m", (u, v), xytext=(6, -6), textcoords="offset points", color=RADAR_COLOR,
                        fontsize=7)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    return dots


def _pca_rgb(features: np.ndarray) -> np.ndarray:
    """(C, h, w) features -> (h, w, 3) in [0, 1], the first three principal components."""
    c, h, w = features.shape
    f = features.reshape(c, -1).T.astype(np.float64)
    f -= f.mean(0)
    _, _, vt = np.linalg.svd(f[:: max(1, len(f) // 20000)], full_matrices=False)
    proj = f @ vt[:3].T
    lo, hi = np.percentile(proj, [2, 98], axis=0)
    return np.clip((proj - lo) / np.where(hi > lo, hi - lo, 1), 0, 1).reshape(h, w, 3)


def entry_figure(entry: Entry, labels: LabelConfig = LabelConfig(), radar: RadarConfig = RadarConfig()) -> Figure:
    """One cache entry as a 3 x 2 figure. Each grid cell shows as one block, on the image it covers."""
    meta, a = entry.meta, entry.load()
    key = entry.key
    gw, gh = meta["grid_size"]
    w, h = meta["image"]["width"], meta["image"]["height"]
    extent = (-0.5, w - 0.5, h - 0.5, -0.5)
    path = entry.image_path
    rgb = np.asarray(Image.open(path).convert("RGB")) if path.is_file() else np.full((h, w, 3), 40, np.uint8)
    gray = rgb.mean(-1) / 255

    fit = fit_of(meta, labels.align)
    abs_rel, _, mask = lidar_labels(meta, a, labels)
    residual, valid = radar_channels(meta, a, radar)
    count = a["temporal_count"]

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

    dots = draw_points(axes[0, 0], rgb, lidar=a["lidar"], min_depth=labels.min_depth, max_depth=labels.max_depth)
    axes[0, 0].set_title("image and keyframe lidar", fontsize=10)
    if dots is not None:
        fig.colorbar(dots, ax=axes[0, 0], shrink=0.9, label="lidar depth (m)")

    if fit is None:
        show(axes[0, 1], a["disparity_grid"], f"predicted disparity on the {gw} x {gh} grid, no {labels.align} fit",
             "turbo", "disparity (relative)")
    else:
        depth = fit.depth(a["disparity_grid"].astype(np.float64), labels.min_depth, labels.max_depth)
        show(axes[0, 1], depth, f"predicted depth on the {gw} x {gh} grid, {labels.align} fit",
             "turbo_r", "depth (m)", norm=LogNorm(labels.min_depth, labels.max_depth))

    temporal = meta["temporal"]
    title = (f"temporal spread over {len(temporal['neighbors'])} neighbor frames" if temporal.get("status") == "ok"
             else f"temporal spread: {temporal.get('status', 'off')}")
    show(axes[1, 0], np.ma.masked_where(count < 2, a["temporal_relstd"].astype(np.float32)), title,
         "magma", "std / mean of inverse depth", under_image=True, vmin=0, vmax=0.1)

    axes[1, 1].imshow(_pca_rgb(a["features"].astype(np.float32)), extent=extent, interpolation="nearest")
    axes[1, 1].set_title(f"features of {meta['tap']}, first 3 principal components", fontsize=10)

    positive = (abs_rel[mask] > labels.tau).mean() if mask.any() else 0.0
    show(axes[2, 0], np.ma.masked_where(~mask, abs_rel),
         f"label: lidar AbsRel, {mask.mean():.1%} of cells, {positive:.0%} above tau {labels.tau:g}",
         "RdYlGn_r", "AbsRel", under_image=True, vmin=0, vmax=3 * labels.tau)
    on = valid > 0
    show(axes[2, 1], np.ma.masked_where(~on, residual), f"input: radar residual, {int(on.sum())} cells",
         "RdYlGn_r", "|pred - z| / z", under_image=True, vmin=0, vmax=3 * labels.tau)
    return fig
