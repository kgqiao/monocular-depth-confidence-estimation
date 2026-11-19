
"""
Figures of the evaluation results. Each function returns a matplotlib Figure; the caller saves it.

The numbers come from `evals/metrics.py`; nothing here computes a metric.

- reliability_figure: per-bin accuracy against confidence, with the cell count per bin
- sparsification_figure: mean AbsRel of the cells kept as the least trusted are removed;
  the head and any baselines against the oracle and random
- qualitative_figure: one image, its lidar AbsRel on labeled cells, and p_error on every cell

"""

import numpy as np
from matplotlib.figure import Figure

from confhead.plot import draw_map, draw_points    # helper functions from confhead.plot


def reliability_figure(bin_conf: np.ndarray, bin_acc: np.ndarray, bin_count: np.ndarray, title: str = "") -> Figure:
    fig = Figure(figsize=(5, 6), layout="constrained")
    top, bottom = fig.subplots(2, 1, sharex=True, height_ratios=[3, 1])

    filled = bin_count > 0
    top.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    top.plot(bin_conf[filled], bin_acc[filled], marker="o", label="confidence head")

    top.set_ylabel("share correct")
    top.set_xlim(0, 1) # fix both axes to 0 to 1
    top.set_ylim(0, 1)
    top.set_aspect("equal")
    top.legend(loc="upper left")
    if title:
        top.set_title(title)

    n_bins = len(bin_count)
    centers = (np.arange(n_bins) + 0.5) / n_bins
    bottom.bar(centers, bin_count, width=0.9 / n_bins)
    bottom.set_xlabel("confidence")
    bottom.set_ylabel("cells")
    return fig


def sparsification_figure(fractions: np.ndarray, oracle_curve: np.ndarray, random_curve: np.ndarray,
                          methods: dict[str, np.ndarray], title: str = "") -> Figure:
    fig = Figure(figsize=(6, 4.5), layout="constrained")
    ax = fig.subplots()  # one panel

    ax.plot(fractions, random_curve, linestyle="--", color="gray", label="random")
    ax.plot(fractions, oracle_curve, color="black", label="oracle (best possible)")
    for name, curve in methods.items(): # the head, plus any baselines
        ax.plot(fractions, curve, label=name)

    first = next(iter(methods.values()))  # the first method passed in, normally the head
    ax.fill_between(fractions, oracle_curve, first, alpha=0.15, label="AUSE (gap to oracle)")

    ax.set_xlabel("fraction of cells removed (least trusted first)")
    ax.set_ylabel("mean AbsRel of the cells kept")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend()
    if title:
        ax.set_title(title)
    return fig


def qualitative_figure(rgb: np.ndarray, lidar: np.ndarray, abs_rel: np.ndarray, mask: np.ndarray,
                       p_error: np.ndarray, tau: float = 0.1, title: str = "") -> Figure:
    h, w = rgb.shape[:2]
    size = (w, h)   # draw_map wants (width, height)
    fig = Figure(figsize=(16, 3.6), layout="constrained")
    axes = fig.subplots(1, 3) # three panels side by side

    draw_points(axes[0], rgb, lidar=lidar) # photo + lidar dots colored by depth
    axes[0].set_title(title or "image and lidar")

    shown = draw_map(axes[1], abs_rel, size, image=rgb, mask=mask, cmap="RdYlGn_r", vmin=0, vmax=3 * tau)
    axes[1].set_title("lidar AbsRel (labeled cells)")
    fig.colorbar(shown, ax=axes[1], shrink=0.8)

    shown = draw_map(axes[2], p_error, size, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[2].set_title("p_error (every cell)")
    fig.colorbar(shown, ax=axes[2], shrink=0.8)
    return fig