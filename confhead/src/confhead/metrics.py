"""Scores of a confidence map against the lidar error, over the labeled cells.

    auroc           how well a score separates the cells above tau from the rest
    sparsification  the mean error left after the most uncertain cells are removed
    ause_aurg       the area between that curve and the oracle's, and between random removal and it

Sparsification is per image, as in Poggi et al., "On the uncertainty of
self-supervised monocular depth estimation" (CVPR 2020); average over images.
"""

from __future__ import annotations

import numpy as np


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """The chance that a random positive scores above a random negative, ties counted half; nan without both."""
    labels = np.asarray(labels).astype(bool).ravel()
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inverse, counts = np.unique(np.asarray(scores).ravel(), return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def sparsification(errors: np.ndarray, uncertainty: np.ndarray, steps: int = 50) -> np.ndarray:
    """The mean error of the cells kept after the most uncertain fraction k / steps is removed, k = 0 .. steps - 1."""
    errors = np.asarray(errors, dtype=np.float64).ravel()
    if errors.size == 0:
        raise ValueError("sparsification needs at least one cell")
    kept_sum = np.cumsum(errors[np.argsort(np.asarray(uncertainty).ravel(), kind="stable")])
    keep = errors.size - np.floor(np.arange(steps) * errors.size / steps).astype(np.int64)
    return kept_sum[keep - 1] / keep


def ause_aurg(errors: np.ndarray, uncertainty: np.ndarray, steps: int = 50) -> tuple[float, float]:
    """AUSE and AURG of one image, in the units of the error, over removed fractions 0 to 1.

    AUSE is the area between the curve and the oracle's, which removes the largest
    errors first: 0 is a perfect ranking. AURG is the area between random removal,
    which keeps the mean error, and the curve: above 0 is better than random.
    """
    curve = sparsification(errors, uncertainty, steps)
    oracle = sparsification(errors, errors, steps)
    return float((curve - oracle).mean()), float(curve[0] - curve.mean())
