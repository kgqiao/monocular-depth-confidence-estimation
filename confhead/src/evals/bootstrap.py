"""Scene bootstrap for the per-image eval metrics.

Why scenes and not images
-------------------------
The val split has 36,114 images but only 150 scenes (day 111, rain 24,
night 15). Images from one scene are near copies of each other (same road,
same light, neighbouring frames), so they are not independent. A bootstrap
that resamples images treats 3,578 night images as 3,578 pieces of evidence
when the real number is closer to 15. Its interval comes out far too narrow.
Thus the unit I resample is the scene: draw scenes with replacement, keep every
image of each drawn scene, recompute the mean.

What it computes
----------------
The mean is image-weighted, the same as the results table (mean of the
per-image metric over all images of the condition). For each resample that is
(sum of the drawn scenes' totals) / (sum of the drawn scenes' image counts).
I precompute one total and one count per scene, so a resample never loops over
images and 2,000 resamples take a fraction of a second.

Use it on one condition at a time (pass only the day rows, or only the night
rows), the same way the results table filters by condition.

Example (CSV from examples/evaluate_head.py, columns scene, condition, ause):

    import pandas as pd
    from evals.bootstrap import scene_bootstrap_ci

    df = pd.read_csv("results/run-v1-radar/run-v1-radar.csv")
    night = df[df["condition"] == "night"]
    mean, lo, hi = scene_bootstrap_ci(night["ause"], night["scene"])

Three questions it answers:
- one model, one condition:      scene_bootstrap_ci
- night vs day (different scenes): scene_bootstrap_diff_ci
- model A vs model B (same images): paired_scene_bootstrap_ci
"""

from __future__ import annotations

import numpy as np

N_BOOT = 2000
CI_LEVEL = 0.95


def _check_args(n_boot: int, ci: float) -> None:
    """Reject settings that would silently give nonsense."""
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be between 0 and 1, got {ci}")


def _scene_sums_counts(values, scenes) -> tuple[np.ndarray, np.ndarray]:
    """Per scene: the sum of the metric and the number of images.

    Rows whose value is NaN or infinite are dropped first. That matches
    pandas .mean(), which the results table uses, so the point estimate here
    equals the table value.

    Returns two float arrays with one entry per scene (same order).
    """
    values = np.asarray(values, dtype=float)
    scenes = np.asarray(scenes)
    if values.ndim != 1 or values.shape != scenes.shape:
        raise ValueError(
            f"values and scenes must be 1-D and the same length, "
            f"got {values.shape} and {scenes.shape}"
        )

    keep = np.isfinite(values)
    values, scenes = values[keep], scenes[keep]
    if values.size == 0:
        raise ValueError("no finite values to bootstrap")

    # inverse maps every image to the index of its scene (0 .. n_scenes - 1)
    _, inverse = np.unique(scenes, return_inverse=True)
    inverse = inverse.reshape(-1)
    n_scenes = int(inverse.max()) + 1

    sums = np.bincount(inverse, weights=values, minlength=n_scenes)
    counts = np.bincount(inverse, minlength=n_scenes).astype(float)
    return sums, counts


def _resampled_means(sums, counts, n_boot, seed) -> np.ndarray:
    """n_boot bootstrap means, each from a resample of the scenes."""
    rng = np.random.default_rng(seed)
    n_scenes = len(sums)
    # one row per resample: which scenes were drawn (with replacement)
    drawn = rng.integers(0, n_scenes, size=(n_boot, n_scenes))
    return sums[drawn].sum(axis=1) / counts[drawn].sum(axis=1)


def _interval(boot_means, ci) -> tuple[float, float]:
    """Percentile interval of the bootstrap means."""
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(boot_means, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return float(lo), float(hi)


def scene_bootstrap_means(values, scenes, n_boot=N_BOOT, seed=0) -> np.ndarray:
    """All n_boot resampled means (use this to plot or inspect the spread).

    values: per-image metric, one number per image
    scenes: scene label per image (any hashable labels, same length as values)
    n_boot: number of resamples
    seed:   seed for numpy's random generator, so the result is reproducible
    """
    _check_args(n_boot, 0.5)
    sums, counts = _scene_sums_counts(values, scenes)
    return _resampled_means(sums, counts, n_boot, seed)


def scene_bootstrap_ci(
    values, scenes, n_boot=N_BOOT, ci=CI_LEVEL, seed=0
) -> tuple[float, float, float]:
    """Image-weighted mean of the metric with a scene-bootstrap interval.

    Returns (mean, lower, upper). The mean is computed on the original data,
    not on the resamples, so it equals the value in the results table.
    """
    _check_args(n_boot, ci)
    sums, counts = _scene_sums_counts(values, scenes)
    point = float(sums.sum() / counts.sum())
    lo, hi = _interval(_resampled_means(sums, counts, n_boot, seed), ci)
    return point, lo, hi


def scene_bootstrap_diff_ci(
    values_a,
    scenes_a,
    values_b,
    scenes_b,
    n_boot=N_BOOT,
    ci=CI_LEVEL,
    seed=0,
) -> tuple[float, float, float]:
    """Mean(a) minus mean(b) when a and b are DIFFERENT scenes.

    Use for night vs day: the two conditions share no scenes, so each one is
    resampled on its own (separate random streams) and the difference is taken
    resample by resample.

    Returns (difference, lower, upper). If the interval does not contain 0,
    the gap is not explained by which scenes happened to be in the split.
    """
    _check_args(n_boot, ci)
    sums_a, counts_a = _scene_sums_counts(values_a, scenes_a)
    sums_b, counts_b = _scene_sums_counts(values_b, scenes_b)

    # two independent streams from one seed
    seed_a, seed_b = np.random.SeedSequence(seed).spawn(2)
    boot_a = _resampled_means(sums_a, counts_a, n_boot, seed_a)
    boot_b = _resampled_means(sums_b, counts_b, n_boot, seed_b)

    point = float(sums_a.sum() / counts_a.sum() - sums_b.sum() / counts_b.sum())
    lo, hi = _interval(boot_a - boot_b, ci)
    return point, lo, hi


def paired_scene_bootstrap_ci(
    values_a, values_b, scenes, n_boot=N_BOOT, ci=CI_LEVEL, seed=0
) -> tuple[float, float, float]:
    """Mean(a) minus mean(b) when a and b are scored on the SAME images.

    Use for model vs model (for example a head vs the flip baseline). The rows
    of values_a, values_b and scenes must line up image by image: sort both
    CSVs by the same key first. Both models are evaluated on the scenes that
    get drawn, so the scene-to-scene differences cancel and the interval is
    tighter than comparing two separate intervals.

    Images where either model has a NaN are dropped from both.
    Returns (difference, lower, upper).
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    scenes = np.asarray(scenes)
    if not (a.ndim == 1 and a.shape == b.shape == scenes.shape):
        raise ValueError(
            f"values_a, values_b and scenes must be 1-D and the same length, "
            f"got {a.shape}, {b.shape} and {scenes.shape}"
        )

    # mean(a) - mean(b) over the same images equals the mean of the per-image
    # difference, so one ordinary scene bootstrap on (a - b) does it
    keep = np.isfinite(a) & np.isfinite(b)
    return scene_bootstrap_ci(
        (a - b)[keep], scenes[keep], n_boot=n_boot, ci=ci, seed=seed
    )