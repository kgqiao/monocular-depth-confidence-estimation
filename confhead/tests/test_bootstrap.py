"""Tests for evals.bootstrap, all on tiny tables where the answers are now known."""

import numpy as np
import pytest

from evals.bootstrap import (
    paired_scene_bootstrap_ci,
    scene_bootstrap_ci,
    scene_bootstrap_diff_ci,
    scene_bootstrap_means,
)


def test_point_estimate_is_image_weighted_mean():
    # scene "a" has 3 images, scene "b" has 1. The mean over images is
    # (1 + 2 + 3 + 10) / 4 = 4. The mean of the scene means would be
    # (2 + 10) / 2 = 6, so this catches a wrong weighting.
    values = [1.0, 2.0, 3.0, 10.0]
    scenes = ["a", "a", "a", "b"]
    mean, _, _ = scene_bootstrap_ci(values, scenes, n_boot=50)
    assert mean == pytest.approx(4.0)


def test_identical_scenes_give_zero_width():
    # every scene has the same size and the same values, so every resample
    # has the same mean and the interval collapses to a point
    values = [0.2, 0.4] * 5
    scenes = np.repeat(["s1", "s2", "s3", "s4", "s5"], 2)
    mean, lo, hi = scene_bootstrap_ci(values, scenes, n_boot=200)
    assert mean == pytest.approx(0.3)
    assert lo == pytest.approx(0.3)
    assert hi == pytest.approx(0.3)


def test_two_scenes_zero_and_one():
    # scene A is all 0 and scene B is all 1, 10 images each. A resample of 2
    # scenes gives 0 (25%), 0.5 (50%) or 1 (25%), so the 95% interval is
    # the full range [0, 1]
    values = [0.0] * 10 + [1.0] * 10
    scenes = ["A"] * 10 + ["B"] * 10
    mean, lo, hi = scene_bootstrap_ci(values, scenes, n_boot=2000)
    assert mean == pytest.approx(0.5)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(1.0)


def test_scene_interval_is_wider_than_image_interval():
    # This is the test that catches resampling images by mistake.
    # One scene has 1 image (value 0), the other has 1,000 images (value 1).
    # Resampling the two scenes gives anything from 0 to 1. Resampling the
    # 1,001 images gives a very narrow interval around 0.999.
    values = np.array([0.0] + [1.0] * 1000)
    scenes = np.array(["small"] + ["big"] * 1000)

    _, lo, hi = scene_bootstrap_ci(values, scenes, n_boot=2000)
    scene_width = hi - lo

    # the image-level bootstrap I do NOT want, for comparison
    rng = np.random.default_rng(0)
    image_means = [
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(2000)
    ]
    image_width = np.percentile(image_means, 97.5) - np.percentile(image_means, 2.5)

    assert scene_width > 0.9
    assert image_width < 0.01
    assert scene_width > 50 * image_width


def test_same_seed_same_result_and_other_seed_differs():
    rng = np.random.default_rng(1)
    values = rng.random(60)
    scenes = np.repeat(np.arange(12), 5)
    first = scene_bootstrap_ci(values, scenes, n_boot=300, seed=7)
    again = scene_bootstrap_ci(values, scenes, n_boot=300, seed=7)
    other = scene_bootstrap_ci(values, scenes, n_boot=300, seed=8)
    assert first == again
    assert first != other
    # the point estimate does not depend on the seed
    assert first[0] == other[0]


def test_means_array_has_one_entry_per_resample_and_brackets_the_mean():
    rng = np.random.default_rng(2)
    values = rng.random(40)
    scenes = np.repeat(np.arange(8), 5)
    boot = scene_bootstrap_means(values, scenes, n_boot=500, seed=3)
    assert boot.shape == (500,)
    assert boot.min() <= values.mean() <= boot.max()


def test_nan_rows_are_dropped_like_pandas_mean():
    values = [1.0, np.nan, 3.0, 5.0]
    scenes = ["a", "a", "b", "b"]
    mean, _, _ = scene_bootstrap_ci(values, scenes, n_boot=50)
    # mean of 1, 3, 5
    assert mean == pytest.approx(3.0)


def test_diff_of_two_constant_conditions():
    # night is always 0.05, day is always 0.02, so the difference is 0.03
    # in every resample and the interval has no width
    night_vals, night_scenes = [0.05] * 6, ["n1", "n1", "n2", "n2", "n3", "n3"]
    day_vals, day_scenes = [0.02] * 4, ["d1", "d1", "d2", "d2"]
    diff, lo, hi = scene_bootstrap_diff_ci(
        night_vals, night_scenes, day_vals, day_scenes, n_boot=200
    )
    assert diff == pytest.approx(0.03)
    assert lo == pytest.approx(0.03)
    assert hi == pytest.approx(0.03)


def test_diff_interval_contains_zero_for_same_distribution():
    # two conditions drawn from the same noise: the true gap is 0, so the
    # interval should straddle 0
    rng = np.random.default_rng(4)
    a_vals, a_scenes = rng.normal(0.03, 0.01, 80), np.repeat(np.arange(16), 5)
    b_vals, b_scenes = rng.normal(0.03, 0.01, 80), np.repeat(np.arange(16), 5)
    _, lo, hi = scene_bootstrap_diff_ci(a_vals, a_scenes, b_vals, b_scenes, seed=5)
    assert lo < 0.0 < hi


def test_paired_constant_offset_has_no_width():
    # model b is exactly 0.1 worse than model a on every image, so the paired
    # difference is 0.1 whichever scenes are drawn
    rng = np.random.default_rng(6)
    a = rng.random(50)
    b = a + 0.1
    scenes = np.repeat(np.arange(10), 5)
    diff, lo, hi = paired_scene_bootstrap_ci(a, b, scenes, n_boot=300)
    assert diff == pytest.approx(-0.1)
    assert lo == pytest.approx(-0.1)
    assert hi == pytest.approx(-0.1)


def test_paired_is_tighter_than_unpaired_when_models_move_together():
    # both models have a big scene effect (some scenes are just harder),
    # and model b is 0.01 worse everywhere. Unpaired, the scene effect hides
    # the 0.01. Paired, it cancels, so the interval stays clear of 0.
    rng = np.random.default_rng(8)
    scene_effect = np.repeat(rng.normal(0.0, 0.05, 12), 5)
    a = 0.03 + scene_effect + rng.normal(0, 0.001, 60)
    b = a + 0.01
    scenes = np.repeat(np.arange(12), 5)

    _, p_lo, p_hi = paired_scene_bootstrap_ci(a, b, scenes, seed=9)
    _, u_lo, u_hi = scene_bootstrap_diff_ci(a, scenes, b, scenes, seed=9)

    assert p_hi < 0.0 or p_lo > 0.0  # paired interval excludes 0
    assert u_lo < 0.0 < u_hi         # unpaired interval does not


def test_paired_drops_rows_where_either_model_is_nan():
    a = np.array([1.0, np.nan, 3.0, 4.0])
    b = np.array([0.0, 2.0, np.nan, 2.0])
    scenes = ["x", "x", "y", "y"]
    # only rows 0 and 3 survive: differences 1.0 and 2.0, mean 1.5
    diff, _, _ = paired_scene_bootstrap_ci(a, b, scenes, n_boot=50)
    assert diff == pytest.approx(1.5)


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        scene_bootstrap_ci([1.0, 2.0], ["a"])            # lengths differ
    with pytest.raises(ValueError):
        scene_bootstrap_ci([np.nan, np.nan], ["a", "b"])  # nothing finite
    with pytest.raises(ValueError):
        scene_bootstrap_ci([1.0, 2.0], ["a", "b"], ci=1.5)
    with pytest.raises(ValueError):
        scene_bootstrap_ci([1.0, 2.0], ["a", "b"], n_boot=0)
    with pytest.raises(ValueError):
        paired_scene_bootstrap_ci([1.0, 2.0], [1.0], ["a", "b"])