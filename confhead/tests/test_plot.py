"""
Tests for the figures in evals/plot.py: reliability, sparsification, qualitative.

A test can't judge whether a figure looks right, so these check that each one builds
without an error and has the expected parts (panels, lines, points). To look at the
figures themselves, save one from made-up data and open it.

- Reliability: two panels (diagram and cell counts); empty bins are not drawn; saves to a file
- Sparsification: one line per curve (random, oracle, and every method passed in)
- Qualitative: three panels plus two colorbars
"""

import numpy as np
from evals.metrics import compute_reliability_bins, compute_sparsification_curves
from evals.plot import qualitative_figure, reliability_figure, sparsification_figure


def test_reliability_figure_draws_and_saves(tmp_path):
    rng = np.random.default_rng(0)
    confidence = rng.random(1_000)
    is_correct = (rng.random(1_000) < confidence).astype(int)
    fig = reliability_figure(*compute_reliability_bins(is_correct, confidence), title="test")
    assert len(fig.axes) == 2 # diagram on top, counts below
    assert len(fig.axes[0].lines) == 2  # the diagonal and the head
    fig.savefig(tmp_path / "rel.png") # pytest's temporary folder, deleted after
    assert (tmp_path / "rel.png").stat().st_size > 0


def test_reliability_figure_skips_empty_bins():
    is_correct = np.array([0, 0, 1, 0])
    confidence = np.array([0.2, 0.2, 0.8, 0.8]) # only 2 of 10 bins have cells
    fig = reliability_figure(*compute_reliability_bins(is_correct, confidence))
    xs = fig.axes[0].lines[1].get_xdata()  # x of the head's points
    assert len(xs) == 2


def test_sparsification_figure_draws_every_method():
    rng = np.random.default_rng(1)
    err = rng.random(500)
    fractions, head, oracle, rand = compute_sparsification_curves(err, err + rng.normal(0, 0.1, 500))
    fig = sparsification_figure(fractions, oracle, rand, {"head": head, "baseline": rand})
    assert len(fig.axes[0].lines) == 4 # random, oracle, and the two methods


def test_qualitative_figure_has_three_panels():
    rng = np.random.default_rng(2)
    rgb = np.zeros((90, 160, 3), dtype=np.uint8)  # a tiny black image
    lidar = np.column_stack([rng.uniform(0, 160, 50), rng.uniform(0, 90, 50), rng.uniform(2, 70, 50)])
    grid = rng.random((9, 16))
    fig = qualitative_figure(rgb, lidar, grid, grid > 0.5, grid)
    assert len(fig.axes) == 5  # 3 panels and 2 colorbars