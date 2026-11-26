"""Pooled sparsification and reliability figures for scored runs.

Reads the cell cache that evaluate_head.py wrote for each run
(<root>/<run>/<run>_cells.npz: p_error, abs_rel, target, scene for every
labeled val cell) and draws two figures with src/evals/plot.py:

  1. sparsification curves of all the runs given, against the oracle and
     random curves (the first run is the head: its AUSE gap is shaded),
  2. the reliability diagram of the first run.

Every cell is pooled into one curve. The results table averages per-image
AUSE instead, so the pooled numbers printed here will not equal the table.
p_error is shifted by pos_weight, so the reliability diagram is too.

Run it as a job (the cache is 4 GB):
    sbatch examples/carc-plot.sbatch run-v1-radar [more runs ...]
"""
import argparse
import os
from pathlib import Path

import numpy as np

from evals.metrics import (
    calculate_ause_and_aurg,
    compute_reliability_bins,
    compute_sparsification_curves,
)
from evals.plot import reliability_figure, sparsification_figure


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("runs", nargs="+", help="run names; the first one is the head")
    p.add_argument("--root", default=f"/scratch1/{os.environ.get('USER', 'me')}/confhead-eval",
                   help="folder that holds <run>/<run>_cells.npz")
    p.add_argument("--out", default="results/figures", help="where the figures go")
    return p.parse_args()


def load_cells(root, run):
    """p_error, abs_rel, target of one run. scene is never read (it is 3.5 GB)."""
    path = Path(root) / run / f"{run}_cells.npz"
    with np.load(path) as z:
        return z["p_error"], z["abs_rel"], z["target"]


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    curves = {}      # run name -> sparsification curve
    shared = None    # fractions, oracle, random, taken from the head
    head_cells = None

    for i, run in enumerate(args.runs):
        print(f"loading {run} ...", flush=True)
        p_error, abs_rel, target = load_cells(args.root, run)
        if i == 0:
            head_cells = (p_error, target)
            head_abs_rel = abs_rel
        elif len(abs_rel) != len(head_abs_rel) or not np.array_equal(abs_rel, head_abs_rel):
            raise ValueError(f"{run} was scored on other cells than {args.runs[0]}; "
                             "its curve cannot share the oracle")

        # uncertainty = 1 - conf = p_error, true error = abs_rel (as in evaluate_head.py)
        fracs, curve, oracle, rand = compute_sparsification_curves(abs_rel, p_error)
        ause, aurg = calculate_ause_and_aurg(fracs, curve, oracle, rand)
        print(f"  {run}: pooled AUSE {ause:.5f}, AURG {aurg:.5f}, cells {len(abs_rel):,}")
        curves[run] = curve
        if i == 0:
            shared = (fracs, oracle, rand)
        del p_error, abs_rel, target

    fracs, oracle, rand = shared
    fig = sparsification_figure(fracs, oracle, rand, curves,
                                title="Sparsification, pooled val cells")
    fig.savefig(out / "sparsification.png", dpi=200)

    p_error, target = head_cells
    bin_conf, bin_acc, bin_count = compute_reliability_bins(1 - target, 1.0 - p_error)
    fig = reliability_figure(bin_conf, bin_acc, bin_count,
                             title=f"{args.runs[0]}, pooled val cells\n(confidence shifted by pos_weight)")
    fig.savefig(out / "reliability.png", dpi=200)

    # the numbers behind the plots, small enough to keep
    np.savez(out / "curves.npz", fractions=fracs, oracle=oracle, random=rand,
             bin_conf=bin_conf, bin_acc=bin_acc, bin_count=bin_count,
             **{f"curve_{k}": v for k, v in curves.items()})
    print(f"wrote sparsification.png, reliability.png, curves.npz to {out}")


if __name__ == "__main__":
    main()
