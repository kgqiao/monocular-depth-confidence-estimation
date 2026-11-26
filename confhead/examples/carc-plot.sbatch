#!/bin/bash
# Draw the pooled sparsification and reliability figures from the cell caches.
# From the confhead folder:  sbatch examples/carc-plot.sbatch run-v1-radar [more runs ...]
# Writes results/figures/sparsification.png, reliability.png and curves.npz.
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=confhead-plot
#SBATCH --time=01:00:00
#SBATCH --output=logs/confhead-plot-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export UV_PROJECT_ENVIRONMENT="${TMPDIR:-/tmp}/confhead-venv-${SLURM_JOB_ID}"
export UV_LINK_MODE=copy
trap 'rm -rf "${UV_PROJECT_ENVIRONMENT}"' EXIT

uv run --locked python -m examples.plot_curves "$@"
