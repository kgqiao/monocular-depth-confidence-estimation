#!/bin/bash
# Score one trained head on the val split, on one CARC GPU. From the confhead folder:
# sbatch examples/carc-eval.sbatch --checkpoint /scratch1/naday/confhead-runs/<run>/best.pt
# Writes results/<run>/<run>.csv and /scratch1/$USER/confhead-eval/<run>/<run>_cells.npz.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=confhead-eval
#SBATCH --time=02:00:00
#SBATCH --output=logs/confhead-eval-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export UV_PROJECT_ENVIRONMENT="${TMPDIR:-/tmp}/confhead-venv-${SLURM_JOB_ID}"
export UV_LINK_MODE=copy
trap 'rm -rf "${UV_PROJECT_ENVIRONMENT}"' EXIT

uv run --locked python -m examples.evaluate_head "$@"
