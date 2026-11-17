#!/bin/bash
# Train one run of the run matrix on one CARC GPU. `confhead train run` submits
# it, with the job name, the time limit, and the log file of the run; do not
# submit it by hand. guides/training-on-carc.md has the steps.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export UV_PROJECT_ENVIRONMENT="${TMPDIR:-/tmp}/confhead-venv-${SLURM_JOB_ID}"
export UV_LINK_MODE=copy
trap 'rm -rf "${UV_PROJECT_ENVIRONMENT}"' EXIT

uv run --locked confhead train worker "$@"
