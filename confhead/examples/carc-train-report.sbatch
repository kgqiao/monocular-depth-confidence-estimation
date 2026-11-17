#!/bin/bash
# Write the report of a set of training jobs, on CPU, when they have all ended.
# `confhead train run` submits it; do not submit it by hand.
#SBATCH --job-name=confhead-report
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export UV_PROJECT_ENVIRONMENT="${TMPDIR:-/tmp}/confhead-venv-${SLURM_JOB_ID}"
export UV_LINK_MODE=copy
trap 'rm -rf "${UV_PROJECT_ENVIRONMENT}"' EXIT

uv run --locked confhead train status "$@"
