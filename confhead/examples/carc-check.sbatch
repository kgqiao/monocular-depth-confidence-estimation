#!/bin/bash
# Write the index of the cache and check every entry, on CPU. It builds nothing.
# examples/carc-submit.sh submits it to start when the build tasks end.
#SBATCH --job-name=confhead-check
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/confhead-check-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
CONFIG="${1:?usage: sbatch examples/carc-check.sbatch <config> [build_cache.py arguments]}"
shift

uv run examples/build_cache.py --config "${CONFIG}" --check "$@"
