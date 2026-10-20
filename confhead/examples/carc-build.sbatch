#!/bin/bash
# Build the cache on one CARC GPU. examples/carc-submit.sh submits this as a job
# array, one shard per task. Without --array it builds every entry, so a short
# test run is, from the confhead folder:
#
#   sbatch --time=00:30:00 examples/carc-build.sbatch configs/nuscenes-trainval.toml --limit 50
#
# Arguments after the config go to examples/build_cache.py.
#SBATCH --job-name=confhead-build
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/confhead-build-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
CONFIG="${1:?usage: sbatch examples/carc-build.sbatch <config> [build_cache.py arguments]}"
shift

uv run examples/build_cache.py --config "${CONFIG}" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}" "$@"
