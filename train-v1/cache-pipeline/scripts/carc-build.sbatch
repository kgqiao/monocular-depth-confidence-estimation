#!/bin/bash
# Build the cache on CARC as a job array, one shard per array task.
#
#   mkdir -p logs
#   sbatch --account=<project> --array=0-7 --export=ALL,CONFIG=configs/nuscenes-mini.toml scripts/carc-build.sbatch
#
# Each task builds every 8th item and writes the index when it ends. A task that
# stops early continues where it was when you submit it again. Extra arguments
# go to `confcache build`, for example `--set run.precision=fp16`.
#SBATCH --job-name=confcache
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/confcache-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
CONFIG="${CONFIG:-configs/nuscenes-mini.toml}"

uv run confcache build --config "${CONFIG}" \
    --shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}" \
    --set run.device=cuda \
    --set run.num_workers=$((SLURM_CPUS_PER_TASK - 1)) \
    "$@"
