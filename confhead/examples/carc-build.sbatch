#!/bin/bash
# Build the cache on CARC as a job array, one shard per array task. From the confhead folder:
#
#   mkdir -p logs
#   sbatch --account=<project> --array=0-7 --export=ALL,CONFIG=configs/nuscenes-mini.toml examples/carc-build.sbatch
#
# Each task builds every 8th item. A task that stops early continues where it was
# when you submit it again. When every task has ended, run
# `uv run examples/build_cache.py --config <the same config>` once to write the
# index and check the cache. Extra arguments go to examples/build_cache.py,
# for example `--set run.precision=fp16`.
#SBATCH --job-name=confhead-cache
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/confhead-cache-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
CONFIG="${CONFIG:-configs/nuscenes-mini.toml}"

uv run examples/build_cache.py --config "${CONFIG}" \
    --shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}" \
    --set run.device=cuda \
    --set run.num_workers=$((SLURM_CPUS_PER_TASK - 1)) \
    "$@"
