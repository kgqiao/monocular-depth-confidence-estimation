#!/bin/bash
# Back up a cache to Hugging Face, on CPU. A job that stops early sends only the
# shards that are not up yet when you submit it again. From the confhead folder:
#
#   sbatch examples/carc-upload.sbatch /scratch1/$USER/confhead-cache/nuscenes-trainval
#
# The repository is in configs/huggingface.toml. Log in once on the login node
# first, with `uv run hf auth login`. Arguments after the cache go to
# `confhead cache upload`, for example `--config <file>`.
#SBATCH --job-name=confhead-upload
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/confhead-upload-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
CACHE="${1:?usage: sbatch examples/carc-upload.sbatch <cache folder> [confhead cache upload arguments]}"
shift

uv run confhead cache upload "${CACHE}" "$@"
