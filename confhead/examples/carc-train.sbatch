#!/bin/bash
# Train the confidence head on one CARC GPU. From the confhead folder:
#
#   sbatch --time=01:00:00 examples/carc-train.sbatch configs/train-trainval.toml --estimate
#   sbatch examples/carc-train.sbatch configs/train-trainval.toml
#
# Arguments after the config go to examples/train_head.py. A run folder that
# holds last.pt continues after its last epoch, so submit the same command again
# to finish a run that reached its time limit.
#SBATCH --job-name=confhead-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/confhead-train-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
CONFIG="${1:?usage: sbatch examples/carc-train.sbatch <config> [train_head.py arguments]}"
shift

uv run examples/train_head.py --config "${CONFIG}" "$@"
