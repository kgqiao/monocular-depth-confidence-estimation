#!/bin/bash
# Download and extract the full nuScenes v1.0 on CARC. From the utilities/dataset folder:
#
#   mkdir -p logs
#   sbatch --account=<project> download_nuscenes.sbatch
#
# The nuScenes account is read from ~/.config/nuscenes.env (chmod 600):
#
#   NUSCENES_EMAIL=you@usc.edu
#   NUSCENES_PASSWORD=...
#
# The data lands in $NUSCENES_DATAROOT, or /scratch1/$USER/nuscenes when it is
# not set. A job that stops early continues where it was when you submit it
# again. Extra arguments go to download_nuscenes.py, for example
# `--archives v1.0-trainval_meta.tgz v1.0-trainval01_blobs.tgz` or `--check`.
#SBATCH --job-name=nuscenes-download
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=logs/nuscenes-download-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

uv run download_nuscenes.py --dest "${NUSCENES_DATAROOT:-/scratch1/${USER}/nuscenes}" "$@"
