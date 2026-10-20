#!/bin/bash
# Submit the cache build on CARC: one GPU task per shard, then one CPU job that
# checks the cache when every task has ended. Submit it again to finish what a
# task left: built entries are skipped. From the confhead folder:
#
#   examples/carc-submit.sh configs/nuscenes-trainval.toml [shards]

set -euo pipefail
CONFIG="$(realpath "${1:?usage: examples/carc-submit.sh <config> [shards]}")"
SHARDS="${2:-16}"
: "${SBATCH_ACCOUNT:?set it to your CARC project account first, for example: export SBATCH_ACCOUNT=<project>}"

cd "$(dirname "$0")/.."
mkdir -p logs
uv sync --quiet

build=$(sbatch --parsable --array="0-$((SHARDS - 1))" examples/carc-build.sbatch "${CONFIG}")
check=$(sbatch --parsable --dependency="afterany:${build}" examples/carc-check.sbatch "${CONFIG}")

echo "build job ${build}: ${SHARDS} GPU tasks, logs in logs/confhead-build-${build}_<task>.out"
echo "check job ${check}: starts when the build ends, log in logs/confhead-check-${check}.out"
