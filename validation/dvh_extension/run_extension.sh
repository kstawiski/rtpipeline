#!/usr/bin/env bash
set -euo pipefail
export DVH_RUN_NAME="extension-${1:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/benchmark_env.sh"
exec nice -n 10 "$RTPIPELINE_PYTHON" -B "$DVH_BENCH/extension.py" \
  --design "$DVH_BENCH/DESIGN.json" --out "$DVH_BENCH/results/shard-${1:-0}" \
  --shard "${1:-0}" --shards "${2:-1}" "${@:3}"
