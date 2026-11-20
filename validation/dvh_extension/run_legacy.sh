#!/usr/bin/env bash
set -euo pipefail
export DVH_RUN_NAME="legacy-${1:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/benchmark_env.sh"
exec nice -n 10 "$RTPIPELINE_PYTHON" -B "$DVH_BENCH/legacy_batch.py" "${1:-0}" "${2:-1}" "${@:3}"
