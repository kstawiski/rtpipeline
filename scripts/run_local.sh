#!/usr/bin/env bash
# Run the Snakemake workflow through the local dual-environment installation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly PROJECT_DIR

CONFIG_FILE=""
CORES="all"
DRY_RUN=false
TARGETS=()
EXTRA_ARGS=()
MAMBA_ROOT="${RTPIPELINE_MAMBA_ROOT:-${HOME}/micromamba}"

usage() {
    cat <<'EOF'
Usage: bash scripts/run_local.sh --config PATH [options] [TARGET ...]

Options:
  --config PATH       YAML configuration file (required)
  --cores N|all       Snakemake cores (default: all)
  --dry-run           Resolve the complete DAG without running jobs
  --mamba-root PATH   Micromamba root prefix (default: ~/micromamba)
  --                  Pass remaining arguments directly to Snakemake
  -h, --help          Show this help

Examples:
  bash scripts/run_local.sh --config config.local.yaml --cores 8 --dry-run
  bash scripts/run_local.sh --config config.local.yaml --cores 8
  bash scripts/run_local.sh --config config.local.yaml --cores 8 radiomics_ct
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

while (($#)); do
    case "$1" in
        --config)
            (($# >= 2)) || die "--config requires a path"
            CONFIG_FILE="$2"
            shift 2
            ;;
        --cores)
            (($# >= 2)) || die "--cores requires a value"
            CORES="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --mamba-root)
            (($# >= 2)) || die "--mamba-root requires a path"
            MAMBA_ROOT="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            die "unknown option: $1 (put Snakemake options after --)"
            ;;
        *)
            TARGETS+=("$1")
            shift
            ;;
    esac
done

[[ -n "$CONFIG_FILE" ]] || die "--config is required"
[[ -f "$CONFIG_FILE" ]] || die "configuration file not found: ${CONFIG_FILE}"
CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"

MAMBA_BIN="${RTPIPELINE_MICROMAMBA:-$(command -v micromamba || true)}"
if [[ -z "$MAMBA_BIN" ]]; then
    MAMBA_BIN="${HOME}/.local/bin/micromamba"
fi
[[ -x "$MAMBA_BIN" ]] || die "micromamba not found; run bash scripts/install_local.sh"

export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"
MAMBA_BIN_DIR="$(dirname "$MAMBA_BIN")"
export PATH="${MAMBA_BIN_DIR}:${PATH}"
export RTPIPELINE_CONDA_EXE="$MAMBA_BIN"
export RTPIPELINE_RADIOMICS_ENV="rtpipeline-radiomics"

command=(
    "$MAMBA_BIN" run --name rtpipeline
    snakemake
    --snakefile "${PROJECT_DIR}/Snakefile"
    --directory "$PROJECT_DIR"
    --configfile "$CONFIG_FILE"
    --cores "$CORES"
    --rerun-incomplete
    --printshellcmds
)
if [[ "$DRY_RUN" == true ]]; then
    command+=(--dry-run)
fi
if ((${#TARGETS[@]})); then
    command+=("${TARGETS[@]}")
fi
if ((${#EXTRA_ARGS[@]})); then
    command+=("${EXTRA_ARGS[@]}")
fi

printf '[rtpipeline] running:'
printf ' %q' "${command[@]}"
printf '\n'
exec "${command[@]}"
