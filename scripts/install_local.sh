#!/usr/bin/env bash
# Install RTpipeline's supported local environments on macOS, Linux, or WSL2.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly PROJECT_DIR
readonly MAIN_ENV="rtpipeline"
readonly RADIOMICS_ENV="rtpipeline-radiomics"
readonly DEFAULT_MAMBA_ROOT="${HOME}/micromamba"

DEVICE="auto"
INSTALL_SYSTEM_PACKAGES=true
PLAN_ONLY=false
VERIFY_ONLY=false
MAMBA_ROOT="${RTPIPELINE_MAMBA_ROOT:-${DEFAULT_MAMBA_ROOT}}"
INSTALL_TEMP_DIR=""

usage() {
    cat <<'EOF'
Usage: bash scripts/install_local.sh [options]

Options:
  --device auto|cpu|gpu|mps  Select PyTorch backend (default: auto)
  --mamba-root PATH           Micromamba root prefix (default: ~/micromamba)
  --no-system-packages        Do not install missing curl/tar/bzip2 prerequisites
  --plan                      Print the detected installation plan and exit
  --verify-only               Verify existing environments without changing them
  -h, --help                  Show this help

The installer creates two named environments: rtpipeline (NumPy 2.x) and
rtpipeline-radiomics (NumPy 1.26 + PyRadiomics). Re-running it updates them.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

log() {
    printf '[rtpipeline] %s\n' "$*"
}

sha256_file() {
    local path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$path" | awk '{print $1}'
    else
        shasum -a 256 "$path" | awk '{print $1}'
    fi
}

ensure_install_temp_dir() {
    if [[ -z "$INSTALL_TEMP_DIR" ]]; then
        INSTALL_TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/rtpipeline-install.XXXXXX")"
        trap '[[ -n "${INSTALL_TEMP_DIR:-}" && -d "$INSTALL_TEMP_DIR" ]] && rm -rf -- "$INSTALL_TEMP_DIR"' EXIT
    fi
}

while (($#)); do
    case "$1" in
        --device)
            (($# >= 2)) || die "--device requires a value"
            DEVICE="$2"
            shift 2
            ;;
        --mamba-root)
            (($# >= 2)) || die "--mamba-root requires a path"
            MAMBA_ROOT="$2"
            shift 2
            ;;
        --no-system-packages)
            INSTALL_SYSTEM_PACKAGES=false
            shift
            ;;
        --plan)
            PLAN_ONLY=true
            shift
            ;;
        --verify-only)
            VERIFY_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

case "$DEVICE" in
    auto|cpu|gpu|mps) ;;
    *) die "--device must be auto, cpu, gpu, or mps" ;;
esac

OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
IS_WSL=false
if [[ "$OS_NAME" == "Linux" ]] && { [[ -n "${WSL_DISTRO_NAME:-}" ]] || grep -qi microsoft /proc/version 2>/dev/null; }; then
    IS_WSL=true
fi

case "$OS_NAME" in
    Darwin|Linux) ;;
    *) die "supported platforms are macOS, Linux, and Linux inside WSL2 (found ${OS_NAME})" ;;
esac

find_nvidia_smi() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        command -v nvidia-smi
    elif [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
        printf '%s\n' /usr/lib/wsl/lib/nvidia-smi
    else
        return 1
    fi
}

if [[ "$DEVICE" == "auto" ]]; then
    if [[ "$OS_NAME" == "Darwin" && "$ARCH_NAME" == "arm64" ]]; then
        DEVICE="mps"
    elif find_nvidia_smi >/dev/null; then
        DEVICE="gpu"
    else
        DEVICE="cpu"
    fi
fi

if [[ "$DEVICE" == "mps" && "$OS_NAME" != "Darwin" ]]; then
    die "the MPS backend is available only on macOS"
fi
if [[ "$DEVICE" == "gpu" && "$OS_NAME" != "Linux" ]]; then
    die "the CUDA backend is supported by this installer only on Linux/WSL2"
fi
if [[ "$DEVICE" == "gpu" ]] && ! find_nvidia_smi >/dev/null; then
    die "--device gpu requested, but nvidia-smi is not available"
fi

if [[ "$IS_WSL" == true ]]; then
    log "platform: ${OS_NAME}/${ARCH_NAME} (WSL2)"
else
    log "platform: ${OS_NAME}/${ARCH_NAME}"
fi
log "selected compute backend: ${DEVICE}"
log "micromamba root: ${MAMBA_ROOT}"

if [[ "$PLAN_ONLY" == true ]]; then
    log "plan: install prerequisites, micromamba, ${MAIN_ENV}, and ${RADIOMICS_ENV}; then run import/GPU checks"
    exit 0
fi

install_linux_prerequisites() {
    local missing=()
    local command_name
    for command_name in curl tar bzip2; do
        command -v "$command_name" >/dev/null 2>&1 || missing+=("$command_name")
    done
    ((${#missing[@]} == 0)) && return 0
    [[ "$INSTALL_SYSTEM_PACKAGES" == true ]] || die "missing prerequisites: ${missing[*]}"

    local elevate=()
    if [[ "$(id -u)" -ne 0 ]]; then
        command -v sudo >/dev/null 2>&1 || die "missing prerequisites (${missing[*]}) and sudo is unavailable"
        elevate=(sudo)
    fi
    if command -v apt-get >/dev/null 2>&1; then
        "${elevate[@]}" apt-get update
        "${elevate[@]}" apt-get install -y ca-certificates curl tar bzip2
    elif command -v dnf >/dev/null 2>&1; then
        "${elevate[@]}" dnf install -y ca-certificates curl tar bzip2
    elif command -v yum >/dev/null 2>&1; then
        "${elevate[@]}" yum install -y ca-certificates curl tar bzip2
    elif command -v zypper >/dev/null 2>&1; then
        "${elevate[@]}" zypper --non-interactive install ca-certificates curl tar bzip2
    elif command -v pacman >/dev/null 2>&1; then
        "${elevate[@]}" pacman -Sy --needed --noconfirm ca-certificates curl tar bzip2
    else
        die "no supported package manager found; install curl, tar, and bzip2, then rerun"
    fi
}

if [[ "$VERIFY_ONLY" != true ]]; then
    if [[ "$OS_NAME" == "Linux" ]]; then
        install_linux_prerequisites
    else
        for command_name in curl tar bzip2; do
            command -v "$command_name" >/dev/null 2>&1 || die "macOS prerequisite missing: ${command_name}"
        done
    fi
fi

MAMBA_BIN="${RTPIPELINE_MICROMAMBA:-}"
if [[ -z "$MAMBA_BIN" ]]; then
    MAMBA_BIN="$(command -v micromamba || true)"
fi
if [[ -z "$MAMBA_BIN" ]]; then
    MAMBA_BIN="${HOME}/.local/bin/micromamba"
fi

if [[ ! -x "$MAMBA_BIN" ]]; then
    [[ "$VERIFY_ONLY" != true ]] || die "micromamba is not installed"
    case "${OS_NAME}/${ARCH_NAME}" in
        Linux/x86_64) mamba_platform="linux-64" ;;
        Linux/aarch64|Linux/arm64) mamba_platform="linux-aarch64" ;;
        Darwin/x86_64) mamba_platform="osx-64" ;;
        Darwin/arm64) mamba_platform="osx-arm64" ;;
        *) die "no micromamba build mapping for ${OS_NAME}/${ARCH_NAME}" ;;
    esac
    log "installing micromamba from the official mamba-org endpoint"
    mkdir -p "$(dirname "$MAMBA_BIN")"
    ensure_install_temp_dir
    curl --fail --location --silent --show-error \
        "https://micro.mamba.pm/api/micromamba/${mamba_platform}/latest" \
        --output "${INSTALL_TEMP_DIR}/micromamba.tar.bz2"
    tar -xjf "${INSTALL_TEMP_DIR}/micromamba.tar.bz2" -C "$INSTALL_TEMP_DIR" bin/micromamba
    install -m 0755 "${INSTALL_TEMP_DIR}/bin/micromamba" "$MAMBA_BIN"
fi

export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"
MAMBA_BIN_DIR="$(dirname "$MAMBA_BIN")"
export PATH="${MAMBA_BIN_DIR}:${PATH}"

env_exists() {
    [[ -f "${MAMBA_ROOT}/envs/$1/conda-meta/history" ]]
}

sync_environment() {
    local env_name="$1"
    local env_file="$2"
    if env_exists "$env_name"; then
        # These are RTpipeline-managed environments. Pruning prevents stale pip/
        # conda packages from an older release from contaminating verification.
        "$MAMBA_BIN" env update --yes --name "$env_name" --file "$env_file" --prune
    else
        "$MAMBA_BIN" create --yes --name "$env_name" --file "$env_file"
    fi
}

if [[ "$VERIFY_ONLY" != true ]]; then
    log "creating/updating the main NumPy 2.x environment"
    sync_environment "$MAIN_ENV" "${PROJECT_DIR}/envs/rtpipeline-local.yaml"

    log "installing the pinned PyTorch backend"
    torch_specs=("pytorch=2.3.*" "torchvision=0.18.*" "torchaudio=2.3.*")
    case "$DEVICE" in
        gpu)
            "$MAMBA_BIN" install --yes --name "$MAIN_ENV" --channel pytorch --channel nvidia \
                --channel conda-forge "${torch_specs[@]}" "pytorch-cuda=12.1"
            ;;
        cpu)
            if [[ "$OS_NAME" == "Linux" ]]; then
                "$MAMBA_BIN" install --yes --name "$MAIN_ENV" --channel pytorch \
                    --channel conda-forge "${torch_specs[@]}" cpuonly
            else
                "$MAMBA_BIN" install --yes --name "$MAIN_ENV" --channel pytorch \
                    --channel conda-forge "${torch_specs[@]}"
            fi
            ;;
        mps)
            "$MAMBA_BIN" install --yes --name "$MAIN_ENV" --channel pytorch \
                --channel conda-forge "${torch_specs[@]}"
            ;;
    esac

    "$MAMBA_BIN" run --name "$MAIN_ENV" python -m pip install --upgrade pip
    "$MAMBA_BIN" run --name "$MAIN_ENV" python -m pip install --editable "$PROJECT_DIR"

    log "creating/updating the isolated NumPy 1.26 radiomics environment"
    sync_environment "$RADIOMICS_ENV" "${PROJECT_DIR}/envs/rtpipeline-radiomics-local.yaml"
    if [[ "$OS_NAME" == "Linux" && "$ARCH_NAME" == "x86_64" ]]; then
        radiomics_source="${PROJECT_DIR}/third_party/wheels/pyradiomics-3.0.1-cp310-cp310-linux_x86_64.whl"
    else
        ensure_install_temp_dir
        radiomics_source="${INSTALL_TEMP_DIR}/pyradiomics-v3.0.1.tar.gz"
        curl --fail --location --silent --show-error \
            "https://github.com/AIM-Harvard/pyradiomics/archive/refs/tags/v3.0.1.tar.gz" \
            --output "$radiomics_source"
    fi
    "$MAMBA_BIN" run --name "$RADIOMICS_ENV" python -m pip install \
        --no-deps --no-build-isolation "$radiomics_source"
    # The radiomics helper intentionally does not install RTpipeline as a Python
    # distribution. Its package metadata declares the full NumPy 2.x/main
    # runtime (including TotalSegmentator), which is incompatible with this
    # isolated NumPy 1.26 environment and would make `pip check` fail. Local
    # launchers execute the checked-out source with PROJECT_DIR on PYTHONPATH.
    "$MAMBA_BIN" run --name "$RADIOMICS_ENV" python -m pip uninstall \
        --yes rtpipeline
fi

log "verifying both environments"
"$MAMBA_BIN" run --name "$MAIN_ENV" python -c \
    "import numpy, pydicom, SimpleITK, torch, rtpipeline; assert numpy.__version__.split('.')[0] == '2'; print('main', rtpipeline.__version__, numpy.__version__, torch.__version__)"
env PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "$MAMBA_BIN" run --name "$RADIOMICS_ENV" python -c \
    "import numpy, radiomics; assert numpy.__version__.split('.')[0] == '1'; print('radiomics', radiomics.__version__, numpy.__version__)"
env PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "$MAMBA_BIN" run --name "$RADIOMICS_ENV" python -m rtpipeline.cli --help >/dev/null
"$MAMBA_BIN" run --name "$MAIN_ENV" python -m pip check
"$MAMBA_BIN" run --name "$RADIOMICS_ENV" python -m pip check
"$MAMBA_BIN" run --name "$MAIN_ENV" snakemake --version

case "$DEVICE" in
    gpu)
        "$MAMBA_BIN" run --name "$MAIN_ENV" python -c \
            "import torch; assert torch.cuda.is_available(), 'CUDA is not available to PyTorch'; print(torch.cuda.get_device_name(0))"
        ;;
    mps)
        "$MAMBA_BIN" run --name "$MAIN_ENV" python -c \
            "import torch; assert torch.backends.mps.is_available(), 'MPS is not available to PyTorch'; print('MPS available')"
        ;;
esac

receipt_dir="${XDG_STATE_HOME:-${HOME}/.local/state}/rtpipeline"
mkdir -p "$receipt_dir"
receipt_path="${receipt_dir}/install-receipt-v2.2.1.txt"
{
    printf 'RTpipeline local installation receipt\n'
    printf 'created_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    printf 'platform=%s/%s\n' "$OS_NAME" "$ARCH_NAME"
    printf 'wsl2=%s\n' "$IS_WSL"
    printf 'device=%s\n' "$DEVICE"
    printf 'project=%s\n' "$PROJECT_DIR"
    printf 'micromamba_version=%s\n' "$("$MAMBA_BIN" --version)"
    if command -v git >/dev/null 2>&1 && git -C "$PROJECT_DIR" rev-parse HEAD >/dev/null 2>&1; then
        printf 'git_commit=%s\n' "$(git -C "$PROJECT_DIR" rev-parse HEAD)"
        printf 'git_dirty=%s\n' "$(if [[ -n "$(git -C "$PROJECT_DIR" status --porcelain)" ]]; then printf true; else printf false; fi)"
    fi
    printf '\n[rtpipeline explicit environment]\n'
    "$MAMBA_BIN" list --name "$MAIN_ENV" --explicit
    printf '\n[rtpipeline pip freeze]\n'
    "$MAMBA_BIN" run --name "$MAIN_ENV" python -m pip freeze --all
    printf '\n[rtpipeline pip inspect]\n'
    "$MAMBA_BIN" run --name "$MAIN_ENV" python -m pip inspect --local
    printf '\n[rtpipeline-radiomics explicit environment]\n'
    "$MAMBA_BIN" list --name "$RADIOMICS_ENV" --explicit
    printf '\n[rtpipeline-radiomics pip freeze]\n'
    "$MAMBA_BIN" run --name "$RADIOMICS_ENV" python -m pip freeze --all
    printf '\n[rtpipeline-radiomics pip inspect]\n'
    "$MAMBA_BIN" run --name "$RADIOMICS_ENV" python -m pip inspect --local
    if [[ -f "${radiomics_source:-}" ]]; then
        printf '\npyradiomics_source=%s\n' "$radiomics_source"
        printf 'pyradiomics_source_sha256=%s\n' "$(sha256_file "$radiomics_source")"
    fi
} > "$receipt_path"

log "installation verified"
log "receipt: ${receipt_path}"
log "next: bash scripts/run_local.sh --config /absolute/path/to/config.yaml --cores all --dry-run"
