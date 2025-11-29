#!/usr/bin/env bash
# RTpipeline Docker project setup (interactive)
# Generates: config.yaml, docker-compose.project.yml, run_docker.sh, monitor.sh, README.md

set -euo pipefail

VERSION="3.0.0"

# Ensure we have an interactive stdin (handles curl | bash)
if [ ! -t 0 ]; then
    if [ -r /dev/tty ]; then
        exec < /dev/tty
    else
        echo "Interactive TTY is required. Run: bash setup_docker_project.sh" >&2
        exit 1
    fi
fi

# ---------------------------
# Helpers
# ---------------------------

abort() {
    echo "Error: $*" >&2
    exit 1
}

abs_path() {
    local p="$1"
    if [ -z "$p" ]; then
        echo ""
        return
    fi
    if command -v python3 >/dev/null 2>&1; then
        python3 - "$p" <<'PY'
import os, sys
p = sys.argv[1]
print(os.path.abspath(os.path.expanduser(p)))
PY
    else
        case "$p" in
            /*) echo "$p" ;;
            *) echo "$(pwd)/${p#./}" ;;
        esac
    fi
}

prompt_input() {
    local question="$1"
    local default="${2:-}"
    local ans
    if [ -n "$default" ]; then
        read -r -p "$question [$default]: " ans
        if [ -z "$ans" ]; then ans="$default"; fi
    else
        read -r -p "$question: " ans
    fi
    echo "$ans"
}

prompt_yes_no() {
    local question="$1"
    local default="${2:-yes}"
    local prompt
    if [ "$default" = "yes" ]; then
        prompt="[Y/n]"
    else
        prompt="[y/N]"
    fi
    local ans
    read -r -p "$question $prompt: " ans
    if [ -z "$ans" ]; then ans="$default"; fi
    case "$ans" in
        [Yy]* ) echo "yes" ;;
        [Nn]* ) echo "no" ;;
        * ) echo "$default" ;;
    esac
}

cpu_count() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    elif command -v sysctl >/dev/null 2>&1; then
        sysctl -n hw.ncpu 2>/dev/null || echo 8
    else
        echo 8
    fi
}

gpu_count() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo 0
        return
    fi
    local out
    if ! out="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"; then
        echo 0
        return
    fi
    if [ -z "$out" ]; then
        echo 0
    else
        printf '%s\n' "$out" | wc -l | tr -d '[:space:]'
    fi
}

clamp_int() {
    local value="$1"
    local min_val="$2"
    local max_val="$3"
    if [ "$value" -lt "$min_val" ]; then
        echo "$min_val"
    elif [ "$value" -gt "$max_val" ]; then
        echo "$max_val"
    else
        echo "$value"
    fi
}

detect_compose() {
    if docker compose version >/dev/null 2>&1; then
        echo "docker compose"
    elif command -v docker-compose >/dev/null 2>&1; then
        echo "docker-compose"
    else
        echo ""
    fi
}

# ---------------------------
# Step 0: preflight
# ---------------------------

echo "RTpipeline Docker Project Setup (v${VERSION})"
echo "This wizard will:"
echo "  - Check Docker + Compose"
echo "  - Ask for your DICOM folder and a project folder"
echo "  - Let you reuse an existing config or build a new one"
echo "  - Generate: config.yaml, docker-compose.project.yml, run_docker.sh, monitor.sh, README.md"
echo ""
echo "Press Enter to accept defaults. Ctrl+C to cancel at any time."
echo ""

command -v docker >/dev/null 2>&1 || abort "Docker not found. Install https://docs.docker.com/get-docker/"
if ! docker info >/dev/null 2>&1; then
    abort "Docker daemon not running or permission denied."
fi
COMPOSE_BIN="$(detect_compose)"
[ -n "$COMPOSE_BIN" ] || abort "Docker Compose not found. Install https://docs.docker.com/compose/install/"

echo ":: Environment Check ::"
echo "  - Docker: $(docker --version | awk '{print $3}' | sed 's/,//')"
echo "  - Compose: ${COMPOSE_BIN}"

GPU_AVAILABLE="no"
GPU_COUNT=0
GPU_DEVICE="cpu"
USE_GPU_RUNTIME="no"
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_AVAILABLE="yes"
    GPU_DEVICE="gpu"
    USE_GPU_RUNTIME="yes"
    GPU_COUNT="$(gpu_count)"
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
    echo "  - GPU detected: ${GPU_NAME:-NVIDIA GPU} (x${GPU_COUNT:-1})"
else
    echo "  - GPU: not detected (CPU mode)"
fi

if ! docker image inspect kstawiski/rtpipeline:latest >/dev/null 2>&1; then
    echo "  - rtpipeline image: not present locally (will pull on first run)"
else
    echo "  - rtpipeline image: found locally"
fi
echo ""

# ---------------------------
# Step 1: DICOM path
# ---------------------------

while true; do
    DICOM_DIR_RAW="$(prompt_input 'Step 1/5 - DICOM root path' '')"
    DICOM_DIR_RAW="$(printf '%s' "$DICOM_DIR_RAW" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [ -n "$DICOM_DIR_RAW" ] || { echo "  Path cannot be empty."; continue; }
    DICOM_DIR="$(abs_path "$DICOM_DIR_RAW")"
    if [ -d "$DICOM_DIR" ]; then
        echo "  Using DICOM directory: $DICOM_DIR"
        break
    fi
    echo "  Path not found: $DICOM_DIR"
    CHOICE="$(prompt_input '    Enter [r]e-enter, [c]reate with mkdir -p, or [q]uit' 'r')"
    case "$CHOICE" in
        c|C)
            mkdir -p "$DICOM_DIR" || abort "Could not create $DICOM_DIR"
            echo "  Created $DICOM_DIR"
            break
            ;;
        q|Q)
            abort "DICOM directory is required."
            ;;
        *)
            ;;
    esac
done

# ---------------------------
# Step 2: Project directory
# ---------------------------

PARENT_DIR="$(dirname "$DICOM_DIR")"
PROJECT_DEFAULT="${PARENT_DIR}/$(basename "$DICOM_DIR")_rtpipeline"
PROJECT_DIR_RAW="$(prompt_input 'Step 2/5 - Project directory (will hold config/output/logs)' "$PROJECT_DEFAULT")"
PROJECT_DIR="$(abs_path "$PROJECT_DIR_RAW")"
mkdir -p "$PROJECT_DIR" "$PROJECT_DIR/Output" "$PROJECT_DIR/Logs" "$PROJECT_DIR/.snakemake_cache"
echo "  Project directory: $PROJECT_DIR"

# ---------------------------
# Step 3: Config source
# ---------------------------

CONFIG_MODE="new"
if [ "$(prompt_yes_no 'Step 3/5 - Use existing config.yaml?' 'no')" = "yes" ]; then
    EXISTING_CONFIG_RAW="$(prompt_input 'Path to existing config.yaml' '')"
    EXISTING_CONFIG="$(abs_path "$EXISTING_CONFIG_RAW")"
    [ -f "$EXISTING_CONFIG" ] || abort "Config not found: $EXISTING_CONFIG"
    CONFIG_MODE="existing"
    echo "  Will copy config from: $EXISTING_CONFIG"
fi

# ---------------------------
# Step 4: Feature selection (only if new config)
# ---------------------------

ENABLE_RADIOMICS="yes"
ENABLE_ROBUSTNESS="no"
ROBUSTNESS_INTENSITY="standard"
ENABLE_CT_CROPPING="yes"
CROPPING_REGION="pelvis"
ENABLE_CUSTOM_MODELS="no"

if [ "$CONFIG_MODE" = "new" ]; then
    if [ "$(prompt_yes_no 'Step 4/5 - Enable radiomics?' 'yes')" = "no" ]; then
        ENABLE_RADIOMICS="no"
    fi
    if [ "$ENABLE_RADIOMICS" = "yes" ]; then
        if [ "$(prompt_yes_no 'Enable radiomics robustness (perturbations)?' 'no')" = "yes" ]; then
            ENABLE_ROBUSTNESS="yes"
            INT_CHOICE="$(prompt_input 'Robustness intensity (mild/standard/aggressive)' 'standard')"
            case "$INT_CHOICE" in
                mild|standard|aggressive) ROBUSTNESS_INTENSITY="$INT_CHOICE" ;;
                *) ROBUSTNESS_INTENSITY="standard" ;;
            esac
        fi
    fi
    if [ "$(prompt_yes_no 'Enable CT cropping?' 'yes')" = "no" ]; then
        ENABLE_CT_CROPPING="no"
    else
        CROPPING_REGION="$(prompt_input 'Cropping region (pelvis/thorax/abdomen/head_neck/brain)' "$CROPPING_REGION")"
    fi
    if [ "$(prompt_yes_no 'Enable custom nnU-Net models?' 'no')" = "yes" ]; then
        ENABLE_CUSTOM_MODELS="yes"
    fi
fi

# ---------------------------
# Step 5: Resources & profiles
# ---------------------------

CPU_COUNT="$(cpu_count)"

set_profile_custom() {
    PROFILE_ID="custom"
    PROFILE_LABEL="Custom"
    PROFILE_DESC="Manual tuning (same defaults as upstream config)."
    PROFILE_USE_GPU="inherit"
    PROFILE_RESERVED_CORES=1
    PROFILE_PARALLEL_COURSES=1
    PROFILE_DVH_THREADS="null"
    PROFILE_RAD_THREADS="null"
    PROFILE_QC_THREADS="null"
    PROFILE_CROP_THREADS="null"
    PROFILE_ROB_THREADS="null"
    PROFILE_SEG_DEVICE_OVERRIDE=""
    PROFILE_SEG_MAX_WORKERS="null"
    PROFILE_SEG_RESAMPLE_THREADS=1
    PROFILE_SEG_SAVE_THREADS=1
    PROFILE_RADIOMICS_SEQUENTIAL="false"
    PROFILE_AGG_THREADS="auto"
    local cpu_default="$CPU_COUNT"
    if [ "$cpu_default" -gt 1 ]; then
        cpu_default=$((cpu_default - 1))
    fi
    PROFILE_MAX_CORES_DEFAULT="$cpu_default"
    PROFILE_MAX_MEMORY_DEFAULT="32G"
    PROFILE_MAX_WORKERS_DEFAULT="null"
}

set_profile_single_gpu() {
    PROFILE_ID="single_gpu"
    PROFILE_LABEL="Single GPU workstation"
    PROFILE_DESC="Serialize segmentation on one GPU and overlap CPU-bound stages."
    PROFILE_USE_GPU="yes"
    PROFILE_RESERVED_CORES=2
    PROFILE_PARALLEL_COURSES=2
    PROFILE_DVH_THREADS="null"
    PROFILE_RAD_THREADS="null"
    PROFILE_QC_THREADS="null"
    PROFILE_CROP_THREADS="null"
    PROFILE_ROB_THREADS="null"
    PROFILE_SEG_DEVICE_OVERRIDE="gpu"
    PROFILE_SEG_MAX_WORKERS="null"
    PROFILE_SEG_RESAMPLE_THREADS="$(clamp_int $((CPU_COUNT / 2)) 2 8)"
    PROFILE_SEG_SAVE_THREADS="$(clamp_int $((CPU_COUNT / 4)) 1 6)"
    PROFILE_RADIOMICS_SEQUENTIAL="false"
    PROFILE_AGG_THREADS="auto"
    PROFILE_MAX_CORES_DEFAULT="$CPU_COUNT"
    PROFILE_MAX_MEMORY_DEFAULT="64G"
    PROFILE_MAX_WORKERS_DEFAULT="null"
}

set_profile_multi_gpu() {
    PROFILE_ID="multi_gpu"
    PROFILE_LABEL="Multi-GPU server"
    PROFILE_DESC="Parallelize segmentation across GPUs and stagger multiple CPU courses."
    PROFILE_USE_GPU="yes"
    PROFILE_RESERVED_CORES=2
    PROFILE_PARALLEL_COURSES=4
    PROFILE_DVH_THREADS=8
    PROFILE_RAD_THREADS=10
    PROFILE_QC_THREADS=4
    PROFILE_CROP_THREADS=4
    PROFILE_ROB_THREADS=8
    PROFILE_SEG_DEVICE_OVERRIDE="gpu"
    local seg_workers="$GPU_COUNT"
    if [ -z "$seg_workers" ] || [ "$seg_workers" -lt 1 ]; then
        seg_workers=1
    fi
    PROFILE_SEG_MAX_WORKERS="$seg_workers"
    PROFILE_SEG_RESAMPLE_THREADS="$(clamp_int $((CPU_COUNT / 2)) 4 16)"
    PROFILE_SEG_SAVE_THREADS="$(clamp_int $((CPU_COUNT / 4)) 2 12)"
    PROFILE_RADIOMICS_SEQUENTIAL="false"
    PROFILE_AGG_THREADS="auto"
    PROFILE_MAX_CORES_DEFAULT="$CPU_COUNT"
    PROFILE_MAX_MEMORY_DEFAULT="128G"
    local mw=$((CPU_COUNT - 16))
    if [ "$mw" -lt 1 ]; then
        PROFILE_MAX_WORKERS_DEFAULT="null"
    else
        PROFILE_MAX_WORKERS_DEFAULT="$mw"
    fi
}

set_profile_cpu_only() {
    PROFILE_ID="cpu_only"
    PROFILE_LABEL="CPU-only / constrained node"
    PROFILE_DESC="Disable GPU usage and keep segmentation fan-out conservative."
    PROFILE_USE_GPU="no"
    PROFILE_RESERVED_CORES=1
    PROFILE_PARALLEL_COURSES=1
    PROFILE_DVH_THREADS=4
    PROFILE_RAD_THREADS=4
    PROFILE_QC_THREADS=2
    PROFILE_CROP_THREADS=2
    PROFILE_ROB_THREADS=4
    PROFILE_SEG_DEVICE_OVERRIDE="cpu"
    local seg_workers=$((CPU_COUNT / 4))
    if [ "$seg_workers" -lt 1 ]; then
        seg_workers=1
    fi
    if [ "$seg_workers" -gt 4 ]; then
        seg_workers=4
    fi
    PROFILE_SEG_MAX_WORKERS="$seg_workers"
    PROFILE_SEG_RESAMPLE_THREADS="$(clamp_int $((CPU_COUNT / 3)) 2 6)"
    PROFILE_SEG_SAVE_THREADS="$(clamp_int $((CPU_COUNT / 4)) 1 4)"
    PROFILE_RADIOMICS_SEQUENTIAL="true"
    PROFILE_AGG_THREADS="auto"
    local cores="$CPU_COUNT"
    if [ "$cores" -gt 1 ]; then
        cores=$((cores - 1))
    fi
    PROFILE_MAX_CORES_DEFAULT="$cores"
    PROFILE_MAX_MEMORY_DEFAULT="32G"
    local mw=$((CPU_COUNT - 2))
    if [ "$mw" -lt 2 ]; then
        mw=2
    fi
    if [ "$mw" -gt 8 ]; then
        mw=8
    fi
    PROFILE_MAX_WORKERS_DEFAULT="$mw"
}

echo ""
echo "Step 5/5 - Resource tuning & optimization profile"
echo "  - Detected CPU cores: $CPU_COUNT"
if [ "$GPU_AVAILABLE" = "yes" ]; then
    echo "  - Detected GPUs: ${GPU_COUNT:-1}"
fi

PROFILE_DEFAULT_ID="cpu_only"
if [ "$GPU_AVAILABLE" = "yes" ]; then
    if [ "$GPU_COUNT" -gt 1 ]; then
        PROFILE_DEFAULT_ID="multi_gpu"
    else
        PROFILE_DEFAULT_ID="single_gpu"
    fi
fi
if [ "$USE_GPU_RUNTIME" = "no" ]; then
    PROFILE_DEFAULT_ID="cpu_only"
fi

echo "Available profiles:"
if [ "$GPU_AVAILABLE" = "yes" ]; then
    echo "  [single_gpu] Single GPU workstation (serialize segmentation, overlap CPU stages)"
    if [ "$GPU_COUNT" -gt 1 ]; then
        echo "  [multi_gpu] Multi-GPU server (parallel segmentation across GPUs)"
    fi
fi
echo "  [cpu_only] CPU-only / low-RAM node"
echo "  [custom] Custom (manual tuning in config.yaml later)"

PROFILE_SELECTION_RAW="$(prompt_input 'Choose optimization profile' "$PROFILE_DEFAULT_ID")"
PROFILE_SELECTION="$(printf '%s' "$PROFILE_SELECTION_RAW" | tr '[:upper:]' '[:lower:]')"
case "$PROFILE_SELECTION" in
    1|single|single_gpu) SELECTED_PROFILE="single_gpu" ;;
    2|multi|multi_gpu) SELECTED_PROFILE="multi_gpu" ;;
    3|cpu|cpu_only) SELECTED_PROFILE="cpu_only" ;;
    4|custom|manual) SELECTED_PROFILE="custom" ;;
    *) SELECTED_PROFILE="$PROFILE_DEFAULT_ID" ;;
esac

if [ "$SELECTED_PROFILE" = "multi_gpu" ] && [ "$GPU_AVAILABLE" != "yes" ]; then
    echo "  Multi-GPU profile requested but no GPU detected; using CPU-only profile."
    SELECTED_PROFILE="cpu_only"
fi
if [ "$SELECTED_PROFILE" = "multi_gpu" ] && [ "$GPU_AVAILABLE" = "yes" ] && [ "${GPU_COUNT:-0}" -lt 2 ]; then
    echo "  Only ${GPU_COUNT:-0} GPU detected; using single GPU profile instead."
    SELECTED_PROFILE="single_gpu"
fi
if [ "$SELECTED_PROFILE" = "single_gpu" ] && [ "$GPU_AVAILABLE" != "yes" ]; then
    echo "  No GPU detected; using CPU-only profile."
    SELECTED_PROFILE="cpu_only"
fi

case "$SELECTED_PROFILE" in
    multi_gpu) set_profile_multi_gpu ;;
    single_gpu) set_profile_single_gpu ;;
    cpu_only) set_profile_cpu_only ;;
    *) set_profile_custom ;;
esac

if [ "$PROFILE_USE_GPU" = "yes" ]; then
    if [ "$GPU_AVAILABLE" = "yes" ]; then
        if [ "$USE_GPU_RUNTIME" = "no" ]; then
            echo "  Re-enabling GPU runtime to match selected profile."
        fi
        USE_GPU_RUNTIME="yes"
        GPU_DEVICE="gpu"
    else
        echo "  GPU profile selected but hardware unavailable; forcing CPU-only settings."
        set_profile_cpu_only
        USE_GPU_RUNTIME="no"
        GPU_DEVICE="cpu"
    fi
elif [ "$PROFILE_USE_GPU" = "no" ]; then
    USE_GPU_RUNTIME="no"
    GPU_DEVICE="cpu"
fi

echo "Selected profile: ${PROFILE_LABEL}"
echo "  ${PROFILE_DESC}"
echo "  Scheduler: reserved_cores=${PROFILE_RESERVED_CORES}, parallel_courses=${PROFILE_PARALLEL_COURSES}"

MAX_CORES_DEFAULT="${PROFILE_MAX_CORES_DEFAULT:-$CPU_COUNT}"
if [ -z "$MAX_CORES_DEFAULT" ]; then
    MAX_CORES_DEFAULT="$CPU_COUNT"
fi
MAX_MEMORY_DEFAULT="${PROFILE_MAX_MEMORY_DEFAULT:-32G}"
MAX_CORES="$(prompt_input 'Max CPU cores for container' "$MAX_CORES_DEFAULT")"
MAX_MEMORY="$(prompt_input 'Max memory for container (e.g., 64G)' "$MAX_MEMORY_DEFAULT")"

WORKERS_PROMPT_DEFAULT="${PROFILE_MAX_WORKERS_DEFAULT:-null}"
WORKERS_INPUT="$(prompt_input 'Max workers (null for auto or number)' "$WORKERS_PROMPT_DEFAULT")"
if [ "$WORKERS_INPUT" = "null" ] || [ "$WORKERS_INPUT" = "auto" ]; then
    MAX_WORKERS="null"
else
    MAX_WORKERS="$WORKERS_INPUT"
fi

PROFILE_LABEL="${PROFILE_LABEL:-Custom}"
PROFILE_DESC="${PROFILE_DESC:-Manual tuning}"
PROFILE_ID="${PROFILE_ID:-custom}"
PROFILE_SEG_DEVICE_OVERRIDE="${PROFILE_SEG_DEVICE_OVERRIDE:-}"

SEG_DEVICE_CONFIG="$GPU_DEVICE"
if [ -n "$PROFILE_SEG_DEVICE_OVERRIDE" ]; then
    SEG_DEVICE_CONFIG="$PROFILE_SEG_DEVICE_OVERRIDE"
fi
GPU_DEVICE="$SEG_DEVICE_CONFIG"

SCHED_RESERVED_CORES="${PROFILE_RESERVED_CORES:-1}"
SCHED_PARALLEL_COURSES="${PROFILE_PARALLEL_COURSES:-1}"
SCHED_DVH_THREADS="${PROFILE_DVH_THREADS:-null}"
SCHED_RAD_THREADS="${PROFILE_RAD_THREADS:-null}"
SCHED_QC_THREADS="${PROFILE_QC_THREADS:-null}"
SCHED_CROP_THREADS="${PROFILE_CROP_THREADS:-null}"
SCHED_ROB_THREADS="${PROFILE_ROB_THREADS:-null}"
SEG_MAX_WORKERS_CONFIG="${PROFILE_SEG_MAX_WORKERS:-null}"
SEG_NR_THREADS_RESAMPLE="${PROFILE_SEG_RESAMPLE_THREADS:-1}"
SEG_NR_THREADS_SAVE="${PROFILE_SEG_SAVE_THREADS:-1}"
RAD_SEQUENTIAL_VALUE="${PROFILE_RADIOMICS_SEQUENTIAL:-false}"
AGG_THREADS_VALUE="${PROFILE_AGG_THREADS:-auto}"

# Optional weights mount
TOTALSEG_WEIGHTS_DIR=""
if [ "$(prompt_yes_no 'Optional: mount TotalSegmentator weights dir for caching?' 'no')" = "yes" ]; then
    TOTALSEG_WEIGHTS_DIR="$PROJECT_DIR/totalseg_weights"
    mkdir -p "$TOTALSEG_WEIGHTS_DIR"
    echo "  Weights cache: $TOTALSEG_WEIGHTS_DIR"
fi

# ---------------------------
# Generate config.yaml
# ---------------------------

CONFIG_PATH="$PROJECT_DIR/config.yaml"

if [ "$CONFIG_MODE" = "existing" ]; then
    cp "$EXISTING_CONFIG" "$CONFIG_PATH"
    # best-effort path remap
    tmpcfg="$(mktemp)"
    sed -e 's|^dicom_root:.*|dicom_root: "/data/input"|' \
        -e 's|^output_dir:.*|output_dir: "/data/output"|' \
        -e 's|^logs_dir:.*|logs_dir: "/data/logs"|' "$CONFIG_PATH" > "$tmpcfg" || true
    mv "$tmpcfg" "$CONFIG_PATH"
    echo "Created: $CONFIG_PATH (copied existing config, remapped paths)"
else
    rad_bool=$([ "$ENABLE_RADIOMICS" = "yes" ] && echo true || echo false)
    rob_bool=$([ "$ENABLE_ROBUSTNESS" = "yes" ] && echo true || echo false)
    crop_bool=$([ "$ENABLE_CT_CROPPING" = "yes" ] && echo true || echo false)
    custom_bool=$([ "$ENABLE_CUSTOM_MODELS" = "yes" ] && echo true || echo false)

    cat > "$CONFIG_PATH" <<EOF
# RTpipeline Docker Project Configuration
# Generated by setup_docker_project.sh v${VERSION} on $(date)
# Optimization profile: ${PROFILE_LABEL}
# Notes: ${PROFILE_DESC}

dicom_root: "/data/input"
output_dir: "/data/output"
logs_dir: "/data/logs"

max_workers: ${MAX_WORKERS}  # null = auto

scheduler:
  reserved_cores: ${SCHED_RESERVED_CORES}
  parallel_courses: ${SCHED_PARALLEL_COURSES}
  dvh_threads_per_job: ${SCHED_DVH_THREADS}
  radiomics_threads_per_job: ${SCHED_RAD_THREADS}
  qc_threads_per_job: ${SCHED_QC_THREADS}
  crop_ct_threads_per_job: ${SCHED_CROP_THREADS}
  robustness_threads_per_job: ${SCHED_ROB_THREADS}
  prioritize_short_courses: true

segmentation:
  device: "${SEG_DEVICE_CONFIG}"  # gpu or cpu
  fast: false
  force: false
  roi_subset: null
  extra_models: []
  max_workers: ${SEG_MAX_WORKERS_CONFIG}
  force_split: true
  nr_threads_resample: ${SEG_NR_THREADS_RESAMPLE}
  nr_threads_save: ${SEG_NR_THREADS_SAVE}
  num_proc_preprocessing: 1
  num_proc_export: 1

custom_models:
  enabled: ${custom_bool}
  root: "custom_models"
  models: []
  max_workers: null
  force: false
  retain_weights: true

radiomics:
  enabled: ${rad_bool}
  sequential: ${RAD_SEQUENTIAL_VALUE}
  params_file: "rtpipeline/radiomics_params.yaml"
  mr_params_file: "rtpipeline/radiomics_params_mr.yaml"
  skip_rois:
    - body
    - couchsurface
    - couchinterior
    - couchexterior
  max_voxels: 1500000000
  min_voxels: 10

radiomics_robustness:
  enabled: ${rob_bool}
  modes:
    - segmentation_perturbation
  segmentation_perturbation:
    intensity: "${ROBUSTNESS_INTENSITY}"
    apply_to_structures:
      - "GTV*"
      - "CTV*"
      - "PTV*"
      - "urinary_bladder"
      - "colon"
      - "prostate"
      - "rectum"
      - "femur*"
    small_volume_changes: [-0.15, 0.0, 0.15]
    max_translation_mm: 3.0
    n_random_contour_realizations: 2
    noise_levels: [0.0]
  metrics:
    icc:
      implementation: "pingouin"
      icc_type: "ICC3"
      ci: true
    cov:
      enabled: true
    qcd:
      enabled: true
  thresholds:
    icc:
      robust: 0.90
      acceptable: 0.75
    cov:
      robust_pct: 10.0
      acceptable_pct: 20.0

ct_cropping:
  enabled: ${crop_bool}
  region: "${CROPPING_REGION}"
  superior_margin_cm: 2.0
  inferior_margin_cm: 10.0
  use_cropped_for_dvh: true
  use_cropped_for_radiomics: true
  keep_original: true

aggregation:
  threads: ${AGG_THREADS_VALUE}

environments:
  main: "rtpipeline"
  radiomics: "rtpipeline-radiomics"

custom_structures: "custom_structures_pelvic.yaml"
EOF
    echo "Created: $CONFIG_PATH"
fi

# ---------------------------
# Generate docker-compose.project.yml
# ---------------------------

COMPOSE_PATH="$PROJECT_DIR/docker-compose.project.yml"
cat > "$COMPOSE_PATH" <<EOF
services:
  rtpipeline:
    image: kstawiski/rtpipeline:latest
    container_name: $(basename "$PROJECT_DIR")_rtpipeline
    hostname: $(basename "$PROJECT_DIR")_rtpipeline
    user: "1000:1000"
    working_dir: /app
    shm_size: "8gb"

    volumes:
      - "${DICOM_DIR}:/data/input:ro"
      - "${PROJECT_DIR}/Output:/data/output:rw"
      - "${PROJECT_DIR}/Logs:/data/logs:rw"
      - "${PROJECT_DIR}/config.yaml:/app/config.custom.yaml:ro"
      - "${PROJECT_DIR}/.snakemake_cache:/home/rtpipeline/.snakemake:rw"
EOF
if [ -n "$TOTALSEG_WEIGHTS_DIR" ]; then
cat >> "$COMPOSE_PATH" <<EOF
      - "${TOTALSEG_WEIGHTS_DIR}:/home/rtpipeline/.totalsegmentator:rw"
EOF
else
cat >> "$COMPOSE_PATH" <<'EOF'
      # Optional weights cache:
      # - /path/to/totalseg_weights:/home/rtpipeline/.totalsegmentator:rw
EOF
fi

cat >> "$COMPOSE_PATH" <<EOF

    environment:
      - PYTHONPATH=/app
      - NUMBA_CACHE_DIR=/tmp/cache
      - MPLCONFIGDIR=/tmp/cache
      - CONDA_DIR=/opt/conda
      - PATH=/opt/conda/bin:$PATH
      - TOTALSEG_TIMEOUT=7200
      - DCM2NIIX_TIMEOUT=300
      - RTPIPELINE_RADIOMICS_TASK_TIMEOUT=1200
EOF
if [ "$USE_GPU_RUNTIME" = "yes" ]; then
cat >> "$COMPOSE_PATH" <<'EOF'
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
EOF
fi

cat >> "$COMPOSE_PATH" <<EOF

    command: >
      snakemake
        --cores ${MAX_CORES}
        --use-conda
        --configfile /app/config.custom.yaml
        --rerun-triggers mtime input
        --printshellcmds

    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL

    deploy:
      resources:
        limits:
          cpus: "${MAX_CORES}"
          memory: "${MAX_MEMORY}"
        reservations:
EOF
if [ "$USE_GPU_RUNTIME" = "yes" ]; then
cat >> "$COMPOSE_PATH" <<'EOF'
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
EOF
else
cat >> "$COMPOSE_PATH" <<'EOF'
          cpus: "1.0"
EOF
fi

if [ "$USE_GPU_RUNTIME" = "yes" ]; then
cat >> "$COMPOSE_PATH" <<'EOF'

    runtime: nvidia
EOF
fi

cat >> "$COMPOSE_PATH" <<'EOF'

    restart: "no"

networks:
  default:
    name: rtpipeline_network
EOF
echo "Created: $COMPOSE_PATH"

# ---------------------------
# Runner scripts
# ---------------------------

cat > "$PROJECT_DIR/run_docker.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

detect_compose() {
  if docker compose version >/dev/null 2>&1; then
    echo "docker compose"
  elif command -v docker-compose >/dev/null 2>&1; then
    echo "docker-compose"
  else
    echo ""
  fi
}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COMPOSE_BIN="$(detect_compose)"
if [ -z "$COMPOSE_BIN" ]; then
  echo "Docker Compose not found. Install https://docs.docker.com/compose/install/"
  exit 1
fi

PULL_PROGRESS_FLAG=""
if $COMPOSE_BIN version 2>/dev/null | head -1 | grep -qi "docker compose version"; then
  PULL_PROGRESS_FLAG="--progress=plain"
fi

cd "$SCRIPT_DIR"
echo "Running rtpipeline via Docker Compose..."
if [ "${SKIP_PULL:-0}" != "1" ]; then
  echo "  Pulling image (set SKIP_PULL=1 to skip)..."
  $COMPOSE_BIN ${PULL_PROGRESS_FLAG:+$PULL_PROGRESS_FLAG} -f docker-compose.project.yml pull rtpipeline || true
fi
echo "  Starting pipeline (this may take several minutes on first run)..."
$COMPOSE_BIN -f docker-compose.project.yml run --rm rtpipeline "$@"
EOF
chmod +x "$PROJECT_DIR/run_docker.sh"

cat > "$PROJECT_DIR/monitor.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

detect_compose() {
  if docker compose version >/dev/null 2>&1; then
    echo "docker compose"
  elif command -v docker-compose >/dev/null 2>&1; then
    echo "docker-compose"
  else
    echo ""
  fi
}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COMPOSE_BIN="$(detect_compose)"
if [ -z "$COMPOSE_BIN" ]; then
  echo "Docker Compose not found. Install https://docs.docker.com/compose/install/"
  exit 1
fi

cd "$SCRIPT_DIR"
CID=$($COMPOSE_BIN -f docker-compose.project.yml ps -q rtpipeline 2>/dev/null || true)
if [ -z "$CID" ]; then
  echo "No running pipeline container. Start with ./run_docker.sh"
  exit 1
fi

echo "Tailing logs for container: $CID (Ctrl+C to stop viewing)"
docker logs -f "$CID"
EOF
chmod +x "$PROJECT_DIR/monitor.sh"

echo "Created: $PROJECT_DIR/run_docker.sh"
echo "Created: $PROJECT_DIR/monitor.sh"

# ---------------------------
# README
# ---------------------------

cat > "$PROJECT_DIR/README.md" <<EOF
# RTpipeline Docker Project

Generated: $(date)

## Paths
- DICOM input: ${DICOM_DIR}
- Project: ${PROJECT_DIR}
- Output: ${PROJECT_DIR}/Output
- Logs: ${PROJECT_DIR}/Logs

## Run
\`\`\`bash
cd ${PROJECT_DIR}
./run_docker.sh
\`\`\`

Monitor in another terminal:
\`\`\`bash
cd ${PROJECT_DIR}
./monitor.sh
\`\`\`

## Feature summary
- GPU hardware detected: ${GPU_AVAILABLE}
- GPU runtime: ${USE_GPU_RUNTIME} (${GPU_DEVICE})
- Optimization profile: ${PROFILE_LABEL}
- Scheduler: reserved=${SCHED_RESERVED_CORES}, parallel_courses=${SCHED_PARALLEL_COURSES}
- Segmentation: device=${SEG_DEVICE_CONFIG}, max_workers=${SEG_MAX_WORKERS_CONFIG}
- Radiomics: ${ENABLE_RADIOMICS}
- Robustness: ${ENABLE_ROBUSTNESS} (${ROBUSTNESS_INTENSITY})
- CT cropping: ${ENABLE_CT_CROPPING} (${CROPPING_REGION})
- Custom nnU-Net models: ${ENABLE_CUSTOM_MODELS}
- Max cores: ${MAX_CORES}, Max memory: ${MAX_MEMORY}

## Docs
- Docker guide: docs/DOCKER_SETUP_GUIDE.md
- Troubleshooting: docs/TROUBLESHOOTING.md
- Output reference: output_format_quick_ref.md
EOF
echo "Created: $PROJECT_DIR/README.md"

# ---------------------------
# Summary
# ---------------------------

echo ""
echo "Setup complete."
echo "Project directory: $PROJECT_DIR"
echo "DICOM directory:   $DICOM_DIR"
echo "Next: cd $PROJECT_DIR && ./run_docker.sh"
