#!/bin/bash
# RTpipeline Docker Execution Script
# Optimized for performance with timeout safety
#
# Usage:
#   ./run_rtpipeline.sh [OPTIONS]
#
# Options:
#   --timeout HOURS   Set pipeline timeout (default: 24 hours)
#   --cores N         Override CPU cores (default: auto-detect - 1)
#   --cpu-only        Run without GPU
#   --dry-run         Show what would run without executing
#   --build           Rebuild Docker image before running
#   --input DIR       Input DICOM directory
#   --output DIR      Output directory
#   --config FILE     Custom config file
#   -h, --help        Show this help

set -euo pipefail

# Defaults
TIMEOUT_HOURS="${RTPIPELINE_TIMEOUT_HOURS:-24}"
CPU_ONLY=false
DRY_RUN=false
BUILD_IMAGE=false
VALIDATE=false
VALIDATE_ONLY=false
VALIDATION_TIMEOUT_SECONDS="${RTPIPELINE_VALIDATE_TIMEOUT:-600}"
INPUT_DIR="${RTPIPELINE_INPUT_DIR:-./Input}"
OUTPUT_DIR="${RTPIPELINE_OUTPUT_DIR:-./Output}"
LOG_DIR="${RTPIPELINE_LOG_DIR:-./Logs}"
CONFIG_FILE=""
CORES=""
DOCKER_IMAGE="${RTPIPELINE_IMAGE:-kstawiski/rtpipeline:latest}"
CONTAINER_NAME="${RTPIPELINE_CONTAINER_NAME:-rtpipeline-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_OVERRIDES="--config dicom_root=/data/input output_dir=/data/output logs_dir=/data/logs"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

show_help() {
    cat << 'EOF'
RTpipeline Docker Execution Script

USAGE:
    ./run_rtpipeline.sh [OPTIONS]

OPTIONS:
    --timeout HOURS   Pipeline timeout in hours (default: 24)
    --cores N         Number of CPU cores to use (default: auto = N-1)
    --cpu-only        Run without GPU (slower segmentation)
    --build           Rebuild Docker image before running (required after code changes)
    --dry-run         Show command without executing
    --input DIR       Input DICOM directory
    --output DIR      Output directory
    --config FILE     Custom config.yaml file to use
    --validate        Run a dry-run DAG validation inside Docker before execution
    --validate-only   Only run the in-container validation and exit (no pipeline run)
    --container-name  Override Docker container name (default: rtpipeline-run)
    -h, --help        Show this help

ENVIRONMENT VARIABLES:
    RTPIPELINE_TIMEOUT_HOURS   Default timeout (hours)
    RTPIPELINE_INPUT_DIR       Default input directory
    RTPIPELINE_OUTPUT_DIR      Default output directory
    RTPIPELINE_IMAGE           Docker image to use

EXAMPLES:
    # Run with defaults (24h timeout, all GPUs, auto cores)
    ./run_rtpipeline.sh --input /path/to/dicom --output /path/to/results

    # Rebuild image after code changes, then run
    ./run_rtpipeline.sh --build --input /path/to/dicom --output /path/to/results

    # Run with 48h timeout on CPU only
    ./run_rtpipeline.sh --timeout 48 --cpu-only --input /path/to/dicom --output /path/to/results

    # Dry run to check configuration
    ./run_rtpipeline.sh --dry-run --build --input ./Input --output ./Output

WORKFLOW:
    1. First time: ./run_rtpipeline.sh --build --input /data --output /results
    2. Subsequent runs: ./run_rtpipeline.sh --input /data --output /results
    3. After code changes: ./run_rtpipeline.sh --build --input /data --output /results

NOTE: For interactive project setup, use ./setup_docker_project.sh instead
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --timeout)
            TIMEOUT_HOURS="$2"
            shift 2
            ;;
        --cores)
            CORES="$2"
            shift 2
            ;;
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --build)
            BUILD_IMAGE=true
            shift
            ;;
        --validate)
            VALIDATE=true
            shift
            ;;
        --validate-only)
            VALIDATE=true
            VALIDATE_ONLY=true
            shift
            ;;
        --input)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --container-name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ ! -d "$INPUT_DIR" ]]; then
    log_error "Input directory not found: $INPUT_DIR"
    exit 1
fi

# Create output/log directories if needed
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

if [[ ! -f "$SCRIPT_DIR/Snakefile" ]]; then
    log_error "Snakefile not found at $SCRIPT_DIR/Snakefile"
    exit 1
fi

if [[ ! -d "$SCRIPT_DIR/rtpipeline" ]]; then
    log_error "rtpipeline package directory not found at $SCRIPT_DIR/rtpipeline"
    exit 1
fi

if [[ ! -f "$SCRIPT_DIR/custom_structures_pelvic.yaml" ]]; then
    log_error "custom_structures_pelvic.yaml not found at $SCRIPT_DIR/custom_structures_pelvic.yaml"
    exit 1
fi

# Auto-detect cores if not specified (use N-1)
if [[ -z "$CORES" ]]; then
    TOTAL_CORES=$(nproc)
    CORES=$((TOTAL_CORES > 1 ? TOTAL_CORES - 1 : 1))
    log_info "Auto-detected $TOTAL_CORES cores, using $CORES for pipeline"
fi

# Calculate timeout in seconds
TIMEOUT_SECONDS=$((TIMEOUT_HOURS * 3600))

# Build Docker command
DOCKER_BASE="docker run --rm --name ${CONTAINER_NAME}"

# Add GPU support if not CPU-only
if [[ "$CPU_ONLY" == "false" ]]; then
    if command -v nvidia-smi &> /dev/null; then
        DOCKER_BASE="$DOCKER_BASE --gpus all"
        log_info "GPU detected, enabling CUDA support"
    else
        log_warning "No GPU detected, running in CPU mode (slower)"
        CPU_ONLY=true
    fi
fi

# Add resource limits
DOCKER_BASE="$DOCKER_BASE --cpus=$CORES"
DOCKER_BASE="$DOCKER_BASE --memory=32g"
DOCKER_BASE="$DOCKER_BASE --shm-size=8g"

# Add volumes
INPUT_DIR_ABS=$(realpath "$INPUT_DIR")
OUTPUT_DIR_ABS=$(realpath "$OUTPUT_DIR")
LOG_DIR_ABS=$(realpath "$LOG_DIR")

DOCKER_BASE="$DOCKER_BASE -v ${INPUT_DIR_ABS}:/data/input:ro"
DOCKER_BASE="$DOCKER_BASE -v ${OUTPUT_DIR_ABS}:/data/output:rw"
DOCKER_BASE="$DOCKER_BASE -v ${LOG_DIR_ABS}:/data/logs:rw"
DOCKER_BASE="$DOCKER_BASE -v ${SCRIPT_DIR}/Snakefile:/app/Snakefile:ro"
DOCKER_BASE="$DOCKER_BASE -v ${SCRIPT_DIR}/rtpipeline:/app/rtpipeline:ro"
DOCKER_BASE="$DOCKER_BASE -v ${SCRIPT_DIR}/custom_structures_pelvic.yaml:/app/custom_structures_pelvic.yaml:ro"

# Add config file if specified
if [[ -n "$CONFIG_FILE" ]]; then
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "Config file not found: $CONFIG_FILE"
        exit 1
    fi
    CONFIG_FILE_ABS=$(realpath "$CONFIG_FILE")
    DOCKER_BASE="$DOCKER_BASE -v ${CONFIG_FILE_ABS}:/app/config_custom.yaml:ro"
    CONFIG_ARG="--configfile /app/config_custom.yaml"
else
    CONFIG_ARG="--configfile /app/config.container.yaml"
fi

# Set environment variables for timeouts
DOCKER_BASE="$DOCKER_BASE -e TOTALSEG_TIMEOUT=3600"
DOCKER_BASE="$DOCKER_BASE -e DCM2NIIX_TIMEOUT=300"
DOCKER_BASE="$DOCKER_BASE -e RTPIPELINE_RADIOMICS_TASK_TIMEOUT=600"

# Add user mapping for proper file permissions
DOCKER_BASE="$DOCKER_BASE --user $(id -u):$(id -g)"

# Set device for segmentation
if [[ "$CPU_ONLY" == "true" ]]; then
    DEVICE_CONFIG="--config segmentation.device=cpu"
else
    DEVICE_CONFIG=""
fi

# Add image and command
DOCKER_BASE="$DOCKER_BASE $DOCKER_IMAGE"
PIPELINE_CMD="snakemake --cores $CORES --rerun-incomplete $CONFIG_ARG $DEVICE_CONFIG $CONFIG_OVERRIDES"
FULL_CMD="$DOCKER_BASE $PIPELINE_CMD"
VALIDATION_CMD="$DOCKER_BASE snakemake --cores 1 -n $CONFIG_ARG $DEVICE_CONFIG --quiet"

# Print configuration
echo ""
echo "=============================================="
echo "         RTpipeline Docker Runner"
echo "=============================================="
log_info "Input directory: $INPUT_DIR_ABS"
log_info "Output directory: $OUTPUT_DIR_ABS"
log_info "CPU cores: $CORES"
log_info "Timeout: ${TIMEOUT_HOURS}h (${TIMEOUT_SECONDS}s)"
log_info "GPU mode: $([ "$CPU_ONLY" == "false" ] && echo "enabled" || echo "disabled")"
log_info "Docker image: $DOCKER_IMAGE"
log_info "Rebuild image: $BUILD_IMAGE"
log_info "Container name: $CONTAINER_NAME"
log_info "Pre-run validation: $([ "$VALIDATE" == "true" ] && echo "enabled" || echo "disabled")"
echo "=============================================="
echo ""

# Build image if requested
if [[ "$BUILD_IMAGE" == "true" ]]; then
    log_info "Building Docker image..."
    if [[ "$DRY_RUN" == "true" ]]; then
        log_warning "DRY RUN - Would execute: docker build -t $DOCKER_IMAGE $SCRIPT_DIR"
    else
        if docker build -t "$DOCKER_IMAGE" "$SCRIPT_DIR"; then
            log_success "Docker image built successfully"
        else
            log_error "Docker image build failed"
            exit 1
        fi
    fi
fi

if [[ "$DRY_RUN" == "true" ]]; then
    log_warning "DRY RUN - Commands that would be executed:"
    echo ""
    if [[ "$VALIDATE" == "true" ]]; then
        echo "timeout ${VALIDATION_TIMEOUT_SECONDS}s $VALIDATION_CMD"
        echo ""
    fi
    echo "timeout ${TIMEOUT_SECONDS}s $FULL_CMD"
    echo ""
    exit 0
fi

if [[ "$VALIDATE" == "true" ]]; then
    log_info "Running dataset validation inside Docker for $INPUT_DIR_ABS (timeout: ${VALIDATION_TIMEOUT_SECONDS}s)..."
    if timeout ${VALIDATION_TIMEOUT_SECONDS}s $VALIDATION_CMD; then
        log_success "Validation completed successfully"
    else
        log_error "Validation failed. Fix issues before running the full pipeline."
        exit 1
    fi
fi

if [[ "$VALIDATE_ONLY" == "true" ]]; then
    log_info "Validation-only mode requested; skipping full pipeline execution."
    exit 0
fi

# Run with timeout
log_info "Starting pipeline (timeout: ${TIMEOUT_HOURS}h)..."
START_TIME=$(date +%s)

# Trap for clean shutdown
cleanup() {
    log_warning "Received interrupt signal, stopping container..."
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    exit 130
}
trap cleanup INT TERM

# Run the pipeline with timeout
if timeout ${TIMEOUT_SECONDS}s $FULL_CMD; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    DURATION_MIN=$((DURATION / 60))
    log_success "Pipeline completed successfully in ${DURATION_MIN} minutes"
    log_success "Results saved to: $OUTPUT_DIR_ABS"
else
    EXIT_CODE=$?
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    if [[ $EXIT_CODE -eq 124 ]]; then
        log_error "Pipeline timed out after ${TIMEOUT_HOURS} hours"
        log_warning "Consider increasing timeout with --timeout HOURS"
    else
        log_error "Pipeline failed with exit code: $EXIT_CODE"
    fi
    exit $EXIT_CODE
fi

echo ""
log_success "Done!"
