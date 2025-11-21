#!/bin/bash
#
# RTpipeline Docker Project Setup Wizard
#
# Interactive tool for setting up new rtpipeline projects using Docker.
# Generates docker-compose configuration, custom config.yaml, and run scripts.
#
# Usage:
#   ./setup_docker_project.sh                    # Interactive wizard
#   ./setup_docker_project.sh --quick /path      # Quick setup with defaults
#   ./setup_docker_project.sh --preset research  # Use preset configuration
#

set -euo pipefail

# ============================================================================
# UI Constants & Helpers
# ============================================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m' # No Color

# Icons
ICON_CHECK="${GREEN}✓${NC}"
ICON_X="${RED}✗${NC}"
ICON_WARN="${YELLOW}⚠${NC}"
ICON_INFO="${BLUE}ℹ${NC}"
ICON_Q="${CYAN}?${NC}"
ICON_GEAR="${MAGENTA}⚙${NC}"
ICON_DOCKER="${BLUE}🐳${NC}"

# Script variables
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VERSION="1.0.0"
PRESET=""
QUICK_MODE=false
DRY_RUN=false
VERBOSE=0

# Print banner
print_banner() {
    clear
    echo -e "${BLUE}${BOLD}"
    cat << "EOF"
    ____  ______       _            _ _
   |  _ \|_   _|     (_)          | (_)
   | |_) | | |_ __    _ _ __   ___| |_ _ __   ___
   |  _ <  | | '_ \  | | '_ \ / _ \ | | '_ \ / _ \
   | |_) | | | |_) | | | |_) |  __/ | | | | |  __/
   |____/  |_| .__/  |_| .__/ \___|_|_|_| |_|\___|
             | |       | |
             |_|       |_|

         🐳 Docker Project Setup Wizard 🐳
EOF
    echo -e "${NC}"
    echo -e "${DIM}   Radiotherapy DICOM Processing Pipeline - Docker Edition v${VERSION}${NC}"
    echo -e "${DIM}   ---------------------------------------------------------------${NC}"
    echo ""
}

# Print section header
print_header() {
    echo -e "\n${BOLD}${MAGENTA}:: $1 ::${NC}"
    echo -e "${DIM}$(printf '%.0s-' {1..70})${NC}"
}

# Print info message
print_info() {
    echo -e "  ${ICON_INFO} $1"
}

# Print warning
print_warning() {
    echo -e "  ${ICON_WARN} ${YELLOW}$1${NC}"
}

# Print error
print_error() {
    echo -e "  ${ICON_X} ${RED}$1${NC}"
}

# Print success
print_success() {
    echo -e "  ${ICON_CHECK} ${GREEN}$1${NC}"
}

# Ask a question with default value
ask_question() {
    local question="$1"
    local default="$2"
    local response

    echo -en "  ${ICON_Q} $question"
    if [ -n "$default" ]; then
        echo -e " ${DIM}[$default]${NC}"
    else
        echo ""
    fi
    read -p "    > " response
    echo "${response:-$default}"
}

# Ask yes/no question
ask_yes_no() {
    local question="$1"
    local default="$2"
    local response
    local prompt_suffix

    if [ "$default" = "yes" ]; then
        prompt_suffix="[Y/n]"
    else
        prompt_suffix="[y/N]"
    fi

    echo -e "  ${ICON_Q} $question ${DIM}$prompt_suffix${NC}"
    read -p "    > " response

    # If empty, use default
    if [ -z "$response" ]; then
        response="$default"
    fi

    case "$response" in
        [Yy]|[Yy][Ee][Ss]) echo "yes" ;;
        *) echo "no" ;;
    esac
}

# Ask for selection from list
ask_select() {
    local question="$1"
    shift
    local options=("$@")
    local PS3="    > "

    echo -e "  ${ICON_Q} $question"
    select opt in "${options[@]}"
    do
        if [ -n "$opt" ]; then
            echo "$opt"
            break
        else
            echo -e "    ${RED}Invalid selection.${NC}"
        fi
    done
}

# ============================================================================
# Validation & Checks
# ============================================================================

check_prerequisites() {
    print_header "Docker Environment Check"
    local missing=0

    # Docker
    if command -v docker &> /dev/null; then
        local docker_ver=$(docker --version 2>&1 | awk '{print $3}' | sed 's/,//')
        print_success "Docker $docker_ver found"

        if docker ps &> /dev/null 2>&1; then
            print_success "Docker daemon running"
        else
            print_error "Docker daemon not running or permission denied"
            print_info "Try: sudo systemctl start docker"
            print_info "Or: sudo usermod -aG docker $USER (then logout/login)"
            missing=$((missing + 1))
        fi
    else
        print_error "Docker not found - Docker is required for this setup"
        print_info "Install from: https://docs.docker.com/get-docker/"
        missing=$((missing + 1))
    fi

    # Docker Compose
    if command -v docker-compose &> /dev/null || docker compose version &> /dev/null 2>&1; then
        print_success "Docker Compose found"
    else
        print_warning "Docker Compose not found (optional but recommended)"
        print_info "Install from: https://docs.docker.com/compose/install/"
    fi

    # GPU
    if command -v nvidia-smi &> /dev/null; then
        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
        print_success "GPU detected: $gpu_name"

        # Check NVIDIA Docker runtime
        if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
            print_success "NVIDIA Docker runtime working"
        else
            print_warning "NVIDIA Docker runtime not working"
            print_info "Install nvidia-container-toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        fi
    else
        print_warning "No NVIDIA GPU detected - pipeline will run in CPU mode"
    fi

    # Check for rtpipeline image
    if docker image inspect kstawiski/rtpipeline:latest &> /dev/null; then
        print_success "rtpipeline Docker image found"
    else
        print_warning "rtpipeline Docker image not found locally"
        print_info "Will pull from Docker Hub on first run"
    fi

    if [ $missing -gt 0 ]; then
        print_error "Critical prerequisites missing. Please install them and retry."
        exit 1
    fi
    echo ""
}

# ============================================================================
# Configuration Presets
# ============================================================================

load_preset() {
    local preset_name="$1"

    case "$preset_name" in
        quick)
            PRESET_NAME="Quick DVH Analysis"
            PRESET_RADIOMICS="no"
            PRESET_ROBUSTNESS="no"
            PRESET_CT_CROPPING="yes"
            PRESET_CROPPING_REGION="pelvis"
            PRESET_CUSTOM_MODELS="no"
            ;;
        standard)
            PRESET_NAME="Standard Clinical Analysis"
            PRESET_RADIOMICS="yes"
            PRESET_ROBUSTNESS="no"
            PRESET_CT_CROPPING="yes"
            PRESET_CROPPING_REGION="pelvis"
            PRESET_CUSTOM_MODELS="no"
            ;;
        research)
            PRESET_NAME="Research-Grade Comprehensive Analysis"
            PRESET_RADIOMICS="yes"
            PRESET_ROBUSTNESS="yes"
            PRESET_ROBUSTNESS_INTENSITY="standard"
            PRESET_CT_CROPPING="yes"
            PRESET_CROPPING_REGION="pelvis"
            PRESET_CUSTOM_MODELS="no"
            ;;
        aggressive)
            PRESET_NAME="Aggressive Research Analysis"
            PRESET_RADIOMICS="yes"
            PRESET_ROBUSTNESS="yes"
            PRESET_ROBUSTNESS_INTENSITY="aggressive"
            PRESET_CT_CROPPING="yes"
            PRESET_CROPPING_REGION="pelvis"
            PRESET_CUSTOM_MODELS="no"
            ;;
        *)
            return 1
            ;;
    esac
    return 0
}

# ============================================================================
# Main Configuration Logic
# ============================================================================

main() {
    print_banner

    # Parse args
    while [[ $# -gt 0 ]]; do
        case $1 in
            --quick) QUICK_MODE=true; shift ;;
            --preset) PRESET="$2"; shift 2 ;;
            --dry-run) DRY_RUN=true; shift ;;
            *) if [ -z "${QUICK_DICOM_DIR:-}" ]; then QUICK_DICOM_DIR="$1"; fi; shift ;;
        esac
    done

    check_prerequisites

    # Load preset if specified
    if [ -n "$PRESET" ]; then
        if load_preset "$PRESET"; then
            print_success "Loaded preset: $PRESET_NAME"
        else
            print_error "Unknown preset: $PRESET"
            print_info "Available presets: quick, standard, research, aggressive"
            exit 1
        fi
    fi

    # --- Project Configuration ---
    print_header "Project Configuration"

    print_info "This wizard will set up a Docker-based rtpipeline project."
    print_info "All analysis will run inside Docker containers - no local installation needed!"
    echo ""

    # Ask if user has existing config or wants to create one
    print_header "Configuration Method"
    print_info "You can either:"
    print_info "  1. Use an existing config.yaml file"
    print_info "  2. Generate a new config interactively (recommended)"
    echo ""

    local use_existing_config=$(ask_yes_no "Do you have an existing config.yaml to use?" "no")
    local existing_config_path=""
    local skip_config_generation=false

    if [ "$use_existing_config" = "yes" ]; then
        existing_config_path=$(ask_question "Path to your config.yaml:" "")
        while [ ! -f "$existing_config_path" ]; do
            print_error "File not found: $existing_config_path"
            local retry=$(ask_yes_no "Try another path?" "yes")
            if [ "$retry" = "yes" ]; then
                existing_config_path=$(ask_question "Path to your config.yaml:" "")
            else
                print_info "Switching to interactive config generation"
                use_existing_config="no"
                break
            fi
        done

        if [ "$use_existing_config" = "yes" ]; then
            # Expand to absolute path
            existing_config_path="${existing_config_path/#\~/$HOME}"
            if [[ "$existing_config_path" != /* ]]; then
                existing_config_path="$(pwd)/$existing_config_path"
            fi
            print_success "Will use config: $existing_config_path"
            skip_config_generation=true
        fi
    fi

    # Select preset or custom (only if generating new config)
    local use_preset="no"
    if [ "$skip_config_generation" = "false" ] && [ -z "$PRESET" ] && [ "$QUICK_MODE" = "false" ]; then
        echo ""
        use_preset=$(ask_yes_no "Use a configuration preset?" "yes")

        if [ "$use_preset" = "yes" ]; then
            print_info "Available presets:"
            echo -e "    ${BOLD}1) quick${NC}      - Fast DVH-only analysis (no radiomics)"
            echo -e "    ${BOLD}2) standard${NC}   - Clinical analysis with radiomics (no robustness)"
            echo -e "    ${BOLD}3) research${NC}   - Research-grade with robustness analysis (recommended)"
            echo -e "    ${BOLD}4) aggressive${NC} - Maximum robustness (30-60 perturbations per ROI)"
            echo ""
            local preset_choice=$(ask_question "Select preset [1-4]" "3")

            case "$preset_choice" in
                1) load_preset "quick" ;;
                2) load_preset "standard" ;;
                3) load_preset "research" ;;
                4) load_preset "aggressive" ;;
                *) load_preset "research" ;;
            esac
            print_success "Using preset: $PRESET_NAME"
        fi
    fi

    # --- Directory Setup ---
    print_header "Directory Setup"

    local dicom_dir
    if [ -n "${QUICK_DICOM_DIR:-}" ]; then
        dicom_dir="$QUICK_DICOM_DIR"
        print_info "Using input directory: $dicom_dir"
    else
        dicom_dir=$(ask_question "Where are your DICOM files located?" "")
        while [ -z "$dicom_dir" ]; do
            print_error "Path cannot be empty."
            dicom_dir=$(ask_question "Where are your DICOM files located?" "")
        done
    fi

    # Convert to absolute path and handle non-existent directories
    # First, expand ~ if present
    dicom_dir="${dicom_dir/#\~/$HOME}"

    # Make path absolute if relative
    if [[ "$dicom_dir" != /* ]]; then
        dicom_dir="$(pwd)/$dicom_dir"
    fi

    # Check if directory exists
    if [ ! -d "$dicom_dir" ]; then
        print_warning "Directory does not exist: $dicom_dir"
        local create_dir=$(ask_yes_no "Create DICOM input directory?" "yes")
        if [ "$create_dir" = "yes" ]; then
            mkdir -p "$dicom_dir" || {
                print_error "Failed to create directory: $dicom_dir"
                exit 1
            }
            print_success "Created: $dicom_dir"
        else
            print_error "DICOM directory is required. Exiting."
            exit 1
        fi
    fi

    # Now resolve to absolute canonical path
    dicom_dir=$(cd "$dicom_dir" && pwd)

    local parent_dir=$(dirname "$dicom_dir")
    local project_name=$(basename "$dicom_dir")
    local default_project_dir="$parent_dir/${project_name}_rtpipeline"

    local project_dir=$(ask_question "Project directory (will contain config and outputs)" "$default_project_dir")

    # Create project structure
    mkdir -p "$project_dir"/{Output,Logs}
    project_dir=$(cd "$project_dir"; pwd)

    print_success "Project directory: $project_dir"
    print_success "DICOM input: $dicom_dir"

    # TotalSegmentator weights handling
    print_header "TotalSegmentator Weights"
    print_info "The Docker image includes TotalSegmentator weights baked-in."
    print_info "You can optionally mount a weights directory to:"
    print_info "  - Update weights without rebuilding the image"
    print_info "  - Share weights across multiple projects"
    echo ""

    local mount_weights=$(ask_yes_no "Mount TotalSegmentator weights directory?" "no")
    local totalseg_weights_dir=""
    local weights_mount_line=""

    if [ "$mount_weights" = "yes" ]; then
        totalseg_weights_dir="$SCRIPT_DIR/totalseg_weights"
        if [ ! -d "$totalseg_weights_dir" ]; then
            mkdir -p "$totalseg_weights_dir"
            print_success "Created weights cache: $totalseg_weights_dir"
        fi
        weights_mount_line="      # TotalSegmentator weights (optional - for caching/updating without image rebuild)
      - ${totalseg_weights_dir}:/home/rtpipeline/.totalsegmentator:rw
"
        print_success "Will mount: $totalseg_weights_dir"
    else
        weights_mount_line="      # TotalSegmentator weights (optional - weights are baked into image)
      # Uncomment to cache/update weights:
      # - /path/to/totalseg_weights:/home/rtpipeline/.totalsegmentator:rw
"
        print_info "Using baked-in weights (simpler setup)"
    fi

    # --- GPU Configuration ---
    print_header "GPU Configuration"
    local has_gpu="no"
    local gpu_device="cpu"

    if command -v nvidia-smi &> /dev/null; then
        has_gpu=$(ask_yes_no "Use GPU for segmentation?" "yes")
        if [ "$has_gpu" = "yes" ]; then
            gpu_device="gpu"
        fi
    else
        print_info "No GPU detected - using CPU mode"
        gpu_device="cpu"
    fi

    # --- Processing Configuration ---
    # Skip if using existing config
    if [ "$skip_config_generation" = "true" ]; then
        print_header "Configuration"
        print_success "Using existing configuration file"
        print_info "Skipping interactive configuration..."

        # Set dummy values since we're using existing config
        enable_radiomics="yes"
        enable_robustness="yes"
        robustness_intensity="standard"
        enable_ct_cropping="yes"
        cropping_region="pelvis"
        enable_custom_models="no"
        max_workers="null"
        max_cores="16"
        max_memory="32G"
    else
        print_header "Processing Features"

        # Radiomics
        local enable_radiomics="${PRESET_RADIOMICS:-}"
        if [ -z "$enable_radiomics" ]; then
            print_info "Radiomics extracts 100+ features per ROI (shape, texture, intensity)"
            enable_radiomics=$(ask_yes_no "Enable radiomics extraction?" "yes")
        else
            print_info "Radiomics: ${enable_radiomics} (from preset)"
        fi

    # Radiomics Robustness
    local enable_robustness="${PRESET_ROBUSTNESS:-}"
    local robustness_intensity="${PRESET_ROBUSTNESS_INTENSITY:-standard}"

    if [ "$enable_radiomics" = "yes" ]; then
        if [ -z "$enable_robustness" ]; then
            echo ""
            print_info "Radiomics Robustness evaluates feature stability via perturbations"
            print_info "Recommended for research/ML models - adds significant processing time"
            enable_robustness=$(ask_yes_no "Enable radiomics robustness analysis?" "yes")

            if [ "$enable_robustness" = "yes" ]; then
                print_info "Perturbation intensity:"
                echo -e "    ${BOLD}1) mild${NC}       - 10-15 perturbations per ROI (quick QA)"
                echo -e "    ${BOLD}2) standard${NC}   - 15-30 perturbations per ROI (recommended)"
                echo -e "    ${BOLD}3) aggressive${NC} - 30-60 perturbations per ROI (research-grade)"
                local intensity_choice=$(ask_question "Select intensity [1-3]" "2")

                case "$intensity_choice" in
                    1) robustness_intensity="mild" ;;
                    3) robustness_intensity="aggressive" ;;
                    *) robustness_intensity="standard" ;;
                esac
            fi
        else
            print_info "Robustness: ${enable_robustness} (${robustness_intensity}) (from preset)"
        fi
    else
        enable_robustness="no"
    fi

    # CT Cropping
    local enable_ct_cropping="${PRESET_CT_CROPPING:-}"
    local cropping_region="${PRESET_CROPPING_REGION:-pelvis}"

    if [ -z "$enable_ct_cropping" ]; then
        echo ""
        print_info "CT Cropping ensures consistent anatomical volumes for comparison"
        print_info "Recommended for multi-patient DVH studies"
        enable_ct_cropping=$(ask_yes_no "Enable systematic CT cropping?" "yes")

        if [ "$enable_ct_cropping" = "yes" ]; then
            print_info "Select anatomical region:"
            cropping_region=$(ask_select "Region:" "pelvis" "thorax" "abdomen" "head_neck" "brain")
        fi
    else
        print_info "CT Cropping: ${enable_ct_cropping} (${cropping_region}) (from preset)"
    fi

    # Custom Models
    local enable_custom_models="${PRESET_CUSTOM_MODELS:-}"
    if [ -z "$enable_custom_models" ]; then
        echo ""
        print_info "Custom nnU-Net models require separately downloaded weights"
        enable_custom_models=$(ask_yes_no "Enable custom nnU-Net models?" "no")
    fi

        # Parallelization
        print_header "Parallelization Settings"
        local max_workers="null"
        local max_cores="16"

        if [ "$QUICK_MODE" = "false" ]; then
            local avail_cores=$(nproc 2>/dev/null || echo "16")
            print_info "Available CPU cores: $avail_cores"
            max_cores=$(ask_question "Max CPU cores for Docker container" "$avail_cores")

            print_info "Max workers controls internal parallelism (null = auto-detect)"
            local worker_input=$(ask_question "Max workers (null/auto or number)" "null")
            if [ "$worker_input" != "null" ] && [ "$worker_input" != "auto" ]; then
                max_workers="$worker_input"
            fi
        fi

        # Memory
        local max_memory="32G"
        if [ "$QUICK_MODE" = "false" ]; then
            max_memory=$(ask_question "Max memory for Docker container (e.g., 32G)" "32G")
        fi
    fi  # End of skip_config_generation check

    # Generate files
    if [ "$DRY_RUN" = "true" ]; then
        print_header "Dry Run Summary"
        echo "Project Dir: $project_dir"
        echo "DICOM Dir: $dicom_dir"
        echo "GPU: $gpu_device"
        echo "Radiomics: $enable_radiomics"
        echo "Robustness: $enable_robustness ($robustness_intensity)"
        echo "CT Cropping: $enable_ct_cropping ($cropping_region)"
        exit 0
    fi

    print_header "Generating Project Files"

    # Convert yes/no to true/false
    local radiomics_bool="false"
    if [ "$enable_radiomics" = "yes" ]; then radiomics_bool="true"; fi

    local robustness_bool="false"
    if [ "$enable_robustness" = "yes" ]; then robustness_bool="true"; fi

    local ct_cropping_bool="false"
    if [ "$enable_ct_cropping" = "yes" ]; then ct_cropping_bool="true"; fi

    local custom_models_bool="false"
    if [ "$enable_custom_models" = "yes" ]; then custom_models_bool="true"; fi

    # 1. Generate or copy config.yaml
    local config_path="$project_dir/config.yaml"

    if [ "$skip_config_generation" = "true" ]; then
        # Copy existing config
        cp "$existing_config_path" "$config_path"
        print_success "Copied config from: $existing_config_path"

        # Update paths in the copied config to use container paths
        if command -v sed &> /dev/null; then
            sed -i 's|^dicom_root:.*|dicom_root: "/data/input"|' "$config_path" 2>/dev/null || true
            sed -i 's|^output_dir:.*|output_dir: "/data/output"|' "$config_path" 2>/dev/null || true
            sed -i 's|^logs_dir:.*|logs_dir: "/data/logs"|' "$config_path" 2>/dev/null || true
            print_info "Updated paths to use Docker container paths"
        fi
    else
        # Generate new config
        cat > "$config_path" <<EOF
# RTpipeline Docker Project Configuration
# Generated by setup_docker_project.sh v$VERSION on $(date)
# Project: $project_name

# =============================================================================
# Directory Paths (Docker container paths)
# =============================================================================
dicom_root: "/data/input"
output_dir: "/data/output"
logs_dir: "/data/logs"

# =============================================================================
# Processing Configuration
# =============================================================================
max_workers: $max_workers  # null = auto-detect (CPU cores - 1)

# Scheduler settings
scheduler:
  reserved_cores: 1
  dvh_threads_per_job: 4
  radiomics_threads_per_job: 6
  qc_threads_per_job: 2
  prioritize_short_courses: true

# =============================================================================
# Segmentation (TotalSegmentator)
# =============================================================================
segmentation:
  device: "$gpu_device"  # gpu or cpu
  fast: false  # Set true for 3x faster (lower quality)
  force: false  # Re-run even if outputs exist
  roi_subset: null  # null = all structures
  extra_models: []  # e.g., ["lung_vessels"]
  workers: null  # null = auto (sequential on GPU, parallel on CPU)
  force_split: true  # Reduce memory spikes
  nr_threads_resample: 1
  nr_threads_save: 1
  num_proc_preprocessing: 1
  num_proc_export: 1

# =============================================================================
# Custom nnU-Net Models
# =============================================================================
custom_models:
  enabled: $custom_models_bool
  root: "custom_models"
  models: []  # Empty = run all available models
  workers: null
  force: false
  retain_weights: true

# =============================================================================
# Radiomics Extraction
# =============================================================================
radiomics:
  enabled: $radiomics_bool
  sequential: false
  params_file: "rtpipeline/radiomics_params.yaml"
  mr_params_file: "rtpipeline/radiomics_params_mr.yaml"
  skip_rois:
    - body
    - couchsurface
    - couchinterior
    - couchexterior
  max_voxels: 1500000000
  min_voxels: 10

# =============================================================================
# Radiomics Robustness Analysis
# =============================================================================
# Evaluates feature stability under systematic perturbations
# Based on IBSI guidelines (Zwanenburg 2019, Lo Iacono 2024)
radiomics_robustness:
  enabled: $robustness_bool

  modes:
    - segmentation_perturbation

  segmentation_perturbation:
    # Intensity: mild (~10-15 perts), standard (~15-30 perts), aggressive (~30-60 perts)
    intensity: "$robustness_intensity"

    # Structures to analyze (supports wildcards)
    apply_to_structures:
      - "GTV*"
      - "CTV*"
      - "PTV*"
      - "urinary_bladder"
      - "colon"
      - "prostate"
      - "rectum"
      - "femur*"

    # Volume perturbations (±15%)
    small_volume_changes: [-0.15, 0.0, 0.15]

    # Translation perturbations (mm)
    max_translation_mm: 3.0

    # Contour randomization (number of realizations)
    n_random_contour_realizations: 2

    # Noise injection (HU standard deviation)
    noise_levels: [0.0]

  # Robustness metrics
  metrics:
    icc:
      implementation: "pingouin"
      icc_type: "ICC3"
      ci: true
    cov:
      enabled: true
    qcd:
      enabled: true

  # Classification thresholds
  thresholds:
    icc:
      robust: 0.90
      acceptable: 0.75
    cov:
      robust_pct: 10.0
      acceptable_pct: 20.0

# =============================================================================
# Systematic CT Cropping
# =============================================================================
# Crops all CTs to consistent anatomical boundaries
# Ensures percentage DVH metrics are comparable across patients
ct_cropping:
  enabled: $ct_cropping_bool

  # Region: pelvis, thorax, abdomen, head_neck, brain
  region: "$cropping_region"

  # Margins (cm)
  superior_margin_cm: 2.0
  inferior_margin_cm: 10.0

  # Apply cropped volumes to analysis
  use_cropped_for_dvh: true
  use_cropped_for_radiomics: true

  # Keep original uncropped files
  keep_original: true

# =============================================================================
# Aggregation
# =============================================================================
aggregation:
  threads: auto

# =============================================================================
# Environments
# =============================================================================
environments:
  main: "rtpipeline"
  radiomics: "rtpipeline-radiomics"

# =============================================================================
# Custom Structures
# =============================================================================
custom_structures: "custom_structures_pelvic.yaml"
EOF
        print_success "Created config.yaml"
    fi

    # 2. Generate docker-compose.project.yml
    local compose_path="$project_dir/docker-compose.project.yml"

    # Build GPU section if needed
    local gpu_section=""
    if [ "$has_gpu" = "yes" ]; then
        gpu_section="      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
        limits:
          cpus: '${max_cores}'
          memory: ${max_memory}"
    else
        gpu_section="      resources:
        limits:
          cpus: '${max_cores}'
          memory: ${max_memory}"
    fi

    cat > "$compose_path" <<EOF
# RTpipeline Docker Compose Configuration
# Project: $project_name
# Generated: $(date)

version: '3.8'

services:
  rtpipeline:
    image: kstawiski/rtpipeline:latest
    container_name: ${project_name}_rtpipeline
    hostname: ${project_name}_rtpipeline
    user: "1000:1000"

    volumes:
      # Input DICOM directory (read-only)
      - ${dicom_dir}:/data/input:ro

      # Output directory (read-write)
      - ${project_dir}/Output:/data/output:rw

      # Logs directory (read-write)
      - ${project_dir}/Logs:/data/logs:rw

      # Custom configuration
      - ${project_dir}/config.yaml:/app/config.custom.yaml:ro

${weights_mount_line}
    environment:
      - PYTHONPATH=/app
      - NUMBA_CACHE_DIR=/tmp/cache
      - MPLCONFIGDIR=/tmp/cache
      - CONDA_DIR=/opt/conda
      - PATH=/opt/conda/bin:\$PATH
      - TOTALSEG_TIMEOUT=7200
      - DCM2NIIX_TIMEOUT=300
      - RTPIPELINE_RADIOMICS_TASK_TIMEOUT=1200

    working_dir: /app

    # Command to run pipeline
    command: >
      snakemake
        --cores ${max_cores}
        --use-conda
        --configfile /app/config.custom.yaml
        --rerun-triggers mtime input
        --printshellcmds

    # Security
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL

    # Resource limits
    deploy:
${gpu_section}

    shm_size: '8gb'

    # Restart policy
    restart: "no"

networks:
  default:
    name: ${project_name}_network
EOF

    print_success "Created docker-compose.project.yml"

    # 3. Generate run_docker.sh
    local run_script="$project_dir/run_docker.sh"
    cat > "$run_script" <<'EOF'
#!/bin/bash
#
# RTpipeline Docker Execution Script
# Run the pipeline using Docker Compose
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "================================================"
echo "  RTpipeline Docker Runner"
echo "================================================"
echo ""
echo "Project directory: $SCRIPT_DIR"
echo "Starting pipeline..."
echo ""

# Pull latest image (optional - comment out to use cached image)
# docker-compose -f docker-compose.project.yml pull

# Run pipeline
docker-compose -f docker-compose.project.yml run --rm rtpipeline "$@"

echo ""
echo "================================================"
echo "  Pipeline Complete!"
echo "================================================"
echo ""
echo "Results available in: $SCRIPT_DIR/Output"
echo "Logs available in: $SCRIPT_DIR/Logs"
echo ""
EOF

    chmod +x "$run_script"
    print_success "Created run_docker.sh"

    # 4. Generate monitoring script
    local monitor_script="$project_dir/monitor.sh"
    cat > "$monitor_script" <<'EOF'
#!/bin/bash
#
# RTpipeline Monitoring Script
# Monitor running pipeline
#

CONTAINER_NAME=$(docker-compose -f docker-compose.project.yml ps -q rtpipeline 2>/dev/null)

if [ -z "$CONTAINER_NAME" ]; then
    echo "Pipeline container not running."
    echo "Start with: ./run_docker.sh"
    exit 1
fi

echo "Monitoring pipeline container: $CONTAINER_NAME"
echo "Press Ctrl+C to exit monitoring (pipeline will continue running)"
echo ""

docker logs -f "$CONTAINER_NAME"
EOF

    chmod +x "$monitor_script"
    print_success "Created monitor.sh"

    # 5. Generate README
    local readme_path="$project_dir/README.md"
    cat > "$readme_path" <<EOF
# RTpipeline Docker Project: $project_name

**Generated:** $(date)
**Configuration:** $PRESET_NAME

---

## Project Structure

\`\`\`
${project_dir}/
├── config.yaml                  # Pipeline configuration
├── docker-compose.project.yml   # Docker Compose configuration
├── run_docker.sh               # Execute pipeline
├── monitor.sh                  # Monitor running pipeline
├── Output/                     # Pipeline results (created on run)
├── Logs/                       # Pipeline logs (created on run)
└── README.md                   # This file
\`\`\`

**Input DICOM directory:** \`${dicom_dir}\`

---

## Configuration Summary

| Feature | Status |
|---------|--------|
| **GPU Acceleration** | ${has_gpu} (${gpu_device}) |
| **Radiomics Extraction** | ${enable_radiomics} |
| **Radiomics Robustness** | ${enable_robustness} (${robustness_intensity}) |
| **CT Cropping** | ${enable_ct_cropping} (${cropping_region}) |
| **Custom Models** | ${enable_custom_models} |
| **Max CPU Cores** | ${max_cores} |
| **Max Memory** | ${max_memory} |

---

## Quick Start

### 1. Run Pipeline

\`\`\`bash
cd ${project_dir}
./run_docker.sh
\`\`\`

### 2. Monitor Progress

In another terminal:

\`\`\`bash
cd ${project_dir}
./monitor.sh
\`\`\`

Or view logs:

\`\`\`bash
tail -f Logs/*.log
\`\`\`

### 3. Access Results

Results will be in:
- \`Output/_RESULTS/\` - Aggregated results (DVH, radiomics, QC)
- \`Output/<patient>/<course>/\` - Per-patient results

---

## Advanced Usage

### Modify Configuration

Edit \`config.yaml\` to customize:

\`\`\`bash
nano config.yaml
\`\`\`

Key settings:
- \`max_workers\`: Parallelization (null = auto)
- \`radiomics_robustness.enabled\`: Enable/disable robustness
- \`ct_cropping.region\`: Anatomical region
- \`segmentation.fast\`: Fast mode (lower quality)

### Run Specific Targets

\`\`\`bash
# Only run segmentation
./run_docker.sh segmentation

# Only run DVH analysis
./run_docker.sh dvh

# Run everything up to radiomics
./run_docker.sh radiomics
\`\`\`

### Debug Mode

\`\`\`bash
# Run with verbose output
./run_docker.sh --printshellcmds --verbose

# Run with dry-run (show what would be executed)
./run_docker.sh --dryrun
\`\`\`

### Re-run Specific Stages

To force re-running completed stages:

\`\`\`bash
# Delete stage marker files
rm -rf Output/*/.segmentation_done
rm -rf Output/*/.dvh_done
rm -rf Output/*/.radiomics_done

# Re-run
./run_docker.sh
\`\`\`

---

## Troubleshooting

### Permission Errors

If you see "Permission denied" errors:

\`\`\`bash
# Fix ownership (container runs as UID 1000)
sudo chown -R 1000:1000 Output/ Logs/
\`\`\`

### Out of Memory

Reduce parallelism in \`config.yaml\`:

\`\`\`yaml
max_workers: 4  # Lower value

segmentation:
  workers: 1  # One at a time
\`\`\`

### GPU Not Working

Check NVIDIA Docker runtime:

\`\`\`bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
\`\`\`

If fails, install nvidia-container-toolkit:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

### Pipeline Hangs

Increase timeouts in \`docker-compose.project.yml\`:

\`\`\`yaml
environment:
  - TOTALSEG_TIMEOUT=14400  # 4 hours
  - RTPIPELINE_RADIOMICS_TASK_TIMEOUT=3600  # 1 hour per ROI
\`\`\`

---

## Documentation

- **Docker Setup Guide:** [docs/DOCKER_SETUP_GUIDE.md](https://github.com/kstawiski/rtpipeline/blob/main/docs/DOCKER_SETUP_GUIDE.md)
- **General Documentation:** [README.md](https://github.com/kstawiski/rtpipeline/blob/main/README.md)
- **Output Format:** [output_format_quick_ref.md](https://github.com/kstawiski/rtpipeline/blob/main/output_format_quick_ref.md)
- **Troubleshooting:** [docs/TROUBLESHOOTING.md](https://github.com/kstawiski/rtpipeline/blob/main/docs/TROUBLESHOOTING.md)

---

## What Features Are Enabled?

EOF

    # Add feature explanations
    if [ "$enable_radiomics" = "yes" ]; then
        cat >> "$readme_path" <<'EOF'

### ✅ Radiomics Extraction

Extracts 100+ radiomic features per ROI:
- **Shape features**: Volume, surface area, compactness, sphericity
- **First-order features**: Mean, median, variance, energy, entropy
- **Texture features**: GLCM, GLRLM, GLSZM, GLDM, NGTDM

**Output:** `Output/_RESULTS/radiomics_ct.xlsx`, `Output/_RESULTS/radiomics_mr.xlsx`
EOF
    fi

    if [ "$enable_robustness" = "yes" ]; then
        cat >> "$readme_path" <<EOF

### ✅ Radiomics Robustness (${robustness_intensity})

Evaluates feature stability via systematic perturbations:
- **Volume changes:** ±15% erosion/dilation
- **Translations:** ±3mm shifts
- **Contour randomization:** 2 realizations
- **Metrics:** ICC (≥0.90 = robust), CoV (≤10% = robust), QCD

**Processing time:** ~${robustness_intensity} intensity adds 3-10x processing time

**Output:** \`Output/<patient>/<course>/radiomics_robustness/\`
EOF
    fi

    if [ "$enable_ct_cropping" = "yes" ]; then
        cat >> "$readme_path" <<EOF

### ✅ Systematic CT Cropping (${cropping_region})

Crops all CTs to consistent anatomical boundaries:
- **Region:** ${cropping_region}
- **Effect:** Makes percentage DVH metrics (V95%, V20Gy) comparable across patients
- **Original files:** Preserved (keep_original: true)

**Output:** Cropped NIfTI files with \`_cropped\` suffix
EOF
    fi

    cat >> "$readme_path" <<'EOF'

### ✅ TotalSegmentator

Automatic segmentation:
- **CT:** 104 anatomical structures
- **MR:** 59 anatomical structures
- **Output formats:** NIfTI masks + DICOM RTSTRUCT

### ✅ DVH Analysis

Dose-volume histograms for all structures:
- Absolute and relative metrics
- Provenance tracking (manual/TotalSegmentator/custom)

**Output:** `Output/_RESULTS/dvh_metrics.xlsx`

---

## Next Steps

1. **Verify DICOM files** are in the input directory
2. **Review config.yaml** if needed
3. **Run the pipeline:** `./run_docker.sh`
4. **Monitor progress:** `./monitor.sh` or check `Logs/`
5. **Analyze results:** Check `Output/_RESULTS/`

---

For help, see documentation or open an issue:
https://github.com/kstawiski/rtpipeline/issues
EOF

    print_success "Created README.md"

    # Final summary
    print_header "Setup Complete! 🎉"
    echo ""
    echo -e "${BOLD}Your Docker project is ready!${NC}"
    echo ""
    echo -e "${CYAN}Project directory:${NC}"
    echo -e "  ${project_dir}"
    echo ""
    echo -e "${CYAN}Generated files:${NC}"
    echo -e "  ${ICON_CHECK} config.yaml"
    echo -e "  ${ICON_CHECK} docker-compose.project.yml"
    echo -e "  ${ICON_CHECK} run_docker.sh"
    echo -e "  ${ICON_CHECK} monitor.sh"
    echo -e "  ${ICON_CHECK} README.md"
    echo ""
    echo -e "${CYAN}Input DICOM:${NC}"
    echo -e "  ${dicom_dir}"
    echo ""
    echo -e "${CYAN}Configuration:${NC}"
    echo -e "  GPU: ${gpu_device}"
    echo -e "  Radiomics: ${enable_radiomics}"
    echo -e "  Robustness: ${enable_robustness} (${robustness_intensity})"
    echo -e "  CT Cropping: ${enable_ct_cropping} (${cropping_region})"
    echo ""
    echo -e "${BOLD}${GREEN}Next steps:${NC}"
    echo -e "  1. ${CYAN}cd ${project_dir}${NC}"
    echo -e "  2. ${CYAN}./run_docker.sh${NC}"
    echo ""
    echo -e "Or review the generated ${CYAN}README.md${NC} for detailed instructions."
    echo ""
}

main "$@"
