#!/bin/bash
#
# RTpipeline Interactive Project Setup v2.0
#
# This script helps you set up the RTpipeline for processing DICOM files
# in a new directory by creating a config.yaml and run_pipeline.sh script.
#
# Usage:
#   ./setup_new_project.sh                    # Interactive mode
#   ./setup_new_project.sh --quick /path      # Quick mode with defaults
#   ./setup_new_project.sh --preset prostate  # Use preset configuration
#   ./setup_new_project.sh --dry-run          # Preview without creating files
#   ./setup_new_project.sh --validate FILE    # Validate existing config
#   ./setup_new_project.sh --edit FILE        # Edit existing config
#

set -euo pipefail

# Script version
VERSION="2.0.0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Mode flags
DRY_RUN=false
QUICK_MODE=false
QUICK_DICOM_DIR=""
VALIDATE_MODE=false
EDIT_MODE=false
PRESET=""
CONFIG_FILE=""
PROGRESS_FILE="/tmp/rtpipeline_setup_progress_$$.json"
VERBOSE=0

# ============================================================================
# Parse Command Line Arguments
# ============================================================================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --quick)
                QUICK_MODE=true
                shift
                ;;
            --quick=*)
                QUICK_MODE=true
                QUICK_DICOM_DIR="${1#--quick=}"
                shift
                ;;
            --preset)
                PRESET="$2"
                shift 2
                ;;
            --validate)
                VALIDATE_MODE=true
                CONFIG_FILE="$2"
                shift 2
                ;;
            --edit)
                EDIT_MODE=true
                CONFIG_FILE="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=$((VERBOSE + 1))
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            --version)
                echo "RTpipeline Setup Script v$VERSION"
                exit 0
                ;;
            *)
                if [ "$QUICK_MODE" = "true" ] && [ -z "$QUICK_DICOM_DIR" ] && [[ "$1" != -* ]]; then
                    QUICK_DICOM_DIR="$1"
                    shift
                else
                    print_error "Unknown option: $1"
                    show_help
                    exit 1
                fi
                ;;
        esac
    done
}

show_help() {
    cat << 'EOF'
RTpipeline Interactive Project Setup

USAGE:
    ./setup_new_project.sh [OPTIONS]

OPTIONS:
    --quick PATH            Quick setup with sensible defaults (requires DICOM directory PATH)
    --preset NAME           Use preset configuration (prostate, lung, brain, head_neck, thorax)
    --dry-run              Preview configuration without creating files
    --validate FILE        Validate existing config.yaml
    --edit FILE            Edit existing config.yaml interactively
    -v, --verbose          Increase verbosity
    -h, --help             Show this help message
    --version              Show version

PRESETS:
    prostate    Pelvis cropping, bladder/rectum focus
    lung        Thorax cropping, lung/heart focus
    brain       Brain cropping, minimal margins
    head_neck   Head/neck cropping, extended margins
    thorax      Thorax cropping, lung/mediastinum

EXAMPLES:
    # Interactive setup (recommended for first-time users)
    ./setup_new_project.sh

    # Quick setup with defaults
    ./setup_new_project.sh --quick /path/to/dicom

    # Use prostate preset
    ./setup_new_project.sh --preset prostate

    # Preview configuration without creating files
    ./setup_new_project.sh --preset lung --dry-run

    # Validate existing config
    ./setup_new_project.sh --validate /path/to/config.yaml

    # Edit existing config
    ./setup_new_project.sh --edit /path/to/config.yaml
EOF
}

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo -e "\n${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}\n"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_explanation() {
    echo -e "${CYAN}ℹ${NC}  $1"
}

print_debug() {
    if [ $VERBOSE -ge 1 ]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

ask_question() {
    local question="$1"
    local default="$2"
    local response

    if [ -n "$default" ]; then
        read -p "$(echo -e "${GREEN}?${NC}") $question [default: $default]: " response
        echo "${response:-$default}"
    else
        read -p "$(echo -e "${GREEN}?${NC}") $question: " response
        echo "$response"
    fi
}

ask_yes_no() {
    local question="$1"
    local default="$2"
    local response

    if [ "$default" = "yes" ]; then
        read -p "$(echo -e "${GREEN}?${NC}") $question [Y/n]: " response
        response="${response:-y}"
    else
        read -p "$(echo -e "${GREEN}?${NC}") $question [y/N]: " response
        response="${response:-n}"
    fi

    case "$response" in
        [Yy]|[Yy][Ee][Ss]) echo "yes" ;;
        *) echo "no" ;;
    esac
}

check_command() {
    local cmd="$1"
    local name="$2"

    if command -v "$cmd" &> /dev/null; then
        print_info "✓ $name found: $(command -v $cmd)"
        return 0
    else
        print_warning "✗ $name not found"
        return 1
    fi
}

check_python_package() {
    local package="$1"
    local import_name="${2:-$package}"

    if python3 -c "import $import_name" 2>/dev/null; then
        # Try to get version
        local version=$(python3 -c "import $import_name; print(getattr($import_name, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
        print_info "✓ Python package '$package' found (version: $version)"
        return 0
    else
        print_warning "✗ Python package '$package' not found"
        return 1
    fi
}

check_gpu_functional() {
    if command -v nvidia-smi &> /dev/null; then
        print_info "Testing GPU functionality..."

        # Check NVIDIA driver
        if nvidia-smi &> /dev/null; then
            local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
            local gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
            print_info "✓ GPU detected: $gpu_name (${gpu_memory}MB VRAM)"

            # Check PyTorch CUDA
            if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
                print_info "✓ PyTorch CUDA support available"
                return 0
            else
                print_warning "PyTorch CUDA not available (CPU-only mode will be used)"
                return 1
            fi
        else
            print_warning "NVIDIA driver not working properly"
            return 1
        fi
    fi
    return 1
}

save_progress() {
    local step="$1"
    local data="$2"

    print_debug "Saving progress: $step"

    # Create or update progress file
    if [ -f "$PROGRESS_FILE" ]; then
        # Append to existing file
        python3 -c "
import json
try:
    with open('$PROGRESS_FILE', 'r') as f:
        progress = json.load(f)
except:
    progress = {}
progress['$step'] = $data
with open('$PROGRESS_FILE', 'w') as f:
    json.dump(progress, f, indent=2)
"
    else
        echo "{\"$step\": $data}" > "$PROGRESS_FILE"
    fi
}

load_progress() {
    if [ -f "$PROGRESS_FILE" ]; then
        print_info "Found previous setup session. Would you like to resume?"
        local resume=$(ask_yes_no "Resume from last checkpoint?" "yes")

        if [ "$resume" = "yes" ]; then
            return 0
        else
            rm -f "$PROGRESS_FILE"
            return 1
        fi
    fi
    return 1
}

cleanup_progress() {
    if [ -f "$PROGRESS_FILE" ]; then
        rm -f "$PROGRESS_FILE"
        print_debug "Cleaned up progress file"
    fi
}

validate_directory_path() {
    local path="$1"
    local name="$2"

    # Expand path
    path="${path/#\~/$HOME}"

    # Check if path contains spaces (warn but allow)
    if [[ "$path" =~ [[:space:]] ]]; then
        print_warning "$name path contains spaces: '$path'"
        print_warning "This may cause issues with some tools. Consider using paths without spaces."
    fi

    # Check if directory exists
    if [ ! -d "$path" ]; then
        print_warning "$name directory does not exist: $path"
        return 1
    fi

    # Check write permissions
    if [ ! -w "$path" ]; then
        print_error "No write permission for $name directory: $path"
        return 1
    fi

    return 0
}

check_directory_overlap() {
    local dir1="$1"
    local dir2="$2"
    local name1="$3"
    local name2="$4"

    # Resolve to absolute paths
    dir1=$(cd "$dir1" 2>/dev/null && pwd || echo "$dir1")
    dir2=$(cd "$dir2" 2>/dev/null && pwd || echo "$dir2")

    # Normalize paths by removing trailing slashes
    local dir1_norm="${dir1%/}"
    local dir2_norm="${dir2%/}"

    # Check if directories are identical or nested within each other
    if [ "$dir1_norm" = "$dir2_norm" ] || [[ "$dir2_norm/" == "$dir1_norm/"* ]] || [[ "$dir1_norm/" == "$dir2_norm/"* ]]; then
        print_error "$name1 and $name2 overlap!"
        print_error "  $name1: $dir1"
        print_error "  $name2: $dir2"
        print_explanation "This can cause data corruption. Please use separate directories."
        return 1
    fi

    return 0
}

estimate_disk_requirements() {
    local dicom_dir="$1"

    print_info "Estimating disk requirements..."

    # Count DICOM files
    local dicom_count=$(find "$dicom_dir" -type f -name "*.dcm" 2>/dev/null | wc -l)

    if [ "$dicom_count" -eq 0 ]; then
        print_warning "No DICOM files found. Cannot estimate disk requirements."
        return
    fi

    # Estimate total DICOM size
    local dicom_size_kb=$(du -sk "$dicom_dir" 2>/dev/null | awk '{print $1}')
    local dicom_size_gb=$((dicom_size_kb / 1024 / 1024))

    # Rough estimate: output is 3-5x input size
    local estimated_output_gb=$((dicom_size_gb * 4))

    print_info "DICOM files: $dicom_count (${dicom_size_gb}GB)"
    print_info "Estimated output size: ~${estimated_output_gb}GB"
    print_warning "Ensure you have at least ${estimated_output_gb}GB free space in output directory"
}

# ============================================================================
# Preset Configurations
# ============================================================================

load_preset() {
    local preset="$1"

    case "$preset" in
        prostate)
            PRESET_CROP_ENABLED="yes"
            PRESET_CROP_REGION="pelvis"
            PRESET_CROP_SUPERIOR="2.0"
            PRESET_CROP_INFERIOR="10.0"
            PRESET_RADIOMICS="yes"
            PRESET_SKIP_ROIS="body,couchsurface,bones"
            print_info "Loaded prostate preset (pelvis cropping, bladder/rectum focus)"
            ;;
        lung)
            PRESET_CROP_ENABLED="yes"
            PRESET_CROP_REGION="thorax"
            PRESET_CROP_SUPERIOR="2.0"
            PRESET_CROP_INFERIOR="2.0"
            PRESET_RADIOMICS="yes"
            PRESET_SKIP_ROIS="body,couchsurface,bones"
            print_info "Loaded lung preset (thorax cropping)"
            ;;
        brain)
            PRESET_CROP_ENABLED="yes"
            PRESET_CROP_REGION="brain"
            PRESET_CROP_SUPERIOR="1.0"
            PRESET_CROP_INFERIOR="1.0"
            PRESET_RADIOMICS="yes"
            PRESET_SKIP_ROIS="body,couchsurface"
            print_info "Loaded brain preset (brain cropping, minimal margins)"
            ;;
        head_neck)
            PRESET_CROP_ENABLED="yes"
            PRESET_CROP_REGION="head_neck"
            PRESET_CROP_SUPERIOR="2.0"
            PRESET_CROP_INFERIOR="2.0"
            PRESET_RADIOMICS="yes"
            PRESET_SKIP_ROIS="body,couchsurface,bones"
            print_info "Loaded head/neck preset"
            ;;
        thorax)
            PRESET_CROP_ENABLED="yes"
            PRESET_CROP_REGION="thorax"
            PRESET_CROP_SUPERIOR="2.0"
            PRESET_CROP_INFERIOR="2.0"
            PRESET_RADIOMICS="yes"
            PRESET_SKIP_ROIS="body,couchsurface,bones"
            print_info "Loaded thorax preset"
            ;;
        *)
            print_error "Unknown preset: $preset"
            print_info "Available presets: prostate, lung, brain, head_neck, thorax"
            exit 1
            ;;
    esac
}

# ============================================================================
# Configuration Validation
# ============================================================================

validate_config_file() {
    local config_file="$1"

    print_header "Validating Configuration"

    if [ ! -f "$config_file" ]; then
        print_error "Config file not found: $config_file"
        return 1
    fi

    local errors=0

    # Check YAML syntax
    print_info "Checking YAML syntax..."
    if python3 -c "import yaml; yaml.safe_load(open('$config_file'))" 2>/dev/null; then
        print_success "✓ Valid YAML syntax"
    else
        print_error "✗ Invalid YAML syntax"
        errors=$((errors + 1))
    fi

    # Check required fields
    print_info "Checking required fields..."
    local required_fields=("dicom_root" "output_dir" "logs_dir" "workers")

    for field in "${required_fields[@]}"; do
        if grep -q "^${field}:" "$config_file"; then
            print_success "✓ Found required field: $field"
        else
            print_error "✗ Missing required field: $field"
            errors=$((errors + 1))
        fi
    done

    # Check directory existence
    print_info "Checking directory paths..."
    local dicom_root=$(grep "^dicom_root:" "$config_file" | awk '{print $2}' | tr -d '"')
    local output_dir=$(grep "^output_dir:" "$config_file" | awk '{print $2}' | tr -d '"')
    local logs_dir=$(grep "^logs_dir:" "$config_file" | awk '{print $2}' | tr -d '"')

    if [ -d "$dicom_root" ]; then
        print_success "✓ DICOM directory exists: $dicom_root"
    else
        print_warning "⚠ DICOM directory not found: $dicom_root"
    fi

    # Check for custom_structures file
    local custom_struct=$(grep "^custom_structures:" "$config_file" | awk '{print $2}' | tr -d '"')
    if [ -n "$custom_struct" ] && [ "$custom_struct" != "null" ]; then
        if [ -f "$SCRIPT_DIR/$custom_struct" ]; then
            print_success "✓ Custom structures file exists: $custom_struct"
        else
            print_warning "⚠ Custom structures file not found: $custom_struct"
        fi
    fi

    echo ""
    if [ $errors -eq 0 ]; then
        print_success "Configuration validation PASSED"
        return 0
    else
        print_error "Configuration validation FAILED with $errors error(s)"
        return 1
    fi
}

# ============================================================================
# Prerequisites Check (Enhanced)
# ============================================================================

check_prerequisites() {
    print_header "Checking Prerequisites"

    local missing_prereqs=0
    local optional_missing=0

    # Essential checks
    print_info "Checking essential prerequisites..."

    if ! check_command "python3" "Python 3"; then
        print_error "Python 3 is required. Please install Python 3.11 or later."
        missing_prereqs=$((missing_prereqs + 1))
    else
        python_version=$(python3 --version | awk '{print $2}')
        print_info "Python version: $python_version"

        # Check if version >= 3.11
        if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"; then
            print_success "✓ Python version is 3.11 or later"
        else
            print_warning "Python 3.11+ recommended (you have $python_version)"
        fi
    fi

    # Check for Conda (recommended)
    if check_command "conda" "Conda"; then
        conda_version=$(conda --version 2>/dev/null || echo "unknown")
        print_info "Conda version: $conda_version"
    else
        print_warning "Conda not found. Conda is recommended for environment management."
        optional_missing=$((optional_missing + 1))
    fi

    # Check for Docker (alternative method)
    if check_command "docker" "Docker"; then
        docker_version=$(docker --version 2>/dev/null || echo "unknown")
        print_info "Docker version: $docker_version"

        # Check for docker-compose
        if check_command "docker-compose" "Docker Compose"; then
            docker_compose_version=$(docker-compose --version 2>/dev/null || echo "unknown")
            print_info "Docker Compose version: $docker_compose_version"
        fi

        # Check for NVIDIA Docker (GPU support)
        check_gpu_functional || true
    else
        print_warning "Docker not found. Docker provides easier deployment."
        optional_missing=$((optional_missing + 1))
    fi

    # Check for Snakemake
    if check_command "snakemake" "Snakemake"; then
        snakemake_version=$(snakemake --version 2>/dev/null || echo "unknown")
        print_info "Snakemake version: $snakemake_version"
    else
        print_warning "Snakemake not found. Required for workflow execution."
        print_info "Install with: conda install -c bioconda snakemake"
        missing_prereqs=$((missing_prereqs + 1))
    fi

    # Check Python packages (if Python available)
    if command -v python3 &> /dev/null; then
        print_info "\nChecking Python packages..."
        check_python_package "numpy" "numpy" || missing_prereqs=$((missing_prereqs + 1))
        check_python_package "pandas" "pandas" || missing_prereqs=$((missing_prereqs + 1))
        check_python_package "pydicom" "pydicom" || missing_prereqs=$((missing_prereqs + 1))
        check_python_package "SimpleITK" "SimpleITK" || optional_missing=$((optional_missing + 1))
        check_python_package "yaml" "yaml" || optional_missing=$((optional_missing + 1))
    fi

    # Check disk space
    print_info "\nChecking disk space..."
    available_space=$(df -BG "$SCRIPT_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')
    print_info "Available disk space: ${available_space}GB"

    if [ "$available_space" -lt 20 ]; then
        print_warning "Less than 20GB available. Processing may require significant space."
    else
        print_success "✓ Sufficient disk space available"
    fi

    # Summary
    echo ""
    if [ $missing_prereqs -gt 0 ]; then
        print_error "$missing_prereqs critical prerequisites missing!"
        print_info "Please install missing prerequisites before continuing."

        offer_install=$(ask_yes_no "Would you like guidance on installing missing prerequisites?" "no")
        if [ "$offer_install" = "yes" ]; then
            show_installation_guide
        fi

        exit 1
    else
        print_success "✓ All critical prerequisites satisfied!"

        if [ $optional_missing -gt 0 ]; then
            print_warning "$optional_missing optional prerequisites missing."
            print_info "Pipeline will work but some features may be limited."
        fi
    fi

    echo ""
    if [ "$QUICK_MODE" = "false" ] && [ "$DRY_RUN" = "false" ]; then
        read -p "Press Enter to continue with setup..."
    fi
}

show_installation_guide() {
    print_header "Installation Guide"

    cat << 'EOF'
To install RTpipeline prerequisites:

1. Install Miniconda/Anaconda:
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh

2. Create RTpipeline environment:
   cd /path/to/rtpipeline
   conda env create -f envs/rtpipeline.yaml -n rtpipeline
   conda activate rtpipeline

3. Install Snakemake:
   conda install -c bioconda snakemake

4. (Optional) Install Docker:
   # Follow instructions at https://docs.docker.com/engine/install/

5. (Optional) Install NVIDIA Docker for GPU:
   # Follow instructions at https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

For more details, see the README.md in the rtpipeline repository.
EOF
}

# ============================================================================
# Configuration (Enhanced)
# ============================================================================

configure_project() {
    print_header "Project Configuration"

    # Get DICOM directory
    if [ "$QUICK_MODE" = "false" ]; then
        print_explanation "Specify the directory containing your DICOM files."
        print_explanation "This should contain subdirectories for each patient or a flat structure."
        echo ""
    fi

    local dicom_dir
    while true; do
        if [ "$QUICK_MODE" = "true" ] && [ $# -gt 0 ]; then
            dicom_dir="$1"
            shift
        else
            dicom_dir=$(ask_question "Path to DICOM directory" "")
        fi

        if [ -z "$dicom_dir" ]; then
            print_error "DICOM directory cannot be empty"
            continue
        fi

        # Expand path
        dicom_dir="${dicom_dir/#\~/$HOME}"

        if [ ! -d "$dicom_dir" ]; then
            print_error "Directory does not exist: $dicom_dir"
            create_dir=$(ask_yes_no "Create directory?" "no")
            if [ "$create_dir" = "yes" ]; then
                mkdir -p "$dicom_dir"
                print_info "Created directory: $dicom_dir"
                break
            fi
        else
            # Validate directory
            if validate_directory_path "$dicom_dir" "DICOM"; then
                # Check if directory contains DICOM files
                dicom_count=$(find "$dicom_dir" -type f -name "*.dcm" 2>/dev/null | wc -l)
                if [ "$dicom_count" -eq 0 ]; then
                    print_warning "No .dcm files found in $dicom_dir"
                    confirm=$(ask_yes_no "Continue anyway?" "yes")
                    if [ "$confirm" = "yes" ]; then
                        break
                    fi
                else
                    print_info "Found $dicom_count DICOM files"

                    # Estimate disk requirements
                    estimate_disk_requirements "$dicom_dir"
                    break
                fi
            fi
        fi
    done

    # Get output directory
    if [ "$QUICK_MODE" = "false" ]; then
        print_explanation "\nSpecify where to store pipeline results."
        print_explanation "This will contain processed data, segmentations, and analysis results."
        echo ""
    fi

    local output_dir
    output_dir=$(ask_question "Output directory name" "$(dirname "$dicom_dir")/rtpipeline_output")
    output_dir="${output_dir/#\~/$HOME}"

    # Get logs directory
    local logs_dir
    logs_dir=$(ask_question "Logs directory name" "$(dirname "$dicom_dir")/rtpipeline_logs")
    logs_dir="${logs_dir/#\~/$HOME}"

    # Check for directory overlap (BUG FIX #3)
    if ! check_directory_overlap "$dicom_dir" "$output_dir" "DICOM" "Output"; then
        print_error "Cannot continue with overlapping directories"
        exit 1
    fi

    # Create directories (check permissions first - BUG FIX #4)
    for dir in "$output_dir" "$logs_dir"; do
        parent_dir=$(dirname "$dir")
        if [ ! -w "$parent_dir" ]; then
            print_error "No write permission in parent directory: $parent_dir"
            exit 1
        fi
    done

    mkdir -p "$output_dir"
    mkdir -p "$logs_dir"
    print_info "Created output directory: $output_dir"
    print_info "Created logs directory: $logs_dir"

    # Worker configuration
    if [ "$QUICK_MODE" = "false" ]; then
        print_header "Performance Configuration"
        print_explanation "Workers control parallel processing."
        print_explanation "More workers = faster processing but more CPU/memory usage."
    fi

    cpu_count=$(python3 -c "import os; print(os.cpu_count() or 4)")
    print_info "Detected $cpu_count CPU cores"

    local workers
    if [ "$QUICK_MODE" = "true" ]; then
        workers="auto"
    else
        workers=$(ask_question "Number of workers" "auto")
    fi

    # Segmentation configuration
    if [ "$QUICK_MODE" = "false" ]; then
        print_header "Segmentation Configuration"
        print_explanation "\nTotalSegmentator automatically segments 100+ organs/tissues."
        print_explanation "It can use GPU (faster) or CPU (slower but works everywhere)."
        echo ""
    fi

    local seg_device
    if check_gpu_functional &>/dev/null; then
        if [ "$QUICK_MODE" = "true" ]; then
            seg_device="gpu"
        else
            print_info "NVIDIA GPU detected"
            seg_device=$(ask_question "Segmentation device (gpu/cpu)" "gpu")
        fi
    else
        if [ "$QUICK_MODE" = "false" ]; then
            print_warning "No NVIDIA GPU detected"
        fi
        seg_device="cpu"
    fi

    local seg_workers
    if [ "$QUICK_MODE" = "true" ]; then
        seg_workers="1"
    else
        print_explanation "\nSegmentation workers: ALWAYS use 1 for GPU safety (prevents GPU OOM)."
        print_explanation "Parallelization happens at the course level via Snakemake, not per-worker."
        seg_workers=$(ask_question "Segmentation workers" "1")
    fi

    local seg_fast seg_force_split
    if [ "$QUICK_MODE" = "true" ]; then
        seg_fast="no"
        seg_force_split="yes"
    else
        print_explanation "\nFast mode: Uses CPU-friendly settings (lower quality but faster)."
        seg_fast=$(ask_yes_no "Enable fast mode?" "no")

        print_explanation "\nForce split: Chunks inference to reduce memory usage (recommended)."
        seg_force_split=$(ask_yes_no "Enable force split?" "yes")
    fi

    # Radiomics configuration
    local enable_radiomics radiomics_threads skip_rois

    if [ -n "$PRESET" ]; then
        enable_radiomics="${PRESET_RADIOMICS:-yes}"
        skip_rois="${PRESET_SKIP_ROIS:-body,couchsurface,bones}"
    elif [ "$QUICK_MODE" = "true" ]; then
        enable_radiomics="yes"
        skip_rois="body,couchsurface,bones"
    else
        print_header "Radiomics Configuration"
        print_explanation "\nRadiomics extracts 150+ quantitative features from images."
        print_explanation "This includes shape, intensity, and texture features."
        enable_radiomics=$(ask_yes_no "Enable radiomics feature extraction?" "yes")
    fi

    # BUG FIX #6: Always set radiomics_threads (with default if disabled)
    if [ "$enable_radiomics" = "yes" ]; then
        if [ "$QUICK_MODE" = "true" ]; then
            radiomics_threads="4"
        else
            print_explanation "\nThread limit: CPU threads per radiomics worker."
            radiomics_threads=$(ask_question "Radiomics thread limit" "4")

            print_explanation "\nSome structures should be skipped (e.g., body, couch, bones)."
            print_info "Default skip list: body, couchsurface, bones"
            local custom_skip
            custom_skip=$(ask_yes_no "Use custom skip list?" "no")

            if [ "$custom_skip" = "yes" ]; then
                print_explanation "Enter comma-separated structure names to skip:"
                read -p "> " skip_rois_input
                skip_rois="$skip_rois_input"
            else
                skip_rois="body,couchsurface,couchinterior,couchexterior,bones,m1,m2"
            fi
        fi
    else
        radiomics_threads="4"  # Set default even when disabled
        skip_rois="body,couchsurface,bones"
    fi

    # CT Cropping configuration
    local enable_cropping crop_region crop_superior crop_inferior crop_use_dvh crop_use_radiomics crop_keep_orig

    if [ -n "$PRESET" ]; then
        enable_cropping="${PRESET_CROP_ENABLED:-no}"
        crop_region="${PRESET_CROP_REGION:-pelvis}"
        crop_superior="${PRESET_CROP_SUPERIOR:-2.0}"
        crop_inferior="${PRESET_CROP_INFERIOR:-10.0}"
        crop_use_dvh="yes"
        crop_use_radiomics="yes"
        crop_keep_orig="yes"
    elif [ "$QUICK_MODE" = "true" ]; then
        enable_cropping="no"
        crop_region="pelvis"
        crop_superior="2.0"
        crop_inferior="10.0"
        crop_use_dvh="yes"
        crop_use_radiomics="yes"
        crop_keep_orig="yes"
    else
        print_header "CT Cropping Configuration"
        print_explanation "\nSystematic CT cropping normalizes anatomical volumes across patients."
        print_explanation "This makes percentage DVH metrics (V95%, V50Gy) meaningful."
        print_explanation "Crops all CTs to the same anatomical boundaries (e.g., L1 to femoral heads)."
        echo ""

        enable_cropping=$(ask_yes_no "Enable systematic CT cropping?" "no")

        if [ "$enable_cropping" = "yes" ]; then
            print_explanation "\nSupported regions:"
            print_explanation "  - pelvis: L1 vertebra to femoral heads (for prostate, bladder, rectum)"
            print_explanation "  - thorax: C7/lung apex to L1/diaphragm (for lung, esophagus)"
            print_explanation "  - abdomen: T12/L1 to L5 vertebra (for liver, pancreas)"
            print_explanation "  - head_neck: Brain/skull apex to C7/clavicles"
            print_explanation "  - brain: Brain boundaries (minimal margins)"
            crop_region=$(ask_question "Cropping region" "pelvis")

            print_explanation "\nMargins (in cm) add buffer around anatomical landmarks."
            crop_superior=$(ask_question "Superior margin (cm)" "2.0")
            crop_inferior=$(ask_question "Inferior margin (cm)" "10.0")

            crop_use_dvh=$(ask_yes_no "Use cropped volumes for DVH analysis?" "yes")
            crop_use_radiomics=$(ask_yes_no "Use cropped volumes for radiomics?" "yes")
            crop_keep_orig=$(ask_yes_no "Keep original uncropped files?" "yes")
        else
            # Set defaults
            crop_region="pelvis"
            crop_superior="2.0"
            crop_inferior="10.0"
            crop_use_dvh="yes"
            crop_use_radiomics="yes"
            crop_keep_orig="yes"
        fi
    fi

    # Custom structures
    local custom_structures_file
    if [ "$QUICK_MODE" = "true" ]; then
        custom_structures_file="custom_structures_pelvic.yaml"
    else
        print_header "Custom Structures Configuration"
        print_explanation "\nCustom structures allow Boolean operations (union, intersection, margins)."
        print_explanation "Example: Combine left+right iliac arteries into single structure."
        custom_structures_file=$(ask_question "Custom structures config file" "custom_structures_pelvic.yaml")
    fi

    # BUG FIX #2: Validate custom structures file existence
    if [ ! -f "$SCRIPT_DIR/$custom_structures_file" ]; then
        print_warning "Custom structures file not found: $SCRIPT_DIR/$custom_structures_file"
        if [ "$QUICK_MODE" = "false" ]; then
            use_anyway=$(ask_yes_no "Continue without custom structures?" "yes")
            if [ "$use_anyway" = "no" ]; then
                print_info "Please create the custom structures file or specify an existing one."
                exit 1
            fi
        fi
    else
        print_success "✓ Custom structures file found: $custom_structures_file"
    fi

    # Custom models
    local enable_custom_models
    if [ "$QUICK_MODE" = "true" ]; then
        enable_custom_models="no"
    else
        print_header "Custom Models Configuration"
        print_explanation "\nCustom models: Use your own trained nnUNet models for segmentation."
        print_explanation "WARNING: This requires model weights and proper setup."
        enable_custom_models=$(ask_yes_no "Enable custom models?" "no")
    fi

    # Generate config
    if [ "$DRY_RUN" = "true" ]; then
        print_header "Dry Run - Configuration Preview"
        print_info "The following configuration would be generated:"
        echo ""
    else
        print_header "Generating Configuration"
    fi

    generate_config_file \
        "$dicom_dir" \
        "$output_dir" \
        "$logs_dir" \
        "$workers" \
        "$seg_device" \
        "$seg_workers" \
        "$seg_fast" \
        "$seg_force_split" \
        "$enable_radiomics" \
        "$radiomics_threads" \
        "$skip_rois" \
        "$enable_cropping" \
        "$crop_region" \
        "$crop_superior" \
        "$crop_inferior" \
        "$crop_use_dvh" \
        "$crop_use_radiomics" \
        "$crop_keep_orig" \
        "$custom_structures_file" \
        "$enable_custom_models" \
        "$DRY_RUN"

    if [ "$DRY_RUN" = "false" ]; then
        # Generate run script
        generate_run_script "$dicom_dir" "$output_dir"

        # Validate generated config
        if validate_config_file "$dicom_dir/config.yaml"; then
            print_success "Configuration validated successfully!"
        fi

        # Summary
        print_header "Setup Complete!"

        print_info "Configuration saved to: $dicom_dir/config.yaml"
        print_info "Run script saved to: $dicom_dir/run_pipeline.sh"

        echo ""
        print_info "To run the pipeline:"
        echo ""
        echo -e "${GREEN}cd \"$dicom_dir\"${NC}"  # BUG FIX #5: Quote paths with spaces
        echo -e "${GREEN}chmod +x run_pipeline.sh${NC}"
        echo -e "${GREEN}./run_pipeline.sh${NC}"
        echo ""

        print_info "Or use Snakemake directly:"
        echo ""
        echo -e "${GREEN}cd \"$SCRIPT_DIR\"${NC}"
        echo -e "${GREEN}snakemake --cores all --use-conda --configfile \"$dicom_dir/config.yaml\"${NC}"
        echo ""

        if [ "$enable_radiomics" = "yes" ]; then
            print_warning "Note: Radiomics requires the rtpipeline-radiomics conda environment."
            print_info "The workflow will automatically switch environments as needed."
        fi

        echo ""
        print_info "Expected output:"
        print_info "  - Aggregated results: $output_dir/_RESULTS/"
        print_info "  - Per-patient data: $output_dir/{PatientID}/{CourseID}/"
        print_info "  - Logs: $logs_dir/"

        echo ""
        print_explanation "For more information, see output_format.md in the rtpipeline repository."

        # Cleanup progress file on success
        cleanup_progress
    else
        print_info "\nDry run complete. No files were created."
        print_info "Remove --dry-run flag to create actual configuration."
    fi
}

generate_config_file() {
    local dicom_dir="$1"
    local output_dir="$2"
    local logs_dir="$3"
    local workers="$4"
    local seg_device="$5"
    local seg_workers="$6"
    local seg_fast="$7"
    local seg_force_split="$8"
    local enable_radiomics="$9"
    local radiomics_threads="${10}"
    local skip_rois="${11}"
    local enable_cropping="${12}"
    local crop_region="${13}"
    local crop_superior="${14}"
    local crop_inferior="${15}"
    local crop_use_dvh="${16}"
    local crop_use_radiomics="${17}"
    local crop_keep_orig="${18}"
    local custom_structures_file="${19}"
    local enable_custom_models="${20}"
    local dry_run="${21:-false}"

    # Convert yes/no to true/false
    [ "$seg_fast" = "yes" ] && seg_fast="true" || seg_fast="false"
    [ "$seg_force_split" = "yes" ] && seg_force_split="true" || seg_force_split="false"
    [ "$enable_cropping" = "yes" ] && enable_cropping="true" || enable_cropping="false"
    [ "$crop_use_dvh" = "yes" ] && crop_use_dvh="true" || crop_use_dvh="false"
    [ "$crop_use_radiomics" = "yes" ] && crop_use_radiomics="true" || crop_use_radiomics="false"
    [ "$crop_keep_orig" = "yes" ] && crop_keep_orig="true" || crop_keep_orig="false"
    [ "$enable_custom_models" = "yes" ] && enable_custom_models="true" || enable_custom_models="false"

    # BUG FIX #1: Properly escape special characters in YAML
    # Convert skip_rois to YAML list with proper quoting
    local skip_rois_yaml=""
    if [ -n "$skip_rois" ]; then
        IFS=',' read -ra SKIP_ARRAY <<< "$skip_rois"
        for roi in "${SKIP_ARRAY[@]}"; do
            # Trim whitespace and quote if necessary
            roi=$(echo "$roi" | xargs)
            # Escape special YAML characters
            roi=$(echo "$roi" | sed 's/"/\\"/g')
            skip_rois_yaml="${skip_rois_yaml}    - \"${roi}\"\n"
        done
    else
        skip_rois_yaml="    - body\n    - couchsurface\n"
    fi

    local config_content="# RTpipeline Configuration
# Generated by setup_new_project.sh v$VERSION on $(date)

# Input/Output directories
dicom_root: \"$dicom_dir\"
output_dir: \"$output_dir\"
logs_dir: \"$logs_dir\"

# Processing parameters
workers: $workers

segmentation:
  workers: $seg_workers
  force: false
  fast: $seg_fast
  roi_subset: null
  extra_models: []
  device: \"$seg_device\"
  force_split: $seg_force_split
  nr_threads_resample: 1
  nr_threads_save: 1
  num_proc_preprocessing: 1
  num_proc_export: 1

custom_models:
  enabled: $enable_custom_models
  root: \"custom_models\"
  models: []
  workers: 1
  force: false
  nnunet_predict: \"nnUNet_predict\"
  retain_weights: true
  conda_activate: null

radiomics:
  sequential: false
  params_file: \"$SCRIPT_DIR/rtpipeline/radiomics_params.yaml\"
  mr_params_file: \"$SCRIPT_DIR/rtpipeline/radiomics_params_mr.yaml\"
  thread_limit: $radiomics_threads
  skip_rois:
$(echo -e "$skip_rois_yaml")
  max_voxels: 1500000000
  min_voxels: 10

aggregation:
  threads: auto

environments:
  main: \"rtpipeline\"
  radiomics: \"rtpipeline-radiomics\"

custom_structures: \"$custom_structures_file\"

ct_cropping:
  enabled: $enable_cropping
  region: \"$crop_region\"
  superior_margin_cm: $crop_superior
  inferior_margin_cm: $crop_inferior
  use_cropped_for_dvh: $crop_use_dvh
  use_cropped_for_radiomics: $crop_use_radiomics
  keep_original: $crop_keep_orig
"

    if [ "$dry_run" = "true" ]; then
        echo "$config_content"
    else
        echo "$config_content" > "$dicom_dir/config.yaml"
        print_success "Generated config.yaml"
    fi
}

generate_run_script() {
    local dicom_dir="$1"
    local output_dir="$2"

    # BUG FIX #5: Properly quote paths with spaces
    cat <<RUNSCRIPT > "$dicom_dir/run_pipeline.sh"
#!/bin/bash
#
# RTpipeline Execution Script
# Generated by setup_new_project.sh
#

set -euo pipefail

# Get script directory
SCRIPT_DIR="\$( cd "\$( dirname "\${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_FILE="\$SCRIPT_DIR/config.yaml"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "\${GREEN}Starting RTpipeline...\${NC}"
echo ""

# Check if config exists
if [ ! -f "\$CONFIG_FILE" ]; then
    echo -e "\${RED}Error: config.yaml not found in \$SCRIPT_DIR\${NC}"
    exit 1
fi

# Check if rtpipeline repository is accessible
RTPIPELINE_DIR="$SCRIPT_DIR"
if [ ! -d "\$RTPIPELINE_DIR" ]; then
    echo -e "\${RED}Error: RTpipeline repository not found at \$RTPIPELINE_DIR\${NC}"
    echo "Please update RTPIPELINE_DIR in this script to point to the rtpipeline repository."
    exit 1
fi

cd "\$RTPIPELINE_DIR"

# Check for running processes
if pgrep -f "snakemake" >/dev/null; then
    echo -e "\${YELLOW}Warning: Another snakemake process is currently running.\${NC}"
    read -p "Stop it and continue? [y/N]: " response
    if [[ "\$response" =~ ^[Yy]$ ]]; then
        pkill -f "snakemake" || true
        sleep 2
    else
        echo "Aborted."
        exit 1
    fi
fi

# Unlock workflow (in case of previous crash)
echo "Unlocking workflow (if needed)..."
snakemake --unlock --configfile "\$CONFIG_FILE" >/dev/null 2>&1 || true

# Run pipeline
echo "Running Snakemake workflow..."
echo "Config: \$CONFIG_FILE"
echo ""

snakemake \\
    --cores all \\
    --use-conda \\
    --configfile "\$CONFIG_FILE" \\
    --rerun-incomplete \\
    --keep-going \\
    --printshellcmds

EXITCODE=\$?

echo ""
if [ \$EXITCODE -eq 0 ]; then
    echo -e "\${GREEN}Pipeline completed successfully!\${NC}"
    echo ""
    echo "Results available at:"
    grep "output_dir:" "\$CONFIG_FILE" | awk '{print $2}' | tr -d '"' | xargs echo "  "
    echo ""
    echo "Check aggregated results in: {output_dir}/_RESULTS/"
else
    echo -e "\${RED}Pipeline failed with exit code \$EXITCODE\${NC}"
    echo ""
    echo "Check logs at:"
    grep "logs_dir:" "\$CONFIG_FILE" | awk '{print $2}' | tr -d '"' | xargs echo "  "
fi

exit \$EXITCODE
RUNSCRIPT

    chmod +x "$dicom_dir/run_pipeline.sh"

    print_success "Generated run_pipeline.sh (executable)"
}

# ============================================================================
# Main
# ============================================================================

main() {
    # Parse command line arguments
    parse_arguments "$@"

    # Handle validation mode
    if [ "$VALIDATE_MODE" = "true" ]; then
        validate_config_file "$CONFIG_FILE"
        exit $?
    fi

    # Handle edit mode
    if [ "$EDIT_MODE" = "true" ]; then
        print_error "Edit mode not yet implemented"
        print_info "For now, manually edit: $CONFIG_FILE"
        exit 1
    fi

    # Validate quick mode requires DICOM directory argument
    if [ "$QUICK_MODE" = "true" ] && [ -z "$QUICK_DICOM_DIR" ]; then
        print_error "Quick mode requires a DICOM directory path"
        echo "Usage: $0 --quick /path/to/dicom"
        exit 1
    fi

    clear

    cat << 'EOF'
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║           RTpipeline Interactive Project Setup v2.0            ║
║                                                                ║
║  This wizard will help you configure the RTpipeline for        ║
║  processing DICOM radiotherapy data in a new directory.        ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
EOF

    echo ""

    if [ "$DRY_RUN" = "true" ]; then
        print_warning "DRY RUN MODE - No files will be created"
    fi

    if [ "$QUICK_MODE" = "true" ]; then
        print_info "QUICK MODE - Using sensible defaults"
    fi

    if [ -n "$PRESET" ]; then
        print_info "PRESET MODE - Using '$PRESET' configuration"
        load_preset "$PRESET"
    fi

    echo ""

    if [ "$QUICK_MODE" = "false" ] && [ "$DRY_RUN" = "false" ]; then
        read -p "Press Enter to begin..."
    fi

    # Check if we can resume
    if [ "$QUICK_MODE" = "false" ] && [ "$DRY_RUN" = "false" ]; then
        load_progress || true
    fi

    # Step 1: Check prerequisites
    check_prerequisites

    # Step 2: Configure project
    if [ "$QUICK_MODE" = "true" ]; then
        configure_project "$QUICK_DICOM_DIR"
    else
        configure_project
    fi

    echo ""
    print_success "Setup wizard completed successfully!"
    echo ""
}

# Trap errors and cleanup
trap cleanup_progress EXIT

# Run main function
main "$@"
