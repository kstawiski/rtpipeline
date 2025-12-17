#!/bin/bash
#
# RTpipeline Interactive Setup Wizard
#
# A comprehensive setup tool for initializing new RTpipeline projects.
# Configures directory structure, processing options, and robustness analysis.
#
# Usage:
#   ./setup_new_project.sh                    # Interactive wizard
#   ./setup_new_project.sh --quick /path      # Quick setup with defaults
#   ./setup_new_project.sh --preset prostate  # Use preset configuration
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

# Script variables
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VERSION="2.1.0"
CONFIG_FILE=""
PRESET=""
QUICK_MODE=false
QUICK_DICOM_DIR=""
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
EOF
    echo -e "${NC}"
    echo -e "${DIM}   Radiotherapy DICOM Processing Pipeline Setup v${VERSION}${NC}"
    echo -e "${DIM}   --------------------------------------------------${NC}"
    echo ""
}

# Print section header
print_header() {
    echo -e "\n${BOLD}${MAGENTA}:: $1 ::${NC}"
    echo -e "${DIM}$(printf '%.0s-' {1..60})${NC}"
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
    print_header "System Check"
    local missing=0

    # Python
    if command -v python3 &> /dev/null; then
        local py_ver=$(python3 -V 2>&1 | awk '{print $2}')
        print_success "Python $py_ver found"
    else
        print_error "Python 3 not found"
        missing=$((missing + 1))
    fi

    # Docker
    if command -v docker &> /dev/null; then
        print_success "Docker found"
        if docker ps &> /dev/null; then
             print_success "Docker daemon running"
        else
             print_warning "Docker daemon not running (or no permission)"
        fi
    else
        print_warning "Docker not found (recommended)"
    fi

    # GPU
    if command -v nvidia-smi &> /dev/null; then
        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
        print_success "GPU detected: $gpu_name"
    else
        print_warning "No NVIDIA GPU detected (pipeline will run in CPU mode)"
    fi

    # Memory
    if [ -f /proc/meminfo ]; then
        local total_mem_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
        local total_mem_gb=$((total_mem_kb / 1024 / 1024))
        if [ "$total_mem_gb" -lt 16 ]; then
            print_warning "Low RAM detected (${total_mem_gb}GB). >16GB recommended."
        else
            print_success "RAM: ${total_mem_gb}GB"
        fi
    fi
    
    if [ $missing -gt 0 ]; then
        print_error "Critical prerequisites missing. Please install them and retry."
        exit 1
    fi
    echo ""
}

# ============================================================================
# Main Logic
# ============================================================================

main() {
    print_banner
    
    # Parse args
    while [[ $# -gt 0 ]]; do
        case $1 in
            --quick) QUICK_MODE=true; shift ;; 
            --quick=*) QUICK_MODE=true; QUICK_DICOM_DIR="${1#--quick=}"; shift ;; 
            --dry-run) DRY_RUN=true; shift ;; 
            --preset) PRESET="$2"; shift 2 ;; 
            *) if [ -z "$QUICK_DICOM_DIR" ]; then QUICK_DICOM_DIR="$1"; fi; shift ;; 
        esac
    done

    check_prerequisites

    # --- Directory Setup ---
    print_header "Project Directories" 
    
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
    
    # Resolve absolute path
    dicom_dir=$(cd "$(dirname "$dicom_dir")"; pwd)/$(basename "$dicom_dir")
    
    local parent_dir=$(dirname "$dicom_dir")
    local default_out="$parent_dir/rtpipeline_output"
    local output_dir="$default_out"
    if [ "$QUICK_MODE" = "true" ]; then
        print_info "Outputs will be saved to: $output_dir"
    else
        output_dir=$(ask_question "Where should outputs be saved?" "$default_out")
    fi
    
    # --- Processing Configuration ---
    print_header "Processing Options"
    
    local worker_cap="auto"
    if [ "$QUICK_MODE" = "false" ]; then
        print_info "Parallel processing settings:"
        worker_cap=$(ask_question "Max concurrent workers (auto = CPU cores - 1)" "auto")
    fi
    
    # --- Segmentation ---
    print_header "Segmentation"
    local seg_device="gpu"
    if ! command -v nvidia-smi &> /dev/null; then
        seg_device="cpu"
        print_info "Using CPU for segmentation (no GPU detected)."
    fi
    
    local seg_fast="no"
    if [ "$seg_device" = "cpu" ] && [ "$QUICK_MODE" = "false" ]; then
        seg_fast=$(ask_yes_no "Enable TotalSegmentator 'fast' mode (lower quality, faster)?" "no")
    fi

    # --- Radiomics Robustness (New Feature) ---
    print_header "Radiomics Robustness"
    print_info "Evaluates feature stability using systematic perturbations (NTCV chain)."
    print_info "Highly recommended for reliable models."
    
    local robustness_enabled="yes"
    local robustness_intensity="standard"
    
    if [ "$QUICK_MODE" = "false" ]; then
        robustness_enabled=$(ask_yes_no "Enable Radiomics Robustness Analysis?" "yes")
        
        if [ "$robustness_enabled" = "yes" ]; then
            print_info "Select perturbation intensity:"
            echo -e "    1) ${BOLD}mild${NC}      (10-15 perts/ROI) - Quick QA"
            echo -e "    2) ${BOLD}standard${NC}  (15-30 perts/ROI) - Recommended for clinical use"
            echo -e "    3) ${BOLD}aggressive${NC} (30-60 perts/ROI) - Research grade"
            
            local intensity_sel=$(ask_question "Select intensity [1-3]" "2")
            case "$intensity_sel" in
                1) robustness_intensity="mild" ;; 
                3) robustness_intensity="aggressive" ;; 
                *) robustness_intensity="standard" ;; 
            esac
        fi
    fi

    # --- Generate Config ---
    if [ "$DRY_RUN" = "true" ]; then
        print_header "Dry Run Summary"
        echo "DICOM Dir: $dicom_dir"
        echo "Output Dir: $output_dir"
        echo "Robustness: $robustness_enabled ($robustness_intensity)"
        exit 0
    fi
    
    print_header "Generating Configuration"
    
    # Convert yes/no to true/false
    local rob_bool="false"
    if [ "$robustness_enabled" = "yes" ]; then rob_bool="true"; fi
    
    local seg_fast_bool="false"
    if [ "$seg_fast" = "yes" ]; then seg_fast_bool="true"; fi

    local config_path="$dicom_dir/config.yaml"
    
    # Create config content
    cat > "$config_path" <<EOF
# RTpipeline Configuration
# Generated by setup_new_project.sh v$VERSION on $(date)

# Paths
dicom_root: "$dicom_dir"
output_dir: "$output_dir"
logs_dir: "${output_dir}_logs"

# Processing
max_workers: $worker_cap  # Controls overall CPU parallelism (auto = cores - 1)

segmentation:
  device: "$seg_device"
  fast: $seg_fast_bool
  # Parallelism note:
  # - GPU mode: defaults to 1 concurrent worker to avoid OOM
  # - CPU mode: defaults to 25% of cores to avoid OOM (TotalSegmentator is RAM heavy)
  max_workers: null  # null = auto-detect based on device

radiomics:
  params_file: "rtpipeline/radiomics_params.yaml"
  mr_params_file: "rtpipeline/radiomics_params_mr.yaml"
  # Parallelism note:
  # - Uses process-based parallelism (cpu_count - 1)
  # - Sets OMP_NUM_THREADS=1 per worker to prevent oversubscription

# Radiomics Robustness
# Evaluates stability via NTCV perturbation chain (Noise, Translation, Contour, Volume)
radiomics_robustness:
  enabled: $rob_bool
  modes:
    - segmentation_perturbation
  segmentation_perturbation:
    intensity: "$robustness_intensity"  # mild, standard, aggressive
    apply_to_structures:
      - "GTV*"
      - "CTV*"
      - "PTV*"
      - "BLADDER"
      - "RECTUM"
      - "PROSTATE"
    small_volume_changes: [-0.15, 0.0, 0.15]
    max_translation_mm: 3.0
    n_random_contour_realizations: 2

# Systematic CT Cropping
ct_cropping:
  enabled: true
  region: "pelvis"
  superior_margin_cm: 2.0
  inferior_margin_cm: 10.0
  use_cropped_for_radiomics: true

EOF
    
    print_success "Configuration saved to: $config_path"
    
    # Create run script
    local run_script="$dicom_dir/run_pipeline.sh"
    cat > "$run_script" <<EOF
#!/bin/bash
# RTpipeline Execution Script
set -e

# Ensure we use the rtpipeline environment
# If running in Docker, this is automatic.
# If running locally, ensure 'rtpipeline' conda env is active.

# Run Snakemake
# --cores all: Uses all available cores (pipeline manages internal concurrency)
# --use-conda: Ensures dependencies are met
snakemake --cores all --use-conda --configfile config.yaml "$@"
EOF
    chmod +x "$run_script"
    print_success "Run script created: $run_script"
    
    echo ""
    echo -e "${BOLD}Next Steps:${NC}"
    echo "1. Review settings: ${CYAN}nano $config_path${NC}"
    echo "2. Run pipeline:    ${CYAN}cd $dicom_dir && ./run_pipeline.sh${NC}"
    echo ""
}

main "$@"
