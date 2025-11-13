#!/bin/bash
#
# RTpipeline Interactive Project Setup
#
# This script helps you set up the RTpipeline for processing DICOM files
# in a new directory by creating a config.yaml and run_pipeline.sh script.
#
# Usage: ./setup_new_project.sh
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

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

print_explanation() {
    echo -e "${BLUE}ℹ${NC}  $1"
}

ask_question() {
    local question="$1"
    local default="$2"
    local response

    if [ -n "$default" ]; then
        read -p "$(echo -e ${GREEN}?)${NC} $question [default: $default]: " response
        echo "${response:-$default}"
    else
        read -p "$(echo -e ${GREEN}?)${NC} $question: " response
        echo "$response"
    fi
}

ask_yes_no() {
    local question="$1"
    local default="$2"
    local response

    if [ "$default" = "yes" ]; then
        read -p "$(echo -e ${GREEN}?)${NC} $question [Y/n]: " response
        response="${response:-y}"
    else
        read -p "$(echo -e ${GREEN}?)${NC} $question [y/N]: " response
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

    if python3 -c "import $package" 2>/dev/null; then
        print_info "✓ Python package '$package' found"
        return 0
    else
        print_warning "✗ Python package '$package' not found"
        return 1
    fi
}

# ============================================================================
# Prerequisites Check
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
            print_info "✓ Python version is 3.11 or later"
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
        if command -v nvidia-smi &> /dev/null; then
            print_info "✓ NVIDIA GPU detected"
            if docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null 2>&1; then
                print_info "✓ NVIDIA Docker support available"
            else
                print_warning "NVIDIA Docker support not configured. GPU acceleration will not be available."
                print_info "Install nvidia-container-toolkit to enable GPU support in Docker."
            fi
        fi
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
        check_python_package "numpy" || missing_prereqs=$((missing_prereqs + 1))
        check_python_package "pandas" || missing_prereqs=$((missing_prereqs + 1))
        check_python_package "pydicom" || missing_prereqs=$((missing_prereqs + 1))
        check_python_package "SimpleITK" || optional_missing=$((optional_missing + 1))
    fi

    # Check disk space
    print_info "\nChecking disk space..."
    available_space=$(df -BG "$SCRIPT_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')
    print_info "Available disk space: ${available_space}GB"

    if [ "$available_space" -lt 20 ]; then
        print_warning "Less than 20GB available. Processing may require significant space."
    else
        print_info "✓ Sufficient disk space available"
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
        print_info "✓ All critical prerequisites satisfied!"

        if [ $optional_missing -gt 0 ]; then
            print_warning "$optional_missing optional prerequisites missing."
            print_info "Pipeline will work but some features may be limited."
        fi
    fi

    echo ""
    read -p "Press Enter to continue with setup..."
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
# Configuration
# ============================================================================

configure_project() {
    print_header "Project Configuration"

    # Get DICOM directory
    print_explanation "Specify the directory containing your DICOM files."
    print_explanation "This should contain subdirectories for each patient or a flat structure."
    echo ""

    local dicom_dir
    while true; do
        dicom_dir=$(ask_question "Path to DICOM directory" "")

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
                break
            fi
        fi
    done

    # Get output directory
    print_explanation "\nSpecify where to store pipeline results."
    print_explanation "This will contain processed data, segmentations, and analysis results."
    echo ""

    local output_dir
    output_dir=$(ask_question "Output directory name" "$(dirname "$dicom_dir")/rtpipeline_output")
    output_dir="${output_dir/#\~/$HOME}"

    # Get logs directory
    local logs_dir
    logs_dir=$(ask_question "Logs directory name" "$(dirname "$dicom_dir")/rtpipeline_logs")
    logs_dir="${logs_dir/#\~/$HOME}"

    # Create directories
    mkdir -p "$output_dir"
    mkdir -p "$logs_dir"
    print_info "Created output directory: $output_dir"
    print_info "Created logs directory: $logs_dir"

    # Worker configuration
    print_header "Performance Configuration"

    print_explanation "Workers control parallel processing."
    print_explanation "More workers = faster processing but more CPU/memory usage."
    cpu_count=$(python3 -c "import os; print(os.cpu_count() or 4)")
    print_info "Detected $cpu_count CPU cores"

    local workers
    workers=$(ask_question "Number of workers" "auto")

    # Segmentation configuration
    print_header "Segmentation Configuration"

    print_explanation "\nTotalSegmentator automatically segments 100+ organs/tissues."
    print_explanation "It can use GPU (faster) or CPU (slower but works everywhere)."
    echo ""

    local seg_device
    if command -v nvidia-smi &> /dev/null; then
        print_info "NVIDIA GPU detected"
        seg_device=$(ask_question "Segmentation device (gpu/cpu)" "gpu")
    else
        print_warning "No NVIDIA GPU detected"
        seg_device="cpu"
    fi

    print_explanation "\nSegmentation workers: Number of parallel segmentation jobs."
    print_explanation "For GPU: 1-4 workers (depending on VRAM). For CPU: 1-2 workers."
    local seg_workers
    if [ "$seg_device" = "gpu" ]; then
        seg_workers=$(ask_question "Segmentation workers" "4")
    else
        seg_workers=$(ask_question "Segmentation workers" "1")
    fi

    print_explanation "\nFast mode: Uses CPU-friendly settings (lower quality but faster)."
    local seg_fast
    seg_fast=$(ask_yes_no "Enable fast mode?" "no")

    print_explanation "\nForce split: Chunks inference to reduce memory usage (recommended)."
    local seg_force_split
    seg_force_split=$(ask_yes_no "Enable force split?" "yes")

    # Radiomics configuration
    print_header "Radiomics Configuration"

    print_explanation "\nRadiomics extracts 150+ quantitative features from images."
    print_explanation "This includes shape, intensity, and texture features."
    local enable_radiomics
    enable_radiomics=$(ask_yes_no "Enable radiomics feature extraction?" "yes")

    local radiomics_threads
    if [ "$enable_radiomics" = "yes" ]; then
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

    # CT Cropping configuration
    print_header "CT Cropping Configuration"

    print_explanation "\nSystematic CT cropping normalizes anatomical volumes across patients."
    print_explanation "This makes percentage DVH metrics (V95%, V50Gy) meaningful."
    print_explanation "Crops all CTs to the same anatomical boundaries (e.g., L1 to femoral heads)."
    echo ""

    local enable_cropping
    enable_cropping=$(ask_yes_no "Enable systematic CT cropping?" "no")

    local crop_region crop_superior crop_inferior crop_use_dvh crop_use_radiomics crop_keep_orig
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
    fi

    # Custom structures
    print_header "Custom Structures Configuration"

    print_explanation "\nCustom structures allow Boolean operations (union, intersection, margins)."
    print_explanation "Example: Combine left+right iliac arteries into single structure."
    local custom_structures_file
    custom_structures_file=$(ask_question "Custom structures config file" "custom_structures_pelvic.yaml")

    if [ ! -f "$SCRIPT_DIR/$custom_structures_file" ]; then
        print_warning "Custom structures file not found: $custom_structures_file"
        print_info "You can create it later or use an existing one."
    fi

    # Custom models
    print_header "Custom Models Configuration"

    print_explanation "\nCustom models: Use your own trained nnUNet models for segmentation."
    print_explanation "WARNING: This requires model weights and proper setup."
    local enable_custom_models
    enable_custom_models=$(ask_yes_no "Enable custom models?" "no")

    # Generate config
    print_header "Generating Configuration"

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
        "$enable_custom_models"

    # Generate run script
    generate_run_script "$dicom_dir" "$output_dir"

    # Summary
    print_header "Setup Complete!"

    print_info "Configuration saved to: $dicom_dir/config.yaml"
    print_info "Run script saved to: $dicom_dir/run_pipeline.sh"

    echo ""
    print_info "To run the pipeline:"
    echo ""
    echo -e "${GREEN}cd $dicom_dir${NC}"
    echo -e "${GREEN}chmod +x run_pipeline.sh${NC}"
    echo -e "${GREEN}./run_pipeline.sh${NC}"
    echo ""

    print_info "Or use Snakemake directly:"
    echo ""
    echo -e "${GREEN}cd $SCRIPT_DIR${NC}"
    echo -e "${GREEN}snakemake --cores all --use-conda --configfile $dicom_dir/config.yaml${NC}"
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

    # Convert yes/no to true/false
    [ "$seg_fast" = "yes" ] && seg_fast="true" || seg_fast="false"
    [ "$seg_force_split" = "yes" ] && seg_force_split="true" || seg_force_split="false"
    [ "$enable_cropping" = "yes" ] && enable_cropping="true" || enable_cropping="false"
    [ "$crop_use_dvh" = "yes" ] && crop_use_dvh="true" || crop_use_dvh="false"
    [ "$crop_use_radiomics" = "yes" ] && crop_use_radiomics="true" || crop_use_radiomics="false"
    [ "$crop_keep_orig" = "yes" ] && crop_keep_orig="true" || crop_keep_orig="false"
    [ "$enable_custom_models" = "yes" ] && enable_custom_models="true" || enable_custom_models="false"

    # Convert skip_rois to YAML list
    if [ -n "$skip_rois" ]; then
        skip_rois_yaml=""
        IFS=',' read -ra SKIP_ARRAY <<< "$skip_rois"
        for roi in "${SKIP_ARRAY[@]}"; do
            skip_rois_yaml="${skip_rois_yaml}    - $(echo "$roi" | xargs)\n"
        done
    else
        skip_rois_yaml="    - body\n    - couchsurface\n"
    fi

    cat > "$dicom_dir/config.yaml" << EOF
# RTpipeline Configuration
# Generated by setup_new_project.sh on $(date)

# Input/Output directories
dicom_root: "$dicom_dir"
output_dir: "$output_dir"
logs_dir: "$logs_dir"

# Processing parameters
workers: $workers

segmentation:
  workers: $seg_workers
  threads_per_worker: null
  force: false
  fast: $seg_fast
  roi_subset: null
  extra_models: []
  device: "$seg_device"
  force_split: $seg_force_split
  nr_threads_resample: 1
  nr_threads_save: 1
  num_proc_preprocessing: 1
  num_proc_export: 1

custom_models:
  enabled: $enable_custom_models
  root: "custom_models"
  models: []
  workers: 1
  force: false
  nnunet_predict: "nnUNet_predict"
  retain_weights: true
  conda_activate: null

radiomics:
  sequential: false
  params_file: "$SCRIPT_DIR/rtpipeline/radiomics_params.yaml"
  mr_params_file: "$SCRIPT_DIR/rtpipeline/radiomics_params_mr.yaml"
  thread_limit: $radiomics_threads
  skip_rois:
$(echo -e "$skip_rois_yaml")
  max_voxels: 1500000000
  min_voxels: 10

aggregation:
  threads: auto

environments:
  main: "rtpipeline"
  radiomics: "rtpipeline-radiomics"

custom_structures: "$custom_structures_file"

ct_cropping:
  enabled: $enable_cropping
  region: "$crop_region"
  superior_margin_cm: $crop_superior
  inferior_margin_cm: $crop_inferior
  use_cropped_for_dvh: $crop_use_dvh
  use_cropped_for_radiomics: $crop_use_radiomics
  keep_original: $crop_keep_orig
EOF

    print_info "Generated config.yaml"
}

generate_run_script() {
    local dicom_dir="$1"
    local output_dir="$2"

    cat > "$dicom_dir/run_pipeline.sh" << 'RUNSCRIPT'
#!/bin/bash
#
# RTpipeline Execution Script
# Generated by setup_new_project.sh
#

set -euo pipefail

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting RTpipeline...${NC}"
echo ""

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: config.yaml not found in $SCRIPT_DIR${NC}"
    exit 1
fi

# Check if rtpipeline repository is accessible
RTPIPELINE_DIR="RTPIPELINE_PLACEHOLDER"
if [ ! -d "$RTPIPELINE_DIR" ]; then
    echo -e "${RED}Error: RTpipeline repository not found at $RTPIPELINE_DIR${NC}"
    echo "Please update RTPIPELINE_DIR in this script to point to the rtpipeline repository."
    exit 1
fi

cd "$RTPIPELINE_DIR"

# Check for running processes
if pgrep -f "snakemake" >/dev/null; then
    echo -e "${RED}Another snakemake process is currently running.${NC}"
    echo "Please stop it before running the pipeline."
    exit 1
fi

# Unlock workflow (in case of previous crash)
snakemake --unlock --configfile "$CONFIG_FILE" >/dev/null 2>&1 || true

# Run pipeline
echo "Running Snakemake workflow..."
echo "Config: $CONFIG_FILE"
echo ""

snakemake \
    --cores all \
    --use-conda \
    --configfile "$CONFIG_FILE" \
    --rerun-incomplete \
    --keep-going \
    --printshellcmds

echo ""
echo -e "${GREEN}Pipeline completed!${NC}"
echo ""
echo "Results available at:"
grep "output_dir:" "$CONFIG_FILE" | awk '{print $2}' | xargs echo "  "
echo ""
echo "Check aggregated results in: {output_dir}/_RESULTS/"
RUNSCRIPT

    # Replace placeholder with actual rtpipeline directory
    sed -i "s|RTPIPELINE_PLACEHOLDER|$SCRIPT_DIR|g" "$dicom_dir/run_pipeline.sh"

    chmod +x "$dicom_dir/run_pipeline.sh"

    print_info "Generated run_pipeline.sh (executable)"
}

# ============================================================================
# Main
# ============================================================================

main() {
    clear

    cat << 'EOF'
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║           RTpipeline Interactive Project Setup                 ║
║                                                                ║
║  This wizard will help you configure the RTpipeline for        ║
║  processing DICOM radiotherapy data in a new directory.        ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
EOF

    echo ""
    read -p "Press Enter to begin..."

    # Step 1: Check prerequisites
    check_prerequisites

    # Step 2: Configure project
    configure_project

    echo ""
    print_info "Setup wizard completed successfully!"
    echo ""
}

# Run main function
main "$@"
