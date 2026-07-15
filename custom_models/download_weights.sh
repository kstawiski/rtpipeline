#!/bin/bash
# Script for downloading custom model weights.
#
# Run from the custom_models directory or the repository root. Optional
# positional arguments restrict the download, for example:
#   custom_models/download_weights.sh lung_tumor_pancancer_lung

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${GREEN}Custom Model Weights Download Script${NC}"
echo "====================================="
echo ""

if command -v wget >/dev/null 2>&1; then
    DOWNLOAD_CMD="wget"
elif command -v curl >/dev/null 2>&1; then
    DOWNLOAD_CMD="curl"
else
    echo -e "${RED}ERROR: Neither wget nor curl is installed.${NC}"
    exit 1
fi

DOWNLOAD_FAILED=0
REQUESTED_TARGETS=("$@")

should_download() {
    local target="$1"
    if [ ${#REQUESTED_TARGETS[@]} -eq 0 ]; then
        return 0
    fi
    for requested in "${REQUESTED_TARGETS[@]}"; do
        if [ "$requested" = "$target" ] || [ "$requested" = "all" ]; then
            return 0
        fi
    done
    return 1
}

download_file() {
    local url="$1"
    local output="$2"
    local expected_size="$3"

    if [ -f "$output" ]; then
        echo -e "  ${GREEN}ok${NC} $output already exists, skipping"
        return 0
    fi

    mkdir -p "$(dirname "$output")"
    echo -e "  ${YELLOW}Downloading${NC} $(basename "$output")..."

    local success=false
    if [ "$DOWNLOAD_CMD" = "wget" ]; then
        if wget -q --show-progress -O "$output" "$url"; then
            success=true
        fi
    else
        if curl -L -# -o "$output" "$url"; then
            success=true
        fi
    fi

    if [ "$success" = true ]; then
        if [ -n "$expected_size" ]; then
            actual_size=$(du -m "$output" 2>/dev/null | cut -f1)
            if [ -n "$actual_size" ] && [ "$actual_size" -lt "$expected_size" ]; then
                echo -e "  ${YELLOW}WARNING:${NC} Downloaded file is smaller than expected ($actual_size MB < $expected_size MB)"
            fi
        fi
        echo -e "  ${GREEN}ok${NC} Downloaded $(basename "$output")"
        return 0
    fi

    echo -e "  ${RED}failed${NC} Failed to download $(basename "$output")"
    rm -f "$output"
    return 1
}

clone_repo_once() {
    local repo_url="$1"
    local output_dir="$2"

    if [ -d "$output_dir/.git" ]; then
        echo -e "  ${GREEN}ok${NC} $output_dir already exists, skipping"
        return 0
    fi
    if ! command -v git >/dev/null 2>&1; then
        echo -e "  ${RED}ERROR:${NC} git is required to clone $repo_url"
        return 1
    fi
    rm -rf "$output_dir"
    mkdir -p "$(dirname "$output_dir")"
    echo -e "  ${YELLOW}Cloning${NC} $repo_url..."
    git clone --depth 1 "$repo_url" "$output_dir"
}

download_hf_snapshot() {
    local repo_id="$1"
    local local_dir="$2"
    local allow_pattern="$3"

    mkdir -p "$local_dir"
    python - "$repo_id" "$local_dir" "$allow_pattern" <<'PY'
from pathlib import Path
import sys
from huggingface_hub import snapshot_download

repo_id, local_dir, allow_pattern = sys.argv[1:4]
snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    allow_patterns=[allow_pattern, "README.md", ".gitattributes"],
)
print(Path(local_dir).resolve())
PY
}

if should_download "cardiac_STOPSTORM"; then
    echo ""
    echo -e "${YELLOW}1. Downloading cardiac_STOPSTORM model weights${NC}"
    echo "   Source: https://zenodo.org/records/14033123"
    echo "----------------------------------------------"

    CARDIAC_DIR="cardiac_STOPSTORM"
    ZENODO_BASE="https://zenodo.org/records/14033123/files"
    download_file "${ZENODO_BASE}/model603.zip?download=1" "$CARDIAC_DIR/model603.zip" 100 || DOWNLOAD_FAILED=1
    download_file "${ZENODO_BASE}/model604.zip?download=1" "$CARDIAC_DIR/model604.zip" 100 || DOWNLOAD_FAILED=1
    download_file "${ZENODO_BASE}/model605.zip?download=1" "$CARDIAC_DIR/model605.zip" 100 || DOWNLOAD_FAILED=1
fi

if should_download "lung_tumor_totalseg_lung_nodules"; then
    echo ""
    echo -e "${YELLOW}Downloading TotalSegmentator lung_nodules weights${NC}"
    echo "   Source: TotalSegmentator task lung_nodules"
    echo "-----------------------------------------------------"
    TS_DIR="lung_tumor_totalseg_lung_nodules"
    mkdir -p "$TS_DIR"
    if command -v totalseg_download_weights >/dev/null 2>&1; then
        TOTALSEG_HOME_DIR="$SCRIPT_DIR/$TS_DIR/.totalsegmentator" \
            totalseg_download_weights -t lung_nodules || DOWNLOAD_FAILED=1
    else
        echo -e "  ${RED}ERROR:${NC} totalseg_download_weights not found"
        DOWNLOAD_FAILED=1
    fi
fi

if should_download "lung_tumor_pancancer_lung"; then
    echo ""
    echo -e "${YELLOW}Downloading PanCancerSeg specialized lung weights${NC}"
    echo "   Source: https://huggingface.co/KS987/PanCancerSeg-Specialized-weights"
    echo "-----------------------------------------------------------------------"
    download_hf_snapshot \
        "KS987/PanCancerSeg-Specialized-weights" \
        "lung_tumor_pancancer_lung/weights/PanCancerSeg-Specialized-weights" \
        "Dataset105_Lung/*" || DOWNLOAD_FAILED=1
fi

if should_download "lung_tumor_medsam_boxprompt"; then
    echo ""
    echo -e "${YELLOW}Downloading MedSAM box-prompt weights/source${NC}"
    echo "   Source: https://github.com/bowang-lab/MedSAM and Zenodo 10.5281/zenodo.10689643"
    echo "-----------------------------------------------------------------------"
    MEDSAM_DIR="lung_tumor_medsam_boxprompt"
    clone_repo_once \
        "https://github.com/bowang-lab/MedSAM.git" \
        "$MEDSAM_DIR/weights/MedSAM" || DOWNLOAD_FAILED=1
    download_file \
        "https://zenodo.org/records/10689643/files/medsam_vit_b.pth?download=1" \
        "$MEDSAM_DIR/weights/medsam_vit_b.pth" \
        300 || DOWNLOAD_FAILED=1
fi

if should_download "lung_tumor_aimi_nsclc_rg_disabled"; then
    echo ""
    echo -e "${YELLOW}Downloading disabled AIMI/BAMF NSCLC-RG nnU-Net v1 weights${NC}"
    echo "   Source: https://zenodo.org/records/8290169"
    echo "-------------------------------------------------------"
    download_file \
        "https://zenodo.org/records/8290169/files/Task775_CT_NSCLC_RG.zip?download=1" \
        "_disabled_lung_tumor_aimi_nsclc_rg/Task775_CT_NSCLC_RG.zip" \
        1000 || DOWNLOAD_FAILED=1
fi

if should_download "HN_lymph_nodes"; then
    echo ""
    echo -e "${YELLOW}2. HN_lymph_nodes model weights${NC}"
    echo "-------------------------------------------"
    echo -e "  ${YELLOW}!${NC} HN_lymph_nodes weights must be obtained separately"
    echo "    Contact the model authors or check the publication for download links"
    echo "    Expected location: HN_lymph_nodes/HNLNL_autosegmentation_trained_models/"
fi

echo ""
echo "====================================="
if [ $DOWNLOAD_FAILED -eq 0 ]; then
    echo -e "${GREEN}Download script completed successfully.${NC}"
else
    echo -e "${YELLOW}Download script completed with failures.${NC}"
fi
echo ""
echo "Test discovery:"
echo "  python -c \"from rtpipeline.custom_models import discover_custom_models; from pathlib import Path; print([m.name for m in discover_custom_models(Path('custom_models'))])\""
exit $DOWNLOAD_FAILED
