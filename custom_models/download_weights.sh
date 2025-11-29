#!/bin/bash
# Script for downloading custom model weights
#
# This script downloads model weights from public repositories.
# Run from the custom_models directory or the repository root.

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Determine script location and set working directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${GREEN}Custom Model Weights Download Script${NC}"
echo "====================================="
echo ""

# Check if wget or curl is available
if command -v wget &> /dev/null; then
    DOWNLOAD_CMD="wget"
elif command -v curl &> /dev/null; then
    DOWNLOAD_CMD="curl"
else
    echo -e "${RED}ERROR: Neither wget nor curl is installed.${NC}"
    echo "Please install one of them: apt install wget"
    exit 1
fi

# Function to download and verify file
download_file() {
    local url="$1"
    local output="$2"
    local expected_size="$3"  # Optional size check in MB

    if [ -f "$output" ]; then
        echo -e "  ${GREEN}✓${NC} $output already exists, skipping"
        return 0
    fi

    # Ensure output directory exists
    local output_dir=$(dirname "$output")
    mkdir -p "$output_dir"

    echo -e "  ${YELLOW}Downloading$(NC) $(basename $output)..."

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
                echo "  File may be incomplete or corrupted"
            fi
        fi
        echo -e "  ${GREEN}✓${NC} Downloaded $(basename $output)"
        return 0
    else
        echo -e "  ${RED}✗${NC} Failed to download $(basename $output)"
        rm -f "$output"
        return 1
    fi
}

DOWNLOAD_FAILED=0

# ==========================================
# cardiac_STOPSTORM model weights (Zenodo)
# ==========================================
echo ""
echo -e "${YELLOW}1. Downloading cardiac_STOPSTORM model weights${NC}"
echo "   Source: https://zenodo.org/records/14033123"
echo "----------------------------------------------"

CARDIAC_DIR="cardiac_STOPSTORM"
ZENODO_BASE="https://zenodo.org/records/14033123/files"

# Model 603 - Small structures (valves, coronary arteries)
download_file "${ZENODO_BASE}/model603.zip?download=1" "$CARDIAC_DIR/model603.zip" 100 || DOWNLOAD_FAILED=1

# Model 604 - Medium structures
download_file "${ZENODO_BASE}/model604.zip?download=1" "$CARDIAC_DIR/model604.zip" 100 || DOWNLOAD_FAILED=1

# Model 605 - Large structures (chambers, great vessels)
download_file "${ZENODO_BASE}/model605.zip?download=1" "$CARDIAC_DIR/model605.zip" 100 || DOWNLOAD_FAILED=1

# ==========================================
# HN_lymph_nodes model weights
# ==========================================
echo ""
echo -e "${YELLOW}2. HN_lymph_nodes model weights${NC}"
echo "-------------------------------------------"

HN_DIR="HN_lymph_nodes"

# Note: HN_lymph_nodes weights need to be obtained from the original publication
# or institutional repository. Uncomment and update when URLs become available.
echo -e "  ${YELLOW}!${NC} HN_lymph_nodes weights must be obtained separately"
echo "    Contact the model authors or check the publication for download links"
echo "    Expected location: $HN_DIR/HNLNL_autosegmentation_trained_models/"

# Example (uncomment when URL is available):
# HN_WEIGHTS_URL="https://example.com/HNLNL_autosegmentation_trained_models.tar.gz"
# download_file "$HN_WEIGHTS_URL" "hnln_weights.tar.gz" 500 || DOWNLOAD_FAILED=1
# if [ -f "hnln_weights.tar.gz" ]; then
#     echo "  Extracting to $HN_DIR..."
#     tar -xzf "hnln_weights.tar.gz" -C "$HN_DIR/"
#     rm "hnln_weights.tar.gz"
#     echo -e "  ${GREEN}✓${NC} Extraction complete"
# fi

echo ""
echo "====================================="
if [ $DOWNLOAD_FAILED -eq 0 ]; then
    echo -e "${GREEN}Download script completed successfully!${NC}"
else
    echo -e "${YELLOW}Download script completed with some failures${NC}"
fi
echo ""
echo "Next steps:"
echo "1. Verify weights are in place:"
echo "   ls -lh $CARDIAC_DIR/*.zip"
echo ""
echo "2. Test model discovery:"
echo "   python -c \"from rtpipeline.segment_custom_models import discover_custom_models; from pathlib import Path; models = discover_custom_models(Path('.')); print(f'Discovered {len(models)} model(s)')\""
echo ""
echo "3. Build Docker image (will include these weights):"
echo "   cd .. && ./build.sh"
echo ""
