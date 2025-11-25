#!/bin/bash
# Example script for downloading custom model weights
#
# IMPORTANT: Replace URLs with actual download locations for your models
# This is a TEMPLATE - update with real download links before using

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Custom Model Weights Download Script"
echo "====================================="
echo ""

# Check if wget or curl is available
if command -v wget &> /dev/null; then
    DOWNLOAD_CMD="wget -O"
elif command -v curl &> /dev/null; then
    DOWNLOAD_CMD="curl -L -o"
else
    echo "ERROR: Neither wget nor curl is installed. Please install one of them."
    exit 1
fi

# Function to download and verify file
download_file() {
    local url="$1"
    local output="$2"
    local expected_size="$3"  # Optional size check in MB

    if [ -f "$output" ]; then
        echo "✓ $output already exists, skipping download"
        return 0
    fi

    echo "Downloading $output..."
    if $DOWNLOAD_CMD "$output" "$url"; then
        if [ -n "$expected_size" ]; then
            actual_size=$(du -m "$output" | cut -f1)
            if [ "$actual_size" -lt "$expected_size" ]; then
                echo "WARNING: Downloaded file is smaller than expected ($actual_size MB < $expected_size MB)"
                echo "File may be incomplete or corrupted"
            fi
        fi
        echo "✓ Downloaded $output successfully"
        return 0
    else
        echo "✗ Failed to download $output"
        return 1
    fi
}

# cardiac_STOPSTORM model weights
echo ""
echo "1. Downloading cardiac_STOPSTORM model weights"
echo "----------------------------------------------"

# TODO: Replace these URLs with actual download locations
# Example: https://zenodo.org/record/XXXXX/files/model603.zip
# Example: https://institution.edu/models/cardiac/model605.zip

CARDIAC_DIR="cardiac_STOPSTORM"
CARDIAC_603_URL="https://REPLACE_WITH_ACTUAL_URL/model603.zip"
CARDIAC_605_URL="https://REPLACE_WITH_ACTUAL_URL/model605.zip"

if [ "$CARDIAC_603_URL" = "https://REPLACE_WITH_ACTUAL_URL/model603.zip" ]; then
    echo "⚠️  WARNING: Download URLs not configured!"
    echo "   Please edit this script and replace REPLACE_WITH_ACTUAL_URL with real URLs"
    echo "   Skipping cardiac_STOPSTORM downloads..."
else
    download_file "$CARDIAC_603_URL" "$CARDIAC_DIR/model603.zip" 100  # Expected ~100 MB
    download_file "$CARDIAC_605_URL" "$CARDIAC_DIR/model605.zip" 100  # Expected ~100 MB
fi

# HN_lymph_nodes model weights
echo ""
echo "2. Downloading HN_lymph_nodes model weights"
echo "-------------------------------------------"

# TODO: Replace these URLs with actual download locations
HN_DIR="HN_lymph_nodes"
HN_WEIGHTS_URL="https://REPLACE_WITH_ACTUAL_URL/hnln_weights.tar.gz"

if [ "$HN_WEIGHTS_URL" = "https://REPLACE_WITH_ACTUAL_URL/hnln_weights.tar.gz" ]; then
    echo "⚠️  WARNING: Download URLs not configured!"
    echo "   Please edit this script and replace REPLACE_WITH_ACTUAL_URL with real URLs"
    echo "   Skipping HN_lymph_nodes downloads..."
else
    download_file "$HN_WEIGHTS_URL" "hnln_weights.tar.gz" 500  # Expected ~500 MB

    # Extract if download successful
    if [ -f "hnln_weights.tar.gz" ]; then
        echo "Extracting hnln_weights.tar.gz to $HN_DIR..."
        tar -xzf "hnln_weights.tar.gz" -C "$HN_DIR/"
        echo "✓ Extraction complete"

        # Optional: remove archive after extraction
        # rm "hnln_weights.tar.gz"
    fi
fi

echo ""
echo "====================================="
echo "Download script completed!"
echo ""
echo "Next steps:"
echo "1. Verify weights are in place:"
echo "   ls -lh $CARDIAC_DIR/"
echo "   ls -lh $HN_DIR/"
echo ""
echo "2. Test model discovery:"
echo "   rtpipeline doctor"
echo ""
echo "3. Or run test command:"
echo "   python -c \"from rtpipeline.custom_models import discover_custom_models; from pathlib import Path; models = discover_custom_models(Path('custom_models')); print(f'Discovered {len(models)} model(s)')\""
echo ""
