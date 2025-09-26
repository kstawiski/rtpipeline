#!/bin/bash
set -e

echo "=== rtpipeline COMPLETE Environment Setup Script ==="
echo "This script will set up a FULLY COMPATIBLE environment for rtpipeline"
echo "with ALL features including pyradiomics, TotalSegmentator, and radiomics"
echo

# Detect system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    SYSTEM="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    SYSTEM="macos"
else
    SYSTEM="unknown"
fi

echo "Detected system: $SYSTEM"
echo "Python compatibility: Using Python 3.11 for maximum compatibility"

# Check if conda/mamba is available
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
else
    echo "Error: conda or mamba not found. Please install conda first."
    echo "Install miniconda from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "Using package manager: $CONDA_CMD"

# Check if environment already exists
ENV_NAME="rtpipeline-full"
if $CONDA_CMD env list | grep -q "^$ENV_NAME\s"; then
    echo "Environment '$ENV_NAME' already exists."
    read -p "Do you want to recreate it for a fresh install? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        $CONDA_CMD env remove -n $ENV_NAME -y
    else
        echo "Using existing environment. To activate:"
        echo "  conda activate $ENV_NAME"
        echo "To test: python -c \"import rtpipeline; print('✓ rtpipeline ready')\""
        exit 0
    fi
fi

echo "Creating conda environment '$ENV_NAME' with Python 3.11..."
$CONDA_CMD create -n $ENV_NAME python=3.11 -y

echo "Activating environment..."
# Handle different conda initialization paths
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    # Try to find conda.sh in common locations
    for path in /opt/miniconda3 /opt/anaconda3 /usr/local/miniconda3 /usr/local/anaconda3; do
        if [[ -f "$path/etc/profile.d/conda.sh" ]]; then
            source "$path/etc/profile.d/conda.sh"
            break
        fi
    done
fi

eval "$($CONDA_CMD shell.bash hook)"
$CONDA_CMD activate $ENV_NAME

echo "Installing scientific computing stack..."
$CONDA_CMD install -c conda-forge \
    "numpy>=1.20,<2.0" \
    "pandas>=1.3" \
    "scipy>=1.7" \
    "matplotlib>=3.3" \
    "scikit-image>=0.18" \
    "scikit-learn>=1.0" \
    -y

echo "Installing medical imaging and DICOM packages..."
pip install \
    "pydicom>=3.0,<4" \
    "SimpleITK>=2.1" \
    "dicompyler-core==0.5.6" \
    "pydicom-seg==0.4.1" \
    "rt-utils" \
    "dicom2nifti>=2.4"

echo "Installing visualization and data packages..."
pip install \
    "openpyxl" \
    "plotly>=5.0" \
    "nested-lookup" \
    "nbformat" \
    "ipython" \
    "jupyter"

echo "Attempting to install pyradiomics..."
if pip install pyradiomics; then
    echo "✓ pyradiomics installed successfully"
else
    echo "⚠ pyradiomics installation failed (common with Python 3.12+)"
    echo "  Radiomics features will be disabled"
    echo "  Use --no-radiomics flag when running rtpipeline"
fi

echo "Installing dcm2niix..."
if [[ "$SYSTEM" == "linux" ]]; then
    $CONDA_CMD install -c conda-forge dcm2niix -y
elif [[ "$SYSTEM" == "macos" ]]; then
    $CONDA_CMD install -c conda-forge dcm2niix -y
else
    echo "Warning: Please install dcm2niix manually for your system"
fi

echo "Installing TotalSegmentator with proper numpy constraint..."
pip install "TotalSegmentator==2.4.0" --no-deps
pip install \
    "numpy>=1.20,<2.0" \
    "torch>=1.10" \
    "torchvision" \
    "nibabel" \
    "tqdm" \
    "requests" \
    "nnunet" \
    --force-reinstall

echo "Installing rtpipeline package..."
cd "$SCRIPT_DIR"
pip install -e .

echo "Validating installation..."
cd "$SCRIPT_DIR"
python -c "
import sys
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
print('✓ Core packages imported successfully')

try:
    import radiomics
    print('✓ pyradiomics available')
except ImportError:
    print('✗ pyradiomics not available (normal for Python 3.12+)')

try:
    from dicompylercore import dvhcalc
    print('✓ dicompyler-core available')
except ImportError as e:
    print(f'✗ dicompyler-core error: {e}')

print(f'NumPy version: {np.__version__}')
print(f'Python version: {sys.version.split()[0]}')

# Check TotalSegmentator
try:
    import totalsegmentator
    print('✓ TotalSegmentator Python package available')
except ImportError as e:
    print(f'✗ TotalSegmentator import error: {e}')

# Test rtpipeline CLI
try:
    import rtpipeline.cli
    print('✓ rtpipeline package available')
except ImportError as e:
    print(f'✗ rtpipeline import error: {e}')
"

echo
echo "Running rtpipeline doctor..."
if rtpipeline doctor; then
    echo "✓ rtpipeline doctor passed"
else
    echo "⚠️ rtpipeline doctor found issues (check output above)"
fi

echo
echo "=== Setup Complete ==="
echo "To use the environment:"
echo "  conda activate $ENV_NAME"
echo
echo "To install rtpipeline:"
echo "  cd /path/to/rtpipeline"
echo "  pip install -e ."
echo
echo "To validate the environment:"
echo "  rtpipeline doctor"
echo
echo "For troubleshooting, see TROUBLESHOOTING.md"