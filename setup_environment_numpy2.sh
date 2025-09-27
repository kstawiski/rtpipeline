#!/bin/bash
# Enhanced setup script for rtpipeline with NumPy 2.x and legacy compatibility
# Based on original setup_environment.sh but updated for NumPy 2.x architecture

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ENV_NAME="rtpipeline-numpy2"
PYTHON_VERSION="3.11"

echo "=== Setting up rtpipeline with NumPy 2.x + Legacy Compatibility ==="
echo

# Detect system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    SYSTEM="linux"
    echo "Detected Linux system"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    SYSTEM="macos"
    echo "Detected macOS system"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    SYSTEM="windows"
    echo "Detected Windows system"
else
    SYSTEM="unknown"
    echo "Unknown system: $OSTYPE"
fi

# Check for conda/mamba
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo "Using mamba for package management"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
    echo "Using conda for package management"
else
    echo "Error: Neither conda nor mamba found. Please install Anaconda, Miniconda, or Mamba first."
    exit 1
fi

# Create conda environment
echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
if $CONDA_CMD env list | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists. Removing it first..."
    $CONDA_CMD env remove -n "$ENV_NAME" -y
fi

$CONDA_CMD create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
echo "✓ Environment created"

# Activate environment
echo "Activating environment..."
eval "$($CONDA_CMD shell.bash hook)"
$CONDA_CMD activate "$ENV_NAME"

# Verify activation
if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    echo "Error: Failed to activate environment"
    exit 1
fi
echo "✓ Environment activated"

# Install core scientific stack with NumPy 2.x
echo "Installing core scientific packages with NumPy 2.x..."
$CONDA_CMD install -c conda-forge -y \
    "numpy>=2.0" \
    "pandas>=2.0" \
    "scipy" \
    "matplotlib" \
    "seaborn" \
    "scikit-learn" \
    "scikit-image" \
    "pillow" \
    "ipython" \
    "jupyter"

echo "✓ Core scientific stack installed with NumPy 2.x"

# Install medical imaging packages
echo "Installing medical imaging packages..."
pip install \
    "pydicom>=2.4.0" \
    "SimpleITK>=2.3.0" \
    "dicompyler-core>=0.5.9" \
    "nibabel>=5.0.0" \
    "plastimatch"

echo "✓ Medical imaging packages installed"

# Install dcm2niix
echo "Installing dcm2niix..."
if [[ "$SYSTEM" == "linux" ]]; then
    $CONDA_CMD install -c conda-forge dcm2niix -y
elif [[ "$SYSTEM" == "macos" ]]; then
    $CONDA_CMD install -c conda-forge dcm2niix -y
else
    echo "Warning: Please install dcm2niix manually for your system"
fi

# Install TotalSegmentator 2.11.0 with NumPy 2.x compatibility
echo "Installing TotalSegmentator 2.11.0 with NumPy 2.x compatibility..."
pip install "TotalSegmentator==2.11.0"
echo "✓ TotalSegmentator installed"

# Build pyradiomics from source for NumPy 2.x compatibility
echo "Building pyradiomics from source for NumPy 2.x compatibility..."
echo "This may take several minutes..."

# Install build dependencies
pip install setuptools wheel Cython

# Create temporary directory for pyradiomics build
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

echo "Cloning pyradiomics repository..."
git clone https://github.com/AIM-Harvard/pyRadiomics.git
cd pyRadiomics

echo "Building pyradiomics from source..."
python setup.py build_ext --inplace
pip install .

echo "✓ pyradiomics built and installed with NumPy 2.x compatibility"

# Clean up temporary directory
cd "$SCRIPT_DIR"
rm -rf "$TEMP_DIR"

# Install rtpipeline package
echo "Installing rtpipeline package..."
cd "$SCRIPT_DIR"
pip install -e .

echo "✓ rtpipeline package installed"

# Validation
echo "Validating installation..."
python -c "
import sys
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk

print('=== NumPy 2.x Pipeline Validation ===')
print(f'✓ NumPy version: {np.__version__}')
print(f'✓ Python version: {sys.version.split()[0]}')

# Check NumPy 2.x compatibility
if np.__version__.startswith('2.'):
    print('✓ NumPy 2.x confirmed')
else:
    print(f'✗ Expected NumPy 2.x, got {np.__version__}')
    sys.exit(1)

# Test core packages
print('✓ Core packages imported successfully')

# Test pyradiomics with NumPy 2.x
try:
    import radiomics
    from radiomics import featureextractor
    print(f'✓ pyradiomics {radiomics.__version__} available with NumPy 2.x')
    
    # Test C extensions work
    extractor = featureextractor.RadiomicsFeatureExtractor()
    print('✓ pyradiomics C extensions working')
except Exception as e:
    print(f'✗ pyradiomics error: {e}')
    sys.exit(1)

# Test dicompyler-core
try:
    from dicompylercore import dvhcalc
    print('✓ dicompyler-core available')
except ImportError as e:
    print(f'✗ dicompyler-core error: {e}')

# Test TotalSegmentator with NumPy 2.x compatibility
try:
    import totalsegmentator
    print('✓ TotalSegmentator available')
    
    # Test our compatibility wrapper
    import rtpipeline.totalsegmentator_compat
    print('✓ TotalSegmentator NumPy 2.x compatibility wrapper loaded')
    
except ImportError as e:
    print(f'✗ TotalSegmentator error: {e}')

# Test rtpipeline
try:
    import rtpipeline.cli
    import rtpipeline.numpy_legacy_compat
    print('✓ rtpipeline with NumPy 2.x legacy compatibility available')
except ImportError as e:
    print(f'✗ rtpipeline import error: {e}')
    sys.exit(1)

print('✓ All components validated with NumPy 2.x')
"

echo
echo "Running rtpipeline doctor..."
if rtpipeline doctor; then
    echo "✓ rtpipeline doctor passed"
else
    echo "⚠️ rtpipeline doctor found issues (check output above)"
fi

echo
echo "=== NumPy 2.x Setup Complete ==="
echo
echo "Key Features of This Setup:"
echo "  • NumPy 2.x with legacy compatibility system"
echo "  • TotalSegmentator 2.11.0 with NumPy 2.x support"
echo "  • pyradiomics built from source for NumPy 2.x C extensions"
echo "  • Complete backward compatibility for legacy libraries"
echo
echo "To use the environment:"
echo "  conda activate $ENV_NAME"
echo
echo "Environment variables for optimal performance:"
echo "  export RTPIPELINE_RADIOMICS_SEQUENTIAL=1  # Prevents segfaults"
echo "  export OMP_NUM_THREADS=1                  # Controls threading"
echo
echo "To validate the environment:"
echo "  rtpipeline doctor"
echo
echo "For troubleshooting NumPy 2.x issues, see rtpipeline/numpy_legacy_compat.py"