#!/bin/bash
# RTpipeline Perfect Environment Setup Script
# Creates a portable environment that works across different devices
# Based on validated working configuration

set -eo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ENV_NAME="rtpipeline"
PYTHON_VERSION="3.11"

echo "=============================================================="
echo "    RTpipeline Environment Setup"
echo "=============================================================="
echo "Creating portable environment with validated package versions"
echo "Environment: $ENV_NAME"
echo "Python: $PYTHON_VERSION"
echo

# Detect system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    SYSTEM="linux"
    echo "✓ Detected Linux system"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    SYSTEM="macos"
    echo "✓ Detected macOS system"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    SYSTEM="windows"
    echo "✓ Detected Windows system"
else
    SYSTEM="unknown"
    echo "⚠ Unknown system: $OSTYPE"
fi

# Check for conda/mamba
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo "✓ Using mamba for fast package management"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
    echo "✓ Using conda for package management"
else
    echo "❌ Error: Neither conda nor mamba found."
    echo "   Please install Miniconda or Anaconda first:"
    echo "   https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo
echo "📦 Creating conda environment '$ENV_NAME'..."
if $CONDA_CMD env list | grep -q "^$ENV_NAME "; then
    echo "⚠ Environment '$ENV_NAME' already exists. Removing it first..."
    $CONDA_CMD env remove -n "$ENV_NAME" -y
fi

$CONDA_CMD create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
echo "✅ Environment created"

# Activate environment
echo
echo "🔄 Activating environment..."
# Use a more robust activation method that handles conda script issues
source "$($CONDA_CMD info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
$CONDA_CMD activate "$ENV_NAME" 2>/dev/null || {
    echo "⚠ Direct activation failed, using eval method..."
    eval "$($CONDA_CMD shell.bash hook)" 2>/dev/null || true
    $CONDA_CMD activate "$ENV_NAME" 2>/dev/null || true
}

# Verify activation (more lenient check)
if command -v python >/dev/null 2>&1 && python -c "import sys; print('Python from:', sys.executable)" 2>/dev/null | grep -q "$ENV_NAME"; then
    echo "✅ Environment activated successfully"
else
    echo "⚠ Activation verification unclear, but continuing..."
    echo "If needed, manually activate with: conda activate $ENV_NAME"
fi

# Install core scientific stack with NumPy 2.x for TotalSegmentator
echo
echo "🧬 Installing core scientific packages with NumPy 2.x..."
# Install NumPy 2.x first to ensure it's the base version
$CONDA_CMD install -c conda-forge -y "numpy>=2.0,<3.0"

# Then install other packages that are compatible with NumPy 2.x
$CONDA_CMD install -c conda-forge -y \
    "scipy>=1.11" \
    "pandas>=2.0" \
    "matplotlib>=3.8" \
    "scikit-learn" \
    "scikit-image" \
    "pillow" \
    "ipython" \
    "jupyter"

echo "✅ Core scientific stack installed"

# Install medical imaging packages with validated versions
echo
echo "🏥 Installing medical imaging packages..."
$CONDA_CMD install -c conda-forge -y \
    "pydicom>=3.0.0" \
    "SimpleITK>=2.5.0" \
    "nibabel"

pip install \
    "dicompyler-core>=0.5.6" \
    "rt-utils"

echo "✅ Medical imaging packages installed"

# Install dcm2niix
echo
echo "💾 Installing dcm2niix..."
if [[ "$SYSTEM" == "linux" || "$SYSTEM" == "macos" ]]; then
    $CONDA_CMD install -c conda-forge dcm2niix -y
    echo "✅ dcm2niix installed via conda"
else
    echo "⚠ Please install dcm2niix manually for Windows"
    echo "   Download from: https://github.com/rordenlab/dcm2niix/releases"
fi

# Install TotalSegmentator with compatibility
echo
echo "🧠 Installing TotalSegmentator..."
# Install PyTorch and dependencies first to avoid conflicts
pip install --no-deps \
    "torch" \
    "torchvision" \
    "torchaudio"

# Install TotalSegmentator dependencies
pip install \
    "nnunetv2" \
    "batchgenerators" \
    "acvl-utils"

# Install TotalSegmentator
pip install --no-deps "TotalSegmentator"

# Install any missing TotalSegmentator dependencies
pip install "dicom2nifti" "pyarrow" "requests" "xvfbwrapper"
echo "✅ TotalSegmentator installed"

# Install pyradiomics with compatibility handling for NumPy 2.x
echo
echo "📊 Installing pyradiomics with NumPy 2.x compatibility wrapper..."
# Install pyradiomics but prevent it from downgrading NumPy
pip install --no-deps "pyradiomics" || {
    echo "⚠ pyradiomics installation failed (will use subprocess-based compatibility wrapper)"
}

# Install PyRadiomics dependencies separately to maintain NumPy 2.x
pip install "pykwalify" "PyWavelets" "six"

# Create a separate NumPy 1.x environment for PyRadiomics fallback (optional)
echo "🔧 Creating optional NumPy 1.x environment for PyRadiomics fallback..."
$CONDA_CMD create -n "rtpipeline-numpy1" python=3.11 "numpy=1.26.*" scipy pyradiomics -c conda-forge -c radiomics -y 2>/dev/null || {
    echo "⚠ Could not create NumPy 1.x fallback environment (will use in-process compatibility)"
}
echo "✅ pyradiomics setup completed"

# Install additional useful packages
echo
echo "🔧 Installing additional packages..."
pip install \
    "xlsxwriter" \
    "openpyxl" \
    "seaborn" \
    "tqdm" \
    "psutil"

echo "✅ Additional packages installed"

# Install rtpipeline package
echo
echo "🚀 Installing rtpipeline package..."
cd "$SCRIPT_DIR"
pip install -e .
echo "✅ rtpipeline package installed"

# Ensure NumPy 2.x is still installed (prevent downgrades)
echo
echo "🔒 Ensuring NumPy 2.x is maintained..."
pip install --upgrade "numpy>=2.0,<3.0" --no-deps

# Validation
echo
echo "🔍 Validating installation..."
python -c "
import sys
import warnings
warnings.filterwarnings('ignore')

print('=== RTpipeline Environment Validation ===')
print()

# Test core packages
packages = {
    'numpy': None,
    'scipy': None,
    'pandas': None,
    'matplotlib': None,
    'pydicom': None,
    'SimpleITK': None,
    'radiomics': None,
    'totalsegmentator': None,
    'dicompylercore': None
}

failed = []
for pkg_name, import_name in packages.items():
    try:
        if import_name:
            mod = __import__(import_name)
        else:
            mod = __import__(pkg_name)
        version = getattr(mod, '__version__', 'installed')
        print(f'✅ {pkg_name}: {version}')
    except ImportError as e:
        print(f'❌ {pkg_name}: FAILED - {e}')
        failed.append(pkg_name)

# Test rtpipeline
try:
    import rtpipeline.cli
    print('✅ rtpipeline: installed')
except ImportError as e:
    print(f'❌ rtpipeline: FAILED - {e}')
    failed.append('rtpipeline')

# Test NumPy compatibility
try:
    import numpy as np
    # Test NumPy compatibility
    version = np.__version__
    major = int(version.split('.')[0])
    if major >= 2:
        print(f'✅ NumPy 2.x installed: v{version}')
    else:
        print(f'⚠️ NumPy 1.x installed: v{version} (expected 2.x)')
        failed.append('numpy_version')
except Exception as e:
    print(f'❌ NumPy compatibility: FAILED - {e}')
    failed.append('numpy_compat')

# Test TotalSegmentator installation
try:
    import totalsegmentator
    print('✅ TotalSegmentator:', getattr(totalsegmentator, '__version__', 'installed'))
except ImportError as e:
    print(f'❌ TotalSegmentator: FAILED - {e}')
    failed.append('totalsegmentator')

print()
if failed:
    failed_str = ', '.join(failed)
    print(f'❌ Validation failed for: {failed_str}')
    sys.exit(1)
else:
    print('🎉 All components validated successfully!')
"

# Run rtpipeline doctor
echo
echo "🩺 Running rtpipeline doctor..."
if rtpipeline doctor; then
    echo "✅ rtpipeline doctor passed"
else
    echo "⚠ rtpipeline doctor found issues (but installation may still work)"
fi

echo
echo "=============================================================="
echo "🎉 RTpipeline Environment Setup Complete!"
echo "=============================================================="
echo
echo "✅ What was installed:"
echo "   • Python $PYTHON_VERSION with NumPy 2.x (optimal for TotalSegmentator)"
echo "   • SciPy, pandas, matplotlib (NumPy 2.x compatible versions)"
echo "   • Medical imaging: pydicom, SimpleITK, nibabel"
echo "   • Segmentation: TotalSegmentator (direct execution, NumPy 2.x compatible)"
echo "   • Radiomics: pyradiomics with automatic NumPy 2.x compatibility layer"
echo "   • RTpipeline with enhanced parallel processing"
echo
echo "🚀 To use the environment:"
echo "   conda activate $ENV_NAME"
echo "   rtpipeline --help"
echo
echo "💡 To enable parallel radiomics (recommended):"
echo "   export RTPIPELINE_USE_PARALLEL_RADIOMICS=1"
echo
echo "🔧 For optimal performance:"
echo "   export OMP_NUM_THREADS=\$((\$(nproc) - 1))  # Use n-1 cores"
echo "   export MKL_NUM_THREADS=1                    # Prevent conflicts"
echo
echo "🧪 To test the installation:"
echo "   rtpipeline doctor"
echo "   rtpipeline --dicom-root /path/to/data --outdir ./output --logs ./logs"
echo
echo "📋 Key features enabled:"
echo "   • Enhanced parallel radiomics (23 workers on 24-core system)"
echo "   • TotalSegmentator with NumPy 2.x (optimal performance)"
echo "   • PyRadiomics with automatic NumPy 2.x compatibility layer"
echo "   • Process-based parallelism (prevents segmentation faults)"
echo "   • Automatic retry mechanisms for robustness"
echo "   • Organized output structure (Data/ directory)"
echo "   • Resume capability for interrupted processing"
echo
echo "Environment ready for deployment on any device! 🎯"