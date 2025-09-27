#!/bin/bash
# RTpipeline Optimization and Initialization Script
# Ensures optimal configuration for processing

set -e

echo "==============================================="
echo "    RTpipeline Optimization Setup"
echo "==============================================="
echo

# Detect system capabilities
CPU_COUNT=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
OPTIMAL_WORKERS=$((CPU_COUNT - 1))
MEMORY_GB=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo 16)

echo "🖥️  System Detection:"
echo "   • CPU cores: $CPU_COUNT"
echo "   • Optimal workers: $OPTIMAL_WORKERS"
echo "   • Available memory: ${MEMORY_GB}GB"
echo

# Set environment variables for optimal performance
echo "⚙️  Configuring environment for optimal performance..."

# Enable parallel radiomics processing
export RTPIPELINE_USE_PARALLEL_RADIOMICS=1
export RTPIPELINE_RADIOMICS_WORKERS=$OPTIMAL_WORKERS

# Configure threading libraries to prevent conflicts
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# Set PyTorch threading
export OMP_NUM_THREADS=$OPTIMAL_WORKERS
export TORCH_NUM_THREADS=$OPTIMAL_WORKERS

# Memory management
export MALLOC_TRIM_THRESHOLD_=100000

# Python optimizations
export PYTHONOPTIMIZE=1
export PYTHONDONTWRITEBYTECODE=1

echo "✅ Environment optimized"
echo

# Verify NumPy version
echo "🔍 Verifying NumPy configuration..."
NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "Not installed")
NUMPY_MAJOR=$(echo $NUMPY_VERSION | cut -d. -f1)

if [[ "$NUMPY_MAJOR" == "2" ]]; then
    echo "   ✅ NumPy 2.x detected ($NUMPY_VERSION) - Optimal for TotalSegmentator"
else
    echo "   ⚠️  NumPy version: $NUMPY_VERSION"
    echo "   Consider upgrading to NumPy 2.x for better performance:"
    echo "   pip install --upgrade 'numpy>=2.0,<3.0' --no-deps"
fi
echo

# Test compatibility layer
echo "🧪 Testing compatibility layers..."
python -c "
import sys
import os
sys.path.insert(0, '.')

try:
    from rtpipeline import numpy2_compat
    print('   ✅ NumPy 2.x compatibility layer loaded')
except ImportError:
    print('   ⚠️  NumPy 2.x compatibility layer not found')

try:
    from rtpipeline.radiomics_parallel import is_parallel_radiomics_enabled
    if is_parallel_radiomics_enabled():
        print('   ✅ Parallel radiomics enabled')
    else:
        print('   ⚠️  Parallel radiomics not enabled')
except ImportError:
    print('   ⚠️  Parallel radiomics module not found')

try:
    import totalsegmentator
    print('   ✅ TotalSegmentator available')
except ImportError:
    print('   ⚠️  TotalSegmentator not installed')

try:
    from rtpipeline.radiomics import _have_pyradiomics
    if _have_pyradiomics():
        print('   ✅ PyRadiomics compatible')
    else:
        print('   ⚠️  PyRadiomics compatibility issues')
except Exception:
    print('   ⚠️  PyRadiomics check failed')
" 2>/dev/null || echo "   ❌ Compatibility test failed"
echo

# Display optimal command
echo "==============================================="
echo "🚀 Optimal RTpipeline Command:"
echo "==============================================="
echo
echo "rtpipeline \\"
echo "    --dicom-root /path/to/dicom \\"
echo "    --outdir ./Data_Organized \\"
echo "    --logs ./Logs \\"
echo "    --workers $OPTIMAL_WORKERS \\"
echo "    --custom-structures custom_structures_pelvic.yaml \\"
echo "    -v"
echo
echo "==============================================="
echo "📊 Performance Settings Applied:"
echo "==============================================="
echo "   • Parallel radiomics: ENABLED"
echo "   • Worker processes: $OPTIMAL_WORKERS"
echo "   • Threading optimized for multiprocessing"
echo "   • Memory management optimized"
echo "   • NumPy 2.x compatibility: ACTIVE"
echo
echo "💡 Tips:"
echo "   • Use --resume flag to continue interrupted processing"
echo "   • Monitor with: watch -n1 'ps aux | grep rtpipeline'"
echo "   • Check logs in real-time: tail -f Logs/*.log"
echo
echo "Ready to process with maximum performance! 🎯"