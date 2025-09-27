#!/bin/bash
# Optimized rtpipeline command with NumPy 2.x compatibility and parallel processing
# This script ensures maximum performance with all compatibility layers active

# Source optimization settings
if [ -f "optimize_pipeline.sh" ]; then
    echo "Loading optimization settings..."
    source optimize_pipeline.sh > /dev/null 2>&1
fi

# Enable parallel radiomics for maximum performance
export RTPIPELINE_USE_PARALLEL_RADIOMICS=1

# Set optimal workers based on CPU cores
CPU_COUNT=$(nproc 2>/dev/null || echo 4)
OPTIMAL_WORKERS=$((CPU_COUNT - 1))

echo "Starting RTpipeline with optimized settings:"
echo "  • NumPy version: $(python -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo 'unknown')"
echo "  • Workers: $OPTIMAL_WORKERS"
echo "  • Parallel radiomics: ENABLED"
echo

# Run pipeline with optimal settings
rtpipeline \
    --dicom-root Example_data \
    --outdir ./Data_Organized \
    --logs ./Logs \
    --workers $OPTIMAL_WORKERS \
    --custom-structures custom_structures_pelvic.yaml \
    -v