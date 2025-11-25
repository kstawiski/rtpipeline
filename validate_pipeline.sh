#!/bin/bash
# Validate pipeline configuration before execution

set -e

echo "Validating pipeline configuration..."

# Check Snakemake dry-run
echo "1. Checking DAG validity..."
snakemake --cores 1 -n --quiet

# Check conda environments
echo "2. Validating conda environments..."
for env in envs/*.yaml; do
    if [ -f "$env" ]; then
         echo "   - $env (exists)"
    fi
done

# Check input directory
echo "3. Checking DICOM input..."
if [ ! -d "Example_data" ] && [ ! -d "Input" ]; then
    echo "WARNING: No Example_data or Input directory found."
else
    echo "   - Input data directory found (OK)"
fi

# Check disk space
echo "4. Checking disk space..."
AVAILABLE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE" -lt 50 ]; then
    echo "WARNING: Less than 50GB available (have ${AVAILABLE}GB)."
else
    echo "   - ${AVAILABLE}GB available (OK)"
fi

echo "✅ Validation complete!"
