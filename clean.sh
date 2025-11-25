#!/bin/bash
set -e

echo "Cleaning up pipeline outputs and logs..."

# Remove Snakemake outputs and working directories
rm -rf Data_Snakemake
rm -rf Logs_Snakemake
rm -rf .snakemake
rm -rf Data_Snakemake_fallback
rm -rf Logs_Snakemake_fallback

# Remove Docker/Container outputs
rm -rf Output
rm -rf Input
rm -rf Logs
rm -rf Uploads

# Remove aggregated results
rm -rf _RESULTS

# Remove development artifacts
rm -rf .genkit
rm -rf .cache
rm -rf .ruff_cache
rm -rf .mypy_cache
rm -rf .pytest_cache

# Remove Python cache files
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name ".ipynb_checkpoints" -type d -exec rm -rf {} + 2>/dev/null || true

# Remove egg-info (will be regenerated on install)
rm -rf rtpipeline.egg-info

# Remove pipeline report if exists
rm -f pipeline_report.html

# Remove any temporary NIfTI files in root
rm -f *.nii *.nii.gz 2>/dev/null || true

echo "Cleanup complete."
echo ""
echo "Note: Example_data/, custom_models/, and totalseg_weights/ are preserved."
