#!/bin/bash
set -euo pipefail

if pgrep -f "snakemake" >/dev/null; then
  echo "Another snakemake process is currently running. Please stop it before running tests." >&2
  exit 1
fi

# Ensure pip can reuse conda-provided numpy when building pyradiomics wheels.
export PIP_NO_BUILD_ISOLATION=1

# rm -rf Data_Snakemake Data_Snakemake_fallback _RESULTS Logs_Snakemake Logs_Snakemake_fallback .snakemake || true

snakemake --unlock >/dev/null 2>&1 || true

XDG_CACHE_HOME=$PWD/.cache snakemake --cores all --use-conda --rerun-incomplete --conda-prefix "$HOME/.snakemake_conda_store"
