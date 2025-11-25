#!/bin/bash
# RTpipeline Execution Script
# Platform: Docker
set -e

echo "Starting RTpipeline in Docker..."
docker run -it --rm --gpus all -v "/projekty/Odbytnice/DICOM":/data/input:ro -v "/projekty/Odbytnice/DICOM_rtpipeline":/data/output:rw kstawiski/rtpipeline:latest snakemake --cores all --use-conda --configfile /data/output/config.yaml "$@"
