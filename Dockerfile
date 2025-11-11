# Dockerfile for rtpipeline - Radiotherapy DICOM Processing Pipeline
# Compatible with Docker and Singularity
# Supports both CPU and GPU execution

FROM condaforge/mambaforge:latest

LABEL maintainer="kstawiski"
LABEL description="DICOM-RT pipeline with TotalSegmentator, nnUNet, and Snakemake"
LABEL version="1.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH

# Install system dependencies including tini for proper process management
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libglu1-mesa \
    pigz \
    unzip \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Use tini as init system for proper signal handling and zombie process reaping
# Critical for parallel processing and timeout mechanisms
ENTRYPOINT ["/usr/bin/tini", "--"]

# Install dcm2niix (required for DICOM to NIfTI conversion)
RUN wget -q https://github.com/rordenlab/dcm2niix/releases/download/v1.0.20230411/dcm2niix_lnx.zip \
    && unzip dcm2niix_lnx.zip -d /usr/local/bin/ \
    && chmod +x /usr/local/bin/dcm2niix \
    && rm dcm2niix_lnx.zip

# Configure conda with strict channel priority
RUN conda config --set channel_priority strict && \
    conda config --add channels conda-forge && \
    conda config --add channels bioconda && \
    conda config --add channels defaults

# Install Snakemake in base environment
RUN mamba install -y -c conda-forge -c bioconda \
    snakemake>=7.0 \
    && mamba clean -afy

# Create app directory
WORKDIR /app

# Copy environment files first for better layer caching
COPY envs/ /app/envs/
COPY .condarc /root/.condarc

# Create conda environments (Snakemake will activate them as needed)
RUN mamba env create -f /app/envs/rtpipeline.yaml && \
    mamba env create -f /app/envs/rtpipeline-radiomics.yaml && \
    mamba env create -f /app/envs/rtpipeline-custom-models.yaml && \
    mamba clean -afy

# Copy project files
COPY . /app/

# Install rtpipeline package in base and rtpipeline environments
RUN pip install -e . && \
    /opt/conda/envs/rtpipeline/bin/pip install -e .

# Install Web UI dependencies
RUN pip install -r /app/webui/requirements.txt

# Install psutil for better CPU detection in containers
RUN pip install psutil && \
    /opt/conda/envs/rtpipeline/bin/pip install psutil

# Create professional directory structure
RUN mkdir -p \
    /data/input \
    /data/output \
    /data/logs \
    /data/models \
    /data/uploads \
    /tmp/cache

# Create container-specific config that uses proper paths
RUN cat > /app/config.container.yaml << 'EOF'
# Container-optimized configuration for rtpipeline
# This config uses professional container paths

# Input/Output directories (container paths)
dicom_root: "/data/input"
output_dir: "/data/output"
logs_dir: "/data/logs"

# Processing parameters
workers: auto

segmentation:
  workers: 4
  threads_per_worker: null
  force: false
  fast: false
  roi_subset: null
  extra_models: []

custom_models:
  enabled: false
  root: "/data/models"
  models: []
  workers: 1
  force: false
  nnunet_predict: "nnUNet_predict"
  retain_weights: true
  conda_activate: null

radiomics:
  sequential: false
  params_file: "rtpipeline/radiomics_params.yaml"
  mr_params_file: "rtpipeline/radiomics_params_mr.yaml"
  thread_limit: 4
  skip_rois:
    - body
    - couchsurface
    - couchinterior
    - couchexterior
    - bones
    - m1
    - m2
  max_voxels: 1500000000
  min_voxels: 10

aggregation:
  threads: auto

environments:
  main: "rtpipeline"
  radiomics: "rtpipeline-radiomics"

custom_structures: "custom_structures_pelvic.yaml"

ct_cropping:
  enabled: true
  region: "pelvis"
  superior_margin_cm: 2.0
  inferior_margin_cm: 10.0
  use_cropped_for_dvh: true
  use_cropped_for_radiomics: true
  keep_original: true
EOF

# Set working directory
WORKDIR /app

# Environment variables for runtime
ENV SNAKEMAKE_OUTPUT_CACHE="" \
    TMPDIR=/tmp \
    HOME=/root \
    NUMBA_CACHE_DIR=/tmp/cache \
    MPLCONFIGDIR=/tmp/cache \
    TOTALSEG_WEIGHTS_PATH=/root/.totalsegmentator/nnunet/results \
    TOTALSEG_TIMEOUT=3600 \
    DCM2NIIX_TIMEOUT=300 \
    RTPIPELINE_RADIOMICS_TASK_TIMEOUT=600

# Pre-create TotalSegmentator weights directory to allow host mounts/caching
RUN mkdir -p /root/.totalsegmentator/nnunet/results

# Expose ports
EXPOSE 8888 8080

# Default command - interactive bash
CMD ["/bin/bash"]

# Health check
HEALTHCHECK --interval=300s --timeout=30s --start-period=120s --retries=3 \
    CMD conda info && python -c "import rtpipeline" || exit 1

# Singularity-specific labels
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="rtpipeline" \
      org.label-schema.description="Radiotherapy DICOM processing pipeline" \
      org.label-schema.url="https://github.com/kstawiski/rtpipeline" \
      org.label-schema.vcs-url="https://github.com/kstawiski/rtpipeline" \
      org.label-schema.schema-version="1.0"

# Add help for Singularity users
LABEL org.label-schema.usage.singularity.run.command="singularity run rtpipeline.sif snakemake --cores all --configfile /app/config.container.yaml" \
      org.label-schema.usage.singularity.exec.command="singularity exec rtpipeline.sif snakemake --help"
