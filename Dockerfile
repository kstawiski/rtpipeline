# Dockerfile for rtpipeline - Radiotherapy DICOM Processing Pipeline
# Compatible with Docker and Singularity
# Supports both CPU and GPU execution

# Stage 1: Builder
FROM condaforge/mambaforge:24.3.0-0 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install dcm2niix
RUN wget -q https://github.com/rordenlab/dcm2niix/releases/download/v1.0.20230411/dcm2niix_lnx.zip \
    && unzip dcm2niix_lnx.zip -d /usr/local/bin/ \
    && chmod +x /usr/local/bin/dcm2niix \
    && rm dcm2niix_lnx.zip

# Create app directory
WORKDIR /app

# Copy environment files first for better layer caching
COPY envs/ /app/envs/

# Create conda environments
RUN mamba env create -f /app/envs/rtpipeline.yaml && \
    mamba env create -f /app/envs/rtpipeline-radiomics.yaml && \
    mamba env create -f /app/envs/rtpipeline-custom-models.yaml && \
    mamba clean -afy

# Stage 2: Runtime
FROM condaforge/mambaforge:24.3.0-0

LABEL maintainer="kstawiski"
LABEL description="DICOM-RT pipeline with TotalSegmentator, nnUNet, and Snakemake"
LABEL version="1.1"

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH

# Install runtime dependencies including tini and GL libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libglu1-mesa \
    pigz \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Use tini as init system
ENTRYPOINT ["/usr/bin/tini", "--"]

# Copy dcm2niix from builder
COPY --from=builder /usr/local/bin/dcm2niix /usr/local/bin/dcm2niix

# Configure conda
RUN conda config --set channel_priority strict && \
    conda config --add channels conda-forge && \
    conda config --add channels bioconda && \
    conda config --add channels defaults

# Install Snakemake in base environment
RUN mamba install -y -c conda-forge -c bioconda snakemake>=7.0 && mamba clean -afy

# Copy conda environments from builder
COPY --from=builder /opt/conda/envs /opt/conda/envs

# Create non-root user
RUN groupadd -r rtpipeline && \
    useradd -r -g rtpipeline -u 1000 -m rtpipeline

# Create professional directory structure with correct permissions
RUN mkdir -p \
    /data/input \
    /data/output \
    /data/logs \
    /data/models \
    /data/uploads \
    /tmp/cache \
    /app && \
    chown -R rtpipeline:rtpipeline /data /tmp/cache /app

WORKDIR /app

# Copy project files
COPY --chown=rtpipeline:rtpipeline . /app/
COPY --chown=rtpipeline:rtpipeline .condarc /home/rtpipeline/.condarc

# Install rtpipeline package in base and rtpipeline environments
# We use 'pip install -e .' to allow read-only mounting of code if needed,
# but in this container image we copy the code, so we install it.
RUN pip install -e . && \
    /opt/conda/envs/rtpipeline/bin/pip install -e . && \
    /opt/conda/envs/rtpipeline/bin/pip install psutil

# Install Web UI dependencies
RUN pip install -r /app/webui/requirements.txt

# Create container-specific config
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

# Radiomics robustness analysis
radiomics_robustness:
  enabled: true
  modes:
    - segmentation_perturbation
  segmentation_perturbation:
    apply_to_structures:
      - "GTV*"
      - "CTV*"
      - "PTV*"
      - "pelvic_bones*"
      - "iliac_*"
      - "bowel_bag"
      - "urinary_bladder"
      - "colon"
      - "prostate"
      - "rectum"
    small_volume_changes: [-0.15, 0.0, 0.15]
    large_volume_changes: [-0.30, 0.0, 0.30]
    max_translation_mm: 0.0
    n_random_contour_realizations: 0
    noise_levels: [0.0]
    intensity: "standard"
  metrics:
    icc:
      implementation: "pingouin"
      icc_type: "ICC3"
      ci: true
    cov:
      enabled: true
    qcd:
      enabled: true
  thresholds:
    icc:
      robust: 0.90
      acceptable: 0.75
    cov:
      robust_pct: 10.0
      acceptable_pct: 20.0

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

RUN chown rtpipeline:rtpipeline /app/config.container.yaml

# Environment variables for runtime
ENV SNAKEMAKE_OUTPUT_CACHE="" \
    TMPDIR=/tmp/cache \
    HOME=/home/rtpipeline \
    NUMBA_CACHE_DIR=/tmp/cache \
    MPLCONFIGDIR=/tmp/cache \
    TOTALSEG_WEIGHTS_PATH=/home/rtpipeline/.totalsegmentator/nnunet/results \
    TOTALSEG_TIMEOUT=3600 \
    DCM2NIIX_TIMEOUT=300 \
    RTPIPELINE_RADIOMICS_TASK_TIMEOUT=600

# Pre-create TotalSegmentator weights directory to allow host mounts/caching
RUN mkdir -p /home/rtpipeline/.totalsegmentator/nnunet/results && \
    chown -R rtpipeline:rtpipeline /home/rtpipeline/.totalsegmentator

# Switch to non-root user
USER rtpipeline

# Expose ports
EXPOSE 8888 8080

# Default command - interactive bash
CMD ["/bin/bash"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

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