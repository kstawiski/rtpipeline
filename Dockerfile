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
COPY third_party/ /app/third_party/

# Provide stable absolute path expected by conda env specs
RUN mkdir -p /projekty && ln -s /app /projekty/rtpipeline

# Create conda environments with aggressive cleanup
RUN mamba env create -f /app/envs/rtpipeline.yaml && \
    mamba env create -f /app/envs/rtpipeline-radiomics.yaml && \
    mamba env create -f /app/envs/rtpipeline-custom-models.yaml && \
    mamba clean -afy && \
    find /opt/conda -follow -type f -name '*.a' -delete && \
    find /opt/conda -follow -type f -name '*.pyc' -delete && \
    find /opt/conda -follow -type f -name '*.js.map' -delete && \
    rm -rf /opt/conda/pkgs/*

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
    PATH=/opt/conda/bin:$PATH \
    LD_LIBRARY_PATH=/opt/conda/lib

# Install runtime dependencies, create user, and setup directories in one layer
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
    && rm -rf /var/lib/apt/lists/* && \
    groupadd -r rtpipeline && \
    useradd -r -g rtpipeline -u 1000 -m rtpipeline && \
    mkdir -p \
    /data/input \
    /data/output \
    /data/logs \
    /data/models \
    /data/uploads \
    /tmp/cache \
    /app && \
    mkdir -p /projekty && \
    ln -s /app /projekty/rtpipeline && \
    chown -R rtpipeline:rtpipeline /data /tmp/cache /app

# Use tini as init system
ENTRYPOINT ["/usr/bin/tini", "--"]

# Copy dcm2niix from builder
COPY --from=builder /usr/local/bin/dcm2niix /usr/local/bin/dcm2niix

# Configure conda
RUN conda config --set channel_priority strict && \
    conda config --add channels conda-forge && \
    conda config --add channels bioconda && \
    conda config --add channels defaults

# Install Snakemake and Web UI dependencies in base environment
# We install these via mamba to ensure GLIBCXX compatibility (pandas/cpp issues with pip wheels)
# pulp + coinor-cbc enable Snakemake's ILP scheduler for optimal job ordering
RUN mamba install -y -c conda-forge -c bioconda \
    python=3.11 \
    'snakemake>=8.4' \
    pandas \
    flask \
    psutil \
    pydicom \
    pyyaml \
    werkzeug \
    openpyxl \
    pynetdicom \
    libstdcxx-ng \
    pulp \
    coinor-cbc \
    && mamba clean -afy && \
    find /opt/conda -follow -type f -name '*.a' -delete && \
    find /opt/conda -follow -type f -name '*.pyc' -delete && \
    find /opt/conda -follow -type f -name '*.js.map' -delete && \
    rm -rf /opt/conda/pkgs/*

# Copy conda environments from builder
COPY --from=builder /opt/conda/envs /opt/conda/envs

# Copy TotalSegmentator weights for offline support
# NOTE: Ensure the 'totalseg_weights/' directory exists in the build context before building.
# The source directory should contain: nnunet/results/Dataset*/...
# Omit this COPY if you do not need offline TotalSegmentator execution.
COPY --chown=rtpipeline:rtpipeline totalseg_weights/ /home/rtpipeline/.totalsegmentator/

# Copy custom models into the image
# This ensures the image is self-contained
COPY --chown=rtpipeline:rtpipeline custom_models /data/models/

WORKDIR /app

# Copy project files
COPY --chown=rtpipeline:rtpipeline . /app/

# Install rtpipeline package in base and rtpipeline environments
# We use 'pip install -e .' to allow read-only mounting of code if needed,
# but in this container image we copy the code, so we install it.
RUN pip install -e . && \
    /opt/conda/envs/rtpipeline/bin/pip install -e . && \
    /opt/conda/envs/rtpipeline/bin/pip install psutil



# Create container-specific config
RUN cat > /app/config.container.yaml << 'EOF'
# Container-optimized configuration for rtpipeline
# This config uses professional container paths

# Input/Output directories (container paths)
dicom_root: "/data/input"
output_dir: "/data/output"
logs_dir: "/data/logs"

# Processing parameters
max_workers: null

scheduler:
  reserved_cores: 1
  dvh_threads_per_job: 4
  radiomics_threads_per_job: 6
  qc_threads_per_job: 2
  prioritize_short_courses: true

segmentation:
  device: "gpu"
  fast: false
  max_workers: null
  force: false
  roi_subset: null
  extra_models: []
  force_split: true
  nr_threads_resample: 1
  nr_threads_save: 1
  num_proc_preprocessing: 1
  num_proc_export: 1

custom_models:
  enabled: true
  root: "/data/models"
  models: []
  max_workers: 1
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
    TOTALSEG_WEIGHTS_PATH=/home/rtpipeline/.totalsegmentator \
    nnUNet_results=/home/rtpipeline/.totalsegmentator/nnunet/results \
    nnUNet_raw=/home/rtpipeline/.totalsegmentator/nnunet/raw \
    nnUNet_preprocessed=/home/rtpipeline/.totalsegmentator/nnunet/preprocessed \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TOTALSEG_TIMEOUT=3600 \
    DCM2NIIX_TIMEOUT=300 \
    RTPIPELINE_RADIOMICS_TASK_TIMEOUT=600 \
    SNAKEMAKE_CONDA_PREFIX=/home/rtpipeline/.snakemake/conda

# Pre-create TotalSegmentator weights and nnUNet directories
RUN mkdir -p /home/rtpipeline/.totalsegmentator/nnunet/results \
             /home/rtpipeline/.totalsegmentator/nnunet/raw \
             /home/rtpipeline/.totalsegmentator/nnunet/preprocessed \
             $SNAKEMAKE_CONDA_PREFIX && \
    chown -R rtpipeline:rtpipeline /home/rtpipeline/.totalsegmentator /home/rtpipeline/.snakemake

# Switch to non-root user
USER rtpipeline

# Warm up Snakemake-managed conda envs so runtime jobs reuse the baked environments
RUN snakemake \
    --cores 1 \
    --use-conda \
    --conda-prefix "$SNAKEMAKE_CONDA_PREFIX" \
    --conda-create-envs-only \
    --configfile /app/config.container.yaml

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
