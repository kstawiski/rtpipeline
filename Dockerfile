# Dockerfile for rtpipeline - Radiotherapy DICOM Processing Pipeline
# Compatible with Docker and Singularity
# Supports both CPU and GPU execution

FROM condaforge/mambaforge:latest

LABEL maintainer="rtpipeline"
LABEL description="DICOM-RT pipeline with TotalSegmentator, nnUNet, and Snakemake"
LABEL version="1.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH

# Install system dependencies
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
    && rm -rf /var/lib/apt/lists/*

# Install dcm2niix
RUN wget -q https://github.com/rordenlab/dcm2niix/releases/download/v1.0.20230411/dcm2niix_lnx.zip \
    && unzip dcm2niix_lnx.zip -d /usr/local/bin/ \
    && chmod +x /usr/local/bin/dcm2niix \
    && rm dcm2niix_lnx.zip

# Configure conda
RUN conda config --set channel_priority strict && \
    conda config --add channels conda-forge && \
    conda config --add channels defaults

# Install Snakemake in base environment
RUN mamba install -y -c conda-forge -c bioconda \
    snakemake>=7.0 \
    snakemake-minimal \
    && mamba clean -afy

# Create app directory
WORKDIR /app

# Copy environment files first for better layer caching
COPY envs/ /app/envs/
COPY .condarc /root/.condarc

# Create conda environments (they will be activated as needed by Snakemake)
# Note: We don't activate them here as Snakemake will manage them
RUN mamba env create -f /app/envs/rtpipeline.yaml && \
    mamba env create -f /app/envs/rtpipeline-radiomics.yaml && \
    mamba env create -f /app/envs/rtpipeline-custom-models.yaml && \
    mamba clean -afy

# Copy project files
COPY . /app/

# Install rtpipeline package in the base environment and rtpipeline conda env
RUN pip install -e . && \
    /opt/conda/envs/rtpipeline/bin/pip install -e .

# Create necessary directories
RUN mkdir -p /app/Data_Snakemake \
    /app/Logs_Snakemake \
    /app/Example_data \
    /data \
    /output \
    /models

# Set working directory
WORKDIR /app

# Environment variables for Singularity compatibility
ENV SNAKEMAKE_OUTPUT_CACHE="" \
    TMPDIR=/tmp \
    HOME=/root

# Expose Jupyter port (optional)
EXPOSE 8888

# Default command - interactive bash
CMD ["/bin/bash"]

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=3 \
    CMD conda info || exit 1

# Singularity-specific labels
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="rtpipeline" \
      org.label-schema.description="Radiotherapy DICOM processing pipeline" \
      org.label-schema.url="https://github.com/kstawiski/rtpipeline" \
      org.label-schema.vcs-url="https://github.com/kstawiski/rtpipeline" \
      org.label-schema.schema-version="1.0"

# Add help for Singularity users
LABEL org.label-schema.usage.singularity.run.command="singularity run rtpipeline.sif snakemake --cores all" \
      org.label-schema.usage.singularity.exec.command="singularity exec rtpipeline.sif snakemake --help"
