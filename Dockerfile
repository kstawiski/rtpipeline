# Multi-stage Docker build for rtpipeline with full feature support
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    pkg-config \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dcm2niix
RUN wget https://github.com/rordenlab/dcm2niix/releases/download/v1.0.20230411/dcm2niix_lnx.zip \
    && unzip dcm2niix_lnx.zip -d /usr/local/bin/ \
    && chmod +x /usr/local/bin/dcm2niix \
    && rm dcm2niix_lnx.zip

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip wheel setuptools

# Install scientific computing stack with version constraints
RUN pip install \
    "numpy>=1.20,<2.0" \
    "pandas>=1.3" \
    "scipy>=1.7" \
    "matplotlib>=3.3" \
    "scikit-image>=0.18" \
    "scikit-learn>=1.0"

# Install medical imaging packages
RUN pip install \
    "pydicom>=3.0,<4" \
    "SimpleITK>=2.1" \
    "dicompyler-core==0.5.6" \
    "pydicom-seg==0.4.1" \
    "rt-utils" \
    "dicom2nifti>=2.4"

# Install visualization and data packages
RUN pip install \
    "openpyxl" \
    "plotly>=5.0" \
    "nested-lookup" \
    "nbformat" \
    "ipython" \
    "jupyter"

# Install TotalSegmentator with proper dependencies
RUN pip install "TotalSegmentator==2.4.0" --no-deps && \
    pip install \
        "torch>=1.10" \
        "torchvision" \
        "nibabel" \
        "tqdm" \
        "requests" \
        "nnunet"

# Install pyradiomics (should work on Python 3.11)
RUN pip install "pyradiomics>=3.0" || echo "pyradiomics installation failed"

# Final stage
FROM python:3.11-slim

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /usr/local/bin/dcm2niix /usr/local/bin/dcm2niix

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# Create app directory
WORKDIR /app

# Copy rtpipeline source
COPY . /app/

# Install rtpipeline in development mode
RUN pip install -e .

# Create data and output directories
RUN mkdir -p /app/data /app/output /app/logs

# Expose port for Jupyter (optional)
EXPOSE 8888

# Set default command
CMD ["bash"]

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import rtpipeline; rtpipeline.cli.main(['doctor'])" || exit 1