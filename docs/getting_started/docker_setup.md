# Docker Setup Guide for rtpipeline

## Overview

This guide provides a comprehensive walkthrough for setting up and running **rtpipeline** using Docker. It covers everything from basic setup to advanced configuration, addressing common questions about configuration, default behavior, and real-world usage patterns.

**Target Audience:**
- Users who want to run rtpipeline without installing dependencies locally
- HPC/cluster users who need reproducible environments
- Users who need to process multiple patients in batch mode
- Researchers who want comprehensive, research-grade analysis

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding Docker Deployment Options](#understanding-docker-deployment-options)
3. [Default Configuration Explained](#default-configuration-explained)
4. [How to Pass Configuration](#how-to-pass-configuration)
5. [Setup New Analysis Using Docker](#setup-new-analysis-using-docker)
6. [Real-World Examples](#real-world-examples)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Topics](#advanced-topics)

---

## Quick Start

**The fastest way to get started is using the interactive setup tool:**

```bash
# One-line installer (no git clone needed)
curl -sSL https://raw.githubusercontent.com/kstawiski/rtpipeline/main/setup_docker_project.sh | bash

# Or use local script if you've cloned the repository
./setup_docker_project.sh
```

This will guide you through:
- Selecting your DICOM input directory
- Configuring output locations
- Choosing analysis features (radiomics robustness, CT cropping, etc.)
- Generating Docker configuration files
- Creating ready-to-run scripts

**Or use Web UI (simplest):**

```bash
# Create required directories
mkdir -p Input Output Logs Uploads

# Start Web UI
docker-compose up -d

# Open browser to http://localhost:8080
# Upload DICOM files and click "Start Processing"
```

---

## Understanding Docker Deployment Options

rtpipeline offers three Docker deployment modes:

### 1. **Web UI Mode** (Recommended for Beginners)

**Best for:** Interactive use, one-off analyses, users unfamiliar with command line

```bash
docker-compose up -d
# Access at http://localhost:8080
```

**Features:**
- Drag-and-drop DICOM upload
- Interactive configuration via web interface
- Real-time progress monitoring
- One-click results download
- No command-line knowledge required

**How it works:**
1. Upload DICOM files (zip or folder)
2. Configure analysis options in web form
3. Web UI generates configuration automatically
4. Pipeline runs in background
5. Download results when complete

---

### 2. **CLI Mode with docker run** (For Power Users)

**Best for:** Automated workflows, scripting, batch processing, integration with other tools

```bash
docker run -it --rm --gpus all --shm-size=8g \
  -v /path/to/dicom:/data/input:ro \
  -v /path/to/output:/data/output:rw \
  -v /path/to/logs:/data/logs:rw \
  -v /path/to/config.yaml:/app/config.custom.yaml:ro \
  kstawiski/rtpipeline:latest \
  snakemake --cores all --use-conda --configfile /app/config.custom.yaml

# Optional: -v /path/to/totalseg_weights:/home/rtpipeline/.totalsegmentator:rw (for caching/updating weights)
```

**Features:**
- Full control over configuration
- Easy to script and automate
- Direct access to Snakemake commands
- Suitable for CI/CD pipelines

---

### 3. **docker-compose CLI Mode** (For Reproducible Projects)

**Best for:** Team projects, reproducible analyses, version-controlled configurations

```bash
# docker-compose.custom.yml
version: '3.8'
services:
  rtpipeline-analysis:
    image: kstawiski/rtpipeline:latest
    volumes:
      - ./MyProject/DICOM:/data/input:ro
      - ./MyProject/Output:/data/output:rw
      - ./MyProject/config.yaml:/app/config.custom.yaml:ro
    command: snakemake --cores 16 --use-conda --configfile /app/config.custom.yaml
```

```bash
docker-compose -f docker-compose.custom.yml run --rm rtpipeline-analysis
```

**Features:**
- Configuration version-controlled in git
- Easy to share with collaborators
- Reproducible across machines
- Can define multiple services for different analyses

---

## Default Configuration Explained

### What Does the Default Configuration Do?

The default `config.yaml` in the Docker container is located at `/app/config.yaml` and enables a **comprehensive research-grade analysis** with the following features:

#### **Enabled by Default:**

| Feature | Default Setting | What It Does | When to Disable |
|---------|----------------|--------------|-----------------|
| **TotalSegmentator** | `enabled: true` | Automatic segmentation of 104 anatomical structures (CT) and 59 structures (MR) | Never (core feature) |
| **Radiomics Extraction** | `enabled: true` | Extracts 100+ radiomic features per ROI (shape, intensity, texture) | For quick DVH-only analysis |
| **Radiomics Robustness** | `enabled: true` | Evaluates feature stability via perturbations (ICC, CoV, QCD metrics) | For quick testing or clinical DVH analysis |
| **CT Cropping** | `enabled: true` | Crops all CTs to consistent anatomical boundaries (e.g., L1 to femoral heads) | When analyzing non-standard regions or single patients |
| **DVH Metrics** | `enabled: true` | Generates dose-volume histograms for all structures | Never (core feature) |
| **Custom Structures** | `enabled: true` | Boolean synthesis of composite structures (e.g., bowel_bag, pelvic_bones) | When using standard TotalSegmentator structures only |

#### **Disabled by Default:**

| Feature | Default Setting | Reason | How to Enable |
|---------|----------------|--------|---------------|
| **Custom nnU-Net Models** | `enabled: false` | Model weights not included in image (size limitations) | Download weights, set `custom_models.enabled: true` |
| **Fast Mode** | `fast: false` | Prioritizes accuracy over speed | Set `segmentation.fast: true` for 3x faster (lower quality) |
| **Force Re-segmentation** | `force: false` | Avoids re-running completed stages | Set `segmentation.force: true` to re-run |

---

### Default Parallelization Behavior

The pipeline automatically adjusts parallelization based on available resources:

```yaml
# config.yaml defaults
max_workers: null  # Auto-detects: (CPU cores - 1)

scheduler:
  reserved_cores: 1              # Keep 1 core for OS
  dvh_threads_per_job: 4         # 4 threads per DVH job
  radiomics_threads_per_job: 6   # 6 threads per radiomics job
  prioritize_short_courses: true # Process small courses first

segmentation:
  workers: null  # GPU: sequential, CPU: uses max_workers
  device: "gpu"  # Use GPU if available
  force_split: true  # Reduce memory spikes
```

**What this means in practice:**

- **16-core system with GPU:**
  - Segmentation: 1 patient at a time (GPU sequential)
  - DVH: Up to 3-4 concurrent patients (15 cores ÷ 4 threads)
  - Radiomics: Up to 2 concurrent patients (15 cores ÷ 6 threads)

- **32-core system without GPU:**
  - Segmentation: Up to 7-8 concurrent patients (CPU mode uses ~25% of cores per job)
  - DVH: Up to 7 concurrent patients
  - Radiomics: Up to 5 concurrent patients

---

### What Does Radiomics Robustness Do?

**Default configuration:**

```yaml
radiomics_robustness:
  enabled: true  # ⚠️ Adds significant processing time!
  segmentation_perturbation:
    intensity: "standard"  # 15-30 perturbations per ROI
    apply_to_structures:
      - "GTV*"  # Tumor volumes
      - "CTV*"  # Clinical targets
      - "PTV*"  # Planning targets
      - "urinary_bladder"
      - "colon"
      - "prostate"
      - "rectum"
    small_volume_changes: [-0.15, 0.0, 0.15]  # ±15% erosion/dilation
```

**What happens:**

1. For each ROI (e.g., `GTV_1`), the pipeline creates 15-30 perturbed versions:
   - Volume changes: -15%, 0%, +15%
   - Translation shifts: ±3mm in X/Y/Z
   - Contour randomization: 2 random realizations

2. Radiomics features are extracted from each perturbed version

3. Robustness metrics are computed:
   - **ICC (Intraclass Correlation)**: ICC ≥ 0.90 = "robust", ICC < 0.75 = "poor"
   - **CoV (Coefficient of Variation)**: CoV ≤ 10% = "robust", CoV > 20% = "poor"
   - **QCD (Quartile Coefficient of Dispersion)**: Additional stability metric

4. Results saved to: `<course>/radiomics_robustness/`

**Processing time impact:**

- Standard intensity: ~3-5x longer than basic radiomics
- Aggressive intensity: ~6-10x longer

**When to disable:**

```yaml
radiomics_robustness:
  enabled: false  # For quick DVH analysis only
```

---

### What Does CT Cropping Do?

**Default configuration:**

```yaml
ct_cropping:
  enabled: true
  region: "pelvis"  # L1 vertebra → femoral heads + 10cm
  superior_margin_cm: 2.0
  inferior_margin_cm: 10.0
  use_cropped_for_dvh: true
  use_cropped_for_radiomics: true
  keep_original: true
```

**Problem it solves:**

Without cropping, percentage DVH metrics (V95%, V20Gy) are meaningless because CT field-of-view varies:

```
Patient A: CT volume = 18,000 cm³ → V20Gy = 500 cm³ / 18,000 cm³ = 2.8%
Patient B: CT volume = 15,000 cm³ → V20Gy = 500 cm³ / 15,000 cm³ = 3.3%
Same absolute dose, different percentages! ❌
```

**After cropping** (all patients cropped to same boundaries):

```
Patient A: CT volume = 12,000 cm³ → V20Gy = 500 cm³ / 12,000 cm³ = 4.2%
Patient B: CT volume = 12,000 cm³ → V20Gy = 500 cm³ / 12,000 cm³ = 4.2%
Now comparable! ✅
```

**When to disable:**

```yaml
ct_cropping:
  enabled: false  # For single-patient analysis or non-standard regions
```

---

## How to Pass Configuration

There are **four methods** to pass custom configuration to Docker:

### Method 1: Volume Mount a Custom config.yaml (Recommended)

**Create your configuration:**

```yaml
# /path/to/my_custom_config.yaml
dicom_root: "/data/input"
output_dir: "/data/output"
logs_dir: "/data/logs"

max_workers: 15  # Custom worker count

radiomics_robustness:
  enabled: false  # Disable for faster analysis

ct_cropping:
  enabled: true
  region: "thorax"  # Different region
```

**Mount and use:**

```bash
docker run -it --rm --gpus all \
  -v /path/to/dicom:/data/input:ro \
  -v /path/to/output:/data/output:rw \
  -v /path/to/my_custom_config.yaml:/app/config.custom.yaml:ro \
  kstawiski/rtpipeline:latest \
  snakemake --cores all --use-conda --configfile /app/config.custom.yaml
```

---

### Method 2: Override with CLI Parameters

**Override specific values without a config file:**

```bash
docker run -it --rm --gpus all \
  -v /path/to/dicom:/data/input:ro \
  -v /path/to/output:/data/output:rw \
  kstawiski/rtpipeline:latest \
  snakemake --cores all --use-conda \
    --config \
      dicom_root=/data/input \
      output_dir=/data/output \
      max_workers=12 \
      radiomics_robustness.enabled=false
```

---

### Method 3: Environment Variables (Limited)

**Some settings can be overridden via environment:**

```bash
docker run -it --rm --gpus all \
  -e TOTALSEG_TIMEOUT=7200 \
  -e RTPIPELINE_RADIOMICS_TASK_TIMEOUT=1200 \
  -v /path/to/dicom:/data/input:ro \
  -v /path/to/output:/data/output:rw \
  kstawiski/rtpipeline:latest \
  snakemake --cores all --use-conda
```

**Available environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `TOTALSEG_TIMEOUT` | 3600 | TotalSegmentator timeout (seconds) |
| `DCM2NIIX_TIMEOUT` | 300 | DICOM conversion timeout (seconds) |
| `RTPIPELINE_RADIOMICS_TASK_TIMEOUT` | 600 | Per-ROI radiomics timeout (seconds) |

---

### Method 4: docker-compose with Custom Config

**Create docker-compose.custom.yml:**

```yaml
version: '3.8'
services:
  rtpipeline-custom:
    image: kstawiski/rtpipeline:latest
    user: "1000:1000"
    volumes:
      - ./MyProject/DICOM:/data/input:ro
      - ./MyProject/Output:/data/output:rw
      - ./MyProject/Logs:/data/logs:rw
      - ./MyProject/config.yaml:/app/config.custom.yaml:ro
      - ./totalseg_weights:/home/rtpipeline/.totalsegmentator:rw
    environment:
      - TOTALSEG_TIMEOUT=7200
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    shm_size: '8gb'
    command: snakemake --cores 16 --use-conda --configfile /app/config.custom.yaml
```

**Run:**

```bash
docker-compose -f docker-compose.custom.yml run --rm rtpipeline-custom
```

---

## Setup New Analysis Using Docker

### Interactive Setup (Recommended)

**Use the interactive setup tool to generate all necessary files:**

```bash
# One-line installer (easiest)
curl -sSL https://raw.githubusercontent.com/kstawiski/rtpipeline/main/setup_docker_project.sh | bash

# Or if you've cloned the repository
./setup_docker_project.sh
```

**The wizard will:**
1. Ask for your DICOM directory location
2. Configure output directory
3. Select analysis features:
   - Radiomics robustness (disabled/mild/standard/aggressive)
   - CT cropping region (pelvis/thorax/abdomen/head_neck/brain)
   - Custom models (if weights are available)
   - Parallelization settings
4. Generate:
   - `docker-compose.project.yml` (ready to run)
   - `config.yaml` (customized configuration)
   - `run_docker.sh` (convenience script)
   - `README.md` (project documentation)

**Example output:**

```
Your project is ready!

Directory structure:
  /home/user/MyStudy/
    ├── DICOM/               ← Your input files
    ├── Output/              ← Pipeline results (created on run)
    ├── Logs/                ← Pipeline logs (created on run)
    ├── config.yaml          ← Generated configuration
    ├── docker-compose.project.yml
    └── run_docker.sh        ← Execute this!

To start processing:
  cd /home/user/MyStudy
  ./run_docker.sh
```

---

### Manual Setup

**Step-by-step manual setup:**

#### 1. Create directory structure:

```bash
mkdir -p MyProject/{DICOM,Output,Logs}
cd MyProject
```

#### 2. Place your DICOM files:

```bash
# Copy DICOM files to DICOM/ directory
# Organize by patient/course (optional, pipeline auto-organizes):
MyProject/
  └── DICOM/
      ├── Patient001/
      │   ├── CT/
      │   ├── RTDOSE/
      │   ├── RTSTRUCT/
      │   └── RTPLAN/
      └── Patient002/
          └── ...
```

#### 3. Create custom configuration:

```bash
cat > config.yaml <<'EOF'
# MyProject Configuration
dicom_root: "/data/input"
output_dir: "/data/output"
logs_dir: "/data/logs"

# Enable comprehensive analysis
max_workers: null  # Auto-detect

radiomics_robustness:
  enabled: true
  segmentation_perturbation:
    intensity: "standard"
    apply_to_structures:
      - "GTV*"
      - "CTV*"
      - "PTV*"

ct_cropping:
  enabled: true
  region: "pelvis"
  use_cropped_for_dvh: true
  use_cropped_for_radiomics: true
EOF
```

#### 4. Create docker-compose file:

```bash
cat > docker-compose.project.yml <<'EOF'
version: '3.8'
services:
  rtpipeline:
    image: kstawiski/rtpipeline:latest
    user: "1000:1000"
    volumes:
      - ./DICOM:/data/input:ro
      - ./Output:/data/output:rw
      - ./Logs:/data/logs:rw
      - ./config.yaml:/app/config.custom.yaml:ro
      - ../totalseg_weights:/home/rtpipeline/.totalsegmentator:rw
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
        limits:
          cpus: '16.0'
          memory: 32G
    shm_size: '8gb'
    command: snakemake --cores 16 --use-conda --configfile /app/config.custom.yaml
EOF
```

#### 5. Run pipeline:

```bash
docker-compose -f docker-compose.project.yml run --rm rtpipeline
```

---

## Real-World Examples

### Example 1: Quick DVH Analysis (No Radiomics)

**Use case:** Fast DVH extraction for 10 prostate patients

**Configuration:**

```yaml
# config.quick_dvh.yaml
dicom_root: "/data/input"
output_dir: "/data/output"
logs_dir: "/data/logs"

max_workers: 15

segmentation:
  fast: false  # Keep high quality
  device: "gpu"

radiomics:
  enabled: false  # SKIP radiomics for speed

radiomics_robustness:
  enabled: false  # SKIP robustness

ct_cropping:
  enabled: true
  region: "pelvis"
  use_cropped_for_dvh: true
```

**Run:**

```bash
docker run -it --rm --gpus all --shm-size=8g \
  -v ./ProstateStudy:/data/input:ro \
  -v ./ProstateOutput:/data/output:rw \
  -v ./config.quick_dvh.yaml:/app/config.custom.yaml:ro \
  kstawiski/rtpipeline:latest \
  snakemake --cores all --use-conda --configfile /app/config.custom.yaml
```

**Expected time:** ~10-20 minutes for 10 patients (GPU mode)

---

### Example 2: Research-Grade Radiomics Study

**Use case:** Extract robust radiomics features for machine learning

**Configuration:**

```yaml
# config.research_radiomics.yaml
dicom_root: "/data/input"
output_dir: "/data/output"
logs_dir: "/data/logs"

max_workers: 30

radiomics_robustness:
  enabled: true
  segmentation_perturbation:
    intensity: "aggressive"  # 30-60 perturbations per ROI
    apply_to_structures:
      - "GTV*"
      - "CTV*"
      - "PTV*"
      - "prostate"
      - "urinary_bladder"
      - "rectum"
    small_volume_changes: [-0.15, 0.0, 0.15]
    large_volume_changes: [-0.30, 0.0, 0.30]
    max_translation_mm: 5.0
    n_random_contour_realizations: 3
    noise_levels: [0.0, 10.0, 20.0]

ct_cropping:
  enabled: true
  region: "pelvis"
  use_cropped_for_radiomics: true
```

**Run:**

```bash
docker run -it --rm --gpus all --shm-size=8g \
  -v ./RadiomicsStudy:/data/input:ro \
  -v ./RadiomicsOutput:/data/output:rw \
  -v ./Logs:/data/logs:rw \
  -v ./config.research_radiomics.yaml:/app/config.custom.yaml:ro \
  kstawiski/rtpipeline:latest \
  snakemake --cores 32 --use-conda --configfile /app/config.custom.yaml
```

**Expected time:** ~2-6 hours per patient (depends on number of ROIs)

---

### Example 3: Batch Processing Multiple Studies

**Use case:** Process 50 patients overnight

**Directory structure:**

```
BatchProcessing/
├── Study01/
│   ├── DICOM/
│   ├── config.yaml
│   └── docker-compose.yml
├── Study02/
│   ├── DICOM/
│   ├── config.yaml
│   └── docker-compose.yml
├── ...
└── run_all.sh
```

**Batch script (run_all.sh):**

```bash
#!/bin/bash
set -e

STUDIES=(Study01 Study02 Study03 ... Study50)

for study in "${STUDIES[@]}"; do
  echo "Processing $study..."
  cd "$study"
  docker-compose run --rm rtpipeline
  cd ..
  echo "$study complete!"
done

echo "All studies processed!"
```

**Run:**

```bash
chmod +x run_all.sh
nohup ./run_all.sh > batch_processing.log 2>&1 &
```

---

### Example 4: CPU-Only Processing (No GPU)

**Use case:** Running on a server without GPU

**Configuration:**

```yaml
# config.cpu_only.yaml
dicom_root: "/data/input"
output_dir: "/data/output"
logs_dir: "/data/logs"

max_workers: 30  # Use all 32 cores (keep 2 for OS)

segmentation:
  device: "cpu"
  workers: 8  # 8 concurrent CPU-based segmentations
  fast: false

radiomics_robustness:
  enabled: false  # Disable to save time on CPU
```

**Run with docker-compose:**

```bash
docker-compose --profile cpu-only up
```

**Or with docker run:**

```bash
docker run -it --rm \
  --cpus 32 --memory 64g --shm-size=8g \
  -v ./DICOM:/data/input:ro \
  -v ./Output:/data/output:rw \
  -v ./config.cpu_only.yaml:/app/config.custom.yaml:ro \
  kstawiski/rtpipeline:latest \
  snakemake --cores 32 --use-conda --configfile /app/config.custom.yaml
```

**Expected time:** ~3-5x slower than GPU mode for segmentation

---

### Example 5: Custom Anatomical Region (Thorax)

**Use case:** Lung cancer study with thorax cropping

**Configuration:**

```yaml
# config.thorax.yaml
dicom_root: "/data/input"
output_dir: "/data/output"
logs_dir: "/data/logs"

ct_cropping:
  enabled: true
  region: "thorax"  # C7/lung apex → L1/diaphragm
  superior_margin_cm: 2.0
  inferior_margin_cm: 2.0
  use_cropped_for_dvh: true
  use_cropped_for_radiomics: true

radiomics_robustness:
  enabled: true
  segmentation_perturbation:
    intensity: "standard"
    apply_to_structures:
      - "GTV*"
      - "CTV*"
      - "lung*"
      - "heart"
      - "esophagus"
```

---

## Troubleshooting

### Issue: "Permission denied" errors

**Cause:** Container runs as UID 1000, but your files have different ownership

**Solution:**

```bash
# Option 1: Change file ownership
sudo chown -R 1000:1000 Input/ Output/ Logs/

# Option 2: Run container as your UID
docker run --user $(id -u):$(id -g) ...
```

---

### Issue: Pipeline hangs or runs very slowly

**Possible causes:**

1. **Insufficient workers:** Check if max_workers is too low
2. **GPU memory exhausted:** Reduce concurrent segmentations
3. **Slow storage (NFS):** Reduce max_workers to 2-4

**Solutions:**

```yaml
# For slow storage
max_workers: 4

# For GPU memory issues
segmentation:
  workers: 1
  force_split: true

# Increase timeouts
TOTALSEG_TIMEOUT=7200
```

---

### Issue: Out of memory errors

**Cause:** Too many concurrent jobs or large datasets

**Solutions:**

```bash
# Increase shared memory
docker run --shm-size=16g ...

# Reduce parallelism
max_workers: 4

segmentation:
  workers: 1
```

---

### Issue: "Config file not found"

**Cause:** Volume mount path incorrect or config not mounted

**Check mount:**

```bash
docker run --rm kstawiski/rtpipeline:latest ls -la /app/

# Should show config.yaml
```

**Verify your mount syntax:**

```bash
# Correct:
-v $(pwd)/config.yaml:/app/config.custom.yaml:ro

# Incorrect:
-v config.yaml:/app/config.custom.yaml:ro  # ❌ Must be absolute path
```

---

### Issue: Results not appearing in Output directory

**Possible causes:**

1. **Volume mount incorrect**
2. **Permissions issue**
3. **Pipeline failed silently**

**Debug steps:**

```bash
# 1. Check logs
docker logs rtpipeline

# 2. Verify mount
docker run --rm -v ./Output:/data/output kstawiski/rtpipeline:latest ls -la /data/output

# 3. Check Snakemake logs
cat Logs/*.log
```

---

## Advanced Topics

### TotalSegmentator Weights: Baked-In vs. Volume Mount

**Important:** The rtpipeline Docker image includes TotalSegmentator weights **baked into the image** during build (Dockerfile line 120). This means:

✅ **You do NOT need to mount weights** if:
- You're using the pre-built image from Docker Hub
- Your locally built image includes weights in `totalseg_weights/` directory

❌ **You DO need to mount weights** if:
- You want to update model weights without rebuilding the image
- You want to test different model versions
- You're sharing weights across multiple containers
- You built the image without weights in the build context

**Check if weights are in your image:**

```bash
# Check if weights exist in container
docker run --rm kstawiski/rtpipeline:latest ls -la /home/rtpipeline/.totalsegmentator/nnunet/results/

# If you see Dataset291, Dataset292, Dataset293 directories, weights are baked in!
```

**To use baked-in weights** (simpler):

```bash
# No weights mount needed
docker run --gpus all \
  -v ./Input:/data/input:ro \
  -v ./Output:/data/output:rw \
  kstawiski/rtpipeline:latest \
  snakemake --cores all --use-conda
```

**To override/cache weights** (optional):

```bash
# Mount weights directory
docker run --gpus all \
  -v ./Input:/data/input:ro \
  -v ./Output:/data/output:rw \
  -v ./totalseg_weights:/home/rtpipeline/.totalsegmentator:rw \
  kstawiski/rtpipeline:latest \
  snakemake --cores all --use-conda
```

---

### Using Custom nnU-Net Models

**Step 1: Download model weights**

See `custom_models/README.md` for weight download instructions

**Step 2: Configure models**

```yaml
# config.yaml
custom_models:
  enabled: true
  root: "/data/models"  # Mount your models here
  models: ["cardiac_STOPSTORM", "HN_lymph_nodes"]
  workers: null
```

**Step 3: Mount models directory**

```bash
docker run -it --rm --gpus all \
  -v ./custom_models:/data/models:ro \
  -v ./DICOM:/data/input:ro \
  -v ./Output:/data/output:rw \
  -v ./config.yaml:/app/config.custom.yaml:ro \
  kstawiski/rtpipeline:latest \
  snakemake --cores all --use-conda --configfile /app/config.custom.yaml
```

---

### Multi-GPU Support

**For systems with multiple GPUs:**

```yaml
segmentation:
  workers: 2  # Run 2 concurrent segmentations (1 per GPU)
  device: "gpu"
```

**Specify GPUs:**

```bash
docker run --gpus '"device=0,1"' ...
```

---

### HPC/SLURM Integration

See `docs/DOCKER.md` for comprehensive Singularity and SLURM examples

**Quick example:**

```bash
#!/bin/bash
#SBATCH --job-name=rtpipeline
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

singularity exec --nv \
  --bind ./DICOM:/data/input:ro \
  --bind ./Output:/data/output:rw \
  rtpipeline.sif \
  snakemake --cores 16 --use-conda
```

---

### Validation Before Running

**Validate your configuration:**

```bash
docker run --rm \
  -v ./config.yaml:/app/config.custom.yaml:ro \
  kstawiski/rtpipeline:latest \
  python -c "
import yaml
with open('/app/config.custom.yaml') as f:
    config = yaml.safe_load(f)
print('Config valid!')
print('Output dir:', config['output_dir'])
print('Robustness enabled:', config['radiomics_robustness']['enabled'])
"
```

---

## Summary

**Key takeaways:**

1. **Three deployment modes:** Web UI (easiest), docker run (flexible), docker-compose (reproducible)

2. **Four ways to pass config:** Volume mount (best), CLI override, environment variables, docker-compose

3. **Default config enables:** TotalSegmentator, DVH, Radiomics, Robustness, CT Cropping

4. **Interactive setup tool:** `./setup_docker_project.sh` generates everything you need

5. **Real-world examples:** Quick DVH, research radiomics, batch processing, CPU-only, custom regions

**Next steps:**

- For **beginners**: Use `docker-compose up -d` and Web UI
- For **power users**: Use `./setup_docker_project.sh` and generated docker-compose files
- For **HPC users**: See `docs/DOCKER.md` for Singularity guide
- For **troubleshooting**: See `docs/TROUBLESHOOTING.md` and logs in `Logs/`

**Need help?** See:
- General docs: `docs/README.md`
- Web UI guide: `WEBUI.md`
- Output format: `output_format_quick_ref.md`
- Docker technical details: `docs/DOCKER.md`
