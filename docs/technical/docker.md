# Docker and Singularity Guide for rtpipeline

## Overview
This document provides comprehensive guidance for running rtpipeline in Docker and Singularity containers, including compatibility information for optimization features, hang prevention, web UI support, and the latest radiomics robustness module.

## ✅ Container Features

All pipeline features work correctly in Docker and Singularity containers:

### Latest Updates (2024)
- **Radiomics Robustness Module**: Comprehensive feature stability assessment with ICC, CoV, and QCD metrics
- **Web UI**: Browser-based interface for drag-and-drop DICOM upload and processing
- **Enhanced Configuration**: Container-specific config includes all recent pipeline features

### Core Features

### 1. **Subprocess Timeouts** ✅
- `subprocess.run()` with timeout parameter works in Docker
- SIGTERM/SIGKILL signals properly handled with tini init system
- Tested with TotalSegmentator and dcm2niix operations

### 2. **Process Spawning** ✅
- multiprocessing with `spawn` context works in containers
- No fork-related issues (spawn is safer than fork in Docker)
- Process pool handling is container-safe

### 3. **CPU Detection** ✅
- `os.cpu_count()` correctly detects available CPUs
- Works with Docker `--cpus` limits (when using psutil)
- Respects cgroup CPU quotas in Docker/Kubernetes

### 4. **Signal Handling** ✅
- Tini init system properly forwards signals
- Graceful shutdown on SIGTERM from `docker stop`
- Prevents zombie processes from multiprocessing

### 5. **GPU Support** ✅
- CUDA_VISIBLE_DEVICES and NVIDIA_VISIBLE_DEVICES work
- Timeout mechanisms don't interfere with GPU operations
- GPU memory management works as expected

## Docker-Specific Enhancements

### Tini Init System
**Added to Dockerfile:**
```dockerfile
RUN apt-get install -y tini
ENTRYPOINT ["/usr/bin/tini", "--"]
```

**Benefits:**
- Properly reaps zombie processes from multiprocessing
- Forwards signals (SIGTERM, SIGINT) to child processes
- Essential for parallel processing in containers

### Psutil for Better CPU Detection
**Added to Dockerfile:**
```dockerfile
RUN pip install psutil
```

**Benefits:**
- Respects Docker CPU limits (`--cpus`)
- Detects actual available CPUs vs. host CPUs
- Works with Kubernetes CPU requests/limits

### Timeout Environment Variables
**Added to Dockerfile:**
```dockerfile
ENV TOTALSEG_TIMEOUT=3600 \
    DCM2NIIX_TIMEOUT=300 \
    RTPIPELINE_RADIOMICS_TASK_TIMEOUT=600
```

**Benefits:**
- Set reasonable defaults for containers
- Can be overridden in docker-compose.yml or at runtime
- Prevents indefinite hangs in containerized environments

## Usage Examples

### Running with Docker

#### Basic Usage (GPU)
```bash
docker run --gpus all \
  -v ./Input:/data/input:ro \
  -v ./Output:/data/output:rw \
  -v ./Logs:/data/logs:rw \
  kstawiski/rtpipeline:latest \
  rtpipeline \
    --dicom-root /data/input \
    --outdir /data/output \
    --logs /data/logs
```

#### Custom Timeouts
```bash
docker run --gpus all \
  -e TOTALSEG_TIMEOUT=7200 \
  -e DCM2NIIX_TIMEOUT=600 \
  -e RTPIPELINE_RADIOMICS_TASK_TIMEOUT=1200 \
  -v ./Input:/data/input:ro \
  -v ./Output:/data/output:rw \
  kstawiski/rtpipeline:latest \
  rtpipeline --dicom-root /data/input --outdir /data/output
```

#### CPU-Limited Container
```bash
docker run \
  --cpus 8 \
  --memory 16g \
  -e TOTALSEG_TIMEOUT=7200 \
  -v ./Input:/data/input:ro \
  -v ./Output:/data/output:rw \
  kstawiski/rtpipeline:latest \
  rtpipeline \
    --dicom-root /data/input \
    --outdir /data/output \
    --totalseg-device cpu \
    --max-workers 7
```

### Running with Docker Compose

#### GPU Mode (default)
```bash
docker-compose up rtpipeline
```

#### CPU-Only Mode
```bash
docker-compose --profile cpu-only up rtpipeline-cpu
```

#### Custom Configuration
```yaml
# docker-compose.override.yml
version: '3.8'
services:
  rtpipeline:
    environment:
      - TOTALSEG_TIMEOUT=7200  # 2 hours
      - RTPIPELINE_RADIOMICS_TASK_TIMEOUT=1800  # 30 min per ROI
    deploy:
      resources:
        limits:
          cpus: '12.0'
          memory: 48G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Testing Docker Compatibility

We provide a comprehensive test script to verify all features work in your container:

```bash
# Run inside container
docker run --rm -it kstawiski/rtpipeline:latest python docker_test.py

# Or with docker-compose
docker-compose run --rm rtpipeline python docker_test.py
```

**Tests performed:**
- ✅ CPU detection (os.cpu_count(), cgroup limits, psutil)
- ✅ Subprocess timeouts (subprocess.run with timeout)
- ✅ Signal handling (SIGTERM, SIGINT)
- ✅ Multiprocessing (spawn context with Pool)
- ✅ Environment variables (timeout config)
- ✅ GPU detection (CUDA availability)

## CPU Detection in Containers

### How It Works

**Without Docker limits:**
```python
os.cpu_count()  # Returns host CPU count (e.g., 16)
effective_workers = cpu_count - 1  # 15 workers
```

**With Docker `--cpus 8` limit:**
```python
os.cpu_count()  # Still returns 16 (host count)
# But psutil respects cgroup limits:
psutil.Process().cpu_affinity()  # [0,1,2,3,4,5,6,7] (8 CPUs)
```

**Recommendation:**
When using Docker CPU limits, manually set workers:
```bash
docker run --cpus 8 ... rtpipeline --max-workers 7 ...
```

Or use environment variable:
```bash
docker run --cpus 8 -e RTPIPELINE_WORKERS=7 ...
```

### Cgroup CPU Limits

The pipeline respects cgroup CPU quotas:
- **cgroups v1**: `/sys/fs/cgroup/cpu/cpu.cfs_quota_us`
- **cgroups v2**: `/sys/fs/cgroup/cpu.max`

To see your container's CPU limit:
```bash
# Inside container
cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us
cat /sys/fs/cgroup/cpu/cpu.cfs_period_us
# Actual CPUs = quota / period
```

## Signal Handling and Graceful Shutdown

### With Tini (Recommended)
```bash
# Graceful stop (10 second timeout)
docker stop rtpipeline

# Tini forwards SIGTERM to pipeline
# Pipeline finishes current task and exits cleanly
```

### Without Tini (Not Recommended)
```bash
# Without tini, signals may not propagate properly
# Can leave zombie processes
# May need docker kill -9 (forceful)
```

**Always use tini** - it's included in our Dockerfile!

## Kubernetes Compatibility

The pipeline works in Kubernetes with proper resource limits:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: rtpipeline
spec:
  containers:
  - name: rtpipeline
    image: kstawiski/rtpipeline:latest
    command: ["rtpipeline"]
    args:
      - "--dicom-root"
      - "/data/input"
      - "--outdir"
      - "/data/output"
      - "--max-workers"
      - "7"  # Set based on CPU limits
    resources:
      requests:
        cpu: "4"
        memory: "16Gi"
      limits:
        cpu: "8"
        memory: "32Gi"
        nvidia.com/gpu: "1"
    env:
    - name: TOTALSEG_TIMEOUT
      value: "3600"
    - name: CUDA_VISIBLE_DEVICES
      value: "0"
```

## Common Issues and Solutions

### Issue: Container uses all host CPUs despite `--cpus` limit

**Solution:**
Manually set `--max-workers` to respect the limit:
```bash
docker run --cpus 8 ... rtpipeline --max-workers 7 ...
```

### Issue: Pipeline doesn't shut down gracefully

**Solution:**
Ensure tini is being used:
```bash
# Check ENTRYPOINT
docker inspect kstawiski/rtpipeline:latest | grep -A1 Entrypoint

# Should show: ["/usr/bin/tini", "--"]
```

### Issue: Timeouts not working in container

**Solution:**
Check environment variables are set:
```bash
docker run --rm kstawiski/rtpipeline:latest env | grep TIMEOUT

# Should show:
# TOTALSEG_TIMEOUT=3600
# DCM2NIIX_TIMEOUT=300
# RTPIPELINE_RADIOMICS_TASK_TIMEOUT=600
```

### Issue: GPU not detected in container

**Solution:**
Check NVIDIA runtime and environment variables:
```bash
# Verify GPU runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check environment in our container
docker run --rm --gpus all kstawiski/rtpipeline:latest \
  bash -c "nvidia-smi && python -c 'import torch; print(torch.cuda.is_available())'"
```

### Issue: Out of shared memory errors

**Solution:**
Increase shared memory size:
```bash
docker run --shm-size=4g ...
# or in docker-compose.yml:
shm_size: '4gb'
```

## Performance Tuning for Docker

### Memory-Constrained Containers
```bash
docker run \
  --memory 16g \
  --memory-swap 16g \
  kstawiski/rtpipeline:latest \
  rtpipeline \
    --max-workers 4 \
    --seg-workers 1 \
    --sequential-radiomics
```

### CPU-Heavy Workload
```bash
docker run \
  --cpus 32 \
  kstawiski/rtpipeline:latest \
  rtpipeline \
    --max-workers 31 \
    --totalseg-device cpu \
    --seg-workers 2
```

### GPU-Optimized
```bash
docker run \
  --gpus all \
  --shm-size=8g \
  kstawiski/rtpipeline:latest \
  rtpipeline \
    --max-workers 15 \
    --seg-workers 1 \
    --totalseg-device gpu \
    --totalseg-force-split
```

## Monitoring

### Watch Pipeline Progress
```bash
# Stream logs
docker logs -f rtpipeline

# Watch for heartbeat messages
docker logs -f rtpipeline 2>&1 | grep "Still processing"

# Monitor resource usage
docker stats rtpipeline
```

### Check for Timeouts
```bash
# Look for timeout errors
docker logs rtpipeline 2>&1 | grep -i timeout

# Should see entries like:
# "ERROR: Command timed out after 3600s"
# "ERROR: Segmentation: task #5 timed out after 7200s"
```

## Best Practices

1. **Always use tini** - included in our Dockerfile ✅
2. **Set explicit workers** when using `--cpus` limit
3. **Increase timeouts** for large datasets or slow systems
4. **Monitor logs** for heartbeat messages and timeouts
5. **Use `--shm-size`** for GPU workloads (4-8GB)
6. **Cache TotalSegmentator weights** with volume mount
7. **Use restart policies** for production:
   ```yaml
   restart: unless-stopped
   ```

## Singularity Support

rtpipeline fully supports Singularity for HPC and secure computing environments.

### Building Singularity Containers

#### Option 1: From Docker Hub (Recommended)
```bash
singularity pull rtpipeline.sif docker://kstawiski/rtpipeline:latest
```

#### Option 2: From Local Docker Image
```bash
# First build Docker image
./build.sh

# Convert to Singularity
singularity build rtpipeline.sif docker-daemon://kstawiski/rtpipeline:latest
```

#### Option 3: From Definition File (Advanced)
```bash
# Build from rtpipeline.def
# Note: Requires repository files in build context
singularity build --fakeroot rtpipeline.sif rtpipeline.def
```

### Running with Singularity

#### Interactive Shell
```bash
singularity shell --nv \
  --bind /path/to/input:/data/input:ro \
  --bind /path/to/output:/data/output:rw \
  --bind /path/to/logs:/data/logs:rw \
  rtpipeline.sif
```

#### Execute Pipeline
```bash
singularity exec --nv \
  --bind /path/to/input:/data/input:ro \
  --bind /path/to/output:/data/output:rw \
  --bind /path/to/logs:/data/logs:rw \
  rtpipeline.sif \
  snakemake --cores all --use-conda --configfile /app/config.container.yaml
```

#### Web UI Mode
```bash
singularity run --nv \
  --bind /path/to/uploads:/data/uploads:rw \
  --bind /path/to/input:/data/input:rw \
  --bind /path/to/output:/data/output:rw \
  --bind /path/to/logs:/data/logs:rw \
  rtpipeline.sif
# Access at http://localhost:8080
```

### HPC/SLURM Integration

**Example Job Script:**
```bash
#!/bin/bash
#SBATCH --job-name=rtpipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

module load singularity

# Set paths
INPUT_DIR=/scratch/$USER/dicom_input
OUTPUT_DIR=/scratch/$USER/rtpipeline_output
LOGS_DIR=/scratch/$USER/rtpipeline_logs

# Create output directories
mkdir -p $OUTPUT_DIR $LOGS_DIR

# Run pipeline
singularity exec --nv \
  --bind ${INPUT_DIR}:/data/input:ro \
  --bind ${OUTPUT_DIR}:/data/output:rw \
  --bind ${LOGS_DIR}:/data/logs:rw \
  rtpipeline.sif \
  snakemake --cores $SLURM_CPUS_PER_TASK --use-conda --configfile /app/config.container.yaml

echo "Pipeline completed at $(date)"
```

**With Custom Configuration:**
```bash
#!/bin/bash
#SBATCH --job-name=rtpipeline-robustness
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=72:00:00

module load singularity

# Custom config with radiomics robustness enabled
cat > /tmp/config.custom.yaml << 'EOF'
dicom_root: "/data/input"
output_dir: "/data/output"
logs_dir: "/data/logs"
workers: 30

radiomics_robustness:
  enabled: true
  modes:
    - segmentation_perturbation
  segmentation_perturbation:
    apply_to_structures:
      - "GTV*"
      - "CTV*"
      - "PTV*"
    intensity: "aggressive"
EOF

singularity exec \
  --bind /tmp/config.custom.yaml:/tmp/config.custom.yaml:ro \
  --bind ${INPUT_DIR}:/data/input:ro \
  --bind ${OUTPUT_DIR}:/data/output:rw \
  rtpipeline.sif \
  snakemake --cores 30 --use-conda --configfile /tmp/config.custom.yaml
```

### Singularity-Specific Notes

1. **GPU Access**: Use `--nv` flag for NVIDIA GPU support
2. **Writable Overlays**: For caching TotalSegmentator weights:
   ```bash
   # Create overlay for persistent weights
   singularity overlay create --size 10240 totalseg_cache.img

   # Use overlay
   singularity exec --nv \
     --overlay totalseg_cache.img \
     --bind /data:/data \
     rtpipeline.sif \
     snakemake --cores all --use-conda --configfile /app/config.container.yaml
   ```

3. **Environment Variables**: Pass via `--env` or `SINGULARITYENV_`:
   ```bash
   export SINGULARITYENV_TOTALSEG_TIMEOUT=7200
   export SINGULARITYENV_RTPIPELINE_RADIOMICS_TASK_TIMEOUT=1200
   singularity exec --nv rtpipeline.sif ...
   ```

4. **Conda Environments**: All environments are pre-built in the container
5. **Web UI**: Requires binding appropriate ports and directories

## Summary

✅ **All optimization and hang prevention features work correctly in Docker and Singularity**

The pipeline has been enhanced with:
- **Radiomics Robustness Module**: ICC-based feature stability assessment (latest update)
- **Web UI**: Browser-based upload and processing interface
- **Tini init system** for proper signal handling
- **Timeout environment variables** with sensible defaults
- **Psutil** for accurate CPU detection
- **Singularity support** for HPC environments

**Container-specific features:**
- Respects container CPU/memory limits
- Proper signal forwarding for graceful shutdown
- Zombie process reaping for parallel operations
- GPU support (NVIDIA Docker runtime / Singularity --nv)
- Pre-built conda environments
- Container-optimized configuration

**Recommended deployment:**

**Docker (Development/Production):**
```bash
# Web UI mode (recommended)
docker-compose up -d

# Access at http://localhost:8080
```

**Singularity (HPC/Secure Environments):**
```bash
# Pull from Docker Hub
singularity pull rtpipeline.sif docker://kstawiski/rtpipeline:latest

# Run pipeline
singularity exec --nv \
  --bind /data:/data \
  rtpipeline.sif \
  snakemake --cores all --use-conda --configfile /app/config.container.yaml
```

For production, see docker-compose.yml configuration with restart policies, resource limits, and health checks.
