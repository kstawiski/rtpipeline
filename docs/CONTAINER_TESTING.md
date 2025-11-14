# Container Testing Guide

This guide provides comprehensive testing procedures for Docker and Singularity containers.

## Quick Testing Checklist

- [ ] Docker image builds successfully
- [ ] All conda environments are created
- [ ] Web UI starts and is accessible
- [ ] Pipeline runs with container config
- [ ] Singularity conversion works
- [ ] All recent features are integrated

## Docker Testing

### 1. Build the Image

```bash
# Clean build (recommended for testing)
./build.sh --no-cache

# Or standard build
./build.sh
```

**Expected output:**
- Build completes without errors
- Final message shows image tags
- Image size should be ~5-8 GB

**Verify build:**
```bash
docker images kstawiski/rtpipeline
```

### 2. Test Container Health

```bash
# Run health check
docker run --rm kstawiski/rtpipeline:latest bash -c "
  conda info &&
  python -c 'import rtpipeline' &&
  echo 'Health check passed'
"
```

**Expected output:**
```
conda version: X.X.X
Health check passed
```

### 3. Test Conda Environments

```bash
# Check all environments exist
docker run --rm kstawiski/rtpipeline:latest bash -c "
  conda env list
"
```

**Expected environments:**
- `base` (with Snakemake)
- `rtpipeline`
- `rtpipeline-radiomics` (with pingouin, pyarrow)
- `rtpipeline-custom-models`

**Verify radiomics environment:**
```bash
docker run --rm kstawiski/rtpipeline:latest bash -c "
  conda run -n rtpipeline-radiomics python -c '
import pyradiomics
import pingouin
import pyarrow
print(\"Radiomics environment OK\")
'
"
```

### 4. Test Container Configuration

```bash
# Check config.container.yaml exists and is valid
docker run --rm kstawiski/rtpipeline:latest bash -c "
  cat /app/config.container.yaml
"
```

**Verify configuration includes:**
- ✅ `dicom_root: /data/input`
- ✅ `output_dir: /data/output`
- ✅ `radiomics_robustness` section
- ✅ `ct_cropping` section
- ✅ All recent features

### 5. Test Web UI

#### 5.1 Start Web UI with docker-compose

```bash
# Start container
docker-compose up -d

# Check logs
docker logs rtpipeline

# Wait for startup (30-60 seconds)
sleep 30
```

**Expected in logs:**
```
Starting rtpipeline Web UI on port 8080
 * Running on http://0.0.0.0:8080
```

#### 5.2 Test Web UI Endpoints

```bash
# Test health endpoint
curl http://localhost:8080/health

# Expected output: {"status":"healthy","timestamp":"...","version":"1.0.0"}

# Test main page
curl -I http://localhost:8080/

# Expected: HTTP/1.1 200 OK
```

#### 5.3 Manual Browser Test

1. Open browser to http://localhost:8080
2. Verify UI loads
3. Test file upload (small DICOM file)
4. Check job creation
5. Verify logs are accessible

#### 5.4 Cleanup

```bash
docker-compose down
```

### 6. Test Pipeline Execution

```bash
# Create test directories
mkdir -p Input Output Logs

# Copy test DICOM data (if available)
# cp -r Example_data/* Input/

# Run pipeline
docker run --rm --gpus all \
  -v $(pwd)/Input:/data/input:ro \
  -v $(pwd)/Output:/data/output:rw \
  -v $(pwd)/Logs:/data/logs:rw \
  kstawiski/rtpipeline:latest \
  bash -c "cd /app && snakemake --cores 4 --use-conda --configfile /app/config.container.yaml organize_courses"
```

**Expected:**
- Snakemake initializes
- Conda environments activate
- Pipeline executes (or reports no input files if none provided)
- No Python import errors
- Logs written to `/data/logs`

### 7. Test CPU Detection

```bash
# Test with CPU limits
docker run --rm --cpus 8 kstawiski/rtpipeline:latest bash -c "
  python -c 'import os, psutil; print(f\"CPU count: {os.cpu_count()}, Psutil cores: {len(psutil.Process().cpu_affinity())}\")'
"
```

**Expected:**
- Reports CPU availability correctly
- psutil respects Docker limits

### 8. Test Environment Variables

```bash
docker run --rm kstawiski/rtpipeline:latest bash -c "
  env | grep -E 'TOTALSEG|DCM2NIIX|RTPIPELINE|CONDA'
"
```

**Expected variables:**
```
TOTALSEG_TIMEOUT=3600
DCM2NIIX_TIMEOUT=300
RTPIPELINE_RADIOMICS_TASK_TIMEOUT=600
CONDA_DIR=/opt/conda
```

### 9. Test GPU Support (if available)

```bash
# Test NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Test GPU in rtpipeline
docker run --rm --gpus all kstawiski/rtpipeline:latest bash -c "
  python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'
"
```

## Singularity Testing

### 1. Build Singularity Image

#### Option A: From Docker Hub

```bash
# Pull pre-built image
singularity pull rtpipeline.sif docker://kstawiski/rtpipeline:latest
```

#### Option B: From Local Docker

```bash
# First ensure Docker image is built
./build.sh

# Convert to Singularity
singularity build rtpipeline.sif docker-daemon://kstawiski/rtpipeline:latest
```

#### Option C: From Definition File

```bash
# Build from rtpipeline.def
# Note: This requires repository files in build context
singularity build --fakeroot rtpipeline.sif rtpipeline.def
```

**Expected:**
- Build completes successfully
- Creates rtpipeline.sif file (~4-7 GB)
- No errors about missing files

### 2. Test Singularity Image

```bash
# Test basic execution
singularity exec rtpipeline.sif python --version

# Test conda
singularity exec rtpipeline.sif conda info

# Test rtpipeline package
singularity exec rtpipeline.sif python -c "import rtpipeline; print('rtpipeline loaded')"
```

### 3. Test Conda Environments in Singularity

```bash
# List environments
singularity exec rtpipeline.sif conda env list

# Test radiomics environment
singularity exec rtpipeline.sif bash -c "
  source /opt/conda/etc/profile.d/conda.sh &&
  conda activate rtpipeline-radiomics &&
  python -c 'import pingouin; import pyarrow; print(\"Radiomics env OK\")'
"
```

### 4. Test Pipeline in Singularity

```bash
# Create test directories
mkdir -p Input Output Logs

# Test pipeline execution
singularity exec \
  --bind $(pwd)/Input:/data/input:ro \
  --bind $(pwd)/Output:/data/output:rw \
  --bind $(pwd)/Logs:/data/logs:rw \
  rtpipeline.sif \
  bash -c "cd /app && snakemake --version"
```

### 5. Test Web UI in Singularity

```bash
# Start web UI
singularity run \
  --bind $(pwd)/Uploads:/data/uploads:rw \
  --bind $(pwd)/Input:/data/input:rw \
  --bind $(pwd)/Output:/data/output:rw \
  --bind $(pwd)/Logs:/data/logs:rw \
  rtpipeline.sif &

# Wait for startup
sleep 30

# Test endpoint
curl http://localhost:8080/health

# Stop
killall -9 python
```

### 6. Test GPU in Singularity (if available)

```bash
# Test with --nv flag
singularity exec --nv rtpipeline.sif bash -c "
  python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'
"
```

### 7. Test HPC Integration

Create a test SLURM script:

```bash
cat > test_slurm.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=rtpipeline-test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00

module load singularity

singularity exec \
  --bind $(pwd)/Input:/data/input:ro \
  --bind $(pwd)/Output:/data/output:rw \
  rtpipeline.sif \
  bash -c "
    echo 'Testing rtpipeline in SLURM'
    conda info
    python -c 'import rtpipeline; print(\"rtpipeline OK\")'
    echo 'Test completed'
  "
EOF

# Submit (if on HPC)
sbatch test_slurm.sh
```

## Integration Testing

### Full Pipeline Test with Sample Data

If you have test DICOM data:

```bash
# 1. Prepare test data
mkdir -p TestData/Input TestData/Output TestData/Logs
cp -r path/to/test/dicom/* TestData/Input/

# 2. Test with Docker
docker run --rm \
  -v $(pwd)/TestData/Input:/data/input:ro \
  -v $(pwd)/TestData/Output:/data/output:rw \
  -v $(pwd)/TestData/Logs:/data/logs:rw \
  kstawiski/rtpipeline:latest \
  bash -c "cd /app && snakemake --cores 4 --use-conda --configfile /app/config.container.yaml"

# 3. Test with Singularity
singularity exec \
  --bind $(pwd)/TestData/Input:/data/input:ro \
  --bind $(pwd)/TestData/Output:/data/output:rw \
  --bind $(pwd)/TestData/Logs:/data/logs:rw \
  rtpipeline.sif \
  bash -c "cd /app && snakemake --cores 4 --use-conda --configfile /app/config.container.yaml"

# 4. Verify outputs
ls -R TestData/Output/
```

**Expected outputs:**
- Course directories created
- NIFTI conversions
- Segmentation results (if data is processable)
- Logs in TestData/Logs/

## Troubleshooting

### Docker Build Fails

**Issue:** Environment creation fails
```bash
# Check specific environment
docker run --rm kstawiski/rtpipeline:latest conda env list
```

**Issue:** Package conflicts
```bash
# Rebuild with no cache
./build.sh --no-cache
```

### Web UI Not Accessible

**Issue:** Port already in use
```bash
# Check what's using port 8080
sudo lsof -i :8080

# Use different port
docker run -p 8081:8080 ...
```

**Issue:** Container not starting
```bash
# Check logs
docker logs rtpipeline

# Interactive debug
docker run -it --rm kstawiski/rtpipeline:latest bash
cd /app/webui
python app.py
```

### Singularity Build Fails

**Issue:** Permission denied
```bash
# Use --fakeroot
singularity build --fakeroot rtpipeline.sif docker://kstawiski/rtpipeline:latest
```

**Issue:** Disk space
```bash
# Check available space
df -h

# Set temporary directory
export SINGULARITY_TMPDIR=/path/to/large/tmp
singularity build ...
```

### Pipeline Execution Fails

**Issue:** Conda environments not found
```bash
# Verify environments inside container
docker run --rm kstawiski/rtpipeline:latest conda env list
```

**Issue:** Import errors
```bash
# Test specific package
docker run --rm kstawiski/rtpipeline:latest \
  conda run -n rtpipeline-radiomics python -c "import pingouin"
```

## Continuous Testing

### Automated Testing Script

Save as `test_containers.sh`:

```bash
#!/bin/bash
set -e

echo "=== rtpipeline Container Testing ==="

echo "1. Testing Docker build..."
./build.sh --no-cache

echo "2. Testing health check..."
docker run --rm kstawiski/rtpipeline:latest bash -c "conda info && python -c 'import rtpipeline'"

echo "3. Testing conda environments..."
docker run --rm kstawiski/rtpipeline:latest bash -c "
  conda env list | grep -q rtpipeline &&
  conda env list | grep -q rtpipeline-radiomics &&
  conda env list | grep -q rtpipeline-custom-models
"

echo "4. Testing configuration..."
docker run --rm kstawiski/rtpipeline:latest bash -c "
  grep -q 'radiomics_robustness' /app/config.container.yaml &&
  grep -q 'ct_cropping' /app/config.container.yaml
"

echo "5. Testing web UI startup..."
docker-compose up -d
sleep 30
curl -f http://localhost:8080/health || exit 1
docker-compose down

echo "6. Testing Singularity build..."
singularity build rtpipeline.sif docker-daemon://kstawiski/rtpipeline:latest

echo "7. Testing Singularity execution..."
singularity exec rtpipeline.sif python -c "import rtpipeline"

echo "=== All tests passed! ==="
```

Run with:
```bash
chmod +x test_containers.sh
./test_containers.sh
```

## Summary

✅ **Complete testing checklist:**

**Docker:**
- [ ] Image builds successfully
- [ ] All conda environments created
- [ ] rtpipeline package imports
- [ ] Web UI starts and responds
- [ ] Configuration includes all features
- [ ] Pipeline can execute

**Singularity:**
- [ ] Image builds from Docker
- [ ] Conda environments accessible
- [ ] Pipeline execution works
- [ ] Web UI can start
- [ ] GPU support (if available)

**Integration:**
- [ ] Full pipeline run with test data
- [ ] All outputs generated correctly
- [ ] No import or dependency errors

For issues, consult:
- [docs/DOCKER.md](DOCKER.md) - Container deployment guide
- [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md) - General troubleshooting
- GitHub Issues: https://github.com/kstawiski/rtpipeline/issues
