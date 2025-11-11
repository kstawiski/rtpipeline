# RT Pipeline - Parallelization Quick Reference Guide

## Command-Line Quick Reference

### 1. Basic Usage (Auto-Optimization)
```bash
# Uses all available CPU cores - 1 for parallelization
rtpipeline --dicom-root /data/dicom --outdir /data/output
```

### 2. Maximum GPU Performance
For systems with high-end GPUs and plenty of RAM:
```bash
rtpipeline \
  --dicom-root /data/dicom \
  --outdir /data/output \
  --seg-workers 4 \
  --workers 8 \
  --totalseg-device gpu \
  --totalseg-force-split
```

**What it does:**
- 4 concurrent TotalSegmentator jobs on GPU
- 8 parallel worker threads for DVH/Visualization/Radiomics
- GPU memory optimized with force-split chunking

### 3. Memory-Constrained System (Single GPU/4-8GB RAM)
```bash
rtpipeline \
  --dicom-root /data/dicom \
  --outdir /data/output \
  --workers 2 \
  --seg-workers 1 \
  --seg-proc-threads 2 \
  --radiomics-proc-threads 1 \
  --sequential-radiomics
```

**Characteristics:**
- 1 TotalSegmentator job at a time (memory safe)
- Sequential radiomics processing (no parallelism overhead)
- Limited thread spawning within processes

### 4. CPU-Only Processing (No GPU)
```bash
rtpipeline \
  --dicom-root /data/dicom \
  --outdir /data/output \
  --totalseg-device cpu \
  --workers 8 \
  --seg-workers 2 \
  --totalseg-fast
```

**Notes:**
- Uses TotalSegmentator's fast mode (less accurate)
- 2 concurrent CPU-based segmentations
- 8 parallel threads for other stages

### 5. Docker Container - GPU Default
```bash
# Start with GPU support (default)
docker-compose up -d

# Access Web UI: http://localhost:8080
```

### 6. Docker Container - CPU Only
```bash
# Start CPU-only version
docker-compose --profile cpu-only up -d

# Limits: 4-16 CPU cores, 8-32GB RAM
```

### 7. Single-Machine Maximum Throughput
For multi-day datasets on high-end systems (32+ cores, multiple GPUs):
```bash
rtpipeline \
  --dicom-root /data/dicom \
  --outdir /data/output \
  --workers 16 \
  --seg-workers 4 \
  --custom-model-workers 2 \
  --radiomics-proc-threads 4 \
  --seg-proc-threads 2
```

### 8. Distributed Processing (Resume Mode)
```bash
# First run: organize and segment
rtpipeline \
  --dicom-root /data/dicom \
  --outdir /data/output \
  --stage organize segmentation

# Later: continue with other stages (automatic resume)
rtpipeline \
  --dicom-root /data/dicom \
  --outdir /data/output \
  --stage dvh radiomics visualize
```

---

## Configuration File (config.yaml) Examples

### High-Performance Setup
```yaml
workers: 16

segmentation:
  workers: 4
  threads_per_worker: 4
  nr_threads_resample: 2
  nr_threads_save: 2
  num_proc_preprocessing: 2
  num_proc_export: 2

radiomics:
  sequential: false
  thread_limit: 4

custom_models:
  workers: 2
```

### Conservative Setup (Low Memory)
```yaml
workers: 2

segmentation:
  workers: 1
  threads_per_worker: 1
  nr_threads_resample: 1
  nr_threads_save: 1
  num_proc_preprocessing: 1
  num_proc_export: 1

radiomics:
  sequential: true
  thread_limit: 1

custom_models:
  workers: 1
```

### Balanced Setup (Typical System)
```yaml
workers: auto  # CPU count - 1

segmentation:
  workers: 2
  threads_per_worker: null  # No limit
  nr_threads_resample: 1
  nr_threads_save: 1
  num_proc_preprocessing: 1
  num_proc_export: 1

radiomics:
  sequential: false
  thread_limit: 4

custom_models:
  workers: 1
```

---

## Environment Variables

### Radiomics Parallelization
```bash
# Enable parallel radiomics (default if no --sequential-radiomics)
export RTPIPELINE_USE_PARALLEL_RADIOMICS=1

# Override worker count (default: CPU count - 1)
export RTPIPELINE_RADIOMICS_WORKERS=8

# Thread limit per worker (prevents OpenMP overload)
export RTPIPELINE_RADIOMICS_THREAD_LIMIT=2

# Force sequential mode (override parallel)
export RTPIPELINE_RADIOMICS_SEQUENTIAL=1
```

### GPU/CUDA Settings
```bash
# GPU selection
export CUDA_VISIBLE_DEVICES=0,1  # Use GPUs 0 and 1
export CUDA_VISIBLE_DEVICES=0    # Use only GPU 0

# TotalSegmentator specific
export TOTALSEG_DEVICE=gpu       # or 'cpu', 'mps'
export TOTALSEG_WEIGHTS_PATH=/path/to/weights

# PyTorch
export TORCH_NUM_THREADS=4
export OMP_NUM_THREADS=2
```

### OpenMP/BLAS Thread Control
```bash
# Single-threaded per process (prevents nested parallelism)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
```

---

## Monitoring & Troubleshooting

### Check System Capacity
```bash
# View CPU information
rtpipeline doctor

# Expected output:
# - Python version
# - Package versions (numpy, PyTorch, TotalSegmentator)
# - CUDA availability and device count
# - nvidia-smi GPU info
# - dcm2niix location
```

### Monitor During Execution
```bash
# Watch log file in real-time
tail -f Logs/rtpipeline.log

# Expected log entries:
# "Segmentation: starting batch with 4 worker(s) for 10 task(s)"
# "Segmentation: 3/10 (30%) elapsed 12.5s ETA 29.2s"
# "memory pressure detected (task #5): CUDA out of memory"
# "retrying 1 task(s) with 2 worker(s) after memory pressure"
```

### Performance Metrics
```bash
# Completion rate: tasks/worker/second
# Typical: 2-5 courses/worker/hour (segmentation)
#          10-50 courses/worker/hour (DVH/Radiomics)

# Memory usage per worker:
# - Segmentation: 2-6 GB (GPU VRAM)
# - Radiomics: 1-3 GB (RAM)
# - DVH: 0.5-1 GB (RAM)
```

---

## Common Performance Issues & Solutions

### Issue 1: CUDA Out of Memory (OOM)
**Symptom:** "CUDA out of memory" or "cublas status alloc failed"

**Automatic Solution:** Pipeline auto-retries with 50% fewer workers

**Manual Solutions:**
```bash
# Option 1: Reduce segmentation workers
rtpipeline --seg-workers 1 ...

# Option 2: Enable force-split for large volumes
rtpipeline --totalseg-force-split ...

# Option 3: Limit threads per worker
rtpipeline --seg-proc-threads 2 ...

# Option 4: Switch to CPU segmentation
rtpipeline --totalseg-device cpu ...
```

### Issue 2: Radiomics Segmentation Faults
**Symptom:** Random crashes during radiomics extraction

**Solution (Automatic):** Uses process-based parallelism (separate processes prevent state sharing)

**Override if needed:**
```bash
# Use sequential processing
rtpipeline --sequential-radiomics ...

# Or reduce workers
export RTPIPELINE_RADIOMICS_WORKERS=1
rtpipeline ...
```

### Issue 3: Slow Processing
**Symptom:** Pipeline not fully utilizing available resources

**Diagnostics:**
```bash
# Check current configuration
rtpipeline doctor

# Monitor during run
watch -n 1 "ps aux | grep python | wc -l"  # Count active processes
nvidia-smi                                   # Check GPU utilization
```

**Solutions:**
```bash
# Increase workers
rtpipeline --workers 16 --seg-workers 4 ...

# Reduce thread limits to prevent contention
rtpipeline --radiomics-proc-threads 2 ...

# Check for network/disk bottlenecks
iostat -x 1        # Disk utilization
nethogs             # Network usage
```

### Issue 4: System Unresponsive (Oversubscribed)
**Symptom:** High load average, system swap usage, UI lag

**Solution:**
```bash
# Emergency: Reduce parallelism
rtpipeline --workers 2 --seg-workers 1 ...

# Prevent nested parallelism
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
rtpipeline ...
```

---

## Performance Tuning Checklist

- [ ] Verify GPU availability: `rtpipeline doctor`
- [ ] Check system resources: `free -h`, `lscpu`
- [ ] Start with conservative settings if unsure
- [ ] Monitor first run: `tail -f Logs/rtpipeline.log`
- [ ] Increase workers if underutilized
- [ ] Decrease workers if memory pressure detected
- [ ] Set thread limits only if needed (default: no limit)
- [ ] Use GPU for segmentation (fastest)
- [ ] Enable parallel radiomics if stable
- [ ] Test with small dataset first

---

## Key Parallelization Parameters Reference

| Parameter | CLI Arg | Default | Purpose |
|-----------|---------|---------|---------|
| Overall Workers | `--workers N` | CPU-1 | Parallelism for DVH/Viz/Radiomics |
| Segmentation Workers | `--seg-workers N` | 1 | TotalSegmentator concurrency |
| Segmentation Threads | `--seg-proc-threads N` | None | Threads per TotalSegmentator |
| Radiomics Threads | `--radiomics-proc-threads N` | None | Threads per radiomics worker |
| Custom Model Workers | `--custom-model-workers N` | 1 | Concurrent custom model courses |
| GPU Device | `--totalseg-device` | gpu | gpu/cpu/mps |
| Force Split | `--totalseg-force-split` | True | Chunk volumes for GPU memory |
| Sequential Radiomics | `--sequential-radiomics` | False | Disable radiomics parallelism |

---

## Real-World Examples

### Example 1: Desktop with 8-core CPU, RTX 3060 (12GB)
```bash
rtpipeline \
  --dicom-root ~/data \
  --outdir ~/results \
  --workers 4 \
  --seg-workers 2 \
  --radiomics-proc-threads 2 \
  --seg-proc-threads 1
```

### Example 2: Server with 64-core CPU, 128GB RAM, no GPU
```bash
rtpipeline \
  --dicom-root /data/dicom \
  --outdir /data/results \
  --workers 32 \
  --seg-workers 4 \
  --totalseg-device cpu \
  --totalseg-fast \
  --seg-proc-threads 4
```

### Example 3: Medical imaging cluster with 4x GPU nodes
```bash
# Node 1:
rtpipeline --dicom-root /data --outdir /results/node1 --stage organize

# Nodes 2-4 (parallel):
rtpipeline --dicom-root /data --outdir /results/node2 --stage segmentation --seg-workers 4 &
rtpipeline --dicom-root /data --outdir /results/node3 --stage dvh radiomics --workers 16 &
rtpipeline --dicom-root /data --outdir /results/node4 --stage dvh radiomics --workers 16 &
```

---

## See Also
- `PIPELINE_ARCHITECTURE.md` - Complete technical reference
- `README.md` - General pipeline documentation
- `Logs/rtpipeline.log` - Detailed execution logs

