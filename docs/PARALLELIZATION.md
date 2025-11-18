# RT Pipeline - Parallelization Guide (SIMPLIFIED)

## Overview

**NEW SIMPLIFIED APPROACH (2024):**
- **All modules** use `(cpu_count - 1)` workers by default
- **Exception:** Segmentation uses 1 worker (GPU-safe, sequential)
- **One setting controls everything:** `workers: auto` in config.yaml

This eliminates the previous confusion between cores, workers, and threads.

---

## Quick Reference

### Default Behavior (Recommended)
```bash
# Uses (cpu_count - 1) for all modules automatically
rtpipeline --dicom-root /data/dicom --outdir /data/output
```

On an 8-core system:
- **7 workers** for organization, DVH, radiomics, QC, aggregation
- **1 worker** for segmentation (GPU-safe, sequential)

### Override for Specific Systems
```bash
# Manually set to 4 workers if needed
rtpipeline --dicom-root /data/dicom --outdir /data/output --workers 4
```

---

## Configuration Examples

### Standard Setup (Recommended)
```yaml
# config.yaml
workers: auto  # Uses (cpu_count - 1) for all modules

segmentation:
  workers: 1  # Always 1 for GPU safety (do not increase)
  nr_threads_resample: 1
  nr_threads_save: 1
  num_proc_preprocessing: 1
  num_proc_export: 1

radiomics:
  sequential: false  # Use parallel processing
  thread_limit: null  # No artificial limit

custom_models:
  workers: 1  # GPU-safe, sequential

aggregation:
  threads: auto  # Uses (cpu_count - 1)
```

### Memory-Constrained System
```yaml
workers: 2  # Manually limit to 2 workers

segmentation:
  workers: 1

radiomics:
  sequential: true  # Disable parallelism if memory issues
  thread_limit: 1

custom_models:
  workers: 1
```

---

## How Parallelization Works

### Module-by-Module Breakdown

| Module | Workers | Type | Notes |
|--------|---------|------|-------|
| **Organization** | cpu_count - 1 | ThreadPoolExecutor | DICOM file I/O operations |
| **Segmentation** | 1 (fixed) | GPU subprocess | Sequential for GPU safety |
| **DVH** | cpu_count - 1 | ThreadPoolExecutor | Per-ROI DVH calculation |
| **Radiomics** | cpu_count - 1 | ProcessPoolExecutor | CPU isolation via processes |
| **QC** | cpu_count - 1 | Per-course | Quality control validation |
| **Aggregation** | cpu_count - 1 | ThreadPoolExecutor | Results compilation |

### Why CPU - 1?

**Best practices from Python concurrent.futures:**
- **ThreadPoolExecutor** (I/O-bound): Can handle `cpu_count + 4` workers
- **ProcessPoolExecutor** (CPU-bound): Should use `cpu_count` or `cpu_count - 1`
- **Leave 1 core free** for system operations and responsiveness

**Our approach:**
- Standardize on `cpu_count - 1` for simplicity
- Works well for both I/O-bound (organization, DVH) and CPU-bound (radiomics) tasks
- Prevents system overload

### Why Segmentation is Sequential (1 Worker)

**GPU memory constraints:**
- TotalSegmentator loads entire models into GPU memory (2-8 GB VRAM)
- Running multiple instances risks GPU OOM errors
- Sequential execution ensures stable, predictable performance

**Performance note:**
- Segmentation is the slowest stage anyway (5-30 seconds per course)
- Parallelizing other stages provides better overall throughput

---

## Environment Variables

### Radiomics Parallelization Override
```bash
# Override worker count (default: cpu_count - 1)
export RTPIPELINE_RADIOMICS_WORKERS=4

# Thread limit per worker (prevents OpenMP overload)
export RTPIPELINE_RADIOMICS_THREAD_LIMIT=2

# Force sequential mode
export RTPIPELINE_RADIOMICS_SEQUENTIAL=1
```

### GPU/CUDA Settings
```bash
# GPU selection
export CUDA_VISIBLE_DEVICES=0    # Use only GPU 0

# TotalSegmentator
export TOTALSEG_DEVICE=gpu       # or 'cpu', 'mps'
```

### OpenMP/BLAS Thread Control
```bash
# Prevent nested parallelism (recommended for radiomics)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

---

## Snakemake Workflow Parallelization

**How it works:**
```bash
# Run with all available cores
snakemake --cores all --use-conda

# Run with specific core count
snakemake --cores 8 --use-conda
```

**Snakemake schedules:**
- Multiple courses run in parallel (up to `--cores` limit)
- Each course uses the configured number of workers internally
- Segmentation rules reserve 1 worker (sequential GPU safety)
- Other rules can run in parallel across courses

**Example on 8-core system with `workers: auto`:**
```
Course 1: Organization (7 workers) → Segmentation (1 worker) → DVH (7 workers)
Course 2:                           → Segmentation (1 worker) → DVH (7 workers)
Course 3:                                                      → DVH (7 workers)
```

---

## Monitoring & Diagnostics

### Check Configuration
```bash
# View current settings
cat config.yaml | grep -A5 "workers:"

# Expected output:
# workers: auto  # Uses (cpu_count - 1) for all modules
```

### Monitor During Execution
```bash
# Watch log file
tail -f Logs_Snakemake/*/segmentation*.log

# Check CPU utilization
htop  # Should see (cpu_count - 1) cores active during DVH/radiomics

# Check GPU utilization
nvidia-smi -l 1  # Should see 1 TotalSegmentator process at a time
```

---

## Common Issues & Solutions

### Issue 1: CUDA Out of Memory (OOM)
**Symptom:** "CUDA out of memory" during segmentation

**Solution:**
```bash
# Segmentation already uses 1 worker (sequential)
# If still occurring, reduce batch size:
# Edit config.yaml:
segmentation:
  force_split: true  # Enable chunked inference (default)
```

### Issue 2: System Unresponsive (Too Many Workers)
**Symptom:** High load average, system lag

**Solution:**
```bash
# Reduce workers manually
rtpipeline --workers 2 ...

# Or in config.yaml:
workers: 2  # Instead of auto
```

### Issue 3: Underutilized CPU
**Symptom:** CPU cores idle during processing

**Solution:**
```bash
# Check if workers: auto is set
grep "workers:" config.yaml

# Verify Snakemake is using all cores
snakemake --cores all  # Not --cores 1
```

### Issue 4: Radiomics Crashes
**Symptom:** Segmentation faults during radiomics

**Solution:**
```bash
# Already uses ProcessPoolExecutor (process isolation)
# If still crashing, try sequential mode:
rtpipeline --sequential-radiomics ...
```

---

## Real-World Examples

### Desktop: 8-core CPU, RTX 3060 (12GB VRAM)
```yaml
# config.yaml
workers: auto  # Uses 7 workers

segmentation:
  workers: 1
  device: "gpu"

radiomics:
  sequential: false
```

**Expected performance:**
- Organization: ~50 courses/hour (7 workers)
- Segmentation: ~10 courses/hour (1 worker, GPU)
- DVH: ~100 courses/hour (7 workers)
- Radiomics: ~20 courses/hour (7 workers, process-based)

### Server: 32-core CPU, 128GB RAM, No GPU
```yaml
workers: auto  # Uses 31 workers

segmentation:
  workers: 1
  device: "cpu"
  fast: true  # CPU mode benefits from fast mode

radiomics:
  sequential: false
```

**Expected performance:**
- Segmentation: ~5 courses/hour (CPU mode is slower)
- DVH/Radiomics: Much faster due to 31 workers

### Cluster: 64-core CPU, 256GB RAM, 4x GPUs
```yaml
workers: auto  # Uses 63 workers

segmentation:
  workers: 1  # Still 1 - sequential for GPU safety

radiomics:
  sequential: false
  thread_limit: 2  # Prevent thread oversubscription
```

**Notes:**
- Even with 4 GPUs, keep segmentation workers at 1
- TotalSegmentator uses CUDA device 0 by default
- For multi-GPU, use CUDA_VISIBLE_DEVICES environment variable

---

## Migration from Old Configuration

### Old Approach (Pre-2024)
```yaml
workers: auto  # Was cpu_count - 2 (min 4, max 32)

segmentation:
  workers: 1
  threads_per_worker: null  # Not used anywhere

radiomics:
  thread_limit: 4  # Complex nested threading
```

### New Simplified Approach
```yaml
workers: auto  # Now cpu_count - 1 (simpler, better performance)

segmentation:
  workers: 1  # Unchanged, still sequential

radiomics:
  thread_limit: null  # Simplified, no artificial limits
```

**Changes:**
- Snakefile: Now uses `cpu_count - 1` instead of `cpu_count - 2`
- DVH: Now uses `cpu_count - 1` instead of `cpu_count / 2`
- Removed unused `threads_per_worker` parameter
- Simplified configuration comments
- One rule for everything: `cpu_count - 1`

**Impact:**
- **4-core system**: 3 workers (was 2) → 50% improvement
- **8-core system**: 7 workers (was 6 or 4) → 17-75% improvement
- **16-core system**: 15 workers (was 14 or 8) → 7-88% improvement
- **32-core system**: 31 workers (was 30 or 16) → 3-94% improvement

---

## Performance Tuning Checklist

- [ ] Use `workers: auto` in config.yaml (recommended)
- [ ] Verify GPU availability: `nvidia-smi`
- [ ] Keep segmentation workers at 1 (GPU safety)
- [ ] Monitor first run: `tail -f Logs_Snakemake/*/segmentation*.log`
- [ ] Check CPU utilization: `htop` (should see cpu_count - 1 cores active)
- [ ] Only override if you have memory constraints
- [ ] Don't set thread limits unless experiencing thread oversubscription

---

## Key Parameters Reference

| Parameter | Config Key | Default | Purpose |
|-----------|------------|---------|---------|
| Overall Workers | `workers` | cpu_count - 1 | All modules except segmentation |
| Segmentation Workers | `segmentation.workers` | 1 | Always sequential (GPU-safe) |
| Radiomics Mode | `radiomics.sequential` | false | Parallel by default |
| Aggregation Threads | `aggregation.threads` | cpu_count - 1 | Results compilation |

**Removed Parameters:**
- ~~`segmentation.threads_per_worker`~~ - Not used, removed
- ~~`--seg-proc-threads`~~ - TotalSegmentator internal threading is fixed

---

## Summary

**SIMPLIFIED RULE:**
1. Set `workers: auto` in config.yaml
2. This uses `cpu_count - 1` for all modules
3. Segmentation is always sequential (1 worker, GPU-safe)
4. No need to configure cores/workers/threads separately

**That's it!** The pipeline handles everything else automatically.

---

## See Also
- [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md) - Technical deep-dive
- [README.md](../README.md) - General pipeline documentation
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions
