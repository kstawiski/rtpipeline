# RTpipeline Parallelization Parameters - Complete Reference

## Summary: One Simple Rule

**USE `cpu_count - 1` FOR EVERYTHING**
(except GPU segmentation which uses 1 worker)

---

## Configuration Parameters

### config.yaml

```yaml
# Main parallelization control
workers: auto  # cpu_count - 1 for all modules

# GPU-safe segmentation (always sequential)
segmentation:
  workers: 1  # Always 1 (GPU safety)
  thread_limit: null  # Optional: OMP_NUM_THREADS for TotalSegmentator

# Radiomics
radiomics:
  sequential: false  # Use parallel processing
  thread_limit: null  # Optional: OMP_NUM_THREADS per worker

# Custom models (GPU-safe)
custom_models:
  workers: 1  # Always 1 (GPU safety)

# Aggregation
aggregation:
  threads: auto  # cpu_count - 1
```

---

## Command-Line Arguments

### Main Workers
```bash
--workers N              # Override auto (cpu_count-1)
                         # Controls: organization, DVH, radiomics, QC, aggregation
```

### Segmentation
```bash
--seg-workers N          # Always use 1 (GPU safety)
--seg-proc-threads N     # OpenMP thread limit for TotalSegmentator
                         # Sets OMP_NUM_THREADS inside TotalSegmentator
                         # Default: no limit
```

### Radiomics
```bash
--radiomics-proc-threads N   # OpenMP thread limit per radiomics worker
                             # Sets OMP_NUM_THREADS for each worker process
                             # Default: no limit
```

### Custom Models
```bash
--custom-model-workers N     # Always use 1 (GPU safety)
```

---

## Environment Variables

### Worker Control (Process/Thread Count)
```bash
# Override auto-detected worker count for radiomics
export RTPIPELINE_RADIOMICS_WORKERS=4

# Force sequential radiomics (disable parallelization)
export RTPIPELINE_RADIOMICS_SEQUENTIAL=1
```

### Thread Limits (OpenMP/BLAS)
```bash
# Limit threads per radiomics worker
export RTPIPELINE_RADIOMICS_THREAD_LIMIT=2

# General OpenMP/BLAS control (affects all modules)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
```

### GPU Control
```bash
# GPU selection
export CUDA_VISIBLE_DEVICES=0  # Use only GPU 0

# TotalSegmentator device
export TOTALSEG_DEVICE=gpu  # or 'cpu', 'mps'
```

---

## Complete Parameter Mapping

| Config File | Snakefile Variable | CLI Argument | Environment Variable | PipelineConfig Attribute | Used By Module |
|-------------|-------------------|--------------|---------------------|-------------------------|----------------|
| `workers: auto` | `WORKERS` | `--workers` | - | `workers` | All modules (via `effective_workers()`) |
| `segmentation.workers` | `SEG_MAX_WORKERS` | `--seg-workers` | - | `segmentation_workers` | Segmentation (Snakemake resource) |
| `segmentation.thread_limit` | `SEG_THREADS_PER_WORKER` | `--seg-proc-threads` | - | `segmentation_thread_limit` | TotalSegmentator (OMP_NUM_THREADS) |
| `radiomics.sequential` | `RADIOMICS_SEQUENTIAL` | - | `RTPIPELINE_RADIOMICS_SEQUENTIAL` | - | Radiomics (mode selection) |
| `radiomics.thread_limit` | `RADIOMICS_THREAD_LIMIT` | `--radiomics-proc-threads` | `RTPIPELINE_RADIOMICS_THREAD_LIMIT` | `radiomics_thread_limit` | Radiomics (OMP_NUM_THREADS) |
| - | - | - | `RTPIPELINE_RADIOMICS_WORKERS` | - | Radiomics (worker override) |
| `custom_models.workers` | `CUSTOM_MODELS_WORKERS` | `--custom-model-workers` | - | `custom_models_workers` | Custom models (Snakemake resource) |
| `aggregation.threads` | `AGGREGATION_THREADS` | - | - | - | Aggregation |

---

## Module-by-Module Worker Usage

### Organization (`rtpipeline/organize.py`)
- **Type:** ThreadPoolExecutor (I/O-bound)
- **Workers:** `config.effective_workers()` → `cpu_count - 1`
- **Used for:** DICOM file linking and RT details extraction
- **Parallelization level:** Per-course tasks

### Segmentation (`rtpipeline/segmentation.py`)
- **Type:** Subprocess (GPU-bound)
- **Workers:** 1 (sequential, GPU-safe)
- **Thread control:** `segmentation_thread_limit` → Sets OMP_NUM_THREADS
- **Parallelization level:** Sequential (one course at a time)

### DVH (`rtpipeline/dvh.py`)
- **Type:** ThreadPoolExecutor (I/O-bound)
- **Workers:** `config.effective_workers()` → `cpu_count - 1`
- **Used for:** Per-ROI DVH calculation
- **Parallelization level:** Per-ROI within each course

### Radiomics (`rtpipeline/radiomics_parallel.py`)
- **Type:** ProcessPoolExecutor (CPU-bound)
- **Workers:** `_calculate_optimal_workers()` → `cpu_count - 1`
  - Override: `RTPIPELINE_RADIOMICS_WORKERS`
- **Thread control:** `RTPIPELINE_RADIOMICS_THREAD_LIMIT` → Sets OMP_NUM_THREADS per worker
- **Parallelization level:** Per-ROI with process isolation

### Radiomics (Legacy) (`rtpipeline/radiomics.py`)
- **Type:** ThreadPoolExecutor (fallback)
- **Workers:** `config.effective_workers()` → `cpu_count - 1`
  - Override: `RTPIPELINE_RADIOMICS_WORKERS`
- **Note:** Legacy thread-based implementation (prone to segfaults)

### QC (`rtpipeline/quality_control.py`)
- **Type:** Per-course sequential
- **Workers:** `config.effective_workers()` (course-level parallelization in CLI)
- **Parallelization level:** Per-course (single-threaded within course)

### Custom Models (`rtpipeline/custom_models.py`)
- **Type:** Subprocess (GPU-bound)
- **Workers:** 1 (sequential, GPU-safe)
- **Thread control:** `segmentation_thread_limit` → Sets OMP_NUM_THREADS
- **Parallelization level:** Sequential (one course at a time)

### Aggregation (`rtpipeline/aggregation.py` - via Snakefile)
- **Type:** ThreadPoolExecutor (I/O-bound)
- **Workers:** `AGGREGATION_THREADS` → `cpu_count - 1`
- **Used for:** Combining results into Excel files
- **Parallelization level:** Per-file metadata extraction

---

## Precedence Rules

### Worker Count Precedence (highest to lowest):
1. CLI argument (`--workers N`)
2. Config file (`workers: N`)
3. Auto-detection (`cpu_count - 1`)

### Thread Limit Precedence (for radiomics):
1. CLI argument (`--radiomics-proc-threads N`)
2. Environment variable (`RTPIPELINE_RADIOMICS_THREAD_LIMIT=N`)
3. Config file (`radiomics.thread_limit: N`)
4. No limit (default)

### Worker Override for Radiomics:
1. Environment variable (`RTPIPELINE_RADIOMICS_WORKERS=N`)
2. `config.effective_workers()` → `cpu_count - 1`

---

## Deprecated/Removed Parameters

### ❌ REMOVED:
- `RTPIPELINE_RADIOMICS_MAX_WORKERS` → Use `RTPIPELINE_RADIOMICS_WORKERS` instead
- Hardcoded 4-worker cap in radiomics.py → Removed

### ⚠️ CONFUSING NAMES (kept for compatibility):
- `--seg-proc-threads` → Actually controls OMP_NUM_THREADS, not worker processes
- `--radiomics-proc-threads` → Actually controls OMP_NUM_THREADS, not worker processes
- Config: `segmentation.thread_limit` (properly named) vs CLI: `--seg-proc-threads` (confusing)

**Recommendation:** In future major version, rename to:
- `--seg-thread-limit`
- `--radiomics-thread-limit`

---

## Quick Reference: What Each Parameter Does

| Parameter | What It Controls | Recommended Value |
|-----------|------------------|-------------------|
| `workers` | Number of parallel workers for all non-GPU modules | `auto` (cpu_count-1) |
| `segmentation.workers` | Concurrent segmentation courses (GPU) | `1` (always) |
| `segmentation.thread_limit` | OpenMP threads in TotalSegmentator | `null` (no limit) |
| `radiomics.sequential` | Disable radiomics parallelization | `false` (enable parallel) |
| `radiomics.thread_limit` | OpenMP threads per radiomics worker | `null` (no limit) |
| `custom_models.workers` | Concurrent custom model courses (GPU) | `1` (always) |
| `aggregation.threads` | Threads for result aggregation | `auto` (cpu_count-1) |

---

## Validation

To verify your configuration:
```bash
# Check workers calculation
python -c "import os; print(f'CPU count: {os.cpu_count()}, Workers: {os.cpu_count()-1}')"

# Check config.yaml syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Check effective workers in Python
python -c "
from rtpipeline.config import PipelineConfig
from pathlib import Path
cfg = PipelineConfig(
    dicom_root=Path('.'),
    output_root=Path('.'),
    logs_root=Path('.')
)
print(f'Effective workers: {cfg.effective_workers()}')
"
```

---

## Summary

**YOU ONLY NEED TO SET ONE PARAMETER:**
```yaml
workers: auto  # Everything else is automatic
```

This gives you `cpu_count - 1` workers for all modules, with GPU-safe sequential segmentation.

All other parameters are for advanced use cases only:
- Thread limits: Only if experiencing thread oversubscription
- Worker overrides: Only for specific system constraints
- Sequential modes: Only for debugging or memory issues
