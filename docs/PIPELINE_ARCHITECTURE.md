# RT Pipeline - Architecture & Parallelization Analysis

## Executive Summary
The rtpipeline is a comprehensive radiotherapy data processing pipeline with:
- **Multi-stage parallel execution** using ThreadPoolExecutor for inter-course parallelization
- **Process-based parallelism** for radiomics with segmentation fault protection
- **GPU acceleration** for TotalSegmentator segmentation (CUDA/cuBLAS support)
- **Adaptive worker scaling** with automatic fallback on memory pressure
- **Thread-limited processing** to prevent OpenMP/threading conflicts

---

## 1. MAIN PIPELINE IMPLEMENTATION FILES

### Core Entry Points
- **`/home/user/rtpipeline/rtpipeline/cli.py`** (924 lines)
  - Main command-line interface with full argument parsing
  - Orchestrates all pipeline stages: organize, segmentation, dvh, visualize, radiomics, qc
  - Implements adaptive worker management per stage
  - Subcommands: `doctor` (environment check), `validate` (config validation)

- **`/home/user/rtpipeline/rtpipeline/config.py`** (89 lines)
  - Central configuration dataclass: `PipelineConfig`
  - Defines all pipeline parameters and defaults
  - Key method: `effective_workers()` - auto-calculates workers (CPU count - 1)

### Configuration Files
- **`config.yaml`** - Master configuration (YAML format)
  - Workers: auto/manual setting
  - Device selection (gpu/cpu/mps)
  - Thread limits per stage
  - Resource allocation parameters

---

## 2. CURRENT PARALLEL PROCESSING MECHANISMS

### A. ThreadPoolExecutor-Based Inter-Course Parallelization
**File:** `/home/user/rtpipeline/rtpipeline/utils.py` (lines 323-465)

**Function:** `run_tasks_with_adaptive_workers()`

**Key Features:**
```python
- Parallel execution of independent course processing tasks
- Uses concurrent.futures.ThreadPoolExecutor
- Intelligent memory pressure detection
- Automatic worker scaling: reduces workers when MemoryError detected
- Fallback strategy:
  * Half workers on first memory error
  * Single worker on continued errors
  * Complete failure only after exhausting all options
```

**Worker Scaling Logic:**
```
Initial Workers = min(max_workers, len(tasks))
If memory_error detected AND workers > min_workers:
  new_workers = max(min_workers, workers // 2)
  if new_workers == workers:
    new_workers = workers - 1
Retry with reduced workers
```

**Progress Monitoring:**
- Tracks: completed/total, percentage, elapsed time, ETA
- Logs every completion if `show_progress=True`

---

### B. Process-Based Radiomics Parallelization
**File:** `/home/user/rtpipeline/rtpipeline/radiomics_parallel.py` (590 lines)

**Process Pool Implementation:**
```python
- Uses multiprocessing.Pool with 'spawn' context (safer than 'fork')
- Function: parallel_radiomics_for_course()
- Optimal workers: _calculate_optimal_workers() → CPU count - 1
- Task distribution: One task per ROI/source combination
- Retry mechanism: Up to 3 attempts per task with exponential backoff
```

**Key Advantages Over Threading:**
- Avoids segmentation faults from pyradiomics/OpenMP interactions
- Process isolation prevents memory leaks
- Each process: dedicated thread limit (OMP_NUM_THREADS=1)

**Task Processing:**
```python
_isolated_radiomics_extraction():
  1. Load task data from temporary pickle file (RestrictedUnpickler for security)
  2. Reconstruct SimpleITK image
  3. Apply per-process thread limit (OMP_NUM_THREADS=1)
  4. Execute radiomics feature extraction
  5. Serialize results to DataFrame
  6. Clean up temp file
```

**Security Feature:** RestrictedUnpickler
- Whitelist approach: only allows safe types
- Prevents arbitrary code execution from pickle files
- Allowed: numpy arrays, Path, dict, list, basic types

---

## 3. GPU UTILIZATION CODE

### A. TotalSegmentator GPU Configuration
**File:** `/home/user/rtpipeline/rtpipeline/segmentation.py` (lines 243-363)

**GPU Device Selection:**
```python
# From config:
totalseg_device: str = "gpu"  # choices: "gpu", "cpu", "mps"
totalseg_force_split: bool = True  # Force chunked inference

# Environment variables set:
env["TOTALSEG_DEVICE"] = device
env["TOTALSEG_ACCELERATOR"] = "cuda" if device=="gpu" else device
env["CUDA_VISIBLE_DEVICES"] = "all"  # Default
```

**CUDA Environment Variables:**
```python
# Fallback strategy for OOM:
if TotalSegmentator fails with GPU:
    retry with: -d cpu + CUDA_VISIBLE_DEVICES='-1'
```

**Memory Optimization:**
- `--force_split`: Chunks large volumes for GPU memory reduction
- `TOTALSEG_PRELOAD_WEIGHTS`: "1" (pre-load model into GPU)
- Thread constraints to prevent GPU overload

### B. PyTorch/CUDA Initialization
**File:** `/home/user/rtpipeline/rtpipeline/cli.py` (lines 207-229)

**Doctor Command - CUDA Diagnostics:**
```bash
rtpipeline doctor

Checks:
- torch.cuda availability
- CUDA device count
- nvidia-smi GPU info
- TotalSegmentator version
```

---

## 4. CPU/CORE ALLOCATION SETTINGS

### Configuration Hierarchy (Priority Order)

1. **Explicit Command-Line Arguments**
   ```
   --workers N                    # Overall parallelism
   --seg-workers N                # TotalSegmentator parallelism
   --seg-proc-threads N           # CPU threads per TotalSegmentator
   --radiomics-proc-threads N     # CPU threads per radiomics worker
   --custom-model-workers N       # Concurrent custom model courses
   ```

2. **Config File (config.yaml)**
   ```yaml
   workers: auto                  # Default: CPU count - 1

   segmentation:
     workers: 1                   # Always 1 for GPU safety (do not increase)
     nr_threads_resample: 1       # nnUNet preprocessing threads
     nr_threads_save: 1           # nnUNet export threads
     num_proc_preprocessing: 1    # nnUNet preprocessing workers
     num_proc_export: 1           # nnUNet export workers
   
   radiomics:
     thread_limit: 4              # CPU threads per radiomics worker
   
   custom_models:
     workers: 1                   # Concurrent custom model courses
   ```

### Auto-Calculation Logic
**Method:** `PipelineConfig.effective_workers()` (config.py, lines 83-88)
```python
def effective_workers(self) -> int:
    import os
    if self.workers and self.workers > 0:
        return int(self.workers)
    cpu = os.cpu_count() or 2
    return max(1, cpu - 1)
```

### Per-Stage Worker Allocation
```
Stage                Calculation                    Default
─────────────────────────────────────────────────────────────
Segmentation        min(seg_workers, effective)    1
Custom Models       min(custom_workers, effective) 1
DVH                 effective_workers              CPU-1
Radiomics           min(optimal, effective)        CPU-1
Visualization       effective_workers              CPU-1
CT Cropping         effective_workers              CPU-1
```

### Thread Limit Management

**Environment Variables Controlled:**
```python
_THREAD_VARS = (
    'OMP_NUM_THREADS',            # OpenMP
    'OPENBLAS_NUM_THREADS',       # OpenBLAS
    'MKL_NUM_THREADS',            # Intel MKL
    'NUMEXPR_NUM_THREADS',        # NumExpr
    'NUMBA_NUM_THREADS',          # Numba
)
```

**Application Points:**
1. TotalSegmentator execution (segmentation.py)
2. Radiomics extraction per process (radiomics_parallel.py)
3. Radiomics extraction per worker (radiomics.py)

---

## 5. PARALLELIZATION CONFIGURATION

### Configuration File Locations
```
/home/user/rtpipeline/config.yaml              # Main runtime config
/home/user/rtpipeline/rtpipeline/config.py     # Config dataclass definition
/home/user/rtpipeline/envs/rtpipeline.yaml    # Conda environment
/home/user/rtpipeline/docker-compose.yml       # Container orchestration
```

### Stage-by-Stage Parallelization Strategy

#### 1. ORGANIZE Stage (Sequential)
- Single-threaded discovery and metadata extraction
- No parallelization (I/O bound, metadata parsing)

#### 2. SEGMENTATION Stage
```python
Workers: min(segmentation_workers, effective_workers)
Default: 1 worker (GPU memory constraint)
Concurrency: Multiple TotalSegmentator jobs per course
Memory Management: Adaptive fallback on OOM
```

**Code Location:** cli.py lines 679-713

#### 3. CUSTOM SEGMENTATION Stage
```python
Workers: min(custom_models_workers, effective_workers)
Default: 1 worker
Per-task: One course per worker (GPU intensive)
```

**Code Location:** cli.py lines 715-762

#### 4. DVH Stage
```python
Workers: effective_workers
Method: ThreadPoolExecutor
Parallelism: One course per thread
Per-course: radiomics feature extraction parallelized
```

**Code Location:** cli.py lines 799-828

#### 5. RADIOMICS Stage
```python
Workers: _calculate_optimal_workers() [CPU - 1]
Method: ProcessPoolExecutor (spawn context)
Per-course: ROI extraction tasks parallelized
Thread limit: Applied per worker process
```

**Code Location:** cli.py lines 859-902

#### 6. VISUALIZE Stage
```python
Workers: effective_workers
Method: ThreadPoolExecutor
Per-course: HTML/image generation parallelized
```

**Code Location:** cli.py lines 830-857

#### 7. QC Stage
```python
Method: Sequential loop over courses
Single-threaded quality control report generation
```

**Code Location:** cli.py lines 904-919

### Radiomics Parallelization Control

**Environment Variables:**
```python
RTPIPELINE_USE_PARALLEL_RADIOMICS = '1'  # Enable parallel mode
RTPIPELINE_RADIOMICS_WORKERS = 'N'       # Override worker count
RTPIPELINE_RADIOMICS_THREAD_LIMIT = 'N'  # Thread limit per worker
RTPIPELINE_RADIOMICS_SEQUENTIAL = '1'    # Force sequential (legacy)
```

**Sequential Mode Trigger:**
```python
# CLI: --sequential-radiomics flag
# Sets: os.environ['RTPIPELINE_RADIOMICS_SEQUENTIAL'] = '1'
# Uses: Original radiomics.py implementation (no parallelization)
```

---

## 6. DOCKER CONTAINER RESOURCE ALLOCATION

**File:** `/home/user/rtpipeline/docker-compose.yml`

### GPU Configuration (Default)
```yaml
rtpipeline (GPU-enabled):
  environment:
    - NVIDIA_VISIBLE_DEVICES=all     # All GPUs visible
    - CUDA_VISIBLE_DEVICES=all       # All GPUs accessible
  shm_size: '4gb'                    # PyTorch shared memory
  restart: unless-stopped
```

### CPU-Only Configuration
```yaml
rtpipeline-cpu (profile: cpu-only):
  deploy:
    resources:
      limits:
        cpus: '16.0'
        memory: 32G
      reservations:
        cpus: '4.0'
        memory: 8G
  profiles:
    - cpu-only
```

**Activation:** `docker-compose --profile cpu-only up`

### Jupyter Service
```yaml
jupyter:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1              # Single GPU
            capabilities: [gpu]
```

---

## 7. PERFORMANCE MONITORING & LOGGING

### Log Output Files
```
/home/user/rtpipeline/Logs/rtpipeline.log  # Main pipeline log
Per-stage logs with progress tracking
```

### Adaptive Worker Progress Logging
**Function:** `_log_progress()` (utils.py, lines 306-320)

```
Format: "Label: X/Y (Z%) elapsed Ats ETA Bs"
Updates: Per task completion (if show_progress=True)
Example: "Segmentation: 3/10 (30%) elapsed 12.5s ETA 29.2s"
```

### Memory Pressure Detection
**Pattern Matching:** (utils.py, lines 286-303)
```python
_MEMORY_PATTERNS = (
    "out of memory",
    "cuda out of memory",
    "cublas status alloc failed",
    "std::bad_alloc",
    "cannot allocate memory",
    "failed to allocate",
    "not enough memory",
    "mmap failed",
    "oom",
)
```

---

## 8. KEY CONFIGURATION DEFAULTS

### Absolute Defaults (Hardcoded)
```python
workers:                           CPU count - 1
segmentation_workers:              1 (GPU constraint)
segmentation_thread_limit:         None (no limit per worker)
radiomics_thread_limit:           4 (if config specifies)
radiomics max_workers:            _calculate_optimal_workers()
custom_models_workers:            1 (GPU constraint)
totalseg_device:                  "gpu"
totalseg_force_split:             True
totalseg_nr_thr_resamp:           1
totalseg_nr_thr_saving:           1
totalseg_num_proc_pre:            1
totalseg_num_proc_export:         1
```

### Conda Dependencies (GPU-enabled)
**File:** `/home/user/rtpipeline/envs/rtpipeline.yaml`
```yaml
- pytorch=2.3.*
- pytorch-cuda=12.1         # CUDA 12.1 support
- torchvision=0.18.*
- torchaudio=2.3.*
- TotalSegmentator>=2.4.0
```

---

## 9. CURRENT OPTIMIZATION OPPORTUNITIES

### Identified in Codebase
1. **Thread Limit Per Worker**: Can be set but defaults to unlimited
2. **GPU Memory Optimization**: force_split already enabled
3. **Process Pool Caching**: Weights pre-loading enabled
4. **Sequential Fallback**: Available via --sequential-radiomics flag

### Configuration Examples

**Maximum GPU Utilization:**
```bash
rtpipeline \
  --dicom-root data/ \
  --outdir output/ \
  --seg-workers 4 \
  --workers 8 \
  --totalseg-device gpu
```

**Memory-Constrained System:**
```bash
rtpipeline \
  --dicom-root data/ \
  --outdir output/ \
  --workers 2 \
  --seg-workers 1 \
  --seg-proc-threads 4 \
  --radiomics-proc-threads 2 \
  --sequential-radiomics
```

**CPU-Only Mode:**
```bash
rtpipeline \
  --dicom-root data/ \
  --outdir output/ \
  --totalseg-device cpu \
  --workers 8 \
  --seg-workers 2
```

---

## 10. SUMMARY TABLE

| Component | Type | Location | Key Parameters |
|-----------|------|----------|-----------------|
| **Main Orchestrator** | CLI | cli.py | --workers, --seg-workers |
| **Inter-course Parallelism** | ThreadPoolExecutor | utils.py:323 | max_workers, min_workers |
| **Radiomics Parallelism** | ProcessPoolExecutor | radiomics_parallel.py:311 | max_workers, thread_limit |
| **GPU Segmentation** | External (TotalSegmentator) | segmentation.py:243 | --totalseg-device |
| **Memory Adaptation** | Dynamic Scaling | utils.py:363-460 | Memory error detection |
| **Thread Management** | Environment Variables | Across modules | OMP_NUM_THREADS, etc. |
| **Configuration** | YAML + CLI | config.yaml, cli.py | All parameters |
| **Container Orchestration** | Docker Compose | docker-compose.yml | GPU/CPU profiles |
| **Logging** | File + Console | Logs/rtpipeline.log | Progress tracking |

---

## File Reference Summary

### Core Pipeline Files
- **config.py** (89 lines) - Configuration dataclass
- **cli.py** (924 lines) - Command-line interface & orchestration
- **utils.py** (466 lines) - Adaptive worker pool implementation
- **segmentation.py** (836 lines) - TotalSegmentator integration
- **radiomics_parallel.py** (590 lines) - Process-based radiomics
- **radiomics.py** (first 150+ lines) - Feature extraction
- **custom_models.py** (100+ lines) - Custom model execution

### Configuration Files
- **config.yaml** - Runtime configuration
- **docker-compose.yml** - Container orchestration
- **envs/rtpipeline.yaml** - Conda environment definition
- **custom_structures_*.yaml** - Structure definitions

---

