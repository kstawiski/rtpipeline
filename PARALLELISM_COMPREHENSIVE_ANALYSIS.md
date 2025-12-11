# RT Pipeline Parallelism Implementation - Comprehensive Analysis

## Executive Summary

The rtpipeline implements a sophisticated multi-level parallelism strategy that adapts to different hardware configurations and workload characteristics. Each pipeline stage uses different parallelism techniques optimized for its specific computational characteristics (GPU-bound vs. I/O-bound vs. CPU-bound).

---

## 1. How run_pipeline.sh Orchestrates Snakemake Execution

### Entry Point: Snakefile (Main Workflow)

The pipeline uses **Snakemake** as its orchestration engine. Key orchestration mechanisms:

#### 1.1 Global Resource Configuration (Snakefile lines 206-213)
```python
SEG_WORKER_POOL = SEG_MAX_WORKERS if SEG_MAX_WORKERS is not None else max(1, WORKERS)
CUSTOM_SEG_WORKER_POOL = CUSTOM_MODELS_WORKERS if CUSTOM_MODELS_WORKERS is not None else 1
workflow.global_resources.update(
    {
        "seg_workers": max(1, SEG_WORKER_POOL),
        "custom_seg_workers": max(1, CUSTOM_SEG_WORKER_POOL),
    }
)
```

Snakemake uses **resource limits** to control concurrency:
- `seg_workers`: Limits concurrent segmentation tasks (GPU memory-intensive)
- `custom_seg_workers`: Limits concurrent custom model tasks (also GPU-intensive)

#### 1.2 Workers Configuration (Snakefile lines 51-57)
```python
WORKERS_CFG = config.get("workers", "auto")
if isinstance(WORKERS_CFG, str) and WORKERS_CFG.lower() == "auto":
    detected_cpus = os.cpu_count() or 4
    WORKERS = max(4, min(32, detected_cpus - 2))
else:
    WORKERS = int(WORKERS_CFG)
```

**Adaptive default**: `WORKERS = CPU_count - 2` (bounded 4-32)
- Min 4 to ensure parallelism on small systems
- Max 32 for safety on large systems (prevents resource exhaustion)

#### 1.3 Checkpoint-Based DAG (Lines 281-319)
Uses Snakemake **checkpoints** for dynamic workflow:
```python
checkpoint organize_courses:
    output: manifest=str(COURSE_MANIFEST)
    threads: max(1, WORKERS)
    ...
```

The `organize_courses` checkpoint creates a manifest of discovered courses, which other rules depend on dynamically:
- Enables per-course parallelization
- Allows downstream rules to process courses in parallel
- Creates sentinel files (`.organized`, `.segmentation_done`, etc.) to track progress

#### 1.4 Per-Course Rules Pattern

Each processing stage has a `{patient}_{course}` rule pair:
```python
rule segmentation_course:
    input: manifest=_manifest_input, organized=ORGANIZED_SENTINEL_PATTERN
    output: sentinel=SEGMENTATION_SENTINEL_PATTERN
    threads: max(1, SEG_THREADS_PER_WORKER if SEG_THREADS_PER_WORKER is not None else WORKERS)
    resources: seg_workers=1
    ...
```

**How it works**:
- `resources: seg_workers=1` declares that each rule invocation needs 1 unit of `seg_workers`
- Snakemake scheduler respects the global `workflow.global_resources` limits
- If `seg_workers=2`, Snakemake runs max 2 segmentation courses in parallel
- The `threads` parameter (CPU threads) is passed to CLI via `--workers` argument

---

## 2. Pipeline Stage Parallelism Implementation

### Stage Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ Snakemake Scheduler (controls inter-course parallelism)         │
│ - Manages resource limits (seg_workers, custom_seg_workers)     │
│ - Spawns Python subprocess for each course/stage combination    │
└─────────────────────────────────────────────────────────────────┘
         │
         ├─→ CLI (rtpipeline.cli.main)
         │   - Parses arguments
         │   - Initializes PipelineConfig
         │   - Calls stage-specific functions
         │
         └─→ Stage Handler (intra-course parallelism)
             - Uses ThreadPoolExecutor or ProcessPoolExecutor
             - Implements adaptive worker fallback
             - Handles memory-aware scaling

```

---

### 2.1 ORGANIZE Stage (Sequential per-dataset, threaded ROI processing)

**File**: `rtpipeline/organize.py`

**Parallelism**: 
- **Inter-course**: Single course at a time (checkpoint consolidation point)
- **Intra-course**: ThreadPoolExecutor for parallel DICOM discovery/organization

**Configuration** (Snakefile line 287):
```python
threads: max(1, WORKERS)  # Full worker pool for organize
```

**How it works**:
1. Reads DICOM files from disk
2. Groups by patient and course (merge criteria: `same_ct_study` or `frame_of_reference`)
3. Uses ThreadPoolExecutor for parallel file I/O operations
4. Outputs `manifest.json` with course metadata

**Resource Profile**: **I/O-bound**
- Quick scans through file metadata
- Parallelism helps with network/slow storage
- Minimal memory overhead

---

### 2.2 SEGMENTATION Stage (GPU-limited, concurrent per patient/course)

**File**: `rtpipeline/segmentation.py`

**Configuration** (config.yaml lines 30-44):
```yaml
segmentation:
  workers: 1              # TotalSegmentator concurrency
  threads_per_worker: null # CPU threads per worker
  device: "gpu"
  force_split: true       # Memory optimization
  nr_threads_resample: 1  # Threads for resampling
  nr_threads_save: 1      # Threads for saving
  num_proc_preprocessing: 1
  num_proc_export: 1
```

**Parallelism Levels**:

#### Level 1: Inter-Course (Snakemake Resource Limit)
```python
# Snakefile line 332
resources: seg_workers=1  # Only 1 course at a time (GPU memory)

# Snakefile line 206
SEG_WORKER_POOL = SEG_MAX_WORKERS if SEG_MAX_WORKERS is not None else max(1, WORKERS)
workflow.global_resources.update({"seg_workers": max(1, SEG_WORKER_POOL)})
```

**Default**: 1 concurrent course (safe for single GPU)
**Can increase to**: 2-4 on multi-GPU systems, but risks OOM

#### Level 2: Intra-Course (TotalSegmentator Internal Parallelism)

TotalSegmentator has built-in parallelism controlled via environment variables:

```python
# Snakefile lines 303-322 (passed to subprocess)
env['TOTALSEG_NUM_PROCESSES_PREPROCESSING'] = max_workers
env['TOTALSEG_NUM_PROCESSES_SEGMENTATION_EXPORT'] = max_workers
env['TOTALSEG_FORCE_TORCH_NUM_THREADS'] = '1'  # Prevent thread proliferation
```

**Key environment variables** (segmentation.py lines 328-371):
```python
thread_vars = (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'NUMBA_NUM_THREADS',
)

if thread_limit is not None:
    for var in thread_vars:
        env[var] = str(thread_limit_int)
```

**Thread Limiting Strategy**:
- If `seg_proc_threads` set: limits CPU threads per worker
- Prevents OpenMP thread oversubscription
- Crucial when multiple workers run in parallel

#### Level 3: CLI Argument Passing
```python
# cli.py lines 377-380 (passed from Snakefile)
if SEG_MAX_WORKERS:
    cmd.extend(["--seg-workers", str(max(1, SEG_MAX_WORKERS))])
if SEG_THREADS_PER_WORKER is not None:
    cmd.extend(["--seg-proc-threads", str(SEG_THREADS_PER_WORKER)])
```

**Resource Profile**: **GPU-bound, Memory-intensive**
- Deep learning inference (nnUNet backbone)
- Requires significant VRAM per concurrent job
- Must limit concurrency (default: 1)

---

### 2.3 CUSTOM MODELS Stage (Similar GPU controls)

**Configuration** (config.yaml lines 46-54):
```yaml
custom_models:
  enabled: false
  workers: 1           # Concurrent courses
  force: false
  nnunet_predict: "nnUNetv2_predict"
  retain_weights: true
  conda_activate: null
```

**Parallelism**: 
- Inter-course: Limited by `custom_models_workers` (default 1)
- Intra-course: nnUNet's built-in parallelism
- Adaptive backoff logic in cli.py (lines 970-987)

```python
# cli.py lines 970-987 (adaptive worker selection)
if cfg.custom_models_workers is not None:
    custom_worker_limit = cfg.custom_models_workers
elif cfg.totalseg_device == "cpu":
    custom_worker_limit = max(1, min(2, cfg.effective_workers() // 2))
else:
    custom_worker_limit = 1  # Conservative for GPU

logger.info("Custom segmentation stage: using %d parallel workers", custom_worker_limit)
```

**Resource Profile**: **GPU-bound, Highly Memory-intensive**
- Often larger model weights than TotalSegmentator
- GPU memory can be the limiting factor
- Default conservative: 1 worker at a time

---

### 2.4 DVH Stage (I/O-bound, ROI-level parallelism)

**File**: `rtpipeline/dvh.py`

**Per-Course Parallelism** (lines 537-682):
```python
def dvh_for_course(
    course_dir: Path,
    custom_structures_config: Optional[Union[str, Path]] = None,
    parallel_workers: Optional[int] = None,
    use_cropped: bool = True,
) -> Optional[Path]:
    ...
    worker_cap = max(1, parallel_workers or max(1, (os.cpu_count() or 2) // 2))
    task_results = run_tasks_with_adaptive_workers(
        f"DVH ({source})",
        tasks,
        _calc_roi,
        max_workers=worker_cap,
        logger=logger,
    )
```

**Parallelism Levels**:

#### Inter-Course (Snakemake)
- Multiple courses in parallel (controlled by main `WORKERS`)
- No resource limit, Snakemake uses `threads` parameter

#### Intra-Course (ThreadPoolExecutor)
- **ROI-level parallelism**: Each structure (ROI) is a separate task
- Uses `run_tasks_with_adaptive_workers()` (utils.py lines 323-510)
- Default workers: `CPU_count // 2`

**DVH Computation Tasks** (per RTSTRUCT, per ROI):
```python
# dvh.py lines 646-681
def process_struct(rs_path: Path, source: str, rx_dose: float) -> None:
    # Process: Manual RTSTRUCT, Auto RTSTRUCT, Custom structures
    tasks = []
    for roi in rois:
        tasks.append((roi_number, roi_name, source, rx_dose, rtstruct, builder))
    
    # Parallel ROI processing
    task_results = run_tasks_with_adaptive_workers(
        f"DVH ({source})",
        tasks,
        _calc_roi,  # Function that computes DVH for one ROI
        max_workers=worker_cap,
        logger=logger,
    )
```

**Resource Profile**: **I/O-bound, Memory-moderate**
- DVH computation is lightweight (mathematical operations)
- RTSTRUCT reading/parsing is I/O-bound
- Parallelism benefits from parallel I/O and cache efficiency
- ThreadPoolExecutor suitable (no GIL issues for I/O)

---

### 2.5 RADIOMICS Stage (CPU-intensive, Process-based parallelism)

**Files**: `rtpipeline/radiomics_parallel.py`, `rtpipeline/radiomics.py`

**Configuration** (config.yaml lines 56-73):
```yaml
radiomics:
  sequential: false
  params_file: "rtpipeline/radiomics_params.yaml"
  mr_params_file: "rtpipeline/radiomics_params_mr.yaml"
  thread_limit: null  # Limits OpenMP/BLAS threads per radiomics worker
  skip_rois: [...]
  max_voxels: 1500000000
  min_voxels: 10
```

**Parallelism Levels**:

#### Inter-Course (Snakemake)
- Multiple courses in parallel (no resource limit)
- Threads parameter: `max(1, WORKERS)` (default: CPU_count - 2)

#### Intra-Course (ProcessPoolExecutor - NOT ThreadPoolExecutor!)

**Why ProcessPoolExecutor?**
- PyRadiomics uses OpenMP/BLAS internally → thread-unsafe in multi-threaded context
- Segmentation faults can occur with ThreadPoolExecutor
- Solution: Process-based parallelism (spawn context) avoids shared memory issues

**Implementation** (radiomics_parallel.py lines 312-535):
```python
def parallel_radiomics_for_course(
    config: Any,
    course_dir: Path,
    custom_structures_config: Optional[Path] = None,
    max_workers: Optional[int] = None,
    use_cropped: bool = True
) -> Optional[Path]:
    
    # Determine workers (defaults to cpu_count - 1)
    if max_workers is None:
        max_workers = _calculate_optimal_workers()  # CPU count - 1
    
    # Prepare tasks (ROI-level extraction)
    prepared_tasks = []
    for source, roi, mask, cropped in tasks:
        task_file, task_params = _prepare_radiomics_task(
            img, mask, config, source, roi, course_dir, temp_dir, cropped
        )
        prepared_tasks.append((task_file, task_params))
    
    # Process in parallel using spawn context
    ctx = get_context('spawn')  # Not fork() to avoid issues
    with ctx.Pool(max_workers) as pool:
        for result in pool.imap_unordered(
            _isolated_radiomics_extraction_with_retry, 
            prepared_tasks
        ):
            if result:
                results.append(result)
```

**Thread Limiting Per Process** (radiomics_parallel.py lines 580-606):
```python
def _apply_thread_limit(limit: Optional[int]) -> None:
    if limit is None:
        for var in _THREAD_VARS:
            os.environ.pop(var, None)
        return
    limit = max(1, int(limit))
    value = str(limit)
    for var in _THREAD_VARS:
        os.environ[var] = value
```

**Task Serialization** (Process-safe temporary files):
```python
# radiomics_parallel.py lines 230-305
def _prepare_radiomics_task(...) -> Tuple[str, Dict[str, Any]]:
    # Pickle task data to temporary file
    # Reason: Process can't share memory, so data must be serialized
    temp_fd, temp_file_path = tempfile.mkstemp(suffix='.pkl', dir=temp_dir)
    task_data = {
        'img_array': img_array,
        'img_info': img_info,
        'mask': mask,
        'config': config,
        ...
    }
    with os.fdopen(temp_fd, 'wb') as f:
        pickle.dump(task_data, f)  # Secure serialization with RestrictedUnpickler
    
    return temp_file_path, task_params
```

**Resource Profile**: **CPU-intensive, Memory-moderate**
- Heavy feature computation (texture analysis, morphometry, etc.)
- Requires serialization overhead for process-based parallelism
- Benefits from high core counts
- Thread limit prevents thread explosion (PyRadiomics + NumPy + BLAS)

---

### 2.6 QC Stage (I/O-bound, Report generation)

**File**: `rtpipeline/quality_control.py`

**Parallelism**:
- Inter-course: Via main `WORKERS` (Snakemake)
- Intra-course: Could use ThreadPoolExecutor for parallel checks

**Pattern** (cli.py lines 1153-1185):
```python
if "qc" in stages:
    from .quality_control import generate_qc_report

    courses = ensure_courses()
    selected_courses = _filter_courses(courses)

    if not selected_courses:
        _log_skip("QC")
    else:
        def _run_qc(course):
            try:
                qc_dir = course.dirs.qc_reports
                qc_dir.mkdir(parents=True, exist_ok=True)
                generate_qc_report(course.dirs.root, qc_dir)
                return True
            except Exception as exc:
                logger.warning("QC stage failed for %s: %s", course.dirs.root, exc)
                return None

        effective_workers = cfg.effective_workers()
        logger.info("QC stage: using %d parallel workers", effective_workers)

        run_tasks_with_adaptive_workers(
            "QC",
            selected_courses,
            _run_qc,
            max_workers=effective_workers,
            logger=logging.getLogger(__name__),
            show_progress=True,
            task_timeout=args.task_timeout,
        )
```

**Resource Profile**: **I/O-bound, Lightweight**
- Report generation, validation checks
- Minimal CPU usage
- High inter-course parallelism acceptable

---

### 2.7 RADIOMICS ROBUSTNESS Stage (CPU-intensive, Perturbation-based)

**File**: `rtpipeline/radiomics_robustness.py`

**Configuration** (config.yaml lines 75-147):
```yaml
radiomics_robustness:
  enabled: false  # Disabled by default (long-running)
  modes:
    - segmentation_perturbation  # Mask erosion/dilation
  segmentation_perturbation:
    apply_to_structures:
      - "GTV*"
      - "CTV*"
      - "PTV*"
    small_volume_changes: [-0.15, 0.0, 0.15]
    large_volume_changes: [-0.30, 0.0, 0.30]
    max_translation_mm: 0.0
    n_random_contour_realizations: 0
    noise_levels: [0.0]
    intensity: "standard"  # mild / standard / aggressive
```

**Parallelism**:
- Per-course parallelism (inter-course)
- Intra-course: Process-based (similar to radiomics)
- Perturbation-level parallelism (multiple perturbations in parallel)

**Resource Profile**: **CPU-intensive, Very time-consuming**
- Multiple radiomics extractions per ROI (due to perturbations)
- Standard intensity: 15-30 perturbations per ROI
- Aggressive: 30-60 perturbations
- Should only be enabled for critical research studies

---

### 2.8 AGGREGATION Stage (Final consolidation, I/O-bound)

**File**: `Snakefile` lines 680-932

**Pattern** (Snakefile lines 696-932):
```python
rule aggregate_results:
    input:
        manifest=_manifest_input,
        dvh=_per_course_sentinels(".dvh_done"),
        radiomics=_per_course_sentinels(".radiomics_done"),
        qc=_per_course_sentinels(".qc_done"),
        custom=_per_course_sentinels(".custom_models_done")
    output:
        dvh=str(AGG_OUTPUTS["dvh"]),
        radiomics=str(AGG_OUTPUTS["radiomics"]),
        ...
    conda: "envs/rtpipeline.yaml"
    run:
        import json
        import os
        from concurrent.futures import ThreadPoolExecutor
        import pandas as pd

        worker_count = _max_workers(os.cpu_count() or 4)
        
        # Single thread pool for all I/O operations
        with ThreadPoolExecutor(max_workers=max(1, worker_count)) as pool:
            for course_data in pool.map(_load_all, courses):
                for key, df in course_data.items():
                    if df is not None and not df.empty:
                        results[key].append(df)
```

**Parallelism**:
- Read Excel files in parallel (I/O-bound)
- Single thread pool for all file types
- Configuration (config.yaml line 149): `threads: auto` (CPU count)

**Resource Profile**: **I/O-bound, Lightweight**
- Fast Excel reading in parallel
- Minimal CPU computation (mostly I/O)
- Can use full CPU count for workers

---

## 3. Executor Configuration and Worker Limits

### 3.1 Default Worker Calculations

**Global Default** (Snakefile lines 51-57):
```python
WORKERS_CFG = config.get("workers", "auto")
if isinstance(WORKERS_CFG, str) and WORKERS_CFG.lower() == "auto":
    detected_cpus = os.cpu_count() or 4
    WORKERS = max(4, min(32, detected_cpus - 2))
else:
    WORKERS = int(WORKERS_CFG)
```

**Formula**: `WORKERS = clamp(CPU_count - 2, min=4, max=32)`

**Rationale**:
- Reserve 1-2 cores for OS/system
- Min 4: ensure parallelism on 2-core systems
- Max 32: prevent resource exhaustion on large systems (>34 cores)

### 3.2 Per-Stage Configuration

| Stage | Config Field | Default | Type | Purpose |
|-------|--------------|---------|------|---------|
| **Segmentation** | `workers` (seg) | 1 | int | Concurrent courses (GPU memory limit) |
| | `threads_per_worker` | null | int/null | CPU threads per course |
| **Custom Models** | `workers` | 1 | int | Concurrent courses |
| **DVH** | Uses main `WORKERS` | auto | int | Intra-course ROI parallelism |
| **Radiomics** | Uses main `WORKERS` | auto | int | Intra-course ROI parallelism (processes) |
| **Radiomics** | `thread_limit` | null | int/null | OpenMP/BLAS threads per process |
| **QC** | Uses main `WORKERS` | auto | int | Inter-course parallelism |
| **Aggregation** | `threads` | auto | int | Excel file reading parallelism |

### 3.3 Resource Limits in Snakemake

```python
# Line 208-213 of Snakefile
workflow.global_resources.update(
    {
        "seg_workers": max(1, SEG_WORKER_POOL),          # Segmentation
        "custom_seg_workers": max(1, CUSTOM_SEG_WORKER_POOL),  # Custom models
    }
)
```

**How it works**:
- Each `segmentation_course` rule has `resources: seg_workers=1`
- Snakemake ensures total concurrent `seg_workers` ≤ global limit
- If limit is 1, only 1 course runs at a time (regardless of CPU count)
- If limit is 4, up to 4 courses run concurrently

---

## 4. Adaptive Logic for Resource Allocation

### 4.1 Memory-Aware Task Backoff (utils.py lines 323-510)

**Function**: `run_tasks_with_adaptive_workers()`

**Algorithm**:
```python
def run_tasks_with_adaptive_workers(
    label: str,
    items: Sequence[T],
    func: Callable[[T], R],
    *,
    max_workers: int,
    min_workers: int = 1,
    logger: Optional[logging.Logger] = None,
    show_progress: bool = False,
    task_timeout: Optional[int] = None,
) -> List[Optional[R]]:
```

**Behavior**:
1. **Start with full concurrency**: Uses `max_workers` to run tasks in parallel
2. **Detect memory errors**: Catches memory exceptions
   ```python
   if _is_memory_error(exc):
       mem_failures.append(idx)
   ```
3. **Reduce concurrency**: Halves worker count on memory errors
   ```python
   new_workers = max(min_workers, workers // 2)
   if new_workers == workers:
       new_workers = max(min_workers, workers - 1)
   ```
4. **Retry failed tasks**: Retries with reduced workers
5. **Give up at min_workers**: If memory error at single worker, task fails

**Example Flow**:
- Max workers = 8, task fails with MemoryError
- Retry with 4 workers
- If success, continue with 4 workers for remaining tasks
- If fail again, retry with 2 workers
- Continue until min_workers or success

**Memory Error Detection** (utils.py lines 286-303):
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

def _is_memory_error(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).lower()
    return any(pat in msg for pat in _MEMORY_PATTERNS)
```

### 4.2 Per-Stage Adaptive Selection

**Segmentation** (cli.py lines 908-937):
```python
# Adaptive worker limit based on device
if cfg.segmentation_workers is not None:
    seg_worker_limit = cfg.segmentation_workers
elif cfg.totalseg_device == "cpu":
    # CPU mode: can parallelize more since no GPU memory constraint
    seg_worker_limit = max(1, min(2, cfg.effective_workers() // 2))
else:
    # GPU mode: conservative default to avoid OOM
    seg_worker_limit = 1

seg_worker_limit = int(seg_worker_limit)
if seg_worker_limit < 1:
    seg_worker_limit = 1
seg_worker_limit = min(seg_worker_limit, max(1, cfg.effective_workers()))

logger.info("Segmentation stage: using %d parallel workers (device: %s, per-worker threads: resample=%d, save=%d)",
           seg_worker_limit, cfg.totalseg_device, totalseg_nr_thr_resamp, totalseg_nr_thr_saving)
```

**Custom Models** (cli.py lines 970-987):
```python
if cfg.custom_models_workers is not None:
    custom_worker_limit = cfg.custom_models_workers
elif cfg.totalseg_device == "cpu":
    custom_worker_limit = max(1, min(2, cfg.effective_workers() // 2))
else:
    custom_worker_limit = 1

custom_worker_limit = int(custom_worker_limit)
if custom_worker_limit < 1:
    custom_worker_limit = 1
custom_worker_limit = min(custom_worker_limit, max(1, cfg.effective_workers()))
```

**Radiomics Parallel** (radiomics_parallel.py lines 89-108):
```python
def _calculate_optimal_workers() -> int:
    """Calculate optimal number of workers (cpu_count - 1) with caching."""
    global _OPTIMAL_WORKERS_CACHE
    if _OPTIMAL_WORKERS_CACHE is not None:
        return _OPTIMAL_WORKERS_CACHE

    cpu_count = os.cpu_count() or 2
    optimal = max(1, cpu_count - 1)

    # Check if user has set a specific override
    env_workers = int(os.environ.get('RTPIPELINE_RADIOMICS_WORKERS', '0'))
    if env_workers > 0:
        optimal = env_workers
    else:
        logger.info("Calculated optimal radiomics workers: %d (CPU cores: %d, using: %d)",
                   optimal, cpu_count, optimal)

    _OPTIMAL_WORKERS_CACHE = optimal
    return optimal
```

### 4.3 TotalSegmentator Thread Management

**Internal Parallelism** (segmentation.py lines 303-371):
```python
# Environment variables controlling TotalSegmentator parallelism
nr_thr_resamp = getattr(config, "totalseg_nr_thr_resamp", None)
if nr_thr_resamp:
    try:
        nr_thr_resamp_int = max(1, int(nr_thr_resamp))
        cmd_parts.extend(["--nr_thr_resamp", str(nr_thr_resamp_int)])
    except (TypeError, ValueError):
        pass

nr_thr_saving = getattr(config, "totalseg_nr_thr_saving", None)
if nr_thr_saving:
    try:
        nr_thr_saving_int = max(1, int(nr_thr_saving))
        cmd_parts.extend(["--nr_thr_saving", str(nr_thr_saving_int)])
    except (TypeError, ValueError):
        pass

# Thread limiting environment variables
thread_vars = (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'NUMBA_NUM_THREADS',
)
if thread_limit is not None:
    thread_limit_int = max(1, int(thread_limit))
    thread_str = str(thread_limit_int)
    for var in thread_vars:
        env[var] = thread_str
```

**Multiprocessing Workers** (segmentation.py lines 359-360):
```python
env.setdefault("TOTALSEG_NUM_PROCESSES_PREPROCESSING", _to_env_int(getattr(config, "totalseg_num_proc_pre", None), 1))
env.setdefault("TOTALSEG_NUM_PROCESSES_SEGMENTATION_EXPORT", _to_env_int(getattr(config, "totalseg_num_proc_export", None), 1))
```

**Default Values** (cli.py lines 686-695):
```python
cpu_count = os.cpu_count() or 2
optimal_threads = max(2, min(8, cpu_count // 2))  # 25-50% of cores, min 2, max 8

totalseg_nr_thr_resamp = _positive_or_none(args.totalseg_nr_thr_resamp) or optimal_threads
totalseg_nr_thr_saving = _positive_or_none(args.totalseg_nr_thr_saving) or optimal_threads
totalseg_num_proc_pre = _positive_or_none(args.totalseg_num_proc_pre) or 1
totalseg_num_proc_export = _positive_or_none(args.totalseg_num_proc_export) or 1
```

**Rationale for defaults**:
- `nr_thr_resamp` & `nr_thr_saving`: Use 25-50% of cores (optimal for nnUNet)
- `num_proc_pre` & `num_proc_export`: Set to 1 (Docker compatibility, POSIX fork safety)

---

## 5. Configuration Defaults Related to Parallelism

### 5.1 config.yaml Defaults

**File**: `/projekty/rtpipeline/config.yaml`

```yaml
# Global worker count
workers: auto  # CPU_count - 2 (min 4, max 32)

# Segmentation
segmentation:
  workers: 1                      # Concurrent courses (GPU memory limit)
  threads_per_worker: null        # No artificial limit per course
  force: false
  fast: false
  roi_subset: null
  extra_models: []
  device: "gpu"                   # gpu / cpu / mps
  force_split: true               # Memory optimization
  nr_threads_resample: 1          # TotalSegmentator internal
  nr_threads_save: 1
  num_proc_preprocessing: 1
  num_proc_export: 1

# Custom models
custom_models:
  enabled: false                  # DISABLED (no weights)
  root: "custom_models"
  models: []
  workers: 1                      # Concurrent courses
  force: false
  nnunet_predict: "nnUNetv2_predict"
  retain_weights: true
  conda_activate: null

# Radiomics
radiomics:
  sequential: false               # Use parallel processing
  params_file: "rtpipeline/radiomics_params.yaml"
  mr_params_file: "rtpipeline/radiomics_params_mr.yaml"
  thread_limit: null              # No artificial limit (auto)
  skip_rois: [body, couchsurface, couchinterior, couchexterior, bones, m1, m2]
  max_voxels: 1500000000          # Skip large ROIs
  min_voxels: 10                  # Skip tiny ROIs

# Radiomics robustness
radiomics_robustness:
  enabled: false                  # DISABLED (long-running)
  modes:
    - segmentation_perturbation
  segmentation_perturbation:
    apply_to_structures:
      - "GTV*"
      - "CTV*"
      - "PTV*"
    small_volume_changes: [-0.15, 0.0, 0.15]
    large_volume_changes: [-0.30, 0.0, 0.30]
    max_translation_mm: 0.0
    n_random_contour_realizations: 0
    noise_levels: [0.0]
    intensity: "standard"         # mild / standard / aggressive

# Aggregation
aggregation:
  threads: auto                   # CPU count (I/O-bound)
```

### 5.2 CLI Argument Defaults

**File**: `rtpipeline/cli.py` lines 69-196

Key parallelism arguments:
```python
p.add_argument("--workers", type=int, default=None, help="Parallel workers for non-segmentation phases (default: auto)")
p.add_argument("--seg-workers", type=int, default=None, help="Maximum concurrent courses for TotalSegmentator (default: 1)")
p.add_argument("--seg-proc-threads", type=int, default=None, help="CPU threads per TotalSegmentator invocation (<=0 to disable limit)")
p.add_argument("--custom-model-workers", type=int, default=None, help="Maximum concurrent courses for custom segmentation models")
p.add_argument("--radiomics-proc-threads", type=int, default=None, help="CPU threads per radiomics worker (<=0 to disable limit)")
p.add_argument("--sequential-radiomics", action="store_true", help="Use sequential radiomics processing (parallel is default)")
p.add_argument("--task-timeout", type=int, default=None, help="Timeout for general pipeline tasks in seconds (default: None = no timeout)")
p.add_argument("--totalseg-timeout", type=int, default=3600, help="Timeout for TotalSegmentator operations in seconds (default: 3600 = 1 hour)")
```

### 5.3 Environment Variable Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `WORKERS` | auto | Global worker count |
| `TOTALSEG_TIMEOUT` | 3600 | TotalSegmentator timeout (1 hour) |
| `DCM2NIIX_TIMEOUT` | 300 | DICOM conversion timeout (5 min) |
| `RTPIPELINE_RADIOMICS_TASK_TIMEOUT` | 600 | Radiomics per-ROI timeout (10 min) |
| `RTPIPELINE_RADIOMICS_SEQUENTIAL` | 0 | Force sequential radiomics |
| `RTPIPELINE_RADIOMICS_WORKERS` | 0 (auto) | Override optimal radiomics workers |
| `RTPIPELINE_RADIOMICS_THREAD_LIMIT` | null | Limit OpenMP threads in radiomics |
| `OMP_NUM_THREADS` | 1 (TotalSeg) | OpenMP thread count |
| `OPENBLAS_NUM_THREADS` | 1 (TotalSeg) | OpenBLAS thread count |
| `MKL_NUM_THREADS` | 1 (TotalSeg) | Intel MKL thread count |
| `TOTALSEG_NUM_PROCESSES_PREPROCESSING` | 1 | nnUNet preprocessing workers |
| `TOTALSEG_NUM_PROCESSES_SEGMENTATION_EXPORT` | 1 | nnUNet export workers |
| `TOTALSEG_FORCE_TORCH_NUM_THREADS` | 1 | Prevent PyTorch threading |

---

## 6. Summary: Parallelism Strategy by Stage

| Stage | Inter-Course | Intra-Course | Worker Pool | Adaptive? | Timeout |
|-------|--------------|--------------|-------------|-----------|---------|
| **Organize** | Sequential (checkpoint) | ThreadPoolExecutor | WORKERS | No | 300s |
| **Segmentation** | Limited (GPU) | nnUNet internal | SEG_WORKERS (default 1) | Yes (CPU-aware) | 3600s |
| **Custom Models** | Limited (GPU) | nnUNet internal | CUSTOM_WORKERS (default 1) | Yes (CPU-aware) | 3600s |
| **DVH** | WORKERS | ThreadPoolExecutor (ROIs) | WORKERS//2 | Yes (memory) | task_timeout |
| **Radiomics** | WORKERS | ProcessPoolExecutor (ROIs) | CPU_count-1 | Yes (memory) | 600s per ROI |
| **QC** | WORKERS | Sequential | WORKERS | Yes (memory) | task_timeout |
| **Robustness** | WORKERS | ProcessPoolExecutor | CPU_count-1 | Yes (memory) | 600s per ROI |
| **Aggregation** | N/A (final) | ThreadPoolExecutor (files) | CPU_count | No | 60s per file |

---

## 7. Key Design Principles

### 7.1 Conservative GPU Defaults
- **Segmentation**: 1 course at a time (GPU memory safety)
- **Custom Models**: 1 course at a time (often larger models)
- **Rationale**: Deep learning inference is memory-intensive; overrun = OOM crash

### 7.2 Aggressive CPU/I/O Defaults
- **DVH**: CPU_count // 2 workers for ROI parallelism
- **Radiomics**: CPU_count - 1 workers (process-based)
- **Aggregation**: Full CPU_count (I/O-bound)
- **Rationale**: I/O and CPU-bound tasks scale well with core count

### 7.3 Memory Awareness
- **Adaptive backoff**: Automatic worker reduction on MemoryError
- **Thread limiting**: OpenMP/BLAS limits prevent thread explosion
- **Timeout handling**: Long-running tasks fail gracefully instead of hanging

### 7.4 Process-Based for Radiomics
- **Why not threads?** PyRadiomics uses OpenMP → segmentation faults with threading
- **Why ProcessPoolExecutor?** Isolates memory/threads per process
- **Serialization safety**: RestrictedUnpickler prevents code injection attacks

### 7.5 Resume Mode
- **Default: resume=True**: Reuse existing outputs, skip completed stages
- **Force redo**: `--force-redo` flag to regenerate everything
- **Manifest caching**: Store course list to speed up re-runs

---

## 8. Performance Tuning Recommendations

### Scenario 1: Single GPU, 16GB VRAM, 8 cores
```yaml
workers: 4              # Reduce from auto (6) to save memory
segmentation:
  workers: 1
  threads_per_worker: 2
radiomics:
  thread_limit: 2       # Limit OpenMP to prevent oversubscription
```

### Scenario 2: Dual GPU, 32GB VRAM, 32 cores
```yaml
workers: 16
segmentation:
  workers: 2            # One per GPU
  threads_per_worker: 4
  nr_threads_resample: 4
  nr_threads_save: 4
radiomics:
  thread_limit: 4
custom_models:
  workers: 2
```

### Scenario 3: CPU-only, 64 cores, 128GB RAM
```yaml
workers: 32
segmentation:
  workers: 4            # CPU can handle more parallelism
  threads_per_worker: 8
  device: "cpu"
  fast: true
radiomics:
  thread_limit: 8
```

### Scenario 4: NFS storage, slow I/O
```yaml
workers: 2              # More courses in parallel to hide I/O latency
segmentation:
  workers: 1
radiomics:
  thread_limit: 2       # Prevent I/O contention
```

---

## 9. Monitoring and Debugging

### 9.1 Log Files
- **Main log**: `Logs_Snakemake/rtpipeline.log`
- **Stage logs**: `Logs_Snakemake/stage_organize.log`, `segmentation/<patient>_<course>.log`, etc.

### 9.2 Key Log Messages
```
[INFO] ==============================================
[INFO] CPU cores: 32, using 30 workers by default
[INFO] TotalSegmentator: resampling threads=8, saving threads=8
[INFO] GPU device: gpu (force_split=true)
[INFO] Timeouts: TotalSegmentator=3600s, dcm2niix=300s, radiomics=600s
```

### 9.3 Debugging Hangs
- Check `top`/`htop` for hung processes
- Look for threads with 100% CPU but no progress in logs
- Increase task timeouts if hitting legitimate long jobs
- Reduce `--workers` if seeing contention

### 9.4 Memory Issues
- Monitor `free` or `nvidia-smi` during runs
- If OOM, automatic backoff should trigger
- Check logs for `"memory pressure detected"`
- Increase swap or reduce workers manually

---

## 10. Conclusion

The rtpipeline implements a sophisticated, **adaptive parallelism strategy** that:

1. **Respects hardware constraints** (GPU memory, available cores)
2. **Adapts to workload characteristics** (GPU-bound vs. I/O-bound vs. CPU-bound)
3. **Automatically scales on memory pressure** (adaptive worker backoff)
4. **Uses appropriate parallelism models**:
   - **Snakemake**: Inter-course orchestration with resource limits
   - **ThreadPoolExecutor**: I/O-bound tasks (DVH, Aggregation)
   - **ProcessPoolExecutor**: CPU-intensive tasks (Radiomics, Robustness)
   - **nnUNet internal**: TotalSegmentator deep learning inference
5. **Provides fine-grained control** via configuration and CLI arguments
6. **Defaults are conservative** for GPU-limited stages, aggressive for CPU/I/O stages

This design ensures the pipeline runs safely and efficiently across diverse hardware configurations while providing expert users with fine-grained tuning knobs.

