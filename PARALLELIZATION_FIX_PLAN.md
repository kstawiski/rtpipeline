# RTpipeline Parallelization Fix Plan

## Problem Statement

Current parallelization is inconsistent and confusing:
- **Snakefile**: Uses `cpu_count - 2` (min 4, max 32)
- **config.effective_workers()**: Uses `cpu_count - 1`
- **DVH**: Uses `cpu_count / 2` by default
- **Radiomics**: Uses `cpu_count - 1`
- Configuration parameters (cores, workers, threads) are unclear and inconsistent

## Research-Based Best Practices (2024-2025)

### ThreadPoolExecutor (I/O-bound tasks)
- Default in Python 3.8+: `min(32, cpu_count + 4)`
- For I/O operations: Can safely use more workers than CPU cores
- Examples: DICOM file reading, metadata extraction, file organization

### ProcessPoolExecutor (CPU-bound tasks)
- Default: `cpu_count`
- **Best practice**: `cpu_count - 1` to leave one core for system
- Avoid more workers than cores (causes context switching overhead)
- Examples: Radiomics calculations, image processing

### Snakemake Workflows
- `threads` directive reserves cores from scheduler
- **CRITICAL**: Must pass `{threads}` to actual command, not just declare it
- Snakemake auto-scales: `threads = min(declared_threads, available_cores)`

## Simplified Strategy

### Single Source of Truth
**Use `cpu_count - 1` for ALL modules** (except GPU-bound segmentation)

### Module-by-Module Changes

| Module | Current | New | Type | Justification |
|--------|---------|-----|------|---------------|
| **Snakefile WORKERS** | CPU-2 (min 4, max 32) | CPU-1 (min 1) | Per-course orchestration | Consistent with best practices |
| **Segmentation** | workers=1, threads=WORKERS | workers=1, threads=1 | GPU subprocess | GPU-safe, must be sequential |
| **DVH** | CPU/2 (default) | CPU-1 (via effective_workers) | ThreadPoolExecutor | I/O-bound, standardize |
| **Radiomics** | CPU-1 | CPU-1 (no change) | ProcessPoolExecutor | Already correct |
| **Organization** | CPU-1 | CPU-1 (no change) | ThreadPoolExecutor | Already uses effective_workers |
| **QC** | CPU-1 | CPU-1 (no change) | Per-course | Already uses effective_workers |
| **Metadata** | CPU-1 | CPU-1 (no change) | ThreadPoolExecutor | Already uses effective_workers |
| **Aggregation** | CPU (all) | CPU-1 | ThreadPoolExecutor | Standardize |

## Implementation Changes

### 1. Snakefile (PRIORITY 1)

**Line 53-57: Simplify WORKERS calculation**
```python
# BEFORE
WORKERS_CFG = config.get("workers", "auto")
if isinstance(WORKERS_CFG, str) and WORKERS_CFG.lower() == "auto":
    detected_cpus = os.cpu_count() or 4
    WORKERS = max(4, min(32, detected_cpus - 2))
else:
    WORKERS = int(WORKERS_CFG)

# AFTER
WORKERS_CFG = config.get("workers", "auto")
if isinstance(WORKERS_CFG, str) and WORKERS_CFG.lower() == "auto":
    detected_cpus = os.cpu_count() or 2
    WORKERS = max(1, detected_cpus - 1)
else:
    WORKERS = int(WORKERS_CFG)
```

**Line 329-332: Fix segmentation thread allocation**
```python
# BEFORE
rule segmentation_course:
    threads: max(1, WORKERS)
    resources:
        seg_workers=1

# AFTER
rule segmentation_course:
    threads: 1  # GPU-safe, sequential execution
    resources:
        seg_workers=1
```

**Line 182-186: Simplify aggregation threads**
```python
# BEFORE
if isinstance(_agg_threads_raw, str) and _agg_threads_raw.lower() == "auto":
    AGGREGATION_THREADS = os.cpu_count() or 4
else:
    AGGREGATION_THREADS = _coerce_int(_agg_threads_raw, None)

# AFTER
if isinstance(_agg_threads_raw, str) and _agg_threads_raw.lower() == "auto":
    AGGREGATION_THREADS = max(1, (os.cpu_count() or 2) - 1)
else:
    AGGREGATION_THREADS = _coerce_int(_agg_threads_raw, None)
```

### 2. DVH Module (PRIORITY 2)

**File: rtpipeline/dvh.py, Line 671**
```python
# BEFORE
worker_cap = max(1, parallel_workers or max(1, (os.cpu_count() or 2) // 2))

# AFTER
worker_cap = max(1, parallel_workers or max(1, (os.cpu_count() or 2) - 1))
```

**Ensure callers pass parallel_workers parameter**

### 3. Configuration Documentation

**Update config.yaml comments:**
```yaml
# Parallelization (SIMPLIFIED)
# - All modules use (cpu_count - 1) workers by default
# - Exception: segmentation uses 1 worker (GPU-safe)
# - Set workers: N to override with specific count
# - Set workers: auto for automatic detection
workers: auto

segmentation:
  workers: 1  # Always 1 for GPU safety (sequential execution)
  # Internal TotalSegmentator threading (keep low for memory safety)
  nr_threads_resample: 1
  nr_threads_save: 1
  num_proc_preprocessing: 1
  num_proc_export: 1

radiomics:
  sequential: false  # Use parallel processing (cpu_count - 1 workers)
  thread_limit: null  # Set to 1-2 if you encounter memory issues
```

### 4. Remove Dead Configuration

**Delete unused parameters:**
- `segmentation.threads_per_worker` - NOT used anywhere in code
- Clarify that `workers` is the only parallelization parameter needed

## Testing Plan

1. **Verify Snakefile calculates WORKERS correctly**
   ```bash
   snakemake --dry-run --cores all
   # Should show threads=1 for segmentation, threads=WORKERS for others
   ```

2. **Test with different CPU counts**
   - 4 cores: Should use 3 workers
   - 8 cores: Should use 7 workers
   - 16 cores: Should use 15 workers

3. **Verify segmentation remains sequential**
   ```bash
   # Check that only 1 segmentation runs at a time
   ps aux | grep TotalSegmentator
   ```

4. **Monitor resource usage**
   ```bash
   htop  # Should see (cpu_count - 1) cores utilized during DVH/radiomics
   ```

## Documentation Updates

1. **README.md**: Update parallelization section
2. **Configuration guide**: Simplify worker configuration instructions
3. **Delete**: PARALLELIZATION_*.md files (outdated analysis)
4. **Add**: Simple one-page guide: "Parallelization: Use cpu_count-1 everywhere, except GPU (1 worker)"

## Benefits

1. **Simplicity**: One rule (`cpu_count - 1`) instead of 3-4 different formulas
2. **Consistency**: All modules use the same approach
3. **Performance**: Better utilization than `cpu_count / 2` or `cpu_count - 2`
4. **Predictability**: Easy to understand and debug
5. **Safety**: GPU segmentation remains sequential, radiomics uses process isolation

## Migration Notes

- Existing configs with `workers: auto` will work better (use CPU-1 instead of CPU-2)
- Existing configs with numeric values (e.g., `workers: 4`) remain unchanged
- Performance should improve for systems with 4-8 cores (was overly conservative)
