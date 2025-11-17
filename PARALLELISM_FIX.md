# Parallelism Fix - November 2025

## Problem Summary

The pipeline was not utilizing parallelism properly, causing DVH computation to take 3 hours per course when it should take ~5-10 minutes. The issue was caused by conflating two different parallelism levels.

## Root Cause

The `workers: 1` config parameter was being used for BOTH:
1. **Snakemake thread allocation** (controls how many courses run concurrently)
2. **Python worker count** (controls how many ROIs process in parallel within each course)

This caused:
- Only 1 course running at a time (due to thread allocation)
- Only 1 ROI being processed at a time within that course (due to worker count)
- Total parallelism: effectively single-threaded

## Solution

Decoupled the two parallelism levels:

### 1. Snakemake Thread Allocation (Inter-Course Parallelism)
- **New parameter**: `snakemake_threads_io_bound: 2`
- Controls how many courses can run concurrently
- For I/O-bound tasks (DVH, radiomics, QC), use LOW values
- Example: 2 threads per job on 24-core system → ~11 concurrent courses

### 2. Python Worker Count (Intra-Course Parallelism)
- **Existing parameter**: `workers: "auto"`
- Controls how many ROIs process in parallel within each course
- Now defaults to `cpu_count - 1` when set to "auto"
- Example: 23 workers on 24-core system → 23 ROIs processed concurrently per course

## Changes Made

### Snakefile Updates

1. **Added new configuration** (lines 51-71):
   ```python
   # Thread allocation strategy
   SNAKEMAKE_THREADS_IO_BOUND = 2  # Default for I/O-bound tasks
   WORKERS = cpu_count - 1  # When workers: "auto"
   ```

2. **Fixed critical bug** (line 404):
   - Added missing `worker_count` variable in `segmentation_custom_models` rule

3. **Updated all I/O-bound rules** to use proper parallelism:
   - `organize_courses`: threads=2, workers=auto
   - `dvh_course`: threads=2, workers=auto
   - `radiomics_course`: threads=2, workers=auto
   - `qc_course`: threads=2, workers=auto
   - `radiomics_robustness_course`: threads=2, workers=auto
   - `segmentation_custom_models`: threads=2, workers=auto

4. **Preserved GPU-bound settings**:
   - `segmentation_course`: threads=8 (from config), workers=1, resources=1
   - Ensures only 1 segmentation runs at a time (GPU limitation)

### Configuration File

Created `config_example_parallelism.yaml` with:
- Clear documentation of parallelism strategy
- Recommended settings for different system sizes
- Usage examples
- Expected performance metrics

## Recommended Configuration

### For Your 24-Core System

```yaml
# Snakemake thread allocation for I/O-bound tasks
snakemake_threads_io_bound: 2

# Worker count for parallel ROI processing
workers: "auto"  # Resolves to 23 workers

segmentation:
  workers: 1  # GPU limitation
  threads_per_worker: 8  # CPU parallelism within TotalSegmentator

radiomics:
  thread_limit: 4  # Prevent per-ROI CPU oversubscription
```

### Run Command

```bash
snakemake --cores 23 --configfile pipeline_config.yaml
```

## Expected Performance

### Before Fix
- **DVH per course**: ~3 hours
- **Concurrent courses**: 1
- **ROIs in parallel**: 1
- **Total CPU usage**: ~4% (1 core of 24)

### After Fix
- **DVH per course**: ~5-10 minutes
- **Concurrent courses**: ~11 (with 24 cores)
- **ROIs in parallel per course**: 23
- **Total CPU usage**: ~95% (23 cores of 24)

### Speedup Calculation
For 100 ROIs taking 2 seconds each:
- **Before**: 100 ROIs × 2 sec = 200 sec = 3.3 minutes (but only 1 course at a time)
- **After**: 100 ROIs / 23 workers × 2 sec = ~9 seconds per course (11 courses in parallel)
- **Effective speedup**: 36× faster overall throughput

## GPU-Bound Tasks (Segmentation)

Segmentation remains properly serialized:
- Only 1 course runs at a time (GPU memory limitation)
- Uses 8 CPU threads per segmentation (for data loading/preprocessing)
- Controlled by Snakemake resources, not thread allocation

## Migration Guide

### Update Your Config File

Replace your current `pipeline_config.yaml` with these settings:

```yaml
# NEW: Add this parameter
snakemake_threads_io_bound: 2

# CHANGE: Set workers to auto (instead of 1)
workers: "auto"

# KEEP: Segmentation settings unchanged
segmentation:
  workers: 1
  threads_per_worker: 8
```

### No Code Changes Required

The Snakefile has been updated automatically. Just update your config file and run.

## Troubleshooting

### Issue: Not seeing parallelism improvement

**Check**:
1. Verify `workers: "auto"` in config (not `workers: 1`)
2. Verify `snakemake --cores N-1` where N is your CPU count
3. Check logs for actual worker count used

**Debug**:
```bash
# Check CPU count detected
python -c "import os; print(f'CPUs: {os.cpu_count()}')"

# Monitor actual CPU usage during run
htop  # or top
```

### Issue: Out of memory errors

**Solution**: Reduce concurrent courses
```yaml
snakemake_threads_io_bound: 4  # Fewer concurrent courses
```

### Issue: Too many open files

**Solution**: Increase system limits
```bash
ulimit -n 4096
```

## Performance Tuning

### For Smaller Systems (8 cores)
```yaml
snakemake_threads_io_bound: 2
workers: "auto"  # → 7 workers
# Run with: snakemake --cores 7
# Result: ~3 concurrent courses, each using 7 workers
```

### For Larger Systems (64 cores)
```yaml
snakemake_threads_io_bound: 2
workers: "auto"  # → 63 workers
# Run with: snakemake --cores 63
# Result: ~31 concurrent courses, each using 63 workers
```

### For Memory-Constrained Systems
```yaml
snakemake_threads_io_bound: 4  # Reduce concurrent courses
workers: 32  # Cap workers to prevent memory issues
radiomics:
  thread_limit: 2  # Reduce per-ROI CPU usage
```

## Technical Details

### Two-Level Parallelism Architecture

```
Snakemake Scheduler (--cores 23)
    │
    ├─ Course A (threads=2) ──────┐
    │   └─ DVH: 23 workers        │
    │       ├─ ROI 1 (parallel)   │
    │       ├─ ROI 2 (parallel)   │
    │       └─ ... (23 at once)   │
    │                              │
    ├─ Course B (threads=2) ──────┤ ~11 courses
    │   └─ DVH: 23 workers        │ running
    │       └─ ...                │ concurrently
    │                              │
    ├─ ...                        │
    │                              │
    └─ Course K (threads=2) ──────┘
        └─ DVH: 23 workers
            └─ ...
```

### Thread Allocation Math

With `snakemake --cores 23` and `snakemake_threads_io_bound: 2`:
- Available threads: 23
- Threads per DVH job: 2
- Max concurrent DVH jobs: 23 / 2 = 11 (with 1 thread spare)

Each DVH job uses:
- Snakemake threads: 2 (reserved but not actively used)
- Python workers: 23 (actual parallel ROI processing)

The "thread" reservation is minimal overhead since DVH is I/O-bound and doesn't fully utilize the reserved threads.

## Files Modified

1. **Snakefile**:
   - Lines 51-71: Added parallelism configuration
   - Line 301: Updated `organize_courses` thread allocation
   - Line 404: Fixed `worker_count` bug in `segmentation_custom_models`
   - Line 469: Updated `dvh_course` thread allocation
   - Line 509: Updated `radiomics_course` thread allocation
   - Line 562: Updated `qc_course` thread allocation
   - Line 599: Updated `radiomics_robustness_course` thread allocation

2. **New Files**:
   - `config_example_parallelism.yaml`: Example configuration with documentation
   - `PARALLELISM_FIX.md`: This document

## Testing Checklist

- [x] Fix compiles without errors
- [ ] Pipeline runs with new config
- [ ] DVH completes in <10 minutes per course
- [ ] Multiple courses run concurrently
- [ ] Segmentation remains serialized (1 at a time)
- [ ] CPU usage near 95% during DVH/radiomics
- [ ] No out-of-memory errors
- [ ] Results match previous runs (numerical correctness)

## Contact

For issues or questions about this fix, please refer to:
- GitHub Issue: [Link to issue]
- Documentation: `config_example_parallelism.yaml`
- Code: See `Snakefile` lines 51-71 for parallelism strategy
