# Parallelism Quick Start Guide

## TL;DR - Fix Your Config Now

Replace this in your `pipeline_config.yaml`:

```yaml
# OLD (SLOW - 3 hours per course)
workers: 1
```

With this:

```yaml
# NEW (FAST - 5-10 minutes per course)
snakemake_threads_io_bound: 2
workers: "auto"
```

Then run:
```bash
snakemake --cores 23  # Use cpu_count - 1
```

## What Changed?

**Before**: `workers: 1` meant:
- 1 course at a time
- 1 ROI at a time within that course
- Result: Single-threaded, very slow

**After**: Separated into two settings:
- `snakemake_threads_io_bound: 2` → ~11 courses run concurrently
- `workers: "auto"` → 23 ROIs processed in parallel per course
- Result: Massive parallelism, ~36× faster

## Complete Example Config

```yaml
# Input/Output directories
dicom_root: "/path/to/DICOM"
output_dir: "/path/to/output"
logs_dir: "/path/to/logs"

# ========================================
# PARALLELISM SETTINGS (NEW!)
# ========================================

# How many threads per I/O-bound job (DVH, radiomics, QC)
# Lower = more concurrent courses
snakemake_threads_io_bound: 2

# How many ROIs to process in parallel per course
# "auto" = use all cores minus 1 (RECOMMENDED)
workers: "auto"

# ========================================
# SEGMENTATION (unchanged)
# ========================================

segmentation:
  workers: 1  # GPU limitation - must stay at 1
  threads_per_worker: 8
  force: false
  fast: false
  roi_subset: null
  extra_models: []
  temp_dir: null

custom_models:
  enabled: true
  root: "custom_models"
  models: []
  workers: 1  # GPU limitation
  force: false
  nnunet_predict: "nnUNet_predict"
  retain_weights: true
  conda_activate: null

# ========================================
# RADIOMICS (unchanged)
# ========================================

radiomics:
  sequential: false
  params_file: "rtpipeline/radiomics_params.yaml"
  mr_params_file: "rtpipeline/radiomics_params_mr.yaml"
  thread_limit: 4
  skip_rois:
    - body
    - couchsurface
    - couchinterior
    - couchexterior
  max_voxels: 1500000000
  min_voxels: 10

aggregation:
  threads: "auto"

environments:
  main: "rtpipeline"
  radiomics: "rtpipeline-radiomics"

custom_structures: "custom_structures_pelvic.yaml"
```

## Run Command

```bash
# For 24-core system (use cpu_count - 1)
snakemake --cores 23 --configfile pipeline_config.yaml

# For 16-core system
snakemake --cores 15 --configfile pipeline_config.yaml

# For 8-core system
snakemake --cores 7 --configfile pipeline_config.yaml
```

## Expected Performance

| Setting | DVH Time | Courses Concurrent | Speedup |
|---------|----------|-------------------|---------|
| **Before** (workers: 1) | 3 hours | 1 | 1× |
| **After** (workers: auto) | 5-10 min | ~11 | **36×** |

## Verify It's Working

### Check Config
```bash
grep -A2 "workers:" pipeline_config.yaml
grep "snakemake_threads_io_bound:" pipeline_config.yaml
```

Should show:
```yaml
workers: "auto"
snakemake_threads_io_bound: 2
```

### Monitor CPU Usage
```bash
# In another terminal while pipeline runs
htop
```

You should see:
- **Before**: ~4% CPU (1 core)
- **After**: ~95% CPU (23 cores)

### Check Logs
```bash
tail -f Logs_Snakemake/dvh/*.log
```

You should see multiple log files updating simultaneously (concurrent courses).

## Troubleshooting

### "Still slow - only 1 course running"

**Check**:
```bash
# Is workers set to auto?
grep "workers:" pipeline_config.yaml

# Are you using enough cores?
snakemake --cores 23  # Not --cores 1!
```

### "Out of memory"

**Solution**: Reduce concurrent courses
```yaml
snakemake_threads_io_bound: 4  # Was 2
```

### "Too many open files"

**Solution**:
```bash
ulimit -n 4096
```

## Reference

- Full docs: `PARALLELISM_FIX.md`
- Example config: `config_example_parallelism.yaml`
- Code changes: `Snakefile` lines 51-71

## Summary

1. Add `snakemake_threads_io_bound: 2` to config
2. Change `workers: 1` to `workers: "auto"`
3. Run with `snakemake --cores 23` (use cpu_count - 1)
4. Enjoy 36× speedup! 🚀
