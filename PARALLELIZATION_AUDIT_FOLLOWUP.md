# Parallelization Parameters - Complete Audit

## Issues Found After Initial Fix

The user correctly identified that there are STILL multiple parallelization parameters scattered throughout the codebase. Here's the complete audit:

## 1. Hardcoded Worker Limits (CRITICAL BUG)

### radiomics.py Lines 510, 927
```python
max_workers = max(1, min(max_workers, 4))  # CAPS AT 4 WORKERS!
```

**Problem:** This OVERRIDES our cpu_count-1 approach and limits radiomics to max 4 workers!

**Impact:**
- 8-core system: Should use 7 workers, actually uses 4 (43% underutilization)
- 16-core system: Should use 15 workers, actually uses 4 (73% underutilization)
- 32-core system: Should use 31 workers, actually uses 4 (87% underutilization)

**Fix:** Remove the hardcoded cap of 4

## 2. Duplicate Environment Variables

### RTPIPELINE_RADIOMICS_WORKERS vs RTPIPELINE_RADIOMICS_MAX_WORKERS

**Current state:**
- `RTPIPELINE_RADIOMICS_WORKERS` - Used in `radiomics_parallel.py` (modern, process-based)
- `RTPIPELINE_RADIOMICS_MAX_WORKERS` - Used in `radiomics.py` (legacy, thread-based)

**Problem:** Confusing - two different variables for same purpose

**Recommendation:**
- Keep `RTPIPELINE_RADIOMICS_WORKERS` (used by modern implementation)
- Remove `RTPIPELINE_RADIOMICS_MAX_WORKERS` (legacy)
- Update radiomics.py to use RTPIPELINE_RADIOMICS_WORKERS

## 3. Unused Snakefile Parameter

### SEG_THREADS_PER_WORKER (Snakefile lines 84-86)

**Current behavior:**
```python
# Snakefile
SEG_THREADS_PER_WORKER = _coerce_int(SEG_CONFIG.get("threads_per_worker"), None)

# Line 379-380
if SEG_THREADS_PER_WORKER is not None:
    cmd.extend(["--seg-proc-threads", str(SEG_THREADS_PER_WORKER)])
```

**Analysis:**
- IS actually used! Passed to CLI as `--seg-proc-threads`
- Sets `segmentation_thread_limit` in PipelineConfig
- Used in segmentation.py line 327 to set OMP_NUM_THREADS environment variables

**Problem with naming:**
- Config file calls it: `segmentation.threads_per_worker`
- But it controls OpenMP thread limits, not worker count

**Fix:**
- Rename in config.yaml: `threads_per_worker` → `thread_limit`
- This matches the radiomics parameter naming

## 4. CLI Arguments Audit

### Current CLI Arguments:
```python
--workers                   # Main workers (cpu_count-1) ✅ CORRECT
--seg-workers               # Segmentation workers (always 1) ✅ CORRECT
--seg-proc-threads          # TotalSegmentator OMP thread limit ⚠️ CONFUSING NAME
--radiomics-proc-threads    # Radiomics OMP thread limit ✅ CORRECT
--custom-model-workers      # Custom model workers ✅ CORRECT
```

**Issues:**
- `--seg-proc-threads` name suggests it controls worker processes, but actually controls OpenMP threads
- Should be `--seg-thread-limit` to match `--radiomics-proc-threads` → should be `--radiomics-thread-limit`

## 5. Backup Files in Repository

Found these files that should be removed:
- `Snakefile.modified`
- `Snakefile.orig`
- `Snakefile.temp`

## Complete Parameter Map

| Config File | Snakefile Var | CLI Argument | PipelineConfig | Actual Usage | Status |
|-------------|---------------|--------------|----------------|--------------|--------|
| `workers: auto` | `WORKERS` | `--workers` | `workers` | All modules (cpu_count-1) | ✅ GOOD |
| `segmentation.workers: 1` | `SEG_MAX_WORKERS` | `--seg-workers` | `segmentation_workers` | Segmentation (always 1) | ✅ GOOD |
| `segmentation.threads_per_worker` | `SEG_THREADS_PER_WORKER` | `--seg-proc-threads` | `segmentation_thread_limit` | OMP_NUM_THREADS for TotalSeg | ⚠️ RENAME |
| `radiomics.thread_limit` | N/A | `--radiomics-proc-threads` | `radiomics_thread_limit` | OMP_NUM_THREADS for radiomics | ✅ GOOD |
| `custom_models.workers: 1` | `CUSTOM_MODELS_WORKERS` | `--custom-model-workers` | `custom_models_workers` | Custom models (always 1) | ✅ GOOD |
| `aggregation.threads: auto` | `AGGREGATION_THREADS` | N/A | N/A | Aggregation (cpu_count-1) | ✅ GOOD |
| N/A (env var) | N/A | N/A | N/A | RTPIPELINE_RADIOMICS_WORKERS | ✅ GOOD |
| N/A (env var) | N/A | N/A | N/A | RTPIPELINE_RADIOMICS_MAX_WORKERS | ❌ REMOVE |
| N/A (env var) | N/A | N/A | N/A | RTPIPELINE_RADIOMICS_THREAD_LIMIT | ✅ GOOD |

## Recommended Fixes

### Priority 1: Remove Hardcoded Limits
```python
# radiomics.py line 510, 927
# BEFORE
max_workers = max(1, min(max_workers, 4))

# AFTER
max_workers = max(1, max_workers)  # No artificial cap
```

### Priority 2: Consolidate Environment Variables
```python
# radiomics.py line 507, 924
# BEFORE
env_limit = int(os.environ.get('RTPIPELINE_RADIOMICS_MAX_WORKERS', '0') or 0)

# AFTER
env_limit = int(os.environ.get('RTPIPELINE_RADIOMICS_WORKERS', '0') or 0)
```

### Priority 3: Rename Config Parameters
```yaml
# config.yaml
segmentation:
  # BEFORE
  threads_per_worker: null

  # AFTER
  thread_limit: null  # Matches radiomics naming
```

### Priority 4: Clean Up CLI Argument Names (Breaking Change)
```python
# cli.py
# Consider for next major version:
p.add_argument("--seg-thread-limit", ...)  # Instead of --seg-proc-threads
p.add_argument("--radiomics-thread-limit", ...)  # Instead of --radiomics-proc-threads
```

## Summary

**Found 5 issues:**
1. ❌ **Hardcoded 4-worker limit in radiomics.py** (CRITICAL - breaks cpu_count-1 approach)
2. ⚠️ **Duplicate env vars** (RTPIPELINE_RADIOMICS_MAX_WORKERS vs RTPIPELINE_RADIOMICS_WORKERS)
3. ⚠️ **Confusing parameter naming** (threads_per_worker actually controls thread limits)
4. ⚠️ **CLI argument names don't match purpose** (--seg-proc-threads sounds like workers)
5. 🧹 **Backup files** (Snakefile.*, need cleanup)

**Next steps:**
1. Fix hardcoded 4-worker limit (IMMEDIATE)
2. Consolidate env vars (IMMEDIATE)
3. Rename config parameters (RECOMMENDED)
4. Update CLI argument names (CONSIDER for v2.0)
5. Clean up backup files (IMMEDIATE)
