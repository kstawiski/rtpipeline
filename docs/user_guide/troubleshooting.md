# Pipeline Hang Prevention & Timeout Mechanisms

## Overview
This document describes the comprehensive timeout and hang prevention mechanisms implemented to prevent the pipeline from hanging indefinitely during long-running operations.

## Problem Statement
The pipeline was experiencing hanging issues where:
- **TotalSegmentator** operations would hang indefinitely on GPU/CPU issues
- **dcm2niix** conversions would stall on corrupted DICOM files
- **Radiomics parallel processing** would block if any worker hung
- **No progress indicators** made it impossible to distinguish between slow progress and actual hangs
- **No watchdog mechanisms** to detect and handle stuck processes

## Solutions Implemented

### 1. Subprocess Timeout Protection

#### TotalSegmentator Timeout
**File:** `rtpipeline/segmentation.py:32-68`

All TotalSegmentator operations now have configurable timeouts:
```python
# Default: 3600 seconds (1 hour) per segmentation task
timeout = int(os.environ.get('TOTALSEG_TIMEOUT', '3600'))
```

**CLI Option:**
```bash
--totalseg-timeout 3600  # Default: 1 hour
```

**Behavior:**
- Automatically detects TotalSegmentator commands and applies appropriate timeout
- Logs detailed timeout errors when operations exceed the limit
- Continues with next task instead of hanging indefinitely

#### dcm2niix Timeout
**File:** `rtpipeline/segmentation.py:32-68`

DICOM to NIfTI conversions have shorter timeouts:
```python
# Default: 300 seconds (5 minutes) per conversion
timeout = int(os.environ.get('DCM2NIIX_TIMEOUT', '300'))
```

**CLI Option:**
```bash
--dcm2niix-timeout 300  # Default: 5 minutes
```

**Behavior:**
- Protects against corrupted DICOM files that cause dcm2niix to hang
- Logs timeout and continues with next operation
- Pipeline falls back to DICOM-mode segmentation if NIfTI conversion fails

### 2. Task-Level Timeout Protection

#### Adaptive Workers Timeout
**File:** `rtpipeline/utils.py:323-333, 410-433`

All pipeline stages using `run_tasks_with_adaptive_workers()` now support per-task timeouts:

```python
def run_tasks_with_adaptive_workers(
    ...,
    task_timeout: Optional[int] = None,
)
```

**CLI Option:**
```bash
--task-timeout 7200  # 2 hours per course (segmentation/DVH/etc.)
```

**Behavior:**
- Each course/task has its own timeout
- Times out and logs error if task exceeds limit
- Continues processing other courses instead of hanging
- Applies to: Segmentation, Custom Models, CT Cropping, DVH, Visualization

**Example timeout errors:**
```
ERROR: Segmentation: task #5 (Patient123/Course1) timed out after 7200s
```

### 3. Radiomics Parallel Processing

#### Progress Monitoring
**File:** `rtpipeline/radiomics_parallel.py:86`

The radiomics processing includes configurable timeout and progress monitoring:
```python
_TASK_TIMEOUT = int(os.environ.get('RTPIPELINE_RADIOMICS_TASK_TIMEOUT', '600'))
```

**CLI Option:**
```bash
--radiomics-task-timeout 600  # Default: 10 minutes per ROI
```

**Behavior:**
- Uses `imap_unordered()` for progress monitoring and efficient task distribution
- Progress is logged periodically as tasks complete
- Individual task retries are handled within worker processes
- Note: Per-task timeout enforcement is handled at the subprocess level via the `_TASK_TIMEOUT` configuration within worker functions, not by the `imap_unordered()` iterator itself

**Progress Logging:**
```
INFO: Radiomics progress: 50/100 (50.0%), ETA: 234.5s
INFO: Radiomics progress: 100/100 (100.0%), ETA: 0.0s
```

### 4. Heartbeat & Progress Monitoring

#### Periodic Heartbeat Logging
**File:** `rtpipeline/utils.py:403-408`

Pipeline now logs heartbeat messages every 60 seconds:
```python
if now - last_heartbeat > 60:  # Log every 60 seconds
    log.info("%s: Still processing... %d/%d completed (%.1f%%)",
            label, completed, total, 100 * completed / total)
```

**Benefits:**
- Distinguishes between slow progress and actual hangs
- Shows pipeline is still alive and making progress
- Helps identify which stage is slow

#### Slow Task Detection
**File:** `rtpipeline/utils.py:418-422`

Tasks taking longer than 5 minutes trigger warnings:
```python
if task_duration > 300:  # Warn if task took more than 5 minutes
    log.warning("%s: task #%d (%s) took %.1fs (slow)",
               label, idx + 1, item_desc, task_duration)
```

**Benefits:**
- Identifies slow courses/tasks for investigation
- Helps tune timeout values
- Provides performance insights

### 5. Enhanced Error Handling

#### Timeout Error Handling
**File:** `rtpipeline/utils.py:424-433`

Explicit handling for timeout errors:
```python
except TimeoutError:
    log.error("%s: task #%d (%s) timed out after %ds",
             label, idx + 1, item_desc, task_timeout)
    completed += 1
    results[idx] = None
```

**Behavior:**
- Logs detailed timeout information
- Marks task as failed (None result)
- **Continues processing remaining tasks**
- Doesn't crash entire pipeline

#### Memory Error Handling
Already present, now combined with timeout handling:
- Detects OOM errors
- Automatically reduces worker count
- Retries with lower parallelism

## Configuration Matrix

### Default Timeouts

| Operation | Default Timeout | Environment Variable | CLI Option |
|-----------|----------------|---------------------|------------|
| TotalSegmentator | 3600s (1 hour) | `TOTALSEG_TIMEOUT` | `--totalseg-timeout` |
| dcm2niix | 300s (5 min) | `DCM2NIIX_TIMEOUT` | `--dcm2niix-timeout` |
| General tasks | None (disabled) | N/A | `--task-timeout` |
| Radiomics ROI | 600s (10 min) | `RTPIPELINE_RADIOMICS_TASK_TIMEOUT` | `--radiomics-task-timeout` |

### Recommended Settings

#### For Fast Systems (32+ cores, GPU)
```bash
rtpipeline \
  --totalseg-timeout 1800 \
  --dcm2niix-timeout 300 \
  --task-timeout 3600 \
  --radiomics-task-timeout 300
```

#### For Slow Systems or Large Datasets
```bash
rtpipeline \
  --totalseg-timeout 7200 \
  --dcm2niix-timeout 600 \
  --task-timeout 14400 \
  --radiomics-task-timeout 1200
```

#### For Debugging (catch hangs quickly)
```bash
rtpipeline \
  --totalseg-timeout 600 \
  --dcm2niix-timeout 120 \
  --task-timeout 1800 \
  --radiomics-task-timeout 180
```

#### For Production (balanced)
```bash
rtpipeline \
  --totalseg-timeout 3600 \
  --dcm2niix-timeout 300 \
  --task-timeout 7200 \
  --radiomics-task-timeout 600
```

## Monitoring for Hangs

### Log Patterns to Watch For

#### Normal Progress
```
INFO: Segmentation stage: using 1 parallel workers
INFO: Segmentation: Still processing... 5/10 completed (50.0%)
INFO: DVH stage: using 15 parallel workers
INFO: DVH: Still processing... 100/200 completed (50.0%)
```

#### Timeout Detected (Good - prevented hang!)
```
ERROR: Command timed out after 3600s: TotalSegmentator...
ERROR: This usually indicates a hung process or insufficient resources.
ERROR: Segmentation: task #5 (Patient123/Course1) timed out after 7200s
```

#### Slow Task Warning
```
WARNING: Segmentation: task #3 (Patient456/Course2) took 1234.5s (slow)
```

#### Actual Hang (Bad - needs investigation)
```
INFO: Segmentation stage: using 1 parallel workers
... no further logs for > 60 seconds despite heartbeat
```

### Diagnostic Commands

#### Check if pipeline is stuck
```bash
# Watch log file for heartbeat messages
tail -f ./Logs/rtpipeline.log | grep -E "(Still processing|completed|timeout)"

# Check system resources
htop  # CPU usage
nvidia-smi -l 1  # GPU usage (should show activity)

# Check for zombie processes
ps aux | grep -E "(TotalSegmentator|dcm2niix|python)"
```

#### Kill hung pipeline safely
```bash
# Find pipeline process
ps aux | grep rtpipeline

# Send SIGTERM (graceful)
kill -TERM <pid>

# If doesn't respond after 30s, force kill
kill -KILL <pid>

# Clean up any orphaned TotalSegmentator processes
pkill -f TotalSegmentator
```

## Troubleshooting

### Issue: Timeouts occurring frequently

**Diagnosis:**
- Check log for which stage is timing out
- Look at task durations in logs
- Check system resources (CPU, GPU, memory, disk I/O)

**Solutions:**
1. **Increase timeout:**
   ```bash
   --totalseg-timeout 7200  # Double the timeout
   ```

2. **Reduce parallelism (GPU memory):**
   ```bash
   --seg-workers 1 --totalseg-force-split
   ```

3. **Check for corrupted data:**
   - Examine courses that consistently timeout
   - Validate DICOM files
   - Check for extremely large volumes

### Issue: Pipeline still hangs despite timeouts

**Diagnosis:**
- Check if heartbeat messages stopped
- Look for patterns in which stage hangs
- Check system logs for hardware issues

**Possible causes:**
1. **Kernel/driver hang (GPU):**
   - Check `dmesg` for GPU errors
   - May need to reboot system
   - Update NVIDIA drivers

2. **Network filesystem hang:**
   - Check if output directory is on NFS/network share
   - May need to remount filesystem
   - Consider using local storage

3. **Docker/container issues:**
   - Check Docker logs: `docker logs <container>`
   - May hit container resource limits
   - Try running outside Docker

### Issue: False timeouts (tasks failing that should succeed)

**Diagnosis:**
- Check if tasks are genuinely slow or timing out too early
- Look at successful task durations
- Compare with similar datasets

**Solutions:**
1. **Increase timeouts appropriately:**
   ```bash
   # Large dataset example
   --totalseg-timeout 10800  # 3 hours
   --radiomics-task-timeout 1800  # 30 minutes
   ```

2. **Reduce data size (if appropriate):**
   - Use `--totalseg-roi-subset` to limit ROIs
   - Use `--radiomics-skip-roi` to skip large structures

3. **Increase resources:**
   - Reduce parallelism: `--max-workers 4`
   - Give tasks more memory
   - Use faster storage

## Implementation Details

### Files Modified

1. **rtpipeline/segmentation.py**
   - Added timeout parameter to `_run()` function
   - Automatic timeout detection based on command type
   - Timeout error handling with detailed logging

2. **rtpipeline/utils.py**
   - Added `task_timeout` parameter to `run_tasks_with_adaptive_workers()`
   - Implemented heartbeat logging (60s intervals)
   - Slow task detection (>5 min warning)
   - Per-task timeout enforcement with TimeoutError handling

3. **rtpipeline/radiomics_parallel.py**
   - Changed from `pool.map()` to `pool.imap_unordered()` for progress monitoring
   - Added per-ROI timeout constant `_TASK_TIMEOUT`
   - Implemented progress logging (every 10 tasks)
   - Better error handling for hung workers

4. **rtpipeline/cli.py**
   - Added CLI arguments: `--totalseg-timeout`, `--dcm2niix-timeout`, `--task-timeout`, `--radiomics-task-timeout`
   - Set environment variables from CLI args
   - Pass `task_timeout` to all adaptive worker calls
   - Enhanced parallelization logging

### Backward Compatibility

All timeout features are **backward compatible**:
- Default timeouts are generous (won't break existing workflows)
- Can be disabled: `--task-timeout 0` or omit the option
- Environment variables take precedence for automation
- CLI options override environment variables

### Testing Recommendations

#### Unit Testing
```python
# Test timeout enforcement
def test_subprocess_timeout():
    result = _run("sleep 10", timeout=1)
    assert result is False  # Should timeout

# Test task timeout
def test_task_timeout():
    def slow_func(x):
        time.sleep(10)
        return x
    results = run_tasks_with_adaptive_workers(
        "Test", [1], slow_func, max_workers=1, task_timeout=1
    )
    assert results[0] is None  # Should timeout
```

#### Integration Testing
```bash
# Test with short timeouts to catch hangs quickly
rtpipeline \
  --dicom-root ./test_data \
  --totalseg-timeout 60 \
  --task-timeout 300 \
  --max-workers 2

# Monitor for timeouts in logs
grep -i "timeout" ./Logs/rtpipeline.log
```

## Future Enhancements

Potential improvements for hang prevention:
1. **Global pipeline timeout** - entire pipeline must complete within X hours
2. **Watchdog thread** - separate thread monitors main pipeline health
3. **Auto-recovery** - automatically retry timed-out tasks with different settings
4. **Adaptive timeouts** - learn optimal timeouts from successful runs
5. **Telemetry** - track timeout frequencies and patterns
6. **Resource monitoring** - detect system resource exhaustion before timeout
7. **Process tree monitoring** - detect and kill orphaned subprocesses

## Summary

The pipeline now has comprehensive hang prevention:
- ✅ **Subprocess timeouts** - TotalSegmentator, dcm2niix protected
- ✅ **Task-level timeouts** - each course/ROI can timeout individually
- ✅ **Progress monitoring** - heartbeat logging every 60s
- ✅ **Slow task detection** - warnings for tasks >5 minutes
- ✅ **Graceful degradation** - continues with other tasks after timeout
- ✅ **Configurable** - all timeouts adjustable via CLI
- ✅ **Well-logged** - detailed timeout information for debugging

**Result:** Pipeline should never hang indefinitely. Timeouts will catch stuck operations and allow processing to continue with remaining data.
