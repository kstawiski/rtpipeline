# RT Pipeline Parallelism Analysis - Documentation Index

## Overview

This repository contains a comprehensive analysis of how the rtpipeline implements parallelism across its eight processing stages. The analysis covers orchestration, per-stage strategies, adaptive resource allocation, and configuration defaults.

## Main Analysis Document

**File**: `PARALLELISM_COMPREHENSIVE_ANALYSIS.md` (1024 lines, 35KB)

This is the primary analysis document containing:

### Sections:

1. **How run_pipeline.sh Orchestrates Snakemake Execution** (1.1-1.4)
   - Global resource configuration
   - Workers calculation and bounds
   - Checkpoint-based dynamic DAG
   - Per-course rule patterns

2. **Pipeline Stage Parallelism Implementation** (2.1-2.8)
   - ORGANIZE: Sequential per-dataset, threaded file I/O
   - SEGMENTATION: GPU-limited, 3-level parallelism (inter-course, intra-course, task-level)
   - CUSTOM MODELS: Similar GPU controls as segmentation
   - DVH: I/O-bound, ROI-level ThreadPoolExecutor parallelism
   - RADIOMICS: CPU-intensive, ProcessPoolExecutor to avoid PyRadiomics thread issues
   - QC: I/O-bound report generation
   - RADIOMICS ROBUSTNESS: Perturbation-based, CPU-intensive
   - AGGREGATION: Final consolidation, parallel Excel file I/O

3. **Executor Configuration and Worker Limits** (3.1-3.3)
   - Default worker calculations (CPU_count - 2, bounded 4-32)
   - Per-stage configuration table
   - Snakemake resource limit mechanism

4. **Adaptive Logic for Resource Allocation** (4.1-4.3)
   - Memory-aware task backoff algorithm
   - Per-stage adaptive worker selection
   - TotalSegmentator thread management

5. **Configuration Defaults** (5.1-5.3)
   - config.yaml defaults with explanations
   - CLI argument defaults
   - Environment variable configuration table

6. **Summary: Parallelism Strategy by Stage** (6)
   - Quick reference table
   - Inter-course vs. intra-course strategies
   - Worker pools and timeouts

7. **Key Design Principles** (7)
   - Conservative GPU, aggressive CPU
   - Process-based for PyRadiomics
   - Memory-aware retry logic
   - Fine-grained control for experts
   - Sensible defaults

8. **Performance Tuning Recommendations** (8)
   - Single GPU scenario
   - Dual GPU scenario
   - CPU-only scenario
   - NFS/slow storage scenario

9. **Monitoring and Debugging** (9)
   - Log file locations
   - Key log messages
   - Hang debugging
   - Memory issue diagnosis

10. **Conclusion** (10)
    - Summary of strategies
    - Key design features

---

## Key Code Files Referenced

### Orchestration & Control Flow
- **Snakefile** (37KB) - Main workflow DAG, resource limits
- **rtpipeline/cli.py** (1192 lines) - CLI argument parsing, stage orchestration, adaptive worker selection
- **rtpipeline/config.py** - PipelineConfig dataclass with defaults

### Resource Management
- **rtpipeline/utils.py** - `run_tasks_with_adaptive_workers()` function implementing memory-aware backoff
- **config.yaml** - Configuration file with parallelism defaults

### Stage Implementations
- **rtpipeline/segmentation.py** - TotalSegmentator execution, thread limiting
- **rtpipeline/dvh.py** - DVH computation with ROI-level parallelism
- **rtpipeline/radiomics_parallel.py** - ProcessPoolExecutor-based radiomics with thread safety
- **rtpipeline/radiomics_robustness.py** - Perturbation-based robustness analysis
- **rtpipeline/quality_control.py** - QC report generation
- **rtpipeline/organize.py** - Course organization with DICOM processing

---

## Quick Facts

### Parallelism Levels
- **Level 1 (Inter-course)**: Snakemake manages concurrent courses (1-32 depending on stage)
- **Level 2 (Intra-course)**: ThreadPoolExecutor or ProcessPoolExecutor for per-ROI/per-file tasks
- **Level 3 (Task-level)**: Built-in parallelism in TotalSegmentator, nnUNet, OpenMP/BLAS

### Default Worker Counts
- Global: `CPU_count - 2` (min 4, max 32)
- Segmentation inter-course: 1 (GPU safety)
- DVH intra-course: `CPU_count // 2`
- Radiomics inter-course: `CPU_count` (full pool)
- Radiomics intra-course: `CPU_count - 1` (process-based)

### Key Adaptive Features
1. **Memory-aware backoff**: Halves workers on MemoryError, retries
2. **Device-aware defaults**: Different settings for GPU vs. CPU mode
3. **Thread limiting**: Prevents OpenMP/BLAS explosion via environment variables
4. **Timeout handling**: 1-3600 seconds depending on stage

### Conservative vs. Aggressive Defaults
- **Conservative** (GPU stages): seg_workers=1, custom_models_workers=1
- **Aggressive** (I/O stages): radiomics uses full CPU count - 1
- **Rationale**: GPU memory is the bottleneck; CPU is generally sufficient

---

## How to Use This Analysis

### For Understanding Architecture
1. Read the Executive Summary (above)
2. Review "Three-Level Parallelism Architecture" section
3. Look at the stage-specific implementation sections (2.1-2.8)

### For Performance Tuning
1. Check "Performance Tuning Recommendations" (Section 8)
2. Review relevant configuration options in "Configuration Defaults" (Section 5)
3. Use "Monitoring and Debugging" (Section 9) if issues arise

### For Modifying Pipeline
1. Understand Snakemake resource limits (Section 3.3)
2. Review adaptive worker selection logic (Section 4.2)
3. Check memory-aware backoff algorithm (Section 4.1)
4. Modify config.yaml or CLI arguments accordingly

### For Debugging
1. Check log messages in "Monitoring and Debugging" (Section 9)
2. Review timeout settings for each stage (Section 6 table)
3. Use memory error patterns for diagnosis (Section 4.1)
4. Increase verbosity with `--verbose` flag

---

## Configuration Quick Reference

### Minimal Configuration (defaults)
```yaml
workers: auto              # CPU_count - 2
segmentation:
  workers: 1             # 1 course at a time
radiomics:
  thread_limit: null     # Auto
```

### Medium-Scale System (16 cores, 32GB RAM)
```yaml
workers: 8
segmentation:
  workers: 1
  threads_per_worker: 4
  nr_threads_resample: 4
  nr_threads_save: 4
radiomics:
  thread_limit: 4
custom_models:
  workers: 1
```

### Large-Scale System (64 cores, 256GB RAM)
```yaml
workers: 32
segmentation:
  workers: 2
  threads_per_worker: 8
  nr_threads_resample: 8
  nr_threads_save: 8
custom_models:
  workers: 2
radiomics:
  thread_limit: 8
```

---

## Related Documentation

The repository also contains:
- **PARALLELIZATION_QUICK_REFERENCE.md** - Command-line examples
- **PIPELINE_ARCHITECTURE.md** - General pipeline design
- **HANG_PREVENTION.md** - Debugging hung processes
- **OPTIMIZATION_SUMMARY.md** - General optimization notes

---

## Generated: 2025-11-16
**Analysis Depth**: Comprehensive technical walkthrough with code citations
**Coverage**: All 8 pipeline stages + orchestration layer
**Content**: 1024 lines, 10 major sections, code examples throughout

---

## Questions to Consider

After reading this analysis, you should be able to answer:

1. **Architecture**: How does Snakemake control inter-course parallelism?
   - Answer: Via resource limits (`seg_workers`, `custom_seg_workers`)

2. **Parallelism Model**: Why does radiomics use ProcessPoolExecutor instead of ThreadPoolExecutor?
   - Answer: PyRadiomics uses OpenMP internally, causing segmentation faults with threads

3. **Memory Safety**: What happens when a task runs out of memory?
   - Answer: Adaptive backoff halves workers and retries the failed task

4. **GPU Safety**: Why are segmentation workers default to 1?
   - Answer: GPU memory exhaustion risk; deep learning inference is very memory-intensive

5. **Configuration**: What does `radiomics.thread_limit` do?
   - Answer: Limits OpenMP/BLAS threads per radiomics worker process to prevent oversubscription

6. **Timeouts**: What is the default timeout for radiomics per-ROI extraction?
   - Answer: 600 seconds (10 minutes) per ROI

7. **Adaptive Logic**: How does the pipeline determine optimal worker count?
   - Answer: Auto = `CPU_count - 1` with min/max bounds based on stage characteristics

---

