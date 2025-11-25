# Parallelization & Resource Guide

The pipeline now has a single, predictable parallelism model. Snakemake decides how many jobs may run at once, while each stage of `rtpipeline` fans out internally using the *same* worker budget. This document replaces the older parallelism notes and reflects the current implementation (Snakefile + `rtpipeline.cli`).

---

## 1. How Throughput Is Controlled

1. Run Snakemake with the number of logical cores you want to devote to the pipeline:
   ```bash
   snakemake --cores 16 --use-conda --rerun-incomplete
   ```
2. The scheduler reserves `scheduler.reserved_cores` (default 1) for the OS / background I/O. Worker budget = `cores - reserved`.
3. Each rule declares Snakemake `threads:` either to that full budget (GPU segmentation) or to a stage-specific cap (DVH, radiomics, QC).
4. Every CLI invocation receives `--max-workers <threads>`, so `PipelineConfig.effective_workers()` matches Snakemake’s allocation.
5. You can cap the global budget at any time:
   - `config.yaml`: `max_workers: 8`
   - CLI: `snakemake ... --config max_workers=8`
   - Environment: `export RTPIPELINE_MAX_WORKERS=8`

Result: All CPU-bound stages use up to `min(max_workers, cores - reserved)` workers. Only GPU segmentation is serialized by default unless you explicitly raise `segmentation.max_workers`.

---

## 2. Stage-by-Stage Behavior

| Stage (rule)           | Snakemake `threads:`                             | Snakemake resource gate              | Internal fan-out                    | Notes |
|------------------------|--------------------------------------------------|--------------------------------------|-------------------------------------|-------|
| `organize_courses`     | `cores - reserved_cores`                         | —                                    | Threaded file I/O                   | Runs once per workflow; prepares manifest |
| `segmentation_course`  | `cores - reserved_cores`                         | `seg_workers=1` on GPU (auto)        | TotalSegmentator uses GPU + tuned CPU threads | Serialized for single GPU/MPS. CPU mode fans out to worker budget. |
| `segmentation_custom`  | `cores - reserved_cores`                         | `custom_seg_workers` (auto: 1 on GPU, budget on CPU) | nnUNet weights staged per-course | Inherits same auto logic as primary segmentation. |
| `dvh_course`           | `scheduler.dvh_threads_per_job` (default 4, clipped to budget) | — | ROI-level ThreadPool (up to job threads) | Starts per course when segmentation completes. |
| `radiomics_course`     | `scheduler.radiomics_threads_per_job` (default 6, clipped) | — | ProcessPool (PyRadiomics)           | Honors `radiomics.sequential` flag. |
| `radiomics_robustness_course` | `scheduler.radiomics_threads_per_job`     | —                                    | CLI subcommand per course           | Follows same CPU cap as radiomics. |
| `qc_course`            | `scheduler.qc_threads_per_job` (default 2, clipped) | —                                    | ThreadPool                          | Lightweight I/O, still respects worker cap. |
| Aggregation rules      | `aggregation.threads` (`auto` ⇒ CPU count)       | —                                    | ThreadPool for Excel/pandas work    | Waits for per-course sentinels before collating. |

Snakemake automatically clips each `threads:` value to the available worker budget (`cores - reserved_cores - throttled by max_workers`). **Per-course flow:** once a course reaches `.segmentation_done`, downstream rules for that *same* course may start even if other courses are still segmenting. This keeps the CPU saturated while the GPU processes the next course.

---

## 3. Scheduler Controls & Worker Pools

Snakemake uses custom resources named `seg_workers` and `custom_seg_workers` to prevent multiple GPU-heavy jobs from running at the same time. A shared `scheduler` block in `config.yaml` controls CPU reservations, per-stage thread caps, and course ordering:

```yaml
scheduler:
  reserved_cores: 1                # Keep a core free for the OS / I/O
  dvh_threads_per_job: 4           # Snakemake threads per DVH job
  radiomics_threads_per_job: 6     # Snakemake threads per radiomics job
  qc_threads_per_job: 2            # Snakemake threads per QC job
  prioritize_short_courses: true   # Sort manifest by estimated DICOM count
```

- **Reserved cores** trims the global worker budget to `cores - reserved`. Set it to `0` if you want the older `(cores - 1)` behavior or increase it if your workstation becomes sluggish.
- **Stage thread caps** allow Snakemake to run multiple per-course jobs concurrently instead of always giving them the entire CPU budget. The numbers above are defaults; they are still clipped to the current worker budget so you can never over-commit.
- **Course prioritization** counts DICOM slices per course during `organize` and sorts the manifest so short cases finish first. Disable it by setting `prioritize_short_courses: false` if you prefer alphabetical order.

Both segmentation pools share the same auto-detection logic based on the `segmentation.device` setting.

### Primary Segmentation (`seg_workers`)

`SEG_MAX_WORKERS` is derived from `segmentation.max_workers` (or the legacy `segmentation.workers`) in `config.yaml`:

```python
_seg_device_token = SEG_DEVICE.strip().lower()
if SEG_MAX_WORKERS is not None:
    SEG_WORKER_POOL = max(1, SEG_MAX_WORKERS)
elif _seg_device_token in {"gpu", "cuda", "mps"}:
    SEG_WORKER_POOL = 1  # serialize jobs
else:
    SEG_WORKER_POOL = max(1, WORKER_BUDGET)  # CPU mode can fan out
```

### Custom Model Segmentation (`custom_seg_workers`)

Custom models now inherit the same auto logic. `CUSTOM_MODELS_WORKERS` is derived from `custom_models.max_workers` (or legacy `custom_models.workers`):

```python
# Custom models inherit same auto logic: user setting > 1 on GPU/MPS > WORKER_BUDGET on CPU
if CUSTOM_MODELS_WORKERS is not None:
    CUSTOM_SEG_WORKER_POOL = max(1, CUSTOM_MODELS_WORKERS)
elif _seg_device_token in {"gpu", "cuda", "mps"}:
    CUSTOM_SEG_WORKER_POOL = 1  # serialize on GPU/MPS
else:
    CUSTOM_SEG_WORKER_POOL = max(1, WORKER_BUDGET)  # CPU mode can fan out
```

**Key behaviors:**
- Both segmentation stages declare `threads: cores - 1` to inform Snakemake they consume the full CPU budget during execution, preventing CPU starvation from concurrent jobs.
- The resource gate (not `threads:`) controls how many segmentation jobs run simultaneously.
- *Multi-GPU hosts*: set `segmentation.max_workers` and `custom_models.max_workers` to the number of GPUs you want to use concurrently. Each job still receives the full `--max-workers` CPU budget for preprocessing/postprocessing.
- *CPU-only hosts*: set `segmentation.device: cpu`. Both primary and custom segmentation will automatically fan out to the worker budget, just like other stages.
- *Mixed scenarios*: if you run custom models with a different device than primary segmentation (not recommended), the auto logic still bases pool size on the `segmentation.device` setting.

---

## 4. Configuration Examples

### Default (single GPU, best for most users)
```yaml
max_workers: null          # auto → cores - 1
scheduler:
  reserved_cores: 1
  dvh_threads_per_job: 4
  radiomics_threads_per_job: 6
  qc_threads_per_job: 2
  prioritize_short_courses: true
segmentation:
  device: gpu              # default
  max_workers: null        # auto → 1 on GPU, worker budget on CPU
custom_models:
  max_workers: null        # auto → inherits same logic as segmentation.max_workers
radiomics:
  sequential: false        # process pool enabled
```

### Multi-GPU workstation
```yaml
max_workers: 24            # cap CPU usage if desired
scheduler:
  reserved_cores: 2
  dvh_threads_per_job: 6
  radiomics_threads_per_job: 8
  qc_threads_per_job: 2
  prioritize_short_courses: true
segmentation:
  device: gpu
  workers: 2               # run two courses at a time
custom_models:
  workers: 2               # match GPU count
radiomics:
  sequential: false
```

### CPU-only node / CI host
```yaml
max_workers: 8
scheduler:
  reserved_cores: 1
  dvh_threads_per_job: 3
  radiomics_threads_per_job: 4
  qc_threads_per_job: 2
  prioritize_short_courses: true
segmentation:
  device: cpu
  workers: null            # auto → worker budget (fan-out allowed)
custom_models:
  workers: null            # auto → also fans out to worker budget on CPU
radiomics:
  sequential: false        # still parallel, but limited to 7 workers here
```

---

## 5. Troubleshooting & Tuning Tips

- **GPU OOM or CUDA context errors**: you probably set `segmentation.max_workers` > number of GPUs. Drop it back to 1 (default) or match your GPU count.
- **Disk thrash / NFS saturation**: lower `max_workers` to 2–4 so fewer per-course jobs hammer storage simultaneously.
- **Desktop becomes sluggish**: increase `scheduler.reserved_cores` to keep spare CPU capacity for the OS / window manager.
- **Radiomics taking forever**: ensure `radiomics.sequential` is `false` (default) and that you did not override `RTPIPELINE_RADIOMICS_SEQUENTIAL=1`.
- **Quick smoke test**: `./test.sh` runs against `Example_data/` using the defaults (`--cores all`) and is the fastest way to validate scheduler changes.
- **Dry-run sanity check**: `snakemake --cores 8 -n` prints the DAG with resource usage so you can confirm `seg_workers` is 1 on GPU systems.

---

## 6. Key Takeaways

1. Run Snakemake with the core count you can actually spare.
2. Let the workflow derive `cores - 1`; override only when you need to throttle.
3. GPU segmentation is serialized automatically unless you explicitly raise `segmentation.max_workers`.
4. Once a course clears segmentation, downstream CPU-heavy stages start immediately, so the machine stays busy end-to-end.
5. All other documentation about "CPU_count - 2" or per-stage worker flags is obsolete and has been removed.
