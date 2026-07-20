from __future__ import annotations

import json
import os
import sys
from pathlib import Path

configfile: "config.yaml"

ROOT_DIR = Path.cwd()

CPU_COUNT = os.cpu_count() or 2
try:
    _workflow_cores = getattr(workflow, "cores", None)
except NameError:
    _workflow_cores = None
if _workflow_cores in (None, "all"):
    WORKFLOW_CORES = CPU_COUNT
else:
    try:
        WORKFLOW_CORES = int(_workflow_cores)
    except Exception:
        WORKFLOW_CORES = CPU_COUNT
WORKFLOW_CORES = max(1, min(WORKFLOW_CORES, CPU_COUNT))
def _coerce_positive_int(value):
    if value in (None, "", "auto"):
        return None
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return None
    return ivalue if ivalue > 0 else None

def _ensure_writable_dir(candidate: Path, fallback_name: str) -> Path:
    fallback = ROOT_DIR / fallback_name
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".write_test"
        probe.write_text("", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return candidate
    except OSError as exc:
        sys.stderr.write(
            f"[rtpipeline] Warning: unable to write to {candidate}: {exc}. "
            f"Using fallback {fallback}\n"
        )
        fallback.mkdir(parents=True, exist_ok=True)
        probe = fallback / ".write_test"
        probe.write_text("", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return fallback


DICOM_ROOT = (ROOT_DIR / config.get("dicom_root", "Example_data")).resolve()
OUTPUT_DIR = _ensure_writable_dir((ROOT_DIR / config.get("output_dir", "Data_Snakemake")).resolve(), "Data_Snakemake_fallback")
LOGS_DIR = _ensure_writable_dir((ROOT_DIR / config.get("logs_dir", "Logs_Snakemake")).resolve(), "Logs_Snakemake_fallback")
RESULTS_DIR = OUTPUT_DIR / "_RESULTS"


def _workflow_configfiles() -> list[Path]:
    try:
        configfiles = getattr(workflow, "configfiles", [])
    except NameError:
        configfiles = []
    paths: list[Path] = []
    for fp in configfiles or []:
        try:
            paths.append(Path(str(fp)).resolve())
        except Exception:
            paths.append(Path(str(fp)))
    return paths


def _materialize_effective_config() -> Path:
    """Persist the merged Snakemake config for subprocess stages."""
    target = LOGS_DIR / "_workflow" / "effective_config.yaml"
    try:
        import yaml

        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        return target.resolve()
    except Exception as exc:
        configfiles = _workflow_configfiles()
        fallback = configfiles[-1] if configfiles else (ROOT_DIR / "config.yaml").resolve()
        sys.stderr.write(
            f"[rtpipeline] Warning: unable to write merged config to {target}: {exc}. "
            f"Falling back to {fallback}\n"
        )
        return fallback


EFFECTIVE_CONFIGFILE = _materialize_effective_config()
os.environ.update(RTPIPELINE_CONFIGFILE=str(EFFECTIVE_CONFIGFILE))

# Ensure pip build isolation is disabled so packages like pyradiomics can reuse the conda-provided numpy.
os.environ.setdefault("PIP_NO_BUILD_ISOLATION", "1")


def _resolve_env_python(env_name: str) -> str:
    """Resolve the full path to python for a named conda/micromamba environment.

    Searches configured and common user-level prefixes. Falls back to
    sys.executable if not found.
    """
    candidates = []
    # Try CONDA_PREFIX-based sibling envs
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix).parent / env_name / "bin" / "python")
    # Try common micromamba/conda locations
    home = Path.home()
    mamba_root = os.environ.get("MAMBA_ROOT_PREFIX")
    if mamba_root:
        candidates.append(Path(mamba_root) / "envs" / env_name / "bin" / "python")
    configured_prefix = config.get("container_env_prefix")
    if configured_prefix:
        candidates.append(Path(configured_prefix) / env_name / "bin" / "python")
    for base in [
        home / "micromamba" / "envs",
        home / "miniforge3" / "envs",
        home / "miniconda3" / "envs",
    ]:
        candidates.append(base / env_name / "bin" / "python")
    for c in candidates:
        if c.exists():
            return str(c)
    return sys.executable


# Resolve python executables for configured environments
_envs_cfg = config.get("environments", {})
_ENV_MAIN = _envs_cfg.get("main", "rtpipeline")
_ENV_RADIOMICS = _envs_cfg.get("radiomics", "rtpipeline-radiomics")
os.environ.update(RTPIPELINE_RADIOMICS_ENV=_ENV_RADIOMICS)
PYTHON_MAIN = _resolve_env_python(_ENV_MAIN)
PYTHON_RADIOMICS = _resolve_env_python(_ENV_RADIOMICS)
# Derive bin dirs for PATH export in shell rules
PYTHON_MAIN_BIN = str(Path(PYTHON_MAIN).parent)
PYTHON_RADIOMICS_BIN = str(Path(PYTHON_RADIOMICS).parent)
CONTAINER_ENV_PREFIX = Path(
    config.get("container_env_prefix")
    or (Path(os.sep) / "opt" / "conda" / "envs")
)
CONTAINER_MAIN_PYTHON = str(CONTAINER_ENV_PREFIX / _ENV_MAIN / "bin" / "python")
CONTAINER_RADIOMICS_PYTHON = str(
    CONTAINER_ENV_PREFIX / _ENV_RADIOMICS / "bin" / "python"
)
CONTAINER_MAIN_BIN = str(Path(CONTAINER_MAIN_PYTHON).parent)


def _coerce_int(value, default=None):
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default

SCHEDULER_CONFIG = config.get("scheduler", {})

_reserved_cfg = _coerce_positive_int(SCHEDULER_CONFIG.get("reserved_cores"))
if _reserved_cfg is None:
    RESERVED_CORES = 1
else:
    RESERVED_CORES = max(0, _reserved_cfg)
RESERVED_CORES = min(RESERVED_CORES, max(0, WORKFLOW_CORES - 1))
AUTO_WORKER_BUDGET = max(1, WORKFLOW_CORES - RESERVED_CORES)

config_max_workers = _coerce_positive_int(config.get("max_workers"))
env_max_workers = _coerce_positive_int(os.environ.get("RTPIPELINE_MAX_WORKERS"))
requested_max_workers = config_max_workers or env_max_workers

WORKER_BUDGET = AUTO_WORKER_BUDGET
if requested_max_workers is not None:
    WORKER_BUDGET = max(1, min(WORKFLOW_CORES, requested_max_workers))

if "workers" in config or "snakemake_job_threads" in config:
    sys.stderr.write(
        "[rtpipeline] Warning: 'workers' and 'snakemake_job_threads' config keys are deprecated. "
        "Parallelism now follows 'snakemake --cores N' (uses N-1). "
        "Use 'max_workers' if you need to cap concurrency.\n"
    )

SNAKEMAKE_THREADS = WORKER_BUDGET

_parallel_courses_raw = _coerce_positive_int(SCHEDULER_CONFIG.get("parallel_courses"))
if _parallel_courses_raw is None:
    # Auto-calculate parallel courses to maximize CPU utilization (target: n_cores - 1)
    #
    # Key insight: CPU-bound stages (DVH, CT crop) need sufficient threads per job
    # to efficiently parallelize internal work (ROI processing, I/O). Too few threads
    # per course hurts throughput more than additional course parallelism helps.
    #
    # Formula: parallel_courses = min(threads // MIN_THREADS, threads // TARGET_THREADS + 1)
    # - MIN_THREADS_PER_JOB = 4 ensures each job has enough internal parallelism
    # - TARGET_THREADS = 5 balances course concurrency with per-job efficiency
    #
    # CPU utilization calculation (threads × courses = active cores):
    #   24 cores → 23 threads → min(23//4=5, 23//5+1=5) = 5 courses × 4 threads = 20 (~87%)
    #   16 cores → 15 threads → min(15//4=3, 15//5+1=4) = 3 courses × 5 threads = 15 (100%)
    #   12 cores → 11 threads → min(11//4=2, 11//5+1=3) = 2 courses × 5 threads = 10 (~91%)
    #    8 cores →  7 threads → min(7//4=1, 7//5+1=2)  = 1 course  × 7 threads =  7 (100%)
    #
    # This formula ensures:
    # 1. Each job gets at least 4 threads for efficient internal parallelization
    # 2. Multiple courses run in parallel when resources allow
    # 3. Total thread usage approaches n_cores - 1
    MIN_THREADS_PER_JOB = 4
    TARGET_THREADS_PER_JOB = 5
    max_courses_by_min = max(1, SNAKEMAKE_THREADS // MIN_THREADS_PER_JOB)
    target_courses = max(1, SNAKEMAKE_THREADS // TARGET_THREADS_PER_JOB + 1)
    DEFAULT_PARALLEL_COURSES = max(
        1, min(SNAKEMAKE_THREADS, max_courses_by_min, target_courses)
    )
else:
    # Honor the override without declaring more concurrent courses than the
    # workflow can schedule with at least one thread apiece.
    DEFAULT_PARALLEL_COURSES = max(
        1, min(SNAKEMAKE_THREADS, _parallel_courses_raw)
    )

# Log the parallelization settings for debugging
_threads_per_course = SNAKEMAKE_THREADS // DEFAULT_PARALLEL_COURSES
_active_threads = DEFAULT_PARALLEL_COURSES * _threads_per_course
_utilization_pct = round(100 * _active_threads / max(1, SNAKEMAKE_THREADS))
sys.stderr.write(
    f"[rtpipeline] Parallelization: {DEFAULT_PARALLEL_COURSES} courses × {_threads_per_course} threads = "
    f"{_active_threads}/{SNAKEMAKE_THREADS} threads ({_utilization_pct}% utilization)\n"
)


def _auto_stage_threads() -> int:
    # Always divide threads among parallel courses for proper Snakemake scheduling
    return max(1, SNAKEMAKE_THREADS // DEFAULT_PARALLEL_COURSES)


def _stage_thread_target(key: str, fallback: int | None) -> int:
    value = _coerce_positive_int(SCHEDULER_CONFIG.get(key))
    if value is not None:
        target = value
    elif fallback is not None:
        target = fallback
    else:
        target = _auto_stage_threads()
    return max(1, min(SNAKEMAKE_THREADS, target))


# Stage-level thread budgets default to a shared pool (≈ cores / DEFAULT_PARALLEL_COURSES)
DVH_RULE_THREADS = _stage_thread_target("dvh_threads_per_job", None)
RAD_RULE_THREADS = _stage_thread_target("radiomics_threads_per_job", None)
QC_RULE_THREADS = _stage_thread_target("qc_threads_per_job", None)
CROP_CT_RULE_THREADS = _stage_thread_target("crop_ct_threads_per_job", None)
ROBUSTNESS_RULE_THREADS = _stage_thread_target("robustness_threads_per_job", None)
PRIORITIZE_SHORT_COURSES = bool(SCHEDULER_CONFIG.get("prioritize_short_courses", True))

SEG_CONFIG = config.get("segmentation", {})
SEG_EXTRA_MODELS = SEG_CONFIG.get("extra_models") or []
if isinstance(SEG_EXTRA_MODELS, str):
    SEG_EXTRA_MODELS = [m.strip() for m in SEG_EXTRA_MODELS.replace(",", " ").split() if m.strip()]
else:
    SEG_EXTRA_MODELS = [str(m).strip() for m in SEG_EXTRA_MODELS if str(m).strip()]
SEG_FAST = bool(SEG_CONFIG.get("fast", False))
_seg_subset = SEG_CONFIG.get("roi_subset")
if isinstance(_seg_subset, str):
    SEG_ROI_SUBSET = _seg_subset
elif _seg_subset:
    SEG_ROI_SUBSET = ",".join(str(x) for x in _seg_subset)
else:
    SEG_ROI_SUBSET = None
_seg_workers_raw = SEG_CONFIG.get("workers")
if _seg_workers_raw is None:
    _seg_workers_raw = SEG_CONFIG.get("max_workers")
if isinstance(_seg_workers_raw, str):
    seg_token = _seg_workers_raw.strip().lower()
    if seg_token in {"", "auto"}:
        SEG_MAX_WORKERS = None
    elif seg_token == "all":
        SEG_MAX_WORKERS = WORKER_BUDGET
    else:
        SEG_MAX_WORKERS = _coerce_int(_seg_workers_raw, None)
else:
    SEG_MAX_WORKERS = _coerce_int(_seg_workers_raw, None)
if SEG_MAX_WORKERS is not None and SEG_MAX_WORKERS < 1:
    SEG_MAX_WORKERS = 1
SEG_FORCE_SEGMENTATION = bool(SEG_CONFIG.get("force", False))
SEG_DEVICE = str(SEG_CONFIG.get("device") or "gpu")
SEG_FORCE_SPLIT = bool(SEG_CONFIG.get("force_split", True))
SEG_NR_THR_RESAMP = _coerce_int(SEG_CONFIG.get("nr_threads_resample"), None)
SEG_NR_THR_SAVING = _coerce_int(SEG_CONFIG.get("nr_threads_save"), None)
SEG_NUM_PROC_PRE = _coerce_int(SEG_CONFIG.get("num_proc_preprocessing"), None)
SEG_NUM_PROC_EXPORT = _coerce_int(SEG_CONFIG.get("num_proc_export"), None)

_seg_tmp_dir = SEG_CONFIG.get("temp_dir")
if _seg_tmp_dir:
    seg_tmp_path = Path(_seg_tmp_dir)
    if not seg_tmp_path.is_absolute():
        seg_tmp_path = ROOT_DIR / seg_tmp_path
    try:
        SEG_TEMP_DIR = str(seg_tmp_path.resolve())
    except FileNotFoundError:
        SEG_TEMP_DIR = str(seg_tmp_path)
else:
    SEG_TEMP_DIR = ""

CUSTOM_MODELS_CONFIG = config.get("custom_models", {})
CUSTOM_MODELS_ENABLED = bool(CUSTOM_MODELS_CONFIG.get("enabled", True))
_custom_root = CUSTOM_MODELS_CONFIG.get("root")
if _custom_root:
    cm_path = Path(_custom_root)
    if not cm_path.is_absolute():
        cm_path = ROOT_DIR / cm_path
    CUSTOM_MODELS_ROOT = str(cm_path.resolve())
else:
    default_custom_root = ROOT_DIR / "custom_models"
    CUSTOM_MODELS_ROOT = str(default_custom_root.resolve()) if default_custom_root.exists() else ""
_custom_models_raw = CUSTOM_MODELS_CONFIG.get("models") or []
if isinstance(_custom_models_raw, str):
    CUSTOM_MODELS_SELECTED = [item.strip() for item in _custom_models_raw.replace(",", " ").split() if item.strip()]
else:
    CUSTOM_MODELS_SELECTED = [str(item).strip() for item in _custom_models_raw if str(item).strip()]
CUSTOM_MODELS_FORCE = bool(CUSTOM_MODELS_CONFIG.get("force", False))
_custom_workers_raw = CUSTOM_MODELS_CONFIG.get("workers")
if _custom_workers_raw is None:
    _custom_workers_raw = CUSTOM_MODELS_CONFIG.get("max_workers")
if isinstance(_custom_workers_raw, str):
    custom_token = _custom_workers_raw.strip().lower()
    if custom_token in {"", "auto"}:
        CUSTOM_MODELS_WORKERS = None
    elif custom_token == "all":
        CUSTOM_MODELS_WORKERS = WORKER_BUDGET
    else:
        CUSTOM_MODELS_WORKERS = _coerce_int(_custom_workers_raw, None)
else:
    CUSTOM_MODELS_WORKERS = _coerce_int(_custom_workers_raw, None)
if CUSTOM_MODELS_WORKERS is not None and CUSTOM_MODELS_WORKERS < 1:
    CUSTOM_MODELS_WORKERS = 1
CUSTOM_MODELS_PREDICT = str(CUSTOM_MODELS_CONFIG.get("nnunet_predict") or "nnUNetv2_predict")
CUSTOM_MODELS_CONDA = CUSTOM_MODELS_CONFIG.get("conda_activate")
CUSTOM_MODELS_RETAIN = bool(CUSTOM_MODELS_CONFIG.get("retain_weights", True))

RADIOMICS_CONFIG = config.get("radiomics", {})
RADIOMICS_ENABLED = bool(RADIOMICS_CONFIG.get("enabled", False))
RADIOMICS_SEQUENTIAL = bool(RADIOMICS_CONFIG.get("sequential", False))
_radiomics_params = RADIOMICS_CONFIG.get("params_file")
if _radiomics_params:
    params_path = Path(_radiomics_params)
    if not params_path.is_absolute():
        params_path = ROOT_DIR / params_path
    RADIOMICS_PARAMS = str(params_path.resolve())
else:
    RADIOMICS_PARAMS = ""

_radiomics_params_mr_cfg = RADIOMICS_CONFIG.get("mr_params_file")
if _radiomics_params_mr_cfg:
    params_mr_path = Path(_radiomics_params_mr_cfg)
    if not params_mr_path.is_absolute():
        params_mr_path = ROOT_DIR / params_mr_path
    RADIOMICS_PARAMS_MR = str(params_mr_path.resolve())
else:
    default_mr = ROOT_DIR / "rtpipeline" / "radiomics_params_mr.yaml"
    RADIOMICS_PARAMS_MR = str(default_mr.resolve()) if default_mr.exists() else ""

_radiomics_skip_cfg = RADIOMICS_CONFIG.get("skip_rois") or []
if isinstance(_radiomics_skip_cfg, str):
    RADIOMICS_SKIP_ROIS = [item.strip() for item in _radiomics_skip_cfg.replace(";", ",").split(",") if item.strip()]
else:
    RADIOMICS_SKIP_ROIS = [str(item).strip() for item in _radiomics_skip_cfg if str(item).strip()]

RADIOMICS_MAX_VOXELS = _coerce_int(RADIOMICS_CONFIG.get("max_voxels"), 15_000_000)
if RADIOMICS_MAX_VOXELS is not None and RADIOMICS_MAX_VOXELS < 1:
    RADIOMICS_MAX_VOXELS = 15_000_000
RADIOMICS_MIN_VOXELS = _coerce_int(RADIOMICS_CONFIG.get("min_voxels"), 120)
if RADIOMICS_MIN_VOXELS is not None and RADIOMICS_MIN_VOXELS < 1:
    RADIOMICS_MIN_VOXELS = 1

_custom_structures_cfg = config.get("custom_structures")
if _custom_structures_cfg:
    cs_path = Path(_custom_structures_cfg)
    if not cs_path.is_absolute():
        cs_path = ROOT_DIR / cs_path
    CUSTOM_STRUCTURES_CONFIG = str(cs_path.resolve())
else:
    default_pelvic = ROOT_DIR / "custom_structures_pelvic.yaml"
    CUSTOM_STRUCTURES_CONFIG = str(default_pelvic.resolve()) if default_pelvic.exists() else ""

AGGREGATION_CONFIG = config.get("aggregation", {})
_agg_threads_raw = AGGREGATION_CONFIG.get("threads")
if isinstance(_agg_threads_raw, str) and _agg_threads_raw.lower() == "auto":
    # Auto: use all CPUs for aggregation (I/O bound)
    AGGREGATION_THREADS = os.cpu_count() or 4
else:
    AGGREGATION_THREADS = _coerce_int(_agg_threads_raw, None)
if AGGREGATION_THREADS is not None and AGGREGATION_THREADS < 1:
    AGGREGATION_THREADS = 1

if AGGREGATION_THREADS is not None:
    AGG_THREADS_RESERVED = max(1, min(SNAKEMAKE_THREADS, AGGREGATION_THREADS))
else:
    AGG_THREADS_RESERVED = SNAKEMAKE_THREADS

ROBUSTNESS_CONFIG = config.get("radiomics_robustness", {})
ROBUSTNESS_REQUESTED = bool(ROBUSTNESS_CONFIG.get("enabled", False))
ROBUSTNESS_ENABLED = RADIOMICS_ENABLED and ROBUSTNESS_REQUESTED
if ROBUSTNESS_REQUESTED and not RADIOMICS_ENABLED:
    sys.stderr.write(
        "[rtpipeline] Warning: radiomics_robustness.enabled=true ignored because "
        "radiomics.enabled=false.\n"
    )

CT_CROPPING_CONFIG = config.get("ct_cropping", {})
CT_CROPPING_ENABLED = bool(CT_CROPPING_CONFIG.get("enabled", False))

COURSE_META_DIR = OUTPUT_DIR / "_COURSES"
COURSE_MANIFEST = COURSE_META_DIR / "manifest.json"

AGG_OUTPUTS = {
    "dvh": RESULTS_DIR / "dvh_metrics.xlsx",
    "fractions": RESULTS_DIR / "fractions.xlsx",
    "metadata": RESULTS_DIR / "case_metadata.xlsx",
    "qc": RESULTS_DIR / "qc_reports.xlsx",
}

if RADIOMICS_ENABLED:
    AGG_OUTPUTS["radiomics"] = RESULTS_DIR / "radiomics_ct.xlsx"
    AGG_OUTPUTS["radiomics_mr"] = RESULTS_DIR / "radiomics_mr.xlsx"

if ROBUSTNESS_ENABLED:
    AGG_OUTPUTS["radiomics_robustness"] = RESULTS_DIR / "radiomics_robustness_summary.xlsx"

# Snakemake resource pool for segmentation jobs: serialize GPU by default, allow fan-out when safe
def _detect_gpu_count() -> int:
    """Detect number of available GPUs."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        pass

    # Fallback: check CUDA_VISIBLE_DEVICES
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices:
        if cuda_devices.lower() == 'all':
             pass
        else:
            return len(cuda_devices.split(','))

    # Fallback: nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--list-gpus'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return len(result.stdout.strip().split('\n'))
    except Exception:
        pass
    return 0

_seg_device_token = SEG_DEVICE.strip().lower()
if SEG_MAX_WORKERS is not None:
    SEG_WORKER_POOL = max(1, SEG_MAX_WORKERS)
elif _seg_device_token in {"gpu", "cuda", "mps"}:
    gpu_count = _detect_gpu_count()
    if gpu_count > 0:
        SEG_WORKER_POOL = gpu_count
    else:
        SEG_WORKER_POOL = 1
else:
    # CPU mode: conservative default to avoid OOM (TotalSegmentator is RAM heavy)
    # Allow using up to 25% of cores/workers concurrently
    SEG_WORKER_POOL = max(1, WORKER_BUDGET // 4)

# Custom models inherit same auto logic: user setting > 1 on GPU/MPS > WORKER_BUDGET on CPU
if CUSTOM_MODELS_WORKERS is not None:
    CUSTOM_SEG_WORKER_POOL = max(1, CUSTOM_MODELS_WORKERS)
elif _seg_device_token in {"gpu", "cuda", "mps"}:
    CUSTOM_SEG_WORKER_POOL = 1
else:
    CUSTOM_SEG_WORKER_POOL = max(1, WORKER_BUDGET)

# Radiomics worker pool: serialize radiomics jobs to prevent nested parallelization explosion
# Each radiomics job spawns ProcessPoolExecutor workers → conda subprocesses
# Running 2+ jobs concurrently causes CPU thrashing and memory pressure
_rad_workers_raw = _coerce_positive_int(RADIOMICS_CONFIG.get("max_parallel_courses"))
if _rad_workers_raw is not None:
    RADIOMICS_WORKER_POOL = max(1, _rad_workers_raw)
else:
    # Default: serialize radiomics jobs to avoid nested parallelization
    # Each job gets full thread budget internally
    RADIOMICS_WORKER_POOL = 1

# Robustness uses same constraint as radiomics
ROBUSTNESS_WORKER_POOL = RADIOMICS_WORKER_POOL

workflow.global_resources.update(
    {
        "seg_workers": max(1, SEG_WORKER_POOL),
        "custom_seg_workers": max(1, CUSTOM_SEG_WORKER_POOL),
        "radiomics_workers": max(1, RADIOMICS_WORKER_POOL),
        "robustness_workers": max(1, ROBUSTNESS_WORKER_POOL),
    }
)

# Log resource pools for debugging
sys.stderr.write(
    f"[rtpipeline] Resource pools: seg_workers={SEG_WORKER_POOL}, custom_seg={CUSTOM_SEG_WORKER_POOL}, "
    f"radiomics_workers={RADIOMICS_WORKER_POOL}, robustness_workers={ROBUSTNESS_WORKER_POOL}\n"
)

def course_dir(patient: str, course: str) -> Path:
    return OUTPUT_DIR / patient / course

def course_sentinel_path(patient: str, course: str, name: str) -> Path:
    return course_dir(patient, course) / name

def _iter_course_dirs():
    for patient_dir in sorted(OUTPUT_DIR.iterdir()):
        if not patient_dir.is_dir():
            continue
        if patient_dir.name.startswith("_") or patient_dir.name.startswith("."):
            continue
        if patient_dir.name in {"Data", "Data_Snakemake_fallback", "Logs_Snakemake_fallback", "_RESULTS"}:
            continue
        for course_dir in sorted(patient_dir.iterdir()):
            if not course_dir.is_dir():
                continue
            if course_dir.name.startswith("_"):
                continue
            yield patient_dir.name, course_dir.name, course_dir

def _manifest_path() -> Path:
    return Path(checkpoints.organize_courses.get().output.manifest)

def _load_course_records() -> list[dict[str, str]]:
    manifest = _manifest_path()
    if not manifest.exists():
        return []
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return []
    records: list[dict[str, str]] = []
    for entry in data.get("courses", []):
        patient = str(entry.get("patient") or "").strip()
        course = str(entry.get("course") or "").strip()
        if not patient or not course:
            continue
        records.append(
            {
                "patient": patient,
                "course": course,
                "path": str(entry.get("path") or "").strip(),
                "complexity": int(entry.get("complexity") or 0),
            }
        )
    return records

def _per_course_sentinels(suffix: str):
    def _inner(wildcards):
        return [
            str(course_sentinel_path(rec["patient"], rec["course"], suffix))
            for rec in _load_course_records()
        ]
    return _inner

def _manifest_input(wildcards):
    return str(_manifest_path())


ORGANIZED_SENTINEL_PATTERN = str(OUTPUT_DIR / "{patient}" / "{course}" / ".organized")
SEGMENTATION_SENTINEL_PATTERN = str(OUTPUT_DIR / "{patient}" / "{course}" / ".segmentation_done")
CUSTOM_SEG_SENTINEL_PATTERN = str(OUTPUT_DIR / "{patient}" / "{course}" / ".custom_models_done")
CROP_CT_SENTINEL_PATTERN = str(OUTPUT_DIR / "{patient}" / "{course}" / ".crop_ct_done")
DVH_SENTINEL_PATTERN = str(OUTPUT_DIR / "{patient}" / "{course}" / ".dvh_done")
RADIOMICS_SENTINEL_PATTERN = str(OUTPUT_DIR / "{patient}" / "{course}" / ".radiomics_done")
RADIOMICS_ROBUSTNESS_SENTINEL_PATTERN = str(OUTPUT_DIR / "{patient}" / "{course}" / ".radiomics_robustness_done")
CROP_CT_SENTINEL_PATTERN = str(OUTPUT_DIR / "{patient}" / "{course}" / ".crop_ct_done")
QC_SENTINEL_PATTERN = str(OUTPUT_DIR / "{patient}" / "{course}" / ".qc_done")

AGGREGATE_RESULTS_INPUTS = {
    "manifest": _manifest_input,
    "dvh": _per_course_sentinels(".dvh_done"),
    "qc": _per_course_sentinels(".qc_done"),
    "custom": _per_course_sentinels(".custom_models_done"),
}

AGGREGATE_RESULTS_OUTPUTS = {
    "dvh": str(AGG_OUTPUTS["dvh"]),
    "fractions": str(AGG_OUTPUTS["fractions"]),
    "metadata": str(AGG_OUTPUTS["metadata"]),
    "qc": str(AGG_OUTPUTS["qc"]),
}

if RADIOMICS_ENABLED:
    AGGREGATE_RESULTS_INPUTS["radiomics"] = _per_course_sentinels(".radiomics_done")
    AGGREGATE_RESULTS_OUTPUTS["radiomics"] = str(AGG_OUTPUTS["radiomics"])
    AGGREGATE_RESULTS_OUTPUTS["radiomics_mr"] = str(AGG_OUTPUTS["radiomics_mr"])


rule all:
    input:
        *(str(path) for path in AGG_OUTPUTS.values())


checkpoint organize_courses:
    output:
        manifest=str(COURSE_MANIFEST)
    log:
        str(LOGS_DIR / "stage_organize.log")
    threads:
        SNAKEMAKE_THREADS
    conda:
        "envs/rtpipeline.yaml"
    params:
        python=PYTHON_MAIN,
        python_bin=PYTHON_MAIN_BIN,
        root_dir=lambda w: str(ROOT_DIR),
        configfile=str(EFFECTIVE_CONFIGFILE),
        radiomics_env=_ENV_RADIOMICS,
        dicom_root=str(DICOM_ROOT),
        output_dir=lambda w, output: str(Path(output.manifest).parents[1]),
        logs_dir=str(LOGS_DIR),
        custom_structures=CUSTOM_STRUCTURES_CONFIG,
        prioritize_short_courses=PRIORITIZE_SHORT_COURSES
    script:
        "workflow/scripts/organize_courses.py"


_seg_params_lambda = lambda w: (
    (f"--seg-temp-dir '{SEG_TEMP_DIR}' " if SEG_TEMP_DIR else "") +
    ("--totalseg-fast " if SEG_FAST else "") +
    " ".join(f"--extra-seg-models '{m}'" for m in SEG_EXTRA_MODELS) + " " +
    (f"--totalseg-roi-subset '{SEG_ROI_SUBSET}' " if SEG_ROI_SUBSET else "") +
    (f"--totalseg-device '{SEG_DEVICE}' " if SEG_DEVICE else "") +
    ("--totalseg-force-split " if SEG_FORCE_SPLIT else "--no-totalseg-force-split ") +
    (f"--totalseg-nr-thr-resamp {SEG_NR_THR_RESAMP} " if SEG_NR_THR_RESAMP else "") +
    (f"--totalseg-nr-thr-saving {SEG_NR_THR_SAVING} " if SEG_NR_THR_SAVING else "") +
    (f"--totalseg-num-proc-pre {SEG_NUM_PROC_PRE} " if SEG_NUM_PROC_PRE else "") +
    (f"--totalseg-num-proc-export {SEG_NUM_PROC_EXPORT} " if SEG_NUM_PROC_EXPORT else "") +
    (f"--seg-workers {max(1, SEG_MAX_WORKERS)} " if SEG_MAX_WORKERS else "") +
    ("--force-segmentation" if SEG_FORCE_SEGMENTATION else "")
)

if config.get("container_mode", False):
    rule segmentation_course:
        input:
            manifest=_manifest_input,
            organized=ORGANIZED_SENTINEL_PATTERN
        output:
            sentinel=SEGMENTATION_SENTINEL_PATTERN
        log:
            str(LOGS_DIR / "segmentation" / "{patient}_{course}.log")
        threads:
            4
        resources:
            seg_workers=1
        conda:
            "envs/rtpipeline.yaml"
        params:
            extra_args=_seg_params_lambda,
            python=CONTAINER_MAIN_PYTHON,
            python_bin=CONTAINER_MAIN_BIN,
            dicom_root=str(DICOM_ROOT),
            output_dir=lambda w, output: str(Path(output.sentinel).parents[2]),
            logs_dir=str(LOGS_DIR)
        shell:
            """
            set -e
            export PATH="{params.python_bin}:$PATH"
            mkdir -p $(dirname {output.sentinel})
            mkdir -p $(dirname {log})
            
            if "{params.python}" -m rtpipeline.cli \
                --dicom-root "{params.dicom_root}" \
                --outdir "{params.output_dir}" \
                --logs "{params.logs_dir}" \
                --stage segmentation \
                --course-filter "{wildcards.patient}/{wildcards.course}" \
                --max-workers {threads} \
                --manifest "{input.manifest}" \
                {params.extra_args} > {log} 2>&1; then
                echo "ok" > {output.sentinel}
            else
                echo "failed: see log" > {output.sentinel}
            fi
            """
else:
    rule segmentation_course:
        input:
            manifest=_manifest_input,
            organized=ORGANIZED_SENTINEL_PATTERN
        output:
            sentinel=SEGMENTATION_SENTINEL_PATTERN
        log:
            str(LOGS_DIR / "segmentation" / "{patient}_{course}.log")
        threads:
            4
        resources:
            seg_workers=1
        conda:
            "envs/rtpipeline.yaml"
        params:
            extra_args=_seg_params_lambda,
            python=PYTHON_MAIN,
            python_bin=PYTHON_MAIN_BIN,
            dicom_root=str(DICOM_ROOT),
            output_dir=lambda w, output: str(Path(output.sentinel).parents[2]),
            logs_dir=str(LOGS_DIR)
        shell:
            """
            set -e
            export PATH="{params.python_bin}:$PATH"
            mkdir -p $(dirname {output.sentinel})
            mkdir -p $(dirname {log})

            if "{params.python}" -m rtpipeline.cli \
                --dicom-root "{params.dicom_root}" \
                --outdir "{params.output_dir}" \
                --logs "{params.logs_dir}" \
                --stage segmentation \
                --course-filter "{wildcards.patient}/{wildcards.course}" \
                --max-workers {threads} \
                --manifest "{input.manifest}" \
                {params.extra_args} > {log} 2>&1; then
                echo "ok" > {output.sentinel}
            else
                echo "failed: see log" > {output.sentinel}
            fi
            """


_custom_params_lambda = lambda w: (
    (f"--custom-models-root '{CUSTOM_MODELS_ROOT}' " if CUSTOM_MODELS_ROOT else "") +
    " ".join(f"--custom-model '{m}'" for m in CUSTOM_MODELS_SELECTED) + " " +
    ("--force-custom-models " if CUSTOM_MODELS_FORCE else "") +
    (f"--custom-model-workers {max(1, CUSTOM_MODELS_WORKERS)} " if CUSTOM_MODELS_WORKERS else "") +
    (f"--nnunet-predict '{CUSTOM_MODELS_PREDICT}' " if CUSTOM_MODELS_PREDICT else "") +
    (f"--custom-model-conda-activate '{CUSTOM_MODELS_CONDA}' " if CUSTOM_MODELS_CONDA else "") +
    ("--purge-custom-model-weights " if not CUSTOM_MODELS_RETAIN else "")
)

if config.get("container_mode", False):
    rule segmentation_custom_models:
        input:
            manifest=_manifest_input,
            segmentation=SEGMENTATION_SENTINEL_PATTERN
        output:
            sentinel=CUSTOM_SEG_SENTINEL_PATTERN
        log:
            str(LOGS_DIR / "segmentation_custom" / "{patient}_{course}.log")
        threads:
            4
        resources:
            custom_seg_workers=1
        conda:
            "envs/rtpipeline.yaml"
        params:
            enabled=str(CUSTOM_MODELS_ENABLED),
            extra_args=_custom_params_lambda,
            python=CONTAINER_MAIN_PYTHON,
            python_bin=CONTAINER_MAIN_BIN,
            dicom_root=str(DICOM_ROOT),
            output_dir=lambda w, output: str(Path(output.sentinel).parents[2]),
            logs_dir=str(LOGS_DIR)
        shell:
            """
            set -e
            mkdir -p $(dirname {output.sentinel})
            mkdir -p $(dirname {log})

            if [ -f "{input.segmentation}" ] && grep -qi "^failed" "{input.segmentation}"; then
                echo "skipped: upstream segmentation failed" > {output.sentinel}
                exit 0
            fi
            
            if [ "{params.enabled}" = "False" ]; then
                echo "disabled" > {output.sentinel}
                exit 0
            fi
            
            export PATH="{params.python_bin}:$PATH"
            
            if "{params.python}" -m rtpipeline.cli \
                --dicom-root "{params.dicom_root}" \
                --outdir "{params.output_dir}" \
                --logs "{params.logs_dir}" \
                --stage segmentation_custom \
                --course-filter "{wildcards.patient}/{wildcards.course}" \
                --max-workers {threads} \
                --manifest "{input.manifest}" \
                {params.extra_args} > {log} 2>&1; then
                echo "ok" > {output.sentinel}
            else
                echo "failed: see log" > {output.sentinel}
            fi
            """
else:
    rule segmentation_custom_models:
        input:
            manifest=_manifest_input,
            segmentation=SEGMENTATION_SENTINEL_PATTERN
        output:
            sentinel=CUSTOM_SEG_SENTINEL_PATTERN
        log:
            str(LOGS_DIR / "segmentation_custom" / "{patient}_{course}.log")
        threads:
            4
        resources:
            custom_seg_workers=1
        conda:
            "envs/rtpipeline.yaml"
        params:
            enabled=str(CUSTOM_MODELS_ENABLED),
            extra_args=_custom_params_lambda,
            python=PYTHON_MAIN,
            python_bin=PYTHON_MAIN_BIN,
            dicom_root=str(DICOM_ROOT),
            output_dir=lambda w, output: str(Path(output.sentinel).parents[2]),
            logs_dir=str(LOGS_DIR)
        shell:
            """
            set -e
            mkdir -p $(dirname {output.sentinel})
            mkdir -p $(dirname {log})

            if [ -f "{input.segmentation}" ] && grep -qi "^failed" "{input.segmentation}"; then
                echo "skipped: upstream segmentation failed" > {output.sentinel}
                exit 0
            fi
            
            if [ "{params.enabled}" = "False" ]; then
                echo "disabled" > {output.sentinel}
                exit 0
            fi
            
            export PATH="{params.python_bin}:$PATH"
            if "{params.python}" -m rtpipeline.cli \
                --dicom-root "{params.dicom_root}" \
                --outdir "{params.output_dir}" \
                --logs "{params.logs_dir}" \
                --stage segmentation_custom \
                --course-filter "{wildcards.patient}/{wildcards.course}" \
                --max-workers {threads} \
                --manifest "{input.manifest}" \
                {params.extra_args} > {log} 2>&1; then
                echo "ok" > {output.sentinel}
            else
                echo "failed: see log" > {output.sentinel}
            fi
            """


rule crop_ct_course:
    input:
        manifest=_manifest_input,
        segmentation=SEGMENTATION_SENTINEL_PATTERN,
        custom=CUSTOM_SEG_SENTINEL_PATTERN
    output:
        sentinel=CROP_CT_SENTINEL_PATTERN
    log:
        str(LOGS_DIR / "crop_ct" / "{patient}_{course}.log")
    threads:
        CROP_CT_RULE_THREADS
    conda:
        "envs/rtpipeline.yaml"
    params:
        stage="crop_ct",
        python=PYTHON_MAIN,
        python_bin=PYTHON_MAIN_BIN,
        root_dir=lambda w: str(ROOT_DIR),
        configfile=str(EFFECTIVE_CONFIGFILE),
        radiomics_env=_ENV_RADIOMICS,
        dicom_root=str(DICOM_ROOT),
        output_dir=lambda w, output: str(Path(output.sentinel).parents[2]),
        logs_dir=str(LOGS_DIR),
        custom_structures=""
    script:
        "workflow/scripts/run_course_stage.py"





rule dvh_course:
    input:
        manifest=_manifest_input,
        segmentation=SEGMENTATION_SENTINEL_PATTERN,
        custom=CUSTOM_SEG_SENTINEL_PATTERN,
        crop=CROP_CT_SENTINEL_PATTERN
    output:
        sentinel=DVH_SENTINEL_PATTERN
    log:
        str(LOGS_DIR / "dvh" / "{patient}_{course}.log")
    threads:
        DVH_RULE_THREADS
    conda:
        "envs/rtpipeline.yaml"
    params:
        stage="dvh",
        python=PYTHON_MAIN,
        python_bin=PYTHON_MAIN_BIN,
        root_dir=lambda w: str(ROOT_DIR),
        configfile=str(EFFECTIVE_CONFIGFILE),
        radiomics_env=_ENV_RADIOMICS,
        dicom_root=str(DICOM_ROOT),
        output_dir=lambda w, output: str(Path(output.sentinel).parents[2]),
        logs_dir=str(LOGS_DIR),
        custom_structures=CUSTOM_STRUCTURES_CONFIG
    script:
        "workflow/scripts/run_course_stage.py"


_radiomics_input_dict = {
    "manifest": _manifest_input,
    "segmentation": SEGMENTATION_SENTINEL_PATTERN,
    "custom": CUSTOM_SEG_SENTINEL_PATTERN,
    "crop": CROP_CT_SENTINEL_PATTERN
}

_radiomics_params_lambda = lambda w: (
    ("--sequential-radiomics " if RADIOMICS_SEQUENTIAL else "") +
    (f"--radiomics-params '{RADIOMICS_PARAMS}' " if RADIOMICS_PARAMS else "") +
    (f"--radiomics-params-mr '{RADIOMICS_PARAMS_MR}' " if RADIOMICS_PARAMS_MR else "") +
    (f"--radiomics-max-voxels {RADIOMICS_MAX_VOXELS} " if RADIOMICS_MAX_VOXELS else "") +
    (f"--radiomics-min-voxels {RADIOMICS_MIN_VOXELS} " if RADIOMICS_MIN_VOXELS else "") +
    " ".join(f"--radiomics-skip-roi '{roi}'" for roi in RADIOMICS_SKIP_ROIS) + " " +
    (f"--custom-structures '{CUSTOM_STRUCTURES_CONFIG}'" if CUSTOM_STRUCTURES_CONFIG else "")
)

if config.get("container_mode", False):
    rule radiomics_course:
        input:
            **_radiomics_input_dict
        output:
            sentinel=RADIOMICS_SENTINEL_PATTERN
        log:
            str(LOGS_DIR / "radiomics" / "{patient}_{course}.log")
        threads:
            RAD_RULE_THREADS
        resources:
            radiomics_workers=1  # Serialize radiomics jobs to avoid nested parallelization
        conda:
            "envs/rtpipeline-radiomics.yaml"
        params:
            extra_args=_radiomics_params_lambda,
            python=CONTAINER_RADIOMICS_PYTHON,
            dicom_root=str(DICOM_ROOT),
            output_dir=lambda w, output: str(Path(output.sentinel).parents[2]),
            logs_dir=str(LOGS_DIR),
            workflow_threads=SNAKEMAKE_THREADS
        shell:
            """
            set -e
            # Export worker budget for subprocess coordination
            export RTPIPELINE_MAX_WORKERS={threads}
            export RTPIPELINE_RADIOMICS_THREAD_LIMIT=1
            # BLAS thread limits to prevent internal parallelism explosion
            export OMP_NUM_THREADS=1
            export OPENBLAS_NUM_THREADS=1
            export MKL_NUM_THREADS=1
            mkdir -p $(dirname {log})
            echo "DEBUG: threads={threads} SNAKEMAKE_THREADS={params.workflow_threads}" >> {log}
            mkdir -p $(dirname {output.sentinel})

            if [ -f "{input.segmentation}" ] && grep -qi "^failed" "{input.segmentation}"; then
                echo "skipped: upstream segmentation failed" > {output.sentinel}
                exit 0
            fi
            
            if "{params.python}" -m rtpipeline.cli \
                --dicom-root "{params.dicom_root}" \
                --outdir "{params.output_dir}" \
                --logs "{params.logs_dir}" \
                --stage radiomics \
                --course-filter "{wildcards.patient}/{wildcards.course}" \
                --max-workers {threads} \
                --manifest "{input.manifest}" \
                {params.extra_args} >> {log} 2>&1; then
                echo "ok" > {output.sentinel}
            else
                echo "failed: see log" > {output.sentinel}
            fi
            """
else:
    rule radiomics_course:
        input:
            **_radiomics_input_dict
        output:
            sentinel=RADIOMICS_SENTINEL_PATTERN
        log:
            str(LOGS_DIR / "radiomics" / "{patient}_{course}.log")
        threads:
            RAD_RULE_THREADS
        resources:
            radiomics_workers=1  # Serialize radiomics jobs to avoid nested parallelization
        conda:
            "envs/rtpipeline-radiomics.yaml"
        params:
            extra_args=_radiomics_params_lambda,
            python=PYTHON_RADIOMICS,
            python_bin=PYTHON_RADIOMICS_BIN,
            dicom_root=str(DICOM_ROOT),
            output_dir=lambda w, output: str(Path(output.sentinel).parents[2]),
            logs_dir=str(LOGS_DIR),
            workflow_threads=SNAKEMAKE_THREADS
        shell:
            """
            set -e
            # Export worker budget for subprocess coordination
            export RTPIPELINE_MAX_WORKERS={threads}
            export RTPIPELINE_RADIOMICS_THREAD_LIMIT=1
            # BLAS thread limits to prevent internal parallelism explosion
            export OMP_NUM_THREADS=1
            export OPENBLAS_NUM_THREADS=1
            export MKL_NUM_THREADS=1
            mkdir -p $(dirname {log})
            echo "DEBUG: threads={threads} SNAKEMAKE_THREADS={params.workflow_threads}" >> {log}
            mkdir -p $(dirname {output.sentinel})

            if [ -f "{input.segmentation}" ] && grep -qi "^failed" "{input.segmentation}"; then
                echo "skipped: upstream segmentation failed" > {output.sentinel}
                exit 0
            fi
            
            export PATH="{params.python_bin}:$PATH"
            if "{params.python}" -m rtpipeline.cli \
                --dicom-root "{params.dicom_root}" \
                --outdir "{params.output_dir}" \
                --logs "{params.logs_dir}" \
                --stage radiomics \
                --course-filter "{wildcards.patient}/{wildcards.course}" \
                --max-workers {threads} \
                --manifest "{input.manifest}" \
                {params.extra_args} >> {log} 2>&1; then
                echo "ok" > {output.sentinel}
            else
                echo "failed: see log" > {output.sentinel}
            fi
            """


rule qc_course:
    input:
        manifest=_manifest_input,
        segmentation=SEGMENTATION_SENTINEL_PATTERN,
        crop=CROP_CT_SENTINEL_PATTERN
    output:
        sentinel=QC_SENTINEL_PATTERN
    log:
        str(LOGS_DIR / "qc" / "{patient}_{course}.log")
    threads:
        1
    conda:
        "envs/rtpipeline.yaml"
    params:
        stage="qc",
        python=PYTHON_MAIN,
        python_bin=PYTHON_MAIN_BIN,
        root_dir=lambda w: str(ROOT_DIR),
        configfile=str(EFFECTIVE_CONFIGFILE),
        radiomics_env=_ENV_RADIOMICS,
        dicom_root=str(DICOM_ROOT),
        output_dir=lambda w, output: str(Path(output.sentinel).parents[2]),
        logs_dir=str(LOGS_DIR),
        custom_structures=""
    script:
        "workflow/scripts/run_course_stage.py"



_robustness_input_dict = {
    "manifest": _manifest_input,
    "radiomics": RADIOMICS_SENTINEL_PATTERN
}

if config.get("container_mode", False):
    rule radiomics_robustness_course:
        input:
            **_robustness_input_dict
        output:
            sentinel=RADIOMICS_ROBUSTNESS_SENTINEL_PATTERN
        log:
            str(LOGS_DIR / "radiomics_robustness" / "{patient}_{course}.log")
        threads:
            ROBUSTNESS_RULE_THREADS
        resources:
            robustness_workers=1  # Serialize robustness jobs like radiomics
        conda:
            "envs/rtpipeline-radiomics.yaml"
        params:
            enabled=str(ROBUSTNESS_ENABLED),
            config=str(EFFECTIVE_CONFIGFILE),
            python=CONTAINER_RADIOMICS_PYTHON,
            course_dir=lambda w, output: str(Path(output.sentinel).parent),
            parquet=lambda w, output: str(
                Path(output.sentinel).parent / "radiomics_robustness_ct.parquet"
            )
        shell:
            """
            mkdir -p $(dirname {output.sentinel})
            mkdir -p $(dirname {log})

            if [ "{params.enabled}" = "False" ]; then
                echo "disabled" > {output.sentinel}
                exit 0
            fi

            if [ -f "{input.radiomics}" ] && grep -qi "^failed" "{input.radiomics}"; then
                rm -f {output.sentinel}
                echo "Radiomics robustness cannot run because upstream radiomics failed" >&2
                exit 1
            fi

            # Export worker budget for subprocess coordination
            export RTPIPELINE_MAX_WORKERS={threads}
            export RTPIPELINE_RADIOMICS_THREAD_LIMIT=1
            # BLAS thread limits to prevent internal parallelism explosion
            export OMP_NUM_THREADS=1
            export MKL_NUM_THREADS=1
            export OPENBLAS_NUM_THREADS=1

            if "{params.python}" -m rtpipeline.cli radiomics-robustness \
                --course-dir "{params.course_dir}" \
                --config "{params.config}" \
                --output "{params.parquet}" \
                --max-workers {threads} > {log} 2>&1; then
                echo "ok" > {output.sentinel}
            else
                rm -f {output.sentinel}
                echo "Radiomics robustness failed; see {log}" >&2
                exit 1
            fi
            """
else:
    rule radiomics_robustness_course:
        input:
            **_robustness_input_dict
        output:
            sentinel=RADIOMICS_ROBUSTNESS_SENTINEL_PATTERN
        log:
            str(LOGS_DIR / "radiomics_robustness" / "{patient}_{course}.log")
        threads:
            ROBUSTNESS_RULE_THREADS
        resources:
            robustness_workers=1  # Serialize robustness jobs like radiomics
        conda:
            "envs/rtpipeline-radiomics.yaml"
        params:
            enabled=str(ROBUSTNESS_ENABLED),
            config=str(EFFECTIVE_CONFIGFILE),
            python=PYTHON_RADIOMICS,
            python_bin=PYTHON_RADIOMICS_BIN,
            course_dir=lambda w, output: str(Path(output.sentinel).parent),
            parquet=lambda w, output: str(
                Path(output.sentinel).parent / "radiomics_robustness_ct.parquet"
            )
        shell:
            """
            mkdir -p $(dirname {output.sentinel})
            mkdir -p $(dirname {log})

            if [ "{params.enabled}" = "False" ]; then
                echo "disabled" > {output.sentinel}
                exit 0
            fi

            if [ -f "{input.radiomics}" ] && grep -qi "^failed" "{input.radiomics}"; then
                rm -f {output.sentinel}
                echo "Radiomics robustness cannot run because upstream radiomics failed" >&2
                exit 1
            fi

            # Export worker budget for subprocess coordination
            export RTPIPELINE_MAX_WORKERS={threads}
            export RTPIPELINE_RADIOMICS_THREAD_LIMIT=1
            # BLAS thread limits to prevent internal parallelism explosion
            export OMP_NUM_THREADS=1
            export MKL_NUM_THREADS=1
            export OPENBLAS_NUM_THREADS=1

            export PATH="{params.python_bin}:$PATH"
            if "{params.python}" -m rtpipeline.cli radiomics-robustness \
                --course-dir "{params.course_dir}" \
                --config "{params.config}" \
                --output "{params.parquet}" \
                --max-workers {threads} > {log} 2>&1; then
                echo "ok" > {output.sentinel}
            else
                rm -f {output.sentinel}
                echo "Radiomics robustness failed; see {log}" >&2
                exit 1
            fi
            """



rule aggregate_radiomics_robustness:
    input:
        manifest=_manifest_input,
        robustness=_per_course_sentinels(".radiomics_robustness_done")
    output:
        summary=str(AGG_OUTPUTS.get("radiomics_robustness", RESULTS_DIR / "radiomics_robustness_summary.xlsx"))
    log:
        str(LOGS_DIR / "aggregate_radiomics_robustness.log")
    threads:
        AGG_THREADS_RESERVED
    params:
        output_dir=lambda w, output: str(Path(output.summary).parent.parent),
        root_dir=lambda w, input: str(ROOT_DIR),
        configfile=str(EFFECTIVE_CONFIGFILE),
        robustness_enabled=ROBUSTNESS_ENABLED,
        python=PYTHON_MAIN,
        python_bin=PYTHON_MAIN_BIN
    conda:
        "envs/rtpipeline.yaml"
    shell:
        """
        set -euo pipefail
        mkdir -p $(dirname {log})
        mkdir -p $(dirname {output.summary})

        export PATH="{params.python_bin}:$PATH"
        if [ "{params.robustness_enabled}" != "True" ]; then
            # Create empty output if robustness is disabled
            "{params.python}" -c "import pandas as pd; pd.DataFrame().to_excel('{output.summary}', index=False)"
            echo "Robustness disabled - empty output created" > {log}
            exit 0
        fi

        # Every successful sentinel must bind to a readable course parquet.
        PARQUET_FILES=()
        while IFS= read -r -d '' sentinel; do
            if head -1 "$sentinel" 2>/dev/null | grep -q "^ok"; then
                course_dir=$(dirname "$sentinel")
                pf="$course_dir/radiomics_robustness_ct.parquet"
                if [ -f "$pf" ]; then
                    PARQUET_FILES+=("$pf")
                else
                    echo "Successful robustness sentinel has no parquet: $sentinel" > {log}
                    exit 1
                fi
            else
                echo "Non-success robustness sentinel encountered: $sentinel" > {log}
                exit 1
            fi
        done < <(find {params.output_dir} -name ".radiomics_robustness_done" -type f -print0 2>/dev/null)

        if [ "${{#PARQUET_FILES[@]}}" -eq 0 ]; then
            echo "Robustness is enabled but no complete course parquets were found" > {log}
            exit 1
        fi

        # Run the aggregation CLI command with the conda environment's Python
        PYTHONPATH="{params.root_dir}:${{PYTHONPATH:-}}" "{params.python}" -m rtpipeline.cli radiomics-robustness-aggregate \
            --inputs "${{PARQUET_FILES[@]}}" \
            --output {output.summary} \
            --config {params.configfile} \
            > {log} 2>&1
        """



rule aggregate_results:
    input:
        **AGGREGATE_RESULTS_INPUTS
    output:
        **AGGREGATE_RESULTS_OUTPUTS
    log:
        str(LOGS_DIR / "aggregate_results.log")
    threads:
        AGG_THREADS_RESERVED
    conda:
        "envs/rtpipeline.yaml"
    params:
        output_dir=lambda w, input: str(Path(input.manifest).parents[1]),
        results_dir=lambda w, output: str(Path(output.dvh).parent),
        radiomics_enabled=RADIOMICS_ENABLED,
        worker_budget=WORKER_BUDGET,
        auto_worker_budget=AUTO_WORKER_BUDGET,
        aggregation_threads=AGGREGATION_THREADS or 0
    script:
        "workflow/scripts/aggregate_results.py"


rule generate_report:
    input:
        *(str(path) for path in AGG_OUTPUTS.values())
    output:
        "pipeline_report.html"
    log:
        str(LOGS_DIR / "report.log")
    run:
        shell(f"snakemake --report {{output}} --configfile {EFFECTIVE_CONFIGFILE}")
