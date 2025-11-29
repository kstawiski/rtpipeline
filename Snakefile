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


def _rt_env():
    env = os.environ.copy()
    repo_path = str(ROOT_DIR)
    existing_py_path = env.get("PYTHONPATH")
    if existing_py_path:
        if repo_path not in existing_py_path.split(os.pathsep):
            env["PYTHONPATH"] = os.pathsep.join([repo_path, existing_py_path])
    else:
        env["PYTHONPATH"] = repo_path
    return env

# Ensure pip build isolation is disabled so packages like pyradiomics can reuse the conda-provided numpy.
os.environ.setdefault("PIP_NO_BUILD_ISOLATION", "1")


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
    # Auto-calculate parallel courses based on available cores
    # Target: each course gets ~4-6 threads, with minimum 2 parallel courses
    if SNAKEMAKE_THREADS >= 16:
        DEFAULT_PARALLEL_COURSES = max(3, SNAKEMAKE_THREADS // 6)  # ~6 threads per course
    elif SNAKEMAKE_THREADS >= 8:
        DEFAULT_PARALLEL_COURSES = max(2, SNAKEMAKE_THREADS // 4)  # ~4 threads per course
    else:
        DEFAULT_PARALLEL_COURSES = 2  # Minimum 2 parallel courses
else:
    # User override - but enforce minimum of 2 for parallelism
    DEFAULT_PARALLEL_COURSES = max(2, _parallel_courses_raw)

# Log the parallelization settings for debugging
sys.stderr.write(
    f"[rtpipeline] Parallelization: {DEFAULT_PARALLEL_COURSES} parallel courses, "
    f"{SNAKEMAKE_THREADS} total threads, ~{SNAKEMAKE_THREADS // DEFAULT_PARALLEL_COURSES} threads/course\n"
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


def _max_worker_args(count: int) -> list[str]:
    return ["--max-workers", str(max(1, count))]


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
ROBUSTNESS_ENABLED = bool(ROBUSTNESS_CONFIG.get("enabled", False))

CT_CROPPING_CONFIG = config.get("ct_cropping", {})
CT_CROPPING_ENABLED = bool(CT_CROPPING_CONFIG.get("enabled", False))

COURSE_META_DIR = OUTPUT_DIR / "_COURSES"
COURSE_MANIFEST = COURSE_META_DIR / "manifest.json"

AGG_OUTPUTS = {
    "dvh": RESULTS_DIR / "dvh_metrics.xlsx",
    "radiomics": RESULTS_DIR / "radiomics_ct.xlsx",
    "radiomics_mr": RESULTS_DIR / "radiomics_mr.xlsx",
    "fractions": RESULTS_DIR / "fractions.xlsx",
    "metadata": RESULTS_DIR / "case_metadata.xlsx",
    "qc": RESULTS_DIR / "qc_reports.xlsx",
}

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


def _estimate_course_complexity(course_path: Path) -> int:
    dicom_root = course_path / "DICOM"
    search_root = dicom_root if dicom_root.exists() else course_path
    count = 0
    for root, _, files in os.walk(search_root):
        for name in files:
            suffix = name.lower()
            if suffix.endswith(".dcm") or suffix.endswith(".ima"):
                count += 1
    if count == 0:
        for root, _, files in os.walk(course_path):
            count += len(files)
    return max(1, count)

ORGANIZED_SENTINEL_PATTERN = str(OUTPUT_DIR / "{patient}" / "{course}" / ".organized")
SEGMENTATION_SENTINEL_PATTERN = str(OUTPUT_DIR / "{patient}" / "{course}" / ".segmentation_done")
CUSTOM_SEG_SENTINEL_PATTERN = str(OUTPUT_DIR / "{patient}" / "{course}" / ".custom_models_done")
CROP_CT_SENTINEL_PATTERN = str(OUTPUT_DIR / "{patient}" / "{course}" / ".crop_ct_done")
DVH_SENTINEL_PATTERN = str(OUTPUT_DIR / "{patient}" / "{course}" / ".dvh_done")
RADIOMICS_SENTINEL_PATTERN = str(OUTPUT_DIR / "{patient}" / "{course}" / ".radiomics_done")
RADIOMICS_ROBUSTNESS_SENTINEL_PATTERN = str(OUTPUT_DIR / "{patient}" / "{course}" / ".radiomics_robustness_done")
CROP_CT_SENTINEL_PATTERN = str(OUTPUT_DIR / "{patient}" / "{course}" / ".crop_ct_done")
QC_SENTINEL_PATTERN = str(OUTPUT_DIR / "{patient}" / "{course}" / ".qc_done")


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
    run:
        import subprocess

        job_threads = max(1, threads)
        manifest_path = Path(output.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        skip_existing = False
        if manifest_path.exists():
            try:
                manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                course_entries = manifest_data.get("courses", [])
            except Exception:
                course_entries = []
            if course_entries:
                missing_flags = []
                for entry in course_entries:
                    try:
                        course_dir = Path(entry.get("path", ""))
                    except Exception:
                        missing_flags.append(entry)
                        continue
                    flag = course_dir / ".organized"
                    if not flag.exists():
                        missing_flags.append(entry)
                skip_existing = not missing_flags
        if skip_existing:
            log_path = Path(log[0])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("Organize stage skipped (manifest already present).\n", encoding="utf-8")
            return

        env = _rt_env()
        cmd = [
            sys.executable,
            "-m",
            "rtpipeline.cli",
            "--dicom-root", str(DICOM_ROOT),
            "--outdir", str(OUTPUT_DIR),
            "--logs", str(LOGS_DIR),
            "--stage", "organize",
        ]
        cmd.extend(_max_worker_args(job_threads))
        if CUSTOM_STRUCTURES_CONFIG:
            cmd.extend(["--custom-structures", CUSTOM_STRUCTURES_CONFIG])
        LOGS_DIR.mkdir(parents=True, exist_ok=True) # Ensure logs dir exists before writing to logfile
        with open(log[0], "w") as logf:
            logf.write("DEBUG: Starting rtpipeline.cli organize stage...\n")
            logf.flush()
            subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT, env=env)
        courses = []
        for patient_id, course_id, course_path in _iter_course_dirs():
            flag = course_sentinel_path(patient_id, course_id, ".organized")
            flag.parent.mkdir(parents=True, exist_ok=True)
            flag.write_text("ok\n", encoding="utf-8")
            complexity = _estimate_course_complexity(course_path)
            courses.append(
                {
                    "patient": patient_id,
                    "course": course_id,
                    "path": str(course_path),
                    "complexity": complexity,
                }
            )
        if PRIORITIZE_SHORT_COURSES:
            courses.sort(key=lambda entry: (entry.get("complexity", 0), entry["patient"], entry["course"]))
        COURSE_META_DIR.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({"courses": courses}, indent=2), encoding="utf-8")


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
        params:
            extra_args=_seg_params_lambda
        shell:
            """
            set -e
            export PATH="/opt/conda/envs/rtpipeline/bin:$PATH"
            
            /opt/conda/envs/rtpipeline/bin/python -m rtpipeline.cli \
                --dicom-root "{DICOM_ROOT}" \
                --outdir "{OUTPUT_DIR}" \
                --logs "{LOGS_DIR}" \
                --stage segmentation \
                --course-filter "{wildcards.patient}/{wildcards.course}" \
                --max-workers {threads} \
                --manifest "{input.manifest}" \
                {params.extra_args} > {log} 2>&1
            
            echo "ok" > {output.sentinel}
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
            extra_args=_seg_params_lambda
        shell:
            """
            set -e
            python -m rtpipeline.cli \
                --dicom-root "{DICOM_ROOT}" \
                --outdir "{OUTPUT_DIR}" \
                --logs "{LOGS_DIR}" \
                --stage segmentation \
                --course-filter "{wildcards.patient}/{wildcards.course}" \
                --max-workers {threads} \
                --manifest "{input.manifest}" \
                {params.extra_args} > {log} 2>&1
            
            echo "ok" > {output.sentinel}
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
        params:
            enabled=str(CUSTOM_MODELS_ENABLED),
            extra_args=_custom_params_lambda
        shell:
            """
            set -e
            mkdir -p $(dirname {output.sentinel})
            
            if [ "{params.enabled}" = "False" ]; then
                echo "disabled" > {output.sentinel}
                exit 0
            fi
            
            export PATH="/opt/conda/envs/rtpipeline/bin:$PATH"
            
            /opt/conda/envs/rtpipeline/bin/python -m rtpipeline.cli \
                --dicom-root "{DICOM_ROOT}" \
                --outdir "{OUTPUT_DIR}" \
                --logs "{LOGS_DIR}" \
                --stage segmentation_custom \
                --course-filter "{wildcards.patient}/{wildcards.course}" \
                --max-workers {threads} \
                {params.extra_args} > {log} 2>&1
            
            echo "ok" > {output.sentinel}
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
            extra_args=_custom_params_lambda
        shell:
            """
            set -e
            mkdir -p $(dirname {output.sentinel})
            
            if [ "{params.enabled}" = "False" ]; then
                echo "disabled" > {output.sentinel}
                exit 0
            fi
            
            python -m rtpipeline.cli \
                --dicom-root "{DICOM_ROOT}" \
                --outdir "{OUTPUT_DIR}" \
                --logs "{LOGS_DIR}" \
                --stage segmentation_custom \
                --course-filter "{wildcards.patient}/{wildcards.course}" \
                --max-workers {threads} \
                {params.extra_args} > {log} 2>&1
            
            echo "ok" > {output.sentinel}
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
    run:
        import subprocess
        job_threads = max(1, threads)
        sentinel_path = Path(output.sentinel)
        sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = Path(log[0])
        log_path.parent.mkdir(parents=True, exist_ok=True)

        env = _rt_env()
        cmd = [
            sys.executable,
            "-m",
            "rtpipeline.cli",
            "--dicom-root", str(DICOM_ROOT),
            "--outdir", str(OUTPUT_DIR),
            "--logs", str(LOGS_DIR),
            "--stage", "crop_ct",
            "--course-filter", f"{wildcards.patient}/{wildcards.course}",
        ]
        cmd.extend(_max_worker_args(job_threads))


        with log_path.open("w") as logf:
            subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT, env=env)
        sentinel_path.write_text("ok\n", encoding="utf-8")





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
    run:
        import subprocess
        job_threads = max(1, threads)
        sentinel_path = Path(output.sentinel)
        sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = Path(log[0])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = _rt_env()
        cmd = [
            sys.executable,
            "-m",
            "rtpipeline.cli",
            "--dicom-root", str(DICOM_ROOT),
            "--outdir", str(OUTPUT_DIR),
            "--logs", str(LOGS_DIR),
            "--stage", "dvh",
            "--course-filter", f"{wildcards.patient}/{wildcards.course}",
        ]
        cmd.extend(_max_worker_args(job_threads))
        if CUSTOM_STRUCTURES_CONFIG:
            cmd.extend(["--custom-structures", CUSTOM_STRUCTURES_CONFIG])
        with log_path.open("w") as logf:
            subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT, env=env)
        sentinel_path.write_text("ok\n", encoding="utf-8")


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
            SNAKEMAKE_THREADS  # Full thread budget when serialized
        resources:
            radiomics_workers=1  # Serialize radiomics jobs to avoid nested parallelization
        params:
            extra_args=_radiomics_params_lambda
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
            echo "DEBUG: threads={threads} SNAKEMAKE_THREADS={SNAKEMAKE_THREADS}" >> {log}
            /opt/conda/envs/rtpipeline-radiomics/bin/python -m rtpipeline.cli \
                --dicom-root "{DICOM_ROOT}" \
                --outdir "{OUTPUT_DIR}" \
                --logs "{LOGS_DIR}" \
                --stage radiomics \
                --course-filter "{wildcards.patient}/{wildcards.course}" \
                --max-workers {threads} \
                --manifest "{input.manifest}" \
                {params.extra_args} >> {log} 2>&1

            echo "ok" > {output.sentinel}
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
            SNAKEMAKE_THREADS  # Full thread budget when serialized
        resources:
            radiomics_workers=1  # Serialize radiomics jobs to avoid nested parallelization
        conda:
            "envs/rtpipeline-radiomics.yaml"
        params:
            extra_args=_radiomics_params_lambda
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
            echo "DEBUG: threads={threads} SNAKEMAKE_THREADS={SNAKEMAKE_THREADS}" >> {log}
            python -m rtpipeline.cli \
                --dicom-root "{DICOM_ROOT}" \
                --outdir "{OUTPUT_DIR}" \
                --logs "{LOGS_DIR}" \
                --stage radiomics \
                --course-filter "{wildcards.patient}/{wildcards.course}" \
                --max-workers {threads} \
                --manifest "{input.manifest}" \
                {params.extra_args} >> {log} 2>&1

            echo "ok" > {output.sentinel}
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
    run:
        import subprocess
        job_threads = max(1, threads)
        sentinel_path = Path(output.sentinel)
        sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = Path(log[0])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = _rt_env()
        cmd = [
            sys.executable,
            "-m",
            "rtpipeline.cli",
            "--dicom-root", str(DICOM_ROOT),
            "--outdir", str(OUTPUT_DIR),
            "--logs", str(LOGS_DIR),
            "--stage", "qc",
            "--course-filter", f"{wildcards.patient}/{wildcards.course}",
        ]
        cmd.extend(_max_worker_args(job_threads))
        with log_path.open("w") as logf:
            subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT, env=env)
        sentinel_path.write_text("ok\n", encoding="utf-8")



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
            SNAKEMAKE_THREADS  # Full thread budget when serialized
        resources:
            robustness_workers=1  # Serialize robustness jobs like radiomics
        params:
            enabled=str(ROBUSTNESS_ENABLED),
            config=str(ROOT_DIR / "config.yaml")
        shell:
            """
            mkdir -p $(dirname {output.sentinel})

            if [ "{params.enabled}" = "False" ]; then
                echo "disabled" > {output.sentinel}
                exit 0
            fi

            # Export worker budget for subprocess coordination
            export RTPIPELINE_MAX_WORKERS={threads}
            export RTPIPELINE_RADIOMICS_THREAD_LIMIT=1

            if /opt/conda/envs/rtpipeline-radiomics/bin/python -m rtpipeline.cli radiomics-robustness \
                --course-dir "{OUTPUT_DIR}/{wildcards.patient}/{wildcards.course}" \
                --config "{params.config}" \
                --output "{OUTPUT_DIR}/{wildcards.patient}/{wildcards.course}/radiomics_robustness_ct.parquet" \
                --max-workers {threads} > {log} 2>&1; then
                echo "ok" > {output.sentinel}
            else
                echo "failed: see log" > {output.sentinel}
                # Don't fail the pipeline for robustness
                exit 0
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
            SNAKEMAKE_THREADS  # Full thread budget when serialized
        resources:
            robustness_workers=1  # Serialize robustness jobs like radiomics
        conda:
            "envs/rtpipeline-radiomics.yaml"
        params:
            enabled=str(ROBUSTNESS_ENABLED),
            config=str(ROOT_DIR / "config.yaml")
        shell:
            """
            mkdir -p $(dirname {output.sentinel})

            if [ "{params.enabled}" = "False" ]; then
                echo "disabled" > {output.sentinel}
                exit 0
            fi

            # Export worker budget for subprocess coordination
            export RTPIPELINE_MAX_WORKERS={threads}
            export RTPIPELINE_RADIOMICS_THREAD_LIMIT=1

            if python -m rtpipeline.cli radiomics-robustness \
                --course-dir "{OUTPUT_DIR}/{wildcards.patient}/{wildcards.course}" \
                --config "{params.config}" \
                --output "{OUTPUT_DIR}/{wildcards.patient}/{wildcards.course}/radiomics_robustness_ct.parquet" \
                --max-workers {threads} > {log} 2>&1; then
                echo "ok" > {output.sentinel}
            else
                echo "failed: see log" > {output.sentinel}
                # Don't fail the pipeline for robustness
                exit 0
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

    run:
        import subprocess
        if not ROBUSTNESS_ENABLED:
            # Create empty file if disabled
            import pandas as pd
            summary_path = Path(output.summary)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame().to_excel(summary_path, index=False)
            return

        log_path = Path(log[0])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = _rt_env()

        # Collect all parquet files
        parquet_files = []
        for patient_id, course_id, course_path in _iter_course_dirs():
            parquet_path = course_path / "radiomics_robustness_ct.parquet"
            if parquet_path.exists():
                parquet_files.append(str(parquet_path))

        if not parquet_files:
            # Create empty output
            import pandas as pd
            summary_path = Path(output.summary)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame().to_excel(summary_path, index=False)
            with log_path.open("w") as logf:
                logf.write("No robustness parquet files found\n")
            return

        cmd = [
            sys.executable,
            "-m",
            "rtpipeline.cli",
            "radiomics-robustness-aggregate",
            "--inputs",
        ] + parquet_files + [
            "--output", str(output.summary),
            "--config", str(ROOT_DIR / "config.yaml"),
        ]

        with log_path.open("w") as logf:
            subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT, env=env)



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
        radiomics_mr=str(AGG_OUTPUTS["radiomics_mr"]),
        fractions=str(AGG_OUTPUTS["fractions"]),
        metadata=str(AGG_OUTPUTS["metadata"]),
        qc=str(AGG_OUTPUTS["qc"])
    threads:
        AGG_THREADS_RESERVED
    run:
        import json
        import os
        import shutil
        from concurrent.futures import ThreadPoolExecutor
        import pandas as pd  # type: ignore

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        courses = list(_iter_course_dirs())

        def _max_workers(default: int) -> int:
            if not courses:
                return 1
            effective_cap = min(len(courses), max(1, WORKER_BUDGET))
            if AGGREGATION_THREADS is not None:
                return min(effective_cap, max(1, AGGREGATION_THREADS))
            return min(effective_cap, max(1, default))

        worker_count = _max_workers(AUTO_WORKER_BUDGET)

        # Optimized: use single thread pool for all I/O operations
        def _collect_all_frames():
            """Collect all file types in parallel using single thread pool."""
            import pandas as pd
            from collections import defaultdict

            results = defaultdict(list)
            errors = []

            def _load_all(course):
                """Load all file types for a single course.

                Prefers Parquet format when available for 10-50x faster I/O.
                Falls back to Excel for compatibility.
                """
                pid, cid, cdir = course
                course_results = {}

                def _read_prefer_parquet(xlsx_path: Path, name: str) -> pd.DataFrame:
                    """Try Parquet first, fall back to Excel."""
                    parquet_path = xlsx_path.with_suffix('.parquet')
                    if parquet_path.exists():
                        try:
                            return pd.read_parquet(parquet_path)
                        except Exception:
                            pass  # Fall through to Excel
                    if xlsx_path.exists():
                        return pd.read_excel(xlsx_path)
                    return None

                # DVH
                dvh_path = cdir / "dvh_metrics.xlsx"
                try:
                    df = _read_prefer_parquet(dvh_path, "DVH")
                    if df is not None:
                        if "patient_id" in df.columns:
                            df["patient_id"] = df["patient_id"].fillna(pid)
                        else:
                            df.insert(0, "patient_id", pid)
                        if "course_id" in df.columns:
                            df["course_id"] = df["course_id"].fillna(cid)
                        else:
                            df.insert(1, "course_id", cid)
                        if "structure_cropped" not in df.columns:
                            df["structure_cropped"] = False
                        course_results['dvh'] = df
                except Exception as e:
                    errors.append(f"DVH error {pid}/{cid}: {e}")

                # Radiomics CT (prefer Parquet)
                rad_path = cdir / "radiomics_ct.xlsx"
                try:
                    df = _read_prefer_parquet(rad_path, "Radiomics CT")
                    if df is not None:
                        if "patient_id" not in df.columns:
                            df.insert(0, "patient_id", pid)
                        else:
                            df["patient_id"] = df["patient_id"].fillna(pid)
                        if "course_id" not in df.columns:
                            df.insert(1, "course_id", cid)
                        else:
                            df["course_id"] = df["course_id"].fillna(cid)
                        if "structure_cropped" not in df.columns:
                            df["structure_cropped"] = False
                        course_results['radiomics'] = df
                except Exception as e:
                    errors.append(f"Radiomics error {pid}/{cid}: {e}")

                # Radiomics MR (prefer Parquet)
                rad_mr_path = cdir / "MR" / "radiomics_mr.xlsx"
                try:
                    df = _read_prefer_parquet(rad_mr_path, "Radiomics MR")
                    if df is not None:
                        if "patient_id" not in df.columns:
                            df.insert(0, "patient_id", pid)
                        else:
                            df["patient_id"] = df["patient_id"].fillna(pid)
                        if "course_id" not in df.columns:
                            df.insert(1, "course_id", cid)
                        else:
                            df["course_id"] = df["course_id"].fillna(cid)
                        course_results['radiomics_mr'] = df
                except Exception as e:
                    errors.append(f"Radiomics MR error {pid}/{cid}: {e}")

                # Fractions
                frac_path = cdir / "fractions.xlsx"
                if frac_path.exists():
                    try:
                        df = pd.read_excel(frac_path)
                        if "patient_id" in df.columns:
                            df["patient_id"] = df["patient_id"].fillna(pid)
                        else:
                            df.insert(0, "patient_id", pid)
                        if "course_id" in df.columns:
                            df["course_id"] = df["course_id"].fillna(cid)
                        else:
                            df.insert(1, "course_id", cid)
                        course_results['fractions'] = df
                    except Exception as e:
                        errors.append(f"Fractions error {pid}/{cid}: {e}")

                # Metadata
                meta_path = cdir / "metadata" / "case_metadata.xlsx"
                if meta_path.exists():
                    try:
                        df = pd.read_excel(meta_path)
                        if "patient_id" in df.columns:
                            df["patient_id"] = df["patient_id"].fillna(pid)
                        else:
                            df.insert(0, "patient_id", pid)
                        if "course_id" in df.columns:
                            df["course_id"] = df["course_id"].fillna(cid)
                        else:
                            df.insert(1, "course_id", cid)
                        course_results['metadata'] = df
                    except Exception as e:
                        errors.append(f"Metadata error {pid}/{cid}: {e}")

                return course_results

            if not courses:
                return results, []

            # Load all files in parallel
            with ThreadPoolExecutor(max_workers=max(1, worker_count)) as pool:
                for course_data in pool.map(_load_all, courses):
                    for key, df in course_data.items():
                        if df is not None and not df.empty:
                            results[key].append(df)

            return results, errors

        # Collect all frames at once
        all_frames, agg_errors = _collect_all_frames()
        
        if agg_errors:
            print(f"Aggregation warnings ({len(agg_errors)}):")
            for err in agg_errors[:20]:
                 print(f" - {err}")
            if len(agg_errors) > 20:
                print(f" ... and {len(agg_errors)-20} more.")
            
            try:
                error_log_path = RESULTS_DIR / "aggregation_errors.log"
                with open(error_log_path, "w") as f:
                    for err in agg_errors:
                        f.write(f"{err}\n")
                print(f"Full error log written to {error_log_path}")
            except Exception:
                pass

        dvh_frames = all_frames.get('dvh', [])
        rad_frames = all_frames.get('radiomics', [])
        rad_mr_frames = all_frames.get('radiomics_mr', [])
        frac_frames = all_frames.get('fractions', [])
        meta_frames = all_frames.get('metadata', [])

        # DVH frames already collected above
        if dvh_frames:
            dvh_all = pd.concat(dvh_frames, ignore_index=True)
            if "Segmentation_Source" not in dvh_all.columns:
                dvh_all["Segmentation_Source"] = "Unknown"
            if "ROI_Name" in dvh_all.columns:
                roi_series = dvh_all["ROI_Name"].astype(str)
            else:
                roi_series = pd.Series(["" for _ in range(len(dvh_all))], index=dvh_all.index)
                dvh_all.insert(len(dvh_all.columns), "ROI_Name", roi_series)
            dvh_all["_roi_key"] = roi_series.str.strip().str.lower()
            manual_keys = set(
                dvh_all.loc[
                    dvh_all["Segmentation_Source"].astype(str).str.lower() == "manual",
                    "_roi_key",
                ].dropna()
            )
            drop_mask = (
                dvh_all["Segmentation_Source"].astype(str).str.lower().isin({"custom", "merged"})
                & dvh_all["_roi_key"].isin(manual_keys)
            )
            if drop_mask.any():
                dvh_all = dvh_all.loc[~drop_mask].copy()
            dvh_all.drop(columns=["_roi_key"], errors="ignore", inplace=True)
            dvh_all.to_excel(output.dvh, index=False)
        else:
            pd.DataFrame(columns=["patient_id", "course_id", "ROI_Name", "structure_cropped"]).to_excel(output.dvh, index=False)

        # Radiomics frames already collected above
        if rad_frames:
            pd.concat(rad_frames, ignore_index=True).to_excel(output.radiomics, index=False)
        else:
            pd.DataFrame(columns=["patient_id", "course_id", "roi_name", "structure_cropped"]).to_excel(output.radiomics, index=False)

        # Radiomics MR frames already collected above
        if rad_mr_frames:
            pd.concat(rad_mr_frames, ignore_index=True).to_excel(output.radiomics_mr, index=False)
        else:
            pd.DataFrame(columns=["patient_id", "course_id", "roi_name"]).to_excel(output.radiomics_mr, index=False)

        # Fraction frames already collected above
        if frac_frames:
            pd.concat(frac_frames, ignore_index=True).to_excel(output.fractions, index=False)
        else:
            pd.DataFrame(columns=["patient_id", "course_id", "treatment_date", "source_path"]).to_excel(output.fractions, index=False)

        # Metadata frames already collected above
        if meta_frames:
            pd.concat(meta_frames, ignore_index=True).to_excel(output.metadata, index=False)
        else:
            pd.DataFrame(columns=["patient_id", "course_id"]).to_excel(output.metadata, index=False)

        supplemental_sources = {
            "plans.xlsx": OUTPUT_DIR / "Data" / "plans.xlsx",
            "structure_sets.xlsx": OUTPUT_DIR / "Data" / "structure_sets.xlsx",
            "dosimetrics.xlsx": OUTPUT_DIR / "Data" / "dosimetrics.xlsx",
            "fractions.xlsx": OUTPUT_DIR / "Data" / "fractions.xlsx",
            "metadata.xlsx": OUTPUT_DIR / "Data" / "metadata.xlsx",
            "CT_images.xlsx": OUTPUT_DIR / "Data" / "CT_images.xlsx",
        }
        for fname, src_path in supplemental_sources.items():
            if src_path.exists():
                dst_path = RESULTS_DIR / fname
                try:
                    shutil.copy2(src_path, dst_path)
                except Exception as exc:
                    print(f"[aggregate_results] Warning: failed to copy {src_path} -> {dst_path}: {exc}")

        qc_rows = []
        for pid, cid, cdir in courses:
            qc_dir = cdir / "qc_reports"
            if not qc_dir.exists():
                continue
            for report_path in qc_dir.glob("*.json"):
                try:
                    data = json.loads(report_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                qc_rows.append(
                    {
                        "patient_id": pid,
                        "course_id": cid,
                        "report_name": report_path.name,
                        "overall_status": data.get("overall_status"),
                        "structure_cropping": json.dumps(data.get("checks", {}).get("structure_cropping", {})),
                        "checks": json.dumps(data.get("checks", {})),
                    }
                )
        if qc_rows:
            pd.DataFrame(qc_rows).to_excel(output.qc, index=False)
        else:
            pd.DataFrame(columns=["patient_id", "course_id", "report_name", "overall_status"]).to_excel(output.qc, index=False)


rule generate_report:
    input:
        *(str(path) for path in AGG_OUTPUTS.values())
    output:
        "pipeline_report.html"
    log:
        str(LOGS_DIR / "report.log")
    run:
        shell("snakemake --report {output} --configfile config.yaml")
