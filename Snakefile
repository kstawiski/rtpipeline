from __future__ import annotations

import json
import os
import sys
from pathlib import Path

configfile: "config.yaml"

ROOT_DIR = Path.cwd()


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


WORKERS_CFG = config.get("workers", "auto")
if isinstance(WORKERS_CFG, str) and WORKERS_CFG.lower() == "auto":
    # Auto-detect: use CPU count - 2, minimum 4, maximum 32 for safety
    detected_cpus = os.cpu_count() or 4
    WORKERS = max(4, min(32, detected_cpus - 2))
else:
    WORKERS = int(WORKERS_CFG)

def _coerce_int(value, default=None):
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default

SEG_CONFIG = config.get("segmentation", {}) or {}
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
SEG_MAX_WORKERS = _coerce_int(SEG_CONFIG.get("workers") or SEG_CONFIG.get("max_workers"), 1)
if SEG_MAX_WORKERS is not None and SEG_MAX_WORKERS < 1:
    SEG_MAX_WORKERS = 1
SEG_THREADS_PER_WORKER = _coerce_int(SEG_CONFIG.get("threads_per_worker"), None)
if SEG_THREADS_PER_WORKER is not None and SEG_THREADS_PER_WORKER < 1:
    SEG_THREADS_PER_WORKER = 1
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

CUSTOM_MODELS_CONFIG = config.get("custom_models", {}) or {}
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
CUSTOM_MODELS_WORKERS = _coerce_int(CUSTOM_MODELS_CONFIG.get("workers") or CUSTOM_MODELS_CONFIG.get("max_workers"), 1)
if CUSTOM_MODELS_WORKERS is not None and CUSTOM_MODELS_WORKERS < 1:
    CUSTOM_MODELS_WORKERS = 1
CUSTOM_MODELS_PREDICT = str(CUSTOM_MODELS_CONFIG.get("nnunet_predict") or "nnUNetv2_predict")
CUSTOM_MODELS_CONDA = CUSTOM_MODELS_CONFIG.get("conda_activate")
CUSTOM_MODELS_RETAIN = bool(CUSTOM_MODELS_CONFIG.get("retain_weights", True))

RADIOMICS_CONFIG = config.get("radiomics", {}) or {}
RADIOMICS_SEQUENTIAL = bool(RADIOMICS_CONFIG.get("sequential", False))
RADIOMICS_THREAD_LIMIT = _coerce_int(RADIOMICS_CONFIG.get("thread_limit"), None)
if RADIOMICS_THREAD_LIMIT is not None and RADIOMICS_THREAD_LIMIT < 1:
    RADIOMICS_THREAD_LIMIT = 1
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

AGGREGATION_CONFIG = config.get("aggregation", {}) or {}
_agg_threads_raw = AGGREGATION_CONFIG.get("threads")
if isinstance(_agg_threads_raw, str) and _agg_threads_raw.lower() == "auto":
    # Auto: use all CPUs for aggregation (I/O bound)
    AGGREGATION_THREADS = os.cpu_count() or 4
else:
    AGGREGATION_THREADS = _coerce_int(_agg_threads_raw, None)
if AGGREGATION_THREADS is not None and AGGREGATION_THREADS < 1:
    AGGREGATION_THREADS = 1

ROBUSTNESS_CONFIG = config.get("radiomics_robustness", {}) or {}
ROBUSTNESS_ENABLED = bool(ROBUSTNESS_CONFIG.get("enabled", False))

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

SEG_WORKER_POOL = SEG_MAX_WORKERS if SEG_MAX_WORKERS is not None else max(1, WORKERS)
CUSTOM_SEG_WORKER_POOL = CUSTOM_MODELS_WORKERS if CUSTOM_MODELS_WORKERS is not None else 1
workflow.global_resources.update(
    {
        "seg_workers": max(1, SEG_WORKER_POOL),
        "custom_seg_workers": max(1, CUSTOM_SEG_WORKER_POOL),
    }
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
        records.append({"patient": patient, "course": course})
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
DVH_SENTINEL_PATTERN = str(OUTPUT_DIR / "{patient}" / "{course}" / ".dvh_done")
RADIOMICS_SENTINEL_PATTERN = str(OUTPUT_DIR / "{patient}" / "{course}" / ".radiomics_done")
RADIOMICS_ROBUSTNESS_SENTINEL_PATTERN = str(OUTPUT_DIR / "{patient}" / "{course}" / ".radiomics_robustness_done")
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
        max(1, WORKERS)
    conda:
        "envs/rtpipeline.yaml"
    run:
        import subprocess
        worker_count = max(1, int(threads))
        manifest_path = Path(output.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        env = _rt_env()
        cmd = [
            sys.executable,
            "-m",
            "rtpipeline.cli",
            "--dicom-root", str(DICOM_ROOT),
            "--outdir", str(OUTPUT_DIR),
            "--logs", str(LOGS_DIR),
            "--workers", str(worker_count),
            "--stage", "organize",
        ]
        if CUSTOM_STRUCTURES_CONFIG:
            cmd.extend(["--custom-structures", CUSTOM_STRUCTURES_CONFIG])
        with open(log[0], "w") as logf:
            subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT, env=env)
        courses = []
        for patient_id, course_id, course_path in _iter_course_dirs():
            flag = course_sentinel_path(patient_id, course_id, ".organized")
            flag.parent.mkdir(parents=True, exist_ok=True)
            flag.write_text("ok\n", encoding="utf-8")
            courses.append({"patient": patient_id, "course": course_id, "path": str(course_path)})
        COURSE_META_DIR.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({"courses": courses}, indent=2), encoding="utf-8")


rule segmentation_course:
    input:
        manifest=_manifest_input,
        organized=ORGANIZED_SENTINEL_PATTERN
    output:
        sentinel=SEGMENTATION_SENTINEL_PATTERN
    log:
        str(LOGS_DIR / "segmentation" / "{patient}_{course}.log")
    threads:
        max(1, SEG_THREADS_PER_WORKER if SEG_THREADS_PER_WORKER is not None else WORKERS)
    resources:
        seg_workers=1
    conda:
        "envs/rtpipeline.yaml"
    run:
        import subprocess
        worker_count = max(1, int(threads))
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
            "--workers", str(worker_count),
            "--stage", "segmentation",
            "--course-filter", f"{wildcards.patient}/{wildcards.course}",
        ]
        cmd.extend(["--manifest", str(input.manifest)])
        if SEG_TEMP_DIR:
            cmd.extend(["--seg-temp-dir", SEG_TEMP_DIR])
        if SEG_FAST:
            cmd.append("--totalseg-fast")
        for model in SEG_EXTRA_MODELS:
            cmd.extend(["--extra-seg-models", model])
        if SEG_ROI_SUBSET:
            cmd.extend(["--totalseg-roi-subset", SEG_ROI_SUBSET])
        if SEG_DEVICE:
            cmd.extend(["--totalseg-device", SEG_DEVICE])
        if SEG_FORCE_SPLIT:
            cmd.append("--totalseg-force-split")
        else:
            cmd.append("--no-totalseg-force-split")
        if SEG_NR_THR_RESAMP is not None and SEG_NR_THR_RESAMP > 0:
            cmd.extend(["--totalseg-nr-thr-resamp", str(SEG_NR_THR_RESAMP)])
        if SEG_NR_THR_SAVING is not None and SEG_NR_THR_SAVING > 0:
            cmd.extend(["--totalseg-nr-thr-saving", str(SEG_NR_THR_SAVING)])
        if SEG_NUM_PROC_PRE is not None and SEG_NUM_PROC_PRE > 0:
            cmd.extend(["--totalseg-num-proc-pre", str(SEG_NUM_PROC_PRE)])
        if SEG_NUM_PROC_EXPORT is not None and SEG_NUM_PROC_EXPORT > 0:
            cmd.extend(["--totalseg-num-proc-export", str(SEG_NUM_PROC_EXPORT)])
        if SEG_MAX_WORKERS:
            cmd.extend(["--seg-workers", str(max(1, SEG_MAX_WORKERS))])
        if SEG_THREADS_PER_WORKER is not None:
            cmd.extend(["--seg-proc-threads", str(SEG_THREADS_PER_WORKER)])
        if SEG_FORCE_SEGMENTATION:
            cmd.append("--force-segmentation")
        with log_path.open("w") as logf:
            subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT, env=env)
        sentinel_path.write_text("ok\n", encoding="utf-8")


rule segmentation_custom_models:
    input:
        manifest=_manifest_input,
        segmentation=SEGMENTATION_SENTINEL_PATTERN
    output:
        sentinel=CUSTOM_SEG_SENTINEL_PATTERN
    log:
        str(LOGS_DIR / "segmentation_custom" / "{patient}_{course}.log")
    threads:
        max(1, WORKERS)
    resources:
        custom_seg_workers=1
    conda:
        "envs/rtpipeline.yaml"
    run:
        import subprocess
        worker_count = max(1, int(threads))
        sentinel_path = Path(output.sentinel)
        sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = Path(log[0])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not CUSTOM_MODELS_ENABLED:
            sentinel_path.write_text("disabled\n", encoding="utf-8")
            return
        env = _rt_env()
        cmd = [
            sys.executable,
            "-m",
            "rtpipeline.cli",
            "--dicom-root", str(DICOM_ROOT),
            "--outdir", str(OUTPUT_DIR),
            "--logs", str(LOGS_DIR),
            "--workers", str(worker_count),
            "--stage", "segmentation_custom",
            "--course-filter", f"{wildcards.patient}/{wildcards.course}",
        ]
        if CUSTOM_MODELS_ROOT:
            cmd.extend(["--custom-models-root", CUSTOM_MODELS_ROOT])
        if CUSTOM_MODELS_SELECTED:
            for name in CUSTOM_MODELS_SELECTED:
                cmd.extend(["--custom-model", name])
        if CUSTOM_MODELS_FORCE:
            cmd.append("--force-custom-models")
        if CUSTOM_MODELS_WORKERS:
            cmd.extend(["--custom-model-workers", str(max(1, CUSTOM_MODELS_WORKERS))])
        if CUSTOM_MODELS_PREDICT:
            cmd.extend(["--nnunet-predict", CUSTOM_MODELS_PREDICT])
        if CUSTOM_MODELS_CONDA:
            cmd.extend(["--custom-model-conda-activate", CUSTOM_MODELS_CONDA])
        if not CUSTOM_MODELS_RETAIN:
            cmd.append("--purge-custom-model-weights")
        with log_path.open("w") as logf:
            subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT, env=env)
        sentinel_path.write_text("ok\n", encoding="utf-8")


rule dvh_course:
    input:
        manifest=_manifest_input,
        segmentation=SEGMENTATION_SENTINEL_PATTERN,
        custom=CUSTOM_SEG_SENTINEL_PATTERN
    output:
        sentinel=DVH_SENTINEL_PATTERN
    log:
        str(LOGS_DIR / "dvh" / "{patient}_{course}.log")
    threads:
        max(1, WORKERS)
    conda:
        "envs/rtpipeline.yaml"
    run:
        import subprocess
        worker_count = max(1, int(threads))
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
            "--workers", str(worker_count),
            "--stage", "dvh",
            "--course-filter", f"{wildcards.patient}/{wildcards.course}",
        ]
        if CUSTOM_STRUCTURES_CONFIG:
            cmd.extend(["--custom-structures", CUSTOM_STRUCTURES_CONFIG])
        with log_path.open("w") as logf:
            subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT, env=env)
        sentinel_path.write_text("ok\n", encoding="utf-8")


rule radiomics_course:
    input:
        manifest=_manifest_input,
        segmentation=SEGMENTATION_SENTINEL_PATTERN,
        custom=CUSTOM_SEG_SENTINEL_PATTERN
    output:
        sentinel=RADIOMICS_SENTINEL_PATTERN
    log:
        str(LOGS_DIR / "radiomics" / "{patient}_{course}.log")
    threads:
        max(1, WORKERS)
    conda:
        "envs/rtpipeline-radiomics.yaml"
    run:
        import subprocess
        worker_count = max(1, int(threads))
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
            "--workers", str(worker_count),
            "--stage", "radiomics",
            "--course-filter", f"{wildcards.patient}/{wildcards.course}",
        ]
        if RADIOMICS_SEQUENTIAL:
            cmd.append("--sequential-radiomics")
        if RADIOMICS_PARAMS:
            cmd.extend(["--radiomics-params", RADIOMICS_PARAMS])
        if RADIOMICS_PARAMS_MR:
            cmd.extend(["--radiomics-params-mr", RADIOMICS_PARAMS_MR])
        if RADIOMICS_MAX_VOXELS:
            cmd.extend(["--radiomics-max-voxels", str(RADIOMICS_MAX_VOXELS)])
        if RADIOMICS_MIN_VOXELS:
            cmd.extend(["--radiomics-min-voxels", str(RADIOMICS_MIN_VOXELS)])
        for roi in RADIOMICS_SKIP_ROIS:
            cmd.extend(["--radiomics-skip-roi", roi])
        if CUSTOM_STRUCTURES_CONFIG:
            cmd.extend(["--custom-structures", CUSTOM_STRUCTURES_CONFIG])
        if RADIOMICS_THREAD_LIMIT is not None:
            cmd.extend(["--radiomics-proc-threads", str(RADIOMICS_THREAD_LIMIT)])
        with log_path.open("w") as logf:
            subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT, env=env)
        sentinel_path.write_text("ok\n", encoding="utf-8")


rule qc_course:
    input:
        manifest=_manifest_input,
        segmentation=SEGMENTATION_SENTINEL_PATTERN
    output:
        sentinel=QC_SENTINEL_PATTERN
    log:
        str(LOGS_DIR / "qc" / "{patient}_{course}.log")
    threads:
        max(1, WORKERS)
    conda:
        "envs/rtpipeline.yaml"
    run:
        import subprocess
        worker_count = max(1, int(threads))
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
            "--workers", str(worker_count),
            "--stage", "qc",
            "--course-filter", f"{wildcards.patient}/{wildcards.course}",
        ]
        with log_path.open("w") as logf:
            subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT, env=env)
        sentinel_path.write_text("ok\n", encoding="utf-8")


rule radiomics_robustness_course:
    input:
        manifest=_manifest_input,
        radiomics=RADIOMICS_SENTINEL_PATTERN
    output:
        sentinel=RADIOMICS_ROBUSTNESS_SENTINEL_PATTERN
    log:
        str(LOGS_DIR / "radiomics_robustness" / "{patient}_{course}.log")
    threads:
        max(1, WORKERS)
    conda:
        "envs/rtpipeline-radiomics.yaml"
    run:
        import subprocess
        if not ROBUSTNESS_ENABLED:
            # Skip if disabled
            sentinel_path = Path(output.sentinel)
            sentinel_path.parent.mkdir(parents=True, exist_ok=True)
            sentinel_path.write_text("disabled\n", encoding="utf-8")
            return

        worker_count = max(1, int(threads))
        sentinel_path = Path(output.sentinel)
        sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = Path(log[0])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = _rt_env()

        course_path = OUTPUT_DIR / wildcards.patient / wildcards.course
        output_parquet = course_path / "radiomics_robustness_ct.parquet"

        cmd = [
            sys.executable,
            "-m",
            "rtpipeline.cli",
            "radiomics-robustness",
            "--course-dir", str(course_path),
            "--config", str(ROOT_DIR / "config.yaml"),
            "--output", str(output_parquet),
        ]

        try:
            with log_path.open("w") as logf:
                subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT, env=env)
            sentinel_path.write_text("ok\n", encoding="utf-8")
        except subprocess.CalledProcessError as e:
            # Log error but don't fail the entire pipeline
            with log_path.open("a") as logf:
                logf.write(f"\nRobustness analysis failed: {e}\n")
            sentinel_path.write_text(f"failed: {e}\n", encoding="utf-8")


rule aggregate_radiomics_robustness:
    input:
        manifest=_manifest_input,
        robustness=_per_course_sentinels(".radiomics_robustness_done")
    output:
        summary=str(AGG_OUTPUTS.get("radiomics_robustness", RESULTS_DIR / "radiomics_robustness_summary.xlsx"))
    log:
        str(LOGS_DIR / "aggregate_radiomics_robustness.log")
    threads:
        max(1, AGGREGATION_THREADS or 4)
    conda:
        "envs/rtpipeline-radiomics.yaml"
    run:
        import subprocess
        if not ROBUSTNESS_ENABLED:
            # Create empty file if disabled
            import pandas as pd
            pd.DataFrame().to_excel(output.summary, index=False)
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
            pd.DataFrame().to_excel(output.summary, index=False)
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
    conda:
        "envs/rtpipeline.yaml"
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
            if AGGREGATION_THREADS is not None:
                return min(len(courses), max(1, AGGREGATION_THREADS))
            return min(len(courses), max(1, default))

        worker_count = _max_workers(os.cpu_count() or 4)

        # Optimized: use single thread pool for all I/O operations
        def _collect_all_frames():
            """Collect all file types in parallel using single thread pool."""
            import pandas as pd
            from collections import defaultdict

            results = defaultdict(list)

            def _load_all(course):
                """Load all file types for a single course."""
                pid, cid, cdir = course
                course_results = {}

                # DVH
                dvh_path = cdir / "dvh_metrics.xlsx"
                if dvh_path.exists():
                    try:
                        df = pd.read_excel(dvh_path)
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
                    except Exception:
                        pass

                # Radiomics CT
                rad_path = cdir / "radiomics_ct.xlsx"
                if rad_path.exists():
                    try:
                        df = pd.read_excel(rad_path)
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
                    except Exception:
                        pass

                # Radiomics MR
                rad_mr_path = cdir / "MR" / "radiomics_mr.xlsx"
                if rad_mr_path.exists():
                    try:
                        df = pd.read_excel(rad_mr_path)
                        if "patient_id" not in df.columns:
                            df.insert(0, "patient_id", pid)
                        else:
                            df["patient_id"] = df["patient_id"].fillna(pid)
                        if "course_id" not in df.columns:
                            df.insert(1, "course_id", cid)
                        else:
                            df["course_id"] = df["course_id"].fillna(cid)
                        course_results['radiomics_mr'] = df
                    except Exception:
                        pass

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
                    except Exception:
                        pass

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
                    except Exception:
                        pass

                return course_results

            if not courses:
                return results

            # Load all files in parallel
            with ThreadPoolExecutor(max_workers=max(1, worker_count)) as pool:
                for course_data in pool.map(_load_all, courses):
                    for key, df in course_data.items():
                        if df is not None and not df.empty:
                            results[key].append(df)

            return results

        # Collect all frames at once
        all_frames = _collect_all_frames()
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
