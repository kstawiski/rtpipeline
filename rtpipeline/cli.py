from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


from .config import PipelineConfig
from .organize import CourseOutput, organize_and_merge, _hydrate_existing_course
from .utils import run_tasks_with_adaptive_workers

logger = logging.getLogger(__name__)


class ConfigValidationError(ValueError):
    """A project config file asked for something invalid.

    Kept distinct from the errors raised when the config file is merely absent,
    unreadable, or malformed YAML. Those are tolerable and fall back to defaults;
    an invalid user-supplied setting must stop the run instead of silently
    disabling the analysis that was requested.
    """


class ManifestReprocessingRequiredError(RuntimeError):
    """A rejected manifest needs organize, but the stage is unavailable."""


def _config_number(value, converter, key: str):
    """Convert a user-supplied config value, naming the key when it is invalid.

    A bare ``float()``/``int()`` here would raise ``ValueError``, which the broad
    handler around the config block absorbs -- leaving the run on defaults with
    no indication the requested setting was rejected.
    """
    expected = "an integer" if converter is int else "a number"
    # YAML ``true``/``false`` would otherwise coerce to 1/0 -- a silent, wrong
    # value rather than the rejection the user needs to see.
    if isinstance(value, bool):
        raise ConfigValidationError(f"{key} must be {expected}, got {value!r}")
    try:
        return converter(value)
    except (TypeError, ValueError):
        raise ConfigValidationError(f"{key} must be {expected}, got {value!r}") from None


def _apply_dose_yaml_config(cfg: PipelineConfig, yaml_config: dict[str, Any]) -> None:
    """Apply the authoritative top-level dose plausibility setting."""
    if "max_total_dose_gy" not in yaml_config:
        return
    value = _config_number(yaml_config["max_total_dose_gy"], float, "max_total_dose_gy")
    if not math.isfinite(value) or value <= 0:
        raise ConfigValidationError(
            f"max_total_dose_gy must be a finite number greater than zero, got {yaml_config['max_total_dose_gy']!r}"
        )
    cfg.max_total_dose_gy = value


def _positive_dose_threshold(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number") from None
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


@dataclass(slots=True)
class _SegmentTask:
    cfg: PipelineConfig
    course: CourseOutput
    force_segmentation: bool

    @property
    def dir(self) -> Path:
        return self.course.dirs.root


@dataclass(slots=True)
class _AllSeriesSegmentTask:
    cfg: PipelineConfig
    patient_id: str
    force_segmentation: bool

    @property
    def dir(self) -> Path:
        return Path(self.cfg.output_root) / str(self.patient_id) / "all_series"


@dataclass(slots=True)
class _PetSuvTask:
    cfg: PipelineConfig
    patient_id: str
    force: bool

    @property
    def dir(self) -> Path:
        return Path(self.cfg.output_root) / str(self.patient_id) / "all_series"


@dataclass(slots=True)
class _CustomSegmentationTask:
    cfg: PipelineConfig
    course: CourseOutput
    models: tuple
    force_custom: bool

    @property
    def dir(self) -> Path:
        return self.course.dirs.root


@dataclass(slots=True)
class _CropTask:
    course: CourseOutput
    region: str
    superior_margin_cm: float | None
    inferior_margin_cm: float | None
    keep_original: bool = True

    @property
    def dir(self) -> Path:
        return self.course.dirs.root


@dataclass(slots=True)
class _DVHTask:
    course: CourseOutput
    custom_structures: Path | None
    use_cropped: bool
    parallel_workers: int
    max_total_dose_gy: float

    @property
    def dir(self) -> Path:
        return self.course.dirs.root


@dataclass(frozen=True, slots=True)
class _DVHStageOutcome:
    """A successful computed or contract-declared no-metrics DVH outcome."""

    status: str
    output_path: Path | None
    reason_code: str | None = None


@dataclass(slots=True)
class _VisualizationTask:
    course: CourseOutput

    @property
    def dir(self) -> Path:
        return self.course.dirs.root


@dataclass(slots=True)
class _QCTask:
    course: CourseOutput

    @property
    def dir(self) -> Path:
        return self.course.dirs.root


def _execute_segment_task(task: _SegmentTask) -> bool:
    from .segmentation import (
        failed_segmentation_outcome,
        publish_course_segmentation_status,
        segment_course,
    )
    from .auto_rtstruct import build_auto_rtstruct

    course_dir = task.course.dirs.root
    try:
        segment_course(task.cfg, course_dir, force=task.force_segmentation)
        build_auto_rtstruct(course_dir)
        outcome = publish_course_segmentation_status(course_dir)
    except Exception as exc:  # noqa: BLE001 - persist a fail-closed course outcome
        logger.exception("Segmentation stage crashed for %s", course_dir)
        outcome = publish_course_segmentation_status(
            course_dir,
            failed_segmentation_outcome(
                course_dir,
                f"segmentation stage crashed: {type(exc).__name__}: {exc}",
            ),
        )
    return outcome["status"] == "ok"


def _execute_all_series_segment_task(task: _AllSeriesSegmentTask) -> bool:
    from .segmentation import segment_all_series_for_patient

    summary = segment_all_series_for_patient(
        task.cfg, task.patient_id, force=task.force_segmentation
    )
    # A per-series failure is recorded in the summary and processing continues, so
    # the worker must report it. Returning True unconditionally let a run whose
    # eligible series failed exit 0 and look complete to downstream automation.
    return not _all_series_segment_summary_has_failure(summary)


def _all_series_segment_summary_has_failure(summary: object) -> bool:
    """True when any image-class bucket recorded a real segmentation failure.

    ``segment_all_series_for_patient`` returns ``{image_class: {"attempted": int,
    "segmented": int, "failed": int, "skipped": int}}``. Only ``failed`` marks
    something that went wrong; ``skipped`` is idempotent reuse, not an error.
    """
    if not isinstance(summary, dict):
        return False
    for bucket in summary.values():
        if isinstance(bucket, dict) and int(bucket.get("failed", 0) or 0) > 0:
            return True
    return False


def _execute_pet_suv_task(task: _PetSuvTask) -> bool:
    from .pet_suv import ingest_pet_suv_for_patient

    summary = ingest_pet_suv_for_patient(task.cfg, task.patient_id, force=task.force)
    # ``excluded`` counts series that eligibility rules intentionally leave out and
    # must not fail the run; only ``failed`` marks an actual processing error.
    return not (isinstance(summary, dict) and int(summary.get("failed", 0) or 0) > 0)


def _execute_custom_segmentation_task(task: _CustomSegmentationTask) -> bool:
    from .custom_models import run_custom_models_for_course

    run_custom_models_for_course(task.cfg, task.course, task.models, force=task.force_custom)
    return True


def _execute_crop_task(task: _CropTask) -> bool:
    from .anatomical_cropping import apply_systematic_cropping

    apply_systematic_cropping(
        task.course.dirs.root,
        region=task.region,
        superior_margin_cm=task.superior_margin_cm,
        inferior_margin_cm=task.inferior_margin_cm,
        keep_original=task.keep_original,
    )
    return True


def _execute_dvh_task(task: _DVHTask) -> _DVHStageOutcome | None:
    from .course_contract import load_course_contract
    from .dvh import dvh_for_course

    output = dvh_for_course(
        task.course.dirs.root,
        task.custom_structures,
        parallel_workers=task.parallel_workers,
        use_cropped=task.use_cropped,
        rx_dose_gy=task.course.total_prescription_gy,
        max_total_dose_gy=task.max_total_dose_gy,
    )
    contract = load_course_contract(task.course.dirs.root)
    decision = contract.data["dvh"]
    if output is not None:
        output_path = Path(output)
        if not output_path.is_file() or decision.get("metrics_status") != "computed":
            return None
        return _DVHStageOutcome(status="computed", output_path=output_path)

    if (
        contract.dose_qc.get("pass") is not True
        or decision.get("metrics_status") != "not_computed"
        or decision.get("output") is not None
    ):
        return None
    qc_path = task.course.dirs.root / "metadata" / "dvh_qc.json"
    try:
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    recorded = (qc.get("dose_resolution") or {}).get("dvh")
    if qc.get("status") != "skipped" or recorded != decision:
        return None
    return _DVHStageOutcome(
        status="not_computed",
        output_path=None,
        reason_code=str(decision.get("reason_code") or "") or None,
    )


def _execute_visualization_task(task: _VisualizationTask) -> bool:
    from .visualize import generate_axial_review, visualize_course

    try:
        visualize_course(task.course.dirs.root)
    finally:
        try:
            generate_axial_review(task.course.dirs.root)
        except Exception as exc:  # noqa: BLE001 - best-effort
            logger.debug("Axial review failed for %s: %s", task.course.dirs.root, exc)
    return True


def _execute_qc_task(task: _QCTask) -> bool | None:
    from .quality_control import generate_qc_report

    try:
        qc_dir = task.course.dirs.qc_reports
        qc_dir.mkdir(parents=True, exist_ok=True)
        generate_qc_report(task.course.dirs.root, qc_dir)
        return True
    except Exception as exc:  # noqa: BLE001 - log and continue
        logger.warning("QC stage failed for %s: %s", task.course.dirs.root, exc)
        return None


def _derive_segmentation_thread_limit(device: str, worker_budget: int) -> int:
    device = (device or "").lower()
    if device == "cpu":
        return max(1, worker_budget)
    # GPU: limit CPU-bound helpers to a small number to avoid contention
    return max(1, min(4, worker_budget))


def _derive_radiomics_thread_limit(worker_budget: int) -> int:
    # Keep OpenMP/BLAS usage modest to avoid oversubscription.
    limit = max(1, worker_budget // 4)
    return max(1, min(worker_budget, limit))


def _load_courses_from_manifest(
    cfg: PipelineConfig,
    manifest_path: Path,
    patient_filter_ids: set[str],
    course_filter_pairs: set[tuple[str, str]],
) -> list[CourseOutput]:
    path = manifest_path.expanduser()
    if not path.exists():
        logger.warning("Manifest %s not found; falling back to full organize stage", path)
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - log and fallback
        logger.warning("Unable to parse manifest %s: %s", path, exc)
        return []

    filters_active = bool(patient_filter_ids or course_filter_pairs)
    results: list[CourseOutput] = []
    rejected: list[str] = []

    entries = data.get("courses", [])
    if not isinstance(entries, list):
        logger.warning("Manifest %s has no valid courses list; falling back to full organize stage", path)
        return []

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            rejected.append(f"entry {index}: not an object")
            continue
        patient = str(entry.get("patient") or "").strip()
        course = str(entry.get("course") or "").strip()
        if not patient or not course:
            rejected.append(f"entry {index}: missing patient or course")
            continue
        if filters_active and (patient, course) not in course_filter_pairs and patient not in patient_filter_ids:
            continue

        path_str = entry.get("path")
        course_dir = None
        if isinstance(path_str, str) and path_str.strip():
            course_dir = Path(path_str).expanduser()
        if course_dir is None or not course_dir.exists():
            course_dir = cfg.output_root / patient / course
        if not course_dir.exists():
            logger.debug("Manifest course missing directory %s/%s at %s", patient, course, course_dir)
            rejected.append(f"{patient}/{course}: missing course directory")
            continue

        hydrated = _hydrate_existing_course(patient, course, course_dir, {"dir_name": course_dir.name})
        if hydrated is None:
            logger.debug("Manifest course hydration failed for %s/%s", patient, course)
            rejected.append(f"{patient}/{course}: reprocessing required")
            continue
        results.append(hydrated)

    if rejected:
        logger.warning(
            "Manifest %s is incomplete for %d selected course(s); rejecting all %d hydrated course(s) "
            "and falling back to full organize stage (%s)",
            path,
            len(rejected),
            len(results),
            "; ".join(rejected[:5]),
        )
        return []
    if results:
        logger.info("Loaded %d course(s) from manifest %s", len(results), path)
    else:
        logger.warning("Manifest %s did not yield any usable courses", path)

    return results


def _organize_after_manifest_rejection(cfg: PipelineConfig) -> list[CourseOutput]:
    """Rediscover rejected manifest entries or fail with an actionable error."""
    if not callable(organize_and_merge):
        raise ManifestReprocessingRequiredError(
            "Manifest resume found a course requiring reprocessing, but the organize stage "
            "is unavailable. Reprocessing is required before downstream stages can run."
        )
    return organize_and_merge(cfg)


def _cuda_capability_is_executable(capability: tuple[int, int], arch_list: list[str]) -> bool:
    """Return True when a CUDA device can actually execute this build's kernels.

    ``torch.cuda.is_available()`` reports driver and device presence, not whether
    the installed wheel carries code the device can run. A PyTorch build whose
    architectures start at ``sm_70`` loads happily on a Pascal card and then
    raises "no kernel image is available for execution on the device" at the
    first convolution. Compare the device capability against the compiled
    architecture list instead, accepting an exact SASS match, a lower minor
    revision inside the same major generation, or forward-JIT from embedded PTX.
    """
    major, minor = capability
    device_sm = major * 10 + minor
    for arch in arch_list:
        try:
            if arch.startswith("sm_"):
                arch_sm = int(arch[3:])
                if arch_sm == device_sm:
                    return True
                if arch_sm // 10 == major and arch_sm % 10 <= minor:
                    return True
            elif arch.startswith("compute_"):
                if int(arch[8:]) <= device_sm:
                    return True
        except ValueError:  # pragma: no cover - unexpected arch spelling
            continue
    return False


def _usable_cuda_device_count() -> int | None:
    """Count CUDA devices this build can target, or None if undeterminable.

    A definitive ``0`` means torch saw CUDA devices and none can execute its
    kernels. That verdict must outrank the nvidia-smi fallback below, which
    counts physical cards without knowing whether they are runnable.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        arch_list = list(torch.cuda.get_arch_list())
        total = torch.cuda.device_count()
        if not arch_list:
            return total
        usable = 0
        unusable: list[str] = []
        for index in range(total):
            capability = torch.cuda.get_device_capability(index)
            if _cuda_capability_is_executable(capability, arch_list):
                usable += 1
            else:
                name = torch.cuda.get_device_name(index)
                unusable.append(f"{name} (sm_{capability[0]}{capability[1]})")
        if unusable:
            logger.error(
                "CUDA reports %d device(s) but %d cannot execute this PyTorch build: %s. "
                "Compiled architectures are %s. torch.cuda.is_available() is True even so, "
                "and running on these devices fails at the first kernel launch. "
                "Install a PyTorch build covering this architecture or pass --totalseg-device cpu.",
                total,
                len(unusable),
                ", ".join(unusable),
                ", ".join(arch_list),
            )
        return usable
    except Exception:  # pragma: no cover - optional dependency
        return None


def _detect_gpu_count() -> int:
    """Best-effort GPU counter used to fan out segmentation on multi-GPU hosts."""
    usable = _usable_cuda_device_count()
    if usable is not None:
        # torch inspected the devices directly; its verdict is authoritative,
        # including a verdict of zero usable devices.
        return usable

    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_devices and cuda_devices.lower() != "all":
        tokens = [item.strip() for item in cuda_devices.split(",") if item.strip()]
        if tokens:
            return len(tokens)

    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = [line for line in (result.stdout or "").splitlines() if line.strip()]
            if lines:
                return len(lines)
    except Exception:  # pragma: no cover - nvidia-smi not available
        pass

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rtpipeline", description="End-to-end DICOM-RT pipeline")
    p.add_argument("--dicom-root", required=True, help="Path to root with DICOM files")
    p.add_argument("--outdir", default="./Data_Organized", help="Output directory for organized data")
    p.add_argument("--logs", default="./Logs", help="Logs directory")
    p.add_argument("--manifest", default=None, help="Optional manifest JSON produced by organize stage to reuse course metadata")
    p.add_argument(
        "--merge-criteria",
        choices=["same_ct_study", "frame_of_reference"],
        default="same_ct_study",
        help="Course grouping criterion. Default: same_ct_study",
    )
    p.add_argument("--max-days", type=int, default=None, help="Optional max days within a course")
    p.add_argument(
        "--max-total-dose-gy",
        type=_positive_dose_threshold,
        default=None,
        help="Warn when prescribed or delivered course dose exceeds this value in Gy (default: 100)",
    )
    p.add_argument(
        "--allow-ct-only-courses",
        action="store_true",
        help="Permit intentional diagnostic CT-only cohorts; default fails loudly when no RT course is found",
    )
    p.add_argument("--no-segmentation", action="store_true", help="Skip TotalSegmentator")
    p.add_argument("--force-segmentation", action="store_true", help="Re-run TotalSegmentator even if outputs exist")
    p.add_argument("--no-dvh", action="store_true", help="Skip DVH computation")
    p.add_argument("--no-visualize", action="store_true", help="Skip HTML visualization")
    p.add_argument("--no-radiomics", action="store_true", help="Skip pyradiomics extraction")
    p.add_argument("--radiomics-params", default=None, help="Path to custom pyradiomics YAML parameter file")
    p.add_argument("--radiomics-params-mr", default=None, help="Path to pyradiomics YAML parameters for MR segmentation")
    p.add_argument("--sequential-radiomics", action="store_true", help="Use sequential radiomics processing (parallel is default)")
    p.add_argument(
        "--radiomics-skip-roi",
        action="append",
        default=[],
        help="ROI name(s) to exclude from radiomics. Accepts comma-separated values; provide multiple times to add more.",
    )
    p.add_argument(
        "--radiomics-max-voxels",
        type=int,
        default=None,
        help="Skip radiomics for ROIs exceeding this voxel count (default: 15,000,000)",
    )
    p.add_argument(
        "--radiomics-min-voxels",
        type=int,
        default=None,
        help="Skip radiomics for ROIs smaller than this voxel count (default: 120)",
    )
    p.add_argument("--custom-structures", default=None, help="Path to YAML configuration file for custom structures (uses pelvic template by default)")
    p.add_argument("--no-metadata", action="store_true", help="Skip XLSX metadata extraction")
    p.add_argument("--conda-activate", default=None, help="Prefix shell with conda activate (segmentation)")
    p.add_argument("--dcm2niix", default="dcm2niix", help="dcm2niix command name")
    p.add_argument("--totalseg", default="TotalSegmentator", help="TotalSegmentator command name")
    p.add_argument("--totalseg-license", default=None, help="TotalSegmentator license key (if required)")
    p.add_argument("--totalseg-weights", default=None, help="Path to pretrained weights (nnUNet_pretrained_models) for TotalSegmentator (offline)")
    p.add_argument(
        "--extra-seg-models",
        action="append",
        default=[],
        help="Extra TotalSegmentator tasks to run in addition to 'total' (comma-separated or repeat)"
    )
    p.add_argument("--totalseg-fast", action="store_true", help="Add --fast for CPU runs to improve runtime")
    p.add_argument("--totalseg-roi-subset", default=None, help="Restrict to subset of ROIs (comma-separated)")
    p.add_argument("--totalseg-device", default="gpu", choices=["gpu", "cpu", "mps"], help="Device to run TotalSegmentator on (default: gpu)")
    p.add_argument("--totalseg-nr-thr-resamp", type=int, default=None, help="Threads for TotalSegmentator resampling step")
    p.add_argument("--totalseg-nr-thr-saving", type=int, default=None, help="Threads for TotalSegmentator saving step")
    p.add_argument("--totalseg-num-proc-pre", type=int, default=None, help="Worker processes for TotalSegmentator preprocessing")
    p.add_argument("--totalseg-num-proc-export", type=int, default=None, help="Worker processes for TotalSegmentator export")
    p.add_argument("--totalseg-force-split", dest="totalseg_force_split", action="store_true", help="Force TotalSegmentator to split volumes into chunks")
    p.add_argument("--no-totalseg-force-split", dest="totalseg_force_split", action="store_false", help="Disable forced chunking in TotalSegmentator")
    p.set_defaults(totalseg_force_split=None)
    p.add_argument("--totalseg-allow-fallback", action="store_true", help="Allow fallback to CPU if GPU segmentation fails (slow!)")
    p.add_argument("--custom-models-root", default=None, help="Directory containing custom segmentation model definitions (default: ./custom_models)")
    p.add_argument(
        "--custom-model",
        action="append",
        default=[],
        help="Restrict custom segmentation to selected model names (comma-separated or repeat)",
    )
    p.add_argument(
        "--custom-model-only",
        action="append",
        default=[],
        help="Alias for --custom-model; restrict custom segmentation to selected model names.",
    )
    p.add_argument("--custom-model-workers", type=int, default=None, help="Maximum concurrent courses for custom segmentation models")
    p.add_argument("--force-custom-models", action="store_true", help="Force re-run custom segmentation models even if outputs exist")
    p.add_argument("--custom-model-conda-activate", default=None, help="Override conda activation prefix for custom segmentation models")
    p.add_argument("--nnunet-predict", default="nnUNetv2_predict", help="nnUNetv2 prediction command (default: nnUNetv2_predict)")
    p.add_argument("--purge-custom-model-weights", action="store_true", help="Delete extracted nnUNet caches after each custom model run")
    p.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Override automatic worker count (cores-1). Leave unset to auto-detect.",
    )
    p.add_argument("--seg-workers", type=int, default=None, help="Maximum concurrent courses for TotalSegmentator (default: 1)")
    p.add_argument("--seg-temp-dir", default=None, help="Optional scratch directory for TotalSegmentator intermediates (defaults to course directory)")
    p.add_argument(
        "--course-filter",
        action="append",
        default=[],
        help="Restrict stages to specific courses. Accepts PATIENT or PATIENT/COURSE; may be passed multiple times.",
    )
    p.add_argument(
        "--totalseg-timeout",
        type=int,
        default=3600,
        help="Timeout for TotalSegmentator operations in seconds (default: 3600 = 1 hour)",
    )
    p.add_argument(
        "--dcm2niix-timeout",
        type=int,
        default=300,
        help="Timeout for dcm2niix operations in seconds (default: 300 = 5 minutes)",
    )
    p.add_argument(
        "--task-timeout",
        type=int,
        default=None,
        help="Timeout for general pipeline tasks in seconds (default: None = no timeout)",
    )
    p.add_argument(
        "--radiomics-task-timeout",
        type=int,
        default=600,
        help="Timeout for individual radiomics ROI extractions in seconds (default: 600 = 10 minutes)",
    )
    p.add_argument(
        "--radiomics-env-probe-timeout",
        dest="radiomics_env_probe_timeout",
        type=int,
        default=None,
        help=(
            "Seconds allowed for each dedicated radiomics environment import probe "
            "(default: RTPIPELINE_RADIOMICS_ENV_PROBE_TIMEOUT or 180)"
        ),
    )
    p.add_argument("--force-redo", action="store_true", help="Force redo all steps, even if outputs exist (resume is default)")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity")
    p.add_argument(
        "--stage",
        action="append",
        choices=[
            "organize",
            "segmentation",
            "segmentation_custom",
            "segmentation_custom_models",
            "crop_ct",
            "dvh",
            "visualize",
            "radiomics",
            "qc",
        ],
        help="Execute only the selected pipeline stage(s); may be provided multiple times. Default: full pipeline.",
    )
    return p


def _doctor(argv: list[str]) -> int:
    import platform
    import shutil
    from importlib import metadata as importlib_metadata
    p = argparse.ArgumentParser(prog="rtpipeline doctor", description="Check environment for rtpipeline")
    p.add_argument("--logs", default="./Logs", help="Logs directory (used for fallback extraction)")
    p.add_argument("--conda-activate", default=None, help="Conda activation prefix to consider")
    p.add_argument("--dcm2niix", default="dcm2niix", help="dcm2niix command name to check")
    p.add_argument("--totalseg", default="TotalSegmentator", help="TotalSegmentator command name to check")
    args = p.parse_args(argv)

    print("rtpipeline doctor")
    print(f"- Python: {platform.python_version()} on {platform.system()} {platform.release()}")
    # Core Python packages and versions
    def ver(name: str) -> str:
        try:
            return importlib_metadata.version(name)
        except Exception:
            return "not installed"
    print(f"- pydicom: {ver('pydicom')}")
    print(f"- SimpleITK: {ver('SimpleITK')}")
    print(f"- dicompyler-core: {ver('dicompyler-core')}")
    print(f"- pydicom-seg: {ver('pydicom-seg')}")
    print(f"- rt-utils: {ver('rt-utils')}")
    print(f"- TotalSegmentator (pkg): {ver('TotalSegmentator')}")

    # CLI tools
    conda_prefix = args.conda_activate
    dcm2 = shutil.which(args.dcm2niix) if not conda_prefix else None
    totseg = shutil.which(args.totalseg) if not conda_prefix else None
    print(f"- dcm2niix in PATH: {dcm2 or ('requires conda env' if conda_prefix else 'not found')}")
    print(f"- TotalSegmentator in PATH: {totseg or ('requires conda env' if conda_prefix else 'not found')}")

    # GPU diagnostics
    try:
        import subprocess as _sp
        out = _sp.check_output(["python","-c","import torch;print('torch_cuda',getattr(torch,'cuda',None) is not None);print('cuda_available',getattr(torch,'cuda',None) and torch.cuda.is_available());print('cuda_count',getattr(torch,'cuda',None) and torch.cuda.device_count())"], stderr=_sp.STDOUT)
        print(out.decode().strip())
    except Exception:
        print("- PyTorch CUDA diagnostics: unavailable")
    # Weights path
    try:
        from pathlib import Path as _P
        weights = _P(args.logs) / 'nnunet'
        print(f"- Expected TS weights dir: {weights} (exists={weights.exists()})")
    except Exception:
        pass
    try:
        import shutil as _sh
        if _sh.which('nvidia-smi'):
            out = _sp.check_output(['nvidia-smi','-L'])
            print('- nvidia-smi:', out.decode().strip().splitlines()[0])
        else:
            print('- nvidia-smi not found in PATH')
    except Exception:
        print('- nvidia-smi check failed')

    # Bundled zips
    from importlib import resources as importlib_resources
    bundled = []
    for nm in ("dcm2niix_lnx.zip", "dcm2niix_mac.zip", "dcm2niix_win.zip"):
        try:
            res = importlib_resources.files('rtpipeline').joinpath('ext', nm)
            if res.is_file():
                bundled.append(nm)
        except Exception:
            pass
    print(f"- Bundled dcm2niix zips in package: {', '.join(bundled) if bundled else 'none'}")

    # Fallback decision
    cfg = PipelineConfig(
        dicom_root=Path('.'),
        output_root=Path('.'),
        logs_root=Path(args.logs).resolve(),
        conda_activate=conda_prefix,
        segmentation_workers=None,
    )
    # Do not actually run any conversion; just see if fallback can be prepared
    fallback_possible = False
    if not conda_prefix and dcm2 is None:
        # Try dry-run: if bundle exists, we can extract when needed
        for nm in ("dcm2niix_lnx.zip", "dcm2niix_mac.zip", "dcm2niix_win.zip"):
            try:
                res = importlib_resources.files('rtpipeline').joinpath('ext', nm)
                if res.is_file():
                    fallback_possible = True
                    break
            except Exception:
                continue
    print(f"- dcm2niix fallback available: {'yes' if fallback_possible else 'no'}")
    if not conda_prefix and dcm2 is None and not fallback_possible:
        print("  -> NIfTI conversion will be skipped; DICOM-mode segmentation still runs.")
    return 0


def _validate(argv: list[str]) -> int:
    """Validate configuration and environment before running pipeline."""
    import shutil
    from pathlib import Path

    p = argparse.ArgumentParser(prog="rtpipeline validate", description="Validate pipeline configuration and environment")
    p.add_argument("--dicom-root", default="Example_data", help="Path to root with DICOM files")
    p.add_argument("--config", default="config.yaml", help="Path to configuration file")
    p.add_argument("--strict", action="store_true", help="Exit with error on any validation failure")
    args = p.parse_args(argv)

    print("rtpipeline validation")
    print("=" * 60)

    errors = []
    warnings = []

    # Check DICOM root
    dicom_root = Path(args.dicom_root)
    if not dicom_root.exists():
        errors.append(f"DICOM root directory not found: {dicom_root}")
        print(f"❌ DICOM root: {dicom_root} (NOT FOUND)")
    elif not dicom_root.is_dir():
        errors.append(f"DICOM root is not a directory: {dicom_root}")
        print(f"❌ DICOM root: {dicom_root} (NOT A DIRECTORY)")
    else:
        print(f"✅ DICOM root: {dicom_root}")

    # Check configuration file
    config_path = Path(args.config)
    if not config_path.exists():
        warnings.append(f"Configuration file not found: {config_path} (will use defaults)")
        print(f"⚠️  Config file: {config_path} (NOT FOUND, will use defaults)")
    else:
        print(f"✅ Config file: {config_path}")

        # Parse configuration
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Check custom models configuration
            custom_models = config.get("custom_models", {})
            if custom_models.get("enabled", False):
                models_root = Path(custom_models.get("root", "custom_models"))
                if not models_root.exists():
                    errors.append(f"Custom models enabled but root directory not found: {models_root}")
                    print(f"❌ Custom models root: {models_root} (NOT FOUND)")
                else:
                    print(f"✅ Custom models root: {models_root}")

                    # Check for model weights
                    model_dirs = [d for d in models_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
                    if not model_dirs:
                        warnings.append(f"No custom model directories found in {models_root}")
                        print(f"⚠️  Custom models: No model directories found")
                    else:
                        for model_dir in model_dirs:
                            config_file = model_dir / "custom_model.yaml"
                            if not config_file.exists():
                                warnings.append(f"Model {model_dir.name} missing config file")
                                print(f"⚠️  Model {model_dir.name}: missing custom_model.yaml")
                                continue

                            # Check for weight files
                            weight_files = list(model_dir.glob("*.zip")) + list(model_dir.glob("model*/"))
                            if not weight_files:
                                errors.append(f"Model {model_dir.name} missing weight files")
                                print(f"❌ Model {model_dir.name}: missing weight files")
                            else:
                                print(f"✅ Model {model_dir.name}: found {len(weight_files)} weight file(s)")
            else:
                print(f"ℹ️  Custom models: disabled")

        except Exception as e:
            errors.append(f"Failed to parse configuration: {e}")
            print(f"❌ Config parsing failed: {e}")

    # Check for required external tools
    print("\nExternal tools:")
    tools = {
        "dcm2niix": "DICOM to NIfTI conversion",
        "TotalSegmentator": "Auto-segmentation",
        "nnUNet_predict": "Custom model predictions (optional)",
        "nnUNetv2_predict": "Custom model predictions v2 (optional)",
    }

    for tool, description in tools.items():
        if shutil.which(tool):
            print(f"✅ {tool}: found ({description})")
        else:
            if tool.startswith("nnUNet"):
                warnings.append(f"{tool} not found - custom models may not work")
                print(f"⚠️  {tool}: not found ({description})")
            else:
                warnings.append(f"{tool} not found in PATH")
                print(f"⚠️  {tool}: not found in PATH ({description})")

    # Check Python packages
    print("\nPython packages:")
    required_packages = ["numpy", "pandas", "pydicom", "SimpleITK", "matplotlib"]
    optional_packages = ["radiomics"]

    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"✅ {pkg}: installed")
        except ImportError:
            errors.append(f"Required package {pkg} not installed")
            print(f"❌ {pkg}: NOT INSTALLED (required)")

    for pkg in optional_packages:
        try:
            __import__(pkg)
            print(f"✅ {pkg}: installed (optional)")
        except ImportError:
            print(f"ℹ️  {pkg}: not installed (optional)")

    # Summary
    print("\n" + "=" * 60)
    print(f"Validation summary:")
    print(f"  Errors: {len(errors)}")
    print(f"  Warnings: {len(warnings)}")

    if errors:
        print("\n❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")

    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")

    if not errors and not warnings:
        print("\n✅ All checks passed! Pipeline is ready to run.")
        return 0
    elif errors:
        print("\n❌ Validation failed. Please fix the errors above before running the pipeline.")
        return 1 if args.strict else 0
    else:
        print("\n⚠️  Validation completed with warnings. Pipeline may run with limited functionality.")
        return 0


def _radiomics_robustness_course(argv: list[str]) -> int:
    """Run radiomics robustness analysis for a single course."""
    p = argparse.ArgumentParser(
        prog="rtpipeline radiomics-robustness",
        description="Radiomics robustness analysis (segmentation perturbation)",
    )
    p.add_argument("--course-dir", required=True, help="Course directory path")
    p.add_argument("--config", default="config.yaml", help="Path to config YAML")
    p.add_argument("--output", required=True, help="Output parquet file path")
    p.add_argument("--max-workers", type=int, default=None, help="Override automatic worker budget (cores-1)")
    p.add_argument(
        "--radiomics-env-probe-timeout",
        type=int,
        default=None,
        help="Seconds allowed for each dedicated radiomics environment import probe",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = p.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Load config
    course_dir = Path(args.course_dir).resolve()
    output_path = Path(args.output).resolve()
    config_path = Path(args.config)

    # Parse config.yaml for robustness and radiomics settings
    import yaml
    rob_config_data = {}
    yaml_config: dict[str, Any] = {}
    root_dir = config_path.parent.resolve()
    if config_path.exists():
        try:
            with open(config_path) as f:
                yaml_config = yaml.safe_load(f) or {}
            rob_config_data = yaml_config.get("radiomics_robustness", {})
        except Exception as e:
            logger.warning("Failed to parse config.yaml: %s", e)

    from .radiomics_robustness import RobustnessConfig, robustness_for_course
    rob_config = RobustnessConfig.from_dict(rob_config_data)

    if not rob_config.enabled:
        logger.warning("Radiomics robustness is disabled in config; enable with 'radiomics_robustness.enabled: true'")
        return 1

    def _resolve_path(raw: Any, default_name: str) -> Path:
        candidate = Path(str(raw)) if raw else Path(default_name)
        if not candidate.is_absolute():
            candidate = (root_dir / candidate).resolve()
        return candidate

    dicom_root = _resolve_path(yaml_config.get("dicom_root", "Example_data"), "Example_data")
    output_root = _resolve_path(yaml_config.get("output_dir", "Data_Snakemake"), "Data_Snakemake")
    logs_root = _resolve_path(yaml_config.get("logs_dir", "Logs_Snakemake"), "Logs_Snakemake")

    radiomics_cfg = yaml_config.get("radiomics", {}) or {}

    def _coerce_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            ivalue = int(value)
            return ivalue
        except (TypeError, ValueError):
            return None

    env_probe_timeout = args.radiomics_env_probe_timeout
    if env_probe_timeout is None:
        env_probe_timeout = _coerce_int(radiomics_cfg.get("env_probe_timeout"))

    params_file = radiomics_cfg.get("params_file")
    if params_file:
        params_path = Path(params_file)
        if not params_path.is_absolute():
            params_path = (root_dir / params_path).resolve()
    else:
        params_path = None

    params_file_mr = radiomics_cfg.get("mr_params_file")
    if params_file_mr:
        params_path_mr = Path(params_file_mr)
        if not params_path_mr.is_absolute():
            params_path_mr = (root_dir / params_path_mr).resolve()
    else:
        params_path_mr = None

    skip_rois_cfg = radiomics_cfg.get("skip_rois") or []
    if isinstance(skip_rois_cfg, str):
        skip_rois = [item.strip() for item in skip_rois_cfg.replace(";", ",").split(",") if item.strip()]
    else:
        skip_rois = [str(item).strip() for item in skip_rois_cfg if str(item).strip()]

    if args.max_workers and args.max_workers > 0:
        os.environ['RTPIPELINE_MAX_WORKERS'] = str(args.max_workers)

    pipeline_config = PipelineConfig(
        dicom_root=dicom_root,
        output_root=output_root,
        logs_root=logs_root,
        max_workers_override=args.max_workers,
        radiomics_params_file=params_path,
        radiomics_params_file_mr=params_path_mr,
        radiomics_skip_rois=skip_rois,
        radiomics_max_voxels=_coerce_int(radiomics_cfg.get("max_voxels")),
        radiomics_min_voxels=_coerce_int(radiomics_cfg.get("min_voxels")),
        radiomics_analysis_contract={},
        radiomics_env_probe_timeout=env_probe_timeout,
        radiomics_robustness_enabled=rob_config.enabled,
        radiomics_robustness_config=rob_config_data,
    )

    worker_budget = pipeline_config.effective_workers()
    pipeline_config.radiomics_thread_limit = _derive_radiomics_thread_limit(worker_budget)

    # Run robustness analysis
    try:
        result = robustness_for_course(pipeline_config, rob_config, course_dir, output_path=output_path)
        if result is None:
            logger.warning("Robustness analysis produced no output")
            return 1
        logger.info("Robustness analysis complete: %s", result)
        return 0
    except Exception as e:
        if getattr(e, "code", None) == "RADIOMICS_ENV_PROBE_TIMEOUT":
            raise
        logger.error("Robustness analysis failed: %s", e, exc_info=True)
        return 1


def _radiomics_robustness_aggregate(argv: list[str]) -> int:
    """Aggregate radiomics robustness results from multiple courses."""
    p = argparse.ArgumentParser(
        prog="rtpipeline radiomics-robustness-aggregate",
        description="Aggregate radiomics robustness results",
    )
    p.add_argument("--inputs", nargs="+", required=True, help="Input parquet files")
    p.add_argument("--output", required=True, help="Output Excel file")
    p.add_argument("--config", default="config.yaml", help="Path to config YAML")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = p.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Load config
    input_paths = [Path(p) for p in args.inputs]
    output_path = Path(args.output).resolve()
    config_path = Path(args.config)

    # Parse config.yaml for robustness settings
    import yaml
    rob_config_data = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                yaml_config = yaml.safe_load(f) or {}
            rob_config_data = yaml_config.get("radiomics_robustness", {})
        except Exception as e:
            logger.warning("Failed to parse config.yaml: %s", e)

    from .radiomics_robustness import RobustnessConfig, aggregate_robustness_results
    rob_config = RobustnessConfig.from_dict(rob_config_data)

    # Run aggregation
    try:
        aggregate_robustness_results(input_paths, output_path, rob_config)
        logger.info("Aggregation complete: %s", output_path)
        return 0
    except Exception as e:
        logger.error("Aggregation failed: %s", e, exc_info=True)
        return 1


def _apply_dicom_copy_yaml_config(cfg: PipelineConfig, organize_config: dict) -> None:
    """Populate cfg.dicom_copy_* fields from the parsed 'organize' section of a
    project YAML config. Extracted from main() for unit testability. Defaults
    mirror the PipelineConfig dataclass fields (config.py) exactly, so omitting a
    key in the YAML falls back to the same default as an unconfigured pipeline run.
    """
    cfg.dicom_copy_dedup_by_sop_uid = organize_config.get("dedup_by_sop_uid", True)
    cfg.dicom_copy_use_hardlinks = organize_config.get("use_hardlinks", False)
    cfg.dicom_copy_verify_checksum = organize_config.get("verify_checksum", False)
    cfg.dicom_copy_cache_headers = organize_config.get("cache_headers", True)
    if "allow_ct_only_courses" in organize_config:
        cfg.allow_ct_only_courses = bool(organize_config["allow_ct_only_courses"])


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    # Lightweight subcommand dispatch to preserve backward compatibility
    if argv and argv[0] == "doctor":
        return _doctor(argv[1:])
    if argv and argv[0] == "validate":
        return _validate(argv[1:])
    if argv and argv[0] == "tcia-rider-lung-ct":
        from .tcia_acquisition import rider_acquisition_main

        return rider_acquisition_main(argv[1:])
    if argv and argv[0] == "rider-manifest":
        from .tcia_acquisition import rider_manifest_main

        return rider_manifest_main(argv[1:])
    if argv and argv[0] == "radiomics-robustness":
        return _radiomics_robustness_course(argv[1:])
    if argv and argv[0] == "radiomics-robustness-aggregate":
        return _radiomics_robustness_aggregate(argv[1:])
    if argv and argv[0] == "federation":
        from .federation import main as federation_main

        return federation_main(argv[1:])
    args = build_parser().parse_args(argv)
    level = logging.INFO if args.verbose == 0 else logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    skip_rois: list[str] = []
    for item in args.radiomics_skip_roi or []:
        if not item:
            continue
        parts = [seg.strip() for seg in str(item).replace(";", ",").split(",") if seg.strip()]
        skip_rois.extend(parts)

    raw_course_filters = args.course_filter or []
    patient_filter_ids: set[str] = set()
    course_filter_pairs: set[tuple[str, str]] = set()
    for raw_entry in raw_course_filters:
        if not raw_entry:
            continue
        token = str(raw_entry).strip()
        if not token:
            continue
        if "/" in token:
            patient_part, course_part = token.split("/", 1)
            patient_part = patient_part.strip()
            course_part = course_part.strip()
            if patient_part and course_part:
                course_filter_pairs.add((patient_part, course_part))
        else:
            patient_filter_ids.add(token)

    filters_active = bool(course_filter_pairs or patient_filter_ids)

    def _filter_courses(seq):
        selected = []
        for course in seq:
            pid = str(getattr(course, "patient_id", ""))
            cid = str(getattr(course, "course_id", ""))
            if not filters_active:
                selected.append(course)
            elif (pid, cid) in course_filter_pairs or pid in patient_filter_ids:
                selected.append(course)
        return selected

    def _log_skip(stage_name: str) -> None:
        if filters_active:
            logger.info("%s: no courses matched course filter; skipping stage", stage_name)

    seg_temp_root: Path | None = None
    if args.seg_temp_dir:
        seg_temp_candidate = Path(args.seg_temp_dir).expanduser()
        if not seg_temp_candidate.is_absolute():
            seg_temp_candidate = Path.cwd() / seg_temp_candidate
        try:
            seg_temp_root = seg_temp_candidate.resolve(strict=False)
        except TypeError:
            try:
                seg_temp_root = seg_temp_candidate.resolve()
            except FileNotFoundError:
                seg_temp_root = seg_temp_candidate
        except FileNotFoundError:
            seg_temp_root = seg_temp_candidate

    # TotalSegmentator worker/thread defaults
    def _positive_or_none(value: int | None) -> int | None:
        if value is None:
            return None
        return value if value > 0 else None

    # Calculate optimal threading defaults for TotalSegmentator
    import os
    cpu_count = os.cpu_count() or 2
    optimal_threads = max(2, min(8, cpu_count // 2))  # 25-50% of cores, min 2, max 8

    totalseg_force_split = args.totalseg_force_split if args.totalseg_force_split is not None else True
    totalseg_nr_thr_resamp = _positive_or_none(args.totalseg_nr_thr_resamp) or optimal_threads
    totalseg_nr_thr_saving = _positive_or_none(args.totalseg_nr_thr_saving) or optimal_threads
    totalseg_num_proc_pre = _positive_or_none(args.totalseg_num_proc_pre) or 1
    totalseg_num_proc_export = _positive_or_none(args.totalseg_num_proc_export) or 1

    # Set timeout environment variables for subprocess communication
    # Note: These environment variables are used by subprocess calls and persist for the
    # entire process lifetime. This is intentional for subprocess timeout configuration.
    # If the pipeline is imported as a module, these values will affect subsequent operations.
    os.environ['TOTALSEG_TIMEOUT'] = str(args.totalseg_timeout)
    os.environ['DCM2NIIX_TIMEOUT'] = str(args.dcm2niix_timeout)
    if args.radiomics_task_timeout:
        os.environ['RTPIPELINE_RADIOMICS_TASK_TIMEOUT'] = str(args.radiomics_task_timeout)
    if args.radiomics_env_probe_timeout:
        os.environ['RTPIPELINE_RADIOMICS_ENV_PROBE_TIMEOUT'] = str(args.radiomics_env_probe_timeout)

    # Log parallelization settings
    logger.info("=== Parallelization Configuration ===")
    logger.info("CPU cores detected: %d", cpu_count)
    if args.max_workers:
        logger.info("Requested worker cap via --max-workers: %d", args.max_workers)
    logger.info("TotalSegmentator: resampling threads=%d, saving threads=%d",
                totalseg_nr_thr_resamp, totalseg_nr_thr_saving)
    logger.info("GPU device: %s (force_split=%s)", args.totalseg_device, totalseg_force_split)
    logger.info("Timeouts: TotalSegmentator=%ds, dcm2niix=%ds, task=%s, radiomics=%ds, radiomics-env-probe=%s",
                args.totalseg_timeout, args.dcm2niix_timeout,
                args.task_timeout or "None", args.radiomics_task_timeout,
                args.radiomics_env_probe_timeout or "environment/default (180)")

    cfg = PipelineConfig(
        dicom_root=Path(args.dicom_root).resolve(),
        output_root=Path(args.outdir).resolve(),
        logs_root=Path(args.logs).resolve(),
        max_workers_override=args.max_workers,
        merge_criteria=args.merge_criteria,
        max_days_between_plans=args.max_days,
        max_total_dose_gy=args.max_total_dose_gy if args.max_total_dose_gy is not None else 100.0,
        allow_ct_only_courses=args.allow_ct_only_courses,
        do_segmentation=not args.no_segmentation,
        do_dvh=not args.no_dvh,
        do_visualize=not args.no_visualize,
        do_radiomics=not args.no_radiomics,
        resume=not args.force_redo,  # Resume is default, disable only with --force-redo
        conda_activate=args.conda_activate,
        dcm2niix_cmd=args.dcm2niix,
        totalseg_cmd=args.totalseg,
        totalseg_license_key=args.totalseg_license,
        totalseg_weights_dir=Path(args.totalseg_weights).resolve() if args.totalseg_weights else None,
        totalseg_device=args.totalseg_device,
        totalseg_force_split=totalseg_force_split,
        totalseg_nr_thr_resamp=totalseg_nr_thr_resamp,
        totalseg_nr_thr_saving=totalseg_nr_thr_saving,
        totalseg_num_proc_pre=totalseg_num_proc_pre,
        totalseg_num_proc_export=totalseg_num_proc_export,
        totalseg_allow_fallback=args.totalseg_allow_fallback,
        extra_seg_models=[m.strip() for part in (args.extra_seg_models or []) for m in part.split(",") if m.strip()],
        segmentation_workers=args.seg_workers,
        segmentation_temp_root=seg_temp_root,
        totalseg_fast=args.totalseg_fast,
        totalseg_roi_subset=args.totalseg_roi_subset,
        radiomics_params_file=Path(args.radiomics_params).resolve() if args.radiomics_params else None,
        radiomics_params_file_mr=Path(args.radiomics_params_mr).resolve() if args.radiomics_params_mr else None,
        radiomics_skip_rois=skip_rois,
        radiomics_max_voxels=args.radiomics_max_voxels,
        radiomics_min_voxels=args.radiomics_min_voxels,
        radiomics_env_probe_timeout=args.radiomics_env_probe_timeout,
        custom_structures_config=None,  # Will be set below
        task_timeout=args.task_timeout,
    )

    worker_budget = cfg.effective_workers()
    if cfg.segmentation_thread_limit is None:
        cfg.segmentation_thread_limit = _derive_segmentation_thread_limit(cfg.totalseg_device, worker_budget)
    if cfg.radiomics_thread_limit is None:
        cfg.radiomics_thread_limit = _derive_radiomics_thread_limit(worker_budget)
    logger.info("Effective worker budget (non-segmentation stages): %d", worker_budget)

    # Parse custom structures configuration
    custom_structures_config = None
    if args.custom_structures:
        # User provided custom config
        custom_structures_config = Path(args.custom_structures).resolve()
        if not custom_structures_config.exists():
            logger.warning("Custom structures config file not found: %s", custom_structures_config)
            custom_structures_config = None
    else:
        # Use default pelvic template if it exists
        packaged_pelvic_config = Path(__file__).with_name("custom_structures_pelvic.yaml")
        repo_root_pelvic_config = Path(__file__).parent.parent / "custom_structures_pelvic.yaml"
        if packaged_pelvic_config.exists():
            custom_structures_config = packaged_pelvic_config
            logger.info("Using packaged pelvic custom structures template: %s", custom_structures_config)
        elif repo_root_pelvic_config.exists():
            custom_structures_config = repo_root_pelvic_config
            logger.info("Using default pelvic custom structures template: %s", custom_structures_config)
        else:
            logger.info("No custom structures configuration provided")

    # Update config object
    cfg.custom_structures_config = custom_structures_config

    # Load project YAML settings (e.g., ct_cropping, organize) for CLI-driven stages.
    #
    # In Docker projects we mount the project configuration as config.custom.yaml and
    # Snakemake is invoked with `--configfile /app/config.custom.yaml`. The CLI must
    # therefore prefer config.custom.yaml (or an explicit env override) over the
    # repository's default config.yaml to avoid silently using mismatched settings.
    try:
        import yaml
        config_file_path: Path | None = None
        env_config_path = os.environ.get("RTPIPELINE_CONFIGFILE")
        if env_config_path:
            candidate = Path(env_config_path).expanduser()
            if candidate.exists():
                config_file_path = candidate
            else:
                logger.warning("RTPIPELINE_CONFIGFILE points to missing path: %s", candidate)
        if config_file_path is None:
            for candidate in (Path("config.custom.yaml"), Path("config.yaml")):
                if candidate.exists():
                    config_file_path = candidate
                    break

        if config_file_path is not None:
            with open(config_file_path) as f:
                yaml_config = yaml.safe_load(f) or {}
            logger.info("Loaded pipeline YAML settings from %s", config_file_path)
            radiomics_yaml = yaml_config.get("radiomics", {}) or {}
            cfg.radiomics_analysis_contract = (
                radiomics_yaml.get("analysis_contract")
                or yaml_config.get("radiomics_analysis_contract")
                or {}
            )
            if args.max_total_dose_gy is None:
                _apply_dose_yaml_config(cfg, yaml_config)

            ct_cropping = yaml_config.get("ct_cropping", {})
            cfg.ct_cropping_enabled = ct_cropping.get("enabled", False)
            cfg.ct_cropping_region = ct_cropping.get("region", "pelvis")
            # None (unspecified) lets apply_systematic_cropping apply the region-specific default.
            cfg.ct_cropping_superior_margin_cm = ct_cropping.get("superior_margin_cm", None)
            cfg.ct_cropping_inferior_margin_cm = ct_cropping.get("inferior_margin_cm", None)
            cfg.ct_cropping_use_for_dvh = ct_cropping.get("use_cropped_for_dvh", True)
            cfg.ct_cropping_use_for_radiomics = ct_cropping.get("use_cropped_for_radiomics", False)
            cfg.ct_cropping_keep_original = ct_cropping.get("keep_original", True)

            if cfg.ct_cropping_enabled:
                logger.info("CT cropping enabled: region=%s, superior_margin=%scm, inferior_margin=%scm",
                           cfg.ct_cropping_region, cfg.ct_cropping_superior_margin_cm,
                           cfg.ct_cropping_inferior_margin_cm)

            # Load organize/dicom_copy settings
            organize_config = yaml_config.get("organize", {})
            _apply_dicom_copy_yaml_config(cfg, organize_config)
            cfg.do_segment_all_series = bool(organize_config.get("do_segment_all_series", False))
            _seg_classes = organize_config.get("all_series_segment_classes", None)
            if isinstance(_seg_classes, str):
                cfg.all_series_segment_classes = [
                    s.strip() for s in _seg_classes.replace(";", ",").split(",") if s.strip()
                ]
            elif isinstance(_seg_classes, (list, tuple, set)):
                cfg.all_series_segment_classes = [str(c).strip() for c in _seg_classes if str(c).strip()]
            elif _seg_classes is not None:
                raise ConfigValidationError(
                    "organize.all_series_segment_classes must be a YAML list (or comma-separated string) of "
                    f"image_class names, got {type(_seg_classes).__name__}"
                )
            cfg.all_series_fourdct_single_representative = bool(
                organize_config.get("all_series_fourdct_single_representative", False)
            )
            _mat_classes = organize_config.get("all_series_materialize_classes", None)
            if isinstance(_mat_classes, str):
                _mat_list = [s.strip() for s in _mat_classes.replace(";", ",").split(",") if s.strip()]
            elif isinstance(_mat_classes, (list, tuple, set)):
                _mat_list = [str(c).strip() for c in _mat_classes if str(c).strip()]
            elif _mat_classes is None:
                _mat_list = None
            else:
                raise ConfigValidationError(
                    "organize.all_series_materialize_classes must be a YAML list (or comma-separated string) of "
                    f"image_class names, got {type(_mat_classes).__name__}"
                )
            if _mat_list is not None:
                # Store the raw allow-list. The all-series materialization stage (inventory.py) unions
                # in the effective segmentation scope so a segmented class is never skip-materialized.
                cfg.all_series_materialize_classes = _mat_list
            _bc_classes = organize_config.get(
                "body_composition_classes",
                yaml_config.get("body_composition_classes", None),
            )
            if isinstance(_bc_classes, str):
                _bc_list = [s.strip() for s in _bc_classes.replace(";", ",").split(",") if s.strip()]
            elif isinstance(_bc_classes, (list, tuple, set)):
                _bc_list = [str(c).strip() for c in _bc_classes if str(c).strip()]
            elif _bc_classes is None:
                _bc_list = None
            else:
                raise ConfigValidationError(
                    "organize.body_composition_classes must be a YAML list (or comma-separated string) of "
                    f"image_class names, got {type(_bc_classes).__name__}"
                )
            if _bc_list is not None:
                allowed_body_classes = {"planning_ct", "diagnostic_ct", "petct_ct"}
                invalid = sorted(set(_bc_list) - allowed_body_classes)
                if invalid:
                    raise ConfigValidationError(
                        "organize.body_composition_classes supports only planning_ct, diagnostic_ct, "
                        f"petct_ct; got {invalid}"
                    )
                cfg.body_composition_classes = _bc_list
            cfg.mr_functional_sampling = bool(
                organize_config.get(
                    "mr_functional_sampling",
                    yaml_config.get("mr_functional_sampling", False),
                )
            )
            pet_config = yaml_config.get("pet", {}) or {}
            cfg.do_ingest_pet_suv = bool(
                pet_config.get("do_ingest_pet_suv", organize_config.get("do_ingest_pet_suv", False))
            )
            cfg.pet_suv_structures = bool(
                pet_config.get(
                    "pet_suv_structures",
                    organize_config.get(
                        "pet_suv_structures",
                        yaml_config.get("pet_suv_structures", False),
                    ),
                )
            )
            cfg.suv_decay_guard_tol = _config_number(
                pet_config.get("suv_decay_guard_tol", organize_config.get("suv_decay_guard_tol", cfg.suv_decay_guard_tol)),
                float,
                "pet.suv_decay_guard_tol",
            )
            cfg.suv_zextent_primary_fraction = _config_number(
                pet_config.get(
                    "suv_zextent_primary_fraction",
                    organize_config.get("suv_zextent_primary_fraction", cfg.suv_zextent_primary_fraction),
                ),
                float,
                "pet.suv_zextent_primary_fraction",
            )
            cfg.pet_clinical_weight_window_days = _config_number(
                pet_config.get(
                    "pet_clinical_weight_window_days",
                    organize_config.get("pet_clinical_weight_window_days", cfg.pet_clinical_weight_window_days),
                ),
                int,
                "pet.pet_clinical_weight_window_days",
            )
            inventory_db = organize_config.get("inventory_db_path")
            if inventory_db:
                cfg.inventory_db_path = Path(inventory_db).expanduser().resolve()
            scan_run_id = organize_config.get("inventory_scan_run_id")
            if scan_run_id not in (None, ""):
                cfg.inventory_scan_run_id = _config_number(
                    scan_run_id, int, "organize.inventory_scan_run_id"
                )
            patient_ids = organize_config.get("inventory_patient_ids") or []
            if isinstance(patient_ids, str):
                patient_ids = [item.strip() for item in patient_ids.split(",") if item.strip()]
            cfg.inventory_patient_ids = [str(item) for item in patient_ids]
    except ConfigValidationError:
        # The user asked for something invalid. Never fall back to defaults here:
        # doing so silently disables the analysis they requested and still exits 0.
        raise
    except Exception as e:
        # Tolerated: no usable config file (absent, unreadable, or malformed YAML).
        # The pipeline has working defaults for that case.
        logger.debug("Could not load project YAML config overrides: %s", e)

    # Scope organize-stage discovery walks to the requested cohort when explicit
    # course/patient filters are active. This avoids statting the entire DICOM
    # root (potentially tens of millions of NFS files) for a run that targets
    # only a handful of patients. The scope is exactly the set of patients that
    # will be PROCESSED (the filter set); all-series inventory materialization
    # reads absolute paths from the inventory DB and does not depend on this
    # discovery walk, so inventory_patient_ids is intentionally NOT unioned in
    # (that would over-scope a single-patient smoke to the whole inventory).
    # Without active filters, discover_patient_ids stays empty and discovery
    # walks the whole tree (unchanged behaviour).
    if filters_active:
        _discover_ids = set(patient_filter_ids)
        _discover_ids |= {pid for (pid, _cid) in course_filter_pairs}
        cfg.discover_patient_ids = sorted(_discover_ids)

    # Configure custom segmentation models
    custom_models_root: Path | None = None
    if args.custom_models_root:
        custom_models_root = Path(args.custom_models_root).resolve()
    else:
        default_models_root = Path.cwd() / "custom_models"
        if default_models_root.exists():
            custom_models_root = default_models_root
    if custom_models_root and not custom_models_root.exists():
        logger.warning("Custom models root not found: %s", custom_models_root)
        custom_models_root = None

    selected_models: list[str] = []
    for entry in (args.custom_model or []) + (args.custom_model_only or []):
        if not entry:
            continue
        parts = [seg.strip() for seg in str(entry).replace(";", ",").split(",") if seg.strip()]
        selected_models.extend(parts)

    cfg.custom_models_root = custom_models_root
    cfg.custom_model_names = selected_models
    cfg.custom_models_force = bool(args.force_custom_models)
    cfg.custom_models_workers = args.custom_model_workers
    if args.nnunet_predict:
        cfg.nnunet_predict_cmd = args.nnunet_predict
    cfg.custom_models_conda_activate = args.custom_model_conda_activate or cfg.conda_activate
    cfg.custom_models_retain_weights = not bool(args.purge_custom_model_weights)

    # Log resume behavior
    if cfg.resume:
        logging.getLogger(__name__).info("Resume mode enabled (default): skipping existing outputs")
    else:
        logging.getLogger(__name__).info("Force redo mode enabled: regenerating all outputs")

    # Ensure directories and also route logs to a file for traceability
    try:
        cfg.ensure_dirs()
        fh = logging.FileHandler(cfg.logs_root / "rtpipeline.log", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logging.getLogger().addHandler(fh)
    except Exception:
        # Non-fatal; continue with console-only logging
        pass

    default_order = ["organize", "segmentation", "segmentation_custom", "crop_ct", "dvh", "visualize", "radiomics", "qc"]
    stage_aliases = {"segmentation_custom_models": "segmentation_custom"}
    requested = [stage_aliases.get(stage.lower(), stage.lower()) for stage in (args.stage or default_order)]
    stages = [stage for stage in default_order if stage in requested]
    if not stages:
        stages = default_order

    manifest_path: Path | None = None
    if args.manifest:
        manifest_candidate = Path(args.manifest).expanduser()
        if not manifest_candidate.is_absolute():
            manifest_candidate = Path.cwd() / manifest_candidate
        try:
            manifest_path = manifest_candidate.resolve(strict=False)
        except TypeError:  # Python <3.6 compatibility
            try:
                manifest_path = manifest_candidate.resolve()
            except FileNotFoundError:
                manifest_path = manifest_candidate
        except FileNotFoundError:
            manifest_path = manifest_candidate

    manifest_attempted = False
    courses: list | None = None

    def ensure_courses() -> list:
        nonlocal courses, manifest_attempted
        if courses is None:
            if manifest_path and cfg.resume and not manifest_attempted:
                manifest_attempted = True
                manifest_courses = _load_courses_from_manifest(
                    cfg,
                    manifest_path,
                    patient_filter_ids,
                    course_filter_pairs,
                )
                if manifest_courses:
                    courses = manifest_courses
            if courses is None:
                courses = (
                    _organize_after_manifest_rejection(cfg)
                    if manifest_attempted
                    else organize_and_merge(cfg)
                )
        return courses

    if "organize" in stages:
        metadata_snapshot = {} if not args.no_metadata else None
        courses = organize_and_merge(cfg, metadata_snapshot=metadata_snapshot)
        if not args.no_metadata:
            from .meta import export_metadata
            export_metadata(cfg, source_snapshot=metadata_snapshot)

    had_failures = False

    if "segmentation" in stages:
        courses = ensure_courses()
        selected_courses = _filter_courses(courses)
        inventory_patient_ids = {
            str(patient_id)
            for patient_id in (getattr(cfg, "inventory_patient_ids", []) or [])
            if str(patient_id)
        }
        all_series_patient_ids = sorted(
            inventory_patient_ids
            | {str(course.patient_id) for course in selected_courses if course.patient_id}
        )
        has_patient_level_work = bool(all_series_patient_ids) and bool(
            getattr(cfg, "do_segment_all_series", False)
            or getattr(cfg, "do_ingest_pet_suv", False)
        )
        if not selected_courses and not has_patient_level_work:
            _log_skip("Segmentation")
        else:
            # Segmentation workers: sequential on GPU, fan out on CPU
            device = (cfg.totalseg_device or "").lower()
            detected_gpu_count = None
            if cfg.segmentation_workers is not None:
                seg_worker_limit = cfg.segmentation_workers
            elif device == "cpu":
                # CPU mode: allow full worker fan-out
                seg_worker_limit = cfg.effective_workers()
            else:
                # GPU mode: fan out to detected devices when possible
                detected_gpu_count = _detect_gpu_count()
                seg_worker_limit = detected_gpu_count if detected_gpu_count and detected_gpu_count > 0 else 1

            try:
                seg_worker_limit = int(seg_worker_limit)
            except Exception:
                seg_worker_limit = 1
            if seg_worker_limit < 1:
                seg_worker_limit = 1
            seg_worker_limit = min(seg_worker_limit, max(1, cfg.effective_workers()))

            logger.info(
                "Segmentation stage: using %d parallel workers (device: %s, detected_gpus=%s, per-worker threads: resample=%d, save=%d)",
                seg_worker_limit,
                cfg.totalseg_device,
                detected_gpu_count if detected_gpu_count is not None else "n/a",
                totalseg_nr_thr_resamp,
                totalseg_nr_thr_saving,
            )

            segment_tasks = [
                _SegmentTask(cfg=cfg, course=course, force_segmentation=args.force_segmentation)
                for course in selected_courses
            ]

            seg_results = (
                run_tasks_with_adaptive_workers(
                    "Segmentation",
                    segment_tasks,
                    _execute_segment_task,
                    max_workers=seg_worker_limit,
                    logger=logging.getLogger(__name__),
                    show_progress=True,
                    progress_success_only=True,
                    task_timeout=args.task_timeout,
                )
                if segment_tasks
                else []
            )
            if any(not result for result in seg_results):
                had_failures = True

            if getattr(cfg, "do_segment_all_series", False):
                all_series_tasks = [
                    _AllSeriesSegmentTask(
                        cfg=cfg,
                        patient_id=patient_id,
                        force_segmentation=args.force_segmentation,
                    )
                    for patient_id in all_series_patient_ids
                ]
                all_series_results = run_tasks_with_adaptive_workers(
                    "All-series segmentation",
                    all_series_tasks,
                    _execute_all_series_segment_task,
                    max_workers=seg_worker_limit,
                    logger=logging.getLogger(__name__),
                    show_progress=True,
                    task_timeout=args.task_timeout,
                )
                # ``None`` means the worker crashed or timed out; ``False`` means it
                # ran but recorded a per-series failure. Both leave the cohort
                # incomplete and must be reported.
                if any(not r for r in all_series_results):
                    had_failures = True

                # Aggregate the per-series body-composition JSONs into the global
                # Data/body_composition.csv ONCE, serially, now that all parallel
                # all-series workers have joined. Doing it here (rather than per
                # series inside each worker) avoids a cross-process write race and
                # an O(N^2) full-tree rescan. Aggregation failure is non-fatal.
                if getattr(cfg, "body_composition_classes", None):
                    try:
                        from .body_composition import write_body_composition_csv

                        bc_csv = write_body_composition_csv(Path(cfg.output_root))
                        if bc_csv is not None:
                            logging.getLogger(__name__).info("Body-composition CSV written: %s", bc_csv)
                    except Exception as exc:
                        logging.getLogger(__name__).warning("Body-composition CSV aggregation failed: %s", exc)

                # B3: functional-MR sampling under anatomic total_mr masks (opt-in, default
                # off). Runs post-segmentation so masks exist; per-patient (rigid-MI is heavy),
                # then a serial CSV aggregation. Never flips a series to seg_failed. Failure is
                # non-fatal (logged). Precondition warn-loud: mr_anatomic must be in the
                # effective segmentation scope, else every functional series -> anatomic_out_of_scope.
                if getattr(cfg, "mr_functional_sampling", False):
                    _log = logging.getLogger(__name__)
                    _eff_scope = cfg.all_series_segment_classes
                    _anat_in_scope = (_eff_scope is None) or ("mr_anatomic" in set(_eff_scope))
                    if not _anat_in_scope:
                        _log.warning(
                            "mr_functional_sampling=True but mr_anatomic is NOT in "
                            "all_series_segment_classes (%s) — no total_mr masks will exist; "
                            "every functional series will be flagged anatomic_out_of_scope.",
                            _eff_scope,
                        )
                    try:
                        from .mr_functional import (
                            sample_patient_mr_functional,
                            write_mr_functional_structures_csv,
                        )

                        _ver = getattr(cfg, "rtpipeline_version", "") or ""
                        for _pid in all_series_patient_ids:
                            try:
                                sample_patient_mr_functional(
                                    Path(cfg.output_root), _pid,
                                    rtpipeline_version=_ver,
                                    anatomic_in_scope=_anat_in_scope,
                                    force=args.force_segmentation,
                                )
                            except Exception as exc:
                                _log.warning("B3 functional-MR sampling failed for %s: %s", _pid, exc)
                        mrf_csv = write_mr_functional_structures_csv(Path(cfg.output_root))
                        if mrf_csv is not None:
                            _log.info("Functional-MR structures CSV written: %s", mrf_csv)
                    except Exception as exc:
                        _log.warning("B3 functional-MR stage failed: %s", exc)

            if getattr(cfg, "do_ingest_pet_suv", False):
                pet_suv_tasks = [
                    _PetSuvTask(
                        cfg=cfg,
                        patient_id=patient_id,
                        force=args.force_redo,
                    )
                    for patient_id in all_series_patient_ids
                ]
                pet_suv_results = run_tasks_with_adaptive_workers(
                    "PET SUV ingestion",
                    pet_suv_tasks,
                    _execute_pet_suv_task,
                    max_workers=max(1, cfg.effective_workers()),
                    logger=logging.getLogger(__name__),
                    show_progress=True,
                    task_timeout=args.task_timeout,
                )
                # ``None`` means the worker crashed or timed out; ``False`` means it
                # ran but recorded a real ingestion failure. Intentional eligibility
                # exclusions are not counted here.
                if any(not r for r in pet_suv_results):
                    had_failures = True

                # B2: per-structure PET SUV under the paired PET-CT CT-component's `total`
                # masks (opt-in, default off). Runs after PET-SUV ingestion (SUVbw NIfTI) and
                # all-series segmentation (petct_ct total masks). Per-patient serial + serial
                # CSV aggregation; failure is non-fatal. MTV/TLG excluded (PI-gated).
                if getattr(cfg, "pet_suv_structures", False):
                    _log2 = logging.getLogger(__name__)
                    _seg_scope = cfg.all_series_segment_classes
                    _petct_in_scope = (_seg_scope is None) or ("petct_ct" in set(_seg_scope))
                    if not _petct_in_scope:
                        _log2.warning(
                            "pet_suv_structures=True but petct_ct is NOT in "
                            "all_series_segment_classes (%s) — no `total` masks will exist; "
                            "every PET series will be flagged petct_ct_masks_missing.",
                            _seg_scope,
                        )
                    try:
                        from .pet_structures import (
                            sample_patient_pet_suv,
                            write_pet_suv_structures_csv,
                        )

                        _ver2 = getattr(cfg, "rtpipeline_version", "") or ""
                        for _pid2 in all_series_patient_ids:
                            try:
                                sample_patient_pet_suv(
                                    Path(cfg.output_root), _pid2,
                                    rtpipeline_version=_ver2, force=args.force_redo,
                                )
                            except Exception as exc:
                                _log2.warning("B2 PET-SUV structures failed for %s: %s", _pid2, exc)
                        b2_csv = write_pet_suv_structures_csv(Path(cfg.output_root))
                        if b2_csv is not None:
                            _log2.info("PET-SUV structures CSV written: %s", b2_csv)
                    except Exception as exc:
                        _log2.warning("B2 PET-SUV structures stage failed: %s", exc)

    if "segmentation_custom" in stages:
        from .custom_models import discover_custom_models  # lazy import

        models_root = cfg.custom_models_root
        available_models = []
        if models_root is None:
            logger.info("Custom segmentation stage skipped: no custom models root configured")
        else:
            try:
                available_models = discover_custom_models(models_root, cfg.custom_model_names, cfg.nnunet_predict_cmd)
            except Exception as exc:
                logger.warning("Failed to discover custom models in %s: %s", models_root, exc)
                available_models = []

        if not available_models:
            logger.info("No custom segmentation models found; skipping custom stage")
        else:
            courses = ensure_courses()
            selected_courses = _filter_courses(courses)
            if not selected_courses:
                _log_skip("Custom Segmentation")
            else:
                force_custom = cfg.custom_models_force

                # Custom models: sequential on GPU, CPU can fan out
                device = (cfg.totalseg_device or "").lower()
                custom_detected_gpu_count = None
                if cfg.custom_models_workers is not None:
                    custom_worker_limit = cfg.custom_models_workers
                elif device == "cpu":
                    custom_worker_limit = cfg.effective_workers()
                else:
                    # GPU mode: fan out cautiously across detected GPUs
                    custom_detected_gpu_count = _detect_gpu_count()
                    custom_worker_limit = custom_detected_gpu_count if custom_detected_gpu_count and custom_detected_gpu_count > 0 else 1

                try:
                    custom_worker_limit = int(custom_worker_limit)
                except Exception:
                    custom_worker_limit = 1
                if custom_worker_limit < 1:
                    custom_worker_limit = 1
                custom_worker_limit = min(custom_worker_limit, max(1, cfg.effective_workers()))

                logger.info(
                    "Custom segmentation stage: using %d parallel workers (detected_gpus=%s)",
                    custom_worker_limit,
                    custom_detected_gpu_count if custom_detected_gpu_count is not None else "n/a",
                )

                model_tuple = tuple(available_models)
                custom_tasks = [
                    _CustomSegmentationTask(
                        cfg=cfg,
                        course=course,
                        models=model_tuple,
                        force_custom=force_custom,
                    )
                    for course in selected_courses
                ]

                custom_results = run_tasks_with_adaptive_workers(
                    "CustomSegmentation",
                    custom_tasks,
                    _execute_custom_segmentation_task,
                    max_workers=custom_worker_limit,
                    logger=logging.getLogger(__name__),
                    show_progress=True,
                    task_timeout=args.task_timeout,
                    use_processes=(device == "cpu"),
                )
                if any(r is None for r in custom_results):
                    had_failures = True

    if "crop_ct" in stages:
        if not cfg.ct_cropping_enabled:
            logger.info("CT cropping disabled in config; skipping crop_ct stage")
        else:
            courses = ensure_courses()
            selected_courses = _filter_courses(courses)

            if not selected_courses:
                _log_skip("CT Cropping")
            else:
                crop_tasks = [
                    _CropTask(
                        course=course,
                        region=cfg.ct_cropping_region,
                        superior_margin_cm=cfg.ct_cropping_superior_margin_cm,
                        inferior_margin_cm=cfg.ct_cropping_inferior_margin_cm,
                        keep_original=cfg.ct_cropping_keep_original,
                    )
                    for course in selected_courses
                ]

                crop_results = run_tasks_with_adaptive_workers(
                    "CT_Cropping",
                    crop_tasks,
                    _execute_crop_task,
                    max_workers=cfg.effective_workers(),
                    logger=logging.getLogger(__name__),
                    show_progress=True,
                    task_timeout=args.task_timeout,
                )
                if any(r is None for r in crop_results):
                    had_failures = True

    if "dvh" in stages:
        courses = ensure_courses()
        selected_courses = _filter_courses(courses)

        if not selected_courses:
            _log_skip("DVH")
        else:
            effective_workers = cfg.effective_workers()
            logger.info("DVH stage: using %d parallel workers", effective_workers)

            dvh_tasks = [
                _DVHTask(
                    course=course,
                    custom_structures=cfg.custom_structures_config,
                    use_cropped=cfg.ct_cropping_use_for_dvh,
                    parallel_workers=effective_workers,
                    max_total_dose_gy=cfg.max_total_dose_gy,
                )
                for course in selected_courses
            ]

            dvh_results = run_tasks_with_adaptive_workers(
                "DVH",
                dvh_tasks,
                _execute_dvh_task,
                max_workers=effective_workers,
                logger=logging.getLogger(__name__),
                show_progress=True,
                task_timeout=args.task_timeout,
            )
            if any(r is None for r in dvh_results):
                had_failures = True

    if "visualize" in stages:
        courses = ensure_courses()
        selected_courses = _filter_courses(courses)

        if not selected_courses:
            _log_skip("Visualization")
        else:
            effective_workers = cfg.effective_workers()
            logger.info("Visualization stage: using %d parallel workers", effective_workers)

            viz_tasks = [_VisualizationTask(course=course) for course in selected_courses]

            viz_results = run_tasks_with_adaptive_workers(
                "Visualization",
                viz_tasks,
                _execute_visualization_task,
                max_workers=effective_workers,
                logger=logging.getLogger(__name__),
                show_progress=True,
                task_timeout=args.task_timeout,
            )
            if any(r is None for r in viz_results):
                had_failures = True

    if "radiomics" in stages:
        import os

        courses = ensure_courses()
        selected_courses = _filter_courses(courses)

        if args.sequential_radiomics:
            os.environ['RTPIPELINE_RADIOMICS_SEQUENTIAL'] = '1'
            logger.info("Using sequential radiomics processing (--sequential-radiomics specified)")
        else:
            try:
                from .radiomics_parallel import enable_parallel_radiomics_processing
                enable_parallel_radiomics_processing(cfg.radiomics_thread_limit)
                # Log the actual worker count that will be used
                workers_used = max(1, cpu_count - 1)
                thread_limit = cfg.radiomics_thread_limit or "auto"
                logger.info("Enabled parallel radiomics processing: %d workers (cores-1), thread_limit=%s",
                           workers_used, thread_limit)
            except ImportError:
                logger.debug("Parallel radiomics helpers unavailable; proceeding with default extractor")

        can_use_radiomics = False
        try:
            from .radiomics import _have_pyradiomics
            can_use_radiomics = _have_pyradiomics()
        except ImportError:
            can_use_radiomics = False

        if not can_use_radiomics:
            try:
                from .radiomics_conda import check_radiomics_env
                can_use_radiomics = check_radiomics_env(
                    timeout=cfg.radiomics_env_probe_timeout,
                )
                if can_use_radiomics:
                    logger.info("PyRadiomics will run via dedicated conda environment")
            except ImportError:
                can_use_radiomics = False

        if not can_use_radiomics:
            had_failures = True
            logger.error("Radiomics dependencies unavailable; radiomics stage failed")
        else:
            radiomics_patient_ids = sorted(
                {
                    str(patient_id)
                    for patient_id in (getattr(cfg, "inventory_patient_ids", []) or [])
                    if str(patient_id)
                }
                | {str(course.patient_id) for course in selected_courses if course.patient_id}
            )
            has_all_series_radiomics = bool(
                getattr(cfg, "do_segment_all_series", False) and radiomics_patient_ids
            )
            if not selected_courses and not has_all_series_radiomics:
                _log_skip("Radiomics")
            else:
                from .radiomics import run_radiomics, run_radiomics_all_series
                if selected_courses:
                    try:
                        run_radiomics(cfg, selected_courses, cfg.custom_structures_config)
                    except Exception as exc:  # noqa: BLE001 - report and continue to other stages
                        had_failures = True
                        logger.error("Radiomics stage failed: %s", exc, exc_info=True)
                if has_all_series_radiomics:
                    try:
                        run_radiomics_all_series(cfg, radiomics_patient_ids)
                    except Exception as exc:  # noqa: BLE001 - keep course radiomics independently usable
                        had_failures = True
                        logger.error("All-series radiomics stage failed: %s", exc, exc_info=True)

    if "qc" in stages:
        courses = ensure_courses()
        selected_courses = _filter_courses(courses)

        if not selected_courses:
            _log_skip("QC")
        else:
            effective_workers = cfg.effective_workers()
            logger.info("QC stage: using %d parallel workers", effective_workers)

            qc_tasks = [_QCTask(course=course) for course in selected_courses]

            qc_results = run_tasks_with_adaptive_workers(
                "QC",
                qc_tasks,
                _execute_qc_task,
                max_workers=effective_workers,
                logger=logging.getLogger(__name__),
                show_progress=True,
                task_timeout=args.task_timeout,
            )
            if any(r is None for r in qc_results):
                had_failures = True

    return 1 if had_failures else 0


def console_main(argv: list[str] | None = None) -> int:
    """Entry point for the installed ``rtpipeline`` command.

    ``project.scripts`` calls this rather than ``main`` so that an invalid
    configuration is reported as a readable message and a non-zero exit status
    instead of an uncaught traceback. Putting the handling only under
    ``if __name__ == "__main__"`` would leave the installed command uncovered,
    which is how most users invoke the pipeline.
    """
    try:
        return main(argv)
    except ConfigValidationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        code = getattr(exc, "code", None)
        if code != "RADIOMICS_ENV_PROBE_TIMEOUT":
            raise
        print(
            json.dumps({"code": code, "message": str(exc)}),
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(console_main())
