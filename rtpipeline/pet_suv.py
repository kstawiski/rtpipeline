from __future__ import annotations

import datetime as _dt
import json
import logging
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.dataset import Dataset

from .config import PipelineConfig
from .layout import build_course_dirs
from .utils import read_dicom

logger = logging.getLogger(__name__)

CLINICAL_ROOT = Path("/umed-projekty/KOPERNIK/MIEDNICE/data/clinical")

PET_SUV_CANDIDATE_STATUSES = {
    "materialized",
    "suv_computed",
    "suv_excluded",
    "suv_failed",
    "suv_skipped_idempotent",
}

RECON_TIER = {
    "OSEM-like": 1,
    "PSF": 2,
    "BSREM": 3,
    "FBP": 4,
}


@dataclass(slots=True)
class ValidationResult:
    valid: bool
    reasons: list[str] = field(default_factory=list)
    ledger: list[dict[str, Any]] = field(default_factory=list)
    image_type_tokens: list[str] = field(default_factory=list)
    corrected_image_tokens: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RadiopharmInfo:
    radiopharmaceutical: Any = None
    start_datetime: Any = None
    start_time: Any = None
    total_dose_bq: float | None = None
    half_life_s: float | None = None
    radionuclide_code_meaning: str = ""
    sources: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class TracerResolution:
    compound: str | None
    isotope: str | None
    isotope_source: str
    label_family: str | None
    tracer_source: str
    flags: list[str] = field(default_factory=list)
    needs_manual_review: bool = False
    source_text: str = ""


@dataclass(slots=True)
class WeightInfo:
    weight_kg: float | None
    weight_source: str
    weight_source_date: str | None = None
    height_m: float | None = None
    height_source: str | None = None
    height_source_date: str | None = None


@dataclass(slots=True)
class TimingInfo:
    injection_datetime: _dt.datetime | None
    injection_datetime_source: str
    reference_datetime: _dt.datetime | None
    reference_datetime_source: str
    delta_t_s: float | None
    flags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DecayGuardResult:
    passed: bool
    expected_decay_factor: float | None
    observed_decay_factor: float | None
    relative_difference: float | None
    frame_reference_time_ms: float | None
    half_life_s: float | None
    tolerance: float
    reason: str = ""
    unconverted_expected_decay_factor: float | None = None


@dataclass(slots=True)
class VolumeBuildResult:
    activity_bqml: np.ndarray
    affine: np.ndarray
    geometry: dict[str, Any]
    slice_scaling: list[dict[str, Any]]


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\\".join(_as_text(item) for item in value)
    return str(value).strip()


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = value.split("\\")
    else:
        try:
            parts = list(value)
        except TypeError:
            parts = [value]
    return [str(part).strip().upper() for part in parts if str(part).strip()]


def _safe_token(token: Any, fallback: str = "series") -> str:
    text = _as_text(token) or fallback
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch in {".", "-", "_"}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    out = "".join(cleaned)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")[:96] or fallback


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def _ledger_entry(
    row: dict[str, Any] | None,
    ds: Dataset | None,
    reason: str,
    *,
    severity: str = "exclude",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = row or {}
    details = details or {}
    return {
        "patient_id": str(row.get("patient_id") or ""),
        "study_uid": str(row.get("study_uid") or getattr(ds, "StudyInstanceUID", "") or ""),
        "series_uid": str(row.get("series_uid") or getattr(ds, "SeriesInstanceUID", "") or ""),
        "reason": reason,
        "severity": severity,
        "manufacturer": _as_text(getattr(ds, "Manufacturer", "")) if ds is not None else "",
        "manufacturer_model": str(row.get("manufacturer_model") or getattr(ds, "ManufacturerModelName", "") or ""),
        "image_type": _tokens(getattr(ds, "ImageType", None)) if ds is not None else [],
        "study_description": _as_text(getattr(ds, "StudyDescription", "")) if ds is not None else "",
        "series_description": _as_text(getattr(ds, "SeriesDescription", "")) if ds is not None else "",
        "output_dir": str(row.get("output_dir") or ""),
        "details": details,
    }


def _sequence_first(ds: Dataset, keyword: str) -> Dataset | None:
    seq = getattr(ds, keyword, None)
    if seq is None:
        return None
    try:
        return seq[0] if len(seq) else None
    except Exception:
        return None


def _nested_or_top(ds: Dataset, item: Dataset | None, keyword: str) -> tuple[Any, str]:
    if item is not None and hasattr(item, keyword):
        return getattr(item, keyword), "nested"
    if hasattr(ds, keyword):
        return getattr(ds, keyword), "top_level"
    return None, "missing"


def _code_meaning(container: Dataset | None) -> str:
    if container is None:
        return ""
    seq = getattr(container, "RadionuclideCodeSequence", None)
    try:
        if seq and len(seq):
            return _as_text(getattr(seq[0], "CodeMeaning", ""))
    except Exception:
        return ""
    return ""


def extract_radiopharmaceutical(ds: Dataset) -> RadiopharmInfo:
    item = _sequence_first(ds, "RadiopharmaceuticalInformationSequence")
    raw_label, label_source = _nested_or_top(ds, item, "Radiopharmaceutical")
    raw_start_dt, start_dt_source = _nested_or_top(ds, item, "RadiopharmaceuticalStartDateTime")
    raw_start_time, start_time_source = _nested_or_top(ds, item, "RadiopharmaceuticalStartTime")
    raw_dose, dose_source = _nested_or_top(ds, item, "RadionuclideTotalDose")
    raw_half_life, half_life_source = _nested_or_top(ds, item, "RadionuclideHalfLife")

    code = _code_meaning(item)
    code_source = "nested" if code else "missing"
    if not code:
        code = _code_meaning(ds)
        code_source = "top_level" if code else "missing"

    return RadiopharmInfo(
        radiopharmaceutical=raw_label,
        start_datetime=raw_start_dt,
        start_time=raw_start_time,
        total_dose_bq=_as_float(raw_dose),
        half_life_s=_as_float(raw_half_life),
        radionuclide_code_meaning=code,
        sources={
            "Radiopharmaceutical": label_source,
            "RadiopharmaceuticalStartDateTime": start_dt_source,
            "RadiopharmaceuticalStartTime": start_time_source,
            "RadionuclideTotalDose": dose_source,
            "RadionuclideHalfLife": half_life_source,
            "RadionuclideCodeSequence.CodeMeaning": code_source,
        },
    )


def _normalize_text(text: Any) -> str:
    raw = _as_text(text).lower()
    raw = raw.replace("^", "")
    raw = raw.replace("[", " ").replace("]", " ")
    raw = raw.replace("_", " ").replace("-", " ")
    raw = re.sub(r"[^a-z0-9]+", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


def _is_noninformative_label(text: str) -> bool:
    norm = _normalize_text(text)
    return norm in {"", "solution", "other", "none", "null", "unknown"}


def _isotope_from_code(code_meaning: Any) -> str | None:
    norm = _normalize_text(code_meaning)
    if "11" in norm and "carbon" in norm:
        return "C11"
    if "18" in norm and "fluor" in norm:
        return "F18"
    if "68" in norm and ("gallium" in norm or "galium" in norm or "ga" in norm):
        return "Ga68"
    return None


def _isotope_from_half_life(half_life_s: float | None) -> str | None:
    if half_life_s is None:
        return None
    if 1100 <= half_life_s <= 1350:
        return "C11"
    if 6200 <= half_life_s <= 7000:
        return "F18"
    if 3800 <= half_life_s <= 4300:
        return "Ga68"
    return None


def _label_family(text: Any) -> str | None:
    norm = _normalize_text(text)
    if not norm:
        return None
    compact = norm.replace(" ", "")
    if "fdg" in compact or "fluorodeoxyglucose" in compact:
        return "FDG"
    if "fluorocholine" in compact or "fcholine" in compact or "f18fluorocholine" in compact:
        return "choline"
    if "choline" in compact or "cholina" in compact:
        return "choline"
    if "acetate" in compact:
        return "acetate"
    if "dcfpyl" in compact or "psma" in compact or "gozetotide" in compact:
        return "PSMA"
    if "dotatate" in compact:
        return "DOTATATE"
    if "dotatoc" in compact or "edotreotide" in compact:
        return "DOTATOC"
    return None


def _label_explicit_isotope(text: Any) -> str | None:
    norm = _normalize_text(text)
    compact = norm.replace(" ", "")
    if "11c" in compact or "c11" in compact:
        return "C11"
    if "18f" in compact or "f18" in compact:
        return "F18"
    if "68ga" in compact or "ga68" in compact:
        return "Ga68"
    return None


def _compound_from_family(family: str | None, isotope: str | None) -> str | None:
    if family == "FDG" and isotope == "F18":
        return "F18_FDG"
    if family == "choline" and isotope == "F18":
        return "F18_fluorocholine"
    if family == "choline" and isotope == "C11":
        return "C11_choline"
    if family == "acetate" and isotope == "C11":
        return "C11_acetate"
    if family == "PSMA" and isotope == "Ga68":
        return "Ga68_PSMA11"
    if family == "PSMA" and isotope == "F18":
        return "F18_PSMA_ligand"
    if family == "DOTATOC" and isotope == "Ga68":
        return "Ga68_DOTATOC"
    if family == "DOTATATE" and isotope == "Ga68":
        return "Ga68_DOTATATE"
    return None


def resolve_compound(
    radiopharmaceutical: Any,
    radionuclide_code_meaning: Any = "",
    half_life_s: float | None = None,
    *,
    study_description: Any = "",
    series_description: Any = "",
) -> TracerResolution:
    code_isotope = _isotope_from_code(radionuclide_code_meaning)
    half_life_isotope = _isotope_from_half_life(half_life_s)
    flags: list[str] = []

    isotope = code_isotope or half_life_isotope
    isotope_source = "code_meaning" if code_isotope else ("half_life" if half_life_isotope else "missing")
    if code_isotope and half_life_isotope and code_isotope != half_life_isotope:
        flags.append("radionuclide_code_half_life_conflict")

    label_text = _as_text(radiopharmaceutical)
    tracer_source = "radiopharmaceutical"
    source_text = label_text
    if _is_noninformative_label(label_text):
        fallback = " ".join(part for part in (_as_text(study_description), _as_text(series_description)) if part)
        tracer_source = "description_fallback"
        source_text = fallback

    family = _label_family(source_text)
    explicit_label_isotope = _label_explicit_isotope(source_text)
    if explicit_label_isotope and isotope and explicit_label_isotope != isotope:
        flags.append("tracer_label_isotope_conflict")

    compound = _compound_from_family(family, isotope)
    if compound:
        return TracerResolution(
            compound=compound,
            isotope=isotope,
            isotope_source=isotope_source,
            label_family=family,
            tracer_source=tracer_source,
            flags=flags,
            source_text=source_text,
        )

    if family and isotope:
        flags.append("unknown_compound_mapping")
    return TracerResolution(
        compound=None,
        isotope=isotope,
        isotope_source=isotope_source,
        label_family=family,
        tracer_source=tracer_source,
        flags=flags,
        needs_manual_review=True,
        source_text=source_text,
    )


def validate_pet_series(
    headers: list[Dataset],
    row: dict[str, Any] | None = None,
) -> ValidationResult:
    ds = headers[0] if headers else None
    reasons: list[str] = []
    ledger: list[dict[str, Any]] = []
    if ds is None:
        return ValidationResult(False, ["missing_dicom_headers"], [_ledger_entry(row, None, "missing_dicom_headers")])

    if _as_text(getattr(ds, "Modality", "")).upper() != "PT":
        reasons.append("invalid_modality")

    units = _as_text(getattr(ds, "Units", "")).upper()
    if units != "BQML":
        reasons.append("non_bqml_units")

    corrected = _tokens(getattr(ds, "CorrectedImage", None))
    if not {"ATTN", "SCAT", "DECY"}.issubset(set(corrected)):
        reasons.append("missing_required_corrections")

    decay_correction = _as_text(getattr(ds, "DecayCorrection", "")).upper()
    if decay_correction != "START":
        reasons.append("unsupported_decay_correction")

    frames = _as_float(getattr(ds, "NumberOfFrames", None))
    if frames is not None and frames > 1:
        reasons.append("unsupported_multiframe_pet")

    image_type = _tokens(getattr(ds, "ImageType", None))
    if len(image_type) < 2 or image_type[0] != "ORIGINAL" or image_type[1] != "PRIMARY":
        reasons.append("invalid_image_type")
    if any(token in {"DERIVED", "SECONDARY"} for token in image_type):
        reasons.append("invalid_image_type")

    manufacturer = _normalize_text(getattr(ds, "Manufacturer", ""))
    manufacturer_model = _normalize_text(getattr(ds, "ManufacturerModelName", ""))
    description_text = _normalize_text(
        " ".join(
            [
                _as_text(getattr(ds, "ImageType", "")),
                _as_text(getattr(ds, "StudyDescription", "")),
                _as_text(getattr(ds, "SeriesDescription", "")),
            ]
        )
    )
    software_text = " ".join([manufacturer, manufacturer_model])
    if any(token in software_text for token in ("velocity", "art plan", "artplan", "mim")):
        reasons.append("derived_rt_software_pet_export")
    elif "fusion" in description_text and any(
        token in description_text for token in ("velocity", "art plan", "artplan", "mim")
    ):
        reasons.append("derived_rt_software_pet_export")

    for reason in sorted(set(reasons)):
        ledger.append(_ledger_entry(row, ds, reason))
    return ValidationResult(not reasons, sorted(set(reasons)), ledger, image_type, corrected)


def _parse_date(value: Any) -> _dt.date | None:
    text = _as_text(value)
    if not text:
        return None
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return _dt.datetime.strptime(text[:10], fmt).date()
        except ValueError:
            continue
    return None


def _parse_time(value: Any) -> _dt.time | None:
    text = _as_text(value)
    if not text:
        return None
    text = text.split("+", 1)[0].split("-", 1)[0]
    if "." in text:
        main, frac = text.split(".", 1)
    else:
        main, frac = text, ""
    main = re.sub(r"[^0-9]", "", main)
    if not main:
        return None
    main = main.ljust(6, "0")[:6]
    frac = re.sub(r"[^0-9]", "", frac)[:6].ljust(6, "0") if frac else "000000"
    try:
        return _dt.time(
            hour=int(main[0:2]),
            minute=int(main[2:4]),
            second=int(main[4:6]),
            microsecond=int(frac),
        )
    except ValueError:
        return None


def _parse_datetime(value: Any) -> _dt.datetime | None:
    text = _as_text(value)
    if not text:
        return None
    try:
        parsed = pydicom.valuerep.DT(text)
        if isinstance(parsed, _dt.datetime):
            return parsed.replace(tzinfo=None)
    except Exception:
        pass
    compact = re.sub(r"[^0-9.]", "", text)
    if len(compact) < 8:
        return None
    date = _parse_date(compact[:8])
    time = _parse_time(compact[8:] or "000000")
    if date is None or time is None:
        return None
    return _dt.datetime.combine(date, time)


def _datetime_from_date_time(date_value: Any, time_value: Any) -> _dt.datetime | None:
    date = _parse_date(date_value)
    time = _parse_time(time_value)
    if date is None or time is None:
        return None
    return _dt.datetime.combine(date, time)


def resolve_timing(ds: Dataset, radiopharm: RadiopharmInfo) -> TimingInfo:
    flags: list[str] = []

    ref_dt = _datetime_from_date_time(getattr(ds, "SeriesDate", ""), getattr(ds, "SeriesTime", ""))
    ref_source = "SeriesDate+SeriesTime"
    if ref_dt is None:
        ref_dt = _datetime_from_date_time(getattr(ds, "SeriesDate", ""), getattr(ds, "AcquisitionTime", ""))
        ref_source = "SeriesDate+AcquisitionTime_fallback"
        if ref_dt is not None:
            flags.append("series_time_unusable_acquisition_time_fallback")

    inj_dt = _parse_datetime(radiopharm.start_datetime)
    inj_source = "RadiopharmaceuticalStartDateTime"
    if inj_dt is None:
        date_value = getattr(ds, "StudyDate", None) or getattr(ds, "SeriesDate", None)
        inj_dt = _datetime_from_date_time(date_value, radiopharm.start_time)
        inj_source = "StudyDate_or_SeriesDate+RadiopharmaceuticalStartTime"

    delta = None
    if ref_dt is not None and inj_dt is not None:
        delta = (ref_dt - inj_dt).total_seconds()
        if delta < 0:
            flags.append("negative_unresolved_delta_t")
        elif delta > 24 * 3600:
            flags.append("unrecoverable_timing_anomaly")
    else:
        flags.append("unrecoverable_timing_anomaly")

    return TimingInfo(
        injection_datetime=inj_dt,
        injection_datetime_source=inj_source,
        reference_datetime=ref_dt,
        reference_datetime_source=ref_source,
        delta_t_s=delta,
        flags=flags,
    )


def compute_decay_factor_guard(
    frame_reference_time_ms: Any,
    half_life_s: Any,
    decay_factor: Any,
    *,
    tolerance: float = 0.02,
) -> DecayGuardResult:
    frt_ms = _as_float(frame_reference_time_ms)
    half_life = _as_float(half_life_s)
    observed = _as_float(decay_factor)
    if frt_ms is None or half_life is None or half_life <= 0 or observed is None or observed <= 0:
        return DecayGuardResult(
            passed=False,
            expected_decay_factor=None,
            observed_decay_factor=observed,
            relative_difference=None,
            frame_reference_time_ms=frt_ms,
            half_life_s=half_life,
            tolerance=float(tolerance),
            reason="missing_or_invalid_decay_factor_inputs",
        )

    expected = 2 ** ((frt_ms / 1000.0) / half_life)
    try:
        unconverted = 2 ** (frt_ms / half_life)
    except OverflowError:
        unconverted = math.inf
    rel_diff = abs(expected - observed) / observed
    return DecayGuardResult(
        passed=rel_diff <= float(tolerance),
        expected_decay_factor=float(expected),
        observed_decay_factor=float(observed),
        relative_difference=float(rel_diff),
        frame_reference_time_ms=float(frt_ms),
        half_life_s=float(half_life),
        tolerance=float(tolerance),
        reason="" if rel_diff <= float(tolerance) else "decay_factor_guard_mismatch",
        unconverted_expected_decay_factor=float(unconverted) if math.isfinite(unconverted) else math.inf,
    )


def validate_decay_factor_guard(
    headers: list[Dataset],
    half_life_s: float,
    *,
    tolerance: float = 0.02,
) -> DecayGuardResult:
    if not headers:
        return compute_decay_factor_guard(None, half_life_s, None, tolerance=tolerance)
    worst: DecayGuardResult | None = None
    for ds in headers:
        result = compute_decay_factor_guard(
            getattr(ds, "FrameReferenceTime", None),
            half_life_s,
            getattr(ds, "DecayFactor", None),
            tolerance=tolerance,
        )
        if not result.passed:
            return result
        if worst is None or (result.relative_difference or 0.0) > (worst.relative_difference or 0.0):
            worst = result
    return worst or compute_decay_factor_guard(None, half_life_s, None, tolerance=tolerance)


def compute_decayed_activity_bq(total_dose_bq: float, half_life_s: float, delta_t_s: float) -> float:
    return float(total_dose_bq) * (2 ** (-float(delta_t_s) / float(half_life_s)))


def compute_suv_scale_factor(weight_kg: float, total_dose_bq: float, half_life_s: float, delta_t_s: float) -> float:
    decayed = compute_decayed_activity_bq(total_dose_bq, half_life_s, delta_t_s)
    if decayed <= 0:
        raise ValueError("Decayed activity must be positive")
    return float(weight_kg) * 1000.0 / decayed


def build_suv_volume(activity_bqml: np.ndarray, suv_scale_factor: float) -> np.ndarray:
    return np.asarray(activity_bqml, dtype=np.float64) * float(suv_scale_factor)


def _orientation(ds: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = getattr(ds, "ImageOrientationPatient", None)
    if raw is None or len(raw) != 6:
        row = np.array([1.0, 0.0, 0.0], dtype=float)
        col = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        vals = np.asarray([float(v) for v in raw], dtype=float)
        row = vals[:3]
        col = vals[3:]
    normal = np.cross(row, col)
    norm = np.linalg.norm(normal)
    if norm > 0:
        normal = normal / norm
    else:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
    return row, col, normal


def _position(ds: Dataset) -> np.ndarray | None:
    raw = getattr(ds, "ImagePositionPatient", None)
    if raw is None or len(raw) != 3:
        return None
    try:
        return np.asarray([float(v) for v in raw], dtype=float)
    except Exception:
        return None


def _pixel_spacing(ds: Dataset) -> tuple[float, float]:
    raw = getattr(ds, "PixelSpacing", None)
    if raw is None or len(raw) != 2:
        return 1.0, 1.0
    try:
        return float(raw[0]), float(raw[1])
    except Exception:
        return 1.0, 1.0


def _sort_datasets_by_geometry(datasets: list[Dataset]) -> list[Dataset]:
    if not datasets:
        return []
    _, _, normal = _orientation(datasets[0])

    def key(item: tuple[int, Dataset]) -> tuple[int, float, int]:
        idx, ds = item
        pos = _position(ds)
        if pos is not None:
            return (0, float(np.dot(pos, normal)), idx)
        inst = _as_float(getattr(ds, "InstanceNumber", None))
        if inst is not None:
            return (1, inst, idx)
        return (2, float(idx), idx)

    return [ds for _, ds in sorted(enumerate(datasets), key=key)]


def compute_physical_z_extent_mm(
    n_slices: int,
    slice_thickness: Any,
    ipp_positions: list[Any] | None = None,
    slice_normal: Any | None = None,
) -> float | None:
    thickness = _as_float(slice_thickness)
    if thickness is not None and thickness > 0 and n_slices > 0:
        return float(n_slices) * thickness
    if ipp_positions:
        normal = None
        if slice_normal is not None:
            try:
                normal = np.asarray([float(v) for v in slice_normal], dtype=float)
                norm = np.linalg.norm(normal)
                normal = normal / norm if norm > 0 else None
            except Exception:
                normal = None
        values: list[float] = []
        for pos in ipp_positions:
            try:
                arr = np.asarray([float(v) for v in pos], dtype=float)
            except Exception:
                continue
            values.append(float(np.dot(arr, normal)) if normal is not None else float(arr[2]))
        if len(values) >= 2:
            return max(values) - min(values)
    return None


def _affine_from_sorted_datasets(datasets: list[Dataset]) -> np.ndarray:
    first = datasets[0]
    row_cos, col_cos, normal = _orientation(first)
    row_spacing, col_spacing = _pixel_spacing(first)
    origin = _position(first)
    if origin is None:
        origin = np.zeros(3, dtype=float)

    positions = [_position(ds) for ds in datasets]
    real_positions = [pos for pos in positions if pos is not None]
    slice_vec: np.ndarray
    if len(real_positions) >= 2:
        diffs = [real_positions[i + 1] - real_positions[i] for i in range(len(real_positions) - 1)]
        slice_vec = np.mean(np.stack(diffs, axis=0), axis=0)
    else:
        thickness = _as_float(getattr(first, "SliceThickness", None)) or 1.0
        slice_vec = normal * thickness

    affine = np.eye(4, dtype=float)
    # Intermediate RAS affine for the raw pydicom pixel-array order
    # [row, column, slice].
    affine[:3, 0] = col_cos * row_spacing
    affine[:3, 1] = row_cos * col_spacing
    affine[:3, 2] = slice_vec
    affine[:3, 3] = origin
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine


def _to_dcm2niix_voxel_order(volume: np.ndarray, row_col_affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if volume.ndim != 3:
        raise ValueError(f"PET SUV volume must be 3D, got shape {volume.shape}")
    n_rows = volume.shape[0]
    reordered = volume.transpose(1, 0, 2)[:, ::-1, :]
    index_transform = np.array(
        [
            [0.0, -1.0, 0.0, float(n_rows - 1)],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return reordered, np.asarray(row_col_affine, dtype=float) @ index_transform


def build_activity_concentration_volume(dicom_paths: list[Path]) -> VolumeBuildResult:
    datasets = [pydicom.dcmread(str(path), force=True) for path in dicom_paths]
    datasets = _sort_datasets_by_geometry(datasets)
    if not datasets:
        raise ValueError("No DICOM instances available")

    slices: list[np.ndarray] = []
    scaling: list[dict[str, Any]] = []
    shape: tuple[int, ...] | None = None
    for ds in datasets:
        arr = ds.pixel_array.astype(np.float64)
        if shape is None:
            shape = arr.shape
        elif arr.shape != shape:
            raise ValueError(f"Inconsistent PET slice shape: {arr.shape} != {shape}")
        slope = _as_float(getattr(ds, "RescaleSlope", None))
        intercept = _as_float(getattr(ds, "RescaleIntercept", None))
        if slope is None:
            slope = 1.0
        if intercept is None:
            intercept = 0.0
        slices.append(arr * slope + intercept)
        scaling.append(
            {
                "sop_instance_uid": _as_text(getattr(ds, "SOPInstanceUID", "")),
                "instance_number": _as_text(getattr(ds, "InstanceNumber", "")),
                "rescale_slope": float(slope),
                "rescale_intercept": float(intercept),
            }
        )

    volume = np.stack(slices, axis=-1)
    affine = _affine_from_sorted_datasets(datasets)
    volume, affine = _to_dcm2niix_voxel_order(volume, affine)
    first = datasets[0]
    positions = [getattr(ds, "ImagePositionPatient", None) for ds in datasets]
    _, _, slice_normal = _orientation(first)
    z_extent = compute_physical_z_extent_mm(
        len(datasets),
        getattr(first, "SliceThickness", None),
        positions,
        slice_normal,
    )
    ipp_z_values = []
    for pos in positions:
        try:
            ipp_z_values.append(float(pos[2]))
        except Exception:
            continue

    geometry = {
        "n_slices": len(datasets),
        "rows": int(getattr(first, "Rows", volume.shape[1])),
        "columns": int(getattr(first, "Columns", volume.shape[0])),
        "pixel_spacing": list(_pixel_spacing(first)),
        "slice_thickness": _as_float(getattr(first, "SliceThickness", None)),
        "ipp_z_min": min(ipp_z_values) if ipp_z_values else None,
        "ipp_z_max": max(ipp_z_values) if ipp_z_values else None,
        "z_extent_mm": z_extent,
        "image_orientation_patient": [float(v) for v in getattr(first, "ImageOrientationPatient", []) or []],
    }
    return VolumeBuildResult(
        activity_bqml=volume,
        affine=affine,
        geometry=geometry,
        slice_scaling=scaling,
    )


def classify_recon_family(reconstruction_method: Any = "", series_description: Any = "") -> dict[str, Any]:
    sources = {
        "ReconstructionMethod": _as_text(reconstruction_method),
        "SeriesDescription": _as_text(series_description),
    }
    haystacks = {key: _normalize_text(value) for key, value in sources.items()}
    compact = {key: value.replace(" ", "") for key, value in haystacks.items()}

    dictionary = [
        ("BSREM", ["q clear", "qclear", "qchd", "mpdl qchd", "mpdlqchd"]),
        ("PSF", ["truex", "tof psf", "psf", "sharpir"]),
        ("FBP", ["fbp", "filteredbackprojection", "2d filteredbackprojection", "filtered back projection"]),
        ("OSEM-like", ["vue point hd", "vue point fx", "vphds", "vphd", "vpfxs", "blob os", "blob-os", "osem", "osem3d", "osem 3d", "osem2d", "ramla", "3d ramla"]),
    ]
    for family, tokens in dictionary:
        for source_name, text in haystacks.items():
            text_compact = compact[source_name]
            for token in tokens:
                token_norm = _normalize_text(token)
                token_compact = token_norm.replace(" ", "")
                if token_norm and token_norm in text:
                    return {
                        "family": family,
                        "tier": RECON_TIER[family],
                        "matched_token": token,
                        "matched_source": source_name,
                        "needs_manual_review": False,
                        "sources": sources,
                    }
                if token_compact and token_compact in text_compact:
                    return {
                        "family": family,
                        "tier": RECON_TIER[family],
                        "matched_token": token,
                        "matched_source": source_name,
                        "needs_manual_review": False,
                        "sources": sources,
                    }
    return {
        "family": "Unknown",
        "tier": None,
        "matched_token": "",
        "matched_source": "",
        "needs_manual_review": True,
        "sources": sources,
    }


def _clinical_entries(data: Any, name_predicate, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], dict[str, Any]]]:
    found: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    if isinstance(data, dict):
        for key, value in data.items():
            next_path = path + (str(key),)
            if isinstance(value, dict) and "value" in value and name_predicate(str(key).lower(), next_path):
                found.append((next_path, value))
            found.extend(_clinical_entries(value, name_predicate, next_path))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            found.extend(_clinical_entries(value, name_predicate, path + (str(index),)))
    return found


def _parse_source_date(value: Any) -> _dt.date | None:
    if value in (None, ""):
        return None
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    text = _as_text(value)
    return _parse_date(text[:10])


def _clinical_measurement(
    patient_id: str,
    study_date: _dt.date | None,
    *,
    key_predicate,
    window_days: int,
) -> tuple[float | None, str | None, str | None]:
    path = CLINICAL_ROOT / str(patient_id) / "extraction_canonical.json"
    if not path.exists() or study_date is None:
        return None, None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("Unable to read clinical fallback %s: %s", path, exc)
        return None, None, None

    candidates: list[tuple[int, float, str, str]] = []
    for key_path, node in _clinical_entries(data, key_predicate):
        value = _as_float(node.get("value"))
        if value is None or value <= 0:
            continue
        source_date = _parse_source_date(node.get("source_date"))
        if source_date is None:
            continue
        delta_days = abs((source_date - study_date).days)
        if delta_days <= int(window_days):
            candidates.append((delta_days, value, source_date.isoformat(), ".".join(key_path)))
    if not candidates:
        return None, None, None
    candidates.sort(key=lambda item: (item[0], item[3]))
    _, value, source_date, source_key = candidates[0]
    return float(value), source_date, source_key


def _pesel_from_other_patient_ids(ds: Dataset) -> str | None:
    text = _as_text(getattr(ds, "OtherPatientIDs", ""))
    matches = re.findall(r"(?<!\d)(\d{11})(?!\d)", text)
    return matches[0] if matches else None


def resolve_weight_and_height(
    ds: Dataset,
    patient_id: str,
    study_date: Any,
    config: PipelineConfig,
) -> WeightInfo:
    date = _parse_date(study_date)
    clinical_patient_id = _pesel_from_other_patient_ids(ds) or str(patient_id)
    weight = _as_float(getattr(ds, "PatientWeight", None))
    height = _as_float(getattr(ds, "PatientSize", None))
    height_source = "DICOM PatientSize" if height and height > 0 else None
    height_date = None
    if height is not None and height > 3.0:
        height = height / 100.0

    if height is None or height <= 0:
        height_cm, h_date, h_key = _clinical_measurement(
            clinical_patient_id,
            date,
            key_predicate=lambda key, path: "height_cm" in key or key == "height",
            window_days=getattr(config, "pet_clinical_weight_window_days", 30),
        )
        if height_cm is not None:
            height = height_cm / 100.0 if height_cm > 3.0 else height_cm
            height_source = f"clinical:{h_key}"
            height_date = h_date

    if weight is not None and weight > 0:
        return WeightInfo(
            weight_kg=float(weight),
            weight_source="DICOM PatientWeight",
            height_m=float(height) if height and height > 0 else None,
            height_source=height_source,
            height_source_date=height_date,
        )

    fallback_weight, source_date, source_key = _clinical_measurement(
        clinical_patient_id,
        date,
        key_predicate=lambda key, path: key in {"weight_kg", "baseline_weight_kg", "weight_kg_baseline"},
        window_days=getattr(config, "pet_clinical_weight_window_days", 30),
    )
    if fallback_weight is not None:
        return WeightInfo(
            weight_kg=fallback_weight,
            weight_source=f"clinical:{source_key}",
            weight_source_date=source_date,
            height_m=float(height) if height and height > 0 else None,
            height_source=height_source,
            height_source_date=height_date,
        )

    return WeightInfo(
        weight_kg=None,
        weight_source="missing",
        height_m=float(height) if height and height > 0 else None,
        height_source=height_source,
        height_source_date=height_date,
    )


def select_primary_recon(
    entries: list[dict[str, Any]],
    *,
    zextent_fraction: float = 0.90,
) -> list[dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    for entry in entries:
        entry["is_primary_recon"] = False

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in entries:
        compound = str(entry.get("tracer_compound") or "")
        study_uid = str(entry.get("study_uid") or "")
        if not compound or not study_uid:
            continue
        groups.setdefault((study_uid, compound), []).append(entry)

    for (_, _), group in groups.items():
        z_values = [float(item.get("z_extent_mm")) for item in group if _as_float(item.get("z_extent_mm")) is not None]
        z_max = max(z_values) if z_values else None
        eligible: list[dict[str, Any]] = []
        for item in group:
            flags = item.setdefault("primary_selection_flags", [])
            family = str(item.get("recon_family") or "Unknown")
            z_extent = _as_float(item.get("z_extent_mm"))
            if z_max is not None and z_extent is not None and z_extent < float(zextent_fraction) * z_max:
                flags.append("partial_axial_extent_for_primary_selection")
                ledger.append(
                    {
                        "patient_id": item.get("patient_id", ""),
                        "study_uid": item.get("study_uid", ""),
                        "series_uid": item.get("series_uid", ""),
                        "reason": "partial_axial_extent_for_primary_selection",
                        "severity": "manual_review",
                        "details": {"z_extent_mm": z_extent, "z_extent_max_mm": z_max},
                    }
                )
                continue
            tier = RECON_TIER.get(family)
            if tier is None:
                item["needs_manual_review"] = True
                flags.append("unknown_reconstruction_needing_manual_review")
                ledger.append(
                    {
                        "patient_id": item.get("patient_id", ""),
                        "study_uid": item.get("study_uid", ""),
                        "series_uid": item.get("series_uid", ""),
                        "reason": "unknown_reconstruction_needing_manual_review",
                        "severity": "manual_review",
                        "details": {"recon_family": family},
                    }
                )
                continue
            eligible.append(item)

        if not eligible:
            continue

        def rank(item: dict[str, Any]) -> tuple[int, float, float]:
            tier = RECON_TIER[str(item.get("recon_family"))]
            z_extent = _as_float(item.get("z_extent_mm")) or 0.0
            series_number = _as_float(item.get("series_number"))
            if series_number is None:
                series_number = math.inf
            return tier, -z_extent, series_number

        eligible.sort(key=rank)
        best = eligible[0]
        tied = [item for item in eligible if rank(item) == rank(best)]
        if len(tied) > 1:
            for item in tied:
                item["needs_manual_review"] = True
                item.setdefault("primary_selection_flags", []).append("primary_selection_tie_needs_manual_review")
                ledger.append(
                    {
                        "patient_id": item.get("patient_id", ""),
                        "study_uid": item.get("study_uid", ""),
                        "series_uid": item.get("series_uid", ""),
                        "reason": "primary_selection_tie_needs_manual_review",
                        "severity": "manual_review",
                        "details": {"rank": rank(item)},
                    }
                )
            continue
        best["is_primary_recon"] = True

    return ledger


def _dicom_paths(input_dir: Path) -> list[Path]:
    paths = sorted(path for path in input_dir.glob("*.dcm") if path.is_file())
    if paths:
        return paths
    return sorted(path for path in input_dir.iterdir() if path.is_file())


def _read_headers(paths: list[Path]) -> list[Dataset]:
    headers: list[Dataset] = []
    for path in paths:
        ds = read_dicom(path)
        if ds is not None:
            headers.append(ds)
    return headers


def _fatal_exclude(
    summary: dict[str, Any],
    row: dict[str, Any],
    ledger: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    reasons: list[str],
) -> None:
    row["status"] = "suv_excluded"
    row["pet_suv"] = {"status": "suv_excluded", "exclusion_reasons": sorted(set(reasons))}
    summary["excluded"] += 1
    for reason in sorted(set(reasons)):
        summary["per_exclusion_reason"][reason] += 1
    ledger.extend(entries)


def _write_suv_nifti(volume: np.ndarray, affine: np.ndarray, path: Path) -> None:
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(np.asarray(volume, dtype=np.float32), affine)
    nib.save(image, str(path))


def _series_number(ds: Dataset) -> float | None:
    return _as_float(getattr(ds, "SeriesNumber", None))


def _load_existing_provenance(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def ingest_pet_suv_for_patient(config: PipelineConfig, patient_id: str, *, force: bool = False) -> dict[str, Any]:
    patient_series_root = Path(config.output_root) / str(patient_id) / "all_series"
    course_dirs = build_course_dirs(patient_series_root)
    manifest_path = course_dirs.metadata / "series_manifest.json"
    summary: dict[str, Any] = {
        "patient_id": str(patient_id),
        "computed": 0,
        "excluded": 0,
        "failed": 0,
        "skipped": 0,
        "per_compound": Counter(),
        "per_exclusion_reason": Counter(),
    }
    if not manifest_path.exists():
        logger.info("All-series manifest not found for patient %s at %s; skipping PET SUV", patient_id, manifest_path)
        summary["per_compound"] = {}
        summary["per_exclusion_reason"] = {}
        return summary

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Unable to read all-series manifest for patient %s: %s", patient_id, exc)
        summary["failed"] += 1
        summary["per_compound"] = {}
        summary["per_exclusion_reason"] = {}
        return summary

    rows = manifest.get("series", [])
    if not isinstance(rows, list):
        logger.warning("All-series manifest for patient %s has no series list; skipping PET SUV", patient_id)
        summary["per_compound"] = {}
        summary["per_exclusion_reason"] = {}
        return summary

    ledger: list[dict[str, Any]] = []
    pet_entries: list[dict[str, Any]] = []
    provenance_paths: list[tuple[dict[str, Any], Path, dict[str, Any]]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("image_class") or "").lower() != "pt":
            continue
        status = str(row.get("status") or "")
        if status not in PET_SUV_CANDIDATE_STATUSES:
            continue
        input_dir_text = str(row.get("output_dir") or "")
        if not input_dir_text:
            _fatal_exclude(
                summary,
                row,
                ledger,
                [_ledger_entry(row, None, "missing_pet_output_dir")],
                ["missing_pet_output_dir"],
            )
            continue

        input_dir = Path(input_dir_text)
        if not input_dir.exists():
            _fatal_exclude(
                summary,
                row,
                ledger,
                [_ledger_entry(row, None, "missing_pet_output_dir", details={"output_dir": str(input_dir)})],
                ["missing_pet_output_dir"],
            )
            continue

        paths = _dicom_paths(input_dir)
        headers = _read_headers(paths)
        ds0 = headers[0] if headers else None
        validation = validate_pet_series(headers, row)
        if not validation.valid:
            _fatal_exclude(summary, row, ledger, validation.ledger, validation.reasons)
            continue

        assert ds0 is not None
        fatal_reasons: list[str] = []
        fatal_entries: list[dict[str, Any]] = []

        radiopharm = extract_radiopharmaceutical(ds0)
        tracer = resolve_compound(
            radiopharm.radiopharmaceutical,
            radiopharm.radionuclide_code_meaning,
            radiopharm.half_life_s,
            study_description=getattr(ds0, "StudyDescription", ""),
            series_description=getattr(ds0, "SeriesDescription", ""),
        )
        if tracer.needs_manual_review or not tracer.compound:
            fatal_reasons.append("ambiguous_tracer_compound")
            fatal_entries.append(
                _ledger_entry(
                    row,
                    ds0,
                    "ambiguous_tracer_compound",
                    severity="manual_review",
                    details={"source_text": tracer.source_text, "isotope": tracer.isotope},
                )
            )

        if radiopharm.total_dose_bq is None or radiopharm.total_dose_bq <= 0:
            fatal_reasons.append("missing_invalid_dose")
            fatal_entries.append(_ledger_entry(row, ds0, "missing_invalid_dose"))
        if radiopharm.half_life_s is None or radiopharm.half_life_s <= 0:
            fatal_reasons.append("missing_invalid_half_life")
            fatal_entries.append(_ledger_entry(row, ds0, "missing_invalid_half_life"))

        weight = resolve_weight_and_height(ds0, str(patient_id), getattr(ds0, "StudyDate", ""), config)
        if weight.weight_kg is None or weight.weight_kg <= 0:
            fatal_reasons.append("missing_invalid_weight")
            fatal_entries.append(_ledger_entry(row, ds0, "missing_invalid_weight"))

        timing = resolve_timing(ds0, radiopharm)
        if "negative_unresolved_delta_t" in timing.flags:
            fatal_reasons.append("negative_unresolved_delta_t")
            fatal_entries.append(_ledger_entry(row, ds0, "negative_unresolved_delta_t"))
        elif "unrecoverable_timing_anomaly" in timing.flags:
            fatal_reasons.append("unrecoverable_timing_anomaly")
            fatal_entries.append(_ledger_entry(row, ds0, "unrecoverable_timing_anomaly"))

        if (
            tracer.compound == "F18_FDG"
            and timing.delta_t_s is not None
            and 0 <= timing.delta_t_s < 1800
            and "fdg_uptake_time_under_1800s" not in timing.flags
        ):
            timing.flags.append("fdg_uptake_time_under_1800s")

        guard = None
        if radiopharm.half_life_s is not None and radiopharm.half_life_s > 0:
            guard = validate_decay_factor_guard(
                headers,
                radiopharm.half_life_s,
                tolerance=getattr(config, "suv_decay_guard_tol", 0.02),
            )
            if not guard.passed:
                fatal_reasons.append("failed_decay_factor_guard")
                fatal_entries.append(
                    _ledger_entry(
                        row,
                        ds0,
                        "failed_decay_factor_guard",
                        details={
                            "reason": guard.reason,
                            "expected_decay_factor": guard.expected_decay_factor,
                            "observed_decay_factor": guard.observed_decay_factor,
                            "relative_difference": guard.relative_difference,
                            "frame_reference_time_ms": guard.frame_reference_time_ms,
                        },
                    )
                )

        if fatal_reasons:
            _fatal_exclude(summary, row, ledger, fatal_entries, fatal_reasons)
            continue

        safe_series = _safe_token(row.get("series_uid") or getattr(ds0, "SeriesInstanceUID", ""), "pt_series")
        out_dir = patient_series_root / "NIFTI" / "SUV" / safe_series
        suv_path = out_dir / f"{safe_series}_SUVbw.nii.gz"
        prov_path = out_dir / f"{safe_series}_SUVbw.provenance.json"
        existing_prov = _load_existing_provenance(prov_path)
        if not force and suv_path.exists() and existing_prov:
            row["status"] = "suv_skipped_idempotent"
            row["pet_suv"] = {
                "status": "suv_skipped_idempotent",
                "suv_nifti": str(suv_path),
                "provenance": str(prov_path),
                "tracer_compound": existing_prov.get("tracer_compound"),
                "is_primary_recon": bool(existing_prov.get("is_primary_recon", False)),
            }
            summary["skipped"] += 1
            compound = str(existing_prov.get("tracer_compound") or "")
            if compound:
                summary["per_compound"][compound] += 1
            pet_entries.append(existing_prov)
            provenance_paths.append((existing_prov, prov_path, row))
            continue

        try:
            built = build_activity_concentration_volume(paths)
            scale = compute_suv_scale_factor(
                weight.weight_kg,
                radiopharm.total_dose_bq,
                radiopharm.half_life_s,
                timing.delta_t_s,
            )
            suv = build_suv_volume(built.activity_bqml, scale)
            _write_suv_nifti(suv, built.affine, suv_path)

            recon = classify_recon_family(
                getattr(ds0, "ReconstructionMethod", ""),
                getattr(ds0, "SeriesDescription", ""),
            )
            provenance = {
                "patient_id": str(patient_id),
                "study_uid": str(row.get("study_uid") or getattr(ds0, "StudyInstanceUID", "")),
                "series_uid": str(row.get("series_uid") or getattr(ds0, "SeriesInstanceUID", "")),
                "suv_nifti": str(suv_path),
                "source_dicom_dir": str(input_dir),
                "tracer_compound": tracer.compound,
                "tracer_isotope": tracer.isotope,
                "tracer_isotope_source": tracer.isotope_source,
                "tracer_label_family": tracer.label_family,
                "tracer_source": tracer.tracer_source,
                "tracer_source_text": tracer.source_text,
                "tracer_flags": tracer.flags,
                "radiopharmaceutical": _as_text(radiopharm.radiopharmaceutical),
                "radiopharmaceutical_sources": radiopharm.sources,
                "radionuclide_code_meaning": radiopharm.radionuclide_code_meaning,
                "total_dose_bq": radiopharm.total_dose_bq,
                "half_life_s": radiopharm.half_life_s,
                "patient_weight_kg": weight.weight_kg,
                "patient_weight_g": weight.weight_kg * 1000.0,
                "patient_weight_source": weight.weight_source,
                "patient_weight_source_date": weight.weight_source_date,
                "patient_height_m": weight.height_m,
                "patient_height_source": weight.height_source,
                "patient_height_source_date": weight.height_source_date,
                "series_date": _as_text(getattr(ds0, "SeriesDate", "")),
                "series_time": _as_text(getattr(ds0, "SeriesTime", "")),
                "acquisition_time": _as_text(getattr(ds0, "AcquisitionTime", "")),
                "injection_datetime": timing.injection_datetime,
                "injection_datetime_source": timing.injection_datetime_source,
                "reference_datetime": timing.reference_datetime,
                "reference_datetime_source": timing.reference_datetime_source,
                "delta_t_s": timing.delta_t_s,
                "timing_flags": timing.flags,
                "frame_reference_time_ms": _as_float(getattr(ds0, "FrameReferenceTime", None)),
                "decay_factor": _as_float(getattr(ds0, "DecayFactor", None)),
                "decay_correction": _as_text(getattr(ds0, "DecayCorrection", "")),
                "decay_factor_guard": guard,
                "decayed_activity_bq": compute_decayed_activity_bq(
                    radiopharm.total_dose_bq,
                    radiopharm.half_life_s,
                    timing.delta_t_s,
                ),
                "suv_scale_factor": scale,
                "geometry": built.geometry,
                "z_extent_mm": built.geometry.get("z_extent_mm"),
                "slice_scaling": built.slice_scaling,
                "rescale_slope_unique_count": len({item["rescale_slope"] for item in built.slice_scaling}),
                "recon_family": recon["family"],
                "recon_tier": recon["tier"],
                "recon_dictionary_match": {
                    "matched_token": recon["matched_token"],
                    "matched_source": recon["matched_source"],
                    "sources": recon["sources"],
                },
                "series_number": _series_number(ds0),
                "is_primary_recon": False,
                "needs_manual_review": bool(recon["needs_manual_review"]),
                "primary_selection_flags": [],
                "image_type": validation.image_type_tokens,
                "corrected_image": validation.corrected_image_tokens,
                "manufacturer": _as_text(getattr(ds0, "Manufacturer", "")),
                "manufacturer_model": str(row.get("manufacturer_model") or getattr(ds0, "ManufacturerModelName", "") or ""),
                "generated_at": _dt.datetime.now(_dt.timezone.utc),
            }
            if recon["needs_manual_review"]:
                ledger.append(
                    _ledger_entry(
                        row,
                        ds0,
                        "unknown_reconstruction_needing_manual_review",
                        severity="manual_review",
                        details={"reconstruction_method": recon["sources"].get("ReconstructionMethod", "")},
                    )
                )
            _write_json(prov_path, provenance)

            row["status"] = "suv_computed"
            row["pet_suv"] = {
                "status": "suv_computed",
                "suv_nifti": str(suv_path),
                "provenance": str(prov_path),
                "tracer_compound": tracer.compound,
                "is_primary_recon": False,
                "needs_manual_review": bool(recon["needs_manual_review"]),
            }
            summary["computed"] += 1
            summary["per_compound"][tracer.compound] += 1
            pet_entries.append(provenance)
            provenance_paths.append((provenance, prov_path, row))
        except Exception as exc:
            logger.warning(
                "PET SUV ingestion failed for patient %s series %s: %s",
                patient_id,
                row.get("series_uid", ""),
                exc,
            )
            row["status"] = "suv_failed"
            row["pet_suv"] = {"status": "suv_failed", "error": str(exc)}
            summary["failed"] += 1
            ledger.append(_ledger_entry(row, ds0, "suv_failed", details={"error": str(exc)}))

    selection_ledger = select_primary_recon(
        pet_entries,
        zextent_fraction=getattr(config, "suv_zextent_primary_fraction", 0.90),
    )
    ledger.extend(selection_ledger)
    for reason in (entry.get("reason") for entry in selection_ledger):
        if reason:
            summary["per_exclusion_reason"][str(reason)] += 1

    for provenance, prov_path, row in provenance_paths:
        _write_json(prov_path, provenance)
        pet_suv = row.setdefault("pet_suv", {})
        pet_suv["is_primary_recon"] = bool(provenance.get("is_primary_recon", False))
        pet_suv["needs_manual_review"] = bool(provenance.get("needs_manual_review", False))
        pet_suv["primary_selection_flags"] = list(provenance.get("primary_selection_flags", []))

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    pet_manifest = {
        "patient_id": str(patient_id),
        "generated_at": _dt.datetime.now(_dt.timezone.utc),
        "summary": {
            **{k: v for k, v in summary.items() if k not in {"per_compound", "per_exclusion_reason"}},
            "per_compound": dict(summary["per_compound"]),
            "per_exclusion_reason": dict(summary["per_exclusion_reason"]),
        },
        "series": pet_entries,
    }
    exclusions = {
        "patient_id": str(patient_id),
        "generated_at": _dt.datetime.now(_dt.timezone.utc),
        "ledger": ledger,
    }
    _write_json(course_dirs.metadata / "pet_suv_manifest.json", pet_manifest)
    _write_json(course_dirs.metadata / "pet_suv_exclusions.json", exclusions)

    summary["per_compound"] = dict(summary["per_compound"])
    summary["per_exclusion_reason"] = dict(summary["per_exclusion_reason"])
    return summary


__all__ = [
    "CLINICAL_ROOT",
    "RadiopharmInfo",
    "TracerResolution",
    "ValidationResult",
    "WeightInfo",
    "build_activity_concentration_volume",
    "build_suv_volume",
    "classify_recon_family",
    "compute_decay_factor_guard",
    "compute_decayed_activity_bq",
    "compute_physical_z_extent_mm",
    "compute_suv_scale_factor",
    "extract_radiopharmaceutical",
    "ingest_pet_suv_for_patient",
    "resolve_compound",
    "resolve_timing",
    "resolve_weight_and_height",
    "select_primary_recon",
    "validate_decay_factor_guard",
    "validate_pet_series",
]
