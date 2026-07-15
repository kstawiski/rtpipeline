from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


IMAGE_CLASSES = {
    "planning_ct",
    "diagnostic_ct",
    "petct_ct",
    "cbct",
    "fourdct_ave",
    "fourdct_phase",
    "mr_anatomic",
    "mr_functional",
    "pt",
    "exclude",
}

CBCT_MANUFACTURER_MODELS = frozenset(
    {
        "patient verification",
        "halcyon - pva",
        "obi cone-beam ct",
        "acuity cone-beam ct",
        "rds - pva",
    }
)
FOURDCT_MODELS = frozenset({"advanced reconstruction", "aria rtm"})

_DESC_EXCLUDE_PATTERNS = (
    ("description_topogram", re.compile(r"\btopogram\b", re.I)),
    ("description_scout", re.compile(r"\bscout\b", re.I)),
    ("description_surview", re.compile(r"\bsurview\b", re.I)),
    ("description_dose_report", re.compile(r"\bdose\s+report\b", re.I)),
    ("description_patient_protocol", re.compile(r"\bpatient\s+protocol\b", re.I)),
    ("description_oncology_reading", re.compile(r"\boncology\s+reading\b", re.I)),
    ("description_results_mm_oncology", re.compile(r"\bresults\s+mm\s+oncology\b", re.I)),
)
_LOCALIZER_DESC_RE = re.compile(r"\b(locali[sz]er|locator|scout|topogram|surview)\b", re.I)
_MIP_PROJECTION_RE = re.compile(r"\bmip\s*\(", re.I)
_FOURDCT_PROJECTION_TOKENS = frozenset({"mip", "maxip", "minip"})
_FOURDCT_AVE_TOKENS = frozenset({"ave", "avg", "average"})
_VARIAN_PROJ_PAREN_RE = re.compile(r"\b(min|max)\s*\(", re.I)
_PHASE_RE = re.compile(r"\bt\s*=\s*\d{1,3}\s*%|\b\d{1,3}\s*%\b|\bresp", re.I)
_PT_PROJECTION_RE = re.compile(r"\b(mip|maxip|minip)\b", re.I)
_PT_REPORT_RE = re.compile(r"dose\s*report|statistic|screen\s*save|patient\s*protocol", re.I)
# MR sequence classification operates on NORMALIZED tokens (lowercase alphanumeric
# runs) rather than the raw description, so underscore/hyphen-delimited vendor names
# are matched correctly. A literal trailing \b fails before '_' (an underscore is a
# word character), which can drop series such as DWI_b1500, T2W_TSE_sag, or
# t2_blade_sag. The allow-lists span common Siemens (tse/spc/tirm/blade/me2d/mpr/vibe),
# Philips (T1W/T2W_TSE, FFE), and GE (FSE/frFSE) naming conventions.
_MR_EXCLUDE_TOKENS = frozenset({
    "localizer", "localiser", "locator", "loc", "scout", "topogram",
    "surview", "survey", "calibration", "cal", "posdisp",
})
# dwi/adc/dixon are matched as substrings (not \b-anchored) because vendors embed
# them inside single tokens with no separator (IsoDWI, cDWI, mDIXON) — anchoring would
# silently drop them. They are unambiguous in MR naming, so substring matching is safe.
_MR_FUNCTIONAL_RE = re.compile(
    r"dwi|\bdiff|\bep2d|adc|\bperf|\btwist\b|\bdyn|\bdce\b|"
    r"\bttp\b|\bpei\b|\bmipt\b",
    re.I,
)
_MR_ANATOMIC_SEQ_RE = re.compile(
    r"dixon|\b(tse|fse|frfse|frse|spc|space|tirm|stir|flair|blade|vibe|"
    r"mpr|mprage|me2d|haste|trufi|truefisp|ffe)\b",
    re.I,
)
_MR_WEIGHT_TOKENS = frozenset({"t1", "t2", "t1w", "t2w", "t1wi", "t2wi", "pd", "pdw"})
_MR_PLANNING_TOKENS = frozenset({"plan", "staging"})
_CALIBRATED_CT_MANUFACTURER_RE = re.compile(
    r"siemens|ge medical|philips|toshiba|canon|united imaging|uih|nms|neusoft|hitachi",
    re.I,
)
_CALIBRATED_CT_MODEL_RE = re.compile(
    r"sensation|somatom|biograph|discovery|gemini|umi|aquilion|brilliance|"
    r"ingenuity|optima|revolution|lightspeed|big\s*bore",
    re.I,
)


def classify_series(meta: Mapping[str, Any]) -> tuple[str, str | None]:
    """Classify one inventory series for the all-series modality expansion.

    Returns ``(image_class, exclusion_reason)``. Non-excluded classes return
    ``None`` for the reason. Unknown classes fail closed as ``exclude``.
    """
    modality = _text(meta, "modality").upper()
    if modality == "PT":
        return _classify_pt(meta)
    if modality == "MR":
        return _classify_mr(meta)
    if modality == "CT":
        return _classify_ct(meta)
    if not modality:
        return "exclude", "missing_modality_default_deny"
    return "exclude", f"unsupported_modality_{_machine_token(modality)}"


def _classify_ct(meta: Mapping[str, Any]) -> tuple[str, str | None]:
    exclusion = _common_image_exclusion(meta, include_mip=True)
    if exclusion:
        return "exclude", exclusion

    description = _text(meta, "series_description")
    manufacturer = _text(meta, "manufacturer")
    model = _text(meta, "manufacturer_model")
    model_norm = model.strip().lower()

    if model_norm in _configured_set(meta, "fourdct_models", FOURDCT_MODELS):
        return _classify_4dct(meta, by_model=True)

    if _is_4dct_description(description) or _is_4dct_image_type(meta):
        return _classify_4dct(meta, by_model=False)

    if (
        manufacturer.strip().lower() == "varian medical systems"
        and model_norm in _configured_set(meta, "cbct_manufacturer_models", CBCT_MANUFACTURER_MODELS)
    ):
        return "cbct", None

    if bool(meta.get("has_pt_same_study_for") or meta.get("petct_ct_candidate")):
        return "petct_ct", None

    if _is_calibrated_ct(manufacturer, model):
        if bool(meta.get("is_planning_ct", meta.get("rt_linked") or meta.get("rtstruct_linked"))):
            return "planning_ct", None
        return "diagnostic_ct", None

    # RTSTRUCT-bound recovery: a series an RTSTRUCT was directly contoured on is a
    # real planning/diagnostic CT even when the manufacturer/model is a TPS/registration
    # export (Plastimatch, Velocity, ART-Plan, Ethos…). This runs AFTER every junk gate
    # (n<10, localizer, projection, report) and the calibrated-vendor check, so only
    # genuine volumetric CT reaches it. Non-RT-linked unknown vendors stay default-deny.
    if bool(meta.get("is_planning_ct")):
        return "planning_ct", None
    if bool(meta.get("rt_series_linked")):
        return "diagnostic_ct", None
    if _image_type_contains(meta, "SECONDARY") and not _image_type_contains(meta, "PRIMARY"):
        return "exclude", "ct_derived_secondary"
    return "exclude", "ct_unrecognized_default_deny"


def _classify_pt(meta: Mapping[str, Any]) -> tuple[str, str | None]:
    n_instances = _int_or_none(meta.get("n_instances", meta.get("n_slices")))
    if n_instances is not None and n_instances < 10:
        return "exclude", "pt_sub_volumetric_lt10"
    description = _text(meta, "series_description")
    if _PT_REPORT_RE.search(description):
        return "exclude", "pt_report_or_derived"
    if _image_type_contains(meta, "SECONDARY") and not _image_type_contains(meta, "PRIMARY"):
        return "exclude", "pt_secondary_capture"
    if (
        _PT_PROJECTION_RE.search(description)
        or _image_type_contains(meta, "MIP")
        or _image_type_contains(meta, "MAXIP")
        or _image_type_contains(meta, "MINIP")
    ):
        # MIP/MaxIP/MinIP are non-anatomic rotating-cine projections, not SUV-quantifiable
        # reconstructions; keep them out of PET SUV ingestion and radiomics.
        return "exclude", "pt_projection_mip"
    return "pt", None


def _classify_mr(meta: Mapping[str, Any]) -> tuple[str, str | None]:
    exclusion = _common_image_exclusion(meta, include_mip=False)
    if exclusion:
        return "exclude", exclusion

    description = _text(meta, "series_description")
    tokens = re.findall(r"[a-z0-9]+", description.lower())
    tokset = set(tokens)
    norm = " ".join(tokens)

    # Non-imaging / geometric: localizers, scouts, surveys, calibration, derived
    # position-display screenshots. Checked first so they never reach a quantitative class.
    hit = tokset & _MR_EXCLUDE_TOKENS
    if hit:
        return "exclude", f"mr_nonanatomic_{sorted(hit)[0]}"

    # Functional first (DWI/ADC/DCE-perfusion); these must not be routed to total_mr.
    if _MR_FUNCTIONAL_RE.search(norm):
        return "mr_functional", None

    # Structural/anatomic: explicit sequence family, T1/T2/PD weighting, or the
    # Polish planning/staging naming used by some rectal MR-simulation protocols.
    if (
        _MR_ANATOMIC_SEQ_RE.search(norm)
        or (tokset & _MR_WEIGHT_TOKENS)
        or ("odbytnica" in tokset and (tokset & _MR_PLANNING_TOKENS))
    ):
        return "mr_anatomic", None

    # A DERIVED\SECONDARY MR with no anatomic/functional tokens (for example a
    # treatment-system re-export) receives a specific, auditable reason instead of the
    # generic default-deny and remains excluded from total_mr.
    if _image_type_contains(meta, "SECONDARY") and not _image_type_contains(meta, "PRIMARY"):
        return "exclude", "mr_derived_secondary"
    return "exclude", "mr_unrecognized_default_deny"


def _classify_4dct(meta: Mapping[str, Any], *, by_model: bool) -> tuple[str, str | None]:
    description = _text(meta, "series_description")
    tokset = set(re.findall(r"[a-z0-9]+", description.lower()))
    if (
        (tokset & _FOURDCT_PROJECTION_TOKENS)
        or _VARIAN_PROJ_PAREN_RE.search(description)
        or _image_type_contains(meta, "MIP")
        or _image_type_contains(meta, "MAXIP")
        or _image_type_contains(meta, "MINIP")
    ):
        return "exclude", "fourdct_projection"
    if (tokset & _FOURDCT_AVE_TOKENS) or _image_type_contains(meta, "AVE"):
        return "fourdct_ave", None
    if _PHASE_RE.search(description):
        return "fourdct_phase", None
    if by_model:
        return "fourdct_phase", None
    return "exclude", "fourdct_unrecognized_default_deny"


def _common_image_exclusion(meta: Mapping[str, Any], *, include_mip: bool) -> str | None:
    n_instances = _int_or_none(meta.get("n_instances", meta.get("n_slices")))
    if n_instances is not None and n_instances < 10:
        return "sub_volumetric_lt10"

    if _image_type_contains(meta, "LOCALIZER") and not (
        _image_type_contains(meta, "AXIAL") or _image_type_contains(meta, "HELICAL")
    ):
        # A LOCALIZER instance only excludes when the series carries no axial/helical
        # acquisition slices. Vendor CBCT (e.g. Acuity) ship an AXIAL volume + 1 stray
        # LOCALIZER instance; the series-level image_type set must not drop the volume.
        return "image_type_localizer"

    description = _text(meta, "series_description")
    if _LOCALIZER_DESC_RE.search(description):
        return "description_localizer"
    if include_mip and _MIP_PROJECTION_RE.search(description):
        return "description_mip_projection"
    for reason, pattern in _DESC_EXCLUDE_PATTERNS:
        if pattern.search(description):
            return reason

    if _image_type_contains(meta, "DERIVED") and n_instances is None:
        return "non_volumetric_derived_missing_count"

    return None


def _is_4dct_description(description: str) -> bool:
    tokset = set(re.findall(r"[a-z0-9]+", description.lower()))
    if tokset & (_FOURDCT_AVE_TOKENS | _FOURDCT_PROJECTION_TOKENS):
        return True
    if _VARIAN_PROJ_PAREN_RE.search(description):
        return True
    return bool(_PHASE_RE.search(description))


def _is_4dct_image_type(meta: Mapping[str, Any]) -> bool:
    return any(_image_type_contains(meta, t) for t in ("AVE", "MIP", "MAXIP", "MINIP"))


def _is_calibrated_ct(manufacturer: str, model: str) -> bool:
    return bool(
        _CALIBRATED_CT_MANUFACTURER_RE.search(manufacturer)
        or _CALIBRATED_CT_MODEL_RE.search(model)
    )


def _configured_set(meta: Mapping[str, Any], key: str, fallback: frozenset[str]) -> set[str]:
    raw = meta.get(key)
    if not raw:
        return set(fallback)
    if isinstance(raw, str):
        values = [raw]
    else:
        values = list(raw) if isinstance(raw, Sequence) else []
    return {str(value).strip().lower() for value in values if str(value).strip()}


def _image_type_contains(meta: Mapping[str, Any], needle: str) -> bool:
    needle = needle.upper()
    for value in _image_type_values(meta.get("image_type", meta.get("image_types"))):
        if needle in value.upper():
            return True
    return False


def _image_type_values(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, Sequence):
        return [str(item) for item in raw if item is not None]
    return [str(raw)]


def _text(meta: Mapping[str, Any], key: str) -> str:
    value = meta.get(key)
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(item) for item in value if item is not None).strip()
    return str(value).strip()


def _int_or_none(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _machine_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return token or "unknown"


__all__ = ["IMAGE_CLASSES", "classify_series"]
