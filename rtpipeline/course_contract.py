from __future__ import annotations

"""Authoritative per-course decisions emitted by :mod:`rtpipeline.organize`.

Downstream course stages load this contract and validate its declared artifacts.
They must not recover a missing or invalid decision by scanning the course tree.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import pydicom
from pydicom.dataset import Dataset


COURSE_CONTRACT_VERSION = 4
PLAN_LEVEL_DOSE_SUMMATION_TYPES = frozenset({"PLAN", "PLAN_SUM", "MULTI_PLAN"})
BEAM_LEVEL_DOSE_SUMMATION_TYPES = frozenset({"BEAM"})
DOSE_GRID_SEMANTICS = "planned_rtdose_weighted_by_rtrecord_delivered_fraction_counts"
UNKNOWN_DELIVERY_DOSE_GRID_SEMANTICS = "planned_dose_for_selected_plan_set_delivery_unknown"
DOSE_RESPONSE_FIELD = "delivered_dose_gy"
DOSE_RESPONSE_ELIGIBILITY_BASIS = "verified_complete_course_dose_and_rtrecord_delivery_evidence"
DOSE_COMPLETENESS_SCHEMA_VERSION = 1
DOSE_COMPLETENESS_ELIGIBLE_STATUS = "eligible"
DOSE_COMPLETENESS_NOT_DEFENSIBLE_STATUS = "not_defensible"
ALL_SERIES_RADIOMICS_TEMP_SCOPE = "all_series_radiomics_temp"
ALL_SERIES_RADIOMICS_TEMP_AUTHORITY = "all_series_radiomics_materializer"
MANUAL_RTSTRUCT_SOURCE = "Manual"
AUTO_RTSTRUCT_SOURCE = "AutoRTS_total"

from .plan_profiles import (
    DERIVED_RTPLAN_SOP_CLASSES,
    SOURCE_RTPLAN_SOP_CLASSES,
    plan_profile_name,
)
from .clinical_prescription import (
    CLINICAL_EVIDENCE_REGENERATION_SCHEMA,
    CLINICAL_EVIDENCE_SCHEMA,
    CLINICAL_RESOLVED_SCOPE,
    clinical_evidence_content_sha256,
    confirm_two_phase_fractionation,
    parse_kopernik_treatment_description,
)
from .prescription import (
    PRESCRIPTION_GROUP_FIELDS,
    aggregate_course_prescription_values,
    resolve_plan_prescriptions,
    resolved_plan_total_gy,
    source_plan_prescribed_dose_gy,
)

_ROLE_EXPECTATIONS = {
    # A plan exported by the treatment system may use a governed vendor profile;
    # a plan this pipeline synthesises may not claim one. See plan_profiles.
    "RTPLAN": ({"RTPLAN"}, set(SOURCE_RTPLAN_SOP_CLASSES)),
    "RTPLAN_SOURCE": ({"RTPLAN"}, set(SOURCE_RTPLAN_SOP_CLASSES)),
    "RTPLAN_DERIVED": ({"RTPLAN"}, set(DERIVED_RTPLAN_SOP_CLASSES)),
    "RTDOSE": ({"RTDOSE"}, {"1.2.840.10008.5.1.4.1.1.481.2"}),
    "RTSTRUCT": ({"RTSTRUCT"}, {"1.2.840.10008.5.1.4.1.1.481.3"}),
    "CT": (
        {"CT"},
        {
            "1.2.840.10008.5.1.4.1.1.2",  # CT Image Storage
            "1.2.840.10008.5.1.4.1.1.2.1",  # Enhanced CT Image Storage
        },
    ),
}


class CourseContractError(RuntimeError):
    """The organize-stage course contract is missing, malformed, or stale."""


def build_dvh_decision(
    selected_plan_count: int,
    selected_dose_count: int,
    delivery_status: str,
    *,
    dose_response_eligible: bool = True,
    dose_completeness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe whether DVH can compute metrics from the contracted dose sources."""
    if selected_dose_count:
        result = {
            "status": "ready",
            "metrics_status": "computed",
            "reason_code": "authoritative_dose_grid",
            "dose_record_status": "authoritative_rtdose_selected",
            "output": "dvh_metrics.xlsx",
            "delivery_status": delivery_status,
            "reason": "The contract contains an authoritative RTDOSE grid for DVH.",
            "dose_response_eligible": bool(dose_response_eligible),
        }
        if not dose_response_eligible:
            completeness_status = (dose_completeness or {}).get("status")
            completeness_failed = (
                completeness_status != DOSE_COMPLETENESS_ELIGIBLE_STATUS
            )
            result.update(
                {
                    "reason_code": (
                        "dose_response_course_dose_incomplete"
                        if completeness_failed
                        else "dose_response_other_requirements_unresolved"
                    ),
                    "reason": (
                        str(
                            (dose_completeness or {}).get("reason")
                            or "The dose grid is retained for QC, but complete course-level dose is not established for dose-response analysis."
                        )
                        if completeness_failed
                        else "A verified complete course-dose grid is available, but another dose-response requirement such as prescription or delivery status is unresolved."
                    ),
                    "dose_completeness_status": completeness_status,
                    "dose_completeness_reason_code": (dose_completeness or {}).get("reason_code"),
                }
            )
        return result
    if selected_plan_count:
        reason_code = "plan_only_no_authoritative_dose_grid"
    else:
        reason_code = "no_selected_plan_or_dose_grid"
    dose_record_status = (
        "delivery_unknown_no_rtrecord"
        if delivery_status == "no_records_at_all"
        else "no_authoritative_rtdose_selected"
    )
    return {
        "status": "not_computed",
        "metrics_status": "not_computed",
        "reason_code": reason_code,
        "dose_record_status": dose_record_status,
        "output": None,
        "delivery_status": delivery_status,
        "reason": (
            "Organize retained the contracted plan membership but selected no RTDOSE. "
            "DVH emits no dose metrics and records this reason in metadata."
            if selected_plan_count
            else "Organize selected no plan or RTDOSE sources. DVH emits no dose metrics and records this reason in metadata."
        ),
    }


def classify_course_dose_completeness(
    *,
    selected_plans: Iterable[dict[str, Any]],
    selected_doses: Iterable[dict[str, Any]],
    dose_classification: dict[str, Any],
    dose_grid: dict[str, Any] | None,
    per_plan_delivery: Iterable[dict[str, Any]],
    delivery_status: str,
    spatial_mapping_validated: bool | None = None,
) -> dict[str, Any]:
    """Classify whether one course has defensible complete dose evidence.

    The expected plan set is the organizer's selected, revision-aware course
    membership. A plan selected by date-only fallback while another distinct
    plan has treatment records is not accepted. A plan sum is accepted only
    when each selected plan has a positive, bounded delivered-fraction weight.
    This function deliberately does not infer anatomical regions from plan
    isocentres.
    """

    def uid(item: dict[str, Any], *fields: str) -> str:
        for field in fields:
            value = str(item.get(field) or "").strip()
            if value:
                return value
        return ""

    selected = [dict(item) for item in selected_plans]
    doses = [dict(item) for item in selected_doses]
    per_plan = [dict(item) for item in per_plan_delivery]
    selected_uids = {
        uid(item, "sop_instance_uid", "plan_sop_uid") for item in selected
    }
    selected_uids.discard("")
    delivered_uids = {
        uid(item, "plan_sop_uid", "sop_instance_uid")
        for item in per_plan
        if uid(item, "plan_sop_uid", "sop_instance_uid")
        and int(item.get("delivered_fraction_count") or 0) > 0
    }
    unselected_delivered = sorted(delivered_uids - selected_uids)
    classification = str(dose_classification.get("classification") or "").strip()
    source_refs = {
        str(reference).strip()
        for item in doses
        for reference in item.get("referenced_plan_uids") or []
        if str(reference).strip()
    }
    expected = sorted(selected_uids)
    base = {
        "schema_version": DOSE_COMPLETENESS_SCHEMA_VERSION,
        "status": DOSE_COMPLETENESS_NOT_DEFENSIBLE_STATUS,
        "category": "not_defensible",
        "reason_code": "",
        "reason": "",
        "expected_plan_uids": expected,
        "delivered_plan_uids": sorted(delivered_uids),
        "represented_plan_uids": sorted(source_refs),
        "unselected_delivered_plan_uids": unselected_delivered,
        "dose_summation_types": sorted(
            {
                str(item.get("dose_summation_type") or "").strip().upper()
                for item in doses
                if str(item.get("dose_summation_type") or "").strip()
            }
        ),
        "delivered_fraction_weights": {},
        "spatial_mapping_validated": bool(spatial_mapping_validated),
    }

    def reject(code: str, reason: str) -> dict[str, Any]:
        result = dict(base)
        result.update({"reason_code": code, "reason": reason})
        return result

    if delivery_status not in {"fully_delivered", "partially_delivered"}:
        return reject(
            "delivery_evidence_unresolved",
            "Dose-response eligibility requires an estimable delivery status backed by RTRECORD evidence.",
        )
    if not selected_uids:
        return reject(
            "no_selected_course_plan",
            "No authoritative selected RTPLAN membership is available for course-level dose coverage.",
        )
    if unselected_delivered:
        return reject(
            "unselected_delivered_plan_requires_reconciliation",
            "A distinct RTPLAN has delivery records but is omitted from the selected dose membership. "
            "Only an explicit equivalent-plan revision chain can justify treating that plan as a replacement rather than an additional dose contributor.",
        )
    if not doses:
        return reject(
            "no_authoritative_dose_grid",
            "The course has no selected RTDOSE source from which complete ROI dose can be established.",
        )
    if source_refs != selected_uids:
        return reject(
            "dose_plan_uid_coverage_mismatch",
            "Selected RTDOSE references do not exactly cover the authoritative selected RTPLAN membership.",
        )

    dose_types = [
        str(item.get("dose_summation_type") or "").strip().upper() for item in doses
    ]
    plan_by_uid = {
        uid(item, "sop_instance_uid", "plan_sop_uid"): item for item in selected
    }
    weights: dict[str, float] = {}
    for plan_uid in sorted(selected_uids):
        item = plan_by_uid.get(plan_uid)
        if item is None:
            return reject(
                "selected_plan_evidence_missing",
                f"No selected RTPLAN evidence is available for {plan_uid}.",
            )
        try:
            planned = float(item.get("planned_fraction_count") or 0)
            delivered = float(item.get("delivered_fraction_count") or 0)
        except (TypeError, ValueError):
            return reject(
                "delivered_fraction_weight_unresolved",
                f"Delivered fraction weight is nonnumeric for selected RTPLAN {plan_uid}.",
            )
        weight = delivered / planned if planned > 0 else float("nan")
        if not math.isfinite(weight) or weight <= 0 or weight > 1:
            return reject(
                "delivered_fraction_weight_invalid",
                f"Delivered fraction weight for selected RTPLAN {plan_uid} is not in (0, 1].",
            )
        weights[plan_uid] = weight
    if len(doses) == 1 and dose_types == ["MULTI_PLAN"]:
        if spatial_mapping_validated is not True:
            return reject(
                "dose_grid_not_validated",
                "The exact-coverage MULTI_PLAN RTDOSE has not passed physical Gy and patient-coordinate grid validation.",
            )
        if any(not math.isclose(weight, 1.0) for weight in weights.values()):
            return reject(
                "multi_plan_dose_not_delivered_weighted",
                "The MULTI_PLAN RTDOSE represents planned dose, but at least one contributing plan was not fully delivered.",
            )
        result = dict(base)
        result.update(
            {
                "status": DOSE_COMPLETENESS_ELIGIBLE_STATUS,
                "category": "multi_plan_rtdose_exact_uid_coverage",
                "reason_code": "multi_plan_rtdose_exact_uid_coverage",
                "reason": "One MULTI_PLAN RTDOSE references every selected course RTPLAN SOP Instance UID exactly.",
            }
        )
        return result
    if not doses or any(dose_type != "PLAN" for dose_type in dose_types):
        return reject(
            "unsupported_course_dose_sources",
            "The selected dose sources are neither one exact MULTI_PLAN RTDOSE nor per-plan PLAN RTDOSE sources.",
        )

    dose_refs = [
        {
            str(reference).strip()
            for reference in item.get("referenced_plan_uids") or []
            if str(reference).strip()
        }
        for item in doses
    ]
    dose_plan_uids = [next(iter(refs)) for refs in dose_refs if len(refs) == 1]
    if (
        any(len(refs) != 1 for refs in dose_refs)
        or len(dose_plan_uids) != len(selected_uids)
        or len(set(dose_plan_uids)) != len(dose_plan_uids)
        or set(dose_plan_uids) != selected_uids
    ):
        return reject(
            "per_plan_dose_linkage_unresolved",
            "Per-plan PLAN RTDOSE sources do not provide one-to-one coverage of the selected plan UIDs.",
        )
    requires_accumulation = bool(
        len(doses) > 1 or any(not math.isclose(weight, 1.0) for weight in weights.values())
    )
    if requires_accumulation and spatial_mapping_validated is not True:
        return reject(
            "spatial_mapping_not_validated",
            "Per-plan dose sources and fraction weights are available, but the patient-coordinate spatial mapping has not been validated.",
        )
    if not requires_accumulation and spatial_mapping_validated is not True:
        return reject(
            "dose_grid_not_validated",
            "The single PLAN RTDOSE has not passed physical Gy and patient-coordinate grid validation.",
        )
    if len(selected_uids) == 1:
        category = "single_plan_course_dose_delivered_weighted"
        reason = (
            "RTRECORD evidence identifies one clinically contributing plan; its linked "
            "PLAN RTDOSE has a positive bounded delivered-fraction weight."
        )
    else:
        category = "per_plan_dose_accumulation_delivered_weighted"
        reason = (
            "Each selected plan has one linked PLAN RTDOSE and a positive bounded "
            "RTRECORD fraction weight."
        )
    result = dict(base)
    result.update(
        {
            "status": DOSE_COMPLETENESS_ELIGIBLE_STATUS,
            "category": category,
            "reason_code": category,
            "reason": reason,
            "delivered_fraction_weights": weights,
            "spatial_mapping_validated": bool(
                spatial_mapping_validated or not requires_accumulation
            ),
        }
    )
    return result


def _nonnegative_int(value: object) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _plan_treatment_technique_evidence(path: Path, course_dir: Path | None) -> dict[str, Any]:
    dataset = _read_header(path, "treatment_technique.plan_evidence.path")
    modality = str(getattr(dataset, "Modality", "") or "").strip().upper()
    if modality != "RTPLAN":
        raise CourseContractError(
            f"treatment technique evidence is not an RTPLAN: {path} ({modality!r})"
        )
    fraction_groups = list(getattr(dataset, "FractionGroupSequence", None) or [])
    beam_sequence_count = len(getattr(dataset, "BeamSequence", None) or [])
    brachy_setup_sequence_count = len(
        getattr(dataset, "BrachyApplicationSetupSequence", None) or []
    )
    number_of_beams = sum(
        _nonnegative_int(getattr(group, "NumberOfBeams", None))
        for group in fraction_groups
    )
    number_of_brachy_setups = sum(
        _nonnegative_int(
            getattr(group, "NumberOfBrachyApplicationSetups", None)
        )
        for group in fraction_groups
    )
    referenced_beam_count = sum(
        len(getattr(group, "ReferencedBeamSequence", None) or [])
        for group in fraction_groups
    )
    referenced_brachy_count = sum(
        len(
            getattr(
                group,
                "ReferencedBrachyApplicationSetupSequence",
                None,
            )
            or []
        )
        for group in fraction_groups
    )
    sop_class_uid = str(getattr(dataset, "SOPClassUID", "") or "").strip()
    has_ebrt = any((beam_sequence_count, number_of_beams, referenced_beam_count))
    has_brachy = any(
        (
            brachy_setup_sequence_count,
            number_of_brachy_setups,
            referenced_brachy_count,
        )
    )
    if has_ebrt and has_brachy:
        classification = "MIXED"
    elif has_brachy:
        classification = "BRACHYTHERAPY"
    elif has_ebrt:
        classification = "EBRT"
    else:
        classification = "UNKNOWN"
    resolved = path.resolve(strict=False)
    if course_dir is None:
        path_value = str(resolved)
    else:
        try:
            path_value = resolved.relative_to(
                Path(course_dir).resolve(strict=False)
            ).as_posix()
        except ValueError as exc:
            raise CourseContractError(
                f"treatment technique plan escapes the course directory: {resolved}"
            ) from exc
    return {
        "sop_instance_uid": str(
            getattr(dataset, "SOPInstanceUID", "") or ""
        ).strip(),
        "sop_class_uid": sop_class_uid,
        "sop_class_profile": plan_profile_name(sop_class_uid),
        "path": path_value,
        "classification": classification,
        "beam_sequence_count": beam_sequence_count,
        "number_of_beams": number_of_beams,
        "referenced_beam_count": referenced_beam_count,
        "brachy_application_setup_sequence_count": brachy_setup_sequence_count,
        "number_of_brachy_application_setups": number_of_brachy_setups,
        "referenced_brachy_application_setup_count": referenced_brachy_count,
    }


def build_treatment_technique_contract(
    plan_paths: Iterable[Path | str],
    *,
    course_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Classify selected treatment plans from standard DICOM plan sequences."""
    root = Path(course_dir) if course_dir is not None else None
    evidence = [
        _plan_treatment_technique_evidence(Path(path), root)
        for path in dict.fromkeys(Path(path) for path in plan_paths)
    ]
    plan_classes = {str(item["classification"]) for item in evidence}
    has_ebrt = bool(plan_classes.intersection({"EBRT", "MIXED"}))
    has_brachy = bool(plan_classes.intersection({"BRACHYTHERAPY", "MIXED"}))
    if has_ebrt and has_brachy:
        classification = "MIXED"
    elif has_brachy:
        classification = "BRACHYTHERAPY"
    elif has_ebrt:
        classification = "EBRT"
    else:
        classification = "UNKNOWN"
    eligible = classification == "EBRT"
    return {
        "classification": classification,
        "plan_evidence": evidence,
        "dose_response_eligible": eligible,
        "dose_response_exclusion_reason": (
            None if eligible else f"treatment_technique_{classification.lower()}"
        ),
        "prescription_relative_dvh_metrics": (
            "available_when_prescription_resolved" if eligible else "suppressed"
        ),
    }


def _nonempty_text(value: object, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise CourseContractError(f"course contract field {field} is empty")
    return text


def _optional_nonnegative_number(value: object, field: str) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise CourseContractError(f"course contract field {field} must be numeric or null")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CourseContractError(
            f"course contract field {field} must be numeric or null"
        ) from exc
    if not math.isfinite(number) or number < 0:
        raise CourseContractError(
            f"course contract field {field} must be finite and nonnegative"
        )
    return number


def _list_of_dicts(value: object, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise CourseContractError(f"course contract field {field} must be a list of objects")
    return list(value)


def _read_header(path: Path, field: str):
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception as exc:
        raise CourseContractError(
            f"course contract field {field} points to unreadable DICOM: {path}: {exc}"
        ) from exc


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _referenced_sop_uids(dataset: object, sequence_name: str) -> list[str]:
    return _unique(
        str(getattr(item, "ReferencedSOPInstanceUID", "") or "").strip()
        for item in getattr(dataset, sequence_name, []) or []
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CourseContractError(f"cannot hash contracted artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _number_list(value: object) -> list[float] | None:
    if value in (None, ""):
        return None
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _ct_provenance(ct_dir: Path) -> dict[str, Any]:
    """Read the identity and geometry used to create a planning-CT NIfTI."""
    instances: list[str] = []
    series_uids: set[str] = set()
    geometry: dict[str, Any] = {}
    for path in sorted(item for item in ct_dir.iterdir() if item.is_file()):
        try:
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        modality = str(getattr(dataset, "Modality", "") or "").strip().upper()
        if not modality:
            continue
        if modality != "CT":
            raise CourseContractError(
                f"planning CT directory contains a non-CT DICOM object: {path} ({modality or 'missing modality'})"
            )
        sop_class = str(getattr(dataset, "SOPClassUID", "") or "").strip()
        if sop_class not in _ROLE_EXPECTATIONS["CT"][1]:
            raise CourseContractError(
                f"planning CT directory contains an unsupported CT SOP Class UID {sop_class!r}: {path}"
            )
        series_uid = str(getattr(dataset, "SeriesInstanceUID", "") or "").strip()
        if series_uid:
            series_uids.add(series_uid)
        sop_uid = str(getattr(dataset, "SOPInstanceUID", "") or "").strip()
        if sop_uid:
            instances.append(sop_uid)
        if not geometry:
            geometry = {
                "rows": int(getattr(dataset, "Rows", 0) or 0),
                "columns": int(getattr(dataset, "Columns", 0) or 0),
                "pixel_spacing": _number_list(getattr(dataset, "PixelSpacing", None)),
                "image_orientation_patient": _number_list(
                    getattr(dataset, "ImageOrientationPatient", None)
                ),
                "slice_thickness": (
                    float(getattr(dataset, "SliceThickness"))
                    if getattr(dataset, "SliceThickness", None) not in (None, "")
                    else None
                ),
            }
    if not instances or not series_uids:
        raise CourseContractError(f"planning CT contract directory contains no readable CT objects: {ct_dir}")
    if len(series_uids) != 1:
        raise CourseContractError(
            f"planning CT contract directory contains multiple SeriesInstanceUID values: {sorted(series_uids)!r}"
        )
    return {
        "series_instance_uid": next(iter(series_uids)),
        "sop_hash": hashlib.sha256("".join(instances).encode("utf-8")).hexdigest(),
        "geometry": geometry,
    }


def _read_json(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CourseContractError(f"course contract field {field} sidecar is unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CourseContractError(f"course contract field {field} sidecar must be a JSON object: {path}")
    return value


def _validate_nifti_provenance(
    contract: "CourseContract",
    planning_ct: dict[str, Any],
    ct_dir: Path,
    nifti: Path,
    series_uid: str,
) -> None:
    provenance = planning_ct.get("nifti_provenance")
    if not isinstance(provenance, dict):
        raise CourseContractError(
            "planning_ct.nifti_provenance is required to validate NIfTI identity"
        )
    sidecar = contract.resolve_path(
        provenance.get("sidecar_path"),
        "planning_ct.nifti_provenance.sidecar_path",
    )
    assert sidecar is not None
    sidecar_data = _read_json(sidecar, "planning_ct.nifti_provenance")
    expected_ct = _ct_provenance(ct_dir)
    for key in ("series_instance_uid", "sop_hash", "geometry", "nifti_geometry", "nifti_sha256"):
        if key not in provenance or key not in sidecar_data:
            raise CourseContractError(
                f"planning CT NIfTI provenance is incomplete for {key}: {nifti}"
            )
        if sidecar_data.get(key) != provenance.get(key):
            raise CourseContractError(
                f"stale planning CT NIfTI provenance: sidecar {key} does not match the course contract"
            )
    if provenance.get("series_instance_uid") != series_uid:
        raise CourseContractError(
            "stale planning CT NIfTI provenance: SeriesInstanceUID does not match the contract"
        )
    if provenance.get("series_instance_uid") != expected_ct["series_instance_uid"]:
        raise CourseContractError(
            "stale planning CT NIfTI provenance: SeriesInstanceUID does not match the selected DICOM series"
        )
    if provenance.get("sop_hash") != expected_ct["sop_hash"]:
        raise CourseContractError(
            "stale planning CT NIfTI provenance: source CT instance hash does not match the selected series"
        )
    if provenance.get("geometry") != expected_ct["geometry"]:
        raise CourseContractError(
            "stale planning CT NIfTI provenance: source CT geometry does not match the selected series"
        )
    # Geometry and orientation are validated from the conversion sidecar and
    # source-series provenance above. Downstream image readers remain responsible
    # for rejecting an unreadable NIfTI before it is used.
    if provenance.get("nifti_sha256") != _sha256(nifti):
        raise CourseContractError(
            "stale planning CT NIfTI provenance: content hash does not match the NIfTI on disk"
        )


@dataclass(frozen=True)
class CourseContract:
    course_dir: Path
    metadata_path: Path
    data: dict[str, Any]

    def resolve_path(
        self,
        value: object,
        field: str,
        *,
        required: bool = True,
        directory: bool = False,
    ) -> Path | None:
        text = str(value or "").strip()
        if not text:
            if required:
                raise CourseContractError(f"course contract field {field} is empty")
            return None
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = self.course_dir / candidate
        resolved = candidate.resolve(strict=False)
        root = self.course_dir.resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise CourseContractError(
                f"course contract field {field} escapes the course directory: {text}"
            ) from exc
        exists = resolved.is_dir() if directory else resolved.is_file()
        if not exists:
            expected = "directory" if directory else "file"
            raise CourseContractError(
                f"course contract field {field} points to a missing {expected}: {resolved}"
            )
        return resolved

    @property
    def selected_plans(self) -> list[dict[str, Any]]:
        return _list_of_dicts(self.data.get("selected_plans"), "selected_plans")

    @property
    def selected_doses(self) -> list[dict[str, Any]]:
        return _list_of_dicts(self.data.get("selected_doses"), "selected_doses")

    @property
    def treatment_technique(self) -> dict[str, Any]:
        value = self.data.get("treatment_technique")
        if not isinstance(value, dict):
            raise CourseContractError(
                "course contract field treatment_technique must be an object"
            )
        return value

    @property
    def delivery(self) -> dict[str, Any]:
        value = self.data.get("delivery")
        if not isinstance(value, dict):
            raise CourseContractError("course contract field delivery must be an object")
        return value

    @property
    def dose_qc(self) -> dict[str, Any]:
        value = self.data.get("dose_qc")
        if not isinstance(value, dict):
            raise CourseContractError("course contract field dose_qc must be an object")
        return value

    @property
    def planning_ct(self) -> dict[str, Any]:
        value = self.data.get("planning_ct")
        if not isinstance(value, dict):
            raise CourseContractError("course contract field planning_ct must be an object")
        return value

    @property
    def prescribed_dose_gy(self) -> float | None:
        value = self.delivery.get("prescribed_dose_gy")
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise CourseContractError("course contract prescribed_dose_gy is not numeric") from exc

    @property
    def resolved_prescribed_dose_total_gy(self) -> float | None:
        value = self.delivery.get("resolved_prescribed_dose_total_gy")
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise CourseContractError(
                "course contract resolved_prescribed_dose_total_gy is not numeric"
            ) from exc

    @property
    def delivered_dose_gy(self) -> float | None:
        value = self.delivery.get("delivered_dose_gy")
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise CourseContractError("course contract delivered_dose_gy is not numeric") from exc

    @property
    def authoritative_rtstruct_path(self) -> Path | None:
        value = self.data.get("authoritative_rtstruct")
        if value is None:
            return None
        if not isinstance(value, dict):
            raise CourseContractError("course contract field authoritative_rtstruct must be an object or null")
        return self.resolve_path(value.get("path"), "authoritative_rtstruct.path")

    @property
    def authoritative_rtstruct_source(self) -> str:
        """Return the provenance label attached to the contracted RTSTRUCT.

        Organizer contracts predate an explicit source field and represent the
        selected clinical structure set, so their compatible default is
        ``Manual``. Scoped all-series contracts must declare their generated
        source explicitly and are validated below.
        """
        value = self.data.get("authoritative_rtstruct")
        if value is None:
            return MANUAL_RTSTRUCT_SOURCE
        if not isinstance(value, dict):
            raise CourseContractError(
                "course contract field authoritative_rtstruct must be an object or null"
            )
        source = str(value.get("segmentation_source") or "").strip()
        return source or MANUAL_RTSTRUCT_SOURCE

    @property
    def planning_ct_dir(self) -> Path | None:
        return self.resolve_path(
            self.planning_ct.get("dicom_dir"),
            "planning_ct.dicom_dir",
            required=False,
            directory=True,
        )

    @property
    def planning_ct_nifti(self) -> Path | None:
        return self.resolve_path(
            self.planning_ct.get("nifti_path"),
            "planning_ct.nifti_path",
            required=False,
        )

    @property
    def plan_artifact_path(self) -> Path | None:
        value = self.data.get("plan_artifact")
        if value is None:
            return None
        if not isinstance(value, dict):
            raise CourseContractError("course contract field plan_artifact must be an object or null")
        return self.resolve_path(value.get("path"), "plan_artifact.path")

    @property
    def dose_grid_path(self) -> Path | None:
        value = self.data.get("dose_grid")
        if value is None:
            return None
        if not isinstance(value, dict):
            raise CourseContractError("course contract field dose_grid must be an object or null")
        return self.resolve_path(value.get("path"), "dose_grid.path")

    def require_planning_ct(self) -> tuple[Path, Path]:
        ct_dir = self.planning_ct_dir
        nifti = self.planning_ct_nifti
        if ct_dir is None or nifti is None:
            raise CourseContractError(
                "course contract has no complete planning CT decision (DICOM directory and NIfTI are required)"
            )
        return ct_dir, nifti

    def require_dvh_artifacts(self) -> tuple[Path, Path | None]:
        plan = self.plan_artifact_path
        dose = self.dose_grid_path
        if plan is None:
            raise CourseContractError(
                "course contract has no authoritative plan artifact for DVH"
            )
        return plan, dose

    def require_computable_dvh_artifacts(self) -> tuple[Path, Path]:
        """Return the RP/RD pair required for dose-based DVH computation.

        A valid plan-only contract has a plan artifact but no dose grid. That
        state is handled by DVH as an explicit no-metrics outcome rather than
        being treated as a malformed contract.
        """
        plan, dose = self.require_dvh_artifacts()
        if dose is None:
            raise CourseContractError(
                "course contract has no authoritative dose grid for DVH"
            )
        return plan, dose


def _validate_dicom_identity(
    contract: CourseContract,
    item: dict[str, Any],
    field: str,
    *,
    summation_type: bool = False,
    role: str | None = None,
) -> Path:
    path = contract.resolve_path(item.get("path"), f"{field}.path")
    assert path is not None
    dataset = _read_header(path, f"{field}.path")
    expected_uid = _nonempty_text(item.get("sop_instance_uid"), f"{field}.sop_instance_uid")
    actual_uid = str(getattr(dataset, "SOPInstanceUID", "") or "").strip()
    if actual_uid != expected_uid:
        raise CourseContractError(
            f"stale course contract at {field}: SOPInstanceUID {actual_uid!r} on disk does not match {expected_uid!r}"
        )
    if role is not None:
        try:
            modalities, sop_classes = _ROLE_EXPECTATIONS[role]
        except KeyError as exc:
            raise ValueError(f"unknown course-contract DICOM role {role!r}") from exc
        actual_modality = str(getattr(dataset, "Modality", "") or "").strip().upper()
        actual_sop_class = str(getattr(dataset, "SOPClassUID", "") or "").strip()
        if actual_modality not in modalities or actual_sop_class not in sop_classes:
            raise CourseContractError(
                f"stale course contract at {field}: expected {role} DICOM "
                f"(Modality={sorted(modalities)!r}, SOPClassUID={sorted(sop_classes)!r}), "
                f"found (Modality={actual_modality!r}, SOPClassUID={actual_sop_class!r})"
            )
    if summation_type:
        expected_type = _nonempty_text(
            item.get("dose_summation_type"), f"{field}.dose_summation_type"
        ).upper()
        actual_type = str(getattr(dataset, "DoseSummationType", "") or "").strip().upper()
        if actual_type != expected_type:
            raise CourseContractError(
                f"stale course contract at {field}: DoseSummationType {actual_type!r} on disk does not match {expected_type!r}"
            )
    return path


def _validate_plan_prescription(
    item: dict[str, Any], dataset: Dataset, field: str
) -> None:
    """Verify serialized prescription evidence against the exact RTPLAN bytes."""

    actual_groups = _list_of_dicts(
        item.get("prescription_groups"), f"{field}.prescription_groups"
    )
    identity_group = actual_groups[0] if actual_groups else {}
    source_tag_path = str(
        identity_group.get("source_prescribed_dose_tag_path") or ""
    )
    binding_source = (
        None
        if source_tag_path.startswith("FractionGroupSequence[")
        else item.get("prescribed_dose_gy")
    )
    expected_groups = resolve_plan_prescriptions(
        dataset,
        source_prescribed_dose_gy=binding_source,
        source_dose_reference_number=identity_group.get(
            "source_dose_reference_number"
        ),
        source_dose_reference_uid=identity_group.get("source_dose_reference_uid"),
    )
    if actual_groups != expected_groups:
        raise CourseContractError(
            f"stale course contract at {field}: prescription_groups do not "
            "match the RTPLAN BeamDose evidence"
        )
    expected_group = expected_groups[0] if len(expected_groups) == 1 else None
    for name in PRESCRIPTION_GROUP_FIELDS:
        if name not in item:
            raise CourseContractError(f"course contract field {field}.{name} is missing")
        expected = expected_group.get(name) if expected_group else None
        if item.get(name) != expected:
            raise CourseContractError(
                f"stale course contract at {field}.{name}: serialized value "
                "does not match the RTPLAN BeamDose evidence"
            )
    source = _optional_nonnegative_number(
        item.get("prescribed_dose_gy"), f"{field}.prescribed_dose_gy"
    )
    expected_source = source_plan_prescribed_dose_gy(expected_groups)
    if source != expected_source:
        raise CourseContractError(
            f"stale course contract at {field}.prescribed_dose_gy: source "
            "prescription does not match the RTPLAN"
        )
    resolved = _optional_nonnegative_number(
        item.get("resolved_prescribed_dose_total_gy"),
        f"{field}.resolved_prescribed_dose_total_gy",
    )
    expected_resolved = resolved_plan_total_gy(expected_groups)
    if resolved != expected_resolved:
        raise CourseContractError(
            f"stale course contract at {field}.resolved_prescribed_dose_total_gy: "
            "resolved total does not match the RTPLAN BeamDose evidence"
        )


def _same_dose(left: object, right: object, tolerance: float = 0.05) -> bool:
    if left in (None, "") or right in (None, ""):
        return left in (None, "") and right in (None, "")
    try:
        return abs(float(str(left)) - float(str(right))) <= tolerance
    except (TypeError, ValueError):
        return False


def _validate_clinical_prescription_evidence(
    evidence: object,
    *,
    dicom_resolved_total_gy: float | None,
    dicom_prescribed_scope: str,
    prescribed_scope: str,
    per_plan_delivery: list[dict[str, Any]],
) -> float | None:
    """Validate the portable row snapshot and its resolution decision."""

    if evidence is None:
        if prescribed_scope == CLINICAL_RESOLVED_SCOPE:
            raise CourseContractError(
                "a clinically resolved prescription requires clinical_prescription_evidence"
            )
        return None
    if not isinstance(evidence, dict):
        raise CourseContractError(
            "course contract field clinical_prescription_evidence must be an object or null"
        )
    if evidence.get("schema") != CLINICAL_EVIDENCE_SCHEMA:
        raise CourseContractError("unsupported clinical prescription evidence schema")
    source = evidence.get("source")
    if not isinstance(source, dict):
        raise CourseContractError("clinical prescription evidence source must be an object")
    workbook_path = str(source.get("workbook_path") or "").strip()
    workbook_hash = str(source.get("workbook_sha256") or "").strip()
    if not workbook_path or not re.fullmatch(r"[0-9a-f]{64}", workbook_hash):
        raise CourseContractError(
            "clinical prescription evidence requires a source path and SHA-256"
        )
    if not str(source.get("sheet_name") or "").strip():
        raise CourseContractError("clinical prescription evidence source sheet is empty")
    try:
        if int(source.get("row_count") or 0) <= 0:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise CourseContractError(
            "clinical prescription evidence source row_count must be positive"
        ) from exc

    dicom = evidence.get("dicom")
    if not isinstance(dicom, dict):
        raise CourseContractError("clinical prescription DICOM snapshot must be an object")
    dicom_delivery_status = str(dicom.get("delivery_status") or "")
    if dicom_delivery_status not in {
        "fully_delivered",
        "partially_delivered",
        "delivered_but_records_absent",
        "delivery_unresolved",
        "no_records_at_all",
    }:
        raise CourseContractError(
            "clinical prescription DICOM snapshot delivery status is invalid"
        )
    dicom_delivered_dose_gy = _optional_nonnegative_number(
        dicom.get("delivered_dose_gy"),
        "clinical_prescription_evidence.dicom.delivered_dose_gy",
    )
    if dicom_delivery_status in {"fully_delivered", "partially_delivered"}:
        if dicom_delivered_dose_gy is None:
            raise CourseContractError(
                "clinical prescription DICOM snapshot delivery dose is missing"
            )
    elif dicom_delivered_dose_gy is not None:
        raise CourseContractError(
            "clinical prescription DICOM snapshot delivery dose contradicts its status"
        )
    dicom_delivery_method = dicom.get("delivery_method")
    if dicom_delivery_method is not None and not isinstance(
        dicom_delivery_method, str
    ):
        raise CourseContractError(
            "clinical prescription DICOM snapshot delivery method is invalid"
        )
    regeneration = evidence.get("regeneration_provenance")
    if regeneration is not None:
        if not isinstance(regeneration, dict):
            raise CourseContractError(
                "clinical prescription regeneration provenance must be an object"
            )
        if regeneration.get("schema") != CLINICAL_EVIDENCE_REGENERATION_SCHEMA:
            raise CourseContractError(
                "clinical prescription regeneration provenance schema is unsupported"
            )
        if regeneration.get("authority") != "organize" or regeneration.get(
            "reason"
        ) != "organize_resume_republication":
            raise CourseContractError(
                "clinical prescription regeneration provenance authority is invalid"
            )
        previous_sha256 = regeneration.get("previous_evidence_payload_sha256")
        if not isinstance(previous_sha256, str) or re.fullmatch(
            r"[0-9a-f]{64}", previous_sha256
        ) is None:
            raise CourseContractError(
                "clinical prescription regeneration provenance hash is invalid"
            )
        for key in (
            "previous_evidence_payload",
            "current_source",
            "current_dicom_snapshot",
        ):
            if not isinstance(regeneration.get(key), dict):
                raise CourseContractError(
                    f"clinical prescription regeneration provenance {key} must be an object"
                )
        previous_payload = regeneration["previous_evidence_payload"]
        if "regeneration_provenance" in previous_payload:
            raise CourseContractError(
                "clinical prescription prior evidence payload is recursively nested"
            )
        if previous_payload.get("schema") != CLINICAL_EVIDENCE_SCHEMA:
            raise CourseContractError(
                "clinical prescription prior evidence payload schema is unsupported"
            )
        if clinical_evidence_content_sha256(previous_payload) != previous_sha256:
            raise CourseContractError(
                "clinical prescription regeneration provenance hash is stale"
            )
        if regeneration["current_source"] != source:
            raise CourseContractError(
                "clinical prescription regeneration provenance source is stale"
            )
        if regeneration["current_dicom_snapshot"] != dicom:
            raise CourseContractError(
                "clinical prescription regeneration provenance DICOM snapshot is stale"
            )
    evidence_dicom_total_gy = dicom.get("resolved_prescribed_dose_total_gy")
    if not _same_dose(evidence_dicom_total_gy, dicom_resolved_total_gy):
        raise CourseContractError(
            "clinical prescription evidence DICOM total is stale: "
            f"snapshot={evidence_dicom_total_gy!r}, "
            f"recomputed={dicom_resolved_total_gy!r}"
        )
    if str(dicom.get("prescribed_dose_scope") or "") != dicom_prescribed_scope:
        raise CourseContractError("clinical prescription evidence DICOM scope is stale")

    outcome = str(evidence.get("outcome") or "")
    if outcome not in {
        "UNRESOLVED",
        "RESOLVED_FROM_CLINICAL_RECORD",
        "CORROBORATED_DICOM",
        "DISAGREES_WITH_DICOM",
    }:
        raise CourseContractError(
            f"unknown clinical prescription evidence outcome {outcome!r}"
        )
    match = evidence.get("match")
    if not isinstance(match, dict):
        raise CourseContractError("clinical prescription match evidence must be an object")
    record = evidence.get("record")
    parsed = evidence.get("parse")
    sites: list[dict[str, Any]] = []
    if match.get("status") == "MATCHED":
        if not isinstance(record, dict):
            raise CourseContractError("matched clinical evidence requires a source record")
        if record.get("workbook_sha256") != workbook_hash:
            raise CourseContractError(
                "clinical source record hash disagrees with the workbook source"
            )
        if record.get("parsed_field") != "Opis leczenia":
            raise CourseContractError(
                "clinical source record must name the parsed Opis leczenia field"
            )
        if not str(record.get("record_id") or "").strip():
            raise CourseContractError("clinical source record_id is empty")
        try:
            if int(record.get("excel_row") or 0) < 2:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise CourseContractError(
                "clinical source record excel_row must identify a data row"
            ) from exc
        source_text = record.get("source_text")
        if not isinstance(source_text, str):
            raise CourseContractError("clinical source_text must be a string")
        expected_parse = parse_kopernik_treatment_description(source_text)
        if parsed != expected_parse:
            raise CourseContractError(
                "clinical prescription parse does not match the exact source text"
            )
        sites = list(expected_parse.get("sites") or [])
        match_evidence = match.get("match_evidence")
        if not isinstance(match_evidence, dict):
            raise CourseContractError("matched clinical evidence lacks match details")
        record_window = match_evidence.get("record_window")
        expected_window = [
            record.get("treatment_start_date"),
            record.get("treatment_end_date"),
        ]
        if record_window != expected_window:
            raise CourseContractError(
                "clinical prescription match window disagrees with source record dates"
            )
    elif record is not None or parsed is not None:
        raise CourseContractError(
            "unmatched clinical prescription evidence cannot contain a parsed record"
        )

    expected_phase = confirm_two_phase_fractionation(sites, per_plan_delivery)
    if evidence.get("fractionation_classification") != expected_phase:
        raise CourseContractError(
            "clinical fractionation classification disagrees with record and RTPLAN delivery evidence"
        )
    totals = sorted({float(item["total_dose_gy"]) for item in sites})
    clinical_total = evidence.get("clinical_resolved_total_gy")
    effective_total = evidence.get("effective_resolved_total_gy")
    source_label = evidence.get("effective_prescription_source")

    if outcome == "RESOLVED_FROM_CLINICAL_RECORD":
        if dicom_resolved_total_gy is not None or len(totals) != 1:
            raise CourseContractError(
                "clinical resolution requires unresolved DICOM and one unique clinical total"
            )
        if not _same_dose(clinical_total, totals[0]) or not _same_dose(
            effective_total, totals[0]
        ):
            raise CourseContractError(
                "clinical resolved total disagrees with self-checked site evidence"
            )
        if source_label != "CLINICAL_RECORD" or prescribed_scope != CLINICAL_RESOLVED_SCOPE:
            raise CourseContractError(
                "clinical resolution source or prescribed scope is inconsistent"
            )
        return totals[0]

    if prescribed_scope == CLINICAL_RESOLVED_SCOPE:
        raise CourseContractError(
            "clinical resolved scope requires RESOLVED_FROM_CLINICAL_RECORD"
        )
    if clinical_total not in (None, ""):
        raise CourseContractError(
            "non-resolving clinical evidence cannot publish a clinical total"
        )
    if not _same_dose(effective_total, dicom_resolved_total_gy):
        raise CourseContractError(
            "clinical evidence effective total disagrees with DICOM precedence"
        )
    expected_source = "DICOM" if dicom_resolved_total_gy is not None else None
    if source_label != expected_source:
        raise CourseContractError(
            "clinical evidence effective prescription source is inconsistent"
        )
    matching_totals = [
        value for value in totals if _same_dose(value, dicom_resolved_total_gy)
    ]
    if outcome == "CORROBORATED_DICOM" and (
        dicom_resolved_total_gy is None or not matching_totals
    ):
        raise CourseContractError("clinical corroboration does not match DICOM")
    if outcome == "DISAGREES_WITH_DICOM" and (
        dicom_resolved_total_gy is None or matching_totals
    ):
        raise CourseContractError("clinical disagreement is not supported")
    return None


def validate_course_contract(contract: CourseContract) -> CourseContract:
    data = contract.data
    if data.get("version") != COURSE_CONTRACT_VERSION:
        raise CourseContractError(
            f"unsupported course contract version {data.get('version')!r}; expected {COURSE_CONTRACT_VERSION}"
        )
    scope = data.get("scope")
    authority = data.get("authority")
    if scope == ALL_SERIES_RADIOMICS_TEMP_SCOPE:
        if authority != ALL_SERIES_RADIOMICS_TEMP_AUTHORITY:
            raise CourseContractError(
                "all-series radiomics temporary contract has an invalid authority"
            )
        if not str(data.get("scope_reason") or "").strip():
            raise CourseContractError(
                "all-series radiomics temporary contract must document its scope reason"
            )
    elif authority != "organize":
        raise CourseContractError("course contract authority must be 'organize'")

    expected_patient = _nonempty_text(data.get("patient_id"), "patient_id")
    expected_course = _nonempty_text(data.get("course_id"), "course_id")
    if contract.course_dir.parent.name != expected_patient or contract.course_dir.name != expected_course:
        raise CourseContractError(
            "stale course contract identity: "
            f"contract={expected_patient}/{expected_course}, disk={contract.course_dir.parent.name}/{contract.course_dir.name}"
        )

    selected_plans = contract.selected_plans
    selected_doses = contract.selected_doses
    if not isinstance(data.get("dose_classification"), dict):
        raise CourseContractError("course contract field dose_classification must be an object")
    plan_uids: list[str] = []
    treatment_plan_paths: list[Path] = []
    selected_source_doses: list[float | None] = []
    selected_resolved_doses: list[float | None] = []
    for index, item in enumerate(selected_plans):
        field = f"selected_plans[{index}]"
        plan_path = _validate_dicom_identity(
            contract, item, field, role="RTPLAN_SOURCE"
        )
        treatment_plan_paths.append(plan_path)
        _validate_plan_prescription(
            item,
            _read_header(plan_path, f"{field}.path"),
            field,
        )
        uid = _nonempty_text(item.get("sop_instance_uid"), f"{field}.sop_instance_uid")
        if uid in plan_uids:
            raise CourseContractError(f"duplicate selected RTPLAN SOPInstanceUID {uid}")
        plan_uids.append(uid)
        try:
            records = int(item.get("delivered_record_count") or 0)
            fractions = int(item.get("delivered_fraction_count") or 0)
        except (TypeError, ValueError) as exc:
            raise CourseContractError(f"{field} delivery counts must be integers") from exc
        if records < 0 or fractions < 0:
            raise CourseContractError(f"{field} delivery counts must be nonnegative")
        selected_source_doses.append(
            _optional_nonnegative_number(
                item.get("prescribed_dose_gy"), f"{field}.prescribed_dose_gy"
            )
        )
        selected_resolved_doses.append(
            _optional_nonnegative_number(
                item.get("resolved_prescribed_dose_total_gy"),
                f"{field}.resolved_prescribed_dose_total_gy",
            )
        )

    expected_treatment_technique = build_treatment_technique_contract(
        treatment_plan_paths, course_dir=contract.course_dir
    )
    if contract.treatment_technique != expected_treatment_technique:
        raise CourseContractError(
            "stale course contract at treatment_technique: serialized DICOM "
            "technique evidence does not match selected RTPLAN sources"
        )

    dose_uids: list[str] = []
    selected_types: list[str] = []
    for index, item in enumerate(selected_doses):
        field = f"selected_doses[{index}]"
        dose_path = _validate_dicom_identity(
            contract, item, field, summation_type=True, role="RTDOSE"
        )
        uid = _nonempty_text(item.get("sop_instance_uid"), f"{field}.sop_instance_uid")
        if uid in dose_uids:
            raise CourseContractError(f"duplicate selected RTDOSE SOPInstanceUID {uid}")
        dose_uids.append(uid)
        selected_types.append(
            _nonempty_text(item.get("dose_summation_type"), f"{field}.dose_summation_type").upper()
        )
        expected_references = item.get("referenced_plan_uids")
        if not isinstance(expected_references, list) or any(
            not isinstance(value, str) or not value.strip() for value in expected_references
        ):
            raise CourseContractError(f"{field}.referenced_plan_uids must be a list of nonempty strings")
        actual_references = _referenced_sop_uids(
            _read_header(dose_path, f"{field}.path"),
            "ReferencedRTPlanSequence",
        )
        if set(actual_references) != set(expected_references):
            raise CourseContractError(
                f"stale course contract at {field}: referenced RTPLAN UIDs "
                f"{actual_references!r} on disk do not match {expected_references!r}"
            )

    type_set = set(selected_types)
    if type_set & PLAN_LEVEL_DOSE_SUMMATION_TYPES and type_set & BEAM_LEVEL_DOSE_SUMMATION_TYPES:
        raise CourseContractError("course contract mixes PLAN/PLAN_SUM and BEAM RTDOSE objects")
    unknown_types = type_set - PLAN_LEVEL_DOSE_SUMMATION_TYPES - BEAM_LEVEL_DOSE_SUMMATION_TYPES
    if unknown_types:
        raise CourseContractError(
            f"course contract selects unsupported DoseSummationType values: {sorted(unknown_types)}"
        )
    for index, item in enumerate(selected_doses):
        outside = set(item.get("referenced_plan_uids") or []) - set(plan_uids)
        if outside:
            raise CourseContractError(
                f"selected_doses[{index}] references RTPLAN UIDs outside selected membership: "
                + ", ".join(sorted(outside))
            )
    referenced_selected_plans = {
        str(uid)
        for item in selected_doses
        for uid in item.get("referenced_plan_uids") or []
    }
    if selected_doses and referenced_selected_plans != set(plan_uids):
        raise CourseContractError(
            "selected RTDOSE references do not cover exactly the selected RTPLAN membership"
        )
    if type_set and type_set <= BEAM_LEVEL_DOSE_SUMMATION_TYPES:
        if len(plan_uids) != 1 or len(selected_doses) < 2:
            raise CourseContractError(
                "BEAM RTDOSE sources require at least two components for exactly one selected RTPLAN"
            )

    should_sum_prescriptions = bool(
        data["dose_classification"].get("should_sum") and len(selected_doses) > 1
    )

    delivery = contract.delivery
    per_plan = _list_of_dicts(delivery.get("per_plan"), "delivery.per_plan")
    status = _nonempty_text(delivery.get("status"), "delivery.status")
    prescribed = _optional_nonnegative_number(
        delivery.get("prescribed_dose_gy"),
        "delivery.prescribed_dose_gy",
    )
    resolved_prescribed = _optional_nonnegative_number(
        delivery.get("resolved_prescribed_dose_total_gy"),
        "delivery.resolved_prescribed_dose_total_gy",
    )
    prescribed_scope = str(delivery.get("prescribed_dose_scope") or "").strip()
    dose_response_eligible = delivery.get("dose_response_eligible")
    dose_response_eligibility_basis = delivery.get(
        "dose_response_eligibility_basis"
    )
    if dose_response_eligible is not None:
        if not isinstance(dose_response_eligible, bool):
            raise CourseContractError(
                "delivery.dose_response_eligible must be boolean when present"
            )
        if dose_response_eligibility_basis is None:
            raise CourseContractError(
                "delivery.dose_response_eligible requires "
                "delivery.dose_response_eligibility_basis"
            )
    if dose_response_eligibility_basis not in {
        None,
        DOSE_RESPONSE_ELIGIBILITY_BASIS,
    }:
        raise CourseContractError(
            "unknown delivery.dose_response_eligibility_basis "
            f"{dose_response_eligibility_basis!r}"
        )
    if (
        dose_response_eligibility_basis == DOSE_RESPONSE_ELIGIBILITY_BASIS
        and dose_response_eligible is None
    ):
        raise CourseContractError(
            "delivery.dose_response_eligibility_basis requires "
            "dose_response_eligible"
        )
    ineligibility_reason_code = delivery.get(
        "dose_response_ineligibility_reason_code"
    )
    ineligibility_reason = delivery.get("dose_response_ineligibility_reason")
    if dose_response_eligible is True and (
        ineligibility_reason_code is not None or ineligibility_reason is not None
    ):
        raise CourseContractError(
            "dose-response-eligible delivery cannot carry an ineligibility reason"
        )
    if dose_response_eligible is False and (
        not isinstance(ineligibility_reason_code, str)
        or not ineligibility_reason_code.strip()
        or not isinstance(ineligibility_reason, str)
        or not ineligibility_reason.strip()
    ):
        raise CourseContractError(
            "dose-response-ineligible delivery requires a structured reason code and reason"
        )
    prescription_plan_uids = {
        str(uid).strip()
        for uid in data["dose_classification"].get("prescription_plan_uids", [])
        if str(uid).strip()
    }
    course_source_doses = selected_source_doses
    course_resolved_doses = selected_resolved_doses
    if prescription_plan_uids:
        course_source_doses = [
            value
            for uid, value in zip(plan_uids, selected_source_doses)
            if uid in prescription_plan_uids
        ]
        course_resolved_doses = [
            value
            for uid, value in zip(plan_uids, selected_resolved_doses)
            if uid in prescription_plan_uids
        ]

    def _course_value_for_scope(
        values: list[float | None], scope_value: str
    ) -> float | None:
        if scope_value.startswith("UNRESOLVED_"):
            return None
        return aggregate_course_prescription_values(
            values,
            sum_all=(
                should_sum_prescriptions or scope_value == "COURSE_TOTAL_SUMMED"
            ),
        )

    dicom_scope = prescribed_scope
    if prescribed_scope == CLINICAL_RESOLVED_SCOPE:
        dicom_scope = str(
            data["dose_classification"].get("dicom_prescribed_dose_scope") or ""
        ).strip()
        if not dicom_scope:
            raise CourseContractError(
                "clinical resolution must retain the original DICOM prescription scope"
            )
    dicom_resolved = _course_value_for_scope(course_resolved_doses, dicom_scope)
    clinical_resolved = _validate_clinical_prescription_evidence(
        data.get("clinical_prescription_evidence"),
        dicom_resolved_total_gy=dicom_resolved,
        dicom_prescribed_scope=dicom_scope,
        prescribed_scope=prescribed_scope,
        per_plan_delivery=per_plan,
    )

    def _course_value(values: list[float | None]) -> float | None:
        if prescribed_scope == CLINICAL_RESOLVED_SCOPE:
            return clinical_resolved
        return _course_value_for_scope(values, prescribed_scope)

    if prescribed_scope:
        allowed_scopes = {
            "SINGLE_PLAN_TOTAL",
            "COURSE_TOTAL_SUMMED",
            CLINICAL_RESOLVED_SCOPE,
            "UNRESOLVED_COMPONENT",
            "UNRESOLVED_REPLACEMENT_CHAIN",
        }
        if prescribed_scope not in allowed_scopes:
            raise CourseContractError(
                f"unknown delivery.prescribed_dose_scope {prescribed_scope!r}"
            )
        if prescribed_scope.startswith("UNRESOLVED_") and resolved_prescribed is not None:
            raise CourseContractError(
                "an unresolved delivery.prescribed_dose_scope requires "
                "resolved_prescribed_dose_total_gy to be null"
            )
        if prescribed_scope in {
            "SINGLE_PLAN_TOTAL",
            "COURSE_TOTAL_SUMMED",
            CLINICAL_RESOLVED_SCOPE,
        } and resolved_prescribed is None:
            raise CourseContractError(
                "a resolved delivery.prescribed_dose_scope requires a resolved total"
            )
        classified_scope = str(
            data["dose_classification"].get("prescribed_dose_scope") or ""
        ).strip()
        if classified_scope and classified_scope != prescribed_scope:
            raise CourseContractError(
                "delivery.prescribed_dose_scope disagrees with dose_classification"
            )
    if prescribed != _course_value(course_source_doses):
        raise CourseContractError(
            "delivery.prescribed_dose_gy disagrees with authoritative prescription evidence"
        )
    if resolved_prescribed != _course_value(course_resolved_doses):
        raise CourseContractError(
            "delivery.resolved_prescribed_dose_total_gy disagrees with authoritative prescription evidence"
        )
    if "prescription_source" in delivery:
        expected_prescription_source = (
            "CLINICAL_RECORD"
            if prescribed_scope == CLINICAL_RESOLVED_SCOPE
            else "DICOM"
            if resolved_prescribed is not None
            else None
        )
        if delivery.get("prescription_source") != expected_prescription_source:
            raise CourseContractError(
                "delivery.prescription_source disagrees with prescription scope"
            )
    delivered = _optional_nonnegative_number(
        delivery.get("delivered_dose_gy"),
        "delivery.delivered_dose_gy",
    )
    if status not in {
        "fully_delivered",
        "partially_delivered",
        "delivered_but_records_absent",
        "delivery_unresolved",
        "no_records_at_all",
    }:
        raise CourseContractError(f"unknown delivery.status {status!r}")
    if status in {"fully_delivered", "partially_delivered"} and (
        delivered is None or resolved_prescribed is None
    ):
        raise CourseContractError(
            f"delivery.status {status!r} requires delivered_dose_gy and "
            "resolved_prescribed_dose_total_gy"
        )
    if status in {
        "delivered_but_records_absent",
        "delivery_unresolved",
        "no_records_at_all",
    } and delivered is not None:
        raise CourseContractError(
            f"delivery.status {status!r} requires delivered_dose_gy to be null"
        )
    if (
        dose_response_eligibility_basis == DOSE_RESPONSE_ELIGIBILITY_BASIS
        and dose_response_eligible is not None
    ):
        expected_eligibility = bool(
            not prescribed_scope.startswith("UNRESOLVED_")
            and resolved_prescribed is not None
            and delivered is not None
            and status in {"fully_delivered", "partially_delivered"}
        )
        serialized_completeness = data.get("dose_completeness")
        if serialized_completeness is not None:
            if not isinstance(serialized_completeness, dict):
                raise CourseContractError(
                    "course contract field dose_completeness must be an object"
                )
            if serialized_completeness.get("schema_version") != DOSE_COMPLETENESS_SCHEMA_VERSION:
                raise CourseContractError(
                    "unsupported dose_completeness schema version"
                )
            expected_eligibility = bool(
                expected_eligibility
                and serialized_completeness.get("status")
                == DOSE_COMPLETENESS_ELIGIBLE_STATUS
            )
        if dose_response_eligible != expected_eligibility:
            raise CourseContractError(
                "delivery.dose_response_eligible disagrees with "
                "delivery.dose_response_eligibility_basis"
            )
    if delivery.get("dose_response_field") != DOSE_RESPONSE_FIELD:
        raise CourseContractError(
            f"delivery.dose_response_field must be {DOSE_RESPONSE_FIELD!r}"
        )
    per_plan_uids = [
        _nonempty_text(item.get("plan_sop_uid"), f"delivery.per_plan[{index}].plan_sop_uid")
        for index, item in enumerate(per_plan)
    ]
    if len(per_plan_uids) != len(set(per_plan_uids)):
        raise CourseContractError("delivery.per_plan contains duplicate RTPLAN SOPInstanceUIDs")
    for index, (uid, item) in enumerate(zip(per_plan_uids, per_plan)):
        field = f"delivery.per_plan[{index}]"
        plan_path = contract.resolve_path(item.get("plan_path"), f"{field}.plan_path")
        assert plan_path is not None
        dataset = _read_header(plan_path, f"{field}.plan_path")
        _validate_plan_prescription(item, dataset, field)
        actual_modality = str(getattr(dataset, "Modality", "") or "").strip().upper()
        actual_sop_class = str(getattr(dataset, "SOPClassUID", "") or "").strip()
        if (actual_modality, actual_sop_class) not in {
            (modality, sop_class)
            for modality in _ROLE_EXPECTATIONS["RTPLAN"][0]
            for sop_class in _ROLE_EXPECTATIONS["RTPLAN"][1]
        }:
            raise CourseContractError(
                f"stale course contract at {field}: delivery evidence plan is not an RTPLAN "
                f"(Modality={actual_modality!r}, SOPClassUID={actual_sop_class!r})"
            )
        actual_uid = str(getattr(dataset, "SOPInstanceUID", "") or "").strip()
        if actual_uid != uid:
            raise CourseContractError(
                f"stale course contract at {field}: plan SOPInstanceUID {actual_uid!r} on disk does not match {uid!r}"
            )
        try:
            record_count = int(item.get("delivered_record_count") or 0)
            fraction_count = int(item.get("delivered_fraction_count") or 0)
        except (TypeError, ValueError) as exc:
            raise CourseContractError(f"{field} delivery counts must be integers") from exc
        if record_count < 0 or fraction_count < 0:
            raise CourseContractError(f"{field} delivery counts must be nonnegative")
        if item.get("zero_delivery_records") is not (record_count == 0):
            raise CourseContractError(
                f"{field}.zero_delivery_records disagrees with delivered_record_count"
            )
        dates = item.get("treatment_dates")
        if not isinstance(dates, list) or any(
            not isinstance(value, str) or not value.strip() for value in dates
        ):
            raise CourseContractError(f"{field}.treatment_dates must be a list")
        record_paths = item.get("record_paths")
        if not isinstance(record_paths, list) or any(
            not isinstance(value, str) or not value.strip() for value in record_paths
        ):
            raise CourseContractError(f"{field}.record_paths must be a list of paths")
        record_uids: set[str] = set()
        fraction_sessions: set[tuple[str, str, str]] = set()
        observed_dates: set[str] = set()
        for record_index, record_value in enumerate(record_paths):
            record_path = contract.resolve_path(
                record_value,
                f"{field}.record_paths[{record_index}]",
            )
            assert record_path is not None
            record = _read_header(record_path, f"{field}.record_paths[{record_index}]")
            modality = str(getattr(record, "Modality", "") or "").strip().upper()
            if modality != "RTRECORD":
                raise CourseContractError(
                    f"{field}.record_paths[{record_index}] is not an RTRECORD: {record_path}"
                )
            referenced_plans = _referenced_sop_uids(record, "ReferencedRTPlanSequence")
            if uid not in referenced_plans:
                raise CourseContractError(
                    f"{field}.record_paths[{record_index}] does not reference plan {uid}"
                )
            record_uid = str(getattr(record, "SOPInstanceUID", "") or "").strip()
            if not record_uid:
                raise CourseContractError(
                    f"{field}.record_paths[{record_index}] has no SOPInstanceUID"
                )
            record_uids.add(record_uid)
            treatment_date = str(getattr(record, "TreatmentDate", "") or "").strip()
            if treatment_date:
                observed_dates.add(treatment_date)
            fraction_value = getattr(record, "CurrentFractionNumber", None)
            fraction_value = fraction_value or getattr(record, "ReferencedFractionNumber", None)
            if fraction_value not in (None, "") and str(fraction_value).isdigit():
                fraction_sessions.add(("fraction", treatment_date, str(int(fraction_value))))
            elif treatment_date:
                fraction_sessions.add(("date", treatment_date, ""))
            else:
                fraction_sessions.add(("record", record_uid, ""))
            is_summary = bool(
                getattr(record, "TreatmentSummaryCalculatedDoseReferenceSequence", None)
                or str(getattr(record, "SOPClassUID", "") or "")
                == "1.2.840.10008.5.1.4.1.1.481.7"
            )
            if is_summary:
                fraction_sessions.discard(("fraction", treatment_date, str(int(fraction_value)))) if (
                    fraction_value not in (None, "") and str(fraction_value).isdigit()
                ) else fraction_sessions.discard(("date", treatment_date, "")) if treatment_date else fraction_sessions.discard(("record", record_uid, ""))
        if len(record_uids) != record_count:
            raise CourseContractError(
                f"{field}.delivered_record_count does not match the RTRECORD evidence"
            )
        if len(fraction_sessions) != fraction_count:
            raise CourseContractError(
                f"{field}.delivered_fraction_count does not match the RTRECORD evidence"
            )
        if sorted(observed_dates) != sorted(set(dates)):
            raise CourseContractError(
                f"{field}.treatment_dates does not match the RTRECORD evidence"
            )
        if (record_count or fraction_count) and not record_paths:
            raise CourseContractError(
                f"{field} claims delivery but has no auditable RTRECORD paths"
            )
    selected_from_delivery = {
        uid
        for uid, item in zip(per_plan_uids, per_plan)
        if item.get("selected_for_dose_grid") is True
    }
    if selected_from_delivery != set(plan_uids):
        raise CourseContractError(
            "selected RTPLAN membership disagrees between selected_plans and delivery.per_plan"
        )
    if status in {"fully_delivered", "partially_delivered"}:
        zero_record_selected = [
            uid
            for uid, item in zip(per_plan_uids, per_plan)
            if item.get("selected_for_dose_grid") is True
            and int(item.get("delivered_record_count") or 0) == 0
        ]
        if zero_record_selected:
            raise CourseContractError(
                "a plan with zero delivery records is selected for the treatment dose grid: "
                + ", ".join(zero_record_selected)
            )

    serialized_completeness = data.get("dose_completeness")
    if not isinstance(serialized_completeness, dict):
        raise CourseContractError(
            "course contract field dose_completeness must be an object"
        )
    expected_completeness = classify_course_dose_completeness(
        selected_plans=selected_plans,
        selected_doses=selected_doses,
        dose_classification=data["dose_classification"],
        dose_grid=data.get("dose_grid")
        if isinstance(data.get("dose_grid"), dict)
        else None,
        per_plan_delivery=per_plan,
        delivery_status=status,
        spatial_mapping_validated=bool(
            serialized_completeness.get("spatial_mapping_validated", False)
        ),
    )
    comparable_fields = (
        "status",
        "category",
        "reason_code",
        "expected_plan_uids",
        "delivered_plan_uids",
        "represented_plan_uids",
        "unselected_delivered_plan_uids",
        "dose_summation_types",
        "delivered_fraction_weights",
        "spatial_mapping_validated",
    )
    for field in comparable_fields:
        if serialized_completeness.get(field) != expected_completeness.get(field):
            raise CourseContractError(
                "stale course contract at dose_completeness: "
                f"{field} does not match authoritative evidence"
            )

    rtstruct = data.get("authoritative_rtstruct")
    if rtstruct is not None:
        if not isinstance(rtstruct, dict):
            raise CourseContractError("authoritative_rtstruct must be an object or null")
        _validate_dicom_identity(
            contract, rtstruct, "authoritative_rtstruct", role="RTSTRUCT"
        )
        source = contract.authoritative_rtstruct_source
        if not source:
            raise CourseContractError(
                "authoritative_rtstruct.segmentation_source must be nonempty when declared"
            )
        if scope == ALL_SERIES_RADIOMICS_TEMP_SCOPE and source != AUTO_RTSTRUCT_SOURCE:
            raise CourseContractError(
                "all-series temporary authoritative RTSTRUCT must declare "
                f"segmentation_source {AUTO_RTSTRUCT_SOURCE!r}"
            )

    planning_ct = contract.planning_ct
    planning_status = _nonempty_text(planning_ct.get("status"), "planning_ct.status")
    ct_dir = contract.planning_ct_dir
    nifti = contract.planning_ct_nifti
    series_uid = str(planning_ct.get("series_instance_uid") or "").strip()
    if ct_dir is not None:
        if not series_uid:
            raise CourseContractError("planning_ct.series_instance_uid is empty for a declared CT directory")
        readable_series: set[str] = set()
        for path in sorted(item for item in ct_dir.iterdir() if item.is_file()):
            try:
                dataset = pydicom.dcmread(
                    str(path),
                    stop_before_pixels=True,
                    force=True,
                )
            except Exception:
                continue
            modality = str(getattr(dataset, "Modality", "") or "").strip().upper()
            sop_class = str(getattr(dataset, "SOPClassUID", "") or "").strip()
            if not modality and not sop_class:
                # Preserve the course-level unreadable-CT check. A malformed
                # bystander is not itself a contract identity mismatch.
                continue
            if modality != "CT" or sop_class not in _ROLE_EXPECTATIONS["CT"][1]:
                raise CourseContractError(
                    f"stale planning CT contract: {path} is not a supported CT object "
                    f"(Modality={modality!r}, SOPClassUID={sop_class!r})"
                )
            value = str(getattr(dataset, "SeriesInstanceUID", "") or "").strip()
            if value:
                readable_series.add(value)
        if not readable_series:
            raise CourseContractError(
                f"planning CT contract directory contains no readable SeriesInstanceUID: {ct_dir}"
            )
        if readable_series != {series_uid}:
            raise CourseContractError(
                "stale planning CT contract: declared SeriesInstanceUID "
                f"{series_uid!r}, found {sorted(readable_series)!r}"
            )
        allow_dicom_only = (
            data.get("scope") == "all_series_radiomics_temp"
            and planning_ct.get("dicom_only") is True
        )
        if nifti is None and not allow_dicom_only:
            raise CourseContractError("planning CT contract has DICOM data but no NIfTI path")
        if nifti is not None:
            _validate_nifti_provenance(contract, planning_ct, ct_dir, nifti, series_uid)
    elif nifti is not None or series_uid:
        raise CourseContractError(
            "planning CT contract must declare DICOM directory, series UID, and NIfTI together"
        )
    elif planning_status in {"referenced", "fallback_largest"}:
        raise CourseContractError(
            f"planning CT status {planning_status!r} requires a resolved CT series"
        )

    plan_artifact = data.get("plan_artifact")
    if selected_plans and plan_artifact is None:
        raise CourseContractError("selected RTPLAN membership has no plan_artifact")
    if plan_artifact is not None:
        if not isinstance(plan_artifact, dict):
            raise CourseContractError("plan_artifact must be an object or null")
        artifact_uid = _nonempty_text(
            plan_artifact.get("sop_instance_uid"), "plan_artifact.sop_instance_uid"
        )
        _source_uids = {
            str((entry or {}).get("sop_instance_uid") or "").strip()
            for entry in (selected_plans or [])
            if isinstance(entry, dict)
        }
        # A single contracted source copied to the flat artifact path keeps that
        # source's profile. A synthesised summation carries a new UID and must
        # therefore conform to the standard class.
        _artifact_role = "RTPLAN_SOURCE" if artifact_uid in _source_uids else "RTPLAN_DERIVED"
        _validate_dicom_identity(
            contract, plan_artifact, "plan_artifact", role=_artifact_role
        )
        artifact_sources = plan_artifact.get("source_plan_uids")
        if not isinstance(artifact_sources, list) or any(
            not isinstance(value, str) or not value.strip() for value in artifact_sources
        ):
            raise CourseContractError(
                "plan_artifact.source_plan_uids must be a list of nonempty strings"
            )
        source_artifact = artifact_uid in _source_uids
        if source_artifact:
            if artifact_sources != [artifact_uid]:
                raise CourseContractError(
                    "a copied source plan_artifact must declare exactly its own "
                    "SOPInstanceUID in source_plan_uids"
                )
        else:
            expected_sources = plan_uids if plan_uids else [artifact_uid]
            if set(artifact_sources) != set(expected_sources):
                raise CourseContractError(
                    "derived plan_artifact.source_plan_uids disagrees with "
                    "selected RTPLAN membership"
                )
        artifact_path = contract.resolve_path(plan_artifact.get("path"), "plan_artifact.path")
        assert artifact_path is not None
        artifact_refs = set(
            _referenced_sop_uids(
                _read_header(artifact_path, "plan_artifact.path"),
                "ReferencedRTPlanSequence",
            )
        )
        if not source_artifact and artifact_refs != set(artifact_sources):
            raise CourseContractError(
                "derived plan_artifact references do not match "
                "plan_artifact.source_plan_uids"
            )

    dose_grid = data.get("dose_grid")
    if dose_grid is not None:
        if not isinstance(dose_grid, dict):
            raise CourseContractError("dose_grid must be an object or null")
        _validate_dicom_identity(
            contract, dose_grid, "dose_grid", summation_type=True, role="RTDOSE"
        )
        grid_type = _nonempty_text(
            dose_grid.get("dose_summation_type"), "dose_grid.dose_summation_type"
        ).upper()
        if grid_type not in PLAN_LEVEL_DOSE_SUMMATION_TYPES:
            raise CourseContractError(
                f"authoritative dose grid must be PLAN, PLAN_SUM, or MULTI_PLAN, not {grid_type!r}"
            )
        expected_semantics = (
            DOSE_GRID_SEMANTICS
            if serialized_completeness.get("status")
            == DOSE_COMPLETENESS_ELIGIBLE_STATUS
            else UNKNOWN_DELIVERY_DOSE_GRID_SEMANTICS
        )
        if dose_grid.get("semantics") != expected_semantics:
            raise CourseContractError(
                f"dose_grid.semantics must be {expected_semantics!r} for delivery status {status!r}"
            )
        grid_plan_uids = list(dose_grid.get("source_plan_uids") or [])
        grid_dose_uids = list(dose_grid.get("source_dose_uids") or [])
        grid_dose_types = list(dose_grid.get("source_dose_summation_types") or [])
        if grid_plan_uids != plan_uids:
            raise CourseContractError(
                "dose_grid.source_plan_uids disagrees with selected RTPLAN membership"
            )
        if grid_dose_uids != dose_uids:
            raise CourseContractError(
                "dose_grid.source_dose_uids disagrees with selected RTDOSE membership"
            )
        if grid_dose_types != selected_types:
            raise CourseContractError(
                "dose_grid.source_dose_summation_types disagrees with selected RTDOSE types"
            )
        if not plan_uids or not dose_uids:
            raise CourseContractError("dose grid exists without selected RTPLAN and RTDOSE sources")
        grid_uid = _nonempty_text(dose_grid.get("sop_instance_uid"), "dose_grid.sop_instance_uid")
        if grid_uid not in set(dose_uids):
            grid_path = contract.resolve_path(dose_grid.get("path"), "dose_grid.path")
            assert grid_path is not None
            grid_dataset = _read_header(grid_path, "dose_grid.path")
            grid_plan_refs = set(
                _referenced_sop_uids(
                    grid_dataset,
                    "ReferencedRTPlanSequence",
                )
            )
            grid_dose_refs = set(
                _referenced_sop_uids(
                    grid_dataset,
                    "ReferencedInstanceSequence",
                )
            )
            expected_plan_refs = set(plan_uids)
            if grid_plan_refs != expected_plan_refs or grid_dose_refs != set(dose_uids):
                raise CourseContractError(
                    "derived dose_grid references do not match its contracted source membership"
                )
    elif selected_doses:
        raise CourseContractError("selected RTDOSE objects exist but dose_grid is null")

    dvh = data.get("dvh")
    if not isinstance(dvh, dict):
        raise CourseContractError("course contract field dvh must be an object")
    expected_dvh = build_dvh_decision(
        len(plan_uids),
        len(dose_uids),
        status,
        dose_response_eligible=bool(dose_response_eligible),
        dose_completeness=serialized_completeness,
    )
    if dvh != expected_dvh:
        raise CourseContractError(
            "course contract field dvh disagrees with selected plan membership, "
            "selected dose membership, dose-grid availability, or delivery status"
        )

    dose_qc = contract.dose_qc
    qc_status = _nonempty_text(dose_qc.get("status"), "dose_qc.status")
    qc_pass = dose_qc.get("pass")
    if qc_status not in {"pass", "fail"} or not isinstance(qc_pass, bool):
        raise CourseContractError("dose_qc must carry status pass/fail and a boolean pass field")
    if (qc_status == "pass") != qc_pass:
        raise CourseContractError("dose_qc status and pass fields disagree")
    reasons = dose_qc.get("reasons")
    if not isinstance(reasons, list):
        raise CourseContractError("dose_qc.reasons must be a list")
    threshold = _optional_nonnegative_number(
        dose_qc.get("threshold_gy"),
        "dose_qc.threshold_gy",
    )
    if threshold is None or threshold <= 0:
        raise CourseContractError("dose_qc.threshold_gy must be positive")
    expected_qc_failure = any(
        value is not None and value > threshold
        for value in (resolved_prescribed, delivered)
    )
    if (not qc_pass) != expected_qc_failure:
        raise CourseContractError(
            "dose_qc verdict disagrees with resolved prescribed or delivered dose and threshold"
        )
    if qc_status != ("fail" if expected_qc_failure else "pass"):
        raise CourseContractError(
            "dose_qc.status disagrees with resolved prescribed or delivered dose and threshold"
        )
    if expected_qc_failure and not reasons:
        raise CourseContractError("failing dose_qc requires at least one reason")

    return contract


def load_course_contract(course_dir: Path | str) -> CourseContract:
    root = Path(course_dir).resolve(strict=False)
    metadata_path = root / "metadata" / "case_metadata.json"
    if not metadata_path.is_file():
        raise CourseContractError(
            f"authoritative course contract is missing: {metadata_path}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CourseContractError(
            f"authoritative course metadata is unreadable: {metadata_path}: {exc}"
        ) from exc
    if not isinstance(metadata, dict):
        raise CourseContractError(f"course metadata must be a JSON object: {metadata_path}")
    data = metadata.get("course_contract")
    if not isinstance(data, dict):
        raise CourseContractError(
            f"authoritative course contract is missing from {metadata_path}"
        )
    return validate_course_contract(
        CourseContract(course_dir=root, metadata_path=metadata_path, data=data)
    )


def relative_contract_path(course_dir: Path, path: Path | str | None) -> str:
    if path in (None, ""):
        return ""
    root = Path(course_dir).resolve(strict=False)
    resolved = Path(path).resolve(strict=False)
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise CourseContractError(
            f"contract artifact must be inside the course directory: {resolved}"
        ) from exc
