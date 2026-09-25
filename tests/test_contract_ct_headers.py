from __future__ import annotations

"""Equivalence and concurrency tests for planning-CT header reader.

Compares the parallel, single-pass planning-CT header reader and provenance computation
against verbatim reference copies from commit ae60f01 across all edge cases and thread counts.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, UID, generate_uid

import rtpipeline.course_contract as cc
from course_contract_test_utils import (
    write_minimal_course_contract,
    write_synthetic_plan_and_dose,
)

# Bind all course_contract internal helpers so verbatim ae60f01 reference code resolves identically
for _name in dir(cc):
    if not _name.startswith("__"):
        globals()[_name] = getattr(cc, _name)


# ---------------------------------------------------------------------------
# Verbatim reference copies from commit ae60f01
# ---------------------------------------------------------------------------

def _ct_provenance_ae60f01(ct_dir: Path) -> dict[str, Any]:
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



def _validate_nifti_provenance_ae60f01(
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
    expected_ct = _ct_provenance_ae60f01(ct_dir)
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




def validate_course_contract_ae60f01(contract: CourseContract) -> CourseContract:
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
        if approval_status(_read_header(plan_path, f"{field}.path")) != "APPROVED":
            raise CourseContractError(f"{field}: authoritative RTPLAN is not APPROVED")
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
    approved_context_uids = {
        str(item.get("plan_sop_uid") or "") for item in per_plan
        if approved_plan_paths([contract.resolve_path(item.get("plan_path"), "delivery.per_plan.plan_path")])
    }
    if not prescription_plan_uids <= approved_context_uids:
        raise CourseContractError("prescription references a plan that is not APPROVED")
    if not selected_plans and data.get("clinical_prescription_evidence") is not None:
        # A plan-less course cannot resolve a prescription, but it may attest
        # the absence: an UNRESOLVED record with no effective source carries
        # no prescription claim. Anything claiming a source stays rejected.
        absence = data.get("clinical_prescription_evidence") or {}
        if not (
            absence.get("outcome") == "UNRESOLVED"
            and absence.get("effective_prescription_source") is None
            and absence.get("effective_resolved_total_gy") is None
        ):
            raise CourseContractError("clinical prescription requires an eligible selected RTPLAN")
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
        per_plan_delivery=[item for item in per_plan if approved_plan_paths([
            contract.resolve_path(item.get("plan_path"), "delivery.per_plan.plan_path")
        ])],
    )

    def _course_value(values: list[float | None]) -> float | None:
        if prescribed_scope == CLINICAL_RESOLVED_SCOPE:
            return clinical_resolved
        return _course_value_for_scope(values, prescribed_scope)

    if prescribed_scope:
        allowed_scopes = ALLOWED_PRESCRIBED_DOSE_SCOPES
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
    approval = approval_audit([
        contract.resolve_path(item.get("plan_path"), "delivery.per_plan.plan_path")
        for item in per_plan
    ])
    serialized_approval = data["dose_classification"].get("plan_approval")
    if approval["approved_plan_count"] != len(per_plan) or serialized_approval is not None:
        # Order is not clinical evidence. Compare the exact records by SOP identity.
        def ordered_audit(value):
            if not isinstance(value, dict):
                return value
            return {**value, "plans": sorted(value.get("plans", []), key=lambda p: p["sop_instance_uid"])}
        if ordered_audit(approval) != ordered_audit(serialized_approval):
            raise CourseContractError("stale or missing plan approval disposition; rerun organize")
        if not approval["approved_plan_count"]:
            if selected_plans or selected_doses or prescribed is not None or resolved_prescribed is not None:
                raise CourseContractError("no approved plan permits no prescription or dose authority")
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
            record = _read_header(
                record_path, f"{field}.record_paths[{record_index}]", treatment_record=True
            )
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
            session_validated, nested_fraction_number, _reason = (
                _record_delivery_session_evidence(record)
            )
            if treatment_date and session_validated:
                observed_dates.add(treatment_date)
            if _is_treatment_summary_record(record):
                continue
            if session_validated:
                fraction_sessions.add(
                    _record_delivery_session_key(
                        record,
                        record_uid,
                        nested_fraction_number=nested_fraction_number,
                    )
                )
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
            _validate_nifti_provenance_ae60f01(contract, planning_ct, ct_dir, nifti, series_uid)
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



# ---------------------------------------------------------------------------
# Synthetic test case fixtures and helpers
# ---------------------------------------------------------------------------

def _make_ct_slice(
    path: Path,
    series_uid: str,
    sop_uid: str,
    *,
    modality: str | None = "CT",
    sop_class: str | None = str(CTImageStorage),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = UID(sop_class) if sop_class else UID(CTImageStorage)
    file_meta.MediaStorageSOPInstanceUID = UID(sop_uid)
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    if sop_class is not None:
        ds.SOPClassUID = str(sop_class)
    if sop_uid is not None:
        ds.SOPInstanceUID = str(sop_uid)
    if modality is not None:
        ds.Modality = str(modality)
    ds.PatientID = "SYNTHETIC_TEST_CASE"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = series_uid
    ds.Rows = 512
    ds.Columns = 512
    ds.PixelSpacing = [1.0, 1.0]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.SliceThickness = 2.5
    ds.save_as(str(path), enforce_file_format=True)


CASES = [
    "normal",
    "unreadable_garbage_file",
    "non_ct_first",
    "non_ct_not_first",
    "neither_modality_nor_sop_class",
    "two_series_uids",
    "unsupported_sop_class",
    "empty_directory",
]


def _setup_case_fixture(case: str, course_dir: Path) -> tuple[Path, cc.CourseContract]:
    course_dir.mkdir(parents=True, exist_ok=True)
    write_synthetic_plan_and_dose(course_dir)
    ct_dir = course_dir / "DICOM" / "CT"
    ct_dir.mkdir(parents=True, exist_ok=True)
    series_uid = generate_uid()

    # Create baseline valid CT slices and contract
    _make_ct_slice(ct_dir / "ct_001.dcm", series_uid, generate_uid())
    _make_ct_slice(ct_dir / "ct_002.dcm", series_uid, generate_uid())
    write_minimal_course_contract(course_dir)

    # Mutate ct_dir to match each required scenario
    if case == "normal":
        pass
    elif case == "unreadable_garbage_file":
        (ct_dir / "ct_001_garbage.bin").write_bytes(b"NOT_A_VALID_DICOM_STREAM")
    elif case == "non_ct_first":
        _make_ct_slice(
            ct_dir / "000_mr.dcm",
            series_uid,
            generate_uid(),
            modality="MR",
            sop_class="1.2.840.10008.5.1.4.1.1.4",
        )
    elif case == "non_ct_not_first":
        _make_ct_slice(
            ct_dir / "999_mr.dcm",
            series_uid,
            generate_uid(),
            modality="MR",
            sop_class="1.2.840.10008.5.1.4.1.1.4",
        )
    elif case == "neither_modality_nor_sop_class":
        _make_ct_slice(
            ct_dir / "ct_001_nomod.dcm",
            series_uid,
            generate_uid(),
            modality="",
            sop_class="",
        )
    elif case == "two_series_uids":
        _make_ct_slice(ct_dir / "ct_003.dcm", generate_uid(), generate_uid())
    elif case == "unsupported_sop_class":
        for f in list(ct_dir.iterdir()):
            f.unlink()
        _make_ct_slice(
            ct_dir / "ct_001.dcm",
            series_uid,
            generate_uid(),
            sop_class="1.2.840.10008.5.1.4.1.1.99999",
        )
    elif case == "empty_directory":
        for f in list(ct_dir.iterdir()):
            f.unlink()
    else:
        raise ValueError(f"Unknown case: {case}")

    meta_path = course_dir / "metadata" / "case_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    contract = cc.CourseContract(
        course_dir=course_dir,
        metadata_path=meta_path,
        data=meta["course_contract"],
    )
    return ct_dir, contract


# ---------------------------------------------------------------------------
# Equivalence Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("threads", [1, 2, 16])
def test_ct_headers_equivalence(case: str, threads: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert 100% equivalence between ae60f01 reference and new parallel reader across all cases and thread counts."""
    monkeypatch.setenv("RTPIPELINE_CONTRACT_HEADER_THREADS", str(threads))
    ct_dir, contract = _setup_case_fixture(case, tmp_path / f"case_{case}_t{threads}")

    # 1. Compare _ct_provenance
    old_prov, old_prov_exc = None, None
    try:
        old_prov = _ct_provenance_ae60f01(ct_dir)
    except Exception as exc:
        old_prov_exc = (type(exc), str(exc))

    new_prov, new_prov_exc = None, None
    try:
        new_prov = cc._ct_provenance(ct_dir)
    except Exception as exc:
        new_prov_exc = (type(exc), str(exc))

    assert new_prov_exc == old_prov_exc, (
        f"Case {case} (threads={threads}) provenance exception mismatch: "
        f"old={old_prov_exc} vs new={new_prov_exc}"
    )
    assert new_prov == old_prov, (
        f"Case {case} (threads={threads}) provenance return mismatch: "
        f"old={old_prov} vs new={new_prov}"
    )

    # 2. Compare validate_course_contract
    old_c, old_c_exc = None, None
    try:
        old_c = validate_course_contract_ae60f01(contract)
    except Exception as exc:
        old_c_exc = (type(exc), str(exc))

    new_c, new_c_exc = None, None
    try:
        new_c = cc.validate_course_contract(contract)
    except Exception as exc:
        new_c_exc = (type(exc), str(exc))

    assert new_c_exc == old_c_exc, (
        f"Case {case} (threads={threads}) contract validation exception mismatch: "
        f"old={old_c_exc} vs new={new_c_exc}"
    )
    if old_c is not None:
        assert new_c is not None
        assert new_c.data == old_c.data
        assert new_c.course_dir == old_c.course_dir


# ---------------------------------------------------------------------------
# Concurrency Tests
# ---------------------------------------------------------------------------

def test_ct_headers_concurrency_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch header read with fixed sleep to verify parallel latency overlap and result identity."""
    ct_dir = tmp_path / "DICOM" / "CT"
    ct_dir.mkdir(parents=True)
    series_uid = generate_uid()
    file_count = 64
    for i in range(file_count):
        _make_ct_slice(ct_dir / f"slice_{i:03d}.dcm", series_uid, generate_uid())

    orig_read = cc._read_single_ct_header

    def slow_read(p: Path):
        time.sleep(0.02)  # 20 ms latency simulation
        return orig_read(p)

    monkeypatch.setattr(cc, "_read_single_ct_header", slow_read)

    monkeypatch.setenv("RTPIPELINE_CONTRACT_HEADER_THREADS", "1")
    t0 = time.perf_counter()
    prov_serial = cc._ct_provenance(ct_dir)
    t_serial = time.perf_counter() - t0

    monkeypatch.setenv("RTPIPELINE_CONTRACT_HEADER_THREADS", "16")
    t0 = time.perf_counter()
    prov_parallel = cc._ct_provenance(ct_dir)
    t_parallel = time.perf_counter() - t0

    speedup = t_serial / t_parallel
    assert prov_parallel == prov_serial, "Parallel provenance does not match serial provenance"
    # Serial requires 64 * 0.02 = 1.28 s minimum.
    # Parallel on 16 threads requires ~4 rounds * 0.02 = 0.08 s.
    # We assert a generous speedup >= 2.5x to ensure robustness on loaded hosts.
    assert speedup >= 2.5, f"Expected speedup >= 2.5x, observed {speedup:.2f}x (serial={t_serial:.3f}s, parallel={t_parallel:.3f}s)"


# ---------------------------------------------------------------------------
# Configuration & Bounded Threads Tests
# ---------------------------------------------------------------------------

def test_configured_header_threads_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify thread configuration fallback and boundary rules."""
    monkeypatch.delenv("RTPIPELINE_CONTRACT_HEADER_THREADS", raising=False)
    assert cc._configured_header_threads() == 16

    monkeypatch.setenv("RTPIPELINE_CONTRACT_HEADER_THREADS", "1")
    assert cc._configured_header_threads() == 1

    monkeypatch.setenv("RTPIPELINE_CONTRACT_HEADER_THREADS", "8")
    assert cc._configured_header_threads() == 8

    # Non-positive values clamp to 1 (plain serial)
    monkeypatch.setenv("RTPIPELINE_CONTRACT_HEADER_THREADS", "0")
    assert cc._configured_header_threads() == 1

    monkeypatch.setenv("RTPIPELINE_CONTRACT_HEADER_THREADS", "-5")
    assert cc._configured_header_threads() == 1

    # Invalid string falls back to default 16
    monkeypatch.setenv("RTPIPELINE_CONTRACT_HEADER_THREADS", "invalid_value")
    assert cc._configured_header_threads() == 16


def test_thread_worker_pool_never_exceeds_file_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that executor worker count never exceeds the number of files."""
    ct_dir = tmp_path / "DICOM" / "CT"
    ct_dir.mkdir(parents=True)
    series_uid = generate_uid()

    # 3 files with threads=16: workers should be capped at 3
    for i in range(3):
        _make_ct_slice(ct_dir / f"slice_{i:03d}.dcm", series_uid, generate_uid())

    pool_workers_used: list[int] = []
    real_executor = cc.ThreadPoolExecutor

    class MockThreadPoolExecutor(real_executor):
        def __init__(self, *args, **kwargs):
            pool_workers_used.append(kwargs.get("max_workers", args[0] if args else None))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(cc, "ThreadPoolExecutor", MockThreadPoolExecutor)
    monkeypatch.setenv("RTPIPELINE_CONTRACT_HEADER_THREADS", "16")

    headers = cc._read_ct_headers(ct_dir)
    assert len(headers) == 3
    assert pool_workers_used == [3], f"Expected pool max_workers=3, got {pool_workers_used}"


def test_empty_directory_creates_no_thread_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that empty directory returns [] without creating any thread pool."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    pool_created = False
    real_executor = cc.ThreadPoolExecutor

    class MockThreadPoolExecutor(real_executor):
        def __init__(self, *args, **kwargs):
            nonlocal pool_created
            pool_created = True
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(cc, "ThreadPoolExecutor", MockThreadPoolExecutor)
    monkeypatch.setenv("RTPIPELINE_CONTRACT_HEADER_THREADS", "16")

    headers = cc._read_ct_headers(empty_dir)
    assert headers == []
    assert not pool_created, "Thread pool was unexpectedly created for empty directory"
