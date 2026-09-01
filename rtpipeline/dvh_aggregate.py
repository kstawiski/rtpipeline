"""Typed, denominator-preserving cohort aggregation for course DVH tables."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from pathlib import Path
from typing import Any

import pandas as pd

DVH_AGGREGATE_SCHEMA_VERSION = "rtpipeline-dvh-aggregate-v2"
DVH_IDENTIFIER_COLUMNS = (
    "patient_id",
    "course_id",
    "ROI_Number",
    "ROI_Name",
    "ROI_OriginalName",
    "ROI_Interpreted_Type",
    "target_like",
)
DVH_PROVENANCE_COLUMNS = (
    "treatment_technique",
    "treatment_technique_source",
    "rtstruct_sop_instance_uid",
    "rtstruct_path",
    "structure_provenance_type",
    "structure_provenance_status",
    "structure_provenance_reason",
    "structure_provenance_path",
)
DVH_QC_COLUMNS = (
    "relative_metric_status",
    "relative_metric_reason",
    "HI_status",
    "HI_reason",
    "zero_dose_status",
    "zero_dose_reason",
    "zero_dose_trigger_metric",
    "dose_metric_status",
    "dose_metric_reason",
    "Dose_Plan_Scope_Status",
    "Dose_Plan_Scope_Reason",
    "Course_Treatment_Isocenter_Status",
    "Course_Treatment_Isocenter_Reason",
    "Course_Target_Dose_Coverage_Status",
    "Prescribed_Dose_Status",
    "Prescribed_Dose_Reason",
    "Delivered_Dose_Status",
    "Delivered_Dose_Reason",
)
DVH_NUMERIC_COLUMNS = (
    "ROI_Number",
    "DmeanGy",
    "DmaxGy",
    "DminGy",
    "D95Gy",
    "D98Gy",
    "D2Gy",
    "D50Gy",
    "HI",
    "HI%",
    "SpreadGy",
    "Dmean%",
    "Dmax%",
    "Dmin%",
    "D95%",
    "D98%",
    "D2%",
    "D50%",
    "Spread%",
    "V95%Rx (cm³)",
    "V95%Rx (%)",
    "V100%Rx (cm³)",
    "V100%Rx (%)",
    "Volume (cm³)",
    "Prescribed_Dose_Gy",
    "Delivered_Dose_Gy",
    "zero_dose_trigger_value_gy",
    "Course_Treatment_Plan_Count",
    "Dose_Grid_Plan_Count",
    "Unrepresented_Treatment_Plan_Count",
    "Course_Treatment_Isocenter_Count",
    "Course_Treatment_Isocenter_Max_Separation_mm",
    "Course_Treatment_Isocenter_Readable_Plan_Count",
    "Course_Target_Near_Zero_Row_Count",
    "Course_Target_Outside_Dose_Grid_Count",
    "Course_Target_Partly_Inside_Dose_Grid_Count",
    "Course_Target_In_Grid_Near_Zero_Count",
    "Course_Target_Geometry_Unresolved_Count",
)
DVH_BOOLEAN_COLUMNS = (
    "structure_cropped",
    "target_like",
    "dose_metric_usable_for_dose_response",
    "dose_response_eligible",
    "Dose_Response_Eligible",
)
DVH_STRING_COLUMNS = (
    "patient_id",
    "course_id",
    "ROI_Name",
    "ROI_OriginalName",
    "ROI_Interpreted_Type",
    *DVH_PROVENANCE_COLUMNS,
    *DVH_QC_COLUMNS,
    "Unrepresented_Treatment_Plan_UIDs",
    "row_status",
    "failure_reason",
    "aggregate_schema_version",
)
DVH_RELATIVE_COLUMNS = (
    "Dmean%",
    "Dmax%",
    "Dmin%",
    "D95%",
    "D98%",
    "D2%",
    "D50%",
    "HI%",
    "Spread%",
    "V95%Rx (cm³)",
    "V95%Rx (%)",
    "V100%Rx (cm³)",
    "V100%Rx (%)",
)
DVH_DOSE_METRIC_COLUMNS = (
    "DmeanGy",
    "DmaxGy",
    "DminGy",
    "D95Gy",
    "D98Gy",
    "D2Gy",
    "D50Gy",
    "D1ccGy",
    "D0.1ccGy",
    "HI",
    "SpreadGy",
    "IntegralDose_Gycm3",
    *DVH_RELATIVE_COLUMNS,
)
DVH_REQUIRED_COLUMNS = (
    *DVH_IDENTIFIER_COLUMNS,
    "row_status",
    "failure_reason",
    "aggregate_schema_version",
    *DVH_PROVENANCE_COLUMNS,
    *DVH_QC_COLUMNS,
)


def _dose_metric_column(column: object) -> bool:
    name = str(column)
    if name in DVH_DOSE_METRIC_COLUMNS:
        return True
    return name.startswith("V") and any(
        token in name
        for token in ("Gy (cm³)", "Gy (%)", "%Rx (cm³)", "%Rx (%)")
    )


def _course_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("patient_id") or ""), str(row.get("course_id") or "")


def _failure_reason(
    key: tuple[str, str],
    incomplete: Mapping[tuple[str, str], Iterable[str]],
    expected_noncomputed: Mapping[tuple[str, str], str],
) -> tuple[str, str]:
    failures = [str(value) for value in incomplete.get(key, ()) if str(value).strip()]
    if failures:
        return "failed", " | ".join(failures)
    reason = str(expected_noncomputed.get(key) or "").strip()
    if reason:
        return "not_computed", reason
    return "failed", "DVH output is missing from the validated course aggregate inputs."


def _course_treatment_technique(course_dir: Path) -> tuple[object, object]:
    try:
        payload = json.loads(
            (course_dir / "metadata" / "case_metadata.json").read_text(encoding="utf-8")
        )
        contract = payload.get("course_contract") or {}
        technique = contract.get("treatment_technique") or {}
        if isinstance(technique, dict):
            return technique.get("classification"), "DICOM_RTPLAN"
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return pd.NA, pd.NA


def _normalise_types(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in DVH_NUMERIC_COLUMNS:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float64")
    for column in DVH_BOOLEAN_COLUMNS:
        if column in frame:
            frame[column] = frame[column].astype("boolean")
    for column in DVH_STRING_COLUMNS:
        if column in frame:
            frame[column] = frame[column].astype("string")
    return frame


def build_dvh_aggregate(
    frames: Iterable[pd.DataFrame],
    courses: Iterable[tuple[str, str, Path]],
    *,
    incomplete: Mapping[tuple[str, str], Iterable[str]] | None = None,
    expected_noncomputed: Mapping[tuple[str, str], str] | None = None,
) -> pd.DataFrame:
    """Combine DVH rows and retain one explicit row for every failed course."""
    incomplete = incomplete or {}
    expected_noncomputed = expected_noncomputed or {}
    valid_frames: list[pd.DataFrame] = []
    successful_keys: set[tuple[str, str]] = set()
    for source in frames:
        if source is None or source.empty:
            continue
        current = source.copy()
        if "patient_id" not in current:
            current.insert(0, "patient_id", pd.NA)
        if "course_id" not in current:
            current.insert(1, "course_id", pd.NA)
        if "treatment_technique" in current:
            non_ebrt = ~current["treatment_technique"].astype("string").eq("EBRT")
            for column in DVH_RELATIVE_COLUMNS:
                if column in current:
                    current.loc[non_ebrt, column] = pd.NA
            if "relative_metric_status" in current:
                current.loc[non_ebrt, "relative_metric_status"] = "suppressed_non_ebrt"
            if "relative_metric_reason" in current:
                current.loc[non_ebrt, "relative_metric_reason"] = (
                    "Prescription-relative metrics are suppressed for non-EBRT courses."
                )
        if "zero_dose_status" in current:
            outside_grid = current["zero_dose_status"].astype("string").eq(
                "zero_dose_outside_dose_grid"
            )
            if bool(outside_grid.any()):
                for column in current.columns:
                    if _dose_metric_column(column):
                        current.loc[outside_grid, column] = pd.NA
                if "dose_metric_usable_for_dose_response" not in current:
                    current["dose_metric_usable_for_dose_response"] = pd.NA
                current.loc[
                    outside_grid, "dose_metric_usable_for_dose_response"
                ] = False
                if "dose_metric_status" not in current:
                    current["dose_metric_status"] = pd.NA
                current.loc[outside_grid, "dose_metric_status"] = (
                    "not_measurable_outside_dose_grid"
                )
                if "dose_metric_reason" not in current:
                    current["dose_metric_reason"] = pd.NA
                current.loc[outside_grid, "dose_metric_reason"] = (
                    "Dose-derived metrics are null because the ROI does not intersect "
                    "the selected RTDOSE grid."
                )
        current["row_status"] = "computed"
        current["failure_reason"] = pd.NA
        current["aggregate_schema_version"] = DVH_AGGREGATE_SCHEMA_VERSION
        valid_frames.append(current)
        successful_keys.update(
            (str(patient), str(course))
            for patient, course in zip(current["patient_id"], current["course_id"])
        )

    rows: list[dict[str, Any]] = []
    for patient_id, course_id, _course_dir in courses:
        key = (str(patient_id), str(course_id))
        if key in successful_keys:
            continue
        status, reason = _failure_reason(key, incomplete, expected_noncomputed)
        technique, technique_source = _course_treatment_technique(_course_dir)
        rows.append(
            {
                "patient_id": str(patient_id),
                "course_id": str(course_id),
                "ROI_Number": pd.NA,
                "ROI_Name": pd.NA,
                "ROI_OriginalName": pd.NA,
                "row_status": status,
                "failure_reason": reason,
                "aggregate_schema_version": DVH_AGGREGATE_SCHEMA_VERSION,
                "treatment_technique": technique,
                "treatment_technique_source": technique_source,
                "structure_provenance_status": "not_available",
                "structure_provenance_reason": "No DVH row exists because the course was not computed.",
                "relative_metric_status": "not_available",
                "relative_metric_reason": "No DVH row exists because the course was not computed.",
                "HI_status": "not_available",
                "HI_reason": "No DVH row exists because the course was not computed.",
                "zero_dose_status": "not_available",
                "zero_dose_reason": "No DVH row exists because the course was not computed.",
                "zero_dose_trigger_metric": pd.NA,
                "dose_metric_status": "not_available",
                "dose_metric_reason": "No DVH row exists because the course was not computed.",
                "Dose_Plan_Scope_Status": "not_available",
                "Dose_Plan_Scope_Reason": "No DVH row exists because the course was not computed.",
                "Course_Treatment_Isocenter_Status": "not_available",
                "Course_Treatment_Isocenter_Reason": "No DVH row exists because the course was not computed.",
                "Course_Target_Dose_Coverage_Status": "not_available",
                "Prescribed_Dose_Status": "not_available",
                "Prescribed_Dose_Reason": "No DVH row exists because the course was not computed.",
                "Delivered_Dose_Status": "not_available",
                "Delivered_Dose_Reason": "No DVH row exists because the course was not computed.",
            }
        )
    combined = pd.concat([*valid_frames, pd.DataFrame(rows)], ignore_index=True, sort=False)
    for column in DVH_REQUIRED_COLUMNS:
        if column not in combined:
            combined[column] = pd.NA
    ordered = [
        *DVH_IDENTIFIER_COLUMNS,
        "row_status",
        "failure_reason",
        "aggregate_schema_version",
        *[column for column in DVH_PROVENANCE_COLUMNS if column not in DVH_IDENTIFIER_COLUMNS],
        *DVH_QC_COLUMNS,
        *[column for column in combined.columns if column not in DVH_REQUIRED_COLUMNS],
    ]
    combined = combined.loc[:, ordered]
    return _normalise_types(combined)


def write_dvh_aggregate(frame: pd.DataFrame, xlsx_path: Path) -> None:
    """Write the human-readable workbook and a typed Parquet sidecar."""
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_excel(xlsx_path, index=False)
    try:
        frame.to_parquet(xlsx_path.with_suffix(".parquet"), index=False)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to write typed DVH aggregate sidecar {xlsx_path.with_suffix('.parquet')}: {exc}"
        ) from exc
