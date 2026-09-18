"""Plan-less courses must carry documented prescription-absence evidence.

Observed: undated plan-less fragments (0000-00 courses) reached the cohort
ledger with no clinical prescription evidence record at all, because the
organize adjudication only ran when selected plans existed. The readiness
gate requires every validated course to carry explicit prescription state,
and an absent record is indistinguishable from a dropped one. A course with
no plan cannot resolve a prescription, but the absence must be evidenced
against the workbook identity instead of silent.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from rtpipeline.clinical_prescription import (
    ClinicalRecord,
    ClinicalRecordIndex,
    adjudicate_clinical_prescription,
)


def _record() -> ClinicalRecord:
    return ClinicalRecord(
        patient_id="P1",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 2, 1),
        diagnosis_icd10="C67",
        treatment_type="RT",
        diagnosis="bladder cancer",
        recommendations="RT",
        description="bladder 55 Gy in 20 fractions",
        workbook_path="register.xlsx",
        workbook_sha256="0" * 64,
        sheet_name="RT Treatments",
        excel_row=2,
        record_id="R1",
    )


def _index() -> ClinicalRecordIndex:
    return ClinicalRecordIndex(
        source_path=Path("register.xlsx"),
        workbook_sha256="0" * 64,
        sheet_name="RT Treatments",
        row_count=1,
        records_by_patient={"P1": (_record(),)},
    )


def test_plan_absence_yields_workbook_bound_unresolved_evidence() -> None:
    evidence = adjudicate_clinical_prescription(
        _index(),
        patient_id="P1",
        course_id="0000-00",
        course_start_date="",
        course_end_date="",
        plan_dates=[],
        treatment_dates=[],
        dicom_resolved_total_gy=None,
        dicom_prescribed_dose_scope="UNRESOLVED_COMPONENT",
        dicom_classification=None,
        per_plan_delivery=[],
    )
    assert evidence["schema"] == "rtpipeline-clinical-prescription-evidence-v1"
    assert evidence["patient_id"] == "P1"
    assert evidence["course_id"] == "0000-00"
    assert evidence["outcome"] == "UNRESOLVED"
    assert evidence["reason"]
    assert evidence["effective_prescription_source"] is None
    assert evidence["effective_resolved_total_gy"] is None
    assert evidence["source"]["workbook_sha256"] == "0" * 64
    assert evidence["match"]["status"] in {"MATCHED", "REFUSED"}
    assert evidence["dicom"]["resolved_prescribed_dose_total_gy"] is None
    assert (
        evidence["dicom"]["prescribed_dose_scope"] == "UNRESOLVED_COMPONENT"
    )


def test_plan_absence_never_resolves_a_prescription() -> None:
    evidence = adjudicate_clinical_prescription(
        _index(),
        patient_id="P1",
        course_id="0000-00",
        course_start_date="2024-01-10",
        course_end_date="2024-01-20",
        plan_dates=[],
        treatment_dates=[],
        dicom_resolved_total_gy=None,
        dicom_prescribed_dose_scope="UNRESOLVED_COMPONENT",
        dicom_classification=None,
        per_plan_delivery=[],
    )
    assert evidence["outcome"] == "UNRESOLVED"
    assert evidence["effective_prescription_source"] is None
    assert evidence["effective_resolved_total_gy"] is None
