"""A plan-less course may attest its prescription absence, never claim one.

Regression: emitting UNRESOLVED plan-absence evidence tripped the contract
gate ("clinical prescription requires an eligible selected RTPLAN") and
quarantined 9 previously validated undated fragments. The gate guards
against unattributed prescription CLAIMS; an explicit absence attestation
carries none and must validate, while any effective-source claim without
plans must still be rejected.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rtpipeline.course_contract import (
    CourseContractError,
    load_course_contract,
    validate_course_contract,
)

from course_contract_test_utils import write_minimal_course_contract


def _absence_evidence() -> dict:
    return {
        "schema": "rtpipeline-clinical-prescription-evidence-v1",
        "patient_id": "PX",
        "course_id": "0000-00",
        "outcome": "UNRESOLVED",
        "reason": "NO_PLAN_NO_MATCHING_EVIDENCE",
        "effective_prescription_source": None,
        "effective_resolved_total_gy": None,
        "source": {
            "workbook_sha256": "0" * 64,
            "workbook_path": "register.xlsx",
            "sheet_name": "RT Treatments",
            "row_count": 1,
        },
        "match": {"status": "REFUSED"},
        "dicom": {
            "resolved_prescribed_dose_total_gy": None,
            "prescribed_dose_scope": "UNRESOLVED_COMPONENT",
            "delivery_status": "no_records_at_all",
        },
    }


def _contract_with_evidence(tmp_path: Path, evidence: object):
    course_dir = tmp_path / "PX" / "0000-00"
    course_dir.mkdir(parents=True)
    write_minimal_course_contract(course_dir, selected_plans=[], selected_doses=[])
    contract = load_course_contract(course_dir)
    # Mirror production plan-less courses, whose dose classification leaves
    # an UNRESOLVED scope that the evidence snapshot must agree with.
    contract.delivery["prescribed_dose_scope"] = "UNRESOLVED_COMPONENT"
    contract.data.setdefault("dose_classification", {})["prescribed_dose_scope"] = (
        "UNRESOLVED_COMPONENT"
    )
    contract.data["clinical_prescription_evidence"] = evidence
    return contract


def test_plan_less_absence_attestation_validates(tmp_path: Path) -> None:
    contract = _contract_with_evidence(tmp_path, _absence_evidence())
    validate_course_contract(contract)


def test_plan_less_effective_source_claim_still_rejected(tmp_path: Path) -> None:
    evidence = _absence_evidence()
    evidence["effective_prescription_source"] = "CLINICAL_RECORD"
    evidence["effective_resolved_total_gy"] = 50.0
    contract = _contract_with_evidence(tmp_path, evidence)
    with pytest.raises(CourseContractError, match="eligible selected RTPLAN"):
        validate_course_contract(contract)
