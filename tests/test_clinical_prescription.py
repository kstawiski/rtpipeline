from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from rtpipeline import cli
from rtpipeline import organize
from rtpipeline.clinical_prescription import (
    ClinicalRecord,
    ClinicalRecordIndex,
    adjudicate_clinical_prescription,
    clinical_evidence_content_sha256,
    clinical_evidence_matches_source,
    confirm_two_phase_fractionation,
    load_kopernik_treatment_records,
    match_clinical_record,
    parse_kopernik_treatment_description,
    record_clinical_evidence_regeneration,
)
from rtpipeline.config import PipelineConfig
from rtpipeline.course_contract import (
    CourseContract,
    CourseContractError,
    _validate_clinical_prescription_evidence,
)


REAL_DESCRIPTIONS = {
    "351107/2021-12": (
        "ICD9: 92.24: Teleradioterapia 3D niekoplanarna (3DCRT) zmiany "
        "przerzutowej na obszar guza cewki moczowej do dawki 20,0Gy/p.ref "
        "po 4,0Gy technika box"
    ),
    "432976/2020-04": (
        "ICD9: 92.24: Teleradioterapia 3D conformalna fotonami X 6 MeV "
        "na obszar pecherza moczowego i wezlow chlonnych miednicy mniejszej "
        "do dawki 36,0Gy/p.ref po 3,0Gy"
    ),
    "446498/2022-04": (
        "ICD9: 92.24: Teleradioterapia radykalna technika VMAT, energia "
        "fotonow 6MV, na obszar lozy po cystektomii do dawki 50,4Gy/p.ref "
        "po 1,8Gy i na obszar regionalnych wezlow chlonnych do dawki "
        "50,4Gy/p.ref po 1,8Gy"
    ),
    "451645/2025-12": (
        "ICD9: 92.27: Radioterapia paliatywna technika IMRT na obszar "
        "pecherza moczowego, moczowod lewy do dawki 60,0Gy/p.ref po 3,0Gy "
        "oraz na obszar regionalnych ww chlonnych do dawki 44,0Gy/p.ref "
        "po 2,2Gy"
    ),
    "426529/2023-11": (
        "ICD9: 92.24: Teleradioterapia SBRT na obszar trzonu kregu Th7 do "
        "dawki 25,0Gy/p.ref po 5,0Gy oraz teleradioterapia SBRT na obszar "
        "trzonu kregu L3 do dawki 27,0Gy w 3 frakcjach po 9,0Gy"
    ),
    "440657/2024-10": "",
    "419783/2020-02": (
        "ICD9: 92.27: Teleradioterapia radykalna technika IMRT na obszar "
        "pecherza moczowego do dawki 37,0Gy/p.ref po 2,0Gy (6 frakcji) "
        "i 2,5Gy (10 frakcji)"
    ),
}


def _record(
    *,
    record_id: str = "record-1",
    patient_id: str = "P1",
    start: date = date(2020, 2, 10),
    end: date = date(2020, 3, 9),
    description: str = REAL_DESCRIPTIONS["419783/2020-02"],
    excel_row: int = 2,
) -> ClinicalRecord:
    return ClinicalRecord(
        record_id=record_id,
        patient_id=patient_id,
        start_date=start,
        end_date=end,
        diagnosis_icd10="C67",
        treatment_type="Radioterapia",
        diagnosis="nowotwor pecherza",
        recommendations="",
        description=description,
        workbook_path="/source/rt_treatments.xlsx",
        workbook_sha256="a" * 64,
        sheet_name="Arkusz1",
        excel_row=excel_row,
    )


def _index(*records: ClinicalRecord) -> ClinicalRecordIndex:
    by_patient: dict[str, list[ClinicalRecord]] = {}
    for record in records:
        by_patient.setdefault(record.patient_id, []).append(record)
    return ClinicalRecordIndex(
        source_path=Path("/source/rt_treatments.xlsx"),
        workbook_sha256="a" * 64,
        sheet_name="Arkusz1",
        row_count=len(records),
        records_by_patient={key: tuple(value) for key, value in by_patient.items()},
    )


def _plan_delivery() -> list[dict[str, object]]:
    return [
        {
            "plan_sop_uid": "phase-1",
            "prescribed_dose_gy": 50.0,
            "planned_fraction_count": 25,
            "delivered_fraction_count": 6,
        },
        {
            "plan_sop_uid": "phase-2",
            "prescribed_dose_gy": 25.0,
            "planned_fraction_count": 10,
            "delivered_fraction_count": 10,
        },
    ]


@pytest.mark.parametrize(
    ("course_key", "expected"),
    [
        ("351107/2021-12", [(20.0, 5, [4.0])]),
        ("432976/2020-04", [(36.0, 12, [3.0])]),
        ("446498/2022-04", [(50.4, 28, [1.8]), (50.4, 28, [1.8])]),
        ("451645/2025-12", [(60.0, 20, [3.0]), (44.0, 20, [2.2])]),
        ("426529/2023-11", [(25.0, 5, [5.0]), (27.0, 3, [9.0])]),
        ("419783/2020-02", [(37.0, 16, [2.0, 2.5])]),
    ],
)
def test_parser_recovers_real_kopernik_fractionation(
    course_key: str,
    expected: list[tuple[float, int, list[float]]],
) -> None:
    parsed = parse_kopernik_treatment_description(REAL_DESCRIPTIONS[course_key])

    assert parsed["status"] == "PARSED"
    observed = [
        (
            site["total_dose_gy"],
            site["fraction_count"],
            [phase["dose_per_fraction_gy"] for phase in site["phases"]],
        )
        for site in parsed["sites"]
    ]
    assert observed == expected
    assert all(site["site"] for site in parsed["sites"])
    assert all(
        site["self_check"]["matches_stated_total"] for site in parsed["sites"]
    )


def test_empty_real_record_refuses() -> None:
    parsed = parse_kopernik_treatment_description(
        REAL_DESCRIPTIONS["440657/2024-10"]
    )
    assert parsed["status"] == "REFUSED"
    assert parsed["reason"] == "EMPTY_DESCRIPTION"
    assert parsed["sites"] == []


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (
            "na obszar pecherza do dawki 37Gy po 2Gy",
            "NONINTEGRAL_IMPLIED_FRACTION_COUNT",
        ),
        (
            "na obszar pecherza do dawki 37Gy po 2Gy (6 frakcji) i 2.5Gy",
            "INCOMPLETE_MULTIPHASE_FRACTION_COUNTS",
        ),
        (
            "na obszar pecherza do dawki 37Gy po 2Gy (6 frakcji) i 2.5Gy (9 frakcji)",
            "STATED_FRACTIONATION_TOTAL_MISMATCH",
        ),
    ],
)
def test_parser_refuses_failed_arithmetic(text: str, reason: str) -> None:
    parsed = parse_kopernik_treatment_description(text)
    assert parsed["status"] == "REFUSED"
    assert parsed["reason"] == reason
    assert parsed["sites"] == []


def test_matching_uses_direct_temporal_evidence_not_nearest_date() -> None:
    brachy = _record(
        record_id="brachy-2020",
        patient_id="327471",
        start=date(2020, 12, 1),
        end=date(2020, 12, 4),
    )
    palliative = _record(
        record_id="palliative-2021",
        patient_id="327471",
        start=date(2021, 2, 8),
        end=date(2021, 2, 12),
    )
    index = _index(brachy, palliative)

    matched = match_clinical_record(
        index.records_for("327471"),
        course_start_date="2020-12-01",
        course_end_date="2020-12-04",
        plan_dates=["20201201"],
        treatment_dates=["20201201", "20201204"],
    )
    no_match = match_clinical_record(
        index.records_for("327471"),
        course_start_date="",
        course_end_date="",
        plan_dates=["20210131"],
        treatment_dates=[],
    )

    assert matched["status"] == "MATCHED"
    assert matched["matched_record"].record_id == "brachy-2020"
    assert no_match["status"] == "REFUSED"
    assert no_match["reason"] == "NO_TEMPORAL_MATCH"


def test_matching_refuses_equal_temporal_candidates() -> None:
    first = _record(record_id="first", excel_row=2)
    second = replace(_record(record_id="second", excel_row=3), description="")

    matched = match_clinical_record(
        _index(first, second).records_for("P1"),
        course_start_date="2020-02-10",
        course_end_date="2020-03-09",
        plan_dates=["20200224"],
        treatment_dates=["20200210", "20200309"],
    )

    assert matched["status"] == "REFUSED"
    assert matched["reason"] == "AMBIGUOUS_RECORD_MATCH"
    assert matched["candidate_record_ids"] == ["first", "second"]


def test_adjudication_resolves_corroborates_and_flags_disagreement() -> None:
    index = _index(_record())
    common = dict(
        index=index,
        patient_id="P1",
        course_id="2020-02",
        course_start_date="2020-02-10",
        course_end_date="2020-03-09",
        plan_dates=["20200224", "20200312"],
        treatment_dates=["20200210", "20200309"],
        dicom_prescribed_dose_scope="UNRESOLVED_REPLACEMENT_CHAIN",
        dicom_classification="replacement_plan_chain",
        per_plan_delivery=_plan_delivery(),
    )

    resolved = adjudicate_clinical_prescription(
        **common, dicom_resolved_total_gy=None
    )
    corroborated = adjudicate_clinical_prescription(
        **{**common, "dicom_prescribed_dose_scope": "SINGLE_PLAN_TOTAL"},
        dicom_resolved_total_gy=37.0,
    )
    disagrees = adjudicate_clinical_prescription(
        **{**common, "dicom_prescribed_dose_scope": "SINGLE_PLAN_TOTAL"},
        dicom_resolved_total_gy=36.0,
    )

    assert resolved["outcome"] == "RESOLVED_FROM_CLINICAL_RECORD"
    assert resolved["clinical_resolved_total_gy"] == 37.0
    assert resolved["effective_prescription_source"] == "CLINICAL_RECORD"
    assert resolved["record"]["source_text"] == REAL_DESCRIPTIONS["419783/2020-02"]
    assert resolved["record"]["parsed_field"] == "Opis leczenia"
    assert resolved["fractionation_classification"]["classification"] == (
        "TWO_FRACTIONATION_PHASES"
    )
    assert corroborated["outcome"] == "CORROBORATED_DICOM"
    assert corroborated["effective_prescription_source"] == "DICOM"
    assert disagrees["outcome"] == "DISAGREES_WITH_DICOM"
    assert disagrees["effective_resolved_total_gy"] == 36.0
    assert disagrees["disagreement"]["clinical_site_totals"] == [
        {"site": "pecherza moczowego", "total_dose_gy": 37.0}
    ]


@pytest.mark.parametrize(
    ("course_key", "clinical_total_gy"),
    [
        ("432976/2020-04", 36.0),
        ("446498/2022-04", 50.4),
    ],
)
def test_task27_clinical_resume_restores_real_course_dicom_snapshot(
    tmp_path: Path,
    course_key: str,
    clinical_total_gy: float,
) -> None:
    """Reproduce the 36/50.4 Gy versus null Task 27 resume mismatch."""

    previous_evidence = {
        "schema": "rtpipeline-clinical-prescription-evidence-v1",
        "outcome": "RESOLVED_FROM_CLINICAL_RECORD",
        "source": {"workbook_sha256": "a" * 64},
        "dicom": {
            "resolved_prescribed_dose_total_gy": None,
            "prescribed_dose_scope": "UNRESOLVED_COMPONENT",
            "dose_classification": "sequential_phases_summed",
            "delivered_dose_gy": None,
            "delivery_status": "delivery_unresolved",
            "delivery_method": "unresolved_course_prescription_scope",
        },
    }
    data = {
        "delivery": {
            "prescribed_dose_gy": clinical_total_gy,
            "resolved_prescribed_dose_total_gy": clinical_total_gy,
            "delivered_dose_gy": None,
            "status": "delivery_unresolved",
            "method": "course_delivery_scalar_unavailable",
        },
        "dose_classification": {
            "classification": "sequential_phases_summed",
            "should_sum": True,
            "prescribed_dose_scope": "COURSE_TOTAL_CLINICAL_RECORD",
            "dicom_prescribed_dose_scope": "UNRESOLVED_COMPONENT",
        },
        "clinical_prescription_evidence": previous_evidence,
    }
    contract = CourseContract(
        course_dir=tmp_path / course_key,
        metadata_path=tmp_path / course_key / "metadata" / "case_metadata.json",
        data=data,
    )

    # The pre-fix resume path fed the published clinical scalar back into the
    # evidence's DICOM field. Contract validation recomputed null from the plans.
    assert contract.resolved_prescribed_dose_total_gy == clinical_total_gy
    assert previous_evidence["dicom"]["resolved_prescribed_dose_total_gy"] is None

    hydrated = organize._hydrated_preclinical_dose_state(contract)

    assert hydrated["resolved_prescribed_dose_total_gy"] is None
    assert hydrated["prescribed_dose_gy"] is None
    assert hydrated["dose_classification"]["prescribed_dose_scope"] == (
        "UNRESOLVED_COMPONENT"
    )

    regenerated = record_clinical_evidence_regeneration(
        dict(previous_evidence), previous_evidence
    )
    provenance = regenerated["regeneration_provenance"]
    assert provenance["previous_evidence_payload_sha256"] == (
        clinical_evidence_content_sha256(previous_evidence)
    )
    assert provenance["previous_evidence_payload"] == previous_evidence
    assert provenance["current_dicom_snapshot"] == regenerated["dicom"]


def test_task27_stale_dicom_guard_still_rejects_the_real_432976_mismatch() -> None:
    evidence = adjudicate_clinical_prescription(
        _index(
            _record(
                patient_id="432976",
                start=date(2020, 4, 15),
                end=date(2020, 4, 30),
                description=REAL_DESCRIPTIONS["432976/2020-04"],
            )
        ),
        patient_id="432976",
        course_id="2020-04",
        course_start_date="2020-04-15",
        course_end_date="2020-04-30",
        plan_dates=["20200428"],
        treatment_dates=["20200415", "20200430"],
        dicom_resolved_total_gy=None,
        dicom_prescribed_dose_scope="UNRESOLVED_COMPONENT",
        dicom_classification="sequential_phases_summed",
        per_plan_delivery=[],
    )
    evidence["dicom"].update(
        {
            "resolved_prescribed_dose_total_gy": 36.0,
            "delivered_dose_gy": None,
            "delivery_status": "delivery_unresolved",
            "delivery_method": "unresolved_course_prescription_scope",
        }
    )

    with pytest.raises(
        CourseContractError,
        match=r"snapshot=36\.0, recomputed=None",
    ):
        _validate_clinical_prescription_evidence(
            evidence,
            dicom_resolved_total_gy=None,
            dicom_prescribed_scope="UNRESOLVED_COMPONENT",
            prescribed_scope="COURSE_TOTAL_CLINICAL_RECORD",
            per_plan_delivery=[],
        )

    evidence["dicom"]["resolved_prescribed_dose_total_gy"] = None
    regenerated = record_clinical_evidence_regeneration(dict(evidence), evidence)
    regenerated["regeneration_provenance"]["previous_evidence_payload"][
        "dicom"
    ]["resolved_prescribed_dose_total_gy"] = 99.0
    with pytest.raises(
        CourseContractError,
        match="regeneration provenance hash is stale",
    ):
        _validate_clinical_prescription_evidence(
            regenerated,
            dicom_resolved_total_gy=None,
            dicom_prescribed_scope="UNRESOLVED_COMPONENT",
            prescribed_scope="COURSE_TOTAL_CLINICAL_RECORD",
            per_plan_delivery=[],
        )


def test_task27_resume_preserves_recovered_419783_dicom_phase_provenance(
    tmp_path: Path,
) -> None:
    data = {
        "delivery": {
            "prescribed_dose_gy": 37.0,
            "resolved_prescribed_dose_total_gy": 37.0,
            "delivered_dose_gy": 37.0,
            "status": "fully_delivered",
            "method": "clinical_two_phase_fractionation_with_dicom_records",
        },
        "dose_classification": {
            "classification": "TWO_FRACTIONATION_PHASES",
            "prescribed_dose_scope": "COURSE_TOTAL_CLINICAL_RECORD",
            "dicom_prescribed_dose_scope": "UNRESOLVED_REPLACEMENT_CHAIN",
            "dicom_classification": "replacement_plan_chain",
            "dicom_reason": "delivered treatment spans a replacement chain",
            "dicom_warnings": ["replacement plans need clinical adjudication"],
        },
        "clinical_prescription_evidence": {
            "outcome": "RESOLVED_FROM_CLINICAL_RECORD",
            "dicom": {
                "resolved_prescribed_dose_total_gy": None,
                "prescribed_dose_scope": "UNRESOLVED_REPLACEMENT_CHAIN",
                "delivered_dose_gy": 37.0,
                "delivery_status": "partially_delivered",
                "delivery_method": "calculated_dose_reference",
            },
        },
    }
    contract = CourseContract(
        course_dir=tmp_path / "419783" / "2020-02",
        metadata_path=(
            tmp_path
            / "419783"
            / "2020-02"
            / "metadata"
            / "case_metadata.json"
        ),
        data=data,
    )

    hydrated = organize._hydrated_preclinical_dose_state(contract)

    assert hydrated["resolved_prescribed_dose_total_gy"] is None
    assert hydrated["delivered_dose_gy"] == 37.0
    assert hydrated["delivery_status"] == "partially_delivered"
    assert hydrated["delivery_method"] == "calculated_dose_reference"
    assert hydrated["dose_classification"]["classification"] == (
        "replacement_plan_chain"
    )
    assert hydrated["dose_classification"]["prescribed_dose_scope"] == (
        "UNRESOLVED_REPLACEMENT_CHAIN"
    )


def test_partial_treatment_overlap_cannot_resolve_dicom_unknown_course() -> None:
    record = _record(
        patient_id="P1",
        start=date(2024, 1, 10),
        end=date(2024, 1, 12),
        description="na obszar pecherza do dawki 30Gy po 3Gy",
    )
    evidence = adjudicate_clinical_prescription(
        _index(record),
        patient_id="P1",
        course_id="2024-01",
        course_start_date="2024-01-10",
        course_end_date="2024-01-30",
        plan_dates=[],
        treatment_dates=["20240110", "20240120"],
        dicom_resolved_total_gy=None,
        dicom_prescribed_dose_scope="UNRESOLVED_COMPONENT",
        dicom_classification="component",
        per_plan_delivery=[],
    )

    assert evidence["match"]["match_evidence"]["basis"] == "TREATMENT_DATE_OVERLAP"
    assert evidence["outcome"] == "UNRESOLVED"
    assert evidence["reason"] == "INSUFFICIENT_TEMPORAL_EVIDENCE_FOR_RESOLUTION"


def test_distinct_multisite_totals_remain_per_site_and_do_not_resolve() -> None:
    record = _record(description=REAL_DESCRIPTIONS["451645/2025-12"])
    evidence = adjudicate_clinical_prescription(
        _index(record),
        patient_id="P1",
        course_id="2025-12",
        course_start_date="2020-02-10",
        course_end_date="2020-03-09",
        plan_dates=[],
        treatment_dates=["20200210"],
        dicom_resolved_total_gy=None,
        dicom_prescribed_dose_scope="UNRESOLVED_COMPONENT",
        dicom_classification="component",
        per_plan_delivery=[],
    )

    assert evidence["outcome"] == "UNRESOLVED"
    assert evidence["reason"] == "MULTISITE_DISTINCT_TOTALS"
    assert [site["total_dose_gy"] for site in evidence["parse"]["sites"]] == [
        60.0,
        44.0,
    ]


def test_two_phase_confirmation_requires_unique_plan_phase_binding() -> None:
    parsed = parse_kopernik_treatment_description(
        REAL_DESCRIPTIONS["419783/2020-02"]
    )
    confirmation = confirm_two_phase_fractionation(
        parsed["sites"], _plan_delivery()
    )

    assert confirmation is not None
    assert confirmation["clinical_total_gy"] == 37.0
    assert confirmation["dicom_delivered_total_gy"] == 37.0
    assert confirmation["phase_plan_bindings"] == [
        {
            "clinical_fraction_count": 6,
            "clinical_dose_per_fraction_gy": 2.0,
            "plan_sop_uid": "phase-1",
            "delivered_fraction_count": 6,
            "dose_per_fraction_gy": 2.0,
        },
        {
            "clinical_fraction_count": 10,
            "clinical_dose_per_fraction_gy": 2.5,
            "plan_sop_uid": "phase-2",
            "delivered_fraction_count": 10,
            "dose_per_fraction_gy": 2.5,
        },
    ]
    ambiguous = _plan_delivery() + [
        {
            "plan_sop_uid": "duplicate",
            "prescribed_dose_gy": 12.0,
            "planned_fraction_count": 6,
            "delivered_fraction_count": 6,
        }
    ]
    assert confirm_two_phase_fractionation(parsed["sites"], ambiguous) is None
    mismatched_total = [dict(parsed["sites"][0], total_dose_gy=38.0)]
    assert confirm_two_phase_fractionation(mismatched_total, _plan_delivery()) is None


@pytest.mark.parametrize(
    (
        "case",
        "outcome",
        "include_phase_confirmation",
        "scope",
        "course_total",
        "plan_total",
        "raw_delivered",
        "raw_status",
        "expected_delivered",
        "expected_status",
        "expected_method",
    ),
    [
        (
            "clinical-resolved-and-independently-delivered",
            "RESOLVED_FROM_CLINICAL_RECORD",
            True,
            "COURSE_TOTAL_CLINICAL_RECORD",
            37.0,
            50.0,
            None,
            "delivery_unresolved",
            37.0,
            "fully_delivered",
            "clinical_two_phase_fractionation_with_dicom_records",
        ),
        (
            "clinical-resolved-without-delivered-scalar",
            "RESOLVED_FROM_CLINICAL_RECORD",
            False,
            "COURSE_TOTAL_CLINICAL_RECORD",
            36.0,
            36.0,
            36.0,
            "fully_delivered",
            None,
            "delivery_unresolved",
            "course_delivery_scalar_unavailable",
        ),
        (
            "dicom-resolved-single-plan",
            None,
            False,
            "SINGLE_PLAN_TOTAL",
            37.0,
            37.0,
            37.0,
            "fully_delivered",
            37.0,
            "fully_delivered",
            "calculated_dose_reference",
        ),
        (
            "dicom-resolved-summed-course",
            None,
            False,
            "COURSE_TOTAL_SUMMED",
            62.0,
            50.0,
            62.0,
            "fully_delivered",
            62.0,
            "fully_delivered",
            "calculated_dose_reference",
        ),
        (
            "clinical-corroborates-dicom",
            "CORROBORATED_DICOM",
            True,
            "SINGLE_PLAN_TOTAL",
            37.0,
            37.0,
            37.0,
            "fully_delivered",
            37.0,
            "fully_delivered",
            "calculated_dose_reference",
        ),
        (
            "clinical-corroborates-dicom-summed-course",
            "CORROBORATED_DICOM",
            True,
            "COURSE_TOTAL_SUMMED",
            62.0,
            50.0,
            62.0,
            "fully_delivered",
            62.0,
            "fully_delivered",
            "calculated_dose_reference",
        ),
        (
            "clinical-disagrees-with-dicom",
            "DISAGREES_WITH_DICOM",
            True,
            "SINGLE_PLAN_TOTAL",
            36.0,
            36.0,
            36.0,
            "fully_delivered",
            36.0,
            "fully_delivered",
            "calculated_dose_reference",
        ),
        (
            "clinical-disagrees-with-dicom-summed-course",
            "DISAGREES_WITH_DICOM",
            True,
            "COURSE_TOTAL_SUMMED",
            61.0,
            50.0,
            61.0,
            "fully_delivered",
            61.0,
            "fully_delivered",
            "calculated_dose_reference",
        ),
        (
            "clinical-unresolved-dicom-single-plan",
            "UNRESOLVED",
            False,
            "SINGLE_PLAN_TOTAL",
            37.0,
            37.0,
            37.0,
            "fully_delivered",
            37.0,
            "fully_delivered",
            "calculated_dose_reference",
        ),
        (
            "clinical-unresolved-dicom-summed-course",
            "UNRESOLVED",
            False,
            "COURSE_TOTAL_SUMMED",
            62.0,
            50.0,
            62.0,
            "fully_delivered",
            62.0,
            "fully_delivered",
            "calculated_dose_reference",
        ),
        (
            "unresolved-replacement-chain",
            "UNRESOLVED",
            False,
            "UNRESOLVED_REPLACEMENT_CHAIN",
            None,
            50.0,
            37.0,
            "partially_delivered",
            None,
            "delivery_unresolved",
            "unresolved_course_prescription_scope",
        ),
        (
            "unresolved-component",
            "UNRESOLVED",
            False,
            "UNRESOLVED_COMPONENT",
            None,
            36.0,
            36.0,
            "fully_delivered",
            None,
            "delivery_unresolved",
            "unresolved_course_prescription_scope",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_delivery_publication_matrix_across_sources_and_scopes(
    case: str,
    outcome: str | None,
    include_phase_confirmation: bool,
    scope: str,
    course_total: float | None,
    plan_total: float | None,
    raw_delivered: float | None,
    raw_status: str,
    expected_delivered: float | None,
    expected_status: str,
    expected_method: str,
) -> None:
    del case
    evidence = None
    if outcome is not None:
        parsed = parse_kopernik_treatment_description(
            REAL_DESCRIPTIONS["419783/2020-02"]
        )
        evidence = {
            "outcome": outcome,
            "clinical_resolved_total_gy": course_total,
            "fractionation_classification": (
                confirm_two_phase_fractionation(parsed["sites"], _plan_delivery())
                if include_phase_confirmation
                else None
            ),
        }
    clinical_publication = organize._clinical_delivery_publication(
        clinical_prescription_evidence=evidence,
        delivered_dose_gy=raw_delivered,
        delivery_status=raw_status,
        delivery_method="calculated_dose_reference",
    )
    publication = organize._scope_aware_course_dose_publication(
        prescribed_dose_scope=scope,
        course_prescribed_dose_gy=course_total,
        course_resolved_prescribed_dose_total_gy=course_total,
        plan_prescribed_dose_gy=plan_total,
        plan_resolved_prescribed_dose_total_gy=plan_total,
        delivered_dose_gy=clinical_publication["delivered_dose_gy"],
        delivery_status=clinical_publication["delivery_status"],
        delivery_method=clinical_publication["delivery_method"],
    )

    assert publication["delivered_dose_gy"] == expected_delivered
    assert publication["delivery_status"] == expected_status
    assert publication["delivery_method"] == expected_method
    eligibility = organize._course_dose_response_eligible(
        prescribed_dose_scope=scope,
        resolved_prescribed_dose_total_gy=publication[
            "resolved_prescribed_dose_total_gy"
        ],
        delivered_dose_gy=publication["delivered_dose_gy"],
        delivery_status=str(publication["delivery_status"]),
    )
    assert eligibility is (
        expected_delivered is not None
        and expected_status in {"fully_delivered", "partially_delivered"}
    )
    if expected_status in {"fully_delivered", "partially_delivered"}:
        assert publication["delivered_dose_gy"] is not None
        assert publication["resolved_prescribed_dose_total_gy"] is not None
    else:
        assert publication["delivered_dose_gy"] is None


def test_loader_records_workbook_identity_dates_row_and_exact_text(
    tmp_path: Path,
) -> None:
    workbook = tmp_path / "rt_treatments.xlsx"
    frame = pd.DataFrame(
        [
            {
                "ID": 419783,
                "Data rozp lecz": "2020-02-10",
                "Data zak lecz": "2020-03-09",
                "Rozpoznanie wg ICD 10": "C67",
                "Rodzaj Leczenia": "Radioterapia",
                "Rozpoznanie": "pecherz",
                "Zalecenia": "",
                "Opis leczenia": REAL_DESCRIPTIONS["419783/2020-02"],
            }
        ]
    )
    frame.to_excel(workbook, index=False)

    index = load_kopernik_treatment_records(workbook)
    record = index.records_by_patient["419783"][0]

    assert index.row_count == 1
    assert index.source_path == workbook.resolve()
    assert len(index.workbook_sha256) == 64
    assert record.excel_row == 2
    assert record.start_date == date(2020, 2, 10)
    assert record.end_date == date(2020, 3, 9)
    assert record.description == REAL_DESCRIPTIONS["419783/2020-02"]
    assert record.audit_dict()["parsed_field"] == "Opis leczenia"
    assert clinical_evidence_matches_source(
        {"clinical_prescription_evidence": {"source": index.source_dict()}},
        index,
    )
    assert not clinical_evidence_matches_source(
        {"clinical_prescription_evidence": None}, index
    )


def test_clinical_source_config_is_optional_and_resolves_relative_path(
    tmp_path: Path,
) -> None:
    config = PipelineConfig(
        dicom_root=tmp_path / "input",
        output_root=tmp_path / "output",
        logs_root=tmp_path / "logs",
    )

    cli._apply_clinical_prescription_yaml_config(
        config,
        {
            "clinical_prescription_records": {
                "enabled": True,
                "path": "clinical/rt_treatments.xlsx",
            }
        },
        config_dir=tmp_path,
    )

    assert config.clinical_prescription_records_path == (
        tmp_path / "clinical" / "rt_treatments.xlsx"
    ).resolve()
    args = cli.build_parser().parse_args(
        [
            "--dicom-root",
            str(tmp_path / "input"),
            "--clinical-prescription-records",
            str(tmp_path / "source.xlsx"),
        ]
    )
    assert Path(args.clinical_prescription_records) == tmp_path / "source.xlsx"


def test_clinical_source_config_can_be_explicitly_disabled(tmp_path: Path) -> None:
    config = PipelineConfig(
        dicom_root=tmp_path / "input",
        output_root=tmp_path / "output",
        logs_root=tmp_path / "logs",
    )

    cli._apply_clinical_prescription_yaml_config(
        config,
        {"clinical_prescription_records": {"enabled": False}},
        config_dir=tmp_path,
    )

    assert config.clinical_prescription_records_path is None
