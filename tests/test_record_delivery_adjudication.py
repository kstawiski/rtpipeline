"""Synthetic regressions for RTRECORD delivery adjudication in organize.

The tests import ``rtpipeline.organize`` from the checkout that contains this
directory and stop with an error if the package resolves anywhere else. All
plans, records, UIDs, dates and values come from
``record_delivery_synthetic_rt`` and were invented for this suite.
"""
from __future__ import annotations

from pathlib import Path

import pydicom
import pytest

from rtpipeline import organize as org
from record_delivery_synthetic_rt import (
    BEAM_DOSES,
    PER_FRACTION,
    PLAN_A,
    PLAN_B,
    REF_A,
    REF_B,
    UNMATCHED_TARGET_RX,
    Event,
    complete_session,
    make_plan,
    make_record,
    make_summary_record,
)

ROOT = Path(__file__).resolve().parents[1]
TWO_FRACTIONS = 2 * PER_FRACTION
NO_DOSE = (None, None)


@pytest.fixture(scope="module")
def organize():
    """Look up definitions on the organize module imported from this checkout."""
    location = Path(org.__file__).resolve()
    if location != ROOT / "rtpipeline" / "organize.py":
        pytest.fail(f"rtpipeline.organize was imported from {location}, not from the checkout at {ROOT}")
    return lambda name: getattr(org, name)


def _summary(organize, plans, records, **kwargs):
    return organize("_calculate_delivery_summary")(plans, records, **kwargs)


def _detail(summary, plan_uid=PLAN_A):
    return next(item for item in summary["delivery_plan_details"] if item["plan_sop_uid"] == plan_uid)


def _hold_codes(summary):
    return [item["reason_code"] for item in summary.get("delivery_course_holds", [])]


# Record calculated-dose semantics.


@pytest.mark.semantics
def test_interrupted_arc_and_its_continuation_are_one_complete_delivery(tmp_path: Path, organize) -> None:
    plan = make_plan(tmp_path / "plan.dcm")
    records = complete_session(tmp_path, 1)
    records += [
        make_record(tmp_path / "f2_b1.dcm", day=2, fraction=2, events=[Event(1, dose=1.35, start="071600")]),
        make_record(tmp_path / "f2_b2_stop.dcm", day=2, fraction=2, time="071700",
                    events=[Event(2, termination="MACHINE", dose=0.35, start="071700")]),
        make_record(tmp_path / "f2_b2_cont.dcm", day=2, fraction=2, time="072400",
                    events=[Event(2, delivery_type="CONTINUATION", dose=0.5, start="072400")]),
    ]
    summary = _summary(organize, [plan], records)
    detail = _detail(summary)
    assert detail["prescription_resolution_status"] == "TOTAL_CONFIRMED"
    assert summary["delivered_fraction_count"] == 2
    assert summary["delivered_dose_gy"] == pytest.approx(TWO_FRACTIONS)
    assert summary["delivery_status"] == "partially_delivered"
    assert summary["delivery_method"] == "calculated_dose_reference"
    assert detail["delivery_completeness_status"] == "ALL_SESSIONS_COMPLETE"
    assert detail["abandoned_partial_session_count"] == 0


@pytest.mark.semantics
@pytest.mark.helper_parity
def test_reexported_copy_of_one_event_counts_once(tmp_path: Path, organize) -> None:
    plan = make_plan(tmp_path / "plan.dcm")
    records = complete_session(tmp_path, 1) + complete_session(tmp_path, 2)
    records.append(make_record(tmp_path / "copy.dcm", day=1, fraction=1, time="071600",
                               events=[Event(1, dose=1.35, start="071600")]))
    summary = _summary(organize, [plan], records)
    assert summary["delivered_fraction_count"] == 2
    assert summary["delivered_dose_gy"] == pytest.approx(TWO_FRACTIONS)


@pytest.mark.semantics
@pytest.mark.helper_parity
def test_setup_item_value_never_enters_the_session_dose(tmp_path: Path, organize) -> None:
    plan = make_plan(tmp_path / "plan.dcm")
    records = [
        make_record(tmp_path / "f1_setup_b1.dcm", day=1, fraction=1, events=[
            Event(8, delivery_type="SETUP", dose=0.07, start="070500", meterset=2.0),
            Event(1, dose=1.35, start="071600"),
        ]),
        make_record(tmp_path / "f1_b2.dcm", day=1, fraction=1, time="071700", events=[Event(2, dose=0.85, start="071700")]),
    ]
    records += complete_session(tmp_path, 2)
    summary = _summary(organize, [plan], records)
    assert summary["delivered_dose_gy"] == pytest.approx(TWO_FRACTIONS)
    assert summary["delivery_method"] == "calculated_dose_reference"


@pytest.mark.semantics
def test_values_bound_to_another_reference_are_absent_evidence(tmp_path: Path, organize) -> None:
    plan = make_plan(tmp_path / "plan.dcm")
    records = complete_session(tmp_path, 1, reference_number="9") + complete_session(tmp_path, 2, reference_number="9")
    summary = _summary(organize, [plan], records)
    detail = _detail(summary)
    assert detail["record_dose_adjudication_status"] == "ABSENT"
    assert summary["delivered_dose_gy"] == pytest.approx(TWO_FRACTIONS)
    assert summary["delivery_method"] == "record_fraction_weighted_prescription"


@pytest.mark.semantics
def test_unresolved_scope_target_bound_record_dose_stays_plan_evidence(tmp_path: Path, organize) -> None:
    plan = make_plan(tmp_path / "plan.dcm", target_rx=UNMATCHED_TARGET_RX)
    records = complete_session(tmp_path, 1) + complete_session(tmp_path, 2)
    summary = _summary(organize, [plan], records)
    detail = _detail(summary)
    assert detail["prescription_resolution_status"] == "UNRESOLVED_NO_MATCH"
    assert detail["delivered_dose_gy"] == pytest.approx(TWO_FRACTIONS)
    assert detail["record_reference_dose_status"] == "target_bound_record_dose_without_prescription_cross_check"
    assert summary["delivered_dose_gy"] is None
    assert summary["delivery_status"] == "delivery_unresolved"
    assert summary["delivery_method"] == "course_dose_adjudication_hold"
    assert _hold_codes(summary) == ["COURSE_DOSE_PRESCRIPTION_CROSS_CHECK_UNAVAILABLE"]


@pytest.mark.semantics
def test_unresolved_scope_target_without_uid_bound_beamdose_is_not_used(tmp_path: Path, organize) -> None:
    plan = make_plan(tmp_path / "plan.dcm", target_rx=UNMATCHED_TARGET_RX, beam_reference_uid=REF_B)
    records = complete_session(tmp_path, 1) + complete_session(tmp_path, 2)
    summary = _summary(organize, [plan], records)
    detail = _detail(summary)
    assert detail["beam_dose_target_binding"] == "DOSE_REFERENCE_UID_METADATA_ONLY"
    assert detail["delivered_dose_gy"] is None
    assert detail["record_reference_dose_gy"] is None
    assert summary["delivered_dose_gy"] is None
    assert any("not bound by UID" in warning for warning in detail["warning_messages"])


@pytest.mark.semantics
def test_single_non_target_reference_dose_is_an_observation_only(tmp_path: Path, organize) -> None:
    plan = make_plan(tmp_path / "plan.dcm", reference_number="2", reference_type="ORGAN_AT_RISK", target_rx=None)
    records = complete_session(tmp_path, 1, reference_number="2") + complete_session(tmp_path, 2, reference_number="2")
    summary = _summary(organize, [plan], records)
    detail = _detail(summary)
    assert detail["record_reference_binding"] == "SINGLE_NON_TARGET_REFERENCE_BEAMDOSE_BOUND"
    assert detail["record_reference_types"] == ["ORGAN_AT_RISK"]
    assert detail["record_reference_dose_gy"] == pytest.approx(TWO_FRACTIONS)
    assert detail["record_reference_dose_status"] == "non_target_reference_dose_held"
    assert detail["delivered_dose_gy"] is None
    assert summary["delivered_dose_gy"] is None
    assert summary["delivered_dose_provenance"] == "no_record_linked_dose_value"


@pytest.mark.semantics
def test_cumulative_summary_record_path_is_unchanged(tmp_path: Path, organize) -> None:
    plan = make_plan(tmp_path / "plan.dcm")
    records = [make_summary_record(tmp_path / "summary.dcm", day=4, cumulative=6.6)]
    summary = _summary(organize, [plan], records)
    assert summary["delivered_dose_gy"] == pytest.approx(6.6)
    assert summary["delivery_method"] == "cumulative_dose_reference"


@pytest.mark.semantics
def test_per_record_dose_needs_one_reference_and_therapeutic_items(tmp_path: Path, organize) -> None:
    read = lambda path: pydicom.dcmread(str(path), force=True)  # noqa: E731
    record_dose = organize("_record_dose_reference")
    single = make_record(tmp_path / "single.dcm", events=[Event(1, dose=1.35, start="071600"), Event(2, dose=0.85, start="071700")])
    mixed = make_record(tmp_path / "mixed.dcm", events=[Event(1, dose=1.35, start="071600"),
                                                        Event(2, dose=0.4, reference_number="9", start="071700")])
    setup_only = make_record(tmp_path / "setup.dcm", events=[Event(8, delivery_type="SETUP", dose=0.07, meterset=2.0)])
    value, method = record_dose(read(single))
    assert value == pytest.approx(PER_FRACTION) and method == "calculated_dose_reference"
    assert record_dose(read(mixed)) == (None, None)
    assert record_dose(read(setup_only)) == (None, None)


# Concern (b): abandoned partial delivery.


@pytest.mark.concern_b
def test_abandoned_partial_session_blocks_the_delivered_scalar(tmp_path: Path, organize) -> None:
    plan = make_plan(tmp_path / "plan.dcm")
    records = complete_session(tmp_path, 1) + complete_session(tmp_path, 2)
    records.append(make_record(tmp_path / "f3_stop.dcm", day=3, fraction=3,
                               events=[Event(1, termination="MACHINE", dose=0.45, start="071600")]))
    summary = _summary(organize, [plan], records)
    detail = _detail(summary)
    assert summary["delivered_fraction_count"] == 2
    assert detail["abandoned_partial_session_count"] == 1
    assert detail["abandoned_partial_session_dose_gy"] == pytest.approx(0.45)
    assert detail["delivered_dose_gy"] is None
    assert summary["delivered_dose_gy"] is None
    assert summary["delivery_method"] == "delivery_completeness_unresolved"


@pytest.mark.concern_b
@pytest.mark.concern_d
def test_abandoned_partial_without_dose_values_blocks_fraction_weighting(tmp_path: Path, organize) -> None:
    plan = make_plan(tmp_path / "plan.dcm")
    records = complete_session(tmp_path, 1, doses=NO_DOSE) + complete_session(tmp_path, 2, doses=NO_DOSE)
    records.append(make_record(tmp_path / "f3_stop.dcm", day=3, fraction=3,
                               events=[Event(1, termination="OPERATOR", start="071600", meterset=25.0)]))
    summary = _summary(organize, [plan], records)
    detail = _detail(summary)
    assert detail["fraction_weighted_dose_gy"] == pytest.approx(TWO_FRACTIONS)
    assert detail["abandoned_partial_unquantified_session_count"] == 1
    assert detail["delivered_dose_gy"] is None
    assert summary["delivered_dose_gy"] is None


@pytest.mark.concern_b
def test_unvalidated_record_with_zero_meterset_does_not_block(tmp_path: Path, organize) -> None:
    plan = make_plan(tmp_path / "plan.dcm")
    records = complete_session(tmp_path, 1) + complete_session(tmp_path, 2)
    records.append(make_record(tmp_path / "f3_zero.dcm", day=3, fraction=3,
                               events=[Event(1, termination="MACHINE", dose=0.0, start="071600", meterset=0.0)]))
    summary = _summary(organize, [plan], records)
    detail = _detail(summary)
    assert detail["zero_delivery_unvalidated_session_count"] == 1
    assert detail["abandoned_partial_session_count"] == 0
    assert summary["delivered_dose_gy"] == pytest.approx(TWO_FRACTIONS)


# Concern (c): one NORMAL event does not prove every beam completed.


@pytest.mark.concern_c
def test_normal_first_arc_does_not_complete_a_session_whose_second_arc_stopped(tmp_path: Path, organize) -> None:
    plan = make_plan(tmp_path / "plan.dcm")
    records = complete_session(tmp_path, 1)
    records += [
        make_record(tmp_path / "f2_b1.dcm", day=2, fraction=2, events=[Event(1, dose=1.35, start="071600")]),
        make_record(tmp_path / "f2_b2.dcm", day=2, fraction=2, time="071700",
                    events=[Event(2, termination="MACHINE", dose=0.4, start="071700")]),
    ]
    summary = _summary(organize, [plan], records)
    detail = _detail(summary)
    assert summary["delivered_fraction_count"] == 2
    assert detail["delivery_completeness_status"] == "INCOMPLETE"
    assert any("last delivery event for beam 2 ended MACHINE" in item for item in detail["delivery_completeness_reasons"])
    assert detail["delivered_dose_gy"] is None
    assert summary["delivered_dose_gy"] is None
    assert detail["method"] == "delivery_completeness_unresolved"


@pytest.mark.concern_c
@pytest.mark.concern_d
def test_stopped_arc_without_dose_values_blocks_fraction_weighting(tmp_path: Path, organize) -> None:
    plan = make_plan(tmp_path / "plan.dcm")
    records = complete_session(tmp_path, 1, doses=NO_DOSE)
    records.append(make_record(tmp_path / "f2.dcm", day=2, fraction=2, events=[
        Event(1, start="071600", meterset=135.0),
        Event(2, termination="OPERATOR", start="071700", meterset=40.0),
    ]))
    summary = _summary(organize, [plan], records)
    detail = _detail(summary)
    assert detail["fraction_weighted_dose_gy"] == pytest.approx(TWO_FRACTIONS)
    assert detail["delivered_dose_gy"] is None
    assert summary["delivered_dose_gy"] is None


@pytest.mark.concern_c
def test_session_missing_an_arc_is_incomplete(tmp_path: Path, organize) -> None:
    plan = make_plan(tmp_path / "plan.dcm")
    records = complete_session(tmp_path, 1, doses=NO_DOSE)
    records.append(make_record(tmp_path / "f2_b1.dcm", day=2, fraction=2, events=[Event(1, start="071600", meterset=135.0)]))
    summary = _summary(organize, [plan], records)
    detail = _detail(summary)
    assert any("no delivery event for beam 2" in item for item in detail["delivery_completeness_reasons"])
    assert summary["delivered_dose_gy"] is None


@pytest.mark.concern_c
def test_continuation_on_another_day_is_not_a_complete_session(tmp_path: Path, organize) -> None:
    plan = make_plan(tmp_path / "plan.dcm")
    records = complete_session(tmp_path, 1, doses=NO_DOSE)
    records.append(make_record(tmp_path / "f2_d2.dcm", day=2, fraction=2, events=[
        Event(1, start="071600", meterset=135.0),
        Event(2, termination="MACHINE", start="071700", meterset=30.0),
    ]))
    records.append(make_record(tmp_path / "f2_d3.dcm", day=3, fraction=2, events=[
        Event(2, delivery_type="CONTINUATION", start="090500", meterset=55.0),
    ]))
    summary = _summary(organize, [plan], records)
    detail = _detail(summary)
    assert summary["delivered_fraction_count"] == 3
    reasons = detail["delivery_completeness_reasons"]
    assert any("starts with a CONTINUATION" in item for item in reasons)
    assert any("ended MACHINE" in item for item in reasons)
    assert summary["delivered_dose_gy"] is None


@pytest.mark.concern_c
def test_restart_with_new_treatment_event_is_incomplete(tmp_path: Path, organize) -> None:
    plan = make_plan(tmp_path / "plan.dcm")
    records = complete_session(tmp_path, 1, doses=NO_DOSE)
    records.append(make_record(tmp_path / "f2.dcm", day=2, fraction=2, events=[
        Event(1, start="071600", meterset=135.0),
        Event(2, termination="MACHINE", start="071700", meterset=30.0),
        Event(2, start="072900", meterset=85.0),
    ]))
    summary = _summary(organize, [plan], records)
    assert any("restarts with a new TREATMENT" in item for item in _detail(summary)["delivery_completeness_reasons"])
    assert summary["delivered_dose_gy"] is None


@pytest.mark.concern_c
def test_unordered_events_for_one_arc_are_incomplete(tmp_path: Path, organize) -> None:
    plan = make_plan(tmp_path / "plan.dcm")
    records = complete_session(tmp_path, 1, doses=NO_DOSE)
    records.append(make_record(tmp_path / "f2.dcm", day=2, fraction=2, events=[
        Event(1, start="071600", meterset=135.0),
        Event(2, termination="MACHINE", start="071700", meterset=30.0),
        Event(2, delivery_type="CONTINUATION", start="071700", meterset=55.0),
    ]))
    summary = _summary(organize, [plan], records)
    assert any("cannot be ordered" in item for item in _detail(summary)["delivery_completeness_reasons"])
    assert summary["delivered_dose_gy"] is None


@pytest.mark.concern_c
def test_therapeutic_event_outside_the_plan_fraction_group_is_incomplete(tmp_path: Path, organize) -> None:
    plan = make_plan(tmp_path / "plan.dcm")
    records = complete_session(tmp_path, 1, doses=NO_DOSE) + complete_session(tmp_path, 2, doses=NO_DOSE)
    records.append(make_record(tmp_path / "f2_b5.dcm", day=2, fraction=2, events=[Event(5, start="073000", meterset=20.0)]))
    summary = _summary(organize, [plan], records)
    assert any("not in the plan FractionGroup" in item for item in _detail(summary)["delivery_completeness_reasons"])
    assert summary["delivered_dose_gy"] is None


@pytest.mark.concern_c
def test_plan_expected_items_exclude_setup_and_need_one_fraction_group(tmp_path: Path, organize) -> None:
    expected = organize("_plan_expected_delivery_items")
    one_group = pydicom.dcmread(str(make_plan(tmp_path / "one.dcm")), force=True)
    two_groups = pydicom.dcmread(str(make_plan(tmp_path / "two.dcm", extra_fraction_group=True)), force=True)
    assert expected(one_group) == {"status": "DEFINED_SINGLE_FRACTION_GROUP", "items": [("beam", "1"), ("beam", "2")]}
    assert expected(two_groups)["status"] == "UNADJUDICATED_FRACTION_GROUP_COUNT"


@pytest.mark.concern_c
def test_application_setup_events_are_adjudicated_by_setup_number(organize) -> None:
    adjudicate = organize("_adjudicate_plan_record_delivery")
    session = ("fraction", "20910305", "1")

    def event(identity, delivery_type, termination, start):
        return {
            "event_key": ("application_setup", "TreatmentSessionApplicationSetupSequence", identity, delivery_type, termination, start),
            "kind": "application_setup", "identity": identity, "delivery_type": delivery_type,
            "termination_status": termination, "event_time": start, "delivered_meterset": 44.0, "dose_references": [],
        }

    complete = [{"session_key": session, "delivery_events": [event("1", "TREATMENT", "NORMAL", "T1")], "session_components": []}]
    stopped = [{"session_key": session, "delivery_events": [event("1", "TREATMENT", "PATIENT", "T1")], "session_components": []}]
    kwargs = dict(validated_sessions={session}, expected_items=[("application_setup", "1")],
                  expected_items_status="DEFINED_SINGLE_FRACTION_GROUP", reference_numbers=set(), per_fraction_reference_gy=None)
    assert adjudicate(complete, **kwargs)["completeness_status"] == "ALL_SESSIONS_COMPLETE"
    assert adjudicate(stopped, **kwargs)["completeness_status"] == "INCOMPLETE"


# Concern (d): contradictions never fall back to a fraction-weighted dose.


@pytest.mark.concern_d
@pytest.mark.helper_parity
@pytest.mark.parametrize("copy_first", [False, True])
def test_copies_of_one_event_with_near_equal_values_hold_without_fallback(tmp_path: Path, organize, copy_first: bool) -> None:
    plan = make_plan(tmp_path / "plan.dcm")
    copy = make_record(tmp_path / "copy.dcm", day=1, fraction=1, time="071600", events=[Event(1, dose=1.31, start="071600")])
    records = complete_session(tmp_path, 1) + complete_session(tmp_path, 2)
    records = [copy, *records] if copy_first else [*records, copy]
    summary = _summary(organize, [plan], records)
    detail = _detail(summary)
    assert detail["fraction_weighted_dose_gy"] == pytest.approx(TWO_FRACTIONS)
    assert detail["record_dose_adjudication_status"] == "CONTRADICTED"
    assert any("disagree in bound calculated dose" in warning for warning in detail["warning_messages"])
    assert detail["method"] == "record_dose_contradiction_hold"
    assert detail["delivered_dose_gy"] is None
    assert summary["delivered_dose_gy"] is None
    assert summary["delivery_status"] == "delivery_unresolved"


@pytest.mark.concern_d
def test_event_without_bound_value_beside_valued_events_holds(tmp_path: Path, organize) -> None:
    plan = make_plan(tmp_path / "plan.dcm")
    records = complete_session(tmp_path, 1) + complete_session(tmp_path, 2, doses=(1.35, None))
    summary = _summary(organize, [plan], records)
    detail = _detail(summary)
    assert any("carry no bound calculated dose while others do" in warning for warning in detail["warning_messages"])
    assert detail["delivered_dose_gy"] is None
    assert summary["delivered_dose_gy"] is None


@pytest.mark.concern_d
def test_session_dose_outside_plan_tolerance_holds(tmp_path: Path, organize) -> None:
    plan = make_plan(tmp_path / "plan.dcm")
    scaled = (1.62, 1.02)
    records = complete_session(tmp_path, 1, doses=scaled) + complete_session(tmp_path, 2, doses=scaled)
    summary = _summary(organize, [plan], records)
    detail = _detail(summary)
    assert any("disagrees with the plan per-fraction reference dose" in warning for warning in detail["warning_messages"])
    assert detail["delivered_dose_gy"] is None
    assert summary["delivered_dose_gy"] is None


@pytest.mark.concern_d
@pytest.mark.parametrize("bad_value", ["nan", "inf", "-0.5"])
def test_invalid_bound_value_holds(tmp_path: Path, organize, bad_value: str) -> None:
    plan = make_plan(tmp_path / "plan.dcm")
    records = complete_session(tmp_path, 1) + complete_session(tmp_path, 2, doses=(BEAM_DOSES[0], bad_value))
    summary = _summary(organize, [plan], records)
    detail = _detail(summary)
    assert any("not finite and non-negative" in warning for warning in detail["warning_messages"])
    assert detail["delivered_dose_gy"] is None
    assert summary["delivered_dose_gy"] is None


@pytest.mark.concern_d
def test_consistent_records_without_dose_values_still_use_fraction_weighting(tmp_path: Path, organize) -> None:
    plan = make_plan(tmp_path / "plan.dcm")
    records = complete_session(tmp_path, 1, doses=NO_DOSE) + complete_session(tmp_path, 2, doses=NO_DOSE)
    summary = _summary(organize, [plan], records)
    assert summary["delivered_dose_gy"] == pytest.approx(TWO_FRACTIONS)
    assert summary["delivery_method"] == "record_fraction_weighted_prescription"


# Concern (a): DoseReferenceUID identity across summed plans.


def _two_plans(tmp_path: Path, *, second_reference: str, second_days: tuple[int, int], doses) -> tuple[list[Path], list[Path]]:
    first = make_plan(tmp_path / "plan_a.dcm")
    second = make_plan(tmp_path / "plan_b.dcm", plan_uid=PLAN_B, reference_uid=second_reference,
                       beam_reference_uid=second_reference)
    records = complete_session(tmp_path, 1, doses=doses) + complete_session(tmp_path, 2, doses=doses)
    records += complete_session(tmp_path, 1, plan_uid=PLAN_B, day=second_days[0], doses=doses)
    records += complete_session(tmp_path, 2, plan_uid=PLAN_B, day=second_days[1], doses=doses)
    return [first, second], records


@pytest.mark.concern_a
def test_record_doses_at_distinct_reference_uids_are_not_summed(tmp_path: Path, organize) -> None:
    plans, records = _two_plans(tmp_path, second_reference=REF_B, second_days=(10, 11), doses=BEAM_DOSES)
    summary = _summary(organize, plans, records)
    assert _detail(summary, PLAN_A)["delivered_dose_gy"] == pytest.approx(TWO_FRACTIONS)
    assert _detail(summary, PLAN_B)["delivered_dose_gy"] == pytest.approx(TWO_FRACTIONS)
    assert summary["course_dose_reference_identity"] == "DISTINCT_OR_UNBOUND_DOSE_REFERENCE_UIDS"
    assert _hold_codes(summary) == ["COURSE_DOSE_REFERENCE_UID_MISMATCH"]
    assert summary["delivered_dose_gy"] is None
    assert summary["delivery_status"] == "delivery_unresolved"


@pytest.mark.concern_a
def test_record_doses_at_one_reference_uid_are_summed(tmp_path: Path, organize) -> None:
    plans, records = _two_plans(tmp_path, second_reference=REF_A, second_days=(10, 11), doses=BEAM_DOSES)
    summary = _summary(organize, plans, records)
    assert summary["course_dose_reference_identity"] == "IDENTICAL_DOSE_REFERENCE_UIDS"
    assert _hold_codes(summary) == []
    assert summary["delivered_dose_gy"] == pytest.approx(2 * TWO_FRACTIONS)


@pytest.mark.concern_a
@pytest.mark.concern_f
def test_fraction_weighted_plans_at_distinct_references_on_shared_dates_are_not_summed(tmp_path: Path, organize) -> None:
    plans, records = _two_plans(tmp_path, second_reference=REF_B, second_days=(1, 2), doses=NO_DOSE)
    summary = _summary(organize, plans, records)
    assert _detail(summary, PLAN_B)["delivered_dose_gy"] == pytest.approx(TWO_FRACTIONS)
    assert _hold_codes(summary) == ["COURSE_DOSE_CONCURRENT_DISTINCT_REFERENCES"]
    assert summary["delivered_dose_gy"] is None


@pytest.mark.concern_a
def test_fraction_weighted_plans_at_distinct_references_on_separate_dates_publish_a_labelled_total(tmp_path: Path, organize) -> None:
    plans, records = _two_plans(tmp_path, second_reference=REF_B, second_days=(10, 11), doses=NO_DOSE)
    summary = _summary(organize, plans, records)
    assert _hold_codes(summary) == []
    assert summary["delivered_dose_gy"] == pytest.approx(2 * TWO_FRACTIONS)
    assert any("nominal sequential total" in warning for warning in summary["delivery_warnings"])
