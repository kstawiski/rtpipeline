from __future__ import annotations

from decimal import Decimal

import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from rtpipeline.prescription import (
    aggregate_course_prescription_values,
    classify_prescription_scope,
    resolve_plan_prescriptions,
    within_five_percent,
)


def _plan(
    *,
    source_rx: object = "30",
    fractions: int = 3,
    beams: list[tuple[str | None, object, str | None]] | None = None,
    number_of_beams: int | None = None,
    duplicate_reference_number: bool = False,
    duplicate_beam_number: bool = False,
    target_uid: str | None = "1.2.3.4",
) -> Dataset:
    plan = Dataset()
    target = Dataset()
    target.DoseReferenceNumber = 1
    target.DoseReferenceType = "TARGET"
    target.TargetPrescriptionDose = source_rx
    if target_uid is not None:
        target.DoseReferenceUID = target_uid
    plan.DoseReferenceSequence = Sequence([target])

    beam_rows = beams if beams is not None else [("TREATMENT", "10", "PHYSICAL")]
    fraction_group = Dataset()
    fraction_group.FractionGroupNumber = 1
    fraction_group.NumberOfFractionsPlanned = fractions
    fraction_group.NumberOfBeams = len(beam_rows) if number_of_beams is None else number_of_beams
    references = []
    beam_sequence = []
    for index, (delivery_type, beam_dose, beam_dose_type) in enumerate(beam_rows, start=1):
        beam_number = 1 if duplicate_beam_number else index
        reference_number = 1 if duplicate_reference_number else index
        beam = Dataset()
        beam.BeamNumber = beam_number
        if delivery_type is not None:
            beam.TreatmentDeliveryType = delivery_type
        beam_sequence.append(beam)

        reference = Dataset()
        reference.ReferencedBeamNumber = reference_number
        if beam_dose is not None:
            reference.BeamDose = beam_dose
        if beam_dose_type is not None:
            reference.BeamDoseType = beam_dose_type
        if target_uid is not None:
            reference.ReferencedDoseReferenceUID = target_uid
        references.append(reference)
    fraction_group.ReferencedBeamSequence = Sequence(references)
    plan.FractionGroupSequence = Sequence([fraction_group])
    plan.BeamSequence = Sequence(beam_sequence)
    return plan


def _resolution(plan: Dataset) -> dict:
    rows = resolve_plan_prescriptions(plan)
    assert len(rows) == 1
    return rows[0]


@pytest.mark.parametrize("delivery_type", ["TREATMENT", "CONTINUATION"])
def test_treatment_and_continuation_with_usable_beamdose_are_included(delivery_type):
    row = _resolution(_plan(beams=[(delivery_type, "10", "PHYSICAL")]))

    assert row["prescription_resolution_status"] == "TOTAL_CONFIRMED"
    assert row["beam_dose_sum_per_fraction_gy"] == pytest.approx(10.0)
    assert row["prescription_resolution_details"]["included_beams"][0][
        "treatment_delivery_type"
    ] == delivery_type


@pytest.mark.parametrize("delivery_type", ["TREATMENT", "CONTINUATION"])
def test_treatment_and_continuation_without_beamdose_are_unresolved(delivery_type):
    row = _resolution(_plan(beams=[(delivery_type, None, "PHYSICAL")]))

    assert row["prescription_resolution_status"] == "UNRESOLVED"
    assert row["prescription_resolution_method"] == "UNRESOLVED_INCOMPLETE_BEAM_MEMBERSHIP"
    assert row["beam_dose_sum_per_fraction_gy"] is None
    assert row["resolved_prescribed_dose_total_gy"] is None


@pytest.mark.parametrize("setup_dose", [None, "0"])
def test_setup_without_positive_beamdose_is_excluded(setup_dose):
    row = _resolution(
        _plan(beams=[("TREATMENT", "10", "PHYSICAL"), ("SETUP", setup_dose, None)])
    )

    assert row["prescription_resolution_status"] == "TOTAL_CONFIRMED"
    assert row["beam_dose_sum_per_fraction_gy"] == pytest.approx(10.0)
    assert len(row["prescription_resolution_details"]["excluded_setup_beams"]) == 1


def test_setup_with_positive_beamdose_is_contradictory_and_unresolved():
    row = _resolution(
        _plan(beams=[("TREATMENT", "10", "PHYSICAL"), ("SETUP", "0.1", None)])
    )

    assert row["prescription_resolution_status"] == "UNRESOLVED"
    assert "SETUP beam" in row["prescription_resolution_details"]["reason"]


@pytest.mark.parametrize("delivery_type", ["OPEN_PORTFILM", "TRMT_PORTFILM"])
@pytest.mark.parametrize("imaging_dose", [None, "0", "0.25"])
def test_portfilm_is_excluded_and_retained_as_imaging_metadata(delivery_type, imaging_dose):
    row = _resolution(
        _plan(beams=[("TREATMENT", "10", "PHYSICAL"), (delivery_type, imaging_dose, None)])
    )

    assert row["prescription_resolution_status"] == "TOTAL_CONFIRMED"
    imaging = row["prescription_resolution_details"]["imaging_beams"]
    assert len(imaging) == 1
    assert imaging[0]["treatment_delivery_type"] == delivery_type


def test_missing_delivery_type_with_usable_beamdose_is_included_with_provenance():
    row = _resolution(_plan(beams=[(None, "10", "PHYSICAL")]))

    assert row["prescription_resolution_status"] == "TOTAL_CONFIRMED"
    assert row["prescription_resolution_details"]["delivery_type_unclassified_beams"] == ["1"]


def test_missing_delivery_type_and_beamdose_are_unresolved():
    row = _resolution(_plan(beams=[(None, None, None)]))

    assert row["prescription_resolution_status"] == "UNRESOLVED"
    assert "missing TreatmentDeliveryType" in row["prescription_resolution_details"]["reason"]


@pytest.mark.parametrize("beam_dose", [None, "10"])
def test_unknown_nonempty_delivery_type_is_unresolved(beam_dose):
    row = _resolution(_plan(beams=[("QA_ONLY", beam_dose, "PHYSICAL")]))

    assert row["prescription_resolution_status"] == "UNRESOLVED"
    assert "unknown TreatmentDeliveryType" in row["prescription_resolution_details"]["reason"]


@pytest.mark.parametrize("beam_dose", ["-1", "NaN", "Infinity"])
def test_treatment_beamdose_must_be_finite_and_nonnegative(beam_dose):
    row = _resolution(_plan(beams=[("TREATMENT", beam_dose, "PHYSICAL")]))

    assert row["prescription_resolution_status"] == "UNRESOLVED"
    assert row["beam_dose_sum_per_fraction_gy"] is None


@pytest.mark.parametrize(
    ("plan", "reason_fragment"),
    [
        (_plan(number_of_beams=2), "NumberOfBeams"),
        (
            _plan(
                beams=[("TREATMENT", "5", "PHYSICAL"), ("TREATMENT", "5", "PHYSICAL")],
                duplicate_reference_number=True,
            ),
            "not unique",
        ),
        (
            _plan(
                beams=[("TREATMENT", "5", "PHYSICAL"), ("TREATMENT", "5", "PHYSICAL")],
                duplicate_beam_number=True,
            ),
            "exactly one",
        ),
    ],
)
def test_beam_membership_structural_requirements_fail_closed(plan, reason_fragment):
    row = _resolution(plan)

    assert row["prescription_resolution_status"] == "UNRESOLVED"
    assert reason_fragment in row["prescription_resolution_details"]["reason"]


def test_included_beamdose_types_must_be_consistent_when_present():
    row = _resolution(
        _plan(
            beams=[
                ("TREATMENT", "5", "PHYSICAL"),
                ("TREATMENT", "5", "EFFECTIVE"),
            ]
        )
    )

    assert row["prescription_resolution_status"] == "UNRESOLVED"
    assert "inconsistent BeamDoseType" in row["prescription_resolution_details"]["reason"]


@pytest.mark.parametrize(
    ("source", "fractions", "beam_sum", "expected_status", "expected_total"),
    [
        ("30", 3, "10", "TOTAL_CONFIRMED", 30.0),
        ("10", 3, "10", "PER_FRACTION_CONFIRMED", 30.0),
        ("30", 3, "8", "UNRESOLVED_NO_MATCH", None),
    ],
)
def test_multi_fraction_classification_matrix(
    source, fractions, beam_sum, expected_status, expected_total
):
    row = classify_prescription_scope(source, fractions, beam_sum)

    assert row["prescription_resolution_status"] == expected_status
    assert row["resolved_prescribed_dose_total_gy"] == expected_total


@pytest.mark.parametrize(
    ("source", "fractions", "beam_sum", "expected_total"),
    [("2.05062909389991", 25, "2.0", 50.0), ("21.0070463739149", 10, "2.0", 20.0)],
)
def test_resolved_total_uses_beamdose_not_computed_reference_point_value(
    source, fractions, beam_sum, expected_total
):
    row = classify_prescription_scope(source, fractions, beam_sum)

    assert row["resolved_prescribed_dose_total_gy"] == pytest.approx(expected_total)
    assert row["resolved_prescribed_dose_per_fraction_gy"] == pytest.approx(
        expected_total / fractions
    )


def test_multi_fraction_both_match_matrix_branch_is_fail_closed(monkeypatch):
    # This branch is mathematically unreachable for positive Rx, integer fx > 1,
    # and a 5% relative tolerance. Keep the specified defensive branch covered.
    monkeypatch.setattr(
        "rtpipeline.prescription.within_five_percent",
        lambda candidate, source: True,
    )

    row = classify_prescription_scope("30", 3, "10")

    assert row["prescription_resolution_status"] == "UNRESOLVED_BOTH_MATCH"
    assert row["resolved_prescribed_dose_total_gy"] is None


def test_single_fraction_match_is_equivalent_but_not_assigned_temporal_scope():
    row = classify_prescription_scope("8", 1, "8")

    assert row["prescription_resolution_status"] == "INDETERMINATE_SINGLE_FRACTION"
    assert row["prescribed_dose_scope"] == "INDETERMINATE_SINGLE_FRACTION"
    assert row["resolved_prescribed_dose_per_fraction_gy"] == 8.0
    assert row["resolved_prescribed_dose_total_gy"] == 8.0


def test_single_fraction_nonmatch_is_unresolved():
    row = classify_prescription_scope("8", 1, "7")

    assert row["prescription_resolution_status"] == "UNRESOLVED_NO_MATCH"
    assert row["resolved_prescribed_dose_total_gy"] is None


@pytest.mark.parametrize(
    ("source", "fractions", "beam_sum"),
    [(None, 3, "10"), ("NaN", 3, "10"), ("0", 3, "10"), ("30", 0, "10")],
)
def test_invalid_classification_evidence_nulls_normalised_values(source, fractions, beam_sum):
    row = classify_prescription_scope(source, fractions, beam_sum)

    assert row["prescription_resolution_status"] == "UNRESOLVED"
    assert row["resolved_prescribed_dose_per_fraction_gy"] is None
    assert row["resolved_prescribed_dose_total_gy"] is None


def test_five_percent_tolerance_is_inclusive_and_uses_decimal_arithmetic():
    assert within_five_percent(Decimal("10.5"), Decimal("10")) is True
    assert within_five_percent(Decimal("9.5"), Decimal("10")) is True
    assert within_five_percent(Decimal("10.5000000000001"), Decimal("10")) is False


@pytest.mark.parametrize(
    ("case_id", "source", "fractions", "beam_sum", "expected_total"),
    [
        ("10144345427/2023-03", "2.75", 20, "2.7500", 55.0),
        ("10149603697/2022-06", "2.0", 25, "2.0000", 50.0),
    ],
)
def test_measured_dfci_per_fraction_cases_resolve_to_expected_totals(
    case_id, source, fractions, beam_sum, expected_total
):
    row = _resolution(
        _plan(
            source_rx=source,
            fractions=fractions,
            beams=[("TREATMENT", beam_sum, "PHYSICAL")],
        )
    )

    assert case_id
    assert row["source_prescribed_dose_gy"] == pytest.approx(float(source))
    assert row["prescription_resolution_status"] == "PER_FRACTION_CONFIRMED"
    assert row["resolved_prescribed_dose_total_gy"] == pytest.approx(expected_total)


def test_dfci_zphys_calculation_point_does_not_replace_site_prescription():
    """10149603697 carries a computed zPhys point beside the nominal SITE target."""
    plan = _plan(
        source_rx="2.0",
        fractions=25,
        beams=[("TREATMENT", "2.0", "PHYSICAL")],
    )
    nominal = plan.DoseReferenceSequence[0]
    nominal.DoseReferenceStructureType = "SITE"
    calculated = Dataset()
    calculated.DoseReferenceNumber = 2
    calculated.DoseReferenceUID = "1.2.3.5"
    calculated.DoseReferenceType = "TARGET"
    calculated.DoseReferenceStructureType = "COORDINATES"
    calculated.DoseReferenceDescription = "zphysC1A1pelvism"
    calculated.DoseReferencePointCoordinates = [-6.87, -10.08, 32.61]
    calculated.TargetPrescriptionDose = "2.05062909389991"
    plan.DoseReferenceSequence = Sequence([calculated, nominal])

    row = _resolution(plan)

    assert row["source_dose_reference_number"] == "1"
    assert row["source_dose_reference_structure_type"] == "SITE"
    assert row["source_prescribed_dose_gy"] == pytest.approx(2.0)
    assert row["prescription_resolution_status"] == "PER_FRACTION_CONFIRMED"
    assert row["resolved_prescribed_dose_total_gy"] == pytest.approx(50.0)


def test_dfci_zphys_only_target_disagreement_remains_unresolved():
    """10130236267's 47.42 Gy coordinate target disagrees with 45 Gy BeamDose."""
    plan = _plan(
        source_rx="47.4194410847676",
        fractions=25,
        beams=[("TREATMENT", "1.8", "PHYSICAL")],
    )
    target = plan.DoseReferenceSequence[0]
    target.DoseReferenceStructureType = "COORDINATES"
    target.DoseReferenceDescription = "zPhysC1A1"
    target.DoseReferencePointCoordinates = [-1.4, -49.12, -41.4]

    row = _resolution(plan)

    assert row["source_prescribed_dose_gy"] == pytest.approx(47.4194410847676)
    assert row["source_dose_reference_structure_type"] == "COORDINATES"
    assert row["source_dose_reference_description"] == "zPhysC1A1"
    assert row["prescription_resolution_status"] == "UNRESOLVED_NO_MATCH"
    assert row["prescribed_dose_scope"] == "UNRESOLVED"
    assert row["resolved_prescribed_dose_total_gy"] is None


def test_non_zphys_coordinate_prescription_remains_eligible():
    """Kopernik coordinate targets are not the DFCI zPhys exporter convention."""
    plan = _plan(
        source_rx="50",
        fractions=25,
        beams=[("TREATMENT", "2", "PHYSICAL")],
    )
    target = plan.DoseReferenceSequence[0]
    target.DoseReferenceStructureType = "COORDINATES"
    target.DoseReferenceDescription = "PTV 1"
    target.DoseReferencePointCoordinates = [-31.0, -213.9, 34.0]

    row = _resolution(plan)

    assert row["source_prescribed_dose_gy"] == pytest.approx(50.0)
    assert row["prescription_resolution_status"] == "TOTAL_CONFIRMED"


def test_additive_course_prescription_propagates_any_unresolved_component():
    assert aggregate_course_prescription_values(
        [50.0, 12.0, None], sum_all=True
    ) is None
    assert aggregate_course_prescription_values(
        [50.0, 12.0, 32.0], sum_all=True
    ) == pytest.approx(94.0)


@pytest.mark.parametrize(
    ("case_id", "source", "fractions", "beam_sum"),
    [
        ("10102925269/2025-09", "5.5", 20, "2.75"),
        ("10152162227/2023-02", "13.75", 20, "2.75"),
    ],
)
def test_measured_dfci_segment_values_are_not_multiplied_without_their_own_match(
    case_id, source, fractions, beam_sum
):
    row = _resolution(
        _plan(
            source_rx=source,
            fractions=fractions,
            beams=[("TREATMENT", beam_sum, "PHYSICAL")],
        )
    )

    assert case_id
    assert row["prescription_resolution_status"] == "UNRESOLVED_NO_MATCH"
    assert row["resolved_prescribed_dose_total_gy"] is None


@pytest.mark.parametrize(
    ("source", "fractions", "beam_sum"),
    [("27", 12, "3"), ("36", 3, "3")],
)
def test_measured_kopernik_rows_remain_unresolved(source, fractions, beam_sum):
    row = _resolution(
        _plan(
            source_rx=source,
            fractions=fractions,
            beams=[("TREATMENT", beam_sum, "PHYSICAL")],
        )
    )

    assert row["prescription_resolution_status"] == "UNRESOLVED_NO_MATCH"
    assert row["resolved_prescribed_dose_total_gy"] is None


def test_multiple_fraction_groups_do_not_swap_targets_by_numerical_fit():
    plan = _plan(source_rx="27", fractions=12, beams=[("TREATMENT", "3", "PHYSICAL")])
    second_group = Dataset()
    second_group.FractionGroupNumber = 2
    second_group.NumberOfFractionsPlanned = 3
    second_group.NumberOfBeams = 1
    second_reference = Dataset()
    second_reference.ReferencedBeamNumber = 1
    second_reference.BeamDose = "3"
    second_reference.BeamDoseType = "PHYSICAL"
    second_reference.ReferencedDoseReferenceUID = "1.2.3.4"
    second_group.ReferencedBeamSequence = Sequence([second_reference])
    plan.FractionGroupSequence.append(second_group)

    rows = resolve_plan_prescriptions(plan)

    assert len(rows) == 2
    assert {row["prescription_resolution_method"] for row in rows} == {"UNRESOLVED_GROUP_SCOPE"}
    assert all(row["resolved_prescribed_dose_total_gy"] is None for row in rows)


def test_existing_contract_target_identity_controls_resolution_not_numerical_fit():
    plan = _plan(
        source_rx="30",
        fractions=3,
        beams=[("TREATMENT", "10", "PHYSICAL")],
    )
    second = Dataset()
    second.DoseReferenceNumber = "2"
    second.DoseReferenceUID = "1.2.826.0.1.3680043.8.498.2"
    second.DoseReferenceType = "TARGET"
    second.TargetPrescriptionDose = "10"
    plan.DoseReferenceSequence.append(second)
    plan.FractionGroupSequence[0].ReferencedBeamSequence[
        0
    ].ReferencedDoseReferenceUID = second.DoseReferenceUID

    rows = resolve_plan_prescriptions(
        plan,
        source_prescribed_dose_gy="10",
        source_dose_reference_number="2",
        source_dose_reference_uid=second.DoseReferenceUID,
    )

    assert rows[0]["source_prescribed_dose_gy"] == 10.0
    assert rows[0]["source_dose_reference_number"] == "2"
    assert rows[0]["prescription_resolution_status"] == "PER_FRACTION_CONFIRMED"
    assert rows[0]["resolved_prescribed_dose_total_gy"] == 30.0


def test_stale_contract_target_identity_is_unresolved():
    plan = _plan(
        source_rx="30",
        fractions=3,
        beams=[("TREATMENT", "10", "PHYSICAL")],
    )

    rows = resolve_plan_prescriptions(
        plan,
        source_prescribed_dose_gy="29",
        source_dose_reference_number="1",
    )

    assert rows[0]["prescription_resolution_status"] == "UNRESOLVED"
    assert (
        rows[0]["prescription_resolution_method"]
        == "UNRESOLVED_INVALID_SOURCE_PRESCRIPTION"
    )


def test_explicit_group_level_target_mapping_scopes_source_before_classification():
    plan = _plan(source_rx="30", fractions=3, beams=[("TREATMENT", "10", "PHYSICAL")])
    group_reference = Dataset()
    group_reference.ReferencedDoseReferenceNumber = 1
    group_reference.TargetPrescriptionDose = "6"
    plan.FractionGroupSequence[0].ReferencedDoseReferenceSequence = Sequence([group_reference])
    plan.FractionGroupSequence[0].NumberOfFractionsPlanned = 2
    plan.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamDose = "3"

    row = _resolution(plan)

    assert row["source_prescribed_dose_gy"] == 6.0
    assert row["source_prescribed_dose_tag_path"].startswith("FractionGroupSequence[0]")
    assert row["prescription_resolution_status"] == "TOTAL_CONFIRMED"
    assert row["resolved_prescribed_dose_total_gy"] == 6.0


def test_duplicate_group_level_target_mapping_is_unresolved():
    plan = _plan(
        source_rx="30",
        fractions=2,
        beams=[("TREATMENT", "3", "PHYSICAL")],
    )
    first = Dataset()
    first.ReferencedDoseReferenceNumber = 1
    first.TargetPrescriptionDose = "6"
    second = Dataset()
    second.ReferencedDoseReferenceNumber = 1
    plan.FractionGroupSequence[0].ReferencedDoseReferenceSequence = Sequence(
        [first, second]
    )

    row = _resolution(plan)

    assert row["prescription_resolution_status"] == "UNRESOLVED"
    assert row["prescription_resolution_method"] == "UNRESOLVED_GROUP_SCOPE"
    assert row["resolved_prescribed_dose_total_gy"] is None


def test_target_uid_mismatch_is_retained_as_metadata_not_membership_evidence():
    plan = _plan()
    plan.FractionGroupSequence[0].ReferencedBeamSequence[0].ReferencedDoseReferenceUID = "9.9.9"

    row = _resolution(plan)

    assert row["prescription_resolution_status"] == "TOTAL_CONFIRMED"
    assert row["beam_dose_target_binding"] == "DOSE_REFERENCE_UID_METADATA_ONLY"
    assert row["prescription_resolution_details"]["observed_target_uids"] == ["9.9.9"]
