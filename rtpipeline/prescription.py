from __future__ import annotations

"""Target-bound RTPLAN prescription scope resolution from BeamDose evidence."""

from decimal import Decimal, InvalidOperation
from typing import Any

from pydicom.dataset import Dataset


PRESCRIPTION_TOLERANCE = Decimal("0.05")
_TARGET_REFERENCE_TYPES = frozenset({"TARGET", "TREATED_VOLUME", "PLANNED_TARGET_VOLUME"})
_INCLUDED_DELIVERY_TYPES = frozenset({"TREATMENT", "CONTINUATION"})
_IMAGING_DELIVERY_TYPES = frozenset({"OPEN_PORTFILM", "TRMT_PORTFILM"})

PRESCRIPTION_GROUP_FIELDS = (
    "source_prescribed_dose_gy",
    "source_prescribed_dose_tag_path",
    "source_dose_reference_number",
    "source_dose_reference_uid",
    "fraction_group_number",
    "planned_fractions",
    "beam_dose_sum_per_fraction_gy",
    "prescribed_dose_scope",
    "resolved_prescribed_dose_per_fraction_gy",
    "resolved_prescribed_dose_total_gy",
    "prescription_resolution_method",
    "prescription_resolution_status",
    "beam_dose_target_binding",
    "prescription_resolution_details",
)


def _text(value: object) -> str:
    return str(value or "").strip()


def _decimal(value: object) -> Decimal | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _positive_decimal(value: object) -> Decimal | None:
    parsed = _decimal(value)
    return parsed if parsed is not None and parsed > 0 else None


def _nonnegative_decimal(value: object) -> Decimal | None:
    parsed = _decimal(value)
    return parsed if parsed is not None and parsed >= 0 else None


def _integer_at_least_one(value: object) -> int | None:
    parsed = _decimal(value)
    if parsed is None or parsed < 1 or parsed != parsed.to_integral_value():
        return None
    return int(parsed)


def _json_number(value: Decimal | None) -> float | None:
    return float(value) if value is not None else None


def _target_references(ds_plan: Dataset) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for index, item in enumerate(getattr(ds_plan, "DoseReferenceSequence", None) or []):
        reference_type = _text(getattr(item, "DoseReferenceType", None)).upper()
        if reference_type and reference_type not in _TARGET_REFERENCE_TYPES:
            continue
        source = _positive_decimal(getattr(item, "TargetPrescriptionDose", None))
        if source is None:
            continue
        references.append(
            {
                "item": item,
                "index": index,
                "source": source,
                "number": _text(getattr(item, "DoseReferenceNumber", None)) or None,
                "uid": _text(getattr(item, "DoseReferenceUID", None)) or None,
                "tag_path": f"DoseReferenceSequence[{index}].TargetPrescriptionDose",
            }
        )
    return references


def contracted_source_prescription(
    ds_plan: Dataset,
    *,
    source_prescribed_dose_gy: object = None,
    source_dose_reference_number: object = None,
    source_dose_reference_uid: object = None,
) -> dict[str, Any] | None:
    """Bind an established source target without using BeamDose evidence.

    Existing contracts should supply the source value and any available target
    identity. New organizer decisions retain the pre-existing source selection
    when no earlier contract identity exists.
    """

    references = _target_references(ds_plan)
    explicit_source_requested = source_prescribed_dose_gy not in (None, "")
    explicit_source = _positive_decimal(source_prescribed_dose_gy)
    explicit_number = _text(source_dose_reference_number) or None
    explicit_uid = _text(source_dose_reference_uid) or None
    if explicit_source_requested and explicit_source is None:
        return None
    if explicit_source_requested or explicit_number is not None or explicit_uid is not None:
        matches = [
            reference
            for reference in references
            if (explicit_source is None or reference["source"] == explicit_source)
            and (explicit_number is None or reference["number"] == explicit_number)
            and (explicit_uid is None or reference["uid"] == explicit_uid)
        ]
        if len(matches) == 1:
            return matches[0]
        if explicit_number is not None or explicit_uid is not None:
            return None
        assert explicit_source is not None
        return {
            "item": None,
            "index": None,
            "source": explicit_source,
            "number": None,
            "uid": None,
            "tag_path": "course_contract.selected_plans[].prescribed_dose_gy",
        }
    return references[0] if references else None


def within_five_percent(candidate: Decimal, source: Decimal) -> bool:
    if not candidate.is_finite() or not source.is_finite() or source <= 0:
        return False
    return abs(candidate - source) <= PRESCRIPTION_TOLERANCE * abs(source)


def classify_prescription_scope(
    source_prescribed_dose: object,
    planned_fractions: object,
    beam_dose_sum: object,
    *,
    evidence_complete: bool = True,
    incomplete_method: str = "UNRESOLVED_INCOMPLETE_BEAM_MEMBERSHIP",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify source prescription scope using exact decimal arithmetic."""

    source = _positive_decimal(source_prescribed_dose)
    fractions = _integer_at_least_one(planned_fractions)
    beam_sum = _nonnegative_decimal(beam_dose_sum)
    result_details = dict(details or {})
    result_details["relative_tolerance"] = str(PRESCRIPTION_TOLERANCE)

    base = {
        "prescribed_dose_scope": "UNRESOLVED",
        "resolved_prescribed_dose_per_fraction_gy": None,
        "resolved_prescribed_dose_total_gy": None,
        "prescription_resolution_method": incomplete_method,
        "prescription_resolution_status": "UNRESOLVED",
        "prescription_resolution_details": result_details,
    }
    if source is None:
        result_details["reason"] = "source TargetPrescriptionDose is absent, non-finite, or non-positive"
        base["prescription_resolution_method"] = "UNRESOLVED_INVALID_SOURCE_PRESCRIPTION"
        return base
    if fractions is None:
        result_details["reason"] = "NumberOfFractionsPlanned is absent, non-integer, or less than one"
        base["prescription_resolution_method"] = "UNRESOLVED_INVALID_FRACTION_COUNT"
        return base
    if not evidence_complete or beam_sum is None:
        result_details.setdefault("reason", "complete therapeutic BeamDose membership is unavailable")
        return base

    match_per_fraction = within_five_percent(beam_sum, source)
    match_total = within_five_percent(beam_sum * Decimal(fractions), source)
    result_details["match_per_fraction"] = match_per_fraction
    result_details["match_total"] = match_total

    if fractions == 1:
        if not match_per_fraction:
            result_details["reason"] = "single-fraction BeamDose sum does not match source prescription"
            base["prescription_resolution_method"] = "UNRESOLVED_NO_MATCH"
            base["prescription_resolution_status"] = "UNRESOLVED_NO_MATCH"
            return base
        return {
            **base,
            "prescribed_dose_scope": "INDETERMINATE_SINGLE_FRACTION",
            "resolved_prescribed_dose_per_fraction_gy": _json_number(source),
            "resolved_prescribed_dose_total_gy": _json_number(source),
            "prescription_resolution_method": "BEAMDOSE_SINGLE_FRACTION_EQUIVALENT_V1",
            "prescription_resolution_status": "INDETERMINATE_SINGLE_FRACTION",
        }

    if match_total and not match_per_fraction:
        return {
            **base,
            "prescribed_dose_scope": "TOTAL",
            "resolved_prescribed_dose_per_fraction_gy": _json_number(source / Decimal(fractions)),
            "resolved_prescribed_dose_total_gy": _json_number(source),
            "prescription_resolution_method": "BEAMDOSE_TOTAL_5PCT_V1",
            "prescription_resolution_status": "TOTAL_CONFIRMED",
        }
    if match_per_fraction and not match_total:
        return {
            **base,
            "prescribed_dose_scope": "PER_FRACTION",
            "resolved_prescribed_dose_per_fraction_gy": _json_number(source),
            "resolved_prescribed_dose_total_gy": _json_number(source * Decimal(fractions)),
            "prescription_resolution_method": "BEAMDOSE_PER_FRACTION_5PCT_V1",
            "prescription_resolution_status": "PER_FRACTION_CONFIRMED",
        }
    if match_total and match_per_fraction:
        result_details["reason"] = "both total and per-fraction interpretations match"
        base["prescription_resolution_method"] = "UNRESOLVED_BOTH_MATCH"
        base["prescription_resolution_status"] = "UNRESOLVED_BOTH_MATCH"
        return base

    result_details["reason"] = "neither total nor per-fraction interpretation matches"
    base["prescription_resolution_method"] = "UNRESOLVED_NO_MATCH"
    base["prescription_resolution_status"] = "UNRESOLVED_NO_MATCH"
    return base


def _beam_number(value: object) -> str:
    text = _text(value)
    if not text:
        return ""
    try:
        parsed = Decimal(text)
    except InvalidOperation:
        return text
    if parsed.is_finite() and parsed == parsed.to_integral_value():
        return str(int(parsed))
    return text


def _beam_membership(
    ds_plan: Dataset,
    fraction_group: Dataset,
    *,
    target_uid: str | None,
) -> dict[str, Any]:
    referenced = list(getattr(fraction_group, "ReferencedBeamSequence", None) or [])
    declared_count = _integer_at_least_one(getattr(fraction_group, "NumberOfBeams", None))
    details: dict[str, Any] = {
        "included_beams": [],
        "excluded_setup_beams": [],
        "imaging_beams": [],
        "delivery_type_unclassified_beams": [],
    }
    if declared_count is None or declared_count != len(referenced):
        details["reason"] = (
            "NumberOfBeams does not equal ReferencedBeamSequence length"
        )
        return {"complete": False, "sum": None, "binding": "UNRESOLVED", "details": details}

    referenced_numbers = [_beam_number(getattr(item, "ReferencedBeamNumber", None)) for item in referenced]
    if any(not number for number in referenced_numbers) or len(set(referenced_numbers)) != len(referenced_numbers):
        details["reason"] = "ReferencedBeamNumber values are missing or not unique"
        return {"complete": False, "sum": None, "binding": "UNRESOLVED", "details": details}

    beams_by_number: dict[str, list[Dataset]] = {}
    for beam in getattr(ds_plan, "BeamSequence", None) or []:
        beams_by_number.setdefault(_beam_number(getattr(beam, "BeamNumber", None)), []).append(beam)
    unresolved_numbers = [number for number in referenced_numbers if len(beams_by_number.get(number, [])) != 1]
    if unresolved_numbers:
        details["reason"] = "each ReferencedBeamNumber must resolve to exactly one BeamSequence.BeamNumber"
        details["unresolved_beam_numbers"] = unresolved_numbers
        return {"complete": False, "sum": None, "binding": "UNRESOLVED", "details": details}

    included_doses: list[Decimal] = []
    included_types: set[str] = set()
    binding_uids: list[str | None] = []
    for reference, number in zip(referenced, referenced_numbers):
        beam = beams_by_number[number][0]
        delivery_type = _text(getattr(beam, "TreatmentDeliveryType", None)).upper()
        raw_dose = getattr(reference, "BeamDose", None)
        dose = _nonnegative_decimal(raw_dose)
        dose_present = raw_dose not in (None, "")
        beam_dose_type = _text(getattr(reference, "BeamDoseType", None)).upper()
        reference_uid = _text(getattr(reference, "ReferencedDoseReferenceUID", None)) or None
        row = {
            "referenced_beam_number": number,
            "treatment_delivery_type": delivery_type or None,
            "beam_dose_gy": _json_number(dose),
            "beam_dose_type": beam_dose_type or None,
            "referenced_dose_reference_uid": reference_uid,
        }

        if delivery_type in _IMAGING_DELIVERY_TYPES:
            row["raw_beam_dose"] = None if raw_dose in (None, "") else str(raw_dose)
            details["imaging_beams"].append(row)
            continue
        if delivery_type == "SETUP":
            if not dose_present or dose == 0:
                details["excluded_setup_beams"].append(row)
                continue
            details["reason"] = "SETUP beam has a positive or invalid BeamDose"
            details["contradictory_beam"] = row
            return {"complete": False, "sum": None, "binding": "UNRESOLVED", "details": details}
        if delivery_type in _INCLUDED_DELIVERY_TYPES:
            if dose is None:
                details["reason"] = "TREATMENT or CONTINUATION beam lacks a finite non-negative BeamDose"
                details["incomplete_beam"] = row
                return {"complete": False, "sum": None, "binding": "UNRESOLVED", "details": details}
        elif not delivery_type:
            if dose is None:
                details["reason"] = "beam with missing TreatmentDeliveryType also lacks BeamDose"
                details["incomplete_beam"] = row
                return {"complete": False, "sum": None, "binding": "UNRESOLVED", "details": details}
            details["delivery_type_unclassified_beams"].append(number)
        else:
            details["reason"] = f"unknown TreatmentDeliveryType {delivery_type!r}"
            details["unknown_beam"] = row
            return {"complete": False, "sum": None, "binding": "UNRESOLVED", "details": details}

        assert dose is not None
        included_doses.append(dose)
        details["included_beams"].append(row)
        binding_uids.append(reference_uid)
        if beam_dose_type:
            included_types.add(beam_dose_type)

    if not included_doses:
        details["reason"] = "no therapeutic beam has a usable BeamDose"
        return {"complete": False, "sum": None, "binding": "UNRESOLVED", "details": details}
    if len(included_types) > 1:
        details["reason"] = "included beams have inconsistent BeamDoseType values"
        details["beam_dose_types"] = sorted(included_types)
        return {"complete": False, "sum": None, "binding": "UNRESOLVED", "details": details}
    details["beam_dose_type"] = next(iter(included_types), None)

    nonempty_binding_uids = {value for value in binding_uids if value}
    if target_uid and binding_uids and all(value == target_uid for value in binding_uids):
        binding = "DOSE_REFERENCE_UID_BOUND"
    elif nonempty_binding_uids:
        binding = "DOSE_REFERENCE_UID_METADATA_ONLY"
        details["observed_target_uids"] = sorted(nonempty_binding_uids)
    else:
        binding = "DOSE_REFERENCE_UID_ABSENT"
    return {
        "complete": True,
        "sum": sum(included_doses, Decimal("0")),
        "binding": binding,
        "details": details,
    }


def _unresolved_group(
    *,
    source: dict[str, Any] | None,
    fraction_group_number: str | None,
    planned_fractions: int | None,
    method: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "source_prescribed_dose_gy": _json_number(source["source"]) if source else None,
        "source_prescribed_dose_tag_path": source["tag_path"] if source else None,
        "source_dose_reference_number": source["number"] if source else None,
        "source_dose_reference_uid": source["uid"] if source else None,
        "fraction_group_number": fraction_group_number,
        "planned_fractions": planned_fractions,
        "beam_dose_sum_per_fraction_gy": None,
        "prescribed_dose_scope": "UNRESOLVED",
        "resolved_prescribed_dose_per_fraction_gy": None,
        "resolved_prescribed_dose_total_gy": None,
        "prescription_resolution_method": method,
        "prescription_resolution_status": "UNRESOLVED",
        "beam_dose_target_binding": "UNRESOLVED",
        "prescription_resolution_details": {"reason": reason},
    }


def resolve_plan_prescriptions(
    ds_plan: Dataset,
    *,
    source_prescribed_dose_gy: object = None,
    source_dose_reference_number: object = None,
    source_dose_reference_uid: object = None,
) -> list[dict[str, Any]]:
    """Resolve every FractionGroup without using BeamDose to select a target."""

    fraction_groups = list(getattr(ds_plan, "FractionGroupSequence", None) or [])
    source = contracted_source_prescription(
        ds_plan,
        source_prescribed_dose_gy=source_prescribed_dose_gy,
        source_dose_reference_number=source_dose_reference_number,
        source_dose_reference_uid=source_dose_reference_uid,
    )
    if not fraction_groups:
        return [
            _unresolved_group(
                source=source,
                fraction_group_number=None,
                planned_fractions=None,
                method="UNRESOLVED_GROUP_SCOPE",
                reason="RTPLAN has no FractionGroupSequence",
            )
        ]

    results: list[dict[str, Any]] = []
    for group_index, fraction_group in enumerate(fraction_groups):
        fraction_group_number = _text(getattr(fraction_group, "FractionGroupNumber", None)) or None
        planned_fractions = _integer_at_least_one(
            getattr(fraction_group, "NumberOfFractionsPlanned", None)
        )
        group_source = source
        if source is None:
            results.append(
                _unresolved_group(
                    source=None,
                    fraction_group_number=fraction_group_number,
                    planned_fractions=planned_fractions,
                    method="UNRESOLVED_INVALID_SOURCE_PRESCRIPTION",
                    reason="no contracted target TargetPrescriptionDose is available",
                )
            )
            continue

        matching_group_references: list[tuple[int, Dataset]] = []
        if source["number"]:
            for reference_index, reference in enumerate(
                getattr(fraction_group, "ReferencedDoseReferenceSequence", None) or []
            ):
                number = _text(getattr(reference, "ReferencedDoseReferenceNumber", None))
                if number == source["number"]:
                    matching_group_references.append((reference_index, reference))
        if len(matching_group_references) > 1:
            results.append(
                _unresolved_group(
                    source=source,
                    fraction_group_number=fraction_group_number,
                    planned_fractions=planned_fractions,
                    method="UNRESOLVED_GROUP_SCOPE",
                    reason=(
                        "multiple group-level dose-reference items match the "
                        "contracted target"
                    ),
                )
            )
            continue
        nested_sources = [
            (
                reference_index,
                reference,
                _positive_decimal(getattr(reference, "TargetPrescriptionDose", None)),
            )
            for reference_index, reference in matching_group_references
        ]
        nested_sources = [item for item in nested_sources if item[2] is not None]
        if len(nested_sources) == 1:
            reference_index, _reference, nested_value = nested_sources[0]
            group_source = {
                **source,
                "source": nested_value,
                "tag_path": (
                    f"FractionGroupSequence[{group_index}].ReferencedDoseReferenceSequence"
                    f"[{reference_index}].TargetPrescriptionDose"
                ),
            }
        elif len(nested_sources) > 1:
            results.append(
                _unresolved_group(
                    source=source,
                    fraction_group_number=fraction_group_number,
                    planned_fractions=planned_fractions,
                    method="UNRESOLVED_GROUP_SCOPE",
                    reason="multiple group-level target prescription items match the contracted target",
                )
            )
            continue
        elif len(fraction_groups) != 1:
            results.append(
                _unresolved_group(
                    source=source,
                    fraction_group_number=fraction_group_number,
                    planned_fractions=planned_fractions,
                    method="UNRESOLVED_GROUP_SCOPE",
                    reason="multiple FractionGroups lack an explicit group-level prescription for the contracted target",
                )
            )
            continue

        assert group_source is not None
        membership = _beam_membership(
            ds_plan,
            fraction_group,
            target_uid=group_source["uid"],
        )
        classification = classify_prescription_scope(
            group_source["source"],
            planned_fractions,
            membership["sum"],
            evidence_complete=bool(membership["complete"]),
            details=membership["details"],
        )
        results.append(
            {
                "source_prescribed_dose_gy": _json_number(group_source["source"]),
                "source_prescribed_dose_tag_path": group_source["tag_path"],
                "source_dose_reference_number": group_source["number"],
                "source_dose_reference_uid": group_source["uid"],
                "fraction_group_number": fraction_group_number,
                "planned_fractions": planned_fractions,
                "beam_dose_sum_per_fraction_gy": (
                    _json_number(membership["sum"]) if membership["complete"] else None
                ),
                **classification,
                "beam_dose_target_binding": membership["binding"],
            }
        )
    return results


def resolved_plan_total_gy(groups: list[dict[str, Any]]) -> float | None:
    """Return one plan total only when group identity and additivity are resolved."""

    if len(groups) != 1:
        return None
    value = groups[0].get("resolved_prescribed_dose_total_gy")
    return float(value) if value is not None else None


def source_plan_prescribed_dose_gy(groups: list[dict[str, Any]]) -> float | None:
    """Return the verbatim source value for the established single target."""

    values = [
        float(group["source_prescribed_dose_gy"])
        for group in groups
        if group.get("source_prescribed_dose_gy") is not None
    ]
    if not values:
        return None
    first = values[0]
    return first if all(value == first for value in values) else None
