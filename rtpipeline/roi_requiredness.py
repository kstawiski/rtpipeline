"""Analysis-scoped ROI requirements and auditable ROI outcomes.

The RTSTRUCT inventory is deliberately separate from the campaign analysis
contract.  A structure can be declared in an RTSTRUCT and have no contours
without being an analysis failure.  Only a contract-declared ROI is required
for extraction.
"""
from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence


class Requiredness(str, Enum):
    INVENTORY_ONLY = "inventory_only"
    ANALYSIS_OPTIONAL = "analysis_optional"
    ANALYSIS_REQUIRED = "analysis_required"


TAXONOMY_CODES = frozenset({
    "ROI_DECLARED_NO_CONTOUR_ITEM",
    "ROI_DECLARED_EMPTY_CONTOUR_SEQUENCE",
    "ROI_CONTOUR_UNPARSEABLE",
    "ROI_CONTOUR_PARTIALLY_UNPARSEABLE",
    "ROI_CONTOUR_ORPHAN_REFERENCE",
    "ROI_MASK_EMPTY_AFTER_RASTERIZATION",
    "ROI_MASK_BELOW_MIN_VOXELS",
    "REQUIRED_ROI_NOT_DECLARED",
    "REQUIRED_ROI_AMBIGUOUS_MATCH",
    "ROI_EXTRACTION_FAILED",
    "RTSTRUCT_NO_NAMED_ROIS",
})

FAILED_RADIOMICS_RESOURCE_LIMIT = "failed_radiomics_resource_limit"
FAILED_RADIOMICS_FEATURE_COMPLETENESS = "failed_radiomics_feature_completeness"


REASON_CODES = frozenset({
    "extracted",
    "not_applicable_modality",
    "not_applicable_scope",
    "not_applicable_anatomy",
    "insufficient_fov",
    "failed_source_segmentation",
    "failed_source_read",
    "failed_custom_generation",
    "failed_custom_read",
    "failed_radiomics_extraction",
    FAILED_RADIOMICS_RESOURCE_LIMIT,
    FAILED_RADIOMICS_FEATURE_COMPLETENESS,
    "indeterminate_applicability",
    "not_computed_valid_empty_scope",
})

CUSTOM_DEPENDENCY_GRAPH: dict[str, tuple[str, ...]] = {
    "iliac_vess": (
        "iliac_artery_left", "iliac_artery_right",
        "iliac_vena_left", "iliac_vena_right",
    ),
    "iliac_area": ("iliac_vess",),
    "pelvic_bones": ("sacrum", "hip_left", "hip_right", "vertebrae_S1"),
    "pelvic_bones_3mm": ("pelvic_bones",),
}


def _norm(value: Any) -> str:
    return "".join(ch for ch in str(value or "").casefold() if ch.isalnum())


def _number(value: Any) -> Optional[int]:
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number


def _sequence(dataset: Any, name: str) -> list[Any]:
    """Read a DICOM sequence by presence, not by catching AttributeError."""
    if dataset is None:
        return []
    try:
        present = name in dataset
    except (TypeError, AttributeError):
        present = hasattr(dataset, name)
    if not present:
        return []
    value = getattr(dataset, name, None)
    if value is None:
        return []
    return list(value)


@dataclass(frozen=True)
class RequiredROI:
    canonical_name: str
    aliases: tuple[str, ...] = ()
    requiredness: Requiredness = Requiredness.ANALYSIS_REQUIRED
    source: Optional[str] = None

    @property
    def accepted_names(self) -> tuple[str, ...]:
        return (self.canonical_name, *self.aliases)


@dataclass(frozen=True)
class ROIObservation:
    roi_number: Optional[int]
    name: str
    requiredness: Requiredness = Requiredness.INVENTORY_ONLY
    structural_code: Optional[str] = None
    valid_contours: int = 0
    invalid_contours: int = 0
    contour_item_present: bool = False
    contour_sequence_present: bool = False
    referenced_by: tuple[int, ...] = ()

    @property
    def has_readable_contour(self) -> bool:
        return self.valid_contours > 0


def dependency_state_from_observation(observation: ROIObservation) -> dict[str, bool]:
    """Translate an RTSTRUCT observation into custom-source evidence."""
    empty_contour_codes = {
        "ROI_DECLARED_NO_CONTOUR_ITEM",
        "ROI_DECLARED_EMPTY_CONTOUR_SEQUENCE",
    }
    return {
        "readable": (
            observation.structural_code is None
            or observation.structural_code in empty_contour_codes
        ),
        "non_empty": observation.has_readable_contour,
    }


@dataclass(frozen=True)
class RTStructInventory:
    path: Path
    rois: tuple[ROIObservation, ...]
    orphan_references: tuple[int, ...] = ()
    structural_codes: tuple[str, ...] = ()

    @property
    def named_rois(self) -> tuple[ROIObservation, ...]:
        return tuple(roi for roi in self.rois if roi.name)

    @property
    def by_number(self) -> dict[int, ROIObservation]:
        return {
            roi.roi_number: roi for roi in self.rois
            if roi.roi_number is not None
        }


@dataclass(frozen=True)
class RequirementMatch:
    requirement: RequiredROI
    observation: Optional[ROIObservation]
    structural_code: Optional[str] = None
    candidates: tuple[ROIObservation, ...] = ()


@dataclass(frozen=True)
class CustomApplicability:
    roi_name: str
    reason_code: str
    dependencies: tuple[str, ...]
    detail: str = ""
    fatal: bool = False

    def __post_init__(self) -> None:
        if self.reason_code not in REASON_CODES:
            raise ValueError(f"unknown custom ROI reason code: {self.reason_code}")


@dataclass
class DenominatorLedger:
    """Append-only in-memory ledger for course and course-ROI decisions."""

    course_rows: list[dict[str, Any]] = field(default_factory=list)
    roi_rows: list[dict[str, Any]] = field(default_factory=list)
    expected_pairs: set[tuple[str, str]] = field(default_factory=set)

    def expect_course_roi(self, course_id: str, roi_name: str) -> None:
        self.expected_pairs.add((str(course_id), str(roi_name)))

    def record_course(self, course_id: str, patient_id: str, **states: Any) -> dict[str, Any]:
        row = {"entity": "COURSE", "course_id": str(course_id),
               "patient_id": str(patient_id), **states}
        self.course_rows.append(row)
        return row

    def record_roi(
        self, course_id: str, patient_id: str, roi_name: str,
        *, reason_code: str, disposition: str = "excluded",
        **values: Any,
    ) -> dict[str, Any]:
        if reason_code not in REASON_CODES and reason_code not in TAXONOMY_CODES:
            raise ValueError(f"unknown ROI reason/taxonomy code: {reason_code}")
        row = {
            "entity": "COURSE_ROI",
            "course_id": str(course_id),
            "patient_id": str(patient_id),
            "roi_name": str(roi_name),
            "disposition": str(disposition),
            "reason_code": reason_code,
            **values,
        }
        self.roi_rows.append(row)
        return row

    def ensure_expected_pairs(self) -> None:
        present = {(r.get("course_id", ""), r.get("roi_name", "")) for r in self.roi_rows}
        missing = sorted(self.expected_pairs - present)
        if missing:
            raise ValueError("denominator ledger is missing course-ROI pairs: " + repr(missing))

    def patient_rows(self) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in self.course_rows:
            grouped.setdefault(str(row["patient_id"]), []).append(row)
        result = []
        for patient_id, rows in sorted(grouped.items()):
            result.append({
                "entity": "PATIENT",
                "patient_id": patient_id,
                "screened": int(any(bool(r.get("screened")) for r in rows)),
                "in_scope": int(any(bool(r.get("in_scope")) for r in rows)),
                "out_of_scope": int(all(bool(r.get("out_of_scope")) for r in rows)),
                "adequate_coverage": int(any(bool(r.get("adequate_coverage")) for r in rows)),
                "insufficient_coverage": int(any(bool(r.get("insufficient_coverage")) for r in rows)),
                "valid_derivation": int(any(bool(r.get("valid_derivation")) for r in rows)),
                "technical_exclusion": int(any(bool(r.get("technical_exclusion")) for r in rows)),
                "indeterminate": int(any(bool(r.get("indeterminate")) for r in rows)),
                "extracted": int(any(bool(r.get("extracted")) for r in rows)),
                "course_count": len(rows),
            })
        return result

    def summary(self) -> dict[str, Any]:
        self.ensure_expected_pairs()
        states = (
            "screened", "in_scope", "out_of_scope", "adequate_coverage",
            "insufficient_coverage", "valid_derivation", "technical_exclusion",
            "indeterminate", "extracted",
        )
        course = {state: sum(bool(row.get(state)) for row in self.course_rows) for state in states}
        per_roi: dict[str, dict[str, int]] = {}
        for row in self.roi_rows:
            roi_name = str(row["roi_name"])
            modality = str(row.get("modality", ""))
            name = f"{modality}:{roi_name}" if modality else roi_name
            counts = per_roi.setdefault(name, {
                "extracted": 0, "excluded_anatomy": 0, "excluded_technical": 0,
            })
            reason = str(row.get("reason_code", ""))
            if reason == "extracted" or row.get("disposition") == "extracted":
                counts["extracted"] += 1
            elif reason in {"not_applicable_modality", "not_applicable_scope", "not_applicable_anatomy", "insufficient_fov",
                            "not_computed_valid_empty_scope"}:
                counts["excluded_anatomy"] += 1
            else:
                counts["excluded_technical"] += 1
        return {
            "COURSE": course,
            "PATIENT": {state: sum(bool(row.get(state)) for row in self.patient_rows()) for state in states},
            "COURSE_ROI": per_roi,
        }

    def write(self, directory: Path, *, prefix: str = "radiomics") -> tuple[Path, Path, Path]:
        self.ensure_expected_pairs()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        rows_path = directory / f"{prefix}_roi_ledger.json"
        summary_path = directory / f"{prefix}_denominators.json"
        patient_path = directory / f"{prefix}_patient_ledger.json"
        _atomic_json(rows_path, {"course": self.course_rows, "course_roi": self.roi_rows})
        _atomic_json(summary_path, self.summary())
        _atomic_json(patient_path, self.patient_rows())
        return rows_path, summary_path, patient_path


def write_modality_ledger(directory: Path, ledger: DenominatorLedger, modality: str) -> tuple[Path, Path, Path]:
    """Persist a modality ledger and rebuild the non-overwriting combined ledger."""
    directory = Path(directory)
    modality_name = str(modality).upper()
    modality_prefix = f"radiomics_{modality_name.casefold()}"
    modality_ledger = DenominatorLedger(
        course_rows=[{**row, "modality": modality_name} for row in ledger.course_rows],
        roi_rows=[{**row, "modality": modality_name} for row in ledger.roi_rows],
        expected_pairs=set(ledger.expected_pairs),
    )
    paths = modality_ledger.write(directory, prefix=modality_prefix)

    course_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    roi_rows: list[dict[str, Any]] = []
    roi_seen: set[tuple[str, str, str, str, str]] = set()
    for candidate_modality in ("ct", "mr"):
        path = directory / f"radiomics_{candidate_modality}_roi_ledger.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"modality denominator ledger is unreadable: {path}: {exc}") from exc
        for row in payload.get("course", ()):
            key = (str(row.get("course_id", "")), str(row.get("patient_id", "")))
            course_groups.setdefault(key, []).append(dict(row))
        for row in payload.get("course_roi", ()):
            normalized = dict(row)
            normalized.setdefault("modality", candidate_modality.upper())
            key = (
                str(normalized.get("course_id", "")),
                str(normalized.get("patient_id", "")),
                str(normalized.get("modality", "")),
                str(normalized.get("roi_name", "")),
                str(normalized.get("segmentation_source", normalized.get("source", ""))),
            )
            if key in roi_seen:
                continue
            roi_seen.add(key)
            roi_rows.append(normalized)

    states_any = (
        "screened", "in_scope", "adequate_coverage", "insufficient_coverage",
        "valid_derivation", "technical_exclusion", "indeterminate", "extracted",
    )
    course_rows: list[dict[str, Any]] = []
    for (course_id, patient_id), rows in sorted(course_groups.items()):
        merged: dict[str, Any] = {
            "entity": "COURSE",
            "course_id": course_id,
            "patient_id": patient_id,
            "out_of_scope": all(bool(row.get("out_of_scope")) for row in rows),
            "modalities": sorted({str(row.get("modality", "")) for row in rows if row.get("modality")}),
            "modality_reason_codes": {
                str(row.get("modality", "")): str(row.get("reason_code", "")) for row in rows
            },
        }
        for state in states_any:
            merged[state] = any(bool(row.get(state)) for row in rows)
        merged["reason_code"] = (
            "extracted" if merged["extracted"] else
            "indeterminate_applicability" if merged["indeterminate"] else
            "failed_radiomics_extraction"
        )
        course_rows.append(merged)
    combined = DenominatorLedger(course_rows=course_rows, roi_rows=roi_rows)
    combined.write(directory)
    return paths


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _parse_requirement(value: Any, default: Requiredness) -> RequiredROI:
    if isinstance(value, str):
        return RequiredROI(value.strip(), (), default)
    if not isinstance(value, Mapping):
        raise ValueError("ROI contract entries must be names or mappings")
    name = value.get("canonical_name", value.get("name"))
    if not isinstance(name, str) or not name.strip():
        raise ValueError("ROI contract entry has no canonical_name")
    aliases = value.get("approved_aliases", value.get("aliases", ())) or ()
    if isinstance(aliases, str):
        aliases = (aliases,)
    aliases = tuple(str(alias).strip() for alias in aliases if str(alias).strip())
    raw_class = value.get("requiredness", default.value)
    requiredness = Requiredness(str(raw_class))
    source = value.get("source")
    return RequiredROI(name.strip(), aliases, requiredness, str(source) if source else None)


def _contract_section(contract: Any, modality: str) -> Mapping[str, Any]:
    if contract is None:
        return {}
    if hasattr(contract, "radiomics_analysis_contract"):
        contract = getattr(contract, "radiomics_analysis_contract")
    if not isinstance(contract, Mapping):
        return {}
    for key in (modality, modality.casefold(), modality.upper(), modality.lower()):
        section = contract.get(key)
        if isinstance(section, Mapping):
            return section
    return contract


def requirements_from_contract(contract: Any, modality: str = "CT") -> tuple[RequiredROI, ...]:
    section = _contract_section(contract, modality)
    result: list[RequiredROI] = []
    for key, default in (
        ("required_rois", Requiredness.ANALYSIS_REQUIRED),
        ("analysis_required", Requiredness.ANALYSIS_REQUIRED),
        ("optional_rois", Requiredness.ANALYSIS_OPTIONAL),
        ("analysis_optional", Requiredness.ANALYSIS_OPTIONAL),
        ("inventory_only", Requiredness.INVENTORY_ONLY),
    ):
        values = section.get(key, ())
        if isinstance(values, Mapping):
            values = [dict(v, canonical_name=k) if isinstance(v, Mapping) else {"canonical_name": k, "aliases": v}
                      for k, v in values.items()]
        if isinstance(values, str):
            values = [values]
        for value in values or ():
            result.append(_parse_requirement(value, default))
    # Repeated canonical entries with conflicting classes are unsafe.
    seen: dict[str, RequiredROI] = {}
    for requirement in result:
        key = _norm(requirement.canonical_name)
        prior = seen.get(key)
        if prior is not None and prior.requiredness != requirement.requiredness:
            raise ValueError(f"ROI contract has conflicting requiredness for {requirement.canonical_name}")
        seen[key] = requirement
    return tuple(seen.values())


def requiredness_for(
    source: str,
    roi_name: str,
    *,
    contract: Any = None,
    modality: str = "CT",
    explicitly_selected_model: bool = False,
) -> Requiredness:
    for requirement in requirements_from_contract(contract, modality):
        if requirement.source and _norm(requirement.source) != _norm(source):
            continue
        if any(_norm(accepted) == _norm(roi_name) for accepted in requirement.accepted_names):
            return requirement.requiredness
    # Explicit custom-model output is an analysis input only when the campaign
    # selected it. This retains fail-closed validation of a selected model while
    # keeping ordinary declared Manual structures inventory-only.
    if explicitly_selected_model:
        return Requiredness.ANALYSIS_REQUIRED
    return Requiredness.INVENTORY_ONLY


def inspect_rtstruct(path: Optional[Path], dataset: Any = None) -> RTStructInventory:
    if dataset is None:
        import pydicom
        dataset = pydicom.dcmread(str(path), stop_before_pixels=True)
    declared = _sequence(dataset, "StructureSetROISequence")
    observations: list[ROIObservation] = []
    by_number: dict[int, int] = {}
    for item in declared:
        name = str(getattr(item, "ROIName", "") or "").strip()
        number = _number(getattr(item, "ROINumber", None))
        observations.append(ROIObservation(number, name))
        if number is not None and number not in by_number:
            by_number[number] = len(observations) - 1
    contours = _sequence(dataset, "ROIContourSequence")
    refs: dict[int, list[int]] = {}
    orphan: list[int] = []
    valid_counts: dict[int, int] = {}
    invalid_counts: dict[int, int] = {}
    contour_item_present: set[int] = set()
    contour_sequence_present: set[int] = set()
    contour_sequence_nonempty: set[int] = set()
    for item in contours:
        ref = _number(getattr(item, "ReferencedROINumber", None))
        if ref is None or ref not in by_number:
            orphan.append(ref if ref is not None else -1)
            continue
        contour_item_present.add(ref)
        refs.setdefault(ref, []).append(ref)
        if "ContourSequence" not in item:
            continue
        contour_sequence_present.add(ref)
        sequence = getattr(item, "ContourSequence", None)
        sequence_items = list(sequence or [])
        if sequence_items:
            contour_sequence_nonempty.add(ref)
        for contour in sequence_items:
            data = getattr(contour, "ContourData", None) if "ContourData" in contour else None
            valid = False
            try:
                values = [float(value) for value in (data or ())]
                valid = len(values) >= 6 and len(values) % 3 == 0 and all(math.isfinite(v) for v in values)
            except (TypeError, ValueError):
                valid = False
            if valid:
                valid_counts[ref] = valid_counts.get(ref, 0) + 1
            else:
                invalid_counts[ref] = invalid_counts.get(ref, 0) + 1
    final: list[ROIObservation] = []
    codes: list[str] = []
    if not any(roi.name for roi in observations):
        codes.append("RTSTRUCT_NO_NAMED_ROIS")
    for index, roi in enumerate(observations):
        number = roi.roi_number
        if number not in contour_item_present:
            code = "ROI_DECLARED_NO_CONTOUR_ITEM"
        elif number not in contour_sequence_present or number not in contour_sequence_nonempty:
            code = "ROI_DECLARED_EMPTY_CONTOUR_SEQUENCE"
        elif invalid_counts.get(number, 0) and valid_counts.get(number, 0):
            code = "ROI_CONTOUR_PARTIALLY_UNPARSEABLE"
        elif invalid_counts.get(number, 0):
            code = "ROI_CONTOUR_UNPARSEABLE"
        else:
            code = None
        if code:
            codes.append(code)
        final.append(ROIObservation(
            roi_number=number,
            name=roi.name,
            structural_code=code,
            valid_contours=valid_counts.get(number, 0),
            invalid_contours=invalid_counts.get(number, 0),
            contour_item_present=number in contour_item_present,
            contour_sequence_present=number in contour_sequence_present,
            referenced_by=tuple(refs.get(number, ())),
        ))
    if orphan:
        codes.append("ROI_CONTOUR_ORPHAN_REFERENCE")
    return RTStructInventory(Path(path) if path is not None else Path("<in-memory>"), tuple(final), tuple(sorted(set(orphan))), tuple(dict.fromkeys(codes)))


def match_requirements(
    inventory: RTStructInventory, requirements: Iterable[RequiredROI], *, source: Optional[str] = None,
) -> tuple[RequirementMatch, ...]:
    matches: list[RequirementMatch] = []
    for requirement in requirements:
        if requirement.requiredness != Requiredness.ANALYSIS_REQUIRED:
            continue
        if requirement.source and (
            source is None or _norm(requirement.source) != _norm(source)
        ):
            continue
        accepted = {_norm(name) for name in requirement.accepted_names}
        candidates = tuple(roi for roi in inventory.named_rois if _norm(roi.name) in accepted)
        if not candidates:
            matches.append(RequirementMatch(requirement, None, "REQUIRED_ROI_NOT_DECLARED"))
        elif len(candidates) > 1:
            matches.append(RequirementMatch(requirement, None, "REQUIRED_ROI_AMBIGUOUS_MATCH", candidates))
        else:
            roi = candidates[0]
            matches.append(RequirementMatch(requirement, roi, roi.structural_code, candidates))
    return tuple(matches)


def _state(value: Any) -> str:
    if isinstance(value, Mapping):
        if value.get("readable") is False or value.get("status") in {"unreadable", "read_error"}:
            return "unreadable"
        if value.get("non_empty") is True or value.get("has_voxels") is True:
            return "readable_nonempty"
        if value.get("empty") is True or value.get("non_empty") is False:
            return "empty"
        value = value.get("status", value.get("state", "absent"))
    if isinstance(value, bool):
        return "readable_nonempty" if value else "empty"
    if value is None:
        return "absent"
    if isinstance(value, str):
        value = value.casefold()
        if value in {"readable_nonempty", "nonempty", "present", "ok", "extracted"}:
            return "readable_nonempty"
        if value in {"unreadable", "read_error", "failed_read"}:
            return "unreadable"
        if value in {"empty", "absent", "missing", "not_present"}:
            return "empty"
    try:
        return "readable_nonempty" if len(value) and bool(getattr(value, "any", lambda: True)()) else "empty"
    except Exception:
        return "unreadable"


def _fov_contains(fov: Any, region: str, anatomy_bounds: Any = None) -> Optional[bool]:
    if isinstance(fov, Mapping):
        if region in fov.get("contains_regions", ()) or region in fov.get("included_regions", ()):
            return True
        if region in fov.get("excluded_regions", ()) or region in fov.get("excluded_anatomy", ()):
            return False
        for key in ("anatomy_in_fov", "contains_anatomy", "region_in_fov"):
            if key in fov and isinstance(fov[key], Mapping) and region in fov[key]:
                return bool(fov[key][region])
        if "contains" in fov and isinstance(fov["contains"], bool):
            return bool(fov["contains"])
        fov_bounds = fov.get("bounds", fov.get("fov_bounds"))
    else:
        fov_bounds = fov
    if anatomy_bounds is None or fov_bounds is None:
        return None
    try:
        f_pairs = [(float(a), float(b)) for a, b in zip(fov_bounds[::2], fov_bounds[1::2])]
        a_pairs = [(float(a), float(b)) for a, b in zip(anatomy_bounds[::2], anatomy_bounds[1::2])]
        return all(f_min <= a_max and f_max >= a_min for (f_min, f_max), (a_min, a_max) in zip(f_pairs, a_pairs))
    except (TypeError, ValueError):
        return None


def assess_custom_applicability(
    roi_name: str,
    dependency_states: Mapping[str, Any],
    planning_ct_fov: Any = None,
    *,
    generated_state: Any = "absent",
    anatomy_region: str = "pelvis",
    anatomy_bounds: Any = None,
    in_scope: Optional[bool] = None,
    custom_provenance: Optional[Mapping[str, Mapping[str, Any]]] = None,
    generation_outcome: Optional[Mapping[str, Any]] = None,
) -> CustomApplicability:
    """Classify a configured derived ROI without treating anatomy absence as corruption."""
    name = str(roi_name)
    provenance = custom_provenance or {}

    def derivation_spec(derived_name: str) -> tuple[tuple[str, ...], str]:
        item = provenance.get(derived_name)
        if isinstance(item, Mapping):
            configured_sources = item.get("source_structures")
            if isinstance(configured_sources, Sequence) and not isinstance(
                configured_sources, (str, bytes)
            ):
                sources = tuple(str(value) for value in configured_sources if str(value))
            else:
                sources = ()
            operation = str(item.get("operation") or "union").casefold()
            if sources:
                return sources, operation
        return CUSTOM_DEPENDENCY_GRAPH.get(derived_name, ()), "union"

    dependencies, operation = derivation_spec(name)
    if not dependencies and isinstance(dependency_states.get(name), Mapping):
        dependencies = tuple(
            str(value)
            for value in dependency_states[name].get("source_structures", ())
        )
    generated = _state(generated_state)
    outcome_status = str((generation_outcome or {}).get("status") or "").casefold()
    if generated == "readable_nonempty":
        return CustomApplicability(
            name,
            "extracted",
            tuple(dependencies),
            "derived ROI is readable and non-empty",
        )
    if outcome_status in {"generated", "generated_partial"}:
        return CustomApplicability(
            name,
            "failed_custom_generation",
            tuple(dependencies),
            "custom-stage metadata declares a generated ROI but the RTSTRUCT ROI is missing",
            True,
        )
    if outcome_status == "failed_generation":
        return CustomApplicability(
            name,
            "failed_custom_generation",
            tuple(dependencies),
            "custom stage recorded a derivation failure",
            True,
        )
    if in_scope is False:
        return CustomApplicability(
            name,
            "not_applicable_scope",
            tuple(dependencies),
            "campaign scope excludes this ROI",
        )

    def resolve_dependency_state(
        dependency: str,
        seen: frozenset[str] = frozenset(),
    ) -> str:
        if dependency in seen:
            return "unreadable"
        if dependency in dependency_states:
            return _state(dependency_states[dependency])
        child_dependencies, child_operation = derivation_spec(dependency)
        if not child_dependencies:
            return "empty"
        child_states = [
            resolve_dependency_state(child, seen | {dependency})
            for child in child_dependencies
        ]
        if any(item == "unreadable" for item in child_states):
            return "unreadable"
        if child_operation == "union":
            return (
                "readable_nonempty"
                if any(item == "readable_nonempty" for item in child_states)
                else "empty"
            )
        return (
            "readable_nonempty"
            if child_states and all(item == "readable_nonempty" for item in child_states)
            else "empty"
        )

    states = [resolve_dependency_state(dependency) for dependency in dependencies]
    if generated == "unreadable":
        return CustomApplicability(
            name,
            "failed_custom_read",
            tuple(dependencies),
            "derived ROI could not be read",
            True,
        )
    if any(state == "unreadable" for state in states):
        if any(state == "readable_nonempty" for state in states):
            return CustomApplicability(
                name,
                "indeterminate_applicability",
                tuple(dependencies),
                "dependency evidence conflicts",
                True,
            )
        return CustomApplicability(
            name,
            "failed_source_read",
            tuple(dependencies),
            "dependency segmentation cannot be read",
        )

    has_nonempty = any(state == "readable_nonempty" for state in states)
    has_empty = any(state in {"empty", "absent"} for state in states)
    can_generate = (
        has_nonempty
        if operation == "union"
        else bool(states) and all(state == "readable_nonempty" for state in states)
    )
    if outcome_status == "source_unavailable" and has_nonempty:
        return CustomApplicability(
            name,
            "indeterminate_applicability",
            tuple(dependencies),
            "custom-stage source outcome conflicts with current dependency masks",
            True,
        )
    if can_generate:
        detail = (
            "at least one union dependency is readable and non-empty but the derived ROI is missing"
            if operation == "union"
            else "all dependencies are readable and non-empty but the derived ROI is missing"
        )
        return CustomApplicability(
            name,
            "failed_custom_generation",
            tuple(dependencies),
            detail,
            True,
        )
    if has_nonempty and has_empty:
        return CustomApplicability(
            name,
            "indeterminate_applicability",
            tuple(dependencies),
            "dependency evidence conflicts for an operation that requires every source",
            True,
        )
    contains = _fov_contains(planning_ct_fov, anatomy_region, anatomy_bounds)
    if contains is None:
        return CustomApplicability(
            name,
            "insufficient_fov",
            tuple(dependencies),
            "un-cropped planning CT FOV does not resolve anatomy scope",
        )
    if not contains:
        return CustomApplicability(
            name,
            "not_applicable_anatomy",
            tuple(dependencies),
            "dependencies are absent or empty and the FOV excludes the anatomy",
        )
    return CustomApplicability(
        name,
        "failed_source_segmentation",
        tuple(dependencies),
        "dependencies are absent or empty although anatomy is in the FOV",
    )


def classify_rasterized_mask(mask: Any, minimum_voxels: int = 1) -> Optional[str]:
    """Return a controlled structural code for a rasterized ROI mask."""
    try:
        array = getattr(mask, "astype", lambda *_args, **_kwargs: mask)(bool)
        count = int(array.sum())
    except Exception:
        return "ROI_EXTRACTION_FAILED"
    if count == 0:
        return "ROI_MASK_EMPTY_AFTER_RASTERIZATION"
    if count < int(minimum_voxels):
        return "ROI_MASK_BELOW_MIN_VOXELS"
    return None


def taxonomy_is_fatal(code: Optional[str], requiredness: Requiredness) -> bool:
    return requiredness == Requiredness.ANALYSIS_REQUIRED and code in TAXONOMY_CODES


__all__ = [
    "CUSTOM_DEPENDENCY_GRAPH", "CustomApplicability", "DenominatorLedger",
    "REASON_CODES", "RTStructInventory", "ROIObservation", "RequiredROI",
    "Requiredness", "RequirementMatch", "TAXONOMY_CODES", "assess_custom_applicability",
    "classify_rasterized_mask",
    "inspect_rtstruct", "match_requirements", "requirements_from_contract",
    "requiredness_for", "taxonomy_is_fatal", "write_modality_ledger",
]
