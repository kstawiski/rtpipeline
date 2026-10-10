"""Governed multi-plan physical course dose for descriptive DVH (RF19).

Organize retains independently delivered plans under
``independent_delivered_plans_unreconciled`` without a course dose grid. This
module adds a DVH-only composite when RT evidence shows that every retained
plan was delivered on the planning-CT frame of reference and that no plan
replaces another. It never revises organize membership or the contract.

DICOM semantics (PS3.3 RT Dose Module): ``DoseSummationType`` PLAN is the dose
of one whole RT Plan, i.e. all planned fractions of its fraction group;
FRACTION is one whole fraction group (equal to PLAN for a single-group plan).
Pixel values times ``DoseGridScaling`` are dose in ``DoseUnits``; only
PHYSICAL dose in GY is accepted. Each plan grid is weighted by RTRECORD
delivered / planned fractions, resampled onto one union grid with trilinear
interpolation (zero outside a source grid) and summed by organize's validated
``_sum_doses_with_resample``. The output is a MULTI_PLAN RTDOSE.

Selection is read-only. Refusals are counts-safe reason codes and never carry
paths or DICOM identities.
"""
from __future__ import annotations

from dataclasses import dataclass
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pydicom

from .course_contract import CourseContract, CourseContractError, load_course_contract
from .plan_approval import approval_status
from .planning_dose_selection import PlanningDoseError, _local, _sha256

BASIS = "multi_plan_physical_dose_sum_delivered_fraction_weighted"
MODE = "composite_course_dose"
SIDECAR = "metadata/composite_course_dose.json"
OUTPUT = "metadata/composite_course_dose/RD_composite.dcm"
# Organize's plan artifact for these courses is one copied intent plan. The
# composite therefore carries its own evaluation-only plan sum (UNAPPROVED,
# PlanIntent REVIEW) referencing every summed plan, as organize does for
# summed phases.
OUTPUT_PLAN = "metadata/composite_course_dose/RP_composite.dcm"
ELIGIBLE_CLASSIFICATION = "independent_delivered_plans_unreconciled"
# Whole-plan dose objects only. MULTI_PLAN/PLAN_SUM cover several plans, so a
# per-plan delivered-fraction weight would be undefined; BEAM and session
# types are partial plans.
PER_PLAN_SUMMATION_TYPES = frozenset({"PLAN", "FRACTION"})
# Plans sharing one structure set but delivered more than this many days apart
# are treated as separate treatment courses (for example re-irradiation), not
# phases of one course. Planned inter-phase breaks are days, not weeks.
MAX_INTER_PLAN_GAP_DAYS = 28
# Equal-signature plans are accepted only as distinct targets: no treatment
# isocenter of one plan within this distance of an isocenter of the other.
# A shared isocenter with an equal signature may be a revision or copy.
SHARED_ISOCENTER_TOLERANCE_MM = 5.0
INTERPOLATION = {
    "method": "trilinear",
    "implementation": "scipy.ndimage.map_coordinates(order=1, mode='constant', cval=0.0)",
    "frame_axis": "linear interpolation of GridFrameOffsetVector",
    "outside_source_extent": 0.0,
    "grid": "union extent of all sources; in-plane spacing and orientation of the finest-voxel source; "
            "frame spacing = smallest source frame spacing plus every source frame position",
    "output_quantization_gy": "1/DoseGridScaling (uint32)",
}
REFUSAL_CODES = (
    "unsafe_course_path", "invalid_course_contract", "contract_selected_dose",
    "classification_not_eligible", "retained_plan_count_below_two",
    "unresolved_delivery_records", "unrepresented_delivered_plan",
    "plan_without_recorded_delivery", "planned_fractions_unknown",
    "delivered_exceeds_planned", "treatment_dates_unavailable",
    "plan_unreadable", "plan_not_approved", "plan_identity_mismatch",
    "multiple_fraction_groups", "planned_fraction_mismatch",
    "brachytherapy_plan_not_supported", "structure_set_mismatch",
    "replacement_plan_chain_not_summed", "separate_course_suspected",
    "equal_signature_shared_isocenter_unproven",
    "organize_dose_qc_failed", "planning_ct_unavailable",
    "planning_ct_frame_unresolved", "frame_of_reference_mismatch",
    "dose_inventory_unreadable", "dose_identity_invalid", "no_plan_level_dose",
    "ambiguous_plan_level_doses", "dose_not_physical_gy", "dose_ct_frame_mismatch",
    "dose_geometry_qc_failed", "composite_prescription_implausible",
    "composite_pixels_invalid",
)


def _refuse(code):
    raise PlanningDoseError(code)


def _uid_hash(uid):
    return hashlib.sha256(str(uid).encode()).hexdigest()


def _date(value):
    text = str(value).replace("-", "")
    return datetime.date(int(text[:4]), int(text[4:6]), int(text[6:8]))


@dataclass(frozen=True)
class CompositeSelection:
    reason_code: str
    plan_paths: tuple = ()
    dose_paths: tuple = ()
    weights: tuple = ()
    plan_artifact: Path | None = None
    evidence: dict | None = None

    @property
    def accepted(self):
        return self.reason_code == "accepted"


def select_composite_course_dose(course_dir: Path | str) -> CompositeSelection:
    """Return evidence-bound composite inputs or an explicit refusal; write nothing."""
    try:
        return _select(Path(course_dir))
    except PlanningDoseError as exc:
        return CompositeSelection(str(exc))
    except (CourseContractError, OSError, ValueError, TypeError, KeyError, AttributeError):
        return CompositeSelection("invalid_course_contract")


def _relations(date_sets):
    counts = {}
    for i in range(len(date_sets)):
        for j in range(i + 1, len(date_sets)):
            a, b = date_sets[i], date_sets[j]
            kind = ("identical" if a == b else "contained" if (a < b or b < a)
                    else "overlapping" if a & b else "disjoint")
            counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))


def _isocenters(plan):
    points = set()
    for beam in getattr(plan, "BeamSequence", []) or []:
        if str(getattr(beam, "TreatmentDeliveryType", "TREATMENT") or "TREATMENT").upper() != "TREATMENT":
            continue
        for point in getattr(beam, "ControlPointSequence", []) or []:
            if hasattr(point, "IsocenterPosition"):
                points.add(tuple(float(v) for v in point.IsocenterPosition))
    return points


def _signature(item):
    return (round(float(item.get("resolved_prescribed_dose_per_fraction_gy") or 0.0), 2),
            int(item.get("planned_fraction_count") or 0))


def _select(course_dir):
    if course_dir.is_symlink():
        _refuse("unsafe_course_path")
    root = course_dir.resolve()
    metadata = _local(root, root / "metadata/case_metadata.json")
    try:
        data = json.loads(metadata.read_text())["course_contract"]
        contract = CourseContract(root, metadata, data)
        if contract.selected_doses or data.get("dose_grid") is not None:
            _refuse("contract_selected_dose")
        if data["dose_classification"].get("classification") != ELIGIBLE_CLASSIFICATION:
            _refuse("classification_not_eligible")
        items = list(contract.selected_plans)
        if len(items) < 2:
            _refuse("retained_plan_count_below_two")
        delivery = data["delivery"]
        if delivery.get("unresolved_record_plan_uids"):
            _refuse("unresolved_delivery_records")
        if (data.get("dose_completeness") or {}).get("unselected_delivered_plan_uids"):
            _refuse("unrepresented_delivered_plan")
        plan_paths = [_local(root, item["path"]) for item in items]
    except (KeyError, TypeError, AttributeError):
        _refuse("invalid_course_contract")

    # Delivery evidence: every retained plan delivered, never above plan.
    date_sets = []
    for item in items:
        delivered = int(item.get("delivered_fraction_count") or 0)
        planned = int(item.get("planned_fraction_count") or 0)
        if delivered < 1:
            _refuse("plan_without_recorded_delivery")
        if planned < 1:
            _refuse("planned_fractions_unknown")
        if delivered > planned:
            _refuse("delivered_exceeds_planned")
        try:
            dates = {_date(value) for value in item.get("treatment_dates") or []}
        except (ValueError, TypeError):
            _refuse("treatment_dates_unavailable")
        if not dates:
            _refuse("treatment_dates_unavailable")
        date_sets.append(dates)

    # Re-plans: an interrupted plan followed by a later plan is a replacement
    # or remainder chain; equal-signature plans with an interrupted member are
    # revisions. Neither is summed, even with fraction weights.
    for i, item in enumerate(items):
        partial = int(item["delivered_fraction_count"]) < int(item["planned_fraction_count"])
        for j, other in enumerate(items):
            if i == j:
                continue
            other_partial = int(other["delivered_fraction_count"]) < int(other["planned_fraction_count"])
            if partial and min(date_sets[j]) > max(date_sets[i]):
                _refuse("replacement_plan_chain_not_summed")
            if (partial or other_partial) and _signature(item) == _signature(other):
                _refuse("replacement_plan_chain_not_summed")
    # Separate courses sharing one structure set: a long delivery gap.
    blocks = sorted((min(d), max(d)) for d in date_sets)
    reach = blocks[0][1]
    max_gap = 0
    for start, end in blocks[1:]:
        max_gap = max(max_gap, (start - reach).days)
        reach = max(reach, end)
    if max_gap > MAX_INTER_PLAN_GAP_DAYS:
        _refuse("separate_course_suspected")

    plans = []
    for item, path in zip(items, plan_paths):
        try:
            plan = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception:
            _refuse("plan_unreadable")
        if approval_status(plan) != "APPROVED":
            _refuse("plan_not_approved")
        uid = str(getattr(plan, "SOPInstanceUID", "") or "")
        if not uid or uid != item.get("sop_instance_uid"):
            _refuse("plan_identity_mismatch")
        groups = list(getattr(plan, "FractionGroupSequence", []) or [])
        if len(groups) != 1:
            _refuse("multiple_fraction_groups")
        if int(getattr(groups[0], "NumberOfFractionsPlanned", 0) or 0) != int(item["planned_fraction_count"]):
            _refuse("planned_fraction_mismatch")
        if hasattr(plan, "ApplicationSetupSequence") or str(getattr(plan, "BrachyTreatmentType", "") or ""):
            _refuse("brachytherapy_plan_not_supported")
        plans.append((uid, plan))
    # Equal-signature plans must be geometrically distinct targets.
    isocenters = [_isocenters(plan) for _, plan in plans]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if _signature(items[i]) != _signature(items[j]):
                continue
            if not isocenters[i] or not isocenters[j] or any(
                    np.linalg.norm(np.subtract(a, b)) <= SHARED_ISOCENTER_TOLERANCE_MM
                    for a in isocenters[i] for b in isocenters[j]):
                _refuse("equal_signature_shared_isocenter_unproven")
    structure_sets = {tuple(sorted(str(r.ReferencedSOPInstanceUID)
                                   for r in getattr(plan, "ReferencedStructureSetSequence", []) or []))
                      for _, plan in plans}
    if len(structure_sets) != 1 or structure_sets == {()}:
        _refuse("structure_set_mismatch")

    if contract.dose_qc.get("pass") is not True:
        _refuse("organize_dose_qc_failed")
    # Full contract validation: identity, approval, CT provenance, QC.
    contract = load_course_contract(root)
    ct_dir = contract.planning_ct_dir
    if ct_dir is None:
        _refuse("planning_ct_unavailable")
    ct_dir = _local(root, ct_dir)
    ct_frames = set()
    try:
        for path in sorted(ct_dir.rglob("*")):
            _local(root, path)
            if not path.is_file():
                continue
            ct = pydicom.dcmread(path, stop_before_pixels=True)
            if str(getattr(ct, "Modality", "")) != "CT":
                _refuse("planning_ct_frame_unresolved")
            ct_frames.add(str(getattr(ct, "FrameOfReferenceUID", "") or ""))
    except PlanningDoseError:
        raise
    except Exception:
        _refuse("planning_ct_frame_unresolved")
    if len(ct_frames) != 1 or "" in ct_frames:
        _refuse("planning_ct_frame_unresolved")
    frame = next(iter(ct_frames))
    if any(str(getattr(plan, "FrameOfReferenceUID", "") or "") != frame for _, plan in plans):
        _refuse("frame_of_reference_mismatch")

    dose_dir = _local(root, root / "DICOM/RTDOSE")
    inventory = []
    if dose_dir.is_dir():
        for entry in sorted(dose_dir.rglob("*")):
            entry = _local(root, entry)
            if not entry.is_file():
                continue
            try:
                ds = pydicom.dcmread(entry, stop_before_pixels=True)
            except Exception:
                _refuse("dose_inventory_unreadable")
            if str(getattr(ds, "Modality", "")) != "RTDOSE":
                _refuse("dose_inventory_unreadable")
            inventory.append((entry, ds))
    from .organize import _validated_dose_grid_geometry
    doses = []
    for uid, plan in plans:
        candidates = []
        for path, ds in inventory:
            refs = {str(getattr(r, "ReferencedSOPInstanceUID", "") or "")
                    for r in getattr(ds, "ReferencedRTPlanSequence", []) or []}
            if refs == {uid} and str(getattr(ds, "DoseSummationType", "")).upper() in PER_PLAN_SUMMATION_TYPES:
                candidates.append((path, ds))
        if not candidates:
            _refuse("no_plan_level_dose")
        if len(candidates) != 1:
            _refuse("ambiguous_plan_level_doses")
        path, ds = candidates[0]
        if (str(getattr(ds, "SOPClassUID", "")) != "1.2.840.10008.5.1.4.1.1.481.2"
                or not str(getattr(ds, "SOPInstanceUID", "") or "")):
            _refuse("dose_identity_invalid")
        if (str(getattr(ds, "DoseType", "")).upper() != "PHYSICAL"
                or str(getattr(ds, "DoseUnits", "")).upper() != "GY"):
            _refuse("dose_not_physical_gy")
        if str(getattr(ds, "FrameOfReferenceUID", "") or "") != frame:
            _refuse("dose_ct_frame_mismatch")
        try:
            geometry = _validated_dose_grid_geometry(ds, path)
        except (ValueError, TypeError, AttributeError):
            _refuse("dose_geometry_qc_failed")
        doses.append((path, ds, geometry))

    weights = [int(i["delivered_fraction_count"]) / int(i["planned_fraction_count"]) for i in items]
    scaled_rx = [w * float(i.get("resolved_prescribed_dose_total_gy") or 0.0) for w, i in zip(weights, items)]
    threshold = float(contract.dose_qc.get("threshold_gy") or 0.0)
    # A 1 uGy tolerance keeps a sum exactly at the threshold (e.g. 50 + 50 Gy)
    # from being refused by binary floating-point error.
    if threshold <= 0 or sum(scaled_rx) > threshold + 1e-6:
        _refuse("composite_prescription_implausible")

    inputs = []
    for item, (uid, plan), plan_path, (dose_path, ds, geometry), weight, rx in zip(
            items, plans, plan_paths, doses, weights, scaled_rx):
        inputs.append({
            "plan_uid_sha256": _uid_hash(uid),
            "plan_path": plan_path.relative_to(root).as_posix(),
            "plan_sha256": _sha256(plan_path),
            "dose_path": dose_path.relative_to(root).as_posix(),
            "dose_sha256": _sha256(dose_path),
            "dose_summation_type": str(ds.DoseSummationType).upper(),
            "planned_fractions": int(item["planned_fraction_count"]),
            "delivered_fractions": int(item["delivered_fraction_count"]),
            "weight": weight,
            "weight_basis": "rtrecord_delivered_fractions / plan_planned_fractions",
            "prescribed_total_gy": float(item.get("resolved_prescribed_dose_total_gy") or 0.0),
            "weighted_prescribed_gy": rx,
            "grid": {"rows": int(geometry["rows"]), "cols": int(geometry["cols"]),
                     "frames": int(geometry["frames"]),
                     "pixel_spacing_mm": [float(v) for v in geometry["spacing"]]},
        })
    evidence = {
        "schema_version": 1,
        "mode": MODE,
        "basis": BASIS,
        "dose_response_eligible": False,
        "course_metadata_sha256": _sha256(metadata),
        "frame_of_reference_uid_sha256": _uid_hash(frame),
        "inputs": inputs,
        "date_relations": _relations(date_sets),
        "equal_signature_pairs": sum(_signature(a) == _signature(b) for k, a in enumerate(items) for b in items[k + 1:]),
        "max_inter_plan_gap_days": max_gap,
        "weighted_prescription_sum_gy": sum(scaled_rx),
        "dose_qc_threshold_gy": threshold,
        "interpolation": INTERPOLATION,
        "output_dose_path": OUTPUT,
        "output_dose_summation_type": "MULTI_PLAN",
        "output_plan_path": OUTPUT_PLAN,
    }
    return CompositeSelection("accepted", tuple(plan_paths), tuple(d[0] for d in doses),
                              tuple(weights), None, evidence)


def build_composite_dose(selection: CompositeSelection):
    """Return the MULTI_PLAN RTDOSE, its evaluation plan sum and a grid record."""
    if not selection.accepted:
        _refuse(selection.reason_code)
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence
    from .organize import _create_summed_plan, _sum_doses_with_resample, _validated_dose_grid_geometry
    plan_sum, plan_datasets, _ = _create_summed_plan(list(selection.plan_paths))
    plan_datasets = plan_datasets or [pydicom.dcmread(path) for path in selection.plan_paths]
    dose, _, source_uids = _sum_doses_with_resample(
        list(selection.dose_paths), plan_sum, plan_datasets,
        dose_weights=list(selection.weights))
    reference = Dataset()
    reference.ReferencedSOPClassUID = str(dose.SOPClassUID)
    reference.ReferencedSOPInstanceUID = str(dose.SOPInstanceUID)
    plan_sum.ReferencedDoseSequence = Sequence([reference])
    dose.SeriesDescription = "RF19 composite course dose"
    dose.DoseComment = "Physical dose sum of delivered plans weighted by RTRECORD delivered/planned fractions."
    values = dose.pixel_array.astype(float) * float(dose.DoseGridScaling)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        _refuse("composite_pixels_invalid")
    geometry = _validated_dose_grid_geometry(dose, Path(OUTPUT))
    grid = {
        "rows": int(geometry["rows"]), "cols": int(geometry["cols"]), "frames": int(geometry["frames"]),
        "pixel_spacing_mm": [float(v) for v in geometry["spacing"]],
        "origin_mm": [float(v) for v in geometry["origin"]],
        "orientation": [float(v) for v in dose.ImageOrientationPatient],
        "frame_offsets_mm_first_last": [float(geometry["offsets"][0]), float(geometry["offsets"][-1])],
        "dose_grid_scaling": float(dose.DoseGridScaling),
        "max_dose_gy": float(values.max()) if values.size else 0.0,
        "source_dose_count": len(source_uids),
    }
    return dose, plan_sum, grid


def load_composite_sidecar(course_dir: Path | str) -> CompositeSelection:
    """Revalidate inputs and output bytes; an existing sidecar is no waiver."""
    root = Path(course_dir).resolve()
    try:
        path = _local(root, root / SIDECAR)
        if not path.exists():
            return CompositeSelection("sidecar_missing")
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict) or not str(payload.get("code_revision") or "").strip():
            _refuse("sidecar_invalid")
        selection = select_composite_course_dose(root)
        if not selection.accepted:
            return selection
        recorded = {k: v for k, v in payload.items()
                    if k not in {"code_revision", "output_dose_sha256", "output_plan_sha256", "output_grid"}}
        if json.dumps(recorded, sort_keys=True) != json.dumps(selection.evidence, sort_keys=True):
            _refuse("sidecar_stale_or_mismatched")
        output = _local(root, root / OUTPUT)
        if not output.is_file() or _sha256(output) != payload.get("output_dose_sha256"):
            _refuse("composite_output_stale_or_missing")
        plan_output = _local(root, root / OUTPUT_PLAN)
        if not plan_output.is_file() or _sha256(plan_output) != payload.get("output_plan_sha256"):
            _refuse("composite_output_stale_or_missing")
        plan_refs = {str(r.ReferencedSOPInstanceUID) for r in
                     getattr(pydicom.dcmread(plan_output, stop_before_pixels=True), "ReferencedRTPlanSequence", []) or []}
        if {_uid_hash(u) for u in plan_refs} != {i["plan_uid_sha256"] for i in selection.evidence["inputs"]}:
            _refuse("composite_output_stale_or_missing")
        dose = pydicom.dcmread(output, stop_before_pixels=True)
        refs = {str(r.ReferencedSOPInstanceUID) for r in getattr(dose, "ReferencedRTPlanSequence", []) or []}
        if (str(getattr(dose, "DoseSummationType", "")).upper() != "MULTI_PLAN"
                or {_uid_hash(u) for u in refs} != {i["plan_uid_sha256"] for i in selection.evidence["inputs"]}
                or _uid_hash(getattr(dose, "FrameOfReferenceUID", "")) != selection.evidence["frame_of_reference_uid_sha256"]):
            _refuse("composite_output_stale_or_missing")
        return CompositeSelection(selection.reason_code, selection.plan_paths, selection.dose_paths,
                                  selection.weights, plan_output, selection.evidence)
    except PlanningDoseError as exc:
        return CompositeSelection(str(exc))
    except (OSError, ValueError, TypeError, KeyError):
        return CompositeSelection("sidecar_invalid")
