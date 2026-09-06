"""Source-complete plan accounting, independent of imaging-course eligibility.

A delivered plan without an imaging authority is retained as a non-measurable
history record. Similar prescriptions are never enough to transfer dose or to
claim a separate clinical episode. Source identities are not rewritten.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import os
import shutil
import tempfile

import pydicom

from .course_contract import _record_delivery_session_evidence, _record_delivery_session_key
from .organize_ledger import _write_json_atomic
from .utils import _scoped_walk

SCHEMA = "rtpipeline-source-plan-dispositions-v1"
RELATIVE_PATH = Path("_COURSES/source_plan_dispositions.json")


def _text(ds, key):
    return str(getattr(ds, key, "") or "").strip()


def _seq(ds, key):
    return list(getattr(ds, key, []) or [])


def _number(ds, key):
    try:
        value = float(getattr(ds, key))
        return value if math.isfinite(value) else None
    except (AttributeError, TypeError, ValueError):
        return None


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_scope_fingerprint(root, patient_ids=None):
    """Notice added/deleted/replaced files before reusing an organize manifest.

    Content hashes below independently bind the plan and record bytes. This
    directory fingerprint prevents a new source plan from escaping that set.
    """
    root = Path(root)
    # Preserve the CLI's empty/missing-input behavior for downstream probes,
    # without ever treating a missing input as a reusable source history.
    try:
        root.stat()
    except FileNotFoundError:
        return None
    def fail(error):
        raise error
    entries = []
    for base, _, files in _scoped_walk(root, patient_ids, onerror=fail):
        for name in files:
            path = Path(base) / name
            st = path.stat()
            entries.append((str(path.relative_to(root)), st.st_size, st.st_mtime_ns))
    return hashlib.sha256(json.dumps(sorted(entries)).encode()).hexdigest()


def source_dispositions_match_source(payload, root):
    try:
        validate_source_plan_dispositions(payload)
        if payload.get("source_root") != str(Path(root).resolve()):
            return False
        if not payload.get("source_scope_fingerprint"):
            return False
        if source_scope_fingerprint(root, payload.get("discovery_scope_patient_ids")) != payload.get("source_scope_fingerprint"):
            return False
        if payload.get("record_read_errors") or payload.get("unresolved_record_plans"):
            return False
        for row in payload["plans"]:
            if any(_sha(p) != row["source_sha256"] for p in row["source_paths"]):
                return False
            if any(_sha(r["source_path"]) != r["sha256"] for r in row["records"]):
                return False
        return payload.get("course_publication_status") == "complete" and payload.get("source_archive_status") == "complete"
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _archive_detached_delivery(output_root, payload):
    root = Path(output_root)
    for row in payload["plans"]:
        if row["disposition_type"] not in {"non_measurable_delivery", "recorded_beam_variant"}:
            continue
        identity = hashlib.sha256(json.dumps([row["patient"], row["plan_uid"]]).encode()).hexdigest()
        directory = root / "_COURSES" / "non_measurable_delivery" / identity
        directory.mkdir(parents=True, exist_ok=True)
        objects = [(row, row["source_paths"][0], row["source_sha256"])]
        objects.extend((r, r["source_path"], r["sha256"]) for r in row["records"])
        for entry, source, expected in objects:
            target = directory / (expected + ".dcm")
            if not target.is_file() or _sha(target) != expected:
                fd, name = tempfile.mkstemp(dir=directory, prefix=".source-", suffix=".tmp")
                os.close(fd)
                tmp = Path(name)
                try:
                    shutil.copyfile(source, tmp)
                    if _sha(tmp) != expected:
                        raise ValueError("source delivery bytes changed during publication")
                    os.replace(tmp, target)
                finally:
                    tmp.unlink(missing_ok=True)
            entry["archived_path"] = target.relative_to(root).as_posix()


def _target_references(ds):
    return [{"number": _text(x, "DoseReferenceNumber"),
             "uid": _text(x, "DoseReferenceUID"),
             "prescription_gy": _number(x, "TargetPrescriptionDose")}
            for x in _seq(ds, "DoseReferenceSequence")
            if _text(x, "DoseReferenceType") == "TARGET"]


def _beam_groups(ds):
    """Candidate screen only. Totals cannot establish physical equivalence."""
    refs = {_text(b, "ReferencedBeamNumber"): b
            for f in _seq(ds, "FractionGroupSequence")
            for b in _seq(f, "ReferencedBeamSequence")}
    groups = defaultdict(list)
    for b in _seq(ds, "BeamSequence"):
        if _text(b, "TreatmentDeliveryType") != "TREATMENT":
            continue
        cps = _seq(b, "ControlPointSequence")
        if not cps:
            return {}
        key = tuple(_number(cps[0], k) for k in (
            "GantryAngle", "NominalBeamEnergy", "BeamLimitingDeviceAngle", "PatientSupportAngle"))
        ref = refs.get(_text(b, "BeamNumber"))
        groups[key].append({"number": _text(b, "BeamNumber"), "name": _text(b, "BeamName"),
                            "machine": _text(b, "TreatmentMachineName"),
                            "control_points": len(cps), "meterset": _number(ref, "BeamMeterset"),
                            "beam_dose_gy": _number(ref, "BeamDose")})
    return dict(groups)


def _canonical_beams(ds):
    """Strict identical-beam variant proof, allowing only name/number changes.

    Do not normalize machine, leaf positions, patient coordinates, control point
    trajectories or weights. Split/converted beams that fail this proof remain
    candidates, even when aggregate monitor units and prescription agree.
    """
    result = []
    refs = {_text(b, "ReferencedBeamNumber"): b
            for f in _seq(ds, "FractionGroupSequence")
            for b in _seq(f, "ReferencedBeamSequence")}
    for b in _seq(ds, "BeamSequence"):
        if _text(b, "TreatmentDeliveryType") != "TREATMENT":
            continue
        if not _text(b, "TreatmentMachineName") or not _seq(b, "ControlPointSequence"):
            return None
        ref = refs.get(_text(b, "BeamNumber"))
        mu, dose = _number(ref, "BeamMeterset"), _number(ref, "BeamDose")
        if mu is None or mu <= 0 or dose is None or dose <= 0:
            return None
        # Dataset JSON is used only in memory for equality, never published.
        value = b.to_json_dict()
        for key in ("300A00C0", "300A00C2", "300A00C3"):
            value.pop(key, None)
        result.append(json.dumps([value, mu, dose], sort_keys=True))
    return sorted(result) or None


def compare_plan_variant(delivery, planning):
    """Return evidence for a candidate, never infer identity from a label/date."""
    if _text(delivery, "PatientID") != _text(planning, "PatientID"):
        return None
    if _text(planning, "ApprovalStatus") != "APPROVED":
        return None
    dr, pr = _target_references(delivery), _target_references(planning)
    if not dr or any(not r["uid"] or not r["prescription_gy"] for r in dr) or dr != pr:
        return None
    fractions = lambda ds: [_number(f, "NumberOfFractionsPlanned") for f in _seq(ds, "FractionGroupSequence")]
    if not fractions(delivery) or fractions(delivery) != fractions(planning):
        return None
    dg, pg = _beam_groups(delivery), _beam_groups(planning)
    if not dg or set(dg) != set(pg):
        return None
    evidence = []
    for key in dg:
        a, b = dg[key], pg[key]
        if any(v is None for v in key):
            return None
        for field in ("meterset", "beam_dose_gy", "control_points"):
            if any(x[field] is None for x in a + b):
                return None
            if not math.isclose(sum(x[field] for x in a), sum(x[field] for x in b), rel_tol=1e-10, abs_tol=1e-8):
                return None
        evidence.append({"geometry_energy": list(key), "delivery_beams": a, "planning_beams": b})
    dc, pc = _canonical_beams(delivery), _canonical_beams(planning)
    exact = dc is not None and dc == pc
    return {"planning_plan_uid": _text(planning, "SOPInstanceUID"),
            "status": "IDENTICAL_BEAM_VARIANT" if exact else "CANDIDATE_ONLY_NOT_DOSE_AUTHORITY",
            "same_machine": {x["machine"] for g in dg.values() for x in g} == {x["machine"] for g in pg.values() for x in g},
            "strict_beam_equivalence": exact, "shared_target_references": dr, "beam_comparison": evidence}


def build_source_plan_dispositions(plans, record_index, accepted_courses, decline_reasons=None):
    """One disposition per discovered patient/plan identity, including failures."""
    decline_reasons = decline_reasons or {}
    accepted = {(str(pid), item.plan.sop_instance_uid): str(key)
                for (pid, key), items in accepted_courses.items() for item in items}
    datasets, errors, source_paths = {}, {}, defaultdict(list)
    for p in plans:
        identity = (str(p.patient_id), p.sop_instance_uid)
        source_paths[identity].append(Path(p.path))
        try:
            ds = pydicom.dcmread(p.path, stop_before_pixels=True)
            if _text(ds, "SOPInstanceUID") != p.sop_instance_uid or _text(ds, "PatientID") != str(p.patient_id):
                raise ValueError("source plan identity differs from discovery")
            if identity in datasets and _sha(p.path) != _sha(source_paths[identity][0]):
                raise ValueError("conflicting source bytes share one plan identity")
            datasets[identity] = ds
        except Exception as exc:
            errors[identity] = type(exc).__name__ + ": " + str(exc)
    records = defaultdict(dict)
    record_errors = []
    for patient, paths in record_index.items():
        for path in dict.fromkeys(paths):
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
                uid = _text(ds, "SOPInstanceUID")
                if _text(ds, "PatientID") != str(patient) or not uid:
                    raise ValueError("delivery record identity differs from discovery")
                valid, nested, reason = _record_delivery_session_evidence(ds)
                row = {"sop_instance_uid": uid, "source_path": str(path), "sha256": _sha(path),
                       "validated_delivery": valid,
                       "session_key": list(_record_delivery_session_key(ds, uid, nested_fraction_number=nested)),
                       "evidence": reason}
                for ref in _seq(ds, "ReferencedRTPlanSequence"):
                    key = (str(patient), _text(ref, "ReferencedSOPInstanceUID"))
                    if uid in records[key] and records[key][uid]["sha256"] != row["sha256"]:
                        raise ValueError("conflicting delivery bytes share one record identity")
                    records[key][uid] = row
            except Exception as exc:
                record_errors.append({"patient": str(patient), "path": str(path), "reason": type(exc).__name__ + ": " + str(exc)})
    rows = []
    for identity, paths in sorted(source_paths.items()):
        patient, uid = identity
        ds = datasets.get(identity)
        recs = sorted(records.get(identity, {}).values(), key=lambda r: r["sop_instance_uid"])
        valid = [r for r in recs if r["validated_delivery"]]
        course_key = accepted.get(identity)
        row = {"patient": patient, "plan_uid": uid, "source_paths": sorted(set(map(str, paths))),
               "source_sha256": _sha(paths[0]) if paths[0].is_file() else None,
               "approval_status": _text(ds, "ApprovalStatus") or "ABSENT",
               "plan_geometry": _text(ds, "RTPlanGeometry"), "plan_intent": _text(ds, "PlanIntent"),
               "target_prescriptions": _target_references(ds),
               "planned_fractions": [_number(f, "NumberOfFractionsPlanned") for f in _seq(ds, "FractionGroupSequence")],
               "records": recs, "validated_record_count": len(valid),
               "validated_session_count": len({tuple(r["session_key"]) for r in valid}),
               "course_key": course_key, "clinical_exclusion": False,
               "additional_clinical_course_count": None, "authoritative_plan_uid": None,
               "candidate_planning_variants": [], "non_measurements": {}}
        mechanical = decline_reasons.get(identity)
        if not mechanical and course_key is None:
            mechanical = "NO_RTSTRUCT_REFERENCE" if not _seq(ds, "ReferencedStructureSetSequence") else "NO_TARGET_BEARING_AUTHORITATIVE_RTSTRUCT"
        row["mechanical_reason"] = mechanical
        if identity in errors:
            state, reason = "technical_hold", "SOURCE_PLAN_UNREADABLE_OR_IDENTITY_CONFLICT"
            row["technical_error"] = errors[identity]
        elif course_key is not None:
            state, reason = "course_member", "TARGET_BEARING_COURSE_MEMBER"
        elif valid:
            state, reason = "non_measurable_delivery", "DELIVERED_PLAN_WITHOUT_IMAGING_AUTHORITY"
            candidates = []
            for other, planning in datasets.items():
                if other[0] == patient and other in accepted and other not in errors:
                    match = compare_plan_variant(ds, planning)
                    if match:
                        match["course_key"] = accepted[other]
                        planning_records = records.get(other, {})
                        match["delivery_record_identities_already_represented"] = bool(valid) and all(
                            r["sop_instance_uid"] in planning_records
                            and planning_records[r["sop_instance_uid"]]["sha256"] == r["sha256"]
                            for r in valid
                        )
                        candidates.append(match)
            row["candidate_planning_variants"] = candidates
            exact = [x for x in candidates if x["strict_beam_equivalence"]
                     and x["delivery_record_identities_already_represented"]]
            if len(exact) == 1:
                state, reason = "recorded_beam_variant", "IDENTICAL_BEAM_VARIANT_NOT_ADDITIONAL_COURSE"
                row["authoritative_plan_uid"] = exact[0]["planning_plan_uid"]
                row["represented_course_key"] = exact[0]["course_key"]
                row["additional_clinical_course_count"] = 0
            elif candidates:
                reason = "DELIVERED_PLAN_VARIANT_IDENTITY_UNRESOLVED"
            # No imaging authority is borrowed, including for a confirmed variant.
            row["non_measurements"] = {stage: {"status": "not_measurable", "reason": "NO_PLAN_BOUND_STRUCTURE_AND_DOSE_AUTHORITY"}
                                       for stage in ("dvh", "ct_radiomics", "dose_radiomics", "dose_accumulation", "robustness")}
        elif _text(ds, "PlanIntent") == "VERIFICATION":
            state, reason = "excluded", "VERIFICATION_OR_QA_PLAN"
            row["clinical_exclusion"] = True
        elif _text(ds, "ApprovalStatus") == "REJECTED":
            state, reason = "excluded", "REJECTED_PLAN_WITHOUT_VALIDATED_DELIVERY"
        elif _text(ds, "ApprovalStatus") != "APPROVED":
            state, reason = "excluded", "NONAPPROVED_PLAN_WITHOUT_IMAGING_AUTHORITY_OR_VALIDATED_DELIVERY"
        else:
            state, reason = "excluded", "NO_TARGET_BEARING_AUTHORITATIVE_RTSTRUCT"
        row.update(disposition_type=state, reason_code=reason,
                   prescription_accounting="source_intent_only_not_additive" if course_key is None else "course_contract",
                   delivered_dose_gy=None, structure_set_path=None, dose_grid_path=None)
        rows.append(row)
    unresolved = [{"patient": pid, "plan_uid": uid, "records": list(r.values())}
                  for (pid, uid), r in sorted(records.items()) if (pid, uid) not in source_paths]
    return summarize_dispositions(rows, record_errors=record_errors, unresolved_record_plans=unresolved)


def summarize_dispositions(rows, *, record_errors=(), unresolved_record_plans=()):
    identities = [(r["patient"], r["plan_uid"]) for r in rows]
    if len(set(identities)) != len(rows):
        raise ValueError("duplicate source plan disposition")
    if any(not r.get("reason_code") or not r.get("disposition_type") for r in rows):
        raise ValueError("every source plan requires a typed disposition")
    detached = [r for r in rows if r["disposition_type"] in ("non_measurable_delivery", "recorded_beam_variant")]
    return {"schema": SCHEMA, "status": "complete_with_holds" if detached or record_errors or unresolved_record_plans or any(r["disposition_type"] == "technical_hold" for r in rows) else "complete",
            "source_plan_count": len(rows), "disposition_counts": dict(Counter(r["disposition_type"] for r in rows)),
            "non_course_delivery_plan_count": len(detached),
            "non_course_validated_record_count": len({(r["patient"], x["sop_instance_uid"]) for r in detached for x in r["records"] if x["validated_delivery"]}),
            "clinical_course_count_from_detached_plans": None,
            "plans": rows, "record_read_errors": list(record_errors), "unresolved_record_plans": list(unresolved_record_plans)}


def validate_source_plan_dispositions(payload):
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("unsupported source plan disposition schema")
    rows = payload.get("plans")
    if not isinstance(rows, list):
        raise ValueError("source plan dispositions require a plans list")
    allowed = {"course_member", "excluded", "technical_hold", "non_measurable_delivery", "recorded_beam_variant"}
    for row in rows:
        if not row.get("patient") or not row.get("plan_uid") or row.get("disposition_type") not in allowed:
            raise ValueError("invalid source plan identity or disposition")
        if row["disposition_type"] in {"non_measurable_delivery", "recorded_beam_variant"}:
            if not row.get("validated_record_count") or row.get("clinical_exclusion"):
                raise ValueError("non-measurable delivery must retain validated treatment evidence")
            if any(row.get(k) is not None for k in ("structure_set_path", "dose_grid_path", "delivered_dose_gy")):
                raise ValueError("detached delivery cannot borrow structure or dose authority")
            for stage in ("dvh", "ct_radiomics", "dose_radiomics", "dose_accumulation", "robustness"):
                outcome = row.get("non_measurements", {}).get(stage, {})
                if outcome.get("status") != "not_measurable" or not outcome.get("reason"):
                    raise ValueError("detached delivery lacks an explicit stage non-measurement")
        valid = [r for r in row["records"] if r["validated_delivery"]]
        if row["validated_record_count"] != len(valid) or row["validated_session_count"] != len({tuple(r["session_key"]) for r in valid}):
            raise ValueError("source plan delivery counts do not reconcile")
        if row["disposition_type"] == "recorded_beam_variant":
            proof = [x for x in row["candidate_planning_variants"] if x["strict_beam_equivalence"] and x["delivery_record_identities_already_represented"]]
            if len(proof) != 1 or proof[0]["planning_plan_uid"] != row["authoritative_plan_uid"]:
                raise ValueError("beam variant lacks unique beam and record identity proof")
    rebuilt = summarize_dispositions(rows, record_errors=payload.get("record_read_errors", []),
                                     unresolved_record_plans=payload.get("unresolved_record_plans", []))
    for field in ("source_plan_count", "disposition_counts", "non_course_delivery_plan_count", "non_course_validated_record_count", "status"):
        if payload.get(field) != rebuilt[field]:
            raise ValueError("source plan disposition counts/status do not reconcile: " + field)
    return payload


def write_source_plan_dispositions(output_root, payload):
    """Fail loudly on an unwritable register, never silently lose a disposition."""
    validate_source_plan_dispositions(payload)
    # Publish evidence first. A failed archive copy leaves an explicit pending
    # receipt rather than erasing the known delivery history or claiming success.
    payload["source_archive_status"] = "pending"
    _write_json_atomic(Path(output_root) / RELATIVE_PATH, payload)
    _archive_detached_delivery(output_root, payload)
    payload["source_archive_status"] = "complete"
    _write_json_atomic(Path(output_root) / RELATIVE_PATH, payload)
    return payload
