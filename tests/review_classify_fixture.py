"""Deterministic synthetic fixtures copied from the WS270 classifier reproducer."""
from __future__ import annotations

import hashlib

import pydicom

def uid(name: str) -> str:
    return "2.25." + str(int(hashlib.sha256(name.encode()).hexdigest()[:30], 16))


FRAME, STUDY, STRUCT = uid("frame"), uid("study"), uid("struct")

def plan(root, name, *, rx=30.0, fx=10, date="20240101", time="080000", target="T-a",
         iso=(0.0, 0.0, 0.0), gantry=0.0, frame=FRAME, struct=STRUCT, intent="CURATIVE"):
    from synthetic_rt_fixtures import make_plan
    path = make_plan(root / f"{name}.dcm", uid(root.name + name), struct_uid=struct, study_uid=STUDY,
                     frame_uid=frame, date=date, rx_gy=rx, fractions=fx)
    ds = pydicom.dcmread(path)
    ds.RTPlanTime = time
    ds.PlanIntent = intent
    ds.SeriesInstanceUID = uid(root.name + name + "series")
    ref = ds.DoseReferenceSequence[0]
    for beam_ref in ds.FractionGroupSequence[0].ReferencedBeamSequence:
        if target is None:
            if "ReferencedDoseReferenceUID" in beam_ref:
                del beam_ref.ReferencedDoseReferenceUID
        else:
            beam_ref.ReferencedDoseReferenceUID = uid(target)
    if target is None:
        del ref.DoseReferenceUID
    else:
        ref.DoseReferenceUID = uid(target)
    cp = ds.BeamSequence[0].ControlPointSequence
    cp[0].IsocenterPosition = list(iso)
    for point in cp:
        point.GantryAngle = gantry
    ds.save_as(path, enforce_file_format=True)
    return path


def dose(root, name, plan_path, *, gy=30.0, frame=FRAME, summation="PLAN"):
    from synthetic_rt_fixtures import make_dose
    plan_uid = str(pydicom.dcmread(plan_path, stop_before_pixels=True).SOPInstanceUID)
    path = make_dose(root / f"{name}.dcm", uid(root.name + name), plan_uid=plan_uid, study_uid=STUDY,
                     frame_uid=frame, with_pixels=True, dose_gy=gy, summation_type=summation)
    ds = pydicom.dcmread(path)
    ds.SeriesInstanceUID = uid(root.name + name + "series")
    ds.save_as(path, enforce_file_format=True)
    return path


def record(root, name, plan_path, fraction, date):
    from synthetic_rt_fixtures import make_record
    plan_uid = str(pydicom.dcmread(plan_path, stop_before_pixels=True).SOPInstanceUID)
    path = make_record(root / f"{name}.dcm", plan_uid, fraction_number=fraction, date=date,
                       sop_uid=uid(root.name + name))
    ds = pydicom.dcmread(path)
    ds.StudyInstanceUID = uid(root.name + name + "study")
    ds.SeriesInstanceUID = uid(root.name + name + "series")
    ds.save_as(path, enforce_file_format=True)
    return path


def _pair(root, a, b, records=None):
    pa, pb = plan(root, "pA", **a), plan(root, "pB", **b)
    da, db = dose(root, "dA", pa), dose(root, "dB", pb)
    recs = []
    for plan_path, fractions in (records or {}).items():
        target = pa if plan_path == "pA" else pb
        for index, (fraction, day) in enumerate(fractions):
            recs.append(record(root, f"r{plan_path}{index}", target, fraction, day))
    return dict(plans=[pa, pb], doses=[da, db], records=recs)


def c_split_iso(root):
    # The WS269 target shape: same Rx/fx/FoR/RS/date, distinct TARGET UIDs, distinct isocentres.
    return _pair(root, dict(target="T-a", iso=(0, 0, 0)), dict(target="T-b", iso=(100, 0, 0)))


def c_concurrent_same_uid_diff_iso_same_date(root):
    # Two plans contributing to ONE target reference (DICOM: a DoseReferenceUID links the same
    # entity across plans), different isocentres and beams, same date, no RTRECORD.
    return _pair(root, dict(target="T-a", iso=(0, 0, 0), gantry=0.0),
                 dict(target="T-a", iso=(0, 0, 150), gantry=180.0))

