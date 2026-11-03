"""Deterministic synthetic five-course fixture copied from the WS270 review harness."""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian


def uid(name: str) -> str:
    return "2.25." + str(int(hashlib.sha256(name.encode()).hexdigest()[:30], 16))


def _save(ds, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)
    return path


def ct_series(folder, key, patient, study, frame, *, slices=12, size=8, spacing=5.0, origin=(-20.0, -20.0, -10.0)):
    series = uid(key + "ct-series")
    refs = []
    for index in range(slices):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = CTImageStorage
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = uid(f"{key}ct{index}")
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.PatientID = patient
        ds.PatientName = "SYNTH^" + patient
        ds.Modality = "CT"
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frame
        ds.StudyDate = ds.SeriesDate = "20240101"
        ds.SeriesDescription = "Synthetic planning CT"
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        ds.ImagePositionPatient = [origin[0], origin[1], origin[2] + index * spacing]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [spacing, spacing]
        ds.SliceThickness = spacing
        ds.InstanceNumber = index + 1
        ds.Rows = ds.Columns = size
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleIntercept, ds.RescaleSlope = -1000, 1
        ds.PixelData = np.full((size, size), 1000, dtype=np.int16).tobytes()
        _save(ds, folder / f"ct{index}.dcm")
        ref = Dataset()
        ref.ReferencedSOPClassUID = CTImageStorage
        ref.ReferencedSOPInstanceUID = ds.SOPInstanceUID
        refs.append((ref, origin[2] + index * spacing))
    return series, refs


def structure(folder, key, patient, study, frame, series, refs, roi_names=("BODY", "PTV1", "PTV2")):
    from synthetic_rt_fixtures import make_struct
    path = make_struct(folder / "rs.dcm", uid(key + "rs"), study_uid=study, frame_uid=frame,
                       roi_names=roi_names, patient_id=patient)
    ds = pydicom.dcmread(path)
    ds.SeriesInstanceUID = uid(key + "rs-series")
    ref_series = Dataset()
    ref_series.SeriesInstanceUID = series
    ref_series.ContourImageSequence = Sequence([ref for ref, _ in refs])
    ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence = Sequence([ref_series])
    import copy
    for roi in ds.ROIContourSequence:
        template = roi.ContourSequence[0]
        contours = []
        for ref, z in refs[1:-1]:
            item = copy.deepcopy(template)
            points = [-12.0, -12.0, z, 12.0, -12.0, z, 12.0, 12.0, z, -12.0, 12.0, z]
            if roi.ReferencedROINumber == 1:
                points = [-18.0, -18.0, z, 18.0, -18.0, z, 18.0, 18.0, z, -18.0, 18.0, z]
            item.ContourData = points
            item.ContourImageSequence = Sequence([copy.deepcopy(ref)])
            contours.append(item)
        roi.ContourSequence = Sequence(contours)
    return _save(ds, path)


def plan(folder, key, name, patient, study, frame, struct_uid, *, rx, fx, target, roi, date="20240101",
         time="080000", iso=(0.0, 0.0, 0.0), gantry=0.0):
    from synthetic_rt_fixtures import make_plan
    path = make_plan(folder / f"{name}.dcm", uid(key + name), struct_uid=struct_uid, study_uid=study,
                     frame_uid=frame, date=date, rx_gy=rx, fractions=fx, patient_id=patient, label=name)
    ds = pydicom.dcmread(path)
    ds.SeriesInstanceUID = uid(key + name + "series")
    ds.RTPlanTime = time
    ref = ds.DoseReferenceSequence[0]
    ref.DoseReferenceUID = uid(target)
    ref.DoseReferenceStructureType = "VOLUME"
    ref.ReferencedROINumber = roi
    for beam_ref in ds.FractionGroupSequence[0].ReferencedBeamSequence:
        beam_ref.ReferencedDoseReferenceUID = uid(target)
    cps = ds.BeamSequence[0].ControlPointSequence
    cps[0].IsocenterPosition = list(iso)
    for cp in cps:
        cp.GantryAngle = gantry
    return _save(ds, path)


def dose(folder, key, name, patient, study, frame, plan_path, *, values, origin, spacing, summation="PLAN",
         plan_uids=None):
    """Write an RTDOSE whose voxel (k, j, i) holds values[k, j, i] Gy (scaling 1e-4)."""
    from synthetic_rt_fixtures import make_dose
    plan_uid = str(pydicom.dcmread(plan_path, stop_before_pixels=True).SOPInstanceUID) if plan_path else None
    frames, rows, cols = values.shape
    path = make_dose(folder / f"{name}.dcm", uid(key + name), plan_uid=plan_uid, study_uid=study,
                     frame_uid=frame, with_pixels=True, shape=values.shape, patient_id=patient,
                     summation_type=summation)
    ds = pydicom.dcmread(path)
    ds.SeriesInstanceUID = uid(key + name + "series")
    if plan_uids:
        items = []
        for value in plan_uids:
            item = Dataset()
            item.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.5"
            item.ReferencedSOPInstanceUID = value
            items.append(item)
        ds.ReferencedRTPlanSequence = Sequence(items)
    ds.ImagePositionPatient = list(origin)
    ds.PixelSpacing = [spacing[1], spacing[2]]
    ds.SliceThickness = spacing[0]
    ds.GridFrameOffsetVector = [float(spacing[0] * index) for index in range(frames)]
    ds.DoseGridScaling = 1e-4
    ds.PixelData = np.rint(values * 1e4).astype(np.uint16).tobytes()
    return _save(ds, path)


def record(folder, key, name, patient, plan_path, fraction, date):
    from synthetic_rt_fixtures import make_record
    plan_uid = str(pydicom.dcmread(plan_path, stop_before_pixels=True).SOPInstanceUID)
    path = make_record(folder / f"{name}.dcm", plan_uid, fraction_number=fraction, date=date,
                       sop_uid=uid(key + name), patient_id=patient)
    ds = pydicom.dcmread(path)
    ds.StudyInstanceUID = uid(key + name + "study")
    ds.SeriesInstanceUID = uid(key + name + "series")
    return _save(ds, path)


def linear(shape, origin, spacing, a, b):
    """Dose a + b . (x, y, z) sampled at voxel centres of an axis-aligned grid (Gy)."""
    k, j, i = np.indices(shape, dtype=float)
    x = origin[0] + i * spacing[2]
    y = origin[1] + j * spacing[1]
    z = origin[2] + k * spacing[0]
    return a + b[0] * x + b[1] * y + b[2] * z


def course(root, patient, label, plans_spec, *, records=None, variant=None):
    key = f"{patient}-{label}"
    folder = root / patient / label
    study, frame = uid(key + "study"), uid(key + "frame")
    series, refs = ct_series(folder, key, patient, study, frame)
    rs = structure(folder, key, patient, study, frame, series, refs)
    struct_uid = str(pydicom.dcmread(rs, stop_before_pixels=True).SOPInstanceUID)
    written = {}
    for spec in plans_spec:
        p = plan(folder, key, spec["name"], patient, study, frame, struct_uid, rx=spec["rx"], fx=spec["fx"],
                 target=key + spec["target"], roi=spec["roi"], date=spec.get("date", "20240101"),
                 time=spec.get("time", "080000"), iso=spec.get("iso", (0.0, 0.0, 0.0)),
                 gantry=spec.get("gantry", 0.0))
        grid = spec["grid"]
        values = linear(grid["shape"], grid["origin"], grid["spacing"], grid["a"], grid["b"])
        d = dose(folder, key, "d" + spec["name"][1:], patient, study, frame, p,
                 values=values, origin=grid["origin"], spacing=grid["spacing"])
        written[spec["name"]] = (p, d)
    for plan_name, sessions in (records or {}).items():
        for index, (fraction, day) in enumerate(sessions):
            record(folder, key, f"rec{plan_name}{index}", patient, written[plan_name][0], fraction, day)
    return dict(folder=folder, study=study, frame=frame, struct_uid=struct_uid, written=written, key=key)


GRID_A = dict(shape=(4, 6, 6), origin=(-15.0, -15.0, -5.0), spacing=(5.0, 5.0, 5.0), a=1.0, b=(0.02, 0.01, 0.03))
GRID_B = dict(shape=(3, 4, 4), origin=(-12.5, -12.5, -2.5), spacing=(7.5, 7.5, 7.5), a=2.0, b=(-0.01, 0.02, 0.01))


def build(root: Path, variant: str = "none") -> dict:
    info = {}
    info["A"] = course(root, "SYNTH_A", "c0", [dict(name="pA", rx=6.0, fx=3, target="T", roi=2, grid=GRID_A)],
                       records={"pA": [(1, "20240102"), (2, "20240103"), (3, "20240104")]})
    info["B1"] = course(root, "SYNTH_B", "c1", [
        dict(name="pA", rx=50.0, fx=25, target="T1", roi=2, grid=GRID_A),
        dict(name="pC", rx=16.0, fx=8, target="T2", roi=3, date="20240201", grid=GRID_B)])
    info["B2"] = course(root, "SYNTH_B", "c2", [
        dict(name="pA", rx=30.0, fx=10, target="T1", roi=2, grid=GRID_A),
        dict(name="pB", rx=30.0, fx=10, target="T1", roi=2, time="090000", gantry=10.0, grid=GRID_A)])
    info["C"] = course(root, "SYNTH_C", "c0", [
        dict(name="pA", rx=30.0, fx=10, target="T1", roi=2, grid=GRID_A),
        dict(name="pB", rx=30.0, fx=10, target="T2", roi=3, iso=(100.0, 0.0, 0.0), grid=GRID_B)])
    info["D"] = course(root, "SYNTH_D", "c0", [
        dict(name="pA", rx=30.0, fx=10, target="T1", roi=2, grid=GRID_A),
        dict(name="pB", rx=30.0, fx=10, target="T2", roi=3, iso=(100.0, 0.0, 0.0), gantry=90.0, grid=GRID_B)])
    d = info["D"]
    folder, key = d["folder"], d["key"]
    pa, pb = d["written"]["pA"][0], d["written"]["pB"][0]
    if variant == "frame":  # dose B on another frame of reference
        ds = pydicom.dcmread(d["written"]["pB"][1]); ds.FrameOfReferenceUID = uid("other-frame"); _save(ds, d["written"]["pB"][1])
    elif variant == "struct":  # plan B references another RTSTRUCT (same CT) present in the course
        other = pydicom.dcmread(folder / "rs.dcm"); other.SOPInstanceUID = uid(key + "rs2")
        other.file_meta.MediaStorageSOPInstanceUID = other.SOPInstanceUID; _save(other, folder / "rs2.dcm")
        ds = pydicom.dcmread(pb); ds.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID = other.SOPInstanceUID; _save(ds, pb)
    elif variant == "ct":  # plan B's RTSTRUCT cites a second CT series under the same frame
        series2, refs2 = ct_series(folder / "ct2", key + "ct2", "SYNTH_D", d["study"], d["frame"])
        other = pydicom.dcmread(folder / "rs.dcm"); other.SOPInstanceUID = uid(key + "rs3")
        other.file_meta.MediaStorageSOPInstanceUID = other.SOPInstanceUID
        rs_series = other.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0]
        rs_series.SeriesInstanceUID = series2
        rs_series.ContourImageSequence = Sequence([ref for ref, _ in refs2])
        for roi in other.ROIContourSequence:
            for item, (ref, _z) in zip(roi.ContourSequence, refs2[1:-1]):
                item.ContourImageSequence = Sequence([ref])
        _save(other, folder / "rs3.dcm")
        ds = pydicom.dcmread(pb); ds.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID = other.SOPInstanceUID; _save(ds, pb)
    elif variant == "multiplan":  # a TPS MULTI_PLAN dose of both plans is present
        values = linear(GRID_A["shape"], GRID_A["origin"], GRID_A["spacing"], 3.0, (0.0, 0.0, 0.0))
        dose(folder, key, "dMULTI", "SYNTH_D", d["study"], d["frame"], None, values=values,
             origin=GRID_A["origin"], spacing=GRID_A["spacing"], summation="MULTI_PLAN",
             plan_uids=[str(pydicom.dcmread(pa).SOPInstanceUID), str(pydicom.dcmread(pb).SOPInstanceUID)])
    elif variant == "third_approved":  # a third approved plan of the course (with a dose) is left undeclared
        p3 = plan(folder, key, "pE", "SYNTH_D", d["study"], d["frame"], d["struct_uid"], rx=10.0, fx=5,
                  target=key + "T3", roi=1, date="20240301")
        dose(folder, key, "dE", "SYNTH_D", d["study"], d["frame"], p3, values=linear(GRID_A["shape"], GRID_A["origin"],
             GRID_A["spacing"], 0.5, (0, 0, 0)), origin=GRID_A["origin"], spacing=GRID_A["spacing"])
    elif variant == "unapproved_pair":  # two UNAPPROVED plans with doses beside the approved pair
        from synthetic_rt_fixtures import make_plan
        for name in ("pU", "pV"):
            path = make_plan(folder / f"{name}.dcm", uid(key + name), struct_uid=d["struct_uid"], study_uid=d["study"],
                             frame_uid=d["frame"], rx_gy=20.0, fractions=5, patient_id="SYNTH_D", label=name,
                             approval_status="UNAPPROVED")
            dose(folder, key, "d" + name[1:], "SYNTH_D", d["study"], d["frame"], path,
                 values=linear(GRID_A["shape"], GRID_A["origin"], GRID_A["spacing"], 0.5, (0, 0, 0)),
                 origin=GRID_A["origin"], spacing=GRID_A["spacing"])
            d["written"][name] = (path, folder / f"d{name[1:]}.dcm")
    elif variant in ("partial_records", "full_records"):  # RTRECORDs for the declared pair, same session days
        sessions_b = 5 if variant == "partial_records" else 10
        for index in range(10):
            day = f"202401{index + 2:02d}"
            record(folder, key, f"recA{index}", "SYNTH_D", pa, index + 1, day)
            if index < sessions_b:
                record(folder, key, f"recB{index}", "SYNTH_D", pb, index + 1, day)
    plan_uids = {name: {p: str(pydicom.dcmread(v[0], stop_before_pixels=True).SOPInstanceUID)
                        for p, v in c["written"].items()} for name, c in info.items()}
    (root / "plan_uids.json").write_text(json.dumps(plan_uids, indent=1, sort_keys=True))
    return plan_uids
