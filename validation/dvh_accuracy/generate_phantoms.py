"""Deterministic synthetic CT/RTSTRUCT/RTDOSE/RTPLAN plus synthetic RTRECORDs.

All coordinates are patient LPS mm, axial orientation. No clinical input is read.
Circles use 256 vertices; the polygon area deficit is <0.0101%. Contours are
exact cross-sections of each continuous shape at the CT slice centres.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import (CTImageStorage, ExplicitVRLittleEndian, RTDoseStorage,
                         RTPlanStorage, RTStructureSetStorage)
from analytic import Geometry, REFERENCE_GY

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SEED = 20261027  # deterministic identifiers; no random sampling in truth or phantoms
N_VERTICES = 256
SHAPES = ("sphere", "cylinder", "cube", "shell")
OFFSETS = {"aligned": (0., 0., 0.), "offset": (.37, -.61, .43)}


def uid(key):
    return "2.25." + str(int.from_bytes(hashlib.sha256(f"{SEED}:{key}".encode()).digest()[:16], "big"))


def designs():
    for volume, phase, ct, dose, field in itertools.product(
            (2., 50., 1000.), OFFSETS, (1., 3.), (1., 2.5), ("uniform", "linear", "radial")):
        key = f"v{volume:g}_{phase}_ct{ct:g}_dg{dose:g}_{field}"
        shapes = ("sphere", "shell") if field == "radial" else SHAPES
        yield dict(group=key, volume_cc=volume, phase=phase, ct_mm=ct, dose_mm=dose,
                   field=field, center=list(OFFSETS[phase]), shapes=list(shapes))


def ds_numbers(values):
    return [format(float(v), ".10g") for v in np.asarray(values).ravel()]


def base(path, sop_class, key, series, study, frame):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = uid(key)
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = uid("implementation")
    meta.ImplementationVersionName = "SYNTH_DVH_V1"
    d = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    d.SOPClassUID = sop_class
    d.SOPInstanceUID = uid(key)
    d.StudyInstanceUID = study
    d.SeriesInstanceUID = series
    d.FrameOfReferenceUID = frame
    d.PatientName = "SYNTHETIC^DVH"
    d.PatientID = "SYNTH_DVH"
    d.PatientBirthDate = ""
    d.PatientSex = ""
    d.StudyDate = "20200101"
    d.StudyTime = "120000"
    d.StudyID = "SYNTH"
    d.AccessionNumber = ""
    d.ReferringPhysicianName = ""
    d.Manufacturer = "Analytic synthetic benchmark"
    d.InstanceNumber = 1
    d.SeriesNumber = 1
    return d


def save(d, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    d.save_as(str(path), enforce_file_format=True)


def image_fields(d, n, bits=16):
    d.Rows = n
    d.Columns = n
    d.SamplesPerPixel = 1
    d.PhotometricInterpretation = "MONOCHROME2"
    d.BitsAllocated = bits
    d.BitsStored = bits
    d.HighBit = bits - 1
    d.PixelRepresentation = 0
    d.ImageOrientationPatient = [1., 0., 0., 0., 1., 0.]


def polygon(g, z):
    """Return outer then inner polygons and their parity area in mm2."""
    q = z - g.center[2]
    r = g.radius
    if abs(q) >= r:
        return [], 0.
    if g.shape == "cube":
        xy = np.array([[-r, -r], [r, -r], [r, r], [-r, r]]) + g.center[:2]
        return [xy], (2 * r) ** 2
    outer = r if g.shape == "cylinder" else math.sqrt(r * r - q * q)
    radii = [outer]
    if g.shape == "shell" and abs(q) < g.inner_radius:
        radii.append(math.sqrt(g.inner_radius ** 2 - q * q))
    angle = np.arange(N_VERTICES) * (2 * math.pi / N_VERTICES)
    unit = np.column_stack((np.cos(angle), np.sin(angle)))
    polys = [unit * radius + g.center[:2] for radius in radii]
    area = (outer ** 2 - (radii[1] ** 2 if len(radii) == 2 else 0.)) * N_VERTICES / 2 * math.sin(2 * math.pi / N_VERTICES)
    return polys, area


def generate(parent: Path, spec: dict, with_contract=True, ct_cache: Path | None = None):
    course = Path(parent) / "SYNTH_DVH" / spec["group"]
    if course.exists():
        raise FileExistsError(course)
    course.mkdir(parents=True)
    imaging_key = f"v{spec['volume_cc']:g}_{spec['phase']}_ct{spec['ct_mm']:g}"
    study, frame = uid(imaging_key + ":study"), uid(imaging_key + ":frame")
    series = uid(imaging_key + ":ct")
    geoms = [Geometry(shape, spec["volume_cc"], tuple(spec["center"])) for shape in spec["shapes"]]
    # Multiples of 15 make origin common to 1, 2.5 and 3 mm lattices.
    extent = math.ceil((max(g.radius for g in geoms) + 6) / 15) * 15
    ct_zs = np.arange(-extent, extent + .01, spec["ct_mm"])
    ct_n = int(extent) + 1  # CT in-plane spacing = 2 mm
    ct_refs = []
    for i, z in enumerate(ct_zs):
        path = course / "DICOM" / "CT" / f"ct_{i:04d}.dcm"
        cached = ct_cache / path.name if ct_cache is not None else None
        if cached is not None and cached.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            os.link(cached, path)
            ref = Dataset()
            ref.ReferencedSOPClassUID = CTImageStorage
            ref.ReferencedSOPInstanceUID = uid(f"{imaging_key}:ct:{i}")
            ct_refs.append(ref)
            continue
        d = base(path, CTImageStorage, f"{imaging_key}:ct:{i}", series, study, frame)
        d.Modality = "CT"
        d.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        image_fields(d, ct_n)
        d.PixelSpacing = [2., 2.]
        d.SliceThickness = spec["ct_mm"]
        d.SpacingBetweenSlices = spec["ct_mm"]
        d.ImagePositionPatient = [-extent, -extent, float(z)]
        d.SliceLocation = float(z)
        d.InstanceNumber = i + 1
        d.RescaleIntercept, d.RescaleSlope = 0., 1.
        d.RescaleType = "HU"
        d.KVP = 120.
        d.PixelData = np.zeros((ct_n, ct_n), dtype="<u2").tobytes()
        save(d, path)
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, cached)
        ref = Dataset()
        ref.ReferencedSOPClassUID = CTImageStorage
        ref.ReferencedSOPInstanceUID = d.SOPInstanceUID
        ct_refs.append(ref)
    # A matching CT image for the fixture's contract provenance, not a dummy grid.
    if with_contract:
        import SimpleITK as sitk
        im = sitk.Image([ct_n, ct_n, len(ct_zs)], sitk.sitkInt16)
        im.SetSpacing((2., 2., spec["ct_mm"]))
        im.SetOrigin((-float(extent), -float(extent), -float(extent)))
        nifti = course / "NIFTI" / "ct.nii.gz"
        nifti.parent.mkdir()
        sitk.WriteImage(im, str(nifti))

    rs_path = course / "DICOM" / "RTSTRUCT" / "rs.dcm"
    rs = base(rs_path, RTStructureSetStorage, spec["group"] + ":rs", uid(spec["group"] + ":rss"), study, frame)
    rs.Modality = "RTSTRUCT"
    rs.StructureSetLabel = "SYNTH_DVH"
    rs.StructureSetDate, rs.StructureSetTime = "20200101", "120000"
    ser = Dataset()
    ser.SeriesInstanceUID = series
    ser.ContourImageSequence = Sequence(ct_refs)
    st = Dataset()
    st.ReferencedSOPClassUID = "1.2.840.10008.3.1.2.3.1"
    st.ReferencedSOPInstanceUID = study
    st.RTReferencedSeriesSequence = Sequence([ser])
    fr = Dataset()
    fr.FrameOfReferenceUID = frame
    fr.RTReferencedStudySequence = Sequence([st])
    rs.ReferencedFrameOfReferenceSequence = Sequence([fr])
    rois, contours, observations, metadata = [], [], [], []
    for number, g in enumerate(geoms, 1):
        roi = Dataset()
        roi.ROINumber, roi.ROIName = number, g.shape
        roi.ReferencedFrameOfReferenceUID = frame
        roi.ROIGenerationAlgorithm = "AUTOMATIC"
        rois.append(roi)
        rc = Dataset()
        rc.ReferencedROINumber = number
        rc.ROIDisplayColor = [255, 100, 50]
        items, areas, zs = [], [], []
        for i, z in enumerate(ct_zs):
            polygons, area = polygon(g, float(z))
            if not polygons:
                continue
            areas.append(area)
            zs.append(float(z))
            for xy in polygons:
                c = Dataset()
                c.ContourGeometricType = "CLOSEDPLANAR_XOR" if g.shape == "shell" else "CLOSED_PLANAR"
                c.NumberOfContourPoints = len(xy)
                c.ContourData = ds_numbers(np.column_stack((xy, np.full(len(xy), z))))
                c.ContourImageSequence = Sequence([ct_refs[i]])
                items.append(c)
        rc.ContourSequence = Sequence(items)
        contours.append(rc)
        obs = Dataset()
        obs.ObservationNumber = number
        obs.ReferencedROINumber = number
        obs.RTROIInterpretedType = "ORGAN"
        obs.ROIInterpreter = ""
        observations.append(obs)
        metadata.append(dict(roi=number, geometry=asdict(g), radius_mm=g.radius,
                             inner_radius_mm=g.inner_radius, contour_planes=len(zs),
                             contour_slab_volume_cc=sum(areas) * spec["ct_mm"] / 1000,
                             contour_z_mm=zs))
    rs.StructureSetROISequence = Sequence(rois)
    rs.ROIContourSequence = Sequence(contours)
    rs.RTROIObservationsSequence = Sequence(observations)
    save(rs, rs_path)

    # Use the same production-shaped plan and delivery contract helpers as the tests.
    if str(REPO / "tests") not in sys.path:
        sys.path.insert(0, str(REPO / "tests"))
    import dvh_plan_target_fixtures as fx
    rp_path = fx.write_plan(course / "DICOM" / "RTPLAN" / "plan.dcm",
                            [fx.dose_reference(description="synthetic reference", prescription_gy=REFERENCE_GY)],
                            referenced_rtstruct_uid=str(rs.SOPInstanceUID), fractions=5)
    rp = pydicom.dcmread(rp_path)
    rp.SOPInstanceUID = uid(spec["group"] + ":plan")
    rp.file_meta.MediaStorageSOPInstanceUID = rp.SOPInstanceUID
    rp.SeriesInstanceUID = uid(spec["group"] + ":plans")
    rp.StudyInstanceUID, rp.FrameOfReferenceUID = study, frame
    rp.PatientID, rp.PatientName = "SYNTH_DVH", "SYNTHETIC^DVH"
    rp.DoseReferenceSequence[0].DoseReferenceUID = uid(spec["group"] + ":reference")
    rp.FractionGroupSequence[0].ReferencedBeamSequence[0].ReferencedDoseReferenceSequence[0].ReferencedDoseReferenceUID = rp.DoseReferenceSequence[0].DoseReferenceUID
    rp.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamDose = REFERENCE_GY / 5
    cp = Dataset()
    cp.ControlPointIndex = 0
    cp.IsocenterPosition = [0., 0., 0.]
    rp.BeamSequence[0].ControlPointSequence = Sequence([cp])
    save(rp, rp_path)

    dose_path = course / "DICOM" / "RTDOSE" / "dose.dcm"
    d = base(dose_path, RTDoseStorage, spec["group"] + ":dose", uid(spec["group"] + ":doses"), study, frame)
    d.Modality, d.DoseUnits, d.DoseType, d.DoseSummationType = "RTDOSE", "GY", "PHYSICAL", "PLAN"
    step = spec["dose_mm"]
    coords = np.arange(-extent, extent + .01, step)
    n = len(coords)
    image_fields(d, n, bits=32)
    d.PixelSpacing = [step, step]
    d.SliceThickness = step
    d.ImagePositionPatient = [-extent, -extent, -extent]
    d.NumberOfFrames = n
    d.GridFrameOffsetVector = ds_numbers(np.arange(n) * step)
    d.FrameIncrementPointer = [pydicom.tag.Tag(0x3004, 0x000c)]
    d.DoseGridScaling = .0001
    pixels = np.empty((n, n, n), dtype="<u4")
    xx, yy = np.meshgrid(coords - spec["center"][0], coords - spec["center"][1])
    for i, z in enumerate(coords):
        if spec["field"] == "uniform":
            values = np.full((n, n), 20.)
        elif spec["field"] == "linear":
            values = np.full((n, n), 20 + .1 * z)
        else:
            values = 30 - .15 * np.sqrt(xx * xx + yy * yy + (z - spec["center"][2]) ** 2)
        if values.min() < 0:
            raise ValueError("FOV exceeds nonnegative analytic dose support")
        pixels[i] = np.rint(values / float(d.DoseGridScaling)).astype("<u4")
    d.PixelData = pixels.tobytes()
    ref = Dataset()
    ref.ReferencedSOPClassUID, ref.ReferencedSOPInstanceUID = RTPlanStorage, rp.SOPInstanceUID
    d.ReferencedRTPlanSequence = Sequence([ref])
    save(d, dose_path)
    if with_contract:
        from course_contract_test_utils import write_minimal_course_contract
        write_minimal_course_contract(course, selected_plans=[rp_path], selected_doses=[dose_path],
                                      authoritative_rtstruct=rs_path, planning_ct_dir=course / "DICOM" / "CT",
                                      planning_ct_nifti=nifti)
        # Bind to the production serializer's CT identity, as organizer contracts
        # do, rather than assuming the test helper's serialization conventions.
        from rtpipeline.course_contract import _ct_provenance, _read_ct_headers
        contract_path = course / "metadata/case_metadata.json"
        payload = json.loads(contract_path.read_text())
        provenance = payload["course_contract"]["planning_ct"]["nifti_provenance"]
        provenance.update(_ct_provenance(course / "DICOM/CT",
                                         _read_ct_headers(course / "DICOM/CT", threads=4)))
        sidecar_path = course / provenance["sidecar_path"]
        sidecar = json.loads(sidecar_path.read_text())
        sidecar.update({k: v for k, v in provenance.items() if k != "sidecar_path"})
        sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
        contract_path.write_text(json.dumps(payload, indent=2) + "\n")
        fx.mark_fully_delivered(course, fractions=5)
        for i, record_path in enumerate(sorted((course / "DICOM_related/RTRECORD").glob("*.dcm")), 1):
            record = pydicom.dcmread(record_path)
            record.SOPInstanceUID = uid(f"{spec['group']}:record:{i}")
            record.file_meta.MediaStorageSOPInstanceUID = record.SOPInstanceUID
            record.SeriesInstanceUID = uid(spec["group"] + ":records")
            record.StudyInstanceUID, record.FrameOfReferenceUID = study, frame
            record.PatientID, record.PatientName = "SYNTH_DVH", "SYNTHETIC^DVH"
            record.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID = rp.SOPInstanceUID
            save(record, record_path)
        payload = json.loads(contract_path.read_text())
        payload["course_contract"]["delivery"]["delivered_dose_gy"] = REFERENCE_GY
        contract_path.write_text(json.dumps(payload, indent=2) + "\n")
    manifest = dict(spec, extent_mm=extent, circle_vertices=N_VERTICES, rois=metadata,
                    ct_inplane_mm=2., dose_quantization_Gy=.0001, reference_dose_Gy=REFERENCE_GY)
    (course / "phantom.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return course, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "phantoms")
    p.add_argument("--group", help="Exact group ID; omitted generates all 72 groups")
    a = p.parse_args()
    if not a.output.resolve().is_relative_to(ROOT):
        p.error("Output must be inside the authorized benchmark directory")
    found = False
    for spec in designs():
        if a.group and spec["group"] != a.group:
            continue
        found = True
        course, _ = generate(a.output, spec)
        print(course, flush=True)
    if not found:
        p.error("Unknown group")


if __name__ == "__main__":
    main()
