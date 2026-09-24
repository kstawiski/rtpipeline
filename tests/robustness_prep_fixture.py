"""Synthetic multi-source CT courses for robustness preparation tests.

Every course is generated here: a CT series with pixel data, a Manual RTSTRUCT
(RS.dcm), an optional AutoRTS RTSTRUCT (RS_auto.dcm), an optional RS_custom.dcm
and optional custom-model RTSTRUCTs, all built with rt_utils from ellipsoid
masks and then edited with pydicom where a case needs a defect. ``inputs.json``
records the identity catalog the driver serves and per-course driver options.
No clinical data is read or written.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

SOURCE_FILES = {"Manual": "RS.dcm", "AutoRTS_total": "RS_auto.dcm", "Custom": "RS_custom.dcm"}


def write_ct(ct_dir: Path, *, rows: int, columns: int, slices: int,
             spacing: Tuple[float, float, float] = (0.98, 0.98, 3.0), seed: int = 0) -> str:
    """Write a CT series with textured int16 pixel data; return its SeriesInstanceUID."""
    ct_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    series_uid, study_uid, frame_uid = generate_uid(), generate_uid(), generate_uid()
    yy, xx = np.mgrid[0:rows, 0:columns]
    for index in range(slices):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.ImplementationClassUID = generate_uid()
        path = ct_dir / f"ct_{index:04d}.dcm"
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.Modality = "CT"
        ds.PatientID = "SYNTH"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.StudyDate, ds.StudyTime, ds.StudyID = "20240101", "093000", "1"
        ds.SeriesNumber = 1
        ds.InstanceNumber = index + 1
        ds.Rows, ds.Columns = rows, columns
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
        ds.PixelRepresentation = 1
        ds.PixelSpacing = [spacing[1], spacing[0]]
        ds.SliceThickness = spacing[2]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [-0.5 * columns * spacing[0], -0.5 * rows * spacing[1],
                                   index * spacing[2]]
        ds.RescaleSlope, ds.RescaleIntercept = 1.0, -1024.0
        body = ((xx - columns / 2) ** 2 + (yy - rows / 2) ** 2) < (0.45 * min(rows, columns)) ** 2
        pixels = np.where(body, 1024 + 40 + ((xx * 7 + yy * 3 + index * 11) % 60), 24)
        pixels = pixels + rng.integers(-15, 16, size=(rows, columns))
        ds.PixelData = pixels.astype(np.int16).tobytes()
        ds.save_as(str(path), enforce_file_format=True)
    return str(series_uid)


def ellipsoid(shape_rcs: Tuple[int, int, int], center: Sequence[float],
              radii: Sequence[float]) -> np.ndarray:
    """Boolean (rows, columns, slices) ellipsoid; center/radii in voxels."""
    r, c, s = np.ogrid[0:shape_rcs[0], 0:shape_rcs[1], 0:shape_rcs[2]]
    return (((r - center[0]) / radii[0]) ** 2 + ((c - center[1]) / radii[1]) ** 2
            + ((s - center[2]) / radii[2]) ** 2) <= 1.0


def write_rtstruct(ct_dir: Path, path: Path, masks: Iterable[Tuple[str, np.ndarray]]) -> None:
    from rt_utils import RTStructBuilder

    rtstruct = RTStructBuilder.create_new(dicom_series_path=str(ct_dir))
    for name, mask in masks:
        rtstruct.add_roi(mask=mask, name=name)
    rtstruct.save(str(path))


def _next_roi_number(ds) -> int:
    return max(int(r.ROINumber) for r in ds.StructureSetROISequence) + 1


def append_roi(ds, name: str, contours: Optional[List[Dataset]]) -> None:
    number = _next_roi_number(ds)
    roi = Dataset()
    roi.ROINumber = number
    roi.ROIName = name
    roi.ReferencedFrameOfReferenceUID = ds.StructureSetROISequence[0].ReferencedFrameOfReferenceUID
    ds.StructureSetROISequence.append(roi)
    item = Dataset()
    item.ReferencedROINumber = number
    if contours is not None:
        item.ContourSequence = DicomSequence(contours)
    ds.ROIContourSequence.append(item)


def contour(kind: str, data: List[float], count: Optional[int] = None,
            image_uid: Optional[str] = None) -> Dataset:
    item = Dataset()
    item.ContourGeometricType = kind
    item.ContourData = data
    item.NumberOfContourPoints = count if count is not None else len(data) // 3
    if image_uid is not None:
        ref = Dataset()
        ref.ReferencedSOPClassUID = CTImageStorage
        ref.ReferencedSOPInstanceUID = image_uid
        item.ContourImageSequence = DicomSequence([ref])
    return item


def roi_contours(ds, name: str):
    number = next(int(r.ROINumber) for r in ds.StructureSetROISequence if r.ROIName == name)
    return next(c for c in ds.ROIContourSequence if int(c.ReferencedROINumber) == number).ContourSequence


def slice_z_and_uid(ct_dir: Path) -> List[Tuple[float, str]]:
    out = []
    for path in sorted(ct_dir.glob("*.dcm")):
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        out.append((float(ds.ImagePositionPatient[2]), str(ds.SOPInstanceUID)))
    return sorted(out)


def square_contour(x0: float, y0: float, side: float, z: float, uid: str) -> Dataset:
    return contour("CLOSED_PLANAR", [x0, y0, z, x0 + side, y0, z, x0 + side, y0 + side, z,
                                     x0, y0 + side, z], image_uid=uid)


def apply_defects(ct_dir: Path, rs_path: Path, defects: Sequence[Dict[str, Any]]) -> None:
    """Append ROIs carrying the requested defects to an existing RTSTRUCT."""
    if not defects:
        return
    ds = pydicom.dcmread(rs_path)
    slices = slice_z_and_uid(ct_dir)
    mid_z, mid_uid = slices[len(slices) // 2]
    for defect in defects:
        name, kind = defect["name"], defect["kind"]
        if kind == "point":
            append_roi(ds, name, [contour("POINT", [0.0, 0.0, mid_z], image_uid=mid_uid)])
        elif kind == "declared_only":
            append_roi(ds, name, [])
        elif kind == "partially_unparseable":
            # One valid square and one area-bounding item whose declared point
            # count disagrees with its data: ROI_CONTOUR_PARTIALLY_UNPARSEABLE.
            append_roi(ds, name, [
                square_contour(-10.0, -10.0, 12.0, mid_z, mid_uid),
                contour("CLOSED_PLANAR", [-5.0, -5.0, slices[1][0], 5.0, -5.0, slices[1][0],
                                          5.0, 5.0, slices[1][0]], count=7, image_uid=slices[1][1]),
            ])
        elif kind == "outside_fov":
            # Valid geometry far outside the image: rasterizes to an empty mask.
            append_roi(ds, name, [square_contour(5000.0, 5000.0, 10.0, mid_z, mid_uid)])
        elif kind == "unreferenced_slice":
            # A valid contour referencing no image of the series: empty mask.
            append_roi(ds, name, [square_contour(-10.0, -10.0, 12.0, mid_z, generate_uid())])
        elif kind == "no_image_reference":
            # Valid geometry without ContourImageSequence: the rt_utils reader raises.
            append_roi(ds, name, [contour("CLOSED_PLANAR", [-10.0, -10.0, mid_z, 2.0, -10.0, mid_z,
                                                            2.0, 2.0, mid_z])])
        else:
            raise ValueError(kind)
    ds.save_as(rs_path)


def build_course(course: Path, spec: Dict[str, Any]) -> None:
    """Create one synthetic course from ``spec``.

    spec keys: rows, columns, slices, rois (list of {source, name, center,
    radii, catalog}), defects ({source: [defect, ...]}), models ({model: [roi
    dict, ...]}), plus driver options copied into inputs.json.
    """
    rows, columns, slices = spec["rows"], spec["columns"], spec["slices"]
    ct_dir = course / "CT"
    series_uid = write_ct(ct_dir, rows=rows, columns=columns, slices=slices,
                          seed=int(spec.get("seed", 0)))
    by_source: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    catalog: List[List[str]] = []
    for roi in spec["rois"]:
        mask = ellipsoid((rows, columns, slices), roi["center"], roi["radii"])
        if roi.get("slab"):
            mask = np.zeros_like(mask)
            r0, r1, c0, c1, s0, s1 = roi["slab"]
            mask[r0:r1, c0:c1, s0:s1] = True
        by_source.setdefault(roi["source"], []).append((roi["name"], mask))
        if roi.get("catalog", True):
            catalog.append([roi["source"], roi["name"]])
    for source, masks in by_source.items():
        if source.startswith("CustomModel:"):
            model_dir = course / "Segmentation_CustomModels" / source.split(":", 1)[1]
            model_dir.mkdir(parents=True, exist_ok=True)
            write_rtstruct(ct_dir, model_dir / "rtstruct.dcm", masks)
            apply_defects(ct_dir, model_dir / "rtstruct.dcm", spec.get("defects", {}).get(source, []))
            continue
        path = course / SOURCE_FILES[source]
        write_rtstruct(ct_dir, path, masks)
        apply_defects(ct_dir, path, spec.get("defects", {}).get(source, []))
    catalog.extend(spec.get("extra_catalog", []))
    options = {k: v for k, v in spec.items() if k not in {"rois", "defects"}}
    (course / "inputs.json").write_text(json.dumps(
        {**options, "series_uid": series_uid, "catalog": catalog}, indent=1))


def realistic_rois(rows: int, columns: int, slices: int, *, organs: int = 80,
                   manual_oars: int = 12, custom: int = 16, seed: int = 7) -> List[Dict[str, Any]]:
    """About 100+ ROIs across Manual, AutoRTS and Custom with ~10 selected targets."""
    rng = np.random.default_rng(seed)
    rois: List[Dict[str, Any]] = []

    def blob(scale):
        center = (rng.uniform(0.25, 0.75) * rows, rng.uniform(0.25, 0.75) * columns,
                  rng.uniform(0.2, 0.8) * slices)
        radii = (rng.uniform(0.3, 1.0) * scale * rows / 10, rng.uniform(0.3, 1.0) * scale * columns / 10,
                 max(2.0, rng.uniform(0.3, 1.0) * scale * slices / 6))
        return center, radii

    targets = [("GTV1", 0.35), ("GTV2", 0.25), ("CTV1", 0.6), ("CTV2", 0.45), ("PTV1", 0.8),
               ("PTV2", 0.6), ("urinary_bladder", 0.7)]
    for name, scale in targets:
        center, radii = blob(scale)
        rois.append(dict(source="Manual", name=name, center=center, radii=radii))
    for index in range(manual_oars):
        center, radii = blob(0.8)
        rois.append(dict(source="Manual", name=f"OAR_{index:02d}", center=center, radii=radii))
    for index in range(organs):
        center, radii = blob(1.2)
        name = "urinary_bladder" if index == 0 else f"organ_{index:03d}"
        rois.append(dict(source="AutoRTS_total", name=name, center=center, radii=radii))
    for index in range(custom):
        center, radii = blob(1.0)
        name = ["PTV_opt", "CTV_eval"][index] if index < 2 else f"custom_{index:02d}"
        rois.append(dict(source="Custom", name=name, center=center, radii=radii))
    return rois
