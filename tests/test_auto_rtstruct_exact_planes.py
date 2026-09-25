"""RS_auto contours must lie on the CT planes they reference.

TotalSegmentator writes ``<series>--total.dcm`` with rt-utils, which places mask
slice ``i`` at ``z_first + i * (z_last - z_first) / (N - 1)``. On a planning CT
with mixed slice spacing those contours sit off the planes of the images they
reference, and the scoped reader (``rtstruct_geometry``) holds every ROI as
``ROI_UNRESOLVED_SOURCE_SCOPE``. ``build_auto_rtstruct`` used to copy that file
verbatim. It now publishes the copy only when every contour is on its plane,
and otherwise rebuilds RS_auto from the per-structure NIfTI masks on the exact
DICOM planes. All data here are synthetic.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from course_contract_test_utils import write_minimal_course_contract
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from rtpipeline import auto_rtstruct as ar
from rtpipeline.rtstruct_geometry import (
    PLANE_TOLERANCE_MM,
    contours_on_referenced_planes,
    create_scoped_rtstruct,
    plane_offset_mm,
)

CT_SIZE = 16
PIXEL_MM = 2.0
ORIGIN_XY = (-15.0, -15.0)
# Physical boxes (x, y, z ranges in mm) standing in for TotalSegmentator classes.
STRUCTURES = {
    "iliac_artery_left": ((-9.0, -1.0), (-7.0, 3.0), (-22.0, 12.0)),
    "vertebrae_S1": ((1.0, 9.0), (-3.0, 7.0), (-26.0, 4.0)),
}


def _mixed_z(n: int = 20) -> np.ndarray:
    steps = [2.4 if i < n // 2 else 3.0 for i in range(n - 1)]
    return np.concatenate([[0.0], np.cumsum(steps)]) - 30.0


def _uniform_z(n: int = 20) -> np.ndarray:
    return np.arange(n, dtype=float) * 3.0 - 30.0


def _write_ct(ct_dir: Path, zs: np.ndarray) -> None:
    ct_dir.mkdir(parents=True, exist_ok=True)
    series_uid, for_uid, study_uid = generate_uid(), generate_uid(), generate_uid()
    for i, z in enumerate(zs):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.Modality = "CT"
        ds.PatientID = "SYNTHETIC"
        ds.PatientName = "Synthetic^Phantom"
        ds.PatientSex = "O"
        ds.PatientBirthDate = ""
        ds.StudyInstanceUID = study_uid
        ds.StudyDate = "20260101"
        ds.StudyTime = "120000"
        ds.StudyID = "1"
        ds.AccessionNumber = ""
        ds.ReferringPhysicianName = ""
        ds.SeriesInstanceUID = series_uid
        ds.SeriesNumber = 1
        ds.SeriesDate = "20260101"
        ds.SeriesTime = "120000"
        ds.FrameOfReferenceUID = for_uid
        ds.InstanceNumber = i + 1
        ds.ImagePositionPatient = [ORIGIN_XY[0], ORIGIN_XY[1], float(z)]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [PIXEL_MM, PIXEL_MM]
        ds.SliceThickness = 2.4
        ds.Rows = ds.Columns = CT_SIZE
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.RescaleIntercept = 0
        ds.RescaleSlope = 1
        ds.PixelData = np.zeros((CT_SIZE, CT_SIZE), dtype=np.int16).tobytes()
        ds.save_as(ct_dir / f"ct_{i:03d}.dcm", enforce_file_format=True)


def _regular_grid(zs: np.ndarray) -> sitk.Image:
    """The regularized NIfTI grid TotalSegmentator segments on."""
    image = sitk.Image(CT_SIZE, CT_SIZE, len(zs), sitk.sitkUInt8)
    image.SetOrigin((ORIGIN_XY[0], ORIGIN_XY[1], float(zs[0])))
    image.SetSpacing((PIXEL_MM, PIXEL_MM, float(zs[-1] - zs[0]) / (len(zs) - 1)))
    return image


def _grid_mask(grid: sitk.Image, box) -> np.ndarray:
    """Boolean mask in SimpleITK array order (z, y, x)."""
    nx, ny, nz = grid.GetSize()
    ox, oy, oz = grid.GetOrigin()
    sx, sy, sz = grid.GetSpacing()
    x = ox + np.arange(nx) * sx
    y = oy + np.arange(ny) * sy
    z = oz + np.arange(nz) * sz
    (x0, x1), (y0, y1), (z0, z1) = box
    inside = lambda values, lo, hi: (values >= lo) & (values <= hi)  # noqa: E731
    return (
        inside(z, z0, z1)[:, None, None]
        & inside(y, y0, y1)[None, :, None]
        & inside(x, x0, x1)[None, None, :]
    )


def _write_totalseg_outputs(seg_dir: Path, ct_dir: Path, zs: np.ndarray, *, rtstruct=True, masks=True):
    """Mirror the verified directory layout: ``<series>--total.dcm`` written by
    rt-utils from the regularized grid (as TotalSegmentator does) and one
    ``total--<roi>.nii.gz`` per structure, without a multilabel file."""
    from rt_utils import RTStructBuilder

    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (seg_dir / "total--ts_version.json").write_text("{}", encoding="utf-8")
    grid = _regular_grid(zs)
    total = None
    if rtstruct:
        builder = RTStructBuilder.create_new(dicom_series_path=str(ct_dir))
        for name, box in STRUCTURES.items():
            # Regular-grid slice i becomes DICOM slice i, rt-utils layout (x, y, z).
            builder.add_roi(mask=np.transpose(_grid_mask(grid, box), (2, 1, 0)), name=name)
        total = seg_dir / f"{seg_dir.name}--total.dcm"
        builder.save(str(total))
    if masks:
        for name, box in STRUCTURES.items():
            image = sitk.GetImageFromArray(_grid_mask(grid, box).astype(np.uint8))
            image.CopyInformation(grid)
            sitk.WriteImage(image, str(seg_dir / f"total--{name}.nii.gz"))
    # Model outputs predate the RS_auto publication.
    for path in seg_dir.iterdir():
        os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    return total


def _align_contract_ct_geometry(metadata_path: Path, ct_dir: Path) -> None:
    """Record the CT geometry exactly as the contract reader derives it.

    The shared helper records pixel spacing and orientation as lists, while the
    reader derives them from pydicom values. Header-only fixtures never notice;
    these slices carry real geometry, so both records are set from the reader.
    """
    from rtpipeline.course_contract import _ct_provenance

    geometry = _ct_provenance(ct_dir)["geometry"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    provenance = next(
        value["planning_ct"]["nifti_provenance"]
        for value in [metadata, *metadata.values()]
        if isinstance(value, dict) and "planning_ct" in value
    )
    provenance["geometry"] = geometry
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    sidecar = metadata_path.parents[1] / provenance["sidecar_path"]  # course-relative
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    data["geometry"] = geometry
    sidecar.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _course(tmp_path: Path, zs: np.ndarray, **outputs):
    course = tmp_path / "course"
    dirs = ar.build_course_dirs(course)
    _write_ct(dirs.dicom_ct, zs)
    _align_contract_ct_geometry(write_minimal_course_contract(course, planning_ct_dir=dirs.dicom_ct), dirs.dicom_ct)
    for path in [*dirs.dicom_ct.iterdir(), *(course / "NIFTI").iterdir()]:
        os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    series_dir = dirs.segmentation_totalseg / "CT_series"
    total = _write_totalseg_outputs(series_dir, dirs.dicom_ct, zs, **outputs)
    return course, dirs.dicom_ct, series_dir, total


def _series(ct_dir: Path):
    from rt_utils import image_helper

    return image_helper.load_sorted_image_series(str(ct_dir))


def _max_plane_offset(rtstruct_path: Path, ct_dir: Path) -> float:
    ds = pydicom.dcmread(str(rtstruct_path))
    by_uid = {str(s.SOPInstanceUID): s for s in _series(ct_dir)}
    offsets = [
        plane_offset_mm(
            np.asarray(c.ContourData, dtype=float).reshape(-1, 3),
            by_uid[str(c.ContourImageSequence[0].ReferencedSOPInstanceUID)],
        )
        for item in ds.ROIContourSequence
        for c in item.ContourSequence
    ]
    assert offsets
    return max(offsets)


def _roi_names(rtstruct_path: Path) -> set[str]:
    ds = pydicom.dcmread(str(rtstruct_path))
    return {str(roi.ROIName) for roi in ds.StructureSetROISequence}


def _resume_decision(course: Path) -> dict:
    audit = json.loads((course / "metadata" / "segmentation_resume.json").read_text(encoding="utf-8"))
    return audit["decisions"]["RS_auto"]


def _assert_scoped_reader_resolves(rtstruct_path: Path, ct_dir: Path) -> None:
    scoped = create_scoped_rtstruct(ct_dir, rtstruct_path)
    assert {r.code for r in scoped.scopes.values()} == {None}, {
        r.roi_name: (r.code, r.detail) for r in scoped.scopes.values()
    }
    for name in scoped.get_roi_names():
        assert np.any(scoped.get_roi_mask_by_name(name)), name


def test_totalseg_rtstruct_on_mixed_spacing_ct_is_off_plane(tmp_path) -> None:
    """The input ae60f01 copied verbatim: off-plane, and every ROI is held."""
    _course_dir, ct_dir, _seg, total = _course(tmp_path, _mixed_z())
    assert _max_plane_offset(total, ct_dir) > 0.1
    ok, detail = contours_on_referenced_planes(pydicom.dcmread(str(total)), _series(ct_dir))
    assert not ok and "not on the CT plane they reference" in detail
    scoped = create_scoped_rtstruct(ct_dir, total)
    assert {r.code for r in scoped.scopes.values()} == {"ROI_UNRESOLVED_SOURCE_SCOPE"}
    assert all("geometry disagrees" in r.detail for r in scoped.scopes.values())


def test_mixed_spacing_rebuilds_rs_auto_on_exact_planes(tmp_path, caplog) -> None:
    course, ct_dir, _seg, total = _course(tmp_path, _mixed_z())
    with caplog.at_level("WARNING", logger="rtpipeline.auto_rtstruct"):
        out = ar.build_auto_rtstruct(course)
    assert out == course / "RS_auto.dcm"
    assert out.read_bytes() != total.read_bytes()
    assert any("not publishing" in r.getMessage() for r in caplog.records)
    assert _max_plane_offset(out, ct_dir) <= PLANE_TOLERANCE_MM
    # Same ROI names the TotalSegmentator RTSTRUCT carries.
    assert _roi_names(out) == _roi_names(total) == set(STRUCTURES)
    _assert_scoped_reader_resolves(out, ct_dir)
    assert _resume_decision(course)["action"] == "rebuilt"
    # The rebuilt file is reused on the next run.
    before = out.read_bytes()
    assert ar.build_auto_rtstruct(course) == out
    assert out.read_bytes() == before
    assert _resume_decision(course)["action"] == "reused"


def test_rebuilt_masks_sit_on_the_correct_slices(tmp_path) -> None:
    """Each rebuilt slice holds the structure present at that slice's z."""
    course, ct_dir, _seg, _total = _course(tmp_path, _mixed_z())
    out = ar.build_auto_rtstruct(course)
    scoped = create_scoped_rtstruct(ct_dir, out)
    zs = [float(s.ImagePositionPatient[2]) for s in scoped.series_data]
    for name, (_x, _y, (z0, z1)) in STRUCTURES.items():
        occupied = np.any(scoped.get_roi_mask_by_name(name), axis=(0, 1))
        expected = np.asarray([z0 - 1.5 <= z <= z1 + 1.5 for z in zs])
        assert occupied.any()
        assert not np.any(occupied & ~expected), name


def test_existing_misaligned_rs_auto_is_rejected_and_rebuilt(tmp_path) -> None:
    course, ct_dir, _seg, total = _course(tmp_path, _mixed_z())
    stale = course / "RS_auto.dcm"
    shutil.copyfile(total, stale)  # what ae60f01 published
    assert ar.build_auto_rtstruct(course) == stale
    assert _max_plane_offset(stale, ct_dir) <= PLANE_TOLERANCE_MM
    _assert_scoped_reader_resolves(stale, ct_dir)
    decision = _resume_decision(course)
    assert decision["action"] == "rebuilt"
    assert "do not lie on the planning CT planes" in decision["rejected_artifact"]["reason"]


def test_mixed_spacing_without_masks_fails_closed(tmp_path) -> None:
    course, _ct_dir, _seg, _total = _course(tmp_path, _mixed_z(), masks=False)
    assert ar.build_auto_rtstruct(course) is None
    assert not (course / "RS_auto.dcm").exists()
    decision = _resume_decision(course)
    assert decision["action"] == "failed"
    assert "off the planning CT planes" in decision["reason"]
    assert "no usable NIfTI mask" in decision["reason"]


def test_mixed_spacing_multilabel_path_is_on_planes(tmp_path) -> None:
    """The multilabel NIfTI path also places rt-utils contours on exact planes."""
    course, ct_dir, seg_dir, _total = _course(tmp_path, _mixed_z(), rtstruct=False, masks=False)
    grid = _regular_grid(_mixed_z())
    labels = np.zeros(grid.GetSize()[::-1], dtype=np.uint8)
    for value, box in enumerate(STRUCTURES.values(), start=1):
        labels[_grid_mask(grid, box)] = value
    image = sitk.GetImageFromArray(labels)
    image.CopyInformation(grid)
    sitk.WriteImage(image, str(seg_dir / f"{seg_dir.name}_total_multilabel.nii.gz"))
    (seg_dir / f"{seg_dir.name}_total_segmentations.json").write_text(
        json.dumps({name: i for i, name in enumerate(STRUCTURES, start=1)}), encoding="utf-8"
    )
    for path in seg_dir.iterdir():
        os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    out = ar.build_auto_rtstruct(course)
    assert out is not None
    assert _roi_names(out) == set(STRUCTURES)
    assert _max_plane_offset(out, ct_dir) <= PLANE_TOLERANCE_MM
    _assert_scoped_reader_resolves(out, ct_dir)


def test_uniform_spacing_aligned_rtstruct_is_copied_unchanged(tmp_path) -> None:
    """Aligned input: publication bytes and resume decision match ae60f01."""
    course, ct_dir, _seg, total = _course(tmp_path, _uniform_z())
    assert _max_plane_offset(total, ct_dir) <= PLANE_TOLERANCE_MM
    # ae60f01 published copy -> sanitize_rtstruct -> fix_rtstruct_rois.
    expected = tmp_path / "expected.dcm"
    shutil.copyfile(total, expected)
    ar.sanitize_rtstruct(expected)
    ar.fix_rtstruct_rois(ct_dir, expected)
    out = ar.build_auto_rtstruct(course)
    assert out == course / "RS_auto.dcm"
    assert out.read_bytes() == expected.read_bytes() == total.read_bytes()
    decision = _resume_decision(course)
    assert decision["action"] == "rebuilt"
    assert decision["reason"] == (
        "rebuilt from a TotalSegmentator RTSTRUCT referencing the planning CT series"
    )
    before = out.read_bytes()
    assert ar.build_auto_rtstruct(course) == out
    assert out.read_bytes() == before
    assert _resume_decision(course)["action"] == "reused"
    _assert_scoped_reader_resolves(out, ct_dir)


def test_anchoring_leaves_uniform_rt_utils_output_untouched(tmp_path) -> None:
    from rt_utils import RTStructBuilder

    ct_dir = tmp_path / "CT"
    _write_ct(ct_dir, _uniform_z())
    builder = RTStructBuilder.create_new(dicom_series_path=str(ct_dir))
    grid = _regular_grid(_uniform_z())
    for name, box in STRUCTURES.items():
        builder.add_roi(mask=np.transpose(_grid_mask(grid, box), (2, 1, 0)), name=name)
    before = [list(c.ContourData) for i in builder.ds.ROIContourSequence for c in i.ContourSequence]
    assert ar._anchor_contours_to_referenced_planes(builder.ds, builder.series_data) == 0
    after = [list(c.ContourData) for i in builder.ds.ROIContourSequence for c in i.ContourSequence]
    assert after == before


@pytest.mark.parametrize("offset, ok", [(0.0, True), (0.5 * PLANE_TOLERANCE_MM, True), (0.42, False)])
def test_plane_check_uses_scoped_reader_tolerance(tmp_path, offset, ok) -> None:
    from rt_utils import RTStructBuilder

    ct_dir = tmp_path / "CT"
    _write_ct(ct_dir, _uniform_z())
    builder = RTStructBuilder.create_new(dicom_series_path=str(ct_dir))
    mask = np.zeros((CT_SIZE, CT_SIZE, 20), dtype=bool)
    mask[4:9, 4:9, 5] = True
    builder.add_roi(mask=mask, name="box")
    contour = builder.ds.ROIContourSequence[0].ContourSequence[0]
    points = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
    points[:, 2] += offset
    contour.ContourData = points.ravel().tolist()
    result, detail = contours_on_referenced_planes(builder.ds, builder.series_data)
    assert result is ok
    assert (detail == "") is ok
