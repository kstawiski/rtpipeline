"""Regression and equivalence tests for MR scoped RTSTRUCT reader.

Exercises:
1. Mask byte-for-byte and feature row equivalence on clean synthetic MR data.
2. Defect (a) regression: MR TotalSegmentator RTSTRUCT with area-less (1- or 2-point)
   contour items next to normal polygons (fails raw rt-utils / OpenCV fillPoly,
   measured after fix equal to scoped reader).
3. Defect (b) regression: MR RTSTRUCT referencing an SOP outside the MR series
   next to a normal ROI (fails raw rt-utils as a whole, measured normal ROI and
   governed structural_nonmeasurement disposition for out-of-scope ROI after fix).
4. Negative controls: unreadable RTSTRUCT still fails; unexplained ROI read
   failure still fails.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import copy

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
import pytest
import SimpleITK as sitk

from rt_utils import RTStruct, RTStructBuilder
from rtpipeline import radiomics
from rtpipeline.config import PipelineConfig
from rtpipeline.radiomics_outcomes import RadiomicsCourseExtractionError
from rtpipeline.rtstruct_geometry import create_scoped_rtstruct
from robustness_prep_fixture import cylinder, ellipsoid


def _write_synthetic_mr(
    mr_dir: Path,
    *,
    rows: int = 16,
    columns: int = 16,
    slices: int = 5,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 3.0),
) -> Tuple[str, List[str]]:
    """Write a synthetic MR series with DICOM image slices."""
    mr_dir.mkdir(parents=True, exist_ok=True)
    series_uid = generate_uid()
    study_uid = generate_uid()
    frame_uid = generate_uid()
    sop_uids = []
    for index in range(slices):
        sop = generate_uid()
        sop_uids.append(sop)
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = sop
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.ImplementationClassUID = generate_uid()
        path = mr_dir / f"mr_{index:04d}.dcm"
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = MRImageStorage
        ds.SOPInstanceUID = sop
        ds.Modality = "MR"
        ds.PatientID = "SYNTH_MR"
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
        ds.ImagePositionPatient = [0.0, 0.0, float(index * spacing[2])]
        ds.RescaleSlope, ds.RescaleIntercept = 1.0, 0.0
        pixels = np.ones((rows, columns), dtype=np.int16) * 50
        ds.PixelData = pixels.tobytes()
        ds.save_as(str(path), enforce_file_format=True)
    return series_uid, sop_uids


def _write_mr_rtstruct(
    mr_dir: Path, path: Path, masks: Iterable[Tuple[str, np.ndarray]]
) -> Path:
    """Create an RTSTRUCT referencing the synthetic MR series."""
    rtstruct = RTStructBuilder.create_new(dicom_series_path=str(mr_dir))
    for name, mask in masks:
        rtstruct.add_roi(mask=mask, name=name)
    rtstruct.save(str(path))
    return path


class _MockMRExtractor:
    """Deterministic extractor stand-in for tests."""

    def __init__(self):
        self.settings = {}

    def execute(self, img, m_img, *extra):
        return {"_firstorder_Mean": 42.0, "_firstorder_Energy": 1234.5}


def test_clean_mr_masks_and_feature_rows_equivalence(tmp_path: Path, monkeypatch):
    """Equivalence proof: for clean MR ROIs without area-less items or scope problems,
    masks are byte-for-byte identical and feature rows match ae60f01 semantics."""
    mr_dir = tmp_path / "MR"
    series_uid, sop_uids = _write_synthetic_mr(mr_dir, rows=16, columns=16, slices=5)

    mask_target = cylinder((16, 16, 5), (8, 8, 2), (4, 4, 1))
    mask_organ = cylinder((16, 16, 5), (4, 4, 2), (2, 2, 1))
    rs_path = _write_mr_rtstruct(
        mr_dir, tmp_path / "RS.dcm", [("Target", mask_target), ("Organ", mask_organ)]
    )

    # 1. Compare masks byte for byte between raw rt-utils and ScopedRTStruct
    raw_rt = RTStructBuilder.create_from(str(mr_dir), str(rs_path))
    scoped = create_scoped_rtstruct(mr_dir, rs_path)

    for name, expected in [("Target", mask_target), ("Organ", mask_organ)]:
        raw_mask = raw_rt.get_roi_mask_by_name(name)
        scoped_mask = scoped.get_roi_mask_by_name(name)
        assert raw_mask.tobytes() == scoped_mask.tobytes()
        assert np.array_equal(raw_mask, scoped_mask)
        assert np.array_equal(raw_mask, expected)

    # 2. Compare feature rows extracted through _mr_manual_rows
    config = PipelineConfig(
        dicom_root=tmp_path, output_root=tmp_path / "out", logs_root=tmp_path / "logs"
    )
    series = radiomics.MRSeries(
        patient_id="SYNTH_MR", series_uid=series_uid, dir=mr_dir
    )
    img = sitk.Image([16, 16, 5], sitk.sitkInt16)
    extractor = _MockMRExtractor()
    source = radiomics._MRMaskSource(
        "manual", rs_path, (rs_path,), radiomics._mr_content_digest([rs_path])
    )
    provenance = {
        "extraction_arm": "MR:default",
        "run_identifier": "run-test-id",
        "code_revision": "test-rev",
    }

    # Extract with scoped reader (new behaviour)
    rows_scoped = radiomics._mr_manual_rows(
        config, series, img, extractor, source, parameter_provenance=provenance
    )

    # Extract with raw reader (ae60f01 behaviour, monkeypatching _rtstruct_masks scoped_reader=False)
    orig_rtstruct_masks = radiomics._rtstruct_masks

    def raw_rtstruct_masks(*args, **kwargs):
        kwargs["scoped_reader"] = False
        return orig_rtstruct_masks(*args, **kwargs)

    monkeypatch.setattr(radiomics, "_rtstruct_masks", raw_rtstruct_masks)
    rows_raw = radiomics._mr_manual_rows(
        config, series, img, extractor, source, parameter_provenance=provenance
    )

    assert len(rows_scoped) == len(rows_raw) == 2
    for r_scoped, r_raw in zip(rows_scoped, rows_raw):
        assert r_scoped["roi_name"] == r_raw["roi_name"]
        assert r_scoped["extraction_status"] == r_raw["extraction_status"] == "success"
        assert r_scoped["_firstorder_Mean"] == r_raw["_firstorder_Mean"]
        assert r_scoped["_firstorder_Energy"] == r_raw["_firstorder_Energy"]
        assert r_scoped["stable_roi_identifier"] == r_raw["stable_roi_identifier"]


def test_mr_totalsegmentator_arealess_contour_item_regression(tmp_path: Path):
    """Regression (a): MR TotalSegmentator RTSTRUCT with an area-less (1-point) contour item
    fails raw rt-utils / OpenCV fillPoly, but is successfully measured after fix
    with a mask identical to the scoped reader's."""
    mr_dir = tmp_path / "MR"
    series_uid, sop_uids = _write_synthetic_mr(mr_dir, rows=16, columns=16, slices=5)

    mask_femur = cylinder((16, 16, 5), (8, 8, 2), (4, 4, 1))
    seg_dir = tmp_path / "Segmentation_TotalSegmentator"
    seg_dir.mkdir(parents=True)
    rs_path = _write_mr_rtstruct(
        mr_dir, seg_dir / "series--total_mr.dcm", [("femur_left", mask_femur)]
    )

    # Append an area-less 1-point contour item to femur_left
    ds = pydicom.dcmread(str(rs_path))
    contour = Dataset()
    contour.ContourGeometricType = "CLOSED_PLANAR"
    contour.NumberOfContourPoints = 1
    contour.ContourData = [5.0, 5.0, 6.0]
    ref = Dataset()
    ref.ReferencedSOPClassUID = MRImageStorage
    ref.ReferencedSOPInstanceUID = sop_uids[2]
    contour.ContourImageSequence = DicomSequence([ref])
    ds.ROIContourSequence[0].ContourSequence.append(contour)
    ds.save_as(str(rs_path), enforce_file_format=True)

    # 1. On raw rt-utils (ae60f01), reading femur_left raises OpenCV assertion error
    raw_rt = RTStructBuilder.create_from(str(mr_dir), str(rs_path))
    with pytest.raises(Exception) as exc_info:
        raw_rt.get_roi_mask_by_name("femur_left")
    assert "fillPoly" in str(exc_info.value) or "checkVector" in str(exc_info.value)

    # 2. After fix: _collect_total_mr_masks measures the ROI without failure
    failures: List[Dict[str, str]] = []
    masks = radiomics._collect_total_mr_masks(mr_dir, seg_dir, failures)

    assert "femur_left" in masks
    assert not failures
    assert masks["femur_left"].sum() > 0

    # Mask matches the scoped reader and the expected ground truth cylinder
    scoped = create_scoped_rtstruct(mr_dir, rs_path)
    expected_mask = scoped.get_roi_mask_by_name("femur_left")
    assert np.array_equal(masks["femur_left"], expected_mask)
    assert np.array_equal(masks["femur_left"], mask_femur)


def test_mr_rtstruct_unresolved_scope_regression(tmp_path: Path):
    """Regression (b): MR RTSTRUCT referencing an SOP not in the MR series
    fails raw rt-utils as a whole file; after fix, normal ROI is measured and
    the out-of-scope ROI has governed structural_nonmeasurement disposition."""
    mr_dir = tmp_path / "MR"
    series_uid, sop_uids = _write_synthetic_mr(mr_dir, rows=16, columns=16, slices=5)

    mask_normal = cylinder((16, 16, 5), (8, 8, 2), (3, 3, 1))
    mask_foreign = cylinder((16, 16, 5), (4, 4, 2), (2, 2, 1))
    rs_path = _write_mr_rtstruct(
        mr_dir, tmp_path / "RS.dcm", [("Normal_ROI", mask_normal), ("Foreign_ROI", mask_foreign)]
    )

    # Rewrite Foreign_ROI's contour reference to an out-of-series SOP
    ds = pydicom.dcmread(str(rs_path))
    foreign_sop = generate_uid()
    for contour in ds.ROIContourSequence[1].ContourSequence:
        contour.ContourImageSequence[0].ReferencedSOPInstanceUID = foreign_sop
    # Also update study reference sequence
    ref_series = ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0]
    foreign_ref = Dataset()
    foreign_ref.ReferencedSOPClassUID = MRImageStorage
    foreign_ref.ReferencedSOPInstanceUID = foreign_sop
    ref_series.ContourImageSequence.append(foreign_ref)
    ds.save_as(str(rs_path), enforce_file_format=True)

    # 1. On raw rt-utils (ae60f01), RTStructBuilder.create_from fails the WHOLE file
    with pytest.raises(Exception) as exc_info:
        RTStructBuilder.create_from(str(mr_dir), str(rs_path))
    assert "not contained in input series data" in str(exc_info.value) or "SOP" in str(exc_info.value)

    # 2. In _mr_manual_rows (after fix):
    config = PipelineConfig(
        dicom_root=tmp_path, output_root=tmp_path / "out", logs_root=tmp_path / "logs"
    )
    series = radiomics.MRSeries(
        patient_id="SYNTH_MR", series_uid=series_uid, dir=mr_dir
    )
    img = sitk.Image([16, 16, 5], sitk.sitkInt16)
    extractor = _MockMRExtractor()
    source = radiomics._MRMaskSource(
        "manual", rs_path, (rs_path,), radiomics._mr_content_digest([rs_path])
    )
    provenance = {
        "extraction_arm": "MR:default",
        "run_identifier": "run-test-id",
        "code_revision": "test-rev",
    }

    rows = radiomics._mr_manual_rows(
        config, series, img, extractor, source, parameter_provenance=provenance
    )

    by_name = {r["roi_name"]: r for r in rows}
    assert "Normal_ROI" in by_name
    assert by_name["Normal_ROI"]["extraction_status"] == "success"
    assert by_name["Normal_ROI"]["_firstorder_Mean"] == 42.0

    assert "Foreign_ROI" in by_name
    foreign_row = by_name["Foreign_ROI"]
    assert foreign_row["extraction_status"] == "structural_nonmeasurement"
    assert foreign_row["extraction_failure_kind"] == "unresolved_source_scope"
    assert foreign_row["roi_structural_code"] == "ROI_UNRESOLVED_SOURCE_SCOPE"
    assert "does not resolve against" in foreign_row["extraction_status_detail"] or "structural status" in foreign_row["extraction_status_detail"]

    # 3. In _collect_total_mr_masks (TotalSegmentator path):
    seg_dir = tmp_path / "Segmentation_TotalSegmentator"
    seg_dir.mkdir(parents=True, exist_ok=True)
    ts_path = seg_dir / "series--total_mr.dcm"
    ds.save_as(str(ts_path), enforce_file_format=True)

    source_failures: List[Dict[str, str]] = []
    masks = radiomics._collect_total_mr_masks(mr_dir, seg_dir, source_failures)
    assert "Normal_ROI" in masks
    assert "Foreign_ROI" not in masks
    assert any(
        f.get("roi_name") == "Foreign_ROI"
        and f.get("status") == "structural_nonmeasurement"
        and f.get("disposition") == "structural_nonmeasurement"
        and f.get("reason_code") == "ROI_UNRESOLVED_SOURCE_SCOPE"
        for f in source_failures
    )


def test_mr_unreadable_rtstruct_negative_control(tmp_path: Path):
    """Negative control: a corrupted/unreadable RTSTRUCT still fails as before."""
    mr_dir = tmp_path / "MR"
    series_uid, _ = _write_synthetic_mr(mr_dir, rows=16, columns=16, slices=5)

    rs_corrupt = tmp_path / "RS_corrupt.dcm"
    rs_corrupt.write_bytes(b"not a valid dicom file at all")

    config = PipelineConfig(
        dicom_root=tmp_path, output_root=tmp_path / "out", logs_root=tmp_path / "logs"
    )
    series = radiomics.MRSeries(
        patient_id="SYNTH_MR", series_uid=series_uid, dir=mr_dir
    )
    img = sitk.Image([16, 16, 5], sitk.sitkInt16)
    extractor = _MockMRExtractor()
    source = radiomics._MRMaskSource(
        "manual", rs_corrupt, (rs_corrupt,), radiomics._mr_content_digest([rs_corrupt])
    )

    with pytest.raises(RadiomicsCourseExtractionError) as exc_info:
        radiomics._mr_manual_rows(
            config, series, img, extractor, source, parameter_provenance={}
        )
    assert "no usable identity" in str(exc_info.value) or "Cannot read" in str(exc_info.value)

    # In _collect_total_mr_masks, corrupt file is recorded as failed_source_read
    seg_dir = tmp_path / "Segmentation_TotalSegmentator"
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / "series--total_mr.dcm").write_bytes(b"corrupt bytes")
    failures: List[Dict[str, str]] = []
    masks = radiomics._collect_total_mr_masks(mr_dir, seg_dir, failures)
    assert masks == {}
    assert any(f.get("status") == "failed" and f.get("reason_code") == "failed_source_read" for f in failures)


def test_mr_unexplained_read_failure_negative_control(tmp_path: Path, monkeypatch):
    """Negative control: an ROI read failure NOT explained by scope/area-less items
    remains a failure."""
    mr_dir = tmp_path / "MR"
    series_uid, sop_uids = _write_synthetic_mr(mr_dir, rows=16, columns=16, slices=5)

    mask_target = cylinder((16, 16, 5), (8, 8, 2), (4, 4, 1))
    rs_path = _write_mr_rtstruct(mr_dir, tmp_path / "RS.dcm", [("Target", mask_target)])

    config = PipelineConfig(
        dicom_root=tmp_path, output_root=tmp_path / "out", logs_root=tmp_path / "logs"
    )
    series = radiomics.MRSeries(
        patient_id="SYNTH_MR", series_uid=series_uid, dir=mr_dir
    )
    img = sitk.Image([16, 16, 5], sitk.sitkInt16)
    extractor = _MockMRExtractor()
    source = radiomics._MRMaskSource(
        "manual", rs_path, (rs_path,), radiomics._mr_content_digest([rs_path])
    )

    # Simulate unexplained rasterization failure on RTStruct
    def broken_get_mask(self, name):
        raise RuntimeError("Unexplained synthetic rasterizer failure")

    monkeypatch.setattr(RTStruct, "get_roi_mask_by_name", broken_get_mask)

    # For optional / inventory-only ROI: records failed extraction
    monkeypatch.setattr(radiomics, "_roi_requiredness", lambda *a, **k: radiomics.Requiredness.INVENTORY_ONLY)
    rows = radiomics._mr_manual_rows(
        config, series, img, extractor, source, parameter_provenance={}
    )
    assert len(rows) == 1
    assert rows[0]["roi_name"] == "Target"
    assert rows[0]["extraction_status"] == "failed"
    assert rows[0]["extraction_failure_kind"] == "extraction_error"
    assert "Unexplained synthetic rasterizer failure" in rows[0]["extraction_status_detail"]

    # For required ROI: raises fail closed
    monkeypatch.setattr(radiomics, "_roi_requiredness", lambda *a, **k: radiomics.Requiredness.ANALYSIS_REQUIRED)
    with pytest.raises(RadiomicsCourseExtractionError, match="could not be extracted"):
        radiomics._mr_manual_rows(
            config, series, img, extractor, source, parameter_provenance={}
        )
