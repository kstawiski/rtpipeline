"""Synthetic regressions for MR series disposition durability.

These checks exercise the durable-row plumbing of ``radiomics_for_mr_series``:
identity-bound nonmeasurement rows for non-volumetric ROIs, durable failed rows
for real read/extraction errors, and fail-closed handling of stale outputs.
The mask reader and extractor are mocked; real feature extraction and dataset
readiness are NOT established by these checks.
"""
import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pydicom
import pytest
import SimpleITK as sitk

from course_contract_test_utils import write_synthetic_rtstruct
from rtpipeline import radiomics
from rtpipeline.config import PipelineConfig
from rtpipeline.radiomics_outcomes import RadiomicsCourseExtractionError
from rt_utils import RTStructBuilder


PATIENT = "P1"
SERIES_UID = "1.2.3.4"
_IMAGE = object()


def _default_image():
    return sitk.Image([2, 2, 2], sitk.sitkInt16)


class _Extractor:
    """Stand-in for the pyradiomics extractor (never constructed in this env).

    Failure is keyed by masked voxel coordinate so the pool can pick the class
    up deterministically (no call-order or process-local state dependence).
    """

    def __init__(self, fail_mask_coord=None):
        self._fail_mask_coord = fail_mask_coord
        self.settings = {}
        self.enabledImagetypes = {"Original": {}}
        self.enabledFeatures = {"firstorder": ["Mean"]}

    def disableAllImageTypes(self):
        self.enabledImagetypes.clear()

    def enableImageTypeByName(self, name):
        self.enabledImagetypes[name] = {}

    def disableAllFeatures(self):
        self.enabledFeatures.clear()

    def enableFeatureClassByName(self, name):
        self.enabledFeatures[name] = []

    def execute(self, img, m_img, *extra):
        if self._fail_mask_coord is not None:
            mask = sitk.GetArrayFromImage(m_img)
            for coord in np.argwhere(mask != 0):
                if tuple(int(v) for v in coord) == self._fail_mask_coord:
                    raise RuntimeError("synthetic extraction failure")
        return {"_firstorder_Mean": 12.5}


def _write_rs(path, *, names=("Target", "Marker")):
    """Synthetic RTSTRUCT; the last ROI is rewritten to POINT geometry."""
    write_synthetic_rtstruct(path, roi_names=list(names))
    dataset = pydicom.dcmread(path)
    contour = dataset.ROIContourSequence[-1].ContourSequence[0]
    contour.ContourGeometricType = "POINT"
    contour.NumberOfContourPoints = 1
    contour.ContourData = [1.0, 1.0, 1.0]
    dataset.save_as(path, enforce_file_format=True)
    return path


def _reader(monkeypatch, names):
    seen = []

    def mask(name):
        seen.append(name)
        if name != "Target":
            raise AssertionError("Non-volumetric ROI reached rasterizer")
        return np.ones((2, 2, 2), dtype=bool)

    monkeypatch.setattr(
        RTStructBuilder, "create_from",
        lambda **_: SimpleNamespace(
            get_roi_names=lambda: names,
            get_roi_mask_by_name=mask,
        ),
    )
    return seen


def _env(monkeypatch, tmp_path, *, fail_mask_coord=None, image=_IMAGE):
    series_dir = tmp_path / "input" / PATIENT
    series_dir.mkdir(parents=True)
    if image is _IMAGE:
        image = _default_image()
    rs = _write_rs(tmp_path / "input" / "RS_manual.dcm")
    out_root = tmp_path / PATIENT / f"MR_{SERIES_UID}"
    out_feat = out_root / "radiomics_features_MR.xlsx"
    extractor = _Extractor(fail_mask_coord=fail_mask_coord)
    monkeypatch.setattr(radiomics, "_extractor", lambda *_a, **_k: extractor)
    monkeypatch.setattr(radiomics, "_load_series_image", lambda *_a, **_k: image)
    monkeypatch.setattr(radiomics, "_find_mr_manual_rs", lambda *_a, **_k: [rs])
    monkeypatch.setattr(radiomics, "_mr_series_for_uid", lambda *_a, **_k: "1.9")
    monkeypatch.setattr(radiomics, "_infer_mr_weighting", lambda *_a, **_k: "T1")
    monkeypatch.setattr(radiomics, "current_code_revision", lambda: "synthetic-test")
    import rtpipeline.auto_rtstruct as auto_rtstruct
    monkeypatch.setattr(auto_rtstruct, "_load_seg_dicom", None)
    monkeypatch.setattr(auto_rtstruct, "_load_seg_nifti", None)
    monkeypatch.setattr(
        RTStructBuilder, "create_from",
        lambda **_: SimpleNamespace(
            get_roi_names=lambda: ["Target"],
            get_roi_mask_by_name=lambda _: np.ones((2, 2, 2), dtype=bool),
        ),
    )
    config = PipelineConfig(
        dicom_root=tmp_path / "input", output_root=tmp_path,
        logs_root=tmp_path / "logs", max_workers_override=1,
    )
    series = radiomics.MRSeries(
        patient_id=PATIENT, series_uid=SERIES_UID, dir=series_dir,
    )
    return config, series, rs, out_feat


def _plant_stale(out_feat):
    out_feat.parent.mkdir(parents=True, exist_ok=True)
    (out_feat.with_suffix(".parquet")).write_bytes(b"stale")
    out_feat.write_bytes(b"stale")


def _read(out_feat):
    return pd.read_parquet(out_feat.with_suffix(".parquet"), engine="pyarrow")


def test_nonvolumetric_roi_persists_identity_bound_nonmeasurement(tmp_path, monkeypatch):
    config, series, rs, out_feat = _env(monkeypatch, tmp_path)
    result = radiomics.radiomics_for_mr_series(config, series)
    assert result == out_feat
    frame = _read(out_feat)
    assert len(frame) == 2
    marker = frame[frame.roi_name == "Marker"]
    assert len(marker) == 1
    marker = marker.iloc[0]
    assert marker.extraction_status == "nonvolumetric_nonmeasurement"
    assert marker.extraction_failure_kind == "nonvolumetric_geometry"
    assert marker.roi_structural_code == "ROI_NONVOLUMETRIC_POINT"
    assert marker.extraction_status_detail == "ROI_NONVOLUMETRIC_POINT"
    # Source / ROI identity: SOP UID bound, ROI number folded into the stable
    # identifier, content fingerprint of the exact source bytes.
    assert marker.rtstruct_sop_instance_uid == str(pydicom.dcmread(rs).SOPInstanceUID)
    assert marker.stable_roi_identifier == "rtstruct_roi_number:2"
    assert marker.source_content_sha256 == hashlib.sha256(rs.read_bytes()).hexdigest()
    assert marker.segmentation_source == "Manual"
    assert marker.modality == "MR"
    assert marker.series_uid == SERIES_UID
    # No feature measurements on a nonmeasurement row.
    assert pd.isna(marker["_firstorder_Mean"])
    # Existing MR parameter/run provenance is present on the durable row.
    for col in ("extraction_arm", "configured_parameter_hash", "effective_parameter_hash",
                "run_identifier", "code_revision"):
        assert str(marker[col])
    target = frame[frame.roi_name == "Target"].iloc[0]
    assert target.extraction_status == "success"
    assert float(target["_firstorder_Mean"]) == 12.5
    assert target.source_content_sha256 == hashlib.sha256(rs.read_bytes()).hexdigest()
    # No raw sink fields leak into the publication; the ROI number is folded
    # into stable_roi_identifier.
    assert frame.columns.intersection({"roi_number", "status", "failure_kind", "reason"}).empty


def test_manual_extraction_failure_persists_durable_failed_row(tmp_path, monkeypatch):
    config, series, rs, out_feat = _env(monkeypatch, tmp_path, fail_mask_coord=(0, 0, 0))
    _plant_stale(out_feat)
    result = radiomics.radiomics_for_mr_series(config, series)
    assert result == out_feat
    frame = _read(out_feat)
    assert len(frame) == 2
    target = frame[frame.roi_name == "Target"].iloc[0]
    assert target.extraction_status == "failed"
    assert target.extraction_failure_kind == "extraction_error"
    assert "synthetic extraction failure" in str(target.extraction_status_detail)
    assert target.rtstruct_sop_instance_uid == str(pydicom.dcmread(rs).SOPInstanceUID)
    assert target.source_content_sha256 == hashlib.sha256(rs.read_bytes()).hexdigest()
    assert target.stable_roi_identifier == "rtstruct_roi_number:1"
    for col in ("extraction_arm", "configured_parameter_hash", "effective_parameter_hash",
                "run_identifier", "code_revision"):
        assert str(target[col])
    # The stale output from a previous run must not survive the failed attempt.
    assert frame.extraction_status.isin(["failed", "nonvolumetric_nonmeasurement"]).all()


def test_required_manual_roi_failure_is_not_relabelled_and_fails_closed(tmp_path, monkeypatch):
    config, series, rs, out_feat = _env(monkeypatch, tmp_path, fail_mask_coord=(0, 0, 0))
    config.radiomics_analysis_contract = {
        "MR": {"required_rois": [{"canonical_name": "Target", "source": "Manual"}]},
    }
    _plant_stale(out_feat)
    with pytest.raises(RadiomicsCourseExtractionError, match="Target"):
        radiomics.radiomics_for_mr_series(config, series)
    assert not out_feat.exists()
    assert not out_feat.with_suffix(".parquet").exists()


def test_empty_scope_invalidates_stale_output_and_fails_closed(tmp_path, monkeypatch):
    config, series, rs, out_feat = _env(monkeypatch, tmp_path)
    monkeypatch.setattr(radiomics, "_find_mr_manual_rs", lambda *_a, **_k: [])
    config.radiomics_analysis_contract = {
        "MR": {"required_rois": [{"canonical_name": "Target", "source": "Manual"}]},
    }
    _plant_stale(out_feat)
    with pytest.raises(RadiomicsCourseExtractionError):
        radiomics.radiomics_for_mr_series(config, series)
    assert not out_feat.exists()
    assert not out_feat.with_suffix(".parquet").exists()


@pytest.mark.parametrize("guard", ["no_image", "no_extractor"])
def test_series_level_technical_failure_invalidates_stale_output(tmp_path, monkeypatch, guard):
    config, series, rs, out_feat = _env(monkeypatch, tmp_path, image=None if guard == "no_image" else _default_image())
    if guard == "no_extractor":
        monkeypatch.setattr(radiomics, "_extractor", lambda *_a, **_k: None)
    _plant_stale(out_feat)
    assert radiomics.radiomics_for_mr_series(config, series) is None
    assert not out_feat.exists()
    assert not out_feat.with_suffix(".parquet").exists()


def test_auto_label_extraction_failure_persists_durable_failed_row(tmp_path, monkeypatch):
    config, series, rs, out_feat = _env(monkeypatch, tmp_path, fail_mask_coord=(1, 2, 2))
    import rtpipeline.auto_rtstruct as auto_rtstruct
    seg_arr = np.zeros((3, 3, 3), dtype=np.uint8)
    seg_arr[2, 2, 2] = 1  # liver
    seg_arr[1, 2, 2] = 2  # kidney
    monkeypatch.setattr(auto_rtstruct, "_load_seg_nifti", lambda *_a, **_k: (
        sitk.GetImageFromArray(seg_arr),
        {1: "liver", 2: "kidney"},
    ))
    monkeypatch.setattr(radiomics, "_resample_to_reference", lambda seg, ref, **_k: seg)
    nifti_dir = out_feat.parent / "TotalSegmentator_total_mr_NIFTI"
    nifti_dir.mkdir(parents=True)
    (nifti_dir / "total_mr--liver.nii.gz").write_bytes(b"synthetic")
    result = radiomics.radiomics_for_mr_series(config, series)
    assert result == out_feat
    frame = _read(out_feat)
    # Manual Target + Marker (nonmeasurement) plus both auto labels: the failed
    # label keeps a durable row instead of being silently omitted, while the
    # healthy label is still extracted.
    assert len(frame) == 4
    liver = frame[frame.roi_name == "liver"].iloc[0]
    assert liver.segmentation_source == "AutoTS_total_mr"
    assert liver.extraction_status == "success"
    assert float(liver["_firstorder_Mean"]) == 12.5
    kidney = frame[frame.roi_name == "kidney"].iloc[0]
    assert kidney.segmentation_source == "AutoTS_total_mr"
    assert kidney.extraction_status == "failed"
    assert kidney.extraction_failure_kind == "extraction_error"
    assert "synthetic extraction failure" in str(kidney.extraction_status_detail)
    for col in ("extraction_arm", "configured_parameter_hash", "effective_parameter_hash",
                "run_identifier", "code_revision"):
        assert str(kidney[col])


def _seg_array():
    array = np.zeros((3, 3, 3), dtype=np.uint8)
    array[2, 2, 2] = 1
    array[1, 2, 2] = 2
    return array


def _nifti_dir(out_feat, *, contents=b"synthetic nifti mask"):
    directory = out_feat.parent / "TotalSegmentator_total_mr_NIFTI"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "total_mr--liver.nii.gz").write_bytes(contents)
    return directory


def test_dicom_seg_is_authoritative_and_each_label_publishes_once(tmp_path, monkeypatch):
    """A course carrying both auto encodings publishes each label exactly once."""
    config, series, _rs, out_feat = _env(monkeypatch, tmp_path)
    import rtpipeline.auto_rtstruct as auto_rtstruct

    seg_dicom = out_feat.parent / "TotalSegmentator_total_mr_DICOM" / "segmentations.dcm"
    seg_dicom.parent.mkdir(parents=True, exist_ok=True)
    seg_dicom.write_bytes(b"synthetic dicom seg")
    _nifti_dir(out_feat)
    monkeypatch.setattr(auto_rtstruct, "_load_seg_dicom", lambda *_a, **_k: (
        sitk.GetImageFromArray(_seg_array()), {1: "liver", 2: "kidney"},
    ))
    monkeypatch.setattr(auto_rtstruct, "_load_seg_nifti", lambda *_a, **_k: (
        sitk.GetImageFromArray(_seg_array()), {1: "liver", 2: "kidney"},
    ))
    monkeypatch.setattr(radiomics, "_resample_to_reference", lambda seg, ref, **_k: seg)

    assert radiomics.radiomics_for_mr_series(config, series) == out_feat
    frame = _read(out_feat)
    auto = frame[frame.segmentation_source == "AutoTS_total_mr"]
    # Manual Target + Marker, and each TotalSegmentator label once - not twice
    # because the same labels are also encoded as NIfTI.
    assert len(frame) == 4
    assert sorted(auto.roi_name) == ["kidney", "liver"]
    assert set(auto.mask_path_source) == {str(seg_dicom)}
    assert set(auto.source_content_sha256) == {
        hashlib.sha256(seg_dicom.read_bytes()).hexdigest()
    }
    assert sorted(auto.stable_roi_identifier) == ["total_mr_label:1", "total_mr_label:2"]
    assert (auto.extraction_status == "success").all()


def test_nifti_source_identity_covers_every_member_and_binds_resume(tmp_path, monkeypatch):
    """The NIfTI directory identity is content-derived, and resume follows it."""
    config, series, _rs, out_feat = _env(monkeypatch, tmp_path)
    config.resume = True
    import rtpipeline.auto_rtstruct as auto_rtstruct

    directory = _nifti_dir(out_feat)
    monkeypatch.setattr(auto_rtstruct, "_load_seg_nifti", lambda *_a, **_k: (
        sitk.GetImageFromArray(_seg_array()), {1: "liver", 2: "kidney"},
    ))
    monkeypatch.setattr(radiomics, "_resample_to_reference", lambda seg, ref, **_k: seg)

    assert radiomics.radiomics_for_mr_series(config, series) == out_feat
    frame = _read(out_feat)
    auto = frame[frame.segmentation_source == "AutoTS_total_mr"]
    expected_digest = radiomics._mr_content_digest(
        [directory / "total_mr--liver.nii.gz"]
    )
    assert set(auto.mask_path_source) == {str(directory)}
    assert set(auto.source_content_sha256) == {expected_digest}
    run_ids = set(frame.run_identifier)

    # An unchanged rerun reuses the published bytes.
    published = out_feat.with_suffix(".parquet").read_bytes()
    assert radiomics.radiomics_for_mr_series(config, series) == out_feat
    assert out_feat.with_suffix(".parquet").read_bytes() == published

    # A new member file changes the source identity, so the stored table is stale.
    (directory / "total_mr--segmentations.json").write_text("{}", encoding="utf-8")
    assert radiomics.radiomics_for_mr_series(config, series) == out_feat
    refreshed = _read(out_feat)
    assert set(refreshed.run_identifier).isdisjoint(run_ids)
    assert set(refreshed[refreshed.segmentation_source == "AutoTS_total_mr"]
               .source_content_sha256) != {expected_digest}


def test_undecodable_auto_source_keeps_a_durable_technical_row(tmp_path, monkeypatch):
    """An auto source that cannot be decoded is recorded, never silently dropped."""
    config, series, _rs, out_feat = _env(monkeypatch, tmp_path)
    import rtpipeline.auto_rtstruct as auto_rtstruct

    directory = _nifti_dir(out_feat)
    monkeypatch.setattr(auto_rtstruct, "_load_seg_nifti", lambda *_a, **_k: (None, {}))

    assert radiomics.radiomics_for_mr_series(config, series) == out_feat
    frame = _read(out_feat)
    auto = frame[frame.segmentation_source == "AutoTS_total_mr"]
    assert len(auto) == 1
    row = auto.iloc[0]
    assert row.extraction_status == "failed"
    assert row.extraction_failure_kind == "source_read_error"
    assert row.mask_path_source == str(directory)
    assert row.source_content_sha256 == radiomics._mr_content_digest(
        [directory / "total_mr--liver.nii.gz"]
    )
    assert pd.isna(row["_firstorder_Mean"])
    # The manual rows are still published alongside the technical disposition.
    assert sorted(frame[frame.segmentation_source == "Manual"].roi_name) == [
        "Marker", "Target",
    ]


def test_nifti_loader_is_called_with_its_required_base_name_argument(tmp_path, monkeypatch):
    """Regression: the loader takes (seg_dir, base_name), not a single argument."""
    config, series, _rs, out_feat = _env(monkeypatch, tmp_path)
    import rtpipeline.auto_rtstruct as auto_rtstruct

    directory = _nifti_dir(out_feat)
    calls = []

    def _loader(seg_dir, base_name):
        calls.append((seg_dir, base_name))
        return sitk.GetImageFromArray(_seg_array()), {1: "liver", 2: "kidney"}

    monkeypatch.setattr(auto_rtstruct, "_load_seg_nifti", _loader)
    monkeypatch.setattr(radiomics, "_resample_to_reference", lambda seg, ref, **_k: seg)

    assert radiomics.radiomics_for_mr_series(config, series) == out_feat
    assert calls == [(directory, None)]
