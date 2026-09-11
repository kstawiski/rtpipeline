"""Synthetic regressions for explicit RTSTRUCT nonmeasurement handling.

These checks exercise inventory, identity, and publication plumbing. The mask
reader and extractor are mocked. They do not establish real feature extraction.
"""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from pydicom.sequence import Sequence
from rt_utils import RTStructBuilder

from course_contract_test_utils import (
    write_minimal_course_contract,
    write_synthetic_planning_ct,
    write_synthetic_rtstruct,
)
from rtpipeline import radiomics
from rtpipeline.config import PipelineConfig
from rtpipeline.radiomics_ct_contract import CT_EXTRACTION_ARMS
from rtpipeline.radiomics_outcomes import RadiomicsCourseExtractionError


def _source(path, *, only_point=False, malformed=False):
    names = ["Marker"] if only_point else ["Target", "Marker"]
    write_synthetic_rtstruct(path, roi_names=names)
    dataset = pydicom.dcmread(path)
    contour = dataset.ROIContourSequence[-1].ContourSequence[0]
    contour.ContourGeometricType = "POINT"
    contour.NumberOfContourPoints = 1
    contour.ContourData = [1.0, 1.0, 1.0]
    if malformed:
        contour.ContourGeometricType = "CLOSED_PLANAR"
    dataset.save_as(path, enforce_file_format=True)
    return dataset


def _reader(monkeypatch, names):
    seen = []

    def mask(name):
        seen.append(name)
        if name != "Target":
            raise AssertionError("Non-volumetric ROI reached rasterizer")
        return np.ones((2, 2, 2), dtype=bool)

    monkeypatch.setattr(
        RTStructBuilder, "create_from",
        lambda **_: SimpleNamespace(get_roi_names=lambda: names, get_roi_mask_by_name=mask),
    )
    return seen


def test_expected_inventory_does_not_rasterize_recorded_point(tmp_path, monkeypatch):
    source = tmp_path / "rtstruct.dcm"
    dataset = _source(source)
    before = source.read_bytes()
    seen = _reader(monkeypatch, ["Target", "Marker"])
    dispositions = []
    masks = radiomics._rtstruct_masks(
        tmp_path / "CT", source,
        expected_rois=["Target", "Marker"], failure_outcomes=dispositions,
    )
    assert set(masks) == {"Target"}
    assert seen == ["Target"]
    assert len(dispositions) == 1
    assert dispositions[0]["status"] == "nonvolumetric_nonmeasurement"
    assert dispositions[0]["rtstruct_sop_instance_uid"] == str(dataset.SOPInstanceUID)
    assert dispositions[0]["roi_number"] == "2"
    assert dispositions[0]["source_path"] == str(source)
    assert source.read_bytes() == before


@pytest.mark.parametrize("expected", [["Marker", "Missing"], []])
def test_all_point_source_still_checks_expected_inventory(tmp_path, monkeypatch, expected):
    source = tmp_path / "rtstruct.dcm"
    _source(source, only_point=True)
    monkeypatch.setattr(RTStructBuilder, "create_from", lambda **_: pytest.fail("No rasterizer needed"))
    with pytest.raises(RadiomicsCourseExtractionError, match="inventory"):
        radiomics._rtstruct_masks(
            tmp_path / "CT", source, expected_rois=expected, failure_outcomes=[],
        )


def test_all_point_duplicate_names_still_fail_expected_inventory(tmp_path):
    source = tmp_path / "rtstruct.dcm"
    dataset = _source(source, only_point=True)
    dataset.StructureSetROISequence = Sequence([
        dataset.StructureSetROISequence[0], dataset.StructureSetROISequence[0],
    ])
    dataset.save_as(source, enforce_file_format=True)
    with pytest.raises(RadiomicsCourseExtractionError):
        radiomics._rtstruct_masks(
            tmp_path / "CT", source, expected_rois=["Marker"], failure_outcomes=[],
        )


def test_explicit_skip_does_not_add_second_nonmeasurement(tmp_path, monkeypatch):
    source = tmp_path / "rtstruct.dcm"
    _source(source)
    seen = _reader(monkeypatch, ["Target", "Marker"])
    dispositions = []
    masks = radiomics._rtstruct_masks(
        tmp_path / "CT", source, expected_rois=["Target", "Marker"],
        skip_rois={"marker"}, failure_outcomes=dispositions,
    )
    assert set(masks) == {"Target"}
    assert seen == ["Target"]
    assert dispositions == []


def test_malformed_required_geometry_is_not_nonmeasurement(tmp_path):
    source = tmp_path / "rtstruct.dcm"
    _source(source, only_point=True, malformed=True)
    dispositions = []
    with pytest.raises(RadiomicsCourseExtractionError, match="UNPARSEABLE"):
        radiomics._rtstruct_masks(
            tmp_path / "CT", source, expected_rois=["Marker"], failure_outcomes=dispositions,
        )
    assert dispositions[0]["status"] == "failed"


class _Extractor:
    def __init__(self):
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

    def execute(self, *_):
        raise AssertionError("A point source must never invoke feature extraction")


def _course(tmp_path, monkeypatch, *, malformed=False):
    course = tmp_path / "P1" / "C1"
    ct = write_synthetic_planning_ct(course)
    write_minimal_course_contract(course, planning_ct_dir=ct)
    model_dir = course / "Segmentation_CustomModels" / "PointModel"
    model_dir.mkdir(parents=True)
    source = model_dir / "rtstruct.dcm"
    dataset = _source(source, only_point=True, malformed=malformed)
    # Keep the existing required-ROI classification gate in force. This fixture
    # uses an approved name with non-volumetric geometry, not an invented class.
    dataset.StructureSetROISequence[0].ROIName = "PTV"
    dataset.save_as(source, enforce_file_format=True)
    monkeypatch.setattr(radiomics, "_standard_rtstruct_sources", lambda *_: [])
    monkeypatch.setattr(radiomics, "_load_series_image", lambda *_: sitk.Image([2, 2, 2], sitk.sitkInt16))
    monkeypatch.setattr(radiomics, "_extractor", lambda *_a, **_k: _Extractor())
    monkeypatch.setattr(radiomics, "current_code_revision", lambda: "synthetic-test")
    monkeypatch.setattr(radiomics, "validate_custom_model_output_inventory", lambda *_: {"PointModel": ["PTV"]})
    monkeypatch.setattr(radiomics, "list_custom_model_outputs", lambda *_: [("PointModel", model_dir)])
    config = PipelineConfig(
        dicom_root=tmp_path / "input", output_root=tmp_path,
        logs_root=tmp_path / "logs", max_workers_override=1,
    )
    return course, source, dataset, config


def test_custom_model_point_has_persisted_identity_bound_arm_rows(tmp_path, monkeypatch):
    course, source, dataset, config = _course(tmp_path, monkeypatch)
    before = source.read_bytes()
    radiomics.radiomics_for_course(config, course)
    frame = radiomics.read_authoritative_ct_publication(course / "radiomics_ct.parquet")
    assert len(frame) == 2
    assert set(frame.extraction_arm) == set(CT_EXTRACTION_ARMS)
    assert set(frame.extraction_status) == {"nonvolumetric_nonmeasurement"}
    assert set(frame.roi_structural_code) == {"ROI_NONVOLUMETRIC_POINT"}
    assert set(frame.segmentation_source) == {"CustomModel:PointModel"}
    assert set(frame.rtstruct_sop_instance_uid) == {str(dataset.SOPInstanceUID)}
    assert set(frame.stable_roi_identifier) == {"rtstruct_roi_number:1"}
    assert set(frame.roi_original_name) == {"PTV"}
    assert frame.run_identifier.str.len().gt(0).all()
    assert frame.configured_parameter_hash.str.len().gt(0).all()
    assert source.read_bytes() == before


def test_custom_model_malformed_source_invalidates_stale_output(tmp_path, monkeypatch):
    course, _, _, config = _course(tmp_path, monkeypatch, malformed=True)
    for suffix in (".xlsx", ".parquet"):
        (course / f"radiomics_ct{suffix}").write_bytes(b"stale")
    with pytest.raises(RadiomicsCourseExtractionError, match="UNPARSEABLE"):
        radiomics.radiomics_for_course(config, course)
    assert not (course / "radiomics_ct.xlsx").exists()
    assert not (course / "radiomics_ct.parquet").exists()


@pytest.mark.parametrize("empty_kind", ["missing_item", "empty_sequence"])
def test_empty_required_custom_model_invalidates_stale_output(tmp_path, monkeypatch, empty_kind):
    course, source, dataset, config = _course(tmp_path, monkeypatch)
    if empty_kind == "missing_item":
        dataset.ROIContourSequence = Sequence([])
    else:
        dataset.ROIContourSequence[0].ContourSequence = Sequence([])
    dataset.save_as(source, enforce_file_format=True)
    for suffix in (".xlsx", ".parquet"):
        (course / f"radiomics_ct{suffix}").write_bytes(b"stale")
    with pytest.raises(RadiomicsCourseExtractionError, match="ROI_DECLARED_"):
        radiomics.radiomics_for_course(config, course)
    assert not (course / "radiomics_ct.xlsx").exists()
    assert not (course / "radiomics_ct.parquet").exists()


@pytest.mark.parametrize("source_kind", ["custom_model", "standard"])
def test_point_disposition_is_not_measurement_or_technical_exclusion(tmp_path, monkeypatch, source_kind):
    course, source, _, config = _course(tmp_path, monkeypatch)
    expected_source = "CustomModel:PointModel"
    if source_kind == "standard":
        expected_source = "AutoRTS_total"
        monkeypatch.setattr(radiomics, "_standard_rtstruct_sources", lambda *_: [(expected_source, source, ["PTV"])])
        monkeypatch.setattr(radiomics, "validate_custom_model_output_inventory", lambda *_: {})
        monkeypatch.setattr(radiomics, "list_custom_model_outputs", lambda *_: [])
    outcome = radiomics.radiomics_for_course(config, course)
    frame = radiomics.read_authoritative_ct_publication(course / "radiomics_ct.parquet")
    assert frame.extraction_status.eq("nonvolumetric_nonmeasurement").all()
    assert frame.native_mask_voxel_count.isna().all()
    assert outcome.extracted_count == 0
    assert outcome.failed_count == 0
    assert outcome.attempted == 0  # No volumetric extraction was attempted.
    assert not outcome.roi_failures
    ledger = json.loads((course / "metadata" / "radiomics_ct_roi_ledger.json").read_text())
    assert len(ledger["course_roi"]) == 1
    row = ledger["course_roi"][0]
    assert row["segmentation_source"] == expected_source
    assert row["reason_code"] == "ROI_NONVOLUMETRIC_POINT"
    assert row["disposition"] == "nonmeasurement"
    assert ledger["course"][0]["extracted"] is False
    assert ledger["course"][0]["technical_exclusion"] is False
    assert ledger["course"][0]["reason_code"] == "nonvolumetric_nonmeasurement"
    summary = json.loads((course / "metadata" / "radiomics_ct_denominators.json").read_text())
    counts = summary["COURSE_ROI"]["CT:PTV"]
    assert counts["extracted"] == 0
    assert counts["excluded_technical"] == 0
    assert counts["excluded_anatomy"] == 0
    assert counts["nonmeasurement"] == 1
    rebuilt = radiomics.outcome_from_output(course / "radiomics_ct.xlsx")
    assert (rebuilt.attempted, rebuilt.extracted_count, rebuilt.failed_count) == (0, 0, 0)
    assert outcome.status == rebuilt.status


@pytest.mark.parametrize("change", ["source_bytes", "code_revision", "unchanged"])
def test_nonmeasurement_resume_checks_content_and_rebuilds_ledger(tmp_path, monkeypatch, change):
    course, source, dataset, config = _course(tmp_path, monkeypatch)
    radiomics.radiomics_for_course(config, course)
    first = radiomics.read_authoritative_ct_publication(course / "radiomics_ct.parquet")
    first_run = set(first.run_identifier)
    if change == "source_bytes":
        # Same SOP UID, same shape class, but different source bytes.
        dataset.ROIContourSequence[0].ContourSequence[0].ContourData = [2.0, 1.0, 1.0]
        dataset.save_as(source, enforce_file_format=True)
    elif change == "code_revision":
        monkeypatch.setattr(radiomics, "current_code_revision", lambda: "synthetic-next")
    ledger_path = course / "metadata" / "radiomics_ct_roi_ledger.json"
    ledger_path.unlink()
    config.resume = True
    radiomics.radiomics_for_course(config, course)
    final = radiomics.read_authoritative_ct_publication(course / "radiomics_ct.parquet")
    if change == "unchanged":
        assert set(final.run_identifier) == first_run
    else:
        assert set(final.run_identifier) != first_run
    assert set(final.rtstruct_sop_instance_uid) == {str(dataset.SOPInstanceUID)}
    assert set(final.source_content_sha256) == {hashlib.sha256(source.read_bytes()).hexdigest()}
    ledger = json.loads(ledger_path.read_text())
    assert ledger["course"][0]["extracted"] is False
    assert ledger["course"][0]["technical_exclusion"] is False
    assert len(ledger["course_roi"]) == 1
    assert ledger["course_roi"][0]["reason_code"] == "ROI_NONVOLUMETRIC_POINT"


@pytest.mark.parametrize("skip,alias", [(False, "PTV"), (False, "ptv"), (True, "PTV")])
def test_required_alias_inherits_actual_disposition_not_name_presence(tmp_path, monkeypatch, skip, alias):
    course, source, _, config = _course(tmp_path, monkeypatch)
    monkeypatch.setattr(radiomics, "_standard_rtstruct_sources", lambda *_: [("Manual", source, ["PTV"])])
    monkeypatch.setattr(radiomics, "validate_custom_model_output_inventory", lambda *_: {})
    monkeypatch.setattr(radiomics, "list_custom_model_outputs", lambda *_: [])
    config.radiomics_analysis_contract = {"CT": {"required_rois": [{
        "canonical_name": "PlanningTarget", "approved_aliases": [alias],
        "source": "Manual",
    }]}}
    if skip:
        config.radiomics_skip_rois = ["PTV"]
    radiomics.radiomics_for_course(config, course)
    payload = json.loads((course / "metadata" / "radiomics_ct_roi_ledger.json").read_text())
    canonical = [row for row in payload["course_roi"] if row["roi_name"] == "PlanningTarget"]
    assert len(canonical) == 1
    expected = "CONFIGURED_SKIP" if skip else "ROI_NONVOLUMETRIC_POINT"
    assert canonical[0]["reason_code"] == expected
    assert canonical[0]["disposition"] != "extracted"
    assert canonical[0]["alias_used"] is True
    assert canonical[0]["segmentation_source"] == "Manual"
    summary = json.loads((course / "metadata" / "radiomics_ct_denominators.json").read_text())
    assert summary["COURSE_ROI"]["CT:PlanningTarget"]["extracted"] == 0
    assert summary["COURSE_ROI"]["CT:PlanningTarget"]["excluded_technical"] == 0


@pytest.mark.parametrize("required_source", ["Manual", None])
def test_alias_resolution_respects_source_scope_and_ambiguity(tmp_path, monkeypatch, required_source):
    course, source, _, config = _course(tmp_path, monkeypatch)
    monkeypatch.setattr(radiomics, "_standard_rtstruct_sources", lambda *_: [
        ("Manual", source, ["PTV"]), ("CustomModel:PointModel", source, ["PTV"]),
    ])
    monkeypatch.setattr(radiomics, "validate_custom_model_output_inventory", lambda *_: {})
    monkeypatch.setattr(radiomics, "list_custom_model_outputs", lambda *_: [])
    config.radiomics_analysis_contract = {"CT": {"required_rois": [{
        "canonical_name": "PlanningTarget", "approved_aliases": ["PTV"],
        "source": required_source,
    }]}}
    radiomics.radiomics_for_course(config, course)
    payload = json.loads((course / "metadata" / "radiomics_ct_roi_ledger.json").read_text())
    canonical = [row for row in payload["course_roi"] if row["roi_name"] == "PlanningTarget"]
    assert len(canonical) == 1
    expected = "ROI_NONVOLUMETRIC_POINT" if required_source else "REQUIRED_ROI_AMBIGUOUS_MATCH"
    assert canonical[0]["reason_code"] == expected
    assert canonical[0]["disposition"] != "extracted"
    if required_source:
        assert canonical[0]["segmentation_source"] == required_source


def test_optional_mask_failure_has_one_source_roi_denominator_row(tmp_path, monkeypatch):
    course, source, _, config = _course(tmp_path, monkeypatch)
    write_synthetic_rtstruct(source, roi_names=("vertebrae_T8",))
    monkeypatch.setattr(radiomics, "_standard_rtstruct_sources", lambda *_: [("AutoRTS_total", source, ["vertebrae_T8"])])
    monkeypatch.setattr(radiomics, "validate_custom_model_output_inventory", lambda *_: {})
    monkeypatch.setattr(radiomics, "list_custom_model_outputs", lambda *_: [])
    monkeypatch.setattr(RTStructBuilder, "create_from", lambda **_: SimpleNamespace(
        get_roi_names=lambda: ["vertebrae_T8"],
        get_roi_mask_by_name=lambda _: np.zeros((2, 2, 2), dtype=bool),
    ))
    outcome = radiomics.radiomics_for_course(config, course)
    assert outcome.failed_count == 1
    payload = json.loads((course / "metadata" / "radiomics_ct_roi_ledger.json").read_text())
    observed = [row for row in payload["course_roi"] if row["roi_name"] == "vertebrae_T8"]
    assert len(observed) == 1
    assert observed[0]["disposition"] == "excluded"
    summary = json.loads((course / "metadata" / "radiomics_ct_denominators.json").read_text())
    assert summary["COURSE_ROI"]["CT:vertebrae_T8"]["excluded_technical"] == 1


def test_source_fingerprint_read_failure_invalidates_stale_output(tmp_path, monkeypatch):
    course, source, _, config = _course(tmp_path, monkeypatch)
    monkeypatch.setattr(radiomics, "_standard_rtstruct_sources", lambda *_: [("AutoRTS_total", source, ["PTV"])])
    monkeypatch.setattr(radiomics, "validate_custom_model_output_inventory", lambda *_: {})
    monkeypatch.setattr(radiomics, "list_custom_model_outputs", lambda *_: [])
    original_read = type(source).read_bytes

    def read_bytes(path):
        if path == source:
            raise PermissionError("synthetic source hash read denied")
        return original_read(path)

    monkeypatch.setattr(type(source), "read_bytes", read_bytes)
    for suffix in (".xlsx", ".parquet"):
        (course / f"radiomics_ct{suffix}").write_bytes(b"stale")
    with pytest.raises(RadiomicsCourseExtractionError, match="synthetic source hash read denied"):
        radiomics.radiomics_for_course(config, course)
    assert not (course / "radiomics_ct.xlsx").exists()
    assert not (course / "radiomics_ct.parquet").exists()


def test_resume_ledger_failure_invalidates_admitted_table(tmp_path, monkeypatch):
    course, _, _, config = _course(tmp_path, monkeypatch)
    radiomics.radiomics_for_course(config, course)
    config.resume = True

    def fail_ledger(*_):
        raise RadiomicsCourseExtractionError("synthetic ledger write failure")

    monkeypatch.setattr(radiomics, "_write_course_roi_ledger", fail_ledger)
    with pytest.raises(RadiomicsCourseExtractionError, match="synthetic ledger write failure"):
        radiomics.radiomics_for_course(config, course)
    assert not (course / "radiomics_ct.xlsx").exists()
    assert not (course / "radiomics_ct.parquet").exists()
