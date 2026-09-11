"""Manager-owned publication-boundary regressions for the executed MR helper.

These generated fixtures isolate source drift and failure accounting. They do
not validate DICOM-to-NIfTI numerical equivalence or real PyRadiomics extraction.
"""
from pathlib import Path
import json

import pydicom
import pytest

from rtpipeline import radiomics_conda as rc
from rtpipeline.radiomics_outcomes import RadiomicsCourseExtractionError
from test_mr_course_source_contract import (
    _Batch, _Config, _contract, _make_course, _make_series, _write_mask,
)


def _setup(tmp_path, monkeypatch):
    course = _make_course(tmp_path)
    source = _make_series(course)
    batch = _Batch()
    monkeypatch.setattr(rc, "process_radiomics_batch", batch)
    monkeypatch.setattr(rc, "check_radiomics_env", lambda *a, **k: True)
    return course, source, batch


@pytest.mark.parametrize("changed", ["image", "mask", "sidecar", "dicom", "new_mask"])
def test_live_source_drift_invalidates_nominal_mr_publication(tmp_path, monkeypatch, changed):
    course, source, batch = _setup(tmp_path, monkeypatch)
    output = course / "MR" / "radiomics_mr.xlsx"
    # Positive control: the unchanged generated source can be published.
    assert rc.radiomics_for_course_mr(course, _Config()) == output
    assert output.is_file() and output.with_suffix(".parquet").is_file()

    def mutate_after_extraction(tasks, output_path, **kwargs):
        result = batch(tasks, output_path, **kwargs)
        if changed == "image":
            # Alter bytes without changing the file path or recorded series UID.
            path = source["nifti_path"]
            path.write_bytes(path.read_bytes() + b"source changed during extraction")
        elif changed == "mask":
            _write_mask(source["series_root"], "liver", 55)
        elif changed == "sidecar":
            path = source["sidecar"]
            metadata = json.loads(path.read_text(encoding="utf-8"))
            metadata["source_note"] = "changed during extraction"
            path.write_text(json.dumps(metadata), encoding="utf-8")
        elif changed == "dicom":
            path = sorted(source["dicom_dir"].glob("*.dcm"))[0]
            dataset = pydicom.dcmread(path)
            original_uid = str(dataset.SOPInstanceUID)
            dataset.SeriesDescription = "changed during extraction"
            dataset.save_as(path, enforce_file_format=True)
            assert str(pydicom.dcmread(path).SOPInstanceUID) == original_uid
        else:
            _write_mask(source["series_root"], "newly_arrived_roi", 30)
        return result

    monkeypatch.setattr(rc, "process_radiomics_batch", mutate_after_extraction)
    with pytest.raises(RadiomicsCourseExtractionError):
        rc.radiomics_for_course_mr(course, _Config())
    assert not output.exists()
    assert not output.with_suffix(".parquet").exists()
    assert not (course / "MR" / "radiomics_mr_checkpoint.parquet").exists()


def test_required_mr_absence_is_not_optional_modality_skip(tmp_path, monkeypatch):
    course = tmp_path / "PAT" / "COURSE"
    course.mkdir(parents=True)
    batch = _Batch()
    monkeypatch.setattr(rc, "process_radiomics_batch", batch)
    with pytest.raises(RadiomicsCourseExtractionError):
        rc.radiomics_for_course_mr(course, _Config(contract=_contract(required=["liver"])))
    assert batch.calls == 0


def test_mixed_dicom_study_identity_cannot_become_blank_and_pass(tmp_path, monkeypatch):
    course, source, batch = _setup(tmp_path, monkeypatch)
    path = sorted(source["dicom_dir"].glob("*.dcm"))[-1]
    dataset = pydicom.dcmread(path)
    dataset.StudyInstanceUID = "1.2.826.0.1.3680043.9.7.999"
    dataset.save_as(path, enforce_file_format=True)
    assert rc.radiomics_for_course_mr(course, _Config()) is None
    assert batch.calls == 0
    assert not (course / "MR" / "radiomics_mr.parquet").exists()


def test_configuration_binding_exception_invalidates_previous_mr_output(tmp_path, monkeypatch):
    course, _source, _batch = _setup(tmp_path, monkeypatch)
    output = course / "MR" / "radiomics_mr.xlsx"
    assert rc.radiomics_for_course_mr(course, _Config()) == output

    def fail_binding(*args, **kwargs):
        raise RuntimeError("synthetic configuration binding failure")

    monkeypatch.setattr(rc, "configured_parameter_hash", fail_binding)
    with pytest.raises((RuntimeError, RadiomicsCourseExtractionError), match="configuration binding"):
        rc.radiomics_for_course_mr(course, _Config())
    assert not output.exists()
    assert not output.with_suffix(".parquet").exists()
