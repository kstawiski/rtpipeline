"""Synthetic manager acceptance checks for the direct MR series helper.

All sources are generated in tmp_path. Extraction is mocked. The auto task
checks use the existing nested-function thread fallback, not a child pipeline.
"""
from __future__ import annotations

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from rtpipeline import radiomics
from rtpipeline import auto_rtstruct
from rtpipeline.radiomics_outcomes import RadiomicsCourseExtractionError
from rtpipeline.radiomics_schema import RadiomicsFeatureTypeError
from test_mr_series_dispositions import _env, _plant_stale, _read


@pytest.mark.parametrize("change", ["source_bytes", "code", "effective_settings"])
def test_mr_resume_rejects_changed_execution_identity(tmp_path, monkeypatch, change):
    config, series, source, output = _env(monkeypatch, tmp_path)
    config.resume = True
    radiomics.radiomics_for_mr_series(config, series)
    first = _read(output)
    run_ids = set(first.run_identifier)
    unchanged_bytes = output.with_suffix(".parquet").read_bytes()
    radiomics.radiomics_for_mr_series(config, series)
    assert output.with_suffix(".parquet").read_bytes() == unchanged_bytes
    assert set(_read(output).run_identifier) == run_ids
    if change == "source_bytes":
        ds = pydicom.dcmread(source)
        original_uid = str(ds.SOPInstanceUID)
        ds.StructureSetLabel = "CHANGED"
        ds.save_as(source)
        assert str(pydicom.dcmread(source).SOPInstanceUID) == original_uid
    elif change == "code":
        monkeypatch.setattr(radiomics, "current_code_revision", lambda: "changed-test-code")
    else:
        radiomics._extractor(config, "MR").settings["binWidth"] = 71
    radiomics.radiomics_for_mr_series(config, series)
    assert set(_read(output).run_identifier).isdisjoint(run_ids)


@pytest.mark.parametrize("during", ["masks", "extraction"])
def test_mr_source_drift_invalidates_publication(tmp_path, monkeypatch, during):
    config, series, source, output = _env(monkeypatch, tmp_path)
    config.resume = False
    _plant_stale(output)

    def change_source():
        ds = pydicom.dcmread(source)
        ds.StructureSetLabel = "CHANGED"
        ds.save_as(source)

    if during == "masks":
        original = radiomics._rtstruct_masks

        def mutate_masks(*args, **kwargs):
            result = original(*args, **kwargs)
            change_source()
            return result

        monkeypatch.setattr(radiomics, "_rtstruct_masks", mutate_masks)
    else:
        extractor = radiomics._extractor(config, "MR")
        original = extractor.execute

        def mutate_extraction(*args, **kwargs):
            result = original(*args, **kwargs)
            change_source()
            return result

        monkeypatch.setattr(extractor, "execute", mutate_extraction)

    with pytest.raises(RadiomicsCourseExtractionError):
        radiomics.radiomics_for_mr_series(config, series)
    assert not output.exists()
    assert not output.with_suffix(".parquet").exists()


def test_required_absent_roi_cannot_hide_behind_other_rows(tmp_path, monkeypatch):
    config, series, _, output = _env(monkeypatch, tmp_path)
    config.radiomics_analysis_contract = {
        "MR": {"required_rois": [{"canonical_name": "MissingRequired", "source": "Manual"}]}
    }
    _plant_stale(output)
    with pytest.raises(RadiomicsCourseExtractionError):
        radiomics.radiomics_for_mr_series(config, series)
    assert not output.exists()
    assert not output.with_suffix(".parquet").exists()


def test_new_required_contract_rejects_previous_optional_failure(tmp_path, monkeypatch):
    config, series, _, output = _env(monkeypatch, tmp_path, fail_mask_coord=(0, 0, 0))
    radiomics.radiomics_for_mr_series(config, series)
    config.resume = True
    config.radiomics_analysis_contract = {
        "MR": {"required_rois": [{"canonical_name": "Target", "source": "Manual"}]}
    }
    with pytest.raises(RadiomicsCourseExtractionError):
        radiomics.radiomics_for_mr_series(config, series)
    assert not output.exists()
    assert not output.with_suffix(".parquet").exists()


def _auto_source(monkeypatch, output):
    array = np.zeros((3, 3, 3), dtype=np.uint8)
    array[2, 2, 2] = 1
    array[1, 2, 2] = 2
    monkeypatch.setattr(auto_rtstruct, "_load_seg_nifti", lambda *_a, **_k: (
        sitk.GetImageFromArray(array), {1: "liver", 2: "kidney"}
    ))
    monkeypatch.setattr(radiomics, "_resample_to_reference", lambda seg, ref, **_k: seg)
    directory = output.parent / "TotalSegmentator_total_mr_NIFTI"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "total_mr--liver.nii.gz").write_bytes(b"synthetic mask source")


def test_required_auto_roi_failure_is_fatal(tmp_path, monkeypatch):
    config, series, _, output = _env(monkeypatch, tmp_path, fail_mask_coord=(1, 2, 2))
    _auto_source(monkeypatch, output)
    config.radiomics_analysis_contract = {
        "MR": {"required_rois": [{"canonical_name": "kidney", "source": "AutoTS_total_mr"}]}
    }
    _plant_stale(output)
    with pytest.raises(RadiomicsCourseExtractionError):
        radiomics.radiomics_for_mr_series(config, series)
    assert not output.exists()
    assert not output.with_suffix(".parquet").exists()


def test_pooled_feature_type_failure_is_not_dropped(tmp_path, monkeypatch):
    config, series, _, output = _env(monkeypatch, tmp_path)
    _auto_source(monkeypatch, output)
    _plant_stale(output)
    extractor = radiomics._extractor(config, "MR")
    original = extractor.execute

    def fail_selected_label(image, mask, *args):
        values = sitk.GetArrayFromImage(mask)
        if values.shape == (3, 3, 3) and values[1, 2, 2]:
            raise RadiomicsFeatureTypeError("synthetic invalid feature type")
        return original(image, mask, *args)

    monkeypatch.setattr(extractor, "execute", fail_selected_label)
    with pytest.raises((RadiomicsFeatureTypeError, RadiomicsCourseExtractionError)):
        radiomics.radiomics_for_mr_series(config, series)
    assert not output.exists()
    assert not output.with_suffix(".parquet").exists()
