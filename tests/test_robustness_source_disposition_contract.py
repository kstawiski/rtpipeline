"""Manager acceptance checks for robustness source-disposition provenance.

Only generated CT/RTSTRUCT files and mocked feature extraction are used.
The imported helper was inspected before use. No clinical inputs or launches.
"""
from __future__ import annotations

import json

import pandas as pd
import pydicom
import pytest

from rtpipeline import radiomics_robustness as rr
from test_robustness_nonmeasurements import _real_mixed_course


def _artifact(course):
    return course / "metadata" / rr.ROBUSTNESS_SOURCE_DISPOSITIONS_FILENAME


def _completed(tmp_path, monkeypatch):
    course, config, robustness, source = _real_mixed_course(tmp_path, monkeypatch)
    output = rr.robustness_for_course(config, robustness, course)
    assert output is not None
    payload = json.loads(_artifact(course).read_text(encoding="utf-8"))
    # Each negative case first proves the same loader admits unchanged output.
    loaded = rr.load_robustness_source_dispositions(
        course, run_identifier=payload["robustness_run_identifier"], rob_config=robustness
    )
    assert loaded == payload["rows"]
    return course, config, robustness, source, output, payload


@pytest.mark.parametrize("corruption", ["schema", "row_type", "row_identity", "row_status", "duplicate"])
def test_loader_rejects_corrupt_disposition_rows(tmp_path, monkeypatch, corruption):
    course, _, robustness, _, _, payload = _completed(tmp_path, monkeypatch)
    run_id = payload["robustness_run_identifier"]
    if corruption == "schema":
        payload["schema_version"] = 999
    elif corruption == "row_type":
        payload["rows"][0] = "not a disposition record"
    elif corruption == "row_identity":
        payload["rows"][0].pop("rtstruct_sop_instance_uid", None)
    elif corruption == "row_status":
        payload["rows"][0]["status"] = "success"
    else:
        payload["rows"].append(dict(payload["rows"][0]))
        payload["row_count"] = len(payload["rows"])
    _artifact(course).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        rr.load_robustness_source_dispositions(course, run_identifier=run_id, rob_config=robustness)


def test_loader_rejects_changed_source_with_same_sop_uid(tmp_path, monkeypatch):
    course, _, robustness, source, _, payload = _completed(tmp_path, monkeypatch)
    ds = pydicom.dcmread(source)
    original_uid = str(ds.SOPInstanceUID)
    ds.StructureSetLabel = "CHANGED"
    ds.save_as(source)
    assert str(pydicom.dcmread(source).SOPInstanceUID) == original_uid

    with pytest.raises(ValueError):
        rr.load_robustness_source_dispositions(
            course, run_identifier=payload["robustness_run_identifier"], rob_config=robustness
        )


@pytest.mark.parametrize("change", ["missing", "modified"])
def test_loader_rejects_missing_or_changed_measured_output(tmp_path, monkeypatch, change):
    course, _, robustness, _, output, payload = _completed(tmp_path, monkeypatch)
    if change == "missing":
        output.unlink()
    else:
        table = pd.read_parquet(output)
        table.loc[table["robustness_status"].eq("measured"), "value"] = 9.0
        table.to_parquet(output, index=False)

    with pytest.raises(ValueError):
        rr.load_robustness_source_dispositions(
            course, run_identifier=payload["robustness_run_identifier"], rob_config=robustness
        )


def test_failed_feature_extraction_does_not_publish_source_dispositions(tmp_path, monkeypatch):
    course, config, robustness, _ = _real_mixed_course(tmp_path, monkeypatch)

    def fail_extraction(*args, **kwargs):
        raise RuntimeError("synthetic downstream extraction failure")

    monkeypatch.setattr(rr, "extract_features_for_masks", fail_extraction)
    with pytest.raises(RuntimeError, match="synthetic downstream extraction failure"):
        rr.robustness_for_course(config, robustness, course)
    assert not (course / "radiomics_robustness_ct.parquet").exists()
    assert not _artifact(course).exists()


def test_failed_contract_validation_invalidates_old_dispositions(tmp_path, monkeypatch):
    course, config, robustness, _, output, _ = _completed(tmp_path, monkeypatch)

    def fail_contract(*args, **kwargs):
        raise RuntimeError("synthetic invalid current course contract")

    monkeypatch.setattr(rr, "load_course_contract", fail_contract)
    with pytest.raises(RuntimeError, match="synthetic invalid current course contract"):
        rr.robustness_for_course(config, robustness, course)
    assert not _artifact(course).exists()
    assert not output.exists()
