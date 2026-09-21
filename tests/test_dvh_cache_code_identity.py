"""A cached DVH produced by different code must not be reused.

``_is_dvh_up_to_date`` validated the schema, provenance hashes, course contract
and QC metadata -- but never the code that produced the numbers. A corrected
classifier therefore never reached an already-measured course: the stage reused
the old measurement and the completion sentinel then stamped the current
revision onto it, asserting a provenance that never happened.

In one campaign a deployed dose-grid coverage fix reached 6 of 122 courses for
exactly this reason; the other 96 kept three-day-old numbers under a sentinel
that named the new revision.
"""

import json

import pytest

from rtpipeline import dvh as dvh_module
from rtpipeline.dvh import (
    DVH_MEASUREMENT_CODE_SOURCES,
    _current_dvh_measurement_code_sha256,
)


@pytest.fixture(autouse=True)
def _reset_identity_cache():
    dvh_module._DVH_MEASUREMENT_CODE_RESOLVED = False
    dvh_module._DVH_MEASUREMENT_CODE_SHA256 = None
    yield
    dvh_module._DVH_MEASUREMENT_CODE_RESOLVED = False
    dvh_module._DVH_MEASUREMENT_CODE_SHA256 = None


def test_the_identity_is_a_stable_digest_of_real_modules():
    first = _current_dvh_measurement_code_sha256()

    assert isinstance(first, str) and len(first) == 64
    dvh_module._DVH_MEASUREMENT_CODE_RESOLVED = False
    assert _current_dvh_measurement_code_sha256() == first


def test_dvh_py_itself_decides_the_identity():
    """The module holding the classifier must be part of its own cache key."""

    assert "dvh.py" in DVH_MEASUREMENT_CODE_SOURCES


def test_the_snakefile_does_not_decide_the_identity():
    """An unrelated workflow edit must not discard the cohort's measurements."""

    assert "Snakefile" not in DVH_MEASUREMENT_CODE_SOURCES
    assert "cli.py" not in DVH_MEASUREMENT_CODE_SOURCES
    assert "stage_completion.py" not in DVH_MEASUREMENT_CODE_SOURCES


def test_changing_a_source_changes_the_identity(monkeypatch, tmp_path):
    baseline = _current_dvh_measurement_code_sha256()

    real = dvh_module.Path(dvh_module.__file__).resolve().parent
    shadow = tmp_path / "pkg"
    shadow.mkdir()
    for relative in DVH_MEASUREMENT_CODE_SOURCES:
        (shadow / relative).write_bytes((real / relative).read_bytes())
    (shadow / "dvh.py").write_bytes(b"# edited\n" + (real / "dvh.py").read_bytes())

    monkeypatch.setattr(dvh_module, "__file__", str(shadow / "dvh.py"))
    dvh_module._DVH_MEASUREMENT_CODE_RESOLVED = False

    assert _current_dvh_measurement_code_sha256() != baseline


def test_an_unreadable_installation_yields_no_identity(monkeypatch, tmp_path):
    """Absent sources are a broken install, not a stale cache."""

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(dvh_module, "__file__", str(empty / "dvh.py"))
    dvh_module._DVH_MEASUREMENT_CODE_RESOLVED = False

    assert _current_dvh_measurement_code_sha256() is None


def _measured_course(tmp_path, monkeypatch, *, code_sources_sha256):
    """A course whose DVH cache is valid in every respect but its code identity."""

    import pandas as pd
    from types import SimpleNamespace

    from rtpipeline.dvh_support import sha256_file

    metadata = tmp_path / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    contract = metadata / "case_metadata.json"
    contract.write_text("{}", encoding="utf-8")

    row = {key: None for key in [
        "D0.03ccGy", "D0.03cc_status", "dose_grid_coverage_fraction",
        "dose_grid_coverage_status", "dose_response_eligible",
        "dose_metric_usable_for_dose_response", "dose_metric_status", "target_like",
        "ROI_Interpreted_Type", "treatment_technique", "relative_metric_status",
        "structure_provenance_status", "zero_dose_status", "zero_dose_trigger_metric",
        "Dose_Plan_Scope_Status", "Course_Treatment_Isocenter_Status",
        "Course_Treatment_Isocenter_Count", "Course_Target_Dose_Coverage_Status"]}
    pd.DataFrame([row]).to_parquet(tmp_path / "dvh_metrics.parquet")

    qc = {
        "status": "ok",
        "course_contract_sha256": sha256_file(contract),
        "metric_version": dvh_module.DVH_METRIC_VERSION,
        "code_sources_sha256": code_sources_sha256,
        "row_count": 1,
        "rx_relative_metrics_available": True,
        "structure_resolution": {"classification": "bound"},
    }
    (metadata / "dvh_qc.json").write_text(json.dumps(qc), encoding="utf-8")

    workbook = tmp_path / "dvh_metrics.xlsx"
    workbook.write_bytes(b"cache existence fixture")

    monkeypatch.setattr(dvh_module, "load_course_contract", lambda path: SimpleNamespace(
        metadata_path=contract, plan_artifact_path=None, dose_grid_path=None,
        authoritative_rtstruct_path=None))
    monkeypatch.setattr(dvh_module, "list_custom_model_outputs", lambda path: [])
    return workbook


def test_a_cache_from_the_running_code_is_reused(tmp_path, monkeypatch):
    workbook = _measured_course(
        tmp_path, monkeypatch,
        code_sources_sha256=_current_dvh_measurement_code_sha256(),
    )

    assert dvh_module._is_dvh_up_to_date(tmp_path, workbook)


@pytest.mark.parametrize("recorded", [None, "", "0" * 64, "stale"])
def test_a_cache_from_different_code_is_regenerated(tmp_path, monkeypatch, recorded):
    """This is the defect: a deployed fix silently never reached the course."""

    workbook = _measured_course(tmp_path, monkeypatch, code_sources_sha256=recorded)

    assert not dvh_module._is_dvh_up_to_date(tmp_path, workbook)


def test_an_unidentifiable_installation_leaves_the_cache_alone(tmp_path, monkeypatch):
    """No identity is a broken install, not grounds to rebuild the cohort forever."""

    workbook = _measured_course(tmp_path, monkeypatch, code_sources_sha256=None)
    monkeypatch.setattr(dvh_module, "_current_dvh_measurement_code_sha256", lambda: None)

    assert dvh_module._is_dvh_up_to_date(tmp_path, workbook)
