"""Per-series failures and invalid configuration must not exit successfully.

Both defects covered here are "fail open": the pipeline recorded a problem and
then reported success, so an incomplete cohort looked complete to whatever ran
next.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import rtpipeline.cli as cli
from rtpipeline.config import PipelineConfig


def _config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        dicom_root=tmp_path / "dicom",
        output_root=tmp_path / "out",
        logs_root=tmp_path / "logs",
    )


def _seg_task(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        cfg=_config(tmp_path), patient_id="PAT001", force_segmentation=False
    )


def _pet_task(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(cfg=_config(tmp_path), patient_id="PAT001", force=False)


# --- all-series segmentation -------------------------------------------------


def test_all_series_worker_reports_failure_when_a_series_failed(tmp_path, monkeypatch):
    """One failed series among several must make the worker report failure."""
    summary = {
        "planning_ct": {"attempted": 1, "segmented": 1, "failed": 0, "skipped": 0},
        "diagnostic_ct": {"attempted": 1, "segmented": 0, "failed": 1, "skipped": 0},
    }
    monkeypatch.setattr(
        "rtpipeline.segmentation.segment_all_series_for_patient",
        lambda *a, **k: summary,
    )
    assert cli._execute_all_series_segment_task(_seg_task(tmp_path)) is False


def test_all_series_worker_reports_success_when_every_series_succeeded(
    tmp_path, monkeypatch
):
    summary = {
        "planning_ct": {"attempted": 2, "segmented": 2, "failed": 0, "skipped": 0},
    }
    monkeypatch.setattr(
        "rtpipeline.segmentation.segment_all_series_for_patient",
        lambda *a, **k: summary,
    )
    assert cli._execute_all_series_segment_task(_seg_task(tmp_path)) is True


def test_idempotent_skip_is_not_a_failure(tmp_path, monkeypatch):
    """A series skipped because its output already exists is not an error."""
    summary = {
        "planning_ct": {"attempted": 0, "segmented": 0, "failed": 0, "skipped": 3},
    }
    monkeypatch.setattr(
        "rtpipeline.segmentation.segment_all_series_for_patient",
        lambda *a, **k: summary,
    )
    assert cli._execute_all_series_segment_task(_seg_task(tmp_path)) is True


def test_absent_manifest_returns_empty_summary_without_failing(tmp_path, monkeypatch):
    """``segment_all_series_for_patient`` returns {} when there is no manifest."""
    monkeypatch.setattr(
        "rtpipeline.segmentation.segment_all_series_for_patient", lambda *a, **k: {}
    )
    assert cli._execute_all_series_segment_task(_seg_task(tmp_path)) is True


# --- PET SUV ingestion -------------------------------------------------------


def test_pet_worker_reports_failure_when_ingestion_failed(tmp_path, monkeypatch):
    summary = {"computed": 2, "excluded": 0, "failed": 1, "skipped": 0}
    monkeypatch.setattr(
        "rtpipeline.pet_suv.ingest_pet_suv_for_patient", lambda *a, **k: summary
    )
    assert cli._execute_pet_suv_task(_pet_task(tmp_path)) is False


def test_pet_eligibility_exclusion_is_not_a_failure(tmp_path, monkeypatch):
    """Series excluded by eligibility rules must not fail the run.

    This is the distinction that makes the fix safe: a PET series that is
    correctly out of scope is not the same as one that could not be processed.
    """
    summary = {"computed": 1, "excluded": 5, "failed": 0, "skipped": 2}
    monkeypatch.setattr(
        "rtpipeline.pet_suv.ingest_pet_suv_for_patient", lambda *a, **k: summary
    )
    assert cli._execute_pet_suv_task(_pet_task(tmp_path)) is True


# --- a present but unusable manifest is a failure, an absent one is not ------


def test_unreadable_segmentation_manifest_is_reported_as_failure(tmp_path):
    """An existing manifest that cannot be parsed may hide eligible series."""
    from rtpipeline import segmentation

    cfg = _config(tmp_path)
    metadata = cfg.output_root / "PAT001" / "all_series" / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "series_manifest.json").write_text("{not valid json", encoding="utf-8")

    summary = segmentation.segment_all_series_for_patient(cfg, "PAT001")
    assert cli._all_series_segment_summary_has_failure(summary) is True


def test_segmentation_manifest_without_series_list_is_reported_as_failure(tmp_path):
    from rtpipeline import segmentation

    cfg = _config(tmp_path)
    metadata = cfg.output_root / "PAT001" / "all_series" / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "series_manifest.json").write_text(
        '{"series": "not-a-list"}', encoding="utf-8"
    )

    summary = segmentation.segment_all_series_for_patient(cfg, "PAT001")
    assert cli._all_series_segment_summary_has_failure(summary) is True


def test_missing_segmentation_manifest_is_not_a_failure(tmp_path):
    """No manifest at all means nothing was materialised for this patient."""
    from rtpipeline import segmentation

    cfg = _config(tmp_path)
    summary = segmentation.segment_all_series_for_patient(cfg, "PAT404")
    assert summary == {}
    assert cli._all_series_segment_summary_has_failure(summary) is False


# --- configuration validation ------------------------------------------------


def test_config_validation_error_is_narrower_than_value_error():
    """Only validation errors are re-raised past the broad config handler.

    A generic ``ValueError`` from unreadable or malformed YAML still falls back
    to defaults; an invalid user-supplied setting must stop the run.
    """
    assert issubclass(cli.ConfigValidationError, ValueError)
    assert cli.ConfigValidationError is not ValueError
    assert not isinstance(ValueError("malformed yaml"), cli.ConfigValidationError)


@pytest.mark.parametrize(
    "value, converter, key",
    [
        ("not-a-number", float, "pet.suv_decay_guard_tol"),
        ("12.5.3", float, "pet.suv_zextent_primary_fraction"),
        ("many", int, "pet.pet_clinical_weight_window_days"),
        (None, int, "organize.inventory_scan_run_id"),
        # YAML `true` coerces through int() to 1; a boolean is not a scan id.
        (True, int, "organize.inventory_scan_run_id"),
        (False, float, "pet.suv_decay_guard_tol"),
        ([1], int, "organize.inventory_scan_run_id"),
        ("", float, "pet.suv_zextent_primary_fraction"),
        ("3.7", int, "pet.pet_clinical_weight_window_days"),
    ],
)
def test_invalid_numeric_config_values_name_the_offending_key(value, converter, key):
    """A bare float()/int() here would be swallowed by the broad handler."""
    with pytest.raises(cli.ConfigValidationError) as excinfo:
        cli._config_number(value, converter, key)
    assert key in str(excinfo.value)


def test_valid_numeric_config_values_convert_normally():
    assert cli._config_number("0.25", float, "pet.suv_decay_guard_tol") == 0.25
    assert cli._config_number("7", int, "organize.inventory_scan_run_id") == 7


# --- the installed console entry point ---------------------------------------


def test_console_main_reports_config_errors_instead_of_a_traceback(monkeypatch, capsys):
    """``project.scripts`` points at console_main, not main.

    Handling this only under ``if __name__ == "__main__"`` would leave the
    installed ``rtpipeline`` command raising an uncaught traceback.
    """
    def _boom(argv=None):
        raise cli.ConfigValidationError("organize.body_composition_classes is invalid")

    monkeypatch.setattr(cli, "main", _boom)
    assert cli.console_main([]) == 2
    assert "organize.body_composition_classes" in capsys.readouterr().err


def test_console_main_passes_through_a_normal_exit_code(monkeypatch):
    monkeypatch.setattr(cli, "main", lambda argv=None: 1)
    assert cli.console_main([]) == 1
