"""Campaign-mode closure for robustness' own extraction failures (D22b).

Production case: course 477918/2025-11 stopped the Kopernik workflow when
the robustness CLI died inside robustness_for_course and the shell rule
exited 1 with no campaign closure. The CLI now publishes a
failed-extraction receipt in campaign mode; the shell rule converts it to
a ledger record and exit 0; cohort accounting absorbs the course with
zero measurement contribution.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rtpipeline import cli
from rtpipeline import radiomics_robustness as rr
from rtpipeline import robustness_completion as rc
from rtpipeline.radiomics_outcomes import RadiomicsCourseExtractionError

OUTPUT_NAME = "radiomics_robustness_ct.parquet"
SENTINEL_NAME = ".radiomics_robustness_done"


def _stub_contract():
    return SimpleNamespace(
        planning_ct={},
        planning_ct_dir=None,
        planning_ct_nifti=None,
        authoritative_rtstruct_path=None,
        authoritative_rtstruct_source=None,
    )


def _enabled_config(tmp_path: Path) -> Path:
    payload = {
        "radiomics_robustness": {
            "enabled": True,
            "modes": ["segmentation_perturbation"],
            "segmentation_perturbation": {
                "intensity": "standard",
                "apply_to_structures": ["GTV*", "CTV*", "PTV*"],
                "small_volume_changes": [-0.15, 0.0, 0.15],
                "max_translation_mm": 4.0,
                "n_random_contour_realizations": 2,
                "noise_levels": [0.0, 10.0, 20.0],
            },
        }
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _course_dir(tmp_path: Path) -> Path:
    course_dir = tmp_path / "Output" / "P1" / "C1"
    course_dir.mkdir(parents=True, exist_ok=True)
    return course_dir


def test_failure_sidecar_receipt_admit_attrition(tmp_path, monkeypatch):
    """Failed receipt revalidates and admits as non-measured attrition."""
    monkeypatch.setattr(rr, "load_course_contract", lambda _course: _stub_contract())
    course_dir = _course_dir(tmp_path)
    rob_config = rr.RobustnessConfig.from_dict(
        yaml.safe_load(((_enabled_config(tmp_path))).read_text())[
            "radiomics_robustness"
        ]
    )

    sidecar = rr.write_robustness_failure_dispositions(
        course_dir,
        rob_config=rob_config,
        output_name=OUTPUT_NAME,
        run_identifier="failure-run-1",
    )
    sentinel = course_dir / SENTINEL_NAME
    receipt = rc.write_robustness_completion_sentinel(
        sentinel,
        course_dir,
        patient_id="P1",
        course_id="C1",
        run_identifier="failure-run-1",
        measurement_outcome=rr.ROBUSTNESS_FAILED_OUTCOME,
        output_name=OUTPUT_NAME,
        dispositions_path=sidecar,
        measured_output=None,
        source_disposition_count=0,
        effective_configuration_sha256=rr._content_sha256(
            rr.effective_robustness_configuration(
                rob_config, output_name=OUTPUT_NAME
            )
        ),
    )
    assert receipt.measurement_outcome == rr.ROBUSTNESS_FAILED_OUTCOME
    assert receipt.measured is False

    admitted = rr.admit_robustness_cohort_course(
        course_dir, patient_id="P1", course_id="C1", rob_config=rob_config
    )
    assert admitted.measurement_outcome == rr.ROBUSTNESS_FAILED_OUTCOME
    assert admitted.frame is None
    assert admitted.measured_output_sha256 is None

    rows = rr._robustness_course_outcome_rows([admitted])
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["measurement_outcome"] == rr.ROBUSTNESS_FAILED_OUTCOME
    assert bool(row["table_present"]) is False
    assert int(row["table_row_count"]) == 0
    assert int(row["measured_value_row_count"]) == 0


def test_failed_outcome_never_counts_as_measured():
    assert rr.ROBUSTNESS_FAILED_OUTCOME in rc.ROBUSTNESS_COMPLETING_OUTCOMES


def _run_cli(course_dir: Path, config: Path, *, campaign: bool) -> int:
    argv = [
        "radiomics-robustness",
        "--course-dir",
        str(course_dir),
        "--config",
        str(config),
        "--output",
        str(course_dir / OUTPUT_NAME),
        "--sentinel",
        str(course_dir / SENTINEL_NAME),
    ]
    if campaign:
        argv.append("--campaign-mode")
    return cli.main(argv)


def test_cli_campaign_failure_publishes_failed_receipt(
    tmp_path, monkeypatch
):
    """Campaign mode: internal failure leaves a revalidatable failed receipt."""
    monkeypatch.setattr(rr, "load_course_contract", lambda _course: _stub_contract())

    def _raise(*_args, **_kwargs):
        raise RadiomicsCourseExtractionError("synthetic selected-ROI failure")

    monkeypatch.setattr(rr, "run_robustness_course", _raise)
    course_dir = _course_dir(tmp_path)
    config = _enabled_config(tmp_path)

    assert _run_cli(course_dir, config, campaign=True) == 1
    sentinel = course_dir / SENTINEL_NAME
    assert sentinel.is_file()
    receipt = rc.read_robustness_completion_sentinel(sentinel)
    assert receipt.measurement_outcome == rr.ROBUSTNESS_FAILED_OUTCOME
    assert (receipt.patient_id, receipt.course_id) == ("P1", "C1")


def test_cli_non_campaign_failure_leaves_no_receipt(tmp_path, monkeypatch):
    """Outside campaign mode the failure path is byte-identical to before."""
    monkeypatch.setattr(rr, "load_course_contract", lambda _course: _stub_contract())

    def _raise(*_args, **_kwargs):
        raise RadiomicsCourseExtractionError("synthetic selected-ROI failure")

    monkeypatch.setattr(rr, "run_robustness_course", _raise)
    course_dir = _course_dir(tmp_path)
    config = _enabled_config(tmp_path)

    assert _run_cli(course_dir, config, campaign=False) == 1
    assert not (course_dir / SENTINEL_NAME).exists()
