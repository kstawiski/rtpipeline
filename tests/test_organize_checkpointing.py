"""Organize must resume at patient granularity, not restart from zero.

Organize is one Snakemake checkpoint job over the whole cohort, so an
interruption at 99% re-walked every DICOM header. On a 154-patient cohort of
415,562 instances that cost hours before any new work began.

A patient is skippable only when every course discovered for it has a
completion record. That distinction is the point: skipping a partially
finished patient would silently drop its unfinished courses from the manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rtpipeline.config import PipelineConfig
from rtpipeline.organize import (
    _completed_patients,
    _organize_checkpoint_dir,
    _record_course_done,
    _record_expected_courses,
)


@pytest.fixture()
def config(tmp_path):
    return PipelineConfig(
        dicom_root=tmp_path / "dicom",
        output_root=tmp_path / "out",
        logs_root=tmp_path / "logs",
    )


def test_a_patient_with_every_course_done_is_complete(config):
    _record_expected_courses(config, "PT001", ["A", "B"])
    _record_course_done(config, "PT001", "A", config.output_root / "PT001" / "A")
    _record_course_done(config, "PT001", "B", config.output_root / "PT001" / "B")

    complete = _completed_patients(config)
    assert set(complete) == {"PT001"}
    assert len(complete["PT001"]) == 2


def test_a_partially_finished_patient_is_not_complete(config):
    """The hazard: skipping this patient would drop course B entirely."""
    _record_expected_courses(config, "PT001", ["A", "B"])
    _record_course_done(config, "PT001", "A", config.output_root / "PT001" / "A")

    assert _completed_patients(config) == {}


def test_a_patient_with_no_completions_is_not_complete(config):
    _record_expected_courses(config, "PT001", ["A"])
    assert _completed_patients(config) == {}


def test_completions_without_an_expected_record_are_ignored(config):
    """A done record alone cannot prove the patient's full course set."""
    _record_course_done(config, "PT001", "A", config.output_root / "PT001" / "A")
    assert _completed_patients(config) == {}


def test_patients_are_independent(config):
    _record_expected_courses(config, "PT001", ["A"])
    _record_course_done(config, "PT001", "A", config.output_root / "PT001" / "A")
    _record_expected_courses(config, "PT002", ["A", "B"])
    _record_course_done(config, "PT002", "A", config.output_root / "PT002" / "A")

    assert set(_completed_patients(config)) == {"PT001"}


def test_records_are_published_atomically(config):
    """No temporary file may survive, and the record must be valid JSON."""
    _record_expected_courses(config, "PT001", ["A"])
    _record_course_done(config, "PT001", "A", config.output_root / "PT001" / "A")

    root = _organize_checkpoint_dir(config)
    assert list(root.rglob("*.tmp")) == [], "temporary record left behind"
    for f in root.rglob("*.json"):
        json.loads(f.read_text(encoding="utf-8"))


def test_a_truncated_record_does_not_mark_a_patient_complete(config):
    """An interrupted write must fail closed, not certify completion."""
    _record_expected_courses(config, "PT001", ["A"])
    _record_course_done(config, "PT001", "A", config.output_root / "PT001" / "A")
    done = next((_organize_checkpoint_dir(config) / "PT001").glob("*.done.json"))
    done.write_text("{ truncated", encoding="utf-8")

    assert _completed_patients(config) == {}


def test_a_course_key_with_path_characters_is_stored_safely(config):
    """A course key is not a filename and must not escape the record directory."""
    _record_expected_courses(config, "PT001", ["../escape"])
    _record_course_done(config, "PT001", "../escape", config.output_root / "PT001" / "x")

    root = _organize_checkpoint_dir(config)
    assert (root / "PT001").is_dir()
    assert not (root.parent / "escape.done.json").exists()
    complete = _completed_patients(config)
    assert set(complete) == {"PT001"}


def test_nothing_is_complete_before_any_run(config):
    assert _completed_patients(config) == {}
