"""A per-course DVH file must carry the publication key it is measured under.

``patient_id``/``course_id`` existed only on the aggregate, inserted while
reading each course. The per-course ``dvh_metrics.parquet`` therefore could not
be reconciled against the aggregate rows it produced: in one campaign all
17,636 source rows across 102 readable files were key-unusable
(``missing_column:patient_id`` x102, ``missing_column:course_id`` x102) and
source-to-aggregate content matching scored 0/1.

The course directory is the authoritative identity: the manifest confines every
course to ``output_dir/<patient>/<course>``, so these are exactly the values the
aggregation step would otherwise insert.
"""

from pathlib import Path

import pandas as pd
import pytest

from rtpipeline.dvh import _insert_course_publication_key


def _course(tmp_path: Path, patient: str = "402931", course: str = "2021-07") -> Path:
    course_dir = tmp_path / "Output" / patient / course
    course_dir.mkdir(parents=True)
    return course_dir


def test_the_key_is_taken_from_the_course_directory(tmp_path):
    frame = pd.DataFrame([{"ROI_Name": "urinary_bladder"}, {"ROI_Name": "rectum"}])

    _insert_course_publication_key(frame, _course(tmp_path))

    assert frame["patient_id"].tolist() == ["402931", "402931"]
    assert frame["course_id"].tolist() == ["2021-07", "2021-07"]


def test_the_key_leads_the_frame(tmp_path):
    """Reconciliation reads the key first; keep it where the aggregate puts it."""

    frame = pd.DataFrame([{"ROI_Name": "urinary_bladder", "DmeanGy": 41.2}])

    _insert_course_publication_key(frame, _course(tmp_path))

    assert list(frame.columns)[:2] == ["patient_id", "course_id"]


def test_an_empty_measurement_still_gains_the_columns(tmp_path):
    """A course that measured nothing must not read as a schema defect."""

    frame = pd.DataFrame()

    _insert_course_publication_key(frame, _course(tmp_path))

    assert list(frame.columns) == ["patient_id", "course_id"]
    assert len(frame) == 0


def test_an_existing_key_is_filled_not_overwritten(tmp_path):
    """Mirrors the aggregate's own rule: fill gaps, never restate identity."""

    frame = pd.DataFrame([
        {"patient_id": "already", "course_id": "recorded"},
        {"patient_id": None, "course_id": None},
    ])

    _insert_course_publication_key(frame, _course(tmp_path))

    assert frame["patient_id"].tolist() == ["already", "402931"]
    assert frame["course_id"].tolist() == ["recorded", "2021-07"]
    assert list(frame.columns) == ["patient_id", "course_id"]


@pytest.mark.parametrize(
    "patient,course",
    [("00000000001", "0000-00"), ("292929", "2024-06"), ("x", "y")],
)
def test_the_key_is_the_two_path_components_verbatim(tmp_path, patient, course):
    """The aggregate derives the same pair from the same directory, so the
    source and aggregate keys must agree character for character."""

    frame = pd.DataFrame([{"ROI_Name": "urinary_bladder"}])

    _insert_course_publication_key(frame, _course(tmp_path, patient, course))

    assert frame["patient_id"].tolist() == [patient]
    assert frame["course_id"].tolist() == [course]
