"""A non-measurement row asserts no feature, so it cannot break cohort parity.

A geometrically impossible perturbation -- a +/-15% erosion that would annihilate
a thin wall structure -- is recorded with a null feature_name and null value.
Comparing those rows as feature assignments casts None to the string "None",
invents a phantom feature in the union, and fails the entire cohort because
every other subject "lacks" it.
"""

import pandas as pd
import pytest

from rtpipeline.radiomics_robustness import _validate_cohort_feature_sets


def _measured(patient, course, structure, features):
    return pd.DataFrame(
        {
            "patient_id": patient,
            "course_id": course,
            "structure": structure,
            "segmentation_source": "Manual",
            "extraction_arm": "primary_resegmented",
            "feature_name": list(features),
            "robustness_status": "measured",
        }
    )


def _impossible(patient, course, structure, n=2):
    return pd.DataFrame(
        {
            "patient_id": [patient] * n,
            "course_id": [course] * n,
            "structure": [structure] * n,
            "segmentation_source": ["Manual"] * n,
            "extraction_arm": ["primary_resegmented"] * n,
            "feature_name": [None] * n,
            "robustness_status": ["geometrically_impossible"] * n,
        }
    )


def test_an_impossible_condition_does_not_fail_the_cohort():
    frame = pd.concat(
        [
            _measured("p1", "c1", "Wall", ["f1", "f2"]),
            _impossible("p1", "c1", "Wall"),
            _measured("p2", "c1", "Wall", ["f1", "f2"]),
        ],
        ignore_index=True,
    )

    _validate_cohort_feature_sets(frame)


def test_a_genuinely_absent_feature_is_still_rejected():
    frame = pd.concat(
        [
            _measured("p1", "c1", "Wall", ["f1", "f2"]),
            _measured("p2", "c1", "Wall", ["f1"]),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="inconsistent feature sets"):
        _validate_cohort_feature_sets(frame)


def test_the_message_names_the_absent_feature():
    frame = pd.concat(
        [
            _measured("p1", "c1", "Wall", ["f1", "shape_Sphericity"]),
            _measured("p2", "c1", "Wall", ["f1"]),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError) as excinfo:
        _validate_cohort_feature_sets(frame)

    message = str(excinfo.value)
    assert "shape_Sphericity" in message
    assert "union has 2 feature(s)" in message
    assert "None" not in message


def test_impossible_rows_alone_do_not_invent_a_feature():
    """Every subject impossible for a structure means no feature parity to check."""

    frame = pd.concat(
        [_impossible("p1", "c1", "Wall"), _impossible("p2", "c1", "Wall")],
        ignore_index=True,
    )

    _validate_cohort_feature_sets(frame)


def test_a_subject_with_only_impossible_rows_does_not_drag_the_union():
    frame = pd.concat(
        [
            _measured("p1", "c1", "Wall", ["f1", "f2"]),
            _measured("p2", "c1", "Wall", ["f1", "f2"]),
            _impossible("p3", "c1", "Wall"),
        ],
        ignore_index=True,
    )

    _validate_cohort_feature_sets(frame)
