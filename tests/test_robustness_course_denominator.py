"""Count course identities, not course labels, in synthetic robustness summaries.

In-memory fixed-grid values only. No image, extraction, clinical data or files.
The tiny grid is a counting regression, not a full study NTCV execution.
"""
import numpy as np
import pandas as pd
import pytest

from rtpipeline.radiomics_robustness import RobustnessConfig, summarize_feature_stability


def _values(identities):
    return pd.DataFrame([
        {
            "patient_id": patient,
            "course_id": course,
            "structure": "synthetic_roi",
            "segmentation_source": "synthetic_source",
            "feature_name": "original_firstorder_Mean",
            "perturbation_id": str(perturbation),
            "value": 10.0 + 7.0 * index + (0.3 + index / 10) * perturbation,
        }
        for index, (patient, course) in enumerate(identities)
        for perturbation in range(3)
    ])


@pytest.mark.parametrize("identities", [
    [("P1", "C1"), ("P2", "C1")],
    [("P1", "C1"), ("P1", "C2"), ("P2", "C1")],
    [("A_B", "C"), ("A", "B_C")],
])
def test_summary_counts_distinct_patient_course_pairs(identities):
    frame = _values(identities)
    row = summarize_feature_stability(frame, RobustnessConfig()).iloc[0]
    assert row["n_courses"] == len(set(identities))
    assert row["n_subjects"] == len(set(identities))
    assert row["n_subjects_dropped"] == 0
    assert row["n_perturbations"] == 3

    # Relabelling courses uniquely cannot alter any statistic or denominator.
    relabelled = frame.copy()
    labels = {identity: f"unique_{index}" for index, identity in enumerate(identities)}
    relabelled["course_id"] = [labels[(p, c)] for p, c in zip(frame.patient_id, frame.course_id)]
    control = summarize_feature_stability(relabelled, RobustnessConfig()).iloc[0]
    for field in ("icc", "icc_ci95_low", "icc_ci95_high", "cov_pct", "qcd"):
        assert row[field] == pytest.approx(control[field], nan_ok=True), field
    assert row["robustness_label"] == control["robustness_label"]


def test_legacy_no_course_identity_does_not_invent_course_count():
    frame = _values([("P1", "C1"), ("P2", "C1")]).drop(columns="course_id")
    row = summarize_feature_stability(frame, RobustnessConfig()).iloc[0]
    assert np.isnan(row["n_courses"])
    assert row["n_subjects"] == 2
