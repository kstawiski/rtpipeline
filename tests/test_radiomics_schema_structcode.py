"""Every field the resource guard writes must be publishable.

The guard bounds an ROI instead of letting it kill the worker, and records why.
Those provenance fields travel in the feature table, so each one must be in the
string allowlist. Getting this wrong does not fail a test - it fails live
courses: omitting `roi_structural_code` failed 13 Kopernik courses, and the
follow-up fix that added only that field then failed 3 more on
`native_mask_bbox_shape`.

This test therefore derives the expected fields from the guard's OWN payload
rather than restating a hand-written list, so a newly added guard field fails
here instead of in a campaign.
"""
from __future__ import annotations

import pandas as pd
import pytest

from rtpipeline.radiomics_schema import (
    RADIOMICS_TEXT_COLUMNS,
    expected_radiomics_string_columns,
)
from rtpipeline.radiomics_resource_guard import (
    RESAMPLED_BBOX_LIMIT_CODE,
    ResampledBoundingBoxEstimate,
)


def _guard_payload() -> dict:
    est = ResampledBoundingBoxEstimate(
        native_foreground_voxels=1234,
        estimated_resampled_foreground_voxels=5678,
        native_bbox_shape=(276, 456, 155),
        estimated_resampled_bbox_shape=(350, 350, 122),
        estimated_resampled_bbox_voxels=60_648_000,
        pad_distance_voxels=5,
    )
    return est.metadata(limit=15_000_000)


def test_every_guard_payload_field_is_publishable() -> None:
    """The allowlist must cover the guard's whole payload, not one field of it."""
    payload = dict(_guard_payload())
    payload["roi_structural_code"] = RESAMPLED_BBOX_LIMIT_CODE
    df = pd.DataFrame([{k: (str(v) if isinstance(v, (list, tuple)) else v)
                        for k, v in payload.items()} | {"patient_id": "p"}])
    expected_radiomics_string_columns(df)  # must not raise

    for name, value in payload.items():
        if isinstance(value, (list, tuple, str)):
            assert name in RADIOMICS_TEXT_COLUMNS, (
                f"guard emits string-valued {name!r} but the allowlist omits it; "
                "this fails live courses at publication, not in CI"
            )


@pytest.mark.parametrize(
    "code",
    [
        RESAMPLED_BBOX_LIMIT_CODE,
        "ROI_MASK_BELOW_MIN_VOXELS",
        "REQUIRED_ROI_NOT_DECLARED",
        "REQUIRED_ROI_AMBIGUOUS_MATCH",
    ],
)
def test_structural_code_is_publishable(code: str) -> None:
    df = pd.DataFrame({"patient_id": ["p"], "roi_structural_code": [code]})
    assert "roi_structural_code" in expected_radiomics_string_columns(df)


def test_undeclared_string_column_still_rejected() -> None:
    """The allowlist must not become a blanket permit."""
    df = pd.DataFrame({"patient_id": ["p"], "some_unexpected_text": ["x"]})
    with pytest.raises(ValueError, match="Unexpected string-valued"):
        expected_radiomics_string_columns(df)
