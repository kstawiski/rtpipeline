"""A guarded ROI must be publishable.

The resource guard writes `roi_structural_code` on the rows it bounds. When that
column was absent from the string allowlist, every course in which the guard
fired raised at publication and was recorded as a failed course - 13 of them in
one live campaign run, after a merge whose full suite passed.
"""
from __future__ import annotations

import pandas as pd
import pytest

from rtpipeline.radiomics_schema import expected_radiomics_string_columns
from rtpipeline.radiomics_resource_guard import RESAMPLED_BBOX_LIMIT_CODE


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
    df = pd.DataFrame({"patient_id": ["p"], "some_unexpected_text": ["x"]})
    with pytest.raises(ValueError, match="Unexpected string-valued"):
        expected_radiomics_string_columns(df)
