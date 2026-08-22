from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional


def remove_artifact_strict(candidate: Path, *, context: str) -> None:
    """Remove one invalid artifact or raise rather than leave stale evidence."""
    candidate = Path(candidate)
    try:
        candidate.unlink(missing_ok=True)
    except OSError as exc:
        raise RadiomicsCourseExtractionError(
            f"Failed to remove invalid radiomics artifact {candidate} while {context}: {exc}"
        ) from exc


def invalidate_radiomics_outputs(
    output_path: Path,
    *,
    additional_artifacts: Iterable[Path] = (),
) -> None:
    """Strictly remove a workbook, its Parquet sidecar, and named extra artifacts."""
    output_path = Path(output_path)
    candidates = (output_path, output_path.with_suffix(".parquet"), *additional_artifacts)
    for candidate in dict.fromkeys(Path(path) for path in candidates):
        remove_artifact_strict(candidate, context="invalidating failed or ineligible output")


_RADIOMIC_FEATURE_MARKERS = (
    "_firstorder_",
    "_shape_",
    "_shape2D_",
    "_glcm_",
    "_glrlm_",
    "_glszm_",
    "_gldm_",
    "_ngtdm_",
)


def resume_identity_pairs(dataframe: Any) -> set[tuple[str, str]]:
    """Validate a wide radiomics workbook and return its source/ROI identities."""
    import pandas as pd

    required_columns = {"segmentation_source", "roi_original_name"}
    columns = {str(column) for column in dataframe.columns}
    if dataframe.empty:
        raise ValueError("existing radiomics workbook is empty")
    if not required_columns.issubset(columns):
        missing = sorted(required_columns - columns)
        raise ValueError(
            f"existing radiomics workbook lacks required columns: {', '.join(missing)}"
        )
    if not any(
        any(marker in str(column) for marker in _RADIOMIC_FEATURE_MARKERS)
        for column in dataframe.columns
    ):
        raise ValueError("existing radiomics workbook has no radiomic feature columns")

    identities: list[tuple[str, str]] = []
    for source_value, roi_value in zip(
        dataframe["segmentation_source"].tolist(),
        dataframe["roi_original_name"].tolist(),
    ):
        if pd.isna(source_value) or pd.isna(roi_value):
            raise ValueError("existing radiomics workbook has blank source/ROI identities")
        source = str(source_value).strip()
        roi_name = str(roi_value).strip()
        if not source or not roi_name:
            raise ValueError("existing radiomics workbook has blank source/ROI identities")
        identities.append((source, roi_name))

    identity_set = set(identities)
    if len(identity_set) != len(identities):
        raise ValueError("existing radiomics workbook has duplicate source/ROI identities")
    return identity_set


def write_excel_atomic(df: Any, output_path: Path) -> Path:
    """Write and validate an Excel workbook before atomically publishing it."""
    import pandas as pd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp.xlsx", dir=output_path.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df.to_excel(tmp_path, index=False)
        pd.read_excel(tmp_path, engine="openpyxl")
        os.replace(tmp_path, output_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return output_path


class RadiomicsCourseStatus(str, Enum):
    EXTRACTED = "extracted"
    NOTHING_TO_DO = "nothing_to_do"
    FAILED = "failed"


@dataclass(frozen=True)
class RadiomicsCourseOutcome:
    status: RadiomicsCourseStatus
    output_path: Optional[Path] = None
    detail: str = ""

    @classmethod
    def extracted(cls, output_path: Path) -> "RadiomicsCourseOutcome":
        return cls(RadiomicsCourseStatus.EXTRACTED, Path(output_path))

    @classmethod
    def nothing_to_do(cls, detail: str) -> "RadiomicsCourseOutcome":
        return cls(RadiomicsCourseStatus.NOTHING_TO_DO, detail=detail)


class RadiomicsCourseExtractionError(RuntimeError):
    """A course failed and must not contribute to cohort aggregation."""

    def __init__(self, detail: str) -> None:
        self.outcome = RadiomicsCourseOutcome(RadiomicsCourseStatus.FAILED, detail=detail)
        super().__init__(detail)