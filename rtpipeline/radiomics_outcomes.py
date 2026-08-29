from __future__ import annotations

import os
import json
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


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
    EXTRACTED_WITH_FAILURES = "extracted_with_failures"
    DEGRADED = "extracted_with_failures"
    NOTHING_TO_DO = "nothing_to_do"
    FAILED = "failed"


@dataclass(frozen=True)
class RadiomicsCourseOutcome:
    status: RadiomicsCourseStatus
    output_path: Optional[Path] = None
    detail: str = ""
    roi_counts: Dict[str, Dict[str, int]] | None = None
    roi_failures: Tuple[Dict[str, str], ...] = ()

    @classmethod
    def extracted(
        cls,
        output_path: Path,
        *,
        roi_counts: Optional[Mapping[str, Mapping[str, int]]] = None,
        roi_failures: Sequence[Mapping[str, str]] = (),
        detail: str = "",
    ) -> "RadiomicsCourseOutcome":
        counts = {
            str(source): {
                "attempted": int(values.get("attempted", 0)),
                "extracted": int(values.get("extracted", 0)),
                "failed": int(values.get("failed", 0)),
            }
            for source, values in (roi_counts or {}).items()
        }
        failures = tuple(dict(failure) for failure in roi_failures)
        has_failures = any(values["failed"] for values in counts.values()) or bool(failures)
        status = (
            RadiomicsCourseStatus.EXTRACTED_WITH_FAILURES
            if has_failures
            else RadiomicsCourseStatus.EXTRACTED
        )
        return cls(status, Path(output_path), detail, counts, failures)

    @classmethod
    def nothing_to_do(
        cls,
        detail: str,
        *,
        roi_counts: Optional[Mapping[str, Mapping[str, int]]] = None,
    ) -> "RadiomicsCourseOutcome":
        counts = {
            str(source): {
                "attempted": int(values.get("attempted", 0)),
                "extracted": int(values.get("extracted", 0)),
                "failed": int(values.get("failed", 0)),
            }
            for source, values in (roi_counts or {}).items()
        }
        return cls(RadiomicsCourseStatus.NOTHING_TO_DO, detail=detail, roi_counts=counts)

    @property
    def counts_by_source(self) -> Dict[str, Dict[str, int]]:
        """Compatibility name for consumers that describe the field by its role."""
        return dict(self.roi_counts or {})

    @property
    def attempted(self) -> int:
        return sum(values.get("attempted", 0) for values in (self.roi_counts or {}).values())

    @property
    def extracted_count(self) -> int:
        return sum(values.get("extracted", 0) for values in (self.roi_counts or {}).values())

    @property
    def failed_count(self) -> int:
        return sum(values.get("failed", 0) for values in (self.roi_counts or {}).values())


def roi_source_is_required(source: str, *, operator_configured: bool = False) -> bool:
    """Return the fail-closed policy for a source, never for an anatomy name.

    Operator-configured structures are required even if a caller gives them a
    source label that resembles an automatically generated one. TotalSegmentator
    organ outputs are best-effort because their inventory is region-independent.
    All other RTSTRUCT sources remain required by default.
    """
    if operator_configured:
        return True
    normalized = str(source).strip().casefold()
    return not normalized.startswith(("autorts_total", "autots_total"))


def extraction_status_is_nonfatal_for_required(status: Any) -> bool:
    """Return whether a required ROI status preserves the historical course gate.

    A present, non-empty mask below ``min_voxels`` was historically reported as
    an observed status rather than treated as a course-failing extraction error.
    Empty masks, unreadable contours, timeouts, and extractor errors remain fatal
    for required sources.
    """
    return str(status).strip().casefold() == "below_minimum_voxels"


def course_diagnostic_columns(outcome: RadiomicsCourseOutcome) -> Dict[str, Any]:
    """Return scalar workbook columns that make partial extraction explicit."""
    counts = outcome.roi_counts or {}
    return {
        "radiomics_course_status": outcome.status.value,
        "radiomics_roi_attempted": outcome.attempted,
        "radiomics_roi_extracted": outcome.extracted_count,
        "radiomics_roi_failed": outcome.failed_count,
        "radiomics_roi_counts_by_source": json.dumps(
            {source: counts[source] for source in sorted(counts)},
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def outcome_from_output(
    output_path: Path,
    *,
    required_by_identity: Optional[Mapping[Tuple[str, str], bool]] = None,
) -> RadiomicsCourseOutcome:
    """Read persisted outcomes and reject failed or missing required ROIs.

    ``required_by_identity`` supplies the current task inventory during resume.
    Source policy remains the fallback for generic backend reconstruction.
    """
    import pandas as pd

    output_path = Path(output_path)
    dataframe = pd.read_excel(output_path, engine="openpyxl")
    required_map = {
        (str(source), str(roi_name)): bool(required)
        for (source, roi_name), required in (required_by_identity or {}).items()
    }
    counts: Dict[str, Dict[str, int]] = {}
    failures: list[Dict[str, str]] = []
    fatal_failures: list[str] = []
    observed_identities: set[Tuple[str, str]] = set()
    for row in dataframe.to_dict("records"):
        source_value = row.get("segmentation_source", "unknown")
        source = "unknown" if bool(pd.isna(source_value)) else str(source_value)
        roi_value = row.get("roi_original_name", row.get("roi_name", "unknown"))
        roi_name = "unknown" if bool(pd.isna(roi_value)) else str(roi_value)
        identity = (source, roi_name)
        observed_identities.add(identity)
        status = row.get("extraction_status")
        if bool(pd.isna(status)):
            status = None
        if status == "declared_skip":
            continue
        source_counts = counts.setdefault(source, {"attempted": 0, "extracted": 0, "failed": 0})
        source_counts["attempted"] += 1
        if status in (None, "success"):
            source_counts["extracted"] += 1
        else:
            source_counts["failed"] += 1
            failures.append(
                {
                    "roi_name": roi_name,
                    "source": source,
                    "status": str(status),
                    "failure_kind": str(row.get("extraction_failure_kind", "extraction_error")),
                    "reason": str(row.get("extraction_status_detail", "unknown error")),
                }
            )
            required = required_map.get(identity, roi_source_is_required(source))
            if required and not extraction_status_is_nonfatal_for_required(status):
                fatal_failures.append(
                    f"required ROI {source}/{roi_name} has persisted status {status}: "
                    f"{row.get('extraction_status_detail', 'unknown error')}"
                )
    for identity, required in required_map.items():
        if required and identity not in observed_identities:
            fatal_failures.append(
                f"required ROI {identity[0]}/{identity[1]} has no persisted outcome"
            )
    if fatal_failures:
        invalidate_radiomics_outputs(output_path)
        raise RadiomicsCourseExtractionError(
            "Persisted radiomics output is not resumable: " + "; ".join(fatal_failures)
        )
    return RadiomicsCourseOutcome.extracted(
        output_path,
        roi_counts=counts,
        roi_failures=failures,
        detail="reconstructed from persisted per-ROI outcomes",
    )


class RadiomicsCourseExtractionError(RuntimeError):
    """A course failed and must not contribute to cohort aggregation."""

    def __init__(self, detail: str) -> None:
        self.outcome = RadiomicsCourseOutcome(RadiomicsCourseStatus.FAILED, detail=detail)
        super().__init__(detail)