from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


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