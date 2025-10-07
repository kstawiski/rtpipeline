from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class CourseDirs:
    """Canonical subdirectories for a course root."""

    root: Path
    dicom: Path
    dicom_ct: Path
    dicom_mr: Path
    dicom_rtplan: Path
    dicom_rtdose: Path
    dicom_rtstruct: Path
    dicom_related: Path
    nifti: Path
    segmentation_original: Path
    segmentation_totalseg: Path
    segmentation_custom_models: Path
    metadata: Path
    qc_reports: Path

    def ensure(self) -> None:
        for path in (
            self.dicom,
            self.dicom_ct,
            self.dicom_mr,
            self.dicom_rtplan,
            self.dicom_rtdose,
            self.dicom_rtstruct,
            self.dicom_related,
            self.nifti,
            self.segmentation_original,
            self.segmentation_totalseg,
            self.segmentation_custom_models,
            self.metadata,
            self.qc_reports,
        ):
            path.mkdir(parents=True, exist_ok=True)


def build_course_dirs(root: Path) -> CourseDirs:
    dicom = root / "DICOM"
    return CourseDirs(
        root=root,
        dicom=dicom,
        dicom_ct=dicom / "CT",
        dicom_mr=root / "MR",
        dicom_rtplan=dicom / "RTPLAN",
        dicom_rtdose=dicom / "RTDOSE",
        dicom_rtstruct=dicom / "RTSTRUCT",
        dicom_related=root / "DICOM_related",
        nifti=root / "NIFTI",
        segmentation_original=root / "Segmentation_Original",
        segmentation_totalseg=root / "Segmentation_TotalSegmentator",
        segmentation_custom_models=root / "Segmentation_CustomModels",
        metadata=root / "metadata",
        qc_reports=root / "qc_reports",
    )


def course_dir_name(
    course_start: Optional[str],
    fallback_key: str,
    existing_names: set[str],
) -> str:
    """Generate a stable directory name of the form YYYY-MM[_suffix]."""
    base = course_start or "0000-00"
    # enforce YYYY-MM pattern if not already
    if len(base) >= 7 and base[4] == "-":
        slug = base[:7]
    else:
        slug = base.replace("/", "-").replace(" ", "-")[:7]
        if len(slug) != 7 or slug[4] != "-":
            slug = "0000-00"
    candidate = slug
    if candidate not in existing_names:
        existing_names.add(candidate)
        return candidate
    suffix = 1
    sanitized_key = "".join(ch.lower() for ch in fallback_key if ch.isalnum())
    sanitized_key = sanitized_key[:6] or "course"
    while True:
        if suffix == 1:
            cand = f"{slug}_{sanitized_key}"
        else:
            cand = f"{slug}_{sanitized_key}{suffix}"
        if cand not in existing_names:
            existing_names.add(cand)
            return cand
        suffix += 1


__all__ = ["CourseDirs", "build_course_dirs", "course_dir_name"]
