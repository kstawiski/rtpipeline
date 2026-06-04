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
    dicom_ct_diagnostic: Path
    dicom_cbct: Path
    dicom_4dct: Path
    dicom_petct: Path
    dicom_pt: Path
    dicom_mr: Path
    dicom_mr_functional: Path
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

    def ensure_all_series(self) -> None:
        """Create optional per-series modality roots for all-series mode."""
        self.ensure()
        for path in (
            self.dicom_ct_diagnostic,
            self.dicom_cbct,
            self.dicom_4dct,
            self.dicom_petct,
            self.dicom_pt,
            self.dicom_mr_functional,
        ):
            path.mkdir(parents=True, exist_ok=True)


def build_course_dirs(root: Path) -> CourseDirs:
    dicom = root / "DICOM"
    return CourseDirs(
        root=root,
        dicom=dicom,
        dicom_ct=dicom / "CT",
        dicom_ct_diagnostic=dicom / "CT_diagnostic",
        dicom_cbct=dicom / "CBCT",
        dicom_4dct=dicom / "4DCT",
        dicom_petct=dicom / "PETCT",
        dicom_pt=dicom / "PT",
        dicom_mr=root / "MR",
        dicom_mr_functional=root / "MR_functional",
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


def find_dcm(subdir: Path, legacy_name: str, course_dir: Path) -> Path:
    """Find first DICOM file in subdir, fallback to legacy path at course root.

    The organize stage stores DICOM files in subdirectories
    (e.g. DICOM/RTPLAN/RP.12345.plan_name.dcm) but some code references
    the legacy flat layout (e.g. course_dir/RP.dcm).  This helper resolves
    the actual path regardless of layout.

    Handles non-standard extensions (.dicom, no extension) common in
    multi-centre data by checking file size and DICOM magic bytes.
    """
    _DICOM_EXTENSIONS = {".dcm", ".dicom", ".ima"}

    if subdir.is_dir():
        # First pass: standard .dcm extension
        for f in sorted(subdir.iterdir()):
            if f.suffix.lower() == ".dcm" and f.is_file():
                return f
        # Second pass: other DICOM extensions (.dicom, .ima)
        for f in sorted(subdir.iterdir()):
            if f.suffix.lower() in _DICOM_EXTENSIONS and f.is_file():
                return f
        # Third pass: files without recognised extension but likely DICOM
        # (> 1 KB and either has DICM magic or starts with DICOM group tags)
        for f in sorted(subdir.iterdir()):
            if f.is_file() and f.suffix.lower() not in _DICOM_EXTENSIONS and f.stat().st_size > 1024:
                try:
                    with open(f, "rb") as fh:
                        header = fh.read(132)
                        if header[128:132] == b"DICM" or header[:2] in (b"\x08\x00", b"\x00\x08"):
                            return f
                except (OSError, IndexError):
                    pass
    legacy = course_dir / legacy_name
    return legacy


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
