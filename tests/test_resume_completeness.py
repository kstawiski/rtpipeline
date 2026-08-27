"""Resume must skip finished work and redo unfinished work.

Resume decides whether a course is already processed by hydrating its output
directory. Hydration accepted a course that had copied CT DICOM but no NIfTI,
so a course whose conversion had failed was marked done and left without a
NIfTI for the rest of the run. Segmentation and radiomics then had nothing to
read, and no later resume would ever repair it.

Observed after a dcm2niix race on a 154-patient cohort: 352,707 copied DICOM
instances against 441 NIfTI files. Every affected course would have been
skipped on resume, baking the damage in permanently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rtpipeline.organize import _hydrate_existing_course
from rtpipeline.layout import build_course_dirs


def _course(tmp_path: Path, *, with_ct: bool, with_nifti: bool, with_meta: bool = True):
    course_dir = tmp_path / "PT001" / "COURSE_A"
    dirs = build_course_dirs(course_dir)
    dirs.ensure()

    if with_ct:
        dirs.dicom_ct.mkdir(parents=True, exist_ok=True)
        (dirs.dicom_ct / "CT_1.dcm").write_bytes(b"x")
    if with_nifti:
        dirs.nifti.mkdir(parents=True, exist_ok=True)
        (dirs.nifti / "image.nii.gz").write_bytes(b"x")
    if with_meta:
        dirs.metadata.mkdir(parents=True, exist_ok=True)
        (dirs.metadata / "case_metadata.json").write_text(
            json.dumps({"course_id": "COURSE_A"}), encoding="utf-8"
        )
    return course_dir


def test_a_converted_course_is_skipped(tmp_path):
    """Genuinely finished work must not be redone."""
    course_dir = _course(tmp_path, with_ct=True, with_nifti=True)
    result = _hydrate_existing_course("PT001", "COURSE_A", course_dir)
    assert result is not None, "a complete course should hydrate and be skipped"


def test_an_unconverted_course_is_reprocessed(tmp_path):
    """The production failure: CT copied, conversion failed, no NIfTI."""
    course_dir = _course(tmp_path, with_ct=True, with_nifti=False)
    result = _hydrate_existing_course("PT001", "COURSE_A", course_dir)
    assert result is None, "a course with CT but no NIfTI must be reprocessed"


def test_an_unconverted_course_is_reprocessed_even_with_metadata(tmp_path):
    """Metadata JSON alone must not certify a course as converted."""
    course_dir = _course(tmp_path, with_ct=True, with_nifti=False, with_meta=True)
    meta = build_course_dirs(course_dir).metadata / "case_metadata.json"
    meta.write_text(
        json.dumps({"course_id": "COURSE_A", "primary_nifti": "/gone/missing.nii.gz"}),
        encoding="utf-8",
    )
    assert _hydrate_existing_course("PT001", "COURSE_A", course_dir) is None


def test_an_empty_directory_is_not_hydrated(tmp_path):
    """Pre-existing behaviour: ensure() creates dirs, which are not results."""
    course_dir = tmp_path / "PT001" / "COURSE_A"
    build_course_dirs(course_dir).ensure()
    assert _hydrate_existing_course("PT001", "COURSE_A", course_dir) is None


def test_a_course_without_ct_is_unaffected(tmp_path):
    """Only a copied CT implies a NIfTI is owed, so nothing else changes."""
    course_dir = tmp_path / "PT001" / "COURSE_A"
    dirs = build_course_dirs(course_dir)
    dirs.ensure()
    (course_dir / "RS.dcm").write_bytes(b"x")

    result = _hydrate_existing_course("PT001", "COURSE_A", course_dir)
    assert result is not None, "an RTSTRUCT-only course owes no NIfTI"


@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
def test_either_nifti_suffix_counts_as_converted(tmp_path, suffix):
    course_dir = tmp_path / "PT001" / "COURSE_A"
    dirs = build_course_dirs(course_dir)
    dirs.ensure()
    dirs.dicom_ct.mkdir(parents=True, exist_ok=True)
    (dirs.dicom_ct / "CT_1.dcm").write_bytes(b"x")
    dirs.nifti.mkdir(parents=True, exist_ok=True)
    (dirs.nifti / f"image{suffix}").write_bytes(b"x")

    assert _hydrate_existing_course("PT001", "COURSE_A", course_dir) is not None
