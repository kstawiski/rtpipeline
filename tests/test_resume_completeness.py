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

from rtpipeline import cli
from rtpipeline.organize import _hydrate_existing_course
from rtpipeline.layout import build_course_dirs
from rtpipeline.config import PipelineConfig


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


def _plan_checkpoint(
    tmp_path: Path,
    *,
    metadata: dict[str, object],
    plan_layout: str | None,
    patient_id: str = "PT001",
    course_id: str = "COURSE_A",
) -> Path:
    course_dir = tmp_path / patient_id / course_id
    dirs = build_course_dirs(course_dir)
    dirs.ensure()
    if plan_layout == "nested":
        (dirs.dicom_rtplan / "RP.1.dcm").write_bytes(b"placeholder")
    elif plan_layout == "legacy":
        (course_dir / "RP.dcm").write_bytes(b"placeholder")
    elif plan_layout is not None:
        raise ValueError(f"unknown plan layout: {plan_layout}")
    (dirs.metadata / "case_metadata.json").write_text(
        json.dumps({"course_id": course_id, **metadata}),
        encoding="utf-8",
    )
    return course_dir


def test_nested_plan_checkpoint_without_delivery_or_planning_ct_is_reprocessed(tmp_path):
    course_dir = _plan_checkpoint(
        tmp_path,
        metadata={"rp_path": ""},
        plan_layout="nested",
    )

    assert _hydrate_existing_course("PT001", "COURSE_A", course_dir) is None


def test_nested_plan_checkpoint_without_planning_ct_is_reprocessed(tmp_path):
    course_dir = _plan_checkpoint(
        tmp_path,
        metadata={"rp_path": "", "delivery_status": "fully_delivered"},
        plan_layout="nested",
    )

    assert _hydrate_existing_course("PT001", "COURSE_A", course_dir) is None


def test_legacy_root_plan_checkpoint_without_adjudication_is_reprocessed(tmp_path):
    course_dir = _plan_checkpoint(
        tmp_path,
        metadata={"rp_path": ""},
        plan_layout="legacy",
    )

    assert _hydrate_existing_course("PT001", "COURSE_A", course_dir) is None


def test_truthy_metadata_plan_checkpoint_without_adjudication_is_reprocessed(tmp_path):
    course_dir = _plan_checkpoint(
        tmp_path,
        metadata={"rp_path": "/archived/layout/RP.1.dcm"},
        plan_layout=None,
    )

    assert _hydrate_existing_course("PT001", "COURSE_A", course_dir) is None


def test_fully_adjudicated_nested_plan_checkpoint_is_resumable(tmp_path):
    course_dir = _plan_checkpoint(
        tmp_path,
        metadata={
            "rp_path": "",
            "delivery_status": "no_records_at_all",
            "planning_ct_status": "unresolved_reference",
        },
        plan_layout="nested",
    )

    result = _hydrate_existing_course("PT001", "COURSE_A", course_dir)

    assert result is not None
    assert result.rp_path == course_dir / "DICOM" / "RTPLAN" / "RP.1.dcm"
    assert result.delivery_status == "no_records_at_all"
    assert result.planning_ct_status == "unresolved_reference"


@pytest.mark.parametrize("plan_layout", ["nested", "legacy", "metadata"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("delivery_status", None),
        ("delivery_status", ""),
        ("delivery_status", "   "),
        ("delivery_status", "unknown"),
        ("delivery_status", "not_a_status"),
        ("planning_ct_status", None),
        ("planning_ct_status", ""),
        ("planning_ct_status", "   "),
        ("planning_ct_status", "unknown"),
    ],
)
def test_plan_checkpoint_rejects_semantically_missing_adjudication(
    tmp_path, plan_layout, field, value
):
    metadata: dict[str, object] = {
        "rp_path": "/archived/layout/RP.1.dcm" if plan_layout == "metadata" else "",
        "delivery_status": "fully_delivered",
        "planning_ct_status": "referenced",
    }
    metadata[field] = value
    disk_layout = None if plan_layout == "metadata" else plan_layout
    course_dir = _plan_checkpoint(tmp_path, metadata=metadata, plan_layout=disk_layout)

    assert _hydrate_existing_course("PT001", "COURSE_A", course_dir) is None


@pytest.mark.parametrize(
    "delivery_status",
    [
        "fully_delivered",
        "partially_delivered",
        "delivered_but_records_absent",
        "no_records_at_all",
    ],
)
def test_all_documented_delivery_statuses_are_valid_checkpoint_values(tmp_path, delivery_status):
    course_dir = _plan_checkpoint(
        tmp_path,
        metadata={
            "rp_path": "",
            "delivery_status": delivery_status,
            "planning_ct_status": "referenced",
        },
        plan_layout="nested",
    )

    result = _hydrate_existing_course("PT001", "COURSE_A", course_dir)

    assert result is not None
    assert result.delivery_status == delivery_status


def test_plan_free_checkpoint_may_still_use_legacy_defaults(tmp_path):
    course_dir = _plan_checkpoint(tmp_path, metadata={"rp_path": ""}, plan_layout=None)

    result = _hydrate_existing_course("PT001", "COURSE_A", course_dir)

    assert result is not None
    assert result.delivery_status == "no_records_at_all"
    assert result.planning_ct_status == "unknown"


def test_mixed_manifest_rejects_valid_subset_when_one_course_requires_reprocessing(tmp_path, caplog):
    good_dir = _plan_checkpoint(
        tmp_path / "good",
        metadata={
            "rp_path": "",
            "delivery_status": "no_records_at_all",
            "planning_ct_status": "none",
        },
        plan_layout="nested",
        patient_id="GOOD",
        course_id="COURSE_A",
    )
    incomplete_dir = _plan_checkpoint(
        tmp_path / "incomplete",
        metadata={"rp_path": ""},
        plan_layout="nested",
        patient_id="INCOMPLETE",
        course_id="COURSE_B",
    )
    manifest_path = tmp_path / "courses.json"
    manifest_path.write_text(
        json.dumps(
            {
                "courses": [
                    {"patient": "GOOD", "course": "COURSE_A", "path": str(good_dir)},
                    {"patient": "INCOMPLETE", "course": "COURSE_B", "path": str(incomplete_dir)},
                ]
            }
        ),
        encoding="utf-8",
    )
    cfg = PipelineConfig(
        dicom_root=tmp_path / "input",
        output_root=tmp_path / "output",
        logs_root=tmp_path / "logs",
        resume=True,
    )

    with caplog.at_level("WARNING"):
        result = cli._load_courses_from_manifest(cfg, manifest_path, set(), set())

    assert result == [], "a mixed manifest must fall back instead of returning only hydrated courses"
    assert "rejecting all 1 hydrated course(s)" in caplog.text
    assert "reprocessing required" in caplog.text


def test_rejected_manifest_falls_back_to_organize(tmp_path, monkeypatch):
    cfg = PipelineConfig(
        dicom_root=tmp_path / "input",
        output_root=tmp_path / "output",
        logs_root=tmp_path / "logs",
        resume=True,
    )
    sentinel = [object()]
    monkeypatch.setattr(cli, "organize_and_merge", lambda received: sentinel if received is cfg else [])

    assert cli._organize_after_manifest_rejection(cfg) is sentinel


def test_rejected_manifest_fails_loudly_when_organize_is_unavailable(tmp_path, monkeypatch):
    cfg = PipelineConfig(
        dicom_root=tmp_path / "input",
        output_root=tmp_path / "output",
        logs_root=tmp_path / "logs",
        resume=True,
    )
    monkeypatch.setattr(cli, "organize_and_merge", None)

    with pytest.raises(cli.ManifestReprocessingRequiredError, match="Reprocessing is required"):
        cli._organize_after_manifest_rejection(cfg)
