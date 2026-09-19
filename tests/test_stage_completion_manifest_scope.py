"""A stage claims the artifacts it recorded, not every file matching a pattern.

Clinicians name structures freely. A target whose name ends in ``_cropped``
yields a segmentation mask that the crop_ct pattern ``**/*_cropped.nii.gz``
matches even though anatomical cropping never ran. The stage then cannot publish
a completion -- a disabled stage must claim no artifacts -- and every downstream
stage on that course fails with it.
"""

import json
from pathlib import Path

from rtpipeline.stage_completion import (
    _STAGE_DEFINITIONS,
    ArtifactRule,
    _discover_outputs,
    _rule_manifest_paths,
)


def _mask(course_dir: Path, series: str, name: str) -> Path:
    path = course_dir / "Segmentation_Original" / series / f"{name}.nii.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"mask")
    return path


def test_a_clinician_roi_named_cropped_is_not_a_cropping_output(
    tmp_path: Path,
) -> None:
    course_dir = tmp_path / "patient" / "course"
    _mask(course_dir, "SERIES", "target_a_cropped")
    _mask(course_dir, "SERIES", "target_b_cropped")

    assert _discover_outputs(course_dir, _STAGE_DEFINITIONS["crop_ct"]) == []


def test_recorded_cropping_outputs_are_still_bound(tmp_path: Path) -> None:
    course_dir = tmp_path / "patient" / "course"
    produced = _mask(course_dir, "SERIES", "CT_cropped")
    foreign = _mask(course_dir, "SERIES", "target_a_cropped")
    (course_dir / "cropping_metadata.json").write_text(
        json.dumps({"region": "PELVIS", "cropped_files": {"ct": str(produced)}}),
        encoding="utf-8",
    )

    discovered = _discover_outputs(course_dir, _STAGE_DEFINITIONS["crop_ct"])
    paths = {item["path"] for item in discovered}

    assert produced.relative_to(course_dir).as_posix() in paths
    assert foreign.relative_to(course_dir).as_posix() not in paths
    assert "cropping_metadata.json" in paths


def test_manifest_paths_are_accepted_relative_to_the_course(tmp_path: Path) -> None:
    """A manifest written on another host records a path, not a mount."""

    course_dir = tmp_path / "patient" / "course"
    produced = _mask(course_dir, "SERIES", "CT_cropped")
    relative = produced.relative_to(course_dir).as_posix()
    (course_dir / "cropping_metadata.json").write_text(
        json.dumps({"cropped_files": {"ct": relative}}), encoding="utf-8"
    )

    paths = {
        item["path"] for item in _discover_outputs(course_dir, _STAGE_DEFINITIONS["crop_ct"])
    }

    assert relative in paths


def test_an_absent_or_unreadable_manifest_claims_nothing(tmp_path: Path) -> None:
    course_dir = tmp_path / "patient" / "course"
    course_dir.mkdir(parents=True)
    rule = ArtifactRule(
        "cropped_image_or_mask",
        "**/*_cropped.nii.gz",
        "content_sha256",
        manifest="cropping_metadata.json",
        manifest_key="cropped_files",
    )

    assert _rule_manifest_paths(course_dir, rule) == set()

    (course_dir / "cropping_metadata.json").write_text("{ not json", encoding="utf-8")
    assert _rule_manifest_paths(course_dir, rule) == set()

    (course_dir / "cropping_metadata.json").write_text("[]", encoding="utf-8")
    assert _rule_manifest_paths(course_dir, rule) == set()


def test_a_rule_without_a_manifest_is_unscoped(tmp_path: Path) -> None:
    course_dir = tmp_path / "patient" / "course"
    course_dir.mkdir(parents=True)
    rule = ArtifactRule("dvh_qc", "metadata/dvh_qc.json", "content_sha256")

    assert _rule_manifest_paths(course_dir, rule) is None


def test_a_manifest_path_outside_the_course_is_refused(tmp_path: Path) -> None:
    course_dir = tmp_path / "patient" / "course"
    course_dir.mkdir(parents=True)
    outside = tmp_path / "elsewhere_cropped.nii.gz"
    outside.write_bytes(b"mask")
    (course_dir / "cropping_metadata.json").write_text(
        json.dumps({"cropped_files": {"ct": str(outside)}}), encoding="utf-8"
    )
    rule = ArtifactRule(
        "cropped_image_or_mask",
        "**/*_cropped.nii.gz",
        "content_sha256",
        manifest="cropping_metadata.json",
        manifest_key="cropped_files",
    )

    assert _rule_manifest_paths(course_dir, rule) == set()
