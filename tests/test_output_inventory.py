"""Durable state cannot change a scientific inventory or its artifact bytes."""
import hashlib
from pathlib import Path

import pytest

from rtpipeline.output_inventory import is_operational_path, scientific_files


@pytest.mark.parametrize("path", [
    "_CACHE/copy-progress/items/a/commit.json", "_CACHE/dicom_headers.pkl",
    "_RESULTS/custom_reduction.commit.json",
    "_RESULTS/.custom_reduction.commit.json-generations/g/table.parquet",
    "P/C/NIFTI/.conversion-progress/items/a/commit.json",
    "P/C/Segmentation_TotalSegmentator/CT/.model-progress/settings.json",
    "_COURSES/.organize-progress/progress.jsonl", "P/C/.organize-progress.json",
    "P/C/.custom-model-progress/data/mask.nii.gz", "P/C/.weights-progress/cache.json",
    "P/C/.crop-progress/current.json", "P/C/.qc-progress/current.json",
    "P/C/.radiomics-progress/publication.json", "P/C/.robustness-progress/cache/data.json",
    "P/C/.dvh/generations/g/commit.json", "_RESULTS/aggregate_results.commit.json",
    "_RESULTS/radiomics_robustness_summary.commit.json",
    "_RESULTS/.radiomics_robustness_summary.commit.json-generations/g/table.parquet",
])
def test_reserved_operational_paths(path):
    assert is_operational_path(path)


@pytest.mark.parametrize("path", [
    "P/C/metadata/case_metadata.json", "P/C/NIFTI/CT.metadata.json",
    "P/C/Segmentation_TotalSegmentator/CT/manifest.json",
    "P/C/Segmentation_TotalSegmentator/CT/total--liver.nii.gz",
    "P/C/.radiomics_robustness_done", "P/C/.organized",
    "_RESULTS/radiomics_robustness_summary.xlsx", "metadata/commit.json",
    "P/C/metadata/progress.jsonl", "P/C/.unrecognized/file.json",
])
def test_scientific_paths_are_never_blanket_excluded(path):
    assert not is_operational_path(path)


def test_bookkeeping_cannot_change_inventory_or_scientific_bytes(tmp_path):
    science = tmp_path / "P/C/NIFTI/CT.nii.gz"
    science.parent.mkdir(parents=True)
    science.write_bytes(b"synthetic scientific volume")
    def inventory():
        return {p.relative_to(tmp_path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in scientific_files(tmp_path)}
    before = inventory()
    operational = tmp_path / "P/C/NIFTI/.conversion-progress/cache/commit.json"
    operational.parent.mkdir(parents=True)
    operational.write_text('{"generation": 1}')
    assert inventory() == before
    operational.write_text('{"generation": 2}')
    assert inventory() == before
    science.write_bytes(b"changed scientific volume")
    assert inventory() != before


def test_scientific_source_manifest_and_aggregate_use_the_same_rule(tmp_path):
    from rtpipeline.dvh_generation import input_manifest
    from rtpipeline.aggregate_generation import snapshot
    from rtpipeline.organize import _organized_artifacts
    for name in ("P/C/NIFTI/CT.nii.gz", "P/C/NIFTI/.conversion-progress/cache.json",
                 "_COURSES/manifest_extra.json", "_COURSES/.organize-progress/discovery.json"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic")
    course = tmp_path / "P/C"
    assert {p["path"] for p in input_manifest(course)} == {"NIFTI/CT.nii.gz"}
    assert {p.relative_to(course).as_posix() for p in _organized_artifacts(course)} == {"NIFTI/CT.nii.gz"}
    assert {p["path"] for p in snapshot(tmp_path)} == {"_COURSES/manifest_extra.json"}
