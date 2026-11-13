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
    "_RESULTS/.aggregate-progress/partitions/a/commit.json",
    "_RESULTS/.aggregate-progress/reductions/b/payload.json",
    "_RESULTS/.radiomics_robustness_summary-progress/partitions/a/commit.json",
    "P/C/.radiomics_ct-progress/dispositions/a/commit.json",
    "P/C/.rtsegmentator-progress/imports/a/commit.json",
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
    "P/C/metadata/.notes-progress", "P/C/.-progress/file.json",
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


def _durable_root_literals():
    """Every string literal in production code that names a progress root."""
    import ast
    import re
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "rtpipeline").glob("*.py")) + sorted((root / "workflow" / "scripts").glob("*.py"))
    pattern = re.compile(r"(?:^|/)(?:\.[\w.-]+-progress|_CACHE/[\w.-]+-progress)(?:/|$)")
    found = set()
    for source in sources:
        for node in ast.walk(ast.parse(source.read_text(), filename=str(source))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and pattern.search(node.value):
                found.add((source.name, node.value))
    return found


def test_every_durable_root_named_in_code_is_operational():
    literals = _durable_root_literals()
    names = {value for _, value in literals}
    # The scan must see the roots this module exists for, or it proves nothing.
    assert {".aggregate-progress/partitions", ".rtsegmentator-progress/imports",
            ".radiomics-progress", ".robustness-progress"} <= names
    missing = sorted((name, value) for name, value in literals
                     if not is_operational_path(f"P/C/{value.strip('/')}/item/commit.json"))
    assert missing == []


@pytest.mark.parametrize("stem", ["radiomics_ct", "radiomics_mr", "radiomics_robustness_summary"])
def test_derived_progress_roots_are_operational(stem):
    # radiomics_conda and the robustness aggregate derive "." + stem + "-progress".
    assert is_operational_path(f"P/C/.{stem}-progress/partitions/a/commit.json")


def _production_sources():
    root = Path(__file__).resolve().parents[1]
    return sorted((root / "rtpipeline").glob("*.py")) + sorted((root / "workflow" / "scripts").glob("*.py"))


def test_dynamically_built_progress_roots_are_registered():
    # Literal roots are checked above; a root assembled at run time is invisible
    # to that scan, so each one must be registered here and match the generic rule.
    import ast
    registered = {
        ("radiomics_conda.py", '"." + Path(output_path).stem + "-progress"'),
        ("radiomics_robustness_aggregate.py", '"." + output_excel.stem + "-progress"'),
    }
    found = set()
    for source in _production_sources():
        text = source.read_text()
        for node in ast.walk(ast.parse(text, filename=str(source))):
            if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)) or isinstance(node, ast.JoinedStr):
                segment = ast.get_source_segment(text, node) or ""
                # A fragment such as "-progress" (not a complete ".<name>-progress"
                # component, which the literal scan covers) means the name is assembled.
                fragments = [part.value for part in ast.walk(node)
                             if isinstance(part, ast.Constant) and isinstance(part.value, str)
                             and part.value.endswith("-progress") and "/" not in part.value
                             and not part.value.startswith(".")]
                if fragments:
                    found.add((source.name, segment))
    assert found == registered


def test_commit_record_names_belong_to_the_commit_writer():
    # ".commit.json" is reserved for aggregate_generation.commit records; a
    # scientific output with that suffix would silently leave the inventory.
    import ast
    owners = {"aggregate_generation.py", "output_inventory.py", "workflow_aggregate.py",
              "radiomics_robustness.py"}
    names = set()
    for source in _production_sources():
        for node in ast.walk(ast.parse(source.read_text(), filename=str(source))):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and node.value.endswith(".commit.json")):
                assert source.name in owners, (source.name, node.value)
                names.add(node.value)
    assert names == {".commit.json", "aggregate_results.commit.json",
                     "radiomics_robustness_summary.commit.json"}
    assert all(is_operational_path(f"_RESULTS/{name}") for name in names - {".commit.json"})
