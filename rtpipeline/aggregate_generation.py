"""Verified aggregate publication and exact frozen input inventories.

The receipt is the commit target. Artifact bytes are verified before that
target is published; interrupted publication has no committed receipt.
Immutable generations and reduction items survive withdrawal of nominal files.
"""
from __future__ import annotations

import json
from pathlib import Path
import uuid

from .durable_progress import (DurableProgressError, final_result_writer,
                               file_sha256, _fsync_dir)

SCHEMA = "rtpipeline-aggregate-generation-v1"


def standard_code_manifest():
    package = Path(__file__).parent
    return {name: package / name for name in ("workflow_aggregate.py", "dvh_aggregate.py",
        "radiomics_ct_contract.py", "radiomics_schema.py", "radiomics_cohort.py",
        "aggregate_generation.py", "extraction_progress.py", "durable_progress.py",
        "course_manifest.py", "course_contract.py", "stage_completion.py",
        "roi_requiredness.py", "missing_values.py", "acquisition_scale.py")}


def aggregate_environment():
    from .stage_completion import environment_record
    return environment_record(("numpy", "pandas", "pyarrow", "pingouin", "scipy", "openpyxl", "xlsxwriter"))


def preflight(output):
    """Check aggregate commits before DAG planning, including imported code."""
    output = Path(output)
    invalid = []
    for name in ("aggregate_results.commit.json", "radiomics_robustness_summary.commit.json"):
        receipt = output / "_RESULTS" / name
        if not receipt.exists():
            continue
        try:
            record = read_back(receipt)
            inputs = record["inputs"]
            if name == "aggregate_results.commit.json":
                code, environment = standard_code_manifest(), aggregate_environment()
                if inputs["snapshots"] != snapshot(output):
                    raise ValueError("aggregate snapshots changed")
            else:
                from .radiomics_robustness_aggregate import _reduction_code, _reduction_environment
                code, environment = _reduction_code(), _reduction_environment()
                for entry in inputs["sources"]:
                    if file_sha256(output / entry["path"]) != entry["sha256"]:
                        raise ValueError("robustness aggregate sources changed")
            if (inputs["code"] != {key: file_sha256(path) for key, path in code.items()}
                    or inputs["environment"] != environment):
                raise ValueError("aggregate code or runtime changed")
        except (OSError, ValueError, KeyError, TypeError, DurableProgressError):
            receipt.unlink(missing_ok=True)
            invalid.append(name)
    return {"invalidated": invalid}


def snapshot(output_root):
    """Exact membership and standard-aggregate dependencies, excluding robustness.

    Robustness is an independent DAG branch with its own frozen admission and
    commit. Its in-flight evidence must not invalidate standard reductions.
    """
    root = Path(output_root)
    files = set()
    for name in ("_COURSES", "_campaign_ledger/records"):
        files.update(p for p in (root / name).rglob("*") if p.is_file() and p.suffix != ".age"
                     and not p.name.endswith("__radiomics_robustness.json"))
    manifest = root / "_COURSES/manifest.json"
    if manifest.is_file():
        from .course_manifest import read_course_manifest
        courses, _ = read_course_manifest(manifest, output_dir=root)
        for _, _, course in courses:
            for path in course.iterdir():
                if path.is_file() and path.name in {
                    ".organized", ".segmentation_done", ".custom_models_done", ".crop_ct_done",
                    ".dvh_done", ".qc_done", ".radiomics_done", "dvh_metrics.xlsx",
                    "dvh_metrics.parquet", "radiomics_ct.xlsx", "radiomics_ct.parquet", "fractions.xlsx"}:
                    files.add(path)
            for name in ("metadata", "qc_reports"):
                files.update(p for p in (course / name).rglob("*") if p.is_file() and p.suffix != ".age"
                             and not p.name.startswith("radiomics_robustness")
                             and p.name != "segmentation_resume.json")
            files.update(p for p in (course / "MR").glob("radiomics_mr.*") if p.is_file() and p.suffix in {".xlsx", ".parquet"})
    for stem in ("plans", "structure_sets", "dosimetrics", "fractions", "metadata", "CT_images"):
        files.update(p for p in (root / "Data").glob(stem + ".*") if p.is_file() and p.suffix in {".xlsx", ".parquet"})
    return [{"path": p.relative_to(root).as_posix(), "sha256": file_sha256(p)} for p in sorted(files)]


def commit(artifacts, receipt, *, inputs=None, verify=None):
    """Commit an immutable generation, publish verified bytes, then its receipt."""
    receipt = Path(receipt)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.unlink(missing_ok=True)
    root = receipt.parent / ("." + receipt.name + "-generations")
    root.mkdir(exist_ok=True)
    _fsync_dir(receipt.parent.parent)
    _fsync_dir(receipt.parent)
    directory = root / uuid.uuid4().hex
    directory.mkdir()
    _fsync_dir(root)
    inventory = {}
    for name, source in artifacts.items():
        destination = directory / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        final_result_writer(destination, Path(source).read_bytes())
        inventory[name] = file_sha256(destination)
    record = {"schema": SCHEMA, "generation_id": directory.name,
              "inputs": inputs, "artifacts": inventory}
    final_result_writer(directory / "commit.json", record)
    for name, target in artifacts.items():
        final_result_writer(target, (directory / name).read_bytes())
    if verify:
        verify()
    final_result_writer(receipt, record)
    read_back(receipt, artifacts)
    _failpoint("aggregate_committed", record["generation_id"])
    return record


def read_back(receipt, artifacts=None):
    receipt = Path(receipt)
    record = json.loads(receipt.read_text())
    import re
    if (not isinstance(record, dict) or record.get("schema") != SCHEMA
            or not re.fullmatch(r"[0-9a-f]{32}", str(record.get("generation_id", "")))
            or not isinstance(record.get("artifacts"), dict)
            or any(Path(name).name != name or name in {".", ".."}
                   for name in record["artifacts"])):
        raise DurableProgressError("AGGREGATE_COMMIT_INVALID")
    directory = receipt.parent / ("." + receipt.name + "-generations") / record["generation_id"]
    if json.loads((directory / "commit.json").read_text()) != record:
        raise DurableProgressError("AGGREGATE_COMMIT_INVALID")
    for name, expected in record["artifacts"].items():
        if file_sha256(directory / name) != expected:
            raise DurableProgressError("AGGREGATE_GENERATION_READBACK_FAILED")
        target = artifacts[name] if artifacts else receipt.parent / name
        if file_sha256(target) != expected:
            raise DurableProgressError("AGGREGATE_PUBLICATION_READBACK_FAILED")
    return record


def _failpoint(event, key):
    """Inert synthetic interruption seam."""
