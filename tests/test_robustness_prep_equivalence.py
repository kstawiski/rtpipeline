"""Faster robustness preparation publishes what a027945 published.

The base revision's package is exported with ``git archive`` and run, like
the working tree, through ``robustness_prep_driver.py`` (the campaign course
step through the CLI) on the same synthetic courses at the same absolute
paths. The working tree runs twice more: with a different worker budget and
preparation start method, and in-process.

Compared exactly, after replacing run identifiers, the code identity and
digests/sizes of run-bound artifacts with placeholders: the robustness table
content (every column and value, keyed by row; column order and row order are
not compared because the base revision's parallel order depends on worker
timing), its Arrow field types, the identity ledger (its rows as a multiset,
for the same reason), the source-disposition sidecar (dispositions, tolerated
failures, source-only basis) and the completion receipt, and every CLI exit
code.

The courses exercise the changed paths: Manual, AutoRTS, RS_custom and a
custom-model RTSTRUCT; many unselected ROIs, some without identity; an
unselected ROI with unreadable contour data, one outside the image and one
without image references; a point ROI; a selected ROI with a governed
structural disposition; small, thin and edge-touching selected ROIs; a
source-only course; identity failures; producer-declared feature gaps; a
selected ROI whose mask is empty (course fails); a course with no identity.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml

from robustness_prep_fixture import build_course

ROOT = Path(__file__).resolve().parents[1]
BASE_REVISION = "a027945"
DRIVER = Path(__file__).with_name("robustness_prep_driver.py")
GRID = dict(rows=40, columns=40, slices=16)
RUN_BOUND_PARENTS = {"measured_output", "source_dispositions", "failed_evidence"}


def _roi(source, name, center=(20, 20, 8), radii=(5, 6, 3), **extra):
    return dict(source=source, name=name, center=center, radii=radii, shape="cylinder", **extra)


def _slab(source, name, box, **extra):
    return dict(source=source, name=name, center=(0, 0, 0), radii=(1, 1, 1), slab=box, **extra)


def _organs(source, prefix, count, catalog=True):
    return [
        _roi(source, f"{prefix}_{i:02d}", center=(8 + 3 * i % 24, 10 + 5 * i % 20, 4 + i % 8),
             radii=(3 + i % 3, 4, 2 + i % 2), catalog=catalog)
        for i in range(count)
    ]


def _full_course(**options):
    rois = [
        _roi("Manual", "GTV1", radii=(4, 5, 3)),
        _roi("Manual", "CTV1", radii=(7, 8, 4)),
        _slab("Manual", "PTV_zedge", (14, 24, 14, 24, 0, 4)),        # clipped by -4 mm z shift
        _slab("Manual", "PTV_rowedge", (0, 6, 8, 18, 5, 11)),        # touches image row 0
        _slab("Manual", "CTV_thin", (12, 28, 12, 28, 7, 8)),         # a single slice
        _slab("Manual", "GTV_small", (18, 22, 18, 22, 7, 9)),        # 32 voxels
        _roi("Manual", "GTV_nocat", catalog=False),                  # selected, no identity
        *_organs("Manual", "OAR", 6),
        *_organs("Manual", "helper", 3, catalog=False),
        _roi("AutoRTS_total", "urinary_bladder", center=(26, 16, 6), radii=(5, 5, 3)),
        *_organs("AutoRTS_total", "organ", 12),
        _roi("Custom", "CTV_eval", center=(16, 24, 9), radii=(5, 4, 3)),
        *_organs("Custom", "custom", 4),
        _roi("CustomModel:ModelA", "GTV_model", center=(20, 18, 8), radii=(3, 4, 2)),
        *_organs("CustomModel:ModelA", "model", 3),
    ]
    defects = {
        "Manual": [
            {"name": "Bad", "kind": "partially_unparseable"},
            {"name": "GTV_bad", "kind": "partially_unparseable"},
            {"name": "Marker", "kind": "point"},
            {"name": "far_oar", "kind": "outside_fov"},
            {"name": "noref_oar", "kind": "no_image_reference"},
        ],
        "Custom": [{"name": "small_bowel1", "kind": "partially_unparseable"}],
        "CustomModel:ModelA": [{"name": "orphan_model", "kind": "unreferenced_slice"}],
    }
    return dict(GRID, rois=rois, defects=defects, **options)


COURSES = {
    "P1": _full_course(parallel=True, workers=3, prep_start_method="spawn", seed=1),
    "P2": _full_course(parallel=True, workers=1, seed=2),
    "P3": dict(GRID, parallel=False, seed=3, rois=[
        _roi("Manual", "GTV1"), _slab("Manual", "PTV_zedge", (14, 24, 14, 24, 0, 4)),
        *_organs("Manual", "OAR", 5)],
        defects={"Manual": [{"name": "Bad", "kind": "partially_unparseable"},
                            {"name": "far_oar", "kind": "outside_fov"}]}),
    # Selection matches only a point ROI: source-only, with the unselected
    # identity-matched masks counted in collected_mask_count.
    "P4": dict(GRID, parallel=True, workers=2, seed=4, rois=[
        *_organs("Manual", "OAR", 6), *_organs("AutoRTS_total", "organ", 5)],
        defects={"Manual": [{"name": "GTV_point", "kind": "point"},
                            {"name": "noref_oar", "kind": "no_image_reference"}]}),
    "P5": dict(GRID, parallel=True, workers=4, prep_start_method="fork", seed=5,
               identity_mismatch_when="_n10_t0_0_4_c1", rois=[
                   _roi("Manual", "GTV1"), _roi("Manual", "CTV1", radii=(7, 8, 4)),
                   _roi("AutoRTS_total", "urinary_bladder", center=(26, 16, 6)),
                   *_organs("AutoRTS_total", "organ", 4)]),
    "P6": dict(GRID, parallel=True, workers=2, seed=6, primary_below_minimum_when="_v-15",
               rois=[_roi("Manual", "GTV1"), _roi("Manual", "PTV1", radii=(8, 8, 4)),
                     *_organs("Manual", "OAR", 3)]),
    # A selected ROI whose mask is empty is a technical failure of the course.
    "P7": dict(GRID, parallel=True, workers=3, seed=7, rois=[
        _roi("Manual", "GTV1"), *_organs("Manual", "OAR", 3)],
        defects={"Manual": [{"name": "GTV_far", "kind": "outside_fov"}]},
        extra_catalog=[["Manual", "GTV_far"]]),
    "P8": dict(GRID, parallel=True, workers=2, seed=8, rois=[
        _slab("Manual", "GTV_tiny", (19, 21, 19, 21, 7, 8)), _roi("Manual", "CTV1"),
        *_organs("Manual", "OAR", 2)]),
    # No identity at all: the NIfTI fallback is consulted, nothing is selected.
    "P9": dict(GRID, parallel=True, workers=2, seed=9, rois=[
        _roi("Manual", "GTV1", catalog=False), *_organs("Manual", "OAR", 3, catalog=False)]),
}


def _export_base(target: Path) -> Path:
    try:
        archive = subprocess.run(
            ["git", "-C", str(ROOT), "archive", "--format=tar", BASE_REVISION, "rtpipeline"],
            check=True, capture_output=True, timeout=120,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"base revision {BASE_REVISION} is not available: {exc}")
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(target)
    return target


def _run(package_root: Path, work: Path, inputs: Path, config: Path, **env_overrides) -> dict:
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(inputs, work)
    env = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="2", **env_overrides)
    result = subprocess.run(
        [sys.executable, "-B", "-s", str(DRIVER), str(package_root), str(work / "Output"),
         str(config)],
        env=env, capture_output=True, text=True, timeout=1800,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    codes = json.loads(result.stdout.strip().splitlines()[-1])
    return {"codes": codes, "outputs": _collect(work / "Output"), "log": result.stderr}


def _collect(output_root: Path) -> dict:
    collected = {}
    for path in sorted(output_root.rglob("*")):
        relative = path.relative_to(output_root)
        if path.is_dir() or relative.parts[2] in {"CT", "RS.dcm", "RS_auto.dcm", "RS_custom.dcm",
                                                   "Segmentation_CustomModels", "inputs.json"}:
            continue
        if path.suffix == ".parquet":
            collected[str(relative)] = (pd.read_parquet(path), pq.read_schema(path))
        else:
            text = path.read_text(encoding="utf-8")
            try:
                collected[str(relative)] = json.loads(text)
            except json.JSONDecodeError:
                collected[str(relative)] = text
    return collected


def _run_ids(outputs: dict) -> dict:
    ids = {}
    for key, value in outputs.items():
        if key.endswith("radiomics_robustness_source_dispositions.json"):
            ids[value["robustness_run_identifier"]] = f"<run:{key.split('/')[0]}>"
    return ids


def _normalize_text(value, ids):
    if isinstance(value, str):
        for run_id, label in ids.items():
            value = value.replace(run_id, label)
    return value


def _normalize(value, ids, parent=None):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "code_identity":
                out[key] = "<code identity>"
            elif key in {"sha256", "size", "size_bytes", "bytes"} and parent in RUN_BOUND_PARENTS:
                out[key] = "<run-bound artifact>"
            else:
                out[key] = _normalize(item, ids, key)
        return out
    if isinstance(value, list):
        return [_normalize(item, ids, parent) for item in value]
    return _normalize_text(value, ids)


def canonical_table(frame: pd.DataFrame, ids) -> pd.DataFrame:
    """Row content independent of row and column order."""
    frame = frame.copy()
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(lambda v: _normalize_text(v, ids))
    frame = frame.reindex(columns=sorted(frame.columns))
    order = frame.astype(str).agg("\x1f".join, axis=1).sort_values(kind="stable").index
    return frame.loc[order].reset_index(drop=True)


def _assert_equivalent(base: dict, current: dict) -> None:
    assert current["codes"] == base["codes"]
    assert sorted(current["outputs"]) == sorted(base["outputs"])
    base_ids, current_ids = _run_ids(base["outputs"]), _run_ids(current["outputs"])
    assert sorted(base_ids.values()) == sorted(current_ids.values())
    for key, expected in base["outputs"].items():
        observed = current["outputs"][key]
        if isinstance(expected, tuple):
            (expected_frame, expected_schema), (observed_frame, observed_schema) = expected, observed
            assert {f.name: f.type for f in observed_schema} == {
                f.name: f.type for f in expected_schema}, key
            pd.testing.assert_frame_equal(
                canonical_table(observed_frame, current_ids),
                canonical_table(expected_frame, base_ids),
                check_exact=True, obj=key,
            )
        elif key.endswith("radiomics_robustness_identity.json"):
            # The base revision appends perturbation identity failures in
            # worker completion order; compare the ledger rows as a multiset.
            observed, expected = (_normalize(v, ids) for v, ids in
                                  ((observed, current_ids), (expected, base_ids)))
            for ledger in (observed, expected):
                ledger["rows"] = sorted(ledger["rows"], key=lambda r: json.dumps(r, sort_keys=True))
            assert observed == expected, key
        else:
            assert _normalize(observed, current_ids) == _normalize(expected, base_ids), key


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("prep_equivalence")
    inputs = tmp_path / "inputs"
    for patient, spec in COURSES.items():
        build_course(inputs / "Output" / patient / "C1", spec)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"radiomics_robustness": {
        "enabled": True,
        "segmentation_perturbation": {
            "apply_to_structures": ["GTV*", "CTV*", "PTV*", "urinary_bladder"],
        },
    }}))
    base_root = _export_base(tmp_path / "base")
    work = tmp_path / "work"
    return {
        "base": _run(base_root, work, inputs, config),
        "current": _run(ROOT, work, inputs, config),
        "current_other_budget": _run(ROOT, work, inputs, config, ROBUSTNESS_DRIVER_WORKERS="5",
                                     ROBUSTNESS_DRIVER_PREP_START="fork"),
        "current_in_process": _run(ROOT, work, inputs, config, ROBUSTNESS_DRIVER_WORKERS="1"),
    }


def test_fixture_exercises_the_changed_paths(runs):
    base = runs["base"]
    outputs = base["outputs"]
    sidecar = outputs["P1/C1/metadata/radiomics_robustness_source_dispositions.json"]
    tolerated = {row["roi_name"]: row["failure_kind"] for row in sidecar["tolerated_source_failures"]}
    assert tolerated == {
        "Bad": "structural_roi_error", "far_oar": "degenerate_mask",
        "noref_oar": "extraction_error", "small_bowel1": "structural_roi_error",
        "orphan_model": "degenerate_mask",
    }
    governed = {(r["roi_name"], r["failure_kind"]) for r in sidecar["rows"]}
    assert ("GTV_bad", "unmeasurable_source_contour") in governed
    assert ("Marker", "nonvolumetric_geometry") in governed
    table, _ = outputs["P1/C1/radiomics_robustness_ct.parquet"]
    assert set(table.structure) == {
        "GTV1", "CTV1", "PTV_zedge", "PTV_rowedge", "CTV_thin", "GTV_small",
        "urinary_bladder", "CTV_eval", "GTV_model"}
    assert set(table.segmentation_source) == {"Manual", "AutoRTS_total", "Custom",
                                              "CustomModel:ModelA"}
    impossible = table[table.robustness_status == "geometrically_impossible"]
    assert "translation_outside_image" in set(impossible.reason_code)
    ledger = outputs["P1/C1/metadata/radiomics_robustness_identity.json"]
    assert "GTV_nocat" in json.dumps(ledger)
    source_only = outputs["P4/C1/metadata/radiomics_robustness_source_dispositions.json"]
    assert source_only["measurement_outcome"] == "source_only_nonvolumetric"
    assert source_only["source_only_basis"]["collected_mask_count"] == 11
    assert "perturbation_identity_validation_failed" in json.dumps(
        outputs["P5/C1/metadata/radiomics_robustness_identity.json"])
    gaps, _ = outputs["P6/C1/radiomics_robustness_ct.parquet"]
    assert (gaps.robustness_status == "feature_not_evaluable").any()
    assert base["codes"]["P7"] == 1 and outputs[
        "P7/C1/metadata/radiomics_robustness_source_dispositions.json"][
        "measurement_outcome"] == "failed_extraction"
    assert base["codes"]["P9"] == 1
    assert base["codes"]["P1"] == base["codes"]["P4"] == 0
    assert "Preparing perturbations for 9 robustness ROI(s) with 3 worker(s)" in runs["current"]["log"]
    assert "with 5 worker(s)" in runs["current_other_budget"]["log"]


@pytest.mark.parametrize("variant", ["current", "current_other_budget", "current_in_process"])
def test_outputs_match_base_revision(runs, variant):
    _assert_equivalent(runs["base"], runs[variant])


def _ordered_table(frame, ids):
    frame = frame.copy()
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(lambda v: _normalize_text(v, ids))
    return frame


@pytest.mark.parametrize("variant", ["current_other_budget", "current_in_process"])
def test_published_order_does_not_depend_on_workers(runs, variant):
    """Rows, columns and ledger entries come out in one order for any budget."""
    reference, other = runs["current"]["outputs"], runs[variant]["outputs"]
    reference_ids, other_ids = _run_ids(reference), _run_ids(other)
    compared = 0
    for key, value in reference.items():
        if isinstance(value, tuple):
            (frame, schema), (other_frame, other_schema) = value, other[key]
            assert other_schema.equals(schema, check_metadata=True), key
            pd.testing.assert_frame_equal(_ordered_table(other_frame, other_ids),
                                          _ordered_table(frame, reference_ids),
                                          check_exact=True, obj=key)
            compared += 1
        elif key.endswith("radiomics_robustness_identity.json"):
            assert _normalize(other[key], other_ids) == _normalize(value, reference_ids), key
    assert compared >= 6
