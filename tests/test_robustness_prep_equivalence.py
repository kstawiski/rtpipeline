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

Since 2026-09-24 the base is 35af111 and robustness reads selected ROIs
through the scoped reader of main CT radiomics. Courses S1 (a selected ROI on
an image outside the CT series), S2 (a selected ROI with a one-point contour
item) and P7 (a selected ROI beyond the image) fail on the base and complete
now; S3 (a selected ROI with a two-point item) completes on both with that
ROI's mask changed to the main path's. Every other course, including the
negative controls N1-N3, must publish exactly what the base published.
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

from robustness_prep_fixture import build_course, realistic_rois

ROOT = Path(__file__).resolve().parents[1]
# 2026-09-24: the base moves from a027945 to 35af111, whose outputs this
# harness proved equal to a027945's, because the scoped-reader change is
# compared with its direct predecessor and the injected read-failure seam
# (radiomics._rt_utils_indexed_roi_mask) exists only from 8af132e on.
BASE_REVISION = "35af111"
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
    # Scoped reader, scenario (a): a selected Manual ROI with a contour on an
    # image outside the CT series, which the global reference list names,
    # beside a normal selected ROI and an unselected ROI with the same defect.
    "S1": dict(GRID, parallel=True, workers=2, seed=11, rois=[
        _roi("Manual", "GTV1"), *_organs("Manual", "OAR", 3),
        _roi("AutoRTS_total", "urinary_bladder", center=(26, 16, 6), radii=(5, 5, 3)),
        *_organs("AutoRTS_total", "organ", 3)],
        defects={"Manual": [{"name": "CTV_oos", "kind": "foreign_image_reference"},
                            {"name": "oar_oos", "kind": "foreign_image_reference"}]},
        extra_catalog=[["Manual", "CTV_oos"], ["Manual", "oar_oos"]]),
    # Scenario (b): a selected AutoRTS ROI with a one-point contour item.
    "S2": dict(GRID, parallel=True, workers=2, seed=12, rois=[
        _roi("Manual", "GTV1"), *_organs("Manual", "OAR", 2),
        _roi("AutoRTS_total", "urinary_bladder", center=(26, 16, 6), radii=(5, 5, 3)),
        *_organs("AutoRTS_total", "organ", 3)],
        defects={"AutoRTS_total": [{"name": "urinary_bladder", "kind": "one_point_item"},
                                   {"name": "organ_01", "kind": "one_point_item"}]}),
    # A selected ROI with a two-point item: rt_utils draws the segment, the
    # scoped reader (and main CT radiomics) withholds it. The one deliberate
    # change for a course that completes on the base revision.
    "S3": dict(GRID, parallel=True, workers=2, seed=13, rois=[
        _roi("Manual", "GTV1"), _roi("Manual", "CTV1", radii=(7, 8, 4)),
        *_organs("Manual", "OAR", 2)],
        defects={"Manual": [{"name": "GTV1", "kind": "two_point_item"},
                            {"name": "OAR_00", "kind": "two_point_item"}]}),
    # Negative controls: whole-source failures and a technical read failure
    # that scope does not explain stay fatal, identically.
    "N1": dict(GRID, parallel=True, workers=2, seed=14, rois=[
        _roi("Manual", "GTV1"), *_organs("Manual", "OAR", 2)],
        defects={"Manual": [{"name": "", "kind": "drop_roi_observations"}]}),
    "N2": dict(GRID, parallel=True, workers=2, seed=15, rois=[
        _roi("Manual", "GTV1"), _roi("Custom", "CTV_eval", center=(16, 24, 9), radii=(5, 4, 3)),
        *_organs("Custom", "custom", 2)],
        defects={"Custom": [{"name": "", "kind": "not_dicom"}]}),
    "N3": dict(GRID, parallel=True, workers=2, seed=16, inject_read_failure_for=["GTV1"], rois=[
        _roi("Manual", "GTV1"), _roi("Manual", "CTV1", radii=(7, 8, 4)),
        *_organs("Manual", "OAR", 2)]),
}
# Courses whose outputs change on purpose (see test_scoped_reader_* below).
# Every other course, completed or failed, must publish what the base did.
CHANGED_COURSES = {"P7", "S1", "S2", "S3"}


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


def _course_outputs(run: dict, courses) -> dict:
    return {"codes": {c: v for c, v in run["codes"].items() if c in courses},
            "outputs": {k: v for k, v in run["outputs"].items() if k.split("/")[0] in courses}}


def _assert_equivalent(base: dict, current: dict, courses=None) -> None:
    if courses is not None:
        base, current = _course_outputs(base, courses), _course_outputs(current, courses)
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
    _assert_equivalent(runs["base"], runs[variant], set(COURSES) - CHANGED_COURSES)


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


# ---------------------------------------------------------------------------
# Scoped RTSTRUCT reader (2026-09-24). Robustness reads selected ROIs through
# the reader of main CT radiomics (rtstruct_geometry.ScopedRTStruct).
# ---------------------------------------------------------------------------

def _sidecar(outputs, course):
    return outputs[f"{course}/C1/metadata/radiomics_robustness_source_dispositions.json"]


def _dispositions(sidecar):
    return sorted((r["roi_name"], r["status"], r["failure_kind"], r["structural_code"])
                  for r in sidecar["rows"])


def _tolerated(sidecar):
    return sorted((r["roi_name"], r["failure_kind"], r["structural_code"])
                  for r in sidecar["tolerated_source_failures"])


def test_scoped_reader_regressions_fail_on_base(runs):
    base = runs["base"]
    for course in ("S1", "S2", "P7"):
        assert base["codes"][course] == 1, course
        assert _sidecar(base["outputs"], course)["measurement_outcome"] == "failed_extraction"
    log = base["log"]
    assert ("/S1/C1/RS.dcm: Loaded RTStruct references image(s) that are not contained "
            "in input series data") in log
    assert "Expected ROI 'urinary_bladder' in " in log
    assert "/S2/C1/RS_auto.dcm could not be read: OpenCV" in log and "fillPoly" in log
    assert "Expected ROI 'GTV_far' in " in log


@pytest.mark.parametrize("variant", ["current", "current_other_budget", "current_in_process"])
def test_scoped_reader_regressions_pass(runs, variant):
    run = runs[variant]
    outputs = run["outputs"]
    for course in ("S1", "S2", "S3", "P7"):
        assert run["codes"][course] == 0, course
        assert _sidecar(outputs, course)["measurement_outcome"] == "measured", course

    # (a) The out-of-scope selected ROI gets the governed disposition the main
    # path publishes as unresolved_source_scope; the normal ROIs are measured;
    # the unselected out-of-scope ROI is a tolerated failure.
    s1 = _sidecar(outputs, "S1")
    assert _dispositions(s1) == [("CTV_oos", "structural_nonmeasurement",
                                  "unresolved_source_scope", "ROI_UNRESOLVED_SOURCE_SCOPE")]
    assert _tolerated(s1) == [("oar_oos", "structural_roi_error", "ROI_UNRESOLVED_SOURCE_SCOPE")]
    table, _ = outputs["S1/C1/radiomics_robustness_ct.parquet"]
    assert set(zip(table.segmentation_source, table.structure)) == {
        ("Manual", "GTV1"), ("AutoRTS_total", "urinary_bladder")}

    # (b) The selected ROI with a one-point item is measured. The unselected
    # one is read by rt_utils as before and stays a tolerated failure.
    s2 = _sidecar(outputs, "S2")
    assert _dispositions(s2) == []
    assert _tolerated(s2) == [("organ_01", "extraction_error", "ROI_EXTRACTION_FAILED")]
    table, _ = outputs["S2/C1/radiomics_robustness_ct.parquet"]
    assert set(zip(table.segmentation_source, table.structure)) == {
        ("Manual", "GTV1"), ("AutoRTS_total", "urinary_bladder")}

    # A contour beyond the image is out of scope, as in the main path.
    assert _dispositions(_sidecar(outputs, "P7")) == [
        ("GTV_far", "structural_nonmeasurement", "unresolved_source_scope",
         "ROI_UNRESOLVED_SOURCE_SCOPE")]
    table, _ = outputs["P7/C1/radiomics_robustness_ct.parquet"]
    assert set(table.structure) == {"GTV1"}


def test_scoped_reader_negative_controls_still_fail(runs):
    for variant in ("base", "current"):
        run = runs[variant]
        for course in ("N1", "N2", "N3"):
            assert run["codes"][course] == 1, (variant, course)
            assert _sidecar(run["outputs"], course)["measurement_outcome"] == "failed_extraction"
        log = run["log"]
        # rt_utils rejects the whole source for a reason other than scope.
        assert "/N1/C1/RS.dcm: Please check that the existing RTStruct is valid" in log
        # A technical read failure of a selected in-scope ROI.
        assert "could not be read: injected read failure for ROI 'GTV1'" in log


def test_two_point_item_changes_only_that_selected_roi(runs):
    """Deliberate: the main path withholds a two-point item that rt_utils draws."""
    base, current = runs["base"], runs["current"]
    assert base["codes"]["S3"] == current["codes"]["S3"] == 0
    for key, value in base["outputs"].items():
        if key.startswith("S3/") and not key.endswith(".parquet"):
            assert _normalize(current["outputs"][key], _run_ids(current["outputs"])) == \
                _normalize(value, _run_ids(base["outputs"])), key
    key = "S3/C1/radiomics_robustness_ct.parquet"
    ids = ["segmentation_source", "structure", "perturbation_id", "extraction_arm", "feature_name"]
    frames = []
    for run in (base, current):
        frame = run["outputs"][key][0].drop(columns=["run_identifier"]).astype(str)
        frames.append(frame.set_index(ids).sort_index())
    changed = (frames[0] != frames[1])
    assert frames[0].index.equals(frames[1].index)
    assert set(changed.columns[changed.any()]) == {"perturbed_mask_identity"}
    assert set(changed.index[changed.any(axis=1)].get_level_values("structure")) == {"GTV1"}


# Mask-level proofs, independent of the course runs.

def _series_and_raw(course, rs_name):
    import pydicom
    from rt_utils import image_helper
    from rt_utils.rtstruct import RTStruct

    series = image_helper.load_sorted_image_series(str(course / "CT"))
    return series, RTStruct(series, pydicom.dcmread(str(course / rs_name)))


def _scoped_masks(course, rs_name):
    from rtpipeline import radiomics as rm
    from rtpipeline.roi_requiredness import Requiredness, inspect_rtstruct

    rs = course / rs_name
    required = {o.name: Requiredness.ANALYSIS_REQUIRED for o in inspect_rtstruct(rs).named_rois}
    sink = []
    masks = rm._rtstruct_masks(
        course / "CT", rs, failure_outcomes=sink, tolerate_unselected=True,
        requiredness_by_roi=required, unmeasurable_required_is_disposition=True,
        contourless_required_is_absence=True, retain_mask=lambda _name: True,
        scoped_reader=True)
    return masks, sink


def _assert_same_bytes(observed, expected):
    assert observed.dtype == expected.dtype == bool
    assert observed.shape == expected.shape
    assert observed.tobytes() == expected.tobytes()


def test_scoped_masks_are_byte_identical_for_in_scope_rois(tmp_path):
    """Every in-scope ROI without area-less items: scoped mask == rt_utils mask."""
    from rtpipeline.rtstruct_geometry import _contour_rasterizable, create_scoped_rtstruct

    full = tmp_path / "Output" / "B1" / "C1"
    build_course(full, _full_course(seed=21))
    realistic = tmp_path / "Output" / "B2" / "C1"
    build_course(realistic, dict(rows=48, columns=48, slices=20, seed=22,
                                 rois=realistic_rois(48, 48, 20)))
    files = [(full, "RS.dcm"), (full, "RS_auto.dcm"), (full, "RS_custom.dcm"),
             (full, "Segmentation_CustomModels/ModelA/rtstruct.dcm"),
             (realistic, "RS.dcm"), (realistic, "RS_auto.dcm"), (realistic, "RS_custom.dcm")]
    compared = withheld = 0
    for course, rs_name in files:
        scoped = create_scoped_rtstruct(course / "CT", course / rs_name)
        masks, _ = _scoped_masks(course, rs_name)
        _, raw = _series_and_raw(course, rs_name)
        for name, result in scoped.by_name.items():
            if result.code:
                continue
            if not all(_contour_rasterizable(c) for c in result.contours):
                withheld += 1
                continue
            try:
                expected = raw.get_roi_mask_by_name(name)
            except Exception:
                continue  # rt_utils cannot read it at all (no image reference)
            _assert_same_bytes(masks[name], expected)
            _assert_same_bytes(masks[name], scoped.get_roi_mask_by_name(name))
            compared += 1
    print(f"byte-identical in-scope ROI masks: {compared}; with withheld items: {withheld}")
    assert compared >= 120


def test_scoped_mask_for_area_less_items_matches_main_path(tmp_path):
    import copy

    from rt_utils.rtstruct import RTStruct
    from rtpipeline.rtstruct_geometry import create_scoped_rtstruct

    course = tmp_path / "Output" / "S2" / "C1"
    build_course(course, COURSES["S2"])
    series, raw = _series_and_raw(course, "RS_auto.dcm")
    with pytest.raises(Exception, match="fillPoly"):
        raw.get_roi_mask_by_name("urinary_bladder")
    masks, sink = _scoped_masks(course, "RS_auto.dcm")
    main_path = create_scoped_rtstruct(course / "CT", course / "RS_auto.dcm")
    _assert_same_bytes(masks["urinary_bladder"], main_path.get_roi_mask_by_name("urinary_bladder"))
    assert sink == []
    # Without the one-point item rt_utils gives the same mask.
    cleaned = copy.deepcopy(raw.ds)
    for name in ("urinary_bladder",):
        contours = next(c for c in cleaned.ROIContourSequence if int(c.ReferencedROINumber) == next(
            int(r.ROINumber) for r in cleaned.StructureSetROISequence if r.ROIName == name))
        contours.ContourSequence = [c for c in contours.ContourSequence if len(c.ContourData) > 3]
    _assert_same_bytes(masks["urinary_bladder"],
                       RTStruct(series, cleaned).get_roi_mask_by_name("urinary_bladder"))

    course = tmp_path / "Output" / "S3" / "C1"
    build_course(course, COURSES["S3"])
    _, raw = _series_and_raw(course, "RS.dcm")
    masks, _ = _scoped_masks(course, "RS.dcm")
    main_path = create_scoped_rtstruct(course / "CT", course / "RS.dcm")
    _assert_same_bytes(masks["GTV1"], main_path.get_roi_mask_by_name("GTV1"))
    drawn = raw.get_roi_mask_by_name("GTV1")
    # rt_utils draws the two-point segment; the main path does not.
    assert drawn.sum() > masks["GTV1"].sum() and not (masks["GTV1"] & ~drawn).any()


def test_out_of_scope_roi_does_not_void_its_source(tmp_path):
    from rtpipeline.rtstruct_geometry import create_scoped_rtstruct
    from rt_utils import RTStructBuilder

    course = tmp_path / "Output" / "S1" / "C1"
    build_course(course, COURSES["S1"])
    with pytest.raises(Exception, match="not contained in input series data"):
        RTStructBuilder.create_from(str(course / "CT"), str(course / "RS.dcm"))
    masks, sink = _scoped_masks(course, "RS.dcm")
    assert sorted((o["roi_name"], o["status"], o["structural_code"]) for o in sink) == [
        ("CTV_oos", "structural_nonmeasurement", "ROI_UNRESOLVED_SOURCE_SCOPE"),
        ("oar_oos", "structural_nonmeasurement", "ROI_UNRESOLVED_SOURCE_SCOPE")]
    assert "CTV_oos" not in masks
    # The in-scope ROIs read exactly as rt_utils reads them per ROI.
    _, raw = _series_and_raw(course, "RS.dcm")
    main_path = create_scoped_rtstruct(course / "CT", course / "RS.dcm")
    for name in ("GTV1", "OAR_00", "OAR_01", "OAR_02"):
        _assert_same_bytes(masks[name], raw.get_roi_mask_by_name(name))
        _assert_same_bytes(masks[name], main_path.get_roi_mask_by_name(name))


def test_scoped_reader_keeps_whole_source_failures_fatal(tmp_path):
    from rtpipeline import radiomics as rm

    course = tmp_path / "Output" / "N1" / "C1"
    build_course(course, COURSES["N1"])
    with pytest.raises(rm.RadiomicsCourseExtractionError,
                       match="Failed to construct RTSTRUCT reader .*RTStruct is valid"):
        _scoped_masks(course, "RS.dcm")
