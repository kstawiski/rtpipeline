"""Courses that meet neither voiding condition produce the same outputs as f16d72a.

The fix for per-ROI structural statuses and per-perturbation feature gaps must
not change any output of a course that completes today. This test exports the
base revision's package with ``git archive`` and runs it and the working tree
through ``robustness_equivalence_driver.py`` on the same synthetic inputs at
the same absolute paths: the campaign course step (table, disposition sidecar,
identity ledger, completion receipt) for sequential and parallel extraction,
the manifest cohort aggregate and the explicit-input aggregate.

Values that identify a run or the deciding code are replaced by placeholders
before comparison: run identifiers, the code identity and the digests of
artifacts that embed them. Everything else must match exactly, including
column order and Arrow types. Since 2026-09-24 course tables are written in
ROBUSTNESS_TABLE_ROW_ORDER, so Parquet rows are compared in canonical order
and the table byte size is masked; cohort CoV, which depends on that order
in its last bits, is checked against the unchanged aggregation of the base
revision's raw values in the new order.

The inputs cover the paths the fix touches without triggering it: healthy
RS_custom and custom-model sources, an unselected unparseable Manual ROI
(tolerated before and after), a point ROI, a selected contourless declaration
in RS_custom (silently skipped before and after), a primary arm below minimum
in every perturbation and an undefined feature in every perturbation (no
feature-set difference).
"""
from __future__ import annotations

import copy
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pydicom
import pytest
import yaml
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from test_science_batch_c import _build_real_rtstruct

ROOT = Path(__file__).resolve().parents[1]
BASE_REVISION = "f16d72a"
DRIVER = Path(__file__).with_name("robustness_equivalence_driver.py")
INPUT_NAMES = {"CT", "RS.dcm", "RS_custom.dcm", "Segmentation_CustomModels", "inputs.json"}

COURSES = {
    "P1": {"in_cohort": True, "parallel": False, "unselected_bad": True, "custom": True},
    "P2": {"in_cohort": True, "parallel": True, "model": True},
    "P3": {"in_cohort": True, "parallel": False},
    "P4": {"in_cohort": False, "parallel": False, "primary_below_minimum_everywhere": True},
    "P5": {"in_cohort": False, "parallel": True, "mcc_undefined_everywhere": True},
}


def _append_roi(ds, number, name, contours):
    roi = Dataset()
    roi.ROINumber = number
    roi.ROIName = name
    roi.ReferencedFrameOfReferenceUID = ds.StructureSetROISequence[0].ReferencedFrameOfReferenceUID
    ds.StructureSetROISequence.append(roi)
    item = Dataset()
    item.ReferencedROINumber = number
    if contours is not None:
        item.ContourSequence = Sequence(contours)
    ds.ROIContourSequence.append(item)


def _contour(kind, data, count=None):
    contour = Dataset()
    contour.ContourGeometricType = kind
    contour.ContourData = data
    contour.NumberOfContourPoints = count if count is not None else len(data) // 3
    return contour


def _build_course(course: Path, spec: dict) -> None:
    ct = course / "CT"
    ct.mkdir(parents=True)
    rtstruct = _build_real_rtstruct(ct, n_slices=8, side=24)
    mask = np.zeros((24, 24, 8), dtype=bool)
    mask[3:9, 3:9, 2:6] = True
    rtstruct.add_roi(mask=mask, name="small_bowel1")
    rtstruct.save(str(course / "RS.dcm"))
    ds = pydicom.dcmread(course / "RS.dcm")
    next(r for r in ds.StructureSetROISequence if r.ROIName == "PTV").ROIName = "GTV1"
    _append_roi(ds, 50, "Marker1", [_contour("POINT", [2.0, 2.0, 1.0])])
    ds.save_as(course / "RS.dcm")
    if spec.get("custom"):
        custom = copy.deepcopy(ds)
        _append_roi(custom, 60, "GTV_empty", [])
        custom.SOPInstanceUID = pydicom.uid.generate_uid()
        custom.file_meta.MediaStorageSOPInstanceUID = custom.SOPInstanceUID
        custom.save_as(course / "RS_custom.dcm")
    if spec.get("model"):
        model = course / "Segmentation_CustomModels" / "ModelA"
        model.mkdir(parents=True)
        shutil.copyfile(course / "RS.dcm", model / "rtstruct.dcm")
    if spec.get("unselected_bad"):
        _append_roi(ds, 70, "Bad", [_contour("POINT", [2.0, 2.0, 1.0], count=9)])
        ds.save_as(course / "RS.dcm")
    series_uid = str(pydicom.dcmread(next(ct.glob("*.dcm"))).SeriesInstanceUID)
    (course / "inputs.json").write_text(json.dumps({**spec, "series_uid": series_uid}))


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


def _run(package_root: Path, work: Path, inputs: Path, config: Path) -> dict:
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(inputs, work)
    env = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-B", "-s", str(DRIVER), str(package_root), str(work / "Output"),
         str(config)],
        env=env, capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    codes = json.loads(result.stdout.strip().splitlines()[-1])
    return {"codes": codes, "outputs": _collect(work / "Output")}


def _collect(output_root: Path) -> dict:
    collected = {}
    for path in sorted(output_root.rglob("*")):
        relative = path.relative_to(output_root)
        if path.is_dir() or (len(relative.parts) > 2 and relative.parts[2] in INPUT_NAMES):
            continue
        if path.suffix == ".parquet":
            collected[str(relative)] = (pd.read_parquet(path), pq.read_schema(path))
        elif path.suffix == ".xlsx":
            collected[str(relative)] = pd.read_excel(path, sheet_name=None)
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
    """Replace run and code identity, and digests of artifacts embedding them."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "code_identity":
                out[key] = "<code identity>"
            elif key == "sha256" and parent in {"measured_output", "source_dispositions"}:
                out[key] = "<digest of run-bound artifact>"
            elif key == "size_bytes" and parent == "measured_output":
                # 2026-09-24: course tables are now written in
                # ROBUSTNESS_TABLE_ROW_ORDER, so their byte size may change.
                out[key] = "<size of reordered table>"
            else:
                out[key] = _normalize(item, ids, key)
        return out
    if isinstance(value, list):
        return [_normalize(item, ids, parent) for item in value]
    return _normalize_text(value, ids)


def _normalize_frame(frame: pd.DataFrame, ids) -> pd.DataFrame:
    frame = frame.copy()
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(lambda v: _normalize_text(v, ids))
        if column in {"table_sha256", "source_dispositions_sha256"}:
            frame[column] = "<digest of run-bound artifact>"
    return frame


COV_COLUMNS = ["cov_pct", "cov_pct_q1", "cov_pct_q3"]
SUMMARY_KEYS = ["structure", "segmentation_source", "extraction_arm", "feature_name"]


def _with_cov_in_table_order(base_outputs: dict, workbook: str, sheet: str,
                             frame: pd.DataFrame, rob_config) -> pd.DataFrame:
    """Base summary with CoV recomputed from base raw values in table row order.

    2026-09-24: per-subject CoV is np.mean/np.std over a subject's rows in
    table order, so writing course tables in ROBUSTNESS_TABLE_ROW_ORDER moves
    CoV in the last bits. Aggregation code is unchanged, so the expected CoV is
    the unchanged summary applied to the base revision's own raw values, each
    course reordered by that row order. Every other value is the base value.
    """
    from rtpipeline import radiomics_robustness as rr

    if not set(COV_COLUMNS) & set(frame.columns) or frame.empty:
        return frame
    raw, _ = base_outputs[workbook.replace(".xlsx", "_raw_values.parquet")]
    ordered = pd.concat(
        [rr._order_robustness_rows(course) for _, course in
         raw.groupby(["patient_id", "course_id"], sort=False)],
        ignore_index=True,
    )
    group_columns = (
        ["segmentation_source", "structure", "extraction_arm", "feature_name"]
        if sheet.endswith("per_source") or sheet == "per_source_summary" else None
    )
    summary = rr.summarize_feature_stability(ordered, rob_config, group_columns=group_columns)
    cov = [column for column in COV_COLUMNS if column in frame.columns]
    merged = frame.drop(columns=cov).merge(summary[SUMMARY_KEYS + cov], on=SUMMARY_KEYS,
                                           how="left", validate="one_to_one")
    assert len(merged) == len(frame)
    # Store the recomputed values exactly as the cohort workbook stores values.
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        merged[list(frame.columns)].to_excel(writer, sheet_name=sheet[:31], index=False)
    buffer.seek(0)
    return pd.read_excel(buffer, sheet_name=sheet[:31])


def _in_row_order(frame: pd.DataFrame) -> pd.DataFrame:
    order = frame.astype(str).agg("\x1f".join, axis=1).sort_values(kind="stable").index
    return frame.loc[order].reset_index(drop=True)


def _assert_equivalent(base: dict, current: dict, rob_config) -> None:
    assert current["codes"] == base["codes"]
    assert sorted(current["outputs"]) == sorted(base["outputs"])
    base_ids, current_ids = _run_ids(base["outputs"]), _run_ids(current["outputs"])
    assert sorted(base_ids.values()) == sorted(current_ids.values())
    for key, expected in base["outputs"].items():
        observed = current["outputs"][key]
        if isinstance(expected, tuple):
            (expected_frame, expected_schema), (observed_frame, observed_schema) = expected, observed
            assert observed_schema.equals(expected_schema, check_metadata=True), key
            # 2026-09-24: the course table (and so the raw cohort values
            # concatenated from it) is written in ROBUSTNESS_TABLE_ROW_ORDER
            # instead of extraction order. Rows, values, columns and types
            # must still match exactly; only the row order is not compared.
            pd.testing.assert_frame_equal(
                _in_row_order(_normalize_frame(observed_frame, current_ids)),
                _in_row_order(_normalize_frame(expected_frame, base_ids)),
                check_exact=True, obj=key,
            )
        elif isinstance(expected, dict) and expected and all(
            isinstance(v, pd.DataFrame) for v in expected.values()
        ):
            assert list(observed) == list(expected), key
            for sheet, frame in expected.items():
                frame = _with_cov_in_table_order(base["outputs"], key, sheet, frame, rob_config)
                pd.testing.assert_frame_equal(
                    _normalize_frame(observed[sheet], current_ids),
                    _normalize_frame(frame, base_ids),
                    check_exact=True, obj=f"{key}:{sheet}",
                )
        else:
            assert _normalize(observed, current_ids) == _normalize(expected, base_ids), key


def test_non_triggering_courses_match_base_revision(tmp_path):
    inputs = tmp_path / "inputs"
    for patient, spec in COURSES.items():
        _build_course(inputs / "Output" / patient / "C1", spec)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"radiomics_robustness": {
        "enabled": True,
        "segmentation_perturbation": {
            "apply_to_structures": ["GTV*"], "small_volume_changes": [-0.15, 0.0],
            "max_translation_mm": 0.0, "n_random_contour_realizations": 0,
            "noise_levels": [0.0, 10.0],
        },
    }}))
    base_root = _export_base(tmp_path / "base")
    base = _run(base_root, tmp_path / "work", inputs, config)
    current = _run(ROOT, tmp_path / "work", inputs, config)
    # The fixture must exercise what it claims before equality means anything.
    assert set(base["codes"].values()) == {0}
    p1 = base["outputs"]["P1/C1/metadata/radiomics_robustness_source_dispositions.json"]
    assert [t["roi_name"] for t in p1["tolerated_source_failures"]] == ["Bad"]
    assert {(r["segmentation_source"], r["roi_name"]) for r in p1["rows"]} == {
        ("Manual", "Marker1"), ("Custom", "Marker1")}
    assert any(k.startswith("_RESULTS/radiomics_robustness_summary") for k in base["outputs"])
    p4, _ = base["outputs"]["P4/C1/radiomics_robustness_ct.parquet"]
    assert not p4.feature_name[p4.extraction_arm == "primary_resegmented"].str.contains(
        "firstorder").any()
    p5, _ = base["outputs"]["P5/C1/radiomics_robustness_ct.parquet"]
    assert not p5.feature_name[p5.extraction_arm == "sensitivity_raw"].str.endswith("MCC").any()
    from rtpipeline.radiomics_robustness import RobustnessConfig

    rob_config = RobustnessConfig.from_dict(yaml.safe_load(config.read_text())["radiomics_robustness"])
    _assert_equivalent(base, current, rob_config)
