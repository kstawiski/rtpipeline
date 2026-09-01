"""B4+C4 — all-series (non-course) CT radiomics adapter + 4DCT dedup.

Real pyradiomics/conda extraction needs the conda env (out of unit scope), so the per-course
dispatch is mocked with a fake that (a) asserts the materialized temp course-tree shape and
(b) writes a tiny ``radiomics_ct.xlsx``. That exercises everything B4+C4 own: eligible-class
selection, the C3 CBCT denylist guard, C4 per-study 4DCT dedup (ave-only / 50%-phase fallback),
temp-tree materialization, auto-RTSTRUCT discovery, provenance aggregation, temp cleanup,
serial==parallel dispatch parity, and course-path isolation (radiomics_all.xlsx untouched).
"""
import json
import os
import sys
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

import rtpipeline.cli as cli
import rtpipeline.radiomics as rad
import rtpipeline.radiomics_conda as radconda
import rtpipeline.radiomics_parallel as radpar
from rtpipeline.inventory import TS_TASK_BY_CLASS, output_dir_for_image_class
from rtpipeline.layout import build_course_dirs
from rtpipeline.segmentation import _series_artifact_dirs
from course_contract_test_utils import write_synthetic_rtstruct


def _write_valid_ct(path: Path, *, series_uid: str, study_uid: str) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "CT"
    dataset.PatientID = "P1"
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.FrameOfReferenceUID = generate_uid()
    dataset.Rows = 2
    dataset.Columns = 2
    dataset.PixelSpacing = [1.0, 1.0]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.SliceThickness = 1.0
    dataset.save_as(str(path), enforce_file_format=True)


def _write_dual_arm_test_publication(
    output: Path, base_records: list[dict[str, Any]]
) -> Path:
    from rtpipeline.radiomics_ct_contract import (
        classify_ct_roi,
        disposition_rows_for_arms,
        write_ct_publication_atomic,
    )

    records: list[dict[str, Any]] = []
    for index, base in enumerate(base_records, start=1):
        source = str(base.get("segmentation_source") or "AutoRTS_total")
        roi_name = str(base.get("roi_original_name") or base.get("roi_name") or "liver")
        normalized = {
            "modality": "CT",
            "segmentation_source": source,
            "roi_name": roi_name,
            "roi_original_name": roi_name,
            "patient_id": str(base.get("patient_id") or "P1"),
            "course_id": str(base.get("course_id") or "C1"),
            "series_uid": str(base.get("series_uid") or f"series-{index}"),
            "mask_identity": str(base.get("mask_identity") or f"mask-{index}"),
            "stable_roi_identifier": str(
                base.get("stable_roi_identifier") or f"roi-{index}"
            ),
            **base,
        }
        pair = disposition_rows_for_arms(
            normalized,
            decision=classify_ct_roi(source, roi_name),
            disposition="success",
            detail="test success",
            failure_kind="none",
            run_identifier="test-run",
            code_revision="test-revision",
            native_voxel_count=120,
            required=False,
            configured_parameter_hashes={
                "primary_resegmented": "test-primary",
                "sensitivity_raw": "test-sensitivity",
            },
            effective_hashes={
                "primary_resegmented": "effective-primary",
                "sensitivity_raw": "effective-sensitivity",
            },
        )
        for record in pair:
            for key, value in base.items():
                if key not in record or key == "feature":
                    record[key] = value
        records.extend(pair)
    write_ct_publication_atomic(pd.DataFrame(records), output)
    return output


class _Cfg:
    """Minimal config: the dispatch is mocked, so only output_root + effective_workers are used."""

    def __init__(self, output_root):
        self.output_root = Path(output_root)
        self.resume = False

    def effective_workers(self):
        return 1


def _row(image_class, series_uid, *, study_uid="S", series_description="", output_dir="x"):
    return {
        "image_class": image_class,
        "series_uid": series_uid,
        "study_uid": study_uid,
        "series_description": series_description,
        "ts_task": TS_TASK_BY_CLASS.get(image_class, "none"),
        "output_dir": output_dir,
        "status": "segmented",
    }


# --------------------------------------------------------------------------- #
# selection + C4 dedup
# --------------------------------------------------------------------------- #
def test_select_base_classes_and_c3_denylist():
    rows = [
        _row("planning_ct", "u1"),
        _row("diagnostic_ct", "u2"),
        _row("petct_ct", "u3"),
        _row("cbct", "u4"),         # C3 denylist: never radiomic'd
        _row("pt", "u5"),           # PET emission, not a CT radiomics class
        _row("mr_anatomic", "u6"),
        _row("exclude", "u7"),
    ]
    got = {r["series_uid"]: is4d for r, is4d in rad._select_all_series_radiomics_rows(rows)}
    assert got == {"u1": False, "u2": False, "u3": False}


def test_c4_ave_only_when_ave_present():
    rows = [
        _row("fourdct_ave", "ave1", study_uid="S1"),
        _row("fourdct_phase", "ph0", study_uid="S1", series_description="0%"),
        _row("fourdct_phase", "ph50", study_uid="S1", series_description="50%"),
    ]
    sel = rad._select_all_series_radiomics_rows(rows)
    assert len(sel) == 1
    row, is4d = sel[0]
    assert row["series_uid"] == "ave1" and is4d is False


def test_c4_phase_fallback_prefers_50pct():
    rows = [
        _row("fourdct_phase", "ph0", study_uid="S1", series_description="Resp 0%"),
        _row("fourdct_phase", "ph50", study_uid="S1", series_description="Resp 50%"),
        _row("fourdct_phase", "ph90", study_uid="S1", series_description="Resp 90%"),
    ]
    sel = rad._select_all_series_radiomics_rows(rows)
    assert len(sel) == 1
    row, is4d = sel[0]
    assert row["series_uid"] == "ph50" and is4d is True


def test_c4_phase_fallback_first_when_no_50pct():
    rows = [
        _row("fourdct_phase", "phA", study_uid="S1", series_description="phase A"),
        _row("fourdct_phase", "phB", study_uid="S1", series_description="phase B"),
    ]
    sel = rad._select_all_series_radiomics_rows(rows)
    assert len(sel) == 1 and sel[0][0]["series_uid"] == "phA" and sel[0][1] is True


def test_c4_independent_per_study():
    rows = [
        _row("fourdct_ave", "ave1", study_uid="S1"),
        _row("fourdct_phase", "ph1", study_uid="S2", series_description="50%"),
    ]
    got = {r["series_uid"]: is4d for r, is4d in rad._select_all_series_radiomics_rows(rows)}
    assert got == {"ave1": False, "ph1": True}


def test_pick_representative_phase_helper():
    rows = [{"series_description": "0%", "series_uid": "a"},
            {"series_description": "mid 50 %", "series_uid": "b"}]
    # whitespace-stripped match: "50 %" -> "50%"
    assert rad._pick_representative_4dct_phase(rows)["series_uid"] == "b"
    assert rad._pick_representative_4dct_phase([{"series_uid": "x"}])["series_uid"] == "x"


def test_c4_representative_chosen_among_segmented_only():
    # The 50% phase was NOT segmented (no RTSTRUCT); the only segmented volume is ph0. B4 must pick
    # the segmented one, not the (preferred-but-absent) 50% phase, so the study is not silently dropped.
    rows = [
        _row("fourdct_phase", "ph0", study_uid="S1", series_description="0%"),
        _row("fourdct_phase", "ph50", study_uid="S1", series_description="50%"),
    ]
    segmented = {"ph0"}
    sel = rad._select_all_series_radiomics_rows(rows, has_rtstruct=lambda r: r["series_uid"] in segmented)
    assert len(sel) == 1
    row, is4d = sel[0]
    assert row["series_uid"] == "ph0" and is4d is True


def test_c4_prefers_50pct_among_segmented():
    rows = [
        _row("fourdct_phase", "ph0", study_uid="S1", series_description="0%"),
        _row("fourdct_phase", "ph50", study_uid="S1", series_description="50%"),
        _row("fourdct_phase", "ph90", study_uid="S1", series_description="90%"),
    ]
    segmented = {"ph50", "ph90"}  # ph0 unsegmented
    sel = rad._select_all_series_radiomics_rows(rows, has_rtstruct=lambda r: r["series_uid"] in segmented)
    assert len(sel) == 1 and sel[0][0]["series_uid"] == "ph50" and sel[0][1] is True


def test_c4_ave_without_rtstruct_falls_back_to_segmented_phase():
    rows = [
        _row("fourdct_ave", "ave1", study_uid="S1"),
        _row("fourdct_phase", "ph50", study_uid="S1", series_description="50%"),
    ]
    segmented = {"ph50"}  # ave failed to segment
    sel = rad._select_all_series_radiomics_rows(rows, has_rtstruct=lambda r: r["series_uid"] in segmented)
    assert len(sel) == 1 and sel[0][0]["series_uid"] == "ph50" and sel[0][1] is True


def test_c4_study_dropped_when_no_segmented_4dct():
    rows = [_row("fourdct_phase", "ph0", study_uid="S1", series_description="0%")]
    sel = rad._select_all_series_radiomics_rows(rows, has_rtstruct=lambda r: False)
    assert sel == []


def test_empty_study_uid_not_collapsed():
    # two distinct ave-less 4DCT acquisitions with MISSING study_uid must NOT collapse into one bucket
    rows = [
        _row("fourdct_phase", "ph_a", study_uid="", series_description="50%"),
        _row("fourdct_phase", "ph_b", study_uid="", series_description="50%"),
    ]
    got = {r["series_uid"]: is4d for r, is4d in rad._select_all_series_radiomics_rows(rows)}
    assert got == {"ph_a": True, "ph_b": True}


# --------------------------------------------------------------------------- #
# temp-tree materialization + RTSTRUCT discovery
# --------------------------------------------------------------------------- #
def test_materialize_temp_course_tree(tmp_path):
    src = tmp_path / "series"
    src.mkdir()
    series_uid = generate_uid()
    study_uid = generate_uid()
    for i in range(3):
        _write_valid_ct(src / f"s{i}.dcm", series_uid=series_uid, study_uid=study_uid)
    (src / "notdicom.txt").write_text("nope")
    rs = tmp_path / "rtstruct.dcm"
    write_synthetic_rtstruct(
        rs,
        referenced_series_uid=series_uid,
        roi_names=("PTV1",),
    )
    course = tmp_path / "course"
    assert rad._materialize_temp_course_tree(course, src, rs) is True
    ct = course / "DICOM" / "CT"
    assert sorted(p.name for p in ct.glob("*.dcm")) == ["s0.dcm", "s1.dcm", "s2.dcm"]
    rs_auto = course / "RS_auto.dcm"
    assert rs_auto.exists()
    assert pydicom.dcmread(str(rs_auto), stop_before_pixels=True).SOPInstanceUID == pydicom.dcmread(
        str(rs), stop_before_pixels=True
    ).SOPInstanceUID
    contract = json.loads((course / "metadata" / "case_metadata.json").read_text())[
        "course_contract"
    ]
    assert contract["scope"] == "all_series_radiomics_temp"
    assert contract["authority"] == "all_series_radiomics_materializer"
    assert contract["planning_ct"]["series_instance_uid"] == series_uid
    assert contract["authoritative_rtstruct"]["segmentation_source"] == "AutoRTS_total"
    assert contract["dvh"]["metrics_status"] == "not_computed"
    assert rad.load_course_contract(course).authoritative_rtstruct_source == "AutoRTS_total"
    # only DICOM slices are linked
    assert not (ct / "notdicom.txt").exists()


def test_materialize_returns_false_when_no_slices(tmp_path):
    src = tmp_path / "empty"
    src.mkdir()
    rs = tmp_path / "rs.dcm"
    rs.write_bytes(b"r")
    assert rad._materialize_temp_course_tree(tmp_path / "c", src, rs) is False


def test_all_series_temp_contract_is_consumed_only_by_all_series_dispatch(tmp_path, monkeypatch):
    src = tmp_path / "series"
    src.mkdir()
    series_uid = generate_uid()
    _write_valid_ct(src / "slice.dcm", series_uid=series_uid, study_uid=generate_uid())
    rs = write_synthetic_rtstruct(
        tmp_path / "rs.dcm",
        referenced_series_uid=series_uid,
        roi_names=("PTV1",),
    )
    course = tmp_path / ".all_series_radiomics" / "P1" / "series"
    assert rad._materialize_temp_course_tree(course, src, rs) is True

    class _Image:
        def GetSpacing(self):
            return (1.0, 1.0, 1.0)

    class _Extractor:
        def __init__(self):
            self.settings = {"resampledPixelSpacing": [1.0, 1.0, 1.0]}
            self.enabledImagetypes = {"Original": {}}
            self.enabledFeatures = {"firstorder": [], "shape": []}

        def disableAllImageTypes(self):
            self.enabledImagetypes = {}

        def enableImageTypeByName(self, name):
            self.enabledImagetypes[name] = {}

        def disableAllFeatures(self):
            self.enabledFeatures = {}

        def enableFeatureClassByName(self, name):
            self.enabledFeatures[name] = []

        def execute(self, _image, _mask):
            return {
                "original_firstorder_Mean": 1.0,
                "original_shape_VoxelVolume": 4096.0,
            }

    monkeypatch.setattr(rad, "_load_series_image", lambda *_args, **_kwargs: _Image())
    monkeypatch.setattr(rad, "_extractor", lambda *_args, **_kwargs: _Extractor())
    monkeypatch.setattr(rad, "_extractor_large_roi", lambda *_args, **_kwargs: _Extractor())
    import rtpipeline.radiomics_ct_contract as ct_contract

    monkeypatch.setattr(
        ct_contract,
        "resampled_mask_qc",
        lambda *_args, **_kwargs: {
            "morphologic_resampled_voxel_count": 4096,
            "resegment_after_count": 4096,
            "resegment_below_lower_count": 0,
            "resegment_above_upper_count": 0,
            "resegment_nonfinite_count": 0,
            "components_26_before": 1,
            "components_26_after": 1,
            "largest_component_voxel_count_before": 4096,
            "largest_component_voxel_count_after": 4096,
            "resegment_retained_fraction": 1.0,
            "largest_component_retained_fraction": 1.0,
            "largest_component_fraction_after": 1.0,
            "component_count_increased": False,
            "observed_roi_dimensions_after_resegmentation": 3,
        },
    )
    monkeypatch.setattr(
        rad,
        "_rtstruct_masks",
        lambda *_args, **_kwargs: {"PTV1": np.ones((16, 16, 16), dtype=bool)},
    )
    monkeypatch.setattr(rad, "_mask_from_array_like", lambda _image, mask: mask)
    monkeypatch.setattr(rad, "validate_custom_model_output_inventory", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(rad, "list_custom_model_outputs", lambda *_args, **_kwargs: [])

    result = rad.radiomics_for_course(cast(Any, _Cfg(tmp_path)), course, allow_all_series_temp=True)
    assert result.output_path == course / "radiomics_ct.xlsx"
    contract = rad.load_course_contract(course)
    sources = rad._standard_rtstruct_sources(contract, course)
    assert [(source, path.resolve()) for source, path, _ in sources] == [
        ("AutoRTS_total", (course / "RS_auto.dcm").resolve())
    ]
    extracted = pd.read_excel(result.output_path)
    assert set(extracted["segmentation_source"]) == {"AutoRTS_total"}
    with pytest.raises(rad.RadiomicsCourseExtractionError, match="temporary contract"):
        rad.radiomics_for_course(cast(Any, _Cfg(tmp_path)), course)


def test_parallel_all_series_conda_fallback_preserves_scope_opt_in(
    tmp_path, monkeypatch
):
    src = tmp_path / "series"
    src.mkdir()
    series_uid = generate_uid()
    _write_valid_ct(
        src / "slice.dcm", series_uid=series_uid, study_uid=generate_uid()
    )
    rs = write_synthetic_rtstruct(
        tmp_path / "rs.dcm",
        referenced_series_uid=series_uid,
        roi_names=("PTV1",),
    )
    course = tmp_path / ".all_series_radiomics" / "P1" / "series"
    assert rad._materialize_temp_course_tree(course, src, rs) is True

    monkeypatch.setattr(rad, "_load_series_image", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(rad, "_extractor", lambda *_args, **_kwargs: None)
    seen = {}

    def fake_conda(
        course_dir,
        _config,
        _custom=None,
        *,
        allow_all_series_temp=False,
    ):
        seen["allow_all_series_temp"] = allow_all_series_temp
        output = Path(course_dir) / "radiomics_ct.xlsx"
        _write_dual_arm_test_publication(
            output,
            [
                {
                    "roi_name": "PTV",
                    "roi_original_name": "PTV",
                    "segmentation_source": "AutoRTS_total",
                    "patient_id": "P1",
                    "course_id": "series",
                    "series_uid": series_uid,
                }
            ],
        )
        return output

    monkeypatch.setattr(radconda, "radiomics_for_course", fake_conda)

    outcome = radpar.parallel_radiomics_for_course(
        cast(Any, _Cfg(tmp_path)), course, allow_all_series_temp=True
    )
    assert outcome.output_path == course / "radiomics_ct.xlsx"
    assert seen["allow_all_series_temp"] is True

    with pytest.raises(rad.RadiomicsCourseExtractionError, match="temporary contract"):
        radpar.parallel_radiomics_for_course(cast(Any, _Cfg(tmp_path)), course)


def test_all_series_parallel_and_conda_emit_one_auto_source(tmp_path, monkeypatch):
    src = tmp_path / "series"
    src.mkdir()
    series_uid = generate_uid()
    _write_valid_ct(
        src / "slice.dcm", series_uid=series_uid, study_uid=generate_uid()
    )
    rs = write_synthetic_rtstruct(
        tmp_path / "rs.dcm",
        referenced_series_uid=series_uid,
        roi_names=("PTV1",),
    )
    course = tmp_path / ".all_series_radiomics" / "P1" / "series"
    assert rad._materialize_temp_course_tree(course, src, rs) is True
    config = cast(Any, _Cfg(tmp_path))

    monkeypatch.setattr(rad, "_load_series_image", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(rad, "_extractor", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        radpar,
        "effective_parameter_hashes_for_arms",
        lambda *_args, **_kwargs: {
            "primary_resegmented": "effective-primary",
            "sensitivity_raw": "effective-sensitivity",
        },
    )
    monkeypatch.setattr(radpar, "validate_custom_model_output_inventory", lambda *_a, **_k: {})
    monkeypatch.setattr(radpar, "list_custom_model_outputs", lambda *_a, **_k: [])
    monkeypatch.setattr(radpar, "_list_roi_names", lambda *_args: ["PTV"])

    class _SyncExecutor:
        def submit(self, _function, task):
            future = Future()
            future.set_result(
                radpar._status_records(task, "success", "test success")
            )
            return future

        def shutdown(self, wait=True, cancel_futures=False):
            return None

    monkeypatch.setattr(radpar, "ProcessPoolExecutor", lambda **_kwargs: _SyncExecutor())
    parallel = radpar.parallel_radiomics_for_course(
        config, course, max_workers=1, allow_all_series_temp=True
    )
    assert parallel.output_path is not None
    parallel_frame = pd.read_parquet(
        parallel.output_path.with_suffix(".parquet"), engine="pyarrow"
    )
    assert list(
        parallel_frame[["segmentation_source", "roi_original_name"]].itertuples(
            index=False, name=None
        )
    ) == [("AutoRTS_total", "PTV1"), ("AutoRTS_total", "PTV1")]

    parallel.output_path.unlink()
    parallel.output_path.with_suffix(".parquet").unlink()
    monkeypatch.setattr(radconda, "validate_custom_model_output_inventory", lambda *_a, **_k: {})
    monkeypatch.setattr(radconda, "list_custom_model_outputs", lambda *_a, **_k: [])

    class _Image:
        def GetSpacing(self):
            return (1.0, 1.0, 1.0)

        def GetOrigin(self):
            return (0.0, 0.0, 0.0)

        def GetDirection(self):
            return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    class _Reader:
        def GetGDCMSeriesFileNames(self, path):
            return [str(next(Path(path).glob("*.dcm")))]

        def SetFileNames(self, _files):
            return None

        def Execute(self):
            return _Image()

    class _RTStruct:
        def get_roi_names(self):
            return ["PTV"]

        def get_roi_mask_by_name(self, _name):
            return np.ones((16, 16, 16), dtype=bool)

    fake_rt_utils = SimpleNamespace(
        RTStructBuilder=SimpleNamespace(
            create_from=lambda **_kwargs: _RTStruct()
        )
    )
    monkeypatch.setitem(sys.modules, "rt_utils", fake_rt_utils)
    monkeypatch.setattr(radconda.sitk, "ImageSeriesReader", _Reader)
    monkeypatch.setattr(radconda.sitk, "WriteImage", lambda *_a, **_k: None)
    monkeypatch.setattr(radconda, "_write_mask_to_file", lambda *_a, **_k: None)
    captured = {}

    def fake_batch(tasks, output_path, **_kwargs):
        captured["tasks"] = tasks
        rows = []
        for task in tasks:
            rows.append(
                {
                    "roi_name": task["roi_name"],
                    **task["metadata"],
                    "feature": 1.0,
                }
            )
        _write_dual_arm_test_publication(Path(output_path), rows)
        return output_path

    monkeypatch.setattr(radconda, "process_radiomics_batch", fake_batch)
    conda = radconda.radiomics_for_course(
        course, config, allow_all_series_temp=True
    )
    assert conda is not None
    conda_frame = pd.read_parquet(conda.with_suffix(".parquet"), engine="pyarrow")
    assert len(captured["tasks"]) == 1
    assert list(
        conda_frame[["segmentation_source", "roi_original_name"]].itertuples(
            index=False, name=None
        )
    ) == [("AutoRTS_total", "PTV"), ("AutoRTS_total", "PTV")]


def test_find_auto_rtstruct_matches_real_layout(tmp_path):
    # mirror segmentation: input_dir.parent/Segmentation_TotalSegmentator/input_dir.name/<base>/<base>--total.dcm
    input_dir = tmp_path / "DICOM" / "CT" / "uid123"
    input_dir.mkdir(parents=True)
    _, seg_root = _series_artifact_dirs(input_dir)
    base_dir = seg_root / "uid123_total"
    base_dir.mkdir(parents=True)
    rt = base_dir / "uid123_total--total.dcm"
    rt.write_bytes(b"rt")
    (base_dir / "total--liver.nii.gz").write_bytes(b"m")  # a mask must NOT be picked as the RTSTRUCT
    assert rad._find_all_series_auto_rtstruct(input_dir, "total") == rt


def test_find_auto_rtstruct_absent(tmp_path):
    input_dir = tmp_path / "DICOM" / "CT" / "uidX"
    input_dir.mkdir(parents=True)
    assert rad._find_all_series_auto_rtstruct(input_dir, "total") is None


# --------------------------------------------------------------------------- #
# end-to-end run_radiomics_all_series (mocked dispatch)
# --------------------------------------------------------------------------- #
def _build_patient_tree(out_root, pid, specs):
    cdirs = build_course_dirs(out_root / pid / "all_series")
    cdirs.ensure_all_series()
    rows = []
    for spec in specs:
        ic = spec["image_class"]
        uid = spec["series_uid"]
        if ic == "exclude":
            rows.append({
                "image_class": ic, "series_uid": uid, "study_uid": spec.get("study_uid", "S"),
                "series_description": spec.get("series_description", ""), "ts_task": "none",
                "output_dir": "", "status": "excluded",
            })
            continue
        outdir = output_dir_for_image_class(cdirs, ic, uid)
        outdir.mkdir(parents=True, exist_ok=True)
        actual_series_uid = generate_uid()
        actual_study_uid = generate_uid()
        for i in range(spec.get("n_slices", 2)):
            _write_valid_ct(
                outdir / f"img{i}.dcm",
                series_uid=actual_series_uid,
                study_uid=actual_study_uid,
            )
        task = TS_TASK_BY_CLASS.get(ic, "none")
        if spec.get("with_rtstruct", True) and task != "none":
            _, seg_root = _series_artifact_dirs(outdir)
            base_dir = seg_root / f"{uid}_base"
            base_dir.mkdir(parents=True, exist_ok=True)
            write_synthetic_rtstruct(base_dir / f"{uid}_base--{task}.dcm")
        rows.append({
            "image_class": ic, "series_uid": uid, "study_uid": spec.get("study_uid", "S"),
            "series_description": spec.get("series_description", ""), "ts_task": task,
            "output_dir": str(outdir), "status": "segmented",
        })
    (cdirs.metadata / "series_manifest.json").write_text(
        json.dumps({"patient_id": pid, "series": rows}), encoding="utf-8"
    )


@pytest.fixture
def fake_dispatch():
    """A fake per-course CT worker: asserts the temp tree shape and writes a 2-row radiomics_ct.xlsx."""
    calls = []

    def fake(config, course_dir, custom=None, *args, **kwargs):
        course_dir = Path(course_dir)
        contract = rad.load_course_contract(course_dir)
        assert contract.data["scope"] == "all_series_radiomics_temp"
        assert contract.data["planning_ct"]["dicom_only"] is True
        assert contract.selected_plans == []
        assert contract.selected_doses == []
        ct = course_dir / "DICOM" / "CT"
        rs = course_dir / "RS_auto.dcm"
        calls.append({
            "course_dir": str(course_dir),
            "n_slices": len(list(ct.glob("*.dcm"))),
            "rs_auto": rs.exists(),
            "rs_target": os.path.realpath(rs) if rs.exists() else None,
            "custom": custom,
            "use_cropped": kwargs.get("use_cropped"),
            "allow_all_series_temp": kwargs.get("allow_all_series_temp"),
            "contract_scope": contract.data["scope"],
        })
        from rtpipeline.radiomics_ct_contract import (
            classify_ct_roi,
            disposition_rows_for_arms,
            write_ct_publication_atomic,
        )

        records = []
        series_uid = str(contract.data["planning_ct"]["series_instance_uid"])
        for index, (roi_name, feature_value) in enumerate(
            (("liver", 1.0), ("spleen", 2.0)), start=1
        ):
            pair = disposition_rows_for_arms(
                {
                    "modality": "CT",
                    "segmentation_source": "AutoRTS_total",
                    "roi_name": roi_name,
                    "roi_original_name": roi_name,
                    "patient_id": course_dir.parent.name,
                    "course_id": course_dir.name,
                    "series_uid": series_uid,
                    "mask_identity": f"test-mask-{index}",
                    "stable_roi_identifier": f"test-roi-{index}",
                    "course_dir": str(course_dir),
                },
                decision=classify_ct_roi("AutoRTS_total", roi_name),
                disposition="success",
                detail="test success",
                failure_kind="none",
                run_identifier="test-run",
                code_revision="test-revision",
                native_voxel_count=120,
                required=False,
                configured_parameter_hashes={
                    "primary_resegmented": "test-primary",
                    "sensitivity_raw": "test-sensitivity",
                },
                effective_hashes={
                    "primary_resegmented": "effective-primary",
                    "sensitivity_raw": "effective-sensitivity",
                },
            )
            for record in pair:
                record["feature"] = feature_value
            records.extend(pair)
        rad.attach_acquisition_descriptor(
            records,
            rad.describe_contract_planning_ct(contract),
        )
        df = pd.DataFrame(records)
        out = course_dir / "radiomics_ct.xlsx"
        write_ct_publication_atomic(df, out)
        return out

    return calls, fake


def _install(monkeypatch, fake, mode):
    # the new early gate: a backend (native or conda) is "available" — the dispatch is mocked anyway
    monkeypatch.setattr(rad, "_have_pyradiomics", lambda: True)
    if mode == "serial":
        monkeypatch.setattr(radpar, "is_parallel_radiomics_enabled", lambda: False)
        monkeypatch.setattr(rad, "radiomics_for_course", fake)
    else:
        monkeypatch.setattr(radpar, "is_parallel_radiomics_enabled", lambda: True)
        monkeypatch.setattr(radpar, "parallel_radiomics_for_course", fake)


@pytest.mark.parametrize("mode", ["serial", "parallel"])
def test_run_radiomics_all_series_e2e(tmp_path, monkeypatch, fake_dispatch, mode):
    calls, fake = fake_dispatch
    out_root = tmp_path / "out"
    pid = "P1"
    _build_patient_tree(out_root, pid, [
        {"image_class": "planning_ct", "series_uid": "u_plan", "study_uid": "S0"},
        {"image_class": "diagnostic_ct", "series_uid": "u_diag", "study_uid": "S0"},
        {"image_class": "petct_ct", "series_uid": "u_pet", "study_uid": "S0"},
        {"image_class": "cbct", "series_uid": "u_cbct", "study_uid": "S0"},   # C3 skip
        {"image_class": "pt", "series_uid": "u_pt", "study_uid": "S0"},        # not a CT class
        {"image_class": "exclude", "series_uid": "u_exc", "study_uid": "S0"},
        # 4DCT study A: ave present -> only the ave is radiomic'd
        {"image_class": "fourdct_ave", "series_uid": "u_aveA", "study_uid": "S_A"},
        {"image_class": "fourdct_phase", "series_uid": "u_phA50", "study_uid": "S_A",
         "series_description": "50%"},
        # 4DCT study B: no ave -> the 50% phase is promoted (is_4d_phase=True)
        {"image_class": "fourdct_phase", "series_uid": "u_phB0", "study_uid": "S_B",
         "series_description": "0%"},
        {"image_class": "fourdct_phase", "series_uid": "u_phB50", "study_uid": "S_B",
         "series_description": "50%"},
    ])

    _install(monkeypatch, fake, mode)
    out_csv = rad.run_radiomics_all_series(_Cfg(out_root), [pid])

    assert out_csv is not None and out_csv.exists()
    df = pd.read_csv(out_csv)

    # exactly the eligible series are radiomic'd; C3/non-CT/exclude/non-representative-4DCT skipped
    assert set(df["series_uid"]) == {"u_plan", "u_diag", "u_pet", "u_aveA", "u_phB50"}

    # provenance columns present
    for col in ("patient_id", "series_uid", "study_uid", "image_class", "is_4d_phase", "series_dir"):
        assert col in df.columns
    assert set(df["patient_id"]) == {"P1"}

    # is_4d_phase True ONLY for the fallback-promoted phase
    is4d = {u: bool(v) for u, v in zip(df["series_uid"], df["is_4d_phase"].astype(bool))}
    assert is4d["u_phB50"] is True
    assert all(v is False for u, v in is4d.items() if u != "u_phB50")

    # 5 series * 2 ROIs * 2 extraction arms
    assert len(df) == 20
    assert set(df["extraction_arm"]) == {"primary_resegmented", "sensitivity_raw"}

    # The governed publication identity retains a stable course surrogate for every arm.
    assert set(df["course_id"]) == {
        "u_plan",
        "u_diag",
        "u_pet",
        "u_aveA",
        "u_phB50",
    }

    # course-path artifacts NOT created; temp tree cleaned up
    assert not (out_root / "Data" / "radiomics_all.xlsx").exists()
    assert not (out_root / ".all_series_radiomics").exists()

    # dispatch saw the correct minimal temp tree for every series
    assert len(calls) == 5
    for c in calls:
        assert c["n_slices"] == 2
        assert c["rs_auto"] is True
        assert c["custom"] is None
        assert c["use_cropped"] in (False, None)  # parallel passes False; serial defaults False
        assert c["allow_all_series_temp"] is True


def test_missing_manifest_returns_none(tmp_path, monkeypatch, fake_dispatch):
    calls, fake = fake_dispatch
    _install(monkeypatch, fake, "serial")
    assert rad.run_radiomics_all_series(_Cfg(tmp_path / "out"), ["NOPE"]) is None
    assert len(calls) == 0


def test_series_without_rtstruct_is_skipped(tmp_path, monkeypatch, fake_dispatch):
    calls, fake = fake_dispatch
    out_root = tmp_path / "out"
    pid = "P2"
    _build_patient_tree(out_root, pid, [
        {"image_class": "planning_ct", "series_uid": "u_ok", "study_uid": "S", "with_rtstruct": True},
        {"image_class": "planning_ct", "series_uid": "u_nort", "study_uid": "S", "with_rtstruct": False},
    ])
    _install(monkeypatch, fake, "serial")
    out_csv = rad.run_radiomics_all_series(_Cfg(out_root), [pid])
    df = pd.read_csv(out_csv)
    assert set(df["series_uid"]) == {"u_ok"}
    assert len(calls) == 1
    assert not (out_root / ".all_series_radiomics").exists()


def test_e2e_4dct_phase_fallback_through_real_rtstruct_probe(tmp_path, monkeypatch, fake_dispatch):
    # End-to-end exercise of the REAL _row_has_rtstruct probe (not an injected lambda): an ave-less
    # 4DCT study whose preferred 50% phase was NOT segmented (no on-disk RTSTRUCT) but a 0% phase was.
    # B4 must radiomic the segmented 0% phase, not silently drop the study by picking the absent 50%.
    calls, fake = fake_dispatch
    out_root = tmp_path / "out"
    pid = "P3"
    _build_patient_tree(out_root, pid, [
        {"image_class": "fourdct_phase", "series_uid": "ph0", "study_uid": "S_C",
         "series_description": "0%", "with_rtstruct": True},
        {"image_class": "fourdct_phase", "series_uid": "ph50", "study_uid": "S_C",
         "series_description": "50%", "with_rtstruct": False},  # preferred but unsegmented
    ])
    _install(monkeypatch, fake, "serial")
    out_csv = rad.run_radiomics_all_series(_Cfg(out_root), [pid])
    df = pd.read_csv(out_csv)
    assert set(df["series_uid"]) == {"ph0"}                       # segmented phase, not the absent 50%
    assert all(bool(v) for v in df["is_4d_phase"].astype(bool))    # tagged, excluded from pooling
    assert len(calls) == 1
    assert not (out_root / ".all_series_radiomics").exists()


def test_empty_patient_list_returns_none(tmp_path, monkeypatch, fake_dispatch):
    calls, fake = fake_dispatch
    _install(monkeypatch, fake, "serial")
    assert rad.run_radiomics_all_series(_Cfg(tmp_path / "out"), []) is None
    assert len(calls) == 0


def test_no_backend_short_circuits(tmp_path, monkeypatch, fake_dispatch):
    # when neither native nor conda PyRadiomics is available, skip cleanly without materializing anything
    calls, fake = fake_dispatch
    monkeypatch.setattr(rad, "_have_pyradiomics", lambda: False)
    monkeypatch.setattr(radpar, "is_parallel_radiomics_enabled", lambda: False)
    monkeypatch.setattr(rad, "radiomics_for_course", fake)
    out_root = tmp_path / "out"
    _build_patient_tree(out_root, "P9", [
        {"image_class": "planning_ct", "series_uid": "u", "study_uid": "S"},
    ])
    assert rad.run_radiomics_all_series(_Cfg(out_root), ["P9"]) is None
    assert len(calls) == 0
    assert not (out_root / ".all_series_radiomics").exists()


def test_subset_rerun_preserves_other_patients(tmp_path, monkeypatch, fake_dispatch):
    calls, fake = fake_dispatch
    out_root = tmp_path / "out"
    for patient_id in ("P1", "P2"):
        _build_patient_tree(
            out_root,
            patient_id,
            [{"image_class": "planning_ct", "series_uid": f"u_{patient_id}", "study_uid": "S"}],
        )
    _install(monkeypatch, fake, "serial")

    output = rad.run_radiomics_all_series(cast(Any, _Cfg(out_root)), ["P1", "P2"])
    assert output is not None
    assert set(pd.read_csv(output)["patient_id"]) == {"P1", "P2"}

    output = rad.run_radiomics_all_series(cast(Any, _Cfg(out_root)), ["P1"])
    assert output is not None
    assert set(pd.read_csv(output)["patient_id"]) == {"P1", "P2"}


def test_subset_rerun_rejects_legacy_rows_without_descriptor(tmp_path, monkeypatch):
    out_root = tmp_path / "out"
    data_dir = out_root / "Data"
    data_dir.mkdir(parents=True)
    output = data_dir / "radiomics_all_series.csv"
    pd.DataFrame(
        [
            {
                "patient_id": "P2",
                "series_uid": "legacy-series",
                "roi_name": "liver",
                "feature": 1.0,
            }
        ]
    ).to_csv(output, index=False)
    before = output.read_bytes()
    monkeypatch.setattr(rad, "_have_pyradiomics", lambda: True)

    with pytest.raises(
        RuntimeError, match="authoritative all-series CT Parquet is missing"
    ):
        rad.run_radiomics_all_series(cast(Any, _Cfg(out_root)), ["P1"])

    assert output.read_bytes() == before


def test_cli_radiomics_stage_runs_all_series_adapter(tmp_path, monkeypatch):
    dicom_root = tmp_path / "dicom"
    dicom_root.mkdir()
    (tmp_path / "config.yaml").write_text(
        "organize:\n  do_segment_all_series: true\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "out"
    course_dirs = build_course_dirs(output_root / "P1" / "course")
    course_dirs.ensure()
    course = SimpleNamespace(patient_id="P1", course_id="course", dirs=course_dirs)
    seen: dict[str, object] = {}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "organize_and_merge", lambda cfg: [course])
    monkeypatch.setattr(cli, "_detect_gpu_count", lambda: 0)
    monkeypatch.setattr(rad, "_have_pyradiomics", lambda: True)
    monkeypatch.setattr(rad, "run_radiomics", lambda cfg, courses, custom: seen.setdefault("course", True))
    monkeypatch.setattr(
        rad,
        "run_radiomics_all_series",
        lambda cfg, patient_ids: seen.setdefault("patient_ids", list(patient_ids)),
    )

    assert cli.main(
        [
            "--dicom-root",
            str(dicom_root),
            "--outdir",
            str(output_root),
            "--logs",
            str(tmp_path / "logs"),
            "--stage",
            "radiomics",
            "--no-metadata",
        ]
    ) == 0
    assert seen == {"course": True, "patient_ids": ["P1"]}
