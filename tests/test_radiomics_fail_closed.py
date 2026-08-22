"""Fail-closed regression tests for course radiomics preparation and publication."""

from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import numpy as np
import pandas as pd
import pytest

import rtpipeline.radiomics as radiomics
import rtpipeline.radiomics_conda as conda
from rtpipeline.config import PipelineConfig
from rtpipeline.layout import build_course_dirs
from rtpipeline.radiomics_outcomes import (
    RadiomicsCourseExtractionError,
    RadiomicsCourseOutcome,
    RadiomicsCourseStatus,
)


def _config(tmp_path: Path, **overrides) -> PipelineConfig:
    config = PipelineConfig(
        dicom_root=tmp_path / "dicom",
        output_root=tmp_path / "out",
        logs_root=tmp_path / "logs",
        max_workers_override=1,
        radiomics_min_voxels=1,
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def _install_rt_utils(monkeypatch, builder) -> None:
    module = types.SimpleNamespace(
        RTStructBuilder=types.SimpleNamespace(create_from=builder)
    )
    monkeypatch.setitem(sys.modules, "rt_utils", module)


@pytest.mark.parametrize(
    ("mask_result", "message"),
    [
        (None, "did not provide a mask"),
        (np.zeros((2, 2, 2), dtype=bool), "empty required mask"),
    ],
)
def test_direct_rtstruct_missing_or_empty_expected_mask_fails(
    tmp_path, monkeypatch, mask_result, message
):
    class RTStruct:
        def get_roi_names(self):
            return ["PTV"]

        def get_roi_mask_by_name(self, _name):
            return mask_result

    _install_rt_utils(monkeypatch, lambda **_kwargs: RTStruct())

    with pytest.raises(RadiomicsCourseExtractionError, match=message):
        radiomics._rtstruct_masks(tmp_path / "CT", tmp_path / "RS.dcm")


def test_direct_rtstruct_roi_read_failure_is_not_omitted(tmp_path, monkeypatch):
    class RTStruct:
        def get_roi_names(self):
            return ["PTV"]

        def get_roi_mask_by_name(self, _name):
            raise ValueError("corrupt contour")

    _install_rt_utils(monkeypatch, lambda **_kwargs: RTStruct())

    with pytest.raises(RadiomicsCourseExtractionError, match="could not be read.*corrupt contour"):
        radiomics._rtstruct_masks(tmp_path / "CT", tmp_path / "RS.dcm")


def test_direct_rtstruct_configured_skip_is_explicit_ineligibility(tmp_path, monkeypatch):
    class RTStruct:
        def get_roi_names(self):
            return ["DO_NOT_EXTRACT"]

        def get_roi_mask_by_name(self, _name):
            raise AssertionError("configured skip must not request a mask")

    _install_rt_utils(monkeypatch, lambda **_kwargs: RTStruct())

    assert radiomics._rtstruct_masks(
        tmp_path / "CT",
        tmp_path / "RS.dcm",
        skip_rois={"do-not extract"},
    ) == {}


def test_direct_rtstruct_construction_failure_invalidates_stale_course_output(
    tmp_path, monkeypatch
):
    course = tmp_path / "P1" / "C1"
    dirs = build_course_dirs(course)
    dirs.dicom_ct.mkdir(parents=True)
    (dirs.dicom_ct / "image.dcm").write_bytes(b"present")
    (course / "RS_auto.dcm").write_bytes(b"present")
    stale = course / "radiomics_ct.xlsx"
    stale.write_bytes(b"stale")
    (course / "radiomics_ct.parquet").write_bytes(b"stale")

    monkeypatch.setattr(radiomics, "_load_series_image", lambda *_a, **_k: object())
    monkeypatch.setattr(radiomics, "_extractor", lambda *_a, **_k: object())
    monkeypatch.setattr(
        radiomics,
        "_rtstruct_masks",
        lambda *_a, **_k: (_ for _ in ()).throw(
            RadiomicsCourseExtractionError("RTSTRUCT construction failed")
        ),
    )

    with pytest.raises(RadiomicsCourseExtractionError, match="construction failed"):
        radiomics.radiomics_for_course(_config(tmp_path), course)

    assert not stale.exists()
    assert not stale.with_suffix(".parquet").exists()


class _Image:
    def GetSpacing(self):
        return (1.0, 1.0, 1.0)

    def GetOrigin(self):
        return (0.0, 0.0, 0.0)

    def GetDirection(self):
        return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


class _Reader:
    def GetGDCMSeriesFileNames(self, _path):
        return ["slice.dcm"]

    def SetFileNames(self, _files):
        pass

    def Execute(self):
        return _Image()


def _prepare_conda_rtstruct_course(tmp_path, monkeypatch):
    course = tmp_path / "P1" / "C1"
    dirs = build_course_dirs(course)
    dirs.dicom_ct.mkdir(parents=True)
    (dirs.dicom_ct / "slice.dcm").write_bytes(b"present")
    rs_path = course / "RS_auto.dcm"
    rs_path.write_bytes(b"present")
    stale = course / "radiomics_ct.xlsx"
    stale.write_bytes(b"stale")
    monkeypatch.setattr(conda, "_select_usable_rtstruct", lambda *_paths: rs_path)
    monkeypatch.setattr(conda.sitk, "ImageSeriesReader", lambda: _Reader())
    monkeypatch.setattr(conda.sitk, "WriteImage", lambda *_a, **_k: None)
    return course, stale


def test_conda_rtstruct_builder_failure_does_not_fall_back_to_nothing_to_do(
    tmp_path, monkeypatch
):
    course, stale = _prepare_conda_rtstruct_course(tmp_path, monkeypatch)
    _install_rt_utils(
        monkeypatch,
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("broken RTSTRUCT")),
    )
    monkeypatch.setattr(
        conda,
        "radiomics_for_course_ct_nifti_fallback",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("preparation failure must not use absence fallback")
        ),
    )

    with pytest.raises(RadiomicsCourseExtractionError, match="broken RTSTRUCT"):
        conda.radiomics_for_course(course, _config(tmp_path))

    assert not stale.exists()


def test_conda_rtstruct_mask_serialization_failure_fails_course(tmp_path, monkeypatch):
    course, stale = _prepare_conda_rtstruct_course(tmp_path, monkeypatch)

    class RTStruct:
        def get_roi_names(self):
            return ["PTV"]

        def get_roi_mask_by_name(self, _name):
            return np.ones((2, 2, 2), dtype=bool)

    _install_rt_utils(monkeypatch, lambda **_kwargs: RTStruct())
    monkeypatch.setattr(
        conda,
        "_write_mask_to_file",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(RadiomicsCourseExtractionError, match="serialize required mask.*disk full"):
        conda.radiomics_for_course(course, _config(tmp_path))

    assert not stale.exists()


def test_conda_workbook_write_failure_raises_and_invalidates_stale_output(
    tmp_path, monkeypatch
):
    output = tmp_path / "radiomics_ct.xlsx"
    output.write_bytes(b"stale workbook")
    monkeypatch.setattr(conda, "check_radiomics_env", lambda *a, **k: True)
    monkeypatch.setattr(
        conda,
        "extract_radiomics_batch_with_conda",
        lambda tasks, params_file=None: [
            {
                "__status__": "success",
                "__task_index__": 0,
                "original_firstorder_Mean": 1.0,
            }
        ],
    )

    def truncated_then_fail(_self, path, index=False, **_kwargs):
        Path(path).write_bytes(b"truncated")
        raise OSError("workbook write failed")

    monkeypatch.setattr(pd.DataFrame, "to_excel", truncated_then_fail)

    with pytest.raises(RadiomicsCourseExtractionError, match="workbook write failed"):
        conda.process_radiomics_batch(
            [
                {
                    "image_path": "image.nrrd",
                    "mask_path": "mask.nrrd",
                    "roi_name": "PTV",
                    "cleanup": False,
                    "metadata": {
                        "segmentation_source": "Manual",
                        "roi_original_name": "PTV",
                    },
                }
            ],
            output,
            sequential=True,
            max_workers=1,
            enable_heartbeat=False,
        )

    assert not output.exists()
    assert list(tmp_path.glob(".radiomics_ct.xlsx.*.tmp.xlsx")) == []


def test_unreadable_expected_course_workbook_blocks_and_invalidates_aggregate(
    tmp_path, monkeypatch
):
    course_root = tmp_path / "P1" / "C1"
    course_root.mkdir(parents=True)
    workbook = course_root / "radiomics_ct.xlsx"
    workbook.write_bytes(b"not an xlsx")
    course = SimpleNamespace(
        patient_id="P1",
        course_key="C1",
        dirs=SimpleNamespace(root=course_root),
    )
    config = _config(tmp_path)
    aggregate = config.output_root / "Data" / "radiomics_all.xlsx"
    aggregate.parent.mkdir(parents=True)
    aggregate.write_bytes(b"stale aggregate")

    monkeypatch.setattr(radiomics, "_have_pyradiomics", lambda: True)
    import rtpipeline.radiomics_parallel as parallel

    monkeypatch.setattr(parallel, "is_parallel_radiomics_enabled", lambda: False)
    monkeypatch.setattr(
        radiomics,
        "run_tasks_with_adaptive_workers",
        lambda *_a, **_k: [RadiomicsCourseOutcome.extracted(workbook)],
    )
    monkeypatch.setattr(radiomics, "radiomics_for_course_mr", lambda *_a, **_k: None)

    with pytest.raises(RadiomicsCourseExtractionError, match="workbook is unreadable"):
        radiomics.run_radiomics(config, [course])

    assert not aggregate.exists()


def test_missing_configured_mr_parameter_path_is_required_failure(tmp_path):
    config = _config(
        tmp_path,
        radiomics_params_file_mr=tmp_path / "missing-required-mr.yaml",
    )
    course = tmp_path / "P1" / "C1"

    with pytest.raises(RadiomicsCourseExtractionError, match="required MR.*path is missing"):
        radiomics.radiomics_for_course_mr(config, course)


def test_optional_mr_failure_remains_nonfatal(tmp_path, monkeypatch):
    course_root = tmp_path / "P1" / "C1"
    course = SimpleNamespace(dirs=SimpleNamespace(root=course_root))
    config = _config(tmp_path)
    aggregate = config.output_root / "Data" / "radiomics_all.xlsx"
    aggregate.parent.mkdir(parents=True)
    aggregate.write_bytes(b"stale")

    monkeypatch.setattr(radiomics, "_have_pyradiomics", lambda: True)
    import rtpipeline.radiomics_parallel as parallel

    monkeypatch.setattr(parallel, "is_parallel_radiomics_enabled", lambda: False)
    monkeypatch.setattr(
        radiomics,
        "run_tasks_with_adaptive_workers",
        lambda *_a, **_k: [RadiomicsCourseOutcome.nothing_to_do("no CT")],
    )
    monkeypatch.setattr(
        radiomics,
        "radiomics_for_course_mr",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("optional MR failed")),
    )

    radiomics.run_radiomics(config, [course])

    assert not aggregate.exists()


def test_conda_required_mr_corrupt_mask_fails_and_invalidates_stale_output(
    tmp_path, monkeypatch
):
    course = tmp_path / "P1" / "C1"
    series = course / "MR" / "SERIES1"
    nifti_dir = series / "NIFTI"
    seg_dir = series / "Segmentation_TotalSegmentator"
    nifti_dir.mkdir(parents=True)
    seg_dir.mkdir(parents=True)
    image_path = nifti_dir / "image.nii.gz"
    mask_path = seg_dir / "total_mr--PTV.nii.gz"
    image_path.write_bytes(b"image")
    mask_path.write_bytes(b"corrupt mask")
    params = tmp_path / "mr.yaml"
    params.write_text("setting: {}\n", encoding="utf-8")
    stale = course / "MR" / "radiomics_mr.xlsx"
    stale.write_bytes(b"stale")
    config = _config(tmp_path, radiomics_params_file_mr=params)

    def fake_read(path):
        if Path(path) == mask_path:
            raise RuntimeError("cannot decode mask")
        return _Image()

    monkeypatch.setattr(conda.sitk, "ReadImage", fake_read)
    monkeypatch.setattr(conda.sitk, "WriteImage", lambda *_a, **_k: None)

    with pytest.raises(RadiomicsCourseExtractionError, match="mask is unreadable.*cannot decode"):
        conda.radiomics_for_course_mr(course, config)

    assert not stale.exists()


def test_direct_no_ct_invalidates_stale_course_outputs(tmp_path, monkeypatch):
    course = tmp_path / "P1" / "C1"
    stale = course / "radiomics_ct.xlsx"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")
    stale.with_suffix(".parquet").write_bytes(b"stale")
    monkeypatch.setattr(radiomics, "_load_series_image", lambda *_a, **_k: None)

    outcome = radiomics.radiomics_for_course(_config(tmp_path), course)

    assert outcome.status is RadiomicsCourseStatus.NOTHING_TO_DO
    assert not stale.exists()
    assert not stale.with_suffix(".parquet").exists()


def test_direct_unreadable_ct_invalidates_stale_course_outputs(tmp_path, monkeypatch):
    course = tmp_path / "P1" / "C1"
    dirs = build_course_dirs(course)
    dirs.dicom_ct.mkdir(parents=True)
    (dirs.dicom_ct / "slice.dcm").write_bytes(b"present")
    stale = course / "radiomics_ct.xlsx"
    stale.write_bytes(b"stale")
    stale.with_suffix(".parquet").write_bytes(b"stale")
    monkeypatch.setattr(radiomics, "_load_series_image", lambda *_a, **_k: None)

    with pytest.raises(RadiomicsCourseExtractionError, match="present but unreadable"):
        radiomics.radiomics_for_course(_config(tmp_path), course)

    assert not stale.exists()
    assert not stale.with_suffix(".parquet").exists()


def test_conda_nifti_fallback_no_masks_invalidates_stale_outputs(tmp_path):
    course = tmp_path / "P1" / "C1"
    stale = course / "radiomics_ct.xlsx"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")
    stale.with_suffix(".parquet").write_bytes(b"stale")

    assert conda.radiomics_for_course_ct_nifti_fallback(course, _config(tmp_path)) is None
    assert not stale.exists()
    assert not stale.with_suffix(".parquet").exists()


def _prepare_conda_nifti_fallback(tmp_path, monkeypatch):
    course = tmp_path / "P1" / "C1"
    seg_dir = build_course_dirs(course).segmentation_totalseg / "CT"
    seg_dir.mkdir(parents=True)
    mask_path = seg_dir / "lung.nii.gz"
    mask_path.write_bytes(b"mask")
    image_path = course / "ct.nii.gz"
    image_path.write_bytes(b"image")
    stale = course / "radiomics_ct.xlsx"
    stale.write_bytes(b"stale")
    stale.with_suffix(".parquet").write_bytes(b"stale")
    monkeypatch.setattr(conda, "_ct_nifti_candidates", lambda _course: {"CT": image_path})
    monkeypatch.setattr(conda.sitk, "ReadImage", lambda _path: _Image())
    return course, mask_path, stale


def test_conda_nifti_fallback_empty_mask_fails_and_invalidates_stale_output(
    tmp_path, monkeypatch
):
    course, _mask_path, stale = _prepare_conda_nifti_fallback(tmp_path, monkeypatch)
    monkeypatch.setattr(
        conda.sitk,
        "GetArrayFromImage",
        lambda _image: np.zeros((2, 2, 2), dtype=np.uint8),
    )
    monkeypatch.setattr(conda.sitk, "WriteImage", lambda *_a, **_k: None)

    with pytest.raises(RadiomicsCourseExtractionError, match="empty mask"):
        conda.radiomics_for_course_ct_nifti_fallback(course, _config(tmp_path))

    assert not stale.exists()
    assert not stale.with_suffix(".parquet").exists()


def test_conda_nifti_fallback_serialization_failure_invalidates_stale_output(
    tmp_path, monkeypatch
):
    course, _mask_path, stale = _prepare_conda_nifti_fallback(tmp_path, monkeypatch)
    monkeypatch.setattr(
        conda.sitk,
        "GetArrayFromImage",
        lambda _image: np.ones((2, 2, 2), dtype=np.uint8),
    )

    def fail_mask_write(_image, path, **_kwargs):
        if "ct_ts_mask_" in str(path):
            raise OSError("mask serialization failed")

    monkeypatch.setattr(conda.sitk, "WriteImage", fail_mask_write)

    with pytest.raises(RadiomicsCourseExtractionError, match="mask serialization failed"):
        conda.radiomics_for_course_ct_nifti_fallback(course, _config(tmp_path))

    assert not stale.exists()
    assert not stale.with_suffix(".parquet").exists()


def test_invalidation_failure_is_not_silently_ignored(tmp_path, monkeypatch):
    output = tmp_path / "radiomics_ct.xlsx"
    output.write_bytes(b"stale")
    original_unlink = Path.unlink

    def deny_target(self, *args, **kwargs):
        if self == output:
            raise PermissionError("read-only stale output")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", deny_target)

    with pytest.raises(RadiomicsCourseExtractionError, match="read-only stale output"):
        radiomics._invalidate_radiomics_outputs(output)


def test_incomplete_worker_result_vector_blocks_aggregation(tmp_path, monkeypatch):
    course_root = tmp_path / "P1" / "C1"
    course_root.mkdir(parents=True)
    course = SimpleNamespace(dirs=SimpleNamespace(root=course_root))
    config = _config(tmp_path)
    aggregate = config.output_root / "Data" / "radiomics_all.xlsx"
    aggregate.parent.mkdir(parents=True)
    aggregate.write_bytes(b"stale")

    monkeypatch.setattr(radiomics, "_have_pyradiomics", lambda: True)
    import rtpipeline.radiomics_parallel as parallel

    monkeypatch.setattr(parallel, "is_parallel_radiomics_enabled", lambda: False)
    monkeypatch.setattr(
        radiomics, "run_tasks_with_adaptive_workers", lambda *_a, **_k: []
    )

    with pytest.raises(RuntimeError, match="incomplete result vector"):
        radiomics.run_radiomics(config, [course])

    assert not aggregate.exists()


def test_parallel_backend_no_ct_invalidates_stale_course_outputs(tmp_path):
    import rtpipeline.radiomics_parallel as parallel

    course = tmp_path / "P1" / "C1"
    stale = course / "radiomics_ct.xlsx"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")
    stale.with_suffix(".parquet").write_bytes(b"stale")

    outcome = parallel.parallel_radiomics_for_course(_config(tmp_path), course)

    assert outcome.status is RadiomicsCourseStatus.NOTHING_TO_DO
    assert not stale.exists()
    assert not stale.with_suffix(".parquet").exists()


def test_parallel_backend_unreadable_ct_invalidates_stale_course_outputs(
    tmp_path, monkeypatch
):
    import rtpipeline.radiomics_parallel as parallel

    course = tmp_path / "P1" / "C1"
    dirs = build_course_dirs(course)
    dirs.dicom_ct.mkdir(parents=True)
    (dirs.dicom_ct / "slice.dcm").write_bytes(b"present")
    stale = course / "radiomics_ct.xlsx"
    stale.write_bytes(b"stale")
    stale.with_suffix(".parquet").write_bytes(b"stale")
    monkeypatch.setattr(radiomics, "_load_series_image", lambda *_a, **_k: None)

    with pytest.raises(RadiomicsCourseExtractionError, match="present but unreadable"):
        parallel.parallel_radiomics_for_course(_config(tmp_path), course)

    assert not stale.exists()
    assert not stale.with_suffix(".parquet").exists()


def test_parallel_backend_publication_failure_invalidates_stale_outputs(
    tmp_path, monkeypatch
):
    import rtpipeline.radiomics_parallel as parallel

    course = tmp_path / "P1" / "C1"
    dirs = build_course_dirs(course)
    dirs.dicom_ct.mkdir(parents=True)
    dirs.dicom_rtstruct.mkdir(parents=True)
    (dirs.dicom_ct / "slice.dcm").write_bytes(b"present")
    (dirs.dicom_rtstruct / "RS.dcm").write_bytes(b"present")
    stale = course / "radiomics_ct.xlsx"
    stale.write_bytes(b"stale")
    stale.with_suffix(".parquet").write_bytes(b"stale")

    class FakeExecutor:
        def __init__(self, *_args, **_kwargs):
            pass

        def submit(self, _function, _task):
            future = Future()
            future.set_result(
                {
                    "segmentation_source": "Manual",
                    "roi_original_name": "PTV",
                    "patient_id": "P1",
                    "course_id": "C1",
                    "original_firstorder_Mean": 1.0,
                }
            )
            return future

        def shutdown(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(radiomics, "_load_series_image", lambda *_a, **_k: object())
    monkeypatch.setattr(radiomics, "_extractor", lambda *_a, **_k: object())
    monkeypatch.setattr(parallel, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(parallel, "_list_roi_names", lambda _path: ["PTV"])
    monkeypatch.setattr(parallel, "list_custom_model_outputs", lambda _course: [])
    monkeypatch.setattr(parallel, "_calculate_optimal_workers", lambda: 1)
    monkeypatch.setattr(
        parallel,
        "_write_excel_atomic",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("atomic publication failed")),
    )

    with pytest.raises(RadiomicsCourseExtractionError, match="atomic publication failed"):
        parallel.parallel_radiomics_for_course(_config(tmp_path), course, max_workers=1)

    assert not stale.exists()
    assert not stale.with_suffix(".parquet").exists()


def test_no_backend_fails_requested_stage_and_invalidates_stale_outputs(
    tmp_path, monkeypatch
):
    course_root = tmp_path / "P1" / "C1"
    course_root.mkdir(parents=True)
    course_output = course_root / "radiomics_ct.xlsx"
    course_output.write_bytes(b"stale")
    course = SimpleNamespace(dirs=SimpleNamespace(root=course_root))
    config = _config(tmp_path)
    aggregate = config.output_root / "Data" / "radiomics_all.xlsx"
    aggregate.parent.mkdir(parents=True)
    aggregate.write_bytes(b"stale")
    monkeypatch.setattr(radiomics, "_have_pyradiomics", lambda: False)

    with pytest.raises(RuntimeError, match="requested radiomics extraction failed"):
        radiomics.run_radiomics(config, [course])

    assert not course_output.exists()
    assert not aggregate.exists()


def test_direct_resume_readable_empty_workbook_is_not_accepted(tmp_path, monkeypatch):
    course = tmp_path / "P1" / "C1"
    dirs = build_course_dirs(course)
    dirs.dicom_ct.mkdir(parents=True)
    (dirs.dicom_ct / "slice.dcm").write_bytes(b"present")
    output = course / "radiomics_ct.xlsx"
    pd.DataFrame(
        columns=[
            "segmentation_source",
            "roi_original_name",
            "original_firstorder_Mean",
        ]
    ).to_excel(output, index=False)
    output.with_suffix(".parquet").write_bytes(b"stale")
    monkeypatch.setattr(radiomics, "_load_series_image", lambda *_a, **_k: None)

    with pytest.raises(RadiomicsCourseExtractionError, match="present but unreadable"):
        radiomics.radiomics_for_course(_config(tmp_path, resume=True), course)

    assert not output.exists()
    assert not output.with_suffix(".parquet").exists()


def test_parallel_resume_structurally_invalid_workbook_is_not_accepted(
    tmp_path, monkeypatch
):
    import rtpipeline.radiomics_parallel as parallel

    course = tmp_path / "P1" / "C1"
    dirs = build_course_dirs(course)
    dirs.dicom_ct.mkdir(parents=True)
    (dirs.dicom_ct / "slice.dcm").write_bytes(b"present")
    output = course / "radiomics_ct.xlsx"
    pd.DataFrame(
        [{"segmentation_source": "Manual", "roi_original_name": "BODY"}]
    ).to_excel(output, index=False)
    output.with_suffix(".parquet").write_bytes(b"stale")
    monkeypatch.setattr(radiomics, "_load_series_image", lambda *_a, **_k: None)

    with pytest.raises(RadiomicsCourseExtractionError, match="present but unreadable"):
        parallel.parallel_radiomics_for_course(_config(tmp_path, resume=True), course)

    assert not output.exists()
    assert not output.with_suffix(".parquet").exists()


class _FakeExtractor:
    settings = {"resampledPixelSpacing": [1.0, 1.0, 1.0]}

    def execute(self, _image, _mask):
        return {"original_firstorder_Mean": 1.0}


def test_direct_resume_missing_non_body_manual_roi_forces_full_rerun(
    tmp_path, monkeypatch
):
    course = tmp_path / "P1" / "C1"
    dirs = build_course_dirs(course)
    dirs.dicom_ct.mkdir(parents=True)
    dirs.dicom_rtstruct.mkdir(parents=True)
    (dirs.dicom_ct / "slice.dcm").write_bytes(b"present")
    manual_rs = dirs.dicom_rtstruct / "RS.dcm"
    manual_rs.write_bytes(b"present")
    auto_rs = course / "RS_auto.dcm"
    auto_rs.write_bytes(b"present")
    output = course / "radiomics_ct.xlsx"
    pd.DataFrame(
        [
            {
                "segmentation_source": "Manual",
                "roi_original_name": "BODY",
                "original_firstorder_Mean": 9.0,
            }
        ]
    ).to_excel(output, index=False)
    output.with_suffix(".parquet").write_bytes(b"stale")

    masks = {
        manual_rs: {
            "BODY": np.ones((2, 2, 2), dtype=bool),
            "GTV": np.ones((2, 2, 2), dtype=bool),
        },
        auto_rs: {"LUNG": np.ones((2, 2, 2), dtype=bool)},
    }
    extractor = _FakeExtractor()
    monkeypatch.setattr(radiomics, "_load_series_image", lambda *_a, **_k: _Image())
    monkeypatch.setattr(radiomics, "_extractor", lambda *_a, **_k: extractor)
    monkeypatch.setattr(radiomics, "_extractor_large_roi", lambda *_a, **_k: extractor)
    monkeypatch.setattr(radiomics, "_mask_from_array_like", lambda *_a, **_k: object())
    monkeypatch.setattr(
        radiomics, "_rtstruct_masks", lambda _ct, path, **_kwargs: masks[Path(path)]
    )
    monkeypatch.setattr(radiomics, "list_custom_model_outputs", lambda _course: [])
    monkeypatch.setattr(
        radiomics,
        "run_tasks_with_adaptive_workers",
        lambda _label, tasks, function, **_kwargs: [function(task) for task in tasks],
    )

    outcome = radiomics.radiomics_for_course(_config(tmp_path, resume=True), course)

    assert outcome.status is RadiomicsCourseStatus.EXTRACTED
    refreshed = pd.read_excel(output, engine="openpyxl")
    assert set(
        zip(refreshed["segmentation_source"], refreshed["roi_original_name"])
    ) == {("Manual", "BODY"), ("Manual", "GTV"), ("AutoRTS_total", "LUNG")}


def test_parallel_resume_missing_autorts_roi_forces_full_rerun(tmp_path, monkeypatch):
    import rtpipeline.radiomics_parallel as parallel

    course = tmp_path / "P1" / "C1"
    dirs = build_course_dirs(course)
    dirs.dicom_ct.mkdir(parents=True)
    dirs.dicom_rtstruct.mkdir(parents=True)
    (dirs.dicom_ct / "slice.dcm").write_bytes(b"present")
    manual_rs = dirs.dicom_rtstruct / "RS.dcm"
    manual_rs.write_bytes(b"present")
    auto_rs = course / "RS_auto.dcm"
    auto_rs.write_bytes(b"present")
    output = course / "radiomics_ct.xlsx"
    pd.DataFrame(
        [
            {
                "segmentation_source": "Manual",
                "roi_original_name": "BODY",
                "original_firstorder_Mean": 9.0,
            }
        ]
    ).to_excel(output, index=False)
    output.with_suffix(".parquet").write_bytes(b"stale")

    class IdentityExecutor:
        def __init__(self, *_args, **_kwargs):
            pass

        def submit(self, _function, task):
            future = Future()
            future.set_result(
                {
                    "segmentation_source": task.source,
                    "roi_name": task.roi_name,
                    "roi_original_name": task.roi_name,
                    "patient_id": "P1",
                    "course_id": "C1",
                    "original_firstorder_Mean": 1.0,
                }
            )
            return future

        def shutdown(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(radiomics, "_load_series_image", lambda *_a, **_k: _Image())
    monkeypatch.setattr(radiomics, "_extractor", lambda *_a, **_k: _FakeExtractor())
    monkeypatch.setattr(parallel, "ProcessPoolExecutor", IdentityExecutor)
    monkeypatch.setattr(
        parallel,
        "_list_roi_names",
        lambda path: ["BODY"] if Path(path) == manual_rs else ["LUNG"],
    )
    monkeypatch.setattr(parallel, "list_custom_model_outputs", lambda _course: [])
    monkeypatch.setattr(parallel, "_calculate_optimal_workers", lambda: 1)

    outcome = parallel.parallel_radiomics_for_course(
        _config(tmp_path, resume=True), course, max_workers=1
    )

    assert outcome.status is RadiomicsCourseStatus.EXTRACTED
    refreshed = pd.read_excel(output, engine="openpyxl")
    assert set(
        zip(refreshed["segmentation_source"], refreshed["roi_original_name"])
    ) == {("Manual", "BODY"), ("AutoRTS_total", "LUNG")}


def test_radiomics_status_contract_still_exposes_explicit_nothing_to_do():
    outcome = RadiomicsCourseOutcome.nothing_to_do("configured skip")
    assert outcome.status is RadiomicsCourseStatus.NOTHING_TO_DO
