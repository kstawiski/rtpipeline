"""Fail-closed regression tests for course radiomics preparation and publication."""

from concurrent.futures import Future
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import numpy as np
import pandas as pd
import pytest

import rtpipeline.custom_models as custom_models
import rtpipeline.radiomics as radiomics
import rtpipeline.radiomics_conda as conda
from rtpipeline.config import PipelineConfig
from rtpipeline.layout import build_course_dirs
from rtpipeline.radiomics_outcomes import (
    RadiomicsCourseExtractionError,
    RadiomicsCourseOutcome,
    RadiomicsCourseStatus,
    outcome_from_output,
)
from course_contract_test_utils import (
    write_minimal_course_contract,
    write_synthetic_planning_ct,
    write_synthetic_rtstruct,
)


def _write_contract(
    course: Path,
    *,
    rtstruct: Path | None = None,
    no_ct: bool = False,
    planning_nifti: Path | None = None,
) -> None:
    if rtstruct is not None:
        write_synthetic_rtstruct(rtstruct)
    ct_dir = None if no_ct else write_synthetic_planning_ct(course)
    write_minimal_course_contract(
        course,
        authoritative_rtstruct=rtstruct,
        planning_ct_dir=ct_dir,
        planning_ct_nifti=planning_nifti,
    )


def _write_current_auto_rtstruct(course: Path) -> Path:
    contract = radiomics.load_course_contract(course)
    planning_series_uid = str(contract.planning_ct.get("series_instance_uid") or "")
    assert planning_series_uid
    return write_synthetic_rtstruct(
        course / "RS_auto.dcm",
        referenced_series_uid=planning_series_uid,
        roi_names=("PTV",),
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
    ("content", "message"),
    [
        ("custom_structures: [\n", "could not be parsed"),
        ("- name: SharedROI\n", "top level must be a mapping"),
        ("other: []\n", "missing required 'custom_structures' section"),
        ("custom_structures: {}\n", "must be a non-empty list"),
        ("custom_structures: []\n", "must be a non-empty list"),
        ("custom_structures:\n  - SharedROI\n", "entry 1 must be a mapping"),
        ("custom_structures:\n  - operation: union\n", "entry 1 has an invalid 'name'"),
        ("custom_structures:\n  - name: 17\n", "entry 1 has an invalid 'name'"),
        (
            "custom_structures:\n"
            "  - name: SharedROI\n"
            "    source_structures: [PTV]\n"
            "  - name: SharedROI\n"
            "    source_structures: [GTV]\n",
            "duplicate ROI name",
        ),
        (
            "custom_structures:\n  - name: SharedROI\n",
            "invalid 'source_structures'",
        ),
        (
            "custom_structures:\n"
            "  - name: SharedROI\n"
            "    operation: invalid\n"
            "    source_structures: [PTV]\n",
            "invalid 'operation'",
        ),
        (
            "custom_structures:\n"
            "  - name: SharedROI\n"
            "    source_structures: [PTV]\n"
            "    margin: [5]\n",
            "invalid 'margin'",
        ),
    ],
)
def test_custom_roi_config_parser_rejects_malformed_inventory(
    tmp_path, content, message
):
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(content, encoding="utf-8")

    with pytest.raises(RadiomicsCourseExtractionError, match=message):
        radiomics._custom_roi_names_from_config(config_path)


def test_custom_roi_config_read_error_is_fatal(tmp_path):
    missing = tmp_path / "missing-custom.yaml"

    with pytest.raises(RadiomicsCourseExtractionError, match="could not be read"):
        radiomics._custom_roi_names_from_config(missing)


@pytest.mark.parametrize("backend", ["native", "parallel", "conda"])
def test_malformed_custom_roi_config_fails_closed_in_every_backend(
    tmp_path, monkeypatch, backend
):
    course = tmp_path / backend / "P1" / "C1"
    dirs = build_course_dirs(course)
    dirs.dicom_ct.mkdir(parents=True)
    (dirs.dicom_ct / "slice.dcm").write_bytes(b"present")
    _write_contract(course)
    stale = course / "radiomics_ct.xlsx"
    stale.write_bytes(b"stale")
    stale.with_suffix(".parquet").write_bytes(b"stale")
    custom_config = tmp_path / backend / "malformed-custom.yaml"
    custom_config.write_text("custom_structures: [\n", encoding="utf-8")
    config = _config(tmp_path, custom_structures_config=custom_config)

    monkeypatch.setattr(radiomics, "_load_series_image", lambda *_a, **_k: _Image())
    monkeypatch.setattr(radiomics, "_extractor", lambda *_a, **_k: _FakeExtractor())

    if backend == "native":
        call = lambda: radiomics.radiomics_for_course(config, course)
    elif backend == "parallel":
        import rtpipeline.radiomics_parallel as parallel

        call = lambda: parallel.parallel_radiomics_for_course(config, course)
    else:
        call = lambda: conda.radiomics_for_course(course, config)

    with pytest.raises(RadiomicsCourseExtractionError, match="could not be parsed"):
        call()

    assert not stale.exists()
    assert not stale.with_suffix(".parquet").exists()


@pytest.mark.parametrize("backend", ["native", "parallel", "conda"])
def test_missing_expected_custom_model_structure_fails_course_in_every_backend(
    tmp_path, monkeypatch, backend
):
    course = tmp_path / backend / "P1" / "C1"
    dirs = build_course_dirs(course)
    dirs.dicom_ct.mkdir(parents=True)
    (dirs.dicom_ct / "slice.dcm").write_bytes(b"present")
    _write_contract(course)
    model_dir = course / "Segmentation_CustomModels" / "TumorModel"
    model_dir.mkdir(parents=True)
    (model_dir / "rtstruct.dcm").write_bytes(b"partial")
    (model_dir / "manifest.json").write_text(
        "{\n"
        '  "model": "TumorModel",\n'
        '  "expected_structures": ["SharedROI", "MissingROI"],\n'
        '  "produced_structures": ["SharedROI", "MissingROI"],\n'
        '  "missing_structures": []\n'
        "}\n",
        encoding="utf-8",
    )
    stale = course / "radiomics_ct.xlsx"
    stale.write_bytes(b"stale")
    stale.with_suffix(".parquet").write_bytes(b"stale")
    config = _config(tmp_path, custom_model_names=["TumorModel"])

    monkeypatch.setattr(radiomics, "_load_series_image", lambda *_a, **_k: _Image())
    monkeypatch.setattr(radiomics, "_extractor", lambda *_a, **_k: _FakeExtractor())
    monkeypatch.setattr(
        custom_models,
        "_rtstruct_structure_inventory",
        lambda _path: ["SharedROI"],
    )

    if backend == "native":
        call = lambda: radiomics.radiomics_for_course(config, course)
    elif backend == "parallel":
        import rtpipeline.radiomics_parallel as parallel

        call = lambda: parallel.parallel_radiomics_for_course(config, course)
    else:
        call = lambda: conda.radiomics_for_course(course, config)

    with pytest.raises(RadiomicsCourseExtractionError, match="MissingROI"):
        call()

    assert not stale.exists()
    assert not stale.with_suffix(".parquet").exists()


def test_custom_model_definition_supplies_expected_inventory_without_manifest(
    tmp_path, monkeypatch
):
    course = tmp_path / "P1" / "C1"
    model_output = course / "Segmentation_CustomModels" / "TumorModel"
    model_output.mkdir(parents=True)
    (model_output / "rtstruct.dcm").write_bytes(b"present")

    models_root = tmp_path / "models"
    definition_dir = models_root / "TumorModel"
    definition_dir.mkdir(parents=True)
    (definition_dir / "custom_model.yaml").write_text(
        "name: TumorModel\n"
        "nnunet:\n"
        "  networks:\n"
        "    - id: tumor\n"
        "      structures: [SharedROI, SecondROI]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        custom_models.pydicom,
        "dcmread",
        lambda *_a, **_k: SimpleNamespace(
            StructureSetROISequence=[
                SimpleNamespace(ROIName="SharedROI"),
                SimpleNamespace(ROIName="SecondROI"),
            ]
        ),
    )

    assert custom_models.validate_custom_model_output_inventory(
        course,
        ["TumorModel"],
        models_root,
    ) == {"TumorModel": ["SharedROI", "SecondROI"]}


def test_same_named_roi_remains_distinct_across_all_ct_sources(
    tmp_path, monkeypatch
):
    course = tmp_path / "P1" / "C1"
    dirs = build_course_dirs(course)
    dirs.dicom_ct.mkdir(parents=True)
    dirs.dicom_rtstruct.mkdir(parents=True)
    (dirs.dicom_ct / "slice.dcm").write_bytes(b"present")
    manual_rs = dirs.dicom_rtstruct / "RS.dcm"
    auto_rs = course / "RS_auto.dcm"
    custom_rs = course / "RS_custom.dcm"
    for path in (manual_rs, custom_rs):
        path.write_bytes(b"present")
    _write_contract(course, rtstruct=manual_rs)
    _write_current_auto_rtstruct(course)

    custom_config = tmp_path / "custom.yaml"
    custom_config.write_text(
        "custom_structures:\n"
        "  - name: SharedROI\n"
        "    operation: union\n"
        "    source_structures: [PTV]\n",
        encoding="utf-8",
    )
    model_dir = course / "Segmentation_CustomModels" / "TumorModel"
    model_dir.mkdir(parents=True)
    model_rs = model_dir / "rtstruct.dcm"
    model_rs.write_bytes(b"present")
    (model_dir / "manifest.json").write_text(
        "{\n"
        '  "model": "TumorModel",\n'
        '  "expected_structures": ["SharedROI"],\n'
        '  "produced_structures": ["SharedROI"],\n'
        '  "missing_structures": []\n'
        "}\n",
        encoding="utf-8",
    )

    mask = np.ones((2, 2, 2), dtype=bool)

    class CustomRTStruct:
        def get_roi_mask_by_name(self, roi_name):
            assert roi_name == "SharedROI"
            return mask

    _install_rt_utils(monkeypatch, lambda **_kwargs: CustomRTStruct())
    monkeypatch.setattr(
        custom_models,
        "_rtstruct_structure_inventory",
        lambda _path: ["SharedROI"],
    )
    monkeypatch.setattr(radiomics, "_load_series_image", lambda *_a, **_k: _Image())
    monkeypatch.setattr(radiomics, "_extractor", lambda *_a, **_k: _FakeExtractor())
    monkeypatch.setattr(radiomics, "_mask_from_array_like", lambda *_a, **_k: object())
    monkeypatch.setattr(radiomics, "_is_rs_custom_stale", lambda *_a, **_k: False)
    monkeypatch.setattr(radiomics, "_list_roi_names_dicom", lambda _path: ["SharedROI"])
    monkeypatch.setattr(
        radiomics,
        "_rtstruct_masks",
        lambda _ct, _path, **_kwargs: {"SharedROI": mask},
    )
    monkeypatch.setattr(
        radiomics,
        "run_tasks_with_adaptive_workers",
        lambda _label, tasks, function, **_kwargs: [function(task) for task in tasks],
    )

    outcome = radiomics.radiomics_for_course(
        _config(
            tmp_path,
            custom_structures_config=custom_config,
            custom_model_names=["TumorModel"],
        ),
        course,
    )

    assert outcome.status is RadiomicsCourseStatus.EXTRACTED
    result = pd.read_excel(course / "radiomics_ct.xlsx", engine="openpyxl")
    assert set(zip(result["segmentation_source"], result["roi_original_name"])) == {
        ("Manual", "SharedROI"),
        ("AutoRTS_total", "SharedROI"),
        ("Custom", "SharedROI"),
        ("CustomModel:TumorModel", "SharedROI"),
    }


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


def test_direct_auto_rtstruct_empty_mask_is_recorded_as_degenerate(tmp_path, monkeypatch):
    class RTStruct:
        def get_roi_names(self):
            return ["vertebrae_T8", "lung_left"]

        def get_roi_mask_by_name(self, name):
            if name == "vertebrae_T8":
                return np.zeros((2, 2, 2), dtype=bool)
            return np.ones((2, 2, 2), dtype=bool)

    failures = []
    _install_rt_utils(monkeypatch, lambda **_kwargs: RTStruct())

    masks = radiomics._rtstruct_masks(
        tmp_path / "CT",
        tmp_path / "RS_auto.dcm",
        best_effort=True,
        failure_outcomes=failures,
    )

    assert set(masks) == {"lung_left"}
    assert failures == [
        {
            "roi_name": "vertebrae_T8",
            "status": "failed",
            "failure_kind": "degenerate_mask",
            "reason": "Expected ROI 'vertebrae_T8' in "
            f"{tmp_path / 'RS_auto.dcm'} produced an empty mask",
        }
    ]


def test_direct_rtstruct_construction_failure_invalidates_stale_course_output(
    tmp_path, monkeypatch
):
    course = tmp_path / "P1" / "C1"
    dirs = build_course_dirs(course)
    dirs.dicom_ct.mkdir(parents=True)
    (dirs.dicom_ct / "image.dcm").write_bytes(b"present")
    _write_contract(course)
    _write_current_auto_rtstruct(course)
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
    _write_contract(course)
    rs_path = _write_current_auto_rtstruct(course)
    stale = course / "radiomics_ct.xlsx"
    stale.write_bytes(b"stale")
    monkeypatch.setattr(conda, "_select_usable_rtstruct", lambda *_paths: rs_path)
    monkeypatch.setattr(conda.sitk, "ImageSeriesReader", lambda: _Reader())
    monkeypatch.setattr(conda.sitk, "WriteImage", lambda *_a, **_k: None)
    return course, stale


def test_conda_rtstruct_builder_failure_is_recorded_without_absence_fallback(
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

    output = conda.radiomics_for_course(course, _config(tmp_path))

    assert output == stale
    result = pd.read_excel(output, engine="openpyxl")
    assert result.loc[0, "extraction_status"] == "failed"
    assert result.loc[0, "extraction_failure_kind"] == "extraction_error"
    assert "broken RTSTRUCT" in result.loc[0, "extraction_status_detail"]


def test_conda_best_effort_mask_serialization_failure_is_recorded(tmp_path, monkeypatch):
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

    output = conda.radiomics_for_course(course, _config(tmp_path))

    assert output == course / "radiomics_ct.xlsx"
    result = pd.read_excel(output, engine="openpyxl")
    assert result.loc[0, "extraction_status"] == "failed"
    assert result.loc[0, "extraction_failure_kind"] == "extraction_error"
    assert "disk full" in result.loc[0, "extraction_status_detail"]


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


def test_cohort_aggregate_carries_course_source_counts(tmp_path, monkeypatch):
    course_root = tmp_path / "P1" / "C1"
    course_root.mkdir(parents=True)
    workbook = course_root / "radiomics_ct.xlsx"
    pd.DataFrame(
        [
            {
                "segmentation_source": "Manual",
                "roi_original_name": "PTV",
                "original_firstorder_Mean": 1.0,
            },
            {
                "segmentation_source": "AutoRTS_total",
                "roi_original_name": "vertebrae_T8",
                "extraction_status": "failed",
                "extraction_status_detail": "empty mask",
                "extraction_failure_kind": "degenerate_mask",
            },
        ]
    ).to_excel(workbook, index=False)
    course = SimpleNamespace(
        patient_id="P1",
        course_key="C1",
        dirs=SimpleNamespace(root=course_root),
    )
    config = _config(tmp_path)
    outcome = RadiomicsCourseOutcome.extracted(
        workbook,
        roi_counts={
            "Manual": {"attempted": 1, "extracted": 1, "failed": 0},
            "AutoRTS_total": {"attempted": 1, "extracted": 0, "failed": 1},
        },
        roi_failures=[
            {
                "source": "AutoRTS_total",
                "roi_name": "vertebrae_T8",
                "status": "failed",
                "failure_kind": "degenerate_mask",
                "reason": "empty mask",
            }
        ],
    )

    monkeypatch.setattr(radiomics, "_have_pyradiomics", lambda: True)
    import rtpipeline.radiomics_parallel as parallel

    monkeypatch.setattr(parallel, "is_parallel_radiomics_enabled", lambda: False)
    monkeypatch.setattr(
        radiomics,
        "run_tasks_with_adaptive_workers",
        lambda *_a, **_k: [outcome],
    )
    monkeypatch.setattr(radiomics, "radiomics_for_course_mr", lambda *_a, **_k: None)

    radiomics.run_radiomics(config, [course])

    aggregate = pd.read_excel(
        config.output_root / "Data" / "radiomics_all.xlsx",
        engine="openpyxl",
    )
    assert set(aggregate["radiomics_course_status"]) == {"extracted_with_failures"}
    assert set(aggregate["radiomics_roi_attempted"]) == {2}
    assert set(aggregate["radiomics_roi_extracted"]) == {1}
    assert set(aggregate["radiomics_roi_failed"]) == {1}
    counts = json.loads(aggregate.loc[0, "radiomics_roi_counts_by_source"])
    assert counts["AutoRTS_total"] == {"attempted": 1, "extracted": 0, "failed": 1}


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
    _write_contract(course, no_ct=True)
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
    _write_contract(course)
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
    _write_contract(course, no_ct=True)

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
    _write_contract(course, planning_nifti=image_path)
    stale = course / "radiomics_ct.xlsx"
    stale.write_bytes(b"stale")
    stale.with_suffix(".parquet").write_bytes(b"stale")
    monkeypatch.setattr(conda, "_ct_nifti_candidates", lambda _course: {"CT": image_path})
    monkeypatch.setattr(conda.sitk, "ReadImage", lambda _path: _Image())
    return course, mask_path, stale


def test_conda_nifti_fallback_empty_mask_is_recorded_as_degenerate(
    tmp_path, monkeypatch
):
    course, _mask_path, stale = _prepare_conda_nifti_fallback(tmp_path, monkeypatch)
    monkeypatch.setattr(
        conda.sitk,
        "GetArrayFromImage",
        lambda _image: np.zeros((2, 2, 2), dtype=np.uint8),
    )
    monkeypatch.setattr(conda.sitk, "WriteImage", lambda *_a, **_k: None)

    output = conda.radiomics_for_course_ct_nifti_fallback(course, _config(tmp_path))

    assert output == stale
    result = pd.read_excel(output, engine="openpyxl")
    assert result.loc[0, "extraction_status"] == "failed"
    assert result.loc[0, "extraction_failure_kind"] == "degenerate_mask"
    assert "empty mask" in result.loc[0, "extraction_status_detail"]
    assert not stale.with_suffix(".parquet").exists()


def test_conda_nifti_fallback_serialization_failure_is_recorded(
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

    output = conda.radiomics_for_course_ct_nifti_fallback(course, _config(tmp_path))

    assert output == stale
    result = pd.read_excel(output, engine="openpyxl")
    assert result.loc[0, "extraction_status"] == "failed"
    assert result.loc[0, "extraction_failure_kind"] == "extraction_error"
    assert "mask serialization failed" in result.loc[0, "extraction_status_detail"]
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
    _write_contract(course, no_ct=True)

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
    _write_contract(course)
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
    _write_contract(course, rtstruct=dirs.dicom_rtstruct / "RS.dcm")
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
    _write_contract(course)
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
    _write_contract(course)
    monkeypatch.setattr(radiomics, "_load_series_image", lambda *_a, **_k: None)

    with pytest.raises(RadiomicsCourseExtractionError, match="present but unreadable"):
        parallel.parallel_radiomics_for_course(_config(tmp_path, resume=True), course)

    assert not output.exists()
    assert not output.with_suffix(".parquet").exists()


class _FakeExtractor:
    settings = {"resampledPixelSpacing": [1.0, 1.0, 1.0]}

    def execute(self, _image, _mask):
        return {"original_firstorder_Mean": 1.0}


def test_direct_resume_missing_non_body_manual_roi_fails_and_invalidates(
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
    _write_contract(course, rtstruct=manual_rs)
    _write_current_auto_rtstruct(course)
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

    with pytest.raises(RadiomicsCourseExtractionError, match="missing required ROI.*Manual/GTV"):
        radiomics.radiomics_for_course(_config(tmp_path, resume=True), course)

    assert not output.exists()
    assert not output.with_suffix(".parquet").exists()


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
    _write_contract(course, rtstruct=manual_rs)
    _write_current_auto_rtstruct(course)
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


def test_direct_resume_rejects_failed_manual_row(tmp_path, monkeypatch):
    course = tmp_path / "P1" / "C1"
    dirs = build_course_dirs(course)
    dirs.dicom_ct.mkdir(parents=True)
    dirs.dicom_rtstruct.mkdir(parents=True)
    manual_rs = dirs.dicom_rtstruct / "RS.dcm"
    # The resume path loads the authoritative contract, so the fixture writes one.
    _write_contract(course, rtstruct=manual_rs)
    output = course / "radiomics_ct.xlsx"
    pd.DataFrame(
        [
            {
                "segmentation_source": "Manual",
                "roi_original_name": "PTV",
                "extraction_status": "failed",
                "extraction_status_detail": "clinical contour could not be read",
                "extraction_failure_kind": "extraction_error",
                "original_firstorder_Mean": 1.0,
            }
        ]
    ).to_excel(output, index=False)
    output.with_suffix(".parquet").write_bytes(b"stale")

    monkeypatch.setattr(radiomics, "_load_series_image", lambda *_a, **_k: _Image())
    monkeypatch.setattr(radiomics, "_extractor", lambda *_a, **_k: _FakeExtractor())
    monkeypatch.setattr(
        radiomics,
        "_rtstruct_masks",
        lambda _ct, path, **_kwargs: (
            {"PTV": np.ones((2, 2, 2), dtype=bool)}
            if Path(path) == manual_rs
            else {}
        ),
    )
    monkeypatch.setattr(radiomics, "list_custom_model_outputs", lambda _course: [])

    with pytest.raises(RadiomicsCourseExtractionError, match="required ROI Manual/PTV"):
        radiomics.radiomics_for_course(_config(tmp_path, resume=True), course)

    assert not output.exists()
    assert not output.with_suffix(".parquet").exists()


def test_parallel_resume_rejects_failed_configured_custom_row(tmp_path, monkeypatch):
    import rtpipeline.radiomics_parallel as parallel

    course = tmp_path / "P1" / "C1"
    dirs = build_course_dirs(course)
    dirs.dicom_ct.mkdir(parents=True)
    custom_rs = course / "RS_custom.dcm"
    custom_rs.write_bytes(b"present")
    # The resume path loads the authoritative contract, so the fixture writes one.
    _write_contract(course)
    custom_config = tmp_path / "custom.yaml"
    custom_config.write_text(
        "custom_structures:\n"
        "  - name: bowel_bag\n"
        "    operation: union\n"
        "    source_structures: [PTV]\n",
        encoding="utf-8",
    )
    output = course / "radiomics_ct.xlsx"
    pd.DataFrame(
        [
            {
                "segmentation_source": "Custom",
                "roi_original_name": "bowel_bag",
                "extraction_status": "failed",
                "extraction_status_detail": "configured mask could not be read",
                "extraction_failure_kind": "extraction_error",
                "original_firstorder_Mean": 1.0,
            }
        ]
    ).to_excel(output, index=False)
    output.with_suffix(".parquet").write_bytes(b"stale")

    monkeypatch.setattr(radiomics, "_load_series_image", lambda *_a, **_k: _Image())
    monkeypatch.setattr(radiomics, "_extractor", lambda *_a, **_k: _FakeExtractor())
    monkeypatch.setattr(parallel, "_is_rs_custom_stale", lambda *_a, **_k: False)
    monkeypatch.setattr(parallel, "_list_roi_names", lambda _path: ["bowel_bag"])
    monkeypatch.setattr(parallel, "list_custom_model_outputs", lambda _course: [])

    with pytest.raises(RadiomicsCourseExtractionError, match="required ROI Custom/bowel_bag"):
        parallel.parallel_radiomics_for_course(
            _config(
                tmp_path,
                resume=True,
                custom_structures_config=custom_config,
            ),
            course,
            max_workers=1,
        )

    assert not output.exists()
    assert not output.with_suffix(".parquet").exists()


def test_radiomics_status_contract_still_exposes_explicit_nothing_to_do():
    outcome = RadiomicsCourseOutcome.nothing_to_do("configured skip")
    assert outcome.status is RadiomicsCourseStatus.NOTHING_TO_DO


@pytest.mark.parametrize(
    ("source", "roi_name"),
    [("Manual", "PTV"), ("Custom", "bowel_bag")],
)
def test_persisted_required_failure_is_rejected_and_invalidated(
    tmp_path, source, roi_name
):
    output = tmp_path / "radiomics_ct.xlsx"
    pd.DataFrame(
        [
            {
                "segmentation_source": source,
                "roi_original_name": roi_name,
                "extraction_status": "failed",
                "extraction_status_detail": "mask could not be read",
                "extraction_failure_kind": "extraction_error",
                "original_firstorder_Mean": 1.0,
            }
        ]
    ).to_excel(output, index=False)
    output.with_suffix(".parquet").write_bytes(b"stale")

    with pytest.raises(RadiomicsCourseExtractionError, match=f"required ROI {source}/{roi_name}"):
        outcome_from_output(output)

    assert not output.exists()
    assert not output.with_suffix(".parquet").exists()


def test_persisted_missing_required_outcome_is_rejected_and_invalidated(tmp_path):
    output = tmp_path / "radiomics_ct.xlsx"
    pd.DataFrame(
        [
            {
                "segmentation_source": "AutoRTS_total",
                "roi_original_name": "vertebrae_T8",
                "extraction_status": "failed",
                "extraction_status_detail": "empty mask",
                "extraction_failure_kind": "degenerate_mask",
                "original_firstorder_Mean": 1.0,
            }
        ]
    ).to_excel(output, index=False)
    output.with_suffix(".parquet").write_bytes(b"stale")

    with pytest.raises(RadiomicsCourseExtractionError, match="Manual/PTV.*no persisted outcome"):
        outcome_from_output(
            output,
            required_by_identity={
                ("Manual", "PTV"): True,
                ("AutoRTS_total", "vertebrae_T8"): False,
            },
        )

    assert not output.exists()
    assert not output.with_suffix(".parquet").exists()


def test_persisted_required_below_minimum_status_remains_nonfatal(tmp_path):
    output = tmp_path / "radiomics_ct.xlsx"
    pd.DataFrame(
        [
            {
                "segmentation_source": "Manual",
                "roi_original_name": "znacznikAg",
                "extraction_status": "below_minimum_voxels",
                "extraction_status_detail": "ROI contains 3 voxels; configured minimum is 64",
                "extraction_failure_kind": "degenerate_mask",
                "voxel_count": 3,
            }
        ]
    ).to_excel(output, index=False)

    outcome = outcome_from_output(
        output,
        required_by_identity={("Manual", "znacznikAg"): True},
    )

    assert outcome.status is RadiomicsCourseStatus.EXTRACTED_WITH_FAILURES
    assert outcome.roi_counts == {
        "Manual": {"attempted": 1, "extracted": 0, "failed": 1}
    }
    assert output.exists()


def test_persisted_outcome_treats_blank_excel_status_as_success(tmp_path):
    output = tmp_path / "radiomics_ct.xlsx"
    pd.DataFrame(
        [
            {
                "segmentation_source": "Manual",
                "roi_original_name": "PTV",
                "extraction_status": None,
                "original_firstorder_Mean": 1.0,
            },
            {
                "segmentation_source": "AutoRTS_total",
                "roi_original_name": "vertebrae_T8",
                "extraction_status": "failed",
                "extraction_status_detail": "empty mask",
                "extraction_failure_kind": "degenerate_mask",
            },
        ]
    ).to_excel(output, index=False)

    outcome = outcome_from_output(output)

    assert outcome.status is RadiomicsCourseStatus.EXTRACTED_WITH_FAILURES
    assert outcome.roi_counts == {
        "Manual": {"attempted": 1, "extracted": 1, "failed": 0},
        "AutoRTS_total": {"attempted": 1, "extracted": 0, "failed": 1},
    }
    assert outcome.roi_failures == (
        {
            "roi_name": "vertebrae_T8",
            "source": "AutoRTS_total",
            "status": "failed",
            "failure_kind": "degenerate_mask",
            "reason": "empty mask",
        },
    )


def _parallel_course_with_fake_roi_results(tmp_path, monkeypatch, results_by_source):
    import rtpipeline.radiomics_parallel as parallel

    course = tmp_path / "P1" / "C1"
    dirs = build_course_dirs(course)
    dirs.dicom_ct.mkdir(parents=True)
    dirs.dicom_rtstruct.mkdir(parents=True)
    # The course contract is now the authority every stage consumes, so a course
    # fixture must carry one, built from synthetic DICOM exactly as a real course
    # is. The Manual source is only reached when the contract names an
    # authoritative RTSTRUCT, so pass it rather than leaving it unset.
    _write_contract(course, rtstruct=dirs.dicom_rtstruct / "RS.dcm")
    _write_current_auto_rtstruct(course)

    monkeypatch.setattr(radiomics, "_load_series_image", lambda *_a, **_k: object())
    monkeypatch.setattr(radiomics, "_extractor", lambda *_a, **_k: object())
    monkeypatch.setattr(
        parallel,
        "_list_roi_names",
        lambda path: ["PTV"] if Path(path).name == "RS.dcm" else ["vertebrae_T8", "lung_left"],
    )
    monkeypatch.setattr(parallel, "list_custom_model_outputs", lambda _course: [])
    monkeypatch.setattr(parallel, "_calculate_optimal_workers", lambda: 1)

    class FakeExecutor:
        def __init__(self, *_args, **_kwargs):
            pass

        def submit(self, _function, task):
            future = Future()
            action = results_by_source[task.source].pop(0)
            if isinstance(action, BaseException):
                future.set_exception(action)
            else:
                future.set_result(action(task))
            return future

        def shutdown(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", FakeExecutor)
    return course, parallel


def _success_record(task):
    return {
        "segmentation_source": task.source,
        "roi_original_name": task.roi_name,
        "patient_id": "P1",
        "course_id": "C1",
        "original_firstorder_Mean": 1.0,
    }


def test_parallel_auto_roi_failure_is_recorded_without_failing_course(tmp_path, monkeypatch):
    course, parallel = _parallel_course_with_fake_roi_results(
        tmp_path,
        monkeypatch,
        {
            "Manual": [_success_record],
            "AutoRTS_total": [RuntimeError("mask outside field of view"), _success_record],
        },
    )

    outcome = parallel.parallel_radiomics_for_course(
        _config(tmp_path), course, max_workers=1
    )

    assert outcome.status is RadiomicsCourseStatus.EXTRACTED_WITH_FAILURES
    assert outcome.roi_counts == {
        "Manual": {"attempted": 1, "extracted": 1, "failed": 0},
        "AutoRTS_total": {"attempted": 2, "extracted": 1, "failed": 1},
    }
    assert outcome.roi_failures[0]["roi_name"] == "vertebrae_T8"
    assert outcome.roi_failures[0]["source"] == "AutoRTS_total"
    assert "outside field of view" in outcome.roi_failures[0]["reason"]
    result = pd.read_excel(course / "radiomics_ct.xlsx", engine="openpyxl")
    failed = result[result["roi_original_name"] == "vertebrae_T8"].iloc[0]
    assert failed["extraction_status"] == "failed"
    assert failed["radiomics_course_status"] == "extracted_with_failures"
    assert int(failed["radiomics_roi_failed"]) == 1


@pytest.mark.parametrize("source", ["Manual", "Custom"])
def test_parallel_required_roi_failure_still_fails_course(tmp_path, monkeypatch, source):
    import rtpipeline.radiomics_parallel as parallel

    course = tmp_path / source / "P1" / "C1"
    dirs = build_course_dirs(course)
    dirs.dicom_ct.mkdir(parents=True)
    custom_config = None
    rtstruct_for_contract = None
    if source == "Manual":
        dirs.dicom_rtstruct.mkdir(parents=True)
        rtstruct_for_contract = dirs.dicom_rtstruct / "RS.dcm"
    else:
        (course / "RS_custom.dcm").write_bytes(b"present")
        custom_config = tmp_path / "custom.yaml"
        custom_config.write_text(
            "custom_structures:\n"
            "  - name: bowel_bag\n"
            "    operation: union\n"
            "    source_structures: [PTV]\n",
            encoding="utf-8",
        )
    # Every course now needs the authoritative contract; it also supplies the
    # planning CT series, so the placeholder slice above is no longer written.
    _write_contract(course, rtstruct=rtstruct_for_contract)
    _write_current_auto_rtstruct(course)
    monkeypatch.setattr(radiomics, "_load_series_image", lambda *_a, **_k: object())
    monkeypatch.setattr(radiomics, "_extractor", lambda *_a, **_k: object())
    monkeypatch.setattr(parallel, "_calculate_optimal_workers", lambda: 1)
    monkeypatch.setattr(parallel, "list_custom_model_outputs", lambda _course: [])
    monkeypatch.setattr(
        parallel,
        "_list_roi_names",
        lambda path: ["PTV"] if Path(path).name == "RS.dcm" else ["bowel_bag"],
    )
    monkeypatch.setattr(parallel, "_is_rs_custom_stale", lambda *_a, **_k: False)

    class FailingExecutor:
        def __init__(self, *_args, **_kwargs):
            pass

        def submit(self, _function, task):
            future = Future()
            future.set_exception(RuntimeError("configured ROI extraction failed"))
            return future

        def shutdown(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", FailingExecutor)
    config = _config(tmp_path)
    if source == "Custom":
        config.custom_structures_config = custom_config

    with pytest.raises(RadiomicsCourseExtractionError, match="required ROI"):
        parallel.parallel_radiomics_for_course(config, course, max_workers=1)


def test_all_best_effort_failures_are_degraded_and_not_empty_success(tmp_path, monkeypatch):
    course, parallel = _parallel_course_with_fake_roi_results(
        tmp_path,
        monkeypatch,
        {
            "Manual": [_success_record],
            "AutoRTS_total": [RuntimeError("outside field of view"), RuntimeError("empty mask")],
        },
    )
    outcome = parallel.parallel_radiomics_for_course(
        _config(tmp_path), course, max_workers=1
    )

    assert outcome.status is RadiomicsCourseStatus.EXTRACTED_WITH_FAILURES
    assert outcome.attempted == 3
    assert outcome.extracted_count == 1
    assert outcome.failed_count == 2
    result = pd.read_excel(course / "radiomics_ct.xlsx", engine="openpyxl")
    assert len(result) == 3
    assert set(result["extraction_status"].fillna("success")) == {"success", "failed"}


def test_parallel_required_below_minimum_status_is_recorded_without_course_failure(
    tmp_path, monkeypatch
):
    def below_minimum(task):
        return {
            "modality": "CT",
            "segmentation_source": task.source,
            "roi_name": task.roi_name,
            "roi_original_name": task.roi_name,
            "patient_id": "P1",
            "course_id": "C1",
            "extraction_status": "below_minimum_voxels",
            "extraction_status_detail": "ROI contains 3 voxels; configured minimum is 64",
            "extraction_failure_kind": "degenerate_mask",
            "voxel_count": 3,
        }

    course, parallel = _parallel_course_with_fake_roi_results(
        tmp_path,
        monkeypatch,
        {
            "Manual": [below_minimum],
            "AutoRTS_total": [_success_record, _success_record],
        },
    )

    outcome = parallel.parallel_radiomics_for_course(
        _config(tmp_path), course, max_workers=1
    )

    assert outcome.status is RadiomicsCourseStatus.EXTRACTED_WITH_FAILURES
    assert outcome.roi_counts["Manual"] == {
        "attempted": 1,
        "extracted": 0,
        "failed": 1,
    }
    result = pd.read_excel(course / "radiomics_ct.xlsx", engine="openpyxl")
    manual = result[result["segmentation_source"] == "Manual"].iloc[0]
    assert manual["extraction_status"] == "below_minimum_voxels"
    assert int(manual["voxel_count"]) == 3


def test_conda_required_below_minimum_status_is_nonfatal_and_published(
    tmp_path, monkeypatch
):
    output = tmp_path / "radiomics_ct.xlsx"
    monkeypatch.setattr(conda, "check_radiomics_env", lambda **_kwargs: True)
    task = {
        "image_path": "image.nrrd",
        "mask_path": None,
        "roi_name": "znacznikAg",
        "cleanup": False,
        "metadata": {
            "segmentation_source": "Manual",
            "roi_original_name": "znacznikAg",
        },
        "precomputed_failure": {
            "status": "below_minimum_voxels",
            "reason": "ROI contains 3 voxels; configured minimum is 64",
            "failure_kind": "degenerate_mask",
        },
    }

    result_path = conda.process_radiomics_batch(
        [task],
        output,
        sequential=True,
        max_workers=1,
        enable_heartbeat=False,
    )

    assert result_path == output
    result = pd.read_excel(output, engine="openpyxl")
    assert result.loc[0, "extraction_status"] == "below_minimum_voxels"
    assert result.loc[0, "radiomics_course_status"] == "extracted_with_failures"
