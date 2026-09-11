"""Cohort robustness aggregation must admit only certified per-course tables.

Every case below builds its own synthetic course tree under ``tmp_path`` and
runs the real :func:`rtpipeline.radiomics_robustness.robustness_for_course`, so
the per-course parquet *and* its run-bound source-disposition sidecar are
produced by the shipped publication path.

The fixture substitutes everything that would otherwise need DICOM on disk: the
course contract, the main-CT ROI identity catalog, the series image, the
standard RTSTRUCT sources and their masks, the custom-model outputs, and
radiomics feature extraction itself (no PyRadiomics subprocess). What that
leaves real is what these tests are about - the perturbation grid, the
publication path, the sidecar writer, and the admission, sidecar validation and
cohort summary code under test. It is not a receipt for DICOM ingestion or for
real feature extraction. Nothing here reads clinical data, starts a service, or
writes outside the test's temporary directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from rtpipeline import custom_models
from rtpipeline import radiomics as rm
from rtpipeline import radiomics_robustness as rr
from rtpipeline.config import PipelineConfig
from rtpipeline.radiomics_ct_contract import CT_EXTRACTION_ARMS

# One volume state, three translation states, one contour and one noise level.
CONDITIONS = ("ntcv_v0", "ntcv_t0_0_4_v0", "ntcv_t0_0_-4_v0")
SHAPE = (21, 21, 21)


def _roi_array() -> np.ndarray:
    """A centred block, so both ±4 mm translations stay inside the image."""
    array = np.zeros(SHAPE, np.uint8)
    array[8:14, 8:14, 8:14] = 1
    return array


def _build_cohort(
    tmp_path,
    monkeypatch,
    *,
    patients=("P1", "P2"),
    extra_feature_for=(),
):
    """Publish one real robustness table + sidecar per synthetic course."""
    output_root = tmp_path / "Output"
    array = _roi_array()
    mask = sitk.GetImageFromArray(array)
    image = sitk.GetImageFromArray(
        np.arange(array.size, dtype=np.float32).reshape(SHAPE)
    )

    courses = {}
    identities = {}
    for index, patient_id in enumerate(patients):
        course_id = f"C{index + 1}"
        course = output_root / patient_id / course_id
        course.mkdir(parents=True)
        courses[patient_id] = course
        identities[course] = rr.RobustnessRoiIdentity.from_mapping(
            dict(
                patient_id=patient_id,
                course_id=course_id,
                series_uid="1.2.3",
                segmentation_source="Manual",
                mask_identity=f"source-mask-{patient_id}",
                roi_original_name="ROI",
                stable_roi_identifier=f"roi-{patient_id}",
            )
        )

    monkeypatch.setattr(
        rr,
        "load_course_contract",
        lambda course: SimpleNamespace(
            planning_ct_dir=Path(course),
            planning_ct={"series_instance_uid": "1.2.3"},
        ),
    )
    monkeypatch.setattr(
        rr,
        "_load_main_ct_identity_catalog",
        lambda course_dir, **kwargs: (
            {("Manual", "ROI"): identities[Path(course_dir)]},
            {},
        ),
    )
    monkeypatch.setattr(rm, "_load_series_image", lambda _: image)
    monkeypatch.setattr(
        rm,
        "_standard_rtstruct_sources",
        lambda contract, course_dir: [
            ("Manual", Path(course_dir) / "RS.dcm", None)
        ],
    )
    monkeypatch.setattr(rm, "_rtstruct_masks", lambda *a, **k: {"ROI": array})
    monkeypatch.setattr(rm, "_mask_from_array_like", lambda *a: mask)
    monkeypatch.setattr(custom_models, "list_custom_model_outputs", lambda *a: [])
    monkeypatch.setenv("RTPIPELINE_DISABLE_PARALLEL_RADIOMICS", "1")
    monkeypatch.setenv("RTPIPELINE_RADIOMICS_THREAD_LIMIT", "1")
    monkeypatch.setenv("RTPIPELINE_MAX_WORKERS", "1")

    def _features(image_arg, masks, config, **kwargs):
        fields = kwargs["source_identity"].as_dict()
        patient_id = fields["patient_id"]
        base = 10.0 * (list(patients).index(patient_id) + 1)
        feature_names = ["original_firstorder_Mean"]
        if patient_id in extra_feature_for:
            feature_names.append("original_firstorder_Maximum")
        rows = []
        for offset, perturbation_id in enumerate(sorted(masks)):
            identity = rr._perturbed_mask_identity(masks[perturbation_id])
            for arm in CT_EXTRACTION_ARMS:
                for feature_name in feature_names:
                    rows.append(
                        {
                            **fields,
                            "structure": "ROI",
                            "modality": "CT",
                            "measurement_type": rr.ROBUSTNESS_MEASUREMENT_TYPE,
                            "perturbed_mask_identity": identity,
                            "perturbation_id": perturbation_id,
                            "extraction_arm": arm,
                            "run_identifier": kwargs["run_identifier"],
                            "feature_name": feature_name,
                            "value": base + 0.01 * offset,
                        }
                    )
        return pd.DataFrame(rows)

    monkeypatch.setattr(rr, "extract_features_for_masks", _features)

    config = PipelineConfig(tmp_path, tmp_path, tmp_path, max_workers_override=1)
    rob = rr.RobustnessConfig(
        enabled=True,
        perturbation=rr.PerturbationConfig(
            small_volume_changes=[0.0],
            max_translation_mm=4.0,
            n_random_contour_realizations=0,
            noise_levels=[0.0],
            apply_to_structures=["ROI"],
        ),
    )

    tables = []
    for patient_id in patients:
        published = rr.robustness_for_course(config, rob, courses[patient_id])
        assert published is not None
        assert rr.robustness_source_dispositions_path(
            courses[patient_id]
        ).is_file()
        tables.append(Path(published))

    return SimpleNamespace(
        courses=[courses[patient_id] for patient_id in patients],
        tables=tables,
        rob=rob,
        config=config,
    )


def _summary_paths(tmp_path):
    output = tmp_path / "_RESULTS" / "radiomics_robustness_summary.xlsx"
    return output, output.parent / (output.stem + "_raw_values.parquet")


def _plant_stale_outputs(output, raw):
    """A previous cohort's nominal outputs must not survive a rejection."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"stale workbook")
    raw.write_bytes(b"stale raw values")


def _assert_no_nominal_outputs(output, raw):
    assert not output.exists(), "a rejected aggregation left a nominal workbook"
    assert not raw.exists(), "a rejected aggregation left nominal raw values"
    if output.parent.exists():
        assert not list(output.parent.glob("*.tmp")), "temporary outputs survived"


def _sidecar_payload(course):
    return json.loads(
        rr.robustness_source_dispositions_path(course).read_text(encoding="utf-8")
    )


def _write_sidecar_payload(course, payload):
    rr.robustness_source_dispositions_path(course).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Positive control: unchanged, valid aggregation
# ---------------------------------------------------------------------------


def test_certified_course_tables_are_admitted_and_summarized(tmp_path, monkeypatch):
    cohort = _build_cohort(tmp_path, monkeypatch)
    output, raw = _summary_paths(tmp_path)

    rr.aggregate_robustness_results(cohort.tables, output, cohort.rob)

    assert output.is_file() and raw.is_file()
    raw_frame = pd.read_parquet(raw)
    # 2 courses x 3 conditions x 2 CT arms x 1 feature, counted exactly once.
    assert len(raw_frame) == 12
    assert set(raw_frame["patient_id"]) == {"P1", "P2"}
    assert set(raw_frame["perturbation_id"]) == set(CONDITIONS)
    assert set(raw_frame["extraction_arm"]) == set(CT_EXTRACTION_ARMS)

    expected = rr.summarize_feature_stability(
        pd.concat(
            [pd.read_parquet(table) for table in cohort.tables], ignore_index=True
        ),
        cohort.rob,
    )
    summary = pd.read_excel(output, sheet_name="global_summary")
    assert set(summary["extraction_arm"]) == set(CT_EXTRACTION_ARMS)
    assert set(summary["structure"]) == {"ROI"}
    assert set(summary["segmentation_source"]) == {"Manual"}
    assert set(summary["n_subjects"]) == {2}
    assert set(summary["n_courses"]) == {2}
    assert set(summary["n_perturbations"]) == {len(CONDITIONS)}
    assert set(summary["cov_status"]) == {"complete"}
    for column in ("icc", "cov_pct", "qcd"):
        np.testing.assert_allclose(
            summary[column].to_numpy(dtype=float),
            expected[column].to_numpy(dtype=float),
            equal_nan=True,
        )

    per_source = pd.read_excel(output, sheet_name="per_source_summary")
    assert set(per_source["segmentation_source"]) == {"Manual"}
    assert set(per_source["extraction_arm"]) == set(CT_EXTRACTION_ARMS)
    for sheet in (
        "per_structure_source",
        "robust_features",
        "acceptable_features",
        "robust_features_per_source",
    ):
        pd.read_excel(output, sheet_name=sheet)


def test_summary_uses_admitted_bytes_when_an_input_drifts_mid_aggregation(
    tmp_path, monkeypatch
):
    cohort = _build_cohort(tmp_path, monkeypatch)
    reference_output, reference_raw = _summary_paths(tmp_path)
    rr.aggregate_robustness_results(cohort.tables, reference_output, cohort.rob)
    reference = pd.read_parquet(reference_raw)

    original_summarize = rr.summarize_feature_stability

    def _drifting_summarize(frame, config, group_columns=None):
        # An admitted input is destroyed the moment summarising starts. The
        # aggregate must already be bound to the bytes it admitted.
        cohort.tables[0].write_bytes(b"no longer a parquet file")
        return original_summarize(frame, config, group_columns)

    monkeypatch.setattr(rr, "summarize_feature_stability", _drifting_summarize)
    drifted_output = tmp_path / "_DRIFT" / "summary.xlsx"
    rr.aggregate_robustness_results(cohort.tables, drifted_output, cohort.rob)

    drifted = pd.read_parquet(
        drifted_output.parent / (drifted_output.stem + "_raw_values.parquet")
    )
    pd.testing.assert_frame_equal(drifted, reference)


# ---------------------------------------------------------------------------
# Sidecar admission
# ---------------------------------------------------------------------------


def test_table_without_source_dispositions_is_rejected(tmp_path, monkeypatch):
    cohort = _build_cohort(tmp_path, monkeypatch)
    rr.robustness_source_dispositions_path(cohort.courses[1]).unlink()
    output, raw = _summary_paths(tmp_path)
    _plant_stale_outputs(output, raw)

    with pytest.raises(
        FileNotFoundError, match="no robustness source dispositions artifact"
    ):
        rr.aggregate_robustness_results(cohort.tables, output, cohort.rob)

    _assert_no_nominal_outputs(output, raw)


def test_unreadable_source_disposition_sidecar_is_rejected(tmp_path, monkeypatch):
    cohort = _build_cohort(tmp_path, monkeypatch)
    rr.robustness_source_dispositions_path(cohort.courses[0]).write_text(
        "{ this is not json", encoding="utf-8"
    )
    output, raw = _summary_paths(tmp_path)

    with pytest.raises(ValueError):
        rr.aggregate_robustness_results(cohort.tables, output, cohort.rob)

    _assert_no_nominal_outputs(output, raw)


def test_tampered_source_disposition_sidecar_is_rejected(tmp_path, monkeypatch):
    cohort = _build_cohort(tmp_path, monkeypatch)
    payload = _sidecar_payload(cohort.courses[0])
    payload["row_count"] = payload["row_count"] + 1
    _write_sidecar_payload(cohort.courses[0], payload)
    output, raw = _summary_paths(tmp_path)

    with pytest.raises(ValueError, match="corrupt robustness source dispositions"):
        rr.aggregate_robustness_results(cohort.tables, output, cohort.rob)

    _assert_no_nominal_outputs(output, raw)


def test_sidecar_from_another_configuration_is_rejected(tmp_path, monkeypatch):
    cohort = _build_cohort(tmp_path, monkeypatch)
    other = rr.RobustnessConfig(
        enabled=True,
        perturbation=rr.PerturbationConfig(
            small_volume_changes=[0.0],
            max_translation_mm=4.0,
            n_random_contour_realizations=0,
            noise_levels=[0.0],
            apply_to_structures=["ROI"],
        ),
        thresholds=rr.RobustnessThresholds(icc_robust=0.5),
    )
    output, raw = _summary_paths(tmp_path)

    with pytest.raises(ValueError, match="different effective configuration"):
        rr.aggregate_robustness_results(cohort.tables, output, other)

    _assert_no_nominal_outputs(output, raw)


def test_sidecar_does_not_certify_a_twin_table_in_the_same_course(
    tmp_path, monkeypatch
):
    cohort = _build_cohort(tmp_path, monkeypatch)
    twin = cohort.courses[0] / "radiomics_robustness_ct_copy.parquet"
    twin.write_bytes(cohort.tables[0].read_bytes())
    output, raw = _summary_paths(tmp_path)

    with pytest.raises(ValueError, match="different effective configuration"):
        rr.aggregate_robustness_results(
            [twin, cohort.tables[1]], output, cohort.rob
        )

    _assert_no_nominal_outputs(output, raw)


def test_table_bytes_changed_after_publication_are_rejected(tmp_path, monkeypatch):
    cohort = _build_cohort(tmp_path, monkeypatch)
    frame = pd.read_parquet(cohort.tables[0])
    frame.loc[frame.index[0], "value"] = 999.0
    frame.to_parquet(cohort.tables[0], index=False)
    output, raw = _summary_paths(tmp_path)

    with pytest.raises(ValueError, match="changed after publication"):
        rr.aggregate_robustness_results(cohort.tables, output, cohort.rob)

    _assert_no_nominal_outputs(output, raw)


def test_sidecar_from_a_different_run_is_rejected(tmp_path, monkeypatch):
    cohort = _build_cohort(tmp_path, monkeypatch)
    payload = _sidecar_payload(cohort.courses[0])
    payload["robustness_run_identifier"] = "a-different-run"
    _write_sidecar_payload(cohort.courses[0], payload)
    output, raw = _summary_paths(tmp_path)

    with pytest.raises(ValueError, match="stale robustness source dispositions"):
        rr.aggregate_robustness_results(cohort.tables, output, cohort.rob)

    _assert_no_nominal_outputs(output, raw)


# ---------------------------------------------------------------------------
# Table run identity
# ---------------------------------------------------------------------------


def test_table_without_a_run_identifier_column_is_rejected(tmp_path, monkeypatch):
    cohort = _build_cohort(tmp_path, monkeypatch)
    frame = pd.read_parquet(cohort.tables[0]).drop(columns=["run_identifier"])
    frame.to_parquet(cohort.tables[0], index=False)
    output, raw = _summary_paths(tmp_path)

    with pytest.raises(RuntimeError, match="no run_identifier column"):
        rr.aggregate_robustness_results(cohort.tables, output, cohort.rob)

    _assert_no_nominal_outputs(output, raw)


def test_table_with_blank_run_identity_is_rejected(tmp_path, monkeypatch):
    cohort = _build_cohort(tmp_path, monkeypatch)
    frame = pd.read_parquet(cohort.tables[0])
    frame["run_identifier"] = ""
    frame.to_parquet(cohort.tables[0], index=False)
    output, raw = _summary_paths(tmp_path)

    with pytest.raises(RuntimeError, match="no readable robustness run identity"):
        rr.aggregate_robustness_results(cohort.tables, output, cohort.rob)

    _assert_no_nominal_outputs(output, raw)


def test_table_carrying_several_runs_is_rejected(tmp_path, monkeypatch):
    cohort = _build_cohort(tmp_path, monkeypatch)
    frame = pd.read_parquet(cohort.tables[0])
    frame.loc[frame.index[:2], "run_identifier"] = "a-second-run"
    frame.to_parquet(cohort.tables[0], index=False)
    output, raw = _summary_paths(tmp_path)

    with pytest.raises(RuntimeError, match="carries 2 robustness run identities"):
        rr.aggregate_robustness_results(cohort.tables, output, cohort.rob)

    _assert_no_nominal_outputs(output, raw)


def test_table_from_another_course_is_rejected(tmp_path, monkeypatch):
    cohort = _build_cohort(tmp_path, monkeypatch)
    frame = pd.read_parquet(cohort.tables[0])
    frame["patient_id"] = "P9"
    frame.to_parquet(cohort.tables[0], index=False)
    output, raw = _summary_paths(tmp_path)

    with pytest.raises(RuntimeError, match="does not belong to course"):
        rr.aggregate_robustness_results(cohort.tables, output, cohort.rob)

    _assert_no_nominal_outputs(output, raw)


# ---------------------------------------------------------------------------
# Duplicate inputs, technical partials, publication failure
# ---------------------------------------------------------------------------


def test_duplicate_course_input_is_detected_not_double_counted(tmp_path, monkeypatch):
    cohort = _build_cohort(tmp_path, monkeypatch)
    output, raw = _summary_paths(tmp_path)

    with pytest.raises(RuntimeError, match="duplicate robustness aggregation input"):
        rr.aggregate_robustness_results(
            [cohort.tables[0], cohort.tables[1], cohort.tables[0]],
            output,
            cohort.rob,
        )

    _assert_no_nominal_outputs(output, raw)


def test_technical_partial_table_leaves_no_nominal_outputs(tmp_path, monkeypatch):
    cohort = _build_cohort(tmp_path, monkeypatch)
    frame = pd.read_parquet(cohort.tables[1])
    frame["robustness_status"] = frame["robustness_status"].astype(object)
    frame.loc[frame.index[:2], "robustness_status"] = "technical_failure"
    frame.to_parquet(cohort.tables[1], index=False)
    output, raw = _summary_paths(tmp_path)
    _plant_stale_outputs(output, raw)

    with pytest.raises(RuntimeError, match="require recovery"):
        rr.aggregate_robustness_results(cohort.tables, output, cohort.rob)

    _assert_no_nominal_outputs(output, raw)


def test_missing_input_path_is_still_reported(tmp_path, monkeypatch):
    cohort = _build_cohort(tmp_path, monkeypatch)
    output, raw = _summary_paths(tmp_path)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        rr.aggregate_robustness_results(
            [cohort.tables[0], tmp_path / "absent.parquet"], output, cohort.rob
        )

    _assert_no_nominal_outputs(output, raw)


def test_publication_failure_leaves_no_nominal_outputs(tmp_path, monkeypatch):
    cohort = _build_cohort(tmp_path, monkeypatch)
    output, raw = _summary_paths(tmp_path)

    def _failing_writer(*args, **kwargs):
        raise OSError("simulated workbook publication failure")

    monkeypatch.setattr(rr.pd, "ExcelWriter", _failing_writer)

    with pytest.raises(OSError, match="simulated workbook publication failure"):
        rr.aggregate_robustness_results(cohort.tables, output, cohort.rob)

    _assert_no_nominal_outputs(output, raw)


def test_admitted_cohort_still_fails_on_inconsistent_feature_sets(
    tmp_path, monkeypatch
):
    cohort = _build_cohort(tmp_path, monkeypatch, extra_feature_for=("P2",))
    output, raw = _summary_paths(tmp_path)

    with pytest.raises(ValueError, match="inconsistent feature sets across subjects"):
        rr.aggregate_robustness_results(cohort.tables, output, cohort.rob)

    _assert_no_nominal_outputs(output, raw)
