"""Producer-declared per-perturbation feature gaps must not void a course.

Production case (paths withheld): a CT robustness course failed with
"feature columns differ across perturbations for Manual/GTVm" listing only the
primary_resegmented arm of the three noise levels of one geometry
(t0_0_-4, c2, v-15). Mechanism, reproduced with PyRadiomics 3.0.1 on synthetic
data: after erosion and translation, the primary arm's HU window left fewer
than minimumROISize voxels, so extract_ct_roi_arms returned shape features
only with intensity_texture_disposition=below_minimum_voxels. The long table
drops that disposition, so the shorter feature set looked unexplained.

A second path has the same effect: a feature the extractor returns as NaN in
one perturbation only (glcm MCC, declared in radiomics_undefined_features_json)
is dropped from that perturbation's rows.

Here the extractor is replaced by synthetic records with the exact
extract_ct_roi_arms shape; the robustness course, validation, publication and
aggregation code is real. No clinical data.
"""
from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from rtpipeline import custom_models
from rtpipeline import radiomics as rm
from rtpipeline import radiomics_conda
from rtpipeline import radiomics_parallel as rp
from rtpipeline import radiomics_robustness as rr
from rtpipeline.config import PipelineConfig
from rtpipeline.radiomics_ct_contract import CT_EXTRACTION_ARMS, PRIMARY_ARM, SENSITIVITY_ARM

SHAPE = ("original_shape_VoxelVolume", "original_shape_Sphericity")
TEXTURE = ("original_firstorder_Mean", "original_glcm_Contrast", "original_glcm_MCC")
BELOW = ("ntcv_v-15", "ntcv_n10_v-15")

# Per-test extraction plan. The parallel path forks workers after it is set,
# so they inherit it.
PLAN: dict = {}


def _value(name, pid, arm, subject):
    base = {"original_shape_VoxelVolume": 700.0, "original_shape_Sphericity": 0.8,
            "original_firstorder_Mean": 40.0, "original_glcm_Contrast": 3.0,
            "original_glcm_MCC": 0.5}[name]
    jitter = (sum(map(ord, pid + arm + name)) % 17) / 400.0
    return base * (1.0 + 0.3 * subject + jitter)


def _records(pid, identity, run_identifier, mask_identity):
    subject = int(identity["patient_id"][1:]) if identity["patient_id"][1:].isdigit() else 0
    records = []
    for arm in CT_EXTRACTION_ARMS:
        record = {
            **identity, "roi_name": identity["roi_original_name"], "modality": "CT",
            "measurement_type": rr.ROBUSTNESS_MEASUREMENT_TYPE,
            "perturbed_mask_identity": mask_identity, "extraction_arm": arm,
            "roi_class": "target", "roi_class_map_version": "v1", "roi_class_map_hash": "h",
            "effective_parameter_hash": f"e-{arm}", "configured_parameter_hash": f"c-{arm}",
            "run_identifier": run_identifier, "extraction_status": "success",
            "shape_disposition": "success", "intensity_texture_disposition": "success",
            "radiomics_undefined_features_json": "[]",
            "resegment_after_count": 500, "observed_roi_dimensions_after_resegmentation": 3,
        }
        record.update({name: _value(name, pid, arm, subject) for name in SHAPE})
        if arm == PRIMARY_ARM and pid in PLAN.get("below", ()):
            record.update(intensity_texture_disposition="below_minimum_voxels",
                          resegment_after_count=16,
                          effective_resegment_lower_hu=-1000.0,
                          effective_resegment_upper_hu=400.0)
        else:
            record.update({name: _value(name, pid, arm, subject) for name in TEXTURE})
        if (pid, arm) in PLAN.get("undefined", ()):
            record["original_glcm_MCC"] = None
            record["radiomics_undefined_features_json"] = '["original_glcm_MCC"]'
        for drop_pid, drop_arm, drop_name in PLAN.get("drop", ()):
            if (pid, arm) == (drop_pid, drop_arm):
                record.pop(drop_name, None)
        records.append(record)
    return records


def fake_conda_batch(tasks, params_file, timeout_per_roi=120):
    from rtpipeline.radiomics_robustness import ROBUSTNESS_SOURCE_IDENTITY_COLUMNS

    results = []
    for index, task in enumerate(tasks):
        identity = {c: task["metadata"][c] for c in ROBUSTNESS_SOURCE_IDENTITY_COLUMNS}
        results.append({
            "__task_index__": index, "__status__": "success",
            "__records__": _records(task["robustness_perturbation_id"], identity,
                                    task["run_identifier"], "sha256:" + "0" * 64),
        })
    return results


def fake_parallel_worker(task):
    params = task[1]
    pid = params["extra_metadata"]["perturbation_id"]
    identity = {c: params[c] for c in rr.ROBUSTNESS_SOURCE_IDENTITY_COLUMNS}
    return {"__records__": _records(pid, identity, params["run_identifier"],
                                    params["perturbed_mask_identity"]),
            "segmentation_source": params["segmentation_source"],
            "roi_name": params["roi_name"], "perturbation_id": pid}


def _course(tmp_path, monkeypatch, *, patient="P1", parallel=False, plan=None):
    PLAN.clear()
    PLAN.update(plan or {})
    course = tmp_path / patient / "C"
    course.mkdir(parents=True)
    array = np.zeros((17, 17, 17), np.uint8)
    array[4:13, 4:13, 4:13] = 1
    mask = sitk.GetImageFromArray(array)
    image = sitk.GetImageFromArray(
        (np.arange(array.size, dtype=np.float32).reshape(array.shape) % 97) + 20.0
    )

    def catalog(course_dir, **kwargs):
        return ({("Manual", "GTVm"): rr.RobustnessRoiIdentity.from_mapping(dict(
            patient_id=Path(course_dir).parent.name, course_id="C", series_uid="1.2.3",
            segmentation_source="Manual", mask_identity="source-mask",
            roi_original_name="GTVm", stable_roi_identifier="roi-1",
        ))}, {})

    monkeypatch.setattr(rr, "load_course_contract", lambda d: SimpleNamespace(
        planning_ct_dir=d, planning_ct={"series_instance_uid": "1.2.3"}))
    monkeypatch.setattr(rr, "_load_main_ct_identity_catalog", catalog)
    monkeypatch.setattr(rm, "_load_series_image", lambda _: image)
    monkeypatch.setattr(rm, "_standard_rtstruct_sources",
                        lambda contract, course_dir: [("Manual", course_dir / "RS.dcm", None)])
    monkeypatch.setattr(rm, "_rtstruct_masks", lambda *a, **k: {"GTVm": array})
    monkeypatch.setattr(rm, "_mask_from_array_like", lambda *a: mask)
    monkeypatch.setattr(custom_models, "list_custom_model_outputs", lambda *a: [])
    monkeypatch.setattr(radiomics_conda, "check_radiomics_env", lambda *a, **k: True)
    monkeypatch.setattr(radiomics_conda, "extract_radiomics_batch_with_conda", fake_conda_batch)
    monkeypatch.setattr(rp, "_isolated_radiomics_extraction_with_retry", fake_parallel_worker)
    monkeypatch.setattr(rr, "get_context", lambda _: mp.get_context("fork"))
    monkeypatch.setenv("RTPIPELINE_DISABLE_PARALLEL_RADIOMICS", "0" if parallel else "1")
    monkeypatch.setenv("RTPIPELINE_RADIOMICS_THREAD_LIMIT", "1")
    monkeypatch.setenv("RTPIPELINE_MAX_WORKERS", "1")
    cfg = PipelineConfig(tmp_path, tmp_path, tmp_path, max_workers_override=1)
    rob = rr.RobustnessConfig(enabled=True, perturbation=rr.PerturbationConfig(
        small_volume_changes=[-0.15, 0.0], max_translation_mm=0.,
        n_random_contour_realizations=0, noise_levels=[0., 10.],
        apply_to_structures=["GTV*"],
    ))
    return course, cfg, rob


MODES = pytest.mark.parametrize("parallel", [False, True], ids=["sequential", "parallel"])


@MODES
def test_primary_below_minimum_is_published_as_not_evaluable(tmp_path, monkeypatch, parallel):
    """Reproduces the production failure; on f16d72a this course raised."""
    course, cfg, rob = _course(tmp_path, monkeypatch, parallel=parallel, plan={"below": BELOW})
    table = pd.read_parquet(rr.robustness_for_course(cfg, rob, course))
    gaps = table[table.robustness_status.eq(rr.ROBUSTNESS_FEATURE_NOT_EVALUABLE_STATUS)]
    assert set(zip(gaps.perturbation_id, gaps.extraction_arm, gaps.feature_name)) == {
        (pid, PRIMARY_ARM, name) for pid in BELOW for name in TEXTURE
    }
    assert set(gaps.reason_code) == {"primary_resegmented_below_minimum_voxels"}
    assert gaps.value.isna().all()
    evidence = json.loads(gaps[rr.ROBUSTNESS_FEATURE_EVALUABILITY_EVIDENCE_COLUMN].iloc[0])
    assert evidence["resegment_after_count"] == 16
    assert evidence["intensity_texture_disposition"] == "below_minimum_voxels"
    measured = table[table.robustness_status.eq("measured")]
    assert np.isfinite(measured.value).all()
    # Shape features stay measured in every perturbation-arm.
    assert len(measured) == 4 * 2 * len(SHAPE) + (4 * 2 - 2) * len(TEXTURE)
    # The not-evaluable rows keep the exact perturbation-arm identity.
    for column in (*rr.ROBUSTNESS_SOURCE_IDENTITY_COLUMNS, "perturbed_mask_identity",
                   "run_identifier", "effective_parameter_hash"):
        for (pid, arm), group in table.groupby(["perturbation_id", "extraction_arm"]):
            assert group[column].nunique() == 1, (pid, arm, column)
    rr._validate_extracted_feature_frame(table, set(table.perturbation_id), "Manual/GTVm")


@MODES
def test_undefined_feature_in_one_perturbation_is_published(tmp_path, monkeypatch, parallel):
    course, cfg, rob = _course(tmp_path, monkeypatch, parallel=parallel,
                               plan={"undefined": {("ntcv_n10_v0", SENSITIVITY_ARM)}})
    table = pd.read_parquet(rr.robustness_for_course(cfg, rob, course))
    gaps = table[table.robustness_status.eq(rr.ROBUSTNESS_FEATURE_NOT_EVALUABLE_STATUS)]
    assert list(zip(gaps.perturbation_id, gaps.extraction_arm, gaps.feature_name,
                    gaps.reason_code)) == [
        ("ntcv_n10_v0", SENSITIVITY_ARM, "original_glcm_MCC", "extractor_declared_undefined_feature")
    ]


@MODES
@pytest.mark.parametrize("plan", [
    # A gap nothing declared.
    {"drop": {("ntcv_v0", SENSITIVITY_ARM, "original_firstorder_Mean")}},
    # A below-minimum declaration explains the primary arm only.
    {"below": BELOW, "drop": {("ntcv_v-15", SENSITIVITY_ARM, "original_glcm_Contrast")}},
    # A declared-undefined feature does not explain a different missing one.
    {"undefined": {("ntcv_v0", SENSITIVITY_ARM)},
     "drop": {("ntcv_v0", SENSITIVITY_ARM, "original_glcm_Contrast")}},
], ids=["undeclared", "wrong-arm", "undefined-other-feature"])
def test_unexplained_feature_gap_still_fails_closed(tmp_path, monkeypatch, parallel, plan):
    course, cfg, rob = _course(tmp_path, monkeypatch, parallel=parallel, plan=plan)
    with pytest.raises(RuntimeError, match="feature columns differ across perturbations"):
        rr.robustness_for_course(cfg, rob, course)
    assert not (course / "radiomics_robustness_ct.parquet").exists()
    assert not rr.robustness_source_dispositions_path(course).exists()


def test_mixed_explained_and_unexplained_gap_adds_nothing(tmp_path, monkeypatch):
    """One unexplained gap leaves the ROI's rows exactly as before the change,
    so the pre-existing check sees and reports the same frame."""
    course, cfg, rob = _course(tmp_path, monkeypatch, plan={
        "below": BELOW, "drop": {("ntcv_v0", SENSITIVITY_ARM, "original_firstorder_Mean")}})
    declare = rr._declare_not_evaluable_features
    calls = []

    def recording(frame, declarations):
        result = declare(frame, declarations)
        calls.append((result is frame, len(declarations)))
        return result

    monkeypatch.setattr(rr, "_declare_not_evaluable_features", recording)
    with pytest.raises(RuntimeError, match="feature columns differ") as error:
        rr.robustness_for_course(cfg, rob, course)
    assert calls == [(True, len(BELOW))]
    assert repr(("ntcv_v0", SENSITIVITY_ARM)) in str(error.value)


def _published(tmp_path, monkeypatch, plan):
    course, cfg, rob = _course(tmp_path, monkeypatch, plan=plan)
    return pd.read_parquet(rr.robustness_for_course(cfg, rob, course)), rob


@pytest.mark.parametrize("tamper", [
    "shape_feature", "value", "reason", "sensitivity_arm", "unapproved_undefined", "duplicate",
])
def test_forged_not_evaluable_rows_are_rejected(tmp_path, monkeypatch, tamper):
    table, _ = _published(tmp_path, monkeypatch, {"below": BELOW})
    gaps = table.robustness_status.eq(rr.ROBUSTNESS_FEATURE_NOT_EVALUABLE_STATUS)
    index = table.index[gaps][0]
    if tamper == "shape_feature":
        # Claim a shape feature is not evaluable: drop its measured row first
        # so only the reason check can reject it.
        pid, arm = table.loc[index, ["perturbation_id", "extraction_arm"]]
        shape = table.index[(table.perturbation_id == pid) & (table.extraction_arm == arm)
                            & (table.feature_name == SHAPE[1])][0]
        table.loc[index, "feature_name"] = SHAPE[1]
        table = table.drop(index=shape)
    elif tamper == "value":
        table.loc[index, "value"] = 1.0
    elif tamper == "reason":
        table.loc[index, "reason_code"] = "worker_exception"
    elif tamper == "sensitivity_arm":
        table.loc[table.index[gaps], "extraction_arm"] = SENSITIVITY_ARM
    elif tamper == "unapproved_undefined":
        table.loc[index, "reason_code"] = rr.ROBUSTNESS_UNDEFINED_FEATURE_REASON
    else:
        table = pd.concat([table, table.loc[[index]]], ignore_index=True)
    with pytest.raises(RuntimeError):
        rr._validate_extracted_feature_frame(table, set(table.perturbation_id), "Manual/GTVm")


def test_cohort_marks_affected_roi_arm_features_not_evaluable(tmp_path, monkeypatch):
    first, cfg, rob = _course(tmp_path, monkeypatch, patient="P1", plan={"below": BELOW})
    first_output = rr.robustness_for_course(cfg, rob, first)
    second, cfg, rob = _course(tmp_path, monkeypatch, patient="P2")
    second_output = rr.robustness_for_course(cfg, rob, second)
    summary_path = tmp_path / "summary.xlsx"
    rr.aggregate_robustness_results([first_output, second_output], summary_path, rob)
    summary = pd.read_excel(summary_path, sheet_name="per_structure_source")
    affected = summary[(summary.extraction_arm == PRIMARY_ARM)
                       & summary.feature_name.isin(TEXTURE)]
    assert len(affected) == len(TEXTURE)
    assert set(affected.robustness_label) == {"not_evaluable"}
    assert set(affected.cov_status) == set(affected.qcd_status) == {"not_evaluable"}
    assert affected.icc.isna().all() and affected.cov_pct.isna().all()
    assert set(affected.evaluability_reason) == {
        "perturbation_feature_not_evaluable:primary_resegmented_below_minimum_voxels"}
    assert set(affected.n_subjects_not_evaluable) == {1}
    assert set(affected.n_subjects) == {2} and set(affected.n_perturbations) == {4}
    assert not affected.pass_seg_perturb.any()
    unaffected = summary.drop(index=affected.index)
    assert len(unaffected) == len(SHAPE) * 2 + len(TEXTURE)
    assert unaffected.evaluability_reason.isna().all()
    assert unaffected.cov_status.eq("complete").all()
    raw = pd.read_parquet(tmp_path / "summary_raw_values.parquet")
    assert raw.robustness_status.eq(rr.ROBUSTNESS_FEATURE_NOT_EVALUABLE_STATUS).sum() == 6


def test_unaffected_cohort_summary_has_no_evaluability_columns(tmp_path, monkeypatch):
    first, cfg, rob = _course(tmp_path, monkeypatch, patient="P1")
    first_output = rr.robustness_for_course(cfg, rob, first)
    second, cfg, rob = _course(tmp_path, monkeypatch, patient="P2")
    second_output = rr.robustness_for_course(cfg, rob, second)
    frame = pd.concat([pd.read_parquet(first_output), pd.read_parquet(second_output)])
    summary = rr.summarize_feature_stability(frame, rob)
    assert "evaluability_reason" not in summary.columns
    assert "n_subjects_not_evaluable" not in summary.columns
    assert not summary.robustness_label.eq("not_evaluable").any()


def test_not_evaluable_group_requires_a_complete_declared_grid(tmp_path, monkeypatch):
    table, rob = _published(tmp_path, monkeypatch, {"below": BELOW})
    gaps = table.robustness_status.eq(rr.ROBUSTNESS_FEATURE_NOT_EVALUABLE_STATUS)
    with pytest.raises(ValueError, match="incomplete perturbation grid"):
        rr.summarize_feature_stability(
            pd.concat([table.drop(index=table.index[gaps][:1]),
                       table.assign(patient_id="P9")]), rob)


def test_real_pyradiomics_grid_reproduces_and_resolves_production_gap(tmp_path):
    """Default 81-condition NTCV grid, real PyRadiomics 3.0.1, production worker.

    The GTV straddles a bone slab: only the t0_0_-4/c2/v-15 geometry (all
    three noise levels) leaves fewer than minimumROISize voxels in the primary
    HU window, which are the perturbation-arms the production log listed.
    """
    from test_robustness_nonmeasurements import _native_radiomics_payload

    payload = _native_radiomics_payload("primary_below_minimum_grid", tmp_path)
    affected = {"ntcv_t0_0_-4_c2_v-15", "ntcv_n10_t0_0_-4_c2_v-15", "ntcv_n20_t0_0_-4_c2_v-15"}
    assert payload["perturbation_count"] == 81
    assert payload["declared"] == sorted(f"{pid}|{PRIMARY_ARM}" for pid in affected)
    counts = payload["feature_counts"]
    full = counts[f"ntcv_v0|{PRIMARY_ARM}"]
    shape_only = {key for key, n in counts.items() if n != full}
    assert shape_only == {f"{pid}|{PRIMARY_ARM}" for pid in affected}
    gaps = [entry.split("|") for entry in payload["not_evaluable"]]
    assert {(pid, arm) for pid, arm, _, _ in gaps} == {(pid, PRIMARY_ARM) for pid in affected}
    assert {reason for *_, reason in gaps} == {"primary_resegmented_below_minimum_voxels"}
    assert not any("_shape_" in name for _, _, name, _ in gaps)
    assert len(gaps) == len(affected) * (full - counts[f"ntcv_t0_0_-4_c2_v-15|{PRIMARY_ARM}"])
    assert payload["not_evaluable_values_all_nan"] and payload["measured_values_all_finite"]
    assert payload["validation"] == "passed"


def test_real_pyradiomics_undefined_mcc_in_one_condition_is_explicit(tmp_path):
    from test_robustness_nonmeasurements import _native_radiomics_payload

    payload = _native_radiomics_payload("undefined_mcc_one_perturbation", tmp_path)
    assert payload["not_evaluable"] == [
        f"ntcv_n10_v0|{arm}|original_glcm_MCC|extractor_declared_undefined_feature"
        for arm in (PRIMARY_ARM, SENSITIVITY_ARM)
    ]
    assert payload["validation"] == "passed"


def test_declarations_survive_text_encoded_conda_transport():
    """The conda helper serializes NumPy scalars as text (json default=str)."""
    declarations = rr._feature_declarations_from_records([
        {"extraction_arm": PRIMARY_ARM, "extraction_status": "success",
         "intensity_texture_disposition": "below_minimum_dimensions",
         "radiomics_undefined_features_json": "[]", "resegment_after_count": "16",
         "observed_roi_dimensions_after_resegmentation": "1",
         "effective_resegment_lower_hu": "-1000.0", "effective_resegment_upper_hu": 400.0},
        {"extraction_arm": SENSITIVITY_ARM, "extraction_status": "success",
         "intensity_texture_disposition": "success",
         "radiomics_undefined_features_json": "[]"},
    ])
    assert list(declarations) == [PRIMARY_ARM]
    assert declarations[PRIMARY_ARM]["evidence"] == {
        "intensity_texture_disposition": "below_minimum_dimensions",
        "resegment_after_count": 16, "observed_roi_dimensions_after_resegmentation": 1,
        "effective_resegment_lower_hu": -1000.0, "effective_resegment_upper_hu": 400.0,
    }
