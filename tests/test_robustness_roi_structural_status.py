"""A structural status on one ROI must not void a robustness course.

Production case (paths withheld): a CT robustness course ended as
failed_extraction because the custom structure small_bowel1 in RS_custom.dcm
carried ROI_CONTOUR_PARTIALLY_UNPARSEABLE, although the robustness selection
never named it. RS_custom and custom-model RTSTRUCTs were read with no
selection map, so every flagged ROI in them was fatal.

The corrected policy, for every RTSTRUCT source the robustness pass reads:

- an unselected ROI's structural status is recorded as a tolerated failure and
  has no bearing on the course;
- a selected ROI whose source inventory finds its contour data unreadable is a
  governed ``structural_nonmeasurement`` disposition, and the other selected
  ROIs are still measured;
- a technical failure of a selected ROI (its mask cannot be read) still fails
  the course with no published table or sidecar.

Only generated CT/RTSTRUCT files and a stubbed feature extractor are used.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pytest
import yaml

from rtpipeline import cli, custom_models
from rtpipeline import radiomics as rm
from rtpipeline import radiomics_robustness as rr
from rtpipeline import robustness_completion as rc
from rtpipeline.radiomics_outcomes import RadiomicsCourseExtractionError
from rtpipeline.roi_requiredness import inspect_rtstruct
from test_robustness_nonmeasurements import _real_mixed_course


def _add_partially_unparseable_roi(source: Path, target: Path, ct_dir: Path, name: str) -> None:
    """Copy ``source`` to ``target`` with one extra ROI whose contour data is
    only partly readable (one valid slice plus one non-planar item that bounds
    area), the inventory finding the production log reported."""
    from rt_utils import RTStructBuilder

    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=str(ct_dir), rt_struct_path=str(source)
    )
    mask = np.zeros((12, 12, 4), dtype=bool)
    mask[2:5, 2:5, 1:3] = True
    rtstruct.add_roi(mask=mask, name=name)
    rtstruct.save(str(target))
    ds = pydicom.dcmread(str(target))
    names = [r.ROIName for r in ds.StructureSetROISequence]
    numbers = [r.ROINumber for r in ds.StructureSetROISequence]
    item = next(
        i for i in ds.ROIContourSequence
        if names[numbers.index(i.ReferencedROINumber)] == name
    )
    bad = copy.deepcopy(item.ContourSequence[0])
    bad.ContourData = [1.0, 1.0, 0.0, 5.0, 1.0, 3.0, 5.0, 5.0, 0.0, 1.0, 5.0, 4.0]
    bad.NumberOfContourPoints = 4
    item.ContourSequence.append(bad)
    ds.save_as(str(target))
    codes = {o.name: o.structural_code for o in inspect_rtstruct(target).named_rois}
    assert codes[name] == "ROI_CONTOUR_PARTIALLY_UNPARSEABLE"


def _course_with_custom(tmp_path, monkeypatch, *, selection, custom_name="small_bowel1"):
    course, cfg, rob, rs_path = _real_mixed_course(
        tmp_path, monkeypatch, apply_to_structures=tuple(selection)
    )
    _add_partially_unparseable_roi(rs_path, course / "RS_custom.dcm", course / "CT", custom_name)
    extract = rr.extract_features_for_masks

    def features(*args, **kwargs):
        frame = extract(*args, **kwargs)
        frame["run_identifier"] = kwargs["run_identifier"]
        return frame

    monkeypatch.setattr(rr, "extract_features_for_masks", features)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"radiomics_robustness": {
        "enabled": True,
        "segmentation_perturbation": {
            "apply_to_structures": list(selection),
            "small_volume_changes": [0., .15],
            "max_translation_mm": 0.,
            "n_random_contour_realizations": 0,
            "noise_levels": [0.],
        },
    }}))
    return course, cfg, rob, rs_path, config


def _run_cli(course, config):
    output = course / "radiomics_robustness_ct.parquet"
    sentinel = rc.robustness_completion_sentinel_path(course)
    code = cli.main(["radiomics-robustness", "--course-dir", str(course),
                     "--config", str(config), "--output", str(output),
                     "--sentinel", str(sentinel), "--campaign-mode"])
    return code, output, sentinel


def test_unselected_custom_structural_status_does_not_void_course(tmp_path, monkeypatch):
    """Reproduces the production failure; on f16d72a the course failed."""
    course, _, rob, _, config = _course_with_custom(
        tmp_path, monkeypatch, selection=["GTV*", "ROI"]
    )
    code, output, sentinel = _run_cli(course, config)
    assert code == 0
    receipt = rc.read_robustness_completion_sentinel(sentinel)
    assert receipt.measurement_outcome == rr.ROBUSTNESS_MEASURED_OUTCOME
    table = pd.read_parquet(output)
    assert set(table.structure) == {"ROI"}
    assert table.robustness_status.eq("measured").all()
    payload = json.loads(rr.robustness_source_dispositions_path(course).read_text())
    tolerated = [
        (t["segmentation_source"], t["roi_name"], t["structural_code"])
        for t in payload["tolerated_source_failures"]
    ]
    assert ("Custom", "small_bowel1", "ROI_CONTOUR_PARTIALLY_UNPARSEABLE") in tolerated
    # Tolerated evidence is not a disposition and is never admitted as one.
    admitted = rr.admit_robustness_cohort_course(
        course, patient_id="P", course_id="C", rob_config=rob
    )
    assert all(row["roi_name"] != "small_bowel1" for row in admitted.source_dispositions)


def test_selected_structural_status_is_a_governed_disposition(tmp_path, monkeypatch):
    course, _, rob, _, config = _course_with_custom(
        tmp_path, monkeypatch, selection=["ROI", "small_bowel*"]
    )
    code, output, sentinel = _run_cli(course, config)
    assert code == 0
    receipt = rc.read_robustness_completion_sentinel(sentinel)
    # Marker1 (a point ROI) is present in RS.dcm and in its RS_custom copy.
    assert receipt.measured and receipt.source_disposition_count == 3
    admitted = rr.admit_robustness_cohort_course(
        course, patient_id="P", course_id="C", rob_config=rob
    )
    row = next(r for r in admitted.source_dispositions if r["roi_name"] == "small_bowel1")
    assert (row["segmentation_source"], row["status"], row["failure_kind"],
            row["structural_code"]) == (
        "Custom", "structural_nonmeasurement", "unmeasurable_source_contour",
        "ROI_CONTOUR_PARTIALLY_UNPARSEABLE",
    )
    assert row["rtstruct_sop_instance_uid"] == str(
        pydicom.dcmread(course / "RS_custom.dcm").SOPInstanceUID
    )
    assert set(pd.read_parquet(output).structure) == {"ROI"}
    assert rr._admit_robustness_aggregation_input(output, rob).source_dispositions == (
        admitted.source_dispositions
    )


def test_selected_manual_structural_status_is_a_governed_disposition(tmp_path, monkeypatch):
    """The Manual source follows the same policy (previously fatal)."""
    course, cfg, rob, _ = _real_mixed_course(
        tmp_path, monkeypatch, add_bad_roi=True, apply_to_structures=("ROI", "Bad")
    )
    code = {o.name: o.structural_code for o in inspect_rtstruct(course / "RS.dcm").named_rois}
    assert code["Bad"] == "ROI_CONTOUR_UNPARSEABLE"
    result = rr.robustness_for_course(cfg, rob, course)
    assert result is not None and set(pd.read_parquet(result).structure) == {"ROI"}
    payload = json.loads(rr.robustness_source_dispositions_path(course).read_text())
    row = next(r for r in payload["rows"] if r["roi_name"] == "Bad")
    assert (row["status"], row["failure_kind"], row["structural_code"]) == (
        "structural_nonmeasurement", "unmeasurable_source_contour", "ROI_CONTOUR_UNPARSEABLE",
    )
    assert not payload["tolerated_source_failures"]


def test_only_selected_roi_unreadable_closes_as_source_only(tmp_path, monkeypatch):
    course, _, rob, _, config = _course_with_custom(
        tmp_path, monkeypatch, selection=["small_bowel*"]
    )
    code, output, sentinel = _run_cli(course, config)
    assert code == 0
    receipt = rc.read_robustness_completion_sentinel(sentinel)
    assert receipt.measurement_outcome == rr.ROBUSTNESS_SOURCE_ONLY_OUTCOME
    assert not output.exists()
    payload = json.loads(rr.robustness_source_dispositions_path(course).read_text())
    assert "contour data its source inventory cannot read" in payload["source_only_basis"]["reason"]


def test_selected_technical_failure_stays_fail_closed(tmp_path, monkeypatch):
    """An unreadable mask of a selected ROI is technical: the course fails."""
    course, _, _, _, config = _course_with_custom(
        tmp_path, monkeypatch, selection=["ROI"]
    )
    from rt_utils import RTStruct

    real = RTStruct.get_roi_mask_by_name

    def failing(self, name):
        if name == "ROI":
            raise RuntimeError("synthetic rasterizer failure")
        return real(self, name)

    monkeypatch.setattr(RTStruct, "get_roi_mask_by_name", failing)
    code, output, sentinel = _run_cli(course, config)
    assert code == 1
    receipt = rc.read_robustness_completion_sentinel(sentinel)
    assert receipt.measurement_outcome == rr.ROBUSTNESS_FAILED_OUTCOME
    assert not output.exists()


def test_selected_technical_failure_raises_directly(tmp_path, monkeypatch):
    course, cfg, rob, _ = _real_mixed_course(tmp_path, monkeypatch)
    from rt_utils import RTStruct

    def failing(self, name):
        raise RuntimeError("synthetic rasterizer failure")

    monkeypatch.setattr(RTStruct, "get_roi_mask_by_name", failing)
    with pytest.raises(RadiomicsCourseExtractionError, match="synthetic rasterizer failure"):
        rr.robustness_for_course(cfg, rob, course)
    assert not (course / "radiomics_robustness_ct.parquet").exists()
    assert not rr.robustness_source_dispositions_path(course).exists()


def test_unselected_custom_model_structural_status_is_tolerated(tmp_path, monkeypatch):
    course, cfg, rob, rs_path = _real_mixed_course(tmp_path, monkeypatch)
    model_dir = course / "Segmentation_CustomModels" / "ModelA"
    model_dir.mkdir(parents=True)
    _add_partially_unparseable_roi(rs_path, model_dir / "rtstruct.dcm", course / "CT", "liver")
    monkeypatch.setattr(custom_models, "list_custom_model_outputs",
                        lambda *a: [("ModelA", model_dir)])
    result = rr.robustness_for_course(cfg, rob, course)
    assert result is not None
    payload = json.loads(rr.robustness_source_dispositions_path(course).read_text())
    assert ("CustomModel:ModelA", "liver") in {
        (t["segmentation_source"], t["roi_name"]) for t in payload["tolerated_source_failures"]
    }


def test_uninspectable_custom_source_keeps_zero_tolerance(tmp_path, monkeypatch):
    """Without a selection map the old fail-closed read is unchanged."""
    course, cfg, rob, _, _ = _course_with_custom(
        tmp_path, monkeypatch, selection=["ROI"]
    )
    monkeypatch.setattr(rr, "_robustness_selection_requiredness", lambda *a, **k: None)
    with pytest.raises(RadiomicsCourseExtractionError, match="PARTIALLY_UNPARSEABLE"):
        rr.robustness_for_course(cfg, rob, course)
    assert not rr.robustness_source_dispositions_path(course).exists()


def test_main_radiomics_reader_default_is_unchanged(tmp_path, monkeypatch):
    """The opt-in flag is robustness-only; the default reader still raises."""
    course, _, _, rs_path, _ = _course_with_custom(
        tmp_path, monkeypatch, selection=["ROI"]
    )
    requiredness = rr._robustness_selection_requiredness(
        course / "RS_custom.dcm", ["small_bowel*"]
    )
    with pytest.raises(RadiomicsCourseExtractionError, match="PARTIALLY_UNPARSEABLE"):
        rm._rtstruct_masks(course / "CT", course / "RS_custom.dcm", failure_outcomes=[],
                           tolerate_unselected=True, requiredness_by_roi=requiredness)


@pytest.mark.parametrize("failure_kind,structural_code", [
    ("unmeasurable_source_contour", "ROI_DECLARED_NO_CONTOUR_ITEM"),
    ("unmeasurable_source_contour", "ROI_EXTRACTION_FAILED"),
    ("unmeasurable_source_contour", "ROI_MASK_EMPTY_AFTER_RASTERIZATION"),
    ("declared_without_contour_data", "ROI_CONTOUR_UNPARSEABLE"),
    ("structural_roi_error", "ROI_CONTOUR_PARTIALLY_UNPARSEABLE"),
    # 2026-09-24: the scoped-reader kind admits only the two scope codes.
    ("unresolved_source_scope", "ROI_CONTOUR_PARTIALLY_UNPARSEABLE"),
    ("unresolved_source_scope", "ROI_EXTRACTION_FAILED"),
    ("unmeasurable_source_contour", "ROI_UNRESOLVED_SOURCE_SCOPE"),
])
def test_forged_structural_dispositions_are_rejected(
    tmp_path, monkeypatch, failure_kind, structural_code
):
    course, _, rob, _, config = _course_with_custom(
        tmp_path, monkeypatch, selection=["ROI", "small_bowel*"]
    )
    assert _run_cli(course, config)[0] == 0
    path = rr.robustness_source_dispositions_path(course)
    payload = json.loads(path.read_text())
    row = next(r for r in payload["rows"] if r["roi_name"] == "small_bowel1")
    row["failure_kind"], row["structural_code"] = failure_kind, structural_code
    payload["row_count"] = len(payload["rows"])
    payload["rows_sha256"] = rr._content_sha256(payload["rows"])
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        rr.load_robustness_source_dispositions(
            course, run_identifier=payload["robustness_run_identifier"], rob_config=rob
        )
