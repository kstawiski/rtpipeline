from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from rtpipeline.dvh import (
    RELATIVE_DVH_METRIC_COLUMNS,
    _near_zero_dose_geometry_qc,
    _resample_mask_to_ct_if_equivalent,
    annotate_dvh_metrics,
    classify_dvh_dose_plan_scope,
    classify_zero_dose_roi_geometry,
    summarize_plan_isocenter_positions,
    summarize_target_near_zero_rows,
)


def _rtstruct_with_contour(points: list[float]) -> Dataset:
    structure = Dataset()
    structure.ROINumber = 1
    structure.ROIName = "BLADDER"
    contour = Dataset()
    contour.ReferencedROINumber = 1
    contour_sequence = Dataset()
    contour_sequence.ContourData = points
    contour.ContourSequence = Sequence([contour_sequence])
    rtstruct = Dataset()
    rtstruct.StructureSetROISequence = Sequence([structure])
    rtstruct.ROIContourSequence = Sequence([contour])
    return rtstruct


def _dose_grid() -> Dataset:
    dose = Dataset()
    dose.Rows = 10
    dose.Columns = 10
    dose.NumberOfFrames = 3
    dose.PixelSpacing = [1.0, 1.0]
    dose.ImagePositionPatient = [0.0, 0.0, 0.0]
    dose.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dose.GridFrameOffsetVector = [0.0, 1.0, 2.0]
    return dose


def _rtstruct_from_boxes(boxes: list[dict]) -> Dataset:
    structures = []
    contours = []
    observations = []
    for box in boxes:
        number = int(box["roi_number"])
        name = str(box["roi_name"])
        lower = np.asarray(box["translated_min_mm"], dtype=float)
        upper = np.asarray(box["translated_max_mm"], dtype=float)
        structure = Dataset()
        structure.ROINumber = number
        structure.ROIName = name
        structures.append(structure)

        roi_contour = Dataset()
        roi_contour.ReferencedROINumber = number
        contour_items = []
        for z_value in (lower[2], upper[2]):
            contour = Dataset()
            contour.ContourData = [
                lower[0], lower[1], z_value,
                upper[0], lower[1], z_value,
                upper[0], upper[1], z_value,
                lower[0], upper[1], z_value,
            ]
            contour_items.append(contour)
        roi_contour.ContourSequence = Sequence(contour_items)
        contours.append(roi_contour)

        observation = Dataset()
        observation.ReferencedROINumber = number
        observation.RTROIInterpretedType = str(box["interpreted_type"])
        observations.append(observation)

    rtstruct = Dataset()
    rtstruct.StructureSetROISequence = Sequence(structures)
    rtstruct.ROIContourSequence = Sequence(contours)
    rtstruct.RTROIObservationsSequence = Sequence(observations)
    return rtstruct


def _dose_from_task18_fixture(payload: dict) -> Dataset:
    grid = payload["dose_grid"]
    dose = Dataset()
    dose.Rows = int(grid["rows"])
    dose.Columns = int(grid["columns"])
    dose.NumberOfFrames = int(grid["number_of_frames"])
    dose.PixelSpacing = grid["pixel_spacing_mm"]
    dose.ImagePositionPatient = grid["image_position_patient"]
    dose.ImageOrientationPatient = grid["image_orientation_patient"]
    dose.GridFrameOffsetVector = np.arange(
        grid["frame_offset_start_mm"],
        grid["frame_offset_stop_mm"] + grid["frame_offset_step_mm"] / 2.0,
        grid["frame_offset_step_mm"],
    ).tolist()
    return dose


def test_brachy_relative_metrics_are_suppressed_with_a_reason() -> None:
    metrics = {name: 1.0 for name in RELATIVE_DVH_METRIC_COLUMNS}
    metrics["HI%"] = 2.0

    annotated = annotate_dvh_metrics(
        metrics,
        technique="BRACHYTHERAPY",
        structure_name="PTV",
        rtstruct_sop_instance_uid="1.2.3",
        rtstruct_path=Path("RS.dcm"),
        zero_dose_status="not_zero",
        zero_dose_reason="DmaxGy is positive.",
    )

    assert all(annotated[column] is None for column in RELATIVE_DVH_METRIC_COLUMNS)
    assert annotated["treatment_technique"] == "BRACHYTHERAPY"
    assert annotated["relative_metric_status"] == "suppressed_non_ebrt"
    assert annotated["relative_metric_reason"]
    assert annotated["rtstruct_provenance_status"] == "traceable"
    assert annotated["rtstruct_sop_instance_uid"] == "1.2.3"


def test_dose_response_ineligible_course_suppresses_relative_endpoints() -> None:
    metrics = {name: 1.0 for name in RELATIVE_DVH_METRIC_COLUMNS}
    metrics["HI%"] = 2.0

    annotated = annotate_dvh_metrics(
        metrics,
        technique="EBRT",
        structure_name="PTV",
        prescription_resolved=False,
        dose_response_eligible=False,
        rtstruct_sop_instance_uid="1.2.3",
        rtstruct_path=Path("RS.dcm"),
    )

    assert annotated["dose_response_eligible"] is False
    assert all(annotated[column] is None for column in RELATIVE_DVH_METRIC_COLUMNS)
    assert annotated["relative_metric_status"] == (
        "excluded_dose_response_ineligible"
    )
    assert annotated["HI_status"] == "excluded_dose_response_ineligible"


def test_non_target_homogeneity_is_explicitly_not_applicable() -> None:
    metrics = {"HI%": 3.0}

    annotated = annotate_dvh_metrics(
        metrics,
        technique="EBRT",
        structure_name="BLADDER",
        prescription_resolved=True,
        rtstruct_sop_instance_uid=None,
        rtstruct_path=None,
        structure_provenance_type="NIFTI_MASK",
        structure_provenance_path=Path("Segmentation/bladder.nii.gz"),
        zero_dose_status="not_zero",
        zero_dose_reason="DmaxGy is positive.",
    )

    assert annotated["HI%"] is None
    assert annotated["HI_status"] == "not_applicable_non_target"
    assert annotated["HI_reason"]
    assert annotated["structure_provenance_status"] == "traceable"
    assert annotated["structure_provenance_path"] == "Segmentation/bladder.nii.gz"
    assert annotated["rtstruct_provenance_status"] == "not_applicable"


def test_zero_dose_geometry_distinguishes_inside_and_outside_grid() -> None:
    inside = _rtstruct_with_contour([2, 2, 1, 4, 2, 1, 4, 4, 1, 2, 4, 1])
    outside = _rtstruct_with_contour([20, 20, 20, 21, 20, 20, 21, 21, 20, 20, 21, 20])

    assert classify_zero_dose_roi_geometry(inside, 1, _dose_grid())["status"] == (
        "zero_dose_in_grid"
    )
    assert classify_zero_dose_roi_geometry(outside, 1, _dose_grid())["status"] == (
        "zero_dose_outside_dose_grid"
    )


def test_zero_dose_geometry_is_unresolved_without_contours() -> None:
    rtstruct = _rtstruct_with_contour([])
    # Keep a contour item but omit ContourData so geometry cannot be inferred.
    rtstruct.ROIContourSequence[0].ContourSequence[0].ContourData = []

    result = classify_zero_dose_roi_geometry(rtstruct, 1, _dose_grid())

    assert result["status"] == "zero_dose_geometry_unresolved"
    assert result["reason"]


def test_zero_dose_geometry_requires_every_contour_item_to_be_valid() -> None:
    rtstruct = _rtstruct_with_contour(
        [1, 1, 1, 4, 1, 1, 4, 4, 1, 1, 4, 1]
    )
    rtstruct.ROIContourSequence[0].ContourSequence.append(Dataset())

    result = classify_zero_dose_roi_geometry(rtstruct, 1, _dose_grid())

    assert result["status"] == "zero_dose_geometry_unresolved"
    assert "complete contour set" in result["reason"]


def test_zero_dose_geometry_identifies_partial_grid_coverage() -> None:
    partial = _rtstruct_with_contour(
        [-2, 2, 1, 4, 2, 1, 4, 4, 1, -2, 4, 1]
    )

    result = classify_zero_dose_roi_geometry(partial, 1, _dose_grid())

    assert result["status"] == "zero_dose_partly_inside_dose_grid"


def test_oblique_contour_bbox_overlap_without_polygon_overlap_is_outside() -> None:
    outside = _rtstruct_with_contour(
        [5, 20, 0, 20, 5, 2, 20, 20, 1]
    )

    result = classify_zero_dose_roi_geometry(outside, 1, _dose_grid())

    assert result["status"] == "zero_dose_outside_dose_grid"


def test_outside_grid_dose_values_become_null_nonmeasurements() -> None:
    metrics = {
        "Volume (cm³)": 12.0,
        "DmeanGy": 0.0,
        "DmaxGy": 0.0,
        "D95Gy": 0.0,
        "D95%": 0.0,
        "V1Gy (cm³)": 0.0,
        "IntegralDose_Gycm3": 0.0,
    }

    annotated = annotate_dvh_metrics(
        metrics,
        technique="EBRT",
        structure_name="PTV 1",
        structure_interpreted_type="PTV",
        prescription_resolved=True,
        dose_response_eligible=True,
        rtstruct_sop_instance_uid=None,
        rtstruct_path=None,
        zero_dose_status="zero_dose_outside_dose_grid",
        zero_dose_reason="Outside grid.",
        zero_dose_trigger_metric="DmaxGy",
        zero_dose_trigger_value_gy=0.0,
    )

    assert annotated["Volume (cm³)"] == 12.0
    assert annotated["DmeanGy"] is None
    assert annotated["DmaxGy"] is None
    assert annotated["D95Gy"] is None
    assert annotated["D95%"] is None
    assert annotated["V1Gy (cm³)"] is None
    assert annotated["IntegralDose_Gycm3"] is None
    assert annotated["zero_dose_trigger_value_gy"] == 0.0
    assert annotated["dose_metric_status"] == "not_measurable_outside_dose_grid"
    assert annotated["dose_metric_usable_for_dose_response"] is False


def test_actual_two_site_geometry_keeps_selected_plan_near_zero_visible() -> None:
    fixture_path = Path(__file__).parent / "data" / "task18_two_site_geometry.toml"
    payload = tomllib.loads(fixture_path.read_text(encoding="utf-8"))
    dose = _dose_from_task18_fixture(payload)
    rtstruct = _rtstruct_from_boxes(payload["targets"])
    targets = {item["roi_number"]: item for item in payload["targets"]}

    ptv1_qc = _near_zero_dose_geometry_qc(
        {
            "D95Gy": targets[11]["selected_plan_d95_gy"],
            "DmaxGy": targets[11]["selected_plan_dmax_gy"],
        },
        target_like=True,
        rtstruct_ds=rtstruct,
        roi_number=11,
        dose_ds=dose,
    )
    ptv2_geometry = classify_zero_dose_roi_geometry(rtstruct, 12, dose)

    assert ptv1_qc["status"] == "zero_dose_in_grid"
    assert ptv1_qc["trigger_metric"] == "D95Gy"
    assert ptv2_geometry["status"] == "zero_dose_in_grid"

    annotated = annotate_dvh_metrics(
        {
            "D95Gy": targets[11]["selected_plan_d95_gy"],
            "DmaxGy": targets[11]["selected_plan_dmax_gy"],
        },
        technique="EBRT",
        structure_name="PTV 1",
        structure_interpreted_type="PTV",
        rtstruct_sop_instance_uid=None,
        rtstruct_path=None,
        zero_dose_status=str(ptv1_qc["status"]),
        zero_dose_reason=str(ptv1_qc["reason"]),
        zero_dose_trigger_metric=str(ptv1_qc["trigger_metric"]),
        zero_dose_trigger_value_gy=float(ptv1_qc["trigger_value_gy"]),
    )
    assert annotated["D95Gy"] == 0.01
    assert annotated["dose_metric_status"] == "computed_in_grid_near_zero"
    assert annotated["dose_metric_usable_for_dose_response"] is True

    centers = [
        (np.asarray(item["translated_min_mm"]) + np.asarray(item["translated_max_mm"]))
        / 2.0
        for item in payload["targets"]
    ]
    assert np.linalg.norm(centers[0] - centers[1]) > 200.0

    plan_isocenters = payload["plan_isocenters"]
    isocenter_summary = summarize_plan_isocenter_positions(
        {
            "plan_one": plan_isocenters["plan_one"],
            "plan_two": plan_isocenters["plan_two"],
        },
        expected_plan_count=2,
    )
    assert isocenter_summary.status == "multiple_plan_isocenters"
    assert isocenter_summary.isocenter_count == 2
    assert isocenter_summary.max_separation_mm is not None
    assert round(isocenter_summary.max_separation_mm, 3) == plan_isocenters[
        "measured_max_separation_mm"
    ]
    assert "does not by itself establish" in isocenter_summary.reason


def test_incomplete_multi_plan_scope_excludes_course_dose_response() -> None:
    contract = SimpleNamespace(
        delivery={
            "per_plan": [
                {"plan_sop_uid": "1.2.plan-one"},
                {"plan_sop_uid": "1.2.plan-two"},
            ]
        },
        selected_plans=[],
    )

    scope = classify_dvh_dose_plan_scope(contract, ["1.2.plan-two"])

    assert scope.status == "partial_course_plan_set"
    assert scope.complete is False
    assert scope.course_treatment_plan_count == 2
    assert scope.dose_grid_plan_count == 1
    assert scope.unrepresented_treatment_plan_uids == ("1.2.plan-one",)

    mismatch = classify_dvh_dose_plan_scope(contract, ["1.2.unrelated"])
    assert mismatch.status == "dose_grid_plan_set_mismatch"
    assert mismatch.dose_grid_plan_count == 0
    assert set(mismatch.unrepresented_treatment_plan_uids) == {
        "1.2.plan-one",
        "1.2.plan-two",
    }


def test_target_near_zero_summary_uses_mutually_exclusive_row_classes() -> None:
    summary = summarize_target_near_zero_rows(
        [
            {"target_like": True, "zero_dose_status": "zero_dose_in_grid"},
            {
                "target_like": True,
                "zero_dose_status": "zero_dose_outside_dose_grid",
            },
            {
                "target_like": False,
                "zero_dose_status": "zero_dose_outside_dose_grid",
            },
        ]
    )

    assert summary["near_zero_target_row_count"] == 2
    assert summary["inside_dose_grid_count"] == 1
    assert summary["outside_dose_grid_count"] == 1
    assert summary["partly_inside_dose_grid_count"] == 0
    assert summary["geometry_unresolved_count"] == 0


def test_nifti_mask_resampling_rejects_translated_grid() -> None:
    import SimpleITK as sitk

    reference = sitk.Image([8, 8, 8], sitk.sitkUInt8)
    mask = sitk.Image([8, 8, 8], sitk.sitkUInt8)
    mask.SetOrigin((100.0, 0.0, 0.0))

    assert _resample_mask_to_ct_if_equivalent(mask, reference) is None


def test_nifti_mask_resampling_accepts_equivalent_grid() -> None:
    import SimpleITK as sitk

    reference = sitk.Image([8, 8, 8], sitk.sitkUInt8)
    mask = sitk.Image([8, 8, 8], sitk.sitkUInt8)

    rebound = _resample_mask_to_ct_if_equivalent(mask, reference)

    assert rebound is not None
    assert rebound.GetSize() == reference.GetSize()
