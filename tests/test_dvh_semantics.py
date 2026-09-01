from __future__ import annotations

from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from rtpipeline.dvh import (
    RELATIVE_DVH_METRIC_COLUMNS,
    _resample_mask_to_ct_if_equivalent,
    annotate_dvh_metrics,
    classify_zero_dose_roi_geometry,
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
