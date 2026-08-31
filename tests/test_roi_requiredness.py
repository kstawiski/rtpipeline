from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pydicom
from pydicom.dataset import Dataset

from course_contract_test_utils import write_synthetic_rtstruct
from rtpipeline import radiomics
from rtpipeline import radiomics_parallel
from rtpipeline.roi_requiredness import (
    CUSTOM_DEPENDENCY_GRAPH,
    TAXONOMY_CODES,
    DenominatorLedger,
    RequiredROI,
    Requiredness,
    assess_custom_applicability,
    classify_rasterized_mask,
    inspect_rtstruct,
    match_requirements,
    requiredness_for,
    taxonomy_is_fatal,
    write_modality_ledger,
)


def _declared(number, name="ROI"):
    item = Dataset()
    item.ROINumber = number
    item.ROIName = name
    return item


def _contour_item(number, contours=None):
    item = Dataset()
    item.ReferencedROINumber = number
    if contours is not None:
        item.ContourSequence = []
        for values in contours:
            contour = Dataset()
            contour.ContourData = values
            item.ContourSequence.append(contour)
    return item


def _inventory(dataset):
    return inspect_rtstruct(None, dataset)


def test_structural_taxonomy_covers_all_codes_and_parser_uses_sequence_presence():
    assert {
        "ROI_DECLARED_NO_CONTOUR_ITEM",
        "ROI_DECLARED_EMPTY_CONTOUR_SEQUENCE",
        "ROI_CONTOUR_UNPARSEABLE",
        "ROI_CONTOUR_PARTIALLY_UNPARSEABLE",
        "ROI_CONTOUR_ORPHAN_REFERENCE",
        "RTSTRUCT_NO_NAMED_ROIS",
    }.issubset(TAXONOMY_CODES)

    assert "RTSTRUCT_NO_NAMED_ROIS" in _inventory(Dataset()).structural_codes

    dataset = Dataset()
    dataset.StructureSetROISequence = [_declared(1)]
    assert "ROI_DECLARED_NO_CONTOUR_ITEM" in _inventory(dataset).structural_codes

    dataset.ROIContourSequence = [_contour_item(1, [])]
    assert "ROI_DECLARED_EMPTY_CONTOUR_SEQUENCE" in _inventory(dataset).structural_codes

    dataset.ROIContourSequence = [_contour_item(1, [[1, 2]])]
    assert "ROI_CONTOUR_UNPARSEABLE" in _inventory(dataset).structural_codes

    dataset.ROIContourSequence = [
        _contour_item(1, [[0, 0, 0, 1, 1, 1], [1, 2]])
    ]
    assert "ROI_CONTOUR_PARTIALLY_UNPARSEABLE" in _inventory(dataset).structural_codes

    dataset.ROIContourSequence = [_contour_item(99, [[0, 0, 0, 1, 1, 1]])]
    assert "ROI_CONTOUR_ORPHAN_REFERENCE" in _inventory(dataset).structural_codes


def test_serial_mask_path_skips_declared_uncontoured_inventory_roi(tmp_path):
    path = write_synthetic_rtstruct(tmp_path / "RS.dcm", roi_names=("TemplateEmpty",))
    dataset = pydicom.dcmread(path)
    del dataset.ROIContourSequence
    dataset.save_as(path, write_like_original=False)

    failures = []
    masks = radiomics._rtstruct_masks(
        tmp_path / "ct",
        path,
        best_effort=True,
        failure_outcomes=failures,
        requiredness_by_roi={"TemplateEmpty": Requiredness.INVENTORY_ONLY},
    )

    assert masks == {}
    assert failures == []


def test_required_matching_is_name_alias_based_but_identity_is_roi_number():
    empty = _inventory(Dataset())
    missing = match_requirements(empty, [RequiredROI("PlanningTarget")])[0]
    assert missing.structural_code == "REQUIRED_ROI_NOT_DECLARED"

    dataset = Dataset()
    dataset.StructureSetROISequence = [_declared(7, "PTV"), _declared(9, "ptv")]
    ambiguous = match_requirements(
        _inventory(dataset), [RequiredROI("PlanningTarget", ("PTV",))]
    )[0]
    assert ambiguous.structural_code == "REQUIRED_ROI_AMBIGUOUS_MATCH"

    dataset.ROIContourSequence = [_contour_item(9, [[0, 0, 0, 1, 1, 1]])]
    dataset.StructureSetROISequence = [_declared(9, "ptv")]
    matched = match_requirements(
        _inventory(dataset), [RequiredROI("PlanningTarget", ("PTV",))]
    )[0]
    assert matched.observation.roi_number == 9
    assert matched.structural_code is None


def test_requiredness_defaults_declared_manual_to_inventory_only():
    contract = {
        "CT": {
            "required_rois": [{"canonical_name": "PlanningTarget", "approved_aliases": ["PTV"]}],
            "optional_rois": ["Body"],
        }
    }
    assert requiredness_for("Manual", "TemplateEmpty", contract=contract) is Requiredness.INVENTORY_ONLY
    assert requiredness_for("Manual", "PTV", contract=contract) is Requiredness.ANALYSIS_REQUIRED
    assert requiredness_for("AutoRTS_total", "Body", contract=contract) is Requiredness.ANALYSIS_OPTIONAL
    assert not taxonomy_is_fatal("ROI_DECLARED_NO_CONTOUR_ITEM", Requiredness.INVENTORY_ONLY)
    assert taxonomy_is_fatal("ROI_DECLARED_NO_CONTOUR_ITEM", Requiredness.ANALYSIS_REQUIRED)


def test_mask_taxonomy_codes_are_controlled():
    assert classify_rasterized_mask(np.zeros((2, 2))) == "ROI_MASK_EMPTY_AFTER_RASTERIZATION"
    assert classify_rasterized_mask(np.ones((2, 2)), minimum_voxels=5) == "ROI_MASK_BELOW_MIN_VOXELS"
    assert classify_rasterized_mask(object()) == "ROI_EXTRACTION_FAILED"


def test_custom_roi_anatomy_branch_is_not_applicable():
    dependencies = CUSTOM_DEPENDENCY_GRAPH["iliac_vess"]
    result = assess_custom_applicability(
        "iliac_vess",
        {dependency: "empty" for dependency in dependencies},
        {"excluded_regions": ["pelvis"]},
    )
    assert result.reason_code == "not_applicable_anatomy"
    assert not result.fatal


def test_custom_roi_readable_dependencies_missing_derived_is_generation_defect():
    dependencies = CUSTOM_DEPENDENCY_GRAPH["iliac_vess"]
    result = assess_custom_applicability(
        "iliac_vess",
        {dependency: {"readable": True, "non_empty": True} for dependency in dependencies},
        {"contains_regions": ["pelvis"]},
    )
    assert result.reason_code == "failed_custom_generation"
    assert result.fatal


def test_custom_roi_conflicting_evidence_fails_closed():
    result = assess_custom_applicability(
        "pelvic_bones",
        {
            "sacrum": "readable_nonempty",
            "hip_left": "unreadable",
            "hip_right": "empty",
            "vertebrae_S1": "empty",
        },
        {"contains_regions": ["pelvis"]},
    )
    assert result.reason_code == "indeterminate_applicability"
    assert result.fatal


def test_denominator_ledger_keeps_course_patient_and_course_roi_rows():
    ledger = DenominatorLedger()
    ledger.expect_course_roi("C1", "iliac_vess")
    ledger.record_roi(
        "C1", "P1", "iliac_vess", reason_code="not_applicable_anatomy"
    )
    ledger.record_course(
        "C1",
        "P1",
        screened=True,
        in_scope=False,
        out_of_scope=True,
        adequate_coverage=False,
        insufficient_coverage=False,
        valid_derivation=False,
        technical_exclusion=False,
        indeterminate=False,
        extracted=False,
    )
    summary = ledger.summary()
    assert summary["COURSE"]["screened"] == 1
    assert summary["PATIENT"]["out_of_scope"] == 1
    assert summary["COURSE_ROI"]["iliac_vess"]["excluded_anatomy"] == 1


def test_modality_ledgers_merge_without_overwriting_course_or_patient_denominators(tmp_path):
    metadata = tmp_path / "metadata"
    ct = DenominatorLedger()
    ct.expect_course_roi("C1", "PTV")
    ct.record_roi("C1", "P1", "PTV", reason_code="extracted", disposition="extracted")
    ct.record_course("C1", "P1", screened=True, in_scope=True, out_of_scope=False, adequate_coverage=True, insufficient_coverage=False, valid_derivation=False, technical_exclusion=False, indeterminate=False, extracted=True)
    write_modality_ledger(metadata, ct, "CT")

    mr = DenominatorLedger()
    mr.expect_course_roi("C1", "PTV")
    mr.record_roi("C1", "P1", "PTV", reason_code="not_computed_valid_empty_scope", disposition="excluded")
    mr.record_course("C1", "P1", screened=True, in_scope=True, out_of_scope=False, adequate_coverage=False, insufficient_coverage=True, valid_derivation=False, technical_exclusion=False, indeterminate=False, extracted=False)
    write_modality_ledger(metadata, mr, "MR")

    combined = json.loads((metadata / "radiomics_roi_ledger.json").read_text(encoding="utf-8"))
    assert len(combined["course"]) == 1
    assert len(combined["course_roi"]) == 2
    assert {row["modality"] for row in combined["course_roi"]} == {"CT", "MR"}
    patients = json.loads((metadata / "radiomics_patient_ledger.json").read_text(encoding="utf-8"))
    assert len(patients) == 1
    assert patients[0]["course_count"] == 1
    summary = json.loads((metadata / "radiomics_denominators.json").read_text(encoding="utf-8"))
    assert summary["COURSE"]["screened"] == 1
    assert summary["PATIENT"]["screened"] == 1
    assert summary["COURSE_ROI"]["CT:PTV"]["extracted"] == 1
    assert summary["COURSE_ROI"]["MR:PTV"]["excluded_anatomy"] == 1


def test_parallel_ledger_records_configured_roi_realized_by_partial_alias(tmp_path):
    course_dir = tmp_path / "442568" / "2021-07"
    course_dir.mkdir(parents=True)
    task = SimpleNamespace(roi_name="bowel_bag__partial")
    rows = [
        {
            "roi_original_name": "bowel_bag__partial",
            "roi_name": "bowel_bag__partial",
            "extraction_status": "success",
        }
    ]
    applicability = [
        assess_custom_applicability(
            "bowel_bag",
            {},
            generated_state="readable_nonempty",
        )
    ]

    radiomics_parallel._write_parallel_roi_ledger(
        course_dir,
        [task],  # type: ignore[list-item]
        rows,
        applicability,
        extracted=True,
    )

    payload = json.loads(
        (course_dir / "metadata" / "radiomics_roi_ledger.json").read_text(
            encoding="utf-8"
        )
    )
    recorded = {
        row["roi_name"]: row
        for row in payload["course_roi"]
        if row["course_id"] == "2021-07"
    }
    assert recorded["bowel_bag__partial"]["reason_code"] == "extracted"
    assert recorded["bowel_bag"]["reason_code"] == "extracted"
    assert "bowel_bag__partial" in recorded["bowel_bag"]["detail"]


def test_every_declared_taxonomy_code_is_registered():
    assert TAXONOMY_CODES == frozenset(
        {
            "ROI_DECLARED_NO_CONTOUR_ITEM",
            "ROI_DECLARED_EMPTY_CONTOUR_SEQUENCE",
            "ROI_CONTOUR_UNPARSEABLE",
            "ROI_CONTOUR_PARTIALLY_UNPARSEABLE",
            "ROI_CONTOUR_ORPHAN_REFERENCE",
            "ROI_MASK_EMPTY_AFTER_RASTERIZATION",
            "ROI_MASK_BELOW_MIN_VOXELS",
            "REQUIRED_ROI_NOT_DECLARED",
            "REQUIRED_ROI_AMBIGUOUS_MATCH",
            "ROI_EXTRACTION_FAILED",
            "RTSTRUCT_NO_NAMED_ROIS",
        }
    )
