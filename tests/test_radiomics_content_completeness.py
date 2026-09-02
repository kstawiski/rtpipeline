from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rtpipeline import radiomics_ct_contract as contract
from rtpipeline import radiomics_conda, radiomics_parallel


def _rows(*, required: bool = False) -> list[dict[str, object]]:
    decision = contract.classify_ct_roi("Manual", "bladder")
    rows = contract.disposition_rows_for_arms(
        {
            "patient_id": "P1",
            "course_id": "C1",
            "series_uid": "1.2.3",
            "segmentation_source": "Manual",
            "mask_identity": "9.8.7",
            "rtstruct_sop_instance_uid": "9.8.7",
            "roi_original_name": "bladder",
            "roi_name": "bladder",
            "stable_roi_identifier": "roi:1",
            "modality": "CT",
        },
        decision=decision,
        disposition="success",
        detail="",
        failure_kind="",
        run_identifier="run-1",
        code_revision="revision-1",
        native_voxel_count=1000,
        required=required,
        effective_hashes={
            contract.PRIMARY_ARM: "effective-primary",
            contract.SENSITIVITY_ARM: "effective-sensitivity",
        },
        configured_parameter_hashes={
            contract.PRIMARY_ARM: "configured-primary",
            contract.SENSITIVITY_ARM: "configured-sensitivity",
        },
    )
    expected = {
        "original_shape_VoxelVolume",
        "original_firstorder_Mean",
        "wavelet-LLL_glcm_MCC",
    }
    for index, row in enumerate(rows, start=1):
        row["original_shape_VoxelVolume"] = 100.0
        row["original_firstorder_Mean"] = float(index)
        row["wavelet-LLL_glcm_MCC"] = 0.5
        row.update(contract._feature_schema_metadata(expected))
    return rows


def _course_with_inputs(tmp_path: Path) -> Path:
    course = tmp_path / "P1" / "C1"
    ct_dir = course / "CT"
    metadata = course / "metadata"
    ct_dir.mkdir(parents=True)
    metadata.mkdir()
    (ct_dir / "slice-1.dcm").write_bytes(b"ct-input-v1")
    (course / "RS_orig.dcm").write_bytes(b"rtstruct-input-v1")
    (metadata / "case_metadata.json").write_text(
        json.dumps(
            {
                "course_contract": {
                    "planning_ct": {"dicom_dir": "CT"},
                    "authoritative_rtstruct": {
                        "path": "RS_orig.dcm",
                        "segmentation_source": "Manual",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return course


def _publish(course: Path, rows: list[dict[str, object]]) -> Path:
    return contract.write_ct_publication_atomic(
        pd.DataFrame(rows), course / "radiomics_ct.xlsx"
    )


def test_completion_sentinel_rejects_feature_mutation_and_accepts_restoration(
    tmp_path: Path,
) -> None:
    course = _course_with_inputs(tmp_path)
    parquet = _publish(course, _rows())
    sentinel = contract.write_completion_sentinel(course)
    original_bytes = parquet.read_bytes()
    original_payload = contract.validate_completion_sentinel(course, sentinel)

    mutated = pd.read_parquet(parquet, engine="pyarrow")
    mutated.loc[0, "original_firstorder_Mean"] += 1.0
    mutated.to_parquet(parquet, index=False, engine="pyarrow")

    with pytest.raises(ValueError, match="authoritative_parquet_sha256"):
        contract.validate_completion_sentinel(course, sentinel)

    parquet.write_bytes(original_bytes)
    restored_payload = contract.validate_completion_sentinel(course, sentinel)
    assert restored_payload == original_payload


def test_input_closure_binds_only_governed_radiomics_inputs(tmp_path: Path) -> None:
    course = _course_with_inputs(tmp_path)
    _publish(course, _rows())
    sentinel = contract.write_completion_sentinel(course)
    payload = contract.validate_completion_sentinel(course, sentinel)

    (course / "downstream-report.txt").write_text("later stage", encoding="utf-8")
    assert contract.validate_completion_sentinel(course, sentinel) == payload

    (course / "CT" / "slice-1.dcm").write_bytes(b"ct-input-v2")
    with pytest.raises(ValueError, match="input_closure_sha256"):
        contract.validate_completion_sentinel(course, sentinel)


def test_input_closure_is_path_portable_and_contract_decision_bound(
    tmp_path: Path,
) -> None:
    course = _course_with_inputs(tmp_path)
    rows = pd.DataFrame(_rows())
    relative_digest = contract.input_closure_sha256(course, rows)
    contract_path = course / "metadata" / "case_metadata.json"
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    decision = payload["course_contract"]
    decision["planning_ct"]["dicom_dir"] = str(course / "CT")
    decision["authoritative_rtstruct"]["path"] = str(course / "RS_orig.dcm")
    contract_path.write_text(json.dumps(payload), encoding="utf-8")

    assert contract.input_closure_sha256(course, rows) == relative_digest

    decision["planning_ct"]["series_instance_uid"] = "changed-series"
    contract_path.write_text(json.dumps(payload), encoding="utf-8")
    assert contract.input_closure_sha256(course, rows) != relative_digest

    decision["planning_ct"]["dicom_dir"] = str(tmp_path.parent)
    contract_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes the course directory"):
        contract.input_closure_sha256(course, rows)


def test_input_closure_binds_nifti_fallback_image_and_mask(
    tmp_path: Path,
) -> None:
    course = _course_with_inputs(tmp_path)
    mask = course / "Segmentation_TotalSegmentator" / "bladder.nii.gz"
    image = course / "NIFTI" / "ct.nii.gz"
    mask.parent.mkdir()
    image.parent.mkdir()
    mask.write_bytes(b"mask-v1")
    image.write_bytes(b"image-v1")
    rows = pd.DataFrame(_rows(required=False))
    rows["segmentation_source"] = "AutoTS_total_nifti_fallback"
    rows["mask_path_source"] = str(mask)
    rows["nifti_path"] = str(image)
    before = contract.input_closure_sha256(course, rows)

    mask.write_bytes(b"mask-v2")
    assert contract.input_closure_sha256(course, rows) != before

    mask.write_bytes(b"mask-v1")
    image.write_bytes(b"image-v2")
    assert contract.input_closure_sha256(course, rows) != before


def test_required_unexplained_incomplete_vector_fails_publication(
    tmp_path: Path,
) -> None:
    rows = _rows(required=True)
    rows[0]["original_firstorder_Mean"] = float("nan")

    with pytest.raises(ValueError, match="required ROI-arm rows"):
        _publish(_course_with_inputs(tmp_path), rows)


def test_configured_schema_rejects_globally_absent_feature(
    tmp_path: Path,
) -> None:
    rows = _rows(required=True)
    expected = {
        "original_shape_VoxelVolume",
        "original_firstorder_Mean",
        "original_glcm_Contrast",
        "wavelet-LLL_glcm_MCC",
    }
    for row in rows:
        row.update(contract._feature_schema_metadata(expected))

    with pytest.raises(ValueError, match="required ROI-arm rows"):
        _publish(_course_with_inputs(tmp_path), rows)


def test_v2_sentinel_refuses_output_inferred_feature_schema(
    tmp_path: Path,
) -> None:
    course = _course_with_inputs(tmp_path)
    rows = _rows(required=True)
    for row in rows:
        for column in (
            contract.RADIOMICS_EXPECTED_SCHEMA_SHA256_COLUMN,
            contract.RADIOMICS_EXPECTED_COUNT_COLUMN,
            contract.RADIOMICS_EXPECTED_SCHEMA_SOURCE_COLUMN,
            contract.RADIOMICS_EXPECTED_SCHEMA_ZLIB_COLUMN,
        ):
            row.pop(column)
    _publish(course, rows)

    with pytest.raises(ValueError, match="requires configured feature schema"):
        contract.write_completion_sentinel(course)


def test_optional_incomplete_vector_is_recorded_and_not_analysis_eligible(
    tmp_path: Path,
) -> None:
    course = _course_with_inputs(tmp_path)
    rows = _rows(required=False)
    rows[0]["original_firstorder_Mean"] = float("nan")
    rows[0][contract.ENVIRONMENT_FINGERPRINT_COLUMN] = "parent-env"
    rows[1][contract.ENVIRONMENT_FINGERPRINT_COLUMN] = "isolated-env"
    complete_pair = _rows(required=False)
    for row in complete_pair:
        row["roi_original_name"] = "rectum"
        row["roi_name"] = "rectum"
        row["stable_roi_identifier"] = "roi:2"
    parquet = _publish(course, [*rows, *complete_pair])

    published = contract.read_authoritative_ct_publication(parquet)
    assert published[contract.RADIOMICS_FEATURE_COMPLETENESS_COLUMN].tolist() == [
        "incomplete",
        "complete",
        "complete",
        "complete",
    ]
    eligible = contract.analysis_eligible_feature_rows(published)
    assert eligible["roi_original_name"].tolist() == ["rectum", "rectum"]
    contract.write_ct_publication_atomic(
        eligible, tmp_path / "aggregate" / "radiomics_all.xlsx"
    )
    sentinel = contract.write_completion_sentinel(course)
    payload = contract.validate_completion_sentinel(course, sentinel)
    assert payload["environment_fingerprint"].startswith("sha256-set:")
    assert payload["feature_completeness"]["analysis_eligible_row_count"] == 2


def test_extractor_declared_undefined_feature_remains_analysis_eligible(
    tmp_path: Path,
) -> None:
    course = _course_with_inputs(tmp_path)
    rows = _rows(required=True)
    feature = "wavelet-LLL_glcm_MCC"
    rows[0][feature] = float("nan")
    rows[0][contract.RADIOMICS_UNDEFINED_FEATURES_COLUMN] = json.dumps([feature])
    parquet = _publish(course, rows)

    published = contract.read_authoritative_ct_publication(parquet)
    assert published[contract.RADIOMICS_FEATURE_COMPLETENESS_COLUMN].tolist() == [
        "complete_with_undefined",
        "complete",
    ]
    assert len(contract.analysis_eligible_feature_rows(published)) == 2


def test_unapproved_undefined_label_cannot_hide_incomplete_vector(
    tmp_path: Path,
) -> None:
    rows = _rows(required=False)
    feature = "original_firstorder_Mean"
    rows[0][feature] = float("nan")
    rows[0][contract.RADIOMICS_UNDEFINED_FEATURES_COLUMN] = json.dumps([feature])

    with pytest.raises(ValueError, match="without an approved undefined-value contract"):
        _publish(_course_with_inputs(tmp_path), rows)


def test_optional_incomplete_vector_is_recorded_in_roi_ledger(
    tmp_path: Path,
) -> None:
    course = _course_with_inputs(tmp_path)
    rows = _rows(required=False)
    rows[0]["original_firstorder_Mean"] = float("nan")
    parquet = _publish(course, rows)
    published = contract.read_authoritative_ct_publication(parquet)
    task = radiomics_parallel._RoiTask(
        source="Manual",
        rs_path=str(course / "RS_orig.dcm"),
        roi_name="bladder",
        course_dir=str(course),
        series_uid="1.2.3",
        mask_identity="9.8.7",
        stable_roi_identifier="roi:1",
        decision=contract.classify_ct_roi("Manual", "bladder"),
        run_identifier="run-1",
        code_revision="revision-1",
        configured_parameter_hashes={},
        effective_parameter_hashes={},
        required=False,
    )

    radiomics_parallel._write_parallel_roi_ledger(
        course,
        [task],
        published.to_dict("records"),
        extracted=True,
        technical=True,
    )

    ledger = json.loads(
        (course / "metadata" / "radiomics_roi_ledger.json").read_text(
            encoding="utf-8"
        )
    )
    assert ledger["course_roi"][0]["disposition"] == "excluded"
    assert (
        ledger["course_roi"][0]["reason_code"]
        == "failed_radiomics_feature_completeness"
    )
    assert ledger["course"][0]["technical_exclusion"] is True


def test_conda_ledger_records_optional_incomplete_vector(
    tmp_path: Path,
) -> None:
    course = _course_with_inputs(tmp_path)
    rows = _rows(required=False)
    rows[0]["original_firstorder_Mean"] = float("nan")
    published = contract.read_authoritative_ct_publication(_publish(course, rows))
    task = {"roi_name": "bladder"}

    radiomics_conda._write_conda_roi_ledger(
        course,
        [task],
        published.to_dict("records"),
        extracted=True,
    )

    ledger = json.loads(
        (course / "metadata" / "radiomics_roi_ledger.json").read_text(
            encoding="utf-8"
        )
    )
    assert ledger["course_roi"][0]["disposition"] == "excluded"
    assert (
        ledger["course_roi"][0]["reason_code"]
        == "failed_radiomics_feature_completeness"
    )
    assert ledger["course"][0]["technical_exclusion"] is True


def test_nonapplicable_intensity_family_does_not_create_false_incompleteness(
    tmp_path: Path,
) -> None:
    course = _course_with_inputs(tmp_path)
    rows = _rows(required=True)
    rows[0]["intensity_texture_disposition"] = "not_applicable_bone"
    rows[0]["original_firstorder_Mean"] = float("nan")
    rows[0]["wavelet-LLL_glcm_MCC"] = float("nan")
    rows[0].update(
        contract._feature_schema_metadata({"original_shape_VoxelVolume"})
    )
    parquet = _publish(course, rows)

    published = contract.read_authoritative_ct_publication(parquet)
    assert published[contract.RADIOMICS_FEATURE_COMPLETENESS_COLUMN].tolist() == [
        "complete_with_not_applicable",
        "complete",
    ]
    assert len(contract.analysis_eligible_feature_rows(published)) == 2
