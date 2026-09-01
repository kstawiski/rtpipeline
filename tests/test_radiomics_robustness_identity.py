from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from rtpipeline.radiomics_ct_contract import CT_EXTRACTION_ARMS
from rtpipeline.radiomics_parallel import _prepare_radiomics_task
from rtpipeline.radiomics_robustness import (
    ROBUSTNESS_MEASUREMENT_TYPE,
    ROBUSTNESS_SOURCE_IDENTITY_COLUMNS,
    _feature_rows_from_worker_result,
    _identity_catalog_from_main_frame,
    _validate_extracted_feature_frame,
    _write_robustness_identity_ledger,
    extract_features_for_masks,
)


FIXTURE_PATH = (
    Path(__file__).parent
    / "data"
    / "task21_dfci_10022703523_2023-01_identity.json"
)


def _real_failure_identity() -> dict[str, str]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return payload["main_radiomics_identity"]


def _regression_frame(tmp_path: Path) -> tuple[pd.DataFrame, object, dict[str, str]]:
    main_identity = _real_failure_identity()
    main_frame = pd.DataFrame(
        [
            {**main_identity, "extraction_arm": arm}
            for arm in CT_EXTRACTION_ARMS
        ]
    )
    catalog, issues = _identity_catalog_from_main_frame(
        main_frame,
        expected_patient_id=main_identity["patient_id"],
        expected_course_id=main_identity["course_id"],
        expected_series_uid=main_identity["series_uid"],
    )
    assert issues == {}
    identity = catalog[("Manual", "Bladder")]

    image = sitk.GetImageFromArray(np.ones((4, 4, 4), dtype=np.int16))
    mask_array = np.zeros((4, 4, 4), dtype=np.uint8)
    mask_array[1:3, 1:3, 1:3] = 1
    mask = sitk.GetImageFromArray(mask_array)
    mask.CopyInformation(image)
    course_dir = tmp_path / main_identity["patient_id"] / main_identity["course_id"]
    course_dir.mkdir(parents=True)

    _, task_params = _prepare_radiomics_task(
        image,
        mask,
        None,
        "Manual",
        "Bladder",
        course_dir,
        tmp_path,
        False,
        source_identity=identity.as_dict(),
    )
    records = []
    for arm in CT_EXTRACTION_ARMS:
        records.append(
            {
                **identity.as_dict(),
                "modality": "CT",
                "roi_name": "Bladder",
                "measurement_type": task_params["measurement_type"],
                "perturbed_mask_identity": task_params[
                    "perturbed_mask_identity"
                ],
                "extraction_arm": arm,
                "original_firstorder_Mean": 1.25,
            }
        )
    result = {
        "__records__": records,
        "segmentation_source": "Manual",
        "roi_name": "Bladder",
        "patient_id": main_identity["patient_id"],
        "course_id": main_identity["course_id"],
        "perturbation_id": "ntcv_n0_t0_c0_v0",
    }
    frame = pd.DataFrame(_feature_rows_from_worker_result(result))
    return frame, identity, task_params


def test_real_dfci_failure_preserves_main_identity_through_worker_flattening(
    tmp_path: Path,
) -> None:
    frame, identity, task_params = _regression_frame(tmp_path)

    _validate_extracted_feature_frame(
        frame,
        {"ntcv_n0_t0_c0_v0"},
        "Manual/Bladder",
        expected_source_identity=identity,
    )

    for column in ROBUSTNESS_SOURCE_IDENTITY_COLUMNS:
        assert set(frame[column]) == {getattr(identity, column)}
    assert task_params["mask_identity"] == identity.mask_identity
    assert task_params["stable_roi_identifier"] == identity.stable_roi_identifier
    assert set(frame["measurement_type"]) == {ROBUSTNESS_MEASUREMENT_TYPE}
    assert set(frame["perturbed_mask_identity"]) == {
        task_params["perturbed_mask_identity"]
    }
    assert task_params["perturbed_mask_identity"].startswith("sha256:")
    assert task_params["perturbed_mask_identity"] != identity.mask_identity


def test_validator_rejects_identity_that_differs_from_main_radiomics(
    tmp_path: Path,
) -> None:
    frame, identity, _ = _regression_frame(tmp_path)
    frame["mask_identity"] = "different-source-mask"

    with pytest.raises(RuntimeError, match="disagrees with main radiomics"):
        _validate_extracted_feature_frame(
            frame,
            {"ntcv_n0_t0_c0_v0"},
            "Manual/Bladder",
            expected_source_identity=identity,
        )


def test_identity_failure_excludes_one_perturbation_but_keeps_other(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, identity, _ = _regression_frame(tmp_path)
    image = sitk.GetImageFromArray(np.ones((4, 4, 4), dtype=np.int16))
    mask = sitk.GetImageFromArray(np.ones((4, 4, 4), dtype=np.uint8))
    mask.CopyInformation(image)

    import rtpipeline.radiomics as radiomics_module
    import rtpipeline.radiomics_ct_contract as contract_module

    monkeypatch.setattr(radiomics_module, "_extractor", lambda *_args: object())
    monkeypatch.setattr(radiomics_module, "_get_params_file", lambda *_args: None)
    fake_radiomics = types.ModuleType("radiomics")
    fake_radiomics.featureextractor = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "radiomics", fake_radiomics)
    calls = {"count": 0}

    def fake_extract(*_args, common_metadata, **_kwargs):
        calls["count"] += 1
        records = []
        for arm in CT_EXTRACTION_ARMS:
            record = {
                **common_metadata,
                "extraction_arm": arm,
                "original_firstorder_Mean": 1.25,
            }
            if calls["count"] == 1:
                record["mask_identity"] = "not-the-main-mask"
            records.append(record)
        return records

    monkeypatch.setattr(contract_module, "extract_ct_roi_arms", fake_extract)
    masks = {"bad": mask, "good": mask}
    output = extract_features_for_masks(
        image,
        masks,
        object(),
        modality="CT",
        structure_name=identity.roi_original_name,
        patient_id=identity.patient_id,
        course_id=identity.course_id,
        segmentation_source=identity.segmentation_source,
        source_identity=identity,
    )

    assert calls["count"] == 2
    assert set(output["perturbation_id"]) == {"good"}
    assert "bad" in output.attrs["identity_failures"]
    _validate_extracted_feature_frame(
        output,
        {"good"},
        "Manual/Bladder",
        expected_source_identity=identity,
    )


def test_one_unidentifiable_roi_is_audited_without_discarding_valid_identity(
    tmp_path: Path,
) -> None:
    valid = _real_failure_identity()
    invalid = {
        **valid,
        "roi_original_name": "Rectum",
        "stable_roi_identifier": "",
    }
    catalog, issues = _identity_catalog_from_main_frame(
        pd.DataFrame([valid, invalid]),
        expected_patient_id=valid["patient_id"],
        expected_course_id=valid["course_id"],
        expected_series_uid=valid["series_uid"],
    )

    assert ("Manual", "Bladder") in catalog
    assert ("Manual", "Rectum") not in catalog
    assert issues[("Manual", "Rectum")]["reason_code"] == (
        "main_radiomics_identity_incomplete"
    )

    course_dir = tmp_path / valid["patient_id"] / valid["course_id"]
    ledger_path = _write_robustness_identity_ledger(
        course_dir,
        selected_count=2,
        rows=[
            {
                **catalog[("Manual", "Bladder")].as_dict(),
                "identity_status": "resolved",
            },
            {
                "segmentation_source": "Manual",
                "roi_original_name": "Rectum",
                "identity_status": "excluded",
                **issues[("Manual", "Rectum")],
            },
        ],
    )
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["identity_resolved_roi_count"] == 1
    assert ledger["identity_excluded_roi_count"] == 1
    assert ledger["rows"][1]["reason_code"] == "main_radiomics_identity_incomplete"
