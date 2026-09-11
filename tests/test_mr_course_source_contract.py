"""Source-identity, inventory and ledger contract for the conda MR course helper.

Covers ``rtpipeline.radiomics_conda.radiomics_for_course_mr`` — the helper that
``rtpipeline.radiomics.run_radiomics`` actually reaches for MR under NumPy 2.x.
Every input is generated in ``tmp_path``: synthetic single-frame MR DICOM
headers, SimpleITK NIfTI images/masks, and the two metadata sidecar spellings
the two real producers write. ``process_radiomics_batch`` is replaced by a mock
that reproduces its published contract (identity uniqueness, precomputed-failure
rows, required-ROI gate, real Arrow/Excel schema writer) so no conda subprocess,
PyRadiomics run, or clinical input is involved.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pydicom
import pytest
import SimpleITK as sitk
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, PYDICOM_IMPLEMENTATION_UID

import rtpipeline.radiomics_conda as rc
from rtpipeline.radiomics_outcomes import (
    RadiomicsCourseExtractionError,
    RadiomicsCourseOutcome,
    course_diagnostic_columns,
    extraction_status_is_nonfatal_for_required,
    invalidate_radiomics_outputs,
)
from rtpipeline.radiomics_schema import write_radiomics_feature_table_atomic

STUDY_UID = "1.2.826.0.1.3680043.9.7.1"
SERIES_UID = "1.2.826.0.1.3680043.9.7.1.20"
OTHER_SERIES_UID = "1.2.826.0.1.3680043.9.7.1.21"
SERIES_DIR = "mr_20"
IMAGE_SHAPE = (4, 8, 8)  # (z, y, x)


# --------------------------------------------------------------------------
# synthetic sources
# --------------------------------------------------------------------------
def _write_dicom_series(
    dicom_dir: Path,
    *,
    series_uid: str = SERIES_UID,
    study_uid: str = STUDY_UID,
    modality: str = "MR",
    instances: int = 2,
) -> List[str]:
    dicom_dir.mkdir(parents=True, exist_ok=True)
    sop_uids: List[str] = []
    for index in range(instances):
        sop_uid = f"{series_uid}.{index + 1}"
        path = dicom_dir / f"slice_{index:03d}.dcm"
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = MRImageStorage
        file_meta.MediaStorageSOPInstanceUID = sop_uid
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID
        ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.SOPClassUID = MRImageStorage
        ds.SOPInstanceUID = sop_uid
        ds.SeriesInstanceUID = series_uid
        ds.StudyInstanceUID = study_uid
        ds.Modality = modality
        ds.PatientID = "PAT"
        ds.Rows = IMAGE_SHAPE[1]
        ds.Columns = IMAGE_SHAPE[2]
        ds.save_as(path, enforce_file_format=True)
        sop_uids.append(sop_uid)
    return sop_uids


def _write_volume(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 2.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(image, str(path))
    return path


def _write_image(series_root: Path, *, name: str = "mr_series", seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    array = (rng.random(IMAGE_SHAPE) * 100).astype(np.float32)
    return _write_volume(series_root / "NIFTI" / f"{name}.nii.gz", array)


def _write_mask(series_root: Path, roi_name: str, voxels: int) -> Path:
    array = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    if voxels:
        flat = array.reshape(-1)
        flat[:voxels] = 1
        array = flat.reshape(IMAGE_SHAPE)
    seg_dir = series_root / "Segmentation_TotalSegmentator"
    return _write_volume(seg_dir / f"total_mr--{roi_name}.nii.gz", array)


def _sidecar_path(nifti_path: Path, style: str) -> Path:
    if style == "organize":
        return nifti_path.with_name(f"{nifti_path.name[:-7]}.metadata.json")
    if style == "segmentation":
        return nifti_path.with_name(f"{nifti_path.stem}.metadata.json")
    raise ValueError(style)


def _write_sidecar(
    nifti_path: Path,
    dicom_dir: Path,
    sop_uids: List[str],
    *,
    style: str = "organize",
    series_uid: str = SERIES_UID,
    study_uid: str = STUDY_UID,
    modality: str = "MR",
    payload: Optional[Dict[str, Any]] = None,
    raw: Optional[str] = None,
) -> Path:
    path = _sidecar_path(nifti_path, style)
    if raw is not None:
        path.write_text(raw, encoding="utf-8")
        return path
    if style == "organize":
        body: Dict[str, Any] = {
            "study_instance_uid": study_uid,
            "series_instance_uid": series_uid,
            "instances": list(sop_uids),
            "instance_count": len(sop_uids),
            "modality": modality,
            "nifti_path": str(nifti_path),
            "source_directory": str(dicom_dir),
            "nifti_sha256": rc.file_sha256(nifti_path),
            "nifti_geometry": {"size": list(IMAGE_SHAPE[::-1])},
        }
    else:
        # segmentation.py writes a leaner sidecar under the `<base>.nii` spelling.
        body = {
            "modality": modality,
            "nifti_path": str(nifti_path),
            "source_directory": str(dicom_dir),
            "series_instance_uid": series_uid,
        }
    body.update(payload or {})
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return path


def _make_course(tmp_path: Path) -> Path:
    course_dir = tmp_path / "PAT" / "COURSE"
    (course_dir / "MR").mkdir(parents=True, exist_ok=True)
    return course_dir


def _make_series(
    course_dir: Path,
    *,
    dir_name: str = SERIES_DIR,
    series_uid: str = SERIES_UID,
    study_uid: str = STUDY_UID,
    modality: str = "MR",
    dicom_modality: Optional[str] = None,
    sidecar_style: str = "organize",
    rois=(("liver", 40), ("spleen", 32)),
    with_dicom: bool = True,
    with_sidecar: bool = True,
    image_name: str = "mr_series",
) -> Dict[str, Any]:
    series_root = course_dir / "MR" / dir_name
    dicom_dir = series_root / "DICOM"
    sop_uids: List[str] = []
    if with_dicom:
        sop_uids = _write_dicom_series(
            dicom_dir,
            series_uid=series_uid,
            study_uid=study_uid,
            modality=dicom_modality or modality,
        )
    nifti_path = _write_image(series_root, name=image_name)
    sidecar = None
    if with_sidecar:
        sidecar = _write_sidecar(
            nifti_path,
            dicom_dir,
            sop_uids,
            style=sidecar_style,
            series_uid=series_uid,
            study_uid=study_uid,
            modality=modality,
        )
    seg_dir = series_root / "Segmentation_TotalSegmentator"
    seg_dir.mkdir(parents=True, exist_ok=True)
    masks = {roi: _write_mask(series_root, roi, voxels) for roi, voxels in rois}
    return {
        "series_root": series_root,
        "dicom_dir": dicom_dir,
        "nifti_path": nifti_path,
        "sidecar": sidecar,
        "seg_dir": seg_dir,
        "masks": masks,
        "sop_uids": sop_uids,
    }


class _Config:
    def __init__(self, *, contract: Optional[Dict[str, Any]] = None, min_voxels: int = 10) -> None:
        self.radiomics_analysis_contract = contract or {}
        self.radiomics_min_voxels = min_voxels
        self.radiomics_max_voxels = None
        self.radiomics_params_file_mr = None
        self.radiomics_env_probe_timeout = None
        self.radiomics_skip_rois: List[str] = []

    def effective_workers(self) -> int:
        return 1


# --------------------------------------------------------------------------
# process_radiomics_batch mock: reproduces its published contract
# --------------------------------------------------------------------------
class _Batch:
    def __init__(self) -> None:
        self.tasks: List[Dict[str, Any]] = []
        self.checkpoint_path: Optional[Path] = None
        self.calls = 0
        self.corrupt_parquet = False

    def __call__(self, tasks, output_path, **kwargs):
        self.calls += 1
        self.tasks = [dict(task) for task in tasks]
        self.checkpoint_path = kwargs.get("checkpoint_path")
        output_path = Path(output_path)

        seen = set()
        for task in tasks:
            for key in rc._task_expected_keys(task):
                assert key not in seen, f"duplicate publication identity: {key}"
                seen.add(key)

        rows: List[Dict[str, Any]] = []
        fatal: List[str] = []
        for task in tasks:
            metadata = dict(task.get("metadata") or {})
            failure = task.get("precomputed_failure")
            if failure:
                metadata.update(dict(failure.get("metadata") or {}))
                status = str(failure.get("status", "failed"))
                if bool(task.get("required", True)) and not extraction_status_is_nonfatal_for_required(status):
                    fatal.append(f"{metadata.get('roi_original_name')}: {failure.get('reason')}")
                    continue
                metadata.update(
                    {
                        "extraction_status": status,
                        "extraction_status_detail": str(failure.get("reason", "")),
                        "extraction_failure_kind": str(failure.get("failure_kind", "extraction_error")),
                    }
                )
                rows.append(metadata)
            else:
                # the real batch leaves extraction_status absent on success rows
                metadata["original_firstorder_Mean"] = 42.0
                metadata["original_shape_VoxelVolume"] = 100.0
                rows.append(metadata)

        if fatal:
            invalidate_radiomics_outputs(output_path)
            raise RadiomicsCourseExtractionError(
                "Radiomics course extraction is incomplete: " + "; ".join(fatal)
            )
        if not rows:
            invalidate_radiomics_outputs(output_path)
            return None

        counts: Dict[str, Dict[str, int]] = {}
        failures: List[Dict[str, str]] = []
        for row in rows:
            source = str(row.get("segmentation_source", "unknown"))
            bucket = counts.setdefault(source, {"attempted": 0, "extracted": 0, "failed": 0})
            bucket["attempted"] += 1
            if str(row.get("extraction_status") or "success") == "success":
                bucket["extracted"] += 1
            else:
                bucket["failed"] += 1
                failures.append(
                    {
                        "roi_name": str(row.get("roi_original_name", "")),
                        "source": source,
                        "status": str(row.get("extraction_status")),
                        "failure_kind": str(row.get("extraction_failure_kind", "")),
                        "reason": str(row.get("extraction_status_detail", "")),
                    }
                )
        diagnostics = course_diagnostic_columns(
            RadiomicsCourseOutcome.extracted(output_path, roi_counts=counts, roi_failures=failures)
        )
        for row in rows:
            row.update(diagnostics)
        write_radiomics_feature_table_atomic(pd.DataFrame(rows), output_path)
        if self.corrupt_parquet:
            output_path.with_suffix(".parquet").write_bytes(b"not a parquet file")
        return output_path


@pytest.fixture()
def batch(monkeypatch) -> _Batch:
    fake = _Batch()
    monkeypatch.setattr(rc, "process_radiomics_batch", fake)
    monkeypatch.setattr(rc, "check_radiomics_env", lambda *a, **k: True)
    return fake


# --------------------------------------------------------------------------
# readers
# --------------------------------------------------------------------------
def _published(course_dir: Path) -> pd.DataFrame:
    frame = pd.read_parquet(course_dir / "MR" / "radiomics_mr.parquet")
    if "extraction_status" not in frame.columns:
        frame["extraction_status"] = None
    # absent extraction_status means success everywhere in this pipeline
    frame["extraction_status"] = frame["extraction_status"].map(
        lambda value: "success" if value is None or pd.isna(value) else value
    )
    return frame


def _ledger(course_dir: Path) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(
        (course_dir / "metadata" / "radiomics_mr_roi_ledger.json").read_text(encoding="utf-8")
    )
    return {str(row["roi_name"]): row for row in payload["course_roi"]}


def _contract(required=(), optional=()) -> Dict[str, Any]:
    section: Dict[str, Any] = {}
    if required:
        section["required_rois"] = list(required)
    if optional:
        section["optional_rois"] = list(optional)
    return {"MR": section}


# --------------------------------------------------------------------------
# positive identity controls
# --------------------------------------------------------------------------
@pytest.mark.parametrize("sidecar_style", ["organize", "segmentation"])
def test_valid_series_publishes_rows_bound_to_their_current_source(tmp_path, batch, sidecar_style):
    course_dir = _make_course(tmp_path)
    series = _make_series(course_dir, sidecar_style=sidecar_style)

    result = rc.radiomics_for_course_mr(course_dir, _Config())

    assert result == course_dir / "MR" / "radiomics_mr.xlsx"
    frame = _published(course_dir)
    assert set(frame.roi_original_name) == {"liver", "spleen"}
    assert set(frame.extraction_status) == {"success"}
    # identity is the real DICOM series, not the directory name
    assert set(frame.series_uid) == {SERIES_UID}
    assert set(frame.study_uid) == {STUDY_UID}
    assert set(frame.series_dir) == {str(series["dicom_dir"])}
    assert set(frame.nifti_path) == {str(series["nifti_path"])}
    # content fingerprints bind the exact bytes that were measured
    assert set(frame.source_content_sha256) == {rc.file_sha256(series["nifti_path"])}
    assert dict(zip(frame.roi_original_name, frame.mask_identity)) == {
        roi: rc.file_sha256(path) for roi, path in series["masks"].items()
    }
    ledger = _ledger(course_dir)
    assert {name: row["reason_code"] for name, row in ledger.items()} == {
        "liver": "extracted",
        "spleen": "extracted",
    }


def test_absent_mr_directory_is_recorded_as_not_applicable(tmp_path, batch):
    course_dir = tmp_path / "PAT" / "COURSE"
    course_dir.mkdir(parents=True)

    config = _Config(contract=_contract(optional=["liver"]))
    assert rc.radiomics_for_course_mr(course_dir, config) is None
    assert batch.calls == 0
    payload = json.loads(
        (course_dir / "metadata" / "radiomics_mr_roi_ledger.json").read_text(encoding="utf-8")
    )
    assert payload["course"][0]["extracted"] is False
    assert _ledger(course_dir)["liver"]["reason_code"] == "not_applicable_modality"


def test_empty_mr_directory_is_recorded_as_not_applicable(tmp_path, batch):
    course_dir = _make_course(tmp_path)

    config = _Config(contract=_contract(optional=["liver"]))
    assert rc.radiomics_for_course_mr(course_dir, config) is None
    assert batch.calls == 0
    assert _ledger(course_dir)["liver"]["reason_code"] == "not_applicable_modality"


# --------------------------------------------------------------------------
# negative source-identity controls
# --------------------------------------------------------------------------
def test_series_without_authoritative_dicom_is_not_measured(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    _make_series(course_dir, with_dicom=False)

    assert rc.radiomics_for_course_mr(course_dir, _Config()) is None
    assert not (course_dir / "MR" / "radiomics_mr.xlsx").exists()
    ledger = _ledger(course_dir)
    assert set(ledger) == {"liver", "spleen"}
    assert {row["reason_code"] for row in ledger.values()} == {"failed_source_read"}


def test_ambiguous_image_metadata_pairing_fails_closed(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    series = _make_series(course_dir)
    second = _write_image(series["series_root"], name="mr_series_b", seed=7)
    _write_sidecar(second, series["dicom_dir"], series["sop_uids"])

    assert rc.radiomics_for_course_mr(course_dir, _Config()) is None
    ledger = _ledger(course_dir)
    assert {row["reason_code"] for row in ledger.values()} == {"failed_source_read"}


def test_unpaired_image_is_a_contradiction_not_a_silent_choice(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    series = _make_series(course_dir)
    _write_image(series["series_root"], name="mr_series_b", seed=7)  # no sidecar

    assert rc.radiomics_for_course_mr(course_dir, _Config()) is None
    assert {row["reason_code"] for row in _ledger(course_dir).values()} == {"failed_source_read"}


def test_sidecar_naming_another_image_is_rejected(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    series = _make_series(course_dir)
    payload = json.loads(series["sidecar"].read_text(encoding="utf-8"))
    payload["nifti_path"] = str(series["nifti_path"].with_name("somewhere_else.nii.gz"))
    series["sidecar"].write_text(json.dumps(payload), encoding="utf-8")

    assert rc.radiomics_for_course_mr(course_dir, _Config()) is None
    assert {row["reason_code"] for row in _ledger(course_dir).values()} == {"failed_source_read"}


def test_malformed_metadata_is_recorded_not_swallowed(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    series = _make_series(course_dir)
    series["sidecar"].write_text("{ this is not json", encoding="utf-8")

    assert rc.radiomics_for_course_mr(course_dir, _Config()) is None
    ledger = _ledger(course_dir)
    assert set(ledger) == {"liver", "spleen"}
    assert {row["reason_code"] for row in ledger.values()} == {"failed_source_read"}


def test_modality_mismatch_is_recorded_not_silently_skipped(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    _make_series(course_dir, modality="CT")

    assert rc.radiomics_for_course_mr(course_dir, _Config()) is None
    ledger = _ledger(course_dir)
    assert set(ledger) == {"liver", "spleen"}
    assert {row["reason_code"] for row in ledger.values()} == {"failed_source_read"}


def test_metadata_series_uid_must_match_the_dicom_series(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    series = _make_series(course_dir)
    payload = json.loads(series["sidecar"].read_text(encoding="utf-8"))
    payload["series_instance_uid"] = OTHER_SERIES_UID
    series["sidecar"].write_text(json.dumps(payload), encoding="utf-8")

    assert rc.radiomics_for_course_mr(course_dir, _Config()) is None
    assert {row["reason_code"] for row in _ledger(course_dir).values()} == {"failed_source_read"}


def test_two_series_uids_in_one_directory_are_ambiguous(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    series = _make_series(course_dir)
    _write_dicom_series(
        series["dicom_dir"], series_uid=OTHER_SERIES_UID, instances=1
    )

    assert rc.radiomics_for_course_mr(course_dir, _Config()) is None
    assert {row["reason_code"] for row in _ledger(course_dir).values()} == {"failed_source_read"}


def test_image_rewritten_after_its_metadata_is_source_drift(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    series = _make_series(course_dir)
    rng = np.random.default_rng(99)
    _write_volume(series["nifti_path"], (rng.random(IMAGE_SHAPE) * 100).astype(np.float32))

    assert rc.radiomics_for_course_mr(course_dir, _Config()) is None
    assert {row["reason_code"] for row in _ledger(course_dir).values()} == {"failed_source_read"}


def test_dicom_instances_added_after_conversion_are_source_drift(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    series = _make_series(course_dir)
    extra = SERIES_UID + ".99"
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = extra
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID
    path = series["dicom_dir"] / "slice_extra.dcm"
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = extra
    ds.SeriesInstanceUID = SERIES_UID
    ds.StudyInstanceUID = STUDY_UID
    ds.Modality = "MR"
    ds.save_as(path, enforce_file_format=True)

    assert rc.radiomics_for_course_mr(course_dir, _Config()) is None
    assert {row["reason_code"] for row in _ledger(course_dir).values()} == {"failed_source_read"}


def test_unreadable_dicom_instance_is_a_technical_failure(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    series = _make_series(course_dir)
    next(iter(sorted(series["dicom_dir"].glob("*.dcm")))).write_bytes(b"corrupt")

    assert rc.radiomics_for_course_mr(course_dir, _Config()) is None
    assert {row["reason_code"] for row in _ledger(course_dir).values()} == {"failed_source_read"}


# --------------------------------------------------------------------------
# mask inventory
# --------------------------------------------------------------------------
def test_empty_mask_directory_still_inventories_the_series(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    _make_series(course_dir, rois=())

    assert rc.radiomics_for_course_mr(course_dir, _Config()) is None
    ledger = _ledger(course_dir)
    assert set(ledger) == {SERIES_DIR}
    assert ledger[SERIES_DIR]["reason_code"] == "failed_source_segmentation"


def test_missing_segmentation_directory_is_inventoried(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    series = _make_series(course_dir, rois=())
    series["seg_dir"].rmdir()

    assert rc.radiomics_for_course_mr(course_dir, _Config()) is None
    ledger = _ledger(course_dir)
    assert ledger[SERIES_DIR]["reason_code"] == "failed_source_segmentation"


def test_aggregate_label_volumes_are_not_measured_as_rois(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    series = _make_series(course_dir)
    _write_mask(series["series_root"], "multilabel", 60)
    (series["seg_dir"] / "total_mr--segmentations.json").write_text("{}", encoding="utf-8")

    rc.radiomics_for_course_mr(course_dir, _Config())

    assert set(_published(course_dir).roi_original_name) == {"liver", "spleen"}
    assert "multilabel" not in _ledger(course_dir)


def test_mask_geometry_mismatch_is_recorded(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    series = _make_series(course_dir)
    array = np.ones((4, 6, 6), dtype=np.uint8)
    _write_volume(series["masks"]["spleen"], array)

    rc.radiomics_for_course_mr(course_dir, _Config())

    frame = _published(course_dir)
    spleen = frame[frame.roi_original_name == "spleen"].iloc[0]
    assert spleen.extraction_status == "failed"
    assert spleen.roi_structural_code == "failed_source_read"
    assert _ledger(course_dir)["spleen"]["reason_code"] == "failed_source_read"


def test_unreadable_mask_keeps_a_durable_technical_row(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    series = _make_series(course_dir)
    series["masks"]["spleen"].write_bytes(b"not a nifti")

    rc.radiomics_for_course_mr(course_dir, _Config())

    frame = _published(course_dir)
    assert set(frame.roi_original_name) == {"liver", "spleen"}
    spleen = frame[frame.roi_original_name == "spleen"].iloc[0]
    assert spleen.extraction_status == "failed"
    assert spleen.roi_structural_code == "failed_source_read"
    assert spleen.mask_path_source == str(series["masks"]["spleen"])


# --------------------------------------------------------------------------
# non-measurement dispositions keep their own class
# --------------------------------------------------------------------------
def test_valid_empty_and_below_minimum_masks_keep_their_classes(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    _make_series(course_dir, rois=(("liver", 40), ("spleen", 0), ("kidney_left", 3)))

    rc.radiomics_for_course_mr(course_dir, _Config(min_voxels=10))

    frame = _published(course_dir).set_index("roi_original_name")
    assert frame.loc["liver", "extraction_status"] == "success"
    assert frame.loc["spleen", "roi_structural_code"] == "not_computed_valid_empty_scope"
    assert frame.loc["kidney_left", "extraction_status"] == "below_minimum_voxels"
    assert frame.loc["kidney_left", "roi_structural_code"] == "ROI_MASK_BELOW_MIN_VOXELS"
    # non-measurements publish no feature value
    assert pd.isna(frame.loc["spleen", "original_firstorder_Mean"])
    ledger = _ledger(course_dir)
    assert ledger["spleen"]["reason_code"] == "not_computed_valid_empty_scope"
    assert ledger["kidney_left"]["reason_code"] == "ROI_MASK_BELOW_MIN_VOXELS"


def test_below_minimum_required_roi_preserves_the_historical_gate(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    _make_series(course_dir, rois=(("liver", 40), ("spleen", 3)))

    result = rc.radiomics_for_course_mr(
        course_dir, _Config(contract=_contract(required=["spleen"]), min_voxels=10)
    )

    assert result is not None
    assert _ledger(course_dir)["spleen"]["reason_code"] == "ROI_MASK_BELOW_MIN_VOXELS"


def test_empty_required_roi_fails_closed(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    _make_series(course_dir, rois=(("liver", 40), ("spleen", 0)))

    with pytest.raises(RadiomicsCourseExtractionError):
        rc.radiomics_for_course_mr(
            course_dir, _Config(contract=_contract(required=["spleen"]), min_voxels=10)
        )
    assert not (course_dir / "MR" / "radiomics_mr.xlsx").exists()


# --------------------------------------------------------------------------
# requiredness must follow the outcome, not the name
# --------------------------------------------------------------------------
def test_required_roi_failure_is_not_masked_by_name_presence(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    series = _make_series(course_dir)
    series["masks"]["liver"].write_bytes(b"not a nifti")

    with pytest.raises(RadiomicsCourseExtractionError):
        rc.radiomics_for_course_mr(course_dir, _Config(contract=_contract(required=["liver"])))
    assert not (course_dir / "MR" / "radiomics_mr.parquet").exists()


def test_required_roi_from_another_source_does_not_satisfy_the_contract(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    _make_series(course_dir)
    contract = {"MR": {"required_rois": [{"canonical_name": "liver", "source": "Manual"}]}}

    with pytest.raises(RadiomicsCourseExtractionError):
        rc.radiomics_for_course_mr(course_dir, _Config(contract=contract))


def test_required_roi_alias_is_accepted(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    _make_series(course_dir)
    contract = {"MR": {"required_rois": [{"canonical_name": "Liver_ref", "aliases": ["liver"]}]}}

    assert rc.radiomics_for_course_mr(course_dir, _Config(contract=contract)) is not None


def test_required_roi_absent_from_every_series_fails_closed(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    _make_series(course_dir, rois=(("liver", 40),))

    with pytest.raises(RadiomicsCourseExtractionError):
        rc.radiomics_for_course_mr(course_dir, _Config(contract=_contract(required=["prostate"])))


# --------------------------------------------------------------------------
# one good series must not hide a failed one
# --------------------------------------------------------------------------
def test_working_series_does_not_hide_a_failed_series(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    _make_series(course_dir, dir_name="mr_ok", rois=(("liver", 40),))
    _make_series(
        course_dir,
        dir_name="mr_broken",
        series_uid=OTHER_SERIES_UID,
        rois=(("spleen", 40),),
        with_dicom=False,
    )

    result = rc.radiomics_for_course_mr(course_dir, _Config())

    assert result is not None
    frame = _published(course_dir)
    assert set(frame.roi_original_name) == {"liver", "spleen"}
    broken = frame[frame.roi_original_name == "spleen"].iloc[0]
    assert broken.extraction_status == "failed"
    assert broken.roi_structural_code == "failed_source_read"
    # the workbook cannot describe itself as a clean extraction
    assert set(frame.radiomics_course_status) == {"extracted_with_failures"}
    assert int(frame.radiomics_roi_failed.iloc[0]) == 1
    ledger = _ledger(course_dir)
    assert ledger["liver"]["reason_code"] == "extracted"
    assert ledger["spleen"]["reason_code"] == "failed_source_read"


def test_two_series_publish_distinct_identities_for_one_roi(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    _make_series(course_dir, dir_name="mr_a", rois=(("liver", 40),))
    _make_series(
        course_dir,
        dir_name="mr_b",
        series_uid=OTHER_SERIES_UID,
        rois=(("liver", 24),),
        image_name="mr_series_b",
    )

    rc.radiomics_for_course_mr(course_dir, _Config())

    frame = _published(course_dir)
    assert len(frame) == 2
    assert set(frame.series_uid) == {SERIES_UID, OTHER_SERIES_UID}
    assert frame.mask_identity.nunique() == 2


# --------------------------------------------------------------------------
# publication must stay bound to current bytes
# --------------------------------------------------------------------------
def test_unreadable_published_table_is_not_swallowed(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    _make_series(course_dir)
    batch.corrupt_parquet = True

    with pytest.raises(RadiomicsCourseExtractionError):
        rc.radiomics_for_course_mr(course_dir, _Config())
    assert not (course_dir / "MR" / "radiomics_mr.xlsx").exists()
    assert not (course_dir / "MR" / "radiomics_mr.parquet").exists()


def test_stale_checkpoint_is_rejected_before_publication(tmp_path, batch):
    course_dir = _make_course(tmp_path)
    series = _make_series(course_dir)
    rc.radiomics_for_course_mr(course_dir, _Config())
    published = _published(course_dir)
    checkpoint = course_dir / "MR" / "radiomics_mr_checkpoint.parquet"
    published.to_parquet(checkpoint, index=False)

    # the mask changes; the stale checkpoint still names the same ROI/series
    _write_mask(series["series_root"], "liver", 55)
    rc.radiomics_for_course_mr(course_dir, _Config())

    assert not checkpoint.exists()
    frame = _published(course_dir)
    liver = frame[frame.roi_original_name == "liver"].iloc[0]
    assert liver.mask_identity == rc.file_sha256(series["masks"]["liver"])


def test_published_row_outside_the_current_inventory_fails_closed(tmp_path, batch, monkeypatch):
    course_dir = _make_course(tmp_path)
    _make_series(course_dir)
    real = batch.__call__

    def drifting(tasks, output_path, **kwargs):
        result = real(tasks, output_path, **kwargs)
        frame = pd.read_parquet(Path(output_path).with_suffix(".parquet"))
        frame.loc[frame.roi_original_name == "liver", "source_content_sha256"] = "0" * 64
        write_radiomics_feature_table_atomic(frame, Path(output_path))
        return result

    monkeypatch.setattr(rc, "process_radiomics_batch", drifting)

    with pytest.raises(RadiomicsCourseExtractionError):
        rc.radiomics_for_course_mr(course_dir, _Config())
    assert not (course_dir / "MR" / "radiomics_mr.parquet").exists()


# --------------------------------------------------------------------------
# the real batch contract accepts what this helper builds
# --------------------------------------------------------------------------
def test_real_batch_publishes_mr_measurements_and_dispositions(tmp_path, monkeypatch):
    """Run the real ``process_radiomics_batch`` with only the extractor mocked.

    The mock above reproduces the batch contract; this proves the tasks the MR
    helper builds satisfy the real identity, checkpoint and publication gates,
    without a conda subprocess or PyRadiomics.
    """
    monkeypatch.setenv("RTPIPELINE_RADIOMICS_BATCH", "0")
    monkeypatch.setattr(rc, "check_radiomics_env", lambda *a, **k: True)

    def fake_extract(image_path, mask_path, params_file, label, large_roi):
        assert Path(image_path).exists() and Path(mask_path).exists()
        return {
            "original_firstorder_Mean": 12.5,
            "original_shape_VoxelVolume": 250.0,
            "__effective_parameter_hash__": "effective-mr-hash",
        }

    monkeypatch.setattr(rc, "extract_radiomics_with_conda", fake_extract)

    course_dir = _make_course(tmp_path)
    _make_series(course_dir, dir_name="mr_ok", rois=(("liver", 40), ("spleen", 0)))
    _make_series(
        course_dir,
        dir_name="mr_broken",
        series_uid=OTHER_SERIES_UID,
        rois=(("kidney_left", 40),),
        with_dicom=False,
    )

    result = rc.radiomics_for_course_mr(course_dir, _Config(min_voxels=10))

    assert result is not None
    frame = _published(course_dir)
    assert set(frame.roi_original_name) == {"liver", "spleen", "kidney_left"}
    by_roi = frame.set_index("roi_original_name")
    assert by_roi.loc["liver", "extraction_status"] == "success"
    assert by_roi.loc["liver", "effective_parameter_hash"] == "effective-mr-hash"
    assert by_roi.loc["liver", "original_firstorder_Mean"] == 12.5
    assert by_roi.loc["spleen", "roi_structural_code"] == "not_computed_valid_empty_scope"
    assert by_roi.loc["kidney_left", "roi_structural_code"] == "failed_source_read"
    assert set(frame.radiomics_course_status) == {"extracted_with_failures"}
    ledger = _ledger(course_dir)
    assert ledger["liver"]["reason_code"] == "extracted"
    assert ledger["spleen"]["reason_code"] == "not_computed_valid_empty_scope"
    assert ledger["kidney_left"]["reason_code"] == "failed_source_read"
    # a checkpoint written by the real batch must not be reusable across a mask change
    checkpoint = course_dir / "MR" / "radiomics_mr_checkpoint.parquet"
    assert checkpoint.exists()
    _write_mask(course_dir / "MR" / "mr_ok", "liver", 55)
    rc.radiomics_for_course_mr(course_dir, _Config(min_voxels=10))
    assert (
        _published(course_dir).set_index("roi_original_name").loc["liver", "mask_identity"]
        == rc.file_sha256(course_dir / "MR" / "mr_ok" / "Segmentation_TotalSegmentator" / "total_mr--liver.nii.gz")
    )


def test_real_batch_fails_closed_on_a_required_unmeasurable_roi(tmp_path, monkeypatch):
    monkeypatch.setenv("RTPIPELINE_RADIOMICS_BATCH", "0")
    monkeypatch.setattr(rc, "check_radiomics_env", lambda *a, **k: True)
    monkeypatch.setattr(
        rc,
        "extract_radiomics_with_conda",
        lambda *a, **k: {
            "original_firstorder_Mean": 1.0,
            "__effective_parameter_hash__": "effective-mr-hash",
        },
    )
    course_dir = _make_course(tmp_path)
    series = _make_series(course_dir)
    series["masks"]["liver"].write_bytes(b"not a nifti")

    with pytest.raises(RadiomicsCourseExtractionError):
        rc.radiomics_for_course_mr(course_dir, _Config(contract=_contract(required=["liver"])))
    assert not (course_dir / "MR" / "radiomics_mr.parquet").exists()
    assert not (course_dir / "MR" / "radiomics_mr_checkpoint.parquet").exists()
