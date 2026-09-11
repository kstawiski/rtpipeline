"""MR failure publication and accounting boundaries, without a helper subprocess.

``tests/test_mr_real_helper_integration.py`` drives the real conda transport and
so needs the isolated PyRadiomics environment. The boundaries checked here are
about what ``radiomics_for_course_mr`` owes its consumers when extraction does
*not* produce measurements, so they are exercised with generated synthetic
courses and a substituted ``process_radiomics_batch``: no conda, no PyRadiomics,
no network, and no writes outside the per-test temporary directory.

Every course is generated in this file. Nothing here reads clinical data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import SimpleITK as sitk
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, PYDICOM_IMPLEMENTATION_UID

import rtpipeline.radiomics_conda as rc
from rtpipeline.radiomics_outcomes import RadiomicsCourseExtractionError

STUDY_UID = "1.2.826.0.1.3680043.9.7.11"
SERIES_UID = "1.2.826.0.1.3680043.9.7.11.41"
SERIES_DIR = "mr_41"
SEGMENTATION_SOURCE = rc._MR_SEGMENTATION_SOURCE

IMAGE_SHAPE = (6, 10, 10)
SPACING_XYZ = (1.5, 1.5, 3.0)
ORIGIN_XYZ = (-12.0, -7.5, 40.0)
ROI_SLICE = (slice(1, 4), slice(2, 6), slice(3, 7))

PARAMS_YAML = """\
imageType:
  Original: {}
featureClass:
  firstorder:
    - Mean
setting:
  label: 1
  binWidth: 25
"""


@pytest.fixture()
def isolated_runtime(tmp_path, monkeypatch):
    """Pin caches and worker counts; nothing here starts a child process."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    for var in rc._THREAD_ENV_VARS:
        monkeypatch.setenv(var, "1")
    monkeypatch.setenv("RTPIPELINE_MAX_WORKERS", "1")
    monkeypatch.setenv("XDG_CACHE_HOME", str(scratch))
    monkeypatch.setenv("MPLCONFIGDIR", str(scratch))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    return scratch


# --------------------------------------------------------------------------
# generated producer fixture
# --------------------------------------------------------------------------
def _write_volume(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(SPACING_XYZ)
    image.SetOrigin(ORIGIN_XYZ)
    sitk.WriteImage(image, str(path))
    return path


def _write_dicom_headers(dicom_dir: Path, instances: int = 2) -> List[str]:
    dicom_dir.mkdir(parents=True, exist_ok=True)
    sop_uids: List[str] = []
    for index in range(instances):
        sop_uid = f"{SERIES_UID}.{index + 1}"
        path = dicom_dir / f"slice_{index:03d}.dcm"
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = MRImageStorage
        file_meta.MediaStorageSOPInstanceUID = sop_uid
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID
        ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.SOPClassUID = MRImageStorage
        ds.SOPInstanceUID = sop_uid
        ds.SeriesInstanceUID = SERIES_UID
        ds.StudyInstanceUID = STUDY_UID
        ds.Modality = "MR"
        ds.PatientID = "SYNTH"
        ds.Rows = IMAGE_SHAPE[1]
        ds.Columns = IMAGE_SHAPE[2]
        ds.save_as(path, enforce_file_format=True)
        sop_uids.append(sop_uid)
    return sop_uids


def _make_course(tmp_path: Path, *, measurable_roi: bool = True) -> Dict[str, Any]:
    """One MR series: an always-present valid-but-empty ``spleen`` mask, and an
    optional measurable ``liver`` mask."""
    course_dir = tmp_path / "SYNTH" / "COURSE_A"
    series_root = course_dir / "MR" / SERIES_DIR
    dicom_dir = series_root / "DICOM"
    sop_uids = _write_dicom_headers(dicom_dir)

    z, y, x = np.indices(IMAGE_SHAPE)
    image_array = (100 + 7 * z + 3 * y + 11 * x).astype(np.float32)
    nifti_path = _write_volume(series_root / "NIFTI" / "mr_series.nii.gz", image_array)
    nifti_path.with_name("mr_series.metadata.json").write_text(
        json.dumps(
            {
                "study_instance_uid": STUDY_UID,
                "series_instance_uid": SERIES_UID,
                "instances": sop_uids,
                "instance_count": len(sop_uids),
                "modality": "MR",
                "nifti_path": str(nifti_path),
                "source_directory": str(dicom_dir),
                "nifti_sha256": rc.file_sha256(nifti_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    seg_dir = series_root / "Segmentation_TotalSegmentator"
    empty_mask_path = _write_volume(
        seg_dir / "total_mr--spleen.nii.gz", np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    )

    mask_path = None
    if measurable_roi:
        mask_array = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
        mask_array[ROI_SLICE] = 1
        mask_path = _write_volume(seg_dir / "total_mr--liver.nii.gz", mask_array)

    return {
        "course_dir": course_dir,
        "series_root": series_root,
        "seg_dir": seg_dir,
        "nifti_path": nifti_path,
        "mask_path": mask_path,
        "empty_mask_path": empty_mask_path,
    }


class _Config:
    """The exact attribute surface radiomics_for_course_mr reads."""

    def __init__(self, *, params_file: Path, contract: Dict[str, Any]) -> None:
        self.radiomics_params_file_mr = str(params_file)
        self.radiomics_analysis_contract = contract
        self.radiomics_min_voxels = 10
        self.radiomics_max_voxels = None
        self.radiomics_env_probe_timeout = 300
        self.radiomics_skip_rois: List[str] = []

    def effective_workers(self) -> int:
        return 1


def _write_params(tmp_path: Path) -> Path:
    params_file = tmp_path / "mr_params_minimal.yaml"
    params_file.write_text(PARAMS_YAML, encoding="utf-8")
    return params_file


def _contract(*, required: List[str] = (), inventory_only: List[str] = ()) -> Dict[str, Any]:
    section: Dict[str, Any] = {}
    if required:
        section["required_rois"] = [
            {"canonical_name": name, "source": SEGMENTATION_SOURCE} for name in required
        ]
    if inventory_only:
        section["inventory_only"] = list(inventory_only)
    return {"MR": section}


def _plant_stale_publication(course_dir: Path) -> Dict[str, Path]:
    """A superseded publication and checkpoint from an earlier run."""
    mr_root = course_dir / "MR"
    artifacts = {
        "workbook": mr_root / "radiomics_mr.xlsx",
        "parquet": mr_root / "radiomics_mr.parquet",
        "checkpoint": mr_root / "radiomics_mr_checkpoint.parquet",
    }
    for name, path in artifacts.items():
        path.write_bytes(f"STALE-{name.upper()}".encode("ascii"))
    return artifacts


def _assert_withdrawn(artifacts: Dict[str, Path]) -> None:
    for name, path in artifacts.items():
        stale = f"STALE-{name.upper()}".encode("ascii")
        assert not path.exists() or path.read_bytes() != stale, (
            f"{name} survived a failed MR course byte-identically"
        )


def _ledger(course_dir: Path) -> Dict[str, Any]:
    return json.loads(
        (course_dir / "metadata" / "radiomics_mr_roi_ledger.json").read_text(
            encoding="utf-8"
        )
    )


# --------------------------------------------------------------------------
# an unexpected extraction failure is still an accounted outcome
# --------------------------------------------------------------------------
def test_unexpected_extraction_failure_is_typed_accounted_and_withdraws_publication(
    tmp_path, isolated_runtime, monkeypatch
):
    """An untyped error inside extraction must not escape unaccounted.

    The observed instance was a bare ``ValueError`` raised while recording a
    successful ROI, which left the course with no ledger and a byte-identical
    stale workbook and Parquet on disk. Any unexpected failure of the extraction
    call has the same consequences, so the boundary -- not that one exception --
    is what is checked here.
    """
    fixture = _make_course(tmp_path)
    course_dir = fixture["course_dir"]
    artifacts = _plant_stale_publication(course_dir)
    config = _Config(
        params_file=_write_params(tmp_path),
        contract=_contract(required=["liver"], inventory_only=["spleen"]),
    )

    def exploding_batch(*args: Any, **kwargs: Any):
        raise ValueError("isolated extraction omitted effective parameter provenance")

    monkeypatch.setattr(rc, "process_radiomics_batch", exploding_batch)

    with pytest.raises(RadiomicsCourseExtractionError) as excinfo:
        rc.radiomics_for_course_mr(course_dir, config)

    message = str(excinfo.value)
    print("observed course error:", message)
    assert "ValueError" in message
    assert "isolated extraction omitted effective parameter provenance" in message
    assert isinstance(excinfo.value.__cause__, ValueError)

    _assert_withdrawn(artifacts)

    ledger = _ledger(course_dir)
    roi_rows = {str(row["roi_name"]): row for row in ledger["course_roi"]}
    print("ledger course_roi:", json.dumps(ledger["course_roi"], sort_keys=True, indent=1))
    assert set(roi_rows) == {"liver", "spleen"}
    # The ROI that was being measured is a technical failure ...
    assert roi_rows["liver"]["disposition"] == "excluded"
    assert roi_rows["liver"]["reason_code"] == "failed_radiomics_extraction"
    assert roi_rows["liver"]["mask_identity"] == rc.file_sha256(fixture["mask_path"])
    # ... and the valid-but-empty mask keeps its own, non-technical reason.
    assert roi_rows["spleen"]["disposition"] == "excluded"
    assert roi_rows["spleen"]["reason_code"] == "not_computed_valid_empty_scope"

    course_rows = ledger["course"]
    assert len(course_rows) == 1
    assert course_rows[0]["extracted"] is False
    assert course_rows[0]["technical_exclusion"] is True
    assert course_rows[0]["reason_code"] == "failed_radiomics_extraction"


def test_typed_course_failure_is_not_rewrapped(tmp_path, isolated_runtime, monkeypatch):
    """A course error raised by extraction keeps its own message and identity.

    ``process_radiomics_batch`` already withdraws its publication before raising
    this type, so the MR boundary must pass it through rather than restate it as
    a new failure.
    """
    course_dir = _make_course(tmp_path)["course_dir"]
    config = _Config(
        params_file=_write_params(tmp_path),
        contract=_contract(required=["liver"], inventory_only=["spleen"]),
    )

    original = RadiomicsCourseExtractionError("course extraction is incomplete: liver")

    def failing_batch(*args: Any, **kwargs: Any):
        raise original

    monkeypatch.setattr(rc, "process_radiomics_batch", failing_batch)

    with pytest.raises(RadiomicsCourseExtractionError) as excinfo:
        rc.radiomics_for_course_mr(course_dir, config)
    assert excinfo.value is original


# --------------------------------------------------------------------------
# a course that publishes only dispositions still owes a current inventory
# --------------------------------------------------------------------------
def test_disposition_only_course_rechecks_live_sources_before_its_ledger(
    tmp_path, isolated_runtime, monkeypatch
):
    """A source that changed during screening must not be described as accounted.

    The ledger of a course that measures nothing is still evidence about specific
    source bytes. The inventory is bound before discovery, so without a recheck a
    mask that arrived while the course was screened would be missing from a
    ledger that claims to account for the whole modality.
    """
    fixture = _make_course(tmp_path, measurable_roi=False)
    course_dir = fixture["course_dir"]
    artifacts = _plant_stale_publication(course_dir)
    config = _Config(
        params_file=_write_params(tmp_path),
        contract=_contract(inventory_only=["spleen"]),
    )

    real_roi_masks = rc._mr_roi_masks
    arrived: List[Path] = []

    def late_arrival(seg_dir: Path):
        masks = real_roi_masks(seg_dir)
        if not arrived:
            # A producer finishing a mask after this course read the directory.
            late = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
            late[ROI_SLICE] = 1
            arrived.append(_write_volume(Path(seg_dir) / "total_mr--kidney.nii.gz", late))
        return masks

    monkeypatch.setattr(rc, "_mr_roi_masks", late_arrival)

    with pytest.raises(RadiomicsCourseExtractionError) as excinfo:
        rc.radiomics_for_course_mr(course_dir, config)

    message = str(excinfo.value)
    print("observed course error:", message)
    assert arrived, "the regression never simulated a live source change"
    assert "changed while the course was being screened" in message
    assert "total_mr--kidney.nii.gz appeared during extraction" in message

    _assert_withdrawn(artifacts)
    assert not (course_dir / "metadata" / "radiomics_mr_roi_ledger.json").exists(), (
        "a ledger was published from an inventory that no longer describes the course"
    )


def test_disposition_only_course_with_stable_sources_still_publishes_its_ledger(
    tmp_path, isolated_runtime
):
    """The recheck must not turn an ordinary all-disposition course into a failure."""
    fixture = _make_course(tmp_path, measurable_roi=False)
    course_dir = fixture["course_dir"]
    config = _Config(
        params_file=_write_params(tmp_path),
        contract=_contract(inventory_only=["spleen"]),
    )

    assert rc.radiomics_for_course_mr(course_dir, config) is None

    ledger = _ledger(course_dir)
    roi_rows = {str(row["roi_name"]): row for row in ledger["course_roi"]}
    assert set(roi_rows) == {"spleen"}
    assert roi_rows["spleen"]["reason_code"] == "not_computed_valid_empty_scope"
    assert roi_rows["spleen"]["mask_identity"] == rc.file_sha256(
        fixture["empty_mask_path"]
    )
    course_rows = ledger["course"]
    assert len(course_rows) == 1
    assert course_rows[0]["extracted"] is False
    # Nothing technical went wrong: the only mask was validly empty.
    assert course_rows[0]["technical_exclusion"] is False


def test_required_valid_empty_roi_still_fails_the_course_closed(
    tmp_path, isolated_runtime
):
    """Required-vs-optional policy survives the recheck: required is not optional.

    The same valid-but-empty mask that is an accepted disposition when it is
    inventory-only must fail the course when the contract requires it.
    """
    fixture = _make_course(tmp_path, measurable_roi=False)
    course_dir = fixture["course_dir"]
    config = _Config(
        params_file=_write_params(tmp_path),
        contract=_contract(required=["spleen"]),
    )

    with pytest.raises(RadiomicsCourseExtractionError) as excinfo:
        rc.radiomics_for_course_mr(course_dir, config)
    assert "Required MR ROI(s) were not measured: spleen" in str(excinfo.value)

    # The failure is still accounted, and it names the mask it read.
    roi_rows = {str(row["roi_name"]): row for row in _ledger(course_dir)["course_roi"]}
    assert roi_rows["spleen"]["reason_code"] == "not_computed_valid_empty_scope"
    assert roi_rows["spleen"]["mask_identity"] == rc.file_sha256(
        fixture["empty_mask_path"]
    )
