"""Real NumPy2 main -> NumPy1 PyRadiomics helper integration for one MR course.

This is the live-subprocess counterpart to ``tests/test_mr_course_source_contract.py``,
which mocks ``process_radiomics_batch``. Nothing is mocked here: the test drives
``rtpipeline.radiomics_conda.radiomics_for_course_mr`` through the real
``process_radiomics_batch``, the real ``check_radiomics_env`` probe, the real
``conda run -n rtpipeline-radiomics`` transport, and a real PyRadiomics
extraction under NumPy 1.x.

Scope and deliberate limits
---------------------------
* The generated course reproduces the **producer fixture surface** the MR path
  actually reads: one NIfTI volume, one producer metadata sidecar in the
  ``organize`` spelling, and one TotalSegmentator ``total_mr--<roi>`` mask.
* The DICOM instances carry **headers only** (identity attributes, no
  ``PixelData``). They exist because ``_mr_resolve_series_source`` binds the
  series identity through them. This file therefore makes **no claim** that the
  NIfTI was derived from those DICOM instances, and asserts nothing about
  DICOM->NIfTI geometric or numerical consistency.
* The numerical claims are limited to what the approved generated NIfTI
  supports: ``original_shape_VoxelVolume`` and the ``original_firstorder``
  intensity statistics are reconciled against a direct NumPy computation on the
  exact array this test wrote.
* One ROI, ``Original`` image type only, ``firstorder``/``shape`` only, one
  worker, single-threaded BLAS/OpenMP, and small arrays. No segmentation, no
  model download, no radiomics acceleration, no pipeline CLI or Snakemake.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, PYDICOM_IMPLEMENTATION_UID

import rtpipeline.radiomics_conda as rc
from rtpipeline.radiomics_ct_contract import configured_parameter_hash
from rtpipeline.radiomics_schema import is_radiomic_feature_column

STUDY_UID = "1.2.826.0.1.3680043.9.7.9"
SERIES_UID = "1.2.826.0.1.3680043.9.7.9.31"
SERIES_DIR = "mr_31"
SEGMENTATION_SOURCE = rc._MR_SEGMENTATION_SOURCE  # "AutoTS_total_mr"

# (z, y, x) voxels and (x, y, z) mm -- deliberately anisotropic so a spacing
# mix-up cannot pass the VoxelVolume reconciliation by coincidence.
IMAGE_SHAPE = (6, 10, 10)
SPACING_XYZ = (1.5, 1.5, 3.0)
ORIGIN_XYZ = (-12.0, -7.5, 40.0)

# Compact interior box: 3 x 4 x 4 = 48 voxels, above the configured minimum and
# large enough for PyRadiomics' shape mesh.
ROI_SLICE = (slice(1, 4), slice(2, 6), slice(3, 7))
ROI_VOXELS = 48

PARAMS_YAML = """\
imageType:
  Original: {}
featureClass:
  firstorder:
    - Mean
    - Minimum
    - Maximum
    - Range
  shape:
    - VoxelVolume
setting:
  label: 1
  binWidth: 25
"""

_CONDA_EXE_AVAILABLE = bool(shutil.which(rc.CONDA_EXE) or Path(rc.CONDA_EXE).is_file())

requires_radiomics_env = pytest.mark.skipif(
    not _CONDA_EXE_AVAILABLE,
    reason=f"conda-compatible executable {rc.CONDA_EXE!r} is not available",
)


# --------------------------------------------------------------------------
# isolation
# --------------------------------------------------------------------------
@pytest.fixture()
def isolated_runtime(tmp_path, monkeypatch):
    """Keep every byte this test writes, and every child process, task-owned.

    ``_conda_subprocess_env`` copies ``os.environ`` and only ``setdefault``s the
    thread limits, so setting them here is what actually pins the helper
    subprocess to one thread.
    """
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    # radiomics_for_course_mr materializes its NRRD image/mask through
    # tempfile.NamedTemporaryFile; pin that to the per-test directory.
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    for var in rc._THREAD_ENV_VARS:
        monkeypatch.setenv(var, "1")
    monkeypatch.setenv("RTPIPELINE_RADIOMICS_THREAD_LIMIT", "1")
    monkeypatch.setenv("RTPIPELINE_MAX_WORKERS", "1")
    monkeypatch.setenv("CONDA_NO_PLUGINS", "1")
    monkeypatch.setenv("XDG_CACHE_HOME", str(scratch))
    monkeypatch.setenv("MPLCONFIGDIR", str(scratch))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    return scratch


# --------------------------------------------------------------------------
# generated producer fixture
# --------------------------------------------------------------------------
def _reference_image_array() -> np.ndarray:
    """Deterministic, non-constant intensities with no random component."""
    z, y, x = np.indices(IMAGE_SHAPE)
    return (100 + 7 * z + 3 * y + 11 * x).astype(np.float32)


def _write_volume(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(SPACING_XYZ)
    image.SetOrigin(ORIGIN_XYZ)
    sitk.WriteImage(image, str(path))
    return path


def _write_dicom_headers(dicom_dir: Path, instances: int = 2) -> List[str]:
    """Header-only MR instances: identity for the source contract, no pixels."""
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


def _make_course(tmp_path: Path, *, empty_extra_roi: bool = False) -> Dict[str, Any]:
    course_dir = tmp_path / "SYNTH" / "COURSE_A"
    series_root = course_dir / "MR" / SERIES_DIR
    dicom_dir = series_root / "DICOM"
    sop_uids = _write_dicom_headers(dicom_dir)

    image_array = _reference_image_array()
    nifti_path = _write_volume(series_root / "NIFTI" / "mr_series.nii.gz", image_array)

    # organize._convert_related_series spelling: "<base>.metadata.json".
    sidecar = nifti_path.with_name("mr_series.metadata.json")
    sidecar.write_text(
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

    mask_array = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    mask_array[ROI_SLICE] = 1
    assert int(mask_array.sum()) == ROI_VOXELS
    seg_dir = series_root / "Segmentation_TotalSegmentator"
    mask_path = _write_volume(seg_dir / "total_mr--liver.nii.gz", mask_array)

    empty_mask_path = None
    if empty_extra_roi:
        empty_mask_path = _write_volume(
            seg_dir / "total_mr--spleen.nii.gz", np.zeros(IMAGE_SHAPE, dtype=np.uint8)
        )

    return {
        "course_dir": course_dir,
        "series_root": series_root,
        "dicom_dir": dicom_dir,
        "nifti_path": nifti_path,
        "sidecar": sidecar,
        "mask_path": mask_path,
        "empty_mask_path": empty_mask_path,
        "image_array": image_array,
        "mask_array": mask_array,
        "sop_uids": sop_uids,
    }


def _write_params(tmp_path: Path) -> Path:
    params_file = tmp_path / "mr_params_minimal.yaml"
    params_file.write_text(PARAMS_YAML, encoding="utf-8")
    return params_file


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


def _liver_required_contract(extra_inventory: bool = False) -> Dict[str, Any]:
    section: Dict[str, Any] = {
        "required_rois": [{"canonical_name": "liver", "source": SEGMENTATION_SOURCE}]
    }
    if extra_inventory:
        section["inventory_only"] = ["spleen"]
    return {"MR": section}


def _expected_intensity_stats(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    values = image[mask > 0].astype(np.float64)
    return {
        "original_firstorder_Mean": float(values.mean()),
        "original_firstorder_Minimum": float(values.min()),
        "original_firstorder_Maximum": float(values.max()),
        "original_firstorder_Range": float(values.max() - values.min()),
    }


def _ledger(course_dir: Path) -> Dict[str, Any]:
    return json.loads(
        (course_dir / "metadata" / "radiomics_mr_roi_ledger.json").read_text(
            encoding="utf-8"
        )
    )


# --------------------------------------------------------------------------
# environment receipt
# --------------------------------------------------------------------------
@requires_radiomics_env
def test_dual_environment_split_is_real(isolated_runtime):
    """The main interpreter has NumPy 2 and no PyRadiomics; the helper has both.

    This is the premise every other assertion in this file depends on, so it is
    measured rather than assumed.
    """
    assert np.__version__.startswith("2."), np.__version__
    with pytest.raises(ImportError):
        __import__("radiomics")

    probe = subprocess.run(
        [
            rc.CONDA_EXE,
            "run",
            "-n",
            rc.RADIOMICS_ENV,
            "python",
            "-c",
            "import json,sys,numpy,radiomics,SimpleITK;"
            "print(json.dumps({'python': sys.version.split()[0],"
            "'numpy': numpy.__version__, 'radiomics': radiomics.__version__,"
            "'simpleitk': SimpleITK.__version__, 'executable': sys.executable}))",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env=rc._conda_subprocess_env(),
    )
    assert probe.returncode == 0, probe.stderr
    observed = json.loads(probe.stdout.strip().splitlines()[-1])
    print("helper environment:", json.dumps(observed, sort_keys=True))
    assert observed["numpy"].startswith("1."), observed
    assert observed["radiomics"].lstrip("v").startswith("3."), observed
    for var in rc._THREAD_ENV_VARS:
        assert rc._conda_subprocess_env()[var] == "1"

    # check_radiomics_env is the production gate; run it for real.
    assert rc.check_radiomics_env(timeout=300) is True


# --------------------------------------------------------------------------
# the integration itself
# --------------------------------------------------------------------------
@requires_radiomics_env
def test_mr_course_extracts_through_the_real_conda_helper(
    tmp_path, isolated_runtime, caplog
):
    """One MR course end to end: batch conda route, real PyRadiomics numbers."""
    caplog.set_level("INFO", logger="rtpipeline.radiomics_conda")
    fixture = _make_course(tmp_path)
    course_dir = fixture["course_dir"]
    params_file = _write_params(tmp_path)
    config = _Config(params_file=params_file, contract=_liver_required_contract())

    invoked: List[Dict[str, Any]] = []
    real_batch = rc.extract_radiomics_batch_with_conda
    real_single = rc.extract_radiomics_with_conda

    def spy_batch(tasks, params, *args, **kwargs):
        invoked.append({"route": "extract_radiomics_batch_with_conda", "n": len(tasks)})
        return real_batch(tasks, params, *args, **kwargs)

    def spy_single(*args, **kwargs):
        invoked.append({"route": "extract_radiomics_with_conda"})
        return real_single(*args, **kwargs)

    # Observation only: both spies delegate to the real, unmodified functions.
    rc.extract_radiomics_batch_with_conda = spy_batch
    rc.extract_radiomics_with_conda = spy_single
    try:
        result = rc.radiomics_for_course_mr(course_dir, config)
    finally:
        rc.extract_radiomics_batch_with_conda = real_batch
        rc.extract_radiomics_with_conda = real_single

    print("invoked conda routes:", invoked)
    assert invoked, "no conda helper route was reached"
    assert invoked[0]["route"] == "extract_radiomics_batch_with_conda"

    assert result is not None
    workbook = course_dir / "MR" / "radiomics_mr.xlsx"
    parquet = workbook.with_suffix(".parquet")
    assert Path(result) == workbook
    assert workbook.is_file() and parquet.is_file()

    frame = pd.read_parquet(parquet)
    assert len(frame) == 1, frame[["roi_original_name", "extraction_status"]]
    row = frame.iloc[0].to_dict()

    # --- row identity -----------------------------------------------------
    assert row["modality"] == "MR"
    assert row["image_modality"] == "MR"
    assert row["segmentation_source"] == SEGMENTATION_SOURCE
    assert row["patient_id"] == "SYNTH"
    assert row["course_id"] == "COURSE_A"
    assert row["series_uid"] == SERIES_UID
    assert row["study_uid"] == STUDY_UID
    assert row["series_dir"] == str(fixture["dicom_dir"])
    assert row["nifti_path"] == str(fixture["nifti_path"])
    assert row["source_content_sha256"] == rc.file_sha256(fixture["nifti_path"])
    assert row["mask_path_source"] == str(fixture["mask_path"])
    assert row["mask_identity"] == rc.file_sha256(fixture["mask_path"])
    assert row["roi_name"] == "liver"
    assert row["roi_original_name"] == "liver"
    assert row["stable_roi_identifier"] == "liver"
    assert row.get("extraction_status") in (None, "success") or pd.isna(
        row.get("extraction_status")
    )

    # --- parameter provenance --------------------------------------------
    assert row["extraction_arm"] == "mr_configured"
    assert row["configured_parameter_hash"] == configured_parameter_hash(
        params_file, arm="mr_configured", window=None, large_roi=False
    )
    effective = str(row["effective_parameter_hash"])
    assert len(effective) == 64 and effective != row["configured_parameter_hash"]
    assert str(row["run_identifier"])
    assert str(row["code_revision"])

    # --- numbers reconciled against an independent NumPy computation ------
    expected_voxel_volume = ROI_VOXELS * float(np.prod(SPACING_XYZ))
    assert row["original_shape_VoxelVolume"] == pytest.approx(
        expected_voxel_volume, rel=1e-9
    )
    expected_stats = _expected_intensity_stats(
        fixture["image_array"], fixture["mask_array"]
    )
    print("reconciliation:", json.dumps(
        {
            "expected_voxel_volume": expected_voxel_volume,
            "observed_voxel_volume": float(row["original_shape_VoxelVolume"]),
            "expected": expected_stats,
            "observed": {k: float(row[k]) for k in expected_stats},
        },
        sort_keys=True,
    ))
    for name, expected_value in expected_stats.items():
        assert row[name] == pytest.approx(expected_value, rel=1e-6), name

    # The reconciled intensities must actually vary; a constant ROI would make
    # Mean/Min/Max agree for the wrong reason.
    assert expected_stats["original_firstorder_Range"] > 0

    # --- full feature vector is finite ------------------------------------
    feature_columns = [c for c in frame.columns if is_radiomic_feature_column(c)]
    assert set(feature_columns) == {
        "original_firstorder_Maximum",
        "original_firstorder_Mean",
        "original_firstorder_Minimum",
        "original_firstorder_Range",
        "original_shape_VoxelVolume",
    }
    values = frame[feature_columns].to_numpy(dtype=float)
    assert np.isfinite(values).all(), dict(zip(feature_columns, values[0]))

    # --- PyRadiomics really ran (its own diagnostics travelled back) -------
    assert str(row["diagnostics_Versions_PyRadiomics"]).lstrip("v").startswith("3.")
    assert str(row["diagnostics_Versions_Numpy"]).startswith("1.")
    assert int(row["diagnostics_Mask-original_VoxelNum"]) == ROI_VOXELS
    print("diagnostics:", json.dumps(
        {
            k: str(row[k])
            for k in sorted(frame.columns)
            if str(k).startswith("diagnostics_Versions")
        },
        sort_keys=True,
    ))

    # --- course diagnostics ------------------------------------------------
    assert row["radiomics_roi_attempted"] == 1
    assert row["radiomics_roi_extracted"] == 1
    assert row["radiomics_roi_failed"] == 0

    # --- ledger ------------------------------------------------------------
    ledger = _ledger(course_dir)
    roi_rows = ledger["course_roi"]
    assert len(roi_rows) == 1
    liver = roi_rows[0]
    assert liver["roi_name"] == "liver"
    assert liver["modality"] == "MR"
    assert liver["disposition"] == "extracted"
    assert liver["reason_code"] == "extracted"
    assert liver["series_uid"] == SERIES_UID
    assert liver["source_content_sha256"] == rc.file_sha256(fixture["nifti_path"])
    assert liver["mask_identity"] == rc.file_sha256(fixture["mask_path"])
    course_rows = ledger["course"]
    assert len(course_rows) == 1
    assert course_rows[0]["extracted"] is True
    assert course_rows[0]["reason_code"] == "extracted"
    assert course_rows[0]["technical_exclusion"] is False

    # --- no temporary NRRD left behind ------------------------------------
    assert sorted(p.name for p in isolated_runtime.glob("mr_*")) == []


# --------------------------------------------------------------------------
# the single-helper route, reached by an ordinary valid-but-empty mask
# --------------------------------------------------------------------------
@requires_radiomics_env
def test_mr_course_with_a_valid_empty_roi_measures_and_accounts_both(
    tmp_path, isolated_runtime
):
    """One measurable ROI plus one valid-but-empty mask: measure one, account both.

    A valid-but-empty TotalSegmentator mask is an ordinary MR outcome, so this is
    the common shape of a real course, not a contrived input. Its precomputed
    disposition disables batch processing, which is exactly what routes the
    healthy ROI through the per-ROI ``extract_radiomics_with_conda`` helper. That
    route must return real measured values carrying the same parameter provenance
    the batch route carries, the empty ROI must stay an explicit nonmeasurement
    with no feature values, and a superseded publication must not survive.

    This is the repaired form of the reproducer this fixture first exposed: the
    single-helper route omitted ``__effective_parameter_hash__``, so the course
    died on an untyped ValueError, wrote no ledger, and left the stale workbook
    and Parquet on disk byte-identical.
    """
    fixture = _make_course(tmp_path, empty_extra_roi=True)
    course_dir = fixture["course_dir"]
    params_file = _write_params(tmp_path)
    config = _Config(
        params_file=params_file,
        contract=_liver_required_contract(extra_inventory=True),
    )

    # A superseded publication from an earlier run.
    workbook = course_dir / "MR" / "radiomics_mr.xlsx"
    parquet = workbook.with_suffix(".parquet")
    workbook.write_bytes(b"STALE-WORKBOOK")
    parquet.write_bytes(b"STALE-PARQUET")

    invoked: List[Dict[str, Any]] = []
    real_batch = rc.extract_radiomics_batch_with_conda
    real_single = rc.extract_radiomics_with_conda

    def spy_batch(tasks, params, *args, **kwargs):
        invoked.append({"route": "extract_radiomics_batch_with_conda", "n": len(tasks)})
        return real_batch(tasks, params, *args, **kwargs)

    def spy_single(*args, **kwargs):
        invoked.append({"route": "extract_radiomics_with_conda"})
        return real_single(*args, **kwargs)

    # Observation only: both spies delegate to the real, unmodified functions.
    rc.extract_radiomics_batch_with_conda = spy_batch
    rc.extract_radiomics_with_conda = spy_single
    try:
        result = rc.radiomics_for_course_mr(course_dir, config)
    finally:
        rc.extract_radiomics_batch_with_conda = real_batch
        rc.extract_radiomics_with_conda = real_single

    print("invoked conda routes:", invoked)
    # The point of this course: the disposition forces the per-ROI route.
    assert [entry["route"] for entry in invoked] == ["extract_radiomics_with_conda"]

    assert result is not None and Path(result) == workbook
    assert workbook.read_bytes() != b"STALE-WORKBOOK"
    assert parquet.read_bytes() != b"STALE-PARQUET"

    frame = pd.read_parquet(parquet)
    assert len(frame) == 2, frame[["roi_original_name", "extraction_status"]]
    rows = {str(row["roi_original_name"]): row for row in frame.to_dict("records")}
    assert set(rows) == {"liver", "spleen"}
    liver, spleen = rows["liver"], rows["spleen"]

    # --- the healthy ROI really measured ----------------------------------
    assert liver["series_uid"] == SERIES_UID
    assert liver["mask_path_source"] == str(fixture["mask_path"])
    assert liver["mask_identity"] == rc.file_sha256(fixture["mask_path"])
    assert liver.get("extraction_status") in (None, "success") or pd.isna(
        liver.get("extraction_status")
    )
    expected_voxel_volume = ROI_VOXELS * float(np.prod(SPACING_XYZ))
    expected_stats = _expected_intensity_stats(
        fixture["image_array"], fixture["mask_array"]
    )
    print("reconciliation:", json.dumps(
        {
            "expected_voxel_volume": expected_voxel_volume,
            "observed_voxel_volume": float(liver["original_shape_VoxelVolume"]),
            "expected": expected_stats,
            "observed": {k: float(liver[k]) for k in expected_stats},
        },
        sort_keys=True,
    ))
    assert liver["original_shape_VoxelVolume"] == pytest.approx(
        expected_voxel_volume, rel=1e-9
    )
    for name, expected_value in expected_stats.items():
        assert liver[name] == pytest.approx(expected_value, rel=1e-6), name
    assert expected_stats["original_firstorder_Range"] > 0

    feature_columns = [c for c in frame.columns if is_radiomic_feature_column(c)]
    assert set(feature_columns) == {
        "original_firstorder_Maximum",
        "original_firstorder_Mean",
        "original_firstorder_Minimum",
        "original_firstorder_Range",
        "original_shape_VoxelVolume",
    }
    liver_values = np.array([float(liver[c]) for c in feature_columns])
    assert np.isfinite(liver_values).all(), dict(zip(feature_columns, liver_values))

    # PyRadiomics really ran in the helper interpreter for this route too.
    assert str(liver["diagnostics_Versions_PyRadiomics"]).lstrip("v").startswith("3.")
    assert str(liver["diagnostics_Versions_Numpy"]).startswith("1.")
    assert int(liver["diagnostics_Mask-original_VoxelNum"]) == ROI_VOXELS

    # --- provenance the per-ROI route previously omitted -------------------
    assert liver["extraction_arm"] == "mr_configured"
    assert liver["configured_parameter_hash"] == configured_parameter_hash(
        params_file, arm="mr_configured", window=None, large_roi=False
    )
    effective = str(liver["effective_parameter_hash"])
    print("single-route effective_parameter_hash:", effective)
    assert len(effective) == 64 and effective != liver["configured_parameter_hash"]

    # --- the empty ROI is a nonmeasurement, not a feature row --------------
    assert spleen["extraction_status"] == "failed"
    assert spleen["roi_structural_code"] == "not_computed_valid_empty_scope"
    assert spleen["extraction_failure_kind"] == "degenerate_mask"
    assert spleen["mask_path_source"] == str(fixture["empty_mask_path"])
    assert spleen["mask_identity"] == rc.file_sha256(fixture["empty_mask_path"])
    for column in feature_columns:
        assert pd.isna(spleen[column]), f"{column} was invented for an empty mask"

    # --- the course denominator says one of two, not two of two ------------
    assert liver["radiomics_roi_attempted"] == 2
    assert liver["radiomics_roi_extracted"] == 1
    assert liver["radiomics_roi_failed"] == 1

    # --- ledger accounts both outcomes -------------------------------------
    ledger = _ledger(course_dir)
    roi_rows = {str(row["roi_name"]): row for row in ledger["course_roi"]}
    print("ledger course_roi:", json.dumps(ledger["course_roi"], sort_keys=True, indent=1))
    assert set(roi_rows) == {"liver", "spleen"}
    assert roi_rows["liver"]["disposition"] == "extracted"
    assert roi_rows["liver"]["reason_code"] == "extracted"
    assert roi_rows["liver"]["mask_identity"] == rc.file_sha256(fixture["mask_path"])
    assert roi_rows["spleen"]["disposition"] == "excluded"
    assert roi_rows["spleen"]["reason_code"] == "not_computed_valid_empty_scope"
    assert roi_rows["spleen"]["mask_identity"] == rc.file_sha256(
        fixture["empty_mask_path"]
    )
    course_rows = ledger["course"]
    assert len(course_rows) == 1
    assert course_rows[0]["extracted"] is True
    # A valid-but-empty mask is a nonmeasurement, not anatomical proof or an
    # extraction exception. Requiredness is tested separately.
    assert course_rows[0]["technical_exclusion"] is False

    assert sorted(p.name for p in isolated_runtime.glob("mr_*")) == []


# --------------------------------------------------------------------------
# the two helper routes must agree about what actually ran
# --------------------------------------------------------------------------
@requires_radiomics_env
def test_effective_parameter_provenance_is_route_independent(tmp_path, isolated_runtime):
    """The same ROI and parameters must hash the same on either conda route.

    ``extract_radiomics_batch_with_conda`` and ``extract_radiomics_with_conda``
    are selected by whether the course happens to carry a disposition, which is a
    property of the course, not of the measurement. If the two routes reported
    different effective parameters for identical inputs, the recorded provenance
    would describe the dispatcher rather than the extraction.
    """
    params_file = _write_params(tmp_path)

    batch_fixture = _make_course(tmp_path / "batch")
    batch_result = rc.radiomics_for_course_mr(
        batch_fixture["course_dir"],
        _Config(params_file=params_file, contract=_liver_required_contract()),
    )
    assert batch_result is not None
    batch_frame = pd.read_parquet(Path(batch_result).with_suffix(".parquet"))
    assert len(batch_frame) == 1

    single_fixture = _make_course(tmp_path / "single", empty_extra_roi=True)
    single_result = rc.radiomics_for_course_mr(
        single_fixture["course_dir"],
        _Config(
            params_file=params_file,
            contract=_liver_required_contract(extra_inventory=True),
        ),
    )
    assert single_result is not None
    single_frame = pd.read_parquet(Path(single_result).with_suffix(".parquet"))
    single_liver = single_frame[single_frame["roi_original_name"] == "liver"]
    assert len(single_liver) == 1

    batch_row = batch_frame.iloc[0].to_dict()
    single_row = single_liver.iloc[0].to_dict()
    print("batch effective:", batch_row["effective_parameter_hash"])
    print("single effective:", single_row["effective_parameter_hash"])
    assert (
        single_row["effective_parameter_hash"]
        == batch_row["effective_parameter_hash"]
    )
    assert (
        single_row["configured_parameter_hash"]
        == batch_row["configured_parameter_hash"]
    )
    # The measured numbers agree too, so the shared hash is not shared emptiness.
    for column in (
        "original_shape_VoxelVolume",
        "original_firstorder_Mean",
        "original_firstorder_Range",
    ):
        assert single_row[column] == pytest.approx(batch_row[column], rel=1e-12)
    assert float(batch_row["original_firstorder_Range"]) > 0
