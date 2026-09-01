"""Feature rows must record the acquisition scale they were extracted under.

A cohort mixing standard CT (about [-1000, 3071] HU) with Siemens extended-scale
iMAR reconstructions (beyond 8000 HU) discretises into very different numbers of
grey levels under fixed bin-size binning, so features are not comparable between
them. Without a recorded descriptor that confounder is invisible downstream.
"""
from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

import rtpipeline.radiomics_ct_contract as radiomics_ct_contract
import rtpipeline.radiomics as radiomics
import rtpipeline.radiomics_conda as radiomics_conda
import rtpipeline.radiomics_parallel as radiomics_parallel
from course_contract_test_utils import (
    write_minimal_course_contract,
    write_synthetic_rtstruct,
)
from rtpipeline.acquisition_scale import (
    AcquisitionDescriptorError,
    attach_acquisition_descriptor,
    classify_scale,
    describe_planning_ct,
    validate_acquisition_descriptor_table,
)
from rtpipeline.config import PipelineConfig
from rtpipeline.radiomics_outcomes import RadiomicsCourseExtractionError


_TAG_ABSENT = object()


def _ct(
    path: Path,
    *,
    slope: float | None,
    intercept: float | None,
    hi: int,
    desc: str | None,
    background: int = 0,
    series_uid: str | None = None,
    bits_stored: int = 16,
    pixel_representation: int = 0,
    kernel: str | None = "Qr40f",
    contrast_agent: object = _TAG_ABSENT,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = CTImageStorage
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=fm, preamble=b"\0" * 128)
    ds.SOPClassUID = fm.MediaStorageSOPClassUID
    ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
    ds.Modality = "CT"
    ds.SeriesInstanceUID = series_uid or generate_uid()
    ds.StudyInstanceUID = generate_uid()
    if desc is not None:
        ds.SeriesDescription = desc
    ds.Manufacturer = "Siemens Healthineers"
    if kernel is not None:
        ds.ConvolutionKernel = kernel
    ds.KVP = 120.0
    ds.SliceThickness = 3.0
    if slope is not None:
        ds.RescaleSlope = slope
    if intercept is not None:
        ds.RescaleIntercept = intercept
    if contrast_agent is not _TAG_ABSENT:
        ds.ContrastBolusAgent = contrast_agent
    ds.Rows = ds.Columns = 8
    ds.BitsAllocated = 16
    ds.BitsStored = bits_stored
    ds.HighBit = bits_stored - 1
    ds.PixelRepresentation = pixel_representation
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    dtype = np.int16 if pixel_representation else np.uint16
    arr = np.full((8, 8), background, dtype=dtype)
    arr[0, 0] = hi
    ds.PixelData = arr.tobytes()
    ds.save_as(path, enforce_file_format=True)
    return path


def test_scale_class_boundaries() -> None:
    assert classify_scale(-1000, 3065) == "standard"
    assert classify_scale(-8192, 14023) == "extended"
    assert classify_scale(-8192, 3797) == "intermediate"
    assert classify_scale(None, None) == "unknown"


def test_extended_scale_is_detected_from_effective_hu(tmp_path: Path) -> None:
    """Intercept alone must not decide it; the effective mapping must."""
    ct = tmp_path / "DICOM" / "CT"
    _ct(ct / "s.dcm", slope=1.0, intercept=-8192.0, hi=16000, desc="MIEDNICA Qr40 S3 iMAR")
    d = describe_planning_ct(ct)
    assert d["acq_scale_class"] == "extended"
    assert d["acq_effective_hu_max"] == pytest.approx(16000 - 8192)
    assert d["acq_imar_present"] is True
    assert d["acq_rescale_intercept"] == pytest.approx(-8192.0)


def test_standard_scale_despite_unusual_intercept(tmp_path: Path) -> None:
    """A large intercept with a compensating slope is still a standard scan.

    slope=4, intercept=-10240: air (stored 2310) -> -1000 HU, and stored 3310 ->
    +3000 HU. The intercept looks alarming; the effective mapping is ordinary.
    """
    ct = tmp_path / "DICOM" / "CT"
    _ct(ct / "s.dcm", slope=4.0, intercept=-10240.0, hi=3310, desc="PELVIS", background=2310)
    d = describe_planning_ct(ct)
    assert d["acq_effective_hu_min"] == pytest.approx(-1000.0)
    assert d["acq_effective_hu_max"] == pytest.approx(3000.0)
    assert d["acq_scale_class"] == "standard"


def test_missing_series_fails_soft(tmp_path: Path) -> None:
    d = describe_planning_ct(tmp_path / "nope")
    assert d["acq_scale_class"] == "unknown"
    assert d["acq_manufacturer"] is None
    assert d["acq_provenance_status"] == "series_unreadable"


def test_full_series_observed_and_representable_ranges_are_distinct(
    tmp_path: Path,
) -> None:
    ct = tmp_path / "DICOM" / "CT"
    series_uid = generate_uid()
    _ct(
        ct / "a_high.dcm",
        slope=1.0,
        intercept=-8192.0,
        hi=16000,
        desc="PELVIS iMAR",
        series_uid=series_uid,
    )
    _ct(
        ct / "m_middle.dcm",
        slope=1.0,
        intercept=-8192.0,
        hi=9000,
        desc="PELVIS iMAR",
        series_uid=series_uid,
    )
    _ct(
        ct / "z_low.dcm",
        slope=1.0,
        intercept=-8192.0,
        hi=8192,
        desc="PELVIS iMAR",
        series_uid=series_uid,
    )

    d = describe_planning_ct(ct, series_instance_uid=series_uid)

    assert d["acq_provenance_status"] == "ok"
    assert d["acq_series_instance_uid"] == series_uid
    assert d["acq_dicom_instance_count"] == 3
    assert d["acq_observed_hu_min"] == pytest.approx(-8192.0)
    assert d["acq_observed_hu_max"] == pytest.approx(7808.0)
    assert d["acq_effective_hu_max"] == d["acq_observed_hu_max"]
    assert d["acq_representable_hu_min"] == pytest.approx(-8192.0)
    assert d["acq_representable_hu_max"] == pytest.approx(57343.0)
    assert d["acq_scale_class"] == "extended"


def test_extensionless_contracted_instance_is_included(tmp_path: Path) -> None:
    ct = tmp_path / "DICOM" / "CT"
    series_uid = generate_uid()
    _ct(
        ct / "slice-001",
        slope=1.0,
        intercept=-1024.0,
        hi=5000,
        desc="PELVIS",
        series_uid=series_uid,
    )

    d = describe_planning_ct(ct, series_instance_uid=series_uid)

    assert d["acq_provenance_status"] == "ok"
    assert d["acq_dicom_instance_count"] == 1
    assert d["acq_observed_hu_max"] == pytest.approx(3976.0)


def test_missing_rescale_mapping_is_explicitly_unknown(tmp_path: Path) -> None:
    ct = tmp_path / "DICOM" / "CT"
    series_uid = generate_uid()
    _ct(
        ct / "slice.dcm",
        slope=None,
        intercept=-1024.0,
        hi=2000,
        desc="PELVIS",
        series_uid=series_uid,
    )

    d = describe_planning_ct(ct, series_instance_uid=series_uid)

    assert d["acq_provenance_status"] == "mapping_metadata_missing"
    assert d["acq_scale_class"] == "unknown"
    assert d["acq_observed_hu_min"] is None
    assert d["acq_observed_hu_max"] is None
    assert d["acq_representable_hu_min"] is None
    assert "RescaleSlope" in d["acq_provenance_detail"]


def test_signed_bits_define_representable_mapping_range(tmp_path: Path) -> None:
    ct = tmp_path / "DICOM" / "CT"
    series_uid = generate_uid()
    _ct(
        ct / "signed.dcm",
        slope=2.0,
        intercept=-100.0,
        hi=100,
        desc="PELVIS",
        series_uid=series_uid,
        bits_stored=12,
        pixel_representation=1,
    )

    d = describe_planning_ct(ct, series_instance_uid=series_uid)

    assert d["acq_representable_hu_min"] == pytest.approx(-4196.0)
    assert d["acq_representable_hu_max"] == pytest.approx(3994.0)


def test_imar_and_contrast_booleans_preserve_false_and_unknown(
    tmp_path: Path,
) -> None:
    known = tmp_path / "known"
    known_uid = generate_uid()
    _ct(
        known / "s.dcm",
        slope=1.0,
        intercept=-1024.0,
        hi=1024,
        desc="PELVIS",
        kernel="B30f",
        contrast_agent="",
        series_uid=known_uid,
    )
    known_descriptor = describe_planning_ct(
        known,
        series_instance_uid=known_uid,
    )
    assert known_descriptor["acq_imar_present"] is False
    assert known_descriptor["acq_contrast_present"] is False

    unknown = tmp_path / "unknown"
    unknown_uid = generate_uid()
    _ct(
        unknown / "s.dcm",
        slope=1.0,
        intercept=-1024.0,
        hi=1024,
        desc=None,
        kernel=None,
        series_uid=unknown_uid,
    )
    unknown_descriptor = describe_planning_ct(
        unknown,
        series_instance_uid=unknown_uid,
    )
    assert unknown_descriptor["acq_imar_present"] is None
    assert unknown_descriptor["acq_contrast_present"] is None


def test_imar_separator_and_incomplete_negative_evidence(tmp_path: Path) -> None:
    positive = tmp_path / "positive"
    positive_uid = generate_uid()
    _ct(
        positive / "s.dcm",
        slope=1.0,
        intercept=-1024.0,
        hi=1024,
        desc="PELVIS_iMAR_3",
        series_uid=positive_uid,
    )
    assert describe_planning_ct(
        positive,
        series_instance_uid=positive_uid,
    )["acq_imar_present"] is True

    partial = tmp_path / "partial"
    partial_uid = generate_uid()
    _ct(
        partial / "a.dcm",
        slope=1.0,
        intercept=-1024.0,
        hi=1024,
        desc="PELVIS",
        kernel="B30f",
        contrast_agent="",
        series_uid=partial_uid,
    )
    _ct(
        partial / "b.dcm",
        slope=1.0,
        intercept=-1024.0,
        hi=1024,
        desc=None,
        kernel=None,
        series_uid=partial_uid,
    )
    descriptor = describe_planning_ct(partial, series_instance_uid=partial_uid)
    assert descriptor["acq_imar_present"] is None
    assert descriptor["acq_contrast_present"] is None


def test_series_uid_mismatch_is_unknown_not_standard(tmp_path: Path) -> None:
    ct = tmp_path / "DICOM" / "CT"
    _ct(
        ct / "wrong.dcm",
        slope=1.0,
        intercept=-1024.0,
        hi=2000,
        desc="PELVIS",
    )

    d = describe_planning_ct(ct, series_instance_uid=generate_uid())

    assert d["acq_provenance_status"] == "series_mismatch"
    assert d["acq_scale_class"] == "unknown"
    assert d["acq_observed_hu_min"] is None


def test_missing_descriptor_is_a_publication_error() -> None:
    with pytest.raises(AcquisitionDescriptorError, match="required"):
        attach_acquisition_descriptor([{"modality": "CT"}], None)


def test_resume_descriptor_must_match_contracted_series(tmp_path: Path) -> None:
    ct = tmp_path / "DICOM" / "CT"
    series_uid = generate_uid()
    _ct(
        ct / "slice.dcm",
        slope=1.0,
        intercept=-1024.0,
        hi=2000,
        desc="PELVIS",
        series_uid=series_uid,
    )
    descriptor = describe_planning_ct(ct, series_instance_uid=series_uid)

    with pytest.raises(AcquisitionDescriptorError, match="not contracted UID"):
        validate_acquisition_descriptor_table(
            pd.DataFrame([descriptor]),
            expected_series_instance_uid=generate_uid(),
        )

    stale_descriptor = dict(descriptor)
    stale_descriptor["acq_observed_hu_max"] = 99999.0
    stale_descriptor["acq_effective_hu_max"] = 99999.0
    with pytest.raises(AcquisitionDescriptorError, match="does not match.*field"):
        validate_acquisition_descriptor_table(
            pd.DataFrame([stale_descriptor]),
            expected_descriptor=descriptor,
            expected_series_instance_uid=series_uid,
        )


def _contracted_scale_course(tmp_path: Path) -> tuple[Path, Path, str]:
    course = tmp_path / "P1" / "C1"
    selected = course / "DICOM" / "selected_planning_ct"
    selected_uid = generate_uid()
    _ct(
        selected / "a_high.dcm",
        slope=1.0,
        intercept=-8192.0,
        hi=16000,
        desc="SELECTED iMAR",
        series_uid=selected_uid,
    )
    _ct(
        selected / "m_low.dcm",
        slope=1.0,
        intercept=-8192.0,
        hi=8192,
        desc="SELECTED iMAR",
        series_uid=selected_uid,
    )
    decoy = course / "DICOM" / "CT"
    _ct(
        decoy / "decoy.dcm",
        slope=1.0,
        intercept=-1024.0,
        hi=3000,
        desc="UNCONTRACTED STANDARD",
    )
    rtstruct = write_synthetic_rtstruct(
        course / "DICOM" / "RTSTRUCT" / "RS.dcm",
        referenced_series_uid=selected_uid,
        roi_names=("PTV",),
    )
    write_minimal_course_contract(
        course,
        authoritative_rtstruct=rtstruct,
        planning_ct_dir=selected,
    )
    return course, selected, selected_uid


class _OneRowExecutor:
    def __init__(self, *_args, **_kwargs):
        pass

    def submit(self, _function, task):
        future = Future()
        future.set_result(
            radiomics_parallel._status_records(task, "success", "test success")
        )
        return future

    def shutdown(self, *_args, **_kwargs):
        pass


def _config(tmp_path: Path, *, resume: bool = False) -> PipelineConfig:
    return PipelineConfig(
        dicom_root=tmp_path / "dicom",
        output_root=tmp_path / "out",
        logs_root=tmp_path / "logs",
        max_workers_override=1,
        radiomics_min_voxels=1,
        resume=resume,
    )


def test_default_parallel_backend_publishes_contracted_full_series_descriptor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    course, selected, selected_uid = _contracted_scale_course(tmp_path)
    seen_ct_dirs: list[Path] = []

    def fake_load(ct_dir: Path):
        seen_ct_dirs.append(Path(ct_dir))
        return object()

    monkeypatch.setattr(radiomics, "_load_series_image", fake_load)
    monkeypatch.setattr(radiomics, "_extractor", lambda *_a, **_k: object())
    monkeypatch.setattr(
        radiomics_parallel,
        "effective_parameter_hashes_for_arms",
        lambda *_args, **_kwargs: {
            "primary_resegmented": "effective-primary",
            "sensitivity_raw": "effective-sensitivity",
        },
    )
    monkeypatch.setattr(radiomics_parallel, "ProcessPoolExecutor", _OneRowExecutor)
    monkeypatch.setattr(radiomics_parallel, "_calculate_optimal_workers", lambda: 1)

    outcome = radiomics_parallel.parallel_radiomics_for_course(
        _config(tmp_path),
        course,
        max_workers=1,
    )

    assert outcome.output_path is not None
    result = pd.read_excel(outcome.output_path, engine="openpyxl")
    assert seen_ct_dirs == [selected]
    assert result.loc[0, "acq_series_instance_uid"] == selected_uid
    assert result.loc[0, "acq_provenance_status"] == "ok"
    assert result.loc[0, "acq_observed_hu_max"] == pytest.approx(7808.0)
    assert result.loc[0, "acq_scale_class"] == "extended"
    assert result.loc[0, "acq_series_description"] == "SELECTED iMAR"

    result.loc[0, "acq_observed_hu_max"] = 99999.0
    result.loc[0, "acq_effective_hu_max"] = 99999.0
    result.to_excel(outcome.output_path, index=False)
    resumed = radiomics_parallel.parallel_radiomics_for_course(
        _config(tmp_path, resume=True),
        course,
        max_workers=1,
    )
    assert resumed.output_path is not None
    refreshed = pd.read_parquet(
        resumed.output_path.with_suffix(".parquet"), engine="pyarrow"
    )
    assert refreshed.loc[0, "acq_observed_hu_max"] == pytest.approx(7808.0)
    assert seen_ct_dirs == [selected, selected]


def test_native_backend_attaches_descriptor_once_after_feature_extraction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    course, _selected, selected_uid = _contracted_scale_course(tmp_path)

    class Image:
        def GetSpacing(self):
            return (1.0, 1.0, 1.0)

    class Extractor:
        def __init__(self):
            self.settings = {"minimumROISize": 1, "minimumROIDimensions": 1}
            self.enabledFeatures = {"shape": [], "firstorder": []}

        def disableAllImageTypes(self):
            return None

        def enableImageTypeByName(self, *_args, **_kwargs):
            return None

        def disableAllFeatures(self):
            self.enabledFeatures = {}

        def enableFeatureClassByName(self, name):
            self.enabledFeatures = {name: []}

        def execute(self, _image, _mask):
            if set(self.enabledFeatures) == {"shape"}:
                return {"original_shape_MeshVolume": 8.0}
            return {"original_firstorder_Mean": 2.0}

    monkeypatch.setattr(radiomics, "_load_series_image", lambda *_a, **_k: Image())
    monkeypatch.setattr(radiomics, "_extractor", lambda *_a, **_k: Extractor())
    monkeypatch.setattr(
        radiomics,
        "_rtstruct_masks",
        lambda *_a, **_k: {"PTV": np.ones((2, 2, 2), dtype=bool)},
    )
    monkeypatch.setattr(radiomics, "_mask_from_array_like", lambda *_a, **_k: object())
    monkeypatch.setattr(
        radiomics_ct_contract,
        "resampled_mask_qc",
        lambda *_a, **_k: {
            "morphologic_resampled_voxel_count": 8,
            "resegment_after_count": 8,
            "resegment_below_lower_count": 0,
            "resegment_above_upper_count": 0,
            "resegment_nonfinite_count": 0,
            "components_26_before": 1,
            "components_26_after": 1,
            "largest_component_voxel_count_before": 8,
            "largest_component_voxel_count_after": 8,
            "largest_component_retained_fraction": 1.0,
            "resegment_retained_fraction": 1.0,
            "largest_component_fraction_after": 1.0,
            "component_count_increased": False,
            "observed_roi_dimensions_after_resegmentation": 3,
        },
    )
    monkeypatch.setattr(
        radiomics_ct_contract,
        "_runtime_versions",
        lambda: {
            "pyradiomics_version": "test",
            "simpleitk_version": "test",
            "numpy_version": np.__version__,
        },
    )
    monkeypatch.setattr(
        radiomics,
        "run_tasks_with_adaptive_workers",
        lambda _label, tasks, function, **_kwargs: [function(task) for task in tasks],
    )

    outcome = radiomics.radiomics_for_course(_config(tmp_path), course)

    assert outcome.output_path is not None
    result = pd.read_excel(outcome.output_path, engine="openpyxl")
    assert result.loc[0, "acq_series_instance_uid"] == selected_uid
    assert result.loc[0, "acq_observed_hu_max"] == pytest.approx(7808.0)


def test_conda_batch_requires_and_attaches_ct_descriptor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    course, selected, selected_uid = _contracted_scale_course(tmp_path)
    descriptor = describe_planning_ct(
        selected,
        series_instance_uid=selected_uid,
    )
    tasks = [
        {
            "image_path": "image.nrrd",
            "mask_path": "mask.nrrd",
            "roi_name": "PTV",
            "cleanup": False,
            "metadata": {
                "modality": "CT",
                "segmentation_source": "Manual",
                "roi_original_name": "PTV",
                "course_dir": str(course),
                "patient_id": "P1",
                "course_id": "C1",
            },
        }
    ]
    monkeypatch.setattr(radiomics_conda, "check_radiomics_env", lambda **_kwargs: True)
    monkeypatch.setattr(
        radiomics_conda,
        "extract_radiomics_batch_with_conda",
        lambda _tasks, params_file=None: [
            {
                "__status__": "success",
                "__task_index__": 0,
                "original_firstorder_Mean": 3.0,
            }
        ],
    )

    missing_output = tmp_path / "missing" / "radiomics_ct.xlsx"
    with pytest.raises(RadiomicsCourseExtractionError, match="descriptor"):
        radiomics_conda.process_radiomics_batch(
            tasks,
            missing_output,
            sequential=True,
            max_workers=1,
            enable_heartbeat=False,
        )

    output = tmp_path / "with_descriptor" / "radiomics_ct.xlsx"
    result_path = radiomics_conda.process_radiomics_batch(
        tasks,
        output,
        sequential=True,
        max_workers=1,
        enable_heartbeat=False,
        acquisition_descriptor=descriptor,
    )

    assert result_path == output
    result = pd.read_excel(output, engine="openpyxl")
    assert result.loc[0, "acq_series_instance_uid"] == selected_uid
    assert result.loc[0, "acq_observed_hu_max"] == pytest.approx(7808.0)
