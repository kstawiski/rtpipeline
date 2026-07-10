"""Regression tests for three science-critical fixes (batch C):

G - roi_fixer.py: swapped row/column direction cosines in `_world_to_pixel`,
    which transposed x/y pixel coordinates for the rt-utils rasterization fallback.
B - organize.py: a `DoseClassification` that legitimately selects no plans caused
    `_process_course` to substitute EVERY course plan for summation, reintroducing
    ITT-excluded replans.
F - auto_rtstruct.py / custom_structures_rtstruct.py: RS_auto.dcm/RS_custom.dcm were
    written non-atomically and resume checks were existence-only, so a process killed
    mid-write left a truncated file that resume accepted forever (the same class of bug
    3bd8c5d fixed for segmentation masks/manifests).
F2 - utils.sanitize_rtstruct / roi_fixer.fix_rtstruct_rois: post-processing rewrites of
    an already-published RTSTRUCT (sanitization, ROI rebuild) wrote straight to the
    destination path, so a kill mid-rewrite still truncates the final file even though
    the initial write is atomic. Both must publish via temp+os.replace() too.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, RTDoseStorage, RTPlanStorage, generate_uid

import rtpipeline.auto_rtstruct as auto_rtstruct
import rtpipeline.custom_structures_rtstruct as custom_structures_rtstruct
import rtpipeline.organize as organize
import rtpipeline.roi_fixer as roi_fixer
import rtpipeline.utils as utils
from rtpipeline.config import PipelineConfig
from rtpipeline.layout import build_course_dirs
from rtpipeline.metadata import LinkedSet
from rtpipeline.organize import DoseClassification
from rtpipeline.rt_details import DoseInfo, PlanInfo

_RTSTRUCT_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.481.3"


# ---------------------------------------------------------------------------
# Shared DICOM fixture helpers (pattern follows tests/test_dvh_dose_resolution.py)
# ---------------------------------------------------------------------------


def _file_dataset(path: Path, sop_class_uid: str, sop_instance_uid: str) -> FileDataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class_uid
    meta.MediaStorageSOPInstanceUID = sop_instance_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = sop_class_uid
    ds.SOPInstanceUID = sop_instance_uid
    ds.PatientID = "P1"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    return ds


def _write(ds: FileDataset, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), write_like_original=False)
    return path


def _mk_plan(path: Path, plan_uid: str, *, label: str = "plan", plan_date: str = "20240101",
             rx_gy: float | None = None) -> Path:
    ds = _file_dataset(path, RTPlanStorage, plan_uid)
    ds.Modality = "RTPLAN"
    ds.RTPlanLabel = label
    ds.RTPlanName = label
    ds.RTPlanDate = plan_date
    ds.FrameOfReferenceUID = "1.2.826.0.1.3680043.8.498.1"
    if rx_gy is not None:
        dose_ref = Dataset()
        dose_ref.DoseReferenceType = "TARGET"
        dose_ref.TargetPrescriptionDose = rx_gy
        ds.DoseReferenceSequence = Sequence([dose_ref])
    return _write(ds, path)


def _mk_dose(path: Path, dose_uid: str, plan_uid: str, summation_type: str = "PLAN") -> Path:
    ds = _file_dataset(path, RTDoseStorage, dose_uid)
    ds.Modality = "RTDOSE"
    ds.FrameOfReferenceUID = "1.2.826.0.1.3680043.8.498.1"
    ds.DoseSummationType = summation_type
    ref = Dataset()
    ref.ReferencedSOPClassUID = RTPlanStorage
    ref.ReferencedSOPInstanceUID = plan_uid
    ds.ReferencedRTPlanSequence = Sequence([ref])
    return _write(ds, path)


def _mk_dose_multi_plan(path: Path, dose_uid: str, plan_uids: list[str], summation_type: str = "PLAN") -> Path:
    """A dose (e.g. a real PLAN_SUM) whose ReferencedRTPlanSequence lists MULTIPLE
    plans -- used to exercise `_plan_paths_for_doses`' multi-plan resolution."""
    ds = _file_dataset(path, RTDoseStorage, dose_uid)
    ds.Modality = "RTDOSE"
    ds.FrameOfReferenceUID = "1.2.826.0.1.3680043.8.498.1"
    ds.DoseSummationType = summation_type
    refs = Sequence()
    for plan_uid in plan_uids:
        ref = Dataset()
        ref.ReferencedSOPClassUID = RTPlanStorage
        ref.ReferencedSOPInstanceUID = plan_uid
        refs.append(ref)
    ds.ReferencedRTPlanSequence = refs
    return _write(ds, path)


def _write_valid_rtstruct(path: Path, roi_names: list[str]) -> Path:
    """Write a minimal, real, parseable RTSTRUCT DICOM file (>=0 ROIs)."""
    sop_uid = generate_uid()
    ds = _file_dataset(path, _RTSTRUCT_SOP_CLASS_UID, sop_uid)
    ds.Modality = "RTSTRUCT"
    rois = Sequence()
    for i, name in enumerate(roi_names, start=1):
        roi = Dataset()
        roi.ROINumber = i
        roi.ROIName = name
        rois.append(roi)
    ds.StructureSetROISequence = rois
    return _write(ds, path)


def _mk_ct_slice(path: Path, *, series_uid: str, study_uid: str, frame_of_ref_uid: str,
                  z: float, rows: int = 4, columns: int = 4) -> Path:
    """A minimal but real CT slice with pixel data, sufficient for rt_utils'
    RTStructBuilder.create_new (load_dcm_images_from_path + slice-position sort)."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "CT"
    ds.PatientID = "P1"
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.FrameOfReferenceUID = frame_of_ref_uid
    ds.StudyDate = "20240101"
    ds.StudyTime = "093000"
    ds.StudyID = "1"
    ds.SeriesNumber = 1
    ds.InstanceNumber = int(z) + 1
    ds.Rows = rows
    ds.Columns = columns
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, z]
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1024.0
    ds.PixelData = np.full((rows, columns), 0, dtype=np.int16).tobytes()
    ds.save_as(str(path), enforce_file_format=True)
    return path


def _build_real_rtstruct(ct_dir: Path, *, n_slices: int = 4, side: int = 4):
    """Build a real rt_utils RTStruct (RTStructBuilder.create_new + add_roi) against a
    tiny synthetic CT series -- exercises the actual RTStruct.save() auto-'.dcm'-append
    behavior that a lambda-writer test cannot reach."""
    from rt_utils import RTStructBuilder

    series_uid = generate_uid()
    study_uid = generate_uid()
    frame_of_ref_uid = generate_uid()
    for i in range(n_slices):
        _mk_ct_slice(
            ct_dir / f"ct_{i}.dcm",
            series_uid=series_uid, study_uid=study_uid, frame_of_ref_uid=frame_of_ref_uid,
            z=float(i), rows=side, columns=side,
        )
    rtstruct = RTStructBuilder.create_new(dicom_series_path=str(ct_dir))
    mask = np.zeros((side, side, n_slices), dtype=bool)
    mask[1:-1, 1:-1, 1:-1] = True
    rtstruct.add_roi(mask=mask, name="PTV")
    return rtstruct


# ===========================================================================
# G - roi_fixer.py: swapped direction cosines in _world_to_pixel
# ===========================================================================


def _make_rebuilder(pixel_spacing, orientation) -> roi_fixer._ROIRebuilder:
    ds = pydicom.dataset.Dataset()
    slice_ds = pydicom.dataset.Dataset()
    slice_ds.PixelSpacing = list(pixel_spacing)
    slice_ds.ImagePositionPatient = [0.0, 0.0, 0.0]
    slice_ds.ImageOrientationPatient = list(orientation)
    fake_rtstruct = SimpleNamespace(ds=ds, series_data=[slice_ds])
    return roi_fixer._ROIRebuilder(fake_rtstruct)


def test_world_to_pixel_uses_correct_row_col_cosine_mapping():
    # Non-square pixel spacing (row spacing != column spacing) plus a non-symmetric
    # (scalene) triangle of world points: correct vs. swapped row/col mapping produce
    # different pixel coordinates for every point here, so a transposition bug cannot
    # accidentally pass.
    rebuilder = _make_rebuilder(
        pixel_spacing=[2.0, 1.0],  # [row spacing, column spacing]
        orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # standard axial IOP
    )
    world_coords = np.array(
        [
            [10.0, 4.0, 0.0],
            [10.0, 8.0, 0.0],
            [16.0, 4.0, 0.0],
        ],
        dtype=float,
    )

    pixels = rebuilder._world_to_pixel(world_coords, 0)

    # Per DICOM PS3.3 C.7.6.2.1.1: column index (x) = dot(translated, IOP[0:3]) / spacing[1];
    # row index (y) = dot(translated, IOP[3:6]) / spacing[0].
    # x = translated_x / col_spacing(1.0); y = translated_y / row_spacing(2.0)
    expected = np.array([[10, 2], [10, 4], [16, 2]], dtype=np.int32)
    # The pre-fix (swapped) formula would instead compute x = translated_y / 1.0 and
    # y = translated_x / 2.0, giving [[4, 5], [8, 5], [4, 8]] - verifiably different.
    swapped = np.array([[4, 5], [8, 5], [4, 8]], dtype=np.int32)

    assert np.array_equal(pixels, expected)
    assert not np.array_equal(pixels, swapped)


# ===========================================================================
# B - organize.py: fail-closed plan selection when DoseClassification.selected_plans
#     is legitimately empty (must not substitute every course plan for summation)
# ===========================================================================


def test_plan_paths_for_doses_matches_only_the_referenced_plan(tmp_path):
    plan_uid_1 = generate_uid()
    plan_uid_2 = generate_uid()
    plan1 = _mk_plan(tmp_path / "RP1.dcm", plan_uid_1)
    plan2 = _mk_plan(tmp_path / "RP2.dcm", plan_uid_2)
    dose = _mk_dose(tmp_path / "RD1.dcm", generate_uid(), plan_uid_1)

    result = organize._plan_paths_for_doses([plan1, plan2], [dose])

    assert result == [plan1]


def test_plan_paths_for_doses_returns_empty_when_no_plan_matches(tmp_path):
    plan_uid_1 = generate_uid()
    plan_uid_2 = generate_uid()
    plan1 = _mk_plan(tmp_path / "RP1.dcm", plan_uid_1)
    plan2 = _mk_plan(tmp_path / "RP2.dcm", plan_uid_2)
    # Dose references a plan UID that matches neither plan1 nor plan2.
    dose = _mk_dose(tmp_path / "RD1.dcm", generate_uid(), generate_uid())

    result = organize._plan_paths_for_doses([plan1, plan2], [dose])

    assert result == []


def _run_organize_with_fake_classification(tmp_path, monkeypatch, *, plan1, plan2, dose, fake_classification):
    """Drive `organize_and_merge` for a single synthetic course, bypassing real DICOM
    discovery/linking (mocked) but exercising the REAL `_process_course` dose/plan
    selection logic (including the fix under test) and REAL dose-classification consumers.
    """
    dicom_root = tmp_path / "empty_dicom_root"
    dicom_root.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig(
        dicom_root=dicom_root,
        output_root=tmp_path / "out",
        logs_root=tmp_path / "logs",
        max_workers_override=1,
        # Disabled so the canonical RP.dcm/RD.dcm copies below are never SOP-deduped
        # against the raw per-plan copies made earlier in _process_course - unrelated
        # to the fix under test here.
        dicom_copy_dedup_by_sop_uid=False,
    )

    plan_info_1 = PlanInfo(
        path=plan1, patient_id="P1", sop_instance_uid="unused", study_uid=None,
        plan_label="plan1", plan_name="plan1", plan_date="20240101",
        frame_of_reference_uid="FOR1",
    )
    plan_info_2 = PlanInfo(
        path=plan2, patient_id="P1", sop_instance_uid="unused", study_uid=None,
        plan_label="plan2", plan_name="plan2", plan_date="20240201",
        frame_of_reference_uid="FOR1",
    )
    dose_info = DoseInfo(
        path=dose, patient_id="P1", sop_instance_uid="unused", study_uid=None,
        frame_of_reference_uid="FOR1", referenced_plan_sop=None,
    )
    items = [
        LinkedSet(patient_id="P1", plan=plan_info_1, dose=dose_info, struct=None,
                   ct_study_uid=None, frame_of_reference_uid="FOR1"),
        LinkedSet(patient_id="P1", plan=plan_info_2, dose=dose_info, struct=None,
                   ct_study_uid=None, frame_of_reference_uid="FOR1"),
    ]
    fixed_courses = {("P1", "course1"): items}
    monkeypatch.setattr(organize, "group_by_course", lambda *a, **k: fixed_courses)
    monkeypatch.setattr(
        organize, "_classify_doses",
        lambda plan_paths, dose_paths, max_total_dose_gy=100.0: fake_classification,
    )

    summed_calls: list[list[Path]] = []

    class _FakeSummedPlan:
        SOPInstanceUID = "1.2.FAKE.SUMMED"

        def save_as(self, path):
            Path(path).write_bytes(b"FAKE-SUMMED-PLAN")

    def _fake_create_summed_plan(plan_files, total_dose_gy=None):
        summed_calls.append(list(plan_files))
        return _FakeSummedPlan(), [], []

    monkeypatch.setattr(organize, "_create_summed_plan", _fake_create_summed_plan)

    outputs = organize.organize_and_merge(config)
    assert len(outputs) == 1
    return outputs[0], summed_calls


def test_empty_selected_plans_matched_via_dose_reference_does_not_sum_all_plans(tmp_path, monkeypatch):
    """selected_plans=[] but the single selected dose's ReferencedRTPlanSequence
    resolves to exactly one of the course's plans -> that plan must be used directly,
    never summed with the (ITT-excluded) other plan."""
    plan_uid_1 = generate_uid()
    plan_uid_2 = generate_uid()
    plan1 = _mk_plan(tmp_path / "src" / "RP1.dcm", plan_uid_1, plan_date="20240101")
    plan2 = _mk_plan(tmp_path / "src" / "RP2.dcm", plan_uid_2, plan_date="20240201")
    dose = _mk_dose(tmp_path / "src" / "RD1.dcm", generate_uid(), plan_uid_1)

    fake_classification = DoseClassification(
        classification="replan_itt_first",
        selected_doses=[dose],
        selected_plans=[],  # classifier legitimately found no linked plan
        excluded_doses=[],
        should_sum=False,
        warnings=["synthetic: classifier found no linked plan"],
        reason="synthetic test",
    )

    course_output, summed_calls = _run_organize_with_fake_classification(
        tmp_path, monkeypatch, plan1=plan1, plan2=plan2, dose=dose,
        fake_classification=fake_classification,
    )

    assert summed_calls == [], "must not synthesize a summed plan when only one dose was selected"
    assert course_output.rp_path is not None and course_output.rp_path.exists()
    saved_uid = str(pydicom.dcmread(str(course_output.rp_path), stop_before_pixels=True).SOPInstanceUID)
    assert saved_uid == plan_uid_1
    assert saved_uid != plan_uid_2


def test_empty_selected_plans_unmatched_falls_back_to_single_plan_not_all(tmp_path, monkeypatch):
    """selected_plans=[] and the dose's ReferencedRTPlanSequence matches NEITHER course
    plan (fully unresolved) -> fail closed to the single earliest course plan, never to
    a synthesized sum of every plan in the course."""
    plan_uid_1 = generate_uid()
    plan_uid_2 = generate_uid()
    plan1 = _mk_plan(tmp_path / "src" / "RP1.dcm", plan_uid_1, plan_date="20240101")
    plan2 = _mk_plan(tmp_path / "src" / "RP2.dcm", plan_uid_2, plan_date="20240201")
    dose = _mk_dose(tmp_path / "src" / "RD1.dcm", generate_uid(), generate_uid())  # unresolved reference

    fake_classification = DoseClassification(
        classification="PLAN_SUM_used",
        selected_doses=[dose],
        selected_plans=[],  # classifier legitimately found no linked plan
        excluded_doses=[],
        should_sum=False,
        warnings=["synthetic: PLAN_SUM references an unresolved plan UID"],
        reason="synthetic test",
    )

    course_output, summed_calls = _run_organize_with_fake_classification(
        tmp_path, monkeypatch, plan1=plan1, plan2=plan2, dose=dose,
        fake_classification=fake_classification,
    )

    assert summed_calls == [], "must not synthesize a summed plan over the full course plan set"
    assert course_output.rp_path is not None and course_output.rp_path.exists()
    saved_uid = str(pydicom.dcmread(str(course_output.rp_path), stop_before_pixels=True).SOPInstanceUID)
    # Falls back to the single earliest (first/ITT) course plan, not a sum of both.
    assert saved_uid == plan_uid_1
    assert saved_uid != plan_uid_2


def test_resolved_multi_plan_case_computes_total_rx(tmp_path, monkeypatch):
    """B hardening (b): dose_classification.selected_plans=[] but the single selected
    dose's ReferencedRTPlanSequence resolves (via _plan_paths_for_doses) to BOTH course
    plans (e.g. a real PLAN_SUM referencing both) -> total_rx must be computed from
    that FINAL resolved multi-plan selected_plans, not left None because the classifier's
    own (empty) selected_plans gated the old total_rx computation."""
    plan_uid_1 = generate_uid()
    plan_uid_2 = generate_uid()
    plan1 = _mk_plan(tmp_path / "src" / "RP1.dcm", plan_uid_1, plan_date="20240101", rx_gy=20.0)
    plan2 = _mk_plan(tmp_path / "src" / "RP2.dcm", plan_uid_2, plan_date="20240201", rx_gy=30.0)
    # A single dose whose ReferencedRTPlanSequence lists BOTH plans (a genuine PLAN_SUM).
    dose = _mk_dose_multi_plan(tmp_path / "src" / "RD1.dcm", generate_uid(), [plan_uid_1, plan_uid_2])

    fake_classification = DoseClassification(
        classification="PLAN_SUM_used",
        selected_doses=[dose],
        selected_plans=[],  # classifier itself didn't resolve the multi-plan reference
        excluded_doses=[],
        should_sum=False,
        warnings=["synthetic: classifier left selected_plans empty for a multi-ref PLAN_SUM"],
        reason="synthetic test",
    )

    course_output, summed_calls = _run_organize_with_fake_classification(
        tmp_path, monkeypatch, plan1=plan1, plan2=plan2, dose=dose,
        fake_classification=fake_classification,
    )

    # Resolved via _plan_paths_for_doses to both plans -> the "multiple plans but single
    # dose" branch creates a summed plan (unlike the single-unresolved-plan B tests above).
    assert summed_calls == [[plan1, plan2]]
    assert course_output.total_prescription_gy is not None, (
        "total_rx must be computed from the resolved multi-plan selected_plans, not left "
        "None because dose_classification.selected_plans (the classifier's own, empty, "
        "pre-resolution list) gated the computation"
    )
    # sum_all is False here (should_sum=False), matching _infer_rx_from_plan_paths'
    # existing sum_all semantics: the first resolved plan's Rx (date-earliest: plan1).
    assert course_output.total_prescription_gy == pytest.approx(20.0)


def test_earliest_dated_plan_path_skips_missing_date_plans():
    """B hardening (a): items_sorted (sorted by `plan_date or ""`) puts missing-date
    plans FIRST (an empty string sorts before any real date string), so plan_paths[0]
    is not reliably the chronologically earliest plan. The ITT fallback must pick the
    earliest plan that actually HAS a date."""
    plan_missing = PlanInfo(
        path=Path("/plan_missing.dcm"), patient_id="P1", sop_instance_uid="u0",
        study_uid=None, plan_label="missing", plan_name="missing", plan_date=None,
        frame_of_reference_uid="FOR1",
    )
    plan_early = PlanInfo(
        path=Path("/plan_early.dcm"), patient_id="P1", sop_instance_uid="u1",
        study_uid=None, plan_label="early", plan_name="early", plan_date="20240101",
        frame_of_reference_uid="FOR1",
    )
    plan_late = PlanInfo(
        path=Path("/plan_late.dcm"), patient_id="P1", sop_instance_uid="u2",
        study_uid=None, plan_label="late", plan_name="late", plan_date="20240301",
        frame_of_reference_uid="FOR1",
    )
    dose = DoseInfo(path=Path("/dose.dcm"), patient_id="P1", sop_instance_uid="ud",
                     study_uid=None, frame_of_reference_uid="FOR1", referenced_plan_sop=None)

    def _item(plan):
        return LinkedSet(patient_id="P1", plan=plan, dose=dose, struct=None,
                          ct_study_uid=None, frame_of_reference_uid="FOR1")

    items = [_item(plan_missing), _item(plan_early), _item(plan_late)]
    # Same sort _process_course applies: missing-date plans ("") sort first.
    items_sorted = sorted(items, key=lambda it: it.plan.plan_date or "")
    plan_paths = [it.plan.path for it in items_sorted]
    assert plan_paths[0] == plan_missing.path, "sanity check: missing-date plan sorts first"

    result = organize._earliest_dated_plan_path(items_sorted, plan_paths)

    assert result == plan_early.path, (
        "must pick the earliest VALID-date plan, not the missing-date one that sorts first"
    )


def test_earliest_dated_plan_path_falls_back_when_no_plan_has_a_date():
    """B hardening (a), documented limitation: with NO date info on any plan, there is
    no way to order them chronologically, so it falls back to plan_paths[0]."""
    plan_a = PlanInfo(path=Path("/a.dcm"), patient_id="P1", sop_instance_uid="u0",
                       study_uid=None, plan_label="a", plan_name="a", plan_date=None,
                       frame_of_reference_uid="FOR1")
    plan_b = PlanInfo(path=Path("/b.dcm"), patient_id="P1", sop_instance_uid="u1",
                       study_uid=None, plan_label="b", plan_name="b", plan_date=None,
                       frame_of_reference_uid="FOR1")
    dose = DoseInfo(path=Path("/dose.dcm"), patient_id="P1", sop_instance_uid="ud",
                     study_uid=None, frame_of_reference_uid="FOR1", referenced_plan_sop=None)
    items = [
        LinkedSet(patient_id="P1", plan=plan_a, dose=dose, struct=None,
                  ct_study_uid=None, frame_of_reference_uid="FOR1"),
        LinkedSet(patient_id="P1", plan=plan_b, dose=dose, struct=None,
                  ct_study_uid=None, frame_of_reference_uid="FOR1"),
    ]
    plan_paths = [plan_a.path, plan_b.path]

    result = organize._earliest_dated_plan_path(items, plan_paths)

    assert result == plan_paths[0]


# ===========================================================================
# F - auto_rtstruct.py / custom_structures_rtstruct.py: atomic write + parse-validated resume
# ===========================================================================


@pytest.mark.parametrize("module", [auto_rtstruct, custom_structures_rtstruct])
def test_is_valid_rtstruct_rejects_garbage(tmp_path, module):
    path = tmp_path / "RS.dcm"
    path.write_bytes(b"truncated mid write, not a real dicom file")
    assert module._is_valid_rtstruct(path) is False


@pytest.mark.parametrize("module", [auto_rtstruct, custom_structures_rtstruct])
def test_is_valid_rtstruct_rejects_empty_roi_sequence(tmp_path, module):
    path = tmp_path / "RS.dcm"
    _write_valid_rtstruct(path, roi_names=[])
    assert module._is_valid_rtstruct(path) is False


@pytest.mark.parametrize("module", [auto_rtstruct, custom_structures_rtstruct])
def test_is_valid_rtstruct_accepts_real_rtstruct(tmp_path, module):
    path = tmp_path / "RS.dcm"
    _write_valid_rtstruct(path, roi_names=["PTV"])
    assert module._is_valid_rtstruct(path) is True


@pytest.mark.parametrize("module", [auto_rtstruct, custom_structures_rtstruct])
def test_write_rtstruct_atomic_leaves_no_partial_file(tmp_path, module):
    out_path = tmp_path / "RS.dcm"
    module._write_rtstruct_atomic(out_path, lambda p: Path(p).write_bytes(b"CONTENT"))
    assert out_path.read_bytes() == b"CONTENT"
    assert list(tmp_path.glob(".*.tmp*")) == []


@pytest.mark.parametrize("module", [auto_rtstruct, custom_structures_rtstruct])
def test_write_rtstruct_atomic_failure_leaves_no_partial_file(tmp_path, module):
    out_path = tmp_path / "RS.dcm"

    def _boom(_p):
        raise OSError("simulated crash mid-write")

    with pytest.raises(OSError):
        module._write_rtstruct_atomic(out_path, _boom)

    assert not out_path.exists()
    assert list(tmp_path.glob(".*.tmp*")) == []
    assert list(tmp_path.glob("*.tmp*")) == []


@pytest.mark.parametrize("module", [auto_rtstruct, custom_structures_rtstruct])
def test_write_rtstruct_atomic_temp_path_ends_in_dcm(tmp_path, module):
    """The temp path itself must end in .dcm: rt_utils' RTStruct.save() auto-appends
    '.dcm' to any path that doesn't already end with it, so a temp path ending only
    in '.tmp' gets silently redirected to '<tmp>.dcm' and the following os.replace
    raises FileNotFoundError (this is the exact live-reproduced F1 regression)."""
    out_path = tmp_path / "RS.dcm"
    captured: dict = {}

    def _capture_and_write(p: str) -> None:
        captured["tmp_path"] = p
        Path(p).write_bytes(b"CONTENT")

    module._write_rtstruct_atomic(out_path, _capture_and_write)
    assert captured["tmp_path"].endswith(".dcm")


@pytest.mark.parametrize("module", [auto_rtstruct, custom_structures_rtstruct])
def test_write_rtstruct_atomic_drives_real_rt_utils_save_end_to_end(tmp_path, module):
    """F1 CRITICAL regression (live-reproduced): rt_utils' real RTStruct.save() auto-
    appends '.dcm' to any path not already ending in it. A lambda-writer test cannot
    exercise this -- it must drive the actual rt_utils save() through
    _write_rtstruct_atomic against a real RTStruct built from a synthetic CT series.

    Pre-fix, the temp path ended in '.tmp' (no '.dcm'), so save() silently wrote to
    '<tmp>.dcm' instead, and os.replace(tmp_path, out_path) raised FileNotFoundError:
    RS_custom.dcm 100% failed, RS_auto.dcm silently returned None (caught upstream),
    and an orphan '.tmp.dcm' was left behind. This test fails on the un-remediated
    '.tmp' (no trailing .dcm) temp-path suffix.
    """
    ct_dir = tmp_path / "ct"
    ct_dir.mkdir()
    rtstruct = _build_real_rtstruct(ct_dir)

    out_path = tmp_path / "RS.dcm"
    module._write_rtstruct_atomic(out_path, rtstruct.save)

    assert out_path.exists()
    ds = pydicom.dcmread(str(out_path))
    assert str(ds.SOPClassUID) == _RTSTRUCT_SOP_CLASS_UID
    assert len(ds.StructureSetROISequence) == 1
    # No orphan temp file (of any suffix) left behind.
    assert list(tmp_path.glob(".*tmp*")) == []


def test_build_auto_rtstruct_resume_rejects_truncated_file(tmp_path, monkeypatch):
    course_dir = tmp_path / "course"
    course_dirs = build_course_dirs(course_dir)
    course_dirs.dicom_ct.mkdir(parents=True, exist_ok=True)
    out_path = course_dir / "RS_auto.dcm"
    out_path.write_bytes(b"truncated mid write, not a real dicom file")

    load_calls: list[Path] = []
    monkeypatch.setattr(
        auto_rtstruct, "_load_ct_image",
        lambda ct_dir: (load_calls.append(ct_dir), None)[1],
    )

    result = auto_rtstruct.build_auto_rtstruct(course_dir)

    assert load_calls, "a truncated existing RS_auto.dcm must trigger regeneration, not a silent skip"
    assert result != out_path
    assert result is None  # no real CT/segmentation fixture, so regen can't complete - but it must NOT
    # return the stale/truncated file as-is (the pre-fix `if out_path.exists(): return out_path` would).


def test_build_auto_rtstruct_resume_accepts_valid_file_without_regenerating(tmp_path, monkeypatch):
    course_dir = tmp_path / "course"
    course_dirs = build_course_dirs(course_dir)
    course_dirs.dicom_ct.mkdir(parents=True, exist_ok=True)
    out_path = course_dir / "RS_auto.dcm"
    _write_valid_rtstruct(out_path, roi_names=["PTV"])

    def _fail_if_called(_ct_dir):
        raise AssertionError("must not attempt regeneration for a valid existing RS_auto.dcm")

    monkeypatch.setattr(auto_rtstruct, "_load_ct_image", _fail_if_called)

    result = auto_rtstruct.build_auto_rtstruct(course_dir)

    assert result == out_path


def test_is_rs_custom_stale_rejects_truncated_file(tmp_path):
    course_dir = tmp_path / "course"
    course_dir.mkdir(parents=True, exist_ok=True)
    rs_custom_path = course_dir / "RS_custom.dcm"
    rs_custom_path.write_bytes(b"truncated mid write, not a real dicom file")

    # No other staleness trigger (no config, no rs_manual/rs_auto, no seg dirs) - the
    # pre-fix mtime/metadata-only check would have returned False (accepted forever).
    assert custom_structures_rtstruct._is_rs_custom_stale(rs_custom_path, None, None, None) is True


def test_is_rs_custom_stale_accepts_valid_up_to_date_file(tmp_path):
    course_dir = tmp_path / "course"
    course_dir.mkdir(parents=True, exist_ok=True)
    rs_custom_path = course_dir / "RS_custom.dcm"
    _write_valid_rtstruct(rs_custom_path, roi_names=["pelvic_bones"])

    assert custom_structures_rtstruct._is_rs_custom_stale(rs_custom_path, None, None, None) is False


# ===========================================================================
# F2 - post-processing rewrites of an already-published RTSTRUCT must also be atomic
# ===========================================================================


def _write_rtstruct_with_degenerate_contour(path: Path) -> Path:
    """A real, parseable RTSTRUCT with one ROI whose only contour has 1 point
    (< sanitize_rtstruct's minimum_points=3), so sanitize_rtstruct drops it and
    triggers its save/rewrite path (`changed=True`)."""
    sop_uid = generate_uid()
    ds = _file_dataset(path, _RTSTRUCT_SOP_CLASS_UID, sop_uid)
    ds.Modality = "RTSTRUCT"
    roi = Dataset()
    roi.ROINumber = 1
    roi.ROIName = "PTV"
    ds.StructureSetROISequence = Sequence([roi])

    degenerate_contour = Dataset()
    degenerate_contour.ContourData = [0.0, 0.0, 0.0]  # 1 point -> below minimum_points
    degenerate_contour.NumberOfContourPoints = 1
    roi_contour = Dataset()
    roi_contour.ReferencedROINumber = 1
    roi_contour.ContourSequence = Sequence([degenerate_contour])
    ds.ROIContourSequence = Sequence([roi_contour])
    return _write(ds, path)


def test_sanitize_rtstruct_leaves_prior_file_intact_on_simulated_crash(tmp_path, monkeypatch):
    """A crash between writing the temp file and the atomic rename must never leave
    the previously-published RTSTRUCT truncated/corrupted (F2 regression: sanitize_rtstruct
    used to `ds.save_as(str(path))` directly, in place)."""
    path = tmp_path / "RS.dcm"
    _write_rtstruct_with_degenerate_contour(path)
    original_bytes = path.read_bytes()

    def boom(*_args, **_kwargs):
        raise OSError("simulated crash mid-publish")

    monkeypatch.setattr(utils.os, "replace", boom)

    changed = utils.sanitize_rtstruct(path)

    assert changed is False  # errors are caught and logged, not raised
    assert path.read_bytes() == original_bytes, "previously-published RTSTRUCT must be untouched"
    assert list(tmp_path.glob(f".{path.name}.*.tmp*")) == [], "no temp file should remain"


def test_sanitize_rtstruct_leaves_no_temp_file_on_success(tmp_path):
    path = tmp_path / "RS.dcm"
    _write_rtstruct_with_degenerate_contour(path)

    changed = utils.sanitize_rtstruct(path)

    assert changed is True
    assert list(tmp_path.glob(f".{path.name}.*.tmp*")) == []
    # Degenerate contour was actually dropped (rewrite happened, not skipped).
    ds = pydicom.dcmread(str(path))
    assert not getattr(ds.ROIContourSequence[0], "ContourSequence", None)


def _build_and_save_real_rtstruct(ct_dir: Path, rtstruct_path: Path):
    """Build a real rt_utils RTStruct against a synthetic CT series and publish it,
    returning the (ct_dir, rtstruct_path) pair fix_rtstruct_rois would load."""
    rtstruct = _build_real_rtstruct(ct_dir)
    rtstruct.save(str(rtstruct_path))
    return rtstruct_path


def test_fix_rtstruct_rois_leaves_prior_file_intact_on_simulated_crash(tmp_path, monkeypatch):
    """Same F2 atomicity guarantee for roi_fixer.fix_rtstruct_rois: a crash between
    writing the temp file and the atomic rename must not corrupt the already-published
    RTSTRUCT it is rewriting in place (the default output_path == rtstruct_path)."""
    ct_dir = tmp_path / "ct"
    ct_dir.mkdir()
    rtstruct_path = tmp_path / "RS.dcm"
    _build_and_save_real_rtstruct(ct_dir, rtstruct_path)
    original_bytes = rtstruct_path.read_bytes()

    # Force the "needs rewrite" branch without depending on rt_utils' own rasterization
    # failing: get_mask() already returns a real mask for our simple synthetic ROI, so
    # was_rebuilt() would normally stay False. Marking it rebuilt exercises exactly the
    # save/publish path under test (atomicity), independent of the rebuild heuristics.
    monkeypatch.setattr(roi_fixer._ROIRebuilder, "was_rebuilt", lambda self, name: True)

    def boom(*_args, **_kwargs):
        raise OSError("simulated crash mid-publish")

    monkeypatch.setattr(roi_fixer.os, "replace", boom)

    result = roi_fixer.fix_rtstruct_rois(ct_dir, rtstruct_path)

    assert result is None  # errors are caught and logged, not raised
    assert rtstruct_path.read_bytes() == original_bytes, "previously-published RTSTRUCT must be untouched"
    assert list(tmp_path.glob(f".{rtstruct_path.name}.*.tmp*")) == [], "no temp file should remain"


def test_fix_rtstruct_rois_leaves_no_temp_file_on_success(tmp_path, monkeypatch):
    ct_dir = tmp_path / "ct"
    ct_dir.mkdir()
    rtstruct_path = tmp_path / "RS.dcm"
    _build_and_save_real_rtstruct(ct_dir, rtstruct_path)

    monkeypatch.setattr(roi_fixer._ROIRebuilder, "was_rebuilt", lambda self, name: True)

    summary = roi_fixer.fix_rtstruct_rois(ct_dir, rtstruct_path)

    assert summary is not None and summary.changed is True
    assert rtstruct_path.exists()
    assert list(tmp_path.glob(f".{rtstruct_path.name}.*.tmp*")) == []
