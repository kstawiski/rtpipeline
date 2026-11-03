"""Synthetic DICOM only: explicit concurrent components and legacy controls."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
import pytest

from rtpipeline import organize as org
from rtpipeline.config import PipelineConfig
from rtpipeline.plan_disposition import RELATIVE_PATH
from synthetic_rt_fixtures import make_dose, make_plan, make_record, make_struct


ENV = "RTPIPELINE_CONCURRENT_COMPONENTS"


def change(path, edit):
    ds = pydicom.dcmread(path)
    edit(ds)
    ds.save_as(path, enforce_file_format=True)


def course(root, *, resample=False):
    root.mkdir(parents=True, exist_ok=True)
    study, frame, series, struct_uid = [generate_uid() for _ in range(4)]
    ct_uids = []
    for index in range(3):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = generate_uid()
        ct_uids.append(str(ds.SOPInstanceUID))
        ds.PatientID = "P1"
        ds.PatientName = "SYNTH^FIXTURE"
        ds.Modality = "CT"
        ds.StudyInstanceUID = study
        ds.SeriesInstanceUID = series
        ds.FrameOfReferenceUID = frame
        ds.SeriesDescription = "Synthetic planning CT"
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        ds.ImagePositionPatient = [0, 0, index * 5]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [5, 5]
        ds.SliceThickness = 5
        ds.InstanceNumber = index + 1
        ds.Rows = ds.Columns = 5
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleIntercept = -1000
        ds.RescaleSlope = 1
        ds.PixelData = np.zeros((5, 5), dtype=np.int16).tobytes()
        ds.save_as(root / f"ct_{index}.dcm", enforce_file_format=True)
    rs = make_struct(root / "rs.dcm", struct_uid, study_uid=study, frame_uid=frame,
                     roi_names=["PTV1", "PTV2", "HEART"])

    def ct_reference(ds):
        ref_series = Dataset()
        ref_series.SeriesInstanceUID = series
        ref_series.ContourImageSequence = Sequence([])
        for uid in ct_uids:
            ref = Dataset()
            ref.ReferencedSOPClassUID = CTImageStorage
            ref.ReferencedSOPInstanceUID = uid
            ref_series.ContourImageSequence.append(ref)
        ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence = Sequence([ref_series])
        for item in ds.ROIContourSequence:
            item.ContourSequence[0].ContourData = [0, 0, 0, 20, 0, 0, 20, 20, 0, 0, 20, 0]
            item.ContourSequence[0].ContourImageSequence = Sequence([ref_series.ContourImageSequence[0]])
    change(rs, ct_reference)
    plans, doses, uids, arrays = [], [], [], []
    for index in range(2):
        uid = generate_uid()
        plan = make_plan(root / f"p{index}.dcm", uid, struct_uid=struct_uid, study_uid=study,
                         frame_uid=frame, date=f"2024010{index + 1}", rx_gy=60, fractions=30)
        def target(ds):
            ds.DoseReferenceSequence[0].DoseReferenceStructureType = "VOLUME"
            ds.DoseReferenceSequence[0].ReferencedROINumber = index + 1
        change(plan, target)
        shape = (3, 3, 3) if resample and index == 1 else (3, 5, 5)
        step = 10 if resample and index == 1 else 5
        z, y, x = np.indices(shape)
        values = (1 + x / 4 + y / 4 + z / 2) if index == 0 else (2 + x * step / 20 + y * step / 20 + z / 2)
        dose = make_dose(root / f"d{index}.dcm", generate_uid(), plan_uid=uid,
                         study_uid=study, frame_uid=frame, with_pixels=True, shape=shape)
        def pixels(ds):
            ds.ImagePositionPatient = [0, 0, 0]
            ds.PixelSpacing = [step, step]
            ds.DoseGridScaling = .001
            ds.PixelData = np.rint(values * 1000).astype(np.uint16).tobytes()
        change(dose, pixels)
        plans.append(plan)
        doses.append(dose)
        uids.append(uid)
        arrays.append(values)
    return dict(plans=plans, doses=doses, uids=uids, structs=[rs], arrays=arrays)


def declare(tmp_path, monkeypatch, uids, **extra):
    path = tmp_path / "declarations.json"
    path.write_text(json.dumps({"declarations": [{
        "plan_sop_instance_uids": uids,
        "basis": "Synthetic completed concurrent course; 30/30 fractions each, same dates, distinct refpoints",
        "declared_on": "2026-11-02", "issued_by": "Lucjan manager (synthetic)", **extra,
    }]}))
    monkeypatch.setenv(ENV, str(path))
    return path


def classify(fixture):
    return org._classify_doses(fixture["plans"], fixture["doses"], structure_set_paths=fixture["structs"])


@pytest.fixture
def legacy(monkeypatch):
    """Pin undeclared behavior to M-DEDUP-1(a), before the declaration port."""
    import subprocess
    reference = "f0ed849:rtpipeline/organize.py"
    source = subprocess.check_output(["git", "show", reference],
                                     cwd=Path(__file__).resolve().parents[1], text=True)
    # The reference remains read-only; loading source does not create a bytecode file.
    import types
    module = types.ModuleType("rtpipeline._concurrent_legacy_test")
    module.__package__ = "rtpipeline"
    import sys
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(source, reference, "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("resample", [False, True])
def test_opt_in_grid_arithmetic_and_no_declaration_equivalence(tmp_path, monkeypatch, legacy, resample):
    fixture = course(tmp_path / "dicom", resample=resample)
    monkeypatch.delenv(ENV, raising=False)
    baseline = classify(fixture)
    expected = legacy._classify_doses(fixture["plans"], fixture["doses"])
    assert asdict(baseline) == asdict(expected)
    assert baseline.classification == org.INDEPENDENT_DELIVERY_UNRECONCILED
    assert baseline.selected_doses == []
    monkeypatch.setenv(ENV, "")
    assert asdict(classify(fixture)) == asdict(expected)
    declare(tmp_path, monkeypatch, fixture["uids"])
    selected = classify(fixture)
    assert selected.classification == org.DECLARED_CONCURRENT_COMPONENTS
    assert selected.should_sum is True
    assert selected.selected_doses == fixture["doses"]
    assert selected.selected_plans == fixture["plans"]
    assert selected.excluded_doses == []
    plan_sum, plans, _ = org._create_summed_plan(selected.selected_plans)
    summed, _, _ = org._sum_doses_with_resample(selected.selected_doses, plan_sum, plans)
    actual = summed.pixel_array * float(summed.DoseGridScaling)
    z, y, x = np.indices((3, 5, 5))
    expected_grid = fixture["arrays"][0] + 2 + x / 4 + y / 4 + z / 2
    if resample:
        np.testing.assert_allclose(actual, expected_grid, atol=.001, rtol=0)
    else:
        np.testing.assert_array_equal(actual, fixture["arrays"][0] + fixture["arrays"][1])
    monkeypatch.delenv(ENV)
    assert asdict(classify(fixture)) == asdict(baseline)


@pytest.mark.parametrize("failure,reason", [
    ("frame", "FrameOfReferenceUID"), ("plan_frame", "FrameOfReferenceUID"),
    ("target_uid", "DoseReferenceUIDs"), ("missing_dose", "exactly one"),
    ("missing_plan", "not among"), ("duplicate_dose", "exactly one"),
    ("ct_series", "same planning-CT"), ("no_struct", "RTSTRUCT missing"),
    ("no_ct", "one referenced CT"), ("no_target", "no TARGET"),
    ("roi_overlap", "ROI numbers"), ("missing_roi", "need ROI"),
    ("unapproved", "not among"), ("geometry", "PATIENT"),
    ("all_doses_missing", "exactly one"),
    ("missing_summation", "explicitly specify PLAN"),
    ("shared_dose_reference", "singly referenced"),
    ("all_unapproved", "not APPROVED"),
])
def test_rejected_declaration_is_legacy_selection_with_visible_reason(tmp_path, monkeypatch, legacy, failure, reason):
    fixture = course(tmp_path / "dicom")
    uids = fixture["uids"][:]
    if failure == "frame":
        change(fixture["doses"][0], lambda ds: setattr(ds, "FrameOfReferenceUID", generate_uid()))
    elif failure == "plan_frame":
        change(fixture["plans"][0], lambda ds: setattr(ds, "FrameOfReferenceUID", generate_uid()))
    elif failure == "target_uid":
        first = pydicom.dcmread(fixture["plans"][0]).DoseReferenceSequence[0].DoseReferenceUID
        change(fixture["plans"][1], lambda ds: setattr(ds.DoseReferenceSequence[0], "DoseReferenceUID", first))
    elif failure == "missing_dose":
        fixture["doses"] = fixture["doses"][1:]
    elif failure == "all_doses_missing":
        fixture["doses"] = []
    elif failure == "missing_plan":
        uids.append(generate_uid())
    elif failure == "duplicate_dose":
        path = fixture["doses"][0].with_name("duplicate.dcm")
        ds = pydicom.dcmread(fixture["doses"][0])
        ds.SOPInstanceUID = generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.save_as(path, enforce_file_format=True)
        fixture["doses"].append(path)
    elif failure == "ct_series":
        other = fixture["structs"][0].with_name("other_rs.dcm")
        ds = pydicom.dcmread(fixture["structs"][0])
        ds.SOPInstanceUID = generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID = generate_uid()
        ds.save_as(other, enforce_file_format=True)
        fixture["structs"].append(other)
        change(fixture["plans"][1], lambda plan: setattr(plan.ReferencedStructureSetSequence[0], "ReferencedSOPInstanceUID", ds.SOPInstanceUID))
    elif failure == "no_struct":
        fixture["structs"] = []
    elif failure == "no_ct":
        change(fixture["structs"][0], lambda ds: setattr(ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0], "RTReferencedSeriesSequence", Sequence([])))
    elif failure == "no_target":
        change(fixture["plans"][0], lambda ds: setattr(ds.DoseReferenceSequence[0], "DoseReferenceType", "ORGAN_AT_RISK"))
    elif failure in {"roi_overlap", "missing_roi"}:
        def without_uid(ds):
            del ds.DoseReferenceSequence[0].DoseReferenceUID
            if failure == "roi_overlap":
                ds.DoseReferenceSequence[0].ReferencedROINumber = 1
            else:
                del ds.DoseReferenceSequence[0].ReferencedROINumber
        change(fixture["plans"][1], without_uid)
    elif failure == "unapproved":
        change(fixture["plans"][0], lambda ds: setattr(ds, "ApprovalStatus", "UNAPPROVED"))
    elif failure == "all_unapproved":
        for path in fixture["plans"]:
            change(path, lambda ds: setattr(ds, "ApprovalStatus", "UNAPPROVED"))
    elif failure == "geometry":
        change(fixture["plans"][0], lambda ds: setattr(ds, "RTPlanGeometry", "TREATMENT_DEVICE"))
    elif failure == "missing_summation":
        change(fixture["doses"][0], lambda ds: ds.__delattr__("DoseSummationType"))
    elif failure == "shared_dose_reference":
        def shared(ds):
            ref = Dataset()
            ref.ReferencedSOPInstanceUID = fixture["uids"][1]
            ref.ReferencedSOPClassUID = ds.ReferencedRTPlanSequence[0].ReferencedSOPClassUID
            ds.ReferencedRTPlanSequence.append(ref)
        change(fixture["doses"][0], shared)
    monkeypatch.delenv(ENV, raising=False)
    expected = asdict(legacy._classify_doses(fixture["plans"], fixture["doses"]))
    declare(tmp_path, monkeypatch, uids)
    actual = asdict(classify(fixture))
    warnings = actual.pop("warnings")
    legacy_warnings = expected.pop("warnings")
    assert actual == expected
    assert all(message in warnings for message in legacy_warnings)
    assert any("not applied" in message and reason in message for message in warnings)


def test_target_roi_fallback_and_unrelated_course(tmp_path, monkeypatch, legacy):
    fixture = course(tmp_path / "dicom")
    for path in fixture["plans"]:
        change(path, lambda ds: ds.DoseReferenceSequence[0].__delattr__("DoseReferenceUID"))
    declare(tmp_path, monkeypatch, fixture["uids"])
    assert classify(fixture).classification == org.DECLARED_CONCURRENT_COMPONENTS
    declare(tmp_path, monkeypatch, [generate_uid(), generate_uid()])
    assert asdict(classify(fixture)) == asdict(legacy._classify_doses(fixture["plans"], fixture["doses"]))


@pytest.mark.parametrize("payload,reason", [
    ("{", "not applied"),
    ({"declarations": "wrong"}, "declarations list"),
])
def test_bad_declaration_file_is_visible_and_keeps_legacy(tmp_path, monkeypatch, legacy, payload, reason):
    fixture = course(tmp_path / "dicom")
    path = tmp_path / "bad.json"
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    monkeypatch.setenv(ENV, str(path))
    actual = classify(fixture)
    expected = legacy._classify_doses(fixture["plans"], fixture["doses"])
    assert actual.selected_doses == expected.selected_doses
    assert actual.classification == expected.classification
    assert any(reason in message for message in actual.warnings)


@pytest.mark.parametrize("problem,reason", [
    ("unreadable", "not applied"), ("basis", "basis is missing"),
    ("date", "YYYY-MM-DD"), ("duplicate_uid", "unique plan UIDs"),
    ("overlap", "multiple declarations"),
])
def test_ambiguous_or_incomplete_declaration_keeps_legacy(tmp_path, monkeypatch, legacy, problem, reason):
    fixture = course(tmp_path / "dicom")
    path = declare(tmp_path, monkeypatch, fixture["uids"])
    payload = json.loads(path.read_text())
    if problem == "unreadable":
        path.unlink()
    else:
        if problem == "basis":
            payload["declarations"][0]["basis"] = ""
        elif problem == "date":
            payload["declarations"][0]["declared_on"] = "20261102"
        elif problem == "duplicate_uid":
            payload["declarations"][0]["plan_sop_instance_uids"] *= 2
        else:
            payload["declarations"] *= 2
        path.write_text(json.dumps(payload))
    actual = classify(fixture)
    expected = legacy._classify_doses(fixture["plans"], fixture["doses"])
    assert actual.classification == expected.classification
    assert actual.selected_doses == expected.selected_doses
    assert any(reason in message for message in actual.warnings)


def test_declaration_always_holds_scalar_but_retains_plan_reference_doses(tmp_path, monkeypatch):
    fixture = course(tmp_path / "dicom")
    declare(tmp_path, monkeypatch, fixture["uids"])
    selected = classify(fixture)
    no_records = org._calculate_delivery_summary(fixture["plans"], [], classification=selected.classification)
    assert no_records["delivered_dose_gy"] is None
    assert any(item["reason_code"] == "COURSE_DOSE_CONCURRENT_DISTINCT_REFERENCES" for item in no_records["delivery_course_holds"])
    records = []
    for index, uid in enumerate(fixture["uids"]):
        # Disjoint dates deliberately exercise the unconditional declaration hold.
        for fraction in range(1, 31):
            import datetime
            day = datetime.date(2024, 2 + index * 2, 1) + datetime.timedelta(days=fraction - 1)
            records.append(make_record(tmp_path / f"r{index}_{fraction}.dcm", uid,
                                       fraction_number=fraction, date=day.strftime("%Y%m%d")))
    baseline = org._calculate_delivery_summary(fixture["plans"], records)
    assert baseline["delivered_dose_gy"] == 120
    summary = org._calculate_delivery_summary(fixture["plans"], records, classification=selected.classification)
    assert summary["delivered_dose_gy"] is None
    assert summary["delivery_plan_details"] == baseline["delivery_plan_details"]
    assert [row["delivered_dose_gy"] for row in summary["delivery_plan_details"]] == [60, 60]
    assert [row["prescribed_dose_gy"] for row in summary["delivery_plan_details"]] == [60, 60]
    totals = org._course_fraction_totals(selected.classification, summary["delivered_fraction_count"], summary["planned_fraction_count"])
    assert totals["delivered_fraction_count"] is None
    assert totals["planned_fraction_count"] is None
    assert totals["basis"] in org.COURSE_FRACTION_TOTALS_WITHHELD_BASES


def run_organizer(root, output, *, resume=False):
    return org.organize_and_merge(PipelineConfig(
        root, output, output.parent / "logs", resume=resume,
        max_workers_override=1, dicom_copy_use_hardlinks=False,
        dicom_copy_dedup_by_sop_uid=False,
    ))


@pytest.mark.parametrize("with_records", [False, True])
def test_published_grid_dispositions_holds_and_resume(tmp_path, monkeypatch, with_records):
    fixture = course(tmp_path / "input")
    if with_records:
        # Partial exported record evidence must not weight the declared full PLAN grids.
        for index, uid in enumerate(fixture["uids"]):
            make_record(tmp_path / "input" / f"record_{index}.dcm", uid)
    output = tmp_path / "output"
    monkeypatch.delenv(ENV, raising=False)
    baseline = run_organizer(tmp_path / "input", output)
    assert len(baseline) == 1
    legacy_classification = (org.INDEPENDENT_DELIVERY_UNRECONCILED if with_records
                             else org.INDEPENDENT_DELIVERY_UNRECONCILED)
    assert baseline[0].dose_classification["classification"] == legacy_classification
    declare(tmp_path, monkeypatch, fixture["uids"])
    courses = run_organizer(tmp_path / "input", output, resume=True)
    assert len(courses) == 1
    co = courses[0]
    assert co.dose_classification["classification"] == org.DECLARED_CONCURRENT_COMPONENTS
    assert len(co.selected_plan_contract) == len(co.selected_dose_contract) == 2
    assert co.delivered_dose_gy is None
    assert co.planned_fraction_count is None
    assert co.delivered_fraction_count is None
    assert co.course_contract["delivery"]["resolved_prescribed_dose_total_gy"] is None
    dispositions = json.loads((output / RELATIVE_PATH).read_text())
    assert {item["disposition_type"] for item in dispositions["plans"]} == {"course_member"}
    ds = pydicom.dcmread(co.dirs.root / "RD.dcm")
    np.testing.assert_array_equal(ds.pixel_array * float(ds.DoseGridScaling), fixture["arrays"][0] + fixture["arrays"][1])
    plan = pydicom.dcmread(co.dirs.root / "RP.dcm")
    assert plan.FractionGroupSequence[0].NumberOfFractionsPlanned is None
    assert not hasattr(plan.DoseReferenceSequence[0], "TargetPrescriptionDose")
    contract = co.course_contract
    assert contract["delivery"]["delivered_dose_gy"] is None
    assert contract["delivery"]["dose_response_eligible"] is False
    assert {row["planned_fraction_count"] for row in contract["selected_plans"]} == {30}
    resumed = run_organizer(tmp_path / "input", output, resume=True)
    assert resumed[0].planned_fraction_count is None
    monkeypatch.delenv(ENV)
    restored = run_organizer(tmp_path / "input", output, resume=True)
    assert restored[0].dose_classification["classification"] == legacy_classification
    assert len(restored[0].selected_dose_contract) == 0


def test_no_dose_publication_keeps_legacy_membership_and_records_refusal(tmp_path, monkeypatch):
    fixture = course(tmp_path / "input")
    for path in fixture["doses"]:
        path.unlink()
    monkeypatch.delenv(ENV, raising=False)
    baseline = run_organizer(tmp_path / "input", tmp_path / "baseline")
    assert len(baseline) == 1
    declare(tmp_path, monkeypatch, fixture["uids"])
    actual = run_organizer(tmp_path / "input", tmp_path / "declared")
    assert len(actual) == 1
    def membership(co):
        return [item["sop_instance_uid"] for item in co.selected_plan_contract]
    assert membership(actual[0]) == membership(baseline[0])
    assert actual[0].selected_dose_contract == baseline[0].selected_dose_contract == []
    assert any("not applied" in warning and "exactly one" in warning
               for warning in actual[0].dose_classification["warnings"])


# Lucjan review 2026-11-02 (criteria from the rtpipeline maintainer): provenance, one structure set, full coverage
def test_applied_declaration_is_hash_pinned_with_its_issuer(tmp_path, monkeypatch):
    import hashlib
    fixture = course(tmp_path / "dicom")
    path = declare(tmp_path, monkeypatch, fixture["uids"])
    result = classify(fixture)
    assert result.classification == org.DECLARED_CONCURRENT_COMPONENTS
    entry = json.loads(path.read_text())["declarations"][0]
    assert "by Lucjan manager (synthetic)" in result.reason
    assert hashlib.sha256(path.read_bytes()).hexdigest() in result.reason
    assert hashlib.sha256(json.dumps(entry, sort_keys=True).encode("utf-8")).hexdigest() in result.reason


def test_declaration_without_issuer_is_refused(tmp_path, monkeypatch, legacy):
    fixture = course(tmp_path / "dicom")
    path = declare(tmp_path, monkeypatch, fixture["uids"])
    payload = json.loads(path.read_text())
    del payload["declarations"][0]["issued_by"]
    path.write_text(json.dumps(payload))
    result = classify(fixture)
    assert result.classification == org.INDEPENDENT_DELIVERY_UNRECONCILED
    assert any("issued_by" in w for w in result.warnings)


def test_plans_on_different_structure_sets_are_refused(tmp_path, monkeypatch):
    fixture = course(tmp_path / "dicom")
    root = tmp_path / "dicom"
    rs2 = root / "rs2.dcm"
    ds = pydicom.dcmread(fixture["structs"][0])
    ds.SOPInstanceUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.save_as(rs2, enforce_file_format=True)
    change(fixture["plans"][1], lambda p: setattr(p.ReferencedStructureSetSequence[0], "ReferencedSOPInstanceUID",
                                                  ds.SOPInstanceUID))
    fixture["structs"].append(rs2)
    declare(tmp_path, monkeypatch, fixture["uids"])
    result = classify(fixture)
    assert result.classification != org.DECLARED_CONCURRENT_COMPONENTS
    assert any("different RTSTRUCTs" in w for w in result.warnings)


@pytest.mark.parametrize("with_dose", [False, True])
def test_declaration_covering_part_of_the_course_is_refused(tmp_path, monkeypatch, with_dose):
    fixture = course(tmp_path / "dicom")
    root = tmp_path / "dicom"
    uid3 = generate_uid()
    p3 = root / "p2.dcm"
    plan = pydicom.dcmread(fixture["plans"][0])
    plan.SOPInstanceUID = uid3
    plan.file_meta.MediaStorageSOPInstanceUID = uid3
    plan.save_as(p3, enforce_file_format=True)
    d3 = root / "d2.dcm"
    dose = pydicom.dcmread(fixture["doses"][0])
    dose.SOPInstanceUID = generate_uid()
    dose.file_meta.MediaStorageSOPInstanceUID = dose.SOPInstanceUID
    dose.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID = uid3
    dose.save_as(d3, enforce_file_format=True)
    fixture["plans"].append(p3)
    if with_dose:
        fixture["doses"].append(d3)
    declare(tmp_path, monkeypatch, fixture["uids"])          # names two of the three dosed plans
    result = classify(fixture)
    assert result.classification != org.DECLARED_CONCURRENT_COMPONENTS
    assert any("does not cover every approved plan" in w for w in result.warnings)


# Independent review r1 (2026-11-02): findings 1-7.
import hashlib
from pydicom.dataelem import DataElement


def write_declarations(tmp_path, monkeypatch, entries, name="declarations.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"declarations": entries}))
    monkeypatch.setenv(ENV, str(path))
    return path


def entry(uids, **extra):
    return {"plan_sop_instance_uids": uids, "basis": "synthetic", "declared_on": "2026-11-02",
            "issued_by": "Lucjan manager (synthetic)", **extra}


def copy_dicom(source, target, **values):
    ds = pydicom.dcmread(source)
    for key, value in values.items():
        setattr(ds, key, value)
    if "SOPInstanceUID" in values:
        ds.file_meta.MediaStorageSOPInstanceUID = values["SOPInstanceUID"]
    ds.save_as(target, enforce_file_format=True)
    return target


def third_plan(fixture, root):
    """A third approved plan of the course with its own PLAN dose and target reference."""
    uid = generate_uid()
    plan = copy_dicom(fixture["plans"][0], root / "p2.dcm", SOPInstanceUID=uid)
    change(plan, lambda ds: setattr(ds.DoseReferenceSequence[0], "DoseReferenceUID", generate_uid()))
    dose = copy_dicom(fixture["doses"][0], root / "d2.dcm", SOPInstanceUID=generate_uid())
    change(dose, lambda ds: setattr(ds.ReferencedRTPlanSequence[0], "ReferencedSOPInstanceUID", uid))
    return plan, dose


def break_grid_geometry(path):
    """Header stays readable (plan reference, PLAN summation); ImagePositionPatient is not numeric."""
    ds = pydicom.dcmread(path)
    ds[0x00200032] = DataElement(0x00200032, "LO", "abc\\def\\ghi")
    ds.save_as(path, enforce_file_format=True)


def test_sibling_dose_with_unreadable_geometry_still_counts_for_coverage(tmp_path, monkeypatch, legacy):
    """Finding 1: plan membership of a dose comes from its header, not from the geometry-dependent metadata."""
    root = tmp_path / "dicom"
    fixture = course(root)
    plan, dose = third_plan(fixture, root)
    break_grid_geometry(dose)
    assert org._extract_dose_metadata(dose)["referenced_plan_uids"] == []   # the fallback the review describes
    fixture["plans"].append(plan)
    fixture["doses"].append(dose)
    declare(tmp_path, monkeypatch, fixture["uids"])                         # names two of the three dosed plans
    result = classify(fixture)
    assert result.classification != org.DECLARED_CONCURRENT_COMPONENTS
    assert any("not applied" in w and "does not cover every approved plan" in w for w in result.warnings)
    monkeypatch.delenv(ENV)
    expected = asdict(legacy._classify_doses(fixture["plans"], fixture["doses"]))
    declare(tmp_path, monkeypatch, fixture["uids"])
    actual = asdict(classify(fixture))
    actual.pop("warnings"), expected.pop("warnings")
    assert actual == expected


def test_declared_dose_with_unreadable_geometry_is_refused(tmp_path, monkeypatch):
    fixture = course(tmp_path / "dicom")
    break_grid_geometry(fixture["doses"][0])
    declare(tmp_path, monkeypatch, fixture["uids"])
    result = classify(fixture)
    assert result.classification != org.DECLARED_CONCURRENT_COMPONENTS
    assert any("incomplete or invalid grid geometry" in w for w in result.warnings)


def test_unreadable_course_dose_file_refuses_the_declaration(tmp_path, monkeypatch, legacy):
    """A dose file that cannot be read at all might belong to an undeclared plan: refuse, keep legacy."""
    fixture = course(tmp_path / "dicom")
    junk = tmp_path / "dicom" / "d_junk.dcm"
    junk.write_bytes(b"not a DICOM file")
    fixture["doses"].append(junk)
    monkeypatch.delenv(ENV, raising=False)
    expected = asdict(legacy._classify_doses(fixture["plans"], fixture["doses"]))
    declare(tmp_path, monkeypatch, fixture["uids"])
    actual = asdict(classify(fixture))
    warnings = actual.pop("warnings")
    expected.pop("warnings")
    assert actual == expected
    assert any("not applied" in w and "cannot be read" in w for w in warnings)


@pytest.mark.parametrize("other", [
    {}, {"plan_sop_instance_uids": "only-one"}, "not an object",
    "one_uid", "no_issuer", "bad_date", "no_basis",
])
def test_entry_naming_no_plan_of_the_course_has_no_effect_even_when_invalid(tmp_path, monkeypatch, legacy, other):
    """Finding 2: an entry is matched to the course by plan UID before it is validated."""
    fixture = course(tmp_path / "dicom")
    elsewhere = [generate_uid(), generate_uid()]
    bad = {"one_uid": entry(elsewhere[:1]), "no_issuer": entry(elsewhere, issued_by=""),
           "bad_date": entry(elsewhere, declared_on="02.11.2026"), "no_basis": entry(elsewhere, basis=" ")}
    write_declarations(tmp_path, monkeypatch, [bad.get(other, other) if isinstance(other, str) else other])
    assert asdict(classify(fixture)) == asdict(legacy._classify_doses(fixture["plans"], fixture["doses"]))


def test_valid_entry_applies_beside_an_invalid_entry_for_another_course(tmp_path, monkeypatch):
    fixture = course(tmp_path / "dicom")
    write_declarations(tmp_path, monkeypatch, [entry([generate_uid()], issued_by=""), entry(fixture["uids"])])
    result = classify(fixture)
    assert result.classification == org.DECLARED_CONCURRENT_COMPONENTS
    assert not any("not applied" in w for w in result.warnings)


@pytest.mark.parametrize("named", ["string", "padded"])
def test_malformed_entry_naming_this_course_is_refused_visibly(tmp_path, monkeypatch, named):
    fixture = course(tmp_path / "dicom")
    uids = fixture["uids"][0] if named == "string" else [fixture["uids"][0] + " ", fixture["uids"][1]]
    write_declarations(tmp_path, monkeypatch, [entry(uids)])
    result = classify(fixture)
    assert result.classification == org.INDEPENDENT_DELIVERY_UNRECONCILED
    assert any("not applied" in w and "unique plan UIDs" in w for w in result.warnings)


def add_frame_reference(struct_path, *, frame=None, series=None):
    """Append a ReferencedFrameOfReferenceSequence item (for example a registered prior image)."""
    def edit(ds):
        item = Dataset()
        item.FrameOfReferenceUID = frame or generate_uid()
        study = Dataset()
        study.ReferencedSOPClassUID = "1.2.840.10008.3.1.2.3.1"
        study.ReferencedSOPInstanceUID = generate_uid()
        ref_series = Dataset()
        ref_series.SeriesInstanceUID = series or generate_uid()
        ref_series.ContourImageSequence = Sequence([])
        study.RTReferencedSeriesSequence = Sequence([ref_series])
        item.RTReferencedStudySequence = Sequence([study])
        ds.ReferencedFrameOfReferenceSequence.append(item)
    change(struct_path, edit)


def test_structure_set_citing_a_further_frame_still_applies(tmp_path, monkeypatch):
    """Finding 3: the declared frame must be among the structure set's frames, with one CT series under it."""
    fixture = course(tmp_path / "dicom")
    add_frame_reference(fixture["structs"][0])
    declare(tmp_path, monkeypatch, fixture["uids"])
    assert classify(fixture).classification == org.DECLARED_CONCURRENT_COMPONENTS


def test_structure_set_without_the_declared_frame_is_refused(tmp_path, monkeypatch):
    fixture = course(tmp_path / "dicom")
    change(fixture["structs"][0], lambda ds: setattr(ds.ReferencedFrameOfReferenceSequence[0], "FrameOfReferenceUID",
                                                     generate_uid()))
    declare(tmp_path, monkeypatch, fixture["uids"])
    result = classify(fixture)
    assert result.classification != org.DECLARED_CONCURRENT_COMPONENTS
    assert any("does not reference the declared FrameOfReferenceUID" in w for w in result.warnings)


def test_two_ct_series_under_the_declared_frame_are_refused(tmp_path, monkeypatch):
    fixture = course(tmp_path / "dicom")
    frame = str(pydicom.dcmread(fixture["plans"][0]).FrameOfReferenceUID)
    add_frame_reference(fixture["structs"][0], frame=frame)
    declare(tmp_path, monkeypatch, fixture["uids"])
    result = classify(fixture)
    assert result.classification != org.DECLARED_CONCURRENT_COMPONENTS
    assert any("one referenced CT series under the declared frame" in w for w in result.warnings)


def records_on(tmp_path, fixture, *, disjoint):
    import datetime
    records = []
    for index, uid in enumerate(fixture["uids"]):
        for fraction in range(1, 31):
            start = datetime.date(2024, 2 + (index * 2 if disjoint else 0), 1)
            day = start + datetime.timedelta(days=fraction - 1)
            records.append(make_record(tmp_path / f"rec{index}_{fraction}.dcm", uid,
                                       fraction_number=fraction, date=day.strftime("%Y%m%d")))
    return records


@pytest.mark.parametrize("evidence", ["none", "disjoint", "shared"])
def test_declared_course_states_the_withheld_dose_once_and_no_sequential_total(tmp_path, monkeypatch, evidence):
    """Finding 4: no 'nominal sequential total' sentence beside the null scalar, and one hold, not two."""
    fixture = course(tmp_path / "dicom")
    records = [] if evidence == "none" else records_on(tmp_path, fixture, disjoint=evidence == "disjoint")
    summary = org._calculate_delivery_summary(fixture["plans"], records,
                                              classification=org.DECLARED_CONCURRENT_COMPONENTS)
    assert summary["delivered_dose_gy"] is None
    assert not any("sequential total" in w for w in summary["delivery_warnings"])
    codes = [item["reason_code"] for item in summary["delivery_course_holds"]]
    assert codes.count("COURSE_DOSE_CONCURRENT_DISTINCT_REFERENCES") == 1
    if evidence == "disjoint":   # the legacy branch this guards against does publish the sentence
        baseline = org._calculate_delivery_summary(fixture["plans"], records)
        assert any("sequential total" in w for w in baseline["delivery_warnings"])


def test_unreadable_plan_outside_any_declaration_does_not_abort_classification(tmp_path, monkeypatch, legacy):
    """Finding 5: a header read failure of any type is caught; with no relevant entry the course is untouched."""
    fixture = course(tmp_path / "dicom")
    for path in fixture["plans"]:
        change(path, lambda ds: setattr(ds, "ApprovalStatus", "UNAPPROVED"))
    junk = tmp_path / "dicom" / "p_junk.dcm"
    junk.write_bytes(b"not a DICOM file")
    plans = fixture["plans"] + [junk]
    write_declarations(tmp_path, monkeypatch, [entry([generate_uid(), generate_uid()])])
    actual = org._classify_doses(plans, fixture["doses"], structure_set_paths=fixture["structs"])
    assert asdict(actual) == asdict(legacy._classify_doses(plans, fixture["doses"]))


def test_unreadable_plan_beside_a_relevant_declaration_refuses_it(tmp_path, monkeypatch):
    fixture = course(tmp_path / "dicom")
    junk = tmp_path / "dicom" / "p_junk.dcm"
    junk.write_bytes(b"not a DICOM file")
    declare(tmp_path, monkeypatch, fixture["uids"])
    warnings = []
    assert org._declared_concurrent_doses(fixture["plans"] + [junk], fixture["doses"], fixture["structs"],
                                          warnings) is None
    assert any("RTPLAN header cannot be read" in w for w in warnings)
    write_declarations(tmp_path, monkeypatch, [entry([generate_uid(), generate_uid()])], name="other.json")
    quiet = []
    assert org._declared_concurrent_doses(fixture["plans"] + [junk], fixture["doses"], fixture["structs"],
                                          quiet) is None
    assert quiet == []


def test_any_exception_inside_the_declaration_check_keeps_legacy(tmp_path, monkeypatch):
    """An exception of any type inside the check (here from the approval read of the declared plans; unreadable
    structure sets are skipped by design, review r3 F3) refuses the declaration visibly."""
    fixture = course(tmp_path / "dicom")
    declare(tmp_path, monkeypatch, fixture["uids"])

    class Odd(Exception):
        pass

    def failing(dataset):
        raise Odd("synthetic reader failure")

    monkeypatch.setattr(org, "approval_status", failing)
    warnings = []
    assert org._declared_concurrent_doses(fixture["plans"], fixture["doses"], fixture["structs"], warnings) is None
    assert any("not applied" in w and "synthetic reader failure" in w for w in warnings)


def test_one_structure_set_linked_under_two_paths(tmp_path, monkeypatch):
    """Finding 7: identical bytes under two paths are one object; differing bytes under one UID stay ambiguous."""
    fixture = course(tmp_path / "dicom")
    twin = tmp_path / "dicom" / "rs_twin.dcm"
    twin.write_bytes(fixture["structs"][0].read_bytes())
    fixture["structs"].append(twin)
    declare(tmp_path, monkeypatch, fixture["uids"])
    assert classify(fixture).classification == org.DECLARED_CONCURRENT_COMPONENTS
    change(twin, lambda ds: setattr(ds, "StructureSetLabel", "edited"))
    result = classify(fixture)
    assert result.classification != org.DECLARED_CONCURRENT_COMPONENTS
    assert any("missing or ambiguous" in w for w in result.warnings)


# Finding 6: published outputs, not only the classifier.
VOLATILE = {"generated_at"}


def published(output):
    """Every published file's bytes, with run-specific absolute paths and generation times normalized."""
    import openpyxl
    root = str(output)
    files = {}
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(output))
        if path.suffix == ".json":
            def scrub(value):
                if isinstance(value, dict):
                    return {k: scrub(v) for k, v in value.items() if k not in VOLATILE}
                if isinstance(value, list):
                    return [scrub(v) for v in value]
                return value.replace(root, "<OUT>") if isinstance(value, str) else value
            files[rel] = json.dumps(scrub(json.loads(path.read_text())), sort_keys=True)
        elif path.suffix == ".xlsx":
            book = openpyxl.load_workbook(path, read_only=True)
            files[rel] = json.dumps([[str(c).replace(root, "<OUT>") if c is not None else None for c in row]
                                     for sheet in book.worksheets for row in sheet.iter_rows(values_only=True)])
        else:
            files[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


@pytest.mark.parametrize("with_records", [False, True])
def test_undeclared_course_publishes_the_reference_output(tmp_path, monkeypatch, legacy, with_records):
    fixture = course(tmp_path / "input")
    if with_records:
        for index, uid in enumerate(fixture["uids"]):
            make_record(tmp_path / "input" / f"record_{index}.dcm", uid)
    monkeypatch.delenv(ENV, raising=False)
    legacy.organize_and_merge(PipelineConfig(
        tmp_path / "input", tmp_path / "reference" / "output", tmp_path / "reference" / "logs",
        max_workers_override=1, dicom_copy_use_hardlinks=False, dicom_copy_dedup_by_sop_uid=False))
    reference = published(tmp_path / "reference" / "output")
    # with RTRECORDs both plans are independent deliveries and no course grid is published (74c25b9 behaviour)
    assert not any(name.endswith("RD.dcm") for name in reference)
    assert any(name.startswith("P1/") for name in reference)
    other = write_declarations(tmp_path, monkeypatch, [entry([generate_uid(), generate_uid()]),
                                                       entry([generate_uid()], issued_by="")], name="other.json")
    for label, value in [("unset", None), ("empty", ""), ("other", str(other))]:
        if value is None:
            monkeypatch.delenv(ENV, raising=False)
        else:
            monkeypatch.setenv(ENV, value)
        run_organizer(tmp_path / "input", tmp_path / label / "output")
        assert published(tmp_path / label / "output") == reference, label
    # Switching to a declaration for this course must leave the invariant path
    # and select both components; this also makes the opt-in regression fail on
    # a classifier that ignores the environment variable entirely.
    declare(tmp_path, monkeypatch, fixture["uids"])
    assert classify(fixture).classification == org.DECLARED_CONCURRENT_COMPONENTS


def test_unrelated_invalid_entry_keeps_resume_hydration(tmp_path, monkeypatch):
    """Finding 2 on resume: a file naming only other courses must not force a rebuild.

    Two identical trees are resumed, one with the variable unset and one with an unrelated invalid entry."""
    fixture = course(tmp_path / "input")
    monkeypatch.delenv(ENV, raising=False)
    for label in ("unset", "other"):
        run_organizer(tmp_path / "input", tmp_path / label / "output")
    calls = []
    real = org._classify_doses

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(org, "_classify_doses", counting)
    run_organizer(tmp_path / "input", tmp_path / "unset" / "output", resume=True)
    hydrated_calls = len(calls)
    calls.clear()
    write_declarations(tmp_path, monkeypatch, [entry([generate_uid()], issued_by="")])
    run_organizer(tmp_path / "input", tmp_path / "other" / "output", resume=True)
    assert len(calls) == hydrated_calls
    assert published(tmp_path / "other" / "output") == published(tmp_path / "unset" / "output")
    calls.clear()
    declare(tmp_path, monkeypatch, fixture["uids"])            # a relevant declaration does force the rebuild
    courses = run_organizer(tmp_path / "input", tmp_path / "other" / "output", resume=True)
    assert len(calls) > hydrated_calls
    assert courses[0].dose_classification["classification"] == org.DECLARED_CONCURRENT_COMPONENTS


def test_written_contract_carries_issuer_and_both_hashes(tmp_path, monkeypatch):
    fixture = course(tmp_path / "input")
    path = declare(tmp_path, monkeypatch, fixture["uids"])
    courses = run_organizer(tmp_path / "input", tmp_path / "output")
    entry_hash = hashlib.sha256(json.dumps(json.loads(path.read_text())["declarations"][0],
                                           sort_keys=True).encode("utf-8")).hexdigest()
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = json.loads((courses[0].dirs.root / "metadata" / "case_metadata.json").read_text())
    reasons = [metadata["course_contract"]["dose_classification"]["reason"]]
    for text in reasons:
        assert "by Lucjan manager (synthetic)" in text and file_hash in text and entry_hash in text
    assert ({row["sop_instance_uid"] for row in metadata["course_contract"]["selected_plans"]}
            == set(fixture["uids"]))


def test_resampled_declared_grid_is_published_as_the_physical_sum(tmp_path, monkeypatch):
    fixture = course(tmp_path / "input", resample=True)
    declare(tmp_path, monkeypatch, fixture["uids"])
    courses = run_organizer(tmp_path / "input", tmp_path / "output")
    assert courses[0].dose_classification["classification"] == org.DECLARED_CONCURRENT_COMPONENTS
    ds = pydicom.dcmread(courses[0].dirs.root / "RD.dcm")
    z, y, x = np.indices((3, 5, 5))
    expected = fixture["arrays"][0] + 2 + x / 4 + y / 4 + z / 2
    np.testing.assert_allclose(ds.pixel_array * float(ds.DoseGridScaling), expected, atol=.001, rtol=0)


def clinical_workbook(path, description="IMRT na obszar pęcherza moczowego do dawki 60 Gy/p.ref po 2 Gy"):
    """A synthetic register row in the kopernik-rt-treatments format (bladder ICD-10, as the matcher requires)."""
    import datetime
    import pandas as pd
    pd.DataFrame([{
        "ID": "P1", "Data rozp lecz": datetime.datetime(2023, 12, 1), "Data zak lecz": datetime.datetime(2024, 12, 31),
        "Rozpoznanie wg ICD 10": "C67.9", "Rodzaj Leczenia": "Radioterapia", "Rozpoznanie": "synthetic",
        "Zalecenia": "", "Opis leczenia": description,
    }]).to_excel(path, index=False)
    return path


@pytest.mark.parametrize("declared,with_records", [(False, True), (True, True), (True, False)])
def test_clinical_record_total_does_not_override_a_declared_course(tmp_path, monkeypatch, declared, with_records):
    """The clinical-register overlay (not configured for Lucjan) resolves a withheld course total from the register,
    except for declared components, whose course total has no additive meaning."""
    fixture = course(tmp_path / "input")
    if with_records:
        for index, uid in enumerate(fixture["uids"]):
            make_record(tmp_path / "input" / f"record_{index}.dcm", uid)
    if declared:
        declare(tmp_path, monkeypatch, fixture["uids"])
    else:
        monkeypatch.delenv(ENV, raising=False)
    config = PipelineConfig(tmp_path / "input", tmp_path / "output", tmp_path / "logs",
                            max_workers_override=1, dicom_copy_use_hardlinks=False, dicom_copy_dedup_by_sop_uid=False)
    config.clinical_prescription_records_path = clinical_workbook(tmp_path / "register.xlsx")
    courses = org.organize_and_merge(config)
    assert len(courses) == 1, "the course must publish, not be quarantined"
    co = courses[0]
    metadata = json.loads((co.dirs.root / "metadata" / "case_metadata.json").read_text())
    evidence = metadata["course_contract"].get("clinical_prescription_evidence") or {}
    assert evidence["match"]["status"] == "MATCHED"
    assert [site["total_dose_gy"] for site in evidence["parse"]["sites"]] == [60]
    resolved = metadata["course_contract"]["delivery"]["resolved_prescribed_dose_total_gy"]
    if declared:
        assert co.dose_classification["classification"] == org.DECLARED_CONCURRENT_COMPONENTS
        assert evidence["outcome"] == "UNRESOLVED"
        assert evidence["reason"] == "DECLARED_CONCURRENT_COMPONENTS_HAVE_NO_ADDITIVE_COURSE_TOTAL"
        assert "clinical_resolved_total_gy" not in evidence
        assert resolved is None
        assert metadata["prescribed_dose_scope"] != org.CLINICAL_RESOLVED_SCOPE
    else:
        assert co.dose_classification["classification"] == org.INDEPENDENT_DELIVERY_UNRECONCILED
        assert evidence["outcome"] == "RESOLVED_FROM_CLINICAL_RECORD"
        assert resolved == 60
        assert metadata["prescribed_dose_scope"] == org.CLINICAL_RESOLVED_SCOPE
        # Verify the transition as well as the undeclared control: declaring
        # this same course must withhold the register-derived course scalar.
        declare(tmp_path, monkeypatch, fixture["uids"])
        config.resume = True
        declared_course, = org.organize_and_merge(config)
        assert declared_course.dose_classification["classification"] == org.DECLARED_CONCURRENT_COMPONENTS
        assert declared_course.course_contract["delivery"]["resolved_prescribed_dose_total_gy"] is None


# Independent review r2 (2026-11-02): coverage on the publication path, the summation's geometry validator, broken files.
def scrub_refusals(files):
    """Published files with the per-course 'declaration not applied' warning removed; also returns how many were removed."""
    removed = 0

    def walk(value):
        nonlocal removed
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            kept = [v for v in value if not (isinstance(v, str) and v.startswith("Concurrent-components declaration not applied"))]
            removed += len(value) - len(kept)
            return [walk(v) for v in kept]
        return value

    out = {}
    for name, text in files.items():
        if name.endswith(".json"):
            out[name] = json.dumps(walk(json.loads(text)), sort_keys=True)
        elif name.endswith(".xlsx"):
            out[name] = json.dumps([[walk(literal(c)) for c in row] for row in json.loads(text)], sort_keys=True)
        else:
            out[name] = text
    return out, removed


def literal(cell):
    """A workbook cell holding a Python repr (e.g. the course contract) as a value, else the cell itself."""
    import ast
    if isinstance(cell, str) and cell[:1] in "{[":
        try:
            return ast.literal_eval(cell)
        except (ValueError, SyntaxError):
            return cell
    return cell


def parsed_cells(files):
    """Published files with workbook cells parsed as in scrub_refusals, nothing removed."""
    return {name: (json.dumps([[literal(c) for c in row] for row in json.loads(text)], sort_keys=True)
                   if name.endswith(".xlsx") else text) for name, text in files.items()}


def dose_identity(co):
    """Selected doses by SOP UID and summation type (paths differ between output roots)."""
    return [(row["sop_instance_uid"], row.get("dose_summation_type")) for row in co.selected_dose_contract]


def test_publication_coverage_sees_a_dose_the_eligibility_filter_drops(tmp_path, monkeypatch):
    """A third approved plan whose only PLAN dose also names an unknown SOP is dropped by eligible_dose_paths before
    classification; coverage must still see it, so a declaration of the other two plans is refused."""
    root = tmp_path / "input"
    fixture = course(root)
    plan, dose = third_plan(fixture, root)

    def unknown_reference(ds):
        ref = Dataset()
        ref.ReferencedSOPClassUID = ds.ReferencedRTPlanSequence[0].ReferencedSOPClassUID
        ref.ReferencedSOPInstanceUID = generate_uid()
        ds.ReferencedRTPlanSequence.append(ref)
    change(dose, unknown_reference)
    monkeypatch.delenv(ENV, raising=False)
    baseline = run_organizer(root, tmp_path / "baseline" / "output")
    declare(tmp_path, monkeypatch, fixture["uids"])                 # names the first two plans only
    courses = run_organizer(root, tmp_path / "declared" / "output")
    assert len(courses) == len(baseline) == 1
    co, base = courses[0], baseline[0]
    assert co.dose_classification["classification"] == base.dose_classification["classification"]
    assert co.dose_classification["classification"] != org.DECLARED_CONCURRENT_COMPONENTS
    assert dose_identity(co) == dose_identity(base)
    assert any("not applied" in w and "does not cover every approved plan" in w
               for w in co.dose_classification["warnings"])


def test_classifier_coverage_reads_the_unfiltered_course_dose_list(tmp_path, monkeypatch):
    """As _process_course calls it: eligible doses as dose_paths, every course dose file as course_dose_paths."""
    fixture = course(tmp_path / "dicom")
    junk = tmp_path / "dicom" / "d_junk.dcm"
    junk.write_bytes(b"not a DICOM file")
    declare(tmp_path, monkeypatch, fixture["uids"])
    result = org._classify_doses(fixture["plans"], fixture["doses"], structure_set_paths=fixture["structs"],
                                 course_dose_paths=fixture["doses"] + [junk])
    assert result.classification == org.INDEPENDENT_DELIVERY_UNRECONCILED
    assert any("not applied" in w and "cannot be read" in w for w in result.warnings)
    assert org._classify_doses(fixture["plans"], fixture["doses"], structure_set_paths=fixture["structs"],
                               course_dose_paths=fixture["doses"]).classification == org.DECLARED_CONCURRENT_COMPONENTS


def test_declared_dose_without_pixel_spacing_keeps_the_course_on_the_legacy_selection(tmp_path, monkeypatch):
    """The gate applies the summation's own geometry validator; a dose it would reject never reaches the sum."""
    root = tmp_path / "input"
    fixture = course(root)
    change(fixture["doses"][0], lambda ds: ds.__delattr__("PixelSpacing"))   # the earlier plan; legacy keeps the later
    monkeypatch.delenv(ENV, raising=False)
    baseline = run_organizer(root, tmp_path / "baseline" / "output")
    declare(tmp_path, monkeypatch, fixture["uids"])
    courses = run_organizer(root, tmp_path / "declared" / "output")
    assert len(courses) == len(baseline) == 1, "the course must still publish"
    co = courses[0]
    assert co.dose_classification["classification"] == baseline[0].dose_classification["classification"] \
        == org.INDEPENDENT_DELIVERY_UNRECONCILED
    assert dose_identity(co) == dose_identity(baseline[0]) and len(co.selected_dose_contract) == 0
    assert any("incomplete or invalid grid geometry" in w for w in co.dose_classification["warnings"])


@pytest.mark.parametrize("broken", ["missing", "not_json", "no_list"])
def test_broken_declaration_file_adds_only_its_warning(tmp_path, monkeypatch, broken):
    """A declaration file that cannot be used leaves classification, selection and every published byte as without the
    variable, except the one 'declaration not applied' warning on the course (visible by design)."""
    fixture = course(tmp_path / "input")
    monkeypatch.delenv(ENV, raising=False)
    run_organizer(tmp_path / "input", tmp_path / "unset" / "output")
    path = tmp_path / "broken.json"
    if broken == "not_json":
        path.write_text("{")
    elif broken == "no_list":
        path.write_text(json.dumps({"declarations": "none"}))
    monkeypatch.setenv(ENV, str(path))
    run_organizer(tmp_path / "input", tmp_path / "broken" / "output")
    reference = parsed_cells(published(tmp_path / "unset" / "output"))
    actual, removed = scrub_refusals(published(tmp_path / "broken" / "output"))
    assert removed >= 1
    assert actual == reference


# Independent review r3 (2026-11-03): pixel payload, course-level sums, unreadable unrelated structure sets, record-linked
# holds, multi-course isolation.
def test_declared_dose_with_truncated_pixels_keeps_the_course_on_the_legacy_selection(tmp_path, monkeypatch):
    root = tmp_path / "input"
    fixture = course(root)

    def truncate(ds):
        ds.PixelData = ds.PixelData[: len(ds.PixelData) // 2]
    change(fixture["doses"][0], truncate)                       # header valid, payload short; legacy keeps the later plan
    monkeypatch.delenv(ENV, raising=False)
    baseline = run_organizer(root, tmp_path / "baseline" / "output")
    declare(tmp_path, monkeypatch, fixture["uids"])
    courses = run_organizer(root, tmp_path / "declared" / "output")
    assert len(courses) == len(baseline) == 1, "the course must still publish"
    co = courses[0]
    assert co.dose_classification["classification"] == baseline[0].dose_classification["classification"]
    assert dose_identity(co) == dose_identity(baseline[0])
    assert any("invalid grid geometry or pixel data" in w for w in co.dose_classification["warnings"])


def add_dose(fixture, root, name, summation, uids):
    """A further RTDOSE of the course with the given summation type referencing the given plan UIDs."""
    path = copy_dicom(fixture["doses"][0], root / name, SOPInstanceUID=generate_uid())

    def edit(ds):
        ds.DoseSummationType = summation
        template = ds.ReferencedRTPlanSequence[0]
        refs = []
        for uid in uids:
            ref = Dataset()
            ref.ReferencedSOPClassUID = template.ReferencedSOPClassUID
            ref.ReferencedSOPInstanceUID = uid
            refs.append(ref)
        ds.ReferencedRTPlanSequence = Sequence(refs)
    change(path, edit)
    return path


@pytest.mark.parametrize("summation", ["MULTI_PLAN", "PLAN_SUM"])
def test_course_level_sum_of_declared_plans_leaves_the_legacy_selection(tmp_path, monkeypatch, legacy, summation):
    """A TPS course-level dose of the declared plans is what 74c25b9 prefers; the declaration does not override it."""
    root = tmp_path / "dicom"
    fixture = course(root)
    fixture["doses"].append(add_dose(fixture, root, "dsum.dcm", summation, fixture["uids"]))
    monkeypatch.delenv(ENV, raising=False)
    expected = asdict(legacy._classify_doses(fixture["plans"], fixture["doses"]))
    declare(tmp_path, monkeypatch, fixture["uids"])
    actual = asdict(classify(fixture))
    warnings = actual.pop("warnings")
    expected.pop("warnings")
    assert actual == expected
    assert any("not applied" in w and f"course-level RTDOSE ({summation})" in w for w in warnings)


def test_beam_and_unsupported_doses_of_declared_plans_are_excluded_with_the_legacy_warnings(tmp_path, monkeypatch):
    root = tmp_path / "dicom"
    fixture = course(root)
    beam = add_dose(fixture, root, "dbeam.dcm", "BEAM", fixture["uids"][:1])
    brachy = add_dose(fixture, root, "dbrachy.dcm", "BRACHY", fixture["uids"][1:])
    fixture["doses"] += [beam, brachy]
    declare(tmp_path, monkeypatch, fixture["uids"])
    result = classify(fixture)
    assert result.classification == org.DECLARED_CONCURRENT_COMPONENTS
    assert result.selected_doses == fixture["doses"][:2]
    assert set(result.excluded_doses) == {beam, brachy}
    assert "Excluded 1 BEAM dose(s) because matching PLAN dose(s) exist" in result.warnings
    assert "Excluded unsupported RTDOSE summation types: BRACHY" in result.warnings


def test_unreadable_structure_set_the_plans_do_not_reference_is_ignored(tmp_path, monkeypatch):
    fixture = course(tmp_path / "dicom")
    junk = tmp_path / "dicom" / "rs_junk.dcm"
    junk.write_bytes(b"not a DICOM file")
    fixture["structs"].append(junk)
    declare(tmp_path, monkeypatch, fixture["uids"])
    assert classify(fixture).classification == org.DECLARED_CONCURRENT_COMPONENTS
    fixture["structs"] = [junk]                                    # the referenced set itself unreadable: refused
    result = classify(fixture)
    assert result.classification != org.DECLARED_CONCURRENT_COMPONENTS
    assert any("referenced RTSTRUCT missing or ambiguous" in w for w in result.warnings)


@pytest.mark.parametrize("shared_days", [False, True])
def test_record_linked_declared_course_holds(tmp_path, shared_days):
    """With record-linked (calculated) doses at distinct references, a declared course carries the record mismatch hold
    and the declaration hold, both withholding the course scalar; no sequential-total sentence."""
    from record_delivery_synthetic_rt import BEAM_DOSES, PLAN_B, REF_B, complete_session, make_plan as record_plan
    first = record_plan(tmp_path / "plan_a.dcm")
    second = record_plan(tmp_path / "plan_b.dcm", plan_uid=PLAN_B, reference_uid=REF_B, beam_reference_uid=REF_B)
    days = (1, 2) if shared_days else (10, 11)
    records = complete_session(tmp_path, 1, doses=BEAM_DOSES) + complete_session(tmp_path, 2, doses=BEAM_DOSES)
    records += complete_session(tmp_path, 1, plan_uid=PLAN_B, day=days[0], doses=BEAM_DOSES)
    records += complete_session(tmp_path, 2, plan_uid=PLAN_B, day=days[1], doses=BEAM_DOSES)
    summary = org._calculate_delivery_summary([first, second], records,
                                              classification=org.DECLARED_CONCURRENT_COMPONENTS)
    assert [item["reason_code"] for item in summary["delivery_course_holds"]] == [
        "COURSE_DOSE_REFERENCE_UID_MISMATCH", "COURSE_DOSE_CONCURRENT_DISTINCT_REFERENCES"]
    assert summary["delivered_dose_gy"] is None
    assert not any("sequential total" in w for w in summary["delivery_warnings"])


def second_course(root, fixture):
    """Copy the course as a later, separate course of the same patient (new frame, study, structure set, plans, doses)."""
    import shutil
    other = root.parent / "second"
    course(other)
    moved = []
    for path in sorted(other.iterdir()):
        target = root / f"c2_{path.name}"
        shutil.move(str(path), target)
        moved.append(target)
    plans = sorted(p for p in moved if p.name.startswith("c2_p"))
    for index, plan in enumerate(plans):
        change(plan, lambda ds, i=index: setattr(ds, "RTPlanDate", f"2024060{i + 1}"))
    return [str(pydicom.dcmread(p).SOPInstanceUID) for p in plans], plans


def test_declaration_for_one_course_leaves_the_patients_other_course_unchanged(tmp_path, monkeypatch):
    root = tmp_path / "input"
    fixture = course(root)
    second_course(root, fixture)
    monkeypatch.delenv(ENV, raising=False)
    run_organizer(root, tmp_path / "unset" / "output")
    reference = published(tmp_path / "unset" / "output")
    declare(tmp_path, monkeypatch, fixture["uids"])                  # names the first course only
    courses = run_organizer(root, tmp_path / "declared" / "output")
    classes = sorted(co.dose_classification["classification"] for co in courses)
    assert len(courses) == 2 and classes.count(org.DECLARED_CONCURRENT_COMPONENTS) == 1
    declared_dir = next(co.dirs.root for co in courses
                        if co.dose_classification["classification"] == org.DECLARED_CONCURRENT_COMPONENTS)
    prefix = str(declared_dir.relative_to(tmp_path / "declared" / "output")) + "/"
    actual = published(tmp_path / "declared" / "output")
    other = {k: v for k, v in actual.items() if k.startswith("P1/") and not k.startswith(prefix)}
    assert other and other == {k: v for k, v in reference.items() if k in other}


def test_entry_spanning_two_courses_is_refused_on_both(tmp_path, monkeypatch):
    root = tmp_path / "input"
    fixture = course(root)
    second_uids, _ = second_course(root, fixture)
    monkeypatch.delenv(ENV, raising=False)
    baseline = run_organizer(root, tmp_path / "unset" / "output")
    declare(tmp_path, monkeypatch, [fixture["uids"][0], second_uids[0]])
    courses = run_organizer(root, tmp_path / "declared" / "output")
    assert len(courses) == len(baseline) == 2
    for co in courses:
        assert co.dose_classification["classification"] != org.DECLARED_CONCURRENT_COMPONENTS
        assert any("not applied" in w and "not among the course's approved plans" in w
                   for w in co.dose_classification["warnings"])
    assert sorted(dose_identity(co) for co in courses) == sorted(dose_identity(co) for co in baseline)
