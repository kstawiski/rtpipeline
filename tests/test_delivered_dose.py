"""Synthetic regressions for delivered-dose accounting and fail-closed discovery."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pydicom
import pytest
from pydicom.datadict import tag_for_keyword
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import (
    ExplicitVRLittleEndian,
    RTBeamsTreatmentRecordStorage,
    RTPlanStorage,
    RTTreatmentSummaryRecordStorage,
    UID,
    generate_uid,
)

from rtpipeline import meta
from rtpipeline import cli
from rtpipeline import organize
from rtpipeline.config import PipelineConfig


def _file_dataset(path: Path, sop_class: str, sop_uid: str) -> FileDataset:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = UID(sop_class)
    file_meta.MediaStorageSOPInstanceUID = UID(sop_uid)
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = sop_uid
    ds.PatientID = "P1"
    return ds


def _write(ds: FileDataset, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), write_like_original=False)
    return path


def _plan(path: Path, uid: str, rx: float, fractions: int) -> Path:
    ds = _file_dataset(path, RTPlanStorage, uid)
    ds.Modality = "RTPLAN"
    ds.RTPlanDate = "20240101"
    dose_ref = Dataset()
    dose_ref.DoseReferenceNumber = "1"
    dose_ref.DoseReferenceUID = generate_uid()
    dose_ref.DoseReferenceType = "TARGET"
    dose_ref.TargetPrescriptionDose = rx
    ds.DoseReferenceSequence = Sequence([dose_ref])
    fraction_group = Dataset()
    fraction_group.FractionGroupNumber = 1
    fraction_group.NumberOfFractionsPlanned = fractions
    fraction_group.NumberOfBeams = 1
    beam_reference = Dataset()
    beam_reference.ReferencedBeamNumber = 1
    beam_reference.BeamDose = rx / fractions
    beam_reference.BeamDoseType = "PHYSICAL"
    target_binding = Dataset()
    target_binding.ReferencedDoseReferenceUID = dose_ref.DoseReferenceUID
    beam_reference.ReferencedDoseReferenceSequence = Sequence([target_binding])
    fraction_group.ReferencedBeamSequence = Sequence([beam_reference])
    ds.FractionGroupSequence = Sequence([fraction_group])
    beam = Dataset()
    beam.BeamNumber = 1
    beam.TreatmentDeliveryType = "TREATMENT"
    ds.BeamSequence = Sequence([beam])
    return _write(ds, path)


def _record(path: Path, uid: str, plan_uid: str, treatment_date: str, dose_gy: float | None = None) -> Path:
    ds = _file_dataset(path, RTBeamsTreatmentRecordStorage, uid)
    ds.Modality = "RTRECORD"
    ds.TreatmentDate = treatment_date
    ref_plan = Dataset()
    ref_plan.ReferencedSOPClassUID = RTPlanStorage
    ref_plan.ReferencedSOPInstanceUID = plan_uid
    ds.ReferencedRTPlanSequence = Sequence([ref_plan])
    session = Dataset()
    session.CurrentFractionNumber = int(treatment_date[-2:])
    session.TreatmentDeliveryType = "TREATMENT"
    session.TreatmentTerminationStatus = "NORMAL"
    session.ReferencedBeamNumber = 1
    ds.TreatmentSessionBeamSequence = Sequence([session])
    if dose_gy is not None:
        dose_ref = Dataset()
        dose_ref.CalculatedDoseReferenceDoseValue = dose_gy
        dose_ref.ReferencedDoseReferenceNumber = "1"
        ds.ReferencedCalculatedDoseReferenceSequence = Sequence([dose_ref])
    return _write(ds, path)


def test_419783_shape_uses_delivered_fractions_and_keeps_prescription(tmp_path):
    plan_abandoned = _plan(tmp_path / "plan_abandoned.dcm", generate_uid(), 50.0, 25)
    plan_completed = _plan(tmp_path / "plan_completed.dcm", generate_uid(), 25.0, 10)
    abandoned_uid = str(pydicom.dcmread(str(plan_abandoned), stop_before_pixels=True).SOPInstanceUID)
    completed_uid = str(pydicom.dcmread(str(plan_completed), stop_before_pixels=True).SOPInstanceUID)
    records = [
        _record(tmp_path / f"abandoned_{index}.dcm", generate_uid(), abandoned_uid, f"202402{index + 1:02d}", 2.0)
        for index in range(6)
    ]
    records.extend(
        _record(tmp_path / f"completed_{index}.dcm", generate_uid(), completed_uid, f"202403{index + 1:02d}", 2.5)
        for index in range(10)
    )

    summary = organize._calculate_delivery_summary([plan_abandoned, plan_completed], records)

    assert summary["delivered_dose_gy"] == pytest.approx(37.0)
    assert summary["delivery_status"] == "partially_delivered"
    assert summary["delivered_record_count"] == 16
    assert summary["delivery_plan_details"][0]["prescribed_dose_gy"] == pytest.approx(50.0)
    assert sum(float(row["prescribed_dose_gy"]) for row in summary["delivery_plan_details"]) == pytest.approx(75.0)


def test_record_dose_reference_is_preferred_when_available(tmp_path):
    plan = _plan(tmp_path / "plan.dcm", generate_uid(), 25.0, 10)
    import pydicom

    plan_uid = str(pydicom.dcmread(str(plan), stop_before_pixels=True).SOPInstanceUID)
    records = [_record(tmp_path / f"record_{index}.dcm", generate_uid(), plan_uid, f"202402{index + 1:02d}", 2.5) for index in range(10)]

    summary = organize._calculate_delivery_summary([plan], records)

    assert summary["delivered_dose_gy"] == pytest.approx(25.0)
    assert summary["delivery_method"] == "calculated_dose_reference"


def test_no_treatment_records_are_unknown_not_zero(tmp_path):
    plan = _plan(tmp_path / "plan.dcm", generate_uid(), 50.0, 25)

    summary = organize._calculate_delivery_summary([plan], [])

    assert summary["delivered_dose_gy"] is None
    assert summary["delivery_status"] == "no_records_at_all"


@pytest.mark.parametrize(
    ("delivery_type", "termination_status"),
    [("TRMT_PORTFILM", "NORMAL"), ("TREATMENT", "MACHINE")],
)
def test_noncompleted_or_nontreatment_record_is_not_a_delivered_fraction(
    tmp_path,
    delivery_type,
    termination_status,
):
    plan = _plan(tmp_path / "plan.dcm", generate_uid(), 50.0, 25)
    plan_uid = str(pydicom.dcmread(str(plan), stop_before_pixels=True).SOPInstanceUID)
    record = _record(
        tmp_path / "record.dcm",
        generate_uid(),
        plan_uid,
        "20240201",
    )
    dataset = pydicom.dcmread(str(record))
    session = dataset.TreatmentSessionBeamSequence[0]
    session.TreatmentDeliveryType = delivery_type
    session.TreatmentTerminationStatus = termination_status
    dataset.save_as(str(record), write_like_original=False)

    summary = organize._calculate_delivery_summary([plan], [record])

    assert summary["delivered_dose_gy"] is None
    assert summary["delivered_fraction_count"] == 0
    assert summary["delivery_status"] == "delivered_but_records_absent"


def test_records_with_unresolved_prescription_scope_do_not_create_delivery_state(
    tmp_path,
):
    plan = _plan(tmp_path / "plan.dcm", generate_uid(), 50.0, 25)
    dataset = pydicom.dcmread(str(plan))
    del dataset.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamDose
    dataset.save_as(str(plan), write_like_original=False)
    plan_uid = str(dataset.SOPInstanceUID)
    record = _record(
        tmp_path / "record.dcm",
        generate_uid(),
        plan_uid,
        "20240201",
        2.0,
    )

    summary = organize._calculate_delivery_summary([plan], [record])

    assert summary["delivered_dose_gy"] is None
    assert summary["delivery_status"] == "delivery_unresolved"
    details = summary["delivery_plan_details"]
    assert isinstance(details, list)
    assert details[0]["status"] == "delivery_unresolved"
    assert (
        details[0]["prescription_resolution_method"]
        == "UNRESOLVED_INCOMPLETE_BEAM_MEMBERSHIP"
    )
    assert summary["delivered_dose_gy"] != 0.0
    assert summary["delivered_dose_gy"] != 50.0


def test_fully_delivered_plan_matches_prescription(tmp_path):
    plan = _plan(tmp_path / "plan.dcm", generate_uid(), 30.0, 3)
    plan_uid = str(pydicom.dcmread(str(plan), stop_before_pixels=True).SOPInstanceUID)
    records = [_record(tmp_path / f"record_{index}.dcm", generate_uid(), plan_uid, f"202402{index + 1:02d}") for index in range(3)]

    summary = organize._calculate_delivery_summary([plan], records)

    assert summary["delivered_dose_gy"] == pytest.approx(30.0)
    assert summary["delivery_status"] == "fully_delivered"


def test_multiple_beam_records_on_one_date_count_as_one_fraction(tmp_path):
    plan = _plan(tmp_path / "plan.dcm", generate_uid(), 30.0, 3)
    plan_uid = str(pydicom.dcmread(str(plan), stop_before_pixels=True).SOPInstanceUID)
    records = []
    for day in range(3):
        for beam in range(2):
            records.append(
                _record(
                    tmp_path / f"record_{day}_{beam}.dcm",
                    generate_uid(),
                    plan_uid,
                    f"202402{day + 1:02d}",
                )
            )

    summary = organize._calculate_delivery_summary([plan], records)

    assert summary["delivered_record_count"] == 6
    assert summary["delivered_fraction_count"] == 3
    assert summary["delivered_dose_gy"] == pytest.approx(30.0)
    assert summary["delivery_status"] == "fully_delivered"


def test_duplicate_explicit_dose_records_are_not_summed_twice(tmp_path):
    plan = _plan(tmp_path / "plan.dcm", generate_uid(), 30.0, 3)
    plan_uid = str(pydicom.dcmread(str(plan), stop_before_pixels=True).SOPInstanceUID)
    records = []
    for day in range(3):
        for duplicate in range(2):
            records.append(
                _record(
                    tmp_path / f"record_{day}_{duplicate}.dcm",
                    generate_uid(),
                    plan_uid,
                    f"202402{day + 1:02d}",
                    10.0,
                )
            )

    summary = organize._calculate_delivery_summary([plan], records)

    assert summary["delivered_record_count"] == 6
    assert summary["delivered_fraction_count"] == 3
    assert summary["delivered_dose_gy"] == pytest.approx(30.0)


def test_latest_cumulative_dose_is_not_summed_with_prior_observations(tmp_path):
    plan = _plan(tmp_path / "plan.dcm", generate_uid(), 30.0, 3)
    plan_uid = str(pydicom.dcmread(str(plan), stop_before_pixels=True).SOPInstanceUID)
    records = []
    for index, dose in enumerate((10.0, 20.0, 30.0), start=1):
        path = tmp_path / f"summary_{index}.dcm"
        ds = _file_dataset(path, RTTreatmentSummaryRecordStorage, generate_uid())
        ds.Modality = "RTRECORD"
        ds.TreatmentDate = f"202402{index:02d}"
        ref_plan = Dataset()
        ref_plan.ReferencedSOPInstanceUID = plan_uid
        ds.ReferencedRTPlanSequence = Sequence([ref_plan])
        dose_ref = Dataset()
        dose_ref.CumulativeDoseToDoseReference = dose
        dose_ref.ReferencedDoseReferenceNumber = "1"
        ds.TreatmentSummaryCalculatedDoseReferenceSequence = Sequence([dose_ref])
        records.append(_write(ds, path))

    summary = organize._calculate_delivery_summary([plan], records)

    assert summary["delivered_dose_gy"] == pytest.approx(30.0)
    assert summary["delivery_method"] == "cumulative_dose_reference"
    assert summary["delivered_fraction_count"] == 0


def test_record_dose_keyword_is_real_dicom_keyword():
    assert tag_for_keyword("CalculatedDoseReferenceDoseValue") is not None
    assert tag_for_keyword("DeliveredDoseReferenceDoseValue") is None


def test_record_level_dose_from_non_target_reference_falls_back(tmp_path):
    plan = _plan(tmp_path / "plan.dcm", generate_uid(), 30.0, 3)
    plan_ds = pydicom.dcmread(str(plan), stop_before_pixels=True)
    oar = Dataset()
    oar.DoseReferenceNumber = "2"
    oar.DoseReferenceType = "ORGAN_AT_RISK"
    oar.TargetPrescriptionDose = 3.0
    plan_ds.DoseReferenceSequence.append(oar)
    plan_ds.save_as(str(plan), write_like_original=False)
    plan_uid = str(plan_ds.SOPInstanceUID)
    records = []
    for index in range(3):
        path = tmp_path / f"record_{index}.dcm"
        _record(path, generate_uid(), plan_uid, f"202402{index + 1:02d}", 3.0)
        ds = pydicom.dcmread(str(path), force=True)
        ds.ReferencedCalculatedDoseReferenceSequence[0].ReferencedDoseReferenceNumber = "2"
        ds.save_as(str(path), write_like_original=False)
        records.append(path)

    summary = organize._calculate_delivery_summary([plan], records)

    assert summary["delivered_dose_gy"] == pytest.approx(30.0)
    assert summary["delivery_method"] == "record_fraction_weighted_prescription"


def test_selected_recordless_plan_makes_course_dose_unknown(tmp_path):
    observed = _plan(tmp_path / "observed.dcm", generate_uid(), 20.0, 2)
    absent = _plan(tmp_path / "absent.dcm", generate_uid(), 30.0, 3)
    observed_uid = str(pydicom.dcmread(str(observed), stop_before_pixels=True).SOPInstanceUID)
    records = [_record(tmp_path / f"record_{i}.dcm", generate_uid(), observed_uid, f"202402{i + 1:02d}") for i in range(2)]

    summary = organize._calculate_delivery_summary(
        [observed, absent], records, selected_plan_paths=[observed, absent]
    )

    assert summary["delivered_dose_gy"] is None
    assert summary["delivery_plan_details"][1]["delivered_dose_gy"] is None


def test_dose_plausibility_warning_is_emitted_for_prescription_mismatch(tmp_path, caplog):
    plan = _plan(tmp_path / "plan.dcm", generate_uid(), 5.0, 1)
    plan_uid = str(pydicom.dcmread(str(plan), stop_before_pixels=True).SOPInstanceUID)
    records = [
        _record(tmp_path / f"record_{i}.dcm", generate_uid(), plan_uid, f"202402{i + 1:02d}", 5.0)
        for i in range(8)
    ]

    with caplog.at_level(logging.WARNING):
        summary = organize._calculate_delivery_summary([plan], records)

    assert summary["delivered_dose_gy"] == pytest.approx(40.0)
    assert any("exceeds resolved prescribed total" in message for message in caplog.messages)


def test_absent_plan_reference_is_counted_logged_and_not_attributed(tmp_path, caplog):
    plan = _plan(tmp_path / "plan.dcm", generate_uid(), 30.0, 3)
    unknown_uid = generate_uid()
    record = _record(tmp_path / "record.dcm", generate_uid(), unknown_uid, "20240201")

    with caplog.at_level(logging.WARNING):
        audit = organize._delivery_reference_audit([record], [plan.stem])
        summary = organize._calculate_delivery_summary([plan], [record], reference_audit=audit)

    assert audit["unresolved_reference_count"] == 1
    assert audit["unresolved_record_count"] == 1
    assert unknown_uid in audit["unresolved_plan_uids"]
    assert summary["delivered_dose_gy"] is None
    assert summary["delivery_status"] == "delivered_but_records_absent"
    assert unknown_uid in caplog.text
    assert "not be attributed" in caplog.text


def test_non_dicom_input_fails_closed_before_empty_manifest(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    (root / "not-dicom.txt").write_text("not DICOM", encoding="utf-8")
    config = PipelineConfig(
        dicom_root=root,
        output_root=tmp_path / "output",
        logs_root=tmp_path / "logs",
    )

    with pytest.raises(organize.OrganizeDiscoveryError, match="zero supported DICOM objects"):
        organize._raise_if_empty_organize_discovery(config, {}, [], [], [])


def test_symlinked_input_error_names_opt_in(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "object.dcm").write_bytes(b"not needed for discovery diagnostic")
    root = tmp_path / "input"
    root.mkdir()
    (root / "linked-patient").symlink_to(source, target_is_directory=True)
    config = PipelineConfig(
        dicom_root=root,
        output_root=tmp_path / "output",
        logs_root=tmp_path / "logs",
    )
    monkeypatch.delenv(organize.FOLLOW_INPUT_SYMLINKS_ENV, raising=False)

    with pytest.raises(organize.OrganizeDiscoveryError) as error:
        organize._raise_if_empty_organize_discovery(config, {}, [], [], [])

    assert organize.FOLLOW_INPUT_SYMLINKS_ENV in str(error.value)
    assert "symlinked directories" in str(error.value)
    assert "Refusing to write an empty manifest" in str(error.value)


def test_configurable_plausibility_warning_distinguishes_prescribed_and_delivered():
    prescribed_only = organize._dose_plausibility(105.0, 95.0, 100.0)
    delivered_only = organize._dose_plausibility(95.0, 105.0, 100.0)
    neither = organize._dose_plausibility(100.0, None, 100.0)

    assert prescribed_only["dose_plausibility_threshold_gy"] == 100.0
    assert prescribed_only["prescribed_dose_plausibility_warning"] is True
    assert prescribed_only["delivered_dose_plausibility_warning"] is False
    assert prescribed_only["dose_plausibility_warning"] is True
    assert prescribed_only["dose_qc_pass"] is False
    assert prescribed_only["dose_qc_status"] == "fail"
    assert "prescribed dose 105 Gy" in prescribed_only["dose_qc_reasons"][0]
    assert delivered_only["prescribed_dose_plausibility_warning"] is False
    assert delivered_only["delivered_dose_plausibility_warning"] is True
    assert delivered_only["dose_plausibility_warning"] is True
    assert neither["dose_plausibility_warning"] is False


def test_max_total_dose_is_loaded_from_project_yaml():
    config = PipelineConfig(
        dicom_root=Path("/tmp/input"),
        output_root=Path("/tmp/output"),
        logs_root=Path("/tmp/logs"),
    )

    cli._apply_dose_yaml_config(config, {"max_total_dose_gy": 40})

    assert config.max_total_dose_gy == pytest.approx(40.0)


def test_max_total_dose_cli_flag_is_exposed():
    args = cli.build_parser().parse_args(["--dicom-root", "/tmp/input", "--max-total-dose-gy", "40"])

    assert args.max_total_dose_gy == pytest.approx(40.0)


def test_configured_threshold_is_used_for_prescribed_and_delivered_warnings():
    config = PipelineConfig(
        dicom_root=Path("/tmp/input"),
        output_root=Path("/tmp/output"),
        logs_root=Path("/tmp/logs"),
    )
    cli._apply_dose_yaml_config(config, {"max_total_dose_gy": 40})

    warnings = organize._dose_plausibility(41.0, 41.0, config.max_total_dose_gy)

    assert warnings["prescribed_dose_plausibility_warning"] is True
    assert warnings["delivered_dose_plausibility_warning"] is True


def test_unresolved_dose_reference_does_not_use_filename_fallback(caplog):
    plans = pd.DataFrame(
        {
            "file_path": ["/x/RP.123.course.dcm"],
            "_sop_instance_uid": ["1.2.3"],
            "plan_value": ["plan"],
        }
    )
    doses = pd.DataFrame(
        {
            "file_path": ["/x/RD.123.course.dcm"],
            "_referenced_plan_sop_uids": [("9.9.9",)],
            "dose_value": ["dose"],
        }
    )

    with caplog.at_level(logging.WARNING):
        merged = meta._merge_plans_doses(plans, doses)

    assert merged.empty
    assert "RD.123.course.dcm" in caplog.text
    assert "9.9.9" in caplog.text
    assert "no filename fallback was used" in caplog.text
