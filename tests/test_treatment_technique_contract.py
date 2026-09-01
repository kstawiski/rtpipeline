from __future__ import annotations

from pathlib import Path

from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, RTPlanStorage, generate_uid

from course_contract_test_utils import (
    write_minimal_course_contract,
    write_synthetic_plan_and_dose,
)
from rtpipeline.course_contract import (
    CourseContractError,
    build_treatment_technique_contract,
    load_course_contract,
)
from rtpipeline.prescription import resolve_plan_prescriptions


def _write_plan(path: Path, *, brachy: bool, ebrt: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sop_uid = generate_uid()
    meta = FileMetaDataset()
    sop_class = RTPlanStorage
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    plan = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    plan.SOPClassUID = sop_class
    plan.SOPInstanceUID = sop_uid
    plan.Modality = "RTPLAN"
    plan.PatientID = "P1"
    plan.StudyInstanceUID = generate_uid()
    plan.SeriesInstanceUID = generate_uid()

    fraction_group = Dataset()
    fraction_group.FractionGroupNumber = 1
    fraction_group.NumberOfFractionsPlanned = 1
    fraction_group.NumberOfBeams = int(ebrt)
    fraction_group.NumberOfBrachyApplicationSetups = int(brachy)
    if ebrt:
        referenced_beam = Dataset()
        referenced_beam.ReferencedBeamNumber = 1
        referenced_beam.BeamDose = 2.0
        fraction_group.ReferencedBeamSequence = Sequence([referenced_beam])
        beam = Dataset()
        beam.BeamNumber = 1
        beam.TreatmentDeliveryType = "TREATMENT"
        plan.BeamSequence = Sequence([beam])
    if brachy:
        referenced_setup = Dataset()
        referenced_setup.ReferencedBrachyApplicationSetupNumber = 1
        referenced_setup.BrachyApplicationSetupDose = 5.0
        fraction_group.ReferencedBrachyApplicationSetupSequence = Sequence(
            [referenced_setup]
        )
    plan.FractionGroupSequence = Sequence([fraction_group])
    plan.save_as(str(path), enforce_file_format=True)
    return path


def test_brachy_plan_is_classified_from_fraction_group_evidence(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path / "brachy.dcm", brachy=True, ebrt=False)

    contract = build_treatment_technique_contract([plan], course_dir=tmp_path)

    assert contract["classification"] == "BRACHYTHERAPY"
    assert contract["dose_response_eligible"] is False
    assert contract["prescription_relative_dvh_metrics"] == "suppressed"
    evidence = contract["plan_evidence"][0]
    assert evidence["sop_class_uid"] == str(RTPlanStorage)
    assert evidence["sop_class_profile"] == "standard_rt_plan"
    assert evidence["number_of_brachy_application_setups"] == 1
    assert evidence["referenced_brachy_application_setup_count"] == 1
    assert evidence["beam_sequence_count"] == 0
    assert evidence["referenced_beam_count"] == 0


def test_brachy_application_setup_dose_is_a_prescription_without_beams(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path / "brachy.dcm", brachy=True, ebrt=False)

    import pydicom

    groups = resolve_plan_prescriptions(pydicom.dcmread(str(plan)))

    assert groups[0]["source_prescribed_dose_gy"] == 5.0
    assert groups[0]["resolved_prescribed_dose_total_gy"] == 5.0
    assert groups[0]["prescription_resolution_method"] == "BRACHY_APPLICATION_SETUP_DOSE_V1"

def test_ebrt_plan_is_classified_from_beam_evidence(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path / "ebrt.dcm", brachy=False, ebrt=True)

    contract = build_treatment_technique_contract([plan], course_dir=tmp_path)

    assert contract["classification"] == "EBRT"
    assert contract["dose_response_eligible"] is True
    assert contract["prescription_relative_dvh_metrics"] == "available_when_prescription_resolved"


def test_course_with_brachy_and_beam_evidence_is_mixed(tmp_path: Path) -> None:
    ebrt = _write_plan(tmp_path / "ebrt.dcm", brachy=False, ebrt=True)
    brachy = _write_plan(tmp_path / "brachy.dcm", brachy=True, ebrt=False)

    contract = build_treatment_technique_contract(
        [ebrt, brachy], course_dir=tmp_path
    )

    assert contract["classification"] == "MIXED"
    assert contract["dose_response_eligible"] is False
    assert {row["classification"] for row in contract["plan_evidence"]} == {
        "EBRT",
        "BRACHYTHERAPY",
    }


def test_loaded_course_contract_exposes_validated_technique(tmp_path: Path) -> None:
    course = tmp_path / "P1" / "C1"
    plan, dose = write_synthetic_plan_and_dose(course)
    write_minimal_course_contract(
        course, selected_plans=[plan], selected_doses=[dose]
    )

    contract = load_course_contract(course)

    assert contract.treatment_technique["classification"] == "EBRT"


def test_course_contract_rejects_stale_technique_evidence(tmp_path: Path) -> None:
    course = tmp_path / "P1" / "C1"
    plan, dose = write_synthetic_plan_and_dose(course)
    metadata = write_minimal_course_contract(
        course, selected_plans=[plan], selected_doses=[dose]
    )
    payload = __import__("json").loads(metadata.read_text(encoding="utf-8"))
    payload["course_contract"]["treatment_technique"]["classification"] = (
        "BRACHYTHERAPY"
    )
    metadata.write_text(__import__("json").dumps(payload), encoding="utf-8")

    try:
        load_course_contract(course)
    except CourseContractError as exc:
        assert "treatment_technique" in str(exc)
    else:
        raise AssertionError("stale treatment-technique evidence was accepted")
