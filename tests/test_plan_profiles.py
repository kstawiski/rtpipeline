"""Vendor-profile plans are admissible as sources but not as synthesised artifacts.

Varian "RT Plan Varian 1 Storage" (1.2.246.352.70.1.70) is a full RT Plan extended
for the dual-layer Halcyon MLC. Rejecting it dropped real Halcyon/Ethos plans from
a bladder cohort. Accepting it through one flat allowlist would have been worse:
_create_summed_plan clones its first source including the SOP class, so the
pipeline could mint a synthetic object claiming a vendor profile, with Ethos
per-phase fraction counts summed into a whole-course denominator.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from rtpipeline import plan_profiles
from rtpipeline.course_contract import _ROLE_EXPECTATIONS
from rtpipeline.organize import PlanSummationUnsupportedError, _create_summed_plan

VARIAN = plan_profiles.VARIAN_RT_PLAN_1_SOP_CLASS
STANDARD = plan_profiles.STANDARD_RT_PLAN_SOP_CLASS
BRACHY_RECORD = plan_profiles.RT_BRACHY_TREATMENT_RECORD_SOP_CLASS


def _plan(path: Path, sop_class: str, *, rx: float = 20.0, fractions: int = 5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = sop_class
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=fm, preamble=b"\0" * 128)
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
    ds.Modality = "RTPLAN"
    ds.PatientID = "P1"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.RTPlanLabel = "VA107"
    ds.ApprovalStatus = "APPROVED"
    ds.PlanIntent = "CURATIVE"
    dr = Dataset()
    dr.DoseReferenceNumber = 1
    dr.TargetPrescriptionDose = rx
    ds.DoseReferenceSequence = [dr]
    fg = Dataset()
    fg.NumberOfFractionsPlanned = fractions
    ds.FractionGroupSequence = [fg]
    ds.save_as(path, enforce_file_format=True)
    return path


def test_source_role_admits_the_varian_profile() -> None:
    modalities, classes = _ROLE_EXPECTATIONS["RTPLAN_SOURCE"]
    assert VARIAN in classes and STANDARD in classes
    assert "RTPLAN" in modalities


def test_rt_brachy_treatment_record_is_not_an_rtplan_source() -> None:
    _modalities, classes = _ROLE_EXPECTATIONS["RTPLAN_SOURCE"]

    assert BRACHY_RECORD not in classes
    assert plan_profiles.plan_profile_name(BRACHY_RECORD) == "not_rt_plan"


def test_derived_role_refuses_the_varian_profile() -> None:
    _modalities, classes = _ROLE_EXPECTATIONS["RTPLAN_DERIVED"]
    assert STANDARD in classes
    assert VARIAN not in classes, "a synthesised plan must not claim a vendor profile"


def test_plan_sum_refuses_vendor_profile_sources(tmp_path: Path) -> None:
    plans = [
        _plan(tmp_path / "a.dcm", VARIAN),
        _plan(tmp_path / "b.dcm", VARIAN),
    ]
    with pytest.raises(PlanSummationUnsupportedError, match="vendor-profile"):
        _create_summed_plan(plans, 40.0)


def test_plan_sum_still_works_for_standard_sources(tmp_path: Path) -> None:
    plans = [
        _plan(tmp_path / "c.dcm", STANDARD),
        _plan(tmp_path / "d.dcm", STANDARD),
    ]
    plan_sum, _datasets, _uids = _create_summed_plan(plans, 40.0)
    assert str(plan_sum.SOPClassUID) == STANDARD
    assert str(plan_sum.file_meta.MediaStorageSOPInstanceUID) == str(plan_sum.SOPInstanceUID)
    assert str(getattr(plan_sum, "ApprovalStatus", "")) != "APPROVED"


def test_fraction_semantics_flagged_for_vendor_profile() -> None:
    assert plan_profiles.fraction_count_semantics(VARIAN) != "whole_course"
    assert plan_profiles.fraction_count_semantics(STANDARD) == "whole_course"


def test_profile_names_are_reported() -> None:
    assert plan_profiles.plan_profile_name(VARIAN) == "varian_rt_plan_1"
    assert plan_profiles.plan_profile_name(STANDARD) == "standard_rt_plan"
    assert plan_profiles.plan_profile_name("9.9.9") == "unknown"
