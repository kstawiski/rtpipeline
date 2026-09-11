"""The combined ROI ledger must keep one accounted outcome per measured source.

``write_modality_ledger`` rebuilds ``radiomics_roi_ledger.json`` from the
per-modality ledgers.  A modality ledger already keeps the source identity that
makes two same-named ROI outcomes distinct, so the combined rebuild must not
collapse two MR series that measured the same ROI name from the same
segmentation source.  It must still collapse a row repeated verbatim, and a CT
ledger, which never carried these source-identity fields, must keep the rows and
denominators it had before.
"""
from __future__ import annotations

import json

from rtpipeline.roi_requiredness import DenominatorLedger, write_modality_ledger


def _course_states(**overrides):
    states = {
        "screened": True,
        "in_scope": True,
        "out_of_scope": False,
        "adequate_coverage": True,
        "insufficient_coverage": False,
        "valid_derivation": False,
        "technical_exclusion": False,
        "indeterminate": False,
        "extracted": True,
    }
    states.update(overrides)
    return states


def _read_combined(metadata):
    return json.loads((metadata / "radiomics_roi_ledger.json").read_text(encoding="utf-8"))


def test_combined_ledger_keeps_two_mr_series_measuring_the_same_roi(tmp_path):
    metadata = tmp_path / "metadata"
    mr = DenominatorLedger()
    mr.expect_course_roi("C1", "urinary_bladder")
    mr.record_roi(
        "C1", "P1", "urinary_bladder",
        reason_code="extracted", disposition="extracted",
        segmentation_source="AutoTS_total_mr",
        series_uid="1.2.826.0.1.3680043.8.498.1",
        nifti_path="MR_1/image.nii.gz",
        source_content_sha256="a" * 64,
        mask_path_source="MR_1/masks/urinary_bladder.nii.gz",
        mask_identity="b" * 64,
    )
    mr.record_roi(
        "C1", "P1", "urinary_bladder",
        reason_code="failed_radiomics_extraction", disposition="excluded",
        segmentation_source="AutoTS_total_mr",
        series_uid="1.2.826.0.1.3680043.8.498.2",
        nifti_path="MR_2/image.nii.gz",
        source_content_sha256="c" * 64,
        mask_path_source="MR_2/masks/urinary_bladder.nii.gz",
        mask_identity="d" * 64,
    )
    mr.record_course("C1", "P1", **_course_states(technical_exclusion=True))
    write_modality_ledger(metadata, mr, "MR")

    per_modality = json.loads(
        (metadata / "radiomics_mr_roi_ledger.json").read_text(encoding="utf-8")
    )
    assert len(per_modality["course_roi"]) == 2

    combined = _read_combined(metadata)
    assert len(combined["course_roi"]) == 2
    assert {row["series_uid"] for row in combined["course_roi"]} == {
        "1.2.826.0.1.3680043.8.498.1",
        "1.2.826.0.1.3680043.8.498.2",
    }
    assert {row["reason_code"] for row in combined["course_roi"]} == {
        "extracted", "failed_radiomics_extraction",
    }
    summary = json.loads(
        (metadata / "radiomics_denominators.json").read_text(encoding="utf-8")
    )
    assert summary["COURSE_ROI"]["MR:urinary_bladder"]["extracted"] == 1
    assert summary["COURSE_ROI"]["MR:urinary_bladder"]["excluded_technical"] == 1


def test_combined_ledger_distinguishes_a_lone_differing_identity_field(tmp_path):
    """One differing source-identity field is enough to keep two outcomes."""
    for field, other in (
        ("nifti_path", "MR_2/image.nii.gz"),
        ("source_content_sha256", "e" * 64),
        ("mask_path_source", "MR_2/masks/liver.nii.gz"),
        ("mask_identity", "f" * 64),
    ):
        metadata = tmp_path / field / "metadata"
        base = {
            "segmentation_source": "AutoTS_total_mr",
            "series_uid": "1.2.826.0.1.3680043.8.498.9",
            "nifti_path": "MR_1/image.nii.gz",
            "source_content_sha256": "1" * 64,
            "mask_path_source": "MR_1/masks/liver.nii.gz",
            "mask_identity": "2" * 64,
        }
        mr = DenominatorLedger()
        mr.expect_course_roi("C1", "liver")
        mr.record_roi(
            "C1", "P1", "liver",
            reason_code="extracted", disposition="extracted", **base,
        )
        mr.record_roi(
            "C1", "P1", "liver",
            reason_code="extracted", disposition="extracted",
            **{**base, field: other},
        )
        mr.record_course("C1", "P1", **_course_states())
        write_modality_ledger(metadata, mr, "MR")

        combined = _read_combined(metadata)
        assert len(combined["course_roi"]) == 2, field
        assert {row[field] for row in combined["course_roi"]} == {base[field], other}


def test_combined_ledger_collapses_a_row_repeated_verbatim(tmp_path):
    metadata = tmp_path / "metadata"
    identity = {
        "segmentation_source": "AutoTS_total_mr",
        "series_uid": "1.2.826.0.1.3680043.8.498.7",
        "nifti_path": "MR_1/image.nii.gz",
        "source_content_sha256": "3" * 64,
        "mask_path_source": "MR_1/masks/spleen.nii.gz",
        "mask_identity": "4" * 64,
    }
    mr = DenominatorLedger()
    mr.expect_course_roi("C1", "spleen")
    for _ in range(2):
        mr.record_roi(
            "C1", "P1", "spleen",
            reason_code="extracted", disposition="extracted", **identity,
        )
    mr.record_course("C1", "P1", **_course_states())
    write_modality_ledger(metadata, mr, "MR")

    assert len(
        json.loads(
            (metadata / "radiomics_mr_roi_ledger.json").read_text(encoding="utf-8")
        )["course_roi"]
    ) == 2
    combined = _read_combined(metadata)
    assert len(combined["course_roi"]) == 1
    summary = json.loads(
        (metadata / "radiomics_denominators.json").read_text(encoding="utf-8")
    )
    assert summary["COURSE_ROI"]["MR:spleen"]["extracted"] == 1


def test_combined_ledger_keeps_ct_rows_without_source_identity_unchanged(tmp_path):
    """A CT ledger carries none of these fields and must merge exactly as before."""
    metadata = tmp_path / "metadata"
    ct = DenominatorLedger()
    ct.expect_course_roi("C1", "PTV")
    ct.expect_course_roi("C1", "urinary_bladder")
    ct.record_roi("C1", "P1", "PTV", reason_code="extracted", disposition="extracted")
    ct.record_roi(
        "C1", "P1", "urinary_bladder",
        reason_code="extracted", disposition="extracted",
        segmentation_source="AutoTS_total",
    )
    # Two CT rows that differ in nothing the merge can see still collapse.
    ct.record_roi("C1", "P1", "PTV", reason_code="extracted", disposition="extracted")
    ct.record_course("C1", "P1", **_course_states())
    write_modality_ledger(metadata, ct, "CT")

    combined = _read_combined(metadata)
    assert len(combined["course_roi"]) == 2
    assert {row["roi_name"] for row in combined["course_roi"]} == {"PTV", "urinary_bladder"}
    assert {row["modality"] for row in combined["course_roi"]} == {"CT"}
    summary = json.loads(
        (metadata / "radiomics_denominators.json").read_text(encoding="utf-8")
    )
    assert summary["COURSE_ROI"]["CT:PTV"]["extracted"] == 1
    assert summary["COURSE_ROI"]["CT:urinary_bladder"]["extracted"] == 1
    assert summary["COURSE"]["screened"] == 1
    assert summary["PATIENT"]["extracted"] == 1


def test_combined_ledger_merges_ct_and_mr_source_identities_side_by_side(tmp_path):
    metadata = tmp_path / "metadata"
    ct = DenominatorLedger()
    ct.expect_course_roi("C1", "urinary_bladder")
    ct.record_roi(
        "C1", "P1", "urinary_bladder",
        reason_code="extracted", disposition="extracted",
        segmentation_source="AutoTS_total",
    )
    ct.record_course("C1", "P1", **_course_states())
    write_modality_ledger(metadata, ct, "CT")

    mr = DenominatorLedger()
    mr.expect_course_roi("C1", "urinary_bladder")
    for suffix in ("1", "2"):
        mr.record_roi(
            "C1", "P1", "urinary_bladder",
            reason_code="extracted", disposition="extracted",
            segmentation_source="AutoTS_total_mr",
            series_uid=f"1.2.826.0.1.3680043.8.498.{suffix}",
            nifti_path=f"MR_{suffix}/image.nii.gz",
            source_content_sha256=suffix * 64,
            mask_path_source=f"MR_{suffix}/masks/urinary_bladder.nii.gz",
            mask_identity=suffix * 64,
        )
    mr.record_course("C1", "P1", **_course_states())
    write_modality_ledger(metadata, mr, "MR")

    combined = _read_combined(metadata)
    assert len(combined["course_roi"]) == 3
    by_modality = {}
    for row in combined["course_roi"]:
        by_modality.setdefault(row["modality"], []).append(row)
    assert len(by_modality["CT"]) == 1
    assert "series_uid" not in by_modality["CT"][0]
    assert len(by_modality["MR"]) == 2
    assert len(combined["course"]) == 1
    assert combined["course"][0]["modalities"] == ["CT", "MR"]
    summary = json.loads(
        (metadata / "radiomics_denominators.json").read_text(encoding="utf-8")
    )
    assert summary["COURSE_ROI"]["CT:urinary_bladder"]["extracted"] == 1
    assert summary["COURSE_ROI"]["MR:urinary_bladder"]["extracted"] == 2
