"""Manager regression for typed MR errors: preserve identity and account every ROI.

Reuses only the generated course fixture from the inspected failure-accounting
module. Extraction is substituted. No helper subprocess or clinical input.
"""
import pytest

import rtpipeline.radiomics_conda as rc
from rtpipeline.radiomics_outcomes import RadiomicsCourseExtractionError
from test_mr_helper_failure_accounting import (
    _Config,
    _contract,
    _ledger,
    _make_course,
    _plant_stale_publication,
    _write_params,
    isolated_runtime,
)


def test_typed_failure_preserves_exception_and_accounts_all_rois(
    tmp_path, isolated_runtime, monkeypatch
):
    fixture = _make_course(tmp_path)
    course_dir = fixture["course_dir"]
    stale = _plant_stale_publication(course_dir)
    config = _Config(
        params_file=_write_params(tmp_path),
        contract=_contract(required=["liver"], inventory_only=["spleen"]),
    )
    original = RadiomicsCourseExtractionError("required MR extraction failed: liver")

    def fail_extraction(*args, **kwargs):
        # Model a typed producer failure. The course publication boundary must
        # account its tasks regardless of whether a lower-level cleanup ran.
        raise original

    monkeypatch.setattr(rc, "process_radiomics_batch", fail_extraction)
    with pytest.raises(RadiomicsCourseExtractionError) as caught:
        rc.radiomics_for_course_mr(course_dir, config)
    assert caught.value is original
    assert all(not path.exists() for path in stale.values())

    ledger = _ledger(course_dir)
    rows = {row["roi_name"]: row for row in ledger["course_roi"]}
    assert set(rows) == {"liver", "spleen"}
    assert rows["liver"]["reason_code"] == "failed_radiomics_extraction"
    assert rows["liver"]["disposition"] == "excluded"
    assert rows["liver"]["mask_identity"] == rc.file_sha256(fixture["mask_path"])
    assert rows["spleen"]["reason_code"] == "not_computed_valid_empty_scope"
    assert rows["spleen"]["mask_identity"] == rc.file_sha256(fixture["empty_mask_path"])
    assert len(ledger["course"]) == 1
    assert ledger["course"][0]["extracted"] is False
    assert ledger["course"][0]["technical_exclusion"] is True
