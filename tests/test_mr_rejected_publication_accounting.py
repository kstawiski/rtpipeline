"""What a *rejected* MR publication owes the cohort denominator.

``radiomics_for_course_mr`` withdraws the workbook and its Parquet sidecar on
several rejection paths, but ``invalidate_radiomics_outputs`` never touches a
ledger. ``write_modality_ledger`` rebuilds the combined
``metadata/radiomics_roi_ledger.json`` from the per-modality ledger files, and
``_write_radiomics_denominator_aggregate`` copies that combined file into the
cohort denominator. So a rejection that leaves an earlier successful ledger on
disk keeps counting MR ROIs as ``extracted`` for a course whose measurements
have just been withdrawn.

Two different obligations are checked, because the module already distinguishes
them:

* the live sources changed under the run -- no evidence about the bytes that
  were read may be published, so a superseded ledger is *withdrawn*;
* the sources were stable and the publication itself was rejected -- every
  attempted ROI is *accounted* with a technical failure reason, exactly as the
  typed-exception path already does.

Every course here is generated. The only substituted symbol is
``process_radiomics_batch``, used to inject a publication outcome; nothing in
this file claims a real PyRadiomics extraction. No conda, no network, no clinical
input, and no write outside the per-test temporary directory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd
import pytest

import rtpipeline.radiomics_conda as rc
from rtpipeline.radiomics_outcomes import RadiomicsCourseExtractionError

# Source-inspected generated-course utilities. That module writes only into the
# temporary directory it is handed and reads no clinical data.
from test_mr_helper_failure_accounting import (  # noqa: F401  (isolated_runtime is a fixture)
    _Config,
    _contract,
    _ledger,
    _make_course,
    _write_params,
    isolated_runtime,
)

MR_LEDGER_NAMES = ("roi_ledger", "denominators", "patient_ledger")


# --------------------------------------------------------------------------
# publication injection
# --------------------------------------------------------------------------
def _rows_from_tasks(tasks: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The rows a healthy batch would publish for exactly these tasks."""
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        row = dict(task["metadata"])
        failure = task.get("precomputed_failure")
        if failure:
            row["extraction_status"] = failure["status"]
            row["extraction_status_detail"] = failure["reason"]
            row["reason_code"] = failure["reason_code"]
        else:
            row["extraction_status"] = "success"
            row["original_firstorder_Mean"] = 123.5
        rows.append(row)
    return rows


def _publish(output_path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    frame = pd.DataFrame(list(rows))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path.with_suffix(".parquet"), index=False)
    frame.to_excel(output_path, index=False)
    return output_path


def _install_batch(monkeypatch, publisher) -> Dict[str, Any]:
    """Substitute the batch call with ``publisher(tasks, output_path)``."""
    seen: Dict[str, Any] = {"calls": 0, "tasks": None}

    def fake_batch(tasks, output_path, *args: Any, **kwargs: Any) -> Optional[Path]:
        seen["calls"] += 1
        seen["tasks"] = [dict(task) for task in tasks]
        return publisher(tasks, Path(output_path))

    monkeypatch.setattr(rc, "process_radiomics_batch", fake_batch)
    return seen


def _healthy_publisher(tasks, output_path: Path) -> Path:
    return _publish(output_path, _rows_from_tasks(tasks))


# --------------------------------------------------------------------------
# a real earlier successful run, planted through the production writer
# --------------------------------------------------------------------------
def _plant_previous_success(course_dir: Path, config: Any, monkeypatch) -> Dict[str, Any]:
    """Leave the artifacts an earlier *successful* MR run leaves behind.

    The ledger is written by the production writer rather than hand-built, so the
    planted evidence is exactly the shape a consumer would meet on disk.
    """
    with monkeypatch.context() as patched:
        _install_batch(patched, _healthy_publisher)
        result = rc.radiomics_for_course_mr(course_dir, config)
    assert result is not None, "the planted previous run did not publish"
    ledger = _ledger(course_dir)
    assert any(
        row["roi_name"] == "liver" and row["disposition"] == "extracted"
        for row in ledger["course_roi"]
    ), "the planted previous run is not a successful ledger"
    checkpoint = course_dir / "MR" / "radiomics_mr_checkpoint.parquet"
    checkpoint.write_bytes(b"STALE-CHECKPOINT")
    return {
        "workbook": course_dir / "MR" / "radiomics_mr.xlsx",
        "parquet": course_dir / "MR" / "radiomics_mr.parquet",
        "checkpoint": checkpoint,
        "ledger": ledger,
    }


def _combined(course_dir: Path) -> Dict[str, Any]:
    path = course_dir / "metadata" / "radiomics_roi_ledger.json"
    if not path.exists():
        return {"course": [], "course_roi": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _mr_ledger_paths(course_dir: Path) -> List[Path]:
    return [
        course_dir / "metadata" / f"radiomics_mr_{name}.json"
        for name in MR_LEDGER_NAMES
    ]


def _assert_no_stale_mr_success(course_dir: Path) -> None:
    """Nothing on disk may still describe a withdrawn MR measurement."""
    for path in _mr_ledger_paths(course_dir):
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("course_roi", []) if isinstance(payload, dict) else []
        assert not [row for row in rows if row.get("disposition") == "extracted"], (
            f"{path.name} still reports an extracted MR ROI"
        )
        for row in payload.get("course", []) if isinstance(payload, dict) else []:
            assert not row.get("extracted"), f"{path.name} still reports an extracted course"
    combined = _combined(course_dir)
    stale = [
        row
        for row in combined["course_roi"]
        if str(row.get("modality", "")).upper() == "MR"
        and row.get("disposition") == "extracted"
    ]
    assert not stale, (
        "the combined ledger the cohort denominator reads still counts withdrawn "
        f"MR measurements: {stale}"
    )
    for path in (
        course_dir / "MR" / "radiomics_mr.xlsx",
        course_dir / "MR" / "radiomics_mr.parquet",
    ):
        assert not path.exists(), f"{path.name} survived a rejected MR publication"


def _mr_roi_rows(course_dir: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row["roi_name"]): row for row in _ledger(course_dir)["course_roi"]}


# --------------------------------------------------------------------------
# control: a stable, healthy course still publishes and keeps its ledger
# --------------------------------------------------------------------------
def test_stable_course_publishes_and_keeps_its_ledger(tmp_path, isolated_runtime, monkeypatch):
    fixture = _make_course(tmp_path)
    course_dir = fixture["course_dir"]
    config = _Config(
        params_file=_write_params(tmp_path),
        contract=_contract(required=["liver"], inventory_only=["spleen"]),
    )
    _install_batch(monkeypatch, _healthy_publisher)

    assert rc.radiomics_for_course_mr(course_dir, config) is not None

    rows = _mr_roi_rows(course_dir)
    assert rows["liver"]["disposition"] == "extracted"
    assert rows["liver"]["mask_identity"] == rc.file_sha256(fixture["mask_path"])
    assert rows["spleen"]["reason_code"] == "not_computed_valid_empty_scope"
    course_row = _ledger(course_dir)["course"][0]
    assert course_row["extracted"] is True
    combined = _combined(course_dir)
    assert any(
        row["roi_name"] == "liver" and row["disposition"] == "extracted"
        for row in combined["course_roi"]
    )


# --------------------------------------------------------------------------
# the source changed while the course was measured
# --------------------------------------------------------------------------
def test_source_change_during_extraction_withdraws_the_previous_success(
    tmp_path, isolated_runtime, monkeypatch
):
    """A mask rewritten mid-batch invalidates the *whole* MR course record.

    The run already refuses to publish what it measured. What it must not do is
    leave the previous run's successful ledger -- bound to bytes that are gone --
    behind for the cohort denominator to count.
    """
    fixture = _make_course(tmp_path)
    course_dir = fixture["course_dir"]
    config = _Config(
        params_file=_write_params(tmp_path),
        contract=_contract(inventory_only=["liver", "spleen"]),
    )
    planted = _plant_previous_success(course_dir, config, monkeypatch)

    def publisher_that_mutates_its_source(tasks, output_path: Path) -> Path:
        published = _publish(output_path, _rows_from_tasks(tasks))
        # A producer rewriting the segmentation while this course measures it.
        mask_path = Path(fixture["mask_path"])
        mask_path.write_bytes(mask_path.read_bytes() + b"\0")
        return published

    _install_batch(monkeypatch, publisher_that_mutates_its_source)

    with pytest.raises(RadiomicsCourseExtractionError) as caught:
        rc.radiomics_for_course_mr(course_dir, config)
    message = str(caught.value)
    print("observed course error:", message)
    assert "changed while the course was being measured" in message
    assert "total_mr--liver.nii.gz changed during extraction" in message

    assert not planted["checkpoint"].exists()
    _assert_no_stale_mr_success(course_dir)


def test_source_change_during_screening_withdraws_the_previous_success(
    tmp_path, isolated_runtime, monkeypatch
):
    """Same obligation on the disposition-only path, which publishes no workbook."""
    fixture = _make_course(tmp_path)
    course_dir = fixture["course_dir"]
    config = _Config(
        params_file=_write_params(tmp_path),
        contract=_contract(inventory_only=["liver", "spleen"]),
    )
    planted = _plant_previous_success(course_dir, config, monkeypatch)

    # Now make the course measure nothing, so screening reaches the no-task path.
    Path(fixture["mask_path"]).unlink()
    real_roi_masks = rc._mr_roi_masks
    arrived: List[Path] = []

    def late_arrival(seg_dir: Path):
        masks = real_roi_masks(seg_dir)
        if not arrived:
            late = Path(seg_dir) / "total_mr--kidney.nii.gz"
            late.write_bytes(Path(fixture["empty_mask_path"]).read_bytes())
            arrived.append(late)
        return masks

    monkeypatch.setattr(rc, "_mr_roi_masks", late_arrival)
    _install_batch(monkeypatch, _healthy_publisher)

    with pytest.raises(RadiomicsCourseExtractionError) as caught:
        rc.radiomics_for_course_mr(course_dir, config)
    message = str(caught.value)
    print("observed course error:", message)
    assert arrived, "the regression never simulated a live source change"
    assert "changed while the course was being screened" in message

    assert not planted["checkpoint"].exists()
    # The tested contract for a drifted source is that no MR ledger is published.
    for path in _mr_ledger_paths(course_dir):
        assert not path.exists(), f"{path.name} survived a drifted screening"
    _assert_no_stale_mr_success(course_dir)


# --------------------------------------------------------------------------
# the sources were stable and the publication itself was rejected
# --------------------------------------------------------------------------
def test_unreadable_publication_accounts_every_attempted_roi(
    tmp_path, isolated_runtime, monkeypatch
):
    """A Parquet that cannot be read back is a technical failure, not silence."""
    fixture = _make_course(tmp_path)
    course_dir = fixture["course_dir"]
    config = _Config(
        params_file=_write_params(tmp_path),
        contract=_contract(inventory_only=["liver", "spleen"]),
    )
    planted = _plant_previous_success(course_dir, config, monkeypatch)

    def unreadable_publisher(tasks, output_path: Path) -> Path:
        _publish(output_path, _rows_from_tasks(tasks))
        output_path.with_suffix(".parquet").write_bytes(b"not a parquet file")
        return output_path

    _install_batch(monkeypatch, unreadable_publisher)

    with pytest.raises(RadiomicsCourseExtractionError) as caught:
        rc.radiomics_for_course_mr(course_dir, config)
    message = str(caught.value)
    print("observed course error:", message)
    assert "cannot be read back" in message

    assert not planted["checkpoint"].exists()
    _assert_no_stale_mr_success(course_dir)

    rows = _mr_roi_rows(course_dir)
    assert set(rows) == {"liver", "spleen"}
    # The ROI that was being measured is accounted as a technical failure, and it
    # still names the exact bytes the run attempted.
    assert rows["liver"]["disposition"] == "excluded"
    assert rows["liver"]["reason_code"] == "failed_radiomics_extraction"
    assert rows["liver"]["mask_identity"] == rc.file_sha256(fixture["mask_path"])
    assert rows["liver"]["source_content_sha256"] == rc.file_sha256(fixture["nifti_path"])
    # An independently valid empty-mask disposition keeps its own reason.
    assert rows["spleen"]["reason_code"] == "not_computed_valid_empty_scope"
    course_row = _ledger(course_dir)["course"][0]
    assert course_row["extracted"] is False
    assert course_row["technical_exclusion"] is True
    assert course_row["reason_code"] == "failed_radiomics_extraction"


def test_publication_bound_to_a_foreign_source_accounts_every_attempted_roi(
    tmp_path, isolated_runtime, monkeypatch
):
    """A row whose identity is not the current source rejects the whole course."""
    fixture = _make_course(tmp_path)
    course_dir = fixture["course_dir"]
    config = _Config(
        params_file=_write_params(tmp_path),
        contract=_contract(inventory_only=["liver", "spleen"]),
    )
    planted = _plant_previous_success(course_dir, config, monkeypatch)

    def foreign_identity_publisher(tasks, output_path: Path) -> Path:
        rows = _rows_from_tasks(tasks)
        for row in rows:
            if row.get("roi_original_name") == "liver":
                # Still the current ROI instance, but bound to different image
                # bytes than the source this run actually read.
                row["source_content_sha256"] = "0" * 64
        return _publish(output_path, rows)

    _install_batch(monkeypatch, foreign_identity_publisher)

    with pytest.raises(RadiomicsCourseExtractionError) as caught:
        rc.radiomics_for_course_mr(course_dir, config)
    message = str(caught.value)
    print("observed course error:", message)
    assert "not bound to its current sources" in message
    assert "published source_content_sha256 for 'liver'" in message

    assert not planted["checkpoint"].exists()
    _assert_no_stale_mr_success(course_dir)

    rows = _mr_roi_rows(course_dir)
    assert rows["liver"]["reason_code"] == "failed_radiomics_extraction"
    assert rows["liver"]["mask_identity"] == rc.file_sha256(fixture["mask_path"])
    assert rows["spleen"]["reason_code"] == "not_computed_valid_empty_scope"
    assert _ledger(course_dir)["course"][0]["technical_exclusion"] is True


def test_required_roi_still_fails_closed_when_the_publication_is_rejected(
    tmp_path, isolated_runtime, monkeypatch
):
    """Control: rejection accounting does not discharge a required ROI."""
    fixture = _make_course(tmp_path)
    course_dir = fixture["course_dir"]
    config = _Config(
        params_file=_write_params(tmp_path),
        contract=_contract(required=["liver"], inventory_only=["spleen"]),
    )

    def unreadable_publisher(tasks, output_path: Path) -> Path:
        _publish(output_path, _rows_from_tasks(tasks))
        output_path.with_suffix(".parquet").write_bytes(b"not a parquet file")
        return output_path

    _install_batch(monkeypatch, unreadable_publisher)

    with pytest.raises(RadiomicsCourseExtractionError):
        rc.radiomics_for_course_mr(course_dir, config)

    rows = _mr_roi_rows(course_dir)
    assert rows["liver"]["reason_code"] == "failed_radiomics_extraction"
    assert rows["liver"]["disposition"] == "excluded"


# --------------------------------------------------------------------------
# a parameter binding that cannot be resolved invalidates the same evidence
# --------------------------------------------------------------------------
def test_missing_configured_parameter_file_withdraws_the_previous_success(
    tmp_path, isolated_runtime, monkeypatch
):
    """The configured MR parameter file disappeared between two runs.

    Nothing may be measured under a binding this run cannot resolve, and the
    previous run's ledger and checkpoint are exactly as invalid as its workbook.
    """
    fixture = _make_course(tmp_path)
    course_dir = fixture["course_dir"]
    good_config = _Config(
        params_file=_write_params(tmp_path),
        contract=_contract(inventory_only=["liver", "spleen"]),
    )
    planted = _plant_previous_success(course_dir, good_config, monkeypatch)

    broken_config = _Config(
        params_file=tmp_path / "mr_params_removed.yaml",
        contract=_contract(inventory_only=["liver", "spleen"]),
    )
    _install_batch(monkeypatch, _healthy_publisher)

    with pytest.raises(RadiomicsCourseExtractionError) as caught:
        rc.radiomics_for_course_mr(course_dir, broken_config)
    message = str(caught.value)
    print("observed course error:", message)
    assert "parameter path is missing" in message

    assert not planted["checkpoint"].exists(), (
        "a checkpoint measured under an unresolvable parameter binding survived"
    )
    _assert_no_stale_mr_success(course_dir)
    course_row = _ledger(course_dir)["course"][0]
    assert course_row["extracted"] is False
    assert course_row["technical_exclusion"] is True
    assert course_row["reason_code"] == "failed_radiomics_extraction"


def test_unresolvable_parameter_hash_withdraws_the_previous_success(
    tmp_path, isolated_runtime, monkeypatch
):
    """Same failure class one step later: the parameter hash cannot be computed."""
    fixture = _make_course(tmp_path)
    course_dir = fixture["course_dir"]
    config = _Config(
        params_file=_write_params(tmp_path),
        contract=_contract(inventory_only=["liver", "spleen"]),
    )
    planted = _plant_previous_success(course_dir, config, monkeypatch)

    def unresolvable(*args: Any, **kwargs: Any):
        raise RuntimeError("configured MR parameter document is not parseable")

    monkeypatch.setattr(rc, "configured_parameter_hash", unresolvable)
    _install_batch(monkeypatch, _healthy_publisher)

    with pytest.raises(RadiomicsCourseExtractionError) as caught:
        rc.radiomics_for_course_mr(course_dir, config)
    print("observed course error:", str(caught.value))
    assert "configuration binding failed" in str(caught.value)

    assert not planted["checkpoint"].exists()
    _assert_no_stale_mr_success(course_dir)
    assert _ledger(course_dir)["course"][0]["technical_exclusion"] is True


# --------------------------------------------------------------------------
# withdrawing MR must not disturb the CT accounting for the same course
# --------------------------------------------------------------------------
def _plant_ct_ledger(course_dir: Path) -> Dict[str, Any]:
    """A CT ledger for the same course, written by the production writer."""
    from rtpipeline.roi_requiredness import DenominatorLedger, write_modality_ledger

    ledger = DenominatorLedger()
    ledger.record_roi(
        course_dir.name,
        course_dir.parent.name,
        "bladder",
        reason_code="extracted",
        disposition="extracted",
    )
    ledger.record_course(
        course_dir.name,
        course_dir.parent.name,
        screened=True,
        in_scope=True,
        out_of_scope=False,
        adequate_coverage=True,
        insufficient_coverage=False,
        valid_derivation=False,
        technical_exclusion=False,
        indeterminate=False,
        extracted=True,
        reason_code="extracted",
    )
    write_modality_ledger(course_dir / "metadata", ledger, "CT")
    return json.loads(
        (course_dir / "metadata" / "radiomics_ct_roi_ledger.json").read_text(
            encoding="utf-8"
        )
    )


def test_withdrawing_mr_preserves_the_ct_accounting_of_the_same_course(
    tmp_path, isolated_runtime, monkeypatch
):
    """MR withdrawal removes MR evidence only; CT keeps its rows and its place.

    The combined ledger the cohort denominator reads is rebuilt from the modality
    ledgers, so dropping MR must leave the CT ledger byte-equivalent and the
    combined view still carrying the CT course and its ROI.
    """
    fixture = _make_course(tmp_path)
    course_dir = fixture["course_dir"]
    config = _Config(
        params_file=_write_params(tmp_path),
        contract=_contract(inventory_only=["liver", "spleen"]),
    )
    _plant_previous_success(course_dir, config, monkeypatch)
    ct_before = _plant_ct_ledger(course_dir)

    def publisher_that_mutates_its_source(tasks, output_path: Path) -> Path:
        published = _publish(output_path, _rows_from_tasks(tasks))
        mask_path = Path(fixture["mask_path"])
        mask_path.write_bytes(mask_path.read_bytes() + b"\0")
        return published

    _install_batch(monkeypatch, publisher_that_mutates_its_source)

    with pytest.raises(RadiomicsCourseExtractionError):
        rc.radiomics_for_course_mr(course_dir, config)

    _assert_no_stale_mr_success(course_dir)
    for path in _mr_ledger_paths(course_dir):
        assert not path.exists(), f"{path.name} survived a drifted extraction"

    ct_after = json.loads(
        (course_dir / "metadata" / "radiomics_ct_roi_ledger.json").read_text(
            encoding="utf-8"
        )
    )
    assert ct_after == ct_before, "MR withdrawal changed the CT ledger"

    combined = _combined(course_dir)
    assert [row["roi_name"] for row in combined["course_roi"]] == ["bladder"]
    assert [str(row.get("modality", "")) for row in combined["course_roi"]] == ["CT"]
    assert len(combined["course"]) == 1
    assert combined["course"][0]["modalities"] == ["CT"]
    assert combined["course"][0]["extracted"] is True


def _plant_ledger_pair_only(course_dir: Path) -> None:
    """Generate ledger state without images or an extraction call."""
    from rtpipeline.roi_requiredness import DenominatorLedger, write_modality_ledger

    payload = _plant_ct_ledger(course_dir)
    write_modality_ledger(
        course_dir / "metadata",
        DenominatorLedger(
            course_rows=payload["course"], roi_rows=payload["course_roi"]
        ),
        "MR",
    )
    assert {row["modality"] for row in _combined(course_dir)["course_roi"]} == {"CT", "MR"}


@pytest.mark.parametrize("keep_ct", [False, True])
def test_withdrawal_reconciles_orphaned_combined_mr_rows(tmp_path, keep_ct):
    """A prior partial cleanup can leave MR only in the combined view."""
    course_dir = tmp_path / "P" / "C"
    _plant_ledger_pair_only(course_dir)
    metadata = course_dir / "metadata"
    ct_bytes = (metadata / "radiomics_ct_roi_ledger.json").read_bytes()
    for path in _mr_ledger_paths(course_dir):
        path.unlink()
    if not keep_ct:
        for suffix in MR_LEDGER_NAMES:
            (metadata / f"radiomics_ct_{suffix}.json").unlink()

    rc._withdraw_mr_ledger(course_dir, context="reconciling partial prior withdrawal")

    rows = _combined(course_dir)["course_roi"]
    assert not any(row.get("modality") == "MR" for row in rows)
    if keep_ct:
        assert [row["modality"] for row in rows] == ["CT"]
        assert (metadata / "radiomics_ct_roi_ledger.json").read_bytes() == ct_bytes
    else:
        for suffix in MR_LEDGER_NAMES:
            assert not (metadata / f"radiomics_{suffix}.json").exists()


@pytest.mark.parametrize("ct_contents", ["{unreadable", "[]"])
def test_failed_ct_rebuild_cannot_leave_superseded_combined_mr(tmp_path, ct_contents):
    """Reject a bad survivor without retaining the old successful combined view."""
    course_dir = tmp_path / "P" / "C"
    _plant_ledger_pair_only(course_dir)
    metadata = course_dir / "metadata"
    ct_path = metadata / "radiomics_ct_roi_ledger.json"
    ct_path.write_text(ct_contents, encoding="utf-8")

    with pytest.raises(RadiomicsCourseExtractionError):
        rc._withdraw_mr_ledger(course_dir, context="withdrawing rejected MR")

    assert ct_path.read_text(encoding="utf-8") == ct_contents
    for suffix in MR_LEDGER_NAMES:
        assert not (metadata / f"radiomics_{suffix}.json").exists()
    assert all(not path.exists() for path in _mr_ledger_paths(course_dir))
