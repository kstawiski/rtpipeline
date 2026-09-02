from __future__ import annotations

import json
import os
import runpy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from course_contract_test_utils import (
    write_minimal_course_contract,
    write_synthetic_plan_and_dose,
)
from rtpipeline.radiomics_cohort import provenance_from_frame
from rtpipeline.radiomics_ct_contract import (
    _feature_schema_metadata,
    classify_ct_roi,
    disposition_rows_for_arms,
    write_completion_sentinel,
    write_ct_publication_atomic,
)


ROOT = Path(__file__).resolve().parents[1]
AGGREGATE_RESULTS = ROOT / "workflow" / "scripts" / "aggregate_results.py"


def _write_publication(course_dir: Path, patient: str) -> None:
    records = disposition_rows_for_arms(
        {
            "modality": "CT",
            "segmentation_source": "Manual",
            "roi_name": "PTV",
            "roi_original_name": "PTV",
            "patient_id": patient,
            "course_id": "C1",
            "series_uid": f"series-{patient}",
            "mask_identity": f"mask-{patient}",
            "stable_roi_identifier": "roi-PTV",
        },
        decision=classify_ct_roi("Manual", "PTV"),
        disposition="success",
        detail="test success",
        failure_kind="none",
        run_identifier="test-run",
        code_revision="test-revision",
        native_voxel_count=120,
        required=True,
        configured_parameter_hashes={
            "primary_resegmented": "test-primary",
            "sensitivity_raw": "test-sensitivity",
        },
        effective_hashes={
            "primary_resegmented": "effective-primary",
            "sensitivity_raw": "effective-sensitivity",
        },
    )
    for record in records:
        record["original_firstorder_Mean"] = 1.0
        record.update(_feature_schema_metadata({"original_firstorder_Mean"}))
    write_ct_publication_atomic(
        pd.DataFrame(records), course_dir / "radiomics_ct.xlsx"
    )


def _write_course(output_dir: Path, patient: str, *, radiomics_ok: bool) -> None:
    course_dir = output_dir / patient / "C1"
    course_dir.mkdir(parents=True)
    plan, dose = write_synthetic_plan_and_dose(course_dir)
    write_minimal_course_contract(
        course_dir, selected_plans=[plan], selected_doses=[dose]
    )
    pd.DataFrame([{"ROI_Name": "PTV", "DmeanGy": 42.0}]).to_excel(
        course_dir / "dvh_metrics.xlsx", index=False
    )
    (course_dir / ".dvh_done").write_text("ok\n", encoding="utf-8")
    (course_dir / ".qc_done").write_text("ok\n", encoding="utf-8")
    (course_dir / ".custom_models_done").write_text(
        "disabled\n", encoding="utf-8"
    )
    if radiomics_ok:
        _write_publication(course_dir, patient)
        write_completion_sentinel(course_dir, course_dir / ".radiomics_done")
    else:
        (course_dir / ".radiomics_done").write_text(
            "failed\n", encoding="utf-8"
        )


def _workflow(tmp_path: Path, *, record_failure: bool) -> SimpleNamespace:
    output_dir = tmp_path / "output"
    results_dir = output_dir / "_RESULTS"
    manifest = output_dir / "_COURSES" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    quarantine = {
        "patient": "P3",
        "course": "C0",
        "status": "technical_quarantine",
        "disposition_type": "technical_quarantine",
        "clinical_exclusion": False,
        "reason": "authoritative structure set has no target volumes",
    }
    payload = {
        "schema": "rtpipeline-organized-course-manifest-v2",
        "intended_course_count": 3,
        "attempted_course_count": 3,
        "validated_course_count": 2,
        "technical_quarantine_count": 1,
        "technical_quarantines": [quarantine],
        "courses": [
            {"patient": patient, "course": "C1"}
            for patient in ("P1", "P2")
        ],
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    _write_course(output_dir, "P1", radiomics_ok=True)
    _write_course(output_dir, "P2", radiomics_ok=False)

    if record_failure:
        record_dir = output_dir / "_campaign_ledger" / "records"
        record_dir.mkdir(parents=True)
        (record_dir / "P2__C1__radiomics.json").write_text(
            json.dumps(
                {
                    "patient": "P2",
                    "course": "C1",
                    "stage": "radiomics",
                    "status": "failed",
                    "returncode": 1,
                    "detail": "planning CT is unavailable after segmentation failure",
                }
            ),
            encoding="utf-8",
        )

    outputs = {
        "dvh": results_dir / "dvh_metrics.xlsx",
        "dvh_parquet": results_dir / "dvh_metrics.parquet",
        "fractions": results_dir / "fractions.xlsx",
        "metadata": results_dir / "case_metadata.xlsx",
        "qc": results_dir / "qc_reports.xlsx",
        "radiomics": results_dir / "radiomics_ct.xlsx",
        "radiomics_mr": results_dir / "radiomics_mr.xlsx",
    }
    log_path = tmp_path / "logs" / "aggregate.log"
    log_path.parent.mkdir(parents=True)
    return SimpleNamespace(
        input=SimpleNamespace(manifest=str(manifest)),
        output=SimpleNamespace(
            **{name: str(path) for name, path in outputs.items()}
        ),
        log=[str(log_path)],
        params=SimpleNamespace(
            output_dir=str(output_dir),
            results_dir=str(results_dir),
            radiomics_enabled=True,
            campaign_mode=True,
            campaign_min_completion_fraction=0.1,
            campaign_require_all_courses=False,
            worker_budget=2,
            auto_worker_budget=2,
            aggregation_threads=1,
        ),
    )


def test_recorded_exclusions_permit_aggregation_with_denominator_provenance(
    tmp_path: Path,
) -> None:
    workflow = _workflow(tmp_path, record_failure=True)

    runpy.run_path(str(AGGREGATE_RESULTS), init_globals={"snakemake": workflow})

    canonical = Path(workflow.params.output_dir) / "Data" / "radiomics_all.parquet"
    frame = pd.read_parquet(canonical)
    provenance = provenance_from_frame(frame)
    assert len(provenance.pop("denominator_source_sha256")) == 64
    exclusions = provenance.pop("exclusions")
    assert [len(entry.pop("source_record_sha256")) for entry in exclusions] == [
        64,
        64,
    ]
    assert provenance == {
        "schema": "rtpipeline-radiomics-cohort-v1",
        "intended_course_count": 3,
        "validated_course_count": 2,
        "extracted_course_count": 1,
        "excluded_course_count": 2,
        "technical_quarantine_count": 1,
        "downstream_exclusion_count": 1,
    }
    assert exclusions == [
        {
            "course_id": "C1",
            "disposition_type": "downstream_technical_exclusion",
            "patient_id": "P2",
            "reason": (
                "radiomics: planning CT is unavailable after segmentation failure"
            ),
            "source": "campaign_ledger",
            "stages": ["radiomics"],
        },
        {
            "course_id": "C0",
            "disposition_type": "technical_quarantine",
            "patient_id": "P3",
            "reason": "authoritative structure set has no target volumes",
            "source": "organize_ledger",
        },
    ]
    assert set(frame["patient_id"]) == {"P1"}


def test_unrecorded_course_failure_still_blocks_aggregation(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path, record_failure=False)

    with pytest.raises(RuntimeError, match="unrecorded course failure"):
        runpy.run_path(str(AGGREGATE_RESULTS), init_globals={"snakemake": workflow})

    canonical = Path(workflow.params.output_dir) / "Data" / "radiomics_all.parquet"
    assert not canonical.exists()


def test_stale_failed_record_does_not_authorize_aggregation(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path, record_failure=True)
    record = (
        Path(workflow.params.output_dir)
        / "_campaign_ledger"
        / "records"
        / "P2__C1__radiomics.json"
    )
    manifest = Path(workflow.input.manifest)
    newer = record.stat().st_mtime_ns + 1_000_000_000
    os.utime(manifest, ns=(newer, newer))

    with pytest.raises(RuntimeError, match="unrecorded course failure"):
        runpy.run_path(str(AGGREGATE_RESULTS), init_globals={"snakemake": workflow})


def test_unrecorded_rerun_does_not_destroy_valid_prior_aggregate(
    tmp_path: Path,
) -> None:
    workflow = _workflow(tmp_path, record_failure=True)
    runpy.run_path(str(AGGREGATE_RESULTS), init_globals={"snakemake": workflow})
    canonical = Path(workflow.params.output_dir) / "Data" / "radiomics_all.parquet"
    workbook = canonical.with_suffix(".xlsx")
    prior_parquet = canonical.read_bytes()
    prior_workbook = workbook.read_bytes()
    record = (
        Path(workflow.params.output_dir)
        / "_campaign_ledger"
        / "records"
        / "P2__C1__radiomics.json"
    )
    record.unlink()

    with pytest.raises(RuntimeError, match="unrecorded course failure"):
        runpy.run_path(str(AGGREGATE_RESULTS), init_globals={"snakemake": workflow})

    assert canonical.read_bytes() == prior_parquet
    assert workbook.read_bytes() == prior_workbook
