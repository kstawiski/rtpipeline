from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pydicom
import pytest
from pydicom.uid import generate_uid

import rtpipeline.organize as organize
from course_contract_test_utils import (
    write_minimal_course_contract,
    write_synthetic_plan_and_dose,
)
from rtpipeline.organize_ledger import (
    STATUS_TECHNICAL_QUARANTINE,
    STATUS_VALIDATED,
    read_organize_ledger,
    write_organize_ledger,
)


ROOT = Path(__file__).resolve().parents[1]
ORGANIZE_COURSES = ROOT / "workflow" / "scripts" / "organize_courses.py"
AGGREGATE_RESULTS = ROOT / "workflow" / "scripts" / "aggregate_results.py"
CAMPAIGN_LEDGER = ROOT / "workflow" / "scripts" / "campaign_ledger.py"


def _ledger_entry(course: Path, status: str = STATUS_VALIDATED, reason=None) -> dict:
    return {
        "patient": course.parent.name,
        "course": course.name,
        "course_key": course.name,
        "path": str(course),
        "status": status,
        "reason": reason,
        "quarantine_path": None,
    }


def _organize_workflow(tmp_path: Path, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output=SimpleNamespace(
            manifest=str(output_dir / "_COURSES" / "manifest.json")
        ),
        log=[str(tmp_path / "logs" / "organize.log")],
        params=SimpleNamespace(
            output_dir=str(output_dir),
            root_dir=str(ROOT),
            configfile=str(tmp_path / "config.yaml"),
            radiomics_env="rtpipeline-radiomics",
            python=str(Path(sys.executable)),
            python_bin=str(Path(sys.executable).parent),
            dicom_root=str(tmp_path / "input"),
            logs_dir=str(tmp_path / "logs"),
            custom_structures="",
            prioritize_short_courses=False,
        ),
        threads=2,
    )


def test_atomic_generated_plan_write_breaks_existing_hardlink(tmp_path: Path):
    course = tmp_path / "P1" / "C1"
    selected_plan, _ = write_synthetic_plan_and_dose(course)
    original_uid = str(
        pydicom.dcmread(selected_plan, stop_before_pixels=True).SOPInstanceUID
    )
    root_plan = course / "RP.dcm"
    os.link(selected_plan, root_plan)
    assert os.path.samefile(selected_plan, root_plan)

    synthesized = pydicom.dcmread(root_plan)
    synthesized.SOPInstanceUID = generate_uid()
    synthesized.file_meta.MediaStorageSOPInstanceUID = synthesized.SOPInstanceUID
    organize._save_dataset_atomic(synthesized, root_plan)

    assert not os.path.samefile(selected_plan, root_plan)
    assert (
        str(pydicom.dcmread(selected_plan, stop_before_pixels=True).SOPInstanceUID)
        == original_uid
    )
    assert (
        str(pydicom.dcmread(root_plan, stop_before_pixels=True).SOPInstanceUID)
        == str(synthesized.SOPInstanceUID)
    )


def test_invalid_candidate_contract_is_never_published(tmp_path: Path):
    course = tmp_path / "P1" / "C1"
    metadata_path = write_minimal_course_contract(course)
    case_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    case_metadata["course_contract"]["version"] = 0

    with pytest.raises(Exception):
        organize._validate_and_publish_case_metadata(course, case_metadata)

    assert not metadata_path.exists()
    assert not list((course / "metadata").glob("*.candidate.json"))


def test_manifest_writer_quarantines_one_stale_course_and_keeps_later_valid_course(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output_dir = tmp_path / "output"
    stale = output_dir / "P1" / "C1"
    valid = output_dir / "P2" / "C2"
    stale_metadata = write_minimal_course_contract(stale)
    write_minimal_course_contract(valid)
    stale_payload = json.loads(stale_metadata.read_text(encoding="utf-8"))
    stale_payload["course_contract"]["version"] = 0
    stale_metadata.write_text(json.dumps(stale_payload), encoding="utf-8")
    (stale / ".organized").write_text("ok\n", encoding="utf-8")
    (valid / ".organized").write_text("ok\n", encoding="utf-8")
    write_organize_ledger(
        output_dir, [_ledger_entry(stale), _ledger_entry(valid)]
    )

    calls = []

    def _producer_completed(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args[0], 0)

    monkeypatch.setattr(subprocess, "run", _producer_completed)
    workflow = _organize_workflow(tmp_path, output_dir)
    runpy.run_path(str(ORGANIZE_COURSES), init_globals={"snakemake": workflow})

    assert len(calls) == 1
    manifest = json.loads(
        Path(workflow.output.manifest).read_text(encoding="utf-8")
    )
    assert manifest["attempted_course_count"] == 2
    assert manifest["validated_course_count"] == 1
    assert manifest["technical_quarantine_count"] == 1
    assert [(item["patient"], item["course"]) for item in manifest["courses"]] == [
        ("P2", "C2")
    ]
    assert (valid / ".organized").read_text(encoding="utf-8").strip() == "ok"
    assert not stale.exists()
    assert not (stale / ".organized").exists()

    quarantine = manifest["technical_quarantines"][0]
    assert quarantine["disposition_type"] == "technical_quarantine"
    assert quarantine["clinical_exclusion"] is False
    assert "unsupported course contract version" in quarantine["reason"]
    quarantine_path = Path(quarantine["quarantine_path"])
    assert quarantine_path.is_dir()
    assert (quarantine_path / "technical_quarantine.json").is_file()

    ledger = read_organize_ledger(output_dir)
    assert ledger["attempted_course_count"] == 2
    assert ledger["validated_course_count"] == 1
    assert ledger["technical_quarantine_count"] == 1


def test_resume_revalidates_manifest_and_contract_before_skipping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output_dir = tmp_path / "output"
    course = output_dir / "P1" / "C1"
    write_minimal_course_contract(course)
    write_organize_ledger(output_dir, [_ledger_entry(course)])
    workflow = _organize_workflow(tmp_path, output_dir)

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0),
    )
    runpy.run_path(str(ORGANIZE_COURSES), init_globals={"snakemake": workflow})

    def _must_not_run(*args, **kwargs):
        raise AssertionError("validated resume should not rerun the producer")

    monkeypatch.setattr(subprocess, "run", _must_not_run)
    runpy.run_path(str(ORGANIZE_COURSES), init_globals={"snakemake": workflow})
    assert "contract validation" in Path(workflow.log[0]).read_text(encoding="utf-8")


def test_resume_success_flag_cannot_hide_a_stale_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output_dir = tmp_path / "output"
    course = output_dir / "P1" / "C1"
    metadata_path = write_minimal_course_contract(course)
    write_organize_ledger(output_dir, [_ledger_entry(course)])
    workflow = _organize_workflow(tmp_path, output_dir)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0),
    )
    runpy.run_path(str(ORGANIZE_COURSES), init_globals={"snakemake": workflow})
    assert (course / ".organized").is_file()

    stale = json.loads(metadata_path.read_text(encoding="utf-8"))
    stale["course_contract"]["version"] = 0
    metadata_path.write_text(json.dumps(stale), encoding="utf-8")
    producer_calls = 0

    def _producer_rerun(*args, **kwargs):
        nonlocal producer_calls
        producer_calls += 1
        return subprocess.CompletedProcess(args[0], 0)

    monkeypatch.setattr(subprocess, "run", _producer_rerun)
    runpy.run_path(str(ORGANIZE_COURSES), init_globals={"snakemake": workflow})

    assert producer_calls == 1
    assert not course.exists()
    manifest = json.loads(
        Path(workflow.output.manifest).read_text(encoding="utf-8")
    )
    assert manifest["validated_course_count"] == 0
    assert manifest["technical_quarantine_count"] == 1


def test_all_course_campaign_gate_blocks_before_scientific_aggregation(
    tmp_path: Path,
):
    output_dir = tmp_path / "output"
    results_dir = output_dir / "_RESULTS"
    manifest_path = output_dir / "_COURSES" / "manifest.json"
    valid = output_dir / "P1" / "C1"
    write_minimal_course_contract(valid)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "rtpipeline-organized-course-manifest-v2",
                "cohort_status": "complete_with_technical_quarantines",
                "intended_course_count": 2,
                "attempted_course_count": 2,
                "validated_course_count": 1,
                "technical_quarantine_count": 1,
                "courses": [
                    {"patient": "P1", "course": "C1", "path": str(valid)}
                ],
                "technical_quarantines": [
                    {
                        "patient": "P2",
                        "course": "C2",
                        "status": STATUS_TECHNICAL_QUARANTINE,
                        "reason": "CourseContractError: stale selected plan",
                        "disposition_type": "technical_quarantine",
                        "clinical_exclusion": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    outputs = {
        name: str(results_dir / filename)
        for name, filename in {
            "dvh": "dvh.xlsx",
            "fractions": "fractions.xlsx",
            "metadata": "metadata.xlsx",
            "qc": "qc.xlsx",
        }.items()
    }
    workflow = SimpleNamespace(
        input=SimpleNamespace(manifest=str(manifest_path)),
        output=SimpleNamespace(**outputs),
        log=[str(tmp_path / "logs" / "aggregate.log")],
        params=SimpleNamespace(
            output_dir=str(output_dir),
            results_dir=str(results_dir),
            radiomics_enabled=False,
            campaign_mode=True,
            campaign_min_completion_fraction=0.5,
            campaign_require_all_courses=True,
            worker_budget=1,
            auto_worker_budget=1,
            aggregation_threads=1,
        ),
    )

    with pytest.raises(RuntimeError, match="before scientific aggregation"):
        runpy.run_path(str(AGGREGATE_RESULTS), init_globals={"snakemake": workflow})

    gate = json.loads(
        (results_dir / "organization_gate.json").read_text(encoding="utf-8")
    )
    assert gate["status"] == "blocked"
    assert gate["attempted_course_count"] == 2
    assert gate["validated_course_count"] == 1
    assert gate["technical_quarantine_count"] == 1
    attrition = (results_dir / "campaign_attrition.csv").read_text(encoding="utf-8")
    assert "technical_quarantine" in attrition
    assert not any(Path(path).exists() for path in outputs.values())


def test_campaign_rollup_preserves_intended_denominator_and_technical_failure(
    tmp_path: Path,
):
    output_dir = tmp_path / "output"
    valid = output_dir / "P1" / "C1"
    valid.mkdir(parents=True)
    quarantined = output_dir / "P2" / "C2"
    write_organize_ledger(
        output_dir,
        [
            _ledger_entry(valid),
            _ledger_entry(
                quarantined,
                status=STATUS_TECHNICAL_QUARANTINE,
                reason="CourseContractError: stale selected plan",
            ),
        ],
    )
    namespace = runpy.run_path(str(CAMPAIGN_LEDGER))
    summary = namespace["rollup"](output_dir)

    assert summary["course_count"] == 2
    assert summary["visible_course_count"] == 1
    assert summary["intended_course_count"] == 2
    assert summary["producer_validated_course_count"] == 1
    assert summary["technical_quarantine_count"] == 1
    assert summary["failed_units"] == [
        {"patient": "P2", "course": "C2", "stage": "organize"}
    ]
    ledger_csv = (
        output_dir / "_campaign_ledger" / "campaign_ledger.csv"
    ).read_text(encoding="utf-8")
    assert "technical_quarantine" in ledger_csv
    assert "stale selected plan" in ledger_csv
