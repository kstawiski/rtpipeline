"""Robustness code identity must bind every directly executed measurement module.

Every admission of a robustness artifact (course reuse, aggregation input and
manifest cohort admission) funnels through the disposition sidecar loader,
which compares the recorded ``code_identity.sources_sha256`` with the current
on-disk digest of ``ROBUSTNESS_DISPOSITION_CODE_SOURCES``. That list is the
*only* code binding on robustness reuse: the completion receipt binds bytes,
not code, and ``stage_completion`` defines no robustness stage.

The robustness measurement path executes, beyond the originally bound modules,
``radiomics_parallel.py`` (the isolated per-condition extractor that produces
the 81-condition rows in the default parallel mode), ``radiomics_ct_contract.py``
(ROI arm classification and per-arm extraction in both modes),
``robustness_mcc.py`` (the MCC computation installed into PyRadiomics inside the
worker) and ``radiomics_conda.py`` (the sequential helper-environment batch
extraction used when PyRadiomics is not importable natively). A change confined
to any of these alters measured values while leaving the bound digest intact.

These tests simulate a content change of one dependency by intercepting the
module-level digest reader, so the live source is never edited. They assert
disk-content binding only. They do not prove which bytes the interpreter
executed, and the synthetic tables here carry no PyRadiomics measurement.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rtpipeline import radiomics_robustness as rr
from rtpipeline import robustness_completion as rc
from rtpipeline.radiomics_robustness import RobustnessConfig

PACKAGE = Path(rr.__file__).resolve().parent
OUTPUT_NAME = "radiomics_robustness_ct.parquet"
RUN_ID = "run-synthetic-binding"

# Modules executed directly on the robustness measurement path that the
# original four-entry list omitted.
DIRECT_EXECUTION_SOURCES = (
    "radiomics_parallel.py",
    "radiomics_ct_contract.py",
    "robustness_mcc.py",
    "radiomics_conda.py",
)
ORIGINALLY_BOUND_SOURCES = (
    "radiomics_robustness.py",
    "radiomics_robustness_outcomes.py",
    "radiomics.py",
    "rtstruct_geometry.py",
)


def _drift_source_digest(monkeypatch, relative_name: str) -> None:
    """Make one package module read as if its content had changed.

    Only the module-level digest reader is intercepted, and only for the named
    package file, so the measured table and sidecar digests remain genuine.
    """
    real = rr._file_sha256

    def drifted(path):
        digest = real(path)
        candidate = Path(path)
        if candidate.name == relative_name and candidate.resolve().parent == PACKAGE:
            return hashlib.sha256(
                bytes.fromhex(digest) + b"direct measurement dependency changed"
            ).hexdigest()
        return digest

    monkeypatch.setattr(rr, "_file_sha256", drifted)


def _course(tmp_path: Path) -> Path:
    course = tmp_path / "P1" / "C1"
    course.mkdir(parents=True)
    return course


def _publish_measured(course: Path, rob: RobustnessConfig) -> Path:
    """Publish a synthetic measured table with the shipped sidecar writer.

    The bytes are not a measurement; they exist so the sidecar can bind a file.
    """
    table = course / OUTPUT_NAME
    table.write_bytes(b"synthetic robustness table bytes; not a measurement\n")
    rr._write_robustness_source_dispositions(
        course,
        run_identifier=RUN_ID,
        rows=[],
        source_bindings=[],
        effective_configuration=rr.effective_robustness_configuration(
            rob, output_name=OUTPUT_NAME
        ),
        code_identity=rr._current_robustness_code_identity(),
        output_path=table,
        measured_output=table,
    )
    return table


def _publish_zero_measurement_with_receipt(course: Path, rob: RobustnessConfig):
    """Publish a source-only (zero measurement) sidecar and its completion receipt.

    The RTSTRUCT is a synthetic byte file: the source binding is content-hash
    and identity-string bound, and no DICOM is parsed on this path. The single
    disposition row names a non-volumetric geometry for a selected ROI family,
    which is the only zero-measurement outcome that completes the step.
    """
    rtstruct = course / "RS_synthetic.dcm"
    rtstruct.write_bytes(b"synthetic rtstruct bytes; not DICOM; not clinical\n")
    binding = {
        "segmentation_source": "Manual",
        "source_path": str(rtstruct),
        "rtstruct_sop_instance_uid": "1.2.826.0.1.3680043.8.498.synthetic.1",
        "sha256": hashlib.sha256(rtstruct.read_bytes()).hexdigest(),
    }
    row = {
        "segmentation_source": binding["segmentation_source"],
        "source_path": binding["source_path"],
        "rtstruct_sop_instance_uid": binding["rtstruct_sop_instance_uid"],
        "roi_name": "GTV",
        "roi_number": "1",
        "status": "nonvolumetric_nonmeasurement",
        "failure_kind": "geometry",
        "structural_code": "ROI_NONVOLUMETRIC_POINT",
        "reason": "synthetic point contour; no volume to perturb",
    }
    effective = rr.effective_robustness_configuration(rob, output_name=OUTPUT_NAME)
    sidecar = rr._write_robustness_source_dispositions(
        course,
        run_identifier=RUN_ID,
        rows=[row],
        source_bindings=[binding],
        effective_configuration=effective,
        code_identity=rr._current_robustness_code_identity(),
        output_path=course / OUTPUT_NAME,
        measured_output=None,
        nonmeasured_outcome=rr.ROBUSTNESS_SOURCE_ONLY_OUTCOME,
    )
    receipt = rc.write_robustness_completion_sentinel(
        rc.robustness_completion_sentinel_path(course),
        course,
        patient_id="P1",
        course_id="C1",
        run_identifier=RUN_ID,
        measurement_outcome=rr.ROBUSTNESS_SOURCE_ONLY_OUTCOME,
        output_name=OUTPUT_NAME,
        dispositions_path=sidecar,
        measured_output=None,
        source_disposition_count=1,
        effective_configuration_sha256=rr._content_sha256(effective),
    )
    return sidecar, receipt


def test_direct_execution_sources_are_bound_and_present():
    bound = set(rr.ROBUSTNESS_DISPOSITION_CODE_SOURCES)
    missing = [name for name in DIRECT_EXECUTION_SOURCES if name not in bound]
    assert not missing, f"unbound direct measurement dependencies: {missing}"
    assert set(ORIGINALLY_BOUND_SOURCES) <= bound
    for relative in rr.ROBUSTNESS_DISPOSITION_CODE_SOURCES:
        assert (PACKAGE / relative).is_file(), relative
    identity = rr._current_robustness_code_identity()
    assert {entry["path"] for entry in identity["sources"]} == bound


def test_unchanged_measured_table_is_admitted(tmp_path):
    rob = RobustnessConfig(enabled=True)
    course = _course(tmp_path)
    table = _publish_measured(course, rob)
    rows = rr.load_robustness_source_dispositions(
        course, run_identifier=RUN_ID, rob_config=rob, output_name=OUTPUT_NAME
    )
    assert rows == []
    payload = json.loads(
        (course / "metadata" / rr.ROBUSTNESS_SOURCE_DISPOSITIONS_FILENAME).read_text()
    )
    assert payload["measurement_outcome"] == rr.ROBUSTNESS_MEASURED_OUTCOME
    assert payload["measured_output"]["sha256"] == hashlib.sha256(
        table.read_bytes()
    ).hexdigest()
    recorded = {e["path"] for e in payload["code_identity"]["sources"]}
    assert set(DIRECT_EXECUTION_SOURCES) <= recorded


@pytest.mark.parametrize("relative", DIRECT_EXECUTION_SOURCES + ORIGINALLY_BOUND_SOURCES)
def test_measured_table_is_rejected_after_execution_dependency_changes(
    tmp_path, monkeypatch, relative
):
    rob = RobustnessConfig(enabled=True)
    course = _course(tmp_path)
    _publish_measured(course, rob)
    # Positive control before the simulated change.
    assert rr.load_robustness_source_dispositions(
        course, run_identifier=RUN_ID, rob_config=rob, output_name=OUTPUT_NAME
    ) == []

    _drift_source_digest(monkeypatch, relative)
    with pytest.raises(ValueError, match="produced by different code") as excinfo:
        rr.load_robustness_source_dispositions(
            course, run_identifier=RUN_ID, rob_config=rob, output_name=OUTPUT_NAME
        )
    message = str(excinfo.value)
    # A code-binding failure is a technical rejection, never a clinical exclusion.
    assert "clinic" not in message.lower()
    assert "exclu" not in message.lower()


@pytest.mark.parametrize("relative", DIRECT_EXECUTION_SOURCES)
def test_zero_measurement_receipt_admits_unchanged_and_rejects_changed_code(
    tmp_path, monkeypatch, relative
):
    rob = RobustnessConfig(enabled=True)
    course = _course(tmp_path)
    _publish_zero_measurement_with_receipt(course, rob)
    admitted = rr.admit_robustness_cohort_course(
        course, patient_id="P1", course_id="C1", rob_config=rob
    )
    assert not admitted.measured
    assert admitted.measurement_outcome == rr.ROBUSTNESS_SOURCE_ONLY_OUTCOME
    assert len(admitted.source_dispositions) == 1
    assert admitted.source_dispositions[0]["structural_code"] == "ROI_NONVOLUMETRIC_POINT"
    assert not (course / OUTPUT_NAME).exists()

    _drift_source_digest(monkeypatch, relative)
    with pytest.raises(ValueError, match="produced by different code"):
        rr.admit_robustness_cohort_course(
            course, patient_id="P1", course_id="C1", rob_config=rob
        )


def test_publication_refuses_when_execution_dependency_changes_mid_run(
    tmp_path, monkeypatch
):
    rob = RobustnessConfig(enabled=True)
    course = _course(tmp_path)
    captured = rr._capture_robustness_code_identity()
    table = course / OUTPUT_NAME
    table.write_bytes(b"synthetic\n")
    _drift_source_digest(monkeypatch, "radiomics_parallel.py")
    with pytest.raises(RuntimeError, match="deciding code changed on disk"):
        rr._write_robustness_source_dispositions(
            course,
            run_identifier=RUN_ID,
            rows=[],
            source_bindings=[],
            effective_configuration=rr.effective_robustness_configuration(
                rob, output_name=OUTPUT_NAME
            ),
            code_identity=captured,
            output_path=table,
            measured_output=table,
        )
    assert not (course / "metadata" / rr.ROBUSTNESS_SOURCE_DISPOSITIONS_FILENAME).exists()


def test_missing_bound_code_source_fails_closed(monkeypatch):
    monkeypatch.setattr(
        rr,
        "ROBUSTNESS_DISPOSITION_CODE_SOURCES",
        rr.ROBUSTNESS_DISPOSITION_CODE_SOURCES + ("absent_execution_module.py",),
    )
    with pytest.raises(RuntimeError, match="code source is absent"):
        rr._current_robustness_code_identity()


@pytest.mark.parametrize("relative", DIRECT_EXECUTION_SOURCES)
def test_measured_receipt_and_aggregation_reject_changed_execution_code(
    tmp_path, monkeypatch, relative
):
    """Exercise real Parquet/receipt admission with mocked feature extraction.

    The inspected cohort fixture creates three perturbation conditions, not
    the full 81-condition study grid. Its course/image/source identity and
    feature extractor are synthetic. Publication, receipt validation, Parquet
    parsing and aggregation admission remain the shipped implementations.
    """
    from test_robustness_aggregation_admission import _build_cohort

    cohort = _build_cohort(tmp_path, monkeypatch, patients=("P1",))
    course, table = cohort.courses[0], cohort.tables[0]
    sidecar = rr.robustness_source_dispositions_path(course)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    rc.write_robustness_completion_sentinel(
        rc.robustness_completion_sentinel_path(course),
        course,
        patient_id="P1",
        course_id="C1",
        run_identifier=payload["robustness_run_identifier"],
        measurement_outcome=rr.ROBUSTNESS_MEASURED_OUTCOME,
        output_name=table.name,
        dispositions_path=sidecar,
        measured_output=table,
        source_disposition_count=len(payload["rows"]),
        effective_configuration_sha256=rr._content_sha256(
            rr.effective_robustness_configuration(cohort.rob, output_name=table.name)
        ),
    )
    admitted = rr.admit_robustness_cohort_course(
        course, patient_id="P1", course_id="C1", rob_config=cohort.rob
    )
    assert admitted.measured
    assert len(admitted.frame) == 6  # 3 synthetic conditions x 2 CT arms.
    assert len(rr._admit_robustness_aggregation_input(table, cohort.rob).frame) == 6

    _drift_source_digest(monkeypatch, relative)
    with pytest.raises(ValueError, match="produced by different code"):
        rr.admit_robustness_cohort_course(
            course, patient_id="P1", course_id="C1", rob_config=cohort.rob
        )

    output = tmp_path / "rejected_summary.xlsx"
    raw = output.with_name(output.stem + "_raw_values.parquet")
    output.write_bytes(b"stale synthetic summary")
    raw.write_bytes(b"stale synthetic raw output")
    with pytest.raises(ValueError, match="produced by different code"):
        rr.aggregate_robustness_results([table], output, cohort.rob)
    assert not output.exists()
    assert not raw.exists()
