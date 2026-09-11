"""Synthetic regressions for immutable, manifest-bound cohort admission."""
import json
from dataclasses import replace

import pytest

from rtpipeline import radiomics_robustness as rr
from rtpipeline.course_manifest import read_course_manifest
from test_robustness_consumer_integration import (
    _build_courses, _run_course, _write_manifest, _rob_config, _summary_paths,
    _plant_stale_outputs, _assert_no_nominal_outputs,
)


def _fixture(tmp_path, monkeypatch, kinds=("measured", "source_only")):
    fixture = _build_courses(tmp_path, monkeypatch,
                             {(f"P{i}", "C1"): kind for i, kind in enumerate(kinds)})
    for patient, course in fixture.courses:
        assert _run_course(fixture, patient, course) == 0
    manifest = _write_manifest(fixture)
    entries, cohort = read_course_manifest(manifest, output_dir=fixture.output_root,
                                            require_current_schema=True)
    config = _rob_config(fixture.config_path)
    courses = [rr.admit_robustness_cohort_course(d, patient_id=p, course_id=c,
                                                rob_config=config) for p, c, d in entries]
    return fixture, manifest, cohort, config, courses


@pytest.mark.parametrize("tamper", ["frame", "rows", "config", "membership", "counts"])
def test_post_admission_tampering_is_rejected(tmp_path, monkeypatch, tamper):
    fixture, manifest, cohort, config, courses = _fixture(tmp_path, monkeypatch)
    if tamper == "frame":
        courses[0].frame.loc[0, "value"] = 999
    elif tamper == "rows":
        courses[1].source_dispositions[0]["reason"] = "altered"
    elif tamper == "config":
        config.enabled = False
    elif tamper == "membership":
        courses[0] = replace(courses[0], patient_id="P9")
    else:
        cohort["attempted_course_count"] = 999
    output, raw = _summary_paths(tmp_path)
    _plant_stale_outputs(output, raw)
    with pytest.raises((ValueError, RuntimeError)):
        rr.aggregate_robustness_cohort(courses, output, config, cohort=cohort)
    _assert_no_nominal_outputs(output, raw)


@pytest.mark.parametrize("drift", ["source", "manifest", "sidecar", "receipt", "config"])
def test_mid_summary_drift_withdraws_outputs(tmp_path, monkeypatch, drift):
    fixture, manifest, cohort, config, courses = _fixture(tmp_path, monkeypatch)
    summarize = rr.summarize_feature_stability

    def changed(*args, **kwargs):
        result = summarize(*args, **kwargs)
        if drift == "config":
            config.enabled = False
        else:
            path = {"source": courses[0].course_dir / "RS.dcm",
                    "manifest": manifest,
                    "sidecar": rr.robustness_source_dispositions_path(courses[0].course_dir),
                    "receipt": fixture.sentinel("P0")}[drift]
            path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(rr, "summarize_feature_stability", changed)
    output, raw = _summary_paths(tmp_path)
    _plant_stale_outputs(output, raw)
    with pytest.raises((ValueError, RuntimeError)):
        rr.aggregate_robustness_cohort(courses, output, config, cohort=cohort)
    _assert_no_nominal_outputs(output, raw)


@pytest.mark.parametrize("tamper", ["backup", "unmatched", "empty", "unmatched_empty"])
def test_source_only_requires_matching_canonical_nonempty_snapshot(tmp_path, monkeypatch, tamper):
    fixture, manifest, cohort, config, courses = _fixture(tmp_path, monkeypatch, ("source_only",))
    course = courses[0]
    path = rr.robustness_source_dispositions_path(course.course_dir)
    payload = json.loads(path.read_bytes())
    receipt = fixture.receipt("P0")
    if tamper == "backup":
        backup = path.with_name("backup.json")
        backup.write_bytes(path.read_bytes())
        receipt["source_dispositions"]["path"] = backup.name
    if tamper in ("backup", "unmatched", "unmatched_empty"):
        payload["measurement_outcome"] = rr.ROBUSTNESS_UNMATCHED_SELECTION_OUTCOME
    if tamper in ("empty", "unmatched_empty"):
        payload["rows"] = []
        payload["row_count"] = 0
        payload["rows_sha256"] = rr._content_sha256([])
        receipt["source_dispositions"]["row_count"] = 0
    path.write_text(json.dumps(payload))
    if tamper != "backup":
        receipt["source_dispositions"]["sha256"] = rr._file_sha256(path)
    fixture.write_receipt("P0", receipt)
    with pytest.raises((ValueError, RuntimeError)):
        rr.admit_robustness_cohort_course(course.course_dir, patient_id="P0", course_id="C1",
                                         rob_config=config)


@pytest.mark.parametrize("target", ["root", "course", "metadata", "sidecar", "receipt", "output"])
def test_output_aliases_are_rejected(tmp_path, monkeypatch, target):
    fixture, manifest, cohort, config, courses = _fixture(tmp_path, monkeypatch, ("source_only",))
    course = courses[0]
    output, raw = _summary_paths(tmp_path)
    if target == "output":
        output.parent.mkdir(parents=True, exist_ok=True)
        real = output.parent.with_name("real_results")
        output.parent.rename(real)
        output.parent.symlink_to(real, target_is_directory=True)
    else:
        path = {"root": fixture.output_root, "course": course.course_dir,
                "metadata": course.course_dir / "metadata",
                "sidecar": rr.robustness_source_dispositions_path(course.course_dir),
                "receipt": fixture.sentinel("P0")}[target]
        real = path.with_name(path.name + "_real")
        path.rename(real)
        path.symlink_to(real, target_is_directory=real.is_dir())
    with pytest.raises(RuntimeError, match="symlink"):
        rr.aggregate_robustness_cohort(courses, output, config, cohort=cohort)


@pytest.mark.parametrize("kinds", [("source_only", "source_only"), ("measured", "source_only")])
def test_valid_snapshot_publishes(tmp_path, monkeypatch, kinds):
    _, _, cohort, config, courses = _fixture(tmp_path, monkeypatch, kinds)
    output, raw = _summary_paths(tmp_path)
    rr.aggregate_robustness_cohort(courses, output, config, cohort=cohort)
    assert output.is_file() and raw.is_file()


def test_equal_count_valid_nonmanifest_course_cannot_substitute(tmp_path, monkeypatch):
    fixture, _, _, config, courses = _fixture(tmp_path, monkeypatch)
    manifest = _write_manifest(fixture, entries=[{"patient": "P0", "course": "C1"}])
    _, cohort = read_course_manifest(manifest, output_dir=fixture.output_root,
                                    require_current_schema=True)
    output, raw = _summary_paths(tmp_path)
    with pytest.raises(ValueError, match="membership"):
        rr.aggregate_robustness_cohort([courses[1]], output, config, cohort=cohort)
    _assert_no_nominal_outputs(output, raw)


def test_drift_during_workbook_serialization_is_rejected(tmp_path, monkeypatch):
    _, manifest, cohort, config, courses = _fixture(tmp_path, monkeypatch)
    original = rr.pd.DataFrame.to_excel

    def changed(frame, *args, **kwargs):
        result = original(frame, *args, **kwargs)
        manifest.write_bytes(manifest.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(rr.pd.DataFrame, "to_excel", changed)
    output, raw = _summary_paths(tmp_path)
    with pytest.raises(ValueError, match="manifest changed"):
        rr.aggregate_robustness_cohort(courses, output, config, cohort=cohort)
    _assert_no_nominal_outputs(output, raw)


@pytest.mark.parametrize("integrity_only", [False, True])
def test_source_only_unrequested_rows_are_not_anatomical_completion(tmp_path, monkeypatch, integrity_only):
    fixture, _, _, config, courses = _fixture(tmp_path, monkeypatch, ("source_only",))
    course = courses[0]
    path = rr.robustness_source_dispositions_path(course.course_dir)
    payload = json.loads(path.read_bytes())
    payload["rows"][0]["roi_name"] = "Unrequested1"
    payload["rows_sha256"] = rr._content_sha256(payload["rows"])
    path.write_text(json.dumps(payload))
    receipt = fixture.receipt("P0")
    receipt["source_dispositions"]["sha256"] = rr._file_sha256(path)
    fixture.write_receipt("P0", receipt)
    with pytest.raises(ValueError, match="matched no source"):
        if integrity_only:
            rr.inspect_robustness_source_dispositions(course.course_dir,
                                                      run_identifier=course.run_identifier)
        else:
            rr.admit_robustness_cohort_course(course.course_dir, patient_id="P0",
                                             course_id="C1", rob_config=config)


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("basis_present", [False, True])
def test_source_only_case_insensitive_selection_allows_unrelated_rows(
    tmp_path, monkeypatch, mixed, basis_present
):
    fixture, _, _, config, courses = _fixture(tmp_path, monkeypatch, ("source_only",))
    course = courses[0]
    path = rr.robustness_source_dispositions_path(course.course_dir)
    payload = json.loads(path.read_bytes())
    payload["rows"][0]["roi_name"] = "mArKeR22"
    if mixed:
        row = dict(payload["rows"][0], roi_name="Unrequested1", roi_number="3")
        payload["rows"].append(row)
    payload["row_count"] = len(payload["rows"])
    payload["rows_sha256"] = rr._content_sha256(payload["rows"])
    if basis_present:
        payload["source_only_basis"]["source_disposition_count"] = len(payload["rows"])
    else:
        payload.pop("source_only_basis", None)
    path.write_text(json.dumps(payload))
    receipt = fixture.receipt("P0")
    receipt["source_dispositions"].update(sha256=rr._file_sha256(path), row_count=len(payload["rows"]))
    fixture.write_receipt("P0", receipt)
    admitted = rr.admit_robustness_cohort_course(course.course_dir, patient_id="P0",
                                              course_id="C1", rob_config=config)
    inspected = rr.inspect_robustness_source_dispositions(course.course_dir,
                                                        run_identifier=course.run_identifier)
    assert len(admitted.source_dispositions) == 1 + mixed
    assert inspected.rows == admitted.source_dispositions
    assert not inspected.configuration_verified


@pytest.mark.parametrize("field,value", [
    ("selection_matched_nonvolumetric_count", 0),
    ("requested_selection", ["Unrequested*"]),
    ("source_disposition_count", 22),
])
def test_source_only_basis_must_reconcile(tmp_path, monkeypatch, field, value):
    _, _, _, _, courses = _fixture(tmp_path, monkeypatch, ("source_only",))
    course = courses[0]
    path = rr.robustness_source_dispositions_path(course.course_dir)
    payload = json.loads(path.read_bytes())
    payload["source_only_basis"][field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="basis.*does not reconcile"):
        rr.inspect_robustness_source_dispositions(course.course_dir,
                                                run_identifier=course.run_identifier)


def test_explicit_aggregation_preserves_foreign_alias_outputs(tmp_path):
    real = tmp_path / "foreign"
    real.mkdir()
    output = real / "summary.xlsx"
    raw = real / "summary_raw_values.parquet"
    output.write_bytes(b"foreign workbook")
    raw.write_bytes(b"foreign raw")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        rr.aggregate_robustness_results([], alias / output.name, rr.RobustnessConfig())
    assert output.read_bytes() == b"foreign workbook"
    assert raw.read_bytes() == b"foreign raw"


def test_count_only_cohort_is_not_a_manifest_binding(tmp_path, monkeypatch):
    _, _, _, config, courses = _fixture(tmp_path, monkeypatch)
    output, raw = _summary_paths(tmp_path)
    with pytest.raises(ValueError, match="manifest snapshot"):
        rr.aggregate_robustness_cohort(courses, output, config,
                                       cohort={"validated_course_count": len(courses)})
    _assert_no_nominal_outputs(output, raw)
