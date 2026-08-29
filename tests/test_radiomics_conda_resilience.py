"""Regression tests for radiomics_conda reliability failures.

Bug A — checkpoint flush crash:
    PyRadiomics emits diagnostics fields such as
    ``diagnostics_Configuration_EnabledImageTypes`` whose value is a nested dict
    ``{'Original': {}}``. That empty-child struct rode into the checkpoint buffer
    via ``_combine_feature_record`` and made ``DataFrame.to_parquet`` raise
    "Cannot write struct type 'Original' with no child field to Parquet", so EVERY
    checkpoint flush failed (228x in the live log) and intra-course resume was lost.

Bug B — per-course conda env probe times out under load and silently skips courses:
    ``process_radiomics_batch`` called ``check_radiomics_env()`` once per course.
    Each call spawned a ``conda run ... import radiomics`` subprocess with a hard
    60 s timeout. Under the nested worker load the cold-start probe timed out,
    returned False, and the course returned None with no ``radiomics_ct.xlsx`` —
    a silently-dropped course (46 spurious failures in the live log, clustered at
    cold start). The fix caches the first SUCCESSFUL check process-wide and gives
    the probe a generous timeout + one retry.
"""

import inspect
import json
import subprocess
import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

import rtpipeline.radiomics_conda as rc
from course_contract_test_utils import (
    write_minimal_course_contract,
    write_synthetic_planning_ct,
)
import rtpipeline.cli as cli
from rtpipeline.radiomics_conda import (
    RadiomicsCheckpoint,
    _jsonify_nested_columns,
    _roi_instance_key,
    check_radiomics_env,
    process_radiomics_batch,
)


def test_batch_extraction_default_allows_large_roi_runtime():
    default = inspect.signature(rc.extract_radiomics_batch_with_conda).parameters[
        "timeout_per_roi"
    ].default
    assert default == 900


# The exact PyRadiomics diagnostics value that broke Parquet serialization.
_NESTED_DIAGNOSTIC = {"Original": {}}


def _feature_record(roi_name):
    """A record shaped like _combine_feature_record output: scalar features +
    the nested diagnostics struct + scalar metadata."""
    return {
        "original_firstorder_Mean": 42.0,
        "original_shape_VoxelVolume": 1234.5,
        "diagnostics_Configuration_EnabledImageTypes": _NESTED_DIAGNOSTIC,
        "diagnostics_Versions_PyRadiomics": "3.1.0",
        "segmentation_source": "Manual",
        "roi_name": roi_name,
        "roi_original_name": roi_name,
        "modality": "CT",
    }


def _write_nifti(path, array, *, spacing=(1.0, 1.0, 1.0)):
    img = sitk.GetImageFromArray(np.asarray(array))
    img.SetSpacing(tuple(float(x) for x in spacing))
    sitk.WriteImage(img, str(path))


def _minimal_config(**overrides):
    data = {
        "radiomics_params_file": None,
        "radiomics_skip_rois": [],
        "radiomics_max_voxels": 10_000_000,
        "radiomics_min_voxels": 2,
        "effective_workers": lambda: 1,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


# ---- Bug A: checkpoint flush must survive nested diagnostics ----------------------------

def test_nested_diagnostics_is_a_real_parquet_hazard(tmp_path):
    """Document the root cause: writing the raw record to Parquet DOES raise, so the
    sanitizer is load-bearing, not cosmetic."""
    raw = pd.DataFrame([_feature_record("CTV")])
    with pytest.raises(Exception):
        raw.to_parquet(tmp_path / "raw.parquet", index=False)


def test_jsonify_nested_columns_encodes_nested_keeps_scalars():
    import json
    df = pd.DataFrame([_feature_record("CTV")])
    cleaned = _jsonify_nested_columns(df)
    # Nested column is RETAINED (faithful checkpoint) but now a JSON string.
    assert "diagnostics_Configuration_EnabledImageTypes" in cleaned.columns
    val = cleaned["diagnostics_Configuration_EnabledImageTypes"].iloc[0]
    assert isinstance(val, str)
    assert json.loads(val) == _NESTED_DIAGNOSTIC
    # Scalar feature + identity columns are untouched.
    for col in ("roi_name", "original_firstorder_Mean", "diagnostics_Versions_PyRadiomics"):
        assert col in cleaned.columns
    assert cleaned["original_firstorder_Mean"].iloc[0] == 42.0


def test_jsonified_frame_writes_to_parquet(tmp_path):
    cleaned = _jsonify_nested_columns(pd.DataFrame([_feature_record("CTV")]))
    out = tmp_path / "cleaned.parquet"
    cleaned.to_parquet(out, index=False)  # must not raise
    assert out.exists()


def test_checkpoint_flush_survives_and_preserves_resume(tmp_path):
    """End-to-end: a checkpoint fed nested-diagnostics records flushes without error
    and a fresh checkpoint recovers the completed roi_name set (the resume contract)."""
    cp_path = tmp_path / "metadata" / "radiomics_ct_checkpoint.parquet"
    cp = RadiomicsCheckpoint(cp_path, buffer_size=2)
    cp.add_result(_feature_record("CTV"))
    cp.add_result(_feature_record("BODY"))  # triggers buffer flush at size 2
    cp.add_result(_feature_record("BLADDER"))
    cp.flush()

    assert cp_path.exists(), "checkpoint parquet should have been written"

    resumed = RadiomicsCheckpoint(cp_path)
    assert resumed.is_completed(_roi_instance_key(_feature_record("CTV")))
    assert resumed.is_completed(_roi_instance_key(_feature_record("BODY")))
    assert resumed.is_completed(_roi_instance_key(_feature_record("BLADDER")))
    assert resumed.get_completed_count() == 3


def test_checkpoint_source_uses_sanitizer():
    """Anti-regression: the flush path must route through the sanitizer, not call a
    bare DataFrame(...).to_parquet on the raw buffer."""
    import inspect
    src = inspect.getsource(RadiomicsCheckpoint._flush_buffer)
    assert "_jsonify_nested_columns(" in src, (
        "_flush_buffer must sanitize nested columns before to_parquet"
    )


def test_resume_writes_complete_workbook(tmp_path, monkeypatch):
    """REGRESSION (the blocking issue): a course whose checkpoint covers only PART of
    the current task inventory must still write EVERY ROI to radiomics_ct.xlsx — never
    just the ROIs processed in this run. Under the pre-fix code the workbook contained
    only the newly-processed subset (silent partial-data-loss feeding the cohort).

    The checkpoint is now all-or-nothing: a partial one is rejected outright rather
    than resumed, so every ROI is recomputed and no stale checkpoint row can reach the
    published workbook either.
    """
    monkeypatch.setattr(rc, "check_radiomics_env", lambda *a, **k: True)
    extracted = []

    def fake_batch(tasks, params_file=None):
        extracted.extend(task["roi_name"] for task in tasks)
        return [
            {"__status__": "success", "__task_index__": i, "original_firstorder_Mean": float(i + 100)}
            for i, _t in enumerate(tasks)
        ]

    monkeypatch.setattr(rc, "extract_radiomics_batch_with_conda", fake_batch)

    checkpoint_path = tmp_path / "metadata" / "radiomics_ct_checkpoint.parquet"
    output_path = tmp_path / "radiomics_ct.xlsx"

    # Simulate run 1 having completed CTV and BODY (incl. the nested diagnostics).
    cp = RadiomicsCheckpoint(checkpoint_path, buffer_size=1)
    cp.add_result(_feature_record("CTV"))
    cp.add_result(_feature_record("BODY"))
    cp.flush()

    # Run 2: the task set lists all three, so the two-ROI checkpoint is partial.
    tasks = [
        {"image_path": "i", "mask_path": "m", "roi_name": name, "cleanup": False,
         "metadata": {"segmentation_source": "Manual", "roi_original_name": name}}
        for name in ("CTV", "BODY", "BLADDER")
    ]
    result = process_radiomics_batch(
        tasks, str(output_path), sequential=True, max_workers=1,
        checkpoint_path=checkpoint_path, enable_heartbeat=False,
    )
    assert result is not None
    assert sorted(extracted) == ["BLADDER", "BODY", "CTV"], (
        f"a partial checkpoint must be rejected, not resumed; extracted {extracted}"
    )
    published = pd.read_excel(output_path)
    written = set(published["roi_name"].tolist())
    assert written == {"CTV", "BODY", "BLADDER"}, (
        f"resumed workbook must contain all completed ROIs; got {written}"
    )
    means = set(published["original_firstorder_Mean"].tolist())
    assert means == {100.0, 101.0, 102.0}, (
        f"every published row must be recomputed, not a stale checkpoint row; got {means}"
    )


def test_fully_checkpointed_course_regenerates_even_if_env_probe_fails(tmp_path, monkeypatch):
    """When every ROI is already checkpointed, no feature computation is needed, so a
    failing conda env probe must NOT block rebuilding the workbook from the checkpoint.
    (The env gate now runs only when uncompleted tasks remain.) Under the pre-fix order
    the probe ran first and returned None, leaving a checkpointed-but-unwritten course
    with no radiomics_ct.xlsx.
    """
    # Env probe deliberately fails — but there is nothing to compute, so it must be skipped.
    monkeypatch.setattr(rc, "check_radiomics_env", lambda *a, **k: False)

    checkpoint_path = tmp_path / "metadata" / "radiomics_ct_checkpoint.parquet"
    output_path = tmp_path / "radiomics_ct.xlsx"
    cp = RadiomicsCheckpoint(checkpoint_path, buffer_size=1)
    cp.add_result(_feature_record("CTV"))
    cp.add_result(_feature_record("BODY"))
    cp.flush()

    tasks = [
        {"image_path": "i", "mask_path": "m", "roi_name": name, "cleanup": False,
         "metadata": {"segmentation_source": "Manual", "roi_original_name": name}}
        for name in ("CTV", "BODY")
    ]
    result = process_radiomics_batch(
        tasks, str(output_path), sequential=True, max_workers=1,
        checkpoint_path=checkpoint_path, enable_heartbeat=False,
    )
    assert result is not None, "fully-checkpointed course must still write its workbook"
    written = set(pd.read_excel(output_path)["roi_name"].tolist())
    assert written == {"CTV", "BODY"}, f"expected both checkpointed ROIs; got {written}"


def test_roi_instance_key_distinguishes_mr_series():
    """The checkpoint key must combine roi_name with series_uid (task: under metadata;
    record: top-level). Same roi_name + different series_uid → different keys."""
    a = _roi_instance_key({"roi_name": "liver", "metadata": {"series_uid": "1.2.A"}})
    b = _roi_instance_key({"roi_name": "liver", "metadata": {"series_uid": "1.2.B"}})
    rec = _roi_instance_key({"roi_name": "liver", "series_uid": "1.2.A"})
    assert a != b, "distinct MR series must yield distinct keys"
    assert a == rec, "task and flattened record for the same instance must agree"
    # NaN series_uid (as read back from Parquet) collapses to empty, not 'nan'.
    assert _roi_instance_key({"roi_name": "x", "series_uid": float("nan")}) == _roi_instance_key({"roi_name": "x"})


def test_mr_duplicate_roi_name_across_series_not_collapsed(tmp_path, monkeypatch):
    """REGRESSION: MR reuses the mask-derived roi_name across series (only series_uid
    differs). Checkpoint dedup + the workbook union must NOT collapse them — under the
    roi_name-only keying a resumed MR course kept only the last series per ROI.
    """
    monkeypatch.setattr(rc, "check_radiomics_env", lambda *a, **k: True)
    monkeypatch.setattr(
        rc, "extract_radiomics_batch_with_conda",
        lambda tasks, params_file=None: [
            {"__status__": "success", "original_firstorder_Mean": float(i + 1)} for i, _t in enumerate(tasks)
        ],
    )
    cp_path = tmp_path / "metadata" / "radiomics_mr_checkpoint.parquet"
    out = tmp_path / "radiomics_mr.xlsx"

    cp = RadiomicsCheckpoint(cp_path, buffer_size=1)
    cp.add_result({"roi_name": "liver", "series_uid": "1.2.A", "modality": "MR",
                   "segmentation_source": "AutoTS_total_mr", "original_firstorder_Mean": 1.0})
    cp.add_result({"roi_name": "liver", "series_uid": "1.2.B", "modality": "MR",
                   "segmentation_source": "AutoTS_total_mr", "original_firstorder_Mean": 2.0})
    cp.flush()
    assert len(cp.load_records()) == 2, "distinct (roi_name, series_uid) must not be deduped to one"

    # Resume: both liver instances already checkpointed → both filtered → workbook rebuilt
    # from the checkpoint must contain BOTH series.
    tasks = [
        {"image_path": "i", "mask_path": "m", "roi_name": "liver", "cleanup": False,
         "metadata": {"modality": "MR", "series_uid": "1.2.A",
                      "segmentation_source": "AutoTS_total_mr"}},
        {"image_path": "i", "mask_path": "m", "roi_name": "liver", "cleanup": False,
         "metadata": {"modality": "MR", "series_uid": "1.2.B",
                      "segmentation_source": "AutoTS_total_mr"}},
    ]
    result = process_radiomics_batch(
        tasks, str(out), sequential=True, max_workers=1,
        checkpoint_path=cp_path, enable_heartbeat=False,
    )
    assert result is not None
    series = sorted(str(s) for s in pd.read_excel(out)["series_uid"].tolist())
    assert series == ["1.2.A", "1.2.B"], f"both MR series' liver must survive resume; got {series}"


def test_ct_totalseg_nifti_fallback_writes_tagged_workbook(tmp_path, monkeypatch):
    """No generated RS file: CT radiomics should recover directly from TS NIfTI masks."""
    course = tmp_path / "P1" / "2024-12"
    nifti_dir = course / "NIFTI"
    seg_dir = course / "Segmentation_TotalSegmentator" / "CT_SERIES"
    nifti_dir.mkdir(parents=True)
    seg_dir.mkdir(parents=True)
    (course / "metadata").mkdir(parents=True)

    _write_nifti(nifti_dir / "CT_SERIES.nii.gz", np.ones((5, 5, 5), dtype=np.int16))
    ct_dir = write_synthetic_planning_ct(course)
    write_minimal_course_contract(
        course,
        planning_ct_dir=ct_dir,
        planning_ct_nifti=nifti_dir / "CT_SERIES.nii.gz",
    )
    series_uid = json.loads(
        (course / "metadata" / "case_metadata.json").read_text(encoding="utf-8")
    )["course_contract"]["planning_ct"]["series_instance_uid"]
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[1:4, 1:4, 1:4] = 1
    _write_nifti(seg_dir / "total--liver.nii.gz", mask)
    _write_nifti(seg_dir / "total--liver_cropped.nii.gz", mask)
    _write_nifti(seg_dir / "total--skip_me.nii.gz", mask)

    seen = {}

    def fake_batch(tasks, params_file=None):
        seen["tasks"] = tasks
        return [
            {"__status__": "success", "__task_index__": i, "original_firstorder_Mean": float(i + 1)}
            for i, _task in enumerate(tasks)
        ]

    monkeypatch.setattr(rc, "check_radiomics_env", lambda *a, **k: True)
    monkeypatch.setattr(rc, "extract_radiomics_batch_with_conda", fake_batch)
    monkeypatch.setenv("RTPIPELINE_RADIOMICS_SEQUENTIAL", "1")

    out = rc.radiomics_for_course(course, _minimal_config(radiomics_skip_rois=["skip_me"]))

    assert out == course / "radiomics_ct.xlsx"
    tasks = seen["tasks"]
    assert len(tasks) == 1
    assert tasks[0]["roi_name"] == "liver"
    assert tasks[0]["metadata"]["series_uid"] == series_uid
    assert tasks[0]["metadata"]["segmentation_source"] == "AutoTS_total_nifti_fallback"
    assert tasks[0]["metadata"]["roi_original_name"] == "liver"

    df = pd.read_excel(out)
    assert df["segmentation_source"].tolist() == ["AutoTS_total_nifti_fallback"]
    assert df["series_uid"].tolist() == [series_uid]
    assert df["roi_name"].tolist() == ["liver"]
    assert df["original_firstorder_Mean"].tolist() == [1]


def test_ct_totalseg_nifti_fallback_not_used_when_rs_selected(tmp_path, monkeypatch):
    """A selected RS path keeps the RTSTRUCT branch; fallback is only for no usable RS."""
    course = tmp_path / "P1" / "2024-12"
    (course / "DICOM" / "CT").mkdir(parents=True)
    (course / "RS_auto.dcm").write_bytes(b"not read by this test")
    write_minimal_course_contract(course)

    monkeypatch.setattr(rc, "_select_usable_rtstruct", lambda *paths: course / "RS_auto.dcm")

    class _FakeRTStruct:
        def get_roi_names(self):
            return []

    fake_rt_utils = types.SimpleNamespace(
        RTStructBuilder=types.SimpleNamespace(
            create_from=lambda dicom_series_path, rt_struct_path: _FakeRTStruct()
        )
    )
    monkeypatch.setitem(sys.modules, "rt_utils", fake_rt_utils)

    class _FakeReader:
        def GetGDCMSeriesFileNames(self, _path):
            return ["ct_000.dcm"]

        def SetFileNames(self, _paths):
            return None

        def Execute(self):
            return sitk.Image([2, 2, 2], sitk.sitkInt16)

    monkeypatch.setattr(rc.sitk, "ImageSeriesReader", _FakeReader)

    def fail_fallback(*args, **kwargs):
        raise AssertionError("NIfTI fallback must not run when an RS is selected")

    monkeypatch.setattr(rc, "radiomics_for_course_ct_nifti_fallback", fail_fallback)

    assert rc.radiomics_for_course(course, _minimal_config()) is None


# ---- Bug B: env check caching + robust first probe --------------------------------------

@pytest.fixture(autouse=True)
def _reset_env_cache():
    """Each test starts with a cold env cache (the cache is process-global)."""
    rc._ENV_CHECK_OK = None
    yield
    rc._ENV_CHECK_OK = None


class _FakeProc:
    def __init__(self, returncode=0, stdout="OK\n", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_env_check_caches_success(monkeypatch):
    calls = {"n": 0}

    def fake_run(*a, **k):
        calls["n"] += 1
        return _FakeProc()

    monkeypatch.setattr(rc.subprocess, "run", fake_run)
    assert check_radiomics_env() is True
    assert check_radiomics_env() is True
    assert calls["n"] == 1, "a confirmed env must not be re-probed per course"


def test_env_check_retries_past_transient_timeout(monkeypatch):
    """A cold-start timeout on the first attempt must not be fatal: the retry succeeds."""
    calls = {"n": 0}

    def fake_run(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise subprocess.TimeoutExpired(cmd="conda", timeout=k.get("timeout", 180))
        return _FakeProc()

    monkeypatch.setattr(rc.subprocess, "run", fake_run)
    assert check_radiomics_env(retries=1) is True
    assert calls["n"] == 2
    # And it is now cached.
    assert check_radiomics_env() is True
    assert calls["n"] == 2


def test_env_check_does_not_cache_failure_then_recovers(monkeypatch):
    """A genuinely-failing probe returns False and is NOT cached, so a later healthy
    probe can still confirm the env (no permanent poisoning)."""
    state = {"ok": False}

    def fake_run(*a, **k):
        return _FakeProc() if state["ok"] else _FakeProc(returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(rc.subprocess, "run", fake_run)
    assert check_radiomics_env(retries=0) is False
    assert rc._ENV_CHECK_OK is None
    state["ok"] = True
    assert check_radiomics_env(retries=0) is True
    assert rc._ENV_CHECK_OK is True


def test_env_check_timeout_is_configurable_and_classified(monkeypatch):
    """An exhausted probe timeout must remain distinct from extraction failure."""
    seen = []

    def fake_run(command, **kwargs):
        seen.append((command, kwargs["timeout"]))
        raise subprocess.TimeoutExpired(cmd=command, timeout=kwargs["timeout"])

    monkeypatch.setenv("RTPIPELINE_RADIOMICS_ENV_PROBE_TIMEOUT", "321")
    monkeypatch.setattr(rc.subprocess, "run", fake_run)

    with pytest.raises(rc.RadiomicsEnvironmentProbeTimeout) as caught:
        check_radiomics_env(retries=1)

    error = caught.value
    assert error.code == "RADIOMICS_ENV_PROBE_TIMEOUT"
    assert error.timeout == 321
    assert error.attempts == 2
    assert len(seen) == 2
    assert all(timeout == 321 for _command, timeout in seen)
    assert "conda" in str(error) or "micromamba" in str(error) or "mamba" in str(error)
    assert "timeout=321s" in str(error)
    assert "attempts=2" in str(error)


def test_env_check_default_probe_budget_reaches_subprocess(monkeypatch):
    seen = []

    def fake_run(command, **kwargs):
        seen.append(kwargs["timeout"])
        return _FakeProc()

    monkeypatch.delenv("RTPIPELINE_RADIOMICS_ENV_PROBE_TIMEOUT", raising=False)
    monkeypatch.setattr(rc.subprocess, "run", fake_run)

    assert check_radiomics_env(retries=0) is True
    assert seen == [180]


def test_batch_environment_unavailable_fails_and_invalidates_output(monkeypatch, tmp_path):
    output_path = tmp_path / "radiomics_ct.xlsx"
    output_path.write_text("stale", encoding="utf-8")
    monkeypatch.setattr(rc, "check_radiomics_env", lambda **_kwargs: False)

    with pytest.raises(rc.RadiomicsCourseExtractionError, match="environment.*unavailable"):
        rc.process_radiomics_batch(
            [
                {
                    "image_path": str(tmp_path / "image.nrrd"),
                    "mask_path": str(tmp_path / "mask.nrrd"),
                    "roi_name": "PTV",
                    "metadata": {
                        "segmentation_source": "Manual",
                        "roi_original_name": "PTV",
                    },
                }
            ],
            output_path,
            enable_heartbeat=False,
        )

    assert not output_path.exists()


def test_radiomics_cli_probe_timeout_reaches_checker_and_unavailable_env_fails(
    monkeypatch, tmp_path
):
    from rtpipeline import radiomics

    seen = []
    monkeypatch.setattr(radiomics, "_have_pyradiomics", lambda: False)
    monkeypatch.setattr(
        rc,
        "check_radiomics_env",
        lambda *, timeout=None, **_kwargs: seen.append(timeout) or False,
    )

    result = cli.main(
        [
            "--dicom-root", str(tmp_path / "dicom"),
            "--outdir", str(tmp_path / "out"),
            "--logs", str(tmp_path / "logs"),
            "--stage", "radiomics",
            "--radiomics-env-probe-timeout", "321",
            "--no-metadata",
        ]
    )

    assert result == 1
    assert seen == [321]


def test_timeout_then_distinct_probe_failure_is_not_relabelled(monkeypatch, tmp_path):
    from rtpipeline import radiomics

    calls = {"count": 0}

    def fake_run(command, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise subprocess.TimeoutExpired(cmd=command, timeout=kwargs["timeout"])
        return _FakeProc(returncode=1, stdout="", stderr="missing radiomics")

    monkeypatch.setattr(radiomics, "_have_pyradiomics", lambda: False)
    monkeypatch.setattr(rc.subprocess, "run", fake_run)

    result = cli.main(
        [
            "--dicom-root", str(tmp_path / "dicom"),
            "--outdir", str(tmp_path / "out"),
            "--logs", str(tmp_path / "logs"),
            "--stage", "radiomics",
            "--no-metadata",
        ]
    )

    assert result == 1
    assert calls["count"] == 2


def test_env_probe_timeout_code_survives_cli_boundary(monkeypatch, capsys):
    error = rc.RadiomicsEnvironmentProbeTimeout(["conda", "run", "probe"], 321, 2)
    monkeypatch.setattr(cli, "main", lambda _argv=None: (_ for _ in ()).throw(error))

    assert cli.console_main([]) == 3
    emitted = json.loads(capsys.readouterr().err)
    assert emitted == {"code": error.code, "message": str(error)}


def test_env_probe_timeout_survives_robustness_subcommand(monkeypatch, tmp_path, capsys):
    from rtpipeline import radiomics_robustness

    error = rc.RadiomicsEnvironmentProbeTimeout(["conda", "run", "probe"], 321, 2)
    monkeypatch.setattr(
        radiomics_robustness,
        "robustness_for_course",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("radiomics_robustness:\n  enabled: true\n", encoding="utf-8")

    assert cli.console_main(
        [
            "radiomics-robustness",
            "--course-dir",
            str(tmp_path / "course"),
            "--config",
            str(config_path),
            "--output",
            str(tmp_path / "result.parquet"),
        ]
    ) == 3
    emitted = json.loads(capsys.readouterr().err)
    assert emitted == {"code": error.code, "message": str(error)}


def test_robustness_subcommand_accepts_env_probe_timeout(monkeypatch, tmp_path):
    from rtpipeline import radiomics_robustness

    seen = []

    def fake_robustness(config, _rob_config, _course_dir, *, output_path):
        seen.append(config.radiomics_env_probe_timeout)
        return output_path

    monkeypatch.setattr(radiomics_robustness, "robustness_for_course", fake_robustness)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("radiomics_robustness:\n  enabled: true\n", encoding="utf-8")

    assert cli.console_main(
        [
            "radiomics-robustness",
            "--course-dir", str(tmp_path / "course"),
            "--config", str(config_path),
            "--output", str(tmp_path / "result.parquet"),
            "--radiomics-env-probe-timeout", "654",
        ]
    ) == 0
    assert seen == [654]
