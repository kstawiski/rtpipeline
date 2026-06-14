"""Regression tests for two radiomics_conda reliability bugs surfaced by the P5 run.

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
from rtpipeline.radiomics_conda import (
    RadiomicsCheckpoint,
    _jsonify_nested_columns,
    _roi_instance_key,
    check_radiomics_env,
    process_radiomics_batch,
)


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
    assert resumed.is_completed(_roi_instance_key({"roi_name": "CTV"}))
    assert resumed.is_completed(_roi_instance_key({"roi_name": "BODY"}))
    assert resumed.is_completed(_roi_instance_key({"roi_name": "BLADDER"}))
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
    """REGRESSION (the blocking issue): a course that resumes with a pre-seeded
    checkpoint must write EVERY completed ROI to radiomics_ct.xlsx — not just the
    ROIs processed in the resumed run. Under the pre-fix code the workbook contained
    only the newly-processed subset (silent partial-data-loss feeding the cohort).
    """
    monkeypatch.setattr(rc, "check_radiomics_env", lambda *a, **k: True)
    # Stub the extractor so only the NOT-yet-completed ROI is "computed" this run.
    monkeypatch.setattr(
        rc, "extract_radiomics_batch_with_conda",
        lambda tasks, params_file=None: [
            {"__status__": "success", "original_firstorder_Mean": float(i + 100)}
            for i, _t in enumerate(tasks)
        ],
    )

    checkpoint_path = tmp_path / "metadata" / "radiomics_ct_checkpoint.parquet"
    output_path = tmp_path / "radiomics_ct.xlsx"

    # Simulate run 1 having completed CTV and BODY (incl. the nested diagnostics).
    cp = RadiomicsCheckpoint(checkpoint_path, buffer_size=1)
    cp.add_result(_feature_record("CTV"))
    cp.add_result(_feature_record("BODY"))
    cp.flush()

    # Run 2 resumes: the task set lists all three; CTV/BODY are already done.
    tasks = [
        {"image_path": "i", "mask_path": "m", "roi_name": "CTV", "cleanup": False, "metadata": {}},
        {"image_path": "i", "mask_path": "m", "roi_name": "BODY", "cleanup": False, "metadata": {}},
        {"image_path": "i", "mask_path": "m", "roi_name": "BLADDER", "cleanup": False, "metadata": {}},
    ]
    result = process_radiomics_batch(
        tasks, str(output_path), sequential=True, max_workers=1,
        checkpoint_path=checkpoint_path, enable_heartbeat=False,
    )
    assert result is not None
    written = set(pd.read_excel(output_path)["roi_name"].tolist())
    assert written == {"CTV", "BODY", "BLADDER"}, (
        f"resumed workbook must contain all completed ROIs; got {written}"
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
        {"image_path": "i", "mask_path": "m", "roi_name": "CTV", "cleanup": False, "metadata": {}},
        {"image_path": "i", "mask_path": "m", "roi_name": "BODY", "cleanup": False, "metadata": {}},
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
    cp.add_result({"roi_name": "liver", "series_uid": "1.2.A", "modality": "MR", "original_firstorder_Mean": 1.0})
    cp.add_result({"roi_name": "liver", "series_uid": "1.2.B", "modality": "MR", "original_firstorder_Mean": 2.0})
    cp.flush()
    assert len(cp.load_records()) == 2, "distinct (roi_name, series_uid) must not be deduped to one"

    # Resume: both liver instances already checkpointed → both filtered → workbook rebuilt
    # from the checkpoint must contain BOTH series.
    tasks = [
        {"image_path": "i", "mask_path": "m", "roi_name": "liver", "cleanup": False,
         "metadata": {"modality": "MR", "series_uid": "1.2.A"}},
        {"image_path": "i", "mask_path": "m", "roi_name": "liver", "cleanup": False,
         "metadata": {"modality": "MR", "series_uid": "1.2.B"}},
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
    (nifti_dir / "CT_SERIES.metadata.json").write_text(
        json.dumps({"series_instance_uid": "1.2.3.CT"}),
        encoding="utf-8",
    )
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[1:4, 1:4, 1:4] = 1
    _write_nifti(seg_dir / "total--liver.nii.gz", mask)
    _write_nifti(seg_dir / "total--liver_cropped.nii.gz", mask)
    _write_nifti(seg_dir / "total--skip_me.nii.gz", mask)

    seen = {}

    def fake_batch(tasks, params_file=None):
        seen["tasks"] = tasks
        return [
            {"__status__": "success", "original_firstorder_Mean": float(i + 1)}
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
    assert tasks[0]["metadata"]["series_uid"] == "1.2.3.CT"
    assert tasks[0]["metadata"]["segmentation_source"] == "AutoTS_total_nifti_fallback"
    assert tasks[0]["metadata"]["roi_original_name"] == "liver"

    df = pd.read_excel(out)
    assert df["segmentation_source"].tolist() == ["AutoTS_total_nifti_fallback"]
    assert df["series_uid"].tolist() == ["1.2.3.CT"]
    assert df["roi_name"].tolist() == ["liver"]
    assert df["original_firstorder_Mean"].tolist() == [1]


def test_ct_totalseg_nifti_fallback_not_used_when_rs_selected(tmp_path, monkeypatch):
    """A selected RS path keeps the RTSTRUCT branch; fallback is only for no usable RS."""
    course = tmp_path / "P1" / "2024-12"
    (course / "DICOM" / "CT").mkdir(parents=True)
    (course / "RS_auto.dcm").write_bytes(b"not read by this test")

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


def test_env_check_uses_generous_default_timeout():
    """The default timeout must be well above the 60 s that timed out under load."""
    import inspect
    sig = inspect.signature(check_radiomics_env)
    assert sig.parameters["timeout"].default >= 120
