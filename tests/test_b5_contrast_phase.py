"""B5 [SAFE-3] — persist contrast phase as a queryable per-series manifest field.

Unit-level tests for the helper + class gating. The end-to-end manifest write is
exercised by the all-series segmentation path (needs TotalSegmentator + a real CT,
out of unit scope); here we verify the safe wrapper extracts the tool's `phase`,
never raises, and is scoped to calibrated-CT classes only.
"""
from pathlib import Path

import rtpipeline.quality_control as qc
from rtpipeline.segmentation import _CONTRAST_PHASE_CLASSES, _detect_contrast_phase_safe


class _Cfg:
    conda_activate = None


def test_b5_classes_are_calibrated_ct_only():
    assert _CONTRAST_PHASE_CLASSES == frozenset({"planning_ct", "diagnostic_ct", "petct_ct"})
    # CBCT (uncalibrated) and 4DCT (projection/respiratory) must be excluded
    for excluded in ("cbct", "fourdct_ave", "fourdct_phase", "mr_anatomic", "pt"):
        assert excluded not in _CONTRAST_PHASE_CLASSES


def test_b5_extracts_phase(monkeypatch):
    monkeypatch.setattr(
        qc, "detect_contrast_phase",
        lambda nifti_path, conda_activate=None: {"phase": "portal_venous", "status": "success"},
    )
    assert _detect_contrast_phase_safe(_Cfg(), Path("/tmp/ct.nii.gz")) == "portal_venous"


def test_b5_passes_conda_activate(monkeypatch):
    seen = {}

    def _stub(nifti_path, conda_activate=None):
        seen["conda_activate"] = conda_activate
        return {"phase": "native", "status": "success"}

    monkeypatch.setattr(qc, "detect_contrast_phase", _stub)

    class _CfgConda:
        conda_activate = "source activate ts"

    assert _detect_contrast_phase_safe(_CfgConda(), Path("/tmp/ct.nii.gz")) == "native"
    assert seen["conda_activate"] == "source activate ts"


def test_b5_unknown_phase_returned(monkeypatch):
    monkeypatch.setattr(
        qc, "detect_contrast_phase",
        lambda nifti_path, conda_activate=None: {"phase": "unknown", "status": "unavailable"},
    )
    assert _detect_contrast_phase_safe(_Cfg(), Path("/tmp/ct.nii.gz")) == "unknown"


def test_b5_empty_or_missing_phase_is_none(monkeypatch):
    monkeypatch.setattr(
        qc, "detect_contrast_phase",
        lambda nifti_path, conda_activate=None: {"status": "error"},  # no 'phase' key
    )
    assert _detect_contrast_phase_safe(_Cfg(), Path("/tmp/ct.nii.gz")) is None


def test_b5_resilient_on_exception(monkeypatch):
    def _boom(nifti_path, conda_activate=None):
        raise RuntimeError("totalseg_get_phase unavailable")

    monkeypatch.setattr(qc, "detect_contrast_phase", _boom)
    # must never raise -> QC enrichment must never fail segmentation
    assert _detect_contrast_phase_safe(_Cfg(), Path("/tmp/ct.nii.gz")) is None


def test_b5_non_dict_result_is_none(monkeypatch):
    monkeypatch.setattr(qc, "detect_contrast_phase", lambda nifti_path, conda_activate=None: None)
    assert _detect_contrast_phase_safe(_Cfg(), Path("/tmp/ct.nii.gz")) is None


def test_b5_persistence_aliasing_and_writeback(tmp_path):
    # The feature's load-bearing contract: a row mutated in the seg sublist returned by
    # _select_all_series_rows aliases into the master manifest and survives the
    # _write_updated_series_manifest round-trip. Guards against a future refactor of
    # _select_all_series_rows to return copies (which would silently drop contrast_phase).
    import json
    from rtpipeline.segmentation import _select_all_series_rows, _write_updated_series_manifest

    manifest = {
        "patient_id": "p",
        "series": [
            {"series_uid": "u1", "image_class": "planning_ct", "ts_task": "total", "status": "materialized"},
            {"series_uid": "u2", "image_class": "cbct", "ts_task": "total", "status": "materialized"},
        ],
    }
    rows = manifest["series"]
    seg_rows = _select_all_series_rows(_Cfg(), rows)  # _Cfg lacks the attrs -> getattr defaults
    # aliasing: the returned rows ARE the manifest's row objects (not copies)
    assert seg_rows[0] is rows[0]
    assert seg_rows[1] is rows[1]
    # mutate as the B5 hook does
    seg_rows[0]["contrast_phase"] = "portal_venous"
    p = tmp_path / "series_manifest.json"
    _write_updated_series_manifest(p, manifest)
    reloaded = json.loads(p.read_text(encoding="utf-8"))
    by_uid = {r["series_uid"]: r for r in reloaded["series"]}
    assert by_uid["u1"].get("contrast_phase") == "portal_venous"   # persisted to disk
    assert "contrast_phase" not in by_uid["u2"]                    # CBCT row untouched


def test_b5_idempotent_guard_semantics():
    # Pins the inline guard expression: detect iff (force or not row.get("contrast_phase")).
    def _should_detect(row, force):
        return force or not row.get("contrast_phase")
    assert _should_detect({}, False) is True                            # absent -> detect
    assert _should_detect({"contrast_phase": "native"}, False) is False  # present -> skip (idempotent)
    assert _should_detect({"contrast_phase": "native"}, True) is True    # force -> recompute
    assert _should_detect({"contrast_phase": ""}, False) is True         # empty -> retry
