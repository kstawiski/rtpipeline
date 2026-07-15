"""Regression tests for all-series segmentation scope selection.

Pins the contract of ``_select_all_series_rows`` / ``_limit_fourdct_to_representative``
(rtpipeline/segmentation.py): an optional image_class allow-list plus a
one-representative-4DCT reduction. The failure mode is segmenting the wrong set of
series (either silently dropping high-value images, or grinding through every daily
CBCT), so these tests pin exactly which rows are selected for each config.
"""
from __future__ import annotations

from types import SimpleNamespace

from rtpipeline.segmentation import _limit_fourdct_to_representative, _select_all_series_rows


def _row(cls: str, uid: str) -> dict:
    return {"image_class": cls, "series_uid": uid, "ts_task": "total", "status": "materialized"}


def _cfg(classes=None, single_4dct=False) -> SimpleNamespace:
    return SimpleNamespace(
        all_series_segment_classes=classes,
        all_series_fourdct_single_representative=single_4dct,
    )


def _classes(rows):
    return [r["image_class"] for r in rows]


def test_none_allowlist_keeps_all():
    rows = [_row("planning_ct", "a"), _row("cbct", "b"), _row("petct_ct", "c")]
    assert _select_all_series_rows(_cfg(None), rows) == rows


def test_allowlist_filters_classes():
    rows = [_row("planning_ct", "a"), _row("cbct", "b"), _row("petct_ct", "c"), _row("cbct", "d")]
    sel = _select_all_series_rows(_cfg(["planning_ct", "petct_ct"]), rows)
    assert _classes(sel) == ["planning_ct", "petct_ct"]


def test_non_dict_rows_dropped():
    rows = [_row("planning_ct", "a"), "garbage", None, _row("petct_ct", "c")]
    sel = _select_all_series_rows(_cfg(None), rows)
    assert _classes(sel) == ["planning_ct", "petct_ct"]


def test_4dct_representative_prefers_ave_drops_phases():
    rows = [
        _row("planning_ct", "p"),
        _row("fourdct_phase", "ph1"),
        _row("fourdct_ave", "ave"),
        _row("fourdct_phase", "ph2"),
    ]
    sel = _limit_fourdct_to_representative(rows)
    # ave kept, all phases dropped, planning untouched
    assert _classes(sel) == ["planning_ct", "fourdct_ave"]


def test_4dct_multi_ave_keeps_only_first():
    """Truly single representative: with multiple ave reconstructions, keep only the first."""
    rows = [
        _row("planning_ct", "p"),
        _row("fourdct_ave", "ave1"),
        _row("fourdct_phase", "ph1"),
        _row("fourdct_ave", "ave2"),
    ]
    sel = _limit_fourdct_to_representative(rows)
    fourdct = [r for r in sel if r["image_class"].startswith("fourdct")]
    assert len(fourdct) == 1 and fourdct[0]["series_uid"] == "ave1"
    assert _classes(sel) == ["planning_ct", "fourdct_ave"]


def test_empty_allowlist_selects_nothing():
    """`[]` (distinct from None) => segment zero series; None => segment everything."""
    rows = [_row("planning_ct", "a"), _row("cbct", "b")]
    assert _select_all_series_rows(_cfg([]), rows) == []


def test_flag_false_keeps_all_phases():
    """With the representative flag off, every 4DCT phase is retained (legacy)."""
    rows = [_row("fourdct_phase", "ph1"), _row("fourdct_phase", "ph2"), _row("fourdct_phase", "ph3")]
    sel = _select_all_series_rows(_cfg(["fourdct_phase"], single_4dct=False), rows)
    assert len(sel) == 3


def test_flag_noop_when_4dct_excluded_from_allowlist():
    """Representative flag has no effect when 4DCT classes are filtered out by the allow-list."""
    rows = [_row("planning_ct", "p"), _row("fourdct_phase", "ph1"), _row("fourdct_phase", "ph2")]
    sel = _select_all_series_rows(_cfg(["planning_ct"], single_4dct=True), rows)
    assert _classes(sel) == ["planning_ct"]


def test_excluded_rows_are_preserved_in_source_list():
    """Scoped-out rows must remain in the original `rows` (manifest write-back persists them)."""
    rows = [_row("planning_ct", "a"), _row("cbct", "b"), _row("fourdct_phase", "ph1")]
    sel = _select_all_series_rows(_cfg(["planning_ct"], single_4dct=True), rows)
    # selection excludes cbct + 4DCT, but they are untouched in the source manifest list
    assert _classes(sel) == ["planning_ct"]
    assert _classes(rows) == ["planning_ct", "cbct", "fourdct_phase"]
    assert all(r["status"] == "materialized" for r in rows)


def test_4dct_representative_keeps_first_phase_when_no_ave():
    rows = [
        _row("planning_ct", "p"),
        _row("fourdct_phase", "ph1"),
        _row("fourdct_phase", "ph2"),
        _row("fourdct_phase", "ph3"),
    ]
    sel = _limit_fourdct_to_representative(rows)
    kept = [r for r in sel if r["image_class"] == "fourdct_phase"]
    assert len(kept) == 1 and kept[0]["series_uid"] == "ph1"
    assert _classes(sel) == ["planning_ct", "fourdct_phase"]


def test_mixed_modality_scope_keeps_one_4dct_representative():
    """Keep planning/PET CT and one 4DCT phase while excluding CBCT."""
    rows = (
        [_row("planning_ct", f"pl{i}") for i in range(2)]
        + [_row("petct_ct", f"pet{i}") for i in range(3)]
        + [_row("fourdct_phase", f"ph{i}") for i in range(4)]
        + [_row("cbct", f"cb{i}") for i in range(5)]
    )
    cfg = _cfg(["planning_ct", "petct_ct", "fourdct_ave", "fourdct_phase"], single_4dct=True)
    sel = _select_all_series_rows(cfg, rows)
    counts = {c: _classes(sel).count(c) for c in set(_classes(sel))}
    assert counts == {"planning_ct": 2, "petct_ct": 3, "fourdct_phase": 1}
    assert "cbct" not in counts
    assert len(sel) == 6


def test_rows_are_same_objects_not_copies():
    """Selected rows must be the SAME dicts as input (loop mutates them; manifest write-back relies on it)."""
    rows = [_row("planning_ct", "a"), _row("cbct", "b")]
    sel = _select_all_series_rows(_cfg(["planning_ct"]), rows)
    assert sel[0] is rows[0]
