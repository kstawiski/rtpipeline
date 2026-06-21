"""Regression tests for the P5 all-series MATERIALIZATION allow-list.

Pins the contract of ``materialize_patient_series_from_inventory`` (rtpipeline/inventory.py):
an optional ``all_series_materialize_classes`` allow-list controls which classified series are
byte-copied. The failure mode is either over-copying (the daily-CBCT NFS bottleneck this feature
removes) or — worse — skip-materializing a class that segmentation / radiomics / PET-SUV needs.

These tests are DB-free: the SQLite-backed helpers (``enumerate_patient_series``,
``build_patient_series_manifest_rows``) and the byte-copy (``_copy_instances``) are monkeypatched,
so the test exercises only the allow-list decision logic.

Contract (matches the sibling all_series_segment_classes; see test_all_series_scope.py):
  * None      => materialize every non-excluded series (legacy).
  * []         => allow-list with nothing allowed (materialize none) — distinct from None.
  * [classes]  => materialize only those classes.
  * Fail-closed: the EFFECTIVE segmentation scope is always unioned in, so a class that will be
    segmented is never skip-materialized.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from rtpipeline import inventory


CLASSES = ["planning_ct", "petct_ct", "pt", "cbct", "exclude"]


def _cfg(materialize=None, segment=None, do_segment_all_series=True):
    return SimpleNamespace(
        inventory_db_path="/dev/null/fake.sqlite",
        inventory_scan_run_id=1,
        output_root="/unused",
        all_series_materialize_classes=materialize,
        all_series_segment_classes=segment,
        do_segment_all_series=do_segment_all_series,
    )


@pytest.fixture
def patched(monkeypatch, tmp_path):
    """Monkeypatch the DB + copy helpers; record which output_dirs were byte-copied."""
    copied: list[str] = []

    def fake_rows(db_path, patient_id, *, course_dirs, config=None):
        rows = []
        for cls in CLASSES:
            uid = f"uid-{cls}"
            out = "" if cls == "exclude" else str(tmp_path / cls / uid)
            rows.append(
                {
                    "patient_id": patient_id,
                    "series_uid": uid,
                    "image_class": cls,
                    "modality": "CT" if cls != "pt" else "PT",
                    "output_dir": out,
                    "status": "excluded" if cls == "exclude" else "classified",
                }
            )
        return rows

    def fake_enumerate(db_path, patient_id):
        return [
            SimpleNamespace(series_uid=f"uid-{cls}", instances=[object()], modality="CT")
            for cls in CLASSES
            if cls != "exclude"
        ]

    def fake_copy(instances, output_dir, modality, use_hardlinks=False):
        from pathlib import Path

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / f"{(modality or 'DICOM')}_00001.dcm").write_text("x")
        copied.append(str(output_dir))
        return False  # nothing missing

    monkeypatch.setattr(inventory, "build_patient_series_manifest_rows", fake_rows)
    monkeypatch.setattr(inventory, "enumerate_patient_series", fake_enumerate)
    monkeypatch.setattr(inventory, "_copy_instances", fake_copy)
    return SimpleNamespace(copied=copied, tmp_path=tmp_path)


def _materialize(cfg, patched):
    root = patched.tmp_path / "out" / "all_series"
    mpath = inventory.materialize_patient_series_from_inventory(
        cfg, "PID1", patient_series_root=root
    )
    manifest = json.loads(mpath.read_text())
    return {r["image_class"]: r["status"] for r in manifest["series"]}, set(patched.copied)


def _copied_classes(copied_dirs):
    return {d.rstrip("/").split("/")[-2] for d in copied_dirs}  # .../<class>/<uid>


def test_none_materializes_all_non_excluded(patched):
    statuses, copied = _materialize(_cfg(materialize=None), patched)
    assert statuses["exclude"] == "excluded"
    assert all(statuses[c] == "materialized" for c in ("planning_ct", "petct_ct", "pt", "cbct"))
    assert _copied_classes(copied) == {"planning_ct", "petct_ct", "pt", "cbct"}


def test_allowlist_skips_out_of_scope(patched):
    # No segmentation scope unioned (segment=None, do_segment_all_series=False).
    statuses, copied = _materialize(
        _cfg(materialize=["planning_ct", "pt"], segment=None, do_segment_all_series=False), patched
    )
    assert statuses["planning_ct"] == "materialized"
    assert statuses["pt"] == "materialized"
    assert statuses["cbct"] == "materialize_skipped_out_of_scope"
    assert statuses["petct_ct"] == "materialize_skipped_out_of_scope"
    assert _copied_classes(copied) == {"planning_ct", "pt"}


def test_empty_list_materializes_none(patched):
    # [] is an allow-list with nothing allowed; with no segmentation it copies nothing (NOT legacy-all).
    statuses, copied = _materialize(
        _cfg(materialize=[], segment=None, do_segment_all_series=False), patched
    )
    assert all(
        statuses[c] == "materialize_skipped_out_of_scope"
        for c in ("planning_ct", "petct_ct", "pt", "cbct")
    )
    assert copied == set()


def test_explicit_segment_scope_is_unioned_in(patched):
    # Narrow materialize list, but petct_ct is in the segment scope => must still be materialized.
    statuses, copied = _materialize(
        _cfg(materialize=["pt"], segment=["planning_ct", "petct_ct"]), patched
    )
    assert statuses["planning_ct"] == "materialized"  # unioned from segment scope
    assert statuses["petct_ct"] == "materialized"  # unioned from segment scope
    assert statuses["pt"] == "materialized"  # explicit
    assert statuses["cbct"] == "materialize_skipped_out_of_scope"
    assert _copied_classes(copied) == {"planning_ct", "petct_ct", "pt"}


def test_empty_list_still_protects_segment_scope(patched):
    # [] + explicit segment scope => segmented classes are still materialized (fail-closed).
    statuses, copied = _materialize(_cfg(materialize=[], segment=["planning_ct"]), patched)
    assert statuses["planning_ct"] == "materialized"
    assert statuses["cbct"] == "materialize_skipped_out_of_scope"
    assert _copied_classes(copied) == {"planning_ct"}


def test_empty_segment_scope_unions_nothing(patched):
    # segment=[] means "segment nothing" (mirrors _select_all_series_rows `if allowed is not None`).
    # Even with do_segment_all_series=True, the union must add NOTHING — only the explicit list copies.
    statuses, copied = _materialize(
        _cfg(materialize=["pt"], segment=[], do_segment_all_series=True), patched
    )
    assert statuses["pt"] == "materialized"
    assert all(
        statuses[c] == "materialize_skipped_out_of_scope"
        for c in ("planning_ct", "petct_ct", "cbct")
    )
    assert _copied_classes(copied) == {"pt"}


def test_segment_everything_scope_materializes_all_segmentable(patched):
    # segment=None + do_segment_all_series=True => legacy "segment everything": every TS-eligible class
    # must be materialized even with a narrow list. 'pt' (ts_task none) stays skipped.
    statuses, copied = _materialize(
        _cfg(materialize=["planning_ct"], segment=None, do_segment_all_series=True), patched
    )
    # planning_ct, petct_ct, cbct all carry a TS 'total' task => materialized
    assert statuses["planning_ct"] == "materialized"
    assert statuses["petct_ct"] == "materialized"
    assert statuses["cbct"] == "materialized"
    # pt has ts_task 'none' and is not in the list => skipped
    assert statuses["pt"] == "materialize_skipped_out_of_scope"
