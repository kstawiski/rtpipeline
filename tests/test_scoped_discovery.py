"""Regression tests for cohort-scoped organize-stage discovery helpers.

These pin the data-safety contract of ``_resolve_scoped_dirs`` / ``_scoped_walk``
/ ``_scoped_patient_dirs`` (rtpipeline/utils.py), which scope the organize stage's
RT/CT/series walks to a requested patient cohort instead of statting the whole
DICOM root. The failure mode of a bug here is *silent patient-data loss*, so the
key invariant under test is ``discovered ⊇ processed``: every directory the old
full ``os.walk`` would have surfaced for a requested patient must still be walked.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from rtpipeline.utils import _resolve_scoped_dirs, _scoped_patient_dirs, _scoped_walk


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")


def _walk_files(gen) -> list[str]:
    """File basenames yielded by a walk generator, sorted."""
    return sorted(f for _base, _dirs, files in gen for f in files)


def _build_tree(root: Path) -> None:
    _touch(root / "flat_pid" / "RP_flat.dcm")                 # flat: root/<pid>
    _touch(root / "CENTER_A" / "nested_pid" / "RS_a.dcm")     # nested: root/<center>/<pid>
    _touch(root / "CENTER_A" / "coll" / "RS_a2.dcm")          # cross-center collision...
    _touch(root / "CENTER_B" / "coll" / "RS_b.dcm")           # ...same pid under two centers
    _touch(root / "dup" / "RP_flat.dcm")                      # flat+nested duplicate...
    _touch(root / "CENTER_A" / "dup" / "RP_nested.dcm")       # ...same pid flat AND nested


def test_none_is_full_walk(tmp_path: Path) -> None:
    _build_tree(tmp_path)
    assert _walk_files(_scoped_walk(tmp_path, None)) == _walk_files(os.walk(tmp_path))


def test_empty_scopes_to_nothing(tmp_path: Path) -> None:
    _build_tree(tmp_path)
    assert _walk_files(_scoped_walk(tmp_path, [])) == []


def test_flat_scope_subset_of_root(tmp_path: Path) -> None:
    _build_tree(tmp_path)
    dirs, missing = _resolve_scoped_dirs(tmp_path, ["flat_pid"])
    assert missing == []
    assert [p.relative_to(tmp_path).as_posix() for p in dirs] == ["flat_pid"]
    assert _walk_files(_scoped_walk(tmp_path, ["flat_pid"])) == ["RP_flat.dcm"]


def test_nested_center_resolution(tmp_path: Path) -> None:
    _build_tree(tmp_path)
    dirs, missing = _resolve_scoped_dirs(tmp_path, ["nested_pid"])
    assert missing == []
    assert [p.relative_to(tmp_path).as_posix() for p in dirs] == ["CENTER_A/nested_pid"]


def test_cross_center_collision_walks_all(tmp_path: Path) -> None:
    _build_tree(tmp_path)
    dirs, missing = _resolve_scoped_dirs(tmp_path, ["coll"])
    assert missing == []
    assert {p.relative_to(tmp_path).as_posix() for p in dirs} == {"CENTER_A/coll", "CENTER_B/coll"}
    assert _walk_files(_scoped_walk(tmp_path, ["coll"])) == ["RS_a2.dcm", "RS_b.dcm"]


def test_flat_and_nested_duplicate_walks_both(tmp_path: Path) -> None:
    _build_tree(tmp_path)
    dirs, missing = _resolve_scoped_dirs(tmp_path, ["dup"])
    assert missing == []
    assert {p.relative_to(tmp_path).as_posix() for p in dirs} == {"dup", "CENTER_A/dup"}
    assert _walk_files(_scoped_walk(tmp_path, ["dup"])) == ["RP_flat.dcm", "RP_nested.dcm"]


def test_missing_id_falls_back_to_full_walk_and_warns(tmp_path: Path, caplog) -> None:
    _build_tree(tmp_path)
    dirs, missing = _resolve_scoped_dirs(tmp_path, ["ghost"])
    assert dirs == [] and missing == ["ghost"]
    with caplog.at_level(logging.WARNING, logger="rtpipeline.utils"):
        files = _walk_files(_scoped_walk(tmp_path, ["ghost"]))
    # fallback discovers everything (correctness over speed) and warns
    assert files == _walk_files(os.walk(tmp_path))
    assert any("could not locate" in r.getMessage() for r in caplog.records)


def test_unsafe_ids_rejected(tmp_path: Path) -> None:
    _build_tree(tmp_path)
    for bad in ["..", ".", "", "a/b", os.path.join("c", "d"), str(tmp_path)]:
        dirs, missing = _resolve_scoped_dirs(tmp_path, [bad])
        assert dirs == [] and missing == [bad], f"unsafe id not rejected: {bad!r}"


def test_symlink_escape_is_contained(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_root"
    _touch(outside / "secret.dcm")
    root = tmp_path / "root"
    _build_tree(root)
    try:
        os.symlink(outside, root / "escape")
    except OSError:
        return  # platform without symlink support
    dirs, missing = _resolve_scoped_dirs(root, ["escape"])
    assert dirs == [] and missing == ["escape"]
    # even the missing-id full-walk fallback must not descend the escaping symlink
    assert "secret.dcm" not in _walk_files(_scoped_walk(root, ["escape"]))


def test_scoped_patient_dirs_matches_resolver(tmp_path: Path) -> None:
    _build_tree(tmp_path)
    assert _scoped_patient_dirs(tmp_path, ["nested_pid"]) == [tmp_path / "CENTER_A" / "nested_pid"]
    # None -> every immediate child dir
    assert {p.name for p in _scoped_patient_dirs(tmp_path, None)} == {
        "flat_pid", "CENTER_A", "CENTER_B", "dup",
    }


def test_discovered_superset_of_processed(tmp_path: Path) -> None:
    """For every requested patient, the scoped walk surfaces at least the files the
    unscoped full walk would have surfaced for that patient (no silent drops)."""
    _build_tree(tmp_path)
    for pid, expect in [
        ("flat_pid", {"RP_flat.dcm"}),
        ("nested_pid", {"RS_a.dcm"}),
        ("coll", {"RS_a2.dcm", "RS_b.dcm"}),
        ("dup", {"RP_flat.dcm", "RP_nested.dcm"}),
    ]:
        got = set(_walk_files(_scoped_walk(tmp_path, [pid])))
        assert expect <= got, f"{pid}: scoped walk dropped data {expect - got}"
