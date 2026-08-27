"""A symlinked cohort subset must be discovered, not silently skipped.

Scoping a run to a cohort is routinely done by building a directory of symlinks
to the selected patient directories, because the alternative is duplicating
terabytes of DICOM. ``os.walk`` does not descend into symlinked directories by
default, while the ``iterdir``-based fast path in ct.py follows them. The two
discovery paths therefore disagreed: a symlinked cohort yielded zero files and
the run completed "successfully" with an empty manifest.

Observed on a 92-patient Kopernik bladder cohort: os.walk saw 0 files where
os.walk(followlinks=True) saw 340,607.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rtpipeline.utils import (
    FOLLOW_INPUT_SYMLINKS_ENV,
    _scoped_walk,
    _walk_following_symlinks,
    list_files,
)


@pytest.fixture(autouse=True)
def _opt_in(monkeypatch):
    """Every test here describes the opted-in cohort-subset mode."""
    monkeypatch.setenv(FOLLOW_INPUT_SYMLINKS_ENV, "1")


def _cohort(tmp_path: Path, n_patients: int = 3, n_files: int = 4):
    source = tmp_path / "source"
    for p in range(n_patients):
        series = source / f"PT{p:03d}" / f"1.2.3.{p}"
        series.mkdir(parents=True)
        for f in range(n_files):
            (series / f"CT_{f}.dcm").write_bytes(b"x")
    cohort = tmp_path / "cohort"
    cohort.mkdir()
    for p in range(n_patients):
        (cohort / f"PT{p:03d}").symlink_to(source / f"PT{p:03d}", target_is_directory=True)
    return source, cohort, n_patients * n_files


def test_plain_os_walk_would_have_missed_everything(tmp_path):
    """Documents the defect this guards against."""
    _, cohort, _ = _cohort(tmp_path)
    assert sum(len(f) for _, _, f in os.walk(cohort)) == 0


def test_scoped_walk_discovers_a_symlinked_cohort(tmp_path):
    _, cohort, expected = _cohort(tmp_path)
    assert sum(len(f) for _, _, f in _scoped_walk(cohort)) == expected


def test_scoped_walk_matches_the_real_directory_tree(tmp_path):
    source, cohort, _ = _cohort(tmp_path)
    via_source = sum(len(f) for _, _, f in _scoped_walk(source))
    via_links = sum(len(f) for _, _, f in _scoped_walk(cohort))
    assert via_links == via_source


def test_list_files_discovers_a_symlinked_cohort(tmp_path):
    _, cohort, expected = _cohort(tmp_path)
    assert len(list_files(cohort, ["*.dcm"])) == expected


def test_a_symlink_cycle_terminates(tmp_path):
    """followlinks=True alone loops forever here."""
    root = tmp_path / "loop"
    (root / "a").mkdir(parents=True)
    (root / "a" / "f.dcm").write_bytes(b"x")
    (root / "a" / "back").symlink_to(root, target_is_directory=True)

    files = sum(len(f) for _, _, f in _walk_following_symlinks(root))
    assert files == 1


def test_each_real_directory_is_visited_once(tmp_path):
    """Two links to the same target must not double-count its files."""
    source = tmp_path / "src"
    source.mkdir()
    (source / "f.dcm").write_bytes(b"x")
    cohort = tmp_path / "cohort"
    cohort.mkdir()
    (cohort / "first").symlink_to(source, target_is_directory=True)
    (cohort / "second").symlink_to(source, target_is_directory=True)

    assert sum(len(f) for _, _, f in _walk_following_symlinks(cohort)) == 1


def test_scoped_walk_still_scopes_to_requested_patients(tmp_path):
    _, cohort, _ = _cohort(tmp_path, n_patients=3, n_files=4)
    scoped = sum(len(f) for _, _, f in _scoped_walk(cohort, ["PT000"]))
    assert scoped == 4


def test_empty_scope_walks_nothing(tmp_path):
    _, cohort, _ = _cohort(tmp_path)
    assert sum(len(f) for _, _, f in _scoped_walk(cohort, [])) == 0


def test_traversal_via_a_path_component_is_still_rejected(tmp_path):
    """Loosening the containment check must not open a traversal hole."""
    from rtpipeline.utils import _resolve_scoped_dirs

    outside = tmp_path / "outside"
    (outside / "SECRET").mkdir(parents=True)
    root = tmp_path / "root"
    root.mkdir()

    for bad in ("..", "../outside", "/etc", "", ".", "a/b"):
        dirs, missing = _resolve_scoped_dirs(root, [bad])
        assert dirs == [], f"{bad!r} must not resolve to a directory"
        assert missing == [bad]


def test_a_curated_symlink_inside_root_is_accepted(tmp_path):
    from rtpipeline.utils import _resolve_scoped_dirs

    source = tmp_path / "elsewhere" / "PT000"
    source.mkdir(parents=True)
    root = tmp_path / "root"
    root.mkdir()
    (root / "PT000").symlink_to(source, target_is_directory=True)

    dirs, missing = _resolve_scoped_dirs(root, ["PT000"])
    assert missing == []
    assert len(dirs) == 1


def test_a_curated_centre_symlink_is_also_accepted(tmp_path):
    """A patient reached one level down through a curated centre link resolves.

    The earlier fix only exempted a symlink that was itself the patient dir, so
    a linked CENTRE holding real patient dirs was still rejected. Containment is
    now judged lexically, which treats both curation shapes alike.
    """
    from rtpipeline.utils import _resolve_scoped_dirs

    root = tmp_path / "root"
    root.mkdir()
    # a real directory that resolves outside root cannot exist as root/<pid>,
    # so the nearest equivalent is a symlinked PARENT holding the patient dir.
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "center" / "PT000").mkdir(parents=True)
    (root / "center").symlink_to(elsewhere / "center", target_is_directory=True)

    dirs, missing = _resolve_scoped_dirs(root, ["PT000"])
    # reached one level down through a curated center link: allowed, and scoped
    assert missing == []
    assert len(dirs) == 1


def test_default_refuses_to_leave_the_dicom_root(monkeypatch, tmp_path):
    """Without the opt-in, a link out of the root is not followed.

    This is the security default: a symlink inside the DICOM root pointing
    outside it is indistinguishable from a stray or hostile link, so discovery
    stays inside the root the operator named.
    """
    from rtpipeline.utils import _resolve_scoped_dirs

    monkeypatch.delenv(FOLLOW_INPUT_SYMLINKS_ENV, raising=False)
    source = tmp_path / "elsewhere" / "PT000"
    source.mkdir(parents=True)
    root = tmp_path / "root"
    root.mkdir()
    (root / "PT000").symlink_to(source, target_is_directory=True)

    dirs, missing = _resolve_scoped_dirs(root, ["PT000"])
    assert dirs == []
    assert missing == ["PT000"]


def test_default_walk_does_not_descend_a_symlinked_cohort(monkeypatch, tmp_path):
    monkeypatch.delenv(FOLLOW_INPUT_SYMLINKS_ENV, raising=False)
    _, cohort, _ = _cohort(tmp_path)
    assert sum(len(f) for _, _, f in _scoped_walk(cohort)) == 0
