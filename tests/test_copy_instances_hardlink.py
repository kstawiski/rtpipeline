"""Unit tests for the all-series materialization hardlink path (inventory._copy_instances).

Covers the full-cohort optimization: when use_hardlinks=True and source/dest share a
filesystem, os.link is used (shared inode, no byte-copy); when False, copy2 is used
(independent inode); on a cross-device os.link failure, it falls back to copy2.
"""
from __future__ import annotations

import os
from pathlib import Path

from rtpipeline.inventory import InventoryInstance, _copy_instances


def _mk_instance(src: Path, n: int) -> InventoryInstance:
    return InventoryInstance(
        sop_instance_uid=f"1.2.3.{n}",
        path=src,
        instance_number=n,
        image_type="ORIGINAL\\PRIMARY\\AXIAL",
        slice_thickness=1.0,
        file_id=n,
    )


def test_hardlink_used_when_enabled_same_fs(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"; src_dir.mkdir()
    out_dir = tmp_path / "out"; out_dir.mkdir(parents=True, exist_ok=True)
    srcs = []
    for i in range(1, 4):
        p = src_dir / f"i{i}.dcm"; p.write_bytes(b"DICOMDATA" * 100); srcs.append(p)
    instances = [_mk_instance(p, i) for i, p in enumerate(srcs, start=1)]

    missing = _copy_instances(instances, out_dir, "CT", use_hardlinks=True)
    assert missing is False
    outs = sorted(out_dir.glob("*.dcm"))
    assert len(outs) == 3
    # Hardlink => same inode as source and nlink >= 2 (no byte copy).
    for src, dst in zip(srcs, outs):
        assert os.stat(src).st_ino == os.stat(dst).st_ino
        assert os.stat(dst).st_nlink >= 2


def test_copy2_used_when_hardlinks_disabled(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"; src_dir.mkdir()
    out_dir = tmp_path / "out"; out_dir.mkdir(parents=True, exist_ok=True)
    src = src_dir / "a.dcm"; src.write_bytes(b"X" * 50)
    instances = [_mk_instance(src, 1)]

    missing = _copy_instances(instances, out_dir, "CT", use_hardlinks=False)
    assert missing is False
    outs = list(out_dir.glob("*.dcm"))
    assert len(outs) == 1
    # Plain copy => distinct inode, content preserved.
    assert os.stat(src).st_ino != os.stat(outs[0]).st_ino
    assert outs[0].read_bytes() == src.read_bytes()


def test_hardlink_falls_back_to_copy_on_oserror(tmp_path: Path, monkeypatch) -> None:
    src_dir = tmp_path / "src"; src_dir.mkdir()
    out_dir = tmp_path / "out"; out_dir.mkdir(parents=True, exist_ok=True)
    src = src_dir / "a.dcm"; src.write_bytes(b"Y" * 50)
    instances = [_mk_instance(src, 1)]

    def boom(_src, _dst):
        raise OSError(18, "Invalid cross-device link")  # EXDEV

    monkeypatch.setattr("rtpipeline.inventory.os.link", boom)
    missing = _copy_instances(instances, out_dir, "CT", use_hardlinks=True)
    assert missing is False
    outs = list(out_dir.glob("*.dcm"))
    assert len(outs) == 1
    # Fallback copy2 => distinct inode, content preserved.
    assert os.stat(src).st_ino != os.stat(outs[0]).st_ino
    assert outs[0].read_bytes() == src.read_bytes()


def test_missing_source_reported(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"; out_dir.mkdir(parents=True, exist_ok=True)
    instances = [_mk_instance(tmp_path / "does_not_exist.dcm", 1)]
    missing = _copy_instances(instances, out_dir, "CT", use_hardlinks=True)
    assert missing is True
    assert list(out_dir.glob("*.dcm")) == []
