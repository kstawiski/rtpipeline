"""RS_custom cache binding ignores mtimes and covers every input mask."""
import os
from pathlib import Path

import pydicom
import pytest
import SimpleITK as sitk

from rtpipeline import custom_structures_rtstruct as custom
from test_custom_structures_d16_auto_harvest import _publish_clean_course


@pytest.mark.parametrize("changed", ["config", "source", "ct", "second_mask"])
def test_same_mtime_mutation_rejects_custom_cache(tmp_path, monkeypatch, changed):
    item = _publish_clean_course(tmp_path, monkeypatch)
    assert not custom._is_rs_custom_stale(item.out, item.config, item.rs_manual, item.rs_auto)
    if changed == "config":
        path = item.config
        content = path.read_bytes() + b"\n# same-path configuration mutation\n"
    elif changed in {"source", "ct"}:
        path = item.rs_auto if changed == "source" else next(item.ct.glob("*.dcm"))
        stat = path.stat()
        ds = pydicom.dcmread(path)
        ds.SeriesDescription = "synthetic content mutation"
        ds.save_as(path, enforce_file_format=True)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert custom._is_rs_custom_stale(item.out, item.config, item.rs_manual, item.rs_auto)
        return
    else:
        for index in (1, 2):
            path = tmp_path / "Segmentation_TotalSegmentator" / f"unused{index}.nii.gz"
            path.parent.mkdir(exist_ok=True)
            sitk.WriteImage(sitk.Image([3, 3, 3], sitk.sitkUInt8), str(path))
        # Bind both masks, then mutate the second with its mtime preserved.
        item.out.unlink()
        custom._create_custom_structures_rtstruct_unlocked(tmp_path, item.config, item.rs_manual, item.rs_auto)
        assert not custom._is_rs_custom_stale(item.out, item.config, item.rs_manual, item.rs_auto)
        path = tmp_path / "Segmentation_TotalSegmentator/unused2.nii.gz"
        content = path.read_bytes() + b"\x00"
    stat = path.stat()
    path.write_bytes(content)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert custom._is_rs_custom_stale(item.out, item.config, item.rs_manual, item.rs_auto)


def test_main_publication_binding_does_not_depend_on_reader_runtime(tmp_path, monkeypatch):
    item = _publish_clean_course(tmp_path, monkeypatch)
    from rtpipeline import stage_completion
    monkeypatch.setattr(stage_completion, "environment_record", lambda _: {"reader_runtime": "numpy1"})
    assert not custom._is_rs_custom_stale(item.out, item.config, item.rs_manual, item.rs_auto)


def test_atomic_rtstruct_fsyncs_file_and_directory_then_reads_published_bytes(tmp_path, monkeypatch):
    import stat
    from rtpipeline import durable_progress
    target = tmp_path / "RS_custom.dcm"
    target.write_bytes(b"previous synthetic RTSTRUCT")
    steps = []
    original_fsync = os.fsync
    original_replace = os.replace
    original_hash = durable_progress.file_sha256
    def synced(fd):
        steps.append("directory_fsync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file_fsync")
        return original_fsync(fd)
    def replaced(source, destination):
        steps.append("replace")
        return original_replace(source, destination)
    def hashed(path):
        steps.append("published_readback" if path == target else "temporary_readback")
        return original_hash(path)
    monkeypatch.setattr(os, "fsync", synced)
    monkeypatch.setattr(os, "replace", replaced)
    monkeypatch.setattr(durable_progress, "file_sha256", hashed)
    custom._write_rtstruct_atomic(target, lambda name: Path(name).write_bytes(b"new synthetic RTSTRUCT"))
    assert steps == ["file_fsync", "temporary_readback", "replace", "directory_fsync", "published_readback"]
    assert target.read_bytes() == b"new synthetic RTSTRUCT"
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_rtstruct_readback_mismatch_refuses_success(tmp_path, monkeypatch):
    from rtpipeline import durable_progress
    target = tmp_path / "RS_custom.dcm"
    original = durable_progress.file_sha256
    def hashed(path):
        return "0" * 64 if path == target else original(path)
    monkeypatch.setattr(durable_progress, "file_sha256", hashed)
    with pytest.raises(durable_progress.DurableProgressError, match="DVH_RS_CUSTOM_READBACK_FAILED"):
        custom._write_rtstruct_atomic(target, lambda name: Path(name).write_bytes(b"synthetic RTSTRUCT"))
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_rtstruct_fsync_failure_keeps_previous_publication(tmp_path, monkeypatch):
    import errno
    target = tmp_path / "RS_custom.dcm"
    target.write_bytes(b"previous synthetic RTSTRUCT")
    def failed(fd):
        raise OSError(errno.EIO, "synthetic fsync failure")
    monkeypatch.setattr(os, "fsync", failed)
    with pytest.raises(OSError):
        custom._write_rtstruct_atomic(target, lambda name: Path(name).write_bytes(b"new synthetic RTSTRUCT"))
    assert target.read_bytes() == b"previous synthetic RTSTRUCT"
    assert list(tmp_path.iterdir()) == [target]
