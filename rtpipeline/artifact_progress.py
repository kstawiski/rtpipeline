"""Bound artifact items using the shared durable progress transactions.

Producers own their expected inventory and validation. A committed cache record
is reusable only while every named artifact still has its committed bytes.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import uuid
import shutil

from .durable_progress import (DurableProgressError, ProgressJournal,
                               _fsync_dir, file_sha256)
from .extraction_progress import BoundItem


def code_files(*names):
    root = Path(__file__).parent
    return {name: root / name for name in
            ("artifact_progress.py", "extraction_progress.py", "durable_progress.py", *names)}


def propagate_persistence_failure(error):
    current = error
    while current is not None:
        if isinstance(current, (OSError, DurableProgressError)):
            raise error
        current = current.__cause__


def inventory(paths, root=None):
    return {str(p.relative_to(root) if root else p): file_sha256(p)
            for p in sorted(map(Path, paths)) if p.suffix != ".age"}


def _ensure_directory(path):
    path = Path(path)
    missing, ancestor = [], path
    while not ancestor.exists():
        missing.append(ancestor)
        ancestor = ancestor.parent
    path.mkdir(parents=True, exist_ok=True)
    for directory in reversed(missing):
        _fsync_dir(directory.parent)
        _fsync_dir(directory)


class ArtifactItem(BoundItem):
    def __init__(self, root, identity, settings, inputs, *, base, code, environment=None):
        super().__init__(root, identity, settings, inputs, code=code, environment=environment)
        self.identity = identity
        self.base = Path(base).resolve()

    def load(self):
        value = super().load()
        if value is None:
            return None
        try:
            if value.get("artifact_root") != str(self.base):
                raise ValueError("artifact root changed")
            for name, digest in value["artifacts"].items():
                path = (self.base / name).resolve()
                path.relative_to(self.base)
                if file_sha256(path) != digest:
                    raise ValueError("artifact digest changed")
        except (OSError, ValueError, KeyError, TypeError, DurableProgressError):
            # Preserve invalid evidence outside ResultCache's transaction scan.
            # A new transaction may then repair this one item under the same
            # scientific binding without conflicting with the invalid result.
            with self.locked():
                invalid = self.cache.root.parent / (self.key + "-invalid")
                invalid.mkdir(exist_ok=True)
                if self.cache.root.exists():
                    for directory in self.cache.root.iterdir():
                        if directory.is_dir():
                            os.replace(directory, invalid / (directory.name + "-" + uuid.uuid4().hex))
                    _fsync_dir(invalid)
                    _fsync_dir(self.cache.root)
                with ProgressJournal(self._lock_path.with_suffix(".jsonl")) as journal:
                    journal.append({"event": "artifact_invalidated", "item_key": self.key})
            return None
        return value

    def commit_artifacts(self, paths, **result):
        paths = list(map(Path, paths))
        if len(set(paths)) != len(paths):
            raise ValueError("duplicate artifact")
        artifacts = {}
        directories = set()
        for path in paths:
            name = path.resolve().relative_to(self.base).as_posix()
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            parent = path.parent.resolve()
            while True:
                directories.add(parent)
                if parent == self.base.parent:
                    break
                parent = parent.parent
            artifacts[name] = file_sha256(path)
        for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
            _fsync_dir(directory)
        value = self.commit(dict(result, artifacts=artifacts, artifact_root=str(self.base)))
        _failpoint("artifact_committed", self)
        return value


@contextmanager
def staged_file(target):
    target = Path(target)
    _ensure_directory(target.parent)
    fd, name = tempfile.mkstemp(prefix=f".prepare-{os.getpid()}-", suffix="".join(target.suffixes), dir=target.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        digest = file_sha256(temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        _fsync_dir(target.parent)
        if file_sha256(target) != digest:
            raise DurableProgressError("ARTIFACT_PUBLICATION_READBACK_FAILED")
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value, **kwargs):
    text = json.dumps(value, **kwargs)
    path = Path(path)
    if path.is_file() and path.read_text() == text:
        return
    with staged_file(path) as temporary:
        temporary.write_text(text)
        if json.loads(temporary.read_text()) != json.loads(text):
            raise DurableProgressError("ARTIFACT_JSON_READBACK_FAILED")


def publish_workbook(path, frame):
    """Verify a workbook and preserve an unchanged scientific table's mtime."""
    import pandas as pd
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-workbook-", dir=path.parent) as scratch:
        prepared = Path(scratch) / path.name
        frame.to_excel(prepared, index=False)
        parsed = pd.read_excel(prepared)
        if path.is_file():
            try:
                pd.testing.assert_frame_equal(parsed, pd.read_excel(path))
            except Exception:
                pass
            else:
                return
        with staged_file(path) as temporary:
            shutil.copyfile(prepared, temporary)


def _failpoint(event, item):
    """Inert interruption seam used only by synthetic fixtures."""
