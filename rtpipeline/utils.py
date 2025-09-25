from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Iterable

import pydicom
from pydicom.dataset import FileDataset
from pydicom.tag import Tag

logger = logging.getLogger(__name__)


def read_dicom(path: str | os.PathLike) -> FileDataset | None:
    try:
        return pydicom.dcmread(str(path), force=True)
    except Exception as e:
        logger.debug("Failed to read DICOM %s: %s", path, e)
        return None


def get(ds: FileDataset, tag: int | tuple[int, int] | str, default: Any = None) -> Any:
    try:
        if isinstance(tag, str):
            return getattr(ds, tag, default)
        return ds.get(Tag(tag)).value if Tag(tag) in ds else default
    except Exception:
        return default


def file_md5(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_files(root: Path, patterns: Iterable[str] | None = None) -> list[Path]:
    if patterns is None:
        patterns = ["*.dcm", "*"]
    out: list[Path] = []
    for base, _, files in os.walk(root):
        for name in files:
            path = Path(base) / name
            if any(path.match(pat) for pat in patterns):
                out.append(path)
    return out

