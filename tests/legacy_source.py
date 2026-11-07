"""Read pinned historical source after the 2026-11-04 history rewrite.

The manifest pins package files from legacy_blob_availability.json. Only
radiomics_schema.py uses a rewritten blob: the supplied evidence records
one changed comment line and identical ASTs. All other files must match
their pre-rewrite bytes exactly. No rewritten commit is needed. Reads and
archives return bytes, as git show and git archive do.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


_MANIFEST = json.loads(
    Path(__file__).with_name("legacy_revisions.json").read_text(encoding="utf-8")
)
_REVISIONS = _MANIFEST["revisions"]
_FILE_MODES = _MANIFEST["archive_file_modes"]
_SCHEMA_PATH = "rtpipeline/radiomics_schema.py"
_ROOT = Path(__file__).resolve().parents[1]


def _record(revision: str) -> dict:
    record = _REVISIONS.get(revision)
    if record is None:
        record = next((item for item in _REVISIONS.values()
                       if item["old"] == revision), None)
    if record is None:
        raise ValueError(f"Unknown legacy revision {revision!r}; no recorded blob fixture")
    return record


def legacy_show(revision: str, path: str, cwd: str | Path = _ROOT) -> bytes:
    """Read and verify a pinned blob for a recorded legacy short/full ID.

    Unknown revisions and paths raise ValueError. Missing Git objects and
    mismatched evidence fail the caller; they never turn a comparison into
    a skip or fall back to current source.
    """
    record = _record(revision)
    if path not in record["blobs"]:
        raise ValueError(f"Unknown legacy path {path!r} for revision {revision!r}")

    expected = record["blobs"][path]
    if path == _SCHEMA_PATH:
        evidence = record["radiomics_schema"]
        if (evidence["ast_identical"] is not True
                or evidence["equivalence"] != "comment-only, AST-identical"
                or evidence["lines_differing"] != 1
                or evidence["rewritten_blob"] != expected):
            raise AssertionError(f"Unverified comment-only, AST-identical equivalence for {revision}:{path}")

    source = subprocess.check_output(["git", "cat-file", "blob", expected], cwd=cwd)
    actual = hashlib.sha1(b"blob " + str(len(source)).encode("ascii") + b"\0" + source).hexdigest()
    if actual != expected:
        raise AssertionError(
            f"Legacy blob mismatch for {revision}:{path}: "
            f"expected {expected}, got {actual}"
        )
    return source


def legacy_archive(
    revision: str, prefix: str = "rtpipeline", cwd: str | Path = _ROOT,
) -> bytes:
    """Build a tar of verified package blobs, retaining repository paths.

    Like git archive's path argument, prefix selects a file or directory.
    File bytes and directory members are reconstructed from the fixture;
    permissions follow git archive's default tar.umask of 0002. Tar
    timestamps and ownership use deterministic defaults.
    """
    record = _record(revision)
    prefix = prefix.rstrip("/")
    paths = [path for path in record["blobs"]
             if path == prefix or path.startswith(prefix + "/")]
    if not paths:
        raise ValueError(f"Unknown legacy path {prefix!r} for revision {revision!r}")
    directories = {str(parent) for path in paths for parent in Path(path).parents
                   if parent != Path(".")}
    with io.BytesIO() as output:
        with tarfile.open(fileobj=output, mode="w") as archive:
            for path in sorted(directories | set(paths)):
                member = tarfile.TarInfo(path)
                if path in directories:
                    member.type = tarfile.DIRTYPE
                    member.mode = 0o775
                    archive.addfile(member)
                else:
                    source = legacy_show(revision, path, cwd=cwd)
                    member.mode = {"100644": 0o664, "100755": 0o775}[_FILE_MODES[path]]
                    member.size = len(source)
                    archive.addfile(member, io.BytesIO(source))
        return output.getvalue()
