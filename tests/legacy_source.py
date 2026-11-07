"""Read pinned historical source after the 2026-11-04 history rewrite.

The manifest preserves pre-rewrite blob IDs from legacy_oracle_blobs.json
and the unique rewritten twins from legacy_oracle_map.json. Only
radiomics_schema.py uses a rewritten blob: the supplied evidence records
one changed comment line and identical ASTs. All other files must match
their pre-rewrite bytes exactly. Reads return bytes, as git show does.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


_REVISIONS = json.loads(
    Path(__file__).with_name("legacy_revisions.json").read_text(encoding="utf-8")
)["revisions"]
_SCHEMA_PATH = "rtpipeline/radiomics_schema.py"


def legacy_show(revision: str, path: str, cwd: str | Path) -> bytes:
    """Resolve a recorded legacy short/full ID and verify the returned blob.

    Unknown revisions and paths raise ValueError. Missing Git objects and
    mismatched evidence fail the caller; they never turn a comparison into
    a skip or fall back to current source.
    """
    record = _REVISIONS.get(revision)
    if record is None:
        record = next((item for item in _REVISIONS.values()
                       if item["old"] == revision), None)
    if record is None:
        raise ValueError(f"Unknown legacy revision {revision!r}; no recorded rewritten twin")
    if path not in record["old_blobs"]:
        raise ValueError(f"Unknown legacy path {path!r} for revision {revision!r}")

    expected = record["old_blobs"][path]
    if path == _SCHEMA_PATH:
        evidence = record["radiomics_schema"]
        if (evidence["ast_identical"] is not True
                or evidence["equivalence"] != "comment-only, AST-identical"
                or evidence["lines_differing"] != 1):
            raise AssertionError(f"Unverified comment-only, AST-identical equivalence for {revision}:{path}")
        expected = evidence["rewritten_blob"]

    source = subprocess.check_output(["git", "show", f"{record['new']}:{path}"], cwd=cwd)
    actual = hashlib.sha1(b"blob " + str(len(source)).encode("ascii") + b"\0" + source).hexdigest()
    if actual != expected:
        raise AssertionError(
            f"Legacy blob mismatch for {revision}:{path} (rewritten {record['new']}): "
            f"expected {expected}, got {actual}"
        )
    return source
