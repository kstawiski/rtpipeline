"""Compare one course's robustness outputs from two code revisions.

Each argument is a directory holding what one run published for the same
course: radiomics_robustness_ct.parquet (or its failed-evidence table),
metadata/radiomics_robustness_identity.json,
metadata/radiomics_robustness_source_dispositions.json and
.radiomics_robustness_done. Run both revisions on the same course directory
one after the other and copy the artifacts aside after each run; then every
recorded path is identical and the comparison is strict.

Masked before comparison: run identifiers, the code identity, the table's
sha256 and size and digests of files that embed them. The table is compared
by content (every column and value, canonical row order, Arrow field types);
identity-ledger rows as a multiset, because the base revision appended
perturbation identity failures in worker completion order. When the two
directories are different course copies, their path prefixes are unified
and digests over path-bearing content are masked as well.

Only artifact names, row counts and the first differing field name are
printed, never identifiers or values, so the tool can be used on clinical
outputs. Exit status 0 means equivalent.

Usage: python -B -s tests/compare_robustness_course_outputs.py <base-dir> <new-dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ARTIFACTS = (
    "radiomics_robustness_ct.parquet",
    "radiomics_robustness_ct.failed_evidence.parquet",
    "metadata/radiomics_robustness_identity.json",
    "metadata/radiomics_robustness_source_dispositions.json",
    ".radiomics_robustness_done",
)
RUN_BOUND_PARENTS = {"measured_output", "source_dispositions", "failed_evidence"}


def _run_identity(root: Path):
    """(run identifier, course directory the run recorded) from its sidecar."""
    sidecar = root / "metadata" / "radiomics_robustness_source_dispositions.json"
    if not sidecar.is_file():
        return None, str(root)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    paths = [Path(b["source_path"]) for b in payload.get("source_bindings", [])
             if b.get("source_path")]
    # Sources live in the course directory or below it (custom models).
    course = min((p.parent for p in paths), key=lambda p: len(p.parts), default=root)
    return payload.get("robustness_run_identifier"), str(course)


def _normalize(value, run_id, prefix, same_paths, parent=None):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "code_identity":
                out[key] = "<code identity>"
            elif key in {"sha256", "size_bytes"} and parent in RUN_BOUND_PARENTS:
                out[key] = "<run-bound>"
            elif not same_paths and key.endswith("sha256"):
                out[key] = "<path-bound digest>"
            else:
                out[key] = _normalize(item, run_id, prefix, same_paths, key)
        return out
    if isinstance(value, list):
        return [_normalize(item, run_id, prefix, same_paths, parent) for item in value]
    if isinstance(value, str):
        if run_id:
            value = value.replace(run_id, "<run>")
        if prefix:
            value = value.replace(prefix, "<course>")
    return value


def _first_difference(a, b, path="") -> str:
    if type(a) is not type(b):
        return path or "<root>"
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return f"{path}.{key}"
            if a[key] != b[key]:
                return _first_difference(a[key], b[key], f"{path}.{key}")
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}[len]"
        for index, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return _first_difference(x, y, f"{path}[{index}]")
    return path or "<root>"


def _canonical_table(frame: pd.DataFrame, run_id) -> pd.DataFrame:
    frame = frame.copy()
    if "run_identifier" in frame and run_id:
        frame["run_identifier"] = frame["run_identifier"].replace(run_id, "<run>")
    frame = frame.reindex(columns=sorted(frame.columns))
    order = frame.astype(str).agg("\x1f".join, axis=1).sort_values(kind="stable").index
    return frame.loc[order].reset_index(drop=True)


def main() -> int:
    base, new = (Path(arg).resolve() for arg in sys.argv[1:3])
    (base_run, base_course), (new_run, new_course) = _run_identity(base), _run_identity(new)
    same_paths = base_course == new_course
    different = []
    for artifact in ARTIFACTS:
        a, b = base / artifact, new / artifact
        if a.is_file() != b.is_file():
            different.append(f"{artifact}: present in only one run")
            continue
        if not a.is_file():
            continue
        if artifact.endswith(".parquet"):
            fa, fb = pd.read_parquet(a), pd.read_parquet(b)
            ta = {f.name: str(f.type) for f in pq.read_schema(a)}
            tb = {f.name: str(f.type) for f in pq.read_schema(b)}
            if ta != tb:
                different.append(f"{artifact}: column names or Arrow types differ")
                continue
            try:
                pd.testing.assert_frame_equal(_canonical_table(fa, base_run),
                                              _canonical_table(fb, new_run), check_exact=True)
            except AssertionError:
                different.append(f"{artifact}: row content differs")
                continue
            print(f"{artifact}: {len(fa)} rows, {fa.shape[1]} columns equal by content")
            continue
        ja = _normalize(json.loads(a.read_text(encoding="utf-8")), base_run,
                        None if same_paths else base_course, same_paths)
        jb = _normalize(json.loads(b.read_text(encoding="utf-8")), new_run,
                        None if same_paths else new_course, same_paths)
        if artifact.endswith("identity.json"):
            for ledger in (ja, jb):
                ledger["rows"] = sorted(ledger.get("rows", []),
                                        key=lambda row: json.dumps(row, sort_keys=True))
        if ja != jb:
            different.append(f"{artifact}: differs at {_first_difference(ja, jb)}")
        else:
            rows = ja.get("rows") if isinstance(ja, dict) else None
            print(f"{artifact}: equal" + (f" ({len(rows)} rows)" if isinstance(rows, list) else ""))
    for line in different:
        print("DIFFERENT", line)
    print("EQUIVALENT" if not different else "NOT EQUIVALENT")
    return 0 if not different else 1


if __name__ == "__main__":
    raise SystemExit(main())
