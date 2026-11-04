#!/usr/bin/env python3
"""Fail when tracked files contain likely patient course keys.

RTpipeline course directories are named <patient>/<YYYY-MM>, and clinical patient
identifiers are numeric. Code, tests, comments and committed results must use
synthetic keys (for example "SYNTH_A" or "patient_01"), never a real key. This
check needs no restricted data. It flags a standalone 6-digit token next to a
YYYY-MM course component, or in a patient path position such as
tmp_path / "<id>" or Output/<id>/. Explicit synthetic numeric keys may be listed
in scripts/patient_key_allowlist.txt, one per line.

Usage: python3 scripts/check_patient_keys.py [PATH ...]   (default: git ls-files)
Exit status 1 lists each file and line number, with the token masked.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "scripts" / "patient_key_allowlist.txt"
TOKEN = r"(?<![0-9A-Za-z])(\d{6})(?![0-9A-Za-z])"
PATTERNS = [
    re.compile(TOKEN + r"""["']?\s*(?:/|__|_|-|,)\s*["']?(?:19|20)\d{2}-\d{2}"""),  # id/YYYY-MM, id__YYYY-MM, ("id", "YYYY-MM"), [id-YYYY-MM]
    re.compile(r"""tmp_path\s*/\s*["']""" + r"(\d{6})" + r"""["']"""),                 # tmp_path / "id"
    re.compile(r"""Output/""" + r"(\d{6})" + r"/"),                                      # Output/id/
]
SKIP_SUFFIXES = {".lock", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".nii", ".gz", ".dcm", ".zip", ".parquet", ".xlsx", ".pkl", ".npy", ".npz", ".sif"}


def allowlist() -> set[str]:
    if not ALLOWLIST.is_file():
        return set()
    return {line.split("#", 1)[0].strip() for line in ALLOWLIST.read_text().splitlines() if line.split("#", 1)[0].strip()}


def tracked() -> list[Path]:
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, check=True).stdout
    return [ROOT / p for p in out.decode(errors="replace").split("\0") if p]


def scan(paths: list[Path], allowed: set[str]) -> list[tuple[str, int]]:
    hits = []
    for path in paths:
        if path.suffix.lower() in SKIP_SUFFIXES or not path.is_file() or path.stat().st_size > 20_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            for pattern in PATTERNS:
                if any(m.group(1) not in allowed for m in pattern.finditer(line)):
                    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
                    hits.append((str(rel), number))
                    break
    return hits


def main(argv: list[str]) -> int:
    paths = [Path(a).resolve() for a in argv] if argv else tracked()
    hits = scan(paths, allowlist())
    for rel, number in hits:
        print(f"{rel}:{number}: likely patient course key (use a synthetic key)")
    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
