#!/usr/bin/env python3
"""Local Git hook helper: refuse commits or pushes whose content matches the restricted patient-ID list.

The list is never stored in a repository. It is read at hook time from a sealed file with
`phi cat` (RTPIPELINE_RESTRICTED_IDS_AGE, default /umed-projekty/rtpipeline-phi/kopernik_patient_ids.txt.age),
or for tests from --ids-file. Fails closed when the list cannot be read. Prints counts and masked paths only.

  --staged             scan staged blobs (pre-commit)
  --pre-push           scan blobs and messages of every commit being pushed (stdin: git pre-push lines)
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

DEFAULT_SEALED = "/umed-projekty/rtpipeline-phi/kopernik_patient_ids.txt.age"
TOKEN = re.compile(rb"(?<![0-9])[0-9]{5,}(?![0-9])")
ZERO = "0" * 40


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], capture_output=True, check=True).stdout


def load_ids(ids_file: str | None) -> set[bytes]:
    if ids_file:
        data = open(ids_file, "rb").read()
    else:
        phi = shutil.which("phi") or os.path.expanduser("~/.local/bin/phi")
        sealed = os.environ.get("RTPIPELINE_RESTRICTED_IDS_AGE", DEFAULT_SEALED)
        result = subprocess.run([phi, "cat", sealed], capture_output=True)
        if result.returncode != 0:
            sys.stderr.write("restricted-ID check: the sealed ID list could not be read; refusing (fail closed).\n")
            raise SystemExit(1)
        data = result.stdout
    ids = {line.strip() for line in data.splitlines() if line.strip()}
    if not ids:
        sys.stderr.write("restricted-ID check: the ID list is empty; refusing (fail closed).\n")
        raise SystemExit(1)
    return ids


def staged() -> list[tuple[str, bytes]]:
    names = [n for n in git("diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR").decode(errors="replace").split("\0") if n]
    return [(n, git("show", f":{n}")) for n in names]


def pushed(lines: list[str]) -> list[tuple[str, bytes]]:
    out = []
    for line in lines:
        parts = line.split()
        if len(parts) != 4 or parts[1] == ZERO:
            continue
        local, remote = parts[1], parts[3]
        spec = [local, "--not", "--remotes"] if remote == ZERO else [f"{remote}..{local}"]
        for commit in [c for c in git("rev-list", *spec).decode().split() if c]:
            out.append((f"commit {commit[:10]} message", git("log", "-1", "--format=%B", commit)))
            for n in [n for n in git("diff-tree", "-r", "--no-commit-id", "--name-only", "-z", "--diff-filter=ACMR", "--root", commit).decode(errors="replace").split("\0") if n]:
                try:
                    out.append((f"{commit[:10]}:{n}", git("show", f"{commit}:{n}")))
                except subprocess.CalledProcessError:
                    continue
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staged", action="store_true")
    mode.add_argument("--pre-push", action="store_true")
    parser.add_argument("--ids-file")
    args = parser.parse_args()
    ids = load_ids(args.ids_file)
    items = staged() if args.staged else pushed(sys.stdin.read().splitlines())
    hits = [name for name, blob in items if set(TOKEN.findall(blob)) & ids]
    if hits:
        sys.stderr.write(f"restricted-ID check: {len(hits)} item(s) contain a restricted patient identifier; refusing.\n")
        for name in hits[:50]:
            sys.stderr.write("  " + re.sub(r"[0-9]{5,}", "<N>", name) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
