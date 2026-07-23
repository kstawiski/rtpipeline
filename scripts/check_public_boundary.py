#!/usr/bin/env python3
"""Fail if private manuscript workspace material enters this public checkout."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_ROOTS = (
    "archive",
    "compliance",
    "deposit",
    "figures",
    "manuscript",
    "plan",
    "references",
    "revision",
    "source_data",
    "step_prompts",
    "submission",
    "supplement",
    "tables",
    "verification",
)
FORBIDDEN_FILES = (
    ".manuscript-workflow-managed.json",
    "CURRENT_RESULTS.md",
    "Dockerfile." + "reviewer_minimal",
    "config.example.yaml",
    "config_prostata.yaml",
    "radiomics_params.yaml",
    "submission_gaps.md",
)
FORBIDDEN_RELEASE_MARKERS = (
    "blinded " + "reviewer archive",
    "DICOMRT-" + "datasets/rtpipeline_" + "manuscript_",
    "Dockerfile." + "reviewer_minimal",
)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> int:
    failures: list[str] = []
    tracked_result = _git("ls-files")
    if tracked_result.returncode != 0:
        raise RuntimeError(tracked_result.stderr.strip() or "git ls-files failed")
    tracked = set(tracked_result.stdout.splitlines())
    candidate_result = _git("ls-files", "--cached", "--others", "--exclude-standard")
    if candidate_result.returncode != 0:
        raise RuntimeError(
            candidate_result.stderr.strip() or "git candidate enumeration failed"
        )
    candidates = set(candidate_result.stdout.splitlines())

    for relative in FORBIDDEN_ROOTS:
        path = ROOT / relative
        if path.exists() or path.is_symlink():
            failures.append(f"private root exists in public checkout: {relative}")
        prefix = f"{relative}/"
        if any(item == relative or item.startswith(prefix) for item in tracked):
            failures.append(f"private root has tracked files: {relative}")

        probe = f"{relative}/__rtpipeline_private_boundary_probe__.csv"
        ignored = _git("check-ignore", "--quiet", "--no-index", "--", probe)
        if ignored.returncode == 0:
            failures.append(f".gitignore hides private-root probe: {probe}")
        elif ignored.returncode != 1:
            raise RuntimeError(
                ignored.stderr.strip() or f"git check-ignore failed for {probe}"
            )

    for relative in FORBIDDEN_FILES:
        path = ROOT / relative
        if path.exists() or path.is_symlink():
            failures.append(f"private file exists in public checkout: {relative}")
        if relative in tracked:
            failures.append(f"private file is tracked: {relative}")
        ignored = _git("check-ignore", "--quiet", "--no-index", "--", relative)
        if ignored.returncode == 0:
            failures.append(f".gitignore hides private-file probe: {relative}")
        elif ignored.returncode != 1:
            raise RuntimeError(
                ignored.stderr.strip() or f"git check-ignore failed for {relative}"
            )

    for relative in sorted(candidates):
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for marker in FORBIDDEN_RELEASE_MARKERS:
            if marker in content:
                failures.append(
                    f"private/reviewer marker {marker!r} appears in tracked file: {relative}"
                )

    if failures:
        print("Public repository boundary check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Public repository boundary check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
