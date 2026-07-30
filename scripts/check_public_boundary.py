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
# The manuscript workflow's entry check also installs its own helper scripts into
# `scripts/`, alongside this repository's legitimate ones. Those never tripped the
# top-level root/file checks, so the guard printed "passed" while seven private
# workflow scripts sat in the public checkout, one `git add -A` from publication.
FORBIDDEN_SCRIPTS = (
    "emit_submission_receipt.py",
    "make_deposit.sh",
    "reproduction_sandbox.py",
    "review_data_policy.py",
    "sealed_runtime_catalog.py",
    "verify_upstream_results.py",
    "workflow_state.py",
)
FORBIDDEN_RELEASE_MARKERS = (
    "blinded " + "reviewer archive",
    "DICOMRT-" + "datasets/rtpipeline_" + "manuscript_",
    "Dockerfile." + "reviewer_minimal",
    "Manuscript " + "Workspace Identity",
)
# Skip blobs no plausible manuscript artifact would reach; keeps the ignored-file
# sweep cheap next to virtualenvs, caches and model weights.
MAX_SCAN_BYTES = 4 * 1024 * 1024


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )


def check(root: Path | None = None) -> list[str]:
    """Return the boundary failures found under ``root`` (default: this repo)."""
    root = ROOT if root is None else Path(root).resolve()

    def _run(*args: str) -> subprocess.CompletedProcess[str]:
        return _git(root, *args)

    failures: list[str] = []
    tracked_result = _run("ls-files")
    if tracked_result.returncode != 0:
        raise RuntimeError(tracked_result.stderr.strip() or "git ls-files failed")
    tracked = set(tracked_result.stdout.splitlines())
    candidate_result = _run("ls-files", "--cached", "--others", "--exclude-standard")
    if candidate_result.returncode != 0:
        raise RuntimeError(
            candidate_result.stderr.strip() or "git candidate enumeration failed"
        )
    candidates = set(candidate_result.stdout.splitlines())

    for relative in FORBIDDEN_ROOTS:
        path = root / relative
        if path.exists() or path.is_symlink():
            failures.append(f"private root exists in public checkout: {relative}")
        prefix = f"{relative}/"
        if any(item == relative or item.startswith(prefix) for item in tracked):
            failures.append(f"private root has tracked files: {relative}")

        probe = f"{relative}/__rtpipeline_private_boundary_probe__.csv"
        ignored = _run("check-ignore", "--quiet", "--no-index", "--", probe)
        if ignored.returncode == 0:
            failures.append(f".gitignore hides private-root probe: {probe}")
        elif ignored.returncode != 1:
            raise RuntimeError(
                ignored.stderr.strip() or f"git check-ignore failed for {probe}"
            )

    for relative in FORBIDDEN_FILES:
        path = root / relative
        if path.exists() or path.is_symlink():
            failures.append(f"private file exists in public checkout: {relative}")
        if relative in tracked:
            failures.append(f"private file is tracked: {relative}")
        ignored = _run("check-ignore", "--quiet", "--no-index", "--", relative)
        if ignored.returncode == 0:
            failures.append(f".gitignore hides private-file probe: {relative}")
        elif ignored.returncode != 1:
            raise RuntimeError(
                ignored.stderr.strip() or f"git check-ignore failed for {relative}"
            )

    for name in FORBIDDEN_SCRIPTS:
        relative = f"scripts/{name}"
        path = root / relative
        if path.exists() or path.is_symlink():
            failures.append(
                f"manuscript-workflow script exists in public checkout: {relative}"
            )
        if relative in tracked:
            failures.append(f"manuscript-workflow script is tracked: {relative}")
        ignored = _run("check-ignore", "--quiet", "--no-index", "--", relative)
        if ignored.returncode == 0:
            failures.append(f".gitignore hides workflow-script probe: {relative}")
        elif ignored.returncode != 1:
            raise RuntimeError(
                ignored.stderr.strip() or f"git check-ignore failed for {relative}"
            )

    # `git ls-files --exclude-standard` honours .gitignore, .git/info/exclude and
    # the global excludes file, so a private file hidden by any of them never
    # reached the content scan below. That defeated the whole point of this
    # guard: private manuscript material had already *entered* the checkout, and
    # the check still passed. Scan every top-level regular file as well,
    # regardless of ignore status — that is where a stray CURRENT_RESULTS.md,
    # VERSION.md or HANDOFF.md from the manuscript workspace lands.
    scan_targets = set(candidates)
    for entry in sorted(root.iterdir()):
        if entry.name == ".git" or not entry.is_file() or entry.is_symlink():
            continue
        scan_targets.add(entry.name)

    for relative in sorted(scan_targets):
        path = root / relative
        if not path.is_file() or path.is_symlink():
            continue
        try:
            if path.stat().st_size > MAX_SCAN_BYTES:
                continue
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for marker in FORBIDDEN_RELEASE_MARKERS:
            if marker in content:
                where = "tracked file" if relative in tracked else "file in checkout"
                failures.append(
                    f"private/reviewer marker {marker!r} appears in {where}: {relative}"
                )

    return failures


def main() -> int:
    failures = check()
    if failures:
        print("Public repository boundary check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Public repository boundary check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
