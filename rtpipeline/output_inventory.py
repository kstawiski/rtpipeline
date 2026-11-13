"""Exclude durable bookkeeping from scientific output inventories.

Operational state remains available to resume validators. Only these reserved
names are excluded: scientific manifests, hidden completion receipts and all
other output bytes remain part of the scientific inventory. Every durable
progress root is a hidden directory named ``.<name>-progress`` (for example the
conda backend's ``.radiomics_ct-progress`` and the robustness summary's
``.radiomics_robustness_summary-progress``), so that form is reserved as a
whole rather than name by name.
"""
from pathlib import Path


_DIRECTORIES = frozenset({
    "_CACHE",
    ".conversion-progress", ".model-progress", ".organize-progress",
    ".custom-model-progress", ".weights-progress", ".crop-progress",
    ".qc-progress", ".radiomics-progress", ".robustness-progress", ".dvh",
    ".aggregate-progress", ".rtsegmentator-progress",
})
_FILES = frozenset({
    ".organize-progress.json", "aggregate_results.commit.json",
    "radiomics_robustness_summary.commit.json",
})


def _is_progress_directory(part: str, last: bool) -> bool:
    return (not last and part.startswith(".") and part.endswith("-progress")
            and len(part) > len(".-progress"))


def is_operational_path(relative_path) -> bool:
    """Whether a path relative to an output root is reserved bookkeeping."""
    parts = Path(relative_path).parts
    return any(part in _DIRECTORIES or part in _FILES
               or _is_progress_directory(part, index == len(parts) - 1)
               or part.endswith(".commit.json")
               or (part.startswith(".") and part[1:].removesuffix("-generations").endswith(".commit.json")
                   and part.endswith("-generations"))
               for index, part in enumerate(parts))


def scientific_files(root):
    """Enumerate scientific files without changing their names or bytes."""
    root = Path(root)
    return (path for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix != ".age"
            and not is_operational_path(path.relative_to(root)))
