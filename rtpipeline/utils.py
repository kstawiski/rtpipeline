from __future__ import annotations

import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, List, Optional, Sequence, TypeVar

import numpy as np
import pydicom
from pydicom.dataset import FileDataset
from pydicom.sequence import Sequence
from pydicom.tag import Tag

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


def validate_path(path: Path | str, base: Path | str, allow_absolute: bool = False) -> Path:
    """
    Validate that a path doesn't escape the base directory (path traversal protection).

    Args:
        path: Path to validate (can be relative or absolute)
        base: Base directory that path must be within
        allow_absolute: If True, allow absolute paths outside base (default: False)

    Returns:
        Resolved path if validation passes

    Raises:
        ValueError: If path attempts to escape base directory

    Example:
        >>> validate_path(Path("data/file.txt"), Path("/app/data"))
        PosixPath('/app/data/file.txt')

        >>> validate_path(Path("../../etc/passwd"), Path("/app/data"))
        ValueError: Path traversal attempt detected: ../../etc/passwd
    """
    path = Path(path)
    base = Path(base).resolve()

    # If path is absolute and we allow it, just check it exists within base
    if path.is_absolute():
        if not allow_absolute:
            raise ValueError(f"Absolute paths not allowed: {path}")
        resolved = path.resolve()
    else:
        # For relative paths, resolve against base
        resolved = (base / path).resolve()

    # Security check: ensure resolved path is within base directory
    try:
        resolved.relative_to(base)
    except ValueError:
        raise ValueError(
            f"Path traversal attempt detected: {path} resolves to {resolved} "
            f"which is outside base directory {base}"
        )

    return resolved


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


def sanitize_rtstruct(rtstruct_path: Path | str, *, minimum_points: int = 3) -> bool:
    """Remove degenerate contour items from an RTSTRUCT file.

    Contours with fewer than ``minimum_points`` XYZ points (i.e. a single point or
    line) confuse downstream voxelisation code. This helper strips them in-place and
    normalises the ``NumberOfContourPoints`` metadata.

    Returns
    -------
    bool
        True when the file was modified, False otherwise.
    """

    path = Path(rtstruct_path)
    if not path.exists():
        logger.debug("sanitize_rtstruct: %s missing", path)
        return False

    try:
        ds = pydicom.dcmread(str(path))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("sanitize_rtstruct: unable to read %s (%s)", path, exc)
        return False

    changed = False
    roi_names = {
        int(roi.ROINumber): str(getattr(roi, "ROIName", ""))
        for roi in getattr(ds, "StructureSetROISequence", [])
    }

    for roi_contour in list(getattr(ds, "ROIContourSequence", []) or []):
        contour_seq = getattr(roi_contour, "ContourSequence", None)
        if not contour_seq:
            continue

        filtered = []
        for contour in contour_seq:
            try:
                data = list(contour.ContourData)
            except Exception:
                changed = True
                continue
            points = len(data) // 3
            if points >= minimum_points:
                if getattr(contour, "NumberOfContourPoints", points) != points:
                    contour.NumberOfContourPoints = points
                    changed = True
                filtered.append(contour)
            else:
                roi_name = roi_names.get(getattr(roi_contour, "ReferencedROINumber", -1), "")
                logger.debug(
                    "sanitize_rtstruct: dropping contour for ROI '%s' (points=%d) in %s",
                    roi_name,
                    points,
                    path,
                )
                changed = True

        if filtered:
            if len(filtered) != len(contour_seq):
                roi_contour.ContourSequence = Sequence(filtered)
                changed = True
        else:
            del roi_contour.ContourSequence
            changed = True

    if changed:
        try:
            ds.save_as(str(path))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("sanitize_rtstruct: failed saving %s (%s)", path, exc)
            return False

    return changed


def mask_is_cropped(mask: np.ndarray) -> bool:
    """Determine whether a binary mask touches the image boundary.

    Accepts masks in either [z, y, x] or [y, x, z] orientation.
    """

    if mask is None:
        return False
    arr = np.asarray(mask)
    if arr.ndim != 3:
        return False
    arr = arr.astype(bool)
    if not arr.any():
        return False
    for axis in range(arr.ndim):
        slicer = [slice(None)] * arr.ndim
        slicer[axis] = 0
        if arr[tuple(slicer)].any():
            return True
        slicer[axis] = arr.shape[axis] - 1
        if arr[tuple(slicer)].any():
            return True
    return False


def snap_rtstruct_to_dose_grid(
    rtstruct: FileDataset,
    rtdose: FileDataset,
    tolerance_mm: float = 1.0,
) -> bool:
    """Snap RTSTRUCT contour plane positions to the closest dose grid plane.

    Many TPS export contours with sub-millimetre deviations from the dose grid.
    dicompyler-core expects an exact match and otherwise falls back to extrema,
    degrading DVH accuracy. This helper adjusts each contour ``z`` coordinate to
    the nearest plane within ``tolerance_mm``. Returns True when any coordinate
    was updated.
    """

    if rtstruct is None or rtdose is None:
        return False

    try:
        ipp = getattr(rtdose, "ImagePositionPatient", None)
        offsets = getattr(rtdose, "GridFrameOffsetVector", [])
        if ipp is None or len(ipp) < 3:
            return False
        base_z = float(ipp[2])
        planes = base_z + np.asarray([float(v) for v in offsets], dtype=float)
    except Exception:
        return False

    if planes.size == 0:
        return False

    tolerance = abs(float(tolerance_mm)) if tolerance_mm is not None else 0.0
    if tolerance == 0.0:
        tolerance = 0.5
    try:
        spacings = np.diff(np.unique(planes))
        min_spacing = float(np.min(spacings)) if spacings.size else None
    except Exception:
        min_spacing = None
    if min_spacing is not None:
        spacing_tol = min_spacing * 1.5
        if tolerance < spacing_tol:
            tolerance = spacing_tol

    changed = False
    roi_sequence = getattr(rtstruct, "ROIContourSequence", []) or []
    for roi_contour in roi_sequence:
        contour_seq = getattr(roi_contour, "ContourSequence", None)
        if not contour_seq:
            continue
        for contour in contour_seq:
            data = getattr(contour, "ContourData", None)
            if not data:
                continue
            updated = False
            for idx in range(2, len(data), 3):
                try:
                    current_z = float(data[idx])
                except Exception:
                    continue
                nearest = float(planes[np.abs(planes - current_z).argmin()])
                if abs(nearest - current_z) <= tolerance and not np.isclose(nearest, current_z, atol=1e-3):
                    rounded = round(nearest, 6)
                    try:
                        data[idx] = type(data[idx])(rounded)
                    except Exception:
                        data[idx] = rounded
                    updated = True
            if updated:
                changed = True
    return changed


_MEMORY_PATTERNS = (
    "out of memory",
    "cuda out of memory",
    "cublas status alloc failed",
    "std::bad_alloc",
    "cannot allocate memory",
    "failed to allocate",
    "not enough memory",
    "mmap failed",
    "oom",
)


def _is_memory_error(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).lower()
    return any(pat in msg for pat in _MEMORY_PATTERNS)


def _log_progress(logger: logging.Logger, label: str, completed: int, total: int, start: float) -> None:
    if total == 0:
        return
    elapsed = perf_counter() - start
    rate = completed / elapsed if elapsed > 0 else 0
    eta = (total - completed) / rate if rate > 0 else float('inf')
    logger.info(
        "%s: %d/%d (%.0f%%) elapsed %.1fs ETA %.1fs",
        label,
        completed,
        total,
        100 * completed / total,
        elapsed,
        eta,
    )


def run_tasks_with_adaptive_workers(
    label: str,
    items: Sequence[T],
    func: Callable[[T], R],
    *,
    max_workers: int,
    min_workers: int = 1,
    logger: Optional[logging.Logger] = None,
    show_progress: bool = False,
    task_timeout: Optional[int] = None,
) -> List[Optional[R]]:
    """Run tasks with adaptive worker fallback when memory pressure is detected.

    Args:
        label: Human-readable label for logging progress.
        items: Ordered collection of task inputs.
        func: Callable executed for each item.
        max_workers: Initial maximum parallel workers.
        min_workers: Lower bound for workers when backing off.
        logger: Optional logger (defaults to module logger).
        show_progress: Emit progress log entries similar to previous pipeline behaviour.
        task_timeout: Optional timeout per task in seconds (None = no timeout).
                     Use this to prevent individual tasks from hanging indefinitely.

    Returns:
        List of results aligned with ``items`` order. Entries are ``None`` when
        a task failed.
    """

    seq = list(items)
    total = len(seq)
    if total == 0:
        return []

    log = logger or logging.getLogger(__name__)
    min_workers = max(1, min_workers)
    workers = max(min_workers, min(max_workers, total))
    results: List[Optional[R]] = [None] * total
    completed = 0
    start = perf_counter()

    pending = list(range(total))

    while pending:
        current_indices = pending
        pending = []
        mem_failures: List[int] = []
        workers = max(min_workers, min(workers, len(current_indices)))

        if show_progress:
            log.info(
                "%s: starting batch with %d worker(s) for %d task(s)",
                label,
                workers,
                len(current_indices),
            )

        with ThreadPoolExecutor(max_workers=workers) as ex:
            future_to_idx = {ex.submit(func, seq[idx]): idx for idx in current_indices}

            # Track task start times for timeout detection
            task_start_times = {fut: perf_counter() for fut in future_to_idx}
            last_heartbeat = perf_counter()

            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                item = seq[idx]
                item_desc = getattr(item, 'dir', None)
                if item_desc is None:
                    if isinstance(item, tuple):
                        parts = []
                        for part in item:
                            if hasattr(part, 'shape'):
                                parts.append(f"array{tuple(part.shape)}")
                            else:
                                parts.append(str(part))
                        item_desc = ", ".join(parts)
                    else:
                        item_desc = str(item)

                try:
                    # Use timeout if specified
                    if task_timeout is not None:
                        results[idx] = fut.result(timeout=task_timeout)
                    else:
                        results[idx] = fut.result()
                    completed += 1

                    # Log slow tasks after task completes
                    now = perf_counter()
                    task_duration = now - task_start_times[fut]
                    if task_duration > 300:  # Warn if task took more than 5 minutes
                        log.warning("%s: task #%d (%s) took %.1fs (slow)",
                                   label, idx + 1, item_desc, task_duration)

                    # Periodic heartbeat logging to detect hangs
                    if now - last_heartbeat > 60:  # Log every 60 seconds
                        log.info("%s: Still processing... %d/%d completed (%.1f%%)",
                                label, completed, total, 100 * completed / total if total > 0 else 0)
                        last_heartbeat = now

                except TimeoutError:
                    if task_timeout is not None:
                        log.error(
                            "%s: task #%d (%s) timed out after %ds",
                            label,
                            idx + 1,
                            item_desc,
                            task_timeout,
                        )
                    else:
                        log.error(
                            "%s: task #%d (%s) timed out",
                            label,
                            idx + 1,
                            item_desc,
                        )
                    completed += 1
                    results[idx] = None
                except Exception as exc:  # noqa: BLE001 - propagate with logging
                    if _is_memory_error(exc):
                        mem_failures.append(idx)
                        log.warning(
                            "%s: memory pressure detected (task #%d: %s): %s",
                            label,
                            idx + 1,
                            item_desc,
                            exc,
                        )
                    else:
                        log.error(
                            "%s: task #%d (%s) failed: %s",
                            label,
                            idx + 1,
                            item_desc,
                            exc,
                            exc_info=True,
                        )
                        completed += 1
                        results[idx] = None
                if show_progress:
                    _log_progress(log, label, completed, total, start)

        if mem_failures:
            if workers == min_workers == 1:
                for idx in mem_failures:
                    item = seq[idx]
                    item_desc = getattr(item, 'dir', None)
                    if item_desc is None:
                        if isinstance(item, tuple):
                            parts = []
                            for part in item:
                                if hasattr(part, 'shape'):
                                    parts.append(f"array{tuple(part.shape)}")
                                else:
                                    parts.append(str(part))
                            item_desc = ", ".join(parts)
                        else:
                            item_desc = str(item)
                    log.error(
                        "%s: memory error even with single worker; giving up on task #%d (%s)",
                        label,
                        idx + 1,
                        item_desc,
                    )
                    completed += 1
                    results[idx] = None
                mem_failures = []
            else:
                new_workers = max(min_workers, workers // 2)
                if new_workers == workers:
                    new_workers = max(min_workers, workers - 1)
                if new_workers < workers:
                    log.warning(
                        "%s: retrying %d task(s) with %d worker(s) after memory pressure",
                        label,
                        len(mem_failures),
                        new_workers,
                    )
                workers = new_workers
                pending = mem_failures
        else:
            break

    if show_progress and completed < total:
        _log_progress(log, label, completed, total, start)

    return results
