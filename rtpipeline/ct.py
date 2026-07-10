from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

from .utils import read_dicom, get, ensure_dir, _scoped_walk, _scoped_patient_dirs, parallel_map_files, DEFAULT_INDEX_WORKERS

if TYPE_CHECKING:
    from .dicom_copy import DicomCopyManager

logger = logging.getLogger(__name__)


@dataclass
class CTInstance:
    path: Path
    patient_id: str
    study_uid: str
    series_uid: str
    series_number: str | int | None
    instance_number: int | None


def index_ct_series(
    dicom_root: Path,
    patient_ids: Optional[Iterable[str]] = None,
    max_workers: Optional[int] = None,
) -> Dict[str, Dict[str, Dict[str, List[CTInstance]]]]:
    """
    Returns nested dict: patient_id -> study_uid -> series_uid -> [CTInstance...]

    When ``patient_ids`` is provided, only those top-level patient directories
    are indexed (cohort-scoped); otherwise the whole ``dicom_root`` is scanned.

    ``max_workers`` controls how many threads read DICOM headers concurrently in
    the generic fallback loop below (NFS-backed trees are latency- not
    CPU-bound, so concurrent reads win heavily over one file at a time).
    Defaults to ``RTPIPELINE_INDEX_WORKERS`` (see ``utils.DEFAULT_INDEX_WORKERS``)
    when not given. ``max_workers=1`` reproduces the exact single-threaded
    behaviour; the final per-series sort below makes the result independent of
    the order files are read in, so parallel reads cannot change the output.
    """
    fast_index = _index_tcia_patient_series_layout(dicom_root, patient_ids)
    if fast_index is not None:
        return fast_index

    paths: List[Path] = []
    for base, _, files in _scoped_walk(dicom_root, patient_ids):
        for name in files:
            paths.append(Path(base) / name)

    workers = max_workers if max_workers is not None else DEFAULT_INDEX_WORKERS
    datasets = parallel_map_files(paths, read_dicom, workers)

    index: Dict[str, Dict[str, Dict[str, List[CTInstance]]]] = {}
    for p, ds in zip(paths, datasets):
        if ds is None:
            continue
        if getattr(ds, "Modality", None) != "CT":
            continue
        pid = str(get(ds, (0x0010, 0x0020), ""))
        study_uid = str(get(ds, (0x0020, 0x000D), ""))
        series_uid = str(get(ds, (0x0020, 0x000E), ""))
        series_num = get(ds, (0x0020, 0x0011))
        inst_num = get(ds, (0x0020, 0x0013))
        if not pid or not study_uid or not series_uid:
            continue
        entry = CTInstance(p, pid, study_uid, series_uid, series_num, int(inst_num) if inst_num is not None else None)
        index.setdefault(pid, {}).setdefault(study_uid, {}).setdefault(series_uid, []).append(entry)
    # sort by instance number
    for pid in index.values():
        for study in pid.values():
            for series in study.values():
                series.sort(key=lambda x: (x.instance_number is None, x.instance_number or 0))
    logger.info("Indexed CT series for %d patients", len(index))
    return index


def _looks_like_dicom(path: Path) -> bool:
    try:
        if path.stat().st_size < 132:
            return False
        with open(path, "rb") as handle:
            header = handle.read(132)
        return header[128:132] == b"DICM" or header[:2] in (b"\x08\x00", b"\x00\x08")
    except OSError:
        return False


def _index_tcia_patient_series_layout(
    dicom_root: Path,
    patient_ids: Optional[Iterable[str]] = None,
) -> Optional[Dict[str, Dict[str, Dict[str, List[CTInstance]]]]]:
    """Fast path for TCIA downloads laid out as PatientID/SeriesInstanceUID/files.

    The generic indexer reads every DICOM header. On NFS-backed TCIA trees this can
    be very slow. For the patient/series directory layout produced by the TCIA
    acquisition helper, one representative header per series is sufficient; slice
    ordering can be delegated to dcm2niix downstream.
    """

    try:
        patient_dirs = _scoped_patient_dirs(dicom_root, patient_ids)
    except OSError:
        return None
    if not patient_dirs:
        return None

    indexed: Dict[str, Dict[str, Dict[str, List[CTInstance]]]] = {}
    indexed_series = 0

    for patient_dir in patient_dirs:
        try:
            series_dirs = sorted(path for path in patient_dir.iterdir() if path.is_dir())
        except OSError:
            return None
        if not series_dirs:
            return None

        for series_dir in series_dirs:
            try:
                names = sorted(os.listdir(series_dir))
            except OSError:
                return None
            dicom_files = [series_dir / name for name in names if name.lower().endswith(".dcm")]
            if not dicom_files:
                dicom_files = [series_dir / name for name in names if _looks_like_dicom(series_dir / name)]
            if not dicom_files:
                continue
            ds = read_dicom(dicom_files[0])
            if ds is None:
                return None
            study_uid = str(get(ds, (0x0020, 0x000D), ""))
            series_uid = str(get(ds, (0x0020, 0x000E), "")) or series_dir.name
            # TCIA fast-path contract: each <patient>/<dir> is exactly ONE SeriesInstanceUID and the
            # directory is NAMED by that UID (see tcia_acquisition.py: series_dir = input_dir/pid/series_uid).
            # If the directory name does not match the series UID, this is NOT a genuine
            # one-series-per-directory tree — e.g. a STUDY-level directory holding many series and
            # modalities (common in non-TCIA hospital exports). Reading only the first file's header
            # would then lump the entire study into a single bogus CT "series". Bail to the generic
            # per-file indexer in index_ct_series(), which groups by real SeriesInstanceUID and filters
            # Modality==CT. Keying this on the directory-name contract (not the first file's modality)
            # also correctly bails for study dirs whose first sorted file happens to be non-CT.
            if series_dir.name != series_uid:
                logger.info(
                    "CT fast-path: directory %s does not match its SeriesInstanceUID; falling back "
                    "to generic per-series CT indexing for correctness.",
                    series_dir,
                )
                return None
            if getattr(ds, "Modality", None) != "CT":
                # Genuine single-series directory that is not CT (e.g. PET/MR series dir): skip it and
                # keep the fast path for the remaining CT series directories.
                continue
            pid = str(get(ds, (0x0010, 0x0020), "")) or patient_dir.name
            series_num = get(ds, (0x0020, 0x0011))
            if not pid or not study_uid or not series_uid:
                return None
            instances = [
                CTInstance(path=path, patient_id=pid, study_uid=study_uid, series_uid=series_uid, series_number=series_num, instance_number=None)
                for path in dicom_files
            ]
            indexed.setdefault(pid, {}).setdefault(study_uid, {})[series_uid] = instances
            indexed_series += 1

    if indexed_series == 0:
        return None
    logger.info(
        "Fast-indexed TCIA-style CT series for %d patients (%d series)",
        len(indexed),
        indexed_series,
    )
    return indexed


def _clear_dir(path: Path) -> None:
    """Remove all contents of ``path`` (files, symlinks, sub-directories) if it is a directory.

    No-op when ``path`` does not exist or is not a directory. Used to stop per-course
    CT-derived outputs from a previous (incorrect) run surviving a corrected re-run.

    A top-level symlink is treated as a no-op: deleting *through* a symlinked directory
    would remove contents OUTSIDE this tree. Course directories are real ``mkdir``'d paths,
    so this guard is defensive in practice but keeps the "no traversal outside the dir"
    guarantee honest.
    """
    if path.is_symlink() or not path.is_dir():
        return
    for item in path.iterdir():
        try:
            if item.is_dir() and not item.is_symlink():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink()
        except OSError as exc:
            logger.warning("_clear_dir: could not remove %s: %s", item, exc)


def copy_ct_series(
    series: List[CTInstance],
    dst_dir: Path,
    copy_manager: Optional["DicomCopyManager"] = None,
) -> None:
    """Copy CT series to destination directory with optional optimizations.

    Files are named CT_{index}.dcm where index is the instance_number if available
    and unique, otherwise a sequential index is used to prevent filename collisions.
    This handles vendors that set all InstanceNumbers to the same value or omit them.

    The destination directory is PURGED before copying so a re-run with a corrected
    (different) series selection cannot leave stale slices from a previous run behind;
    a mixed leftover folder would otherwise make dcm2niix fail to build one volume.
    Each instance is guaranteed to physically land at ``dst``: when ``copy_manager``
    SOP-deduplication returns an existing copy at a FOREIGN path (e.g. another course
    sharing the same CT, or the all-series materialisation), the file is still
    materialised at ``dst`` (hardlink, else copy) so this course's DICOM/CT folder is
    never left empty.
    """
    # FAIL-SAFE: never purge a populated destination when there is nothing to copy in
    # its place. An empty `series` means no CT instances were resolved for this course
    # (e.g. a wrong/misconfigured --dicom-root, or a transient source outage). Purging
    # here would empty an already-valid DICOM/CT folder and copy nothing, destroying a
    # good course's raw CT. Only purge when we will actually refill it.
    if not series:
        if dst_dir.is_dir() and any(dst_dir.iterdir()):
            logger.warning(
                "copy_ct_series: empty series for %s but destination is already "
                "populated; refusing to purge (fail-safe against source-config errors).",
                dst_dir,
            )
        return
    # Purge stale contents so a corrected re-run does not inherit a prior garbage-bag CT.
    _clear_dir(dst_dir)
    ensure_dir(dst_dir)
    # Track used indices to prevent collisions when InstanceNumber is not unique
    used_indices: set[int] = set()
    for idx, inst in enumerate(series):
        # Prefer instance_number if valid and not already used, else use enumeration index
        if inst.instance_number is not None and inst.instance_number not in used_indices:
            file_idx = inst.instance_number
        else:
            # Find next available index starting from enumeration position
            file_idx = idx
            while file_idx in used_indices:
                file_idx += 1
        used_indices.add(file_idx)
        dst = dst_dir / f"CT_{file_idx:05d}.dcm"
        if copy_manager is not None:
            actual, _copied = copy_manager.copy_dicom(inst.path, dst, skip_if_exists=True)
            # SOP dedup may return a copy at a foreign path without placing a file at dst.
            # The per-course CT folder MUST contain real files for dcm2niix, so materialise dst.
            if Path(actual) != dst and not dst.exists():
                try:
                    os.link(actual, dst)
                except OSError:
                    shutil.copy2(actual, dst)
        else:
            shutil.copy2(inst.path, dst)


def pick_primary_series(series_map: Dict[str, List[CTInstance]]) -> Optional[List[CTInstance]]:
    """Pick a representative series (heuristic: largest number of slices)."""
    if not series_map:
        return None
    best = max(series_map.values(), key=lambda lst: len(lst))
    return best
