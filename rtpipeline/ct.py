from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

from .utils import read_dicom, get, ensure_dir, _scoped_walk, _scoped_patient_dirs

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
) -> Dict[str, Dict[str, Dict[str, List[CTInstance]]]]:
    """
    Returns nested dict: patient_id -> study_uid -> series_uid -> [CTInstance...]

    When ``patient_ids`` is provided, only those top-level patient directories
    are indexed (cohort-scoped); otherwise the whole ``dicom_root`` is scanned.
    """
    fast_index = _index_tcia_patient_series_layout(dicom_root, patient_ids)
    if fast_index is not None:
        return fast_index

    index: Dict[str, Dict[str, Dict[str, List[CTInstance]]]] = {}
    for base, _, files in _scoped_walk(dicom_root, patient_ids):
        for name in files:
            p = Path(base) / name
            ds = read_dicom(p)
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
            if getattr(ds, "Modality", None) != "CT":
                continue
            pid = str(get(ds, (0x0010, 0x0020), "")) or patient_dir.name
            study_uid = str(get(ds, (0x0020, 0x000D), ""))
            series_uid = str(get(ds, (0x0020, 0x000E), "")) or series_dir.name
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


def copy_ct_series(
    series: List[CTInstance],
    dst_dir: Path,
    copy_manager: Optional["DicomCopyManager"] = None,
) -> None:
    """Copy CT series to destination directory with optional optimizations.

    Files are named CT_{index}.dcm where index is the instance_number if available
    and unique, otherwise a sequential index is used to prevent filename collisions.
    This handles vendors that set all InstanceNumbers to the same value or omit them.
    """
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
            copy_manager.copy_dicom(inst.path, dst, skip_if_exists=True)
        else:
            shutil.copy2(inst.path, dst)


def pick_primary_series(series_map: Dict[str, List[CTInstance]]) -> Optional[List[CTInstance]]:
    """Pick a representative series (heuristic: largest number of slices)."""
    if not series_map:
        return None
    best = max(series_map.values(), key=lambda lst: len(lst))
    return best
