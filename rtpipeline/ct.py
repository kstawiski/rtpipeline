from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .utils import read_dicom, get, ensure_dir

logger = logging.getLogger(__name__)


@dataclass
class CTInstance:
    path: Path
    patient_id: str
    study_uid: str
    series_uid: str
    series_number: str | int | None
    instance_number: int | None


def index_ct_series(dicom_root: Path) -> Dict[str, Dict[str, Dict[str, List[CTInstance]]]]:
    """
    Returns nested dict: patient_id -> study_uid -> series_uid -> [CTInstance...]
    """
    index: Dict[str, Dict[str, Dict[str, List[CTInstance]]]] = {}
    for base, _, files in os.walk(dicom_root):
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


def copy_ct_series(series: List[CTInstance], dst_dir: Path) -> None:
    ensure_dir(dst_dir)
    for inst in series:
        dst = dst_dir / f"CT_{inst.instance_number or 0}.dcm"
        shutil.copy2(inst.path, dst)


def pick_primary_series(series_map: Dict[str, List[CTInstance]]) -> Optional[List[CTInstance]]:
    """Pick a representative series (heuristic: largest number of slices)."""
    if not series_map:
        return None
    best = max(series_map.values(), key=lambda lst: len(lst))
    return best

