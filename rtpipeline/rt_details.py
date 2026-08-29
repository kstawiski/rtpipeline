from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pydicom
from pydicom.dataset import FileDataset

from .utils import read_dicom, get, _scoped_walk, parallel_map_files, DEFAULT_INDEX_WORKERS

logger = logging.getLogger(__name__)

DEFAULT_ROI_FAMILY_NAMES = ("GTV", "CTV", "PTV", "BLADDER", "RECTUM")

_TARGET_VOLUME_TOKEN = re.compile(
    r"(?<![A-Z0-9])(?:" + "|".join(DEFAULT_ROI_FAMILY_NAMES[:3]) + r")",
    re.IGNORECASE,
)
_LEADING_MARGIN_HELPER = re.compile(r"^\s*MARG(?:\b|[\s_.-])", re.IGNORECASE)
_LEADING_Z_HELPER = re.compile(
    r"^\s*Z[\s_.-]*(?:GTV|CTV|PTV)",
    re.IGNORECASE,
)
_BOOLEAN_CROP_SEPARATOR = re.compile(r"\s+-\s*$")


@dataclass
class PlanInfo:
    path: Path
    patient_id: str
    sop_instance_uid: str
    study_uid: str | None
    plan_label: str | None
    plan_name: str | None
    plan_date: str | None
    frame_of_reference_uid: str | None
    referenced_struct_sop: str | None = None


@dataclass
class DoseInfo:
    path: Path
    patient_id: str
    sop_instance_uid: str
    study_uid: str | None
    frame_of_reference_uid: str | None
    referenced_plan_sop: str | None
    referenced_plan_sops: tuple[str, ...] = ()


@dataclass
class StructInfo:
    path: Path
    patient_id: str
    sop_instance_uid: str
    study_uid: str | None
    frame_of_reference_uid: str | None
    roi_names: List[str]


def is_target_volume_name(name: object) -> bool:
    """Return whether an ROI name denotes a target rather than a helper or crop."""
    text = str(name or "").strip()
    if not text:
        return False
    if _LEADING_MARGIN_HELPER.search(text) or _LEADING_Z_HELPER.search(text):
        return False
    for match in _TARGET_VOLUME_TOKEN.finditer(text):
        if _BOOLEAN_CROP_SEPARATOR.search(text[: match.start()]):
            continue
        return True
    return False


def target_volume_names(roi_names: List[str]) -> List[str]:
    """Return left-token-bounded GTV, CTV, and PTV names after helper exclusions."""
    return [name for name in roi_names if is_target_volume_name(name)]


def has_target_volumes(roi_names: List[str]) -> bool:
    """Return whether a structure set contains at least one target volume."""
    return bool(target_volume_names(roi_names))


def _frame_uid(ds: FileDataset) -> str | None:
    """Read the standard frame UID, then the legacy referenced-frame tag."""
    return str(get(ds, (0x0020, 0x0052)) or get(ds, (0x3006, 0x0024)) or "") or None


def _referenced_sop_uids(ds: FileDataset, sequence_name: str) -> tuple[str, ...]:
    values: List[str] = []
    for item in getattr(ds, sequence_name, []) or []:
        uid = str(getattr(item, "ReferencedSOPInstanceUID", "") or "")
        if uid and uid not in values:
            values.append(uid)
    return tuple(values)


def _safe_roi_names(ds: FileDataset) -> List[str]:
    names: List[str] = []
    try:
        seq = ds.StructureSetROISequence
        for roi in seq:
            nm = getattr(roi, "ROIName", None)
            if nm:
                names.append(str(nm))
    except Exception:
        pass
    return names


def extract_rt(
    dicom_root: Path,
    patient_ids: Optional[Iterable[str]] = None,
    max_workers: Optional[int] = None,
) -> tuple[List[PlanInfo], List[DoseInfo], List[StructInfo]]:
    """Scan every file under ``dicom_root`` (cohort-scoped when ``patient_ids`` is
    given) and classify RTPLAN/RTDOSE/RTSTRUCT by ``Modality`` -- ALL files are
    scanned (not just RT-named ones), since RT objects are not required to be
    named RT*.

    ``max_workers`` controls how many threads read DICOM headers concurrently
    (defaults to ``utils.DEFAULT_INDEX_WORKERS``; ``max_workers=1`` reproduces
    the exact single-threaded behaviour). The per-file reads are parallelized,
    but classification into ``plans``/``doses``/``structs`` below is done in a
    single pass over the results in the SAME order the files were discovered
    in, so the output does not depend on thread completion order.
    """
    plans: List[PlanInfo] = []
    doses: List[DoseInfo] = []
    structs: List[StructInfo] = []

    paths: List[Path] = []
    for base, _, files in _scoped_walk(dicom_root, patient_ids):
        for name in files:
            paths.append(Path(base) / name)

    workers = max_workers if max_workers is not None else DEFAULT_INDEX_WORKERS
    datasets = parallel_map_files(paths, read_dicom, workers)

    for p, ds in zip(paths, datasets):
        if ds is None:
            continue
        modality = getattr(ds, "Modality", None)
        if modality == "RTPLAN":
            ref_structs = _referenced_sop_uids(ds, "ReferencedStructureSetSequence")
            plans.append(
                PlanInfo(
                    path=p,
                    patient_id=str(get(ds, (0x0010, 0x0020), "")),
                    sop_instance_uid=str(get(ds, (0x0008, 0x0018), "")),
                    study_uid=str(get(ds, (0x0020, 0x000D), "")) or None,
                    plan_label=get(ds, (0x300A, 0x0002)),
                    plan_name=get(ds, (0x300A, 0x0003)),
                    plan_date=get(ds, (0x300A, 0x0006)),
                    frame_of_reference_uid=_frame_uid(ds),
                    referenced_struct_sop=ref_structs[0] if ref_structs else None,
                )
            )
        elif modality == "RTDOSE":
            # The first reference preserves sequence order for legacy callers;
            # all references are retained for ambiguity logging and PLAN_SUM use.
            ref_plan_uids = _referenced_sop_uids(ds, "ReferencedRTPlanSequence")
            ref_plan_uid = ref_plan_uids[0] if ref_plan_uids else None
            doses.append(
                DoseInfo(
                    path=p,
                    patient_id=str(get(ds, (0x0010, 0x0020), "")),
                    sop_instance_uid=str(get(ds, (0x0008, 0x0018), "")),
                    study_uid=str(get(ds, (0x0020, 0x000D), "")) or None,
                    frame_of_reference_uid=_frame_uid(ds),
                    referenced_plan_sop=ref_plan_uid,
                    referenced_plan_sops=ref_plan_uids,
                )
            )
        elif modality == "RTSTRUCT":
            structs.append(
                StructInfo(
                    path=p,
                    patient_id=str(get(ds, (0x0010, 0x0020), "")),
                    sop_instance_uid=str(get(ds, (0x0008, 0x0018), "")),
                    study_uid=str(get(ds, (0x0020, 0x000D), "")) or None,
                    frame_of_reference_uid=_frame_uid(ds),
                    roi_names=_safe_roi_names(ds),
                )
            )
    # Basic logs
    logger.info("Found RTPLAN=%d, RTDOSE=%d, RTSTRUCT=%d", len(plans), len(doses), len(structs))
    return plans, doses, structs

