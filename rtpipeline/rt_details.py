from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pydicom
from pydicom.dataset import FileDataset

from .utils import read_dicom, get

logger = logging.getLogger(__name__)


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


@dataclass
class DoseInfo:
    path: Path
    patient_id: str
    sop_instance_uid: str
    study_uid: str | None
    frame_of_reference_uid: str | None
    referenced_plan_sop: str | None


@dataclass
class StructInfo:
    path: Path
    patient_id: str
    sop_instance_uid: str
    study_uid: str | None
    frame_of_reference_uid: str | None
    roi_names: List[str]


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


def extract_rt(dicom_root: Path) -> tuple[List[PlanInfo], List[DoseInfo], List[StructInfo]]:
    plans: List[PlanInfo] = []
    doses: List[DoseInfo] = []
    structs: List[StructInfo] = []

    for base, _, files in os.walk(dicom_root):
        for name in files:
            p = Path(base) / name
            ds = read_dicom(p)
            if ds is None:
                continue
            modality = getattr(ds, "Modality", None)
            if modality == "RTPLAN":
                plans.append(
                    PlanInfo(
                        path=p,
                        patient_id=str(get(ds, (0x0010, 0x0020), "")),
                        sop_instance_uid=str(get(ds, (0x0008, 0x0018), "")),
                        study_uid=str(get(ds, (0x0020, 0x000D), "")) or None,
                        plan_label=get(ds, (0x300A, 0x0002)),
                        plan_name=get(ds, (0x300A, 0x0003)),
                        plan_date=get(ds, (0x300A, 0x0006)),
                        frame_of_reference_uid=get(ds, (0x3006, 0x0024)),
                    )
                )
            elif modality == "RTDOSE":
                # Reference to RP is usually in ReferencedRTPlanSequence (300C, 0062)
                ref_plan_uid: Optional[str] = None
                try:
                    for ref in ds.ReferencedRTPlanSequence:
                        ref_plan_uid = getattr(ref, "ReferencedSOPInstanceUID", None)
                        if ref_plan_uid:
                            break
                except Exception:
                    pass
                doses.append(
                    DoseInfo(
                        path=p,
                        patient_id=str(get(ds, (0x0010, 0x0020), "")),
                        sop_instance_uid=str(get(ds, (0x0008, 0x0018), "")),
                        study_uid=str(get(ds, (0x0020, 0x000D), "")) or None,
                        frame_of_reference_uid=get(ds, (0x3006, 0x0024)),
                        referenced_plan_sop=ref_plan_uid,
                    )
                )
            elif modality == "RTSTRUCT":
                structs.append(
                    StructInfo(
                        path=p,
                        patient_id=str(get(ds, (0x0010, 0x0020), "")),
                        sop_instance_uid=str(get(ds, (0x0008, 0x0018), "")),
                        study_uid=str(get(ds, (0x0020, 0x000D), "")) or None,
                        frame_of_reference_uid=get(ds, (0x3006, 0x0024)),
                        roi_names=_safe_roi_names(ds),
                    )
                )
    # Basic logs
    logger.info("Found RTPLAN=%d, RTDOSE=%d, RTSTRUCT=%d", len(plans), len(doses), len(structs))
    return plans, doses, structs

