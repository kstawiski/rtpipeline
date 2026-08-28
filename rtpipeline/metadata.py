from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .rt_details import PlanInfo, DoseInfo, StructInfo

logger = logging.getLogger(__name__)


@dataclass
class LinkedSet:
    patient_id: str
    plan: PlanInfo
    dose: Optional[DoseInfo]
    struct: Optional[StructInfo]
    # Derived
    ct_study_uid: Optional[str]
    frame_of_reference_uid: Optional[str]


def parse_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    # Accept common formats
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def link_rt_sets(plans: List[PlanInfo], doses: List[DoseInfo], structs: List[StructInfo]) -> List[LinkedSet]:
    """Resolve RD→RP→RS using explicit DICOM references.

    StudyInstanceUID and FrameOfReferenceUID are geometry context, not
    structure-set identity. They can be shared by setup contours, auto-contours,
    and several clinical courses, so they are never used as an attachment
    fallback here.
    """
    plan_by_sop: Dict[Tuple[str, str], PlanInfo] = {
        (p.patient_id, p.sop_instance_uid): p for p in plans if p.sop_instance_uid
    }
    struct_by_sop: Dict[Tuple[str, str], StructInfo] = {
        (s.patient_id, s.sop_instance_uid): s for s in structs if s.sop_instance_uid
    }

    linked: List[LinkedSet] = []
    for d in doses:
        referenced_plan_uids = tuple(
            dict.fromkeys(d.referenced_plan_sops or ((d.referenced_plan_sop,) if d.referenced_plan_sop else ()))
        )
        resolved_plans: List[PlanInfo] = []
        for plan_uid in referenced_plan_uids:
            plan = plan_by_sop.get((d.patient_id, plan_uid))
            if plan is None:
                logger.error(
                    "Unresolved authoritative RTDOSE→RTPLAN reference: patient=%s dose=%s "
                    "referenced_plan=%s",
                    d.patient_id,
                    d.sop_instance_uid,
                    plan_uid,
                )
            else:
                resolved_plans.append(plan)
        if not referenced_plan_uids:
            logger.error(
                "RTDOSE has no ReferencedRTPlanSequence: patient=%s dose=%s path=%s",
                d.patient_id,
                d.sop_instance_uid,
                d.path,
            )
        if not resolved_plans:
            continue

        referenced_struct_uids = {
            plan.referenced_struct_sop for plan in resolved_plans if plan.referenced_struct_sop
        }
        if len(referenced_struct_uids) > 1:
            logger.error(
                "RTDOSE references plans from multiple authoritative structure sets; "
                "dose not attached: patient=%s dose=%s structures=%s",
                d.patient_id,
                d.sop_instance_uid,
                sorted(referenced_struct_uids),
            )
            continue

        for p in resolved_plans:
            s = None
            if p.referenced_struct_sop:
                s = struct_by_sop.get((p.patient_id, p.referenced_struct_sop))
                if s is None:
                    logger.error(
                        "Unresolved authoritative RTPLAN→RTSTRUCT reference: patient=%s plan=%s "
                        "referenced_struct=%s",
                        p.patient_id,
                        p.sop_instance_uid,
                        p.referenced_struct_sop,
                    )
            else:
                logger.error(
                    "RTPLAN has no ReferencedStructureSetSequence: patient=%s plan=%s path=%s",
                    p.patient_id,
                    p.sop_instance_uid,
                    p.path,
                )
            ct_study_uid = (s.study_uid if s else None) or d.study_uid or p.study_uid
            linked.append(
                LinkedSet(
                    patient_id=p.patient_id,
                    plan=p,
                    dose=d,
                    struct=s,
                    ct_study_uid=ct_study_uid,
                    frame_of_reference_uid=(
                        (s.frame_of_reference_uid if s else None)
                        or d.frame_of_reference_uid
                        or p.frame_of_reference_uid
                    ),
                )
            )

    linked_plan_uids = {(item.patient_id, item.plan.sop_instance_uid) for item in linked}
    for p in plans:
        if (p.patient_id, p.sop_instance_uid) in linked_plan_uids:
            continue
        s = (
            struct_by_sop.get((p.patient_id, p.referenced_struct_sop))
            if p.referenced_struct_sop
            else None
        )
        if p.referenced_struct_sop and s is None:
            logger.error(
                "Unresolved authoritative RTPLAN→RTSTRUCT reference for plan without a linked dose: "
                "patient=%s plan=%s referenced_struct=%s",
                p.patient_id,
                p.sop_instance_uid,
                p.referenced_struct_sop,
            )
        logger.warning(
            "RTPLAN has no resolved RTDOSE; retaining a plan-only course candidate: patient=%s plan=%s",
            p.patient_id,
            p.sop_instance_uid,
        )
        linked.append(
            LinkedSet(
                patient_id=p.patient_id,
                plan=p,
                dose=None,
                struct=s,
                ct_study_uid=(s.study_uid if s else None) or p.study_uid,
                frame_of_reference_uid=(
                    (s.frame_of_reference_uid if s else None)
                    or p.frame_of_reference_uid
                ),
            )
        )

    logger.info("Linked %d Plan-Dose(-Struct) sets", len(linked))
    return linked


def group_by_course(
    linked: List[LinkedSet],
    merge_criteria: str = "same_ct_study",
    max_days_between_plans: Optional[int] = None,
) -> Dict[Tuple[str, str], List[LinkedSet]]:
    """Group by the plan-referenced RTSTRUCT SOPInstanceUID.

    The legacy ``merge_criteria`` and ``max_days_between_plans`` parameters are
    accepted for API compatibility but cannot override an explicit DICOM
    reference chain. Plans that reference one structure set remain one course.
    Different referenced structure sets remain different courses even when they
    share a study, frame, or date.
    """
    if merge_criteria != "same_ct_study" or max_days_between_plans is not None:
        logger.info(
            "Ignoring legacy course merge settings because referenced RTSTRUCT identity is authoritative"
        )
    grouped: Dict[Tuple[str, str], List[LinkedSet]] = {}
    for item in linked:
        key = (
            item.struct.sop_instance_uid
            if item.struct is not None
            else item.plan.referenced_struct_sop
            or f"UNRESOLVED_PLAN:{item.plan.sop_instance_uid}"
        )
        grouped.setdefault((item.patient_id, key), []).append(item)

    # Logging summary
    per_patient: Dict[str, int] = {}
    for (pid, _), items in grouped.items():
        per_patient[pid] = per_patient.get(pid, 0) + 1
    for pid, n in per_patient.items():
        logger.info("Patient %s: %d course(s) detected", pid, n)

    return grouped
