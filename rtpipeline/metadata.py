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
    dose: DoseInfo
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
    # Index plans by SOP Instance UID for dose linking
    plan_by_sop: Dict[str, PlanInfo] = {p.sop_instance_uid: p for p in plans if p.sop_instance_uid}

    # Index structs by frame of reference for convenience
    structs_by_for: Dict[Tuple[str, str], StructInfo] = {}
    for s in structs:
        if s.frame_of_reference_uid:
            key = (s.patient_id, s.frame_of_reference_uid)
            # Prefer the first we see; if multiple, keep the earliest by Study UID string order
            if key not in structs_by_for:
                structs_by_for[key] = s

    linked: List[LinkedSet] = []
    for d in doses:
        p = plan_by_sop.get(d.referenced_plan_sop or "")
        if not p:
            continue
        s = structs_by_for.get((p.patient_id, d.frame_of_reference_uid or ""))
        # Prefer RS study (commonly the CT study), then RD, then RP
        ct_study_uid = (s.study_uid if s else None) or d.study_uid or p.study_uid
        linked.append(
            LinkedSet(
                patient_id=p.patient_id,
                plan=p,
                dose=d,
                struct=s,
                ct_study_uid=ct_study_uid,
                frame_of_reference_uid=d.frame_of_reference_uid or p.frame_of_reference_uid,
            )
        )

    logger.info("Linked %d Plan-Dose(-Struct) sets", len(linked))
    return linked


def group_by_course(
    linked: List[LinkedSet],
    merge_criteria: str = "same_ct_study",
    max_days_between_plans: Optional[int] = None,
) -> Dict[Tuple[str, str], List[LinkedSet]]:
    """
    Returns dict keyed by (patient_id, course_key) with value list of LinkedSet.
    course_key is ct_study_uid when merge_criteria == 'same_ct_study', else frame_of_reference_uid.

    Only merges within same course_key; optional time-window filter excludes outliers.
    """
    grouped: Dict[Tuple[str, str], List[LinkedSet]] = {}
    for item in linked:
        if merge_criteria == "frame_of_reference":
            key = item.frame_of_reference_uid or item.ct_study_uid or ""
        else:
            key = item.ct_study_uid or item.frame_of_reference_uid or ""
        if not key:
            # If nothing reliable, fallback to per-plan grouping (no merge)
            key = f"SOP:{item.plan.sop_instance_uid}"
        grouped.setdefault((item.patient_id, key), []).append(item)

    # Optional: time-window filter within groups based on plan dates
    if max_days_between_plans is not None:
        filtered: Dict[Tuple[str, str], List[LinkedSet]] = {}
        for gk, items in grouped.items():
            dates = [parse_date(it.plan.plan_date) for it in items if it.plan.plan_date]
            dates = [d for d in dates if d is not None]
            if not dates:
                filtered[gk] = items
                continue
            tmin, tmax = min(dates), max(dates)
            if (tmax - tmin).days <= max_days_between_plans:
                filtered[gk] = items
            else:
                # Split by date proximity: simple heuristic around earliest date
                cluster_a, cluster_b = [], []
                pivot = tmin
                for it in items:
                    dt = parse_date(it.plan.plan_date)
                    if dt and (dt - pivot).days <= max_days_between_plans:
                        cluster_a.append(it)
                    else:
                        cluster_b.append(it)
                if cluster_a:
                    filtered[gk] = cluster_a
                if cluster_b:
                    # Use a derived key to separate
                    pid, key = gk
                    filtered[(pid, f"{key}#late")] = cluster_b
        grouped = filtered

    # Logging summary
    per_patient: Dict[str, int] = {}
    for (pid, _), items in grouped.items():
        per_patient[pid] = per_patient.get(pid, 0) + 1
    for pid, n in per_patient.items():
        logger.info("Patient %s: %d course(s) detected", pid, n)

    return grouped
