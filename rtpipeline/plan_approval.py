"""Fail-closed RTPLAN authority, separate from archived planning evidence.

Approval is necessary, not evidence of delivery. Only the exact standard
APPROVED value confers eligibility. Missing, empty and unknown values do not.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pydicom


def approval_status(dataset) -> str:
    value = str(getattr(dataset, "ApprovalStatus", "") or "").strip()
    return value or "ABSENT"


def approval_disposition(status: str) -> str:
    return {
        "APPROVED": "eligible_approved",
        "REJECTED": "ineligible_rejected",
        "UNAPPROVED": "ineligible_unapproved",
        "ABSENT": "ineligible_approval_status_absent",
        "UNREADABLE": "ineligible_plan_unreadable",
    }.get(status, "ineligible_approval_status_unknown")


def plan_approval(path: Path) -> dict:
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        status = approval_status(ds)
        uid = str(getattr(ds, "SOPInstanceUID", "") or "")
        date = str(getattr(ds, "RTPlanDate", "") or "")
    except Exception:
        status, uid, date = "UNREADABLE", "", ""
    return {"sop_instance_uid": uid, "approval_status": status,
            "disposition": approval_disposition(status), "plan_date": date}


def approved_plan_paths(paths: Iterable[Path]) -> list[Path]:
    return [Path(p) for p in paths if plan_approval(p)["approval_status"] == "APPROVED"]


def approval_audit(paths: Iterable[Path]) -> dict:
    plans = [plan_approval(p) for p in dict.fromkeys(paths)]
    approved = [p for p in plans if p["approval_status"] == "APPROVED"]
    statuses = {p["approval_status"] for p in plans}
    if approved:
        disposition = "eligible_approved_plan_available"
    elif len(statuses) == 1:
        disposition = "no_approved_plan_" + {
            "REJECTED": "rejected_only", "UNAPPROVED": "unapproved_only",
            "ABSENT": "approval_status_absent", "UNREADABLE": "unreadable",
        }.get(next(iter(statuses)), "approval_status_unknown")
    else:
        disposition = "no_approved_plan_mixed_status" if plans else "no_plan"
    return {"policy": "approved_only_v1", "disposition": disposition,
            "approved_plan_count": len(approved), "plans": plans}


def eligible_dose_paths(plan_paths: Iterable[Path], dose_paths: Iterable[Path]) -> list[Path]:
    """Never salvage a composite grid containing an ineligible/unknown plan."""
    approved = {p["sop_instance_uid"] for p in map(plan_approval, plan_paths)
                if p["approval_status"] == "APPROVED" and p["sop_instance_uid"]}
    selected = []
    for path in dose_paths:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
            refs = {str(getattr(r, "ReferencedSOPInstanceUID", "") or "")
                    for r in getattr(ds, "ReferencedRTPlanSequence", [])}
            if refs and "" not in refs and refs <= approved:
                selected.append(Path(path))
        except Exception:
            continue
    return selected


def approved_course_start(items):
    """Keep the RTSTRUCT grouping and minimum-plan-date rule, restrict eligibility."""
    from .metadata import parse_date
    dates = [parse_date(item.plan.plan_date) for item in items
             if item.plan.approval_status == "APPROVED"]
    return min((date for date in dates if date is not None), default=None)
