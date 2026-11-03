"""Generate explicit concurrent-component declarations from selected ARIA plans.

Usage: python concurrent_declarations.py --history history.csv
       --selection selection.json --output declarations.json --declared-on YYYY-MM-DD
Only synthetic fixtures accompany this program. Console output contains counts only.
"""
from __future__ import annotations

import argparse
import csv
import datetime
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import sys


REQUIRED_COLUMNS = {
    "case_id", "plan_uid", "course_id", "included_in_course", "plan_status",
    "fractions_planned", "fractions_treated", "refpoint_id", "refpoint_planned",
    "refpoint_delivered", "first_day_of_treatment", "last_day_of_treatment",
}


def _positive_fractions(value: str) -> int:
    number = Decimal(value)
    if not number.is_finite() or number <= 0 or number != number.to_integral_value():
        raise ValueError("invalid fraction count")
    return int(number)


def generate_declarations(history: Path, selection_path: Path, declared_on: str,
                          issued_by: str = 'Lucjan concurrent_declarations.py') -> tuple[dict, dict]:
    """Accept an entire selected case group or skip it; never choose a subset.

    Duplicate selected history rows, missing plans, mixed courses and excluded
    plans leave the case undeclared. Fraction and reference-point values remain
    plan evidence; no course dose or course fraction total is calculated here.
    """
    if datetime.date.fromisoformat(declared_on).isoformat() != declared_on:
        raise ValueError("declared_on must be YYYY-MM-DD")
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    if not isinstance(selection, dict) or any(
        not isinstance(case, str) or not case.strip() or case != case.strip() or not isinstance(uids, list)
        or any(not isinstance(uid, str) or not uid.strip() or uid != uid.strip() for uid in uids)
        or len(set(uids)) != len(uids)
        for case, uids in selection.items()
    ):
        raise ValueError("invalid selection")
    rows_by_plan: dict[tuple[str, str], list[dict]] = {}
    with Path(history).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not REQUIRED_COLUMNS <= set(reader.fieldnames or []):
            raise ValueError("missing CSV columns")
        rows_read = 0
        for row in reader:
            rows_read += 1
            case, uid = ((row.get(key) or "").strip() for key in ("case_id", "plan_uid"))
            if case in selection and uid in selection[case]:
                rows_by_plan.setdefault((case, uid), []).append(
                    {key: (value or "").strip() for key, value in row.items() if key is not None}
                )
    declarations = []
    for case, uids in selection.items():
        if len(uids) < 2 or any(len(rows_by_plan.get((case, uid), [])) != 1 for uid in uids):
            continue
        rows = [rows_by_plan[(case, uid)][0] for uid in uids]
        if any(row["plan_status"] != "Completed" or
               row["included_in_course"].lower() not in {"1", "true", "yes"} for row in rows):
            continue
        courses = {row["course_id"] for row in rows}
        first = {row["first_day_of_treatment"] for row in rows}
        last = {row["last_day_of_treatment"] for row in rows}
        refs = {row["refpoint_id"] for row in rows}
        if (len(courses) != 1 or "" in courses or len(first) != 1 or "" in first
                or len(last) != 1 or "" in last or len(refs) != len(rows) or "" in refs):
            continue
        try:  # treatment days are ISO dates (YYYY-MM-DD), first <= last
            first_day, last_day = (datetime.date.fromisoformat(next(iter(x))) for x in (first, last))
        except ValueError:
            continue
        if first_day.isoformat() != next(iter(first)) or last_day.isoformat() != next(iter(last)) or first_day > last_day:
            continue
        try:
            planned = [_positive_fractions(row["fractions_planned"]) for row in rows]
            treated = [_positive_fractions(row["fractions_treated"]) for row in rows]
        except (ValueError, InvalidOperation):
            continue
        if planned != treated:
            continue
        declarations.append({
            "plan_sop_instance_uids": uids,
            "basis": (
                "ARIA plan history: all selected plans in one case and course "
                f"({next(iter(courses))}) are included_in_course and Completed; "
                "each is fully treated (fractions_treated = fractions_planned > 0; "
                f"per-plan counts in UID order: {', '.join(map(str, planned))}); "
                f"identical first treatment day {next(iter(first))} and last treatment day "
                f"{next(iter(last))}; distinct nonempty refpoint_id for every plan"
            ),
            "declared_on": declared_on,
            "issued_by": issued_by,
        })
    return {"declarations": declarations}, {
        "rows_read": rows_read,
        "cases_selected": len(selection),
        "declarations_written": len(declarations),
        "cases_skipped": len(selection) - len(declarations),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--declared-on", required=True)
    parser.add_argument("--issued-by", required=True, help="who issues the declarations (recorded in each entry)")
    args = parser.parse_args(argv)
    try:
        if not args.issued_by.strip():
            raise ValueError('issued_by is empty')
        payload, counts = generate_declarations(args.history, args.selection, args.declared_on, args.issued_by.strip())
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, InvalidOperation, TypeError):
        print("errors=1", file=sys.stderr)
        return 2
    print(" ".join(f"{key}={value}" for key, value in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
