"""Per-course campaign ledger for resumable, crash-safe cohort runs.

Each course/stage outcome is written as one small JSON record under
``_campaign_ledger/records``. Records are published with ``os.replace`` so a
killed process can never leave a half-written record, and one file per unit
means parallel Snakemake jobs never contend for the same path. The rollup
reconstructs campaign state from those records plus the sentinels already on
disk, so a resumed run knows what completed and an aggregation step can report
honest denominators for what did not.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from rtpipeline.organize_ledger import ledger_path as organize_ledger_path
from rtpipeline.organize_ledger import read_organize_ledger

LEDGER_DIRNAME = "_campaign_ledger"
RECORDS_DIRNAME = "records"

STAGE_SENTINELS = {
    "organize": ".organized",
    "segmentation": ".segmentation_done",
    "custom_models": ".custom_models_done",
    "crop_ct": ".crop_ct_done",
    "dvh": ".dvh_done",
    "radiomics": ".radiomics_done",
    "radiomics_robustness": ".radiomics_robustness_done",
    "qc": ".qc_done",
}

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_DISABLED = "disabled"
STATUS_MISSING = "missing"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ledger_dir(output_dir: Path) -> Path:
    return Path(output_dir) / LEDGER_DIRNAME


def records_dir(output_dir: Path) -> Path:
    return ledger_dir(output_dir) / RECORDS_DIRNAME


def _unit_key(patient: str, course: str, stage: str) -> str:
    safe = [part.replace(os.sep, "_").replace("__", "_") for part in (patient, course, stage)]
    return "__".join(safe)


def _write_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def record(
    output_dir: Path,
    patient: str,
    course: str,
    stage: str,
    status: str,
    *,
    returncode: int | None = None,
    log_path: str | None = None,
    detail: str | None = None,
    started_at: str | None = None,
    duration_seconds: float | None = None,
) -> Path:
    """Publish one course/stage outcome record atomically."""
    entry = {
        "patient": patient,
        "course": course,
        "stage": stage,
        "status": status,
        "returncode": returncode,
        "log_path": log_path,
        "detail": detail,
        "started_at": started_at,
        "finished_at": _utcnow(),
        "duration_seconds": duration_seconds,
    }
    target = records_dir(output_dir) / f"{_unit_key(patient, course, stage)}.json"
    _write_atomic(target, json.dumps(entry, indent=2, sort_keys=True) + "\n")
    return target


def _load_records(output_dir: Path) -> dict[tuple[str, str, str], dict]:
    entries: dict[tuple[str, str, str], dict] = {}
    directory = records_dir(output_dir)
    if not directory.is_dir():
        return entries
    for path in sorted(directory.glob("*.json")):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        key = (
            str(entry.get("patient") or ""),
            str(entry.get("course") or ""),
            str(entry.get("stage") or ""),
        )
        if all(key):
            entries[key] = entry
    return entries


def _sentinel_status(course_path: Path, suffix: str) -> str:
    sentinel = course_path / suffix
    if not sentinel.exists():
        return STATUS_MISSING
    try:
        value = sentinel.read_text(encoding="utf-8").strip().lower()
    except Exception:
        return STATUS_MISSING
    if value == STATUS_DISABLED:
        return STATUS_DISABLED
    return STATUS_OK if value == STATUS_OK else STATUS_MISSING


def _iter_courses(output_dir: Path):
    skip_names = {"Data", "Data_Snakemake_fallback", "Logs_Snakemake_fallback", "_RESULTS"}
    root = Path(output_dir)
    if not root.is_dir():
        return
    for patient_dir in sorted(root.iterdir()):
        if not patient_dir.is_dir():
            continue
        if patient_dir.name.startswith(("_", ".")) or patient_dir.name in skip_names:
            continue
        for course_path in sorted(patient_dir.iterdir()):
            if course_path.is_dir() and not course_path.name.startswith("_"):
                yield patient_dir.name, course_path.name, course_path


def rollup(output_dir: Path) -> dict:
    """Reconstruct campaign state from records and on-disk sentinels."""
    output_dir = Path(output_dir)
    recorded = _load_records(output_dir)
    rows: list[dict] = []

    for patient, course, course_path in _iter_courses(output_dir):
        for stage, suffix in STAGE_SENTINELS.items():
            key = (patient, course, stage)
            entry = recorded.get(key)
            sentinel_state = _sentinel_status(course_path, suffix)

            if entry is not None:
                status = str(entry.get("status") or STATUS_MISSING)
                source = "record"
                # A record claiming success without a sentinel is not success.
                if status == STATUS_OK and sentinel_state == STATUS_MISSING:
                    status = STATUS_FAILED
                    source = "record-sentinel-conflict"
            else:
                status = sentinel_state
                source = "sentinel"

            rows.append(
                {
                    "patient": patient,
                    "course": course,
                    "stage": stage,
                    "status": status,
                    "status_source": source,
                    "sentinel_status": sentinel_state,
                    "returncode": (entry or {}).get("returncode"),
                    "log_path": (entry or {}).get("log_path"),
                    "detail": (entry or {}).get("detail"),
                    "finished_at": (entry or {}).get("finished_at"),
                    "duration_seconds": (entry or {}).get("duration_seconds"),
                }
            )

    visible_course_count = len(
        {(row["patient"], row["course"]) for row in rows}
    )
    organize_ledger = None
    if organize_ledger_path(output_dir).is_file():
        organize_ledger = read_organize_ledger(output_dir)
    if organize_ledger is not None:
        existing_organize = {
            (row["patient"], row["course"])
            for row in rows
            if row["stage"] == "organize"
        }
        for entry in organize_ledger["technical_quarantines"]:
            identity = (entry["patient"], entry["course"])
            if identity in existing_organize:
                continue
            rows.append(
                {
                    "patient": entry["patient"],
                    "course": entry["course"],
                    "stage": "organize",
                    "status": STATUS_FAILED,
                    "status_source": "organize-ledger",
                    "sentinel_status": STATUS_MISSING,
                    "returncode": None,
                    "log_path": None,
                    "detail": entry["reason"],
                    "finished_at": organize_ledger.get("generated_at"),
                    "duration_seconds": None,
                    "disposition_type": "technical_quarantine",
                    "clinical_exclusion": False,
                }
            )

    by_stage: dict[str, dict[str, int]] = {}
    for row in rows:
        counts = by_stage.setdefault(row["stage"], {})
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    courses = {(row["patient"], row["course"]) for row in rows}
    failed_units = sorted(
        {(row["patient"], row["course"], row["stage"]) for row in rows if row["status"] == STATUS_FAILED}
    )

    summary = {
        "generated_at": _utcnow(),
        "output_dir": str(output_dir),
        "course_count": len(courses),
        "stage_counts": by_stage,
        "failed_units": [
            {"patient": p, "course": c, "stage": s} for p, c, s in failed_units
        ],
        "failed_unit_count": len(failed_units),
        "courses_with_any_failure": len({(p, c) for p, c, _ in failed_units}),
    }
    if organize_ledger is not None:
        summary.update(
            {
                "visible_course_count": visible_course_count,
                "course_count": organize_ledger["intended_course_count"],
                "intended_course_count": organize_ledger["intended_course_count"],
                "attempted_course_count": organize_ledger["attempted_course_count"],
                "producer_validated_course_count": organize_ledger[
                    "validated_course_count"
                ],
                "technical_quarantine_count": organize_ledger[
                    "technical_quarantine_count"
                ],
                "technical_quarantines": organize_ledger[
                    "technical_quarantines"
                ],
            }
        )

    target_dir = ledger_dir(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "patient",
        "course",
        "stage",
        "status",
        "status_source",
        "sentinel_status",
        "returncode",
        "log_path",
        "detail",
        "finished_at",
        "duration_seconds",
        "disposition_type",
        "clinical_exclusion",
    ]
    buffer = []
    for row in rows:
        buffer.append({name: row.get(name) for name in fieldnames})
    import io

    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(buffer)
    _write_atomic(target_dir / "campaign_ledger.csv", stream.getvalue())
    _write_atomic(
        target_dir / "campaign_summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="publish one course/stage outcome")
    rec.add_argument("--output-dir", required=True)
    rec.add_argument("--patient", required=True)
    rec.add_argument("--course", required=True)
    rec.add_argument("--stage", required=True)
    rec.add_argument("--status", required=True, choices=[STATUS_OK, STATUS_FAILED, STATUS_DISABLED])
    rec.add_argument("--returncode", type=int, default=None)
    rec.add_argument("--log-path", default=None)
    rec.add_argument("--detail", default=None)

    roll = sub.add_parser("rollup", help="rebuild campaign ledger and summary")
    roll.add_argument("--output-dir", required=True)

    args = parser.parse_args(argv)

    if args.command == "record":
        path = record(
            Path(args.output_dir),
            args.patient,
            args.course,
            args.stage,
            args.status,
            returncode=args.returncode,
            log_path=args.log_path,
            detail=args.detail,
        )
        print(path)
        return 0

    summary = rollup(Path(args.output_dir))
    print(json.dumps(summary["stage_counts"], indent=2, sort_keys=True))
    print(
        f"courses={summary['course_count']} "
        f"failed_units={summary['failed_unit_count']} "
        f"courses_with_any_failure={summary['courses_with_any_failure']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
