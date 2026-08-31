from __future__ import annotations

"""Fail-closed publication ledger for organized radiotherapy courses."""

import datetime
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable


ORGANIZE_LEDGER_SCHEMA = "rtpipeline-organize-ledger-v1"
ORGANIZE_LEDGER_RELATIVE_PATH = Path("_COURSES") / "organize_ledger.json"
STATUS_VALIDATED = "validated"
STATUS_TECHNICAL_QUARANTINE = "technical_quarantine"
_VALID_STATUSES = {STATUS_VALIDATED, STATUS_TECHNICAL_QUARANTINE}


class OrganizeLedgerError(RuntimeError):
    """The organize ledger is missing, malformed, or internally inconsistent."""


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def ledger_path(output_root: Path | str) -> Path:
    return Path(output_root) / ORGANIZE_LEDGER_RELATIVE_PATH


def _course_identity(entry: dict[str, Any]) -> tuple[str, str]:
    patient = str(entry.get("patient") or "").strip()
    course = str(entry.get("course") or "").strip()
    if not patient or not course:
        raise OrganizeLedgerError("every organize-ledger entry requires patient and course")
    return patient, course


def build_organize_ledger(entries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in entries:
        if not isinstance(raw, dict):
            raise OrganizeLedgerError("organize-ledger entries must be objects")
        entry = dict(raw)
        identity = _course_identity(entry)
        if identity in seen:
            raise OrganizeLedgerError(
                f"duplicate organize-ledger course {identity[0]}/{identity[1]}"
            )
        seen.add(identity)
        status = str(entry.get("status") or "").strip()
        if status not in _VALID_STATUSES:
            raise OrganizeLedgerError(
                f"invalid organize-ledger status {status!r} for {identity[0]}/{identity[1]}"
            )
        entry["patient"], entry["course"] = identity
        entry["status"] = status
        if status == STATUS_TECHNICAL_QUARANTINE:
            reason = str(entry.get("reason") or "").strip()
            if not reason:
                raise OrganizeLedgerError(
                    f"technical quarantine {identity[0]}/{identity[1]} has no reason"
                )
            entry["reason"] = reason
            entry["disposition_type"] = STATUS_TECHNICAL_QUARANTINE
            entry["clinical_exclusion"] = False
        else:
            entry["reason"] = None
            entry["disposition_type"] = STATUS_VALIDATED
            entry["clinical_exclusion"] = False
        normalized.append(entry)

    normalized.sort(key=lambda item: (item["patient"], item["course"]))
    validated_count = sum(
        entry["status"] == STATUS_VALIDATED for entry in normalized
    )
    quarantined = [
        dict(entry)
        for entry in normalized
        if entry["status"] == STATUS_TECHNICAL_QUARANTINE
    ]
    attempted_count = len(normalized)
    return {
        "schema": ORGANIZE_LEDGER_SCHEMA,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "status": (
            "complete_with_technical_quarantines" if quarantined else "validated"
        ),
        "intended_course_count": attempted_count,
        "attempted_course_count": attempted_count,
        "validated_course_count": validated_count,
        "technical_quarantine_count": len(quarantined),
        "courses": normalized,
        "technical_quarantines": quarantined,
    }


def validate_organize_ledger(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise OrganizeLedgerError("organize ledger must be a JSON object")
    if payload.get("schema") != ORGANIZE_LEDGER_SCHEMA:
        raise OrganizeLedgerError(
            f"unsupported organize-ledger schema {payload.get('schema')!r}"
        )
    courses = payload.get("courses")
    if not isinstance(courses, list):
        raise OrganizeLedgerError("organize ledger must contain a courses list")
    rebuilt = build_organize_ledger(courses)
    count_fields = (
        "intended_course_count",
        "attempted_course_count",
        "validated_course_count",
        "technical_quarantine_count",
    )
    for field in count_fields:
        if payload.get(field) != rebuilt[field]:
            raise OrganizeLedgerError(
                f"organize ledger {field}={payload.get(field)!r} does not reconcile "
                f"with course entries ({rebuilt[field]})"
            )
    declared_quarantines = payload.get("technical_quarantines")
    if not isinstance(declared_quarantines, list):
        raise OrganizeLedgerError(
            "organize ledger must contain a technical_quarantines list"
        )
    declared_ids = {_course_identity(entry) for entry in declared_quarantines}
    rebuilt_ids = {_course_identity(entry) for entry in rebuilt["technical_quarantines"]}
    if declared_ids != rebuilt_ids:
        raise OrganizeLedgerError(
            "organize ledger technical_quarantines do not match course dispositions"
        )
    result = dict(payload)
    result["courses"] = rebuilt["courses"]
    result["technical_quarantines"] = rebuilt["technical_quarantines"]
    return result


def read_organize_ledger(output_root: Path | str) -> dict[str, Any]:
    path = ledger_path(output_root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OrganizeLedgerError(f"organize ledger is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise OrganizeLedgerError(f"organize ledger is unreadable: {path}: {exc}") from exc
    return validate_organize_ledger(payload)


def write_organize_ledger(
    output_root: Path | str, entries: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    payload = build_organize_ledger(entries)
    _write_json_atomic(ledger_path(output_root), payload)
    return payload


def _safe_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", value).strip(".")
    return safe or "unknown"


def quarantine_course_directory(
    output_root: Path | str,
    course_dir: Path | str,
    *,
    patient: str,
    course: str,
    reason: str,
    phase: str,
) -> Path | None:
    """Move a rejected course outside every consumer-visible course path."""

    root = Path(output_root).resolve(strict=False)
    source = Path(course_dir).resolve(strict=False)
    try:
        relative_source = source.relative_to(root)
    except ValueError as exc:
        raise OrganizeLedgerError(
            f"refusing to quarantine course outside output root: {source}"
        ) from exc
    if source == root or len(relative_source.parts) < 2:
        raise OrganizeLedgerError(
            f"refusing to quarantine non-course output path: {source}"
        )

    organized = source / ".organized"
    organized.unlink(missing_ok=True)
    if not source.exists():
        return None

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%S.%fZ"
    )
    base = (
        root.parent
        / f"{root.name}_QUARANTINE"
        / "organize"
        / _safe_component(patient)
        / _safe_component(course)
    )
    base.mkdir(parents=True, exist_ok=True)
    counter = 0
    while True:
        suffix = f"{timestamp}-{os.getpid()}"
        if counter:
            suffix += f"-{counter}"
        destination = base / suffix
        if not destination.exists():
            break
        counter += 1

    os.replace(source, destination)
    record = {
        "status": STATUS_TECHNICAL_QUARANTINE,
        "disposition_type": STATUS_TECHNICAL_QUARANTINE,
        "clinical_exclusion": False,
        "patient": patient,
        "course": course,
        "phase": phase,
        "reason": str(reason),
        "source_path": str(source),
        "quarantine_path": str(destination),
        "quarantined_at": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(timespec="seconds"),
    }
    _write_json_atomic(destination / "technical_quarantine.json", record)
    if source.exists():
        raise OrganizeLedgerError(
            f"technical quarantine did not revoke consumer-visible course {source}"
        )
    return destination
