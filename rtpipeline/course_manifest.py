"""Pure parsing of the organized-course manifest.

The manifest is the authoritative statement of which courses organize actually
validated, and of the denominator those courses came from. Consumers that need
it must not rediscover courses by scanning the output tree: a directory that no
manifest entry names was never validated, and a manifest entry whose directory
is missing is a missing course, not an absent one.

This module holds no Snakemake state and imports nothing from the executable
workflow scripts, so a consumer (a CLI command, a test) can parse a manifest
without the ``snakemake`` global those scripts require at import time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple


CURRENT_COURSE_MANIFEST_SCHEMA = "rtpipeline-organized-course-manifest-v2"

# Course entries carry the denominator only in the current schema.
_COHORT_COUNT_FIELDS = (
    "attempted_course_count",
    "intended_course_count",
    "validated_course_count",
    "technical_quarantine_count",
)

CourseEntry = Tuple[str, str, Path]


def require_no_output_symlinks(path: Path) -> None:
    """Reject aliases before resolving an output/evidence path, including parents."""
    path = Path(path).absolute()
    for component in (path, *path.parents):
        if component.is_symlink():
            raise RuntimeError(f"symlink output/evidence path is prohibited: {component}")


def _malformed(manifest_path: Any, detail: str) -> RuntimeError:
    return RuntimeError(f"Course manifest is malformed: {detail}")


def _require_safe_identifier(
    value: str, *, field: str, index: int, manifest_path: Any
) -> str:
    """A patient/course identifier is a single path component, or nothing.

    A manifest is data, and data that is joined onto an output root decides
    which directory a consumer reads. An identifier carrying a separator, a
    parent reference or an absolute path would silently redirect that read
    outside the cohort, so it is rejected rather than normalized.
    """
    text = str(value)
    if text != text.strip():
        raise _malformed(
            manifest_path,
            f"entry {index} has a {field} identifier with surrounding whitespace",
        )
    if not text:
        raise _malformed(manifest_path, f"entry {index} has no {field} identifier")
    if text in {os.curdir, os.pardir}:
        raise _malformed(
            manifest_path, f"entry {index} has a relative {field} identifier {text!r}"
        )
    if "\x00" in text:
        raise _malformed(
            manifest_path, f"entry {index} has a {field} identifier with a NUL byte"
        )
    separators = {"/", "\\", os.sep}
    if os.altsep:
        separators.add(os.altsep)
    if any(separator in text for separator in separators):
        raise _malformed(
            manifest_path,
            f"entry {index} has a {field} identifier {text!r} containing a path separator",
        )
    if Path(text).is_absolute():
        raise _malformed(
            manifest_path,
            f"entry {index} has an absolute {field} identifier {text!r}",
        )
    return text


def _require_confined(
    course_dir: Path, *, output_dir: Path, index: int, manifest_path: Any
) -> None:
    root = Path(output_dir).resolve(strict=False)
    try:
        course_dir.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise _malformed(
            manifest_path,
            f"entry {index} resolves to {course_dir} outside the cohort output root {root}",
        ) from exc


def _require_declared_path(
    entry: Mapping[str, Any],
    course_dir: Path,
    *,
    index: int,
    manifest_path: Any,
) -> None:
    declared = entry.get("path")
    if declared is None or not str(declared).strip():
        return
    if Path(str(declared)).resolve(strict=False) != course_dir.resolve(strict=False):
        raise _malformed(
            manifest_path,
            f"entry {index} declares path {declared!r}, which is not the course "
            f"directory {course_dir} its identifiers name",
        )


def _require_exact_count(
    payload: Mapping[str, Any], field: str, *, manifest_path: Any
) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _malformed(
            manifest_path,
            f"{field} must be an integer count, not {value!r}",
        )
    if value < 0:
        raise _malformed(manifest_path, f"{field} is negative: {value!r}")
    return value


def parse_course_manifest(
    payload: Any,
    *,
    output_dir: Path,
    manifest_path: Any,
    require_current_schema: bool = False,
) -> Tuple[List[CourseEntry], Dict[str, Any]]:
    """Return ``(courses, cohort)`` for a parsed manifest payload.

    With ``require_current_schema`` false this reproduces the historical
    aggregation behavior exactly, including the legacy-manifest denominator
    fallback. With it true the payload must be the current schema, identifiers
    must be safe single path components confined to ``output_dir``, and the
    denominator fields must be exact integers (a boolean is not a count).
    """
    output_dir = Path(output_dir)
    if require_current_schema:
        require_no_output_symlinks(output_dir)
        require_no_output_symlinks(Path(manifest_path))
    if not isinstance(payload, dict) or not isinstance(payload.get("courses"), list):
        raise RuntimeError(
            f"Course manifest is malformed: {manifest_path} must contain a courses list"
        )
    schema = payload.get("schema")
    if require_current_schema and schema != CURRENT_COURSE_MANIFEST_SCHEMA:
        raise RuntimeError(
            f"Course manifest {manifest_path} declares schema {schema!r}; this "
            f"consumer requires {CURRENT_COURSE_MANIFEST_SCHEMA!r} and does not "
            "reconstruct a cohort denominator from a manifest that carries none"
        )

    courses: List[CourseEntry] = []
    seen = set()
    for index, entry in enumerate(payload["courses"], start=1):
        if not isinstance(entry, dict):
            raise _malformed(manifest_path, f"entry {index} is not a mapping")
        patient_id = entry.get("patient")
        course_id = entry.get("course")
        if not isinstance(patient_id, str) or not patient_id.strip():
            raise _malformed(manifest_path, f"entry {index} has no patient identifier")
        if not isinstance(course_id, str) or not course_id.strip():
            raise _malformed(manifest_path, f"entry {index} has no course identifier")
        if require_current_schema:
            patient_id = _require_safe_identifier(
                patient_id, field="patient", index=index, manifest_path=manifest_path
            )
            course_id = _require_safe_identifier(
                course_id, field="course", index=index, manifest_path=manifest_path
            )
        key = (patient_id, course_id)
        if key in seen:
            raise _malformed(
                manifest_path, f"duplicate course {patient_id}/{course_id}"
            )
        seen.add(key)
        course_dir = output_dir / patient_id / course_id
        if require_current_schema:
            require_no_output_symlinks(course_dir)
            if entry.get("path"):
                require_no_output_symlinks(Path(entry["path"]))
            _require_confined(
                course_dir, output_dir=output_dir, index=index, manifest_path=manifest_path
            )
            _require_declared_path(
                entry, course_dir, index=index, manifest_path=manifest_path
            )
        courses.append((patient_id, course_id, course_dir))

    if schema == CURRENT_COURSE_MANIFEST_SCHEMA:
        quarantine_entries = payload.get("technical_quarantines")
        if not isinstance(quarantine_entries, list):
            raise _malformed(
                manifest_path, "technical_quarantines must be a list"
            )
        if require_current_schema:
            attempted = _require_exact_count(
                payload, "attempted_course_count", manifest_path=manifest_path
            )
            intended = _require_exact_count(
                payload, "intended_course_count", manifest_path=manifest_path
            )
            validated = _require_exact_count(
                payload, "validated_course_count", manifest_path=manifest_path
            )
            quarantined = _require_exact_count(
                payload, "technical_quarantine_count", manifest_path=manifest_path
            )
        else:
            try:
                attempted = int(payload["attempted_course_count"])
                intended = int(payload["intended_course_count"])
                validated = int(payload["validated_course_count"])
                quarantined = int(payload["technical_quarantine_count"])
            except (KeyError, TypeError, ValueError) as exc:
                raise _malformed(
                    manifest_path, "organize denominator fields are invalid"
                ) from exc
        if intended != attempted:
            raise _malformed(
                manifest_path, "intended and attempted course counts disagree"
            )
        if validated != len(courses):
            raise _malformed(
                manifest_path, "validated count does not match courses"
            )
        if quarantined != len(quarantine_entries):
            raise _malformed(
                manifest_path,
                "technical quarantine count does not match records",
            )
        if attempted != validated + quarantined:
            raise _malformed(
                manifest_path,
                "attempted count does not reconcile with validated and "
                "technically quarantined courses",
            )
        quarantine_ids = set()
        for index, entry in enumerate(quarantine_entries, start=1):
            if not isinstance(entry, dict):
                raise _malformed(
                    manifest_path, f"technical quarantine {index} is not a mapping"
                )
            patient_id = str(entry.get("patient") or "").strip()
            course_id = str(entry.get("course") or "").strip()
            reason = str(entry.get("reason") or "").strip()
            if not patient_id or not course_id or not reason:
                raise _malformed(
                    manifest_path,
                    f"technical quarantine {index} lacks patient, course, or exact reason",
                )
            if (
                entry.get("disposition_type") != "technical_quarantine"
                or entry.get("clinical_exclusion") is not False
            ):
                raise _malformed(
                    manifest_path,
                    f"technical quarantine {index} is not explicitly separated "
                    "from clinical exclusion",
                )
            key = (patient_id, course_id)
            if key in seen or key in quarantine_ids:
                raise _malformed(
                    manifest_path,
                    f"duplicate disposition for {patient_id}/{course_id}",
                )
            quarantine_ids.add(key)
        cohort = {
            "intended_course_count": intended,
            "attempted_course_count": attempted,
            "validated_course_count": validated,
            "technical_quarantine_count": quarantined,
            "technical_quarantines": quarantine_entries,
        }
    else:
        # Legacy manifests did not carry an organize denominator. The current
        # writer always emits the current schema, but retain deterministic
        # compatibility for historical unit artifacts by treating their
        # explicit list as intended.
        cohort = {
            "intended_course_count": len(courses),
            "attempted_course_count": len(courses),
            "validated_course_count": len(courses),
            "technical_quarantine_count": 0,
            "technical_quarantines": [],
        }
    if require_current_schema:
        # Immutable payload bytes bind membership and denominators together.
        # Publication also requires these bytes to match the source manifest.
        cohort["manifest_snapshot"] = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        source = Path(manifest_path)
        if source.is_file():
            source_bytes = source.read_bytes()
            if json.loads(source_bytes) != payload:
                raise RuntimeError("course manifest changed while parsing")
            cohort["manifest_snapshot"] = source_bytes
        cohort["manifest_path"] = str(Path(manifest_path).absolute())
        cohort["output_root"] = str(output_dir.absolute())
        cohort["course_identities"] = frozenset(seen)
    return courses, cohort


def read_course_manifest(
    manifest_path: Path,
    *,
    output_dir: Path,
    require_current_schema: bool = False,
) -> Tuple[List[CourseEntry], Dict[str, Any]]:
    """Read and parse a manifest file, failing closed on unreadable bytes."""
    manifest_path = Path(manifest_path)
    if require_current_schema:
        require_no_output_symlinks(manifest_path)
        require_no_output_symlinks(Path(output_dir))
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Course manifest is unreadable: {manifest_path}: {exc}"
        ) from exc
    return parse_course_manifest(
        payload,
        output_dir=output_dir,
        manifest_path=manifest_path,
        require_current_schema=require_current_schema,
    )
