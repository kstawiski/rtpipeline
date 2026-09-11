"""The per-course robustness completion receipt, and how a consumer revalidates it.

A robustness course step used to complete by writing the literal text ``ok``.
That token says which shell command exited zero; it says nothing about which
course, which run, which outcome, or which bytes it certifies, so a consumer
could only re-derive those by scanning the filesystem and trusting whatever it
found. This receipt records the exact identity instead, and every consumer
re-checks all of it against the artifacts on disk before using a course.

The receipt is a *step* receipt, not a scientific eligibility decision: it says
a robustness course reached a terminal, non-technical outcome and which one.
Deciding what a measured or source-only outcome means for a cohort stays with
the aggregation consumer.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


ROBUSTNESS_COMPLETION_SCHEMA = "rtpipeline-robustness-completion-v1"
ROBUSTNESS_COMPLETION_SENTINEL_NAME = ".radiomics_robustness_done"

# Outcomes that let the workflow step complete. An outcome outside this set is
# not a completion, and the course never receives a receipt for it.
ROBUSTNESS_COMPLETING_OUTCOMES = frozenset(
    {"measured", "source_only_nonvolumetric"}
)


class RobustnessCompletionError(RuntimeError):
    """A completion receipt is absent, malformed, stale, or foreign."""


class LegacyRobustnessCompletionError(RobustnessCompletionError):
    """The sentinel is a legacy ``ok`` token, which certifies nothing.

    It is reported as its own failure so an operator sees that the course must
    be re-run under the current contract, rather than a receipt being invented
    for evidence that was never recorded.
    """


@dataclass(frozen=True)
class RobustnessCompletionReceipt:
    """A revalidated per-course robustness completion."""

    path: Path
    course_dir: Path
    patient_id: str
    course_id: str
    run_identifier: str
    measurement_outcome: str
    output_name: str
    dispositions_path: Path
    dispositions_sha256: str
    measured_output: Optional[Path]
    measured_output_sha256: Optional[str]
    source_disposition_count: int
    effective_configuration_sha256: str

    @property
    def measured(self) -> bool:
        return self.measurement_outcome == "measured"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def robustness_completion_sentinel_path(course_dir: Path) -> Path:
    return Path(course_dir) / ROBUSTNESS_COMPLETION_SENTINEL_NAME


def invalidate_robustness_completion_sentinel(sentinel_path: Path) -> None:
    """Drop any previous receipt before work starts.

    A receipt that survives the start of a new attempt could certify the new
    attempt's failure with the previous attempt's evidence.
    """
    Path(sentinel_path).unlink(missing_ok=True)


def _require_plain_name(output_name: str) -> str:
    text = str(output_name).strip()
    if not text or text != str(output_name):
        raise RobustnessCompletionError(
            f"robustness output name {output_name!r} is not a plain file name"
        )
    if text in {os.curdir, os.pardir} or Path(text).name != text:
        raise RobustnessCompletionError(
            f"robustness output name {output_name!r} is not a plain file name"
        )
    return text


def robustness_completion_payload(
    course_dir: Path,
    *,
    patient_id: str,
    course_id: str,
    run_identifier: str,
    measurement_outcome: str,
    output_name: str,
    dispositions_path: Path,
    measured_output: Optional[Path],
    source_disposition_count: int,
    effective_configuration_sha256: str,
) -> Dict[str, Any]:
    """Build the receipt for one admitted, terminal robustness course outcome."""
    course_dir = Path(course_dir)
    outcome = str(measurement_outcome)
    if outcome not in ROBUSTNESS_COMPLETING_OUTCOMES:
        raise RobustnessCompletionError(
            f"refusing to record robustness completion for outcome {outcome!r}; "
            "only "
            + ", ".join(sorted(ROBUSTNESS_COMPLETING_OUTCOMES))
            + " complete the step"
        )
    if (patient_id, course_id) != (course_dir.parent.name, course_dir.name):
        raise RobustnessCompletionError(
            f"robustness completion identity {patient_id}/{course_id} does not "
            f"match course directory {course_dir}"
        )
    output_name = _require_plain_name(output_name)
    dispositions_path = Path(dispositions_path)
    if not dispositions_path.is_file():
        raise RobustnessCompletionError(
            f"robustness source dispositions {dispositions_path} are absent; a "
            "completion receipt may not cite evidence that does not exist"
        )
    measured_entry: Optional[Dict[str, Any]] = None
    if outcome == "measured":
        if measured_output is None:
            raise RobustnessCompletionError(
                "a measured robustness completion must bind its measurement table"
            )
        measured_output = Path(measured_output)
        if measured_output.name != output_name:
            raise RobustnessCompletionError(
                f"measured robustness output {measured_output.name!r} is not the "
                f"declared output {output_name!r}"
            )
        if not measured_output.is_file():
            raise RobustnessCompletionError(
                f"measured robustness output {measured_output} is absent"
            )
        measured_entry = {
            "path": output_name,
            "sha256": _file_sha256(measured_output),
            "size_bytes": int(measured_output.stat().st_size),
        }
    elif measured_output is not None:
        raise RobustnessCompletionError(
            "a source-only robustness completion must not bind a measurement table"
        )
    return {
        "schema": ROBUSTNESS_COMPLETION_SCHEMA,
        "status": "ok",
        "stage": ROBUSTNESS_COMPLETION_SENTINEL_NAME,
        "stage_name": "radiomics_robustness",
        "patient_id": str(patient_id),
        "course_id": str(course_id),
        "robustness_run_identifier": str(run_identifier),
        "measurement_outcome": outcome,
        "output_path": output_name,
        "measured_output": measured_entry,
        "source_dispositions": {
            "path": dispositions_path.name,
            "sha256": _file_sha256(dispositions_path),
            "row_count": int(source_disposition_count),
        },
        "effective_configuration_sha256": str(effective_configuration_sha256),
    }


def write_robustness_completion_sentinel(
    sentinel_path: Path,
    course_dir: Path,
    **payload_fields: Any,
) -> RobustnessCompletionReceipt:
    """Atomically publish a receipt, then read it back through the consumer path."""
    sentinel_path = Path(sentinel_path)
    course_dir = Path(course_dir)
    from .course_manifest import require_no_output_symlinks
    require_no_output_symlinks(sentinel_path)
    require_no_output_symlinks(course_dir)
    expected = robustness_completion_sentinel_path(course_dir)
    if sentinel_path.resolve(strict=False) != expected.resolve(strict=False):
        raise RobustnessCompletionError(
            f"robustness completion sentinel path mismatch: expected {expected}, "
            f"got {sentinel_path}"
        )
    payload = robustness_completion_payload(course_dir, **payload_fields)
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=str(sentinel_path.parent), prefix=f".{sentinel_path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, sentinel_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    # Read it back through exactly the consumer path: a receipt a consumer
    # cannot revalidate is not a completion, and must not survive the step.
    try:
        return read_robustness_completion_sentinel(sentinel_path, course_dir=course_dir)
    except BaseException:
        sentinel_path.unlink(missing_ok=True)
        raise


def _validate_payload(
    payload: Mapping[str, Any],
    *,
    sentinel_path: Path,
    course_dir: Path,
) -> RobustnessCompletionReceipt:
    if payload.get("schema") != ROBUSTNESS_COMPLETION_SCHEMA:
        raise RobustnessCompletionError(
            f"{sentinel_path} declares schema {payload.get('schema')!r}; this "
            f"consumer reads {ROBUSTNESS_COMPLETION_SCHEMA!r}"
        )
    if payload.get("status") != "ok":
        raise RobustnessCompletionError(
            f"{sentinel_path} records status {payload.get('status')!r}, which is "
            "not a completed robustness course"
        )
    patient_id = str(payload.get("patient_id") or "")
    course_id = str(payload.get("course_id") or "")
    if (patient_id, course_id) != (course_dir.parent.name, course_dir.name):
        raise RobustnessCompletionError(
            f"{sentinel_path} certifies {patient_id}/{course_id}, not the course "
            f"{course_dir.parent.name}/{course_dir.name} it was found in"
        )
    run_identifier = str(payload.get("robustness_run_identifier") or "").strip()
    if not run_identifier:
        raise RobustnessCompletionError(
            f"{sentinel_path} records no robustness run identity"
        )
    outcome = str(payload.get("measurement_outcome") or "")
    if outcome not in ROBUSTNESS_COMPLETING_OUTCOMES:
        raise RobustnessCompletionError(
            f"{sentinel_path} records outcome {outcome!r}, which does not "
            "complete a robustness course"
        )
    output_name = _require_plain_name(str(payload.get("output_path") or ""))

    dispositions = payload.get("source_dispositions")
    if not isinstance(dispositions, dict):
        raise RobustnessCompletionError(
            f"{sentinel_path} binds no robustness source dispositions"
        )
    dispositions_name = _require_plain_name(str(dispositions.get("path") or ""))
    if dispositions_name != "radiomics_robustness_source_dispositions.json":
        raise RobustnessCompletionError("receipt must bind the canonical disposition sidecar")
    dispositions_path = course_dir / "metadata" / dispositions_name
    from .course_manifest import require_no_output_symlinks
    for path in (sentinel_path, dispositions_path, course_dir / output_name):
        require_no_output_symlinks(path)
    if not dispositions_path.is_file():
        raise RobustnessCompletionError(
            f"robustness source dispositions {dispositions_path} certified by "
            f"{sentinel_path} are absent"
        )
    recorded_dispositions_digest = str(dispositions.get("sha256") or "")
    current_dispositions_digest = _file_sha256(dispositions_path)
    if recorded_dispositions_digest != current_dispositions_digest:
        raise RobustnessCompletionError(
            f"robustness source dispositions {dispositions_path} changed after "
            f"completion (recorded sha256 {recorded_dispositions_digest!r}, "
            f"current {current_dispositions_digest!r})"
        )

    measured_output_path: Optional[Path] = None
    measured_output_digest: Optional[str] = None
    measured_output = payload.get("measured_output")
    if outcome == "measured":
        if not isinstance(measured_output, dict):
            raise RobustnessCompletionError(
                f"{sentinel_path} records a measured course but binds no table"
            )
        if _require_plain_name(str(measured_output.get("path") or "")) != output_name:
            raise RobustnessCompletionError(
                f"{sentinel_path} binds table {measured_output.get('path')!r}, "
                f"not the declared output {output_name!r}"
            )
        measured_output_path = course_dir / output_name
        if not measured_output_path.is_file():
            raise RobustnessCompletionError(
                f"measured robustness table {measured_output_path} certified by "
                f"{sentinel_path} is absent"
            )
        measured_output_digest = str(measured_output.get("sha256") or "")
        current_table_digest = _file_sha256(measured_output_path)
        if measured_output_digest != current_table_digest:
            raise RobustnessCompletionError(
                f"measured robustness table {measured_output_path} changed after "
                f"completion (recorded sha256 {measured_output_digest!r}, current "
                f"{current_table_digest!r})"
            )
    else:
        if measured_output is not None:
            raise RobustnessCompletionError(
                f"{sentinel_path} records a source-only course but binds a "
                "measurement table"
            )
        if (course_dir / output_name).exists():
            raise RobustnessCompletionError(
                f"{sentinel_path} records a source-only course, but the "
                f"measurement table {course_dir / output_name} exists"
            )

    row_count = dispositions.get("row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
        raise RobustnessCompletionError(
            f"{sentinel_path} records an unusable source disposition count "
            f"{row_count!r}"
        )
    return RobustnessCompletionReceipt(
        path=sentinel_path,
        course_dir=course_dir,
        patient_id=patient_id,
        course_id=course_id,
        run_identifier=run_identifier,
        measurement_outcome=outcome,
        output_name=output_name,
        dispositions_path=dispositions_path,
        dispositions_sha256=current_dispositions_digest,
        measured_output=measured_output_path,
        measured_output_sha256=measured_output_digest,
        source_disposition_count=row_count,
        effective_configuration_sha256=str(
            payload.get("effective_configuration_sha256") or ""
        ),
    )


def read_robustness_completion_sentinel(
    sentinel_path: Path, *, course_dir: Optional[Path] = None
) -> RobustnessCompletionReceipt:
    """Revalidate a receipt and everything it binds, or fail closed.

    This never migrates a legacy ``ok`` sentinel into a receipt. The token
    records no run, outcome or digest, so treating it as a completion would
    mean inventing evidence that was never captured.
    """
    sentinel_path = Path(sentinel_path)
    course_dir = Path(course_dir) if course_dir is not None else sentinel_path.parent
    from .course_manifest import require_no_output_symlinks
    require_no_output_symlinks(sentinel_path)
    require_no_output_symlinks(course_dir)
    if not sentinel_path.is_file():
        raise RobustnessCompletionError(
            f"no robustness completion receipt at {sentinel_path}"
        )
    text = sentinel_path.read_text(encoding="utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except Exception as exc:
        stripped = text.strip()
        if stripped.startswith("disabled"):
            raise RobustnessCompletionError(
                f"{sentinel_path} marks a disabled robustness step; a disabled "
                "step certifies no course outcome and is not a completion"
            ) from exc
        if stripped.startswith("ok"):
            raise LegacyRobustnessCompletionError(
                f"{sentinel_path} is a legacy {stripped.splitlines()[0]!r} sentinel, "
                "which binds no run, outcome, output or digest; the course must be "
                "re-run under the current robustness contract"
            ) from exc
        raise RobustnessCompletionError(
            f"robustness completion receipt {sentinel_path} is unreadable: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RobustnessCompletionError(
            f"robustness completion receipt {sentinel_path} is not a record: "
            f"{type(payload).__name__}"
        )
    return _validate_payload(
        payload, sentinel_path=sentinel_path, course_dir=course_dir
    )
