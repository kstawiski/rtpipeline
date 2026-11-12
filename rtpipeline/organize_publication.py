"""One verified closure for the organized course set, ledger and manifest."""
import json
from pathlib import Path

from .durable_progress import file_sha256, final_result_writer


MARKER = "_COURSES/.organize-progress/publication.json"


def _expected(output):
    manifest = json.loads((output / "_COURSES/manifest.json").read_text())
    ledger = json.loads((output / "_COURSES/organize_ledger.json").read_text())
    courses = {(entry["patient"], entry["course"]) for entry in manifest["courses"]}
    valid = {(entry["patient"], entry["course"]) for entry in ledger["courses"]
             if entry["status"] == "validated"}
    if courses != valid or len(courses) != len(manifest["courses"]):
        raise ValueError("Organized manifest and ledger have different course sets")
    names = {"_COURSES/manifest.json", "_COURSES/organize_ledger.json",
             "_COURSES/.organize-progress/discovery.json"}
    names.update(f"{patient}/{course}/.organized" for patient, course in courses)
    names.update(f"{patient}/{course}/.organize-progress.json" for patient, course in courses)
    return names


def publish(output):
    output = Path(output)
    record = {"schema": "rtpipeline-organize-publication-v1",
              "artifacts": {name: file_sha256(output / name) for name in sorted(_expected(output))}}
    final_result_writer(output / MARKER, record)
    validate(output)


def validate(output):
    output = Path(output)
    record = json.loads((output / MARKER).read_text())
    if record.get("schema") != "rtpipeline-organize-publication-v1" or set(record["artifacts"]) != _expected(output):
        raise ValueError("Organize publication has an incomplete artifact inventory")
    if any(file_sha256(output / name) != digest for name, digest in record["artifacts"].items()):
        raise ValueError("Organize publication artifact changed")
    return record


def source_bytes_current(output, source_root, patient_ids=None):
    from .artifact_progress import inventory
    from .meta import _source_candidate_paths
    try:
        record = json.loads((Path(output) / "_COURSES/.organize-progress/discovery.json").read_text())
        return record["sources"] == inventory(_source_candidate_paths(Path(source_root), patient_ids), Path(source_root))
    except (OSError, ValueError, KeyError, TypeError):
        return False
