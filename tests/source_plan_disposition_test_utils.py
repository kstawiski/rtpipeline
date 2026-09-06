"""Synthetic source-history fixture for CT-only manifest resume tests."""
from pathlib import Path
from rtpipeline.plan_disposition import build_source_plan_dispositions, source_scope_fingerprint


def empty_source_history(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    payload = build_source_plan_dispositions([], {}, {})
    payload.update(source_root=str(root.resolve()), source_scope_fingerprint=source_scope_fingerprint(root),
                   discovery_scope_patient_ids=None, course_publication_status="complete",
                   source_archive_status="complete")
    return payload
