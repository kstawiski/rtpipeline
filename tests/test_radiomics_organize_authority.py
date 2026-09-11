"""Synthetic checks that an invocation list cannot certify a study census."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from rtpipeline import radiomics
from rtpipeline.organize_ledger import ledger_path, write_organize_ledger


def _course(root, patient, course):
    return SimpleNamespace(
        patient_id=patient, course_key=course,
        dirs=SimpleNamespace(root=root / patient / course),
    )


def _ledger(root):
    return write_organize_ledger(root, [
        {"patient": "P1", "course": "C1", "status": "validated"},
        {"patient": "P2", "course": "C2", "status": "validated"},
        {"patient": "P3", "course": "C3", "status": "technical_quarantine",
         "reason": "synthetic unresolved planning CT"},
    ])


@pytest.mark.parametrize("empty", [False, True])
def test_missing_ledger_never_synthesizes_validated_census(tmp_path, empty):
    courses = [] if empty else [_course(tmp_path, "P1", "C1")]
    assert radiomics._full_organize_cohort(
        SimpleNamespace(output_root=tmp_path), courses
    ) is None
    assert not ledger_path(tmp_path).exists()


@pytest.mark.parametrize("with_ledger", [False, True])
def test_duplicate_invocation_identity_is_rejected(tmp_path, with_ledger):
    if with_ledger:
        _ledger(tmp_path)
    first = _course(tmp_path, "P1", "C1")
    duplicate = _course(tmp_path, " P1 ", "C1")
    second = _course(tmp_path, "P2", "C2")
    with pytest.raises(RuntimeError, match="duplicate.*course"):
        radiomics._full_organize_cohort(
            SimpleNamespace(output_root=tmp_path), [first, duplicate, second]
        )


def test_complete_invocation_keeps_quarantine_and_exact_source_hash(tmp_path):
    _ledger(tmp_path)
    before = ledger_path(tmp_path).read_bytes()
    result = radiomics._full_organize_cohort(
        SimpleNamespace(output_root=tmp_path),
        [_course(tmp_path, "P2", "C2"), _course(tmp_path, "P1", "C1")],
    )
    assert result["intended_course_count"] == 3
    assert result["validated_course_count"] == 2
    assert result["technical_quarantine_count"] == 1
    assert len(result["technical_quarantines"]) == 1
    quarantine = result["technical_quarantines"][0]
    assert quarantine["clinical_exclusion"] is False
    assert quarantine["reason"] == "synthetic unresolved planning CT"
    digest = hashlib.sha256(before).hexdigest()
    assert result["denominator_source_sha256"] == digest
    assert quarantine["source_record_sha256"] == digest
    assert ledger_path(tmp_path).read_bytes() == before


@pytest.mark.parametrize("identities", [
    [("P1", "C1")],
    [("P1", "C1"), ("P4", "C4")],  # equal count is not equal identity
    [("P1", "C1"), ("P2", "C2"), ("P3", "C3")],
])
def test_nonmatching_invocation_cannot_publish_full_census(tmp_path, identities):
    _ledger(tmp_path)
    assert radiomics._full_organize_cohort(
        SimpleNamespace(output_root=tmp_path),
        [_course(tmp_path, *identity) for identity in identities],
    ) is None


def test_malformed_ledger_is_not_treated_as_absent(tmp_path):
    _ledger(tmp_path)
    path = ledger_path(tmp_path)
    payload = json.loads(path.read_text())
    payload["validated_course_count"] = 3
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="organize ledger is invalid"):
        radiomics._full_organize_cohort(
            SimpleNamespace(output_root=tmp_path),
            [_course(tmp_path, "P1", "C1"), _course(tmp_path, "P2", "C2")],
        )


def test_ledger_validation_and_hash_use_one_source_snapshot(tmp_path, monkeypatch):
    _ledger(tmp_path)
    path = ledger_path(tmp_path)
    original_bytes = path.read_bytes()
    changed = json.loads(original_bytes)
    changed["generated_at"] = "synthetic later ledger version"
    changed_bytes = json.dumps(changed).encode()
    original_read = type(path).read_bytes
    original_read_text = type(path).read_text
    reads = []

    def read_text(candidate, *args, **kwargs):
        if candidate == path:
            return changed_bytes.decode()
        return original_read_text(candidate, *args, **kwargs)

    def read_bytes(candidate):
        if candidate != path:
            return original_read(candidate)
        reads.append(candidate)
        return original_bytes if len(reads) == 1 else changed_bytes

    monkeypatch.setattr(type(path), "read_bytes", read_bytes)
    monkeypatch.setattr(type(path), "read_text", read_text)
    result = radiomics._full_organize_cohort(
        SimpleNamespace(output_root=tmp_path),
        [_course(tmp_path, "P1", "C1"), _course(tmp_path, "P2", "C2")],
    )
    # A separately parsed text read could bind different contents to this hash.
    assert len(reads) == 1
    assert result["denominator_source_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert result["generated_at"] == json.loads(original_bytes)["generated_at"]
