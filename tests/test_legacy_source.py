"""Historical lookups must resolve pinned twins and reject changed bytes."""
import copy
from pathlib import Path
import subprocess

import pytest

import legacy_source
from legacy_source import legacy_show


ROOT = Path(__file__).resolve().parents[1]
REVISIONS = legacy_source._REVISIONS
SOURCE_PATH = "rtpipeline/rtstruct_geometry.py"
SCHEMA_PATH = "rtpipeline/radiomics_schema.py"


@pytest.mark.parametrize("revision", REVISIONS)
@pytest.mark.parametrize("full_id", [False, True], ids=["short", "full"])
def test_resolves_recorded_rewritten_twin(revision, full_id, monkeypatch):
    record = REVISIONS[revision]
    expected = subprocess.check_output(
        ["git", "show", f"{record['new']}:{SOURCE_PATH}"], cwd=ROOT,
    )
    calls = []

    def show(command, **kwargs):
        calls.append((command, kwargs))
        return expected

    monkeypatch.setattr(legacy_source.subprocess, "check_output", show)
    assert legacy_show(record["old"] if full_id else revision, SOURCE_PATH, ROOT) == expected
    assert calls == [(["git", "show", f"{record['new']}:{SOURCE_PATH}"], {"cwd": ROOT})]


@pytest.mark.parametrize("revision", REVISIONS)
def test_manifest_pins_every_rewritten_path(revision):
    record = REVISIONS[revision]
    tree = subprocess.check_output(
        ["git", "ls-tree", "-r", record["new"], "--", "rtpipeline/"], cwd=ROOT, text=True,
    )
    rewritten = {line.split("\t", 1)[1]: line.split()[2] for line in tree.splitlines()}
    assert record["old"].startswith(revision)
    assert len(record["old"]) == len(record["new"]) == 40
    assert rewritten.keys() == record["old_blobs"].keys()
    assert {path for path in rewritten if rewritten[path] != record["old_blobs"][path]} == {SCHEMA_PATH}
    assert rewritten[SCHEMA_PATH] == record["radiomics_schema"]["rewritten_blob"]


@pytest.mark.parametrize("revision", REVISIONS)
def test_schema_uses_only_recorded_comment_equivalence(revision):
    record = REVISIONS[revision]
    evidence = record["radiomics_schema"]
    assert evidence["ast_identical"] is True
    assert evidence["equivalence"] == "comment-only, AST-identical"
    assert evidence["lines_differing"] == 1
    source = legacy_show(revision, SCHEMA_PATH, ROOT)
    blob = subprocess.check_output(["git", "hash-object", "--stdin"], input=source, cwd=ROOT).decode().strip()
    assert blob == evidence["rewritten_blob"]
    assert blob != record["old_blobs"][SCHEMA_PATH]


@pytest.mark.parametrize("path", [SOURCE_PATH, SCHEMA_PATH])
def test_one_tampered_byte_fails(path, monkeypatch):
    source = legacy_show("939ad18", path, ROOT)
    tampered = source[:-1] + bytes([source[-1] ^ 1])
    monkeypatch.setattr(legacy_source.subprocess, "check_output", lambda *a, **k: tampered)
    with pytest.raises(AssertionError, match="Legacy blob mismatch.*939ad18"):
        legacy_show("939ad18", path, ROOT)


@pytest.mark.parametrize("field,value", [
    ("ast_identical", False),
    ("ast_identical", "true"),
    ("equivalence", "AST-identical"),
    ("lines_differing", 2),
])
def test_schema_requires_recorded_equivalence(field, value, monkeypatch):
    record = copy.deepcopy(REVISIONS["939ad18"])
    record["radiomics_schema"][field] = value
    monkeypatch.setitem(REVISIONS, "939ad18", record)
    with pytest.raises(AssertionError, match="Unverified comment-only, AST-identical equivalence"):
        legacy_show("939ad18", SCHEMA_PATH, ROOT)


def test_schema_requires_its_recorded_rewritten_blob(monkeypatch):
    record = copy.deepcopy(REVISIONS["939ad18"])
    record["radiomics_schema"]["rewritten_blob"] = record["old_blobs"][SCHEMA_PATH]
    monkeypatch.setitem(REVISIONS, "939ad18", record)
    with pytest.raises(AssertionError, match="Legacy blob mismatch"):
        legacy_show("939ad18", SCHEMA_PATH, ROOT)


@pytest.mark.parametrize("revision,path,match", [
    ("HEAD", SOURCE_PATH, "Unknown legacy revision 'HEAD'"),
    ("939ad1", SOURCE_PATH, "Unknown legacy revision '939ad1'"),
    ("939ad18", "rtpipeline/missing.py", "Unknown legacy path"),
    ("939ad18", "../rtpipeline/rtstruct_geometry.py", "Unknown legacy path"),
    ("879f225", "rtpipeline/planning_ct_localizers.py", "Unknown legacy path"),
])
def test_unknown_revision_or_path_fails_before_git(revision, path, match, monkeypatch):
    def unexpected_git(*args, **kwargs):
        pytest.fail("Unknown revision/path must fail before calling Git")

    monkeypatch.setattr(legacy_source.subprocess, "check_output", unexpected_git)
    with pytest.raises(ValueError, match=match):
        legacy_show(revision, path, ROOT)


def test_missing_git_object_fails_without_fallback(monkeypatch):
    def missing(command, **kwargs):
        raise subprocess.CalledProcessError(128, command, stderr=b"missing object")

    monkeypatch.setattr(legacy_source.subprocess, "check_output", missing)
    with pytest.raises(subprocess.CalledProcessError):
        legacy_show("939ad18", SOURCE_PATH, ROOT)
