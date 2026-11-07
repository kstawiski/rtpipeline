"""Historical source and archives must match pinned blobs without commit twins."""
import copy
import io
from pathlib import Path
import subprocess
import tarfile

import pytest

import legacy_source
from legacy_source import legacy_archive, legacy_show


ROOT = Path(__file__).resolve().parents[1]
REVISIONS = legacy_source._REVISIONS
SOURCE_PATH = "rtpipeline/rtstruct_geometry.py"
SCHEMA_PATH = "rtpipeline/radiomics_schema.py"


@pytest.fixture(scope="module")
def fixture_blobs():
    """Read each distinct expected object independently of the legacy helpers."""
    ids = {blob for record in REVISIONS.values() for blob in record["blobs"].values()}
    return {blob: subprocess.check_output(["git", "cat-file", "blob", blob], cwd=ROOT)
            for blob in ids}


@pytest.mark.parametrize("revision", REVISIONS)
@pytest.mark.parametrize("full_id", [False, True], ids=["short", "full"])
def test_resolves_recorded_blob_without_commit_twin(revision, full_id, monkeypatch, fixture_blobs):
    record = REVISIONS[revision]
    blob = record["blobs"][SOURCE_PATH]
    expected = fixture_blobs[blob]
    calls = []

    def read_blob(command, **kwargs):
        calls.append((command, kwargs))
        return expected

    monkeypatch.setattr(legacy_source.subprocess, "check_output", read_blob)
    assert legacy_show(record["old"] if full_id else revision, SOURCE_PATH, ROOT) == expected
    assert calls == [(["git", "cat-file", "blob", blob], {"cwd": ROOT})]


def test_manifest_covers_all_recorded_revisions():
    assert set(REVISIONS) == {
        "072b146", "1aeb0c5", "2369fbb", "2875df2", "35af111", "3af9f51",
        "4ad1c53", "51c7c65", "80d5d65", "879f225", "923233e", "939ad18",
        "a177c59", "a61b96a", "f0ed849", "f16d72a", "fc53fea", "5de543a",
    }


@pytest.mark.parametrize("revision", REVISIONS)
def test_manifest_records_original_revision_and_schema_exception(revision):
    record = REVISIONS[revision]
    assert record["old"].startswith(revision)
    assert len(record["old"]) == 40
    assert all(path.startswith("rtpipeline/") for path in record["blobs"])
    assert all(len(blob) == 40 for blob in record["blobs"].values())
    evidence = record["radiomics_schema"]
    assert record["blobs"][SCHEMA_PATH] == evidence["rewritten_blob"]
    assert record["blobs"][SCHEMA_PATH] != evidence["old_blob"]


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
    assert blob != evidence["old_blob"]


@pytest.mark.parametrize("revision", REVISIONS)
@pytest.mark.parametrize("full_id", [False, True], ids=["short", "full"])
def test_archive_members_and_bytes_equal_fixture(revision, full_id, fixture_blobs):
    record = REVISIONS[revision]
    revision_id = record["old"] if full_id else revision
    payload = legacy_archive(revision_id, cwd=ROOT)
    directories = {str(parent) for path in record["blobs"] for parent in Path(path).parents
                   if parent != Path(".")}
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        members = archive.getmembers()
        assert len(members) == len(record["blobs"]) + len(directories)
        assert {member.name for member in members if member.isdir()} == directories
        assert {member.name for member in members if member.isfile()} == record["blobs"].keys()
        for member in members:
            if member.isdir():
                assert member.mode == 0o775
            else:
                assert member.isfile()
                assert member.mode & 0o020
                assert member.mode & ~0o020 == int(legacy_source._FILE_MODES[member.name], 8) & 0o777
                assert archive.extractfile(member).read() == fixture_blobs[record["blobs"][member.name]]


@pytest.mark.parametrize("prefix", ["rtpipeline/eval", "rtpipeline/eval/", SOURCE_PATH])
def test_archive_prefix_selects_only_recorded_files(prefix, fixture_blobs):
    record = REVISIONS["a177c59"]
    selected = {path: blob for path, blob in record["blobs"].items()
                if path == prefix.rstrip("/") or path.startswith(prefix.rstrip("/") + "/")}
    with tarfile.open(fileobj=io.BytesIO(legacy_archive("a177c59", prefix))) as archive:
        assert {member.name for member in archive if member.isfile()} == selected.keys()
        for path, blob in selected.items():
            assert archive.extractfile(path).read() == fixture_blobs[blob]


def test_helpers_use_repository_root_by_default(tmp_path, monkeypatch, fixture_blobs):
    monkeypatch.chdir(tmp_path)
    blob = REVISIONS["a177c59"]["blobs"][SOURCE_PATH]
    assert legacy_show("a177c59", SOURCE_PATH) == fixture_blobs[blob]
    with tarfile.open(fileobj=io.BytesIO(legacy_archive("a177c59", SOURCE_PATH))) as archive:
        assert archive.extractfile(SOURCE_PATH).read() == fixture_blobs[blob]


@pytest.mark.parametrize("path", [SOURCE_PATH, SCHEMA_PATH])
@pytest.mark.parametrize("reader", [legacy_show, legacy_archive])
def test_one_tampered_byte_fails(path, reader, monkeypatch):
    source = legacy_show("939ad18", path, ROOT)
    tampered = source[:-1] + bytes([source[-1] ^ 1])
    original_read = legacy_source.subprocess.check_output
    target_blob = REVISIONS["939ad18"]["blobs"][path]

    def read_blob(command, **kwargs):
        if command == ["git", "cat-file", "blob", target_blob]:
            return tampered
        return original_read(command, **kwargs)

    monkeypatch.setattr(legacy_source.subprocess, "check_output", read_blob)
    with pytest.raises(AssertionError, match="Legacy blob mismatch.*939ad18"):
        if reader is legacy_show:
            reader("939ad18", path, ROOT)
        else:
            reader("939ad18", cwd=ROOT)


@pytest.mark.parametrize("field,value", [
    ("ast_identical", False),
    ("ast_identical", "true"),
    ("equivalence", "AST-identical"),
    ("lines_differing", 2),
])
@pytest.mark.parametrize("reader", [legacy_show, legacy_archive])
def test_schema_requires_recorded_equivalence(field, value, reader, monkeypatch):
    record = copy.deepcopy(REVISIONS["939ad18"])
    record["radiomics_schema"][field] = value
    monkeypatch.setitem(REVISIONS, "939ad18", record)
    with pytest.raises(AssertionError, match="Unverified comment-only, AST-identical equivalence"):
        reader("939ad18", SCHEMA_PATH, ROOT)


def test_schema_requires_its_recorded_rewritten_blob(monkeypatch):
    record = copy.deepcopy(REVISIONS["939ad18"])
    record["radiomics_schema"]["rewritten_blob"] = record["radiomics_schema"]["old_blob"]
    monkeypatch.setitem(REVISIONS, "939ad18", record)
    with pytest.raises(AssertionError, match="Unverified comment-only, AST-identical equivalence"):
        legacy_show("939ad18", SCHEMA_PATH, ROOT)


@pytest.mark.parametrize("revision,path,match", [
    ("HEAD", SOURCE_PATH, "Unknown legacy revision 'HEAD'"),
    ("939ad1", SOURCE_PATH, "Unknown legacy revision '939ad1'"),
    ("939ad18", "rtpipeline/missing.py", "Unknown legacy path"),
    ("939ad18", "../rtpipeline/rtstruct_geometry.py", "Unknown legacy path"),
    ("939ad18", "rtpipeline/../rtpipeline/rtstruct_geometry.py", "Unknown legacy path"),
    ("939ad18", "", "Unknown legacy path"),
    ("879f225", "rtpipeline/planning_ct_localizers.py", "Unknown legacy path"),
])
@pytest.mark.parametrize("reader", [legacy_show, legacy_archive])
def test_unknown_revision_or_path_fails_before_git(revision, path, match, reader, monkeypatch):
    def unexpected_git(*args, **kwargs):
        pytest.fail("Unknown revision/path must fail before calling Git")

    monkeypatch.setattr(legacy_source.subprocess, "check_output", unexpected_git)
    with pytest.raises(ValueError, match=match):
        reader(revision, path, ROOT)


@pytest.mark.parametrize("reader", [legacy_show, legacy_archive])
def test_missing_git_object_fails_without_fallback(reader, monkeypatch):
    def missing(command, **kwargs):
        raise subprocess.CalledProcessError(128, command, stderr=b"missing object")

    monkeypatch.setattr(legacy_source.subprocess, "check_output", missing)
    with pytest.raises(subprocess.CalledProcessError):
        reader("939ad18", SOURCE_PATH, ROOT)
