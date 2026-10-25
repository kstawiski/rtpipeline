"""RTsegmentator result packages as an RTpipeline segmentation source.

Synthetic DICOM headers and packages only; no portal, model or network is used.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pydicom
import pytest

from rtpipeline import cli
from rtpipeline import rtsegmentator as rtseg
from rtpipeline.custom_models import list_custom_model_outputs, validate_custom_model_output_inventory
from rtpipeline.segmenters import rtsegmentator_client as client_module

from course_contract_test_utils import (
    write_minimal_course_contract,
    write_synthetic_planning_ct,
    write_synthetic_rtstruct,
)

# The client is vendored byte-for-byte from RTsegmentator (rtseg_client.py).
# Update this digest together with the vendored file.
VENDORED_CLIENT_SHA256 = "31743523c7900db4785a1f8a2e7c00555959bcf1d5535e0921432cfef07be25e"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def course(tmp_path):
    course_dir = tmp_path / "Output" / "P001" / "2024-01"
    ct_dir = write_synthetic_planning_ct(course_dir)
    write_minimal_course_contract(course_dir, planning_ct_dir=ct_dir)
    series_uid = str(pydicom.dcmread(next(ct_dir.iterdir()), stop_before_pixels=True).SeriesInstanceUID)
    return course_dir, series_uid


def make_package(path: Path, series_uid: str, *, rtstructs=None, empty=(), report=(), manifest_series=None,
                 manifest_names=None):
    """Write a result ZIP in the RTsegmentator manifest 1.0 layout."""
    rtstructs = {"total": ["liver", "spleen"]} if rtstructs is None else rtstructs
    scratch = path.parent / (path.stem + "_parts")
    scratch.mkdir(parents=True, exist_ok=True)
    members, models = {}, []
    for name, rois in rtstructs.items():
        rt_path = write_synthetic_rtstruct(scratch / f"{name}_RTSTRUCT.dcm", referenced_series_uid=series_uid,
                                           roi_names=rois)
        members[f"{name}_RTSTRUCT.dcm"] = rt_path.read_bytes()
        models.append({
            "name": name, "structure_kind": "fixed", "catalog_structures": [*rois, "kidney_left"],
            "outcome": "rtstruct", "license": "Apache-2.0", "reference": "https://example.org/model",
            "rtstruct": {"path": f"{name}_RTSTRUCT.dcm", "sop_instance_uid": None,
                         "roi_names": list((manifest_names or {}).get(name, rois)), "roi_count": len(rois),
                         "frame_of_reference_uids": [], "referenced_series_instance_uids": [series_uid]},
        })
    for name in empty:
        models.append({"name": name, "structure_kind": "fixed", "catalog_structures": ["lung_nodule"],
                       "outcome": "no_structures_found"})
    if empty:
        members["NO_STRUCTURES_FOUND.txt"] = b"".join(f"- {name}\n".encode() for name in empty)
    for name in report:
        models.append({"name": name, "structure_kind": "report", "catalog_structures": [], "outcome": "report"})
        members[f"report/{name}.json"] = b"{}"
    manifest = {
        "schema": "rtsegmentator.result-manifest", "schema_version": "1.0",
        "generator": {"name": "RTsegmentator", "version": "1.0.0"}, "created_utc": "2026-01-01T00:00:00Z",
        "job": {"id": "20260101-000000-1-abcd", "models": [m["name"] for m in models], "postprocess": []},
        "source_series": {"series_instance_uid": manifest_series or series_uid, "modality": "CT"},
        "models": models,
        "files": [{"path": name, "size": len(data), "sha256": _sha(data)} for name, data in sorted(members.items())],
        "notes": [],
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        archive.writestr("rtsegmentator_manifest.json", json.dumps(manifest))
    return path


def test_vendored_client_is_unmodified():
    data = Path(client_module.__file__).read_bytes()
    assert _sha(data) == VENDORED_CLIENT_SHA256
    assert client_module.MANIFEST_SCHEMA == "rtsegmentator.result-manifest"


def test_import_writes_custom_model_contract(course, tmp_path):
    course_dir, series_uid = course
    package = make_package(tmp_path / "result.zip", series_uid, empty=["lung_nodules"], report=["nv_reason_ct_report"])
    imported = rtseg.import_result_package(package, course_dir, url="https://portal.example")
    assert [(item.model, item.outcome) for item in imported] == [("total", "rtstruct"),
                                                                  ("lung_nodules", "no_structures_found")]
    output = course_dir / "Segmentation_CustomModels" / "rtseg_total"
    assert (output / "rtstruct.dcm").read_bytes() == zipfile.ZipFile(package).read("total_RTSTRUCT.dcm")
    model_manifest = json.loads((output / "manifest.json").read_text())
    assert model_manifest["expected_structures"] == ["liver", "spleen"]
    assert model_manifest["missing_structures"] == []
    assert model_manifest["absent_catalog_structures"] == ["kidney_left"]
    assert model_manifest["rtsegmentator"]["license"] == "Apache-2.0"
    assert model_manifest["planning_ct"]["series_instance_uid"] == series_uid
    assert (output / "rtsegmentator_manifest.json").is_file()
    assert not (course_dir / "Segmentation_CustomModels" / "rtseg_lung_nodules").exists()
    # Downstream stages enumerate the import through the existing custom-model contract.
    assert validate_custom_model_output_inventory(course_dir) == {"rtseg_total": ["liver", "spleen"]}
    assert [name for name, _ in list_custom_model_outputs(course_dir)] == ["rtseg_total"]
    ledger = json.loads((course_dir / "metadata" / "rtsegmentator_imports.json").read_text())
    assert ledger["models"]["total"]["url"] == "https://portal.example"
    assert ledger["models"]["lung_nodules"]["outcome"] == "no_structures_found"
    assert not list((course_dir / "Segmentation_CustomModels").glob(".*"))


@pytest.mark.parametrize("problem", ["other_series", "names", "tampered", "missing_model", "report_requested"])
def test_rejected_packages_write_nothing(course, tmp_path, problem):
    course_dir, series_uid = course
    kwargs, models = {}, None
    if problem == "other_series":
        kwargs["manifest_series"] = "1.2.3.4"
    if problem == "names":
        kwargs["manifest_names"] = {"total": ["liver", "kidney"]}
    if problem == "report_requested":
        kwargs["report"] = ["nv_reason_ct_report"]
        models = ["nv_reason_ct_report"]
    if problem == "missing_model":
        models = ["total", "pancreas"]
    package = make_package(tmp_path / "result.zip", series_uid, **kwargs)
    if problem == "tampered":
        with zipfile.ZipFile(package) as source:
            items = {name: source.read(name) for name in source.namelist()}
        items["total_RTSTRUCT.dcm"] += b"x"
        with zipfile.ZipFile(package, "w") as archive:
            for name, data in items.items():
                archive.writestr(name, data)
    with pytest.raises(rtseg.RTsegmentatorError):
        rtseg.import_result_package(package, course_dir, models=models)
    assert not (course_dir / "Segmentation_CustomModels").exists()
    assert not (course_dir / "metadata" / "rtsegmentator_imports.json").exists()


def test_rtstruct_for_another_series_is_rejected(course, tmp_path):
    course_dir, series_uid = course
    package = make_package(tmp_path / "result.zip", "9.9.9")
    with pytest.raises(rtseg.RTsegmentatorError, match="not the planning CT"):
        rtseg.import_result_package(package, course_dir)


def test_reimport_replaces_and_empty_outcome_removes(course, tmp_path):
    course_dir, series_uid = course
    rtseg.import_result_package(make_package(tmp_path / "a.zip", series_uid), course_dir)
    rtseg.import_result_package(make_package(tmp_path / "b.zip", series_uid, rtstructs={"total": ["liver"]}),
                                course_dir)
    assert validate_custom_model_output_inventory(course_dir) == {"rtseg_total": ["liver"]}
    rtseg.import_result_package(make_package(tmp_path / "c.zip", series_uid, rtstructs={}, empty=["total"]),
                                course_dir)
    assert not (course_dir / "Segmentation_CustomModels" / "rtseg_total").exists()


@pytest.mark.parametrize("mapping, message", [
    ({"enabled": True, "models": ["total"]}, "url is required"),
    ({"enabled": True, "url": "https://x"}, "at least one model"),
    ({"enabled": "yes", "url": "https://x", "models": ["total"]}, "boolean"),
    ({"enabled": False, "modles": ["total"]}, "Unknown"),
    ({"enabled": False, "models": ["../total"]}, "invalid model name"),
    ({"enabled": False, "models": ["total", "total"]}, "duplicates"),
])
def test_settings_validation(mapping, message):
    with pytest.raises(ValueError, match=message):
        rtseg.RTsegmentatorSettings.from_mapping(mapping)


def test_settings_defaults_and_string_models():
    settings = rtseg.RTsegmentatorSettings.from_mapping(
        {"enabled": True, "url": "https://x", "models": "total, lung_nodules", "job_timeout_seconds": 0})
    assert settings.models == ("total", "lung_nodules")
    assert settings.job_timeout_seconds is None
    assert rtseg.RTsegmentatorSettings.from_mapping(None).enabled is False


class FakeClient:
    """Records the calls the course runner makes and returns a prepared package."""

    def __init__(self, package_factory, series_uid, skipped=None):
        self.package_factory, self.series_uid, self.skipped = package_factory, series_uid, skipped
        self.calls = []

    def upload_files(self, files):
        self.calls.append(("upload", sorted(Path(f).name for f in files)))
        return "session-1"

    def assess(self, session, poll):
        self.calls.append(("assess", session))
        return {"upload_id": "upload-1", "series": [
            {"key": f"{self.series_uid}|CT", "series_uid": self.series_uid, "modality": "CT"},
            {"key": "other|MR", "series_uid": "other", "modality": "MR"}]}

    def submit(self, upload_id, series_keys, models):
        self.calls.append(("submit", upload_id, list(series_keys), list(models)))
        job = {"id": "job-1", "series_key": series_keys[0], "models": list(models)}
        if self.skipped:
            job["skipped_models"] = self.skipped
        return {"batch_id": "b", "ids": ["job-1"], "jobs": [job]}

    def wait(self, job_id, poll, timeout):
        self.calls.append(("wait", job_id))
        return {"status": "complete"}

    def download(self, job_id, destination):
        self.calls.append(("download", job_id))
        return self.package_factory(Path(destination))


def test_run_for_course_submits_planning_ct_once(course, tmp_path):
    course_dir, series_uid = course
    settings = rtseg.RTsegmentatorSettings.from_mapping(
        {"enabled": True, "url": "https://portal.example", "models": ["total"]})
    fake = FakeClient(lambda target: make_package(target, series_uid), series_uid)
    imported = rtseg.run_for_course(settings, course_dir, client=fake)
    assert [item.model for item in imported] == ["total"]
    assert ("submit", "upload-1", [f"{series_uid}|CT"], ["total"]) in fake.calls
    assert fake.calls[0] == ("upload", ["ct_000.dcm"])
    # Current results are reused; force resubmits.
    fake.calls.clear()
    assert rtseg.run_for_course(settings, course_dir, client=fake) == []
    assert fake.calls == []
    rtseg.run_for_course(settings, course_dir, client=fake, force=True)
    assert fake.calls and fake.calls[0][0] == "upload"
    # A different portal is a different model installation.
    moved = rtseg.RTsegmentatorSettings.from_mapping(
        {"enabled": True, "url": "https://other.example", "models": ["total"]})
    fake.calls.clear()
    rtseg.run_for_course(moved, course_dir, client=fake)
    assert fake.calls and fake.calls[0][0] == "upload"
    assert not list(course_dir.glob("rtseg-*"))


def test_run_for_course_refuses_skipped_models(course, tmp_path):
    course_dir, series_uid = course
    settings = rtseg.RTsegmentatorSettings.from_mapping(
        {"enabled": True, "url": "https://portal.example", "models": ["total"]})
    fake = FakeClient(lambda target: make_package(target, series_uid), series_uid,
                      skipped={"total": "requires MR"})
    with pytest.raises(rtseg.RTsegmentatorError, match="skipped"):
        rtseg.run_for_course(settings, course_dir, client=fake)
    assert not (course_dir / "Segmentation_CustomModels").exists()


def test_disabled_settings_do_nothing(course):
    course_dir, _ = course
    assert rtseg.run_for_course(rtseg.RTsegmentatorSettings(), course_dir) == []


def test_cli_import_subcommand(course, tmp_path, capsys):
    course_dir, series_uid = course
    package = make_package(tmp_path / "result.zip", series_uid)
    assert cli.main(["rtsegmentator-import", "--course-dir", str(course_dir), str(package)]) == 0
    record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert record["model"] == "total" and record["rois"] == 2
    bad = make_package(tmp_path / "bad.zip", "9.9.9")
    assert cli.main(["rtsegmentator-import", "--course-dir", str(course_dir), str(bad)]) == 1
