"""RTsegmentator as a segmentation source for RTpipeline courses.

RTsegmentator is a separate DICOM RT segmentation portal that serves many
third-party models behind one versioned API. This module sends a course's
contracted planning CT to a portal, or reads a result package downloaded from
one, verifies the package against its ``rtsegmentator_manifest.json``, and writes
each model's RTSTRUCT under ``Segmentation_CustomModels/rtseg_<model>/``.

That directory is the existing custom-model contract, so the DVH, radiomics
and robustness stages consume RTsegmentator structures without a separate
code path. The ``manifest.json`` written there prespecifies the exact ROI
inventory (the ROI names the portal wrote) and keeps the portal's model,
licence and job provenance next to the structures.

An RTSTRUCT is imported only when it references the course's planning CT
series. A package for another series, a checksum mismatch, a member missing
from the manifest, or an ROI inventory that disagrees with the manifest is
rejected; nothing is written for a rejected package.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import pydicom

from .course_contract import load_course_contract
from .segmenters import rtsegmentator_client as client_module

logger = logging.getLogger(__name__)

OUTPUT_PREFIX = "rtseg_"
SOURCE_NAME = "rtsegmentator"
CUSTOM_MODELS_DIRNAME = "Segmentation_CustomModels"
IMPORT_LEDGER = Path("metadata") / "rtsegmentator_imports.json"
# Model directories are assembled here, beside (never inside) Segmentation_CustomModels,
# so an interrupted import cannot leave a directory that consumers enumerate.
STAGING_DIRNAME = ".rtsegmentator-staging"
CUSTOM_STAGE_SENTINEL = ".custom_models_done"
_MODEL_NAME = re.compile(r"^[A-Za-z0-9_]+$")
_JOB_ID = re.compile(r"^[A-Za-z0-9-]+$")


class RTsegmentatorError(RuntimeError):
    """An RTsegmentator submission or import could not be completed."""


@dataclass(frozen=True)
class RTsegmentatorSettings:
    """The ``rtsegmentator`` section of the pipeline configuration."""

    enabled: bool = False
    url: Optional[str] = None
    models: tuple[str, ...] = ()
    headers_env: str = "RTSEG_CLIENT_HEADERS"
    ca_bundle: Optional[str] = None
    request_timeout_seconds: float = 120.0
    poll_seconds: float = 15.0
    job_timeout_seconds: Optional[float] = 6 * 3600.0
    force: bool = False

    @classmethod
    def from_mapping(cls, mapping: Optional[Mapping[str, Any]]) -> "RTsegmentatorSettings":
        data = dict(mapping or {})
        unknown = sorted(set(data) - {
            "enabled", "url", "models", "headers_env", "ca_bundle", "request_timeout_seconds",
            "poll_seconds", "job_timeout_seconds", "force",
        })
        if unknown:
            raise ValueError("Unknown rtsegmentator setting(s): " + ", ".join(unknown))
        raw_models = data.get("models") or []
        if isinstance(raw_models, str):
            raw_models = [item for item in re.split(r"[\s,]+", raw_models) if item]
        models = tuple(str(item).strip() for item in raw_models if str(item).strip())
        for name in models:
            if not _MODEL_NAME.match(name):
                raise ValueError(f"rtsegmentator.models contains an invalid model name: {name!r}")
        if len(set(models)) != len(models):
            raise ValueError("rtsegmentator.models contains duplicates")
        enabled = data.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("rtsegmentator.enabled must be a YAML boolean (true/false)")
        if not isinstance(data.get("force", False), bool):
            raise ValueError("rtsegmentator.force must be a YAML boolean (true/false)")
        url = str(data["url"]).strip() if data.get("url") else None
        if enabled and not url:
            raise ValueError("rtsegmentator.url is required when rtsegmentator.enabled is true")
        if enabled and not models:
            raise ValueError("rtsegmentator.models must name at least one model when enabled")
        job_timeout = data.get("job_timeout_seconds", cls.job_timeout_seconds)
        return cls(
            enabled=enabled,
            url=url,
            models=models,
            headers_env=str(data.get("headers_env") or cls.headers_env),
            ca_bundle=str(data["ca_bundle"]) if data.get("ca_bundle") else None,
            request_timeout_seconds=float(data.get("request_timeout_seconds", cls.request_timeout_seconds)),
            poll_seconds=float(data.get("poll_seconds", cls.poll_seconds)),
            job_timeout_seconds=None if job_timeout in (None, 0) else float(job_timeout),
            force=data.get("force", False),
        )

    def scientific_identity(self) -> dict[str, Any]:
        """Settings that change which structures a course receives (for stage dependencies)."""
        return {"enabled": self.enabled, "url": self.url, "models": list(self.models)}


@dataclass
class ImportedModel:
    model: str
    outcome: str
    output_dir: Optional[Path] = None
    roi_names: list[str] = field(default_factory=list)


def output_dir_name(model: str) -> str:
    if not _MODEL_NAME.match(model):
        raise RTsegmentatorError(f"Invalid RTsegmentator model name: {model!r}")
    return OUTPUT_PREFIX + model


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def planning_ct_identity(course_dir: Path) -> tuple[Path, str, str]:
    """Return the contracted planning CT directory, its SeriesInstanceUID and a SOP-set digest."""
    contract = load_course_contract(Path(course_dir))
    ct_dir = contract.planning_ct_dir
    series_uid = str(contract.planning_ct.get("series_instance_uid") or "").strip()
    if ct_dir is None or not series_uid:
        raise RTsegmentatorError(f"Course {course_dir} has no contracted planning CT series")
    sop_uids = []
    for path in sorted(Path(ct_dir).iterdir()):
        if not path.is_file():
            continue
        try:
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True, specific_tags=["SOPInstanceUID"])
        except Exception:
            continue
        uid = str(getattr(dataset, "SOPInstanceUID", "") or "").strip()
        if uid:
            sop_uids.append(uid)
    if not sop_uids:
        raise RTsegmentatorError(f"Planning CT directory has no readable DICOM instances: {ct_dir}")
    digest = _sha256_bytes("\n".join(sorted(sop_uids)).encode("utf-8"))
    return Path(ct_dir), series_uid, digest


def _rtstruct_identity(data: bytes) -> tuple[list[str], set[str], set[str]]:
    from io import BytesIO

    dataset = pydicom.dcmread(BytesIO(data), stop_before_pixels=True, force=True)
    names = [str(getattr(roi, "ROIName", "") or "").strip()
             for roi in getattr(dataset, "StructureSetROISequence", None) or []]
    series, frames = set(), set()
    for frame in getattr(dataset, "ReferencedFrameOfReferenceSequence", None) or []:
        if getattr(frame, "FrameOfReferenceUID", None):
            frames.add(str(frame.FrameOfReferenceUID))
        for study in getattr(frame, "RTReferencedStudySequence", None) or []:
            for item in getattr(study, "RTReferencedSeriesSequence", None) or []:
                if getattr(item, "SeriesInstanceUID", None):
                    series.add(str(item.SeriesInstanceUID))
    return names, series, frames


def _validated_inventory(names: Sequence[str], context: str) -> list[str]:
    if not names:
        raise RTsegmentatorError(f"{context} contains no ROI")
    if any(not name for name in names):
        raise RTsegmentatorError(f"{context} contains an ROI without a name")
    duplicates = sorted({name for name in names if list(names).count(name) > 1})
    if duplicates:
        raise RTsegmentatorError(f"{context} has duplicate ROI names: {', '.join(duplicates)}")
    return list(names)


def _read_ledger(course_dir: Path) -> dict[str, Any]:
    path = Path(course_dir) / IMPORT_LEDGER
    if not path.is_file():
        return {"schema": "rtpipeline.rtsegmentator-imports", "version": 1, "models": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RTsegmentatorError(f"RTsegmentator import ledger is unreadable: {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
        raise RTsegmentatorError(f"RTsegmentator import ledger is malformed: {path}")
    return data


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def import_result_package(
    package: Path,
    course_dir: Path,
    *,
    models: Optional[Iterable[str]] = None,
    url: Optional[str] = None,
) -> list[ImportedModel]:
    """Verify an RTsegmentator result ZIP and import its RTSTRUCTs into ``course_dir``.

    ``models`` restricts the import to those model names; every named model must
    appear in the package. Returns one record per imported or empty model.
    """
    course_dir = Path(course_dir)
    package = Path(package)
    try:
        manifest = client_module.verify_result_zip(package)
    except (client_module.ManifestError, zipfile.BadZipFile, OSError) as exc:
        raise RTsegmentatorError(f"RTsegmentator package {package} failed verification: {exc}") from exc
    _ct_dir, series_uid, ct_digest = planning_ct_identity(course_dir)

    source = manifest.get("source_series") or {}
    if source.get("series_instance_uid") != series_uid:
        # Also refuses a package without a source series: even a "no structures found"
        # outcome must be bound to this course's planning CT before it replaces anything.
        raise RTsegmentatorError(
            f"RTsegmentator package {package.name} was produced for series "
            f"{source.get('series_instance_uid')!r}, not the planning CT {series_uid}"
        )
    entries = {str(entry.get("name")): entry for entry in manifest.get("models") or [] if isinstance(entry, dict)}
    if models is not None:
        requested = list(models)
    else:
        # Report-only outputs (for example AI draft reports) carry no structures.
        requested = [name for name, entry in entries.items()
                     if entry.get("outcome") in {"rtstruct", "no_structures_found"}]
    missing = [name for name in requested if name not in entries]
    if missing:
        raise RTsegmentatorError(
            f"RTsegmentator package {package.name} has no result for model(s): {', '.join(missing)}"
        )

    package_sha256 = _sha256_file(package)
    staged: list[tuple[ImportedModel, Optional[bytes], dict[str, Any]]] = []
    with zipfile.ZipFile(package) as archive:
        for name in requested:
            entry = entries[name]
            outcome = str(entry.get("outcome") or "")
            directory = output_dir_name(name)
            if outcome == "no_structures_found":
                staged.append((ImportedModel(name, outcome), None, {}))
                continue
            if outcome != "rtstruct":
                raise RTsegmentatorError(
                    f"RTsegmentator model {name!r} produced no RTSTRUCT (outcome {outcome!r})"
                )
            record = entry.get("rtstruct") or {}
            data = archive.read(str(record.get("path")))
            names, referenced_series, frames = _rtstruct_identity(data)
            context = f"RTsegmentator RTSTRUCT for {name!r}"
            names = _validated_inventory(names, context)
            if names != list(record.get("roi_names") or []):
                raise RTsegmentatorError(f"{context} ROI names disagree with the package manifest")
            if referenced_series != {series_uid}:
                raise RTsegmentatorError(
                    f"{context} references series {sorted(referenced_series)}, not the planning CT {series_uid}"
                )
            catalog = [str(item) for item in entry.get("catalog_structures") or []]
            model_manifest = {
                "model": directory,
                "source": SOURCE_NAME,
                "rtstruct": "rtstruct.dcm",
                "expected_structures": names,
                "produced_structures": names,
                "missing_structures": [],
                "catalog_structures": catalog,
                # Catalog labels the portal did not write: the model did not find them,
                # or the exporter omitted them (the package notes then name them).
                "absent_catalog_structures": [item for item in catalog if item not in names],
                "package_notes": [str(note) for note in manifest.get("notes") or []],
                "generated_at": _utc_now(),
                "planning_ct": {"series_instance_uid": series_uid, "sop_instance_set_sha256": ct_digest,
                                "frame_of_reference_uids": sorted(frames)},
                "rtsegmentator": {
                    "model": name,
                    "url": url,
                    "job_id": (manifest.get("job") or {}).get("id"),
                    "generator": manifest.get("generator"),
                    "manifest_schema_version": manifest.get("schema_version"),
                    "package_sha256": package_sha256,
                    "rtstruct_sha256": _sha256_bytes(data),
                    **{key: entry[key] for key in (
                        "structure_kind", "description", "reference", "license", "license_note", "code_license",
                        "weights_license", "noncommercial", "licensed", "limitations",
                    ) if key in entry},
                },
            }
            staged.append((ImportedModel(name, outcome, course_dir / CUSTOM_MODELS_DIRNAME / directory, names),
                           data, model_manifest))

    # All checks passed; publish each model directory atomically.
    root = course_dir / CUSTOM_MODELS_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    staging = course_dir / STAGING_DIRNAME
    shutil.rmtree(staging, ignore_errors=True)  # leftovers of an interrupted import
    staging.mkdir()
    ledger = _read_ledger(course_dir)
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    try:
        for imported, data, model_manifest in staged:
            target = root / output_dir_name(imported.model)
            if data is None:
                if target.exists():
                    os.replace(target, staging / f"{target.name}.removed")
            else:
                temporary = Path(tempfile.mkdtemp(prefix=f"{target.name}.", dir=staging))
                (temporary / "rtstruct.dcm").write_bytes(data)
                (temporary / "rtsegmentator_manifest.json").write_bytes(manifest_bytes)
                (temporary / "manifest.json").write_text(
                    json.dumps(model_manifest, indent=2) + "\n", encoding="utf-8")
                if target.exists():
                    os.replace(target, staging / f"{target.name}.replaced")
                os.replace(temporary, target)
            ledger["models"][imported.model] = {
                "outcome": imported.outcome,
                "roi_names": imported.roi_names,
                "job_id": (manifest.get("job") or {}).get("id"),
                "generator": manifest.get("generator"),
                "url": url,
                "planning_ct_series_instance_uid": series_uid,
                "planning_ct_sop_instance_set_sha256": ct_digest,
                "package_sha256": package_sha256,
                "imported_at": _utc_now(),
            }
            logger.info("Imported RTsegmentator %s for %s: %s (%d ROI)", imported.model, course_dir,
                        imported.outcome, len(imported.roi_names))
        _write_json_atomic(course_dir / IMPORT_LEDGER, ledger)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return [item for item, _data, _manifest in staged]


def _import_intact(output: Path, record: Mapping[str, Any]) -> bool:
    """The imported directory still holds the RTSTRUCT and manifest the ledger recorded."""
    try:
        model_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        provenance = model_manifest["rtsegmentator"]
        return (
            (output / "rtsegmentator_manifest.json").is_file()
            and provenance.get("job_id") == record.get("job_id")
            and _sha256_file(output / "rtstruct.dcm") == provenance.get("rtstruct_sha256")
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def _current_models(course_dir: Path, settings: RTsegmentatorSettings, series_uid: str, digest: str) -> set[str]:
    """Models whose recorded import still matches this planning CT and portal."""
    ledger = _read_ledger(course_dir)
    current = set()
    for name in settings.models:
        record = ledger["models"].get(name) or {}
        if (record.get("planning_ct_series_instance_uid") != series_uid
                or record.get("planning_ct_sop_instance_set_sha256") != digest
                or record.get("url") != settings.url):
            continue
        if record.get("outcome") == "rtstruct":
            if not _import_intact(Path(course_dir) / CUSTOM_MODELS_DIRNAME / output_dir_name(name), record):
                continue
        elif record.get("outcome") != "no_structures_found":
            continue
        current.add(name)
    return current


def _prune_unconfigured(course_dir: Path, settings: RTsegmentatorSettings) -> list[str]:
    """Remove imports this portal produced for models no longer configured.

    Offline imports (no recorded URL) and imports from another portal are kept.
    """
    if not (Path(course_dir) / IMPORT_LEDGER).is_file():
        return []
    ledger = _read_ledger(course_dir)
    removed = []
    for name, record in sorted(ledger["models"].items()):
        if name in settings.models or not isinstance(record, dict) or record.get("url") != settings.url:
            continue
        target = Path(course_dir) / CUSTOM_MODELS_DIRNAME / output_dir_name(name)
        if target.exists():
            shutil.rmtree(target)
        del ledger["models"][name]
        removed.append(name)
    if removed:
        _write_json_atomic(Path(course_dir) / IMPORT_LEDGER, ledger)
        logger.info("Removed RTsegmentator imports no longer configured for %s: %s", course_dir, ", ".join(removed))
    return removed


def build_client(settings: RTsegmentatorSettings) -> "client_module.RTsegClient":
    headers = client_module.parse_headers([os.environ.get(settings.headers_env, "")])
    return client_module.RTsegClient(
        settings.url or "", headers=headers, timeout=settings.request_timeout_seconds,
        ca_bundle=settings.ca_bundle,
    )


def run_for_course(
    settings: RTsegmentatorSettings,
    course_dir: Path,
    *,
    client: Optional["client_module.RTsegClient"] = None,
    force: bool = False,
) -> list[ImportedModel]:
    """Submit the course's planning CT to RTsegmentator and import the configured models."""
    if not settings.enabled:
        return []
    course_dir = Path(course_dir)
    contract = load_course_contract(course_dir)
    if contract.planning_ct_dir is None:
        # Same disposition as local custom models: nothing to segment.
        logger.info("RTsegmentator skipped for %s: the course contract has no planning CT", course_dir)
        return []
    _prune_unconfigured(course_dir, settings)
    ct_dir, series_uid, digest = planning_ct_identity(course_dir)
    pending = list(settings.models)
    if not (force or settings.force):
        current = _current_models(course_dir, settings, series_uid, digest)
        pending = [name for name in pending if name not in current]
        if not pending:
            logger.info("RTsegmentator results for %s are current; skipping", course_dir)
            return []
    client = client or build_client(settings)
    files = client_module.collect_files([ct_dir])
    try:
        session = client.upload_files(files)
        upload = client.assess(session, poll=max(1.0, settings.poll_seconds / 5))
        chosen = client_module.select_series(upload.get("series", []), series_uids=[series_uid])
        if len(chosen) != 1:
            raise RTsegmentatorError(f"RTsegmentator assessed {len(chosen)} series for {series_uid}; expected one")
        batch = client.submit(upload["upload_id"], [chosen[0]["key"]], pending)
        jobs = batch.get("jobs") or []
        if len(jobs) != 1:
            raise RTsegmentatorError(f"RTsegmentator created {len(jobs)} jobs for one series")
        skipped = jobs[0].get("skipped_models") or {}
        if skipped:
            raise RTsegmentatorError(f"RTsegmentator skipped model(s) for this series: {skipped}")
        job_id = str(jobs[0].get("id") or "")
        if not _JOB_ID.match(job_id):
            raise RTsegmentatorError(f"RTsegmentator returned an invalid job ID: {job_id!r}")
        logger.info("RTsegmentator job %s submitted for %s with %s", job_id, course_dir, ", ".join(pending))
        client.wait(job_id, poll=settings.poll_seconds, timeout=settings.job_timeout_seconds)
        with tempfile.TemporaryDirectory(prefix="rtseg-", dir=course_dir) as scratch:
            package = client.download(job_id, Path(scratch) / f"{job_id}.zip")
            # The packaged manifest must be the one the portal serves for this job.
            client_module.verify_against_portal(client, job_id, package)
            return import_result_package(package, course_dir, models=pending, url=settings.url)
    except client_module.ClientError as exc:
        raise RTsegmentatorError(f"RTsegmentator request failed for {course_dir}: {exc}") from exc


def import_main(argv: Optional[Sequence[str]] = None) -> int:
    """``rtpipeline rtsegmentator-import``: import downloaded result packages into a course."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="rtpipeline rtsegmentator-import",
        description="Verify RTsegmentator result ZIPs and import their RTSTRUCTs into an organized course.",
    )
    parser.add_argument("--course-dir", required=True, help="Organized course directory (Output/PATIENT/COURSE)")
    parser.add_argument("--model", action="append", default=None,
                        help="Import only this model; repeat for several (default: every model in the package)")
    parser.add_argument("packages", nargs="+", help="Result ZIP(s) downloaded from RTsegmentator")
    args = parser.parse_args(argv)
    course_dir = Path(args.course_dir)
    imported_any = False
    try:
        for package in args.packages:
            for item in import_result_package(Path(package), course_dir, models=args.model):
                imported_any = True
                print(json.dumps({"package": str(package), "model": item.model, "outcome": item.outcome,
                                  "rois": len(item.roi_names),
                                  "output": str(item.output_dir) if item.output_dir else None}))
    except RTsegmentatorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if imported_any:
            # The custom-model stage receipt no longer describes Segmentation_CustomModels.
            # Removing it makes the next workflow run repeat that stage for this course and
            # re-run DVH, radiomics and robustness on the imported structures.
            (course_dir / CUSTOM_STAGE_SENTINEL).unlink(missing_ok=True)
    return 0
