"""Bound, immutable DVH generations and crash recovery (one writer per course).

Fixed artifact names are aliases through one atomic current-generation pointer.
Readers verify the commit and all hashes before accepting any alias. Attempt
state is independent of that pointer and of Snakemake's DAG closure.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

from .durable_progress import (Binding, DurableProgressError, ProgressJournal,
                               ResultCache, resume_plan, _fsync_dir, file_sha256,
                               final_result_writer, make_binding)

SCHEMA = "rtpipeline-dvh-generation-v1"
ARTIFACTS = ("dvh_metrics.parquet", "dvh_metrics.xlsx", "dvh_curves.json",
             "metadata/dvh_qc.json", "RS_custom.dcm", "metadata/rs_custom_meta.json",
             "metadata/custom_structure_warnings.json")
_PUBLICATION = ContextVar("dvh_publication", default=None)
_UPSTREAM_RECEIPTS = (".organized", ".segmentation_done", ".custom_models_done", ".crop_ct_done")
_UPSTREAM_FIELDS = ("schema", "status", "stage", "stage_name", "patient_id", "course_id",
                    "content_closure_sha256", "output_set_sha256",
                    "configuration_dependency_sha256", "code_identity_sha256", "environment_fingerprint")


def output_path(course, relative):
    context = _PUBLICATION.get()
    return (context[1] if context and Path(course) == context[0] else Path(course)) / relative


def logical_path(path):
    context = _PUBLICATION.get()
    if context:
        try:
            return context[0] / Path(path).relative_to(context[1])
        except ValueError:
            pass
    return Path(path)


def staging():
    return _PUBLICATION.get() is not None


@contextmanager
def publication(course, destination, roi_progress=None):
    token = _PUBLICATION.set((Path(course), Path(destination), roi_progress))
    try:
        yield
    finally:
        _PUBLICATION.reset(token)


def input_manifest(course, *, include_derived=True):
    """Hash complete source inventories, including every CT and mask byte.

    DICOM bytes include SOP/series/reference identities. Outputs, runtime audit
    logs and the DVH generation itself are deliberately outside this inventory.
    """
    course = Path(course)
    pending = course / ".dvh/upstream-recompute.json"
    if pending.is_file() and any(not (course / name).is_file() for name in _read(pending)["upstream"]):
        raise DurableProgressError("DVH_UPSTREAM_RECOMPUTE_REQUIRED")
    files = set()
    for name in ("DICOM", "CT", "NIFTI", "Segmentation_TotalSegmentator", "Segmentation_CustomModels"):
        root = course / name
        if root.exists():
            files.update(p for p in root.rglob("*") if p.is_file() and p.suffix != ".age")
    if include_derived:
        root = course / "metadata/dvh_derived_masks"
        if root.exists():
            files.update(p for p in root.rglob("*") if p.is_file() and p.suffix != ".age")
    for name in ("RS_auto.dcm", "RS_auto_cropped.dcm", "cropping_metadata.json",
                 "metadata/case_metadata.json", "metadata/rs_auto_origin.json",
                 "metadata/planning_dose_selection.json", "metadata/composite_course_dose.json"):
        path = course / name
        if path.is_file():
            files.add(path)
    manifest = [{"path": p.relative_to(course).as_posix(), "sha256": file_sha256(p)}
                for p in sorted(files)]
    # The upstream generation identity binds content, code, configuration and
    # environment. Receipt creation time and host are audit fields, not inputs.
    for name in _UPSTREAM_RECEIPTS:
        path = course / name
        if path.is_file():
            receipt = _read(path)
            if not isinstance(receipt, dict) or any(field not in receipt for field in _UPSTREAM_FIELDS):
                raise DurableProgressError("DVH_UPSTREAM_LEGACY_UNBOUND")
            manifest.append({"path": name, "generation": {key: receipt[key] for key in _UPSTREAM_FIELDS}})
    return manifest


def dvh_rule_text(snakefile):
    """Extract the single top-level DVH rule, excluding neighboring statements.

    Snakemake's rule suite is indented. Preserve its exact text (including
    comments inside it), but exclude trailing blank lines and all other rules.
    Missing/duplicate rules refuse a binding instead of falling back to a file.
    """
    lines = Path(snakefile).read_text().splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.rstrip() == "rule dvh_course:"]
    if len(starts) != 1:
        raise ValueError("DVH rule is missing or ambiguous")
    start = starts[0]
    end = start + 1
    while end < len(lines) and (not lines[end].strip() or lines[end][0].isspace()):
        end += 1
    return "".join(lines[start:end]).rstrip() + "\n"


def binding(course, settings, custom_config=None, *, custom_only=False, extra_inputs=(),
            _include_derived=True):
    from .dvh import DVH_MEASUREMENT_CODE_SOURCES
    from .custom_structures_rtstruct import RS_CUSTOM_GENERATOR_CODE_SOURCES
    from .stage_completion import environment_record
    package = Path(__file__).resolve().parent
    sources = RS_CUSTOM_GENERATOR_CODE_SOURCES if custom_only else DVH_MEASUREMENT_CODE_SOURCES
    sources = set(sources) | {"dvh_generation.py", "durable_progress.py", "radiomics_rasterizer.py",
                              "layout.py", "utils.py", "stage_completion.py", "plan_profiles.py", "clinical_prescription.py",
                              "planning_ct_localizers.py", "ct_series_hygiene.py", "nifti_provenance.py"}
    if not custom_only:
        sources.update(("custom_models.py", "config.py", "cli.py"))
    config = dict(settings)
    try:
        config["custom_config_sha256"] = file_sha256(custom_config) if custom_config else None
    except OSError as exc:
        raise DurableProgressError("DVH_BINDING_UNREADABLE") from exc
    if custom_only:
        # Radiomics validates this main-runtime publication from NumPy 1. Its
        # caller runtime cannot become the identity of the NumPy 2 producer.
        config["builder_contract"] = "numpy2-opencv5"
    else:
        try:
            parameters = json.loads(os.environ.get("RTPIPELINE_DVH_RULE_PARAMETERS", "{}"))
            if not isinstance(parameters, dict):
                raise ValueError("workflow parameters must be an object")
            config["workflow_parameters"] = parameters
        except (ValueError, TypeError) as exc:
            raise DurableProgressError("DVH_BINDING_UNREADABLE") from exc
        config["environment"] = environment_record(("numpy", "pandas", "pydicom", "SimpleITK",
            "scipy", "xlsxwriter", "openpyxl", "dicompyler-core", "rt-utils", "opencv-python", "opencv-python-headless", "pyarrow"))
    try:
        inputs = input_manifest(course, include_derived=_include_derived and not custom_only)
        inputs.extend({"source": name, "sha256": file_sha256(path)} for name, path in extra_inputs)
        code_files = {name: package / name for name in sources}
        if not custom_only:
            for name in ("workflow/scripts/run_course_stage.py", "workflow/scripts/campaign_ledger.py",
                         "rtpipeline/workflow_delegate.py", "rtpipeline/snakemake_delegate.py"):
                code_files[name] = package.parent / name
        result = make_binding(code_files, config, inputs)
        if not custom_only:
            code = {"modules_sha256": result.code_sha256,
                    "dvh_rule": dvh_rule_text(package.parent / "Snakefile")}
            result = Binding(hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest(),
                             result.config_sha256, result.inputs_sha256)
        return result
    except (OSError, ValueError, TypeError) as exc:
        raise DurableProgressError("DVH_BINDING_UNREADABLE") from exc


def settings_from_config(config):
    crop = config.get("ct_cropping") or {}
    return {"use_cropped": crop.get("use_cropped_for_dvh", True),
            "estimate_rx_from_ctv1": False, "rx_dose_gy": None,
            "max_total_dose_gy": float(config.get("max_total_dose_gy", 100.0)),
            "dvh": {k: v for k, v in (config.get("dvh") or {}).items() if k != "retry_failed"}}


def effective_settings(use_cropped=True, estimate_rx_from_ctv1=False, rx_dose_gy=None,
                       max_total_dose_gy=100.0):
    extra = {}
    configfile = os.environ.get("RTPIPELINE_CONFIGFILE")
    if configfile:
        import yaml
        extra = {k: v for k, v in ((yaml.safe_load(Path(configfile).read_text()) or {}).get("dvh") or {}).items()
                 if k != "retry_failed"}
    return dict(use_cropped=use_cropped, estimate_rx_from_ctv1=estimate_rx_from_ctv1,
                rx_dose_gy=rx_dose_gy, max_total_dose_gy=float(max_total_dose_gy) if max_total_dose_gy is not None else None, dvh=extra)



def _encode_row(value):
    import math
    if isinstance(value, dict):
        return {"__dvh_mapping__": [[key, _encode_row(item)] for key, item in value.items()]}
    if isinstance(value, (tuple, list)):
        return [_encode_row(item) for item in value]
    if hasattr(value, "item"):
        return _encode_row(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"__dvh_float__": "nan" if math.isnan(value) else "inf" if value > 0 else "-inf"}
    return value


def _decode_row(value):
    if isinstance(value, dict):
        if set(value) == {"__dvh_mapping__"}:
            return {key: _decode_row(item) for key, item in value["__dvh_mapping__"]}
        if set(value) == {"__dvh_float__"}:
            return float(value["__dvh_float__"])
        return {key: _decode_row(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_row(item) for item in value]
    return value



def validate_row_sources(row):
    kind = row.get("structure_provenance_type")
    if kind not in {"NIFTI_MASK", "DERIVED_NIFTI_MASK"}:
        return
    from .dvh_support import nifti_identity, validate_derived_mask
    expected = row.get("structure_provenance_sha256")
    source = row.get("structure_provenance_path")
    if not isinstance(expected, str) or len(expected) != 64:
        raise DurableProgressError("DVH_MASK_PROVENANCE_INVALID")
    try:
        identity = (nifti_identity(source) if kind == "NIFTI_MASK"
                    else validate_derived_mask(source, expected))
        if identity["sha256"] != expected:
            raise ValueError("mask identity changed")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        raise DurableProgressError("DVH_MASK_PROVENANCE_INVALID") from exc

def roi_progress():
    context = _PUBLICATION.get()
    return context[2] if context else None


def roi_key(task):
    # Opaque key: no patient/course label or source UID is written in cache metadata.
    identity = ["NIFTI_Direct", task[0]] if len(task) == 2 else [str(task[9]), task[8], task[0], task[3]]
    return hashlib.sha256(json.dumps(identity,
                                    separators=(",", ":")).encode()).hexdigest()


def roi_resume(tasks, progress):
    if not progress:
        return {}
    cache, expected, journal = progress
    keys = [roi_key(task) for task in tasks]
    reused, _, report = resume_plan(keys, cache, expected)
    _event(journal, "roi_resume", **report)
    result = {}
    for key, entry in reused.items():
        row = _decode_row(entry["row"])
        try:
            validate_row_sources(row)
        except DurableProgressError:
            _event(journal, "roi_dependency_invalid", item_key=key, reason_code="DVH_MASK_PROVENANCE_INVALID")
            continue
        result[key] = row
    return result


def commit_roi(task, row, progress):
    if not progress or row is None:
        return
    cache, expected, journal = progress
    key = roi_key(task)
    try:
        cache.put(key, {"row": _encode_row(row)}, expected)
        _event(journal, "roi_committed", item_key=key)
    except DurableProgressError:
        raise
    except Exception as exc:
        raise DurableProgressError("DVH_ROI_PERSISTENCE_FAILED") from exc
    _failpoint("staging")

def _read(path):
    return json.loads(Path(path).read_text())


def verify(directory, expected=None):
    directory = Path(directory)
    marker = _read(directory / "commit.json")
    if not isinstance(marker, dict):
        raise DurableProgressError("DVH_GENERATION_CORRUPT")
    if marker.get("schema") != SCHEMA or marker.get("binding") is None:
        raise DurableProgressError("DVH_LEGACY_UNBOUND")
    saved = Binding(**marker["binding"])
    if expected is not None and saved != expected:
        fields = ("code_sha256", "config_sha256", "inputs_sha256")
        reason = next(k.split("_")[0].upper() for k in fields
                      if saved.as_dict()[k] != expected.as_dict()[k])
        raise DurableProgressError("DVH_BINDING_" + reason + "_CHANGED")
    artifacts = marker["artifacts"]
    if not isinstance(artifacts, dict):
        raise DurableProgressError("DVH_GENERATION_INVENTORY_INVALID")
    if set(artifacts) != set(ARTIFACTS) | {".dvh_done", "ledger.json"}:
        raise DurableProgressError("DVH_GENERATION_INVENTORY_INVALID")
    for name, digest in artifacts.items():
        path = directory / name
        if digest is None:
            if path.exists():
                raise DurableProgressError("DVH_GENERATION_HASH_MISMATCH")
        elif not path.is_file() or file_sha256(path) != digest:
            raise DurableProgressError("DVH_GENERATION_HASH_MISMATCH")
    receipt = _read(directory / ".dvh_done")
    if not isinstance(receipt, dict):
        raise DurableProgressError("DVH_GENERATION_RECEIPT_INVALID")
    outcome = "computed" if artifacts["dvh_metrics.parquet"] else "not_measured"
    if receipt.get("measurement_outcome") != outcome:
        raise DurableProgressError("DVH_GENERATION_OUTCOME_INVALID")
    if receipt.get("binding") != saved.as_dict() or receipt.get("generation_id") != marker["generation_id"]:
        raise DurableProgressError("DVH_GENERATION_RECEIPT_INVALID")
    if artifacts["dvh_metrics.parquet"]:
        import pandas as pd
        for row in pd.read_parquet(directory / "dvh_metrics.parquet").to_dict("records"):
            validate_row_sources(row)
    return marker


def current(course, expected=None):
    try:
        directory = (Path(course) / ".dvh/current").resolve(strict=True)
        directory.relative_to((Path(course) / ".dvh/generations").resolve())
        return verify(directory, expected)
    except FileNotFoundError as exc:
        raise DurableProgressError("DVH_LEGACY_UNBOUND" if (Path(course) / ".dvh_done").exists()
                                   else "DVH_GENERATION_MISSING") from exc


def _event(journal, event, **fields):
    journal.append({"event": event, **fields})


def route_legacy_upstream(course, journal):
    """Invalidate unbound stage receipts so their DAG producers must run again."""
    course = Path(course)
    pending = course / ".dvh/upstream-recompute.json"
    invalid = set(_read(pending)["upstream"] if pending.is_file() else ())
    for name in _UPSTREAM_RECEIPTS:
        path = course / name
        if not path.is_file():
            continue
        try:
            receipt = _read(path)
            bound = isinstance(receipt, dict) and all(key in receipt for key in _UPSTREAM_FIELDS)
        except (ValueError, TypeError):
            bound = False
        if not bound:
            invalid.add(name)
            # Persist the rerouting requirement before invalidating the alias.
            # A second run must still refuse until a fresh receipt is present.
            final_result_writer(pending, {"upstream": sorted(invalid),
                                "reason_code": "DVH_UPSTREAM_RECOMPUTE_REQUIRED"})
            _event(journal, "upstream_recompute_required", upstream=name,
                   reason_code="DVH_UPSTREAM_LEGACY_UNBOUND")
            path.unlink()
            _fsync_dir(path.parent)
        else:
            invalid.discard(name)
    if invalid:
        final_result_writer(pending, {"upstream": sorted(invalid),
                            "reason_code": "DVH_UPSTREAM_RECOMPUTE_REQUIRED"})
    elif pending.exists():
        pending.unlink()
        _fsync_dir(pending.parent)
    return sorted(invalid)


def _failpoint(name):
    """Inert production seam; tests monkeypatch this in owned child processes."""


def _aliases(course):
    for name in (*ARTIFACTS, ".dvh_done"):
        path = course / name
        path.parent.mkdir(parents=True, exist_ok=True)
        target = os.path.relpath(course / ".dvh/current" / name, path.parent)
        if path.is_symlink() and os.readlink(path) == target:
            continue
        temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
        temporary.symlink_to(target)
        os.replace(temporary, path)
        _fsync_dir(path.parent)


def _activate(course, directory):
    root = course / ".dvh"
    _aliases(course)
    temporary = root / (".current-" + uuid.uuid4().hex)
    temporary.symlink_to(os.path.relpath(directory, root))
    os.replace(temporary, root / "current")
    _fsync_dir(root)


def reconcile_ledger(course, marker):
    """Idempotent projection of committed evidence, never evidence of closure."""
    course = Path(course)
    record = _read(course / ".dvh/current/ledger.json")
    directory = course.parents[1] / "_campaign_ledger/records"
    directory.mkdir(parents=True, exist_ok=True)
    _fsync_dir(directory.parent)
    _fsync_dir(directory.parent.parent)
    target = directory / ("__".join((course.parent.name, course.name, "dvh")) + ".json")
    try:
        projected = _read(target)
    except (OSError, ValueError):
        projected = None
    if projected != record:
        final_result_writer(target, record)
    if _read(target) != record:
        raise DurableProgressError("DVH_LEDGER_VERIFY_FAILED")
    # A controller may have died after the generation commit. Reconcile both
    # durable attempt projections without launching a child or repeating a ROI.
    attempt_paths = (course / ".dvh/attempt.json",
        directory.parent / "attempts" / target.name)
    for path in attempt_paths:
        if not path.is_file():
            continue
        attempt = _read(path)
        if attempt.get("binding") == marker["binding"] and attempt.get("state") != "committed":
            final_result_writer(path, {**attempt, "state": "committed", "reason_code": None})


def _validate_products(course, destination):
    import pandas as pd
    qc = _read(destination / "metadata/dvh_qc.json")
    parquet = destination / "dvh_metrics.parquet"
    if parquet.exists():
        frame = pd.read_parquet(parquet)
        workbook = pd.read_excel(destination / "dvh_metrics.xlsx")
        if len(frame) != len(workbook) or list(frame.columns) != list(workbook.columns):
            raise DurableProgressError("DVH_TABLE_READBACK_FAILED")
        if not len(frame) or qc.get("status") not in {"ok", "degraded"}:
            raise DurableProgressError("DVH_OUTCOME_INVALID")
    else:
        from .course_contract import load_course_contract
        contract = load_course_contract(course)
        decision = contract.data.get("dvh") or {}
        resolution = qc.get("dose_resolution") or {}
        if (qc.get("status") != "skipped" or decision.get("metrics_status") != "not_computed"
                or decision.get("output") is not None or contract.selected_doses
                or contract.data.get("dose_grid") is not None or resolution.get("dose_qc_pass") is not True
                or resolution.get("dvh") != decision
                or qc.get("course_contract_sha256") != file_sha256(contract.metadata_path)):
            raise DurableProgressError("DVH_OUTCOME_INVALID")
    curves = destination / "dvh_curves.json"
    if curves.exists():
        _read(curves)
    rs = destination / "RS_custom.dcm"
    if rs.exists():
        from .rtstruct_identity import require_rtstruct_identity
        require_rtstruct_identity(rs, require_derived=True)
        if _read(destination / "metadata/rs_custom_meta.json")["rs_custom_sha256"] != file_sha256(rs):
            raise DurableProgressError("DVH_RS_CUSTOM_READBACK_FAILED")



def discard_incomplete(root, journal):
    for name, reason in (("staging", "DVH_INCOMPLETE_STAGING"),
                         ("generations", "DVH_INCOMPLETE_GENERATION")):
        directory = root / name
        if not directory.exists():
            continue
        for abandoned in sorted(directory.iterdir()):
            if name == "generations" and (abandoned / "commit.json").exists():
                continue
            _event(journal, "discard", reason_code=reason, generation_id=abandoned.name)
            shutil.rmtree(abandoned)
            _fsync_dir(directory)

def run(course, settings, custom_config, compute, *, retry_failed=False):
    course = Path(course)
    root = course / ".dvh"
    root.mkdir(exist_ok=True)
    _fsync_dir(course)
    with (root / "writer.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with ProgressJournal.recover(root / "progress.jsonl") as journal:
            if route_legacy_upstream(course, journal):
                raise DurableProgressError("DVH_UPSTREAM_RECOMPUTE_REQUIRED")
        expected = binding(course, settings, custom_config)
        # Derived masks are materialized by compute(). Bind their determinants
        # for ROI checkpoints and validate each cached row's exact provenance.
        # The generation additionally binds the materialized closure/mask bytes.
        source_expected = binding(course, settings, custom_config, _include_derived=False)
        with ProgressJournal.recover(root / "progress.jsonl") as journal:
            discard_incomplete(root, journal)
            reason = None
            rejected = set()
            current_directory = (root / "current").resolve()
            try:
                marker = current(course, expected)
            except (DurableProgressError, ValueError, KeyError, TypeError, OSError) as exc:
                reason = getattr(exc, "code", "DVH_GENERATION_CORRUPT")
                rejected.add(current_directory)
            else:
                _aliases(course)
                reconcile_ledger(course, marker)
                _event(journal, "reuse", generation_id=marker["generation_id"])
                return course / "dvh_metrics.xlsx" if marker["artifacts"]["dvh_metrics.xlsx"] else None
            _event(journal, "resume_decision", reason_code=reason, binding=expected.as_dict())
            generations = root / "generations"
            generations.mkdir(exist_ok=True)
            # A complete generation can outlive the controller before activation
            # or ledger projection. Reuse only after fresh exact binding checks.
            for candidate in sorted(generations.iterdir()):
                if candidate.resolve() in rejected:
                    continue
                try:
                    recovered = verify(candidate, expected)
                except (DurableProgressError, ValueError, KeyError, TypeError, OSError):
                    continue  # Preserve committed data, including corrupt evidence.
                _activate(course, candidate)
                reconcile_ledger(course, recovered)
                _event(journal, "recovered_commit", generation_id=candidate.name)
                return course / "dvh_metrics.xlsx" if recovered["artifacts"]["dvh_metrics.xlsx"] else None
            prepared = root / "staging"
            prepared.mkdir(exist_ok=True)
            _fsync_dir(root)
            attempt_path = root / "attempt.json"
            previous = _read(attempt_path) if attempt_path.exists() else {}
            if previous.get("binding") == expected.as_dict():
                if previous.get("state") == "failed" and not retry_failed:
                    raise DurableProgressError("DVH_FAILED_RETRY_REQUIRES_EXPLICIT_RULE")
                if previous.get("state") == "started":
                    _event(journal, "interrupted", reason_code="DVH_INTERRUPTED", attempt_id=previous["attempt_id"])
            identity = uuid.uuid4().hex
            attempt = dict(attempt_id=identity, binding=expected.as_dict(), state="started",
                           attempt_number=(previous.get("attempt_number", 0) + 1
                               if previous.get("binding") == expected.as_dict() else 1),
                           previous_attempt_id=previous.get("attempt_id"))
            final_result_writer(attempt_path, attempt)
            _event(journal, "started", **attempt)
            destination = prepared / identity
            (destination / "metadata").mkdir(parents=True)
            try:
                cache_root = root / "roi-cache"
                if not cache_root.exists():
                    cache_root.mkdir()
                    _fsync_dir(root)
                roi_progress = (ResultCache(cache_root), source_expected, journal)
                with publication(course, destination, roi_progress):
                    compute()
                if binding(course, settings, custom_config, _include_derived=False) != source_expected:
                    raise DurableProgressError("DVH_BINDING_CHANGED_DURING_ATTEMPT")
                _validate_products(course, destination)
                finalized = binding(course, settings, custom_config)
                if finalized != expected:
                    _event(journal, "derived_inputs_finalized",
                           reason_code="DVH_BINDING_INPUTS_CHANGED", binding=finalized.as_dict())
                    expected = finalized
                    attempt = {**attempt, "binding": expected.as_dict()}
                    final_result_writer(attempt_path, attempt)
                outcome = "computed" if (destination / "dvh_metrics.parquet").exists() else "not_measured"
                product_hashes = {name: file_sha256(destination / name) if (destination / name).is_file() else None
                                  for name in ARTIFACTS}
                generation_id = hashlib.sha256(json.dumps(
                    {"binding": expected.as_dict(), "artifacts": product_hashes}, sort_keys=True).encode()).hexdigest()
                receipt = dict(schema=SCHEMA, status="ok", stage=".dvh_done", stage_name="dvh",
                               patient_id=course.parent.name, course_id=course.name,
                               generation_id=generation_id, binding=expected.as_dict(), settings=settings,
                               custom_config=str(custom_config) if custom_config else None,
                               measurement_outcome=outcome)
                final_result_writer(destination / ".dvh_done", receipt)
                final_result_writer(destination / "ledger.json", dict(patient=course.parent.name,
                    course=course.name, stage="dvh", status="ok", attempt_state="committed",
                    generation_id=generation_id, binding=expected.as_dict(), returncode=0,
                    detail=outcome))
                artifacts = {name: file_sha256(destination / name) if (destination / name).is_file() else None
                             for name in (*ARTIFACTS, ".dvh_done", "ledger.json")}
                for path in destination.rglob("*"):
                    if path.is_file():
                        with path.open("rb") as stream:
                            os.fsync(stream.fileno())
                _fsync_dir(destination / "metadata")
                _fsync_dir(destination)
                published = generations / (generation_id + "-" + identity)
                os.replace(destination, published)
                _fsync_dir(generations)
                _fsync_dir(prepared)
                _failpoint("published_before_commit")
                marker = dict(schema=SCHEMA, generation_id=generation_id, binding=expected.as_dict(), artifacts=artifacts)
                final_result_writer(published / "commit.json", marker)
                verify(published, expected)
                _failpoint("committed_before_activation")
                _activate(course, published)
                _failpoint("committed_before_ledger")
                reconcile_ledger(course, marker)
                final_result_writer(attempt_path, {**attempt, "state": "committed"})
                _event(journal, "committed", generation_id=generation_id)
                return course / "dvh_metrics.xlsx" if artifacts["dvh_metrics.xlsx"] else None
            except Exception as exc:
                code = getattr(exc, "code", None) or (
                    "DVH_PERSISTENCE_ERROR" if isinstance(exc, OSError) else
                    "DVH_RESOURCE_LIMIT" if isinstance(exc, MemoryError) else
                    "DVH_STRUCTURAL_INPUT_ERROR" if isinstance(exc, ValueError) else "DVH_STAGE_FAILED")
                final_result_writer(attempt_path, {**attempt, "state": "failed", "reason_code": code})
                _event(journal, "failed", reason_code=code, attempt_id=identity)
                raise


def validate_receipt(course, receipt, *, settings=None, custom_config=None):
    course = Path(course)
    if receipt.get("schema") != SCHEMA:
        raise DurableProgressError("DVH_LEGACY_UNBOUND")
    if (receipt.get("patient_id"), receipt.get("course_id")) != (course.parent.name, course.name):
        raise DurableProgressError("DVH_RECEIPT_IDENTITY_INVALID")
    configfile = os.environ.get("RTPIPELINE_CONFIGFILE")
    if settings is None and configfile:
        import yaml
        config = yaml.safe_load(Path(configfile).read_text()) or {}
        settings = settings_from_config(config)
        from .config import resolve_custom_structures_config
        custom_config = resolve_custom_structures_config(config.get("custom_structures"))
    elif settings is None:
        settings = receipt["settings"]
        custom_config = receipt.get("custom_config") if custom_config is None else custom_config
    expected = binding(course, settings, custom_config)
    marker = current(course, expected)
    actual = _read(course / ".dvh/current/.dvh_done")
    if actual != receipt:
        raise DurableProgressError("DVH_GENERATION_RECEIPT_INVALID")
    roles = {"dvh_metrics.parquet": "authoritative_dvh", "dvh_metrics.xlsx": "dvh_workbook",
             "dvh_curves.json": "dvh_curves", "metadata/dvh_qc.json": "dvh_qc",
             "RS_custom.dcm": "custom_rtstruct", "metadata/rs_custom_meta.json": "custom_rtstruct_metadata",
             "metadata/custom_structure_warnings.json": "custom_structure_warnings"}
    outputs = []
    for name in sorted(ARTIFACTS):
        digest = marker["artifacts"][name]
        alias = course / name
        if digest is None:
            if alias.is_file():
                raise DurableProgressError("DVH_PUBLICATION_ALIAS_INVALID")
            continue
        if not alias.is_file() or file_sha256(alias) != digest:
            raise DurableProgressError("DVH_PUBLICATION_ALIAS_INVALID")
        outputs.append(dict(path=name, role=roles[name], binding="content_sha256", kind="file",
                            sha256=digest, size_bytes=alias.stat().st_size))
    from .stage_completion import content_sha256, _content_closure
    return {**receipt, "outputs": outputs, "output_count": len(outputs),
            "output_set_sha256": content_sha256(outputs), "content_closure_sha256": _content_closure(outputs)}


def preflight(output, configfile, custom_config=None):
    """Run in the pipeline runtime before DAG planning, with current config."""
    import yaml
    config = yaml.safe_load(Path(configfile).read_text()) or {}
    settings = settings_from_config(config)
    result = []
    courses = {p.parent for p in Path(output).glob("*/*/.dvh_done")}
    courses.update(p.parent for p in Path(output).glob("*/*/.dvh"))
    for course in sorted(courses):
        root = course / ".dvh"
        root.mkdir(exist_ok=True)
        with (root / "writer.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with ProgressJournal.recover(root / "progress.jsonl") as journal:
                upstream_recompute = route_legacy_upstream(course, journal)
            try:
                if upstream_recompute:
                    raise DurableProgressError("DVH_UPSTREAM_RECOMPUTE_REQUIRED")
                sentinel = course / ".dvh_done"
                try:
                    receipt = _read(sentinel) if sentinel.is_file() else {}
                except ValueError as exc:
                    raise DurableProgressError("DVH_LEGACY_UNBOUND") from exc
                if not isinstance(receipt, dict):
                    raise DurableProgressError("DVH_LEGACY_UNBOUND")
                if receipt.get("status") == "disabled":
                    if receipt.get("dvh_binding") != disabled_binding(course, settings, custom_config).as_dict():
                        raise DurableProgressError("DVH_LEGACY_UNBOUND" if not receipt.get("dvh_binding")
                                                   else "DVH_BINDING_CHANGED")
                    result.append(dict(course=str(course), reusable=True, status="disabled"))
                    continue
                expected = binding(course, settings, custom_config)
                marker = current(course, expected)
                reconcile_ledger(course, marker)
                # Snakemake may have removed the output alias after a killed job.
                _aliases(course)
                result.append(dict(course=str(course), reusable=True))
            except (DurableProgressError, OSError, ValueError, KeyError, TypeError) as exc:
                reason = getattr(exc, "code", "DVH_GENERATION_CORRUPT")
                with ProgressJournal.recover(root / "progress.jsonl") as journal:
                    _event(journal, "preflight", reason_code=reason)
                (course / ".dvh_done").unlink(missing_ok=True)
                result.append(dict(course=str(course), reusable=False, reason_code=reason))
    return {"courses": result}


def launch_binding(course, configfile, custom_config=None):
    import yaml
    config = yaml.safe_load(Path(configfile).read_text()) or {}
    return {"binding": binding(Path(course), settings_from_config(config), custom_config).as_dict()}


def disabled_binding(course, settings, custom_config=None):
    """A disabled DVH executes no numerical backend and needs no main runtime."""
    package = Path(__file__).resolve().parent
    sources = {name: package / name for name in ("dvh_generation.py", "stage_completion.py", "durable_progress.py")}
    sources["workflow/scripts/run_course_stage.py"] = package.parent / "workflow/scripts/run_course_stage.py"
    config = {**settings, "custom_config_sha256": file_sha256(custom_config) if custom_config else None}
    return make_binding(sources, config, input_manifest(Path(course)))


def completion_context(configuration=None):
    """Current effective disabled-stage settings, in either interpreter."""
    configfile = os.environ.get("RTPIPELINE_CONFIGFILE")
    if configfile:
        import yaml
        config = yaml.safe_load(Path(configfile).read_text()) or {}
        from .config import resolve_custom_structures_config
        return settings_from_config(config), resolve_custom_structures_config(config.get("custom_structures"))
    payload = (configuration or {}).get("payload") or {}
    return {"use_cropped": payload.get("use_cropped", True), "estimate_rx_from_ctv1": False,
            "rx_dose_gy": None, "max_total_dose_gy": float(payload.get("max_total_dose_gy", 100.0)),
            "dvh": payload.get("settings") or {}}, None
