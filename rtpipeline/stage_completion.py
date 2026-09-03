from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import __version__


STAGE_COMPLETION_SCHEMA = "rtpipeline-stage-completion-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RADIOMICS_ENVIRONMENT_PACKAGES = (
    "numpy",
    "scipy",
    "PyWavelets",
    "SimpleITK",
    "pyradiomics",
    "pyarrow",
)


@dataclass(frozen=True)
class ArtifactRule:
    role: str
    pattern: str
    binding: str


@dataclass(frozen=True)
class StageDefinition:
    name: str
    sentinel: str
    configuration_stage: str
    binding_policy: str
    rules: tuple[ArtifactRule, ...]
    required_roles: frozenset[str]
    code_sources: tuple[str, ...]
    environment_packages: tuple[str, ...]


_COMMON_CODE_SOURCES = (
    "__init__.py",
    "cli.py",
    "stage_completion.py",
    "workflow_delegate.py",
    "workflow/scripts/run_course_stage.py",
    "Snakefile",
)

_STAGE_DEFINITIONS: dict[str, StageDefinition] = {
    "organize": StageDefinition(
        name="organize",
        sentinel=".organized",
        configuration_stage="organize",
        binding_policy="organized-hybrid-v1",
        rules=(
            ArtifactRule("course_contract", "metadata/case_metadata.json", "content_sha256"),
            ArtifactRule("course_metadata_workbook", "metadata/case_metadata.xlsx", "content_sha256"),
            ArtifactRule("fraction_metadata", "metadata/fractions_raw.xlsx", "content_sha256"),
            ArtifactRule("fraction_table", "fractions.xlsx", "content_sha256"),
            ArtifactRule("organized_dicom", "DICOM/**/*", "inventory"),
            ArtifactRule("organized_related_dicom", "DICOM_related/**/*", "inventory"),
            ArtifactRule("organized_nifti", "NIFTI/**/*", "inventory"),
            ArtifactRule("original_segmentation", "Segmentation_Original/**/*", "inventory"),
            ArtifactRule("selected_plan_alias", "RP.dcm", "inventory"),
            ArtifactRule("selected_dose_alias", "RD.dcm", "inventory"),
            ArtifactRule("selected_rtstruct_alias", "RS.dcm", "inventory"),
        ),
        required_roles=frozenset({"course_contract", "organized_dicom"}),
        code_sources=(
            *_COMMON_CODE_SOURCES,
            "workflow/scripts/organize_courses.py",
            "organize.py",
            "course_contract.py",
            "prescription.py",
            "clinical_prescription.py",
            "organize_ledger.py",
            "nifti_provenance.py",
        ),
        environment_packages=("numpy", "pandas", "pydicom", "SimpleITK", "openpyxl"),
    ),
    "segmentation": StageDefinition(
        name="segmentation",
        sentinel=".segmentation_done",
        configuration_stage="segmentation",
        binding_policy="segmentation-content-v1",
        rules=(
            ArtifactRule("segmentation_artifact", "Segmentation_TotalSegmentator/**/*", "content_sha256"),
            ArtifactRule("segmentation_rtstruct", "RS_auto.dcm", "content_sha256"),
            ArtifactRule("segmentation_status", "metadata/segmentation_status.json", "content_sha256"),
        ),
        required_roles=frozenset({"segmentation_artifact", "segmentation_status"}),
        code_sources=(*_COMMON_CODE_SOURCES, "segmentation.py", "course_contract.py"),
        environment_packages=("numpy", "pydicom", "SimpleITK", "torch", "totalsegmentator"),
    ),
    "segmentation_custom": StageDefinition(
        name="segmentation_custom",
        sentinel=".custom_models_done",
        configuration_stage="custom-models",
        binding_policy="custom-models-content-v1",
        rules=(
            ArtifactRule("custom_model_artifact", "Segmentation_CustomModels/**/*", "content_sha256"),
        ),
        required_roles=frozenset(),
        code_sources=(*_COMMON_CODE_SOURCES, "custom_models.py", "custom_structures_rtstruct.py", "course_contract.py"),
        environment_packages=("numpy", "pydicom", "SimpleITK", "torch", "nnunetv2"),
    ),
    "crop_ct": StageDefinition(
        name="crop_ct",
        sentinel=".crop_ct_done",
        configuration_stage="crop-ct",
        binding_policy="crop-ct-content-v1",
        rules=(
            ArtifactRule("cropped_image_or_mask", "**/*_cropped.nii.gz", "content_sha256"),
            ArtifactRule("cropped_rtstruct", "RS_auto_cropped.dcm", "content_sha256"),
            ArtifactRule("cropping_metadata", "cropping_metadata.json", "content_sha256"),
        ),
        required_roles=frozenset({"cropped_image_or_mask", "cropping_metadata"}),
        code_sources=(*_COMMON_CODE_SOURCES, "anatomical_cropping.py", "course_contract.py"),
        environment_packages=("numpy", "pydicom", "SimpleITK", "rt-utils"),
    ),
    "dvh": StageDefinition(
        name="dvh",
        sentinel=".dvh_done",
        configuration_stage="dvh",
        binding_policy="dvh-content-v1",
        rules=(
            ArtifactRule("authoritative_dvh", "dvh_metrics.parquet", "content_sha256"),
            ArtifactRule("dvh_workbook", "dvh_metrics.xlsx", "content_sha256"),
            ArtifactRule("dvh_qc", "metadata/dvh_qc.json", "content_sha256"),
            ArtifactRule("custom_rtstruct", "RS_custom.dcm", "content_sha256"),
            ArtifactRule("custom_rtstruct_metadata", "metadata/rs_custom_meta.json", "content_sha256"),
        ),
        required_roles=frozenset({"dvh_qc"}),
        code_sources=(
            *_COMMON_CODE_SOURCES,
            "dvh.py",
            "custom_structures.py",
            "custom_structures_rtstruct.py",
            "course_contract.py",
            "prescription.py",
        ),
        environment_packages=("numpy", "pandas", "pydicom", "scipy", "dicompyler-core", "pyarrow"),
    ),
    "qc": StageDefinition(
        name="qc",
        sentinel=".qc_done",
        configuration_stage="qc",
        binding_policy="qc-content-v1",
        rules=(ArtifactRule("qc_report", "qc_reports/**/*.json", "content_sha256"),),
        required_roles=frozenset({"qc_report"}),
        code_sources=(*_COMMON_CODE_SOURCES, "quality_control.py", "course_contract.py"),
        environment_packages=("numpy", "pydicom", "SimpleITK"),
    ),
}

_SENTINEL_TO_STAGE = {definition.sentinel: name for name, definition in _STAGE_DEFINITIONS.items()}
_STAGE_ALIASES = {
    "organized": "organize",
    "custom_models": "segmentation_custom",
    "custom-models": "segmentation_custom",
    "crop-ct": "crop_ct",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_package_version(distribution_name: str) -> str:
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return "unavailable"


def environment_record(package_names: Sequence[str]) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "libc": list(platform.libc_ver()),
        "packages": {
            name: _safe_package_version(name) for name in sorted(set(package_names))
        },
    }


def environment_fingerprint(package_names: Sequence[str]) -> str:
    return content_sha256(environment_record(package_names))


def execution_environment_fingerprint() -> str:
    """Preserve the radiomics environment-fingerprint API and byte contract."""

    record = environment_record(_RADIOMICS_ENVIRONMENT_PACKAGES)
    legacy_record = {
        "python": record["python"],
        "implementation": record["implementation"],
        "system": record["system"],
        "release": record["release"],
        "machine": record["machine"],
        "libc": tuple(record["libc"]),
        "numpy": record["packages"]["numpy"],
        "scipy": record["packages"]["scipy"],
        "pywavelets": record["packages"]["PyWavelets"],
        "simpleitk": record["packages"]["SimpleITK"],
        "pyradiomics": record["packages"]["pyradiomics"],
        "pyarrow": record["packages"]["pyarrow"],
    }
    return hashlib.sha256(canonical_json(legacy_record).encode("utf-8")).hexdigest()


def current_code_revision() -> str:
    configured = os.environ.get("RTPIPELINE_CODE_REVISION", "").strip()
    if configured:
        return configured
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _normalize_stage(stage: str) -> str:
    value = str(stage).strip()
    if value in _SENTINEL_TO_STAGE:
        return _SENTINEL_TO_STAGE[value]
    value = _STAGE_ALIASES.get(value, value)
    if value not in _STAGE_DEFINITIONS:
        raise ValueError(f"Unsupported completion stage {stage!r}")
    return value


def stage_definition(stage: str) -> StageDefinition:
    return _STAGE_DEFINITIONS[_normalize_stage(stage)]


def _configuration_dependency(path: Path, expected_stage: str) -> dict[str, Any]:
    candidate = Path(path)
    try:
        record = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable stage configuration dependency: {candidate}") from exc
    if not isinstance(record, dict) or record.get("schema") != "rtpipeline-stage-config-dependency-v1":
        raise ValueError(f"Invalid stage configuration dependency: {candidate}")
    if record.get("stage") != expected_stage:
        raise ValueError(
            f"Configuration dependency {candidate} is for {record.get('stage')!r}, not {expected_stage!r}"
        )
    payload = record.get("payload")
    digest = str(record.get("sha256") or "")
    if payload is None or not _SHA256_RE.fullmatch(digest) or content_sha256(payload) != digest:
        raise ValueError(f"Invalid stage configuration dependency digest: {candidate}")
    return record


def _code_identity(definition: StageDefinition) -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    repository_root = package_root.parent
    sources: list[dict[str, str]] = []
    for relative in sorted(set(definition.code_sources)):
        source = package_root / relative
        if not source.is_file():
            source = repository_root / relative
        if not source.is_file():
            raise ValueError(f"Completion code source is absent: {source}")
        sources.append({"path": relative, "sha256": file_sha256(source)})
    return {
        "pipeline_version": __version__,
        "revision": current_code_revision(),
        "sources": sources,
        "sources_sha256": content_sha256(sources),
    }


def _artifact_entry(course_dir: Path, path: Path, role: str, binding: str) -> dict[str, Any]:
    relative = path.relative_to(course_dir).as_posix()
    stat = path.lstat()
    entry: dict[str, Any] = {
        "role": role,
        "path": relative,
        "binding": binding,
        "kind": "symlink" if path.is_symlink() else "file",
        "size_bytes": int(stat.st_size),
    }
    if path.is_symlink():
        entry["symlink_target"] = os.readlink(path)
    if binding == "content_sha256":
        if not path.is_file():
            raise ValueError(f"Content-bound output is not a readable file: {path}")
        entry["sha256"] = file_sha256(path)
    elif binding != "inventory":
        raise ValueError(f"Unsupported artifact binding {binding!r}")
    return entry


def _discover_outputs(course_dir: Path, definition: StageDefinition) -> list[dict[str, Any]]:
    root = Path(course_dir)
    selected: dict[str, tuple[Path, str, str]] = {}
    for rule in definition.rules:
        for candidate in sorted(root.glob(rule.pattern)):
            if not (candidate.is_file() or candidate.is_symlink()):
                continue
            relative = candidate.relative_to(root).as_posix()
            previous = selected.get(relative)
            if previous is None or (
                previous[2] == "inventory" and rule.binding == "content_sha256"
            ):
                selected[relative] = (candidate, rule.role, rule.binding)
    return [
        _artifact_entry(root, path, role, binding)
        for _, (path, role, binding) in sorted(selected.items())
    ]


def _content_closure(outputs: Iterable[Mapping[str, Any]]) -> str:
    content = [
        {"path": item["path"], "role": item["role"], "sha256": item["sha256"]}
        for item in outputs
        if item.get("binding") == "content_sha256"
    ]
    return content_sha256(content)


def _validate_required_roles(
    definition: StageDefinition, status: str, outputs: Sequence[Mapping[str, Any]]
) -> None:
    if status != "ok":
        return
    observed = {str(item.get("role") or "") for item in outputs}
    missing = sorted(definition.required_roles - observed)
    if missing:
        raise ValueError(
            f"{definition.name} completion lacks required output roles: {missing}"
        )


def _validate_stage_specific_outcome(
    course_dir: Path,
    definition: StageDefinition,
    status: str,
    outputs: Sequence[Mapping[str, Any]],
) -> None:
    if definition.name != "dvh" or status != "ok":
        return
    roles = {str(item.get("role") or "") for item in outputs}
    if "authoritative_dvh" in roles or "dvh_workbook" in roles:
        return
    qc_path = Path(course_dir) / "metadata" / "dvh_qc.json"
    contract_path = Path(course_dir) / "metadata" / "case_metadata.json"
    try:
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        metadata = json.loads(contract_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            "DVH completion without a table requires readable bound QC and course contract"
        ) from exc
    decision = ((metadata.get("course_contract") or {}).get("dvh") or {})
    if (
        qc.get("status") != "skipped"
        or decision.get("metrics_status") != "not_computed"
        or decision.get("output") is not None
    ):
        raise ValueError(
            "DVH completion without a table requires an authoritative not-computed decision"
        )


def stage_completion_payload(
    course_dir: Path,
    *,
    stage: str,
    status: str,
    configuration_dependency: Path,
) -> dict[str, Any]:
    root = Path(course_dir).resolve(strict=False)
    definition = stage_definition(stage)
    terminal_status = str(status).strip().lower()
    if terminal_status not in {"ok", "disabled"}:
        raise ValueError(f"Completion status must be 'ok' or 'disabled', got {status!r}")
    if root.parent.name == "" or root.name == "":
        raise ValueError(f"Course directory has no patient/course identity: {root}")

    configuration = _configuration_dependency(
        configuration_dependency, definition.configuration_stage
    )
    outputs = [] if terminal_status == "disabled" else _discover_outputs(root, definition)
    _validate_required_roles(definition, terminal_status, outputs)
    _validate_stage_specific_outcome(root, definition, terminal_status, outputs)
    code = _code_identity(definition)
    environment = environment_record(definition.environment_packages)
    return {
        "schema": STAGE_COMPLETION_SCHEMA,
        "status": terminal_status,
        "stage": definition.sentinel,
        "stage_name": definition.name,
        "patient_id": root.parent.name,
        "course_id": root.name,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "binding_policy": definition.binding_policy,
        "outputs": outputs,
        "output_count": len(outputs),
        "output_set_sha256": content_sha256(outputs),
        "content_closure_sha256": _content_closure(outputs),
        "configuration_dependency_sha256": configuration["sha256"],
        "configuration_dependency": configuration,
        "code_revisions": [code["revision"]],
        "code_identity_sha256": code["sources_sha256"],
        "code_identity": code,
        "execution_host": socket.gethostname(),
        "environment_fingerprint": content_sha256(environment),
        "environment": environment,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_stage_completion_sentinel(
    course_dir: Path,
    sentinel_path: Path,
    *,
    stage: str,
    status: str,
    configuration_dependency: Path,
) -> dict[str, Any]:
    root = Path(course_dir).resolve(strict=False)
    destination = Path(sentinel_path).resolve(strict=False)
    expected = root / stage_definition(stage).sentinel
    if destination != expected:
        raise ValueError(
            f"Completion sentinel path mismatch: expected {expected}, got {destination}"
        )
    payload = stage_completion_payload(
        root,
        stage=stage,
        status=status,
        configuration_dependency=configuration_dependency,
    )
    _atomic_json(destination, payload)
    return validate_stage_completion_sentinel(
        destination,
        expected_stage=stage,
        expected_patient=root.parent.name,
        expected_course=root.name,
    )


def _safe_output_path(course_dir: Path, value: object) -> Path:
    text = str(value or "")
    relative = Path(text)
    if not text or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe stage output path {text!r}")
    candidate = course_dir / relative
    try:
        candidate.parent.resolve(strict=False).relative_to(
            course_dir.resolve(strict=False)
        )
    except ValueError as exc:
        raise ValueError(f"Stage output escapes course directory: {text!r}") from exc
    return candidate


def validate_stage_completion_payload(
    course_dir: Path,
    payload: Mapping[str, Any],
    *,
    expected_stage: str,
    expected_patient: str | None = None,
    expected_course: str | None = None,
) -> dict[str, Any]:
    root = Path(course_dir).resolve(strict=False)
    definition = stage_definition(expected_stage)
    if payload.get("schema") != STAGE_COMPLETION_SCHEMA:
        raise ValueError("Unsupported stage completion schema")
    status = str(payload.get("status") or "").strip().lower()
    if status not in {"ok", "disabled"}:
        raise ValueError("Stage completion has no successful terminal status")
    if payload.get("stage") != definition.sentinel or payload.get("stage_name") != definition.name:
        raise ValueError("Stage completion identifies the wrong stage")
    patient = expected_patient if expected_patient is not None else root.parent.name
    course = expected_course if expected_course is not None else root.name
    if str(payload.get("patient_id") or "") != patient or str(payload.get("course_id") or "") != course:
        raise ValueError("Stage completion identifies the wrong patient/course")
    if payload.get("binding_policy") != definition.binding_policy:
        raise ValueError("Stage completion uses the wrong binding policy")

    configuration = payload.get("configuration_dependency")
    if not isinstance(configuration, Mapping):
        raise ValueError("Stage completion lacks its configuration dependency record")
    digest = str(payload.get("configuration_dependency_sha256") or "")
    if (
        configuration.get("schema") != "rtpipeline-stage-config-dependency-v1"
        or configuration.get("stage") != definition.configuration_stage
        or configuration.get("sha256") != digest
        or not _SHA256_RE.fullmatch(digest)
        or content_sha256(configuration.get("payload")) != digest
    ):
        raise ValueError("Stage completion configuration identity is invalid")

    code = payload.get("code_identity")
    revisions = payload.get("code_revisions")
    if not isinstance(code, Mapping) or not isinstance(revisions, list) or not revisions:
        raise ValueError("Stage completion lacks code identity")
    sources = code.get("sources")
    code_digest = str(payload.get("code_identity_sha256") or "")
    if (
        not isinstance(sources, list)
        or not sources
        or not _SHA256_RE.fullmatch(code_digest)
        or content_sha256(sources) != code_digest
        or code.get("sources_sha256") != code_digest
        or revisions != [code.get("revision")]
    ):
        raise ValueError("Stage completion code identity is invalid")
    for source in sources:
        if (
            not isinstance(source, Mapping)
            or not str(source.get("path") or "")
            or not _SHA256_RE.fullmatch(str(source.get("sha256") or ""))
        ):
            raise ValueError("Stage completion has a malformed code-source identity")

    environment = payload.get("environment")
    environment_digest = str(payload.get("environment_fingerprint") or "")
    if (
        not isinstance(environment, Mapping)
        or not _SHA256_RE.fullmatch(environment_digest)
        or content_sha256(environment) != environment_digest
        or not str(payload.get("execution_host") or "").strip()
    ):
        raise ValueError("Stage completion execution environment is invalid")

    outputs = payload.get("outputs")
    if not isinstance(outputs, list) or not all(isinstance(item, Mapping) for item in outputs):
        raise ValueError("Stage completion outputs must be a list of objects")
    if status == "disabled" and outputs:
        raise ValueError("Disabled stage completion must not claim output artifacts")
    if payload.get("output_count") != len(outputs):
        raise ValueError("Stage completion output count is stale")
    if outputs != sorted(outputs, key=lambda item: str(item.get("path") or "")):
        raise ValueError("Stage completion outputs are not canonically ordered")
    if len({str(item.get("path") or "") for item in outputs}) != len(outputs):
        raise ValueError("Stage completion contains duplicate output paths")

    observed: list[dict[str, Any]] = []
    for item in outputs:
        role = str(item.get("role") or "")
        binding = str(item.get("binding") or "")
        if not role or binding not in {"inventory", "content_sha256"}:
            raise ValueError("Stage completion contains a malformed output binding")
        path = _safe_output_path(root, item.get("path"))
        if not (path.is_file() or path.is_symlink()):
            raise ValueError(f"Bound stage output is absent: {item.get('path')}")
        observed.append(_artifact_entry(root, path, role, binding))
    if observed != outputs:
        raise ValueError("Bound stage output bytes or inventory changed")
    expected_outputs = _discover_outputs(root, definition)
    if expected_outputs != outputs:
        raise ValueError("Stage completion no longer declares the complete output set")
    if payload.get("output_set_sha256") != content_sha256(outputs):
        raise ValueError("Stage completion output-set digest is invalid")
    if payload.get("content_closure_sha256") != _content_closure(outputs):
        raise ValueError("Stage completion content-closure digest is invalid")
    _validate_required_roles(definition, status, outputs)
    _validate_stage_specific_outcome(root, definition, status, outputs)
    return dict(payload)


def validate_stage_completion_sentinel(
    sentinel_path: Path,
    *,
    expected_stage: str,
    expected_patient: str | None = None,
    expected_course: str | None = None,
) -> dict[str, Any]:
    path = Path(sentinel_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable structured stage completion sentinel: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Stage completion sentinel is not a JSON object")
    return validate_stage_completion_payload(
        path.parent,
        payload,
        expected_stage=expected_stage,
        expected_patient=expected_patient,
        expected_course=expected_course,
    )
