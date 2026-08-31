from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from base64 import urlsafe_b64encode
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import yaml


DEPENDENCY_SCHEMA = "rtpipeline-stage-config-dependency-v1"
RADIOMICS_DEPENDENCY_SCHEMA = "rtpipeline-radiomics-config-dependency-v1"


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonicalize(item) for item in value]
        return sorted(normalized, key=canonical_json)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def semantic_yaml(path: Path) -> Any:
    """Load YAML so comments and key ordering do not affect its identity."""

    candidate = Path(path)
    try:
        return yaml.safe_load(candidate.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Configured workflow dependency does not exist: {candidate}"
        ) from exc


def semantic_yaml_or_absent(path: Optional[Path]) -> dict[str, Any]:
    if path is None:
        return {"state": "not-configured"}
    candidate = Path(path)
    if not candidate.is_file():
        return {"state": "absent"}
    return {"state": "present", "content": semantic_yaml(candidate)}


def file_content_or_absent(path: Optional[Path]) -> dict[str, Any]:
    if path is None:
        return {"state": "not-configured"}
    candidate = Path(path)
    if not candidate.is_file():
        return {"state": "absent"}
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    return {"state": "present", "sha256": digest}


def stage_dependency_record(stage: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _canonicalize(payload)
    return {
        "schema": DEPENDENCY_SCHEMA,
        "stage": str(stage),
        "sha256": content_sha256(normalized),
        "payload": normalized,
    }


def _record_digest(record: Any) -> Optional[str]:
    if not isinstance(record, Mapping):
        return None
    digest = str(record.get("sha256") or "")
    payload = record.get("payload")
    if len(digest) != 64 or payload is None:
        return None
    if content_sha256(payload) != digest:
        return None
    return digest


def _initial_mtime_ns(source_paths: Iterable[Path]) -> int:
    observed = [
        Path(path).stat().st_mtime_ns
        for path in source_paths
        if Path(path).exists()
    ]
    # A newly introduced stable dependency should behave as if the source had
    # always been a rule input. It must not make every existing output stale merely
    # because the dependency mechanism was installed later.
    return max(observed, default=1)


def materialize_stage_dependency(
    cache_dir: Path,
    stage: str,
    payload: Mapping[str, Any],
    *,
    source_paths: Iterable[Path] = (),
) -> Path:
    """Write a stable DAG input only when its semantic content changes.

    The path is stable so initial adoption does not trigger Snakemake's changed-input
    inventory check for every later configuration. The file mtime changes only when
    the canonical payload hash changes. Rewriting a source YAML file with identical
    content therefore leaves downstream rules current.
    """

    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stage}.json"
    record = stage_dependency_record(stage, payload)

    existing_record: Any = None
    if path.is_file():
        try:
            existing_record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_record = None
    if _record_digest(existing_record) == record["sha256"]:
        return path

    initial_write = not path.exists()
    payload_text = json.dumps(record, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload_text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        if initial_write:
            mtime_ns = _initial_mtime_ns(Path(item) for item in source_paths)
            os.utime(path, ns=(mtime_ns, mtime_ns))
    finally:
        try:
            Path(temporary_name).unlink()
        except FileNotFoundError:
            pass
    return path


def read_stage_dependency(path: Path, *, expected_stage: Optional[str] = None) -> dict[str, Any]:
    candidate = Path(path)
    try:
        record = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable stage configuration dependency: {candidate}") from exc
    if record.get("schema") != DEPENDENCY_SCHEMA or _record_digest(record) is None:
        raise ValueError(f"Invalid stage configuration dependency: {candidate}")
    if expected_stage is not None and record.get("stage") != expected_stage:
        raise ValueError(
            f"Configuration dependency {candidate} is for {record.get('stage')!r}, "
            f"not {expected_stage!r}"
        )
    return record


def _snakemake_record_path(subject: Path, output_path: Path) -> Path:
    encoded = urlsafe_b64encode(str(output_path).encode()).decode()
    try:
        max_name = os.pathconf(subject, "PC_NAME_MAX")
    except (OSError, ValueError):
        max_name = 255
    if not max_name:
        max_name = 255
    chunks = [
        encoded[index : index + max_name - 1]
        for index in range(0, len(encoded), max_name - 1)
    ]
    chunks = ["@" + chunk for chunk in chunks[:-1]] + chunks[-1:]
    return subject.joinpath(*chunks)


def adopt_legacy_snakemake_inputs(
    workdir: Path,
    bindings: Mapping[Path, Path],
) -> int:
    """Add new inputs to legacy metadata without discarding other provenance.

    Snakemake otherwise treats the one-time addition of a configuration input as
    an input-set change even when that input predates a current output. Updating
    only the stored input inventory preserves code, parameter, software, and
    incomplete-job provenance. Missing or malformed records are left untouched so
    Snakemake conservatively schedules those outputs.
    """

    subject = Path(workdir) / ".snakemake" / "metadata"
    adopted = 0
    for output_path, dependency_path in bindings.items():
        record_path = _snakemake_record_path(subject, Path(output_path))
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stored_inputs = record.get("input") if isinstance(record, Mapping) else None
        if not isinstance(stored_inputs, list) or not all(
            isinstance(value, str) for value in stored_inputs
        ):
            continue
        dependency_text = str(Path(dependency_path))
        if dependency_text in stored_inputs:
            continue
        updated_record = dict(record)
        updated_record["input"] = sorted([*stored_inputs, dependency_text])
        record_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=record_path.parent,
            prefix=f".{record_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(updated_record, handle, sort_keys=True)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, record_path)
        adopted += 1
    return adopted


def advance_dependency_past_unbound_outputs(
    dependency_path: Path,
    output_paths: Iterable[Path],
    *,
    binding_field: str,
) -> int:
    """Make legacy outputs older than a dependency until provenance is bound."""

    dependency = read_stage_dependency(Path(dependency_path))
    expected = str(dependency["sha256"])
    legacy: list[Path] = []
    for output_path in output_paths:
        candidate = Path(output_path)
        try:
            observed = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            observed = None
        if not isinstance(observed, Mapping) or str(observed.get(binding_field) or "") != expected:
            legacy.append(candidate)
    if not legacy:
        return 0
    newest_output_ns = max(
        (path.stat().st_mtime_ns for path in legacy if path.exists()),
        default=0,
    )
    current_ns = Path(dependency_path).stat().st_mtime_ns
    target_ns = max(
        time.time_ns(),
        current_ns + 1_000_000_000,
        newest_output_ns + 1_000_000_000,
    )
    os.utime(dependency_path, ns=(target_ns, target_ns))
    return len(legacy)


def _parameter_key(
    arm: str,
    window: Optional[tuple[float, float]],
    *,
    large_roi: bool,
) -> str:
    window_text = (
        "none"
        if window is None
        else ",".join(format(float(value), ".17g") for value in window)
    )
    return f"{arm}|{window_text}|large_roi={int(large_roi)}"


def radiomics_parameter_manifest(
    *,
    ct_params: Path,
    mr_params: Path,
    roi_class_map: Path,
    pet_params: Optional[Mapping[str, Path]] = None,
) -> dict[str, Any]:
    """Build parameter provenance using the same configured hash stored per CT row."""

    from .radiomics_ct_contract import (
        PRIMARY_ARM,
        SENSITIVITY_ARM,
        configured_parameter_hash,
        roi_class_map_identity,
    )

    roi_map_data = semantic_yaml(roi_class_map)
    classes = roi_map_data.get("classes") if isinstance(roi_map_data, Mapping) else None
    windows: set[Optional[tuple[float, float]]] = {None}
    if isinstance(classes, Mapping):
        for entry in classes.values():
            if not isinstance(entry, Mapping):
                continue
            raw_window = entry.get("primary_resegment_range_hu")
            if isinstance(raw_window, (list, tuple)) and len(raw_window) == 2:
                windows.add((float(raw_window[0]), float(raw_window[1])))

    ct_hashes: dict[str, str] = {}
    sorted_windows = sorted(
        windows,
        key=lambda value: () if value is None else (value[0], value[1]),
    )
    for large_roi in (False, True):
        for window in sorted_windows:
            ct_hashes[
                _parameter_key(PRIMARY_ARM, window, large_roi=large_roi)
            ] = configured_parameter_hash(
                ct_params,
                arm=PRIMARY_ARM,
                window=window,
                large_roi=large_roi,
            )
        ct_hashes[
            _parameter_key(SENSITIVITY_ARM, None, large_roi=large_roi)
        ] = configured_parameter_hash(
            ct_params,
            arm=SENSITIVITY_ARM,
            window=None,
            large_roi=large_roi,
        )
    mr_hashes = {
        arm: configured_parameter_hash(
            mr_params,
            arm=arm,
            window=None,
            large_roi=False,
        )
        for arm in ("mr_configured", "mr_native_intensity", "mr_normalized")
    }
    map_version, map_hash = roi_class_map_identity(roi_class_map)

    pet_records = {
        str(name): semantic_yaml_or_absent(Path(path))
        for name, path in sorted((pet_params or {}).items())
    }
    return {
        "schema": RADIOMICS_DEPENDENCY_SCHEMA,
        "ct": {
            "configured_parameter_hashes": ct_hashes,
            "params": semantic_yaml(ct_params),
            "roi_class_map": {
                "version": map_version,
                "sha256": map_hash,
                "content": roi_map_data,
            },
        },
        "mr": {
            "configured_parameter_hashes": mr_hashes,
            "params": semantic_yaml(mr_params),
        },
        "pet": pet_records,
    }


def radiomics_row_parameter_key(
    arm: str,
    lower_hu: Any,
    upper_hu: Any,
    *,
    large_roi: bool,
) -> str:
    if str(arm) == "primary_resegmented" and lower_hu is not None and upper_hu is not None:
        try:
            if not (float(lower_hu) != float(lower_hu) or float(upper_hu) != float(upper_hu)):
                return _parameter_key(
                    str(arm),
                    (float(lower_hu), float(upper_hu)),
                    large_roi=large_roi,
                )
        except (TypeError, ValueError):
            pass
    return _parameter_key(str(arm), None, large_roi=large_roi)
