from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import socket
import subprocess
import tempfile
import uuid
import zlib
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata as importlib_metadata
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import yaml

from .radiomics_schema import (
    assert_radiomics_arrow_schema,
    is_radiomic_feature_column,
    expected_radiomics_string_columns,
    normalize_radiomics_dataframe,
    normalize_radiomics_result,
)


PRIMARY_ARM = "primary_resegmented"
SENSITIVITY_ARM = "sensitivity_raw"
CT_EXTRACTION_ARMS = (PRIMARY_ARM, SENSITIVITY_ARM)
SHAPE_FEATURE_MARKERS = ("_shape_", "_shape2D_")
INTENSITY_TEXTURE_FEATURE_MARKERS = (
    "_firstorder_",
    "_glcm_",
    "_glrlm_",
    "_glszm_",
    "_gldm_",
    "_ngtdm_",
)
RADIOMICS_UNDEFINED_FEATURES_COLUMN = "radiomics_undefined_features_json"
RADIOMICS_FEATURE_COMPLETENESS_COLUMN = "radiomics_feature_completeness"
RADIOMICS_FEATURE_COMPLETENESS_REASON_COLUMN = (
    "radiomics_feature_completeness_reason"
)
RADIOMICS_FEATURE_COMPLETENESS_SCHEMA_COLUMN = (
    "radiomics_feature_completeness_schema"
)
RADIOMICS_MISSING_FEATURES_COLUMN = "radiomics_missing_features_json"
RADIOMICS_MISSING_COUNT_COLUMN = "radiomics_missing_feature_count"
RADIOMICS_EXPECTED_SCHEMA_SHA256_COLUMN = (
    "radiomics_expected_feature_schema_sha256"
)
RADIOMICS_EXPECTED_COUNT_COLUMN = "radiomics_expected_feature_count"
RADIOMICS_EXPECTED_SCHEMA_SOURCE_COLUMN = (
    "radiomics_expected_feature_schema_source"
)
RADIOMICS_EXPECTED_SCHEMA_ZLIB_COLUMN = (
    "radiomics_expected_feature_schema_zlib_b64"
)
EXECUTION_HOST_COLUMN = "execution_host"
ENVIRONMENT_FINGERPRINT_COLUMN = "environment_fingerprint"
FEATURE_COMPLETENESS_SCHEMA = "rtpipeline-radiomics-feature-completeness-v1"
ALLOWED_UNDEFINED_FEATURE_SUFFIXES = ("_glcm_MCC",)
PUBLICATION_KEY_COLUMNS = (
    "patient_id",
    "course_id",
    "series_uid",
    "segmentation_source",
    "mask_identity",
    "roi_original_name",
    "stable_roi_identifier",
    "extraction_arm",
)
BASE_IDENTITY_COLUMNS = PUBLICATION_KEY_COLUMNS[:-1]
_COUNT_COLUMNS = (
    "morphologic_resampled_voxel_count",
    "resegment_after_count",
    "resegment_below_lower_count",
    "resegment_above_upper_count",
    "resegment_nonfinite_count",
)
_PRIMARY_WINDOWS: dict[str, tuple[float, float]] = {
    "target": (-1000.0, 400.0),
    "hollow_pelvic_organ": (-1000.0, 400.0),
    "solid_soft_tissue_neural": (-500.0, 400.0),
}
_NONAPPLICABLE_DISPOSITIONS = {
    "bone": "not_applicable_bone",
    "positioning_support": "not_primary_analysis_anatomy",
    "planning_helper": "not_applicable_planning_helper",
    "vessel": "not_applicable_pending_vessel_adjudication",
    "unresolved_mixed": "unclassified_roi",
}
_COMPLETE_DISPOSITIONS = {
    "success",
    "below_minimum_voxels",
    "below_minimum_dimensions",
    "not_applicable_bone",
    "not_primary_analysis_anatomy",
    "not_applicable_planning_helper",
    "not_applicable_pending_vessel_adjudication",
    "unclassified_roi",
    "declared_skip",
    "failed",
    "failed_shape_physical_validity",
}

FEATURE_POLICY_EXTRACT = "extract"
FEATURE_POLICY_INVENTORY_ONLY = "inventory_only_no_features"
_FEATURE_POLICIES = {FEATURE_POLICY_EXTRACT, FEATURE_POLICY_INVENTORY_ONLY}


@dataclass(frozen=True)
class RoiClassDecision:
    roi_class: str
    map_version: str
    map_hash: str
    map_entry_source: str
    adjudication_status: str
    primary_resegment_range_hu: Optional[tuple[float, float]]
    primary_intensity_texture_disposition: str
    feature_publication_policy: str


def load_roi_class_map(path_text: Optional[str] = None) -> tuple[dict[str, Any], str]:
    dependency_path = (
        ""
        if path_text
        else str(os.environ.get("RTPIPELINE_RADIOMICS_CONFIG_DEPENDENCY") or "")
    )
    return _load_roi_class_map(path_text, dependency_path)


@lru_cache(maxsize=8)
def _load_roi_class_map(
    path_text: Optional[str], dependency_path: str
) -> tuple[dict[str, Any], str]:
    expected_hash = ""
    if path_text:
        path = Path(path_text)
        raw = path.read_bytes()
        data = yaml.safe_load(raw)
    elif dependency_path:
        from .config_dependencies import read_stage_dependency

        record = read_stage_dependency(Path(dependency_path))
        if record.get("stage") != "radiomics":
            raise ValueError("radiomics dependency record has the wrong stage")
        payload = record.get("payload")
        provenance = (
            payload.get("parameter_provenance")
            if isinstance(payload, dict)
            else None
        )
        ct = provenance.get("ct") if isinstance(provenance, dict) else None
        identity = ct.get("roi_class_map") if isinstance(ct, dict) else None
        data = identity.get("content") if isinstance(identity, dict) else None
        expected_hash = (
            str(identity.get("sha256") or "") if isinstance(identity, dict) else ""
        )
        if not isinstance(data, dict) or not expected_hash:
            raise ValueError("radiomics dependency lacks a bound CT ROI class map")
    else:
        resource = importlib_resources.files("rtpipeline").joinpath("roi_class_map_v1.yaml")
        raw = resource.read_bytes()
        data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("CT ROI class map must be a mapping")
    if int(data.get("schema_version", 0)) != 1:
        raise ValueError("unsupported CT ROI class map schema")
    version = str(data.get("map_version") or "").strip()
    if not version:
        raise ValueError("CT ROI class map lacks map_version")
    total = data.get("totalsegmentator")
    canonical_names = total.get("canonical_names") if isinstance(total, dict) else None
    expected_canonical_hash = (
        str(total.get("canonical_names_hash") or "").strip()
        if isinstance(total, dict)
        else ""
    )
    if not isinstance(canonical_names, dict) or not expected_canonical_hash:
        raise ValueError("CT ROI class map lacks a hash-pinned TotalSegmentator vocabulary")
    canonical_names_text = json.dumps(
        canonical_names,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    observed_canonical_hash = hashlib.sha256(
        canonical_names_text.encode("utf-8")
    ).hexdigest()
    if observed_canonical_hash != expected_canonical_hash:
        raise ValueError("CT ROI class map TotalSegmentator vocabulary hash is stale")
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if dependency_path and digest != expected_hash:
        raise ValueError("radiomics dependency CT ROI class map hash is stale")
    return data, digest


def roi_class_map_identity(path: Optional[Path] = None) -> tuple[str, str]:
    data, digest = load_roi_class_map(str(Path(path).resolve()) if path else None)
    return str(data["map_version"]), digest


def _strip_partial(name: str) -> str:
    text = str(name)
    return text[:-9] if text.endswith("__partial") else text


def load_custom_structure_provenance(path: Optional[Path]) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    candidate = Path(path)
    raw = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    items = raw.get("custom_structures") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        raise ValueError(f"custom structure configuration lacks custom_structures: {candidate}")
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"custom structure entry must be a mapping: {candidate}")
        name = str(item.get("name") or "").strip()
        operation = str(item.get("operation") or "").strip()
        sources = item.get("source_structures")
        if not name or operation not in {"union", "intersection", "subtract", "xor"}:
            raise ValueError(f"custom structure provenance is incomplete for {name!r}: {candidate}")
        if not isinstance(sources, list) or not sources or not all(
            isinstance(source, str) and source.strip() for source in sources
        ):
            raise ValueError(f"custom structure base structures are incomplete for {name!r}: {candidate}")
        out[name] = {
            "operation": operation,
            "source_structures": [str(source).strip() for source in sources],
            "margin": item.get("margin"),
        }
    return out


def _decision(
    roi_class: str,
    *,
    data: Mapping[str, Any],
    digest: str,
    source: str,
    status: str,
) -> RoiClassDecision:
    window = _PRIMARY_WINDOWS.get(roi_class)
    disposition = "success" if window is not None else _NONAPPLICABLE_DISPOSITIONS.get(
        roi_class, "unclassified_roi"
    )
    class_spec = data.get("classes", {}).get(roi_class, {})
    feature_policy = str(
        class_spec.get("radiomic_feature_policy") or FEATURE_POLICY_EXTRACT
    )
    if feature_policy not in _FEATURE_POLICIES:
        raise ValueError(
            f"unsupported radiomic feature policy {feature_policy!r} for ROI class {roi_class!r}"
        )
    return RoiClassDecision(
        roi_class=roi_class,
        map_version=str(data["map_version"]),
        map_hash=digest,
        map_entry_source=source,
        adjudication_status=status,
        primary_resegment_range_hu=window,
        primary_intensity_texture_disposition=disposition,
        feature_publication_policy=feature_policy,
    )


def classify_ct_roi(
    segmentation_source: str,
    roi_original_name: str,
    *,
    custom_provenance: Optional[Mapping[str, Mapping[str, Any]]] = None,
    map_path: Optional[Path] = None,
) -> RoiClassDecision:
    data, digest = load_roi_class_map(str(Path(map_path).resolve()) if map_path else None)
    source = str(segmentation_source).strip()
    name = str(roi_original_name)
    base_name = _strip_partial(name)
    source_folded = source.casefold()

    if source_folded.startswith(("autorts_total", "autots_total")):
        entries = data["totalsegmentator"]["canonical_names"]
        entry = entries.get(base_name)
        if isinstance(entry, dict):
            return _decision(
                str(entry["roi_class"]),
                data=data,
                digest=digest,
                source=(
                    f"TotalSegmentator:{data['totalsegmentator']['model']}:"
                    f"{data['totalsegmentator']['version']}"
                ),
                status="versioned_canonical_name",
            )
        return _decision(
            "unresolved_mixed",
            data=data,
            digest=digest,
            source="TotalSegmentator:unrecognized_name",
            status="operator_adjudication_required",
        )

    if source_folded == "custom":
        derived_entries = data.get("derived_crosswalk", {})
        derived_entry = derived_entries.get(base_name)
        provenance_map = custom_provenance or {}
        provenance = provenance_map.get(base_name)
        if isinstance(derived_entry, dict) and bool(derived_entry.get("require_provenance")):
            if not isinstance(provenance, Mapping):
                return _decision(
                    "unresolved_mixed",
                    data=data,
                    digest=digest,
                    source="derived_crosswalk:missing_provenance",
                    status="operator_adjudication_required",
                )
            operation = str(provenance.get("operation") or "")
            bases = provenance.get("source_structures")
            if operation not in {"union", "intersection", "subtract", "xor"} or not isinstance(
                bases, Sequence
            ) or isinstance(bases, (str, bytes)) or not bases:
                return _decision(
                    "unresolved_mixed",
                    data=data,
                    digest=digest,
                    source="derived_crosswalk:invalid_provenance",
                    status="operator_adjudication_required",
                )

            def _recorded_base_class(base: str, seen: set[str]) -> Optional[str]:
                base = _strip_partial(str(base))
                if base in seen:
                    return None
                canonical = data["totalsegmentator"]["canonical_names"].get(base)
                if isinstance(canonical, Mapping):
                    return str(canonical.get("roi_class") or "") or None
                manual = data.get("manual_custom_crosswalk", {}).get(base)
                if isinstance(manual, Mapping):
                    return str(manual.get("roi_class") or "") or None
                nested = derived_entries.get(base)
                nested_provenance = provenance_map.get(base)
                if not isinstance(nested, Mapping) or not isinstance(nested_provenance, Mapping):
                    return None
                nested_operation = str(nested_provenance.get("operation") or "")
                nested_bases = nested_provenance.get("source_structures")
                if nested_operation not in {"union", "intersection", "subtract", "xor"} or not isinstance(
                    nested_bases, Sequence
                ) or isinstance(nested_bases, (str, bytes)) or not nested_bases:
                    return None
                nested_classes = [
                    _recorded_base_class(str(item), seen | {base})
                    for item in nested_bases
                ]
                if any(value is None for value in nested_classes):
                    return None
                inherited = nested_classes[0]
                if nested_operation != "subtract" and any(
                    value != inherited for value in nested_classes[1:]
                ):
                    return None
                if inherited != str(nested.get("roi_class") or ""):
                    return None
                return inherited

            base_classes = [
                _recorded_base_class(str(base), {base_name}) for base in bases
            ]
            if any(value is None for value in base_classes):
                return _decision(
                    "unresolved_mixed",
                    data=data,
                    digest=digest,
                    source="derived_crosswalk:unclassifiable_recorded_base",
                    status="operator_adjudication_required",
                )
            inherited_class = base_classes[0]
            if operation != "subtract" and any(
                value != inherited_class for value in base_classes[1:]
            ):
                return _decision(
                    "unresolved_mixed",
                    data=data,
                    digest=digest,
                    source="derived_crosswalk:mixed_recorded_base_classes",
                    status="operator_adjudication_required",
                )
            if inherited_class != str(derived_entry.get("roi_class") or ""):
                return _decision(
                    "unresolved_mixed",
                    data=data,
                    digest=digest,
                    source="derived_crosswalk:declared_class_disagrees_with_bases",
                    status="operator_adjudication_required",
                )
            return _decision(
                "planning_helper",
                data=data,
                digest=digest,
                source="derived_crosswalk:recorded_boolean_operation",
                status="approved_non_anatomic_derived_structure",
            )

    entry = data.get("manual_custom_crosswalk", {}).get(base_name)
    if isinstance(entry, dict):
        return _decision(
            str(entry["roi_class"]),
            data=data,
            digest=digest,
            source="manual_custom_crosswalk:exact_name",
            status=str(entry.get("adjudication_status") or "approved"),
        )
    return _decision(
        "unresolved_mixed",
        data=data,
        digest=digest,
        source="manual_custom_crosswalk:unlisted_exact_name",
        status="operator_adjudication_required",
    )


def rtstruct_roi_identities(path: Path) -> dict[str, tuple[str, str]]:
    import pydicom

    dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    sop_uid = str(getattr(dataset, "SOPInstanceUID", "") or "").strip()
    if not sop_uid:
        raise ValueError(f"RTSTRUCT has no SOPInstanceUID: {path}")
    identities: dict[str, tuple[str, str]] = {}
    for item in getattr(dataset, "StructureSetROISequence", []) or []:
        name = str(getattr(item, "ROIName", "") or "").strip()
        roi_number = str(getattr(item, "ROINumber", "") or "").strip()
        if not name or not roi_number:
            raise ValueError(f"RTSTRUCT has a blank ROI name or number: {path}")
        if name in identities:
            raise ValueError(f"RTSTRUCT has duplicate ROI name {name!r}: {path}")
        identities[name] = (sop_uid, roi_number)
    if not identities:
        raise ValueError(f"RTSTRUCT contains no stable ROI identities: {path}")
    return identities


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _safe_package_version(distribution_name: str) -> str:
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return "unavailable"


def execution_environment_fingerprint() -> str:
    """Hash the runtime identity that can affect radiomics values."""
    payload = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "libc": platform.libc_ver(),
        "numpy": _safe_package_version("numpy"),
        "scipy": _safe_package_version("scipy"),
        "pywavelets": _safe_package_version("PyWavelets"),
        "simpleitk": _safe_package_version("SimpleITK"),
        "pyradiomics": _safe_package_version("pyradiomics"),
        "pyarrow": _safe_package_version("pyarrow"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _course_input_path(course_dir: Path, value: Any, field: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"radiomics input closure field {field} is empty")
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = course_dir / candidate
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(course_dir.resolve(strict=False))
    except ValueError as exc:
        raise ValueError(
            f"radiomics input closure field {field} escapes the course directory"
        ) from exc
    return resolved


def _input_artifact_entries(role: str, path: Path) -> list[str]:
    """Return path-independent content records for one governed input artifact."""
    if not path.exists():
        return [f"M\t{role}"]
    if path.is_file():
        return [f"F\t{role}\t{file_sha256(path)}"]
    if not path.is_dir():
        return [f"X\t{role}"]
    entries: list[str] = []
    traversal_root = path.resolve(strict=False)
    for directory, dirnames, filenames in os.walk(
        traversal_root, followlinks=False
    ):
        dirnames.sort()
        filenames.sort()
        directory_path = Path(directory)
        for filename in filenames:
            candidate = directory_path / filename
            try:
                relative = candidate.relative_to(traversal_root).as_posix()
            except ValueError as exc:
                raise ValueError(
                    f"radiomics input closure traversal escaped {role}"
                ) from exc
            if candidate.is_file():
                entries.append(
                    f"F\t{role}/{relative}\t{file_sha256(candidate)}"
                )
    if not entries:
        entries.append(f"D\t{role}")
    return entries


def input_closure_sha256(
    course_dir: Path,
    dataframe: Any = None,
) -> str:
    """Hash the exact DICOM and RTSTRUCT inputs used by CT radiomics.

    Logical roles, rather than host-specific absolute paths, make the closure
    comparable across hosts. Downstream course products are deliberately absent.
    """
    root = Path(course_dir)
    artifacts: list[tuple[str, Path]] = []
    contract_entry: Optional[str] = None
    contract_path = root / "metadata" / "case_metadata.json"
    if contract_path.is_file():
        try:
            metadata = json.loads(contract_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                "radiomics input closure course contract is unreadable"
            ) from exc
        if not isinstance(metadata, Mapping):
            raise ValueError(
                "radiomics input closure course metadata is malformed"
            )
        contract = metadata.get("course_contract")
        if not isinstance(contract, Mapping):
            raise ValueError(
                "radiomics input closure course contract is malformed"
            )
        planning_ct = contract.get("planning_ct") or {}
        if not isinstance(planning_ct, Mapping):
            raise ValueError(
                "radiomics input closure planning_ct contract is malformed"
            )
        dicom_dir = planning_ct.get("dicom_dir")
        if dicom_dir:
            artifacts.append(
                (
                    "planning_ct_dicom",
                    _course_input_path(root, dicom_dir, "planning_ct.dicom_dir"),
                )
            )
        authoritative = contract.get("authoritative_rtstruct")
        if authoritative is not None:
            if not isinstance(authoritative, Mapping):
                raise ValueError(
                    "radiomics input closure authoritative_rtstruct is malformed"
                )
            rtstruct_path = authoritative.get("path")
            if rtstruct_path:
                artifacts.append(
                    (
                        "authoritative_rtstruct",
                        _course_input_path(
                            root,
                            rtstruct_path,
                            "authoritative_rtstruct.path",
                        ),
                    )
                )
        decision = {
            "version": contract.get("version"),
            "authority": contract.get("authority"),
            "course_id": contract.get("course_id"),
            "course_key": contract.get("course_key"),
            "planning_ct": {
                key: planning_ct.get(key)
                for key in (
                    "status",
                    "series_instance_uid",
                    "referenced_series_uids",
                )
            },
            "authoritative_rtstruct": (
                {"sop_instance_uid": authoritative.get("sop_instance_uid")}
                if isinstance(authoritative, Mapping)
                else None
            ),
        }
        contract_digest = hashlib.sha256(
            json.dumps(
                decision, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        contract_entry = (
            f"J\tcourse_contract_radiomics_decision\t{contract_digest}"
        )

    sources: set[str] = set()
    if dataframe is not None and "segmentation_source" in dataframe.columns:
        sources = {
            str(value).strip()
            for value in dataframe["segmentation_source"].tolist()
            if not _is_missing(value) and str(value).strip()
        }
    if "Manual" in sources and not any(
        role == "authoritative_rtstruct" for role, _ in artifacts
    ):
        artifacts.append(("authoritative_rtstruct", root / "RS_orig.dcm"))
    if "AutoRTS_total" in sources:
        artifacts.append(("autorts_rtstruct", root / "RS_auto.dcm"))
    if "Custom" in sources:
        artifacts.append(("custom_rtstruct", root / "RS_custom.dcm"))
    for source in sorted(sources):
        if not source.startswith("CustomModel:"):
            continue
        model_name = source.partition(":")[2].strip()
        if not model_name or Path(model_name).name != model_name:
            raise ValueError(
                f"invalid custom-model segmentation source in radiomics input closure: {source}"
            )
        artifacts.append(
            (
                f"custom_model_rtstruct/{model_name}",
                root / "Segmentation_CustomModels" / model_name / "rtstruct.dcm",
            )
        )

    if dataframe is not None and "segmentation_source" in dataframe.columns:
        for _, row in dataframe.iterrows():
            if str(row.get("segmentation_source") or "") != (
                "AutoTS_total_nifti_fallback"
            ):
                continue
            identity = json.dumps(
                {
                    "series_uid": str(row.get("series_uid") or ""),
                    "stable_roi_identifier": str(
                        row.get("stable_roi_identifier")
                        or row.get("roi_original_name")
                        or ""
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            logical_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            mask_source = row.get("mask_path_source")
            if not _is_missing(mask_source) and str(mask_source).strip():
                artifacts.append(
                    (
                        f"nifti_fallback_mask/{logical_id}",
                        _course_input_path(
                            root,
                            mask_source,
                            "mask_path_source",
                        ),
                    )
                )
            nifti_path = row.get("nifti_path")
            if not _is_missing(nifti_path) and str(nifti_path).strip():
                artifacts.append(
                    (
                        f"nifti_fallback_image/{str(row.get('series_uid') or '')}",
                        _course_input_path(root, nifti_path, "nifti_path"),
                    )
                )

    entries: list[str] = [contract_entry] if contract_entry is not None else []
    seen_roles: set[str] = set()
    for role, path in artifacts:
        if role in seen_roles:
            continue
        seen_roles.add(role)
        entries.extend(_input_artifact_entries(role, path))
    encoded = "\n".join(sorted(entries)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_feature_name_list(value: Any) -> set[str]:
    if _is_missing(value):
        return set()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{RADIOMICS_UNDEFINED_FEATURES_COLUMN} is not valid JSON"
            ) from exc
    if not isinstance(value, (list, tuple, set)):
        raise ValueError(f"{RADIOMICS_UNDEFINED_FEATURES_COLUMN} must be a JSON list")
    return {str(item) for item in value}


def _nonfinite_feature_value(value: Any) -> bool:
    if _is_missing(value):
        return True
    try:
        return not bool(np.isfinite(float(value)))
    except (TypeError, ValueError, OverflowError):
        return True


def feature_completeness_records(dataframe: Any) -> list[dict[str, Any]]:
    """Describe the analysis completeness of each published ROI-arm row.

    Expected features are the course publication schema, restricted to shape for
    primary rows whose intensity and texture family has a declared non-applicable
    disposition. Extractor-returned nonfinite values are valid only when named in
    the undefined-feature record.
    """
    feature_columns = sorted(
        str(column)
        for column in dataframe.columns
        if is_radiomic_feature_column(str(column))
    )
    shape_columns = [
        name for name in feature_columns if any(marker in name for marker in SHAPE_FEATURE_MARKERS)
    ]
    records: list[dict[str, Any]] = []
    for _, row in dataframe.iterrows():
        values = row.to_dict()
        undefined = _parse_feature_name_list(
            values.get(RADIOMICS_UNDEFINED_FEATURES_COLUMN)
        )
        unknown_undefined = undefined.difference(feature_columns)
        if unknown_undefined:
            raise ValueError(
                "undefined feature record names unknown features: "
                + ", ".join(sorted(unknown_undefined))
            )
        def _status_value(column: str, default: str) -> str:
            value = values.get(column)
            if _is_missing(value):
                return default
            text = str(value).strip().casefold()
            return text or default

        unsupported_undefined = {
            name
            for name in undefined
            if not name.endswith(ALLOWED_UNDEFINED_FEATURE_SUFFIXES)
        }
        if unsupported_undefined:
            raise ValueError(
                "undefined feature record names features without an approved "
                "undefined-value contract: "
                + ", ".join(sorted(unsupported_undefined))
            )
        extraction_status = _status_value("extraction_status", "success")
        shape_disposition = _status_value("shape_disposition", "success")
        intensity_disposition = _status_value(
            "intensity_texture_disposition", "success"
        )

        if extraction_status != "success":
            missing: list[str] = []
            if extraction_status in {
                "below_minimum_voxels",
                "below_minimum_dimensions",
            }:
                completeness = "excluded_geometry"
                reason = f"extraction_status:{extraction_status}"
            elif extraction_status == "declared_skip":
                completeness = "excluded_declared"
                reason = "extraction_status:declared_skip"
            else:
                completeness = "failed_extraction"
                reason = f"extraction_status:{extraction_status or 'missing'}"
            if undefined:
                raise ValueError(
                    "excluded radiomics row must not declare undefined features"
                )
        else:
            expected = feature_columns
            if intensity_disposition and intensity_disposition != "success":
                expected = shape_columns
            schema_fields = {
                RADIOMICS_EXPECTED_SCHEMA_SHA256_COLUMN,
                RADIOMICS_EXPECTED_COUNT_COLUMN,
                RADIOMICS_EXPECTED_SCHEMA_SOURCE_COLUMN,
                RADIOMICS_EXPECTED_SCHEMA_ZLIB_COLUMN,
            }
            schema_present = {
                field
                for field in schema_fields
                if not _is_missing(values.get(field))
                and str(values.get(field)).strip()
            }
            if schema_present and schema_present != schema_fields:
                raise ValueError(
                    "incomplete configured radiomics feature schema record"
                )
            if schema_present:
                expected_count = int(values[RADIOMICS_EXPECTED_COUNT_COLUMN])
                expected_digest = str(
                    values[RADIOMICS_EXPECTED_SCHEMA_SHA256_COLUMN]
                ).strip()
                schema_source = str(
                    values[RADIOMICS_EXPECTED_SCHEMA_SOURCE_COLUMN]
                ).strip()
                if schema_source != "configured-extractor-v1":
                    raise ValueError(
                        "radiomics expected feature schema lacks configured-extractor provenance"
                    )
                configured_names = _decode_feature_schema(
                    values[RADIOMICS_EXPECTED_SCHEMA_ZLIB_COLUMN]
                )
                if expected_count != len(configured_names):
                    raise ValueError(
                        "radiomics configured feature count does not match schema payload"
                    )
                if expected_digest != _feature_schema_sha256(configured_names):
                    raise ValueError(
                        "radiomics configured feature schema digest is stale"
                    )
                expected = configured_names
            missing = [
                name
                for name in expected
                if _nonfinite_feature_value(values.get(name))
            ]
            stale_undefined = undefined.difference(missing)
            if stale_undefined:
                raise ValueError(
                    "undefined feature record names finite or non-applicable features: "
                    + ", ".join(sorted(stale_undefined))
                )
            unexplained = sorted(set(missing).difference(undefined))
            if shape_disposition != "success":
                completeness = "incomplete"
                reason = "successful_row_lacks_successful_shape_disposition"
            elif unexplained:
                completeness = "incomplete"
                reason = "unexplained_nonfinite_expected_features"
            elif undefined:
                completeness = "complete_with_undefined"
                reason = "extractor_declared_undefined_features"
            elif intensity_disposition and intensity_disposition != "success":
                completeness = "complete_with_not_applicable"
                reason = f"intensity_texture_disposition:{intensity_disposition}"
            else:
                completeness = "complete"
                reason = "all_expected_features_finite"
        records.append(
            {
                "publication_key": publication_key(values),
                "status": completeness,
                "reason": reason,
                "missing_features": sorted(missing),
                "undefined_features": sorted(undefined),
                "missing_count": len(missing),
                "required": str(values.get("roi_required") or "")
                .strip()
                .casefold()
                in {"1", "true", "yes", "on"},
            }
        )
    return records


def validate_feature_completeness_records(
    dataframe: Any,
    *,
    require_no_unexplained: bool = False,
) -> list[dict[str, Any]]:
    """Validate persisted per-ROI completeness metadata when present."""
    metadata_columns = {
        RADIOMICS_FEATURE_COMPLETENESS_COLUMN,
        RADIOMICS_FEATURE_COMPLETENESS_REASON_COLUMN,
        RADIOMICS_FEATURE_COMPLETENESS_SCHEMA_COLUMN,
        RADIOMICS_MISSING_FEATURES_COLUMN,
        RADIOMICS_MISSING_COUNT_COLUMN,
        RADIOMICS_UNDEFINED_FEATURES_COLUMN,
    }
    present = metadata_columns.intersection(set(dataframe.columns))
    records = feature_completeness_records(dataframe)
    if present:
        # The extractor may provide only the undefined-feature list before the
        # publication writer materializes the complete persisted record.
        producer_only = {RADIOMICS_UNDEFINED_FEATURES_COLUMN}
        if present != metadata_columns and present != producer_only:
            missing = sorted(metadata_columns.difference(present))
            raise ValueError(
                "incomplete radiomics feature completeness record, missing: "
                + ", ".join(missing)
            )
    if present == metadata_columns:
        for position, (_, row) in enumerate(dataframe.iterrows()):
            expected = records[position]
            observed_missing = _parse_feature_name_list(
                row[RADIOMICS_MISSING_FEATURES_COLUMN]
            )
            observed_undefined = _parse_feature_name_list(
                row[RADIOMICS_UNDEFINED_FEATURES_COLUMN]
            )
            if str(row[RADIOMICS_FEATURE_COMPLETENESS_SCHEMA_COLUMN]) != FEATURE_COMPLETENESS_SCHEMA:
                raise ValueError("unsupported radiomics feature completeness schema")
            if str(row[RADIOMICS_FEATURE_COMPLETENESS_COLUMN]) != expected["status"]:
                raise ValueError("radiomics feature completeness status is stale")
            if str(row[RADIOMICS_FEATURE_COMPLETENESS_REASON_COLUMN]) != expected["reason"]:
                raise ValueError("radiomics feature completeness reason is stale")
            try:
                observed_count = int(row[RADIOMICS_MISSING_COUNT_COLUMN])
            except (TypeError, ValueError):
                raise ValueError("radiomics missing feature count is invalid") from None
            if observed_count != expected["missing_count"]:
                raise ValueError("radiomics missing feature count is stale")
            if observed_missing != set(expected["missing_features"]):
                raise ValueError("radiomics missing feature record is stale")
            if observed_undefined != set(expected["undefined_features"]):
                raise ValueError("radiomics undefined feature record is stale")
    if require_no_unexplained:
        incomplete = [record for record in records if record["status"] == "incomplete"]
        if incomplete:
            examples = "; ".join(
                f"{record['publication_key']} missing={record['missing_features']}"
                for record in incomplete[:3]
            )
            raise ValueError(
                f"{len(incomplete)} ROI-arm rows have unexplained incomplete feature vectors: {examples}"
            )
    return records


ANALYSIS_ELIGIBLE_COMPLETENESS = frozenset(
    {"complete", "complete_with_undefined", "complete_with_not_applicable"}
)


def analysis_eligible_feature_rows(dataframe: Any) -> Any:
    """Return complete ROI pairs and exclude both arms if either arm is ineligible."""
    records = validate_feature_completeness_records(dataframe)
    if RADIOMICS_FEATURE_COMPLETENESS_COLUMN not in dataframe.columns:
        raise ValueError(
            "radiomics publication lacks per-ROI feature completeness records"
        )
    base_eligibility: dict[tuple[str, ...], bool] = {}
    for record in records:
        base = tuple(record["publication_key"][:-1])
        eligible = record["status"] in ANALYSIS_ELIGIBLE_COMPLETENESS
        base_eligibility[base] = base_eligibility.get(base, True) and eligible
    keep = [
        base_eligibility[tuple(record["publication_key"][:-1])]
        for record in records
    ]
    return dataframe.loc[keep].copy()


def _add_feature_completeness_metadata(dataframe: Any) -> Any:
    output = dataframe.copy()
    for column, default in (
        (EXECUTION_HOST_COLUMN, socket.gethostname()),
        (ENVIRONMENT_FINGERPRINT_COLUMN, execution_environment_fingerprint()),
        (RADIOMICS_UNDEFINED_FEATURES_COLUMN, "[]"),
    ):
        if column not in output.columns:
            output[column] = default
        else:
            output[column] = output[column].map(
                lambda value: default if _is_missing(value) else value
            )
    records = feature_completeness_records(output)
    output[RADIOMICS_FEATURE_COMPLETENESS_SCHEMA_COLUMN] = FEATURE_COMPLETENESS_SCHEMA
    output[RADIOMICS_FEATURE_COMPLETENESS_COLUMN] = [record["status"] for record in records]
    output[RADIOMICS_FEATURE_COMPLETENESS_REASON_COLUMN] = [record["reason"] for record in records]
    output[RADIOMICS_MISSING_FEATURES_COLUMN] = [
        _feature_names_json(record["missing_features"]) for record in records
    ]
    output[RADIOMICS_MISSING_COUNT_COLUMN] = [record["missing_count"] for record in records]
    return output


def _canonical_feature_value(value: Any) -> Any:
    if _is_missing(value):
        return "NaN"
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return str(value)
    if np.isnan(number):
        return "NaN"
    if np.isposinf(number):
        return "+Inf"
    if np.isneginf(number):
        return "-Inf"
    return number.hex()


def canonical_feature_table_sha256(dataframe: Any) -> str:
    feature_columns = sorted(
        str(column) for column in dataframe.columns if is_radiomic_feature_column(str(column))
    )
    rows = []
    for _, row in dataframe.iterrows():
        values = row.to_dict()
        rows.append(
            {
                "key": list(publication_key(values)),
                "features": {
                    name: _canonical_feature_value(values.get(name))
                    for name in feature_columns
                },
            }
        )
    rows.sort(key=lambda item: json.dumps(item["key"], separators=(",", ":")))
    encoded = json.dumps(
        {"columns": feature_columns, "rows": rows},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def missingness_digest(dataframe: Any) -> str:
    feature_columns = sorted(
        str(column) for column in dataframe.columns if is_radiomic_feature_column(str(column))
    )
    rows = []
    for _, row in dataframe.iterrows():
        values = row.to_dict()
        rows.append(
            {
                "key": list(publication_key(values)),
                "missing": [
                    name for name in feature_columns
                    if _nonfinite_feature_value(values.get(name))
                ],
            }
        )
    rows.sort(key=lambda item: json.dumps(item["key"], separators=(",", ":")))
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_rtstruct_roi_identity(path: Path, roi_name: str) -> tuple[str, str]:
    """Return DICOM identities, or content/name identities for legacy inputs."""
    try:
        identities = rtstruct_roi_identities(path)
        return identities[str(roi_name)]
    except (KeyError, OSError, ValueError):
        content_identity = f"sha256:{file_sha256(path)}"
        name_digest = hashlib.sha256(str(roi_name).encode("utf-8")).hexdigest()
        return content_identity, f"roi-name-sha256:{name_digest}"


def configured_parameter_hash(
    params_file: Optional[Path],
    *,
    arm: str,
    window: Optional[tuple[float, float]],
    large_roi: bool,
) -> str:
    raw: Any
    if params_file is None:
        raw = {"source": "pyradiomics-defaults"}
    else:
        raw = yaml.safe_load(Path(params_file).read_text(encoding="utf-8"))
    payload = {
        "configured_parameters": raw,
        "arm": arm,
        "window": list(window) if window is not None else None,
        # Runtime large-ROI overrides are represented by effective_parameter_hash.
        # The configured hash identifies the immutable source configuration so it
        # can be recomputed before mask materialization and compared on resume.
        "shape_contract": "separate_unresegmented_morphological_mask_v1",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def current_code_revision() -> str:
    configured = os.environ.get("RTPIPELINE_CODE_REVISION", "").strip()
    if configured:
        return configured
    try:
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        revision = result.stdout.strip()
        return revision or "unknown"
    except Exception:
        return "unknown"


def new_run_identifier() -> str:
    return str(uuid.uuid4())


def effective_parameter_hash(extractor: Any, *, arm: str, window: Optional[tuple[float, float]]) -> str:
    payload = {
        "arm": arm,
        "window": list(window) if window is not None else None,
        "settings": dict(getattr(extractor, "settings", {}) or {}),
        "image_types": dict(getattr(extractor, "enabledImagetypes", {}) or {}),
        "features": dict(getattr(extractor, "enabledFeatures", {}) or {}),
        "shape_contract": "separate_unresegmented_morphological_mask_v1",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _disable_shape(extractor: Any) -> Any:
    enabled = dict(getattr(extractor, "enabledFeatures", {}) or {})
    enabled.pop("shape", None)
    enabled.pop("shape2D", None)
    extractor.enabledFeatures = enabled
    extractor.settings.pop("resegmentRange", None)
    extractor.settings["resegmentShape"] = False
    return extractor


def _configure_shape_only(extractor: Any) -> Any:
    extractor.settings.pop("resegmentRange", None)
    extractor.settings["resegmentShape"] = False
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("shape")
    return extractor


def build_ct_extractors(
    factory: Callable[[], Any],
    window: Optional[tuple[float, float]],
) -> tuple[Any, Any, Optional[Any]]:
    shape_extractor = _configure_shape_only(factory())
    raw_extractor = _disable_shape(factory())
    primary_extractor = None
    if window is not None:
        primary_extractor = _disable_shape(factory())
        primary_extractor.settings["resegmentRange"] = [float(window[0]), float(window[1])]
        primary_extractor.settings["resegmentMode"] = "absolute"
        primary_extractor.settings["resegmentShape"] = False
    return shape_extractor, raw_extractor, primary_extractor


def effective_parameter_hashes_for_arms(
    factory: Callable[[], Any],
    decision: RoiClassDecision,
) -> dict[str, str]:
    """Hash the materialized extractor settings selected for each CT arm."""
    _, raw_extractor, primary_extractor = build_ct_extractors(
        factory, decision.primary_resegment_range_hu
    )
    return {
        PRIMARY_ARM: effective_parameter_hash(
            primary_extractor or raw_extractor,
            arm=PRIMARY_ARM,
            window=decision.primary_resegment_range_hu,
        ),
        SENSITIVITY_ARM: effective_parameter_hash(
            raw_extractor,
            arm=SENSITIVITY_ARM,
            window=None,
        ),
    }


def _scalarize(result: Mapping[str, Any]) -> dict[str, Any]:
    return normalize_radiomics_result(result)


def _feature_subset(result: Mapping[str, Any], markers: tuple[str, ...]) -> dict[str, Any]:
    return {str(key): value for key, value in result.items() if any(marker in str(key) for marker in markers)}


def _approved_undefined_feature_names(result: Mapping[str, Any]) -> set[str]:
    """Return nonfinite values covered by an approved feature-level contract."""
    names: set[str] = set()
    for key, value in result.items():
        name = str(key)
        if not is_radiomic_feature_column(name) or not name.endswith(
            ALLOWED_UNDEFINED_FEATURE_SUFFIXES
        ):
            continue
        if _is_missing(value):
            names.add(name)
            continue
        try:
            if not np.isfinite(float(value)):
                names.add(name)
        except (TypeError, ValueError, OverflowError):
            names.add(name)
    return names


def _feature_names_json(names: Iterable[str]) -> str:
    return json.dumps(sorted({str(name) for name in names}), separators=(",", ":"))


def _feature_schema_sha256(names: Iterable[str]) -> str:
    encoded = "\n".join(sorted({str(name) for name in names})).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _configured_feature_names(
    extractor: Any,
    *,
    observed_result: Optional[Mapping[str, Any]] = None,
) -> set[str]:
    """Resolve expected PyRadiomics names from configured classes and image types."""
    try:
        import radiomics

        get_feature_classes = getattr(radiomics, "getFeatureClasses")
    except (ImportError, AttributeError):
        if observed_result is None:
            raise
        return {
            str(name)
            for name in observed_result
            if is_radiomic_feature_column(str(name))
        }

    feature_classes = get_feature_classes()
    enabled_features = dict(getattr(extractor, "enabledFeatures", {}) or {})
    enabled_images = dict(getattr(extractor, "enabledImagetypes", {}) or {})
    names_by_class: dict[str, list[str]] = {}
    for class_name, configured in enabled_features.items():
        feature_class = feature_classes.get(str(class_name))
        if feature_class is None:
            raise ValueError(
                f"unsupported configured PyRadiomics feature class: {class_name}"
            )
        available = feature_class.getFeatureNames()
        selected = (
            [str(name) for name in configured]
            if configured
            else [
                str(name)
                for name, deprecated in available.items()
                if not deprecated
            ]
        )
        names_by_class[str(class_name)] = selected

    prefixes: list[str] = []
    for image_name, settings in enabled_images.items():
        name = str(image_name)
        options = dict(settings or {})
        if name == "Original":
            prefixes.append("original")
        elif name == "LoG":
            sigmas = options.get("sigma") or extractor.settings.get("sigma") or []
            prefixes.extend(
                "log-sigma-"
                + str(float(sigma)).replace(".", "-")
                + "-mm-3D"
                for sigma in sigmas
            )
        elif name == "Wavelet":
            level = int(options.get("level", 1))
            if level != 1:
                raise ValueError(
                    "configured CT feature schema supports Wavelet level 1 only"
                )
            prefixes.extend(
                f"wavelet-{band}"
                for band in (
                    "HHH",
                    "HHL",
                    "HLH",
                    "HLL",
                    "LHH",
                    "LHL",
                    "LLH",
                    "LLL",
                )
            )
        else:
            simple_prefixes = {
                "Square": "square",
                "SquareRoot": "squareroot",
                "Logarithm": "logarithm",
                "Exponential": "exponential",
                "Gradient": "gradient",
            }
            if name not in simple_prefixes:
                raise ValueError(
                    f"unsupported configured CT image type for feature schema: {name}"
                )
            prefixes.append(simple_prefixes[name])

    expected: set[str] = set()
    for class_name, feature_names in names_by_class.items():
        if class_name in {"shape", "shape2D"}:
            expected.update(
                f"original_{class_name}_{feature_name}"
                for feature_name in feature_names
            )
            continue
        for prefix in prefixes:
            expected.update(
                f"{prefix}_{class_name}_{feature_name}"
                for feature_name in feature_names
            )
    return expected


def _feature_schema_metadata(names: Iterable[str]) -> dict[str, Any]:
    expected = sorted({str(name) for name in names})
    encoded = json.dumps(expected, separators=(",", ":")).encode("utf-8")
    compressed = base64.b64encode(zlib.compress(encoded, level=9)).decode("ascii")
    return {
        RADIOMICS_EXPECTED_SCHEMA_SHA256_COLUMN: _feature_schema_sha256(expected),
        RADIOMICS_EXPECTED_COUNT_COLUMN: len(expected),
        RADIOMICS_EXPECTED_SCHEMA_SOURCE_COLUMN: "configured-extractor-v1",
        RADIOMICS_EXPECTED_SCHEMA_ZLIB_COLUMN: compressed,
    }


def _decode_feature_schema(value: Any) -> list[str]:
    try:
        compressed = base64.b64decode(str(value), validate=True)
        decoded = zlib.decompress(compressed)
        names = json.loads(decoded.decode("utf-8"))
    except Exception as exc:
        raise ValueError(
            "configured radiomics feature schema payload is invalid"
        ) from exc
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise ValueError("configured radiomics feature schema payload is not a name list")
    if names != sorted(set(names)):
        raise ValueError("configured radiomics feature schema names are not canonical")
    return names


def _component_qc(mask: np.ndarray) -> tuple[int, int]:
    foreground = np.asarray(mask, dtype=bool)
    if not foreground.any():
        return 0, 0
    from scipy import ndimage

    labels, count = ndimage.label(foreground, structure=np.ones((3, 3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())
    largest = int(sizes[1:].max()) if sizes.size > 1 else 0
    return int(count), largest


def _observed_dimensions(mask: np.ndarray) -> int:
    coordinates = np.where(np.asarray(mask, dtype=bool))
    if not coordinates or coordinates[0].size == 0:
        return 0
    return int(sum((int(axis.max()) - int(axis.min()) + 1) > 1 for axis in coordinates))


def resampled_mask_qc(
    image: Any,
    mask: Any,
    extractor: Any,
    window: Optional[tuple[float, float]],
) -> dict[str, Any]:
    from radiomics import imageoperations
    import SimpleITK as sitk

    settings = dict(extractor.settings)
    settings.pop("resegmentRange", None)
    loaded_image, loaded_mask = extractor.loadImage(image, mask, None, **settings)
    _, corrected_mask = imageoperations.checkMask(loaded_image, loaded_mask, **settings)
    if corrected_mask is not None:
        loaded_mask = corrected_mask
    label = int(settings.get("label", 1))
    image_array = sitk.GetArrayFromImage(loaded_image)
    morphologic = sitk.GetArrayFromImage(loaded_mask) == label
    morphologic_count = int(morphologic.sum())
    values = np.asarray(image_array[morphologic])
    finite = np.isfinite(values)
    nonfinite_count = int((~finite).sum())
    if window is None:
        below_count = 0
        above_count = 0
        retained_values = finite
    else:
        lower, upper = window
        below_count = int((finite & (values < lower)).sum())
        above_count = int((finite & (values > upper)).sum())
        retained_values = finite & (values >= lower) & (values <= upper)
    after_count = int(retained_values.sum())
    after_mask = np.zeros_like(morphologic, dtype=bool)
    after_mask[morphologic] = retained_values
    before_components, before_largest = _component_qc(morphologic)
    after_components, after_largest = _component_qc(after_mask)
    if after_count + below_count + above_count + nonfinite_count != morphologic_count:
        raise AssertionError(
            "resegmentation count identity failed: "
            f"{after_count}+{below_count}+{above_count}+{nonfinite_count}!={morphologic_count}"
        )
    return {
        "morphologic_resampled_voxel_count": morphologic_count,
        "resegment_after_count": after_count,
        "resegment_below_lower_count": below_count,
        "resegment_above_upper_count": above_count,
        "resegment_nonfinite_count": nonfinite_count,
        "components_26_before": before_components,
        "components_26_after": after_components,
        "largest_component_voxel_count_after": after_largest,
        "resegment_retained_fraction": (
            float(after_count / morphologic_count) if morphologic_count else None
        ),
        "largest_component_retained_fraction": (
            float(after_largest / before_largest) if before_largest else None
        ),
        "largest_component_fraction_after": (
            float(after_largest / after_count) if after_count else None
        ),
        "component_count_increased": bool(after_components > before_components),
        "observed_roi_dimensions_after_resegmentation": _observed_dimensions(after_mask),
        "largest_component_voxel_count_before": before_largest,
    }


def _runtime_versions() -> dict[str, str]:
    import SimpleITK as sitk
    import radiomics

    return {
        "pyradiomics_version": str(getattr(radiomics, "__version__", "unknown")),
        "simpleitk_version": str(sitk.Version_VersionString()),
        "numpy_version": str(np.__version__),
    }


def _arm_metadata(
    *,
    arm: str,
    decision: RoiClassDecision,
    extractor: Any,
    run_identifier: str,
    code_revision: str,
    native_voxel_count: int,
    required: bool,
    configured_parameter_hash_value: str,
) -> dict[str, Any]:
    window = decision.primary_resegment_range_hu if arm == PRIMARY_ARM else None
    return {
        "extraction_arm": arm,
        "effective_resegment_lower_hu": window[0] if window is not None else None,
        "effective_resegment_upper_hu": window[1] if window is not None else None,
        "roi_class": decision.roi_class,
        "roi_map_version": decision.map_version,
        "roi_map_hash": decision.map_hash,
        "roi_map_entry_source": decision.map_entry_source,
        "roi_class_adjudication_status": decision.adjudication_status,
        "effective_parameter_hash": effective_parameter_hash(extractor, arm=arm, window=window),
        "configured_parameter_hash": configured_parameter_hash_value,
        "code_revision": code_revision,
        "run_identifier": run_identifier,
        "native_mask_voxel_count": int(native_voxel_count),
        "roi_required": bool(required),
        "execution_host": socket.gethostname(),
        "environment_fingerprint": execution_environment_fingerprint(),
        **_runtime_versions(),
    }


def shape_physicality_violations(shape_features: Mapping[str, Any]) -> list[str]:
    """Return objective physical-validity failures from one shape feature set."""
    violations: list[str] = []
    strictly_positive = {
        "original_shape_MeshVolume",
        "original_shape_SurfaceArea",
        "original_shape_SurfaceVolumeRatio",
        "original_shape_VoxelVolume",
    }
    for name, value in sorted(shape_features.items()):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            violations.append(f"{name}=non_numeric")
            continue
        if not np.isfinite(numeric):
            violations.append(f"{name}=non_finite")
            continue
        if name in strictly_positive and numeric <= 0.0:
            violations.append(f"{name}={numeric:.17g} is not positive")
        elif numeric < 0.0:
            violations.append(f"{name}={numeric:.17g} is negative")
        if name == "original_shape_Sphericity" and numeric > 1.0 + 1e-6:
            violations.append(f"{name}={numeric:.17g} exceeds 1")
    return violations


def _effective_hashes_for_built_extractors(
    raw_extractor: Any,
    primary_extractor: Any,
    decision: RoiClassDecision,
) -> dict[str, str]:
    primary_qc_extractor = primary_extractor or raw_extractor
    return {
        SENSITIVITY_ARM: effective_parameter_hash(
            raw_extractor,
            arm=SENSITIVITY_ARM,
            window=None,
        ),
        PRIMARY_ARM: effective_parameter_hash(
            primary_qc_extractor,
            arm=PRIMARY_ARM,
            window=decision.primary_resegment_range_hu,
        ),
    }


def extract_ct_roi_arms(
    image: Any,
    mask: Any,
    *,
    factory: Callable[[], Any],
    decision: RoiClassDecision,
    common_metadata: Mapping[str, Any],
    run_identifier: str,
    code_revision: str,
    native_voxel_count: int,
    required: bool,
    configured_parameter_hashes: Optional[Mapping[str, str]] = None,
) -> list[dict[str, Any]]:
    shape_extractor, raw_extractor, primary_extractor = build_ct_extractors(
        factory, decision.primary_resegment_range_hu
    )
    effective_hashes = _effective_hashes_for_built_extractors(
        raw_extractor,
        primary_extractor,
        decision,
    )
    if decision.feature_publication_policy == FEATURE_POLICY_INVENTORY_ONLY:
        return disposition_rows_for_arms(
            common_metadata,
            decision=decision,
            disposition=decision.primary_intensity_texture_disposition,
            detail=(
                f"ROI class {decision.roi_class} is retained for inventory only. "
                "Radiomic feature publication is prohibited."
            ),
            failure_kind="declared_ineligible",
            run_identifier=run_identifier,
            code_revision=code_revision,
            native_voxel_count=native_voxel_count,
            required=required,
            effective_hashes=effective_hashes,
            configured_parameter_hashes=configured_parameter_hashes,
            runtime_versions=_runtime_versions(),
        )
    shape_result = _scalarize(shape_extractor.execute(image, mask))
    shape_features = _feature_subset(shape_result, SHAPE_FEATURE_MARKERS)
    shape_expected = _configured_feature_names(
        shape_extractor, observed_result=shape_result
    )
    shape_undefined = _approved_undefined_feature_names(shape_result)
    if not shape_features:
        raise RuntimeError("separate shape-only extractor returned no shape features")
    shape_violations = shape_physicality_violations(shape_features)
    if shape_violations:
        return disposition_rows_for_arms(
            common_metadata,
            decision=decision,
            disposition="failed_shape_physical_validity",
            detail="Shape feature physicality failed: " + " | ".join(shape_violations),
            failure_kind="invalid_shape_physicality",
            run_identifier=run_identifier,
            code_revision=code_revision,
            native_voxel_count=native_voxel_count,
            required=required,
            effective_hashes=effective_hashes,
            configured_parameter_hashes=configured_parameter_hashes,
            runtime_versions=_runtime_versions(),
        )

    raw_result = _scalarize(raw_extractor.execute(image, mask))
    raw_expected = shape_expected | _configured_feature_names(
        raw_extractor, observed_result=raw_result
    )
    raw_undefined = _approved_undefined_feature_names(raw_result)
    raw_result.update(shape_features)
    raw_qc = resampled_mask_qc(image, mask, raw_extractor, None)
    sensitivity = dict(raw_result)
    sensitivity.update(common_metadata)
    sensitivity.update(
        _arm_metadata(
            arm=SENSITIVITY_ARM,
            decision=decision,
            extractor=raw_extractor,
            run_identifier=run_identifier,
            code_revision=code_revision,
            native_voxel_count=native_voxel_count,
            required=required,
            configured_parameter_hash_value=str(
                (configured_parameter_hashes or {}).get(SENSITIVITY_ARM, "unavailable")
            ),
        )
    )
    sensitivity.update(raw_qc)
    sensitivity.update(
        {
            "shape_disposition": "success",
            "intensity_texture_disposition": "success",
            "extraction_status": "success",
            RADIOMICS_UNDEFINED_FEATURES_COLUMN: _feature_names_json(
                raw_undefined | shape_undefined
            ),
            **_feature_schema_metadata(raw_expected),
        }
    )

    primary_qc_extractor = primary_extractor or raw_extractor
    primary_expected = set(shape_expected)
    primary_qc = resampled_mask_qc(
        image, mask, primary_qc_extractor, decision.primary_resegment_range_hu
    )
    primary: dict[str, Any] = dict(shape_features)
    primary_undefined = set(shape_undefined)
    primary.update(common_metadata)
    primary.update(
        _arm_metadata(
            arm=PRIMARY_ARM,
            decision=decision,
            extractor=primary_qc_extractor,
            run_identifier=run_identifier,
            code_revision=code_revision,
            native_voxel_count=native_voxel_count,
            required=required,
            configured_parameter_hash_value=str(
                (configured_parameter_hashes or {}).get(PRIMARY_ARM, "unavailable")
            ),
        )
    )
    primary.update(primary_qc)
    primary["shape_disposition"] = "success"
    disposition = decision.primary_intensity_texture_disposition
    if primary_extractor is not None:
        minimum_size = primary_extractor.settings.get("minimumROISize")
        minimum_dimensions = int(primary_extractor.settings.get("minimumROIDimensions", 2))
        after_count = int(primary_qc["resegment_after_count"])
        after_dimensions = int(primary_qc["observed_roi_dimensions_after_resegmentation"])
        if minimum_size is not None and after_count <= int(minimum_size):
            disposition = "below_minimum_voxels"
        elif after_dimensions < minimum_dimensions:
            disposition = "below_minimum_dimensions"
        else:
            primary_result = _scalarize(primary_extractor.execute(image, mask))
            primary_expected.update(
                _configured_feature_names(
                    primary_extractor, observed_result=primary_result
                )
            )
            primary_undefined.update(_approved_undefined_feature_names(primary_result))
            primary_result = {
                key: value
                for key, value in primary_result.items()
                if not any(marker in key for marker in SHAPE_FEATURE_MARKERS)
            }
            primary_result.update(shape_features)
            primary = {**primary_result, **primary}
            disposition = "success"
    primary["intensity_texture_disposition"] = disposition
    primary["extraction_status"] = "success"
    primary[RADIOMICS_UNDEFINED_FEATURES_COLUMN] = _feature_names_json(
        primary_undefined
    )
    primary.update(_feature_schema_metadata(primary_expected))

    rows = [primary, sensitivity]
    assert_paired_shape_identity(rows)
    return rows


def disposition_rows_for_arms(
    common_metadata: Mapping[str, Any],
    *,
    decision: RoiClassDecision,
    disposition: str,
    detail: str,
    failure_kind: str,
    run_identifier: str,
    code_revision: str,
    native_voxel_count: Optional[int],
    required: bool,
    effective_hashes: Optional[Mapping[str, str]] = None,
    configured_parameter_hashes: Optional[Mapping[str, str]] = None,
    runtime_versions: Optional[Mapping[str, str]] = None,
) -> list[dict[str, Any]]:
    if not effective_hashes or any(
        not str(effective_hashes.get(arm) or "").strip()
        or str(effective_hashes.get(arm)).strip() == "unavailable"
        for arm in CT_EXTRACTION_ARMS
    ):
        raise ValueError(
            "disposition rows require runtime effective-parameter hashes for both CT arms"
        )
    versions = dict(runtime_versions or {
        "pyradiomics_version": "unavailable",
        "simpleitk_version": "unavailable",
        "numpy_version": str(np.__version__),
        "execution_host": socket.gethostname(),
        "environment_fingerprint": execution_environment_fingerprint(),
    })
    rows: list[dict[str, Any]] = []
    for arm in CT_EXTRACTION_ARMS:
        window = decision.primary_resegment_range_hu if arm == PRIMARY_ARM else None
        rows.append(
            {
                **common_metadata,
                "extraction_arm": arm,
                "effective_resegment_lower_hu": window[0] if window else None,
                "effective_resegment_upper_hu": window[1] if window else None,
                "roi_class": decision.roi_class,
                "roi_map_version": decision.map_version,
                "roi_map_hash": decision.map_hash,
                "roi_map_entry_source": decision.map_entry_source,
                "roi_class_adjudication_status": decision.adjudication_status,
                "effective_parameter_hash": str(effective_hashes[arm]),
                "configured_parameter_hash": str(
                    (configured_parameter_hashes or {}).get(arm, "unavailable")
                ),
                "code_revision": code_revision,
                "run_identifier": run_identifier,
                "native_mask_voxel_count": native_voxel_count,
                "roi_required": bool(required),
                "shape_disposition": disposition,
                "intensity_texture_disposition": disposition,
                "extraction_status": disposition,
                "extraction_status_detail": detail,
                "extraction_failure_kind": failure_kind,
                RADIOMICS_UNDEFINED_FEATURES_COLUMN: "[]",
                "morphologic_resampled_voxel_count": None,
                "resegment_after_count": None,
                "resegment_below_lower_count": None,
                "resegment_above_upper_count": None,
                "resegment_nonfinite_count": None,
                "components_26_before": None,
                "components_26_after": None,
                "largest_component_voxel_count_before": None,
                "largest_component_voxel_count_after": None,
                "resegment_retained_fraction": None,
                "largest_component_retained_fraction": None,
                "largest_component_fraction_after": None,
                "component_count_increased": None,
                "observed_roi_dimensions_after_resegmentation": None,
                **versions,
            }
        )
    return rows


def publication_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(column) if row.get(column) is not None else "").strip() for column in PUBLICATION_KEY_COLUMNS)


def base_identity_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return publication_key(row)[:-1]


def expected_publication_keys(base_rows: Iterable[Mapping[str, Any]]) -> set[tuple[str, ...]]:
    out: set[tuple[str, ...]] = set()
    for row in base_rows:
        base = tuple(str(row.get(column) if row.get(column) is not None else "").strip() for column in BASE_IDENTITY_COLUMNS)
        if any(not value for value in base):
            raise ValueError("expected CT identity has a blank publication-key field")
        for arm in CT_EXTRACTION_ARMS:
            out.add((*base, arm))
    return out


def _is_missing(value: Any) -> bool:
    try:
        import pandas as pd

        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)):
            return bool(missing)
    except Exception:
        pass
    return value is None


def _row_bool(value: Any) -> bool:
    if _is_missing(value):
        return False
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes"}
    return bool(value)


def assert_paired_shape_identity(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 2 or {str(row.get("extraction_arm")) for row in rows} != set(CT_EXTRACTION_ARMS):
        raise ValueError("paired shape assertion requires exactly the two governed CT arms")
    shape_columns = {
        str(column)
        for row in rows
        for column in row
        if any(marker in str(column) for marker in SHAPE_FEATURE_MARKERS)
    }
    for column in shape_columns:
        left, right = rows[0].get(column), rows[1].get(column)
        if _is_missing(left) and _is_missing(right):
            continue
        if _is_missing(left) != _is_missing(right):
            raise ValueError(f"paired CT shape feature {column} is missing in one extraction arm")
        try:
            equal = bool(np.isclose(float(left), float(right), rtol=0.0, atol=0.0, equal_nan=True))
        except Exception:
            equal = left == right
        if not equal:
            raise ValueError(f"paired CT shape feature differs across arms: {column}: {left!r} != {right!r}")


def validate_ct_publication(
    dataframe: Any,
    *,
    expected_keys: Optional[set[tuple[str, ...]]] = None,
    fail_on_unclassified_required: bool = True,
) -> set[tuple[str, ...]]:
    import pandas as pd

    if dataframe is None or dataframe.empty:
        raise ValueError("CT radiomics publication is empty")
    required_columns = set(PUBLICATION_KEY_COLUMNS) | {
        "shape_disposition",
        "intensity_texture_disposition",
        "effective_resegment_lower_hu",
        "effective_resegment_upper_hu",
        "roi_class",
        "roi_map_version",
        "roi_map_hash",
        "roi_class_adjudication_status",
        "effective_parameter_hash",
        "configured_parameter_hash",
        "code_revision",
        "pyradiomics_version",
        "simpleitk_version",
        "numpy_version",
        "run_identifier",
        "roi_required",
        *_COUNT_COLUMNS,
        "components_26_before",
        "components_26_after",
        "largest_component_retained_fraction",
        "component_count_increased",
    }
    missing_columns = sorted(required_columns - {str(column) for column in dataframe.columns})
    if missing_columns:
        raise ValueError("CT radiomics publication lacks required columns: " + ", ".join(missing_columns))

    records = dataframe.to_dict("records")
    keys = [publication_key(record) for record in records]
    if any(any(not field for field in key) for key in keys):
        raise ValueError("CT radiomics publication has a blank publication-key field")
    if len(keys) != len(set(keys)):
        raise ValueError("CT radiomics publication has duplicate full identities including extraction_arm")
    key_set = set(keys)
    if expected_keys is not None and key_set != expected_keys:
        raise ValueError(
            "CT radiomics publication identity set is incomplete or stale "
            f"(expected {len(expected_keys)}, found {len(key_set)})"
        )

    current_version, current_hash = roi_class_map_identity()
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for record in records:
        arm = str(record["extraction_arm"])
        if arm not in CT_EXTRACTION_ARMS:
            raise ValueError(f"invalid CT extraction_arm: {arm!r}")
        if str(record.get("roi_map_version")) != current_version or str(record.get("roi_map_hash")) != current_hash:
            raise ValueError("CT radiomics publication uses a stale ROI class map identity")
        for column in (
            "effective_parameter_hash",
            "configured_parameter_hash",
            "code_revision",
            "pyradiomics_version",
            "simpleitk_version",
            "numpy_version",
            "run_identifier",
        ):
            if _is_missing(record.get(column)) or not str(record.get(column)).strip():
                raise ValueError(f"CT radiomics row has blank {column}")
        if (
            fail_on_unclassified_required
            and _row_bool(record.get("roi_required"))
            and str(record.get("roi_class_adjudication_status"))
            == "operator_adjudication_required"
        ):
            raise ValueError(
                "required ROI is unclassified; operator adjudication is required before publication: "
                f"{record.get('segmentation_source')}/{record.get('roi_original_name')}"
            )
        disposition = str(record.get("intensity_texture_disposition") or "")
        if disposition not in _COMPLETE_DISPOSITIONS:
            raise ValueError(f"CT radiomics row has incomplete disposition {disposition!r}")
        if all(not _is_missing(record.get(column)) for column in _COUNT_COLUMNS):
            values = [int(record[column]) for column in _COUNT_COLUMNS]
            morphologic, after, below, above, nonfinite = values
            if after + below + above + nonfinite != morphologic:
                raise ValueError(
                    "resegmentation count identity failed in publication row: "
                    f"{after}+{below}+{above}+{nonfinite}!={morphologic}"
                )
        if arm == SENSITIVITY_ARM:
            if not _is_missing(record.get("effective_resegment_lower_hu")) or not _is_missing(
                record.get("effective_resegment_upper_hu")
            ):
                raise ValueError("sensitivity_raw row must not carry an effective resegmentation window")
        else:
            expected_window = _PRIMARY_WINDOWS.get(str(record.get("roi_class")))
            lower = record.get("effective_resegment_lower_hu")
            upper = record.get("effective_resegment_upper_hu")
            if expected_window is None:
                if not _is_missing(lower) or not _is_missing(upper):
                    raise ValueError("non-applicable primary row carries a resegmentation window")
            elif _is_missing(lower) or _is_missing(upper) or (
                float(lower), float(upper)
            ) != expected_window:
                raise ValueError("primary row carries the wrong class-governed resegmentation window")
            if disposition != "success":
                feature_columns = [
                    column
                    for column in dataframe.columns
                    if any(marker in str(column) for marker in INTENSITY_TEXTURE_FEATURE_MARKERS)
                ]
                if any(not _is_missing(record.get(column)) for column in feature_columns):
                    raise ValueError(
                        "primary disposition row contains intensity or texture feature values"
                    )
        grouped.setdefault(base_identity_key(record), []).append(record)

    for base, pair in grouped.items():
        if len(pair) != 2 or {str(row["extraction_arm"]) for row in pair} != set(CT_EXTRACTION_ARMS):
            raise ValueError(f"CT base identity does not have exactly both extraction arms: {base}")
        if len({str(row.get("run_identifier")) for row in pair}) != 1:
            raise ValueError(f"paired CT arms do not share one run_identifier: {base}")
        assert_paired_shape_identity(pair)
    validate_feature_completeness_records(dataframe)
    return key_set


def _jsonify_nested_columns(dataframe: Any) -> Any:
    if dataframe.empty:
        return dataframe
    output = dataframe.copy()
    for column in output.columns:
        if output[column].map(lambda value: isinstance(value, (dict, list, set, tuple))).any():
            output[column] = output[column].map(
                lambda value: json.dumps(value, default=str, sort_keys=True)
                if isinstance(value, (dict, list, set, tuple))
                else value
            )
    return output


def write_ct_publication_atomic(
    dataframe: Any,
    workbook_path: Path,
    *,
    expected_keys: Optional[set[tuple[str, ...]]] = None,
) -> Path:
    import pandas as pd

    workbook_path = Path(workbook_path)
    parquet_path = workbook_path.with_suffix(".parquet")
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    publish_df = _jsonify_nested_columns(normalize_radiomics_dataframe(dataframe))
    publish_df = _add_feature_completeness_metadata(publish_df)
    completeness = validate_feature_completeness_records(publish_df)
    required_incomplete = [
        record
        for record in completeness
        if record["status"] == "incomplete" and record["required"]
    ]
    if required_incomplete:
        examples = "; ".join(
            f"{record['publication_key']} missing={record['missing_features']}"
            for record in required_incomplete[:3]
        )
        raise ValueError(
            f"{len(required_incomplete)} required ROI-arm rows have incomplete "
            f"feature vectors: {examples}"
        )
    expected_strings = expected_radiomics_string_columns(publish_df)
    validate_ct_publication(publish_df, expected_keys=expected_keys)

    fd, parquet_tmp_name = tempfile.mkstemp(
        prefix=f".{parquet_path.name}.", suffix=".tmp.parquet", dir=parquet_path.parent
    )
    os.close(fd)
    parquet_tmp = Path(parquet_tmp_name)
    fd, workbook_tmp_name = tempfile.mkstemp(
        prefix=f".{workbook_path.name}.", suffix=".tmp.xlsx", dir=workbook_path.parent
    )
    os.close(fd)
    workbook_tmp = Path(workbook_tmp_name)
    try:
        publish_df.to_parquet(parquet_tmp, index=False, engine="pyarrow")
        assert_radiomics_arrow_schema(
            parquet_tmp, expected_string_columns=expected_strings
        )
        parquet_check = pd.read_parquet(parquet_tmp, engine="pyarrow")
        validate_ct_publication(parquet_check, expected_keys=expected_keys)
        publish_df.to_excel(workbook_tmp, index=False)
        workbook_check = pd.read_excel(workbook_tmp, engine="openpyxl")
        validate_ct_publication(workbook_check, expected_keys=expected_keys)
        os.replace(parquet_tmp, parquet_path)
        os.replace(workbook_tmp, workbook_path)
    except Exception:
        # Destinations are untouched until both temporary files validate. Preserve a
        # prior published pair if creation or validation of the replacement fails.
        raise
    finally:
        parquet_tmp.unlink(missing_ok=True)
        workbook_tmp.unlink(missing_ok=True)
    return parquet_path


def read_authoritative_ct_publication(path: Path) -> Any:
    import pandas as pd

    candidate = Path(path)
    parquet_path = candidate if candidate.suffix == ".parquet" else candidate.with_suffix(".parquet")
    dataframe = pd.read_parquet(parquet_path, engine="pyarrow")
    assert_radiomics_arrow_schema(
        parquet_path,
        expected_string_columns=expected_radiomics_string_columns(dataframe),
    )
    validate_ct_publication(dataframe)
    return dataframe


def _validate_configuration_dependency(dataframe: Any, dependency_path: Path) -> str:
    from .config_dependencies import (
        RADIOMICS_DEPENDENCY_SCHEMA,
        radiomics_row_parameter_key,
        read_stage_dependency,
    )

    record = read_stage_dependency(dependency_path, expected_stage="radiomics")
    payload = record.get("payload") or {}
    provenance = payload.get("parameter_provenance") or {}
    if provenance.get("schema") != RADIOMICS_DEPENDENCY_SCHEMA:
        raise ValueError("radiomics configuration dependency lacks parameter provenance")
    ct_manifest = provenance.get("ct") or {}
    roi_map = ct_manifest.get("roi_class_map") or {}
    configured_hashes = ct_manifest.get("configured_parameter_hashes") or {}
    if not isinstance(configured_hashes, Mapping) or not configured_hashes:
        raise ValueError("radiomics configuration dependency has no configured CT hashes")

    for row in dataframe.to_dict("records"):
        if str(row.get("roi_map_version")) != str(roi_map.get("version")) or str(
            row.get("roi_map_hash")
        ) != str(roi_map.get("sha256")):
            raise ValueError(
                "CT radiomics row provenance disagrees with the DAG ROI class map"
            )
        keys = [
            radiomics_row_parameter_key(
                str(row.get("extraction_arm") or ""),
                row.get("effective_resegment_lower_hu"),
                row.get("effective_resegment_upper_hu"),
                large_roi=large_roi,
            )
            for large_roi in (False, True)
        ]
        expected_hashes = {
            str(configured_hashes[key])
            for key in keys
            if configured_hashes.get(key)
        }
        if str(row.get("configured_parameter_hash")) not in expected_hashes:
            raise ValueError(
                "CT radiomics row configured_parameter_hash disagrees with the "
                f"DAG extraction configuration for {keys}"
            )
    return str(record["sha256"])


def sentinel_payload(
    dataframe: Any,
    parquet_path: Path,
    *,
    configuration_dependency: Optional[Path] = None,
) -> dict[str, Any]:
    keys = sorted("\x1f".join(key) for key in validate_ct_publication(dataframe))
    schema_fields = (
        RADIOMICS_EXPECTED_SCHEMA_SHA256_COLUMN,
        RADIOMICS_EXPECTED_COUNT_COLUMN,
        RADIOMICS_EXPECTED_SCHEMA_SOURCE_COLUMN,
        RADIOMICS_EXPECTED_SCHEMA_ZLIB_COLUMN,
    )
    for row in dataframe.to_dict("records"):
        if str(row.get("extraction_status") or "success") != "success":
            continue
        missing_schema = [
            field
            for field in schema_fields
            if _is_missing(row.get(field)) or not str(row.get(field)).strip()
        ]
        if missing_schema:
            raise ValueError(
                "completion sentinel requires configured feature schema provenance "
                "for every successful ROI-arm row"
            )
        if str(row[RADIOMICS_EXPECTED_SCHEMA_SOURCE_COLUMN]) != "configured-extractor-v1":
            raise ValueError(
                "completion sentinel refuses inferred feature schema provenance"
            )
    key_digest = hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()
    completeness = validate_feature_completeness_records(dataframe)

    def _single_metadata_value(column: str, fallback: str) -> str:
        if column not in dataframe.columns:
            return fallback
        values = {
            str(value).strip()
            for value in dataframe[column].tolist()
            if not _is_missing(value) and str(value).strip()
        }
        if not values:
            return fallback
        if len(values) != 1:
            raise ValueError(f"radiomics publication has multiple {column} values")
        return next(iter(values))

    def _fingerprint_metadata_value(column: str, fallback: str) -> str:
        if column not in dataframe.columns:
            return fallback
        values = sorted(
            {
                str(value).strip()
                for value in dataframe[column].tolist()
                if not _is_missing(value) and str(value).strip()
            }
        )
        if not values:
            return fallback
        if len(values) == 1:
            return values[0]
        encoded = json.dumps(values, separators=(",", ":")).encode("utf-8")
        return "sha256-set:" + hashlib.sha256(encoded).hexdigest()

    eligible_row_count = len(analysis_eligible_feature_rows(dataframe))
    payload = {
        "status": "ok",
        "schema": "rtpipeline-radiomics-completion-v2",
        "authoritative_parquet": str(Path(parquet_path).name),
        "authoritative_parquet_sha256": file_sha256(Path(parquet_path)),
        "input_closure_sha256": input_closure_sha256(
            Path(parquet_path).parent, dataframe
        ),
        "canonical_feature_table_sha256": canonical_feature_table_sha256(dataframe),
        "missingness_digest": missingness_digest(dataframe),
        "row_count": int(len(dataframe)),
        "identity_set_sha256": key_digest,
        "roi_map_version": str(dataframe["roi_map_version"].iloc[0]),
        "roi_map_hash": str(dataframe["roi_map_hash"].iloc[0]),
        "run_identifiers": sorted({str(value) for value in dataframe["run_identifier"].tolist()}),
        "effective_parameter_hashes": sorted(
            {str(value) for value in dataframe["effective_parameter_hash"].tolist()}
        ),
        "configured_parameter_hashes": sorted(
            {str(value) for value in dataframe["configured_parameter_hash"].tolist()}
        ),
        "code_revisions": sorted(
            {str(value) for value in dataframe["code_revision"].tolist()}
        ),
        "execution_host": _single_metadata_value(
            EXECUTION_HOST_COLUMN, socket.gethostname()
        ),
        "environment_fingerprint": _fingerprint_metadata_value(
            ENVIRONMENT_FINGERPRINT_COLUMN, execution_environment_fingerprint()
        ),
        "feature_completeness": {
            "schema": FEATURE_COMPLETENESS_SCHEMA,
            "row_count": len(completeness),
            "analysis_eligible_row_count": eligible_row_count,
            "complete_row_count": sum(
                record["status"] == "complete" for record in completeness
            ),
            "complete_with_undefined_row_count": sum(
                record["status"] == "complete_with_undefined"
                for record in completeness
            ),
            "complete_with_not_applicable_row_count": sum(
                record["status"] == "complete_with_not_applicable"
                for record in completeness
            ),
            "failed_extraction_row_count": sum(
                record["status"] == "failed_extraction"
                for record in completeness
            ),
            "geometry_exclusion_row_count": sum(
                record["status"] == "excluded_geometry"
                for record in completeness
            ),
            "declared_exclusion_row_count": sum(
                record["status"] == "excluded_declared"
                for record in completeness
            ),
            "incomplete_row_count": sum(
                record["status"] == "incomplete" for record in completeness
            ),
            "undefined_feature_row_count": sum(
                bool(record["undefined_features"]) for record in completeness
            ),
        },
    }
    if configuration_dependency is not None:
        payload["configuration_dependency_sha256"] = _validate_configuration_dependency(
            dataframe, Path(configuration_dependency)
        )
    return payload


def validate_completion_sentinel(
    course_dir: Path,
    sentinel_path: Optional[Path] = None,
    *,
    configuration_dependency: Optional[Path] = None,
) -> dict[str, Any]:
    course_dir = Path(course_dir)
    target = Path(sentinel_path) if sentinel_path else course_dir / ".radiomics_done"
    try:
        observed = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"governed radiomics completion sentinel is unreadable: {target}") from exc
    if not isinstance(observed, dict):
        raise ValueError("governed radiomics completion sentinel must be a JSON object")
    parquet_path = course_dir / "radiomics_ct.parquet"
    dataframe = read_authoritative_ct_publication(parquet_path)
    expected = sentinel_payload(
        dataframe,
        parquet_path,
        configuration_dependency=configuration_dependency,
    )
    fields = [
        "status",
        "schema",
        "authoritative_parquet",
        "authoritative_parquet_sha256",
        "input_closure_sha256",
        "canonical_feature_table_sha256",
        "missingness_digest",
        "row_count",
        "identity_set_sha256",
        "roi_map_version",
        "roi_map_hash",
        "run_identifiers",
        "effective_parameter_hashes",
        "configured_parameter_hashes",
        "code_revisions",
        "execution_host",
        "environment_fingerprint",
        "feature_completeness",
    ]
    if configuration_dependency is not None:
        fields.append("configuration_dependency_sha256")
    for field in fields:
        if observed.get(field) != expected.get(field):
            raise ValueError(
                f"governed radiomics completion sentinel is stale for {field}"
            )
    return observed


def write_completion_sentinel(
    course_dir: Path,
    sentinel_path: Optional[Path] = None,
    *,
    configuration_dependency: Optional[Path] = None,
) -> Path:
    course_dir = Path(course_dir)
    if configuration_dependency is None:
        configured_dependency = os.environ.get(
            "RTPIPELINE_RADIOMICS_CONFIG_DEPENDENCY", ""
        ).strip()
        if configured_dependency:
            configuration_dependency = Path(configured_dependency)
    parquet_path = course_dir / "radiomics_ct.parquet"
    dataframe = read_authoritative_ct_publication(parquet_path)
    payload = sentinel_payload(
        dataframe,
        parquet_path,
        configuration_dependency=configuration_dependency,
    )
    target = Path(sentinel_path) if sentinel_path else course_dir / ".radiomics_done"
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_path, target)
    finally:
        tmp_path.unlink(missing_ok=True)
    return target
