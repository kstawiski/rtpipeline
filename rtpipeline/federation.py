"""Fail-closed packets for distributed aggregate radiomics reliability analysis.

This module exchanges cohort-level reliability summaries only. It is not a
federated-learning framework, secure aggregation, differential privacy, an
anonymity mechanism, or a privacy guarantee.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import __version__


PACKET_SCHEMA_VERSION = 3
METRICS_FILENAME = "metrics.csv.gz"
MANIFEST_FILENAME = "manifest.json"
PRIVACY_SCOPE = (
    "Distributed aggregate radiomics reliability analysis only; not federated "
    "learning, secure aggregation, differential privacy, anonymity, or a privacy guarantee."
)

PACKET_COLUMNS: tuple[str, ...] = (
    "feature_name",
    "roi_name",
    "body_region",
    "segmentation_source",
    "n_subjects",
    "n_raters",
    "n_subjects_cov",
    "n_subjects_qcd",
    "icc",
    "icc_ci_low",
    "icc_ci_high",
    "cov_percent",
    "qcd",
    "classification",
    "feature_family",
    "image_type",
)
INVENTORY_COLUMNS: tuple[str, ...] = (
    "body_region",
    "segmentation_source",
    "roi_name",
    "feature_name",
    "feature_family",
    "image_type",
)
NATIVE_ROBUSTNESS_REQUIRED_COLUMNS: tuple[str, ...] = (
    "structure",
    "segmentation_source",
    "feature_name",
    "n_subjects",
    "n_perturbations",
    "n_subjects_cov",
    "n_subjects_qcd",
    "icc",
    "icc_ci95_low",
    "icc_ci95_high",
    "cov_pct",
    "qcd",
    "robustness_label",
)
STRING_COLUMNS: tuple[str, ...] = (
    "feature_name",
    "roi_name",
    "body_region",
    "segmentation_source",
    "classification",
    "feature_family",
    "image_type",
)
INTEGER_COLUMNS: tuple[str, ...] = (
    "n_subjects",
    "n_raters",
    "n_subjects_cov",
    "n_subjects_qcd",
)
FLOAT_COLUMNS: tuple[str, ...] = ("icc",)
NULLABLE_FLOAT_COLUMNS: tuple[str, ...] = (
    "icc_ci_low",
    "icc_ci_high",
    "cov_percent",
    "qcd",
)
IDENTITY_COLUMNS: tuple[str, ...] = (
    "body_region",
    "segmentation_source",
    "roi_name",
    "feature_name",
)
ALLOWED_CLASSIFICATIONS = frozenset({"Robust", "Acceptable", "Poor", "Not Evaluable"})
NATIVE_CLASSIFICATION_MAP = {
    "robust": "Robust",
    "acceptable": "Acceptable",
    "poor": "Poor",
    "not_evaluable": "Not Evaluable",
}
PACKET_CLASSIFICATION_RULE = {
    "icc_statistic": "icc_ci_low_when_both_ci_limits_finite_else_icc_when_both_null",
    "robust": {"minimum_icc": 0.90, "maximum_cov_percent": 10.0},
    "acceptable": {"minimum_icc": 0.75, "maximum_cov_percent": 20.0},
    "not_evaluable_when": "n_subjects_cov_less_than_n_subjects",
}
EXACT_COMPATIBILITY_POLICY = {
    "processing_config": "exact_sha256",
    "source_artifact": "exact_kind_and_sha256",
    "rtpipeline_version": "exact",
    "feature_roi_inventory": "exact_sha256_including_segmentation_source",
}
ALLOWED_ARTIFACT_KINDS = frozenset({"source_tree", "container_image"})
AUDIT_KEYS: tuple[str, ...] = (
    "forbidden_columns",
    "absolute_path_values",
    "uri_values",
    "dicom_uid_values",
    "date_values",
    "direct_identifier_values",
    "hostname_values",
)
MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "schema_sha256",
        "contract_id",
        "contract_sha256",
        "compatibility_policy",
        "processing_config_sha256",
        "source_artifact_kind",
        "source_artifact_sha256",
        "rtpipeline_version",
        "node_id",
        "metrics_file",
        "metrics_sha256",
        "feature_roi_inventory_sha256",
        "row_count",
        "columns",
        "minimum_subjects",
        "n_subjects_min",
        "n_subjects_max",
        "body_regions",
        "roi_count",
        "feature_count",
        "content_audit",
        "privacy_scope",
    }
)
PACKET_FILENAMES = frozenset({METRICS_FILENAME, MANIFEST_FILENAME})

FORBIDDEN_COLUMN_NAMES = {
    "patient_id",
    "patientid",
    "subject_id",
    "subjectid",
    "course_id",
    "courseid",
    "accession_number",
    "accessionnumber",
    "dicom_uid",
    "study_instance_uid",
    "series_instance_uid",
    "sop_instance_uid",
    "study_date",
    "series_date",
    "acquisition_date",
    "birth_date",
    "date_of_birth",
    "source_path",
    "ct_path",
    "rs_path",
    "hostname",
    "outcome",
    "event",
    "survival_time",
}

NODE_ID_RE = re.compile(r"^node-[A-Za-z0-9_-]{3,64}$")
CONTRACT_ID_RE = re.compile(r"^[A-Za-z0-9._-]{3,128}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][A-Za-z0-9._-]+)?$")
DICOM_UID_RE = re.compile(r"^\d+(?:\.\d+){2,}$")
DATE_RE = re.compile(r"^(?:\d{4}-\d{2}-\d{2}(?:[ T].*)?|\d{8}(?:[ T]?\d{6}(?:\.\d+)?)?)$")
WINDOWS_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
UNC_PATH_RE = re.compile(r"^(?:\\\\|//)[^\\/]+[\\/]")
URI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
DIRECT_IDENTIFIER_RE = re.compile(
    r"^(?:(?:patient|subject|course|mrn|medical[-_ ]?record)[-_: ]*[A-Za-z0-9][A-Za-z0-9._-]{2,}|P\d{4,})$",
    re.IGNORECASE,
)
HOSTNAME_RE = re.compile(
    r"^(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"(?:local|internal|lan|[A-Za-z]{2,63})$",
    re.IGNORECASE,
)


class FederationPacketError(ValueError):
    """Raised when a site packet violates the sharing contract."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FederationPacketError(f"Value is not canonical finite JSON: {exc}") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_processing_config_sha256(config: Mapping[str, object]) -> str:
    """Hash a processing configuration after canonical JSON normalization."""

    if not isinstance(config, Mapping):
        raise FederationPacketError("Processing configuration must be a JSON object")
    normalized = json.loads(_canonical_json_bytes(dict(config)).decode("utf-8"))
    return _sha256_bytes(_canonical_json_bytes(normalized))


def _require_sha256(name: str, value: str) -> str:
    if not SHA256_RE.fullmatch(str(value)):
        raise FederationPacketError(f"{name} must be a lowercase SHA-256 digest")
    return str(value)


def _validate_compatibility_inputs(
    *,
    processing_config_sha256: str,
    source_artifact_kind: str,
    source_artifact_sha256: str,
    expected_rtpipeline_version: str,
    expected_feature_roi_inventory_sha256: str,
) -> None:
    _require_sha256("processing_config_sha256", processing_config_sha256)
    _require_sha256("source_artifact_sha256", source_artifact_sha256)
    _require_sha256(
        "expected_feature_roi_inventory_sha256",
        expected_feature_roi_inventory_sha256,
    )
    if source_artifact_kind not in ALLOWED_ARTIFACT_KINDS:
        raise FederationPacketError(
            "source_artifact_kind must be 'source_tree' or 'container_image'"
        )
    if not VERSION_RE.fullmatch(str(expected_rtpipeline_version)):
        raise FederationPacketError("expected_rtpipeline_version is invalid")


def packet_schema_sha256() -> str:
    schema = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "columns": list(PACKET_COLUMNS),
        "inventory_columns": list(INVENTORY_COLUMNS),
        "native_required_columns": list(NATIVE_ROBUSTNESS_REQUIRED_COLUMNS),
        "string_columns": list(STRING_COLUMNS),
        "integer_columns": list(INTEGER_COLUMNS),
        "float_columns": list(FLOAT_COLUMNS),
        "nullable_float_columns": list(NULLABLE_FLOAT_COLUMNS),
        "identity_columns": list(IDENTITY_COLUMNS),
        "manifest_keys": sorted(MANIFEST_KEYS),
    }
    return _sha256_bytes(_canonical_json_bytes(schema))


def packet_contract_document(
    contract_id: str,
    minimum_subjects: int,
    *,
    processing_config_sha256: str,
    source_artifact_kind: str,
    source_artifact_sha256: str,
    expected_rtpipeline_version: str,
    expected_feature_roi_inventory_sha256: str,
) -> dict[str, object]:
    """Return the canonical, hash-bound schema-v3 packet contract."""

    if not CONTRACT_ID_RE.fullmatch(contract_id):
        raise FederationPacketError(
            "contract_id must contain only letters, numbers, dot, underscore, or hyphen"
        )
    if int(minimum_subjects) < 2:
        raise FederationPacketError("minimum_subjects must be at least 2")
    _validate_compatibility_inputs(
        processing_config_sha256=processing_config_sha256,
        source_artifact_kind=source_artifact_kind,
        source_artifact_sha256=source_artifact_sha256,
        expected_rtpipeline_version=expected_rtpipeline_version,
        expected_feature_roi_inventory_sha256=expected_feature_roi_inventory_sha256,
    )
    return {
        "contract_id": contract_id,
        "packet_schema_version": PACKET_SCHEMA_VERSION,
        "packet_schema_sha256": packet_schema_sha256(),
        "packet_files": sorted(PACKET_FILENAMES),
        "manifest_keys": sorted(MANIFEST_KEYS),
        "audit_keys": list(AUDIT_KEYS),
        "minimum_subjects": int(minimum_subjects),
        "minimum_raters": 2,
        "allowed_classifications": sorted(ALLOWED_CLASSIFICATIONS),
        "classification_rule": PACKET_CLASSIFICATION_RULE,
        "compatibility_policy": EXACT_COMPATIBILITY_POLICY,
        "processing_config_sha256": processing_config_sha256,
        "source_artifact": {
            "kind": source_artifact_kind,
            "sha256": source_artifact_sha256,
        },
        "rtpipeline_version": expected_rtpipeline_version,
        "expected_feature_roi_inventory_sha256": expected_feature_roi_inventory_sha256,
        "semantic_rules": [
            "icc_in_closed_interval_-1_1",
            "icc_confidence_limits_both_finite_or_both_null",
            "finite_ci_limits_in_closed_interval_-1_1",
            "finite_ci_low_le_icc_le_ci_high",
            "point_icc_used_only_when_both_ci_limits_null",
            "cov_percent_and_qcd_nonnegative_when_evaluable",
            "relative_dispersion_denominators_between_zero_and_n_subjects",
            "relative_dispersion_null_if_and_only_if_denominator_zero",
            "incomplete_cov_denominator_requires_not_evaluable_classification",
            "classification_must_match_hash_bound_icc_and_cov_thresholds",
            "unique_body_region_segmentation_source_roi_name_feature_name_identity",
            "observed_inventory_must_equal_expected_inventory",
            "finite_required_numeric_values",
        ],
        "serialization": "deterministic_csv_gzip_float64_17_significant_digits",
        "publication": "sibling_staging_directory_then_atomic_final_replace",
        "privacy_scope": PRIVACY_SCOPE,
    }


def packet_contract_sha256(
    contract_id: str,
    minimum_subjects: int,
    *,
    processing_config_sha256: str,
    source_artifact_kind: str,
    source_artifact_sha256: str,
    expected_rtpipeline_version: str,
    expected_feature_roi_inventory_sha256: str,
) -> str:
    return _sha256_bytes(
        _canonical_json_bytes(
            packet_contract_document(
                contract_id,
                minimum_subjects,
                processing_config_sha256=processing_config_sha256,
                source_artifact_kind=source_artifact_kind,
                source_artifact_sha256=source_artifact_sha256,
                expected_rtpipeline_version=expected_rtpipeline_version,
                expected_feature_roi_inventory_sha256=expected_feature_roi_inventory_sha256,
            )
        )
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json_object(path: Path) -> dict[str, object]:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise FederationPacketError(f"Duplicate JSON key in {path.name}: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys
    )
    if not isinstance(value, dict):
        raise FederationPacketError(f"{path.name} must contain one JSON object")
    return value


def _write_deterministic_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    csv_bytes = frame.to_csv(
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    ).encode("utf-8")
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            zipped.write(csv_bytes)


def _read_table(path: Path) -> pd.DataFrame:
    suffixes = path.suffixes
    if suffixes[-2:] == [".csv", ".gz"] or path.suffix.lower() == ".csv":
        return pd.read_csv(path, float_precision="round_trip")
    if path.suffix.lower() in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise FederationPacketError(
                "Reading Parquet requires a pandas Parquet engine; use CSV or install pyarrow."
            ) from exc
    raise FederationPacketError(f"Unsupported input table format: {path}")


def _normalized_column_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _metadata_audit_values(metadata: dict[str, object]) -> Iterable[str]:
    exempt_keys = {
        "schema_sha256",
        "contract_sha256",
        "metrics_sha256",
        "feature_roi_inventory_sha256",
        "processing_config_sha256",
        "source_artifact_sha256",
        "rtpipeline_version",
        "metrics_file",
        "privacy_scope",
    }

    def walk(value: object, key: str | None = None) -> Iterable[str]:
        if key in exempt_keys:
            return
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                yield str(child_key)
                yield from walk(child_value, str(child_key))
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from walk(child, key)
        elif isinstance(value, str):
            yield value

    yield from walk(metadata)


def _audit_forbidden_content(
    frame: pd.DataFrame, metadata: dict[str, object] | None = None
) -> dict[str, int]:
    forbidden_columns = sum(
        _normalized_column_name(column) in FORBIDDEN_COLUMN_NAMES
        for column in frame.columns
    )
    absolute_paths = 0
    uri_values = 0
    dicom_uids = 0
    date_values = 0
    direct_identifiers = 0
    hostname_values = 0
    values: list[str] = []
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        values.extend(frame[column].dropna().astype(str).str.strip().tolist())
    if metadata is not None:
        forbidden_columns += sum(
            _normalized_column_name(key) in FORBIDDEN_COLUMN_NAMES for key in metadata
        )
        values.extend(value.strip() for value in _metadata_audit_values(metadata))
    for value in values:
        if value.startswith("/") or WINDOWS_PATH_RE.match(value) or UNC_PATH_RE.match(value):
            absolute_paths += 1
        if URI_RE.match(value):
            uri_values += 1
        if DICOM_UID_RE.fullmatch(value):
            dicom_uids += 1
        if DATE_RE.fullmatch(value):
            date_values += 1
        if DIRECT_IDENTIFIER_RE.fullmatch(value):
            direct_identifiers += 1
        if HOSTNAME_RE.fullmatch(value) and not DICOM_UID_RE.fullmatch(value):
            hostname_values += 1
    return {
        "forbidden_columns": int(forbidden_columns),
        "absolute_path_values": int(absolute_paths),
        "uri_values": int(uri_values),
        "dicom_uid_values": int(dicom_uids),
        "date_values": int(date_values),
        "direct_identifier_values": int(direct_identifiers),
        "hostname_values": int(hostname_values),
    }


def _validate_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    actual = list(inventory.columns)
    if actual != list(INVENTORY_COLUMNS):
        missing = [column for column in INVENTORY_COLUMNS if column not in actual]
        extra = [column for column in actual if column not in INVENTORY_COLUMNS]
        raise FederationPacketError(
            "Inventory columns must exactly match the contract and order; "
            f"missing={missing}, extra={extra}, actual={actual}"
        )
    if inventory.empty:
        raise FederationPacketError("Expected feature/ROI inventory is empty")
    validated = inventory.copy()
    for column in INVENTORY_COLUMNS:
        if validated[column].isna().any():
            raise FederationPacketError(f"Inventory column {column} contains missing values")
        validated[column] = validated[column].astype(str).str.strip()
        if validated[column].eq("").any():
            raise FederationPacketError(f"Inventory column {column} contains empty values")
    if validated.duplicated(list(IDENTITY_COLUMNS)).any():
        raise FederationPacketError("Expected feature/ROI inventory contains duplicate identities")
    audit = _audit_forbidden_content(validated)
    if any(audit.values()):
        raise FederationPacketError(f"Forbidden inventory content detected: {audit}")
    return validated.sort_values(list(IDENTITY_COLUMNS), kind="mergesort").reset_index(drop=True)


def feature_roi_inventory_sha256(inventory: pd.DataFrame) -> str:
    """Hash the exact source, body-region, ROI, feature, family, and image inventory."""

    validated = _validate_inventory(inventory)
    return _sha256_bytes(
        _canonical_json_bytes(validated.to_dict(orient="records"))
    )


def _inventory_from_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, INVENTORY_COLUMNS].copy()


def _validate_metrics(frame: pd.DataFrame, minimum_subjects: int) -> pd.DataFrame:
    actual = list(frame.columns)
    expected = list(PACKET_COLUMNS)
    if actual != expected:
        missing = [column for column in expected if column not in actual]
        extra = [column for column in actual if column not in expected]
        raise FederationPacketError(
            "Metrics columns must exactly match the packet contract and order; "
            f"missing={missing}, extra={extra}, actual={actual}"
        )
    if frame.empty:
        raise FederationPacketError("Metrics table is empty")
    validated = frame.copy()
    for column in STRING_COLUMNS:
        if validated[column].isna().any():
            raise FederationPacketError(f"Column {column} contains missing values")
        validated[column] = validated[column].astype(str).str.strip()
        if validated[column].eq("").any():
            raise FederationPacketError(f"Column {column} contains empty values")
    if validated.duplicated(list(IDENTITY_COLUMNS)).any():
        raise FederationPacketError(
            "Metrics contain duplicate normalized body_region/segmentation_source/"
            "roi_name/feature_name identities"
        )

    for column in INTEGER_COLUMNS:
        numeric = pd.to_numeric(validated[column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise FederationPacketError(f"Column {column} must contain finite integers")
        if not np.equal(numeric, np.floor(numeric)).all():
            raise FederationPacketError(f"Column {column} must contain integers")
        validated[column] = numeric.astype("int64")

    for column in FLOAT_COLUMNS:
        numeric = pd.to_numeric(validated[column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise FederationPacketError(f"Column {column} must contain finite numbers")
        validated[column] = numeric.astype("float64")

    for column in NULLABLE_FLOAT_COLUMNS:
        raw = validated[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        invalid = numeric.isna() & raw.notna()
        if invalid.any() or not np.isfinite(numeric.dropna().to_numpy(dtype=float)).all():
            raise FederationPacketError(
                f"Column {column} must contain finite numbers or missing values"
            )
        validated[column] = numeric.astype("float64")

    if int(validated["n_subjects"].min()) < int(minimum_subjects):
        raise FederationPacketError(
            f"Packet contains n_subjects below minimum_subjects={minimum_subjects}"
        )
    if (validated["n_raters"] < 2).any():
        raise FederationPacketError("Packet contains n_raters below 2")
    for metric, denominator in (
        ("cov_percent", "n_subjects_cov"),
        ("qcd", "n_subjects_qcd"),
    ):
        if ((validated[denominator] < 0) | (validated[denominator] > validated["n_subjects"])).any():
            raise FederationPacketError(
                f"Column {denominator} must be between zero and n_subjects"
            )
        metric_missing = validated[metric].isna()
        if (metric_missing != validated[denominator].eq(0)).any():
            raise FederationPacketError(
                f"Column {metric} must be missing if and only if {denominator} is zero"
            )

    invalid_classes = sorted(set(validated["classification"]) - ALLOWED_CLASSIFICATIONS)
    if invalid_classes:
        raise FederationPacketError(
            f"Packet contains unsupported robustness classifications: {invalid_classes}"
        )
    if ((validated["icc"] < -1.0) | (validated["icc"] > 1.0)).any():
        raise FederationPacketError("Column icc must be within [-1, 1]")
    ci_low_present = validated["icc_ci_low"].notna()
    ci_high_present = validated["icc_ci_high"].notna()
    if (ci_low_present != ci_high_present).any():
        raise FederationPacketError(
            "ICC confidence limits must be both finite or both missing"
        )
    ci_present = ci_low_present & ci_high_present
    if (
        ((validated.loc[ci_present, "icc_ci_low"] < -1.0) | (validated.loc[ci_present, "icc_ci_low"] > 1.0)).any()
        or ((validated.loc[ci_present, "icc_ci_high"] < -1.0) | (validated.loc[ci_present, "icc_ci_high"] > 1.0)).any()
    ):
        raise FederationPacketError("Finite ICC confidence limits must be within [-1, 1]")
    if (
        (validated.loc[ci_present, "icc_ci_low"] > validated.loc[ci_present, "icc"])
        | (validated.loc[ci_present, "icc"] > validated.loc[ci_present, "icc_ci_high"])
    ).any():
        raise FederationPacketError("ICC confidence limits must satisfy low <= ICC <= high")
    if (validated[["cov_percent", "qcd"]] < 0).any().any():
        raise FederationPacketError("cov_percent and qcd must be nonnegative")

    incomplete_cov = validated["n_subjects_cov"] < validated["n_subjects"]
    if (incomplete_cov & validated["classification"].ne("Not Evaluable")).any():
        raise FederationPacketError(
            "Incomplete CoV denominators require classification='Not Evaluable'"
        )

    icc_for_threshold = validated["icc_ci_low"].where(ci_present, validated["icc"])
    expected_classification = pd.Series("Poor", index=validated.index, dtype="object")
    acceptable = (
        (icc_for_threshold >= PACKET_CLASSIFICATION_RULE["acceptable"]["minimum_icc"])
        & (validated["cov_percent"] <= PACKET_CLASSIFICATION_RULE["acceptable"]["maximum_cov_percent"])
    )
    robust = (
        (icc_for_threshold >= PACKET_CLASSIFICATION_RULE["robust"]["minimum_icc"])
        & (validated["cov_percent"] <= PACKET_CLASSIFICATION_RULE["robust"]["maximum_cov_percent"])
    )
    expected_classification.loc[acceptable] = "Acceptable"
    expected_classification.loc[robust] = "Robust"
    expected_classification.loc[incomplete_cov] = "Not Evaluable"
    classification_mismatch = validated["classification"].ne(expected_classification)
    if classification_mismatch.any():
        first = validated.loc[
            classification_mismatch, [*IDENTITY_COLUMNS, "classification"]
        ].iloc[0]
        expected_value = expected_classification.loc[first.name]
        identity = "/".join(str(first[column]) for column in IDENTITY_COLUMNS)
        raise FederationPacketError(
            "Robustness classification does not match the hash-bound ICC/CI and CoV "
            f"thresholds for {identity}: expected={expected_value}, "
            f"actual={first['classification']}"
        )

    audit = _audit_forbidden_content(validated)
    if any(audit.values()):
        raise FederationPacketError(f"Forbidden content detected: {audit}")
    return validated.sort_values(list(IDENTITY_COLUMNS), kind="mergesort").reset_index(drop=True)


def adapt_native_robustness_output(
    native_metrics: pd.DataFrame,
    expected_inventory: pd.DataFrame,
    *,
    minimum_subjects: int = 5,
) -> pd.DataFrame:
    """Strictly convert native ``radiomics_robustness`` summary rows to schema v3.

    Column aliases are deliberately unsupported. Body region, feature family,
    and image type come only from the explicit expected inventory, never from
    parsing or guessing feature names.
    """

    missing = [
        column
        for column in NATIVE_ROBUSTNESS_REQUIRED_COLUMNS
        if column not in native_metrics.columns
    ]
    if missing:
        raise FederationPacketError(
            "Native robustness table is missing exact required columns; aliases are not "
            f"guessed: missing={missing}"
        )
    audit = _audit_forbidden_content(native_metrics)
    if any(audit.values()):
        raise FederationPacketError(f"Forbidden native-output content detected: {audit}")
    inventory = _validate_inventory(expected_inventory)
    adapter_keys = ["segmentation_source", "roi_name", "feature_name"]
    if inventory.duplicated(adapter_keys).any():
        raise FederationPacketError(
            "Expected inventory is ambiguous for native segmentation-source/structure/feature "
            "keys across body regions"
        )

    source = native_metrics.loc[:, NATIVE_ROBUSTNESS_REQUIRED_COLUMNS].copy()
    source["structure"] = source["structure"].astype(str).str.strip()
    source["segmentation_source"] = source["segmentation_source"].astype(str).str.strip()
    source["feature_name"] = source["feature_name"].astype(str).str.strip()
    native_keys = ["segmentation_source", "structure", "feature_name"]
    if source.duplicated(native_keys).any():
        raise FederationPacketError(
            "Native robustness table contains duplicate segmentation-source/structure/feature rows"
        )
    labels = source["robustness_label"]
    if labels.isna().any() or not set(labels.astype(str)).issubset(NATIVE_CLASSIFICATION_MAP):
        invalid = sorted(set(labels.dropna().astype(str)) - set(NATIVE_CLASSIFICATION_MAP))
        raise FederationPacketError(
            "Native robustness_label must use exact lowercase native values; "
            f"invalid={invalid}"
        )
    source = source.rename(columns={"structure": "roi_name"})
    observed_keys = set(map(tuple, source[adapter_keys].to_numpy()))
    expected_keys = set(map(tuple, inventory[adapter_keys].to_numpy()))
    if observed_keys != expected_keys:
        missing_keys = sorted(expected_keys - observed_keys)
        extra_keys = sorted(observed_keys - expected_keys)
        raise FederationPacketError(
            "Native output does not exactly match the expected feature/ROI inventory; "
            f"missing={missing_keys}, extra={extra_keys}"
        )
    merged = source.merge(
        inventory,
        on=adapter_keys,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    converted = pd.DataFrame(
        {
            "feature_name": merged["feature_name"],
            "roi_name": merged["roi_name"],
            "body_region": merged["body_region"],
            "segmentation_source": merged["segmentation_source"],
            "n_subjects": merged["n_subjects"],
            "n_raters": merged["n_perturbations"],
            "n_subjects_cov": merged["n_subjects_cov"],
            "n_subjects_qcd": merged["n_subjects_qcd"],
            "icc": merged["icc"],
            "icc_ci_low": merged["icc_ci95_low"],
            "icc_ci_high": merged["icc_ci95_high"],
            "cov_percent": merged["cov_pct"],
            "qcd": merged["qcd"],
            "classification": merged["robustness_label"].map(NATIVE_CLASSIFICATION_MAP),
            "feature_family": merged["feature_family"],
            "image_type": merged["image_type"],
        },
        columns=PACKET_COLUMNS,
    )
    return _validate_metrics(converted, int(minimum_subjects))


def _expected_contract_sha256(
    *,
    contract_id: str,
    minimum_subjects: int,
    processing_config_sha256: str,
    source_artifact_kind: str,
    source_artifact_sha256: str,
    expected_rtpipeline_version: str,
    expected_feature_roi_inventory_sha256: str,
) -> str:
    return packet_contract_sha256(
        contract_id,
        minimum_subjects,
        processing_config_sha256=processing_config_sha256,
        source_artifact_kind=source_artifact_kind,
        source_artifact_sha256=source_artifact_sha256,
        expected_rtpipeline_version=expected_rtpipeline_version,
        expected_feature_roi_inventory_sha256=expected_feature_roi_inventory_sha256,
    )


def _new_staging_directory(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )


def _publish_staged_directory(staging: Path, destination: Path, *, force: bool) -> None:
    if destination.is_symlink():
        raise FederationPacketError(f"Output destination must not be a symlink: {destination}")
    if destination.exists() and not destination.is_dir():
        raise FederationPacketError(f"Output destination is not a directory: {destination}")
    if destination.exists() and not force:
        raise FederationPacketError(f"Output already exists at {destination}")
    if not destination.exists():
        os.replace(staging, destination)
        return

    backup = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.backup-", dir=destination.parent)
    )
    backup.rmdir()
    os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        os.replace(backup, destination)
        raise
    else:
        shutil.rmtree(backup)


def export_site_packet(
    metrics: pd.DataFrame,
    output_dir: str | Path,
    *,
    node_id: str,
    contract_id: str,
    contract_sha256: str,
    processing_config_sha256: str,
    source_artifact_kind: str,
    source_artifact_sha256: str,
    expected_rtpipeline_version: str,
    expected_feature_roi_inventory_sha256: str,
    minimum_subjects: int = 5,
    force: bool = False,
) -> dict[str, object]:
    """Publish one deterministic aggregate-only schema-v3 site packet atomically."""

    if not NODE_ID_RE.fullmatch(node_id):
        raise FederationPacketError(
            "node_id must be opaque and match node-[A-Za-z0-9_-]{3,64}"
        )
    expected_contract_sha256 = _expected_contract_sha256(
        contract_id=contract_id,
        minimum_subjects=minimum_subjects,
        processing_config_sha256=processing_config_sha256,
        source_artifact_kind=source_artifact_kind,
        source_artifact_sha256=source_artifact_sha256,
        expected_rtpipeline_version=expected_rtpipeline_version,
        expected_feature_roi_inventory_sha256=expected_feature_roi_inventory_sha256,
    )
    if contract_sha256 != expected_contract_sha256:
        raise FederationPacketError(
            "Supplied contract_sha256 does not match the canonical contract document"
        )
    if __version__ != expected_rtpipeline_version:
        raise FederationPacketError(
            "Running RTpipeline version does not match contract: "
            f"expected={expected_rtpipeline_version}, actual={__version__}"
        )

    validated = _validate_metrics(metrics, int(minimum_subjects))
    observed_inventory_sha256 = feature_roi_inventory_sha256(
        _inventory_from_metrics(validated)
    )
    if observed_inventory_sha256 != expected_feature_roi_inventory_sha256:
        raise FederationPacketError(
            "Observed feature/ROI inventory does not match contract expected inventory"
        )

    destination = Path(output_dir)
    if destination.exists() and not force:
        raise FederationPacketError(f"Packet already exists at {destination}")
    staging = _new_staging_directory(destination)
    try:
        metrics_path = staging / METRICS_FILENAME
        manifest_path = staging / MANIFEST_FILENAME
        _write_deterministic_csv_gz(validated, metrics_path)
        manifest_without_audit: dict[str, object] = {
            "schema_version": PACKET_SCHEMA_VERSION,
            "schema_sha256": packet_schema_sha256(),
            "contract_id": contract_id,
            "contract_sha256": expected_contract_sha256,
            "compatibility_policy": EXACT_COMPATIBILITY_POLICY,
            "processing_config_sha256": processing_config_sha256,
            "source_artifact_kind": source_artifact_kind,
            "source_artifact_sha256": source_artifact_sha256,
            "rtpipeline_version": __version__,
            "node_id": node_id,
            "metrics_file": METRICS_FILENAME,
            "metrics_sha256": _sha256_file(metrics_path),
            "feature_roi_inventory_sha256": observed_inventory_sha256,
            "row_count": int(len(validated)),
            "columns": list(PACKET_COLUMNS),
            "minimum_subjects": int(minimum_subjects),
            "n_subjects_min": int(validated["n_subjects"].min()),
            "n_subjects_max": int(validated["n_subjects"].max()),
            "body_regions": sorted(validated["body_region"].unique().tolist()),
            "roi_count": int(validated["roi_name"].nunique()),
            "feature_count": int(validated["feature_name"].nunique()),
            "privacy_scope": PRIVACY_SCOPE,
        }
        audit = _audit_forbidden_content(validated, manifest_without_audit)
        if any(audit.values()):
            raise FederationPacketError(f"Forbidden manifest content detected: {audit}")
        manifest = {**manifest_without_audit, "content_audit": audit}
        _write_json(manifest_path, manifest)
        _publish_staged_directory(staging, destination, force=force)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def export_native_site_packet(
    native_metrics: pd.DataFrame,
    expected_inventory: pd.DataFrame,
    output_dir: str | Path,
    *,
    node_id: str,
    contract_id: str,
    contract_sha256: str,
    processing_config_sha256: str,
    source_artifact_kind: str,
    source_artifact_sha256: str,
    expected_rtpipeline_version: str,
    expected_feature_roi_inventory_sha256: str,
    minimum_subjects: int = 5,
    force: bool = False,
) -> dict[str, object]:
    """Strictly adapt native robustness output and atomically publish a packet."""

    adapted = adapt_native_robustness_output(
        native_metrics,
        expected_inventory,
        minimum_subjects=minimum_subjects,
    )
    actual_expected_digest = feature_roi_inventory_sha256(expected_inventory)
    if expected_feature_roi_inventory_sha256 != actual_expected_digest:
        raise FederationPacketError(
            "Supplied expected inventory digest does not match the explicit inventory table"
        )
    return export_site_packet(
        adapted,
        output_dir,
        node_id=node_id,
        contract_id=contract_id,
        contract_sha256=contract_sha256,
        processing_config_sha256=processing_config_sha256,
        source_artifact_kind=source_artifact_kind,
        source_artifact_sha256=source_artifact_sha256,
        expected_rtpipeline_version=expected_rtpipeline_version,
        expected_feature_roi_inventory_sha256=expected_feature_roi_inventory_sha256,
        minimum_subjects=minimum_subjects,
        force=force,
    )


def validate_site_packet(
    packet_dir: str | Path,
    *,
    contract_id: str,
    contract_sha256: str,
    processing_config_sha256: str,
    source_artifact_kind: str,
    source_artifact_sha256: str,
    expected_rtpipeline_version: str,
    expected_feature_roi_inventory_sha256: str,
    minimum_subjects: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Validate one packet against the coordinator's explicit exact-match policy."""

    expected_contract_sha256 = _expected_contract_sha256(
        contract_id=contract_id,
        minimum_subjects=minimum_subjects,
        processing_config_sha256=processing_config_sha256,
        source_artifact_kind=source_artifact_kind,
        source_artifact_sha256=source_artifact_sha256,
        expected_rtpipeline_version=expected_rtpipeline_version,
        expected_feature_roi_inventory_sha256=expected_feature_roi_inventory_sha256,
    )
    if contract_sha256 != expected_contract_sha256:
        raise FederationPacketError(
            "Coordinator contract_sha256 does not match the canonical contract document"
        )

    packet_path = Path(packet_dir)
    if packet_path.is_symlink() or not packet_path.is_dir():
        raise FederationPacketError(f"Packet path must be a real directory: {packet_path}")
    entries = list(packet_path.iterdir())
    if any(entry.is_symlink() for entry in entries):
        raise FederationPacketError("Packet directories must not contain symlinks")
    actual_files = {entry.name for entry in entries if entry.is_file()}
    unexpected_entries = sorted(
        entry.name
        for entry in entries
        if not entry.is_file() or entry.name not in PACKET_FILENAMES
    )
    if actual_files != PACKET_FILENAMES or unexpected_entries:
        raise FederationPacketError(
            "Packet directory must contain exactly manifest.json and metrics.csv.gz; "
            f"files={sorted(actual_files)}, unexpected={unexpected_entries}"
        )

    manifest_path = packet_path / MANIFEST_FILENAME
    manifest = _read_json_object(manifest_path)
    actual_manifest_keys = set(manifest)
    if actual_manifest_keys != MANIFEST_KEYS:
        raise FederationPacketError(
            "Manifest keys must exactly match the packet contract; "
            f"missing={sorted(MANIFEST_KEYS - actual_manifest_keys)}, "
            f"extra={sorted(actual_manifest_keys - MANIFEST_KEYS)}"
        )

    integer_fields = (
        "schema_version",
        "row_count",
        "minimum_subjects",
        "n_subjects_min",
        "n_subjects_max",
        "roi_count",
        "feature_count",
    )
    if any(type(manifest[field]) is not int for field in integer_fields):
        raise FederationPacketError("Manifest count and version fields must be JSON integers")
    if manifest["schema_version"] != PACKET_SCHEMA_VERSION:
        raise FederationPacketError("Unsupported packet schema version")
    if manifest["schema_sha256"] != packet_schema_sha256():
        raise FederationPacketError("Packet schema hash mismatch")
    if manifest["contract_id"] != contract_id:
        raise FederationPacketError("Packet contract_id does not match coordinator contract")
    if manifest["compatibility_policy"] != EXACT_COMPATIBILITY_POLICY:
        raise FederationPacketError("Packet compatibility policy is not the exact-match policy")
    compatibility_expectations = {
        "processing_config_sha256": processing_config_sha256,
        "source_artifact_kind": source_artifact_kind,
        "source_artifact_sha256": source_artifact_sha256,
        "rtpipeline_version": expected_rtpipeline_version,
        "feature_roi_inventory_sha256": expected_feature_roi_inventory_sha256,
    }
    for field, expected in compatibility_expectations.items():
        if manifest[field] != expected:
            raise FederationPacketError(
                f"Packet compatibility mismatch for {field}: expected={expected}, "
                f"actual={manifest[field]}"
            )
    if manifest["contract_sha256"] != expected_contract_sha256:
        raise FederationPacketError("Packet contract hash mismatch")
    if manifest["minimum_subjects"] != int(minimum_subjects):
        raise FederationPacketError("Packet minimum_subjects does not match coordinator policy")
    if manifest["columns"] != list(PACKET_COLUMNS):
        raise FederationPacketError("Packet column contract mismatch")
    if not NODE_ID_RE.fullmatch(str(manifest["node_id"])):
        raise FederationPacketError("Packet node_id is not a valid coded node label")
    if not VERSION_RE.fullmatch(str(manifest["rtpipeline_version"])):
        raise FederationPacketError("Packet rtpipeline_version is invalid")
    if manifest["privacy_scope"] != PRIVACY_SCOPE:
        raise FederationPacketError("Packet privacy_scope is not canonical")
    for field in (
        "schema_sha256",
        "contract_sha256",
        "metrics_sha256",
        "feature_roi_inventory_sha256",
        "processing_config_sha256",
        "source_artifact_sha256",
    ):
        _require_sha256(f"manifest field {field}", str(manifest[field]))

    declared_audit = manifest["content_audit"]
    if not isinstance(declared_audit, dict) or set(declared_audit) != set(AUDIT_KEYS):
        raise FederationPacketError("Manifest content_audit keys do not match the contract")
    if any(type(value) is not int or value < 0 for value in declared_audit.values()):
        raise FederationPacketError("Manifest content_audit values must be nonnegative integers")

    if manifest["metrics_file"] != METRICS_FILENAME:
        raise FederationPacketError("Manifest metrics_file is not the contracted local filename")
    metrics_path = packet_path / METRICS_FILENAME
    if _sha256_file(metrics_path) != manifest["metrics_sha256"]:
        raise FederationPacketError("Packet metrics hash mismatch")

    frame = _read_table(metrics_path)
    validated = _validate_metrics(frame, int(minimum_subjects))
    expected_summary = {
        "row_count": int(len(validated)),
        "n_subjects_min": int(validated["n_subjects"].min()),
        "n_subjects_max": int(validated["n_subjects"].max()),
        "body_regions": sorted(validated["body_region"].unique().tolist()),
        "roi_count": int(validated["roi_name"].nunique()),
        "feature_count": int(validated["feature_name"].nunique()),
    }
    for field, expected in expected_summary.items():
        if manifest[field] != expected:
            raise FederationPacketError(f"Packet manifest summary mismatch for {field}")
    observed_inventory_sha256 = feature_roi_inventory_sha256(
        _inventory_from_metrics(validated)
    )
    if manifest["feature_roi_inventory_sha256"] != observed_inventory_sha256:
        raise FederationPacketError("Packet feature/ROI inventory hash mismatch")

    manifest_without_audit = dict(manifest)
    manifest_without_audit.pop("content_audit")
    recomputed_audit = _audit_forbidden_content(validated, manifest_without_audit)
    if declared_audit != recomputed_audit:
        raise FederationPacketError(
            "Manifest content_audit does not match the independently recomputed audit"
        )
    if any(recomputed_audit.values()):
        raise FederationPacketError(f"Forbidden packet content detected: {recomputed_audit}")
    return manifest, validated


def aggregate_site_packets(
    packet_dirs: Iterable[str | Path],
    output_dir: str | Path,
    *,
    contract_id: str,
    contract_sha256: str,
    processing_config_sha256: str,
    source_artifact_kind: str,
    source_artifact_sha256: str,
    expected_rtpipeline_version: str,
    expected_feature_roi_inventory_sha256: str,
    minimum_subjects: int,
    force: bool = False,
) -> dict[str, object]:
    """Validate and atomically concatenate compatible site rows with node IDs.

    This function computes no pooled cross-site estimator or meta-analysis.
    """

    packets = [Path(path) for path in packet_dirs]
    if len(packets) < 2:
        raise FederationPacketError("At least two site packets are required")
    compatibility_kwargs = {
        "contract_id": contract_id,
        "contract_sha256": contract_sha256,
        "processing_config_sha256": processing_config_sha256,
        "source_artifact_kind": source_artifact_kind,
        "source_artifact_sha256": source_artifact_sha256,
        "expected_rtpipeline_version": expected_rtpipeline_version,
        "expected_feature_roi_inventory_sha256": expected_feature_roi_inventory_sha256,
        "minimum_subjects": minimum_subjects,
    }
    validated_packets: list[tuple[Path, dict[str, object], pd.DataFrame]] = []
    for packet in packets:
        manifest, frame = validate_site_packet(packet, **compatibility_kwargs)
        validated_packets.append((packet, manifest, frame))

    node_ids = [str(manifest["node_id"]) for _, manifest, _ in validated_packets]
    if len(set(node_ids)) != len(node_ids):
        raise FederationPacketError("Duplicate node_id across packets")
    combined_parts = []
    inventory_rows = []
    for packet, manifest, frame in validated_packets:
        node_id = str(manifest["node_id"])
        part = frame.copy()
        part.insert(0, "node_id", node_id)
        combined_parts.append(part)
        inventory_rows.append(
            {
                "node_id": node_id,
                "row_count": int(manifest["row_count"]),
                "packet_bytes": int(
                    (packet / METRICS_FILENAME).stat().st_size
                    + (packet / MANIFEST_FILENAME).stat().st_size
                ),
                "manifest_sha256": _sha256_file(packet / MANIFEST_FILENAME),
                "metrics_sha256": str(manifest["metrics_sha256"]),
                "feature_roi_inventory_sha256": str(
                    manifest["feature_roi_inventory_sha256"]
                ),
                "processing_config_sha256": str(manifest["processing_config_sha256"]),
                "source_artifact_kind": str(manifest["source_artifact_kind"]),
                "source_artifact_sha256": str(manifest["source_artifact_sha256"]),
                "rtpipeline_version": str(manifest["rtpipeline_version"]),
                "n_subjects_min": int(manifest["n_subjects_min"]),
                "n_subjects_max": int(manifest["n_subjects_max"]),
                "roi_count": int(manifest["roi_count"]),
                "feature_count": int(manifest["feature_count"]),
            }
        )

    combined = pd.concat(combined_parts, ignore_index=True).sort_values(
        ["node_id", *IDENTITY_COLUMNS], kind="mergesort"
    )
    inventory = pd.DataFrame(inventory_rows).sort_values("node_id", kind="mergesort")
    destination = Path(output_dir)
    if destination.exists() and not force:
        raise FederationPacketError(f"Aggregate output already exists at {destination}")
    staging = _new_staging_directory(destination)
    try:
        combined_path = staging / "combined_metrics.csv.gz"
        inventory_path = staging / "packet_inventory.csv"
        manifest_path = staging / "aggregate_manifest.json"
        _write_deterministic_csv_gz(combined.reset_index(drop=True), combined_path)
        inventory.to_csv(inventory_path, index=False, lineterminator="\n")
        aggregate_manifest: dict[str, object] = {
            "schema_version": PACKET_SCHEMA_VERSION,
            "schema_sha256": packet_schema_sha256(),
            "contract_id": contract_id,
            "contract_sha256": contract_sha256,
            "compatibility_policy": EXACT_COMPATIBILITY_POLICY,
            "processing_config_sha256": processing_config_sha256,
            "source_artifact_kind": source_artifact_kind,
            "source_artifact_sha256": source_artifact_sha256,
            "rtpipeline_version": expected_rtpipeline_version,
            "expected_feature_roi_inventory_sha256": expected_feature_roi_inventory_sha256,
            "minimum_subjects": int(minimum_subjects),
            "node_count": len(node_ids),
            "node_ids": sorted(node_ids),
            "row_count": int(len(combined)),
            "combined_metrics_file": combined_path.name,
            "combined_metrics_sha256": _sha256_file(combined_path),
            "packet_inventory_file": inventory_path.name,
            "packet_inventory_sha256": _sha256_file(inventory_path),
            "source_packet_metrics_sha256": {
                str(manifest["node_id"]): str(manifest["metrics_sha256"])
                for _, manifest, _ in sorted(
                    validated_packets, key=lambda item: str(item[1]["node_id"])
                )
            },
            "source_packet_manifests_sha256": {
                str(manifest["node_id"]): _sha256_file(packet / MANIFEST_FILENAME)
                for packet, manifest, _ in sorted(
                    validated_packets, key=lambda item: str(item[1]["node_id"])
                )
            },
            "privacy_scope": PRIVACY_SCOPE,
        }
        _write_json(manifest_path, aggregate_manifest)
        _publish_staged_directory(staging, destination, force=force)
        return aggregate_manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _add_compatibility_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--processing-config-sha256", required=True)
    parser.add_argument(
        "--source-artifact-kind",
        required=True,
        choices=sorted(ALLOWED_ARTIFACT_KINDS),
    )
    parser.add_argument("--source-artifact-sha256", required=True)
    parser.add_argument("--rtpipeline-version", required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rtpipeline federation",
        description="Create and combine distributed aggregate radiomics reliability packets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser(
        "config-digest", help="Hash a normalized JSON processing configuration"
    )
    config_parser.add_argument("--input", required=True, help="JSON object file")

    inventory_parser = subparsers.add_parser(
        "inventory-digest", help="Validate and hash an expected feature/ROI inventory"
    )
    inventory_parser.add_argument("--input", required=True, help="Inventory CSV or Parquet")

    contract_parser = subparsers.add_parser(
        "contract", help="Print the canonical schema-v3 contract and digest"
    )
    contract_parser.add_argument("--contract-id", required=True)
    contract_parser.add_argument("--minimum-subjects", type=int, default=5)
    _add_compatibility_arguments(contract_parser)

    export_parser = subparsers.add_parser("export", help="Create one site packet")
    export_parser.add_argument("--input", required=True, help="Schema-v3 packet table CSV or Parquet")
    export_parser.add_argument("--output", required=True, help="Output packet directory")
    export_parser.add_argument("--node-id", required=True, help="Coded node ID such as node-a13f")
    export_parser.add_argument("--contract-id", required=True)
    export_parser.add_argument("--contract-sha256", required=True)
    export_parser.add_argument("--minimum-subjects", type=int, default=5)
    export_parser.add_argument("--force", action="store_true")
    _add_compatibility_arguments(export_parser)

    native_parser = subparsers.add_parser(
        "export-native",
        help="Strictly adapt native radiomics_robustness output and create a packet",
    )
    native_parser.add_argument("--input", required=True, help="Native summary CSV or Parquet")
    native_parser.add_argument("--inventory", required=True, help="Exact expected inventory CSV or Parquet")
    native_parser.add_argument("--output", required=True, help="Output packet directory")
    native_parser.add_argument("--node-id", required=True)
    native_parser.add_argument("--contract-id", required=True)
    native_parser.add_argument("--contract-sha256", required=True)
    native_parser.add_argument("--minimum-subjects", type=int, default=5)
    native_parser.add_argument("--force", action="store_true")
    _add_compatibility_arguments(native_parser)

    aggregate_parser = subparsers.add_parser(
        "aggregate", help="Validate and combine compatible site packets"
    )
    aggregate_parser.add_argument("--packet", action="append", required=True)
    aggregate_parser.add_argument("--output", required=True)
    aggregate_parser.add_argument("--contract-id", required=True)
    aggregate_parser.add_argument("--contract-sha256", required=True)
    aggregate_parser.add_argument("--minimum-subjects", type=int, default=5)
    aggregate_parser.add_argument("--force", action="store_true")
    _add_compatibility_arguments(aggregate_parser)
    return parser


def _compatibility_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "processing_config_sha256": args.processing_config_sha256,
        "source_artifact_kind": args.source_artifact_kind,
        "source_artifact_sha256": args.source_artifact_sha256,
        "expected_rtpipeline_version": args.rtpipeline_version,
        "expected_feature_roi_inventory_sha256": args.expected_inventory_sha256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "config-digest":
            config = _read_json_object(Path(args.input))
            manifest: dict[str, object] = {
                "processing_config_sha256": normalized_processing_config_sha256(config)
            }
        elif args.command == "inventory-digest":
            inventory = _read_table(Path(args.input))
            manifest = {
                "expected_feature_roi_inventory_sha256": feature_roi_inventory_sha256(inventory)
            }
        elif args.command == "contract":
            compatibility = _compatibility_kwargs(args)
            manifest = packet_contract_document(
                args.contract_id,
                args.minimum_subjects,
                **compatibility,
            )
            manifest["contract_sha256"] = packet_contract_sha256(
                args.contract_id,
                args.minimum_subjects,
                **compatibility,
            )
        elif args.command in {"export", "export-native"}:
            compatibility = _compatibility_kwargs(args)
            common = {
                "node_id": args.node_id,
                "contract_id": args.contract_id,
                "contract_sha256": args.contract_sha256,
                "minimum_subjects": args.minimum_subjects,
                "force": args.force,
                **compatibility,
            }
            frame = _read_table(Path(args.input))
            if args.command == "export-native":
                inventory = _read_table(Path(args.inventory))
                manifest = export_native_site_packet(
                    frame, inventory, args.output, **common
                )
            else:
                manifest = export_site_packet(frame, args.output, **common)
        else:
            manifest = aggregate_site_packets(
                args.packet,
                args.output,
                contract_id=args.contract_id,
                contract_sha256=args.contract_sha256,
                minimum_subjects=args.minimum_subjects,
                force=args.force,
                **_compatibility_kwargs(args),
            )
    except (FederationPacketError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "EXACT_COMPATIBILITY_POLICY",
    "FederationPacketError",
    "INVENTORY_COLUMNS",
    "NATIVE_ROBUSTNESS_REQUIRED_COLUMNS",
    "PACKET_COLUMNS",
    "PACKET_SCHEMA_VERSION",
    "adapt_native_robustness_output",
    "aggregate_site_packets",
    "export_native_site_packet",
    "export_site_packet",
    "feature_roi_inventory_sha256",
    "normalized_processing_config_sha256",
    "packet_contract_document",
    "packet_contract_sha256",
    "packet_schema_sha256",
    "validate_site_packet",
]


if __name__ == "__main__":
    raise SystemExit(main())
