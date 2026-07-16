"""Privacy-bounded site packets for distributed RTpipeline analyses.

The module deliberately handles cohort-level reliability summaries only.  It is
not a federated-learning framework, a secure-aggregation protocol, or a claim
that aggregate data are legally anonymous.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from . import __version__


PACKET_SCHEMA_VERSION = 1
METRICS_FILENAME = "metrics.csv.gz"
MANIFEST_FILENAME = "manifest.json"
PRIVACY_SCOPE = (
    "Aggregate-only content audit; not secure aggregation, differential privacy, "
    "legal anonymity, or federated learning."
)

PACKET_COLUMNS: tuple[str, ...] = (
    "feature_name",
    "roi_name",
    "body_region",
    "n_subjects",
    "n_raters",
    "icc",
    "icc_ci_low",
    "icc_ci_high",
    "cov_percent",
    "qcd",
    "classification",
    "feature_family",
    "image_type",
)

STRING_COLUMNS: tuple[str, ...] = (
    "feature_name",
    "roi_name",
    "body_region",
    "classification",
    "feature_family",
    "image_type",
)

INTEGER_COLUMNS: tuple[str, ...] = ("n_subjects", "n_raters")
FLOAT_COLUMNS: tuple[str, ...] = (
    "icc",
    "icc_ci_low",
    "icc_ci_high",
    "cov_percent",
    "qcd",
)

IDENTITY_COLUMNS: tuple[str, ...] = ("body_region", "roi_name", "feature_name")
ALLOWED_CLASSIFICATIONS = frozenset({"Robust", "Acceptable", "Poor"})
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
        "node_id",
        "rtpipeline_version",
        "metrics_file",
        "metrics_sha256",
        "feature_identity_sha256",
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
DATE_RE = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2}(?:[ T].*)?|\d{8}(?:[ T]?\d{6}(?:\.\d+)?)?)$"
)
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
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def packet_schema_sha256() -> str:
    schema = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "columns": list(PACKET_COLUMNS),
        "string_columns": list(STRING_COLUMNS),
        "integer_columns": list(INTEGER_COLUMNS),
        "float_columns": list(FLOAT_COLUMNS),
        "identity_columns": list(IDENTITY_COLUMNS),
    }
    return _sha256_bytes(_canonical_json_bytes(schema))


def packet_contract_document(contract_id: str, minimum_subjects: int) -> dict[str, object]:
    """Return the canonical, hash-bound packet contract used by every node."""

    if not CONTRACT_ID_RE.fullmatch(contract_id):
        raise FederationPacketError(
            "contract_id must contain only letters, numbers, dot, underscore, or hyphen"
        )
    if int(minimum_subjects) < 2:
        raise FederationPacketError("minimum_subjects must be at least 2")
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
        "semantic_rules": [
            "icc_and_confidence_limits_in_closed_interval_-1_1",
            "icc_ci_low_le_icc_le_icc_ci_high",
            "cov_percent_and_qcd_nonnegative",
            "unique_body_region_roi_name_feature_name_identity",
            "finite_required_numeric_values",
        ],
        "serialization": "deterministic_csv_gzip_float64_17_significant_digits",
        "privacy_scope": PRIVACY_SCOPE,
    }


def packet_contract_sha256(contract_id: str, minimum_subjects: int) -> str:
    """Hash the complete contract, including thresholds and semantic rules."""

    return _sha256_bytes(
        _canonical_json_bytes(packet_contract_document(contract_id, minimum_subjects))
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
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
    """Yield metadata strings not already protected by exact/format checks."""

    exempt_keys = {
        "schema_sha256",
        "contract_sha256",
        "metrics_sha256",
        "feature_identity_sha256",
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
        for raw in frame[column].dropna().astype(str):
            values.append(raw.strip())
    if metadata is not None:
        forbidden_columns += sum(
            _normalized_column_name(key) in FORBIDDEN_COLUMN_NAMES
            for key in metadata
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
    if frame.duplicated(list(IDENTITY_COLUMNS)).any():
        raise FederationPacketError(
            "Metrics contain duplicate body_region/roi_name/feature_name identities"
        )

    validated = frame.copy()
    for column in STRING_COLUMNS:
        if validated[column].isna().any():
            raise FederationPacketError(f"Column {column} contains missing values")
        validated[column] = validated[column].astype(str).str.strip()
        if validated[column].eq("").any():
            raise FederationPacketError(f"Column {column} contains empty values")

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

    if int(validated["n_subjects"].min()) < int(minimum_subjects):
        raise FederationPacketError(
            f"Packet contains n_subjects below minimum_subjects={minimum_subjects}"
        )
    if (validated["n_raters"] < 2).any():
        raise FederationPacketError("Packet contains n_raters below 2")
    invalid_classes = sorted(set(validated["classification"]) - ALLOWED_CLASSIFICATIONS)
    if invalid_classes:
        raise FederationPacketError(
            f"Packet contains unsupported robustness classifications: {invalid_classes}"
        )
    for column in ("icc", "icc_ci_low", "icc_ci_high"):
        if ((validated[column] < -1.0) | (validated[column] > 1.0)).any():
            raise FederationPacketError(f"Column {column} must be within [-1, 1]")
    if (
        (validated["icc_ci_low"] > validated["icc"])
        | (validated["icc"] > validated["icc_ci_high"])
    ).any():
        raise FederationPacketError("ICC confidence limits must satisfy low <= ICC <= high")
    if (validated[["cov_percent", "qcd"]] < 0).any().any():
        raise FederationPacketError("cov_percent and qcd must be nonnegative")

    audit = _audit_forbidden_content(validated)
    if any(audit.values()):
        raise FederationPacketError(f"Forbidden content detected: {audit}")

    return validated.sort_values(list(IDENTITY_COLUMNS), kind="mergesort").reset_index(drop=True)


def _feature_identity_sha256(frame: pd.DataFrame) -> str:
    identity = frame.loc[:, IDENTITY_COLUMNS].sort_values(
        list(IDENTITY_COLUMNS), kind="mergesort"
    )
    lines = (
        identity.astype(str).agg("\t".join, axis=1).str.cat(sep="\n") + "\n"
    ).encode("utf-8")
    return _sha256_bytes(lines)


def export_site_packet(
    metrics: pd.DataFrame,
    output_dir: str | Path,
    *,
    node_id: str,
    contract_id: str,
    contract_sha256: str,
    minimum_subjects: int = 5,
    force: bool = False,
) -> dict[str, object]:
    """Write one deterministic, aggregate-only site packet."""

    if not NODE_ID_RE.fullmatch(node_id):
        raise FederationPacketError(
            "node_id must be opaque and match node-[A-Za-z0-9_-]{3,64}"
        )
    expected_contract_sha256 = packet_contract_sha256(contract_id, minimum_subjects)
    if contract_sha256 != expected_contract_sha256:
        raise FederationPacketError(
            "Supplied contract_sha256 does not match the canonical contract document"
        )

    validated = _validate_metrics(metrics, int(minimum_subjects))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / METRICS_FILENAME
    manifest_path = destination / MANIFEST_FILENAME
    if not force and (metrics_path.exists() or manifest_path.exists()):
        raise FederationPacketError(f"Packet already exists at {destination}")

    _write_deterministic_csv_gz(validated, metrics_path)
    manifest_without_audit: dict[str, object] = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "schema_sha256": packet_schema_sha256(),
        "contract_id": contract_id,
        "contract_sha256": expected_contract_sha256,
        "node_id": node_id,
        "rtpipeline_version": __version__,
        "metrics_file": METRICS_FILENAME,
        "metrics_sha256": _sha256_file(metrics_path),
        "feature_identity_sha256": _feature_identity_sha256(validated),
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
    return manifest


def validate_site_packet(
    packet_dir: str | Path,
    *,
    contract_id: str,
    contract_sha256: str,
    minimum_subjects: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Validate one packet against a coordinator-supplied contract."""

    expected_contract_sha256 = packet_contract_sha256(contract_id, minimum_subjects)
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
        "feature_identity_sha256",
    ):
        if not SHA256_RE.fullmatch(str(manifest[field])):
            raise FederationPacketError(f"Manifest field {field} is not a SHA-256 digest")

    declared_audit = manifest["content_audit"]
    if not isinstance(declared_audit, dict) or set(declared_audit) != set(AUDIT_KEYS):
        raise FederationPacketError("Manifest content_audit keys do not match the contract")
    if any(type(value) is not int or value < 0 for value in declared_audit.values()):
        raise FederationPacketError("Manifest content_audit values must be nonnegative integers")

    metrics_file = manifest["metrics_file"]
    if metrics_file != METRICS_FILENAME:
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
    if manifest["feature_identity_sha256"] != _feature_identity_sha256(validated):
        raise FederationPacketError("Packet feature identity hash mismatch")

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
    minimum_subjects: int,
    force: bool = False,
) -> dict[str, object]:
    """Validate and combine aggregate-only site packets."""

    packets = [Path(path) for path in packet_dirs]
    if len(packets) < 2:
        raise FederationPacketError("At least two site packets are required")

    validated_packets: list[tuple[Path, dict[str, object], pd.DataFrame]] = []
    for packet in packets:
        manifest, frame = validate_site_packet(
            packet,
            contract_id=contract_id,
            contract_sha256=contract_sha256,
            minimum_subjects=minimum_subjects,
        )
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
                "feature_identity_sha256": str(manifest["feature_identity_sha256"]),
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
    destination.mkdir(parents=True, exist_ok=True)
    combined_path = destination / "combined_metrics.csv.gz"
    inventory_path = destination / "packet_inventory.csv"
    manifest_path = destination / "aggregate_manifest.json"
    if not force and any(path.exists() for path in (combined_path, inventory_path, manifest_path)):
        raise FederationPacketError(f"Aggregate output already exists at {destination}")

    _write_deterministic_csv_gz(combined.reset_index(drop=True), combined_path)
    inventory.to_csv(inventory_path, index=False, lineterminator="\n")
    aggregate_manifest: dict[str, object] = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "schema_sha256": packet_schema_sha256(),
        "contract_id": contract_id,
        "contract_sha256": contract_sha256,
        "minimum_subjects": int(minimum_subjects),
        "rtpipeline_version": __version__,
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
    return aggregate_manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rtpipeline federation",
        description="Create and combine aggregate-only distributed-analysis packets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Create one site packet")
    export_parser.add_argument("--input", required=True, help="Contract-shaped CSV or Parquet table")
    export_parser.add_argument("--output", required=True, help="Output packet directory")
    export_parser.add_argument("--node-id", required=True, help="Coded node ID such as node-a13f")
    export_parser.add_argument("--contract-id", required=True, help="Shared analysis contract identifier")
    export_parser.add_argument("--contract-sha256", required=True, help="Frozen contract digest")
    export_parser.add_argument("--minimum-subjects", type=int, default=5)
    export_parser.add_argument("--force", action="store_true")

    aggregate_parser = subparsers.add_parser("aggregate", help="Validate and combine site packets")
    aggregate_parser.add_argument("--packet", action="append", required=True, help="Site packet directory; repeat for each node")
    aggregate_parser.add_argument("--output", required=True, help="Aggregate output directory")
    aggregate_parser.add_argument("--contract-id", required=True, help="Coordinator contract identifier")
    aggregate_parser.add_argument("--contract-sha256", required=True, help="Coordinator contract digest")
    aggregate_parser.add_argument("--minimum-subjects", type=int, default=5)
    aggregate_parser.add_argument("--force", action="store_true")

    contract_parser = subparsers.add_parser(
        "contract", help="Print the canonical contract document and digest"
    )
    contract_parser.add_argument("--contract-id", required=True)
    contract_parser.add_argument("--minimum-subjects", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "contract":
            manifest = packet_contract_document(args.contract_id, args.minimum_subjects)
            manifest["contract_sha256"] = packet_contract_sha256(
                args.contract_id, args.minimum_subjects
            )
        elif args.command == "export":
            frame = _read_table(Path(args.input))
            manifest = export_site_packet(
                frame,
                args.output,
                node_id=args.node_id,
                contract_id=args.contract_id,
                contract_sha256=args.contract_sha256,
                minimum_subjects=args.minimum_subjects,
                force=args.force,
            )
        else:
            manifest = aggregate_site_packets(
                args.packet,
                args.output,
                contract_id=args.contract_id,
                contract_sha256=args.contract_sha256,
                minimum_subjects=args.minimum_subjects,
                force=args.force,
            )
    except (FederationPacketError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps(manifest, sort_keys=True))
    return 0


__all__ = [
    "FederationPacketError",
    "PACKET_COLUMNS",
    "PACKET_SCHEMA_VERSION",
    "aggregate_site_packets",
    "export_site_packet",
    "packet_contract_document",
    "packet_contract_sha256",
    "packet_schema_sha256",
    "validate_site_packet",
]


if __name__ == "__main__":
    raise SystemExit(main())
