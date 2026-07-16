import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from rtpipeline.federation import (
    FederationPacketError,
    PACKET_COLUMNS,
    aggregate_site_packets,
    export_site_packet,
    main,
    packet_contract_sha256,
    validate_site_packet,
)


CONTRACT_ID = "ntcv-icc-v1"
MINIMUM_SUBJECTS = 5
CONTRACT_SHA256 = packet_contract_sha256(CONTRACT_ID, MINIMUM_SUBJECTS)


def _export(metrics: pd.DataFrame, output: Path, node_id: str = "node-a01") -> dict[str, object]:
    return export_site_packet(
        metrics,
        output,
        node_id=node_id,
        contract_id=CONTRACT_ID,
        contract_sha256=CONTRACT_SHA256,
        minimum_subjects=MINIMUM_SUBJECTS,
    )


def _validate(packet: Path) -> tuple[dict[str, object], pd.DataFrame]:
    return validate_site_packet(
        packet,
        contract_id=CONTRACT_ID,
        contract_sha256=CONTRACT_SHA256,
        minimum_subjects=MINIMUM_SUBJECTS,
    )


def _aggregate(packets: list[Path], output: Path) -> dict[str, object]:
    return aggregate_site_packets(
        packets,
        output,
        contract_id=CONTRACT_ID,
        contract_sha256=CONTRACT_SHA256,
        minimum_subjects=MINIMUM_SUBJECTS,
    )


def _metrics(offset: float = 0.0) -> pd.DataFrame:
    rows = [
        {
            "feature_name": "original_firstorder_Mean",
            "roi_name": "lung",
            "body_region": "Thorax",
            "n_subjects": 12,
            "n_raters": 81,
            "icc": 0.91 + offset,
            "icc_ci_low": 0.88 + offset,
            "icc_ci_high": 0.94 + offset,
            "cov_percent": 4.1,
            "qcd": 0.03,
            "classification": "Robust",
            "feature_family": "firstorder",
            "image_type": "original",
        },
        {
            "feature_name": "original_glcm_Contrast",
            "roi_name": "lung",
            "body_region": "Thorax",
            "n_subjects": 12,
            "n_raters": 81,
            "icc": 0.82 + offset,
            "icc_ci_low": 0.78 + offset,
            "icc_ci_high": 0.86 + offset,
            "cov_percent": 8.2,
            "qcd": 0.06,
            "classification": "Acceptable",
            "feature_family": "glcm",
            "image_type": "original",
        },
    ]
    return pd.DataFrame(rows, columns=PACKET_COLUMNS)


def test_export_and_validate_round_trip_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = _export(_metrics(), first)
    _export(_metrics(), second)

    validated_manifest, validated = _validate(first)

    assert manifest == validated_manifest
    assert validated["feature_name"].tolist() == sorted(validated["feature_name"])
    assert (first / "metrics.csv.gz").read_bytes() == (second / "metrics.csv.gz").read_bytes()
    assert validated["icc"].tolist() == _metrics().sort_values(
        ["body_region", "roi_name", "feature_name"]
    )["icc"].tolist()
    assert manifest["content_audit"] == {
        "forbidden_columns": 0,
        "absolute_path_values": 0,
        "uri_values": 0,
        "dicom_uid_values": 0,
        "date_values": 0,
        "direct_identifier_values": 0,
        "hostname_values": 0,
    }


def test_export_fails_closed_on_extra_identifier_column(tmp_path: Path) -> None:
    frame = _metrics().assign(patient_id=["P1", "P2"])
    with pytest.raises(FederationPacketError, match="extra=.*patient_id"):
        _export(frame, tmp_path / "packet")


@pytest.mark.parametrize(
    "value",
    [
        "/private/site/data",
        r"C:\\private\\site\\data",
        r"\\server\share\data",
        "s3://private-bucket/data",
        "1.2.840.10008.1.2.1",
        "2.25.12345678901234567890",
        "2026-07-16",
        "20260716",
        "P000001",
        "node01.institution.local",
    ],
)
def test_export_rejects_forbidden_string_values(tmp_path: Path, value: str) -> None:
    frame = _metrics()
    frame.loc[0, "roi_name"] = value
    with pytest.raises(FederationPacketError, match="Forbidden content"):
        _export(frame, tmp_path / "packet")


def test_export_enforces_minimum_subjects(tmp_path: Path) -> None:
    frame = _metrics()
    frame.loc[0, "n_subjects"] = 4
    with pytest.raises(FederationPacketError, match="below minimum_subjects"):
        export_site_packet(
            frame,
            tmp_path / "packet",
            node_id="node-a01",
            contract_id=CONTRACT_ID,
            contract_sha256=CONTRACT_SHA256,
            minimum_subjects=5,
        )


def test_aggregate_validates_contract_and_preserves_only_aggregate_rows(tmp_path: Path) -> None:
    packet_one = tmp_path / "one"
    packet_two = tmp_path / "two"
    _export(_metrics(), packet_one, "node-a01")
    _export(_metrics(-0.02), packet_two, "node-b02")

    aggregate = _aggregate([packet_two, packet_one], tmp_path / "aggregate")

    assert aggregate["node_count"] == 2
    assert aggregate["row_count"] == 4
    with gzip.open(tmp_path / "aggregate" / "combined_metrics.csv.gz", "rt") as handle:
        combined = pd.read_csv(handle)
    assert combined.columns.tolist() == ["node_id", *PACKET_COLUMNS]
    assert combined["node_id"].tolist() == ["node-a01", "node-a01", "node-b02", "node-b02"]
    assert not any("patient" in column.lower() for column in combined.columns)


def test_aggregate_rejects_contract_mismatch(tmp_path: Path) -> None:
    packet_one = tmp_path / "one"
    packet_two = tmp_path / "two"
    _export(_metrics(), packet_one, "node-a01")
    other_sha = packet_contract_sha256("other-contract", MINIMUM_SUBJECTS)
    export_site_packet(
        _metrics(),
        packet_two,
        node_id="node-b02",
        contract_id="other-contract",
        contract_sha256=other_sha,
        minimum_subjects=MINIMUM_SUBJECTS,
    )
    with pytest.raises(FederationPacketError, match="contract_id"):
        _aggregate([packet_one, packet_two], tmp_path / "aggregate")


def test_cli_export_and_tamper_detection(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    _metrics().to_csv(source, index=False)
    packet = tmp_path / "packet"
    assert main(
        [
            "export",
            "--input",
            str(source),
            "--output",
            str(packet),
            "--node-id",
            "node-a01",
            "--contract-id",
            CONTRACT_ID,
            "--contract-sha256",
            CONTRACT_SHA256,
        ]
    ) == 0

    manifest_path = packet / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metrics_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(FederationPacketError, match="hash mismatch"):
        _validate(packet)


def test_validator_rejects_extra_manifest_keys_and_files(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    _export(_metrics(), packet)
    manifest_path = packet / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["patient_id"] = "P000001"
    manifest_path.write_text(json.dumps(manifest))
    (packet / "patient_rows.csv").write_text("patient_id\nP000001\n")
    with pytest.raises(FederationPacketError, match="exactly manifest.json"):
        _validate(packet)


def test_validator_rejects_unexpected_symlink(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    _export(_metrics(), packet)
    (packet / "extra-link").symlink_to(packet / "metrics.csv.gz")
    with pytest.raises(FederationPacketError, match="symlinks"):
        _validate(packet)


@pytest.mark.parametrize("audit", [{}, {"forbidden_columns": 0}])
def test_validator_rejects_incomplete_declared_audit(
    tmp_path: Path, audit: dict[str, int]
) -> None:
    packet = tmp_path / "packet"
    _export(_metrics(), packet)
    manifest_path = packet / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["content_audit"] = audit
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(FederationPacketError, match="content_audit keys"):
        _validate(packet)


def test_validator_recomputes_manifest_summaries(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    _export(_metrics(), packet)
    manifest_path = packet / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["roi_count"] = 999
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(FederationPacketError, match="summary mismatch for roi_count"):
        _validate(packet)


def test_coordinator_enforces_contract_hash_and_minimum_subject_floor(tmp_path: Path) -> None:
    low_count = _metrics()
    low_count["n_subjects"] = 2
    low_contract_sha = packet_contract_sha256(CONTRACT_ID, 2)
    packets = []
    for node_id in ("node-a01", "node-b02"):
        packet = tmp_path / node_id
        export_site_packet(
            low_count,
            packet,
            node_id=node_id,
            contract_id=CONTRACT_ID,
            contract_sha256=low_contract_sha,
            minimum_subjects=2,
        )
        packets.append(packet)
    with pytest.raises(FederationPacketError, match="contract hash mismatch"):
        _aggregate(packets, tmp_path / "aggregate")


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("classification", "Unknown", "unsupported robustness"),
        ("icc", float("nan"), "finite numbers"),
        ("n_raters", 1, "n_raters below 2"),
        ("cov_percent", -0.1, "must be nonnegative"),
        ("icc_ci_low", 0.99, "low <= ICC <= high"),
    ],
)
def test_export_rejects_semantically_invalid_metrics(
    tmp_path: Path, column: str, value: object, message: str
) -> None:
    frame = _metrics()
    frame.loc[0, column] = value
    with pytest.raises(FederationPacketError, match=message):
        _export(frame, tmp_path / "packet")


def test_export_rejects_duplicate_feature_identity(tmp_path: Path) -> None:
    frame = pd.concat([_metrics(), _metrics().iloc[[0]]], ignore_index=True)
    with pytest.raises(FederationPacketError, match="duplicate"):
        _export(frame, tmp_path / "packet")


def test_export_rejects_identity_collision_after_whitespace_normalization(
    tmp_path: Path,
) -> None:
    frame = pd.concat([_metrics(), _metrics().iloc[[0]]], ignore_index=True)
    frame.loc[len(frame) - 1, "roi_name"] = " lung "
    with pytest.raises(FederationPacketError, match="duplicate normalized"):
        _export(frame, tmp_path / "packet")


def test_float_columns_round_trip_bitwise(tmp_path: Path) -> None:
    frame = _metrics()
    frame.loc[0, "icc"] = float(pd.Series([0.91]).map(lambda x: x).iloc[0])
    packet = tmp_path / "packet"
    _export(frame, packet)
    _, validated = _validate(packet)
    expected = frame.sort_values(["body_region", "roi_name", "feature_name"])
    for column in ("icc", "icc_ci_low", "icc_ci_high", "cov_percent", "qcd"):
        expected_bits = expected[column].to_numpy(dtype="float64").view("uint64")
        observed_bits = validated[column].to_numpy(dtype="float64").view("uint64")
        assert observed_bits.tolist() == expected_bits.tolist()
