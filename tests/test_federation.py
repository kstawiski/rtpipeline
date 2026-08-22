import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

import rtpipeline.federation as federation
from rtpipeline import __version__
from rtpipeline.cli import console_main
from rtpipeline.federation import (
    EXACT_COMPATIBILITY_POLICY,
    FederationPacketError,
    INVENTORY_COLUMNS,
    PACKET_COLUMNS,
    PACKET_SCHEMA_VERSION,
    adapt_native_robustness_output,
    aggregate_site_packets,
    export_native_site_packet,
    export_site_packet,
    feature_roi_inventory_sha256,
    normalized_processing_config_sha256,
    packet_contract_document,
    packet_contract_sha256,
    validate_site_packet,
)


CONTRACT_ID = "ntcv-icc-v3"
MINIMUM_SUBJECTS = 5
PROCESSING_CONFIG_SHA256 = normalized_processing_config_sha256(
    {"perturbations": {"translations_mm": [-4, 0, 4]}, "radiomics": {"binWidth": 25}}
)
SOURCE_ARTIFACT_KIND = "container_image"
SOURCE_ARTIFACT_SHA256 = "a" * 64


def _inventory() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "body_region": "Thorax",
                "segmentation_source": "Manual",
                "roi_name": "lung",
                "feature_name": "original_firstorder_Mean",
                "feature_family": "firstorder",
                "image_type": "original",
            },
            {
                "body_region": "Thorax",
                "segmentation_source": "Manual",
                "roi_name": "lung",
                "feature_name": "original_glcm_Contrast",
                "feature_family": "glcm",
                "image_type": "original",
            },
        ],
        columns=INVENTORY_COLUMNS,
    )


EXPECTED_INVENTORY_SHA256 = feature_roi_inventory_sha256(_inventory())
COMPATIBILITY = {
    "processing_config_sha256": PROCESSING_CONFIG_SHA256,
    "source_artifact_kind": SOURCE_ARTIFACT_KIND,
    "source_artifact_sha256": SOURCE_ARTIFACT_SHA256,
    "expected_rtpipeline_version": __version__,
    "expected_feature_roi_inventory_sha256": EXPECTED_INVENTORY_SHA256,
}
CONTRACT_SHA256 = packet_contract_sha256(
    CONTRACT_ID,
    MINIMUM_SUBJECTS,
    **COMPATIBILITY,
)


def _metrics(offset: float = 0.0) -> pd.DataFrame:
    rows = [
        {
            "feature_name": "original_firstorder_Mean",
            "roi_name": "lung",
            "body_region": "Thorax",
            "segmentation_source": "Manual",
            "n_subjects": 12,
            "n_raters": 81,
            "n_subjects_cov": 12,
            "n_subjects_qcd": 12,
            "icc": 0.93 + offset,
            "icc_ci_low": 0.91 + offset,
            "icc_ci_high": 0.95 + offset,
            "cov_percent": 4.1,
            "qcd": 0.03,
            "classification": "Robust" if offset >= -0.01 else "Acceptable",
            "feature_family": "firstorder",
            "image_type": "original",
        },
        {
            "feature_name": "original_glcm_Contrast",
            "roi_name": "lung",
            "body_region": "Thorax",
            "segmentation_source": "Manual",
            "n_subjects": 12,
            "n_raters": 81,
            "n_subjects_cov": 12,
            "n_subjects_qcd": 12,
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


def _native(offset: float = 0.0) -> pd.DataFrame:
    packet = _metrics(offset)
    labels = {
        "Robust": "robust",
        "Acceptable": "acceptable",
        "Poor": "poor",
        "Not Evaluable": "not_evaluable",
    }
    return pd.DataFrame(
        {
            "structure": packet["roi_name"],
            "segmentation_source": packet["segmentation_source"],
            "feature_name": packet["feature_name"],
            "n_subjects": packet["n_subjects"],
            "n_perturbations": packet["n_raters"],
            "n_subjects_cov": packet["n_subjects_cov"],
            "n_subjects_qcd": packet["n_subjects_qcd"],
            "icc": packet["icc"],
            "icc_ci95_low": packet["icc_ci_low"],
            "icc_ci95_high": packet["icc_ci_high"],
            "cov_pct": packet["cov_percent"],
            "qcd": packet["qcd"],
            "robustness_label": packet["classification"].map(labels),
            "n_courses": 12,
        }
    )


def _contract_sha(compatibility: dict[str, object]) -> str:
    return packet_contract_sha256(
        CONTRACT_ID,
        MINIMUM_SUBJECTS,
        **compatibility,
    )


def _export(
    metrics: pd.DataFrame,
    output: Path,
    node_id: str = "node-a01",
    *,
    force: bool = False,
    compatibility: dict[str, object] | None = None,
) -> dict[str, object]:
    selected = COMPATIBILITY if compatibility is None else compatibility
    return export_site_packet(
        metrics,
        output,
        node_id=node_id,
        contract_id=CONTRACT_ID,
        contract_sha256=_contract_sha(selected),
        minimum_subjects=MINIMUM_SUBJECTS,
        force=force,
        **selected,
    )


def _validate(
    packet: Path,
    *,
    compatibility: dict[str, object] | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    selected = COMPATIBILITY if compatibility is None else compatibility
    return validate_site_packet(
        packet,
        contract_id=CONTRACT_ID,
        contract_sha256=_contract_sha(selected),
        minimum_subjects=MINIMUM_SUBJECTS,
        **selected,
    )


def _aggregate(
    packets: list[Path],
    output: Path,
    *,
    compatibility: dict[str, object] | None = None,
) -> dict[str, object]:
    selected = COMPATIBILITY if compatibility is None else compatibility
    return aggregate_site_packets(
        packets,
        output,
        contract_id=CONTRACT_ID,
        contract_sha256=_contract_sha(selected),
        minimum_subjects=MINIMUM_SUBJECTS,
        **selected,
    )


def _cli_compatibility_args() -> list[str]:
    return [
        "--processing-config-sha256",
        PROCESSING_CONFIG_SHA256,
        "--source-artifact-kind",
        SOURCE_ARTIFACT_KIND,
        "--source-artifact-sha256",
        SOURCE_ARTIFACT_SHA256,
        "--rtpipeline-version",
        __version__,
        "--expected-inventory-sha256",
        EXPECTED_INVENTORY_SHA256,
    ]


def test_schema_v3_contract_binds_provenance_inventory_and_policy() -> None:
    contract = packet_contract_document(
        CONTRACT_ID,
        MINIMUM_SUBJECTS,
        **COMPATIBILITY,
    )

    assert PACKET_SCHEMA_VERSION == 3
    assert contract["compatibility_policy"] == EXACT_COMPATIBILITY_POLICY
    assert contract["processing_config_sha256"] == PROCESSING_CONFIG_SHA256
    assert contract["source_artifact"] == {
        "kind": SOURCE_ARTIFACT_KIND,
        "sha256": SOURCE_ARTIFACT_SHA256,
    }
    assert contract["rtpipeline_version"] == __version__
    assert contract["expected_feature_roi_inventory_sha256"] == EXPECTED_INVENTORY_SHA256
    assert contract["classification_rule"]["icc_statistic"] == (
        "icc_ci_low_when_both_ci_limits_finite_else_icc_when_both_null"
    )


def test_normalized_processing_config_digest_ignores_json_key_order() -> None:
    first = {"b": {"y": 2, "x": 1}, "a": [3, 4]}
    second = {"a": [3, 4], "b": {"x": 1, "y": 2}}
    assert normalized_processing_config_sha256(first) == normalized_processing_config_sha256(second)


def test_export_and_validate_round_trip_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = _export(_metrics(), first)
    _export(_metrics(), second)

    validated_manifest, validated = _validate(first)

    assert manifest == validated_manifest
    assert manifest["schema_version"] == 3
    assert manifest["rtpipeline_version"] == __version__
    assert manifest["feature_roi_inventory_sha256"] == EXPECTED_INVENTORY_SHA256
    assert validated["feature_name"].tolist() == sorted(validated["feature_name"])
    assert (first / "metrics.csv.gz").read_bytes() == (second / "metrics.csv.gz").read_bytes()
    assert manifest["content_audit"] == {
        "forbidden_columns": 0,
        "absolute_path_values": 0,
        "uri_values": 0,
        "dicom_uid_values": 0,
        "date_values": 0,
        "direct_identifier_values": 0,
        "hostname_values": 0,
    }


def test_native_adapter_uses_exact_columns_and_explicit_inventory() -> None:
    adapted = adapt_native_robustness_output(
        _native(),
        _inventory(),
        minimum_subjects=MINIMUM_SUBJECTS,
    )
    expected = _metrics().sort_values(["body_region", "roi_name", "feature_name"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(adapted, expected, check_dtype=False)


def test_native_adapter_supports_multi_source_native_summary() -> None:
    native_manual = _native()
    native_auto = native_manual.assign(segmentation_source="AutoTS")
    inventory_manual = _inventory()
    inventory_auto = inventory_manual.assign(segmentation_source="AutoTS")

    adapted = adapt_native_robustness_output(
        pd.concat([native_manual, native_auto], ignore_index=True),
        pd.concat([inventory_manual, inventory_auto], ignore_index=True),
        minimum_subjects=MINIMUM_SUBJECTS,
    )

    assert len(adapted) == 4
    assert set(adapted["segmentation_source"]) == {"Manual", "AutoTS"}
    assert not adapted.duplicated(
        ["body_region", "segmentation_source", "roi_name", "feature_name"]
    ).any()


def test_inventory_digest_binds_segmentation_source() -> None:
    manual_digest = feature_roi_inventory_sha256(_inventory())
    auto_digest = feature_roi_inventory_sha256(_inventory().assign(segmentation_source="AutoTS"))
    assert manual_digest != auto_digest


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(columns="n_perturbations").assign(n_raters=81), "aliases are not guessed"),
        (lambda frame: frame.drop(columns="segmentation_source"), "missing=.*segmentation_source"),
        (lambda frame: frame.assign(robustness_label=["Robust", "acceptable"]), "exact lowercase"),
        (lambda frame: frame.assign(patient_id=["P000001", "P000002"]), "Forbidden native-output"),
    ],
)
def test_native_adapter_fails_closed_on_invalid_native_output(mutation, message: str) -> None:
    with pytest.raises(FederationPacketError, match=message):
        adapt_native_robustness_output(mutation(_native()), _inventory())


def test_native_adapter_rejects_inventory_mismatch() -> None:
    inventory = _inventory().iloc[[0]].reset_index(drop=True)
    with pytest.raises(FederationPacketError, match="exactly match"):
        adapt_native_robustness_output(_native(), inventory)


def test_native_export_rejects_unbound_inventory_provenance(tmp_path: Path) -> None:
    with pytest.raises(FederationPacketError, match="digest does not match"):
        export_native_site_packet(
            _native(),
            _inventory(),
            tmp_path / "packet",
            node_id="node-a01",
            contract_id=CONTRACT_ID,
            contract_sha256=CONTRACT_SHA256,
            minimum_subjects=MINIMUM_SUBJECTS,
            **{**COMPATIBILITY, "expected_feature_roi_inventory_sha256": "f" * 64},
        )


def test_public_cli_native_output_to_packets_to_aggregate(tmp_path: Path) -> None:
    inventory_path = tmp_path / "inventory.csv"
    native_one = tmp_path / "native-one.csv"
    native_two = tmp_path / "native-two.csv"
    _inventory().to_csv(inventory_path, index=False)
    _native().to_csv(native_one, index=False)
    _native(-0.02).to_csv(native_two, index=False)
    packet_one = tmp_path / "packet-one"
    packet_two = tmp_path / "packet-two"

    for source, packet, node_id in (
        (native_one, packet_one, "node-a01"),
        (native_two, packet_two, "node-b02"),
    ):
        assert console_main(
            [
                "federation",
                "export-native",
                "--input",
                str(source),
                "--inventory",
                str(inventory_path),
                "--output",
                str(packet),
                "--node-id",
                node_id,
                "--contract-id",
                CONTRACT_ID,
                "--contract-sha256",
                CONTRACT_SHA256,
                "--minimum-subjects",
                str(MINIMUM_SUBJECTS),
                *_cli_compatibility_args(),
            ]
        ) == 0

    aggregate = tmp_path / "aggregate"
    assert console_main(
        [
            "federation",
            "aggregate",
            "--packet",
            str(packet_one),
            "--packet",
            str(packet_two),
            "--output",
            str(aggregate),
            "--contract-id",
            CONTRACT_ID,
            "--contract-sha256",
            CONTRACT_SHA256,
            "--minimum-subjects",
            str(MINIMUM_SUBJECTS),
            *_cli_compatibility_args(),
        ]
    ) == 0
    manifest = json.loads((aggregate / "aggregate_manifest.json").read_text())
    assert manifest["node_count"] == 2
    assert manifest["row_count"] == 4
    with gzip.open(aggregate / "combined_metrics.csv.gz", "rt") as handle:
        combined = pd.read_csv(handle)
    assert combined.columns.tolist() == ["node_id", *PACKET_COLUMNS]
    assert combined["node_id"].tolist() == ["node-a01", "node-a01", "node-b02", "node-b02"]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("processing_config_sha256", "b" * 64),
        ("source_artifact_kind", "source_tree"),
        ("source_artifact_sha256", "c" * 64),
        ("expected_rtpipeline_version", "9.9.9"),
        ("expected_feature_roi_inventory_sha256", "d" * 64),
    ],
)
def test_validation_rejects_compatibility_mismatch_by_default(
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    packet = tmp_path / "packet"
    _export(_metrics(), packet)
    altered = {**COMPATIBILITY, field: replacement}
    manifest_field = {
        "expected_rtpipeline_version": "rtpipeline_version",
        "expected_feature_roi_inventory_sha256": "feature_roi_inventory_sha256",
    }.get(field, field)
    with pytest.raises(FederationPacketError, match=manifest_field):
        _validate(packet, compatibility=altered)


@pytest.mark.parametrize(
    ("manifest_field", "replacement"),
    [
        ("processing_config_sha256", "b" * 64),
        ("source_artifact_kind", "source_tree"),
        ("source_artifact_sha256", "c" * 64),
        ("rtpipeline_version", "9.9.9"),
        ("feature_roi_inventory_sha256", "d" * 64),
    ],
)
def test_aggregate_rejects_mismatched_compatibility_fields(
    tmp_path: Path,
    manifest_field: str,
    replacement: str,
) -> None:
    packet_one = tmp_path / "one"
    packet_two = tmp_path / "two"
    _export(_metrics(), packet_one, "node-a01")
    _export(_metrics(-0.02), packet_two, "node-b02")
    manifest_path = packet_two / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[manifest_field] = replacement
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(FederationPacketError, match=manifest_field):
        _aggregate([packet_one, packet_two], tmp_path / "aggregate")


def test_aggregate_preserves_only_aggregate_rows_and_compatibility(tmp_path: Path) -> None:
    packet_one = tmp_path / "one"
    packet_two = tmp_path / "two"
    _export(_metrics(), packet_one, "node-a01")
    _export(_metrics(-0.02), packet_two, "node-b02")

    aggregate = _aggregate([packet_two, packet_one], tmp_path / "aggregate")

    assert aggregate["compatibility_policy"] == EXACT_COMPATIBILITY_POLICY
    assert aggregate["processing_config_sha256"] == PROCESSING_CONFIG_SHA256
    assert aggregate["source_artifact_sha256"] == SOURCE_ARTIFACT_SHA256
    assert aggregate["expected_feature_roi_inventory_sha256"] == EXPECTED_INVENTORY_SHA256
    assert aggregate["node_count"] == 2
    assert aggregate["row_count"] == 4


def test_export_accepts_both_missing_ci_and_uses_point_estimate(tmp_path: Path) -> None:
    frame = _metrics()
    frame.loc[0, ["icc_ci_low", "icc_ci_high"]] = float("nan")
    frame.loc[0, "icc"] = 0.93
    frame.loc[0, "classification"] = "Robust"
    packet = tmp_path / "packet"

    _export(frame, packet)
    _, validated = _validate(packet)

    row = validated.loc[validated["feature_name"] == "original_firstorder_Mean"].iloc[0]
    assert pd.isna(row["icc_ci_low"])
    assert pd.isna(row["icc_ci_high"])
    assert row["classification"] == "Robust"


def test_export_rejects_one_missing_ci_limit(tmp_path: Path) -> None:
    frame = _metrics()
    frame.loc[0, "icc_ci_low"] = float("nan")
    with pytest.raises(FederationPacketError, match="both finite or both missing"):
        _export(frame, tmp_path / "packet")


def test_point_estimate_fallback_still_enforces_classification(tmp_path: Path) -> None:
    frame = _metrics()
    frame.loc[0, ["icc_ci_low", "icc_ci_high"]] = float("nan")
    frame.loc[0, "icc"] = 0.74
    frame.loc[0, "classification"] = "Acceptable"
    with pytest.raises(FederationPacketError, match="expected=Poor"):
        _export(frame, tmp_path / "packet")


def test_packet_publication_is_atomic_on_interrupted_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "packet"

    def fail_write(path: Path, value: object) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr(federation, "_write_json", fail_write)
    with pytest.raises(OSError, match="simulated interruption"):
        _export(_metrics(), destination)

    assert not destination.exists()
    assert list(tmp_path.glob(".packet.staging-*")) == []


def test_force_packet_publication_preserves_old_artifact_on_staging_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "packet"
    _export(_metrics(), destination)
    old_manifest = (destination / "manifest.json").read_bytes()

    def fail_write(path: Path, value: object) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr(federation, "_write_json", fail_write)
    with pytest.raises(OSError, match="simulated interruption"):
        _export(_metrics(-0.02), destination, force=True)

    assert (destination / "manifest.json").read_bytes() == old_manifest
    _validate(destination)


def test_aggregate_publication_is_atomic_on_interrupted_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet_one = tmp_path / "one"
    packet_two = tmp_path / "two"
    _export(_metrics(), packet_one, "node-a01")
    _export(_metrics(-0.02), packet_two, "node-b02")
    destination = tmp_path / "aggregate"

    def fail_write(path: Path, value: object) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr(federation, "_write_json", fail_write)
    with pytest.raises(OSError, match="simulated interruption"):
        _aggregate([packet_one, packet_two], destination)

    assert not destination.exists()
    assert list(tmp_path.glob(".aggregate.staging-*")) == []


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


def test_export_round_trips_not_evaluable_relative_dispersion(tmp_path: Path) -> None:
    frame = _metrics()
    frame.loc[0, ["n_subjects_cov", "n_subjects_qcd"]] = 0
    frame.loc[0, ["cov_percent", "qcd"]] = float("nan")
    frame.loc[0, "classification"] = "Not Evaluable"
    packet = tmp_path / "packet"

    _export(frame, packet)
    _, validated = _validate(packet)

    row = validated.loc[validated["feature_name"] == "original_firstorder_Mean"].iloc[0]
    assert row["n_subjects_cov"] == 0
    assert row["n_subjects_qcd"] == 0
    assert pd.isna(row["cov_percent"])
    assert pd.isna(row["qcd"])
    assert row["classification"] == "Not Evaluable"


def test_export_rejects_hidden_partial_cov_denominator(tmp_path: Path) -> None:
    frame = _metrics()
    frame.loc[0, "n_subjects_cov"] = 11
    with pytest.raises(FederationPacketError, match="Not Evaluable"):
        _export(frame, tmp_path / "packet")


@pytest.mark.parametrize(
    ("icc_ci_low", "cov_percent", "classification", "expected"),
    [
        (0.88, 4.1, "Robust", "Acceptable"),
        (0.74, 4.1, "Acceptable", "Poor"),
        (0.91, 4.1, "Poor", "Robust"),
        (0.91, 10.1, "Robust", "Acceptable"),
        (0.76, 20.1, "Acceptable", "Poor"),
    ],
)
def test_export_rejects_threshold_inconsistent_classification(
    tmp_path: Path,
    icc_ci_low: float,
    cov_percent: float,
    classification: str,
    expected: str,
) -> None:
    frame = _metrics()
    frame.loc[0, "icc_ci_low"] = icc_ci_low
    frame.loc[0, "classification"] = classification
    frame.loc[0, "cov_percent"] = cov_percent
    with pytest.raises(
        FederationPacketError,
        match=rf"expected={expected}, actual={classification}",
    ):
        _export(frame, tmp_path / "packet")


def test_export_rejects_metric_denominator_mismatch(tmp_path: Path) -> None:
    frame = _metrics()
    frame.loc[0, "n_subjects_qcd"] = 0
    with pytest.raises(FederationPacketError, match="if and only if"):
        _export(frame, tmp_path / "packet")


def test_export_rejects_duplicate_feature_identity(tmp_path: Path) -> None:
    frame = pd.concat([_metrics(), _metrics().iloc[[0]]], ignore_index=True)
    with pytest.raises(FederationPacketError, match="duplicate"):
        _export(frame, tmp_path / "packet")


def test_validator_rejects_tampering_extra_files_and_symlinks(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    _export(_metrics(), packet)
    manifest_path = packet / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metrics_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(FederationPacketError, match="hash mismatch"):
        _validate(packet)

    packet_two = tmp_path / "packet-two"
    _export(_metrics(), packet_two)
    (packet_two / "patient_rows.csv").write_text("patient_id\nP000001\n")
    with pytest.raises(FederationPacketError, match="exactly manifest.json"):
        _validate(packet_two)

    packet_three = tmp_path / "packet-three"
    _export(_metrics(), packet_three)
    (packet_three / "extra-link").symlink_to(packet_three / "metrics.csv.gz")
    with pytest.raises(FederationPacketError, match="symlinks"):
        _validate(packet_three)


def test_float_columns_round_trip_bitwise(tmp_path: Path) -> None:
    frame = _metrics()
    packet = tmp_path / "packet"
    _export(frame, packet)
    _, validated = _validate(packet)
    expected = frame.sort_values(["body_region", "roi_name", "feature_name"])
    for column in ("icc", "icc_ci_low", "icc_ci_high", "cov_percent", "qcd"):
        expected_bits = expected[column].to_numpy(dtype="float64").view("uint64")
        observed_bits = validated[column].to_numpy(dtype="float64").view("uint64")
        assert observed_bits.tolist() == expected_bits.tolist()
