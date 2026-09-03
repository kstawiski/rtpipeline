from __future__ import annotations

import json
import os
import runpy
from base64 import urlsafe_b64encode
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from rtpipeline import radiomics_ct_contract as contract
from rtpipeline.config_dependencies import (
    adopt_legacy_snakemake_inputs,
    advance_dependency_past_unbound_outputs,
    materialize_stage_dependency,
    radiomics_parameter_manifest,
    read_stage_dependency,
    semantic_yaml,
)


PACKAGE_DIR = Path(contract.__file__).parent


def _parameter_manifest() -> dict[str, Any]:
    return radiomics_parameter_manifest(
        ct_params=PACKAGE_DIR / "radiomics_params.yaml",
        mr_params=PACKAGE_DIR / "radiomics_params_mr.yaml",
        roi_class_map=PACKAGE_DIR / "roi_class_map_v1.yaml",
        pet_params={
            "fdg": PACKAGE_DIR / "radiomics_params_pet_fdg.yaml",
            "psma": PACKAGE_DIR / "radiomics_params_pet_psma.yaml",
        },
    )


def test_stable_dependency_changes_only_for_semantic_content(tmp_path):
    source = tmp_path / "params.yaml"
    source.write_text("setting:\n  binWidth: 25\n  imageTypes: [Original]\n", encoding="utf-8")
    initial_source_ns = 1_700_000_000_000_000_000
    os.utime(source, ns=(initial_source_ns, initial_source_ns))

    cache = tmp_path / "cache"
    marker = materialize_stage_dependency(
        cache,
        "radiomics",
        {"params": semantic_yaml(source)},
        source_paths=[source],
    )
    initial_marker_ns = marker.stat().st_mtime_ns
    output = tmp_path / ".radiomics_done"
    output.write_text("ok\n", encoding="utf-8")
    output_ns = initial_marker_ns + 1_000_000
    os.utime(output, ns=(output_ns, output_ns))

    source.write_text(
        "# comment and mapping order do not change extraction\n"
        "setting:\n  imageTypes: [Original]\n  binWidth: 25\n",
        encoding="utf-8",
    )
    unchanged = materialize_stage_dependency(
        cache,
        "radiomics",
        {"params": semantic_yaml(source)},
        source_paths=[source],
    )
    assert unchanged == marker
    assert marker.stat().st_mtime_ns == initial_marker_ns
    assert output.stat().st_mtime_ns > marker.stat().st_mtime_ns

    source.write_text("setting:\n  binWidth: 10\n  imageTypes: [Original]\n", encoding="utf-8")
    changed = materialize_stage_dependency(
        cache,
        "radiomics",
        {"params": semantic_yaml(source)},
        source_paths=[source],
    )
    assert changed == marker
    assert marker.stat().st_mtime_ns > output.stat().st_mtime_ns
    record = read_stage_dependency(marker, expected_stage="radiomics")
    assert record["payload"]["params"]["setting"]["binWidth"] == 10
    assert len(record["sha256"]) == 64


def test_input_inventory_migration_preserves_other_metadata(tmp_path):
    affected = tmp_path / "Output" / "P1" / "C1" / ".radiomics_done"
    unrelated = tmp_path / "Output" / "_COURSES" / "manifest.json"
    metadata = tmp_path / ".snakemake" / "metadata"
    metadata.mkdir(parents=True)

    def _record(path: Path) -> Path:
        encoded = urlsafe_b64encode(str(path).encode()).decode()
        record = metadata / encoded
        record.write_text(
            '{"code":"preserve-me","input":["existing-input"]}\n',
            encoding="utf-8",
        )
        return record

    affected_record = _record(affected)
    unrelated_record = _record(unrelated)
    incomplete = tmp_path / ".snakemake" / "incomplete"
    incomplete.mkdir(parents=True)
    incomplete_record = incomplete / urlsafe_b64encode(str(affected).encode()).decode()
    incomplete_record.write_text("{}\n", encoding="utf-8")

    dependency = tmp_path / "dependencies" / "radiomics.json"
    assert adopt_legacy_snakemake_inputs(tmp_path, {affected: dependency}) == 1
    adopted = json.loads(affected_record.read_text(encoding="utf-8"))
    assert adopted["code"] == "preserve-me"
    assert adopted["input"] == sorted(["existing-input", str(dependency)])
    assert unrelated_record.exists()
    assert incomplete_record.exists()


def test_unbound_completion_is_invalidated_until_hash_is_recorded(tmp_path):
    dependency = materialize_stage_dependency(
        tmp_path / "dependencies",
        "radiomics",
        {"params": {"binWidth": 25}},
    )
    completion = tmp_path / ".radiomics_done"
    completion.write_text("ok\n", encoding="utf-8")
    completion_ns = dependency.stat().st_mtime_ns + 1_000_000
    os.utime(completion, ns=(completion_ns, completion_ns))

    assert advance_dependency_past_unbound_outputs(
        dependency,
        [completion],
        binding_field="configuration_dependency_sha256",
    ) == 1
    assert dependency.stat().st_mtime_ns > completion.stat().st_mtime_ns

    record = read_stage_dependency(dependency, expected_stage="radiomics")
    completion.write_text(
        '{"configuration_dependency_sha256":"' + record["sha256"] + '"}\n',
        encoding="utf-8",
    )
    marker_ns = dependency.stat().st_mtime_ns
    assert advance_dependency_past_unbound_outputs(
        dependency,
        [completion],
        binding_field="configuration_dependency_sha256",
    ) == 0
    assert dependency.stat().st_mtime_ns == marker_ns


def test_radiomics_manifest_covers_ct_mr_and_pet_parameter_files():
    manifest = _parameter_manifest()

    assert manifest["schema"] == "rtpipeline-radiomics-config-dependency-v1"
    assert set(manifest) == {"schema", "ct", "mr", "pet"}
    assert set(manifest["pet"]) == {"fdg", "psma"}
    assert manifest["pet"]["fdg"]["state"] == "present"
    assert manifest["pet"]["psma"]["state"] == "present"
    assert manifest["mr"]["params"]["setting"]["binWidth"] > 0
    assert set(manifest["mr"]["configured_parameter_hashes"]) == {
        "mr_configured",
        "mr_native_intensity",
        "mr_normalized",
    }
    base_keys = {
        "primary_resegmented|none",
        "primary_resegmented|-1000,400",
        "primary_resegmented|-500,400",
        "sensitivity_raw|none",
    }
    assert set(manifest["ct"]["configured_parameter_hashes"]) == {
        f"{key}|large_roi={flag}" for key in base_keys for flag in (0, 1)
    }


def test_default_roi_map_comes_from_the_dag_dependency(tmp_path, monkeypatch):
    """A live checkout edit cannot change a running radiomics job's ROI map."""

    manifest = _parameter_manifest()
    dependency = materialize_stage_dependency(
        tmp_path / "dependencies",
        "radiomics",
        {"parameter_provenance": manifest},
    )
    monkeypatch.setenv("RTPIPELINE_RADIOMICS_CONFIG_DEPENDENCY", str(dependency))
    monkeypatch.setattr(
        contract.importlib_resources,
        "files",
        lambda _package: (_ for _ in ()).throw(
            AssertionError("the mutable package resource must not be read")
        ),
    )
    contract._load_roi_class_map.cache_clear()

    data, digest = contract.load_roi_class_map()

    identity = manifest["ct"]["roi_class_map"]
    assert data == identity["content"]
    assert digest == identity["sha256"]


def test_mr_rows_reuse_the_manifest_configured_hash():
    from rtpipeline import radiomics

    class _Extractor:
        settings = {"binWidth": 25, "normalize": False}
        enabledImagetypes = {"Original": {}}
        enabledFeatures = {"firstorder": []}

    params = PACKAGE_DIR / "radiomics_params_mr.yaml"
    config = SimpleNamespace(radiomics_params_file_mr=params)
    provenance = radiomics._mr_parameter_provenance(
        cast(Any, config),
        _Extractor(),
        normalize_override=False,
        run_identifier="run-1",
        code_revision="revision-1",
    )
    manifest = _parameter_manifest()

    assert provenance["configured_parameter_hash"] == manifest["mr"][
        "configured_parameter_hashes"
    ]["mr_native_intensity"]
    assert len(provenance["effective_parameter_hash"]) == 64
    assert radiomics._mr_resume_parameter_provenance_is_current(
        pd.DataFrame([provenance]), params
    )


def test_row_hashes_must_agree_with_dag_parameter_manifest(tmp_path, monkeypatch):
    helpers = runpy.run_path(str(Path(__file__).with_name("test_radiomics_ct_dual_arm.py")))
    common_identity = helpers["_common_identity"]
    dual_arm_rows = helpers["_dual_arm_rows"]

    manifest = _parameter_manifest()
    configured = manifest["ct"]["configured_parameter_hashes"]
    rows = dual_arm_rows(monkeypatch)
    for row in rows:
        if row["extraction_arm"] == contract.PRIMARY_ARM:
            row["configured_parameter_hash"] = configured[
                "primary_resegmented|-1000,400|large_roi=0"
            ]
        else:
            row["configured_parameter_hash"] = configured[
                "sensitivity_raw|none|large_roi=1"
            ]
        row.update(common_identity())

    dependency = materialize_stage_dependency(
        tmp_path / "dependencies",
        "radiomics",
        {"parameter_provenance": manifest},
    )
    course = tmp_path / "P1" / "C1"
    contract.write_ct_publication_atomic(
        pd.DataFrame(rows), course / "radiomics_ct.xlsx"
    )
    sentinel = contract.write_completion_sentinel(
        course,
        configuration_dependency=dependency,
    )
    payload = contract.validate_completion_sentinel(
        course,
        sentinel,
        configuration_dependency=dependency,
    )
    dependency_record = read_stage_dependency(
        dependency, expected_stage="radiomics"
    )
    assert payload["configuration_dependency_sha256"] == dependency_record["sha256"]

    monkeypatch.setenv("RTPIPELINE_RADIOMICS_CONFIG_DEPENDENCY", str(dependency))
    environment_bound = contract.write_completion_sentinel(course)
    environment_payload = json.loads(environment_bound.read_text(encoding="utf-8"))
    assert environment_payload["configuration_dependency_sha256"] == dependency_record[
        "sha256"
    ]

    mismatched = [dict(row) for row in rows]
    mismatched[0]["configured_parameter_hash"] = "0" * 64
    contract.write_ct_publication_atomic(
        pd.DataFrame(mismatched), course / "radiomics_ct.xlsx"
    )
    with pytest.raises(ValueError, match="disagrees with the DAG extraction configuration"):
        contract.write_completion_sentinel(
            course,
            configuration_dependency=dependency,
        )


def test_snakefile_exposes_stage_configuration_dependencies():
    snakefile = (PACKAGE_DIR.parent / "Snakefile").read_text(encoding="utf-8")

    for name in (
        "SEGMENTATION_CONFIG_DEPENDENCY",
        "CUSTOM_MODELS_CONFIG_DEPENDENCY",
        "CROP_CT_CONFIG_DEPENDENCY",
        "DVH_CONFIG_DEPENDENCY",
        "RADIOMICS_CONFIG_DEPENDENCY",
        "ROBUSTNESS_CONFIG_DEPENDENCY",
    ):
        assert f"{name} = _dependency_path(" in snakefile
    assert snakefile.count("configuration=SEGMENTATION_CONFIG_DEPENDENCY") == 2
    assert snakefile.count("configuration=CUSTOM_MODELS_CONFIG_DEPENDENCY") == 2
    assert snakefile.count("configuration=CROP_CT_CONFIG_DEPENDENCY") == 1
    assert snakefile.count("configuration=DVH_CONFIG_DEPENDENCY") == 1
    assert '"configuration": RADIOMICS_CONFIG_DEPENDENCY' in snakefile
    assert '"configuration": ROBUSTNESS_CONFIG_DEPENDENCY' in snakefile
    assert "configuration=ROBUSTNESS_CONFIG_DEPENDENCY" in snakefile
    assert 'os.environ["RTPIPELINE_RADIOMICS_CONFIG_DEPENDENCY"]' in snakefile
    assert "adopt_legacy_snakemake_inputs" in snakefile
