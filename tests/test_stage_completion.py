from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping
from importlib import metadata as importlib_metadata
from pathlib import Path

import pytest

from rtpipeline.config_dependencies import materialize_stage_dependency
from rtpipeline.stage_completion import (
    execution_environment_fingerprint,
    validate_stage_completion_sentinel,
    write_stage_completion_sentinel,
)


def _dependency(
    tmp_path: Path,
    stage: str,
    payload: Mapping[str, object] | None = None,
) -> Path:
    return materialize_stage_dependency(
        tmp_path / "dependencies", stage, payload or {"enabled": True}
    )


def _course(tmp_path: Path) -> Path:
    course = tmp_path / "Output" / "p1" / "c1"
    course.mkdir(parents=True)
    return course


def test_dvh_completion_binds_content_config_code_and_environment(tmp_path: Path) -> None:
    course = _course(tmp_path)
    qc = course / "metadata" / "dvh_qc.json"
    qc.parent.mkdir()
    qc.write_text('{"status":"ok"}\n', encoding="utf-8")
    parquet = course / "dvh_metrics.parquet"
    parquet.write_bytes(b"authoritative-dvh")
    dependency = _dependency(tmp_path, "dvh", {"bins": [1, 2, 3]})

    payload = write_stage_completion_sentinel(
        course,
        course / ".dvh_done",
        stage="dvh",
        status="ok",
        configuration_dependency=dependency,
    )

    assert payload["schema"] == "rtpipeline-stage-completion-v1"
    assert payload["binding_policy"] == "dvh-content-v1"
    assert payload["configuration_dependency_sha256"]
    assert payload["code_identity_sha256"]
    source_paths = {entry["path"] for entry in payload["code_identity"]["sources"]}
    assert "Snakefile" in source_paths
    assert "workflow/scripts/run_course_stage.py" in source_paths
    assert "dvh.py" in source_paths
    assert payload["execution_host"]
    assert payload["environment_fingerprint"]
    assert {entry["role"] for entry in payload["outputs"]} == {
        "authoritative_dvh",
        "dvh_qc",
    }
    assert validate_stage_completion_sentinel(
        course / ".dvh_done", expected_stage="dvh"
    ) == payload

    extra = course / "dvh_metrics.xlsx"
    extra.write_bytes(b"not declared at completion")
    with pytest.raises(ValueError, match="complete output set"):
        validate_stage_completion_sentinel(
            course / ".dvh_done", expected_stage="dvh"
        )
    extra.unlink()

    parquet.write_bytes(b"mutated-authoritative-dvh")
    with pytest.raises(ValueError, match="changed"):
        validate_stage_completion_sentinel(
            course / ".dvh_done", expected_stage="dvh"
        )


def test_organized_completion_uses_hybrid_binding(tmp_path: Path) -> None:
    course = _course(tmp_path)
    contract = course / "metadata" / "case_metadata.json"
    contract.parent.mkdir()
    contract.write_text('{"course_contract":{"version":4}}\n', encoding="utf-8")
    dicom = course / "DICOM" / "CT" / "image.dcm"
    dicom.parent.mkdir(parents=True)
    dicom.write_bytes(b"first")
    dependency = _dependency(tmp_path, "organize", {"dicom_root": "/source"})

    payload = write_stage_completion_sentinel(
        course,
        course / ".organized",
        stage="organize",
        status="ok",
        configuration_dependency=dependency,
    )
    outputs = {entry["path"]: entry for entry in payload["outputs"]}
    assert outputs["metadata/case_metadata.json"]["binding"] == "content_sha256"
    assert outputs["DICOM/CT/image.dcm"]["binding"] == "inventory"

    # The bulk DICOM materialization contract binds the declared set and sizes.
    # Scientific selection and object identity remain content-bound in the small
    # authoritative course contract.
    dicom.write_bytes(b"other")
    validate_stage_completion_sentinel(
        course / ".organized", expected_stage="organize"
    )
    contract.write_text('{"course_contract":{"version":3}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        validate_stage_completion_sentinel(
            course / ".organized", expected_stage="organize"
        )


@pytest.mark.parametrize('metadata', [None, '{', '{}',
    '{"schema":"rtpipeline-original-segmentation-v2","geometry_status":"MASK_GRID_NOT_EQUIVALENT"}'])
def test_organized_completion_records_stale_exports_without_certifying_them(tmp_path, metadata):
    course = _course(tmp_path)
    contract = course / 'metadata/case_metadata.json'
    contract.parent.mkdir()
    contract.write_text('{}')
    dicom = course / 'DICOM/CT/image.dcm'
    dicom.parent.mkdir(parents=True)
    dicom.write_bytes(b'synthetic')
    export = course / 'Segmentation_Original/series/rtstruct'
    export.mkdir(parents=True)
    mask = export / 'mask.nii.gz'
    mask.write_bytes(b'stale synthetic mask')
    if metadata is not None:
        (export / 'metadata.json').write_text(metadata)
    (export / 'provenance.json').write_text('{}')
    before = {path: path.read_bytes() for path in export.iterdir()}
    sentinel = course / '.organized'
    payload = write_stage_completion_sentinel(course, sentinel, stage='organize', status='ok',
                                             configuration_dependency=_dependency(tmp_path, 'organize'))
    # Workflow resume accepts organize, then segmentation completion must work.
    assert validate_stage_completion_sentinel(sentinel, expected_stage='organize') == payload
    from rtpipeline.stage_completion import _validate_stage_specific_outcome, stage_definition
    _validate_stage_specific_outcome(course, stage_definition('segmentation'), 'ok', [])
    assert not any(item['role'] == 'original_segmentation' for item in payload['outputs'])
    assert payload['stale_original_exports'] == []
    assert not export.exists()
    archived = next((course / 'Segmentation_Quarantine').rglob('export'))
    assert {p.name: p.read_bytes() for p in archived.iterdir()} == {p.name: v for p,v in before.items()}
    assert validate_stage_completion_sentinel(sentinel, expected_stage='organize') == payload
    from rtpipeline.stage_completion import _validate_stage_specific_outcome, stage_definition
    _validate_stage_specific_outcome(course, stage_definition('segmentation'), 'ok', [])
    # An active stale mask introduced after publication must invalidate organize.
    export.mkdir(parents=True)
    mask.write_bytes(b'new stale mask')
    with pytest.raises(ValueError, match='stale'):
        validate_stage_completion_sentinel(sentinel, expected_stage='organize')


def test_organized_inventory_detects_missing_declared_output(tmp_path: Path) -> None:
    course = _course(tmp_path)
    contract = course / "metadata" / "case_metadata.json"
    contract.parent.mkdir()
    contract.write_text("{}\n", encoding="utf-8")
    dicom = course / "DICOM" / "CT" / "image.dcm"
    dicom.parent.mkdir(parents=True)
    dicom.write_bytes(b"bytes")
    dependency = _dependency(tmp_path, "organize")
    write_stage_completion_sentinel(
        course,
        course / ".organized",
        stage="organize",
        status="ok",
        configuration_dependency=dependency,
    )

    dicom.unlink()
    with pytest.raises(ValueError, match="absent"):
        validate_stage_completion_sentinel(
            course / ".organized", expected_stage="organize"
        )


def test_organized_inventory_supports_external_leaf_symlink(tmp_path: Path) -> None:
    course = _course(tmp_path)
    contract = course / "metadata" / "case_metadata.json"
    contract.parent.mkdir()
    contract.write_text("{}\n", encoding="utf-8")
    source = tmp_path / "source.dcm"
    source.write_bytes(b"source")
    alias = course / "DICOM" / "CT" / "image.dcm"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(source)
    dependency = _dependency(tmp_path, "organize")

    payload = write_stage_completion_sentinel(
        course,
        course / ".organized",
        stage="organize",
        status="ok",
        configuration_dependency=dependency,
    )
    entry = next(item for item in payload["outputs"] if item["path"] == "DICOM/CT/image.dcm")
    assert entry["kind"] == "symlink"
    assert entry["symlink_target"] == str(source)
    validate_stage_completion_sentinel(
        course / ".organized", expected_stage="organize"
    )


def test_disabled_custom_stage_has_structured_zero_output_completion(tmp_path: Path) -> None:
    course = _course(tmp_path)
    dependency = _dependency(tmp_path, "custom-models", {"enabled": False})
    payload = write_stage_completion_sentinel(
        course,
        course / ".custom_models_done",
        stage="segmentation_custom",
        status="disabled",
        configuration_dependency=dependency,
    )
    assert payload["status"] == "disabled"
    assert payload["outputs"] == []
    assert payload["output_count"] == 0
    validate_stage_completion_sentinel(
        course / ".custom_models_done", expected_stage="segmentation_custom"
    )


def test_required_stage_output_role_is_fail_closed(tmp_path: Path) -> None:
    course = _course(tmp_path)
    dependency = _dependency(tmp_path, "qc")
    with pytest.raises(ValueError, match="required output roles"):
        write_stage_completion_sentinel(
            course,
            course / ".qc_done",
            stage="qc",
            status="ok",
            configuration_dependency=dependency,
        )
    assert not (course / ".qc_done").exists()


def test_configuration_dependency_must_match_stage_and_digest(tmp_path: Path) -> None:
    course = _course(tmp_path)
    (course / "metadata").mkdir()
    (course / "metadata" / "dvh_qc.json").write_text("{}\n", encoding="utf-8")
    wrong = _dependency(tmp_path, "qc")
    with pytest.raises(ValueError, match="not 'dvh'"):
        write_stage_completion_sentinel(
            course,
            course / ".dvh_done",
            stage="dvh",
            status="ok",
            configuration_dependency=wrong,
        )

    dependency = _dependency(tmp_path, "dvh")
    record = json.loads(dependency.read_text(encoding="utf-8"))
    record["payload"] = {"changed": True}
    dependency.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        write_stage_completion_sentinel(
            course,
            course / ".dvh_done",
            stage="dvh",
            status="ok",
            configuration_dependency=dependency,
        )


def test_refactored_radiomics_environment_fingerprint_is_byte_compatible() -> None:
    def version(name: str) -> str:
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            return "unavailable"

    legacy = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "libc": platform.libc_ver(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "pywavelets": version("PyWavelets"),
        "simpleitk": version("SimpleITK"),
        "pyradiomics": version("pyradiomics"),
        "pyarrow": version("pyarrow"),
    }
    expected = hashlib.sha256(
        json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert execution_environment_fingerprint() == expected


def test_disabled_segmentation_completion_validates_beside_its_status_record(
    tmp_path: Path,
) -> None:
    # A course with no planning CT closes segmentation as "disabled". The
    # producer always writes metadata/segmentation_status.json with that
    # decision, which the stage's rules discover. The disabled completion binds
    # no outputs (as written), so validation must not demand the discovered set.
    course = _course(tmp_path)
    (course / "metadata").mkdir()
    (course / "metadata" / "segmentation_status.json").write_text(
        json.dumps({"status": "disabled", "reasons": ["nothing applicable"]}),
        encoding="utf-8",
    )
    dependency = _dependency(tmp_path, "segmentation", {"device": "gpu"})
    payload = write_stage_completion_sentinel(
        course,
        course / ".segmentation_done",
        stage="segmentation",
        status="disabled",
        configuration_dependency=dependency,
    )
    assert payload["status"] == "disabled"
    assert payload["outputs"] == []
    validate_stage_completion_sentinel(
        course / ".segmentation_done", expected_stage="segmentation"
    )
