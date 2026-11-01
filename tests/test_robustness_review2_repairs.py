"""Synthetic regressions for source classification, reuse and campaign accounting."""

import ast
import csv
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pydicom
import pytest
import SimpleITK as sitk
import yaml

from rtpipeline import cli, radiomics as rm, radiomics_robustness as rr
from rtpipeline import robustness_completion as rc
from rtpipeline.config_dependencies import migrate_legacy_robustness_receipts
from rtpipeline.course_manifest import read_course_manifest
from test_robustness_consumer_integration import _build_courses, _run_course, _write_manifest
from test_robustness_producer_equivalence import _layout, _produce, main_producer, one_itk_thread  # noqa: F401
from test_robustness_review_repairs import _course
from test_robustness_selection_unmatched import _ledger
from test_robustness_snakefile_parse import SNAKEFILE_PATH, parse_snakefile_source


def _selection(c, patterns):
    data = yaml.safe_load(c.config_path.read_text())
    data["radiomics_robustness"]["segmentation_perturbation"]["apply_to_structures"] = patterns
    c.config_path.write_text(yaml.safe_dump(data))


@pytest.mark.parametrize("nifti_ct", [False, True])
@pytest.mark.parametrize("name", ["PTV1-poprzednio", "GTV3_prechemo"])
def test_selected_voxel_helper_never_unresolved(tmp_path, monkeypatch, nifti_ct, name):
    c = _course(tmp_path, monkeypatch)
    course = c.course_dir("P1")
    path = course / f"Segmentation_CustomModels/M/{name}.nii.gz"
    path.parent.mkdir(parents=True)
    image = sitk.GetImageFromArray(np.ones((17, 17, 17), np.uint8))
    sitk.WriteImage(image, str(path))
    if nifti_ct:
        ct = course / "ct.nii.gz"
        sitk.WriteImage(sitk.Cast(image, sitk.sitkFloat32), str(ct))
        contract = rr.load_course_contract(course)
        contract.planning_ct_dir, contract.planning_ct_nifti = None, ct
        monkeypatch.setattr(rr, "load_course_contract", lambda _: contract)
    assert _run_course(c, "P1", campaign=True) == 0
    assert c.receipt("P1")["measurement_outcome"] == rr.ROBUSTNESS_NO_TARGET_OUTCOME
    inventory = json.loads((course / "metadata/radiomics_robustness_target_inventory.json").read_text())
    roi = next(r for r in inventory["target_inventory"] if r["roi_name"] == name)
    assert not roi["volumetric"] and not roi["detector_rules"]


@pytest.mark.parametrize("name", ["pelvic_bones_3mm", "iliac_area"])
@pytest.mark.parametrize("via_cli", [False, True])
def test_custom_provenance_helper_reader_agrees(tmp_path, monkeypatch, name, via_cli):
    course, config, contract, catalog = _layout(tmp_path, monkeypatch)
    ds = pydicom.dcmread(course / "RS.dcm")
    ds.StructureSetROISequence[0].ROIName = name
    ds.save_as(course / "RS_custom.dcm")
    ds.StructureSetROISequence[0].ROIName = "ODBYTNICA"
    ds.save_as(course / "RS.dcm")
    (course / "RS_auto.dcm").unlink()
    shutil.rmtree(course / "Segmentation_TotalSegmentator")
    catalog.clear()
    custom = tmp_path / "custom.yaml"
    custom.write_bytes((Path(rr.__file__).parent / "custom_structures_pelvic.yaml").read_bytes())
    config.custom_structures_config = custom
    catalog[("Custom", name)] = rr.RobustnessRoiIdentity.from_mapping({
        "patient_id": "P1", "course_id": "C1", "series_uid": contract.planning_ct["series_instance_uid"],
        "segmentation_source": "Custom", "roi_original_name": name,
        "mask_identity": "helper", "stable_roi_identifier": "helper",
    })
    monkeypatch.setattr("rtpipeline.custom_structures_rtstruct._is_rs_custom_stale", lambda *a: False)
    cfg = rr.RobustnessConfig(enabled=True)
    cfg.perturbation.apply_to_structures = [name]
    if via_cli:
        configuration = tmp_path / "config.yaml"
        configuration.write_text(yaml.safe_dump({
            "dicom_root": str(tmp_path / "Input"), "output_dir": str(tmp_path),
            "logs_dir": str(tmp_path / "Logs"), "custom_structures": str(custom),
            "radiomics_robustness": {"enabled": True, "segmentation_perturbation": {"apply_to_structures": [name]}},
        }))
        assert cli.main(["radiomics-robustness", "--course-dir", str(course), "--config", str(configuration), "--output", str(course / "radiomics_robustness_ct.parquet")]) == 0
    else:
        result = rr.run_robustness_course(config, cfg, course, output_path=course / "radiomics_robustness_ct.parquet")
        assert result.measurement_outcome == rr.ROBUSTNESS_NO_TARGET_OUTCOME
    sidecar = json.loads(rr.robustness_source_dispositions_path(course).read_text())
    assert sidecar["measurement_outcome"] == rr.ROBUSTNESS_NO_TARGET_OUTCOME


@pytest.mark.parametrize("measured", [False, True])
def test_contourless_standard_slot_does_not_hide_custom_target(tmp_path, monkeypatch, measured):
    if measured:
        course, config, _, _ = _layout(tmp_path, monkeypatch, "normal")
        ds = pydicom.dcmread(course / "RS.dcm")
        custom = copy.deepcopy(ds)
        custom.StructureSetROISequence[0].ROIName = "HR-CTV"
        custom.save_as(course / "RS_custom.dcm")
        roi, contours = copy.deepcopy(ds.StructureSetROISequence[0]), copy.deepcopy(ds.ROIContourSequence[0])
        roi.ROIName, roi.ROINumber, contours.ReferencedROINumber = "HR-CTV", 9, 9
        contours.ContourSequence = []
        ds.StructureSetROISequence.append(roi)
        ds.ROIContourSequence.append(contours)
        ds.save_as(course / "RS.dcm")
        monkeypatch.setattr("rtpipeline.custom_structures_rtstruct._is_rs_custom_stale", lambda *a: False)
        result = _produce(rr, rc, course, config)
        sidecar = result[1]
        rr.admit_robustness_cohort_course(course, patient_id="P1", course_id="C1", rob_config=rr.RobustnessConfig(enabled=True))
    else:
        c = _course(tmp_path, monkeypatch)
        course = c.course_dir("P1")
        ds = pydicom.dcmread(course / "RS.dcm")
        ds.StructureSetROISequence[0].ROIName = "HR-CTV"
        ds.save_as(course / "RS_custom.dcm")
        ds.ROIContourSequence[0].ContourSequence = []
        ds.save_as(course / "RS.dcm")
        monkeypatch.setattr("rtpipeline.custom_structures_rtstruct._is_rs_custom_stale", lambda *a: False)
        assert _run_course(c, "P1", campaign=True) == 0
        sidecar = json.loads(rr.robustness_source_dispositions_path(course).read_text())
    gaps = sidecar.get("selection_gap_targets", (sidecar.get("source_only_basis") or {}).get("missed_targets"))
    assert [(r["segmentation_source"], r["roi_name"]) for r in gaps] == [("Custom", "HR-CTV")]


def _add_copied_disposition(course, geometry):
    for filename in ("RS.dcm", "RS_custom.dcm"):
        path = course / filename
        ds = pydicom.dcmread(path)
        roi, contours = copy.deepcopy(ds.StructureSetROISequence[0]), copy.deepcopy(ds.ROIContourSequence[0])
        roi.ROIName, roi.ROINumber, contours.ReferencedROINumber = "CTV", 9, 9
        item = contours.ContourSequence[0]
        item.ContourData = item.ContourData[:3] if geometry == "POINT" else [1, 1, 0, 5, 1, 3, 5, 5, 0, 1, 5, 4]
        item.ContourGeometricType = "POINT" if geometry == "POINT" else "CLOSED_PLANAR"
        item.NumberOfContourPoints = len(item.ContourData) // 3
        contours.ContourSequence = [item]
        ds.StructureSetROISequence.append(roi)
        ds.ROIContourSequence.append(contours)
        ds.save_as(path)


@pytest.mark.parametrize("geometry", ["POINT", "UNPARSEABLE"])
def test_copied_custom_disposition_is_not_published_twice(tmp_path, monkeypatch, geometry):
    course, config, _, _ = _layout(tmp_path, monkeypatch, "custom")
    _add_copied_disposition(course, geometry)
    result = _produce(rr, rc, course, config)
    assert [(r["segmentation_source"], r["roi_name"]) for r in result[1]["rows"]] == [("Manual", "CTV")]


def _main_artifact(tmp_path, monkeypatch, main_producer, kind="custom"):
    course, config, contract, catalog = _layout(tmp_path, monkeypatch, kind)
    main, completion = main_producer
    for name in ("load_course_contract", "_load_main_ct_identity_catalog", "extract_features_for_masks"):
        monkeypatch.setattr(main, name, getattr(rr, name))
    return course, config, contract, catalog, main, completion


@pytest.mark.parametrize("real_check", [False, True])
def test_certificate_refuses_stale_custom(tmp_path, monkeypatch, main_producer, real_check):
    from rtpipeline.custom_structures_rtstruct import _is_rs_custom_stale
    course, config, _, _, main, completion = _main_artifact(tmp_path, monkeypatch, main_producer)
    before = _produce(main, completion, course, config)
    check = _is_rs_custom_stale if real_check else lambda *a: True
    monkeypatch.setattr("rtpipeline.custom_structures_rtstruct._is_rs_custom_stale", check)
    assert not rr._compatible_main_measured_code(before[1], rr._current_robustness_code_identity(), course_dir=course)


@pytest.mark.parametrize("geometry", ["POINT", "UNPARSEABLE"])
def test_certificate_refuses_copied_custom_rows(tmp_path, monkeypatch, main_producer, geometry):
    course, config, _, _, main, completion = _main_artifact(tmp_path, monkeypatch, main_producer)
    _add_copied_disposition(course, geometry)
    before = _produce(main, completion, course, config)
    assert [(r["segmentation_source"], r["roi_name"]) for r in before[1]["rows"]] == [("Manual", "CTV"), ("Custom", "CTV")]
    assert not rr._compatible_main_measured_code(before[1], rr._current_robustness_code_identity(), course_dir=course)


def test_certificate_never_loads_ct_and_admits_benign_voxel_model(tmp_path, monkeypatch, main_producer):
    course, config, _, _, main, completion = _main_artifact(tmp_path, monkeypatch, main_producer, "voxel_model")
    path = course / "Segmentation_CustomModels/M/GTV.nii.gz"
    path.rename(path.with_name("colon.nii.gz"))
    before = _produce(main, completion, course, config)
    monkeypatch.setattr(rm, "_load_series_image", lambda *a: pytest.fail("certificate loaded CT"))
    assert rr._compatible_main_measured_code(before[1], rr._current_robustness_code_identity(), course_dir=course)


def test_certificate_refuses_existing_unselected_target(tmp_path, monkeypatch, main_producer):
    course, config, _, _, main, completion = _main_artifact(tmp_path, monkeypatch, main_producer, "normal")
    path = course / "RS.dcm"
    ds = pydicom.dcmread(path)
    roi, contours = copy.deepcopy(ds.StructureSetROISequence[0]), copy.deepcopy(ds.ROIContourSequence[0])
    roi.ROIName, roi.ROINumber, contours.ReferencedROINumber = "HR-CTV", 9, 9
    ds.StructureSetROISequence.append(roi)
    ds.ROIContourSequence.append(contours)
    ds.save_as(path)
    before = _produce(main, completion, course, config)
    assert not rr._compatible_main_measured_code(before[1], rr._current_robustness_code_identity(), course_dir=course)


def test_relocated_physical_export_and_stale_sibling(tmp_path, monkeypatch):
    course, config, contract, _ = _layout(tmp_path, monkeypatch, "nifti_ct", export=True)
    provenance = json.loads((course / "ct.metadata.json").read_text())
    metadata = course / "metadata/case_metadata.json"
    metadata.parent.mkdir(exist_ok=True)
    metadata.write_text(json.dumps({"course_contract": {
        "planning_ct": {"nifti_path": "ct.nii.gz", "nifti_provenance": provenance},
        "authoritative_rtstruct": {"path": "RS.dcm"},
    }}))
    moved = tmp_path / "relocated/P1/C1"
    moved.parent.mkdir(parents=True)
    shutil.move(course, moved)
    contract.authoritative_rtstruct_path, contract.planning_ct_nifti = moved / "RS.dcm", moved / "ct.nii.gz"
    for unused in (moved / "Segmentation_TotalSegmentator").glob("*.nii.gz"):
        unused.unlink()
    export = next((moved / "Segmentation_Original").glob("*/metadata.json")).parent
    stale = export.with_name("000-stale")
    shutil.copytree(export, stale)
    manifest = json.loads((stale / "metadata.json").read_text())
    for entry in manifest["structures"]:
        entry["mask"] = f"000-stale/{Path(entry['mask']).name}"
    (stale / "metadata.json").write_text(json.dumps(manifest))
    (stale / "PTV.nii.gz").write_bytes(b"stale synthetic export")
    result = _produce(rr, rc, moved, config)
    assert result is not None
    inventory = json.loads((moved / "metadata/radiomics_robustness_target_inventory.json").read_text())
    assert [(r["segmentation_source"], r["roi_name"]) for r in inventory["target_inventory"]] == [("Manual", "PTV")]
    assert len(inventory["voxel_source_bindings"]) == 1


@pytest.mark.parametrize("name,unreadable", [("colon", True), ("HR-CTV", True), ("GTV", True), ("colon", False)])
def test_inventory_only_voxel_reading(tmp_path, monkeypatch, name, unreadable):
    course, config, _, _ = _layout(tmp_path, monkeypatch, "voxel_model")
    path = course / "Segmentation_CustomModels/M/GTV.nii.gz"
    renamed = path.with_name(f"{name}.nii.gz")
    path.rename(renamed)
    if unreadable:
        renamed.write_bytes(b"unreadable synthetic mask")
    monkeypatch.setattr(rr, "_nifti_mask_in_reference_order", lambda *a: pytest.fail("inventory resampled a voxel mask"))
    if unreadable and name != "colon":
        with pytest.raises(RuntimeError, match="Unreadable robustness NIfTI"):
            _produce(rr, rc, course, config)
    else:
        result = _produce(rr, rc, course, config)
        assert result is not None
        if unreadable:
            assert [(r["segmentation_source"], r["roi_name"]) for r in result[1]["tolerated_source_failures"]] == [("CustomModel:M", "colon")]
            rr.admit_robustness_cohort_course(course, patient_id="P1", course_id="C1", rob_config=rr.RobustnessConfig(enabled=True))


def test_public_aggregate_handles_upstream_failed_and_gates(tmp_path, monkeypatch):
    c = _build_courses(tmp_path, monkeypatch, {("P1", "C1"): "unmatched"})
    c.sentinel("P1").write_text("failed\n")
    manifest = _write_manifest(c)
    _, cohort = read_course_manifest(manifest, output_dir=c.output_root, require_current_schema=True)
    cfg = rr.RobustnessConfig(enabled=True)
    courses = [rr.admit_robustness_cohort_course(c.course_dir("P1"), patient_id="P1", course_id="C1", rob_config=cfg, campaign_mode=True)]
    rr._verify_cohort_snapshot(courses, cohort, cfg, campaign_mode=True)
    output = tmp_path / "summary.xlsx"
    rr.aggregate_robustness_cohort(courses, output, cfg, cohort=cohort, campaign_mode=True, campaign_min_completion_fraction=0)
    assert pd.read_excel(output, sheet_name="course_outcomes").measurement_outcome.tolist() == ["upstream_failed"]
    with pytest.raises(RuntimeError, match="completion gate"):
        rr.aggregate_robustness_cohort(courses, output, cfg, cohort=cohort, campaign_mode=True, campaign_require_all_courses=True)
    assert not output.exists()


@pytest.mark.parametrize("text", ["ok\n", "not_applicable\n", '"ok"', "garbage"])
def test_unvalidated_text_never_overrides_failure(tmp_path, monkeypatch, text):
    c = _build_courses(tmp_path, monkeypatch, {("P1", "C1"): "unmatched"})
    ledger = _ledger()
    ledger.record(c.output_root, "P1", "C1", "radiomics_robustness", "failed", returncode=1)
    c.sentinel("P1").write_text(text)
    assert ledger._sentinel_status(c.course_dir("P1"), ".radiomics_robustness_done") == "missing"
    assert ledger.rollup(c.output_root)["stage_counts"]["radiomics_robustness"] == {"failed": 1}


@pytest.mark.parametrize("status", ["not_applicable", "technically_missing"])
def test_new_receipt_replaces_all_record_metadata(tmp_path, monkeypatch, status):
    c = _course(tmp_path, monkeypatch)
    ledger = _ledger()
    ledger.record(c.output_root, "P1", "C1", "radiomics_robustness", status, returncode=0, detail="stale reason", log_path="stale.log", duration_seconds=42)
    assert _run_course(c, "P1", campaign=True) == 0
    ledger.rollup(c.output_root)
    with (ledger.ledger_dir(c.output_root) / "campaign_ledger.csv").open() as handle:
        row = next(r for r in csv.DictReader(handle) if r["stage"] == "radiomics_robustness")
    assert row["detail"] == rr.ROBUSTNESS_NO_TARGET_OUTCOME and row["status_source"] == "sentinel"
    assert not row["log_path"] and not row["finished_at"] and not row["duration_seconds"]


def test_migration_quarantines_only_obsolete_across_output_roots_and_dag(tmp_path):
    dependency = tmp_path / "dependency.json"
    from rtpipeline.config_dependencies import stage_dependency_record
    dependency.write_text(json.dumps(stage_dependency_record("radiomics-robustness", {})))
    os.utime(dependency, ns=(1_000_000_000, 1_000_000_000))
    stamp = dependency.stat().st_mtime_ns
    for root_name in ("cohortA", "cohortB"):
        root = tmp_path / root_name
        receipts = []
        for index in range(4):
            receipt = root / f"P{index}/C1/.radiomics_robustness_done"
            receipt.parent.mkdir(parents=True)
            receipt.write_text(json.dumps({"measurement_outcome": "not_applicable_selection_unmatched" if index == 0 else "measured"}))
            os.utime(receipt, ns=(2_000_000_000, 2_000_000_000))
            receipts.append(receipt)
        valid = [(p.read_bytes(), p.stat().st_mtime_ns) for p in receipts[1:]]
        assert migrate_legacy_robustness_receipts(root, dependency) == 1
        assert not receipts[0].exists()
        assert [(p.read_bytes(), p.stat().st_mtime_ns) for p in receipts[1:]] == valid
        assert dependency.stat().st_mtime_ns == stamp
        quarantined = list(root.rglob("receipt.json"))
        assert len(quarantined) == 1 and json.loads(quarantined[0].read_text())["measurement_outcome"] == "not_applicable_selection_unmatched"
        assert len(list(root.rglob("record.json"))) == 1
        snakefile = root / "dag.smk"
        snakefile.write_text(
            'rule all:\n    input: expand("P{p}/C1/.radiomics_robustness_done", p=range(4))\n'
            f'rule course:\n    input: {str(dependency)!r}\n'
            '    output: "P{p}/C1/.radiomics_robustness_done"\n'
            '    shell: "echo run-course-{wildcards.p}"\n'
        )
        result = subprocess.run([sys.executable, "-m", "snakemake", "-n", "-p", "--nolock", "--cores", "1", "-s", str(snakefile), "--directory", str(root)], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "run-course-0" in result.stdout + result.stderr
        assert all(f"run-course-{i}" not in result.stdout + result.stderr for i in range(1, 4))


def test_aggregate_dag_parameters_bind_campaign_gates():
    compiled, _, _ = parse_snakefile_source(SNAKEFILE_PATH.read_text())
    tree = ast.parse(compiled)
    fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "__rule_aggregate_radiomics_robustness")
    params = next(dec for dec in fn.decorator_list if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr == "params")
    values = {kw.arg: kw.value for kw in params.keywords}
    for key in ("campaign_mode", "campaign_require_all_courses", "campaign_min_completion_fraction"):
        assert isinstance(values.get(key), ast.Name) and values[key].id == key.upper()


def test_source_aware_deciding_rule(tmp_path, monkeypatch):
    c = _course(tmp_path, monkeypatch, "colon")
    roi = rr._robustness_target_inventory({"segmentation_source": "AutoRTS_total", "source_path": str(c.course_dir("P1") / "RS.dcm")})[0]
    assert roi["classification_rule"].startswith("class_map:TotalSegmentator:total:2.12.0:")


@pytest.mark.parametrize("source", ["Manual", "Custom", "AutoRTS_total", "CustomModel:M"])
@pytest.mark.parametrize("presence", [False, True])
def test_classifier_producer_and_reader_agreement_table(tmp_path, monkeypatch, source, presence):
    from rtpipeline.radiomics_ct_contract import load_custom_structure_provenance

    c = _course(tmp_path, monkeypatch)
    path = c.course_dir("P1") / "RS.dcm"
    template = pydicom.dcmread(path)
    names = ["CTVN_Prostate", "GTVm - lopatka", "GTVm_SUMA", "GTVn4p", "Jelita w PTV", "Jelita + PTV1",
             "PTV1-poprzednio", "GTV3_prechemo", "z_PTV1", "2cm od PTV1", "pelvic_bones_3mm", "iliac_area", "HR-CTV", "ITV", "colon"]
    provenance = load_custom_structure_provenance(Path(rr.__file__).parent / "custom_structures_pelvic.yaml")
    ds = copy.deepcopy(template)
    ds.StructureSetROISequence, ds.ROIContourSequence = [], []
    for i, name in enumerate(names, 1):
        roi, contours = copy.deepcopy(template.StructureSetROISequence[0]), copy.deepcopy(template.ROIContourSequence[0])
        roi.ROIName, roi.ROINumber, contours.ReferencedROINumber = name, i, i
        if not presence:
            contours.ContourSequence = []
        ds.StructureSetROISequence.append(roi)
        ds.ROIContourSequence.append(contours)
    ds.save_as(path)
    binding = {"segmentation_source": source, "source_path": str(path), "custom_provenance": provenance}
    inventory = rr._robustness_target_inventory(binding)
    voxel_bindings = []
    for name in names:
        image_path = tmp_path / f"{name}.nii.gz"
        sitk.WriteImage(sitk.GetImageFromArray(np.full((2, 2, 2), presence, np.uint8)), str(image_path))
        voxel_bindings.append({"segmentation_source": source, "source_path": str(image_path), "roi_name": name, "custom_provenance": provenance, "sha256": rr._file_sha256(image_path)})
    voxels = {r["roi_name"]: r for r in rr._robustness_voxel_target_inventory(voxel_bindings)}
    for roi in inventory:
        classified = rr.classify_robustness_roi(source, roi["roi_name"], has_contours=presence, volumetric=presence, custom_provenance=provenance)
        if roi["roi_name"] in {"PTV1-poprzednio", "GTV3_prechemo", "z_PTV1", "2cm od PTV1"}:
            assert not classified["in_publication_base"] and not classified["detector_rules"]
        assert roi["volumetric"] == classified["volumetric"] == voxels[roi["roi_name"]]["volumetric"]
        assert roi["detector_rules"] == classified["detector_rules"] == voxels[roi["roi_name"]]["detector_rules"]
        for mode in ("patterns", "targets"):
            expected = presence and classified["in_publication_base"] and rr.robustness_selection_matches(
                roi["roi_name"], ["PTV*", "GTV*"], target_selection=mode, segmentation_source=source, custom_provenance=provenance)
            actual = roi["volumetric"] and rr.robustness_selection_matches(
                roi["roi_name"], ["PTV*", "GTV*"], target_selection=mode, segmentation_source=source, custom_provenance=provenance)
            assert actual == expected


    for bindings, voxel_sources in (([{**rr._rtstruct_source_binding(source, path), "custom_provenance": provenance}], []),
                                     ([], voxel_bindings)):
        config = rr.RobustnessConfig(enabled=True)
        config.perturbation.apply_to_structures = ["Absent"]
        effective = rr.effective_robustness_configuration(config, output_name="radiomics_robustness_ct.parquet")
        inventory = rr._robustness_bound_inventory(bindings, voxel_sources, configuration=effective)
        missed = [roi for roi in inventory if roi["detector_rules"]]
        rr._write_robustness_source_dispositions(
            c.course_dir("P1"), run_identifier="agreement-table", rows=[], source_bindings=bindings,
            effective_configuration=effective, code_identity=rr._current_robustness_code_identity(),
            output_path=c.course_dir("P1") / "radiomics_robustness_ct.parquet", measured_output=None,
            nonmeasured_outcome=rr.ROBUSTNESS_SELECTION_GAP_OUTCOME if missed else rr.ROBUSTNESS_NO_TARGET_OUTCOME,
            voxel_source_bindings=voxel_sources,
            source_only_basis={"selected_structure_count": 0, "unresolved_identity_count": 0,
                               "source_disposition_count": 0, "requested_selection": ["Absent"],
                               "missed_targets": missed, "reason": "No source ROI matches the requested patterns."},
        )
        rr._read_robustness_source_dispositions(c.course_dir("P1"), run_identifier="agreement-table", rob_config=config, output_name="radiomics_robustness_ct.parquet")

    # Execute production as well as its pure selection rule. Isolate the source
    # resolver, while keeping the RTSTRUCT reader, publication and admission real.
    custom_cfg = tmp_path / "custom.yaml"
    custom_cfg.write_bytes((Path(rr.__file__).parent / "custom_structures_pelvic.yaml").read_bytes())
    monkeypatch.setattr(rm, "_standard_rtstruct_sources", lambda *a: [(source, path, None)])
    monkeypatch.setattr("rtpipeline.custom_structures_rtstruct._is_rs_custom_stale", lambda *a: False)
    actual = rr.run_robustness_course(
        SimpleNamespace(custom_structures_config=custom_cfg), config, c.course_dir("P1"),
        output_path=c.course_dir("P1") / "radiomics_robustness_ct.parquet",
    )
    produced = json.loads(rr.robustness_source_dispositions_path(c.course_dir("P1")).read_text())
    expected = rr._robustness_target_inventory(binding)
    assert produced["target_inventory"] == expected
    assert actual.measurement_outcome == (rr.ROBUSTNESS_SELECTION_GAP_OUTCOME if any(r["detector_rules"] for r in expected) else rr.ROBUSTNESS_NO_TARGET_OUTCOME)


def test_selection_docs_match_governed_contract():
    root = Path(rr.__file__).parents[1]
    for relative in ("docs/features/radiomics_robustness.md", "docs/user_guide/campaigns.md", "docs/RADIOMICS_ROBUSTNESS.md"):
        text = (root / relative).read_text()
        assert "The next character\n  must not be a letter" not in text
        assert "token cannot be followed by a letter" not in text
        assert "`GTV`, `CTV`, `PTV` and `ITV`" in text
        assert "selection_gap_targets" in text
        assert "provenance" in text and "contourless" in text.lower()
    campaigns = (root / "docs/user_guide/campaigns.md").read_text()
    assert "campaign_min_completion_fraction" in campaigns and "quarantine" in campaigns.lower()


@pytest.mark.parametrize("config_digest", [None, "configured-but-unrecorded"])
def test_certificate_refuses_unknown_custom_configuration(tmp_path, monkeypatch, main_producer, config_digest):
    course, config, _, _, main, completion = _main_artifact(tmp_path, monkeypatch, main_producer)
    before = _produce(main, completion, course, config)
    meta = course / "metadata/rs_custom_meta.json"
    meta.write_text(json.dumps({"custom_config_sha256": config_digest}))
    assert not rr._compatible_main_measured_code(before[1], rr._current_robustness_code_identity(), course_dir=course)


def test_benign_voxel_omission_never_hides_a_later_target(tmp_path, monkeypatch):
    course, config, _, _ = _layout(tmp_path, monkeypatch, "benign_voxel_model")
    _produce(rr, rc, course, config)
    path = course / "Segmentation_CustomModels/M/colon.nii.gz"
    path.rename(path.with_name("HR-CTV.nii.gz"))
    with pytest.raises(ValueError, match="unrecorded voxel source"):
        rr.admit_robustness_cohort_course(course, patient_id="P1", course_id="C1", rob_config=rr.RobustnessConfig(enabled=True))


@pytest.mark.parametrize("name", ["PTV1-poprzednio", "GTV3_prechemo"])
def test_measured_dicom_course_ignores_selected_inventory_helper(tmp_path, monkeypatch, name):
    course, config, _, _ = _layout(tmp_path, monkeypatch, "voxel_model")
    path = course / "Segmentation_CustomModels/M/GTV.nii.gz"
    path.rename(path.with_name(f"{name}.nii.gz"))
    cfg = rr.RobustnessConfig(enabled=True)
    cfg.perturbation.apply_to_structures = ["PTV*", "GTV*", "BLADDER"]
    result = rr.run_robustness_course(config, cfg, course, output_path=course / "radiomics_robustness_ct.parquet")
    assert result.measurement_outcome == "measured"
    frame = pd.read_parquet(course / "radiomics_robustness_ct.parquet")
    assert set(zip(frame.segmentation_source, frame.roi_original_name)) == {("Manual", "PTV"), ("AutoRTS_total", "BLADDER")}
