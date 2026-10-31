"""Synthetic regressions for course selection and campaign closure."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
import yaml

from rtpipeline import cli, radiomics as rm, radiomics_robustness as rr
from test_robustness_consumer_integration import (
    _build_courses,
    _run_course,
    _run_aggregate,
    _write_manifest,
)
from test_robustness_selection_unmatched import _ledger


@pytest.mark.parametrize(
    "name,target",
    [
        ("CTVN_Prostate", True),
        ("GTVm - lopatka", True),
        ("GTVm_SUMA", True),
        ("GTVn4p", True),
        ("Jelita w PTV", False),
        ("Jelita + PTV1", False),
        ("2cm od PTV1", False),
        ("z_PTV1", False),
        ("PTV1-poprzednio", False),
        ("CTVn", True),
        ("GTVp", True),
        ("ITVn", True),
        ("CTVx", False),
    ],
)
def test_governed_detector(name, target):
    assert bool(rr.detect_robustness_target(name)) == target


def test_itv_type():
    assert rr.detect_robustness_target("InternalTarget", "ITV")


def _course(
    tmp_path, monkeypatch, name="ODBYTNICA", *, mode="patterns", contour="volume"
):
    reader, resolver = rm._rtstruct_masks, rm._standard_rtstruct_sources
    cohort = _build_courses(
        tmp_path,
        monkeypatch,
        {("P1", "C1"): "unmatched"},
        apply_to_structures=["GTV*", "CTV*", "PTV*"],
    )
    path = cohort.course_dir("P1") / "RS.dcm"
    ds = pydicom.dcmread(path)
    ds.StructureSetROISequence[0].ROIName = name
    if contour == "empty":
        ds.ROIContourSequence[0].ContourSequence = []
    elif contour == "point":
        item = ds.ROIContourSequence[0].ContourSequence[0]
        item.ContourGeometricType = "POINT"
        item.ContourData = item.ContourData[:3]
        item.NumberOfContourPoints = 1
        ds.ROIContourSequence[0].ContourSequence = [item]
    ds.save_as(path)
    ct = tmp_path / "_template/CT"
    monkeypatch.setattr(
        "rtpipeline.radiomics_rasterizer.require_radiomics_rasterizer", lambda: None
    )
    monkeypatch.setattr(rm, "_rtstruct_masks", reader)
    monkeypatch.setattr(rm, "_standard_rtstruct_sources", resolver)
    monkeypatch.setattr(
        rr,
        "load_course_contract",
        lambda _: SimpleNamespace(
            planning_ct_dir=ct,
            planning_ct={"series_instance_uid": "1.2.3"},
            authoritative_rtstruct_path=path,
            authoritative_rtstruct_source="Manual",
            planning_ct_nifti=None,
        ),
    )
    data = yaml.safe_load(cohort.config_path.read_text())
    data["radiomics_robustness"]["target_selection"] = mode
    cohort.config_path.write_text(yaml.safe_dump(data))
    return cohort


@pytest.mark.parametrize(
    "name,contour", [("GTV_auto", "empty"), ("PTV1-poprzednio", "volume")]
)
def test_nonmeasurable_selected_roi(tmp_path, monkeypatch, name, contour):
    c = _course(tmp_path, monkeypatch, name, contour=contour)
    assert _run_course(c, "P1", campaign=True) == 0
    assert (
        c.receipt("P1")["measurement_outcome"] == "not_applicable_no_target_structure"
        if contour == "volume"
        else c.receipt("P1")["measurement_outcome"] == "source_only_nonvolumetric"
    )


def test_point_does_not_hide_target(tmp_path, monkeypatch):
    c = _course(tmp_path, monkeypatch, "GTV_ref", contour="point")
    path = c.course_dir("P1") / "RS.dcm"
    ds = pydicom.dcmread(path)
    template = pydicom.dcmread(tmp_path / "_template/RS.dcm")
    roi = copy.deepcopy(template.StructureSetROISequence[0])
    roi.ROINumber = 2
    roi.ROIName = "HR-CTV"
    contours = copy.deepcopy(template.ROIContourSequence[0])
    contours.ReferencedROINumber = 2
    ds.StructureSetROISequence.append(roi)
    ds.ROIContourSequence.append(contours)
    ds.save_as(path)
    assert _run_course(c, "P1", campaign=True) == 0
    assert c.receipt("P1")["measurement_outcome"] == "selection_gap_targets_unselected"


@pytest.mark.parametrize("corrupt", [False, True])
def test_voxel_zero_or_unreadable(tmp_path, monkeypatch, corrupt):
    c = _build_courses(tmp_path, monkeypatch, {("P1", "C1"): "unmatched"})
    path = c.course_dir("P1") / (
        "Segmentation_TotalSegmentator/colon.nii.gz"
        if corrupt
        else "Segmentation_TotalSegmentator/ROI.nii.gz"
    )
    path.parent.mkdir()
    if corrupt:
        path.write_bytes(b"broken nifti")
    else:
        sitk.WriteImage(
            sitk.GetImageFromArray(np.zeros((17, 17, 17), np.uint8)), str(path)
        )
    result = _run_course(c, "P1", campaign=True)
    assert result == (1 if corrupt else 0)
    assert c.receipt("P1")["measurement_outcome"] == (
        "failed_extraction" if corrupt else "not_applicable_no_target_structure"
    )


def test_targets_empty_cli(tmp_path, monkeypatch):
    c = _course(tmp_path, monkeypatch, mode="targets")
    data = yaml.safe_load(c.config_path.read_text())
    data["radiomics_robustness"]["segmentation_perturbation"][
        "apply_to_structures"
    ] = []
    c.config_path.write_text(yaml.safe_dump(data))
    assert _run_course(c, "P1", campaign=True) == 0
    assert _run_aggregate(c, _write_manifest(c), tmp_path / "summary.xlsx") == 0


def test_rollup_validates_measured_receipt(tmp_path, monkeypatch):
    c = _build_courses(tmp_path, monkeypatch, {("P1", "C1"): "measured"})
    assert _run_course(c, "P1") == 0
    c.table("P1").write_bytes(b"tampered")
    assert _ledger().rollup(c.output_root)["stage_counts"]["radiomics_robustness"] == {
        "missing": 1
    }


def test_default_identity_matches_main_configuration():
    cfg = rr.RobustnessConfig()
    effective = rr.effective_robustness_configuration(
        cfg, output_name="radiomics_robustness_ct.parquet"
    )
    assert "target_selection" not in effective["robustness"]
    assert (
        rr._robustness_configured_parameter_hash("extractor", "patterns") == "extractor"
    )


def test_receipt_scan_is_cached(tmp_path, monkeypatch):
    from rtpipeline.config_dependencies import migrate_legacy_robustness_receipts

    dependency = tmp_path / "dependency.json"
    dependency.write_text("{}")
    migrate_legacy_robustness_receipts(tmp_path, dependency)
    monkeypatch.setattr(
        Path, "glob", lambda *a: pytest.fail("cached startup scanned receipts")
    )
    migrate_legacy_robustness_receipts(tmp_path, dependency)


@pytest.mark.parametrize("mode", ["patterns", "targets"])
@pytest.mark.parametrize(
    "name", ["z_PTV1", "Jelita - PTV", "2cm od PTV1", "PTV1-poprzednio"]
)
def test_real_helper_reader(tmp_path, monkeypatch, mode, name):
    c = _course(tmp_path, monkeypatch, name, mode=mode)
    assert _run_course(c, "P1", campaign=True) == 0
    assert (
        c.receipt("P1")["measurement_outcome"] == "not_applicable_no_target_structure"
    )


@pytest.mark.parametrize("nifti_ct", [False, True])
def test_all_sources_inventoried(tmp_path, monkeypatch, nifti_ct):
    c = _course(tmp_path, monkeypatch, "HR-CTV")
    course = c.course_dir("P1")
    path = course / "Segmentation_CustomModels/M/CTVn.nii.gz"
    path.parent.mkdir(parents=True)
    sitk.WriteImage(sitk.GetImageFromArray(np.ones((17, 17, 17), np.uint8)), str(path))
    if nifti_ct:
        ct = course / "ct.nii.gz"
        sitk.WriteImage(
            sitk.GetImageFromArray(np.ones((17, 17, 17), np.float32)), str(ct)
        )
        monkeypatch.setattr(rm, "_load_series_image", lambda _: None)
        monkeypatch.setattr(
            rr,
            "load_course_contract",
            lambda _: SimpleNamespace(
                planning_ct_dir=None,
                planning_ct_nifti=ct,
                planning_ct={"series_instance_uid": "1.2.3"},
                authoritative_rtstruct_path=course / "RS.dcm",
                authoritative_rtstruct_source="Manual",
            ),
        )
    # Unselect both targets. No rasterization or main identity is needed.
    data = yaml.safe_load(c.config_path.read_text())
    data["radiomics_robustness"]["segmentation_perturbation"]["apply_to_structures"] = [
        "Absent"
    ]
    c.config_path.write_text(yaml.safe_dump(data))
    assert _run_course(c, "P1", campaign=True) == 0
    assert {
        (r["segmentation_source"], r["roi_name"])
        for r in c.receipt("P1")["missed_targets"]
    } == {("Manual", "HR-CTV"), ("CustomModel:M", "CTVn")}


@pytest.mark.parametrize("source", ["model", "custom"])
def test_generated_contourless_and_copied_rois(tmp_path, monkeypatch, source):
    from rtpipeline import custom_models, custom_structures_rtstruct

    c = _course(tmp_path, monkeypatch)
    course = c.course_dir("P1")
    ds = pydicom.dcmread(course / "RS.dcm")
    roi = copy.deepcopy(ds.StructureSetROISequence[0])
    roi.ROINumber = 2
    roi.ROIName = "GTV_auto"
    ds.StructureSetROISequence.append(roi)
    contour = pydicom.Dataset()
    contour.ReferencedROINumber = 2
    contour.ContourSequence = []
    ds.ROIContourSequence.append(contour)
    if source == "custom":
        path = course / "RS_custom.dcm"
        monkeypatch.setattr(
            custom_structures_rtstruct, "_is_rs_custom_stale", lambda *a: False
        )
        # Main's catalog defines the independent Custom declaration. Base copies
        # are absent from that namespace and must not duplicate Manual targets.
        identity = rr.RobustnessRoiIdentity.from_mapping(
            dict(
                patient_id="P1",
                course_id="C1",
                series_uid="1.2.3",
                segmentation_source="Custom",
                mask_identity="empty-slot",
                roi_original_name="GTV_auto",
                stable_roi_identifier="custom-slot",
            )
        )
        monkeypatch.setattr(
            rr,
            "_load_main_ct_identity_catalog",
            lambda *a, **k: ({("Custom", "GTV_auto"): identity}, {}),
        )
    else:
        path = course / "Segmentation_CustomModels/M/rtstruct.dcm"
        path.parent.mkdir(parents=True)
        monkeypatch.setattr(
            custom_models, "list_custom_model_outputs", lambda _: [("M", path.parent)]
        )
    ds.save_as(path)
    assert _run_course(c, "P1", campaign=True) == 0
    assert c.receipt("P1")["measurement_outcome"] in {
        "source_only_nonvolumetric",
        "not_applicable_no_target_structure",
    }
    sidecar = json.loads(rr.robustness_source_dispositions_path(course).read_text())
    assert all(not r["detector_rules"] for r in sidecar["target_inventory"])
    if source == "custom":
        assert [
            (r["segmentation_source"], r["roi_name"])
            for r in sidecar["target_inventory"]
        ] == [("Manual", "ODBYTNICA"), ("Custom", "GTV_auto")]


def test_campaign_gates_and_bare_failed(tmp_path, monkeypatch):
    c = _build_courses(tmp_path, monkeypatch, {("P1", "C1"): "unmatched"})
    c.sentinel("P1").write_text("failed\n")
    manifest = _write_manifest(c)
    output = tmp_path / "summary.xlsx"
    args = [
        "radiomics-robustness-aggregate",
        "--manifest",
        str(manifest),
        "--output-root",
        str(c.output_root),
        "--output",
        str(output),
        "--config",
        str(c.config_path),
        "--campaign-mode",
        "--campaign-min-completion-fraction",
        "0",
    ]
    assert cli.main(args) == 0
    import pandas as pd

    row = pd.read_excel(output, sheet_name="course_outcomes").iloc[0]
    assert row.measurement_outcome == "upstream_failed" and row.technically_missing
    assert cli.main(args + ["--campaign-require-all-courses"]) == 1
    assert not output.exists()
    assert cli.main(args[:-1] + ["0.5"]) == 1
    assert not output.exists()


def test_receipt_only_failure_and_stale_record(tmp_path, monkeypatch):
    c = _build_courses(tmp_path, monkeypatch, {("P1", "C1"): "measured"})
    monkeypatch.setattr(
        rr,
        "extract_features_for_masks",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )
    assert _run_course(c, "P1", campaign=True) == 1
    assert "target_selection" not in c.receipt("P1")
    ledger = _ledger()
    assert ledger.rollup(c.output_root)["stage_counts"]["radiomics_robustness"] == {
        "failed": 1
    }
    ledger.record(
        c.output_root,
        "P1",
        "C1",
        "radiomics_robustness",
        "technically_missing",
        returncode=1,
        detail="old gap",
    )
    # A subsequent successful no-target run supersedes every stale record field.
    data = yaml.safe_load(c.config_path.read_text())
    data["radiomics_robustness"]["segmentation_perturbation"]["apply_to_structures"] = [
        "Absent"
    ]
    c.config_path.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(rm, "_rtstruct_masks", lambda *a, **k: {})
    assert _run_course(c, "P1", campaign=True) == 0
    assert ledger.rollup(c.output_root)["stage_counts"]["radiomics_robustness"] == {
        "not_applicable": 1
    }


def test_main_receipt_reused_byte_for_byte(tmp_path, monkeypatch):
    """Run the actual main producer, then admit its untouched artifacts here."""
    import hashlib
    import subprocess
    import sys
    import types

    root = Path(__file__).resolve().parents[1]
    baseline = tmp_path / "main-code"
    baseline.mkdir()
    for relative in rr.ROBUSTNESS_DISPOSITION_CODE_SOURCES:
        content = subprocess.check_output(
            ["git", "show", f"072b146:rtpipeline/{relative}"], cwd=root
        )
        (baseline / relative).write_bytes(content)
    name = "rtpipeline._main_producer_fixture"
    module = types.ModuleType(name)
    module.__file__ = str(baseline / "radiomics_robustness.py")
    monkeypatch.setitem(sys.modules, name, module)
    exec(
        compile(Path(module.__file__).read_bytes(), module.__file__, "exec"),
        module.__dict__,
    )
    c = _build_courses(tmp_path, monkeypatch, {("P1", "C1"): "measured"})
    for attr in (
        "load_course_contract",
        "_load_main_ct_identity_catalog",
        "extract_features_for_masks",
    ):
        monkeypatch.setattr(module, attr, getattr(rr, attr))
    with monkeypatch.context() as scoped:

        def main_run(config, rob_config, course_dir, **kwargs):
            old_config = module.RobustnessConfig.from_dict(
                yaml.safe_load(c.config_path.read_text())["radiomics_robustness"]
            )
            return module.run_robustness_course(
                config, old_config, course_dir, **kwargs
            )

        scoped.setattr(rr, "run_robustness_course", main_run)
        assert _run_course(c, "P1") == 0
    paths = [
        c.table("P1"),
        c.sentinel("P1"),
        rr.robustness_source_dispositions_path(c.course_dir("P1")),
    ]
    before = {str(path): path.read_bytes() for path in paths}
    admitted = rr.admit_robustness_cohort_course(
        c.course_dir("P1"),
        patient_id="P1",
        course_id="C1",
        rob_config=rr.RobustnessConfig.from_dict(
            yaml.safe_load(c.config_path.read_text())["radiomics_robustness"]
        ),
    )
    assert admitted.measured
    assert {str(path): path.read_bytes() for path in paths} == before
    # Keep reproducible byte evidence in pytest's captured output.
    print(
        {
            Path(path).name: hashlib.sha256(value).hexdigest()
            for path, value in before.items()
        }
    )
    payload = json.loads(before[str(paths[-1])])
    current = rr._current_robustness_code_identity()
    current["sources_sha256"] = "dirty-code"
    assert not rr._compatible_main_measured_code(payload, current)


def test_newer_disagreeing_record_is_conflict(tmp_path, monkeypatch):
    c = _build_courses(tmp_path, monkeypatch, {("P1", "C1"): "unmatched"})
    assert _run_course(c, "P1") == 0
    ledger = _ledger()
    ledger.record(
        c.output_root,
        "P1",
        "C1",
        "radiomics_robustness",
        "failed",
        returncode=1,
        detail="newer attempt failed",
    )
    assert ledger.rollup(c.output_root)["stage_counts"]["radiomics_robustness"] == {
        "failed": 1
    }
    import pandas as pd

    row = (
        pd.read_csv(c.output_root / "_campaign_ledger/campaign_ledger.csv")
        .query("stage == 'radiomics_robustness'")
        .iloc[0]
    )
    assert (
        row.status_source == "record-sentinel-conflict"
        and row.detail == "record-sentinel-conflict"
    )


def test_campaign_config_enforces_completion_gate(tmp_path, monkeypatch):
    c = _build_courses(tmp_path, monkeypatch, {("P1", "C1"): "unmatched"})
    c.sentinel("P1").write_text("failed\n")
    data = yaml.safe_load(c.config_path.read_text())
    data.update(campaign_mode=True, campaign_require_all_courses=True)
    c.config_path.write_text(yaml.safe_dump(data))
    assert _run_aggregate(c, _write_manifest(c), tmp_path / "summary.xlsx") == 1


def test_custom_target_without_published_identity_is_inventoried(tmp_path, monkeypatch):
    from rtpipeline import custom_structures_rtstruct
    c = _course(tmp_path, monkeypatch)
    course = c.course_dir('P1')
    ds = pydicom.dcmread(course / 'RS.dcm')
    ds.StructureSetROISequence[0].ROIName = 'HR-CTV'
    ds.save_as(course / 'RS_custom.dcm')
    monkeypatch.setattr(custom_structures_rtstruct, '_is_rs_custom_stale', lambda *a: False)
    assert _run_course(c, 'P1', campaign=True) == 0
    assert [(r['segmentation_source'], r['roi_name']) for r in c.receipt('P1')['missed_targets']] == [('Custom', 'HR-CTV')]


def test_measured_roi_does_not_hide_unselected_target(tmp_path, monkeypatch):
    c = _build_courses(tmp_path, monkeypatch, {('P1', 'C1'): 'measured'})
    path = c.course_dir('P1') / 'RS.dcm'
    ds = pydicom.dcmread(path)
    roi = copy.deepcopy(ds.StructureSetROISequence[0]); roi.ROINumber = 2; roi.ROIName = 'HR-CTV'
    contours = copy.deepcopy(ds.ROIContourSequence[0]); contours.ReferencedROINumber = 2
    ds.StructureSetROISequence.append(roi); ds.ROIContourSequence.append(contours); ds.save_as(path)
    assert _run_course(c, 'P1') == 0
    assert c.receipt('P1')['measurement_outcome'] == 'measured'
    ledger = _ledger()
    ledger.close_robustness_completion(c.output_root, 'P1', 'C1', c.sentinel('P1'))
    assert ledger.rollup(c.output_root)['stage_counts']['radiomics_robustness'] == {'technically_missing': 1}
    output = tmp_path / 'summary.xlsx'
    assert _run_aggregate(c, _write_manifest(c), output) == 0
    import pandas as pd
    row = pd.read_excel(output, sheet_name='course_outcomes').iloc[0]
    assert row.technically_missing and row.measured_value_row_count > 0
    ledger_row = pd.read_csv(c.output_root / '_campaign_ledger/campaign_ledger.csv').query("stage == 'radiomics_robustness'").iloc[0]
    assert ledger_row.detail == 'selection_gap_targets_unselected'


def test_mixed_voxel_and_dicom_rows_keep_their_backend(tmp_path, monkeypatch):
    import pandas as pd
    c = _build_courses(tmp_path, monkeypatch, {('P1', 'C1'): 'measured'}, apply_to_structures=['ROI', 'CTVn'])
    course = c.course_dir('P1')
    path = course / 'Segmentation_TotalSegmentator/CTVn.nii.gz'
    path.parent.mkdir()
    array = np.zeros((17, 17, 17), np.uint8)
    array[6:12, 6:12, 6:12] = 1
    sitk.WriteImage(sitk.GetImageFromArray(array), str(path))
    catalog, issues = rr._load_main_ct_identity_catalog(course)
    catalog[('AutoRTS_total', 'CTVn')] = rr.RobustnessRoiIdentity.from_mapping(dict(
        patient_id='P1', course_id='C1', series_uid='1.2.3', segmentation_source='AutoRTS_total',
        mask_identity='voxel-mask', roi_original_name='CTVn', stable_roi_identifier='voxel-target',
    ))
    monkeypatch.setattr(rr, '_load_main_ct_identity_catalog', lambda *a, **k: (catalog, issues))
    assert _run_course(c, 'P1') == 0
    frame = pd.read_parquet(c.table('P1'))
    assert set(frame.loc[frame.segmentation_source.eq('Manual'), 'rasterizer_backend']) == {'rt-utils/cv2.fillPoly'}
    assert set(frame.loc[frame.segmentation_source.eq('AutoRTS_total'), 'rasterizer_backend']) == {'nifti/no-polygon-fill'}
