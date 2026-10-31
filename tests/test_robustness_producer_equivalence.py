"""Synthetic production comparisons with the exported 072b146 producer."""

import copy
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pydicom
import pytest
import SimpleITK as sitk

from rtpipeline import radiomics as rm, radiomics_robustness as rr
from rtpipeline import robustness_completion as rc
from rtpipeline.radiomics_ct_contract import CT_EXTRACTION_ARMS
from robustness_prep_fixture import cylinder, write_ct, write_rtstruct


@pytest.fixture(autouse=True)
def one_itk_thread():
    previous = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    yield
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(previous)


@pytest.fixture
def main_producer(tmp_path, monkeypatch):
    directory = tmp_path / "main-code"
    directory.mkdir()
    for relative in (*rr.ROBUSTNESS_DISPOSITION_CODE_SOURCES, "robustness_completion.py"):
        (directory / relative).write_bytes(subprocess.check_output(
            ["git", "show", f"072b146:rtpipeline/{relative}"],
            cwd=Path(__file__).resolve().parents[1],
        ))
        if relative not in {"radiomics_robustness.py", "robustness_completion.py"}:
            # Relative imports execute shared helpers. Prove those are exactly
            # main's bytes, rather than silently comparing a hybrid producer.
            assert (Path(rr.__file__).parent / relative).read_bytes() == (directory / relative).read_bytes()
    modules = []
    for filename in ("radiomics_robustness", "robustness_completion"):
        name = f"rtpipeline._s11e_main_{filename}"
        module = types.ModuleType(name)
        module.__file__ = str(directory / f"{filename}.py")
        monkeypatch.setitem(sys.modules, name, module)
        exec(compile(Path(module.__file__).read_bytes(), module.__file__, "exec"), module.__dict__)
        modules.append(module)
    return modules


def _features(image, masks, config, **kwargs):
    """Deterministic mask-sensitive probe in place of unavailable PyRadiomics."""
    fields = kwargs["source_identity"].as_dict()
    rows = []
    for pid, mask in sorted(masks.items()):
        selected = sitk.GetArrayFromImage(mask).astype(bool)
        values = sitk.GetArrayFromImage(kwargs.get("perturbed_images", {}).get(pid, image))[selected]
        for arm in CT_EXTRACTION_ARMS:
            for feature, value in (("original_firstorder_Mean", float(values.mean())),
                                   ("original_shape_VoxelVolume", float(selected.sum()))):
                rows.append({
                    **fields, "structure": fields["roi_original_name"], "modality": "CT",
                    "measurement_type": rr.ROBUSTNESS_MEASUREMENT_TYPE,
                    "perturbed_mask_identity": rr._perturbed_mask_identity(mask),
                    "perturbation_id": pid, "extraction_arm": arm,
                    "run_identifier": kwargs["run_identifier"], "feature_name": feature,
                    "configured_parameter_hash": f"synthetic-probe-{arm}", "value": value,
                })
    return pd.DataFrame(rows)


def _layout(tmp_path, monkeypatch, kind="normal", *, missing_identity=False, export=False):
    from rtpipeline import auto_rtstruct, custom_structures_rtstruct
    from rasterizer_test_utils import stub_synthetic_rasterizer_columns

    stub_synthetic_rasterizer_columns(monkeypatch)
    monkeypatch.setattr("rtpipeline.radiomics_rasterizer.require_radiomics_rasterizer", lambda: None)
    course = tmp_path / "P1/C1"
    course.mkdir(parents=True)
    ct = course / "CT"
    uid = write_ct(ct, rows=31, columns=31, slices=13, spacing=(1, 1, 2))
    image = rm._load_series_image(ct)
    mask = cylinder((31, 31, 13), (15, 15, 6), (7, 6, 3))
    write_rtstruct(ct, course / "RS.dcm", [("PTV", mask)])
    write_rtstruct(ct, course / "RS_auto.dcm", [("BLADDER", np.roll(mask, 2, axis=0))])
    seg = course / "Segmentation_TotalSegmentator"
    seg.mkdir()
    nii_mask = rm._mask_from_array_like(image, mask)
    sitk.WriteImage(nii_mask, str(seg / "BLADDER.nii.gz"))
    # A published voxel identity omitted by RS_auto must not enter a DICOM pass.
    sitk.WriteImage(nii_mask, str(seg / "CTV.nii.gz"))
    identities = [("Manual", "PTV"), ("AutoRTS_total", "BLADDER"), ("AutoRTS_total", "CTV")]
    if kind == "custom":
        write_rtstruct(ct, course / "RS_custom.dcm", [("PTV", mask), ("GTV", mask)])
        identities.append(("Custom", "GTV"))
        monkeypatch.setattr(custom_structures_rtstruct, "_is_rs_custom_stale", lambda *a: False)
    if kind in {"model", "voxel_model"}:
        model = course / "Segmentation_CustomModels/M"
        model.mkdir(parents=True)
        if kind == "model":
            write_rtstruct(ct, model / "rtstruct.dcm", [("GTV", mask)])
        sitk.WriteImage(nii_mask, str(model / "GTV.nii.gz"))
        identities.append(("CustomModel:M", "GTV"))
    if kind == "fallback":
        for path in (course / "RS.dcm", course / "RS_auto.dcm"):
            path.unlink()
        identities = [("AutoRTS_total", "BLADDER"), ("AutoRTS_total", "CTV")]
    nifti = course / "ct.nii.gz"
    if kind == "nifti_ct":
        sitk.WriteImage(image, str(nifti))
        if export:
            from rtpipeline import organize, nifti_provenance
            from test_s9c_repairs import write_sidecar
            write_sidecar(nifti)
            sidecar = nifti.with_name("ct.metadata.json")
            provenance = json.loads(sidecar.read_text())
            provenance.update(series_instance_uid=uid, nifti_geometry=nifti_provenance.nifti_geometry(nifti))
            sidecar.write_text(json.dumps(provenance))
            assert organize._export_original_segmentation_from_paths(
                rs_path=course / "RS.dcm", primary_nifti=nifti, dicom_ct_dir=ct,
                segmentation_original_dir=course / "Segmentation_Original",
                log_root=course, overwrite=False,
            )
        import shutil
        shutil.rmtree(ct)
        (course / "RS_auto.dcm").unlink()
        identities = [] if missing_identity else [("Manual", "PTV")]
    contract = SimpleNamespace(
        planning_ct_dir=None if kind == "nifti_ct" else ct,
        planning_ct_nifti=nifti if kind == "nifti_ct" else None,
        planning_ct={"series_instance_uid": uid},
        authoritative_rtstruct_path=course / "RS.dcm" if (course / "RS.dcm").exists() else None,
        authoritative_rtstruct_source="Manual",
    )
    catalog = {(source, name): rr.RobustnessRoiIdentity.from_mapping({
        "patient_id": "P1", "course_id": "C1", "series_uid": uid,
        "segmentation_source": source, "roi_original_name": name,
        "mask_identity": f"mask-{source}-{name}", "stable_roi_identifier": f"roi-{source}-{name}",
    }) for source, name in identities}
    monkeypatch.setattr(rr, "load_course_contract", lambda _: contract)
    monkeypatch.setattr(rr, "_load_main_ct_identity_catalog", lambda *a, **k: (catalog, {}))
    monkeypatch.setattr(rr, "extract_features_for_masks", _features)
    monkeypatch.setattr(auto_rtstruct, "_derived_rtstruct_dependencies_are_current", lambda *a, **k: kind != "stale_auto")
    monkeypatch.setenv("RTPIPELINE_DISABLE_PARALLEL_RADIOMICS", "1")
    monkeypatch.setenv("RTPIPELINE_RADIOMICS_THREAD_LIMIT", "1")
    monkeypatch.setenv("RTPIPELINE_MAX_WORKERS", "1")
    return course, SimpleNamespace(custom_structures_config=None), contract, catalog


def _produce(module, completion, course, config):
    output = course / "radiomics_robustness_ct.parquet"
    cfg = module.RobustnessConfig(enabled=True)  # All measurement/selection defaults.
    result = module.robustness_for_course(config, cfg, course, output_path=output)
    if result is None:
        return None
    frame = pd.read_parquet(output)
    sidecar = json.loads(module.robustness_source_dispositions_path(course).read_text())
    receipt = completion.robustness_completion_payload(
        course, patient_id="P1", course_id="C1", run_identifier=sidecar["robustness_run_identifier"],
        measurement_outcome="measured", output_name=output.name,
        dispositions_path=module.robustness_source_dispositions_path(course),
        source_disposition_count=sidecar["row_count"],
        effective_configuration_sha256=sidecar["effective_configuration_sha256"], measured_output=output,
    )
    (course / ".radiomics_robustness_done").write_text(json.dumps(receipt))
    ledger = json.loads((course / "metadata/radiomics_robustness_identity.json").read_text())
    return frame, sidecar, ledger, receipt


# These are the ONLY semantic exclusions. Container hashes and sizes below are
# recomputed from normalized content, so a changed field is never hidden by
# ignoring the checksum of its containing artifact.
EXCLUDED_FRAME_FIELDS = {"run_identifier", "code_revision"}
EXCLUDED_SIDECAR_FIELDS = {"robustness_run_identifier", "code_identity"}
EXCLUDED_RECEIPT_FIELDS = {"robustness_run_identifier"}
# Neither producer currently emits a timestamp in these four artifacts.
EXCLUDED_TIMESTAMP_FIELDS = set()


def _canonical(result):
    frame, sidecar, ledger, receipt = copy.deepcopy(result)
    frame = frame.drop(columns=sorted(EXCLUDED_FRAME_FIELDS & set(frame)))
    for key in EXCLUDED_SIDECAR_FIELDS:
        sidecar.pop(key, None)
    for key in EXCLUDED_RECEIPT_FIELDS:
        receipt.pop(key, None)
    table_bytes = frame.to_json(orient="table", double_precision=15).encode()
    table_binding = {"path": receipt["measured_output"]["path"],
                     "sha256": hashlib.sha256(table_bytes).hexdigest(), "size_bytes": len(table_bytes)}
    sidecar["measured_output"] = table_binding
    receipt["measured_output"] = table_binding
    receipt["source_dispositions"]["sha256"] = rr._content_sha256(sidecar)
    return frame, sidecar, ledger, receipt


@pytest.mark.parametrize("kind", ["normal", "custom", "model", "stale_auto", "fallback", "nifti_ct"])
def test_main_producer_equivalence(tmp_path, monkeypatch, main_producer, kind):
    # Actual main radiomics cannot publish Manual RTSTRUCT identities without
    # DICOM CT. Layout 6 therefore exercises its real (absent-identity) state.
    course, config, _, _ = _layout(tmp_path, monkeypatch, kind, missing_identity=True)
    main, completion = main_producer
    for name in ("load_course_contract", "_load_main_ct_identity_catalog", "extract_features_for_masks"):
        monkeypatch.setattr(main, name, getattr(rr, name))
    before = _produce(main, completion, course, config)
    if kind == "nifti_ct":
        old_ledger = json.loads((course / "metadata/radiomics_robustness_identity.json").read_text())
        assert not (course / "radiomics_robustness_ct.parquet").exists()
        assert not rr.robustness_source_dispositions_path(course).exists()
    after = _produce(rr, rc, course, config)
    if kind == "nifti_ct":
        assert before is None and after is None
        assert not (course / ".radiomics_robustness_done").exists()
        assert not (course / "radiomics_robustness_ct.parquet").exists()
        assert not rr.robustness_source_dispositions_path(course).exists()
        ledger = json.loads((course / "metadata/radiomics_robustness_identity.json").read_text())
        # Main's defect: it omitted the selected clinical target entirely.
        assert {(r["segmentation_source"], r["roi_original_name"]) for r in ledger["rows"]} == (
            {(r["segmentation_source"], r["roi_original_name"]) for r in old_ledger["rows"]} | {("Manual", "PTV")})
        return
    old, new = _canonical(before), _canonical(after)
    pd.testing.assert_frame_equal(old[0], new[0], check_exact=True)
    if kind == "fallback":
        # Main defect: its sidecar and receipt omit every voxel input. These
        # are asserted additions, not excluded comparison fields.
        expected_sidecar = copy.deepcopy(old[1])
        bindings = [{"segmentation_source": source, "source_path": str(path),
                     "sha256": rr._file_sha256(path)}
                    for source, path in rr._robustness_nifti_sources(course)]
        expected_sidecar.update(voxel_source_bindings=bindings,
                                target_inventory=rr._robustness_voxel_target_inventory(bindings))
        assert new[1] == expected_sidecar
        assert old[2] == new[2]
        expected_receipt = copy.deepcopy(old[3])
        expected_receipt["source_dispositions"]["sha256"] = rr._content_sha256(expected_sidecar)
        assert new[3] == expected_receipt
    else:
        assert old[1:] == new[1:]
    # Admission must preserve an actual historical artifact byte for byte.
    _produce(main, completion, course, config)
    paths = [course / ".radiomics_robustness_done", course / "radiomics_robustness_ct.parquet",
             rr.robustness_source_dispositions_path(course)]
    snapshot = [path.read_bytes() for path in paths]
    if kind == "fallback":
        # Main never bound voxel inputs. Equal values alone cannot certify reuse.
        with pytest.raises((ValueError, rc.RobustnessCompletionError), match="different code"):
            rr.admit_robustness_cohort_course(course, patient_id="P1", course_id="C1", rob_config=rr.RobustnessConfig(enabled=True))
    else:
        assert rr.admit_robustness_cohort_course(course, patient_id="P1", course_id="C1", rob_config=rr.RobustnessConfig(enabled=True)).measured
    assert snapshot == [path.read_bytes() for path in paths]
    foreign_code = {**rr._current_robustness_code_identity(), "sources_sha256": "unlisted-digest"}
    assert not rr._compatible_main_measured_code(before[1], foreign_code, course_dir=course)
    print({"layout": kind, "rows": len(new[0]),
           "comparison": "voxel binding correction" if kind == "fallback" else "equal", "reuse": kind != "fallback"})


@pytest.mark.parametrize("kind", ["normal", "model", "stale_auto"])
def test_dicom_course_does_not_read_nifti(tmp_path, monkeypatch, kind):
    course, config, _, _ = _layout(tmp_path, monkeypatch, kind)
    # An unreadable unselected voxel file is outside this DICOM source scope.
    (course / "Segmentation_TotalSegmentator/colon.nii.gz").write_bytes(b"broken nifti")
    reads = []
    real_read = sitk.ReadImage
    def counted(path, *args, **kwargs):
        if str(path).endswith(".nii.gz"):
            reads.append(str(path))
        return real_read(path, *args, **kwargs)
    monkeypatch.setattr(sitk, "ReadImage", counted)
    result = _produce(rr, rc, course, config)
    assert reads == []
    print({"layout": kind, "nifti_reads": len(reads)})
    assert result[0].segmentation_source.unique().tolist() == (
        ["AutoRTS_total", "CustomModel:M", "Manual"] if kind == "model"
        else ["Manual"] if kind == "stale_auto" else ["AutoRTS_total", "Manual"]
    )


@pytest.mark.parametrize("broken", [False, True])
def test_certificate_rederives_sources(tmp_path, monkeypatch, main_producer, broken):
    course, config, _, _ = _layout(tmp_path, monkeypatch)
    main, completion = main_producer
    for name in ("load_course_contract", "_load_main_ct_identity_catalog", "extract_features_for_masks"):
        monkeypatch.setattr(main, name, getattr(rr, name))
    _produce(main, completion, course, config)
    # A newly evaluated source is invisible to the old receipt's own list.
    write_rtstruct(course / "CT", course / "RS_custom.dcm", [("GTV", np.ones((31, 31, 13), bool))])
    if broken:
        (course / "RS_custom.dcm").write_bytes(b"malformed synthetic DICOM")
    with pytest.raises((ValueError, rc.RobustnessCompletionError), match="different code"):
        rr.admit_robustness_cohort_course(course, patient_id="P1", course_id="C1", rob_config=rr.RobustnessConfig(enabled=True))


def test_nifti_ct_main_receipt_requires_recompute(tmp_path, monkeypatch, main_producer):
    course, config, contract, catalog = _layout(tmp_path, monkeypatch, "nifti_ct", missing_identity=True)
    # Main measures AutoRTS NIfTI, but omits the selected Manual source binding.
    catalog[("AutoRTS_total", "BLADDER")] = rr.RobustnessRoiIdentity.from_mapping({
        "patient_id": "P1", "course_id": "C1", "series_uid": contract.planning_ct["series_instance_uid"],
        "segmentation_source": "AutoRTS_total", "roi_original_name": "BLADDER",
        "mask_identity": "voxel", "stable_roi_identifier": "voxel",
    })
    main, completion = main_producer
    for name in ("load_course_contract", "_load_main_ct_identity_catalog", "extract_features_for_masks"):
        monkeypatch.setattr(main, name, getattr(rr, name))
    assert _produce(main, completion, course, config) is not None
    with pytest.raises((ValueError, rc.RobustnessCompletionError), match="different code"):
        rr.admit_robustness_cohort_course(course, patient_id="P1", course_id="C1", rob_config=rr.RobustnessConfig(enabled=True))


def test_nifti_ct_unpublished_target_is_unresolved(tmp_path, monkeypatch):
    course, config, _, _ = _layout(tmp_path, monkeypatch, "nifti_ct", missing_identity=True)
    assert _produce(rr, rc, course, config) is None
    assert not (course / ".radiomics_robustness_done").exists()
    ledger = json.loads((course / "metadata/radiomics_robustness_identity.json").read_text())
    assert ("Manual", "PTV", "identity_absent_from_main_radiomics") in [
        (row["segmentation_source"], row["roi_original_name"], row["reason_code"]) for row in ledger["rows"]]


def test_nifti_ct_published_target_uses_physical_export(tmp_path, monkeypatch):
    # A retained main identity can still have its exact, provenance-validated
    # physical mask after DICOM staging is removed. Never redraw on a new grid.
    course, config, _, _ = _layout(tmp_path, monkeypatch, "nifti_ct", export=True)
    for unused in (course / "Segmentation_TotalSegmentator").glob("*.nii.gz"):
        unused.unlink()
    path = next((course / "Segmentation_Original").glob("*/PTV.nii.gz"))
    expected = sitk.ReadImage(str(path))
    result = _produce(rr, rc, course, config)
    assert result is not None
    frame = result[0]
    baseline = frame[frame.perturbation_id.eq("ntcv_v0")]
    assert set(baseline.perturbed_mask_identity) == {rr._perturbed_mask_identity(expected)}
    assert set(frame.rasterizer_backend) == {"rt-utils/cv2.fillPoly"}
    inventory = json.loads((course / "metadata/radiomics_robustness_target_inventory.json").read_text())
    assert str(path) in {binding["source_path"] for binding in inventory["voxel_source_bindings"]}
    assert rr.admit_robustness_cohort_course(
        course, patient_id="P1", course_id="C1", rob_config=rr.RobustnessConfig(enabled=True),
    ).measured
    path.write_bytes(b"changed synthetic mask")
    with pytest.raises((ValueError, rc.RobustnessCompletionError)):
        rr.admit_robustness_cohort_course(
            course, patient_id="P1", course_id="C1", rob_config=rr.RobustnessConfig(enabled=True),
        )


def test_many_unused_masks_have_zero_reads(tmp_path, monkeypatch):
    course, config, _, _ = _layout(tmp_path, monkeypatch)
    content = (course / "Segmentation_TotalSegmentator/BLADDER.nii.gz").read_bytes()
    for index in range(100):
        (course / f"Segmentation_TotalSegmentator/Unused_{index}.nii.gz").write_bytes(content)
    reads = []
    real_read = sitk.ReadImage
    def counted(path, *args, **kwargs):
        if str(path).endswith(".nii.gz"):
            reads.append(str(path))
        return real_read(path, *args, **kwargs)
    monkeypatch.setattr(sitk, "ReadImage", counted)
    assert _produce(rr, rc, course, config) is not None
    print({"nifti_files": 102, "nifti_reads": len(reads)})
    assert len(reads) == 0


def test_model_without_rtstruct_uses_its_own_inventory(tmp_path, monkeypatch):
    course, config, _, _ = _layout(tmp_path, monkeypatch, "voxel_model")
    path = course / "Segmentation_CustomModels/M/GTV.nii.gz"
    path.rename(path.with_name("HR-CTV.nii.gz"))
    result = _produce(rr, rc, course, config)
    assert set(result[0].segmentation_source) == {"Manual", "AutoRTS_total"}
    assert set(result[0].roi_original_name) == {"PTV", "BLADDER"}
    inventory = json.loads((course / "metadata/radiomics_robustness_target_inventory.json").read_text())
    assert {(r["segmentation_source"], r["roi_name"]) for r in inventory["target_inventory"]} == {
        ("Manual", "PTV"), ("AutoRTS_total", "BLADDER"), ("CustomModel:M", "HR-CTV"),
    }
    assert [(r["segmentation_source"], r["roi_name"]) for r in result[1]["selection_gap_targets"]] == [
        ("CustomModel:M", "HR-CTV")]


def test_dispositioned_auto_roi_is_not_filled_from_voxels(tmp_path, monkeypatch):
    course, config, _, _ = _layout(tmp_path, monkeypatch)
    path = course / "RS_auto.dcm"
    ds = pydicom.dcmread(path)
    contour = ds.ROIContourSequence[0].ContourSequence[0]
    contour.ContourGeometricType = "POINT"
    contour.ContourData = contour.ContourData[:3]
    contour.NumberOfContourPoints = 1
    ds.ROIContourSequence[0].ContourSequence = [contour]
    ds.save_as(path)
    result = _produce(rr, rc, course, config)
    assert set(result[0].segmentation_source) == {"Manual"}
    assert [(row["segmentation_source"], row["roi_name"]) for row in result[1]["rows"]] == [
        ("AutoRTS_total", "BLADDER")]


def test_unused_nifti_name_does_not_create_unresolved_identity(tmp_path, monkeypatch):
    course, config, _, _ = _layout(tmp_path, monkeypatch)
    content = (course / "Segmentation_TotalSegmentator/BLADDER.nii.gz").read_bytes()
    (course / "Segmentation_TotalSegmentator/GTV.nii.gz").write_bytes(content)
    result = _produce(rr, rc, course, config)
    assert all(row["identity_status"] == "resolved" for row in result[2]["rows"])
    assert {row["roi_original_name"] for row in result[2]["rows"]} == {"PTV", "BLADDER"}


def test_current_fallback_receipt_binds_used_voxels(tmp_path, monkeypatch):
    course, config, _, _ = _layout(tmp_path, monkeypatch, "fallback")
    _produce(rr, rc, course, config)
    assert rr.admit_robustness_cohort_course(
        course, patient_id="P1", course_id="C1", rob_config=rr.RobustnessConfig(enabled=True),
    ).measured
    (course / "Segmentation_TotalSegmentator/BLADDER.nii.gz").write_bytes(b"changed synthetic voxels")
    with pytest.raises((ValueError, rc.RobustnessCompletionError)):
        rr.admit_robustness_cohort_course(
            course, patient_id="P1", course_id="C1", rob_config=rr.RobustnessConfig(enabled=True),
        )


def test_selected_voxel_only_model_cannot_be_hidden(tmp_path, monkeypatch):
    course, config, _, _ = _layout(tmp_path, monkeypatch, "voxel_model")
    assert _produce(rr, rc, course, config) is None
    assert not (course / ".radiomics_robustness_done").exists()
    ledger = json.loads((course / "metadata/radiomics_robustness_identity.json").read_text())
    assert ("CustomModel:M", "GTV", "main_radiomics_mask_unavailable") in [
        (row["segmentation_source"], row["roi_original_name"], row.get("reason_code")) for row in ledger["rows"]]
