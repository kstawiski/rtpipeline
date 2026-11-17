"""Durable refusal, omission accounting and overlay pixels on synthetic DICOM."""
import base64
import io
import json
import re
from types import SimpleNamespace

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from PIL import Image
from rt_utils.rtstruct import RTStruct

from test_roi_identity_ws554 import source, names, builder, export


def _omit(source, mode, *, all_rois=False):
    """Use real empty contours or a real malformed contour reference."""
    ds = pydicom.dcmread(source[1])
    for item in ds.ROIContourSequence if all_rois else ds.ROIContourSequence[1:]:
        if mode == "empty":
            item.ContourSequence = []
        else:
            # rt-utils cannot bind a contour whose image reference is absent.
            del item.ContourSequence[0].ContourImageSequence
    ds.save_as(source[1])


@pytest.mark.parametrize("values", [["Organ", "Organ"], ["Organ", " Organ"]])
@pytest.mark.parametrize("runner", ["producer", "course_adapter", "campaign_closure"])
@pytest.mark.requires_radiomics_env
@pytest.mark.skipif(int(np.__version__.split(".")[0]) >= 2, reason="requires the native NumPy 1.x radiomics interpreter")
def test_robustness_identity_refusal_is_durable(source, tmp_path, values, runner):
    from course_contract_test_utils import write_minimal_course_contract
    from rtpipeline import radiomics_robustness as rb
    from rtpipeline.config import PipelineConfig
    from rtpipeline.course_contract import _ct_provenance
    from rtpipeline.radiomics import RadiomicsCourseExtractionError
    from rtpipeline.radiomics_rasterizer import require_radiomics_rasterizer

    require_radiomics_rasterizer()
    names(source[1], ["Organ", "Other"])
    contract = write_minimal_course_contract(
        tmp_path, authoritative_rtstruct=source[1], planning_ct_nifti=source[2])
    sidecar = source[2].with_name("ct.metadata.json")
    meta = json.loads(sidecar.read_text())
    meta.update(_ct_provenance(source[0]))
    sidecar.write_text(json.dumps(meta))
    payload = json.loads(contract.read_text())
    payload["course_contract"]["planning_ct"]["nifti_provenance"].update(meta)
    contract.write_text(json.dumps(payload))
    cfg = PipelineConfig(source[0], tmp_path, tmp_path / "logs", max_workers_override=1)
    params = tmp_path / "radiomics.yaml"
    params.write_text("imageType:\n  Original: {}\nfeatureClass:\n  firstorder: [Mean]\n  shape: [VoxelVolume]\nsetting:\n  minimumROIDimensions: 2\n")
    cfg.radiomics_params_file = params
    cfg.radiomics_min_voxels = 1
    # Real upstream publication first; a later source with colliding labels
    # must be refused durably even when no current measurement can be made.
    from rtpipeline import radiomics
    radiomics.radiomics_for_course(cfg, tmp_path)
    names(source[1], values)
    rob = rb.RobustnessConfig(perturbation=rb.PerturbationConfig(apply_to_structures=["Organ"]))
    disposition = rb.robustness_source_dispositions_path(tmp_path)
    disposition.write_text('{"stale": true}')
    output = tmp_path / "radiomics_robustness_ct.parquet"
    output.write_bytes(b"stale synthetic table")
    run = rb.robustness_for_course if runner == "producer" else rb.run_robustness_course
    with pytest.raises(RadiomicsCourseExtractionError, match="RTSTRUCT_AMBIGUOUS_ROI_NAME"):
        run(cfg, rob, tmp_path, output_path=output)
    assert not output.exists()
    assert disposition.is_file(), "identity refusals must survive the course exception"
    if runner == "campaign_closure":
        rb.write_robustness_failure_dispositions(
            tmp_path, rob_config=rob, output_name=output.name)
    refusal = json.loads(disposition.read_text())
    assert refusal["measurement_outcome"] == rb.ROBUSTNESS_FAILED_OUTCOME
    assert refusal["measured_output"] is None
    rows = refusal["source_errors"]
    assert [(row["roi_number"], row["roi_name"]) for row in rows] == list(zip(["1", "2"], values))
    assert all(row["structural_code"] == "RTSTRUCT_AMBIGUOUS_ROI_NAME" for row in rows)
    assert all(row["status"] == "failed" and row["segmentation_source"] == "Manual" for row in rows)
    assert all(row["source_path"] == str(source[1]) for row in rows)
    assert all(row["rtstruct_sop_instance_uid"] == str(pydicom.dcmread(source[1]).SOPInstanceUID) for row in rows)
    # This is valid failure evidence, with no measurement/disposition admitted.
    assert rb.load_robustness_source_dispositions(
        tmp_path, run_identifier=refusal["robustness_run_identifier"], rob_config=rob) == []


@pytest.mark.parametrize("mode,reason", [("empty", "ROI_MASK_EMPTY"), ("read_failed", "ROI_MASK_READ_FAILED")])
@pytest.mark.parametrize("all_rois", [False, True])
def test_export_records_each_omitted_declaration(source, tmp_path, mode, reason, all_rois):
    root = tmp_path / "Segmentation_Original"
    before = export(source, root)
    first_bytes = (root / before["structures"][0]["mask"]).read_bytes()
    _omit(source, mode, all_rois=all_rois)
    result = export(source, root)
    manifest = json.loads((root / "ct/metadata.json").read_text())
    expected = [1, 2] if all_rois else [2]
    assert manifest["skipped_structures"] == [
        {"roi_number": number, "roi_name": "Organ", "reason_code": reason}
        for number in expected]
    if all_rois:
        assert manifest["structures"] == []
        assert result is None
        assert not list((root / "ct").glob("*.nii*"))
    else:
        assert [row["roi_number"] for row in manifest["structures"]] == [1]
        assert (root / manifest["structures"][0]["mask"]).read_bytes() == first_bytes
        from rtpipeline.original_segmentation import require_current_original_export
        require_current_original_export(root / "ct")


@pytest.mark.parametrize("mode,reason", [("empty", "ROI_MASK_EMPTY"), ("read_failed", "ROI_MASK_READ_FAILED")])
@pytest.mark.parametrize("source_label", ["manual", "auto"])
@pytest.mark.parametrize("all_rois", [False, True])
def test_qc_records_each_omitted_declaration(source, tmp_path, mode, reason, source_label, all_rois):
    from rtpipeline.quality_control import DICOMValidator

    # The readable declaration touches a boundary and has a real QC result.
    # Rebuild the first contour with an in-plane boundary, retaining duplicate labels.
    from robustness_prep_fixture import write_rtstruct
    edge = np.zeros_like(source[3]); edge[0:5, 3:8, 1:4] = True
    write_rtstruct(source[0], source[1], [("Organ", edge), ("Organ", source[4])])
    validator = object.__new__(DICOMValidator)
    validator.course_dir = tmp_path
    validator.ct_dir = source[0]
    validator.rs_path = source[1] if source_label == "manual" else tmp_path / "absent.dcm"
    validator.rs_auto_path = source[1] if source_label == "auto" else tmp_path / "absent.dcm"
    before = validator._segmentation_cropping()
    _omit(source, mode, all_rois=all_rois)
    result = validator._segmentation_cropping()
    if all_rois:
        assert result["status"] == "SKIP"
    else:
        assert result["status"] == before["status"]
        assert result["structures"][0] == before["structures"][0]
    assert [row for row in result["structures"] if row.get("status") == "SKIP"] == [
        {"source": source_label, "roi_number": number, "roi_name": "Organ",
         "status": "SKIP", "reason_code": reason}
        for number in ([1, 2] if all_rois else [2])]


@pytest.mark.parametrize("structure_set", ["manual", "auto", "custom"])
@pytest.mark.parametrize("legacy_method", ["get_mask_for_roi", "get_roi_mask"])
def test_axial_overlay_pixels_follow_roi_numbers(source, tmp_path, monkeypatch, structure_set, legacy_method):
    pytest.importorskip("plotly")
    from rtpipeline import visualize

    # Supply a name convenience method alongside the real rt-utils number path.
    # It uses the real first-name rasterizer; selecting it reproduces the defect.
    monkeypatch.setattr(RTStruct, legacy_method, lambda self, name: self.get_roi_mask_by_name(name), raising=False)
    rs = source[1]
    if structure_set != "manual":
        target = tmp_path / ("RS_auto.dcm" if structure_set == "auto" else "RS_custom.dcm")
        target.write_bytes(rs.read_bytes())
    monkeypatch.setattr(visualize, "load_course_contract", lambda _: SimpleNamespace(
        planning_ct_dir=source[0], authoritative_rtstruct_path=rs if structure_set == "manual" else None))
    html = visualize.generate_axial_review(tmp_path).read_text()
    match = re.search(r"const OVERLAYS_" + structure_set.upper() + r" = (.*);", html)
    overlays = json.loads(match.group(1))
    assert set(overlays) == {"Organ [ROI 1]", "Organ [ROI 2]"}
    image = sitk.ReadImage(str(source[2]))
    from rtpipeline.rtstruct_geometry import rtstruct_mask_to_image
    for number, mask in enumerate(source[3:], 1):
        expected = sitk.GetArrayFromImage(rtstruct_mask_to_image(mask, builder(source).series_data, image)).astype(bool)
        actual = np.zeros_like(expected)
        for z, encoded in overlays[f"Organ [ROI {number}]"].items():
            rgba = np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))))
            actual[int(z)] = rgba[:, :, 3] > 0
        assert np.array_equal(actual, expected), f"{structure_set} ROI {number} overlay pixels"
