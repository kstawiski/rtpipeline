from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from rtpipeline.config import PipelineConfig
from rtpipeline.custom_models import (
    NetworkDefinition,
    _postprocess_structures,
    _run_single_model,
    discover_custom_models,
)
from rtpipeline.segmenters.medsam_boxprompt import run_medsam_boxprompt_prediction
from rtpipeline.segmenters.postprocessing import (
    derive_anchor_from_reference,
    select_component_nearest_anchor,
)


@dataclass(slots=True)
class _Dirs:
    root: Path

    @property
    def segmentation_custom_models(self) -> Path:
        return self.root / "Segmentation_CustomModels"

    @property
    def dicom_ct(self) -> Path:
        return self.root / "DICOM" / "CT"

    def ensure(self) -> None:
        self.segmentation_custom_models.mkdir(parents=True, exist_ok=True)
        self.dicom_ct.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class _Course:
    root: Path
    patient_id: str = "phantom"
    course_id: str = "test"

    @property
    def dirs(self) -> _Dirs:
        dirs = _Dirs(self.root)
        dirs.ensure()
        return dirs


def _write_phantom(path: Path) -> None:
    arr = np.zeros((2, 8, 8), dtype=np.int16)
    arr[:, 2:6, 2:6] = 100
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 2.0))
    sitk.WriteImage(img, str(path))


def _write_seg(path: Path, reference: Path) -> None:
    ref = sitk.ReadImage(str(reference))
    arr = np.zeros(sitk.GetArrayFromImage(ref).shape, dtype=np.uint8)
    arr[:, 3:5, 3:5] = 1
    seg = sitk.GetImageFromArray(arr)
    seg.CopyInformation(ref)
    sitk.WriteImage(seg, str(path))


def _model_yaml(interface: str, name: str, source_dir: str) -> str:
    command = "nnUNetv2_predict_from_modelfolder" if interface == "nnunetv2_modelfolder" else "TotalSegmentator"
    if interface == "nnunetv1":
        command = "nnUNet_predict"
    if interface == "medsam_boxprompt":
        command = "MedSAM"
    task_line = "      task: lung_nodules\n" if interface == "totalsegmentator" else ""
    checkpoint_line = "      checkpoint: weights/fake_checkpoint.pth\n" if interface == "medsam_boxprompt" else ""
    prompt_block = (
        "      prompt:\n"
        "        model: lung_tumor_totalseg_lung_nodules\n"
        "        structure: lung_nodules\n"
        "        bbox_margin_voxels: 1\n"
        if interface == "medsam_boxprompt"
        else ""
    )
    return f"""
name: {name}
description: R8 test model
nnunet:
  interface: {interface}
  command: {command}
  model: 3d_fullres
  folds: [0]
  env:
    nnUNet_results: ./weights
    nnUNet_raw: ./raw
    nnUNet_preprocessed: ./preprocessed
  networks:
    - id: test_dataset
      alias: test
{task_line}{checkpoint_line}{prompt_block}      source_directory: {source_dir}
      dataset_directory: {source_dir}
      label_order:
        - lung_tumor
  combine:
    ordered_networks: ["test"]
"""


@pytest.mark.parametrize(
    ("name", "interface"),
    [
        ("lung_tumor_totalseg_lung_nodules", "totalsegmentator"),
        ("lung_tumor_pancancer_lung", "nnunetv2_modelfolder"),
        ("lung_tumor_medsam_boxprompt", "medsam_boxprompt"),
    ],
)
def test_r8_custom_model_discovery_and_phantom_smoke(tmp_path, monkeypatch, name, interface):
    model_dir = tmp_path / name
    source = model_dir / "weights" / "test_source"
    source.mkdir(parents=True)
    (model_dir / "weights" / "fake_checkpoint.pth").write_bytes(b"fake")
    (model_dir / "custom_model.yaml").write_text(
        _model_yaml(interface, name, "weights/test_source"),
        encoding="utf-8",
    )
    models = discover_custom_models(tmp_path, selected_names=[name])
    assert [model.name for model in models] == [name]
    model = models[0]
    assert model.interface == interface
    assert model.expected_structures() == ["lung_tumor"]

    course = _Course(tmp_path / "course")
    nifti = tmp_path / "phantom.nii.gz"
    _write_phantom(nifti)
    if interface == "medsam_boxprompt":
        prompt_dir = course.dirs.segmentation_custom_models / "lung_tumor_totalseg_lung_nodules"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        _write_seg(prompt_dir / "lung_nodules.nii.gz", nifti)

    def fake_nnunet(_cfg, _model, _network, _input_dir, output_dir, _env):
        _write_seg(output_dir / "case.nii.gz", nifti)

    def fake_totalseg(**kwargs):
        _write_seg(Path(kwargs["output_dir"]) / "lung_tumor.nii.gz", nifti)

    def fake_medsam(**kwargs):
        _write_seg(Path(kwargs["output_dir"]) / "lung_tumor.nii.gz", nifti)

    def fake_rtstruct(_ct_dir, _structure_masks, output_dir, _structure_paths):
        out = output_dir / "rtstruct.dcm"
        out.write_bytes(b"fake")
        return out

    monkeypatch.setattr("rtpipeline.custom_models._run_nnunet_prediction", fake_nnunet)
    monkeypatch.setattr("rtpipeline.segmenters.totalsegmentator.run_totalsegmentator_prediction", fake_totalseg)
    monkeypatch.setattr("rtpipeline.segmenters.medsam_boxprompt.run_medsam_boxprompt_prediction", fake_medsam)
    monkeypatch.setattr("rtpipeline.custom_models._build_rtstruct", fake_rtstruct)

    cfg = PipelineConfig(dicom_root=tmp_path, output_root=tmp_path / "out", logs_root=tmp_path / "logs")
    _run_single_model(cfg, course, nifti, course.dirs.dicom_ct, model)
    out_mask = course.dirs.segmentation_custom_models / name / "lung_tumor.nii.gz"
    assert out_mask.exists()
    assert (course.dirs.segmentation_custom_models / name / "manifest.json").exists()


def test_discover_custom_models_skips_underscore_prefixed_dirs(tmp_path):
    model_dir = tmp_path / "_disabled_lung_tumor_aimi_nsclc_rg"
    source = model_dir / "weights" / "test_source"
    source.mkdir(parents=True)
    (model_dir / "custom_model.yaml").write_text(
        _model_yaml("nnunetv1", "lung_tumor_aimi_nsclc_rg", "weights/test_source"),
        encoding="utf-8",
    )
    assert discover_custom_models(tmp_path) == []


def test_derive_anchor_from_reference_glob_unions_gtv_masks(tmp_path):
    arr = np.zeros((5, 12, 12), dtype=np.uint8)
    arr[1:3, 4:6, 6:8] = 1
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((2.0, 2.0, 5.0))
    ref_dir = tmp_path / "Segmentation_Original" / "CT_1"
    ref_dir.mkdir(parents=True)
    sitk.WriteImage(img, str(ref_dir / "GTV-a.nii.gz"))

    anchor = derive_anchor_from_reference(str(tmp_path / "Segmentation_Original" / "CT_*" / "GTV-*.nii.gz"))

    assert anchor is not None
    assert anchor["voxels"] == 8
    assert anchor["bbox_zyx"] == ((1, 4, 6), (2, 5, 7))
    assert np.allclose(anchor["centroid_zyx"], (1.5, 4.5, 6.5), atol=1.0)


def test_derive_anchor_from_reference_indexed_glob_excludes_second_lesion(tmp_path):
    ref_dir = tmp_path / "Segmentation_Original" / "CT_1"
    ref_dir.mkdir(parents=True)

    def write_box(name: str, y0: int, x0: int) -> None:
        arr = np.zeros((4, 24, 24), dtype=np.uint8)
        arr[1:3, y0 : y0 + 3, x0 : x0 + 3] = 1
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((1.0, 1.0, 2.0))
        sitk.WriteImage(img, str(ref_dir / name))

    write_box("GTV-1vis-1.nii.gz", 3, 3)
    write_box("GTV-1auto-1.nii.gz", 4, 4)
    write_box("GTV-2vis-1.nii.gz", 17, 17)
    write_box("GTV-2auto-3.nii.gz", 18, 18)

    mixed_anchor = derive_anchor_from_reference(
        str(tmp_path / "Segmentation_Original" / "CT_*" / "GTV-*.nii.gz")
    )
    primary_anchor = derive_anchor_from_reference(
        str(tmp_path / "Segmentation_Original" / "CT_*" / "GTV-1*-*.nii.gz")
    )

    assert mixed_anchor is not None
    assert primary_anchor is not None
    assert mixed_anchor["bbox_zyx"] == ((1, 3, 3), (2, 20, 20))
    assert primary_anchor["bbox_zyx"] == ((1, 3, 3), (2, 6, 6))
    assert primary_anchor["centroid_zyx"][1] < 6.0
    assert primary_anchor["centroid_zyx"][2] < 6.0
    assert mixed_anchor["centroid_zyx"][1] > 10.0
    assert mixed_anchor["centroid_zyx"][2] > 10.0
    assert all("GTV-1" in Path(path).name for path in primary_anchor["reference_paths"])


def test_select_component_nearest_anchor_keeps_near_component_or_empty(tmp_path):
    arr = np.zeros((6, 14, 14), dtype=np.uint8)
    arr[1:3, 2:5, 2:5] = 1
    arr[3:5, 9:12, 9:12] = 1
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))

    selected, meta = select_component_nearest_anchor(
        img,
        anchor_centroid_zyx=(1.5, 3.0, 3.0),
        max_distance_cm=1.0,
        min_component_volume_cm3=0.001,
    )
    selected_arr = sitk.GetArrayFromImage(selected)
    assert meta["status"] == "selected"
    assert meta["n_components_evaluated"] == 2
    assert selected_arr.sum() == arr[1:3, 2:5, 2:5].sum()
    assert selected_arr[1:3, 2:5, 2:5].all()
    assert not selected_arr[3:5, 9:12, 9:12].any()

    empty, far_meta = select_component_nearest_anchor(
        img,
        anchor_centroid_zyx=(100.0, 100.0, 100.0),
        max_distance_cm=1.0,
        min_component_volume_cm3=0.001,
    )
    assert far_meta["status"] == "no_anchor_match"
    assert sitk.GetArrayFromImage(empty).sum() == 0


def test_postprocess_reports_anchor_selected_then_lung_confine_zeroed(tmp_path):
    arr = np.zeros((4, 12, 12), dtype=np.uint8)
    arr[1:3, 4:7, 4:7] = 1
    mask = sitk.GetImageFromArray(arr)
    mask.SetSpacing((1.0, 1.0, 1.0))

    lung_dir = tmp_path / "Segmentation_TotalSegmentator" / "CT_1"
    lung_dir.mkdir(parents=True)
    empty_lung = sitk.GetImageFromArray(np.zeros_like(arr, dtype=np.uint8))
    empty_lung.CopyInformation(mask)
    sitk.WriteImage(empty_lung, str(lung_dir / "total--lung.nii.gz"))

    network = NetworkDefinition(
        network_id="test",
        alias="test",
        archive_path=None,
        checkpoint_path=None,
        dataset_dir="",
        label_names=["lung_tumor"],
        anchor_source="specialist_reference",
        confine_to_lung_mask=True,
        min_component_volume_cm3=0.001,
    )

    postprocessed, diagnostics = _postprocess_structures(
        {"lung_tumor": mask},
        tmp_path,
        network,
        anchor={"centroid_zyx": (1.5, 5.0, 5.0)},
        diagnostics={},
    )

    struct_diag = diagnostics["structures"]["lung_tumor"]
    assert struct_diag["anchor_selection"]["status"] == "selected"
    assert struct_diag["lung_confinement"]["before_voxels"] > 0
    assert struct_diag["lung_confinement"]["after_voxels"] == 0
    assert struct_diag["failure_reason"] == "anchor_matched_but_lung_confine_zeroed"
    assert sitk.GetArrayFromImage(postprocessed["lung_tumor"]).sum() == 0


def test_medsam_anchor_bbox_empty_prompt_is_graceful(tmp_path):
    nifti = tmp_path / "phantom.nii.gz"
    _write_phantom(nifti)

    metadata = run_medsam_boxprompt_prediction(
        input_path=nifti,
        output_dir=tmp_path / "medsam_out",
        checkpoint_path=tmp_path / "missing_checkpoint.pth",
        source_dir=None,
        prompt_source="anchor_bbox",
        anchor=None,
        output_name="lung_tumor",
        bbox_margin_voxels=1,
    )

    out_path = tmp_path / "medsam_out" / "lung_tumor.nii.gz"
    assert metadata["status"] == "skipped_no_prompt"
    assert out_path.exists()
    assert sitk.GetArrayFromImage(sitk.ReadImage(str(out_path))).sum() == 0


def test_dice_reference_glob_explicit_empty_disables_default(tmp_path):
    name = "lung_tumor_pancancer_lung_rider_test"
    model_dir = tmp_path / name
    source = model_dir / "weights" / "test_source"
    source.mkdir(parents=True)
    (model_dir / "custom_model.yaml").write_text(
        f"""
name: {name}
description: parser-fix regression
nnunet:
  interface: nnunetv2_modelfolder
  command: nnUNetv2_predict_from_modelfolder
  model: 3d_fullres
  folds: [0]
  env:
    nnUNet_results: ./weights
    nnUNet_raw: ./raw
    nnUNet_preprocessed: ./preprocessed
  networks:
    - id: test_dataset
      alias: test
      source_directory: weights/test_source
      dataset_directory: weights/test_source
      postprocessing:
        anchor_source: specialist_reference
        anchor_glob: Segmentation_CustomModels/lung_tumor_totalseg_lung_nodules/lung_nodules.nii.gz
        dice_reference_glob: ""
      label_order:
        - lung_tumor
  combine:
    ordered_networks: ["test"]
""",
        encoding="utf-8",
    )
    [model] = discover_custom_models(tmp_path, selected_names=[name])
    [network] = model.networks
    assert network.dice_reference_glob == ""
    assert network.anchor_glob == "Segmentation_CustomModels/lung_tumor_totalseg_lung_nodules/lung_nodules.nii.gz"
    assert network.anchor_source == "specialist_reference"


def test_dice_reference_glob_omitted_uses_default(tmp_path):
    name = "lung_tumor_default_glob_test"
    model_dir = tmp_path / name
    source = model_dir / "weights" / "test_source"
    source.mkdir(parents=True)
    (model_dir / "custom_model.yaml").write_text(
        _model_yaml("nnunetv2_modelfolder", name, "weights/test_source"),
        encoding="utf-8",
    )
    [model] = discover_custom_models(tmp_path, selected_names=[name])
    [network] = model.networks
    assert network.dice_reference_glob == "Segmentation_Original/CT_*/GTV-1vis-*.nii.gz"
    assert network.anchor_glob == "Segmentation_Original/CT_*/GTV-1*-*.nii.gz"
