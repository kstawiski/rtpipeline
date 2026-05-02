from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from rtpipeline.config import PipelineConfig
from rtpipeline.custom_models import discover_custom_models, _run_single_model


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
    task_line = "      task: lung_nodules\n" if interface == "totalsegmentator" else ""
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
{task_line}      source_directory: {source_dir}
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
        ("lung_tumor_aimi_nsclc_rg", "nnunetv1"),
    ],
)
def test_r8_custom_model_discovery_and_phantom_smoke(tmp_path, monkeypatch, name, interface):
    model_dir = tmp_path / name
    source = model_dir / "weights" / "test_source"
    source.mkdir(parents=True)
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

    def fake_nnunet(_cfg, _model, _network, _input_dir, output_dir, _env):
        _write_seg(output_dir / "case.nii.gz", nifti)

    def fake_totalseg(**kwargs):
        _write_seg(Path(kwargs["output_dir"]) / "lung_tumor.nii.gz", nifti)

    def fake_rtstruct(_ct_dir, _structure_masks, output_dir, _structure_paths):
        out = output_dir / "rtstruct.dcm"
        out.write_bytes(b"fake")
        return out

    monkeypatch.setattr("rtpipeline.custom_models._run_nnunet_prediction", fake_nnunet)
    monkeypatch.setattr("rtpipeline.segmenters.totalsegmentator.run_totalsegmentator_prediction", fake_totalseg)
    monkeypatch.setattr("rtpipeline.custom_models._build_rtstruct", fake_rtstruct)

    cfg = PipelineConfig(dicom_root=tmp_path, output_root=tmp_path / "out", logs_root=tmp_path / "logs")
    _run_single_model(cfg, course, nifti, course.dirs.dicom_ct, model)
    out_mask = course.dirs.segmentation_custom_models / name / "lung_tumor.nii.gz"
    assert out_mask.exists()
    assert (course.dirs.segmentation_custom_models / name / "manifest.json").exists()
