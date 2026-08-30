from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from rtpipeline import body_composition
from rtpipeline.config import PipelineConfig
from rtpipeline.inventory import ts_tasks_for_image_class
from rtpipeline.layout import build_course_dirs
import rtpipeline.cli as cli
import rtpipeline.segmentation as segmentation


def _config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        dicom_root=tmp_path / "dicom",
        output_root=tmp_path / "out",
        logs_root=tmp_path / "logs",
    )


def _write_nifti(path: Path, array: np.ndarray, spacing=(2.0, 3.0, 5.0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(array)
    img.SetSpacing(tuple(float(x) for x in spacing))
    sitk.WriteImage(img, str(path))


def _write_dicom(path: Path, patient_size: float | None = 2.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = generate_uid()
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.ImplementationClassUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    if patient_size is not None:
        ds.PatientSize = patient_size
    pydicom.dcmwrite(str(path), ds, write_like_original=False)


def _write_body_comp_inputs(tmp_path: Path, *, patient_size: float | None = 2.0):
    ct = np.zeros((5, 4, 4), dtype=np.int16)
    muscle = np.zeros((5, 4, 4), dtype=np.uint8)
    torso_fat = np.zeros((5, 4, 4), dtype=np.uint8)
    subq = np.zeros((5, 4, 4), dtype=np.uint8)
    l3 = np.zeros((5, 4, 4), dtype=np.uint8)

    l3[1:4, 1:3, 1:3] = 1
    muscle[2, 0:2, 0:2] = 1
    muscle[2, 2, 2] = 1
    ct[2][muscle[2] > 0] = 42  # muscle HU within [-29, 150]
    torso_fat[2, 0, 2:4] = 1
    subq[2, 3, 0:3] = 1
    ct[2][torso_fat[2] > 0] = -90  # visceral fat HU within [-150, -50]
    ct[2][subq[2] > 0] = -110  # subcutaneous fat HU within [-190, -30]

    root = tmp_path / "series"
    ct_path = root / "NIFTI" / "series.nii.gz"
    seg_dir = root / "Segmentation_TotalSegmentator" / "series"
    dicom_dir = root / "DICOM"
    _write_nifti(ct_path, ct)
    _write_nifti(seg_dir / "total--vertebrae_L3.nii.gz", l3)
    _write_nifti(seg_dir / "tissue_types--skeletal_muscle.nii.gz", muscle)
    _write_nifti(seg_dir / "tissue_types--torso_fat.nii.gz", torso_fat)
    _write_nifti(seg_dir / "tissue_types--subcutaneous_fat.nii.gz", subq)
    _write_dicom(dicom_dir / "IM_0001.dcm", patient_size=patient_size)
    return ct_path, seg_dir, dicom_dir


def test_b1_default_off_keeps_legacy_task_list(tmp_path):
    cfg = _config(tmp_path)

    assert cfg.body_composition_classes is None
    assert ts_tasks_for_image_class("planning_ct", cfg.body_composition_classes) == ["total"]
    assert ts_tasks_for_image_class("petct_ct", cfg.body_composition_classes) == ["total"]
    assert ts_tasks_for_image_class("cbct", cfg.body_composition_classes) == ["total"]


def test_b1_cli_yaml_parses_body_composition_classes(tmp_path, monkeypatch):
    dicom_root = tmp_path / "dicom"
    dicom_root.mkdir()
    (tmp_path / "config.yaml").write_text(
        "organize:\n  body_composition_classes: planning_ct, petct_ct\n",
        encoding="utf-8",
    )
    course_dirs = build_course_dirs(tmp_path / "out" / "P1" / "course")
    course_dirs.ensure()
    course = SimpleNamespace(patient_id="P1", course_id="course", dirs=course_dirs)
    captured: dict[str, PipelineConfig] = {}

    def fake_organize(cfg):
        captured["cfg"] = cfg
        return [course]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "organize_and_merge", fake_organize)
    monkeypatch.setattr(cli, "_detect_gpu_count", lambda: 0)
    monkeypatch.setattr(
        cli,
        "run_tasks_with_adaptive_workers",
        lambda name, tasks, fn, **kwargs: [True for _ in tasks],
    )

    assert cli.main(
        [
            "--dicom-root",
            str(dicom_root),
            "--outdir",
            str(tmp_path / "out"),
            "--logs",
            str(tmp_path / "logs"),
            "--stage",
            "segmentation",
            "--no-metadata",
        ]
    ) == 0
    assert captured["cfg"].body_composition_classes == ["planning_ct", "petct_ct"]


def test_b1_routing_adds_body_tasks_only_for_configured_ct_classes(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    cfg.body_composition_classes = ["planning_ct", "petct_ct"]
    all_series_root = cfg.output_root / "P1" / "all_series"
    manifest_path = build_course_dirs(all_series_root).metadata / "series_manifest.json"
    rows = []
    for idx, image_class in enumerate(("planning_ct", "diagnostic_ct", "petct_ct", "cbct"), start=1):
        output_dir = all_series_root / image_class / f"series-{idx}"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "IM_0001.dcm").write_text("not real dicom", encoding="utf-8")
        rows.append(
            {
                "patient_id": "P1",
                "study_uid": f"study-{idx}",
                "series_uid": f"series-{idx}",
                "modality": "CT",
                "image_class": image_class,
                "manufacturer_model": "",
                "frame_of_reference_uid": f"for-{idx}",
                "n_slices": 25,
                "ts_task": "total",
                "output_dir": str(output_dir),
                "status": "materialized",
                "exclusion_reason": "",
            }
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"patient_id": "P1", "series": rows}, indent=2), encoding="utf-8")

    def fake_ensure_nifti(config, input_dir, nifti_dir, force=False, dcm2niix_depth=None):
        nifti_dir.mkdir(parents=True, exist_ok=True)
        path = nifti_dir / f"{Path(input_dir).name}.nii.gz"
        path.write_text("fake nifti", encoding="utf-8")
        return path

    calls: list[dict] = []

    def fake_totalseg(config, input_path, output_path, output_type, task=None, extra_args=None):
        calls.append(
            {
                "input_path": Path(input_path),
                "output_path": Path(output_path),
                "output_type": output_type,
                "task": task,
                "extra_args": None if extra_args is None else list(extra_args),
            }
        )
        output_path.mkdir(parents=True, exist_ok=True)
        if output_type == "dicom_rtstruct":
            (output_path / "RS.fake.dcm").write_text("fake rtstruct", encoding="utf-8")
        else:
            masks = output_path / "segmentations"
            masks.mkdir(parents=True, exist_ok=True)
            (masks / f"{task}_mask.nii.gz").write_text("fake mask", encoding="utf-8")
        return True

    written_body_json: list[str] = []

    def fake_body_json(**kwargs):
        out = Path(kwargs["segmentation_dir"]) / "body_composition.json"
        out.write_text("{}", encoding="utf-8")
        written_body_json.append(str(out))
        return out

    monkeypatch.setattr(segmentation, "_ensure_ct_nifti", fake_ensure_nifti)
    monkeypatch.setattr(segmentation, "run_totalsegmentator", fake_totalseg)
    monkeypatch.setattr(body_composition, "write_series_body_composition", fake_body_json)
    monkeypatch.setattr(body_composition, "write_body_composition_csv", lambda output_root: None)

    summary = segmentation.segment_all_series_for_patient(cfg, "P1")

    assert summary["planning_ct"]["attempted"] == 1
    assert summary["petct_ct"]["attempted"] == 1
    assert summary["diagnostic_ct"]["attempted"] == 1
    assert summary["cbct"]["attempted"] == 1

    by_class: dict[str, list[str]] = {row["image_class"]: [] for row in rows}
    for call in calls:
        if call["output_type"] != "nifti":
            continue
        for row in rows:
            nifti_root = segmentation._series_artifact_dirs(Path(row["output_dir"]))[0]
            if nifti_root.resolve() in call["input_path"].resolve().parents:
                by_class[row["image_class"]].append(str(call["task"]))

    assert by_class["planning_ct"] == ["total", "tissue_types", "body"]
    assert by_class["petct_ct"] == ["total", "tissue_types", "body"]
    assert by_class["diagnostic_ct"] == ["total"]
    assert by_class["cbct"] == ["total"]
    assert len(written_body_json) == 2


def test_b1_worker_path_bodycomp_failure_keeps_series_segmented_and_writes_no_csv(tmp_path, monkeypatch):
    # Drives the REAL all-series worker control flow (not the writer in isolation):
    # when body-composition computation raises, the segmentation must remain
    # "segmented" (never seg_failed), and the worker path must NOT write the
    # global Data/body_composition.csv (that is aggregated once at stage end).
    cfg = _config(tmp_path)
    cfg.body_composition_classes = ["planning_ct"]
    all_series_root = cfg.output_root / "P1" / "all_series"
    manifest_path = build_course_dirs(all_series_root).metadata / "series_manifest.json"
    output_dir = all_series_root / "planning_ct" / "series-1"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "IM_0001.dcm").write_text("not real dicom", encoding="utf-8")
    row = {
        "patient_id": "P1",
        "study_uid": "study-1",
        "series_uid": "series-1",
        "modality": "CT",
        "image_class": "planning_ct",
        "manufacturer_model": "",
        "frame_of_reference_uid": "for-1",
        "n_slices": 25,
        "ts_task": "total",
        "output_dir": str(output_dir),
        "status": "materialized",
        "exclusion_reason": "",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"patient_id": "P1", "series": [row]}, indent=2), encoding="utf-8")

    def fake_ensure_nifti(config, input_dir, nifti_dir, force=False, dcm2niix_depth=None):
        nifti_dir.mkdir(parents=True, exist_ok=True)
        path = nifti_dir / f"{Path(input_dir).name}.nii.gz"
        path.write_text("fake nifti", encoding="utf-8")
        return path

    def fake_totalseg(config, input_path, output_path, output_type, task=None, extra_args=None):
        output_path.mkdir(parents=True, exist_ok=True)
        if output_type == "dicom_rtstruct":
            (output_path / "RS.fake.dcm").write_text("fake rtstruct", encoding="utf-8")
        else:
            masks = output_path / "segmentations"
            masks.mkdir(parents=True, exist_ok=True)
            (masks / f"{task}_mask.nii.gz").write_text("fake mask", encoding="utf-8")
        return True

    def boom(**kwargs):
        raise RuntimeError("synthetic body-composition failure")

    csv_calls: list = []

    def spy_csv(output_root):
        csv_calls.append(output_root)
        return None

    monkeypatch.setattr(segmentation, "_ensure_ct_nifti", fake_ensure_nifti)
    monkeypatch.setattr(segmentation, "run_totalsegmentator", fake_totalseg)
    monkeypatch.setattr(body_composition, "write_series_body_composition", boom)
    monkeypatch.setattr(body_composition, "write_body_composition_csv", spy_csv)

    summary = segmentation.segment_all_series_for_patient(cfg, "P1")

    # Segmentation succeeded; the body-composition failure must not fail the series.
    assert summary["planning_ct"]["segmented"] == 1
    assert summary["planning_ct"]["failed"] == 0
    # The worker path must not have aggregated the global CSV.
    assert csv_calls == []
    assert not (cfg.output_root / "Data" / "body_composition.csv").exists()
    # The error is recorded on the row for traceability.
    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "body_composition_error" in written["series"][0]
    assert written["series"][0]["status"] == "segmented"


def test_b1_body_composition_math_json_and_csv(tmp_path):
    ct_path, seg_dir, dicom_dir = _write_body_comp_inputs(tmp_path, patient_size=2.0)

    out_json = body_composition.write_series_body_composition(
        ct_nifti=ct_path,
        segmentation_dir=seg_dir,
        dicom_dir=dicom_dir,
        patient_id="P1",
        series_uid="S1",
        image_class="planning_ct",
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    metrics = payload["metrics"]

    assert payload["l3_selection"]["slice_index"] == 2
    assert metrics["skeletal_muscle_area_cm2"] == pytest.approx(0.30)
    assert metrics["skeletal_muscle_radiodensity_hu"] == pytest.approx(42.0)
    assert metrics["visceral_fat_area_cm2"] == pytest.approx(0.12)
    assert metrics["visceral_fat_proxy"] == "torso_fat"
    assert "proxy" in metrics["visceral_fat_proxy_note"]
    assert metrics["subcutaneous_fat_area_cm2"] == pytest.approx(0.18)
    assert metrics["smi_cm2_m2"] == pytest.approx(0.075)
    assert metrics["smi_missing_reason"] is None

    csv_path = body_composition.write_body_composition_csv(tmp_path)
    assert csv_path == tmp_path / "Data" / "body_composition.csv"
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["patient_id"] == "P1"
    assert rows[0]["series_uid"] == "S1"
    assert rows[0]["image_class"] == "planning_ct"
    assert "skeletal_muscle_area_cm2" in rows[0]
    assert "smi_missing_reason" in rows[0]


def test_b1_smi_null_when_patient_size_absent(tmp_path):
    ct_path, seg_dir, dicom_dir = _write_body_comp_inputs(tmp_path, patient_size=None)

    payload = body_composition.compute_body_composition(
        ct_nifti=ct_path,
        segmentation_dir=seg_dir,
        dicom_dir=dicom_dir,
        patient_id="P1",
        series_uid="S1",
        image_class="planning_ct",
    )

    assert payload["metrics"]["skeletal_muscle_area_cm2"] == pytest.approx(0.30)
    assert payload["metrics"]["smi_cm2_m2"] is None
    assert payload["metrics"]["smi_missing_reason"] == "DICOM PatientSize absent"


def test_b1_rejects_same_shape_mask_on_different_physical_grid(tmp_path):
    ct_path, seg_dir, dicom_dir = _write_body_comp_inputs(tmp_path, patient_size=2.0)
    mask_path = seg_dir / "tissue_types--skeletal_muscle.nii.gz"
    image = sitk.ReadImage(str(mask_path))
    image.SetOrigin((5.0, 0.0, 0.0))
    sitk.WriteImage(image, str(mask_path))

    with pytest.raises(ValueError, match="physical geometry"):
        body_composition.compute_body_composition(
            ct_nifti=ct_path,
            segmentation_dir=seg_dir,
            dicom_dir=dicom_dir,
            patient_id="P1",
            series_uid="S1",
            image_class="planning_ct",
        )


def test_b1_hu_windows_exclude_out_of_window_voxels(tmp_path):
    # Single axial slice. Muscle label has one in-window voxel (42 HU) and one
    # air voxel (-1000 HU). The air voxel must be excluded from BOTH the area
    # and the mean radiodensity (regression for the missing-HU-window bug).
    ct = np.zeros((1, 3, 3), dtype=np.int16)
    muscle = np.zeros((1, 3, 3), dtype=np.uint8)
    l3 = np.ones((1, 3, 3), dtype=np.uint8)
    muscle[0, 0, 0] = 1
    muscle[0, 0, 1] = 1
    ct[0, 0, 0] = 42       # in-window muscle
    ct[0, 0, 1] = -1000    # air partial-volume inside the label -> must be dropped

    root = tmp_path / "series"
    ct_path = root / "NIFTI" / "series.nii.gz"
    seg_dir = root / "Segmentation_TotalSegmentator" / "series"
    dicom_dir = root / "DICOM"
    _write_nifti(ct_path, ct, spacing=(10.0, 10.0, 5.0))  # pixel area = 1.0 cm^2
    _write_nifti(seg_dir / "total--vertebrae_L3.nii.gz", l3, spacing=(10.0, 10.0, 5.0))
    _write_nifti(seg_dir / "tissue_types--skeletal_muscle.nii.gz", muscle, spacing=(10.0, 10.0, 5.0))
    _write_dicom(dicom_dir / "IM_0001.dcm", patient_size=2.0)

    payload = body_composition.compute_body_composition(
        ct_nifti=ct_path,
        segmentation_dir=seg_dir,
        dicom_dir=dicom_dir,
        patient_id="P1",
        series_uid="S1",
        image_class="planning_ct",
    )
    metrics = payload["metrics"]
    # Only the 42-HU voxel counts: 1 voxel * 1.0 cm^2.
    assert metrics["skeletal_muscle_area_cm2"] == pytest.approx(1.0)
    assert metrics["skeletal_muscle_radiodensity_hu"] == pytest.approx(42.0)
    assert payload["hu_windows"]["skeletal_muscle_hu"] == [-29.0, 150.0]
    assert payload["hu_windows"]["visceral_fat_hu"] == [-150.0, -50.0]
    assert payload["hu_windows"]["subcutaneous_fat_hu"] == [-190.0, -30.0]


def test_b1_patient_size_in_cm_is_converted(tmp_path):
    # Vendor variant: PatientSize stored in cm (200.0) instead of metres.
    ct_path, seg_dir, dicom_dir = _write_body_comp_inputs(tmp_path, patient_size=200.0)
    payload = body_composition.compute_body_composition(
        ct_nifti=ct_path,
        segmentation_dir=seg_dir,
        dicom_dir=dicom_dir,
        patient_id="P1",
        series_uid="S1",
        image_class="planning_ct",
    )
    # height -> 2.0 m, SMI = 0.30 / 2.0^2 = 0.075
    assert payload["metrics"]["smi_cm2_m2"] == pytest.approx(0.075)
    assert payload["metrics"]["smi_missing_reason"] is None


def _concurrent_csv_worker(args):
    root_str, barrier = args
    from rtpipeline.body_composition import write_body_composition_csv

    barrier.wait()
    try:
        out = write_body_composition_csv(Path(root_str))
        return "ok" if out is not None else "none"
    except Exception as exc:  # pragma: no cover - failure path asserted below
        return f"{type(exc).__name__}: {exc}"


def test_b1_csv_concurrent_writers_no_corruption(tmp_path):
    import multiprocessing as mp

    n_series = 40
    for i in range(n_series):
        d = tmp_path / f"P{i % 8}" / "all_series" / "planning_ct" / f"S{i}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "body_composition.json").write_text(
            json.dumps(
                {
                    "patient_id": f"P{i % 8}",
                    "series_uid": f"S{i}",
                    "image_class": "planning_ct",
                    "status": "ok",
                    "metrics": {},
                    "l3_selection": {},
                    "hu_windows": {},
                }
            ),
            encoding="utf-8",
        )

    ctx = mp.get_context("spawn")
    with ctx.Manager() as mgr:
        barrier = mgr.Barrier(6)
        with ctx.Pool(6) as pool:
            results = pool.map(_concurrent_csv_worker, [(str(tmp_path), barrier)] * 6)

    assert all(r == "ok" for r in results), results
    csv_path = tmp_path / "Data" / "body_composition.csv"
    assert csv_path.exists()
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == n_series  # complete, not truncated/interleaved
    # No shared/leftover temp files remain.
    assert not list((tmp_path / "Data").glob("*.csv.tmp"))
