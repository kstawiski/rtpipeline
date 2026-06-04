from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from rtpipeline.config import PipelineConfig
from rtpipeline.inventory import InventoryInstance, _copy_instances
from rtpipeline.layout import build_course_dirs
import rtpipeline.cli as cli
import rtpipeline.segmentation as segmentation


def _config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        dicom_root=tmp_path / "dicom",
        output_root=tmp_path / "out",
        logs_root=tmp_path / "logs",
    )


def _series_rows(root: Path) -> list[dict]:
    classes = [
        ("planning_ct", "total", "materialized"),
        ("diagnostic_ct", "total", "materialized"),
        ("petct_ct", "total", "materialized"),
        ("fourdct_ave", "total", "materialized"),
        ("fourdct_phase", "total", "materialized"),
        ("cbct", "total", "materialized"),
        ("mr_anatomic", "total_mr", "materialized"),
        ("mr_functional", "none", "materialized"),
        ("pt", "none", "materialized"),
        ("exclude", "none", "excluded"),
    ]
    rows: list[dict] = []
    for index, (image_class, task, status) in enumerate(classes, start=1):
        output_dir = root / image_class / f"series-{index}" / ("DICOM" if image_class.startswith("mr_") else "")
        if image_class != "exclude":
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "IM_0001.dcm").write_text("not real dicom", encoding="utf-8")
        rows.append(
            {
                "patient_id": "P1",
                "study_uid": f"study-{index}",
                "series_uid": f"series-{index}",
                "modality": "MR" if image_class.startswith("mr_") else "CT",
                "image_class": image_class,
                "manufacturer_model": "",
                "frame_of_reference_uid": f"for-{index}",
                "n_slices": 25,
                "ts_task": task,
                "output_dir": str(output_dir) if image_class != "exclude" else "",
                "status": status,
                "exclusion_reason": "test_exclude" if image_class == "exclude" else "",
            }
        )
    return rows


def test_segment_all_series_dispatch_status_and_idempotency(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    all_series_root = cfg.output_root / "P1" / "all_series"
    manifest_path = build_course_dirs(all_series_root).metadata / "series_manifest.json"
    rows = _series_rows(all_series_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"patient_id": "P1", "series": rows}, indent=2), encoding="utf-8")

    def fake_ensure_nifti(config, input_dir, nifti_dir, force=False, dcm2niix_depth=None):
        assert dcm2niix_depth == 0
        nifti_dir.mkdir(parents=True, exist_ok=True)
        path = nifti_dir / f"{Path(input_dir).parent.name if Path(input_dir).name == 'DICOM' else Path(input_dir).name}.nii.gz"
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
            (masks / "liver.nii.gz").write_text("fake mask", encoding="utf-8")
        return True

    monkeypatch.setattr(segmentation, "_ensure_ct_nifti", fake_ensure_nifti)
    monkeypatch.setattr(segmentation, "run_totalsegmentator", fake_totalseg)

    summary = segmentation.segment_all_series_for_patient(cfg, "P1")

    eligible_classes = {
        "planning_ct",
        "diagnostic_ct",
        "petct_ct",
        "fourdct_ave",
        "fourdct_phase",
        "cbct",
        "mr_anatomic",
    }
    assert set(summary) == eligible_classes
    assert all(summary[image_class]["attempted"] == 1 for image_class in eligible_classes)
    assert all(summary[image_class]["segmented"] == 1 for image_class in eligible_classes)
    assert len(calls) == 14

    seg_root_by_class = {
        row["image_class"]: segmentation._series_artifact_dirs(Path(row["output_dir"]))[1]
        for row in rows
        if row["output_dir"]
    }

    def class_for_output(output_path: Path) -> str:
        resolved = output_path.resolve()
        for image_class, seg_root in seg_root_by_class.items():
            if seg_root.resolve() in resolved.parents:
                return image_class
        raise AssertionError(f"Unexpected RTSTRUCT output path: {output_path}")

    dicom_calls = [call for call in calls if call["output_type"] == "dicom_rtstruct"]
    by_class = {class_for_output(call["output_path"]): call for call in dicom_calls}
    for image_class in {"planning_ct", "diagnostic_ct", "petct_ct", "fourdct_ave", "fourdct_phase"}:
        assert by_class[image_class]["task"] == "total"
        assert by_class[image_class]["extra_args"] is None
    assert by_class["cbct"]["task"] == "total"
    assert by_class["cbct"]["extra_args"] == ["--body_seg"]
    assert by_class["mr_anatomic"]["task"] == "total_mr"
    assert by_class["mr_anatomic"]["extra_args"] is None
    assert "mr_functional" not in by_class
    assert "pt" not in by_class
    assert "exclude" not in by_class

    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    status_by_class = {row["image_class"]: row["status"] for row in persisted["series"]}
    for image_class in eligible_classes:
        assert status_by_class[image_class] == "segmented"
    assert status_by_class["mr_functional"] == "materialized"
    assert status_by_class["pt"] == "materialized"
    assert status_by_class["exclude"] == "excluded"

    calls.clear()
    skipped = segmentation.segment_all_series_for_patient(cfg, "P1")
    assert calls == []
    assert all(skipped[image_class]["skipped"] == 1 for image_class in eligible_classes)
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    status_by_class = {row["image_class"]: row["status"] for row in persisted["series"]}
    for image_class in eligible_classes:
        assert status_by_class[image_class] == "seg_skipped_idempotent"

    calls.clear()
    forced = segmentation.segment_all_series_for_patient(cfg, "P1", force=True)
    assert len(calls) == 14
    assert all(forced[image_class]["attempted"] == 1 for image_class in eligible_classes)
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    status_by_class = {row["image_class"]: row["status"] for row in persisted["series"]}
    for image_class in eligible_classes:
        assert status_by_class[image_class] == "segmented"


def test_run_totalsegmentator_extra_args_shell_free_and_shell(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    monkeypatch.setattr(segmentation, "_totalseg_supported_output_types", lambda config: {"nifti"})

    vector_commands: list[list[str]] = []

    def fake_run_vec(cmd, env=None):
        vector_commands.append(list(cmd))
        return True

    monkeypatch.setattr(segmentation, "_run_vec", fake_run_vec)
    assert segmentation.run_totalsegmentator(
        cfg,
        tmp_path / "input.nii.gz",
        tmp_path / "out",
        "nifti",
        task="total",
        extra_args=None,
    )
    baseline = vector_commands[-1]
    assert baseline == [
        "TotalSegmentator",
        "-i",
        str(tmp_path / "input.nii.gz"),
        "-o",
        str(tmp_path / "out"),
        "-ot",
        "nifti",
        "--task",
        "total",
        "--force_split",
        "-d",
        "gpu",
    ]

    assert segmentation.run_totalsegmentator(
        cfg,
        tmp_path / "input.nii.gz",
        tmp_path / "out",
        "nifti",
        task="total",
        extra_args=["--body_seg"],
    )
    with_extra = vector_commands[-1]
    assert with_extra == baseline[:9] + ["--body_seg"] + baseline[9:]

    shell_commands: list[str] = []

    def fake_run(cmd, env=None):
        shell_commands.append(cmd)
        return True

    cfg.conda_activate = "source activate rt"
    monkeypatch.setattr(segmentation, "_run", fake_run)
    assert segmentation.run_totalsegmentator(
        cfg,
        tmp_path / "input.nii.gz",
        tmp_path / "out",
        "nifti",
        task="total",
        extra_args=["--body_seg"],
    )
    assert "--task total --body_seg" in shell_commands[-1]


def test_totalseg_probe_includes_dicom_rtstruct_and_command_uses_output_type(tmp_path, monkeypatch):
    segmentation._totalseg_supported_output_types_cached.cache_clear()
    help_text = """
  -ot OUTPUT_TYPE [OUTPUT_TYPE ...], --output_type OUTPUT_TYPE [OUTPUT_TYPE ...]
                        Select output type(s). Choices: nifti, dicom_rtstruct,
                        dicom_seg. Multiple are allowed e.g. -ot nifti
"""

    def fake_probe(*args, **kwargs):
        return SimpleNamespace(stdout=help_text)

    monkeypatch.setattr(segmentation.subprocess, "run", fake_probe)
    supported = segmentation._totalseg_supported_output_types_cached("", "TotalSegmentator")
    assert {"nifti", "dicom_rtstruct", "dicom_seg"} <= supported

    cfg = _config(tmp_path)
    monkeypatch.setattr(segmentation, "_totalseg_supported_output_types", lambda config: supported)
    commands: list[list[str]] = []

    def fake_run_vec(cmd, env=None):
        commands.append(list(cmd))
        return True

    monkeypatch.setattr(segmentation, "_run_vec", fake_run_vec)
    rtstruct_path = tmp_path / "out" / "RS.total.dcm"
    assert segmentation.run_totalsegmentator(
        cfg,
        tmp_path / "dicom",
        rtstruct_path,
        "dicom_rtstruct",
        task="total",
    )
    assert commands[-1][commands[-1].index("-ot") + 1] == "dicom_rtstruct"
    assert commands[-1][commands[-1].index("-o") + 1] == str(rtstruct_path)


def test_segment_all_series_rtstruct_failure_keeps_successful_nifti_segmented(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    all_series_root = cfg.output_root / "P1" / "all_series"
    manifest_path = build_course_dirs(all_series_root).metadata / "series_manifest.json"
    rows = [row for row in _series_rows(all_series_root) if row["image_class"] == "planning_ct"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"patient_id": "P1", "series": rows}, indent=2), encoding="utf-8")

    def fake_ensure_nifti(config, input_dir, nifti_dir, force=False, dcm2niix_depth=None):
        nifti_dir.mkdir(parents=True, exist_ok=True)
        path = nifti_dir / "planning.nii.gz"
        path.write_text("fake nifti", encoding="utf-8")
        return path

    def fake_totalseg(config, input_path, output_path, output_type, task=None, extra_args=None):
        if output_type == "dicom_rtstruct":
            return False
        output_path.mkdir(parents=True, exist_ok=True)
        masks = output_path / "segmentations"
        masks.mkdir(parents=True, exist_ok=True)
        (masks / "liver.nii.gz").write_text("fake mask", encoding="utf-8")
        return True

    monkeypatch.setattr(segmentation, "_ensure_ct_nifti", fake_ensure_nifti)
    monkeypatch.setattr(segmentation, "run_totalsegmentator", fake_totalseg)

    summary = segmentation.segment_all_series_for_patient(cfg, "P1")
    assert summary["planning_ct"] == {"attempted": 1, "segmented": 1, "failed": 0, "skipped": 0}

    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = persisted["series"][0]
    assert row["status"] == "segmented"

    input_dir = Path(row["output_dir"])
    series_manifest = next((input_dir.parent / "Segmentation_TotalSegmentator" / input_dir.name).glob("*/manifest.json"))
    model_entry = json.loads(series_manifest.read_text(encoding="utf-8"))["models"][0]
    assert model_entry["rtstruct_ok"] is False
    assert model_entry["rtstruct"] == ""
    assert model_entry["masks"]


def test_segment_all_series_nifti_failure_isolates_row(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    all_series_root = cfg.output_root / "P1" / "all_series"
    manifest_path = build_course_dirs(all_series_root).metadata / "series_manifest.json"
    rows = [
        row
        for row in _series_rows(all_series_root)
        if row["image_class"] in {"planning_ct", "diagnostic_ct"}
    ]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"patient_id": "P1", "series": rows}, indent=2), encoding="utf-8")

    def fake_ensure_nifti(config, input_dir, nifti_dir, force=False, dcm2niix_depth=None):
        if Path(input_dir).name == "series-1":
            return None
        nifti_dir.mkdir(parents=True, exist_ok=True)
        path = nifti_dir / "diagnostic.nii.gz"
        path.write_text("fake nifti", encoding="utf-8")
        return path

    def fake_totalseg(config, input_path, output_path, output_type, task=None, extra_args=None):
        if output_type == "nifti":
            output_path.mkdir(parents=True, exist_ok=True)
            masks = output_path / "segmentations"
            masks.mkdir(parents=True, exist_ok=True)
            (masks / "liver.nii.gz").write_text("fake mask", encoding="utf-8")
        elif output_type == "dicom_rtstruct":
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("fake rtstruct", encoding="utf-8")
        return True

    monkeypatch.setattr(segmentation, "_ensure_ct_nifti", fake_ensure_nifti)
    monkeypatch.setattr(segmentation, "run_totalsegmentator", fake_totalseg)

    summary = segmentation.segment_all_series_for_patient(cfg, "P1")
    assert summary["planning_ct"]["failed"] == 1
    assert summary["diagnostic_ct"]["segmented"] == 1

    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    status_by_class = {row["image_class"]: row["status"] for row in persisted["series"]}
    assert status_by_class == {"planning_ct": "seg_failed", "diagnostic_ct": "segmented"}


def test_segment_all_series_malformed_ts_task_fails_closed(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    all_series_root = cfg.output_root / "P1" / "all_series"
    manifest_path = build_course_dirs(all_series_root).metadata / "series_manifest.json"
    rows = [row for row in _series_rows(all_series_root) if row["image_class"] in {"planning_ct", "pt"}]
    rows[0]["ts_task"] = "total_mr"
    rows[1]["ts_task"] = "total"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"patient_id": "P1", "series": rows}, indent=2), encoding="utf-8")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("TotalSegmentator path should not be reached for malformed ts_task")

    monkeypatch.setattr(segmentation, "_ensure_ct_nifti", fail_if_called)
    monkeypatch.setattr(segmentation, "run_totalsegmentator", fail_if_called)

    summary = segmentation.segment_all_series_for_patient(cfg, "P1")
    assert summary["planning_ct"]["failed"] == 1
    assert summary["pt"]["failed"] == 1

    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert {row["image_class"]: row["status"] for row in persisted["series"]} == {
        "planning_ct": "seg_failed",
        "pt": "seg_failed",
    }
    assert all("invalid ts_task" in row["segmentation_error"] for row in persisted["series"])


def test_segment_all_series_force_uses_sibling_outputs_and_depth_zero(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    all_series_root = cfg.output_root / "P1" / "all_series"
    manifest_path = build_course_dirs(all_series_root).metadata / "series_manifest.json"
    rows = [row for row in _series_rows(all_series_root) if row["image_class"] == "planning_ct"]
    input_dir = Path(rows[0]["output_dir"])
    stale_inside = input_dir / "Segmentation_TotalSegmentator" / "stale.dcm"
    stale_inside.parent.mkdir(parents=True, exist_ok=True)
    stale_inside.write_text("stale", encoding="utf-8")
    stale_nifti = input_dir / "NIFTI" / "old.nii.gz"
    stale_nifti.parent.mkdir(parents=True, exist_ok=True)
    stale_nifti.write_text("stale nifti", encoding="utf-8")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"patient_id": "P1", "series": rows}, indent=2), encoding="utf-8")

    seen: dict[str, object] = {}

    def is_under(child: Path, parent: Path) -> bool:
        child = child.resolve()
        parent = parent.resolve()
        return child == parent or parent in child.parents

    def fake_ensure_nifti(config, input_path, nifti_dir, force=False, dcm2niix_depth=None):
        seen["nifti_dir"] = Path(nifti_dir)
        seen["dcm2niix_depth"] = dcm2niix_depth
        assert not is_under(Path(nifti_dir), input_dir)
        nifti_dir.mkdir(parents=True, exist_ok=True)
        path = nifti_dir / "planning.nii.gz"
        path.write_text("fake nifti", encoding="utf-8")
        return path

    def fake_totalseg(config, input_path, output_path, output_type, task=None, extra_args=None):
        assert not is_under(Path(output_path), input_dir)
        if output_type == "dicom_rtstruct":
            seen["rtstruct_input"] = Path(input_path)
            assert Path(input_path) != input_dir
            assert sorted(p.name for p in Path(input_path).iterdir()) == ["IM_0001.dcm"]
            assert not any(p.name == "stale.dcm" for p in Path(input_path).rglob("*"))
            assert not any(p.name == "old.nii.gz" for p in Path(input_path).rglob("*"))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("fake rtstruct", encoding="utf-8")
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            masks = output_path / "segmentations"
            masks.mkdir(parents=True, exist_ok=True)
            (masks / "liver.nii.gz").write_text("fake mask", encoding="utf-8")
        return True

    monkeypatch.setattr(segmentation, "_ensure_ct_nifti", fake_ensure_nifti)
    monkeypatch.setattr(segmentation, "run_totalsegmentator", fake_totalseg)

    summary = segmentation.segment_all_series_for_patient(cfg, "P1", force=True)
    assert summary["planning_ct"]["segmented"] == 1
    assert seen["dcm2niix_depth"] == 0
    assert seen["rtstruct_input"] != input_dir
    assert stale_inside.exists()
    assert stale_nifti.exists()


def test_copy_instances_drops_mixed_localizer_but_keeps_minimum_volume(tmp_path):
    sources = tmp_path / "src"
    sources.mkdir()
    axial_instances = []
    for index in range(1, 11):
        path = sources / f"axial_{index}.dcm"
        path.write_text("axial", encoding="utf-8")
        axial_instances.append(
            InventoryInstance(f"axial-{index}", path, index, "ORIGINAL\\PRIMARY\\AXIAL", None, index)
        )
    localizer_path = sources / "localizer.dcm"
    localizer_path.write_text("localizer", encoding="utf-8")
    localizer = InventoryInstance("localizer", localizer_path, 99, "ORIGINAL\\PRIMARY\\LOCALIZER", None, 99)

    out = tmp_path / "out"
    out.mkdir()
    assert _copy_instances([*axial_instances, localizer], out, "CT") is False
    assert len(list(out.glob("*.dcm"))) == 10
    assert not (out / "CT_00099.dcm").exists()
    (out / "CT_00099.dcm").write_text("stale localizer", encoding="utf-8")
    assert _copy_instances(axial_instances, out, "CT") is False
    assert len(list(out.glob("*.dcm"))) == 10
    assert not (out / "CT_00099.dcm").exists()

    guarded_out = tmp_path / "guarded"
    guarded_out.mkdir()
    assert _copy_instances([*axial_instances[:9], localizer], guarded_out, "CT") is False
    assert len(list(guarded_out.glob("*.dcm"))) == 10
    assert (guarded_out / "CT_00099.dcm").exists()


def test_cli_do_segment_all_series_false_never_calls_all_series_segmentation(tmp_path, monkeypatch):
    dicom_root = tmp_path / "dicom"
    dicom_root.mkdir()
    course_dirs = build_course_dirs(tmp_path / "out" / "P1" / "course")
    course_dirs.ensure()
    course = SimpleNamespace(patient_id="P1", course_id="course", dirs=course_dirs)
    runner_names: list[str] = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "organize_and_merge", lambda cfg: [course])
    monkeypatch.setattr(cli, "_detect_gpu_count", lambda: 0)

    def fake_runner(name, tasks, fn, **kwargs):
        runner_names.append(name)
        assert name != "All-series segmentation"
        return [True for _ in tasks]

    monkeypatch.setattr(cli, "run_tasks_with_adaptive_workers", fake_runner)

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
    assert runner_names == ["Segmentation"]
