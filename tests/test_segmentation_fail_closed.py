from __future__ import annotations

import json
import logging
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import SimpleITK as sitk

import rtpipeline.cli as cli
import rtpipeline.auto_rtstruct as auto_rtstruct
import rtpipeline.segmentation as segmentation
from rtpipeline.config import PipelineConfig
from rtpipeline.layout import build_course_dirs
from rtpipeline.utils import run_tasks_with_adaptive_workers
from course_contract_test_utils import (
    write_minimal_course_contract,
    write_synthetic_planning_ct,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_COURSE_STAGE = ROOT / "workflow" / "scripts" / "run_course_stage.py"


def _course(tmp_path: Path) -> tuple[Path, PipelineConfig]:
    course_dir = tmp_path / "output" / "P1" / "C1"
    planning_ct_dir = write_synthetic_planning_ct(course_dir)
    write_minimal_course_contract(course_dir, planning_ct_dir=planning_ct_dir)
    config = PipelineConfig(
        dicom_root=tmp_path / "input",
        output_root=tmp_path / "output",
        logs_root=tmp_path / "logs",
        segmentation_temp_root=tmp_path / "seg-tmp",
    )
    return course_dir, config


def test_totalsegmentator_failure_publishes_failed_sentinel_and_blocks_downstream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    course_dir, config = _course(tmp_path)
    monkeypatch.setattr(segmentation, "run_totalsegmentator", lambda *args, **kwargs: False)

    task = cli._SegmentTask(
        cfg=config,
        course=SimpleNamespace(dirs=build_course_dirs(course_dir)),
        force_segmentation=True,
    )

    assert cli._execute_segment_task(task) is False

    sentinel = course_dir / ".segmentation_done"
    assert sentinel.read_text(encoding="utf-8").strip() == "failed"
    report = json.loads(
        (course_dir / "metadata" / "segmentation_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "failed"
    assert report["reasons"]
    assert not (course_dir / "RS_auto.dcm").exists()
    assert not any(
        path.is_file()
        for path in (course_dir / "Segmentation_TotalSegmentator").rglob("*.nii*")
    )

    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"courses": []}\n', encoding="utf-8")
    workflow = SimpleNamespace(
        output=SimpleNamespace(sentinel=str(course_dir / ".dvh_done")),
        log=[str(tmp_path / "logs" / "dvh.log")],
        input=SimpleNamespace(manifest=str(manifest), segmentation=str(sentinel)),
        params=SimpleNamespace(
            root_dir=str(ROOT),
            configfile=str(tmp_path / "config.yaml"),
            radiomics_env="",
            python_bin=str(Path(sys.executable).parent),
            campaign_mode=False,
            stage="dvh",
            python=sys.executable,
            dicom_root=str(config.dicom_root),
            output_dir=str(config.output_root),
            logs_dir=str(config.logs_root),
            custom_structures="",
        ),
        threads=1,
        wildcards=SimpleNamespace(patient="P1", course="C1"),
    )

    with pytest.raises(RuntimeError, match="segmentation sentinel is not successful"):
        runpy.run_path(str(RUN_COURSE_STAGE), init_globals={"snakemake": workflow})
    assert not Path(workflow.output.sentinel).exists()

    # A legacy success sentinel from an older run is not trusted either. The
    # downstream gate reassesses content without rewriting the campaign course.
    sentinel.write_text("ok\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="segmentation content is not successful"):
        runpy.run_path(str(RUN_COURSE_STAGE), init_globals={"snakemake": workflow})
    assert sentinel.read_text(encoding="utf-8").strip() == "ok"
    assert not Path(workflow.output.sentinel).exists()


def test_nothing_applicable_publishes_disabled_with_reason(
    tmp_path: Path,
) -> None:
    course_dir = tmp_path / "output" / "P1" / "C1"
    write_minimal_course_contract(course_dir)
    config = PipelineConfig(
        dicom_root=tmp_path / "input",
        output_root=tmp_path / "output",
        logs_root=tmp_path / "logs",
    )
    task = cli._SegmentTask(
        cfg=config,
        course=SimpleNamespace(dirs=build_course_dirs(course_dir)),
        force_segmentation=False,
    )

    assert cli._execute_segment_task(task) is False
    assert (course_dir / ".segmentation_done").read_text(encoding="utf-8").strip() == "disabled"
    report = json.loads(
        (course_dir / "metadata" / "segmentation_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "disabled"
    assert "nothing applicable" in report["reasons"][0]


def test_failed_segmentation_result_is_not_logged_as_completed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="rtpipeline.utils"):
        results = run_tasks_with_adaptive_workers(
            "Segmentation",
            ["course"],
            lambda _item: False,
            max_workers=1,
            use_processes=False,
            show_progress=True,
            progress_success_only=True,
        )

    assert results == [False]
    progress = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Segmentation:")
    ]
    assert any("0/1 (0%)" in message for message in progress)
    assert all("1/1 (100%)" not in message for message in progress)


def test_course_segmentation_uses_nifti_only_and_never_invokes_dicom_totalseg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    course_dir, config = _course(tmp_path)
    contract = segmentation.load_course_contract(course_dir)
    assert contract.planning_ct_nifti is not None
    calls: list[dict[str, object]] = []

    def fake_totalseg(
        config,
        input_path,
        output_path,
        output_type,
        task=None,
        extra_args=None,
    ):
        calls.append(
            {
                "input_path": Path(input_path),
                "output_path": Path(output_path),
                "output_type": output_type,
                "task": task,
            }
        )
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        reference = sitk.ReadImage(str(contract.planning_ct_nifti))
        mask = sitk.Image(reference.GetSize(), sitk.sitkUInt8)
        mask.CopyInformation(reference)
        mask[0, 0, 0] = 1
        sitk.WriteImage(mask, str(output_path / "liver.nii.gz"))
        return True

    monkeypatch.setattr(segmentation, "run_totalsegmentator", fake_totalseg)
    monkeypatch.setattr(
        segmentation, "_ensure_model_rtstruct_from_masks", lambda *args, **kwargs: None
    )

    segmentation.segment_course(config, course_dir, force=True)

    assert len(calls) == 1
    assert calls[0]["input_path"] == contract.planning_ct_nifti
    assert calls[0]["output_type"] == "nifti"


@pytest.mark.parametrize(
    ("non_empty", "expected_status"),
    [(True, "ok"), (False, "failed")],
)
def test_stage_status_is_derived_from_readable_non_empty_masks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    non_empty: bool,
    expected_status: str,
) -> None:
    course_dir, config = _course(tmp_path)
    contract = segmentation.load_course_contract(course_dir)
    assert contract.planning_ct_nifti is not None

    def fake_totalseg(
        config,
        input_path,
        output_path,
        output_type,
        task=None,
        extra_args=None,
    ):
        assert Path(input_path) == contract.planning_ct_nifti
        assert output_type == "nifti"
        output_path = Path(output_path)
        masks = output_path / "segmentations"
        masks.mkdir(parents=True, exist_ok=True)
        reference = sitk.ReadImage(str(contract.planning_ct_nifti))
        mask = sitk.Image(reference.GetSize(), sitk.sitkUInt8)
        mask.CopyInformation(reference)
        if non_empty:
            mask[0, 0, 0] = 1
        sitk.WriteImage(mask, str(masks / "liver.nii.gz"))
        return True

    def fake_build(course_path: Path):
        output = Path(course_path) / "RS_auto.dcm"
        output.write_bytes(b"synthetic validated RTSTRUCT")
        return output

    monkeypatch.setattr(segmentation, "run_totalsegmentator", fake_totalseg)
    monkeypatch.setattr(
        segmentation, "_ensure_model_rtstruct_from_masks", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(auto_rtstruct, "build_auto_rtstruct", fake_build)
    monkeypatch.setattr(auto_rtstruct, "_is_valid_rtstruct", lambda path: Path(path).is_file())
    monkeypatch.setattr(
        auto_rtstruct, "_rtstruct_matches_planning_ct", lambda path, uid: True
    )
    monkeypatch.setattr(
        auto_rtstruct,
        "_derived_rtstruct_dependencies_are_current",
        lambda *args, **kwargs: True,
    )

    task = cli._SegmentTask(
        cfg=config,
        course=SimpleNamespace(dirs=build_course_dirs(course_dir)),
        force_segmentation=True,
    )
    result = cli._execute_segment_task(task)

    assert result is (expected_status == "ok")
    assert (course_dir / ".segmentation_done").read_text(encoding="utf-8").strip() == expected_status
    report = json.loads(
        (course_dir / "metadata" / "segmentation_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == expected_status
    assert report["evidence"]["total_masks_current"] is non_empty

    if not non_empty:
        forged = segmentation.publish_course_segmentation_status(
            course_dir,
            {"status": "ok", "reasons": ["caller asserted success"]},
        )
        assert forged["status"] == "failed"
        assert "caller asserted success" not in forged["reasons"]

    if non_empty:
        manifest_path = Path(report["evidence"]["manifest"])
        reference = sitk.ReadImage(str(contract.planning_ct_nifti))
        stale = sitk.Image(reference.GetSize(), sitk.sitkUInt8)
        stale.CopyInformation(reference)
        stale[0, 0, 0] = 1
        sitk.WriteImage(stale, str(manifest_path.parent / "total--stale.nii.gz"))

        stale_outcome = segmentation.publish_course_segmentation_status(course_dir)
        assert stale_outcome["status"] == "failed"
        assert "unmanifested" in stale_outcome["reasons"][0]
        assert (course_dir / ".segmentation_done").read_text(encoding="utf-8").strip() == "failed"


def test_rs_custom_failure_does_not_close_its_upstream_segmentation_stage(tmp_path: Path):
    course_dir = tmp_path / "P1" / "C1"
    metadata = course_dir / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "segmentation_resume.json").write_text(
        json.dumps(
            {
                "decisions": {
                    "RS_custom": {
                        "action": "failed",
                        "reason": (
                            "configured ROI(s) [bowel_bag] could not be built: "
                            "ValueError: missing source mask"
                        ),
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert segmentation._recorded_segmentation_failures(course_dir) == []


def test_totalsegmentator_cuda_oom_is_recorded_explicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_with_oom(command, **kwargs):
        stderr = kwargs["stderr"]
        stderr.write(b"RuntimeError: CUDA error: out of memory\n")
        stderr.flush()
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(segmentation.subprocess, "run", fail_with_oom)

    with pytest.raises(RuntimeError, match="cuda_out_of_memory"):
        segmentation._run_vec(
            ["TotalSegmentator", "-i", str(tmp_path / "ct.nii.gz")],
            timeout=10,
        )
    assert segmentation._last_totalseg_failure() == {
        "category": "cuda_out_of_memory",
        "reason": "TotalSegmentator cuda_out_of_memory with exit code 1",
    }


def test_cpu_fallback_replaces_gpu_device_instead_of_appending_second_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = PipelineConfig(
        dicom_root=tmp_path / "input",
        output_root=tmp_path / "output",
        logs_root=tmp_path / "logs",
        totalseg_device="gpu",
        totalseg_allow_fallback=True,
    )
    monkeypatch.setattr(
        segmentation, "_totalseg_supported_output_types", lambda _config: {"nifti"}
    )
    commands: list[list[str]] = []

    def fake_run_vec(command, env=None):
        commands.append(list(command))
        return len(commands) == 2

    monkeypatch.setattr(segmentation, "_run_vec", fake_run_vec)

    assert segmentation.run_totalsegmentator(
        config,
        tmp_path / "ct.nii.gz",
        tmp_path / "seg",
        "nifti",
    )
    assert len(commands) == 2
    retry = commands[1]
    assert retry.count("-d") == 1
    assert retry[retry.index("-d") + 1] == "cpu"
    assert "gpu" not in retry
