from __future__ import annotations

from rtpipeline import cli


def test_cli_inventory_only_patient_reaches_all_series_segmentation(tmp_path, monkeypatch):
    dicom_root = tmp_path / "dicom"
    dicom_root.mkdir()
    (tmp_path / "config.yaml").write_text(
        "organize:\n"
        "  do_segment_all_series: true\n"
        "  inventory_db_path: inventory.sqlite\n"
        "  inventory_patient_ids: [P_INV]\n",
        encoding="utf-8",
    )
    captured: list[str] = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "organize_and_merge", lambda cfg: [])
    monkeypatch.setattr(cli, "_detect_gpu_count", lambda: 1)
    monkeypatch.setattr(
        cli,
        "run_tasks_with_adaptive_workers",
        lambda _name, tasks, fn, **_kwargs: [fn(task) for task in tasks],
    )
    monkeypatch.setattr(
        cli,
        "_execute_all_series_segment_task",
        lambda task: captured.append(task.patient_id) or {"status": "ok"},
    )

    rc = cli.main([
        "--dicom-root", str(dicom_root),
        "--outdir", str(tmp_path / "out"),
        "--logs", str(tmp_path / "logs"),
        "--stage", "segmentation",
        "--no-metadata",
    ])

    assert rc == 0
    assert captured == ["P_INV"]
