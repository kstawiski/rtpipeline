from __future__ import annotations

import json
import sqlite3
import sys
import types
from pathlib import Path

import numpy as np

from rtpipeline import organize, segmentation
from rtpipeline.config import PipelineConfig
from rtpipeline.inventory import manual_rtstruct_bindings_from_inventory
from rtpipeline.layout import build_course_dirs
from rtpipeline.organize import CourseOutput


class _FakeImage:
    def GetSize(self):
        return (2, 2, 1)

    def GetSpacing(self):
        return (1.0, 1.0, 1.0)

    def GetDirection(self):
        return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    def GetOrigin(self):
        return (0.0, 0.0, 0.0)


class _FakeMaskImage:
    def SetSpacing(self, value):
        self.spacing = value

    def SetDirection(self, value):
        self.direction = value

    def SetOrigin(self, value):
        self.origin = value


def _patch_manual_export_dependencies(monkeypatch, calls: list[dict]) -> None:
    class _FakeBuilder:
        @classmethod
        def create_from(cls, *, dicom_series_path: str, rt_struct_path: str):
            calls.append(
                {
                    "dicom_series_path": Path(dicom_series_path),
                    "rt_struct_path": Path(rt_struct_path),
                }
            )
            return cls()

        def get_roi_names(self):
            return ["liver"]

        def get_roi_mask_by_name(self, roi_name: str):
            mask = np.zeros((1, 2, 2), dtype=bool)
            mask[0, 0, 0] = True
            return mask

    fake_sitk = types.SimpleNamespace(
        ReadImage=lambda path: _FakeImage(),
        GetImageFromArray=lambda array: _FakeMaskImage(),
        WriteImage=lambda image, path, useCompression=True: Path(path).write_text("mask", encoding="utf-8"),
    )
    monkeypatch.setitem(sys.modules, "rt_utils", types.SimpleNamespace(RTStructBuilder=_FakeBuilder))
    monkeypatch.setattr(organize, "sitk", fake_sitk)


def _cfg(tmp_path: Path, db_path: Path | None = None) -> PipelineConfig:
    return PipelineConfig(
        dicom_root=tmp_path / "dicom",
        output_root=tmp_path / "out",
        logs_root=tmp_path / "logs",
        inventory_db_path=db_path,
    )


def _write_inventory_db(db_path: Path, *, rtstruct_path: Path, target_series_uid: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE studies (study_uid TEXT PRIMARY KEY, patient_id TEXT);
            CREATE TABLE series (series_uid TEXT PRIMARY KEY, study_uid TEXT);
            CREATE TABLE dicom_files (file_id INTEGER PRIMARY KEY, file_path TEXT);
            CREATE TABLE instances (
                sop_instance_uid TEXT PRIMARY KEY,
                series_uid TEXT,
                modality TEXT,
                primary_file_id INTEGER
            );
            CREATE TABLE rt_links (
                source_sop_uid TEXT,
                relationship TEXT,
                target_series_uid TEXT,
                target_for_uid TEXT,
                target_study_uid TEXT
            );
            """
        )
        conn.execute("INSERT INTO studies VALUES (?, ?)", ("RT_STUDY", "P1"))
        conn.execute("INSERT INTO series VALUES (?, ?)", ("RT_SERIES", "RT_STUDY"))
        conn.execute("INSERT INTO dicom_files VALUES (?, ?)", (1, str(rtstruct_path)))
        conn.execute(
            "INSERT INTO instances VALUES (?, ?, ?, ?)",
            ("RT_SOP", "RT_SERIES", "RTSTRUCT", 1),
        )
        conn.execute(
            "INSERT INTO rt_links VALUES (?, ?, ?, ?, ?)",
            ("RT_SOP", "rtstruct_to_series", target_series_uid, "", "PLAN_STUDY"),
        )
        conn.commit()
    finally:
        conn.close()


def _write_manual_binding_inventory_db(db_path: Path, links: list[dict]) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE studies (study_uid TEXT PRIMARY KEY, patient_id TEXT);
            CREATE TABLE series (series_uid TEXT PRIMARY KEY, study_uid TEXT);
            CREATE TABLE dicom_files (file_id INTEGER PRIMARY KEY, file_path TEXT);
            CREATE TABLE instances (
                sop_instance_uid TEXT PRIMARY KEY,
                series_uid TEXT,
                modality TEXT,
                primary_file_id INTEGER
            );
            CREATE TABLE rt_links (
                source_sop_uid TEXT,
                relationship TEXT,
                target_series_uid TEXT,
                target_for_uid TEXT,
                target_study_uid TEXT
            );
            """
        )
        for idx, link in enumerate(links, start=1):
            sop_uid = str(link["source_sop_uid"])
            rtstruct_path = Path(link["source_path"])
            study_uid = f"RT_STUDY_{idx}"
            series_uid = f"RT_SERIES_{idx}"
            conn.execute("INSERT INTO studies VALUES (?, ?)", (study_uid, "P1"))
            conn.execute("INSERT INTO series VALUES (?, ?)", (series_uid, study_uid))
            conn.execute("INSERT INTO dicom_files VALUES (?, ?)", (idx, str(rtstruct_path)))
            conn.execute(
                "INSERT INTO instances VALUES (?, ?, ?, ?)",
                (sop_uid, series_uid, "RTSTRUCT", idx),
            )
            conn.execute(
                "INSERT INTO rt_links VALUES (?, ?, ?, ?, ?)",
                (
                    sop_uid,
                    str(link["relationship"]),
                    str(link.get("target_series_uid") or ""),
                    str(link.get("target_for_uid") or ""),
                    str(link.get("target_study_uid") or ""),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _write_old_inventory_without_rt_links(db_path: Path, rtstruct_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE studies (study_uid TEXT PRIMARY KEY, patient_id TEXT);
            CREATE TABLE series (series_uid TEXT PRIMARY KEY, study_uid TEXT);
            CREATE TABLE dicom_files (file_id INTEGER PRIMARY KEY, file_path TEXT);
            CREATE TABLE instances (
                sop_instance_uid TEXT PRIMARY KEY,
                series_uid TEXT,
                modality TEXT,
                primary_file_id INTEGER
            );
            """
        )
        conn.execute("INSERT INTO studies VALUES (?, ?)", ("RT_STUDY", "P1"))
        conn.execute("INSERT INTO series VALUES (?, ?)", ("RT_SERIES", "RT_STUDY"))
        conn.execute("INSERT INTO dicom_files VALUES (?, ?)", (1, str(rtstruct_path)))
        conn.execute("INSERT INTO instances VALUES (?, ?, ?, ?)", ("RT_SOP", "RT_SERIES", "RTSTRUCT", 1))
        conn.commit()
    finally:
        conn.close()


def test_manual_rtstruct_binding_uses_for_unique_and_single_row_fallback(tmp_path: Path) -> None:
    db_path = tmp_path / "inventory.sqlite"
    rs_unique = tmp_path / "rs_for_unique.dcm"
    rs_single = tmp_path / "rs_single_row_fallback.dcm"
    _write_manual_binding_inventory_db(
        db_path,
        [
            {
                "source_sop_uid": "RT_FOR_UNIQUE",
                "source_path": rs_unique,
                "relationship": "rtstruct_to_for",
                "target_study_uid": "STUDY_FOR_UNIQUE",
                "target_for_uid": "FOR_UNIQUE",
            },
            {
                "source_sop_uid": "RT_FOR_SINGLE",
                "source_path": rs_single,
                "relationship": "rtstruct_to_for",
                "target_study_uid": "STUDY_FOR_SINGLE",
                "target_for_uid": "FOR_SINGLE",
            },
        ],
    )
    rows = [
        {
            "series_uid": "SER_FOR_UNIQUE",
            "study_uid": "STUDY_FOR_UNIQUE",
            "frame_of_reference_uid": "FOR_UNIQUE",
            "rt_link_basis": "rtstruct_to_for_unique",
        },
        {
            "series_uid": "SER_FOR_SINGLE",
            "study_uid": "STUDY_FOR_SINGLE",
            "frame_of_reference_uid": "FOR_SINGLE",
        },
    ]

    bindings = manual_rtstruct_bindings_from_inventory(db_path, "P1", rows)

    assert bindings == {
        "SER_FOR_UNIQUE": rs_unique,
        "SER_FOR_SINGLE": rs_single,
    }


def test_manual_rtstruct_binding_skips_ambiguous_for_matches(tmp_path: Path, caplog) -> None:
    db_path = tmp_path / "inventory.sqlite"
    _write_manual_binding_inventory_db(
        db_path,
        [
            {
                "source_sop_uid": "RT_FOR_A",
                "source_path": tmp_path / "rs_for_a.dcm",
                "relationship": "rtstruct_to_for",
                "target_study_uid": "STUDY_FOR",
                "target_for_uid": "FOR_AMBIG",
            },
            {
                "source_sop_uid": "RT_FOR_B",
                "source_path": tmp_path / "rs_for_b.dcm",
                "relationship": "rtstruct_to_for",
                "target_study_uid": "STUDY_FOR",
                "target_for_uid": "FOR_AMBIG",
            },
        ],
    )
    rows = [
        {
            "series_uid": "SER_FOR",
            "study_uid": "STUDY_FOR",
            "frame_of_reference_uid": "FOR_AMBIG",
            "rt_link_basis": "rtstruct_to_for_unique",
        }
    ]

    caplog.set_level("WARNING", logger="rtpipeline.inventory")
    bindings = manual_rtstruct_bindings_from_inventory(db_path, "P1", rows)

    assert bindings == {}
    assert "Multiple manual RTSTRUCTs reference patient P1 study STUDY_FOR FrameOfReferenceUID FOR_AMBIG" in caplog.text


def test_manual_rtstruct_binding_skips_multiple_exact_series_matches(tmp_path: Path, caplog) -> None:
    db_path = tmp_path / "inventory.sqlite"
    _write_manual_binding_inventory_db(
        db_path,
        [
            {
                "source_sop_uid": "RT_EXACT_A",
                "source_path": tmp_path / "rs_exact_a.dcm",
                "relationship": "rtstruct_to_series",
                "target_series_uid": "SER_PLAN",
            },
            {
                "source_sop_uid": "RT_EXACT_B",
                "source_path": tmp_path / "rs_exact_b.dcm",
                "relationship": "rtstruct_to_series",
                "target_series_uid": "SER_PLAN",
            },
        ],
    )

    caplog.set_level("WARNING", logger="rtpipeline.inventory")
    bindings = manual_rtstruct_bindings_from_inventory(db_path, "P1", [{"series_uid": "SER_PLAN"}])

    assert bindings == {}
    assert "Multiple manual RTSTRUCTs reference patient P1 series SER_PLAN" in caplog.text


def test_manual_rtstruct_binding_missing_or_old_inventory_is_noop(tmp_path: Path, caplog) -> None:
    rows = [{"series_uid": "SER_PLAN", "study_uid": "STUDY", "frame_of_reference_uid": "FOR"}]
    assert manual_rtstruct_bindings_from_inventory(None, "P1", rows) == {}

    old_db_path = tmp_path / "old_inventory.sqlite"
    _write_old_inventory_without_rt_links(old_db_path, tmp_path / "rs.dcm")

    caplog.set_level("WARNING", logger="rtpipeline.inventory")
    assert manual_rtstruct_bindings_from_inventory(old_db_path, "P1", rows) == {}
    assert "D1 original-export disabled" in caplog.text
    assert "inventory schema/rt_links" in caplog.text


def test_original_segmentation_export_reuses_cached_manifest_on_rerun(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict] = []
    _patch_manual_export_dependencies(monkeypatch, calls)

    rs_path = tmp_path / "RS.dcm"
    rs_path.write_text("rs", encoding="utf-8")
    primary_nifti = tmp_path / "ct.nii.gz"
    primary_nifti.write_text("nifti", encoding="utf-8")
    dicom_ct_dir = tmp_path / "DICOM"
    dicom_ct_dir.mkdir()
    (dicom_ct_dir / "IM_0001.dcm").write_text("ct", encoding="utf-8")
    segmentation_original_dir = tmp_path / "Segmentation_Original"

    first = organize._export_original_segmentation_from_paths(
        rs_path=rs_path,
        primary_nifti=primary_nifti,
        dicom_ct_dir=dicom_ct_dir,
        segmentation_original_dir=segmentation_original_dir,
        log_root=tmp_path,
        overwrite=False,
    )
    assert first is not None
    cached = dict(first)
    cached["cached"] = True
    manifest_path = segmentation_original_dir / "ct" / "metadata.json"
    manifest_path.write_text(json.dumps(cached, indent=2), encoding="utf-8")

    second = organize._export_original_segmentation_from_paths(
        rs_path=rs_path,
        primary_nifti=primary_nifti,
        dicom_ct_dir=dicom_ct_dir,
        segmentation_original_dir=segmentation_original_dir,
        log_root=tmp_path,
        overwrite=False,
    )

    assert second == cached
    assert len(calls) == 1


def test_per_course_original_export_keeps_legacy_manifest_schema(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict] = []
    _patch_manual_export_dependencies(monkeypatch, calls)

    course_dir = tmp_path / "course"
    dirs = build_course_dirs(course_dir)
    dirs.ensure()
    rs_path = course_dir / "RS.dcm"
    rs_path.write_text("rs", encoding="utf-8")
    primary_nifti = dirs.nifti / "ct.nii.gz"
    primary_nifti.write_text("nifti", encoding="utf-8")
    (dirs.dicom_ct / "IM_0001.dcm").write_text("ct", encoding="utf-8")
    course = CourseOutput(
        patient_id="P1",
        course_key="C1",
        course_id="C1",
        course_start=None,
        dirs=dirs,
        rp_path=course_dir / "RP.dcm",
        rd_path=course_dir / "RD.dcm",
        rs_path=rs_path,
        primary_nifti=primary_nifti,
        related_dicom=[],
        total_prescription_gy=None,
    )

    manifest = organize._export_original_segmentation(course, overwrite=True)

    expected = {
        "model": "manual",
        "source_rtstruct": str(rs_path),
        "source_nifti": str(primary_nifti),
        "structures": [{"roi_name": "liver", "mask": "ct/liver.nii.gz"}],
    }
    assert manifest == expected
    assert json.loads((dirs.segmentation_original / "ct" / "metadata.json").read_text()) == expected
    assert "source" not in manifest
    assert calls == [{"dicom_series_path": dirs.dicom_ct, "rt_struct_path": rs_path}]


def test_all_series_original_export_binds_only_referenced_series_and_keeps_ai_separate(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    calls: list[dict] = []
    _patch_manual_export_dependencies(monkeypatch, calls)

    rtstruct_path = tmp_path / "source_rs.dcm"
    rtstruct_path.write_text("rs", encoding="utf-8")
    db_path = tmp_path / "inventory.sqlite"
    _write_inventory_db(db_path, rtstruct_path=rtstruct_path, target_series_uid="SER_PLAN")
    cfg = _cfg(tmp_path, db_path=db_path)

    all_series_root = cfg.output_root / "P1" / "all_series"
    cdirs = build_course_dirs(all_series_root)
    cdirs.ensure_all_series()
    plan_dir = cdirs.dicom_ct / "plan"
    diag_dir = cdirs.dicom_ct_diagnostic / "diag"
    plan_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)
    (plan_dir / "IM_0001.dcm").write_text("ct", encoding="utf-8")
    (diag_dir / "IM_0001.dcm").write_text("ct", encoding="utf-8")
    rows = [
        {
            "patient_id": "P1",
            "study_uid": "PLAN_STUDY",
            "series_uid": "SER_PLAN",
            "modality": "CT",
            "image_class": "planning_ct",
            "frame_of_reference_uid": "FOR_PLAN",
            "rt_link_basis": "rtstruct_to_series",
            "n_slices": 1,
            "ts_task": "total",
            "output_dir": str(plan_dir),
            "status": "materialized",
        },
        {
            "patient_id": "P1",
            "study_uid": "DIAG_STUDY",
            "series_uid": "SER_DIAG",
            "modality": "CT",
            "image_class": "diagnostic_ct",
            "frame_of_reference_uid": "FOR_DIAG",
            "n_slices": 1,
            "ts_task": "total",
            "output_dir": str(diag_dir),
            "status": "materialized",
        },
    ]
    manifest_path = cdirs.metadata / "series_manifest.json"
    manifest_path.write_text(json.dumps({"patient_id": "P1", "series": rows}, indent=2), encoding="utf-8")

    def fake_ensure_nifti(config, input_dir, nifti_dir, force=False, dcm2niix_depth=None):
        assert dcm2niix_depth == 0
        nifti_dir.mkdir(parents=True, exist_ok=True)
        path = nifti_dir / f"{Path(input_dir).name}.nii.gz"
        path.write_text("nifti", encoding="utf-8")
        return path

    def fake_totalseg(config, input_path, output_path, output_type, task=None, extra_args=None):
        if output_type == "dicom_rtstruct":
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("auto rtstruct", encoding="utf-8")
        else:
            masks = output_path / "segmentations"
            masks.mkdir(parents=True, exist_ok=True)
            (masks / "liver.nii.gz").write_text("ai liver", encoding="utf-8")
        return True

    monkeypatch.setattr(segmentation, "_ensure_ct_nifti", fake_ensure_nifti)
    monkeypatch.setattr(segmentation, "run_totalsegmentator", fake_totalseg)

    caplog.set_level("DEBUG", logger="rtpipeline.segmentation")
    summary = segmentation.segment_all_series_for_patient(cfg, "P1")

    assert summary["planning_ct"]["segmented"] == 1
    assert summary["diagnostic_ct"]["segmented"] == 1
    assert calls == [{"dicom_series_path": plan_dir, "rt_struct_path": rtstruct_path}]

    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_uid = {row["series_uid"]: row for row in persisted["series"]}
    manual_manifest_path = Path(by_uid["SER_PLAN"]["manual_segmentation_manifest"])
    assert manual_manifest_path.exists()
    assert "manual_segmentation_manifest" not in by_uid["SER_DIAG"]
    assert "event=no_original_available" in caplog.text
    assert "series_uid=SER_DIAG" in caplog.text
    assert any(
        record.levelname == "DEBUG" and "event=no_original_available" in record.getMessage()
        for record in caplog.records
    )

    manual_manifest = json.loads(manual_manifest_path.read_text(encoding="utf-8"))
    assert manual_manifest["model"] == "manual"
    assert manual_manifest["structures"] == [{"roi_name": "liver", "mask": "plan/liver.nii.gz"}]
    manual_mask = manual_manifest_path.parent / "liver.nii.gz"
    assert manual_mask.exists()

    ai_manifest_path = next((plan_dir.parent / "Segmentation_TotalSegmentator" / "plan").glob("*/manifest.json"))
    ai_manifest = json.loads(ai_manifest_path.read_text(encoding="utf-8"))
    assert ai_manifest["models"][0]["model"] == "total"
    assert ai_manifest["models"][0]["masks"] == ["total--liver.nii.gz"]
    ai_mask = ai_manifest_path.parent / "total--liver.nii.gz"
    assert ai_mask.exists()
    assert manual_mask != ai_mask
