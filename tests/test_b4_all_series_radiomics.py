"""B4+C4 — all-series (non-course) CT radiomics adapter + 4DCT dedup.

Real pyradiomics/conda extraction needs the conda env (out of unit scope), so the per-course
dispatch is mocked with a fake that (a) asserts the materialized temp course-tree shape and
(b) writes a tiny ``radiomics_ct.xlsx``. That exercises everything B4+C4 own: eligible-class
selection, the C3 CBCT denylist guard, C4 per-study 4DCT dedup (ave-only / 50%-phase fallback),
temp-tree materialization, auto-RTSTRUCT discovery, provenance aggregation, temp cleanup,
serial==parallel dispatch parity, and course-path isolation (radiomics_all.xlsx untouched).
"""
import json
import os
from pathlib import Path

import pandas as pd
import pytest

import rtpipeline.radiomics as rad
import rtpipeline.radiomics_parallel as radpar
from rtpipeline.inventory import TS_TASK_BY_CLASS, output_dir_for_image_class
from rtpipeline.layout import build_course_dirs
from rtpipeline.segmentation import _series_artifact_dirs


class _Cfg:
    """Minimal config: the dispatch is mocked, so only output_root + effective_workers are used."""

    def __init__(self, output_root):
        self.output_root = Path(output_root)
        self.resume = False

    def effective_workers(self):
        return 1


def _row(image_class, series_uid, *, study_uid="S", series_description="", output_dir="x"):
    return {
        "image_class": image_class,
        "series_uid": series_uid,
        "study_uid": study_uid,
        "series_description": series_description,
        "ts_task": TS_TASK_BY_CLASS.get(image_class, "none"),
        "output_dir": output_dir,
        "status": "segmented",
    }


# --------------------------------------------------------------------------- #
# selection + C4 dedup
# --------------------------------------------------------------------------- #
def test_select_base_classes_and_c3_denylist():
    rows = [
        _row("planning_ct", "u1"),
        _row("diagnostic_ct", "u2"),
        _row("petct_ct", "u3"),
        _row("cbct", "u4"),         # C3 denylist: never radiomic'd
        _row("pt", "u5"),           # PET emission, not a CT radiomics class
        _row("mr_anatomic", "u6"),
        _row("exclude", "u7"),
    ]
    got = {r["series_uid"]: is4d for r, is4d in rad._select_all_series_radiomics_rows(rows)}
    assert got == {"u1": False, "u2": False, "u3": False}


def test_c4_ave_only_when_ave_present():
    rows = [
        _row("fourdct_ave", "ave1", study_uid="S1"),
        _row("fourdct_phase", "ph0", study_uid="S1", series_description="0%"),
        _row("fourdct_phase", "ph50", study_uid="S1", series_description="50%"),
    ]
    sel = rad._select_all_series_radiomics_rows(rows)
    assert len(sel) == 1
    row, is4d = sel[0]
    assert row["series_uid"] == "ave1" and is4d is False


def test_c4_phase_fallback_prefers_50pct():
    rows = [
        _row("fourdct_phase", "ph0", study_uid="S1", series_description="Resp 0%"),
        _row("fourdct_phase", "ph50", study_uid="S1", series_description="Resp 50%"),
        _row("fourdct_phase", "ph90", study_uid="S1", series_description="Resp 90%"),
    ]
    sel = rad._select_all_series_radiomics_rows(rows)
    assert len(sel) == 1
    row, is4d = sel[0]
    assert row["series_uid"] == "ph50" and is4d is True


def test_c4_phase_fallback_first_when_no_50pct():
    rows = [
        _row("fourdct_phase", "phA", study_uid="S1", series_description="phase A"),
        _row("fourdct_phase", "phB", study_uid="S1", series_description="phase B"),
    ]
    sel = rad._select_all_series_radiomics_rows(rows)
    assert len(sel) == 1 and sel[0][0]["series_uid"] == "phA" and sel[0][1] is True


def test_c4_independent_per_study():
    rows = [
        _row("fourdct_ave", "ave1", study_uid="S1"),
        _row("fourdct_phase", "ph1", study_uid="S2", series_description="50%"),
    ]
    got = {r["series_uid"]: is4d for r, is4d in rad._select_all_series_radiomics_rows(rows)}
    assert got == {"ave1": False, "ph1": True}


def test_pick_representative_phase_helper():
    rows = [{"series_description": "0%", "series_uid": "a"},
            {"series_description": "mid 50 %", "series_uid": "b"}]
    # whitespace-stripped match: "50 %" -> "50%"
    assert rad._pick_representative_4dct_phase(rows)["series_uid"] == "b"
    assert rad._pick_representative_4dct_phase([{"series_uid": "x"}])["series_uid"] == "x"


# --------------------------------------------------------------------------- #
# temp-tree materialization + RTSTRUCT discovery
# --------------------------------------------------------------------------- #
def test_materialize_temp_course_tree(tmp_path):
    src = tmp_path / "series"
    src.mkdir()
    for i in range(3):
        (src / f"s{i}.dcm").write_bytes(b"x")
    (src / "notdicom.txt").write_text("nope")
    rs = tmp_path / "rtstruct.dcm"
    rs.write_bytes(b"rs")
    course = tmp_path / "course"
    assert rad._materialize_temp_course_tree(course, src, rs) is True
    ct = course / "DICOM" / "CT"
    assert sorted(p.name for p in ct.glob("*.dcm")) == ["s0.dcm", "s1.dcm", "s2.dcm"]
    rs_auto = course / "RS_auto.dcm"
    assert rs_auto.exists()
    assert os.path.realpath(rs_auto) == os.path.realpath(rs)
    # only DICOM slices are linked
    assert not (ct / "notdicom.txt").exists()


def test_materialize_returns_false_when_no_slices(tmp_path):
    src = tmp_path / "empty"
    src.mkdir()
    rs = tmp_path / "rs.dcm"
    rs.write_bytes(b"r")
    assert rad._materialize_temp_course_tree(tmp_path / "c", src, rs) is False


def test_find_auto_rtstruct_matches_real_layout(tmp_path):
    # mirror segmentation: input_dir.parent/Segmentation_TotalSegmentator/input_dir.name/<base>/<base>--total.dcm
    input_dir = tmp_path / "DICOM" / "CT" / "uid123"
    input_dir.mkdir(parents=True)
    _, seg_root = _series_artifact_dirs(input_dir)
    base_dir = seg_root / "uid123_total"
    base_dir.mkdir(parents=True)
    rt = base_dir / "uid123_total--total.dcm"
    rt.write_bytes(b"rt")
    (base_dir / "total--liver.nii.gz").write_bytes(b"m")  # a mask must NOT be picked as the RTSTRUCT
    assert rad._find_all_series_auto_rtstruct(input_dir, "total") == rt


def test_find_auto_rtstruct_absent(tmp_path):
    input_dir = tmp_path / "DICOM" / "CT" / "uidX"
    input_dir.mkdir(parents=True)
    assert rad._find_all_series_auto_rtstruct(input_dir, "total") is None


# --------------------------------------------------------------------------- #
# end-to-end run_radiomics_all_series (mocked dispatch)
# --------------------------------------------------------------------------- #
def _build_patient_tree(out_root, pid, specs):
    cdirs = build_course_dirs(out_root / pid / "all_series")
    cdirs.ensure_all_series()
    rows = []
    for spec in specs:
        ic = spec["image_class"]
        uid = spec["series_uid"]
        if ic == "exclude":
            rows.append({
                "image_class": ic, "series_uid": uid, "study_uid": spec.get("study_uid", "S"),
                "series_description": spec.get("series_description", ""), "ts_task": "none",
                "output_dir": "", "status": "excluded",
            })
            continue
        outdir = output_dir_for_image_class(cdirs, ic, uid)
        outdir.mkdir(parents=True, exist_ok=True)
        for i in range(spec.get("n_slices", 2)):
            (outdir / f"img{i}.dcm").write_bytes(b"d")
        task = TS_TASK_BY_CLASS.get(ic, "none")
        if spec.get("with_rtstruct", True) and task != "none":
            _, seg_root = _series_artifact_dirs(outdir)
            base_dir = seg_root / f"{uid}_base"
            base_dir.mkdir(parents=True, exist_ok=True)
            (base_dir / f"{uid}_base--{task}.dcm").write_bytes(b"rt")
        rows.append({
            "image_class": ic, "series_uid": uid, "study_uid": spec.get("study_uid", "S"),
            "series_description": spec.get("series_description", ""), "ts_task": task,
            "output_dir": str(outdir), "status": "segmented",
        })
    (cdirs.metadata / "series_manifest.json").write_text(
        json.dumps({"patient_id": pid, "series": rows}), encoding="utf-8"
    )


@pytest.fixture
def fake_dispatch():
    """A fake per-course CT worker: asserts the temp tree shape and writes a 2-row radiomics_ct.xlsx."""
    calls = []

    def fake(config, course_dir, custom=None, *args, **kwargs):
        course_dir = Path(course_dir)
        ct = course_dir / "DICOM" / "CT"
        rs = course_dir / "RS_auto.dcm"
        calls.append({
            "course_dir": str(course_dir),
            "n_slices": len(list(ct.glob("*.dcm"))),
            "rs_auto": rs.exists(),
            "rs_target": os.path.realpath(rs) if rs.exists() else None,
            "custom": custom,
            "use_cropped": kwargs.get("use_cropped"),
        })
        df = pd.DataFrame([
            {"feature": 1.0, "roi_name": "liver", "modality": "CT",
             "course_dir": str(course_dir), "patient_id": course_dir.parent.name,
             "course_id": course_dir.name},
            {"feature": 2.0, "roi_name": "spleen", "modality": "CT",
             "course_dir": str(course_dir), "patient_id": course_dir.parent.name,
             "course_id": course_dir.name},
        ])
        out = course_dir / "radiomics_ct.xlsx"
        df.to_excel(out, index=False)
        return out

    return calls, fake


def _install(monkeypatch, fake, mode):
    if mode == "serial":
        monkeypatch.setattr(radpar, "is_parallel_radiomics_enabled", lambda: False)
        monkeypatch.setattr(rad, "radiomics_for_course", fake)
    else:
        monkeypatch.setattr(radpar, "is_parallel_radiomics_enabled", lambda: True)
        monkeypatch.setattr(radpar, "parallel_radiomics_for_course", fake)


@pytest.mark.parametrize("mode", ["serial", "parallel"])
def test_run_radiomics_all_series_e2e(tmp_path, monkeypatch, fake_dispatch, mode):
    calls, fake = fake_dispatch
    out_root = tmp_path / "out"
    pid = "P1"
    _build_patient_tree(out_root, pid, [
        {"image_class": "planning_ct", "series_uid": "u_plan", "study_uid": "S0"},
        {"image_class": "diagnostic_ct", "series_uid": "u_diag", "study_uid": "S0"},
        {"image_class": "petct_ct", "series_uid": "u_pet", "study_uid": "S0"},
        {"image_class": "cbct", "series_uid": "u_cbct", "study_uid": "S0"},   # C3 skip
        {"image_class": "pt", "series_uid": "u_pt", "study_uid": "S0"},        # not a CT class
        {"image_class": "exclude", "series_uid": "u_exc", "study_uid": "S0"},
        # 4DCT study A: ave present -> only the ave is radiomic'd
        {"image_class": "fourdct_ave", "series_uid": "u_aveA", "study_uid": "S_A"},
        {"image_class": "fourdct_phase", "series_uid": "u_phA50", "study_uid": "S_A",
         "series_description": "50%"},
        # 4DCT study B: no ave -> the 50% phase is promoted (is_4d_phase=True)
        {"image_class": "fourdct_phase", "series_uid": "u_phB0", "study_uid": "S_B",
         "series_description": "0%"},
        {"image_class": "fourdct_phase", "series_uid": "u_phB50", "study_uid": "S_B",
         "series_description": "50%"},
    ])

    _install(monkeypatch, fake, mode)
    out_csv = rad.run_radiomics_all_series(_Cfg(out_root), [pid])

    assert out_csv is not None and out_csv.exists()
    df = pd.read_csv(out_csv)

    # exactly the eligible series are radiomic'd; C3/non-CT/exclude/non-representative-4DCT skipped
    assert set(df["series_uid"]) == {"u_plan", "u_diag", "u_pet", "u_aveA", "u_phB50"}

    # provenance columns present
    for col in ("patient_id", "series_uid", "study_uid", "image_class", "is_4d_phase", "series_dir"):
        assert col in df.columns
    assert set(df["patient_id"]) == {"P1"}

    # is_4d_phase True ONLY for the fallback-promoted phase
    is4d = {u: bool(v) for u, v in zip(df["series_uid"], df["is_4d_phase"].astype(bool))}
    assert is4d["u_phB50"] is True
    assert all(v is False for u, v in is4d.items() if u != "u_phB50")

    # 5 series * 2 feature rows
    assert len(df) == 10

    # course-path artifacts NOT created; temp tree cleaned up
    assert not (out_root / "Data" / "radiomics_all.xlsx").exists()
    assert not (out_root / ".all_series_radiomics").exists()

    # dispatch saw the correct minimal temp tree for every series
    assert len(calls) == 5
    for c in calls:
        assert c["n_slices"] == 2
        assert c["rs_auto"] is True
        assert c["custom"] is None
        assert c["use_cropped"] in (False, None)  # parallel passes False; serial defaults False


def test_missing_manifest_returns_none(tmp_path, monkeypatch, fake_dispatch):
    calls, fake = fake_dispatch
    _install(monkeypatch, fake, "serial")
    assert rad.run_radiomics_all_series(_Cfg(tmp_path / "out"), ["NOPE"]) is None
    assert len(calls) == 0


def test_series_without_rtstruct_is_skipped(tmp_path, monkeypatch, fake_dispatch):
    calls, fake = fake_dispatch
    out_root = tmp_path / "out"
    pid = "P2"
    _build_patient_tree(out_root, pid, [
        {"image_class": "planning_ct", "series_uid": "u_ok", "study_uid": "S", "with_rtstruct": True},
        {"image_class": "planning_ct", "series_uid": "u_nort", "study_uid": "S", "with_rtstruct": False},
    ])
    _install(monkeypatch, fake, "serial")
    out_csv = rad.run_radiomics_all_series(_Cfg(out_root), [pid])
    df = pd.read_csv(out_csv)
    assert set(df["series_uid"]) == {"u_ok"}
    assert len(calls) == 1
    assert not (out_root / ".all_series_radiomics").exists()


def test_empty_patient_list_returns_none(tmp_path, monkeypatch, fake_dispatch):
    calls, fake = fake_dispatch
    _install(monkeypatch, fake, "serial")
    assert rad.run_radiomics_all_series(_Cfg(tmp_path / "out"), []) is None
    assert len(calls) == 0
