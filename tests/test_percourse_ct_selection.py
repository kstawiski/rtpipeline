"""Tests for the per-course CT-selection fix.

Covers:
  Part 1 - CT indexer (`index_ct_series` / `_index_tcia_patient_series_layout`):
    - study-level directory with multiple CT series is grouped by real SeriesInstanceUID
    - genuine TCIA one-series-per-dir layout still groups one series per dir (no regression)
    - study-level dir whose FIRST sorted file is non-CT still indexes the CT series
  Part 1b - `copy_ct_series`: purges stale dst contents; materialises dst even when the
    copy-manager deduplicates a SOP to a foreign path.
  Part 2 - `select_course_ct_series`: RTSTRUCT-referenced planning CT, deterministic
    tie-break, fail-closed on unresolved references, legacy fallback when no references.

All DICOM is synthesised with pydicom; no database or external data is required.
"""
from __future__ import annotations

from pathlib import Path

import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid, ExplicitVRLittleEndian, CTImageStorage

from rtpipeline.ct import index_ct_series, copy_ct_series, CTInstance, _clear_dir
from rtpipeline import organize as org


# --------------------------------------------------------------------------- helpers
def _mk_dcm(path: Path, *, patient: str, study: str, series: str,
            modality: str = "CT", instance: int = 1) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = FileMetaDataset()
    sop = generate_uid()
    fm.MediaStorageSOPClassUID = CTImageStorage
    fm.MediaStorageSOPInstanceUID = sop
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=fm, preamble=b"\0" * 128)
    ds.PatientID = patient
    ds.Modality = modality
    ds.StudyInstanceUID = study
    ds.SeriesInstanceUID = series
    ds.SOPInstanceUID = sop
    ds.InstanceNumber = instance
    ds.save_as(str(path), write_like_original=False)
    return sop


def _mk_rtstruct(path: Path, *, patient: str, study: str, series: str,
                 referenced_ct_series: list[str]) -> None:
    """Minimal RTSTRUCT whose ReferencedFrameOfReferenceSequence points at CT series."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.3"  # RT Structure Set Storage
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=fm, preamble=b"\0" * 128)
    ds.PatientID = patient
    ds.Modality = "RTSTRUCT"
    ds.StudyInstanceUID = study
    ds.SeriesInstanceUID = series
    ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
    rt_series_items = []
    for cs in referenced_ct_series:
        rt_ser = Dataset()
        rt_ser.SeriesInstanceUID = cs
        rt_series_items.append(rt_ser)
    rt_study = Dataset()
    rt_study.RTReferencedSeriesSequence = Sequence(rt_series_items)
    ref_for = Dataset()
    ref_for.FrameOfReferenceUID = generate_uid()
    ref_for.RTReferencedStudySequence = Sequence([rt_study])
    ds.ReferencedFrameOfReferenceSequence = Sequence([ref_for])
    ds.save_as(str(path), write_like_original=False)


def _ct_instances(n: int, *, series: str = "S", study: str = "ST") -> list:
    return [CTInstance(path=Path(f"/dev/null"), patient_id="PT1", study_uid=study,
                       series_uid=series, series_number=1, instance_number=i) for i in range(n)]


# --------------------------------------------------------------------------- Part 1: indexer
def test_study_dir_with_multiple_ct_series_is_grouped_by_real_series(tmp_path):
    """T1: a STUDY-level dir (name != series UID) holding 2 CT series + 1 RTSTRUCT must
    index as 2 CT series (not one lump), excluding the RTSTRUCT."""
    study = generate_uid()
    s1, s2 = generate_uid(), generate_uid()
    d = tmp_path / "PT1" / study  # directory named by STUDY uid (not series)
    for i in range(5):
        _mk_dcm(d / f"CT_a_{i}.dcm", patient="PT1", study=study, series=s1, instance=i)
    for i in range(3):
        _mk_dcm(d / f"CT_b_{i}.dcm", patient="PT1", study=study, series=s2, instance=i)
    _mk_rtstruct(d / "CT_zz_rs.dcm", patient="PT1", study=study, series=generate_uid(),
                 referenced_ct_series=[s1])

    idx = index_ct_series(tmp_path)
    assert "PT1" in idx and study in idx["PT1"]
    series_map = idx["PT1"][study]
    assert set(series_map.keys()) == {s1, s2}, "must group by real SeriesInstanceUID"
    assert len(series_map[s1]) == 5 and len(series_map[s2]) == 3
    # RTSTRUCT (its own series UID) must not appear as a CT series
    assert all(k in (s1, s2) for k in series_map)


def test_genuine_tcia_one_series_per_dir_no_regression(tmp_path):
    """T2: genuine PatientID/SeriesInstanceUID/*.dcm layout still yields one series per dir."""
    study = generate_uid()
    s1, s2 = generate_uid(), generate_uid()
    for i in range(4):
        _mk_dcm(tmp_path / "PT1" / s1 / f"{i}.dcm", patient="PT1", study=study, series=s1, instance=i)
    for i in range(6):
        _mk_dcm(tmp_path / "PT1" / s2 / f"{i}.dcm", patient="PT1", study=study, series=s2, instance=i)

    idx = index_ct_series(tmp_path)
    assert set(idx["PT1"][study].keys()) == {s1, s2}
    assert len(idx["PT1"][study][s1]) == 4 and len(idx["PT1"][study][s2]) == 6


def test_study_dir_first_file_non_ct_still_indexes_ct(tmp_path):
    """T2b: study dir whose first SORTED file is non-CT must still bail to the slow path
    and index the CT series (closes the first-file-non-CT gap)."""
    study = generate_uid()
    s1 = generate_uid()
    d = tmp_path / "PT1" / study
    # '000_' sorts before 'CT_' so the first file the fast path reads is the RTSTRUCT.
    _mk_rtstruct(d / "000_rs.dcm", patient="PT1", study=study, series=generate_uid(),
                 referenced_ct_series=[s1])
    for i in range(5):
        _mk_dcm(d / f"CT_{i}.dcm", patient="PT1", study=study, series=s1, instance=i)

    idx = index_ct_series(tmp_path)
    assert study in idx.get("PT1", {}), "CT series must be indexed despite non-CT first file"
    assert set(idx["PT1"][study].keys()) == {s1}
    assert len(idx["PT1"][study][s1]) == 5


# --------------------------------------------------------------------------- Part 2: selection
def test_selection_prefers_referenced_smaller_series(tmp_path):
    """T3: RTSTRUCT references the SMALLER series -> select that, not the largest."""
    study = "ST"
    big, small = "SER_BIG", "SER_SMALL"
    ct_index = {"PT1": {study: {big: _ct_instances(214, series=big), small: _ct_instances(52, series=small)}}}
    rs = tmp_path / "rs.dcm"
    _mk_rtstruct(rs, patient="PT1", study="rstudy", series=generate_uid(), referenced_ct_series=[small])

    series, status = org.select_course_ct_series(ct_index, "PT1", rs, study)
    assert status == "referenced"
    assert series is not None and series[0].series_uid == small
    assert len(series) == 52


def test_selection_no_references_falls_back_to_largest(tmp_path):
    """T4: no RTSTRUCT references -> legacy pick_primary_series (largest)."""
    study = "ST"
    big, small = "SER_BIG", "SER_SMALL"
    ct_index = {"PT1": {study: {big: _ct_instances(214, series=big), small: _ct_instances(52, series=small)}}}
    # struct_source_path None -> no refs
    series, status = org.select_course_ct_series(ct_index, "PT1", None, study)
    assert status == "fallback_largest"
    assert series[0].series_uid == big and len(series) == 214


def test_selection_unresolved_reference_fails_closed(tmp_path):
    """T5: RTSTRUCT references a CT series that is NOT indexed -> fail closed (None)."""
    study = "ST"
    ct_index = {"PT1": {study: {"SER_A": _ct_instances(100, series="SER_A")}}}
    rs = tmp_path / "rs.dcm"
    _mk_rtstruct(rs, patient="PT1", study="rstudy", series=generate_uid(),
                 referenced_ct_series=["SER_NOT_INDEXED"])
    series, status = org.select_course_ct_series(ct_index, "PT1", rs, study)
    assert status == "unresolved_reference"
    assert series is None


def test_selection_multiple_resolved_deterministic_tiebreak(tmp_path):
    """T6: multiple referenced series resolve -> deterministic tie-break (most slices, then lowest uid)."""
    study = "ST"
    a, b = "SER_A", "SER_B"
    ct_index = {"PT1": {study: {a: _ct_instances(60, series=a), b: _ct_instances(120, series=b)}}}
    rs = tmp_path / "rs.dcm"
    _mk_rtstruct(rs, patient="PT1", study="rstudy", series=generate_uid(), referenced_ct_series=[a, b])
    series, status = org.select_course_ct_series(ct_index, "PT1", rs, study)
    assert status == "referenced_multi"
    assert series[0].series_uid == b and len(series) == 120  # most slices wins


def test_selection_tiebreak_equal_slices_lowest_uid(tmp_path):
    """T6b: equal slice counts -> deterministic tie-break picks the LOWEST series_uid."""
    study = "ST"
    hi, lo = "SER_ZZZ", "SER_AAA"  # equal sizes; 'SER_AAA' < 'SER_ZZZ'
    ct_index = {"PT1": {study: {hi: _ct_instances(80, series=hi), lo: _ct_instances(80, series=lo)}}}
    rs = tmp_path / "rs.dcm"
    _mk_rtstruct(rs, patient="PT1", study="rstudy", series=generate_uid(), referenced_ct_series=[hi, lo])
    series, status = org.select_course_ct_series(ct_index, "PT1", rs, study)
    assert status == "referenced_multi"
    assert series[0].series_uid == lo, "equal slice counts -> lowest series_uid wins"


def test_selection_searches_across_studies(tmp_path):
    """Referenced CT series in a DIFFERENT study than course_study is still resolved."""
    other_study, course_study = "ST_OTHER", "ST_COURSE"
    ref = "SER_REF"
    ct_index = {"PT1": {other_study: {ref: _ct_instances(80, series=ref)},
                        course_study: {"SER_X": _ct_instances(200, series="SER_X")}}}
    rs = tmp_path / "rs.dcm"
    _mk_rtstruct(rs, patient="PT1", study="rstudy", series=generate_uid(), referenced_ct_series=[ref])
    series, status = org.select_course_ct_series(ct_index, "PT1", rs, course_study)
    assert status == "referenced" and series[0].series_uid == ref


# --------------------------------------------------------------------------- Part 1b: copy_ct_series
def test_copy_ct_series_purges_stale_dst(tmp_path):
    """T7: stale files in the destination are removed before copying the new series."""
    src = tmp_path / "src"
    src.mkdir()
    real = src / "real.dcm"
    _mk_dcm(real, patient="PT1", study="ST", series="S", instance=1)
    dst = tmp_path / "dst"
    dst.mkdir()
    (dst / "STALE_garbage.dcm").write_bytes(b"old")  # leftover from a prior wrong run
    inst = CTInstance(path=real, patient_id="PT1", study_uid="ST", series_uid="S",
                      series_number=1, instance_number=1)
    copy_ct_series([inst], dst, copy_manager=None)
    names = sorted(p.name for p in dst.iterdir())
    assert "STALE_garbage.dcm" not in names
    assert any(n.startswith("CT_") for n in names) and len(names) == 1


class _DedupCM:
    """Copy manager stub that simulates SOP dedup returning a FOREIGN existing path."""
    def __init__(self, foreign: Path):
        self.foreign = foreign

    def copy_dicom(self, src, dst, skip_if_exists=True):
        return self.foreign, False  # deduped elsewhere; does NOT create dst


def test_copy_ct_series_materializes_dst_on_foreign_dedup(tmp_path):
    """T8: when the copy manager dedups a SOP to a foreign path, the file is still placed at dst."""
    src = tmp_path / "src"
    src.mkdir()
    foreign = tmp_path / "all_series_copy.dcm"
    _mk_dcm(foreign, patient="PT1", study="ST", series="S", instance=1)
    real = src / "real.dcm"
    _mk_dcm(real, patient="PT1", study="ST", series="S", instance=1)
    dst = tmp_path / "course_ct"
    inst = CTInstance(path=real, patient_id="PT1", study_uid="ST", series_uid="S",
                      series_number=1, instance_number=1)
    copy_ct_series([inst], dst, copy_manager=_DedupCM(foreign))
    files = list(dst.glob("CT_*.dcm"))
    assert len(files) == 1 and files[0].exists(), "per-course CT must be materialised despite dedup"


# --------------------------------------------------------------------------- fail-closed cleanup
class _FakeCourseDirs:
    def __init__(self, root: Path):
        self.dicom_ct = root / "DICOM" / "CT"
        self.nifti = root / "NIFTI"
        self.segmentation_totalseg = root / "Segmentation_TotalSegmentator"
        for d in (self.dicom_ct, self.nifti, self.segmentation_totalseg):
            d.mkdir(parents=True, exist_ok=True)


def test_clear_dir_removes_files_and_subdirs(tmp_path):
    d = tmp_path / "d"
    (d / "sub").mkdir(parents=True)
    (d / "a.dcm").write_bytes(b"x")
    (d / "sub" / "b.dcm").write_bytes(b"y")
    _clear_dir(d)
    assert d.is_dir() and list(d.iterdir()) == []
    # no-op on missing / non-dir
    _clear_dir(tmp_path / "missing")
    f = tmp_path / "file"
    f.write_bytes(b"z")
    _clear_dir(f)
    assert f.exists()
    # symlink-to-dir -> no-op: must NOT delete through the link (would touch files outside the tree)
    target = tmp_path / "target"
    target.mkdir()
    (target / "keep.dcm").write_bytes(b"k")
    link = tmp_path / "link"
    link.symlink_to(target)
    _clear_dir(link)
    assert (target / "keep.dcm").exists(), "must not delete through a symlinked directory"


def test_fail_closed_clears_stale_percourse_ct_outputs(tmp_path):
    """BLOCKER regression: a fail-closed course must not retain a prior run's wrong CT /
    NIfTI / auto-OAR segmentation that a re-run could otherwise segment or hydrate."""
    cd = _FakeCourseDirs(tmp_path / "course")
    # Pre-populate with stale artifacts from a previous (wrong) run.
    (cd.dicom_ct / "CT_00001.dcm").write_bytes(b"stale wrong CT")
    (cd.nifti / "ct.nii.gz").write_bytes(b"stale nifti")
    (cd.segmentation_totalseg / "total--liver.nii.gz").write_bytes(b"stale mask")
    org._clear_course_ct_outputs(cd)
    assert list(cd.dicom_ct.iterdir()) == [], "stale CT must be removed on fail-closed"
    assert list(cd.nifti.iterdir()) == [], "stale NIfTI must be removed on fail-closed"
    assert list(cd.segmentation_totalseg.iterdir()) == [], "stale OAR seg must be removed on fail-closed"
