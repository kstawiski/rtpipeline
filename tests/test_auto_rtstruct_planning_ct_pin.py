"""Regression tests for C1: auto-RTSTRUCT must bind masks to the planning CT.

``build_auto_rtstruct`` builds ``RS_auto.dcm`` against the course planning CT
(``course_dirs.dicom_ct``) but historically chose the mask source as the first
*alphabetically sorted* ``Segmentation_TotalSegmentator`` subdir that had a
``--total.dcm`` (and fell back to ``candidate_dirs[0]``). In all-series mode the
course can have several segmented series (planning_ct, CBCT, 4DCT phases,
diagnostic CT), so the blind first-dir pick can bind masks from the WRONG series /
FrameOfReference onto the planning CT, producing anatomically wrong structures.

C1 fixes this by selecting the seg dir built FROM the planning CT, using the
strongest signal available, fail-closed at every rung:

  1. **Source-series identity** — the seg dir whose ``--total.dcm`` references the
     planning CT's SeriesInstanceUID. This is the only signal that disambiguates
     4DCT phases / reconstructions that *share* a FrameOfReferenceUID but are
     distinct series.
  2. **FrameOfReferenceUID** — coarser fallback when no source-series link is
     readable.
  3. **Lone-candidate back-compat** — a single candidate with no contradicting
     readable identity (legacy single-series / NIfTI-only dirs).

Critically, the rtpipeline ``--total.dcm`` default output type is
``dicom_rtstruct`` (SOP 481.3, an RT Structure Set), whose FrameOfReferenceUID and
source-series link live in the *nested* ``ReferencedFrameOfReferenceSequence`` —
NOT the top-level ``FrameOfReferenceUID`` tag. The fixtures below reproduce that
real structure; an earlier draft used a DICOM-SEG with a top-level FoR, which does
not exercise the production code path.

``_geometry_compatible`` is the universal pre-resample safety net (applied to both
the combined-segmentation path and the per-ROI binary-mask fallback): if the
selected segmentation does not share the CT's physical space (origin / spacing /
direction within tolerance), the build aborts instead of emitting a wrong RTSTRUCT.
"""
from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from rtpipeline.auto_rtstruct import (
    _geometry_compatible,
    _read_for_uid,
    _seg_source_series_uids,
    _select_seg_dir_for_ct,
)

RTSTRUCT_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.3"  # RT Structure Set Storage
SEG_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.66.4"  # DICOM Segmentation Storage


def _new_dataset(sop_class: str, modality: str) -> FileDataset:
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = sop_class
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Modality = modality
    return ds


def _write_total_rtstruct(
    seg_dir: Path, for_uid: str, ref_series_uid: str | None
) -> Path:
    """Write a realistic ``<name>--total.dcm`` RTSTRUCT (the rtpipeline default).

    FrameOfReferenceUID and the referenced source CT series live in the *nested*
    ``ReferencedFrameOfReferenceSequence`` — there is deliberately NO top-level
    ``FrameOfReferenceUID``, matching TotalSegmentator's ``dicom_rtstruct`` output.
    Pass ``ref_series_uid=None`` to omit the source-series link (so only the FoR
    fallback can resolve it).
    """
    seg_dir.mkdir(parents=True, exist_ok=True)
    out = seg_dir / f"{seg_dir.name}--total.dcm"
    ds = _new_dataset(RTSTRUCT_SOP_CLASS, "RTSTRUCT")

    ref_for = Dataset()
    ref_for.FrameOfReferenceUID = for_uid
    if ref_series_uid is not None:
        ref_series = Dataset()
        ref_series.SeriesInstanceUID = ref_series_uid
        rt_ref_study = Dataset()
        rt_ref_study.RTReferencedSeriesSequence = Sequence([ref_series])
        ref_for.RTReferencedStudySequence = Sequence([rt_ref_study])
    ds.ReferencedFrameOfReferenceSequence = Sequence([ref_for])
    ds.save_as(out, enforce_file_format=True)
    return out


def _write_total_seg(seg_dir: Path, for_uid: str, ref_series_uid: str | None = None) -> Path:
    """Write a DICOM-SEG ``<name>--total.dcm`` (legacy ``dicom_seg`` output type).

    SEGs carry a top-level ``FrameOfReferenceUID`` and reference their source
    series via the top-level ``ReferencedSeriesSequence``.
    """
    seg_dir.mkdir(parents=True, exist_ok=True)
    out = seg_dir / f"{seg_dir.name}--total.dcm"
    ds = _new_dataset(SEG_SOP_CLASS, "SEG")
    ds.FrameOfReferenceUID = for_uid
    if ref_series_uid is not None:
        ref_series = Dataset()
        ref_series.SeriesInstanceUID = ref_series_uid
        ds.ReferencedSeriesSequence = Sequence([ref_series])
    ds.save_as(out, enforce_file_format=True)
    return out


def _img(origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 3.0),
         direction=(1, 0, 0, 0, 1, 0, 0, 0, 1), size=(8, 8, 8)) -> sitk.Image:
    im = sitk.Image(int(size[0]), int(size[1]), int(size[2]), sitk.sitkUInt8)
    im.SetOrigin(tuple(float(v) for v in origin))
    im.SetSpacing(tuple(float(v) for v in spacing))
    im.SetDirection(tuple(float(v) for v in direction))
    return im


# --------------------------------------------------------------------------- #
# _read_for_uid — must read the RTSTRUCT *nested* FrameOfReferenceUID           #
# --------------------------------------------------------------------------- #

def test_read_for_uid_rtstruct_nested(tmp_path: Path) -> None:
    """The bug Codex caught: --total.dcm is an RTSTRUCT with no top-level FoR."""
    for_uid = generate_uid()
    seg = _write_total_rtstruct(tmp_path / "planning", for_uid, ref_series_uid=generate_uid())
    assert _read_for_uid(seg) == for_uid


def test_read_for_uid_seg_toplevel(tmp_path: Path) -> None:
    for_uid = generate_uid()
    seg = _write_total_seg(tmp_path / "planning", for_uid)
    assert _read_for_uid(seg) == for_uid


def test_read_for_uid_unreadable(tmp_path: Path) -> None:
    bogus = tmp_path / "not_a_dicom.dcm"
    bogus.write_bytes(b"not dicom")
    assert _read_for_uid(bogus) == ""


# --------------------------------------------------------------------------- #
# _seg_source_series_uids — RTSTRUCT (RTReferencedSeriesSequence) + SEG path    #
# --------------------------------------------------------------------------- #

def test_source_series_from_rtstruct(tmp_path: Path) -> None:
    series_uid = generate_uid()
    seg = _write_total_rtstruct(tmp_path / "planning", generate_uid(), ref_series_uid=series_uid)
    assert _seg_source_series_uids(seg) == {series_uid}


def test_source_series_from_seg(tmp_path: Path) -> None:
    series_uid = generate_uid()
    seg = _write_total_seg(tmp_path / "planning", generate_uid(), ref_series_uid=series_uid)
    assert _seg_source_series_uids(seg) == {series_uid}


def test_source_series_absent_when_no_link(tmp_path: Path) -> None:
    seg = _write_total_rtstruct(tmp_path / "planning", generate_uid(), ref_series_uid=None)
    assert _seg_source_series_uids(seg) == set()


# --------------------------------------------------------------------------- #
# _select_seg_dir_for_ct — source-series identity is primary                   #
# --------------------------------------------------------------------------- #

def test_picks_series_matched_dir_among_same_for(tmp_path: Path) -> None:
    """The killer case: two 4DCT phases share ONE FoR; only series identity
    distinguishes the planning phase. Alpha-first and FoR-only both pick wrong."""
    shared_for = generate_uid()
    planning_series = generate_uid()
    phase_series = generate_uid()
    # "aaa_phase" sorts first; it shares the FoR but is a different series.
    _write_total_rtstruct(tmp_path / "aaa_phase", shared_for, ref_series_uid=phase_series)
    planning = tmp_path / "zzz_planning"
    planning_dcm = _write_total_rtstruct(planning, shared_for, ref_series_uid=planning_series)
    candidates = sorted(p for p in tmp_path.iterdir() if p.is_dir())

    selected_dir, seg_path, base_name = _select_seg_dir_for_ct(
        candidates, planning_series, shared_for
    )
    assert selected_dir == planning
    assert seg_path == planning_dcm
    assert base_name == "zzz_planning"


def test_single_matching_series(tmp_path: Path) -> None:
    series_uid = generate_uid()
    only = tmp_path / "planning"
    only_dcm = _write_total_rtstruct(only, generate_uid(), ref_series_uid=series_uid)
    selected_dir, seg_path, base_name = _select_seg_dir_for_ct([only], series_uid, generate_uid())
    assert (selected_dir, seg_path, base_name) == (only, only_dcm, "planning")


def test_for_fallback_when_no_series_link(tmp_path: Path) -> None:
    """No source-series link readable, but FoR disambiguates one of several dirs."""
    ct_for = generate_uid()
    _write_total_rtstruct(tmp_path / "aaa_other", generate_uid(), ref_series_uid=None)
    planning = tmp_path / "zzz_planning"
    planning_dcm = _write_total_rtstruct(planning, ct_for, ref_series_uid=None)
    candidates = sorted(p for p in tmp_path.iterdir() if p.is_dir())
    # ct_series_uid present but no dir references it -> rung 1 finds nothing,
    # rung 2 (FoR) resolves the planning dir.
    selected_dir, seg_path, base_name = _select_seg_dir_for_ct(
        candidates, generate_uid(), ct_for
    )
    assert selected_dir == planning
    assert seg_path == planning_dcm


def test_fail_closed_same_for_multiple_no_series(tmp_path: Path) -> None:
    """4DCT phases share a FoR, none references the planning series -> fail closed."""
    shared_for = generate_uid()
    _write_total_rtstruct(tmp_path / "phase00", shared_for, ref_series_uid=generate_uid())
    _write_total_rtstruct(tmp_path / "phase50", shared_for, ref_series_uid=generate_uid())
    candidates = sorted(p for p in tmp_path.iterdir() if p.is_dir())
    selected_dir, seg_path, base_name = _select_seg_dir_for_ct(
        candidates, generate_uid(), shared_for
    )
    assert (selected_dir, seg_path, base_name) == (None, None, None)


# --------------------------------------------------------------------------- #
# _select_seg_dir_for_ct — lone-candidate back-compat / NIfTI-only             #
# --------------------------------------------------------------------------- #

def test_lone_nifti_only_fallback(tmp_path: Path) -> None:
    """Single NIfTI-only dir (no --total.dcm): used, downstream geometry-checked.
    This is the regression all three reviewers flagged in the first C1 cut."""
    only = tmp_path / "planning"
    only.mkdir()
    (only / "planning_total_multilabel.nii.gz").write_bytes(b"")  # presence only
    selected_dir, seg_path, base_name = _select_seg_dir_for_ct(
        [only], generate_uid(), generate_uid()
    )
    assert selected_dir == only
    assert seg_path is None
    assert base_name == "planning"


def test_lone_candidate_contradicting_series_fail_closed(tmp_path: Path) -> None:
    """The only seg dir references a DIFFERENT series than the planning CT."""
    only = tmp_path / "diagnostic"
    _write_total_rtstruct(only, generate_uid(), ref_series_uid=generate_uid())
    selected_dir, seg_path, base_name = _select_seg_dir_for_ct(
        [only], generate_uid(), generate_uid()
    )
    assert (selected_dir, seg_path, base_name) == (None, None, None)


def test_lone_candidate_contradicting_for_fail_closed(tmp_path: Path) -> None:
    """Only seg dir: no series link, FoR contradicts the planning CT -> fail closed."""
    only = tmp_path / "cbct"
    _write_total_rtstruct(only, generate_uid(), ref_series_uid=None)
    selected_dir, seg_path, base_name = _select_seg_dir_for_ct(
        [only], generate_uid(), generate_uid()
    )
    assert (selected_dir, seg_path, base_name) == (None, None, None)


def test_single_series_backcompat_when_ct_unreadable(tmp_path: Path) -> None:
    """If neither CT series UID nor CT FoR can be read, a lone candidate is usable."""
    only = tmp_path / "planning"
    only_dcm = _write_total_rtstruct(only, generate_uid(), ref_series_uid=generate_uid())
    selected_dir, seg_path, base_name = _select_seg_dir_for_ct([only], "", "")
    assert (selected_dir, seg_path, base_name) == (only, only_dcm, "planning")


def test_fail_closed_multiple_when_ct_unreadable(tmp_path: Path) -> None:
    """Several candidates + no CT identity to disambiguate -> never guess."""
    _write_total_rtstruct(tmp_path / "aaa", generate_uid(), ref_series_uid=generate_uid())
    _write_total_rtstruct(tmp_path / "bbb", generate_uid(), ref_series_uid=generate_uid())
    candidates = sorted(p for p in tmp_path.iterdir() if p.is_dir())
    selected_dir, seg_path, base_name = _select_seg_dir_for_ct(candidates, "", "")
    assert selected_dir is None


def test_no_candidates(tmp_path: Path) -> None:
    assert _select_seg_dir_for_ct([], generate_uid(), generate_uid()) == (None, None, None)


# --------------------------------------------------------------------------- #
# _geometry_compatible — the pre-resample safety net                           #
# --------------------------------------------------------------------------- #

def test_geometry_identical_is_compatible() -> None:
    assert _geometry_compatible(_img(), _img()) is True


def test_geometry_small_origin_shift_within_tolerance() -> None:
    assert _geometry_compatible(_img(origin=(0.5, 0.0, 0.0)), _img()) is True


def test_geometry_large_origin_shift_incompatible() -> None:
    assert _geometry_compatible(_img(origin=(250.0, 120.0, -400.0)), _img()) is False


def test_geometry_different_direction_incompatible() -> None:
    rotated = _img(direction=(0, 1, 0, 1, 0, 0, 0, 0, 1))
    assert _geometry_compatible(rotated, _img()) is False


def test_geometry_nonidentity_direction_self_compatible() -> None:
    swap = dict(direction=(0, 1, 0, 1, 0, 0, 0, 0, 1), size=(20, 4, 4), spacing=(1, 1, 1))
    assert _geometry_compatible(_img(**swap), _img(**swap)) is True


def test_geometry_nonidentity_direction_extent_is_orientation_safe() -> None:
    """Discriminating case for the _extent fix. Axis-swap direction + non-square
    size: the two volumes are physically DISJOINT once direction cosines are
    honored. The old ``origin + (size-1)*spacing`` math (ignoring direction)
    computed overlapping extents and wrongly returned True (a dangerous false
    positive that would resample a cross-frame mask)."""
    swap = (0, 1, 0, 1, 0, 0, 0, 0, 1)
    a = _img(origin=(0.0, 0.0, 0.0), spacing=(1, 1, 1), direction=swap, size=(20, 4, 4))
    b = _img(origin=(8.0, 0.0, 0.0), spacing=(1, 1, 1), direction=swap, size=(20, 4, 4))
    assert _geometry_compatible(a, b) is False


# --------------------------------------------------------------------------- #
# build_auto_rtstruct — end-to-end selection -> RS_auto wiring                  #
# --------------------------------------------------------------------------- #

def _write_ct_slice(ct_dir: Path, series_uid: str, for_uid: str) -> None:
    ct_dir.mkdir(parents=True, exist_ok=True)
    ds = _new_dataset("1.2.840.10008.5.1.4.1.1.2", "CT")  # CT Image Storage
    ds.SeriesInstanceUID = series_uid
    ds.FrameOfReferenceUID = for_uid
    ds.save_as(ct_dir / "ct_000.dcm", enforce_file_format=True)


def test_build_auto_rtstruct_selects_planning_among_same_for(tmp_path, monkeypatch) -> None:
    """End-to-end: two same-FoR RTSTRUCT seg dirs; RS_auto must copy the one whose
    --total.dcm references the planning CT series, not the alpha-first phase."""
    from rtpipeline import auto_rtstruct as ar

    course = tmp_path / "course"
    dirs = ar.build_course_dirs(course)
    shared_for = generate_uid()
    planning_series = generate_uid()
    phase_series = generate_uid()
    _write_ct_slice(dirs.dicom_ct, planning_series, shared_for)

    seg_root = dirs.segmentation_totalseg
    _write_total_rtstruct(seg_root / "aaa_phase", shared_for, ref_series_uid=phase_series)
    planning_dcm = _write_total_rtstruct(
        seg_root / "zzz_planning", shared_for, ref_series_uid=planning_series
    )

    # Isolate the C1 selection->copy wiring from SimpleITK CT loading and the
    # RTSTRUCT post-processing (not under test here).
    monkeypatch.setattr(ar, "_load_ct_image", lambda d: _img())
    monkeypatch.setattr(ar, "sanitize_rtstruct", lambda p: None)
    monkeypatch.setattr(ar, "fix_rtstruct_rois", lambda ct, p: None)

    out = ar.build_auto_rtstruct(course)
    assert out is not None and out.exists()
    # The copied RS_auto must carry the PLANNING series' provenance.
    assert planning_series in _seg_source_series_uids(out)
    assert phase_series not in _seg_source_series_uids(out)
    # And it is byte-identical to the planning dir's RTSTRUCT (correct source copied).
    assert out.read_bytes() == planning_dcm.read_bytes()


def test_build_auto_rtstruct_fail_closed_when_no_planning_match(tmp_path, monkeypatch) -> None:
    """Several same-FoR phases, none references the planning series -> no RS_auto."""
    from rtpipeline import auto_rtstruct as ar

    course = tmp_path / "course"
    dirs = ar.build_course_dirs(course)
    shared_for = generate_uid()
    _write_ct_slice(dirs.dicom_ct, generate_uid(), shared_for)  # planning series unmatched
    seg_root = dirs.segmentation_totalseg
    _write_total_rtstruct(seg_root / "phase00", shared_for, ref_series_uid=generate_uid())
    _write_total_rtstruct(seg_root / "phase50", shared_for, ref_series_uid=generate_uid())

    monkeypatch.setattr(ar, "_load_ct_image", lambda d: _img())
    monkeypatch.setattr(ar, "sanitize_rtstruct", lambda p: None)
    monkeypatch.setattr(ar, "fix_rtstruct_rois", lambda ct, p: None)

    assert ar.build_auto_rtstruct(course) is None
