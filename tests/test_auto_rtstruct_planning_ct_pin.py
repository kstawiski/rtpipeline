"""Regression tests for C1: auto-RTSTRUCT must bind masks to the planning CT.

``build_auto_rtstruct`` builds ``RS_auto.dcm`` against the course planning CT
(``course_dirs.dicom_ct``) but historically chose the mask source as the first
*alphabetically sorted* ``Segmentation_TotalSegmentator`` subdir that had a
``--total.dcm`` (and fell back to ``candidate_dirs[0]``). In all-series mode the
course can have several segmented series (planning_ct, CBCT, 4DCT phases,
diagnostic CT), so the blind first-dir pick can bind masks from the WRONG series /
FrameOfReference onto the planning CT, producing anatomically wrong structures.
``_resample_to_reference`` only corrects grid/spacing *within one physical space*,
so a cross-FoR pick yields garbage rather than an error.

C1 fixes this with two pure helpers:
  * ``_select_seg_dir_for_ct`` picks the seg dir whose ``FrameOfReferenceUID``
    matches the planning CT, and fail-closes (returns ``None``) when several series
    are present and none matches — it must NEVER fall back to ``candidate_dirs[0]``.
  * ``_geometry_compatible`` is the universal safety net applied before resampling:
    if the selected segmentation does not share the CT's physical space (origin /
    spacing / direction within tolerance), the build aborts instead of emitting a
    wrong RTSTRUCT.

The single-series case (today's common path) must keep working unchanged.
"""
from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from rtpipeline.auto_rtstruct import _geometry_compatible, _select_seg_dir_for_ct

SEG_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.66.4"  # DICOM Segmentation Storage


def _write_total_seg(seg_dir: Path, for_uid: str) -> Path:
    """Write a minimal, readable ``<name>--total.dcm`` carrying a FrameOfReferenceUID."""
    seg_dir.mkdir(parents=True, exist_ok=True)
    out = seg_dir / f"{seg_dir.name}--total.dcm"
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = SEG_SOP_CLASS
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = SEG_SOP_CLASS
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "SEG"
    ds.FrameOfReferenceUID = for_uid
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
# _select_seg_dir_for_ct — FrameOfReferenceUID-based selection                 #
# --------------------------------------------------------------------------- #

def test_picks_for_matched_dir_not_alpha_first(tmp_path: Path) -> None:
    """The bug: alpha-first dir has a different FoR; the planning-CT match is second."""
    ct_for = generate_uid()
    # "aaa_cbct" sorts before "zzz_planning" so candidate_dirs[0] would be the CBCT.
    _write_total_seg(tmp_path / "aaa_cbct", for_uid=generate_uid())
    planning = tmp_path / "zzz_planning"
    planning_dcm = _write_total_seg(planning, for_uid=ct_for)
    candidates = sorted(p for p in tmp_path.iterdir() if p.is_dir())

    selected_dir, seg_path, base_name = _select_seg_dir_for_ct(candidates, ct_for)

    assert selected_dir == planning
    assert seg_path == planning_dcm
    assert base_name == "zzz_planning"


def test_single_matching_series(tmp_path: Path) -> None:
    ct_for = generate_uid()
    only = tmp_path / "planning"
    only_dcm = _write_total_seg(only, for_uid=ct_for)
    selected_dir, seg_path, base_name = _select_seg_dir_for_ct([only], ct_for)
    assert (selected_dir, seg_path, base_name) == (only, only_dcm, "planning")


def test_single_series_backcompat_when_ct_for_unreadable(tmp_path: Path) -> None:
    """If the CT FoR can't be read, a lone candidate is still usable (legacy behavior)."""
    only = tmp_path / "planning"
    only_dcm = _write_total_seg(only, for_uid=generate_uid())
    selected_dir, seg_path, base_name = _select_seg_dir_for_ct([only], "")
    assert (selected_dir, seg_path, base_name) == (only, only_dcm, "planning")


def test_fail_closed_when_multiple_and_none_match(tmp_path: Path) -> None:
    """Several series, none shares the CT FoR -> never guess candidate_dirs[0]."""
    _write_total_seg(tmp_path / "aaa_cbct", for_uid=generate_uid())
    _write_total_seg(tmp_path / "bbb_diag", for_uid=generate_uid())
    candidates = sorted(p for p in tmp_path.iterdir() if p.is_dir())
    selected_dir, seg_path, base_name = _select_seg_dir_for_ct(candidates, generate_uid())
    assert (selected_dir, seg_path, base_name) == (None, None, None)


def test_fail_closed_when_multiple_and_ct_for_unreadable(tmp_path: Path) -> None:
    """Ambiguous (multiple candidates) + no CT FoR to disambiguate -> fail closed."""
    _write_total_seg(tmp_path / "aaa_cbct", for_uid=generate_uid())
    _write_total_seg(tmp_path / "zzz_planning", for_uid=generate_uid())
    candidates = sorted(p for p in tmp_path.iterdir() if p.is_dir())
    selected_dir, seg_path, base_name = _select_seg_dir_for_ct(candidates, "")
    assert selected_dir is None


# --------------------------------------------------------------------------- #
# _geometry_compatible — the pre-resample safety net                           #
# --------------------------------------------------------------------------- #

def test_geometry_identical_is_compatible() -> None:
    assert _geometry_compatible(_img(), _img()) is True


def test_geometry_small_origin_shift_within_tolerance() -> None:
    # < 2 mm shift on a shared grid: still the same physical space.
    assert _geometry_compatible(_img(origin=(0.5, 0.0, 0.0)), _img()) is True


def test_geometry_large_origin_shift_incompatible() -> None:
    # Different series occupying a different region of space (e.g. CBCT vs planning CT).
    assert _geometry_compatible(_img(origin=(250.0, 120.0, -400.0)), _img()) is False


def test_geometry_different_direction_incompatible() -> None:
    rotated = _img(direction=(0, 1, 0, 1, 0, 0, 0, 0, 1))
    assert _geometry_compatible(rotated, _img()) is False
