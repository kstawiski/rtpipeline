"""C2 (MR on-the-fly classify defense) + C3 (CBCT default-deny) — unit tests.

C2: _mr_series_is_anatomic() classifies an MR series from its DICOM headers at the
course-MR loop entry and skips positively-non-anatomic series (DWI/ADC/DCE/localizer)
before total_mr, while preserving back-compat for anatomic AND unrecognized MR.

C3: is_quantitative_image_class() default-denies CBCT from quantitative endpoints.
"""
from pathlib import Path

import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from rtpipeline.modality_classifier import (
    NON_QUANTITATIVE_IMAGE_CLASSES,
    is_quantitative_image_class,
)
from rtpipeline.segmentation import _imagetype_to_list, _mr_series_is_anatomic


def _write_min_dcm(path: Path, description: str, modality: str = "MR", image_type=None) -> None:
    ds = Dataset()
    ds.SeriesDescription = description
    ds.Modality = modality
    ds.SeriesInstanceUID = generate_uid()
    if image_type is not None:
        ds.ImageType = image_type
    fm = FileMetaDataset()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    fm.MediaStorageSOPClassUID = generate_uid()
    fm.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta = fm
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    pydicom.dcmwrite(str(path), ds, write_like_original=False)


def _series_dir(tmp_path: Path, name: str, description: str, n: int = 12, **kw) -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        _write_min_dcm(d / f"s{i:03d}.dcm", description, **kw)
    return d


# ---------------- C3: CBCT default-deny ----------------

def test_c3_cbct_is_non_quantitative():
    assert NON_QUANTITATIVE_IMAGE_CLASSES == frozenset({"cbct"})
    assert is_quantitative_image_class("cbct") is False
    assert is_quantitative_image_class("CBCT") is False  # case-insensitive
    assert is_quantitative_image_class(" cbct ") is False  # whitespace


def test_c3_calibrated_classes_are_quantitative():
    for cls in ("planning_ct", "diagnostic_ct", "petct_ct", "fourdct_ave", "mr_anatomic"):
        assert is_quantitative_image_class(cls) is True


# ---------------- C2: ImageType normalization (pydicom MultiValue) ----------------

def test_imagetype_multivalue_parsed_not_malformed():
    # Regression: pydicom stores multi-valued ImageType as MultiValue (a MutableSequence,
    # NOT list/tuple). A bare isinstance(x,(list,tuple)) check stringifies it to a single
    # malformed token like "['DERIVED', 'SECONDARY']". _imagetype_to_list must yield the
    # real per-value list.
    from pydicom.multival import MultiValue
    mv = MultiValue(str, ["DERIVED", "SECONDARY"])
    assert _imagetype_to_list(mv) == ["DERIVED", "SECONDARY"]
    assert _imagetype_to_list("ORIGINAL\\PRIMARY\\M\\NORM") == ["ORIGINAL", "PRIMARY", "M", "NORM"]
    assert _imagetype_to_list(["A", "B"]) == ["A", "B"]
    assert _imagetype_to_list(None) == []
    # round-trip through a real written/read DICOM (the production path)
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    import tempfile, os
    ds = Dataset()
    ds.ImageType = ["DERIVED", "SECONDARY"]
    ds.SeriesInstanceUID = generate_uid()
    fm = FileMetaDataset(); fm.TransferSyntaxUID = ExplicitVRLittleEndian
    fm.MediaStorageSOPClassUID = generate_uid(); fm.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta = fm; ds.is_little_endian = True; ds.is_implicit_VR = False
    p = tempfile.mktemp(suffix=".dcm")
    pydicom.dcmwrite(p, ds, write_like_original=False)
    try:
        rd = pydicom.dcmread(p, stop_before_pixels=True, force=True)
        assert _imagetype_to_list(getattr(rd, "ImageType", None)) == ["DERIVED", "SECONDARY"]
    finally:
        os.remove(p)


# ---------------- C2: MR header-classify guard ----------------

def test_c2_anatomic_t2_is_segmented(tmp_path):
    d = _series_dir(tmp_path, "anat", "T2 TSE ax pelvis")
    assert _mr_series_is_anatomic(d) is True


def test_c2_functional_dwi_is_skipped(tmp_path):
    d = _series_dir(tmp_path, "dwi", "ep2d diff DWI b1000")
    assert _mr_series_is_anatomic(d) is False


def test_c2_adc_map_is_skipped(tmp_path):
    d = _series_dir(tmp_path, "adc", "ADC map")
    assert _mr_series_is_anatomic(d) is False


def test_c2_localizer_is_skipped(tmp_path):
    d = _series_dir(tmp_path, "loc", "localizer")
    assert _mr_series_is_anatomic(d) is False


def test_c2_unrecognized_mr_is_segmented_backcompat(tmp_path):
    # No anatomic/functional/exclude token -> mr_unrecognized_default_deny -> segment (back-compat)
    d = _series_dir(tmp_path, "unk", "zzz seq 9000")
    assert _mr_series_is_anatomic(d) is True


def test_c2_empty_dir_is_segmented_backcompat(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    assert _mr_series_is_anatomic(d) is True


def test_c2_malformed_dcm_segments_backcompat(tmp_path):
    # Resilience + back-compat: a single malformed/non-DICOM .dcm must never raise and
    # must fail OPEN to segment. (force=True yields an empty Dataset; n_slices=1 -> the
    # adequacy exclusion 'sub_volumetric_lt10' -> segment, never a content skip.)
    d = tmp_path / "bad"
    d.mkdir()
    (d / "garbage.dcm").write_bytes(b"not a dicom file")
    assert _mr_series_is_anatomic(d) is True


def test_c2_thin_anatomic_mr_is_segmented_bc2(tmp_path):
    # BC-2 regression guard: a <10-slice ANATOMIC MR classifies 'sub_volumetric_lt10';
    # it was segmented before C2 and must STILL segment (do not silently drop thin MR).
    d = _series_dir(tmp_path, "thin", "T2 TSE ax pelvis", n=6)
    assert _mr_series_is_anatomic(d) is True


def test_c2_dce_perfusion_is_skipped(tmp_path):
    d = _series_dir(tmp_path, "dce", "DCE perfusion dyn twist")
    assert _mr_series_is_anatomic(d) is False


def test_c2_derived_secondary_is_skipped(tmp_path):
    # DERIVED/SECONDARY MR with no anatomic/functional token -> mr_derived_secondary (skip)
    d = _series_dir(tmp_path, "deriv", "", image_type=["DERIVED", "SECONDARY"])
    assert _mr_series_is_anatomic(d) is False


def test_c2_prefers_slice_with_description(tmp_path):
    # Hardening: a corrupted/empty first slice must not drive the decision. First slice
    # has no SeriesDescription; the rest are functional -> the guard classifies from a
    # described slice and skips (without the read-until-description loop it would wrongly
    # fall to mr_unrecognized -> segment).
    d = tmp_path / "mixed"
    d.mkdir()
    _write_min_dcm(d / "s000.dcm", "")  # empty description (corrupted/odd first frame)
    for i in range(1, 12):
        _write_min_dcm(d / f"s{i:03d}.dcm", "ep2d diff DWI b1000")
    assert _mr_series_is_anatomic(d) is False
