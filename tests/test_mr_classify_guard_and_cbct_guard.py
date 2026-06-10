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
from rtpipeline.segmentation import _mr_series_is_anatomic


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


def test_c2_malformed_dcm_does_not_crash(tmp_path):
    # Resilience contract: a malformed/non-DICOM .dcm must never raise out of the guard
    # (it must never break a previously-working course); it returns a plain bool.
    d = tmp_path / "bad"
    d.mkdir()
    (d / "garbage.dcm").write_bytes(b"not a dicom file")
    assert isinstance(_mr_series_is_anatomic(d), bool)
