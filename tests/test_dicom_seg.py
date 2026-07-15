import sys
import importlib.util
from types import SimpleNamespace

import pydicom
import pytest
import SimpleITK as sitk
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from rtpipeline import dicom_seg


def _seg_dataset() -> Dataset:
    ds = Dataset()
    ds.SOPClassUID = pydicom.uid.SegmentationStorage
    ds.Modality = "SEG"
    segment = Dataset()
    segment.SegmentNumber = 1
    segment.SegmentLabel = "Tumor"
    ds.SegmentSequence = Sequence([segment])
    return ds


def test_optional_reader_imports_with_pydicom_3_compatibility():
    if importlib.util.find_spec("pydicom_seg") is None:
        pytest.skip("optional pydicom-seg dependency is not installed")
    module = dicom_seg.import_pydicom_seg()
    assert hasattr(module, "MultiClassReader")
    assert "pydicom._storage_sopclass_uids" in sys.modules


def test_load_dicom_seg_returns_image_not_reader_wrapper(monkeypatch):
    expected = sitk.Image(3, 4, 5, sitk.sitkUInt8)

    class Reader:
        def read(self, dataset):
            assert dataset.Modality == "SEG"
            return SimpleNamespace(image=expected)

    monkeypatch.setattr(
        dicom_seg,
        "import_pydicom_seg",
        lambda: SimpleNamespace(MultiClassReader=Reader),
    )
    image, labels = dicom_seg.load_dicom_seg_multiclass(_seg_dataset())
    assert image is expected
    assert labels == {1: "Tumor"}


def test_load_dicom_seg_rejects_non_seg_storage():
    ds = Dataset()
    ds.SOPClassUID = pydicom.uid.CTImageStorage
    ds.Modality = "CT"
    with pytest.raises(ValueError, match="not DICOM Segmentation Storage"):
        dicom_seg.load_dicom_seg_multiclass(ds)


def test_load_dicom_seg_propagates_overlap_failure(monkeypatch):
    class Reader:
        def read(self, dataset):
            raise ValueError("overlapping segments")

    monkeypatch.setattr(
        dicom_seg,
        "import_pydicom_seg",
        lambda: SimpleNamespace(MultiClassReader=Reader),
    )
    with pytest.raises(ValueError, match="overlapping"):
        dicom_seg.load_dicom_seg_multiclass(_seg_dataset())
