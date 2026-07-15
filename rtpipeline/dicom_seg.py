"""DICOM Segmentation Storage decoding helpers.

``pydicom-seg`` 0.4.1 still imports a private pydicom module removed in
pydicom 3.x.  RTpipeline supports pydicom 3.x, so importing the optional
decoder directly otherwise fails before a SEG can be read.  This module owns
that narrow compatibility boundary and normalizes the decoder's result to the
multilabel SimpleITK image expected by RTpipeline.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Dict

import pydicom
import SimpleITK as sitk
from pydicom.dataset import Dataset


SEGMENTATION_STORAGE_UID = str(pydicom.uid.SegmentationStorage)


def import_pydicom_seg():
    """Import the optional ``pydicom-seg`` package with pydicom 3 support.

    The compatibility module exposes only the public UID constant required by
    pydicom-seg.  It does not patch pydicom datasets or pixel decoding.
    """

    legacy_module = "pydicom._storage_sopclass_uids"
    if legacy_module not in sys.modules:
        try:
            __import__(legacy_module)
        except ModuleNotFoundError:
            shim = types.ModuleType(legacy_module)
            shim.SegmentationStorage = pydicom.uid.SegmentationStorage
            sys.modules[legacy_module] = shim

    try:
        import pydicom_seg
    except ImportError as exc:
        raise RuntimeError(
            "DICOM-SEG decoding requires the optional 'dcmseg' dependencies "
            "(install rtpipeline[dcmseg])."
        ) from exc
    return pydicom_seg


def load_dicom_seg_multiclass(
    source: str | Path | Dataset,
) -> tuple[sitk.Image, Dict[int, str]]:
    """Decode a non-overlapping binary DICOM-SEG as a multilabel image.

    ``pydicom_seg.MultiClassReader.read`` returns a result wrapper, not a
    SimpleITK image.  Returning ``result.image`` here prevents callers from
    accidentally passing that wrapper to SimpleITK resampling code.

    Overlapping or fractional segments cannot be represented faithfully in a
    single integer label map and therefore fail explicitly.
    """

    dataset = (
        source
        if isinstance(source, Dataset)
        else pydicom.dcmread(str(source), force=True)
    )
    if str(getattr(dataset, "SOPClassUID", "")) != SEGMENTATION_STORAGE_UID:
        raise ValueError(
            f"Dataset is not DICOM Segmentation Storage: "
            f"{getattr(dataset, 'SOPClassUID', None)!r}"
        )
    if str(getattr(dataset, "Modality", "")) != "SEG":
        raise ValueError(f"DICOM Segmentation Storage has invalid modality: {getattr(dataset, 'Modality', None)!r}")

    pydicom_seg = import_pydicom_seg()
    result = pydicom_seg.MultiClassReader().read(dataset)
    image = result.image
    if not isinstance(image, sitk.Image):
        raise TypeError(f"pydicom-seg returned {type(image)!r}, expected SimpleITK.Image")

    labels: Dict[int, str] = {}
    for segment in getattr(dataset, "SegmentSequence", []):
        number = int(getattr(segment, "SegmentNumber", 0) or 0)
        if number <= 0:
            continue
        label = str(getattr(segment, "SegmentLabel", "") or f"Segment_{number}")
        labels[number] = label
    if not labels:
        raise ValueError("DICOM-SEG contains no valid SegmentSequence entries")
    return image, labels
