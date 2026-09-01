import json
from types import SimpleNamespace

import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from rtpipeline.custom_structures_rtstruct import (
    _add_roi_with_unique_number,
    _assert_unique_roi_numbers,
    load_rs_custom_outcomes,
)


def _roi(number: int, name: str) -> Dataset:
    ds = Dataset()
    ds.ROINumber = number
    ds.ROIName = name
    return ds


def _roi_contour(number: int) -> Dataset:
    ds = Dataset()
    ds.ReferencedROINumber = number
    return ds


def _roi_observation(number: int) -> Dataset:
    ds = Dataset()
    ds.ObservationNumber = number
    ds.ReferencedROINumber = number
    return ds


class FakeRTStruct:
    def __init__(self) -> None:
        self.ds = Dataset()
        self.ds.StructureSetROISequence = Sequence([_roi(1, "Native1"), _roi(52, "Native52")])
        self.ds.ROIContourSequence = Sequence([_roi_contour(1), _roi_contour(52)])
        self.ds.RTROIObservationsSequence = Sequence([_roi_observation(1), _roi_observation(52)])

    def add_roi(self, *, mask, name, **kwargs) -> None:
        roi_number = len(self.ds.StructureSetROISequence) + 1
        self.ds.StructureSetROISequence.append(_roi(roi_number, name))
        self.ds.ROIContourSequence.append(_roi_contour(roi_number))
        self.ds.RTROIObservationsSequence.append(_roi_observation(roi_number))


def test_add_roi_uses_max_existing_roi_number_not_sequence_length():
    rtstruct = FakeRTStruct()

    _add_roi_with_unique_number(rtstruct, mask=None, name="custom_a")
    _add_roi_with_unique_number(rtstruct, mask=None, name="custom_b")

    roi_numbers = [int(roi.ROINumber) for roi in rtstruct.ds.StructureSetROISequence]
    contour_numbers = [int(roi.ReferencedROINumber) for roi in rtstruct.ds.ROIContourSequence]
    observation_numbers = [int(roi.ReferencedROINumber) for roi in rtstruct.ds.RTROIObservationsSequence]

    assert roi_numbers == [1, 52, 53, 54]
    assert contour_numbers == roi_numbers
    assert observation_numbers == roi_numbers
    _assert_unique_roi_numbers(rtstruct.ds, "test")


def test_rs_custom_outcomes_are_bound_to_published_sop_instance(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "RS_custom.dcm").write_bytes(b"synthetic")
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "rs_custom_meta.json").write_text(
        json.dumps(
            {
                "rs_custom_sop_instance_uid": "1.2.3",
                "custom_structure_outcomes": {
                    "iliac_area": {"status": "source_unavailable"}
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "rtpipeline.custom_structures_rtstruct.pydicom.dcmread",
        lambda *_args, **_kwargs: SimpleNamespace(SOPInstanceUID="1.2.3"),
    )

    assert load_rs_custom_outcomes(tmp_path)["iliac_area"]["status"] == "source_unavailable"

    monkeypatch.setattr(
        "rtpipeline.custom_structures_rtstruct.pydicom.dcmread",
        lambda *_args, **_kwargs: SimpleNamespace(SOPInstanceUID="9.9.9"),
    )
    assert load_rs_custom_outcomes(tmp_path) == {}
