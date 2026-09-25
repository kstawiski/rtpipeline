"""dcm2niix output selection: the equalized volume wins for uneven slice spacing."""

from pathlib import Path

from rtpipeline.segmentation import _select_dcm2niix_volume


def _write(path: Path, size: int) -> None:
    path.write_bytes(b"\0" * size)


def test_equalized_volume_is_chosen_even_when_smaller(tmp_path):
    _write(tmp_path / "CT_1.nii.gz", 3287)
    _write(tmp_path / "CT_1_Eq_1.nii.gz", 3235)
    assert _select_dcm2niix_volume(tmp_path) == tmp_path / "CT_1_Eq_1.nii.gz"


def test_without_equalized_volume_the_largest_is_chosen(tmp_path):
    _write(tmp_path / "CT_1.nii.gz", 5000)
    _write(tmp_path / "CT_2.nii.gz", 4000)
    (tmp_path / "CT_1.json").write_text("{}")
    assert _select_dcm2niix_volume(tmp_path) == tmp_path / "CT_1.nii.gz"


def test_empty_output_gives_none(tmp_path):
    assert _select_dcm2niix_volume(tmp_path) is None
