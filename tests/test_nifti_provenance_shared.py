"""Every NIfTI sidecar writer must emit the provenance the contract requires.

organize's related-series conversion and segmentation's `_ensure_ct_nifti` used
to build the sidecar independently. Only the latter recorded `nifti_sha256` and
`nifti_geometry`, so a planning CT converted by the former failed contract
validation with "planning CT NIfTI provenance sidecar is incomplete".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rtpipeline import nifti_provenance


def _write_nifti(path: Path) -> Path:
    sitk = pytest.importorskip("SimpleITK")
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(np.zeros((4, 4, 4), dtype=np.int16))
    image.SetSpacing((1.0, 1.0, 3.0))
    sitk.WriteImage(image, str(path))
    return path


def test_annotate_supplies_every_contract_required_key(tmp_path: Path) -> None:
    nifti = _write_nifti(tmp_path / "ct.nii.gz")
    metadata = {"series_instance_uid": "1.2.3", "sop_hash": "abc", "geometry": {"rows": 4}}
    nifti_provenance.annotate(metadata, nifti, tmp_path / "DICOM", regenerated=True)
    for key in nifti_provenance.REQUIRED_SIDECAR_KEYS:
        assert key in metadata, f"shared provenance omitted {key}"
    assert nifti_provenance.sidecar_is_complete(metadata)
    assert metadata["nifti_geometry"]["spacing"][2] == pytest.approx(3.0)


def test_unregenerated_sidecar_preserves_content_timestamps(tmp_path: Path) -> None:
    """Mask reuse keys off these timestamps, so a metadata refresh must not move them."""
    nifti = _write_nifti(tmp_path / "ct.nii.gz")
    existing = {"generated_at": "2020-01-01T00:00:00+00:00",
                "nifti_generated_at": "2020-01-01T00:00:00+00:00"}
    metadata: dict = {}
    nifti_provenance.annotate(
        metadata, nifti, tmp_path / "DICOM", regenerated=False, existing_sidecar=existing
    )
    assert metadata["nifti_generated_at"] == existing["nifti_generated_at"]
    assert metadata["generated_at"] == existing["generated_at"]


def test_contract_required_keys_match_the_validator() -> None:
    """The helper's key list must track course_contract's actual requirement."""
    source = Path("rtpipeline/organize.py").read_text()
    start = source.index("required_nifti_keys = (")
    block = source[start:source.index(")", start)]
    declared = {line.strip().strip('",') for line in block.splitlines()[1:] if line.strip()}
    assert declared == set(nifti_provenance.REQUIRED_SIDECAR_KEYS), (
        "organize's required_nifti_keys drifted from nifti_provenance.REQUIRED_SIDECAR_KEYS"
    )
