"""DN02: a scoped, validated raise of the radiomics allocation envelope."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rtpipeline.radiomics_memory import (
    MEMORY_BUDGET_BYTES, MEMORY_BUDGET_CEILING_BYTES, MEMORY_BUDGET_ENV,
    STAGE2_CODE, memory_budget_bytes, predicted_peak_bytes, preflight_metadata,
)

GiB = 1024**3


def test_default_budget_is_unchanged_without_override(monkeypatch):
    monkeypatch.delenv(MEMORY_BUDGET_ENV, raising=False)
    assert memory_budget_bytes() == MEMORY_BUDGET_BYTES == 8 * GiB


def test_validated_raise_is_returned(monkeypatch):
    monkeypatch.setenv(MEMORY_BUDGET_ENV, str(12 * GiB))
    assert memory_budget_bytes() == 12 * GiB


@pytest.mark.parametrize("bad", ["abc", "1.5e10", str(4 * GiB), str(MEMORY_BUDGET_CEILING_BYTES + 1), "-1"])
def test_lower_malformed_or_excessive_override_is_refused(monkeypatch, bad):
    monkeypatch.setenv(MEMORY_BUDGET_ENV, bad)
    with pytest.raises(ValueError):
        memory_budget_bytes()


def _large_ptv_case():
    """The recorded DN02 bbox class: 512x512x263 CT at 3 mm, padded 1 mm bbox ~37 M voxels.

    Returned as (estimate, native image voxels) without allocating the CT.
    """
    estimate = SimpleNamespace(
        estimated_resampled_bbox_voxels=36_406_000,
        estimated_resampled_bbox_shape=(330, 330, 333),
        native_foreground_voxels=1_764_000,
    )
    return estimate, 512 * 512 * 263


def _preflight(estimate, native):
    # preflight_metadata only needs mask.size; a zero-stride view avoids allocation.
    mask = np.broadcast_to(np.zeros(1, np.uint8), (native,))
    return preflight_metadata(
        estimate, mask, native_spacing_xyz=(0.977, 0.977, 3.0), array_axis_to_xyz=(2, 1, 0),
        settings={"resampledPixelSpacing": [1.0, 1.0, 1.0], "distances": [1]},
        image_types={"Original": {}, "LoG": {"sigma": [1.0]}, "Wavelet": {}},
    )


def test_dn02_large_ptv_refused_at_default_and_admitted_at_12_gib(monkeypatch):
    estimate, native = _large_ptv_case()
    monkeypatch.delenv(MEMORY_BUDGET_ENV, raising=False)
    assert _preflight(estimate, native) is None
    monkeypatch.setenv(MEMORY_BUDGET_ENV, str(12 * GiB))
    meta = _preflight(estimate, native)
    assert meta is not None and meta["resource_guard_reason_code"] == STAGE2_CODE
    assert meta["resource_guard_memory_budget_bytes"] == 12 * GiB
    assert 8 * GiB < meta["resource_guard_predicted_peak_bytes"] <= 12 * GiB


def test_texture_guard_uses_the_raised_budget(monkeypatch):
    # A gray-level range rejected at 8 GiB is admitted at 12 GiB by the same arithmetic.
    native, crop, mask = 512 * 512 * 263, 31_534_000, 2_000_000
    g = 1_200
    peak = predicted_peak_bytes(crop_voxels=crop, native_image_voxels=native, mask_voxels=mask,
                                gray_levels=g, max_gray_frequency=1, max_extent=330)
    assert 8 * GiB < peak <= 12 * GiB
    monkeypatch.delenv(MEMORY_BUDGET_ENV, raising=False)
    assert peak > memory_budget_bytes()
    monkeypatch.setenv(MEMORY_BUDGET_ENV, str(12 * GiB))
    assert peak <= memory_budget_bytes()
