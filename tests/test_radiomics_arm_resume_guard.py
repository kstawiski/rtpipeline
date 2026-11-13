"""A resumed extraction arm must publish the same resource-guard metadata as a fresh one.

The texture budget guard records the predicted peak allocation while the
extractor runs. An arm reused from the durable arm cache does not run the
extractor, so the cached payload has to carry that arm's contribution.
"""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

sitk = pytest.importorskip("SimpleITK")

from rtpipeline import radiomics_ct_contract as contract
from test_radiomics_ct_dual_arm import _FakeExtractor, _common_identity, _qc, _versions

PEAK = "resource_guard_predicted_peak_bytes"
CALLS: list[str] = []


class _GuardedExtractor(_FakeExtractor):
    """Routes texture extraction through computeFeatures so the real guard runs."""

    def computeFeatures(self, image, mask, imageTypeName, **kwargs):
        CALLS.append(imageTypeName)
        values = sitk.GetArrayViewFromImage(image)[sitk.GetArrayViewFromImage(mask) == 1]
        return {"original_firstorder_Mean": float(values.mean()),
                "original_glcm_Contrast": float(values.std())}

    def execute(self, image, mask):
        if set(self.enabledFeatures) == {"shape"}:
            return super().execute(image, mask)
        array = sitk.GetArrayFromImage(image)
        # The resegmented arm sees a wider gray range, so the two texture arms
        # predict different peaks and a partial resume must restore the larger.
        if self.settings.get("resegmentRange") is not None:
            array = array * 3.0
        scaled = sitk.GetImageFromArray(array)
        scaled.CopyInformation(image)
        return self.computeFeatures(scaled, mask, "Original", binWidth=25, label=1)


def _images():
    rng = np.random.default_rng(20261113)
    image = sitk.GetImageFromArray(rng.integers(0, 400, size=(12, 12, 12)).astype(np.float64))
    mask_array = np.zeros((12, 12, 12), np.uint8)
    mask_array[3:9, 3:9, 3:9] = 1
    mask = sitk.GetImageFromArray(mask_array)
    return image, mask


def _extract(monkeypatch, progress_root):
    monkeypatch.setattr(contract, "resampled_mask_qc", _qc)
    monkeypatch.setattr(contract, "_runtime_versions", _versions)
    image, mask = _images()
    CALLS.clear()
    rows = contract.extract_ct_roi_arms(
        image, mask, factory=_GuardedExtractor,
        decision=contract.classify_ct_roi("Manual", "PTV"),
        common_metadata=_common_identity(), run_identifier="run-1",
        code_revision="revision-1", native_voxel_count=216, required=True,
        configured_parameter_hashes={contract.PRIMARY_ARM: "configured-primary",
                                     contract.SENSITIVITY_ARM: "configured-sensitivity"},
        progress_root=progress_root, progress_context={"fixture": "arm-resume-guard"},
    )
    return pd.DataFrame(rows), len(CALLS)


def test_resumed_arms_publish_the_fresh_resource_guard_peak(tmp_path, monkeypatch):
    root = tmp_path / "progress"
    fresh, fresh_calls = _extract(monkeypatch, root)
    # Two texture arms; the guard schedules each feature family separately.
    assert fresh_calls == 4
    assert (fresh[PEAK] > 0).all()
    reference, _ = _extract(monkeypatch, None)
    pd.testing.assert_frame_equal(reference, fresh, check_exact=True)

    # Both texture arms predict different peaks, so the max is discriminating.
    arms = sorted(path for path in (root / "arms").iterdir() if path.is_dir())
    assert len(arms) == 3
    pristine = tmp_path / "pristine"
    shutil.copytree(root, pristine)

    resumed, resumed_calls = _extract(monkeypatch, root)
    assert resumed_calls == 0
    pd.testing.assert_frame_equal(fresh, resumed, check_exact=True)

    # Any single missing arm reruns that arm alone and still publishes the
    # same row, including the peak contributed by the cached arms.
    for index, arm in enumerate(arms):
        work = tmp_path / f"partial-{index}"
        shutil.copytree(pristine, work)
        shutil.rmtree(work / "arms" / arm.name)
        partial, partial_calls = _extract(monkeypatch, work)
        assert partial_calls in {0, 2}
        pd.testing.assert_frame_equal(fresh, partial, check_exact=True)


def test_texture_arm_peaks_differ_so_partial_resume_discriminates():
    # If both texture arms predicted the same peak, a cache that dropped the
    # peak would still publish the right maximum after any partial resume.
    from rtpipeline.radiomics_memory import install_texture_budget
    image, mask = _images()
    peaks = []
    for resegment in (None, [-1000, 3000]):
        extractor = _GuardedExtractor()
        extractor.enabledFeatures = {"firstorder": [], "glcm": []}
        if resegment is not None:
            extractor.settings["resegmentRange"] = resegment
        metadata = {"resource_guard_native_image_voxels": 1728,
                    "resource_guard_crop_voxels_bound": 1728, PEAK: 0}
        install_texture_budget(extractor, metadata).execute(image, mask)
        peaks.append(metadata[PEAK])
    assert 0 < peaks[0] < peaks[1]


def test_arm_cache_rejects_a_payload_without_the_guard_peak(tmp_path, monkeypatch):
    from rtpipeline.durable_progress import DurableProgressError
    from rtpipeline.extraction_progress import BoundItem
    root = tmp_path / "progress"
    _extract(monkeypatch, root)
    original_load = BoundItem.load
    def legacy_shape(self):
        value = original_load(self)
        return value["features"] if isinstance(value, dict) and "features" in value else value
    monkeypatch.setattr(BoundItem, "load", legacy_shape)
    with pytest.raises(DurableProgressError, match="EXTRACTION_ARM_PAYLOAD_INVALID"):
        _extract(monkeypatch, root)
