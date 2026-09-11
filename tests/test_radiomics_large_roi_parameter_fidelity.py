"""Actual-helper method fidelity for a tiny generated ROI marked as large.

The compatibility flag must not prune configured image types/features or change
spacing. No clinical DICOM, segmentation, service, or pipeline launch is used.
Host pytest invokes the installed NumPy1/PyRadiomics conda helper directly.
"""
from __future__ import annotations

import json
import tempfile

import numpy as np
import pytest
import SimpleITK as sitk

from rtpipeline import radiomics_conda as rc

SPACING = (1.25, 1.5, 2.5)
PARAMETERS = """imageType:
  Original: {}
  Square: {}
featureClass:
  firstorder: [Mean, Minimum, Maximum, Range]
  shape: [VoxelVolume]
  glcm: [JointEntropy]
setting:
  label: 1
  binWidth: 25
  resampledPixelSpacing: [1.25, 1.5, 2.5]
  interpolator: sitkBSpline
"""


@pytest.fixture
def generated_case(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    for key in rc._THREAD_ENV_VARS:
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("RTPIPELINE_RADIOMICS_THREAD_LIMIT", "1")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("CONDA_NO_PLUGINS", "1")
    z, y, x = np.indices((10, 12, 12))
    array = (100 + 7*z + 3*y + 11*x).astype(np.float32)
    mask = np.zeros(array.shape, dtype=np.uint8)
    mask[2:6, 3:8, 4:9] = 1
    paths = []
    for name, data in (("image", array), ("mask", mask)):
        image = sitk.GetImageFromArray(data)
        image.SetSpacing(SPACING)
        path = tmp_path / f"{name}.nrrd"
        sitk.WriteImage(image, str(path))
        paths.append(str(path))
    params = tmp_path / "configured.yaml"
    params.write_text(PARAMETERS, encoding="utf-8")
    return paths[0], paths[1], str(params), array[mask == 1]


def _features(result):
    return {k: float(v) for k, v in result.items() if k.startswith(("original_", "square_"))}


def _assert_configured_measurements(result, voxels):
    features = _features(result)
    expected = {f"{image}_{family}_{feature}" for image in ("original", "square")
                for family, names in (("firstorder", ("Mean", "Minimum", "Maximum", "Range")),
                                      ("glcm", ("JointEntropy",))) for feature in names}
    expected.add("original_shape_VoxelVolume")
    assert set(features) == expected
    assert all(np.isfinite(v) for v in features.values())
    assert features["original_firstorder_Mean"] == pytest.approx(float(voxels.mean()))
    assert features["original_firstorder_Minimum"] == pytest.approx(float(voxels.min()))
    assert features["original_firstorder_Maximum"] == pytest.approx(float(voxels.max()))
    assert features["original_firstorder_Range"] == pytest.approx(float(np.ptp(voxels)))
    assert features["original_shape_VoxelVolume"] == pytest.approx(len(voxels) * np.prod(SPACING))
    assert features["original_glcm_JointEntropy"] > 0
    assert result["diagnostics_Versions_Numpy"].startswith("1.")
    assert result["diagnostics_Versions_PyRadiomics"].lstrip("v") == "3.0.1"
    # The shared schema serializes non-scalar diagnostic values as JSON strings.
    assert json.loads(result["diagnostics_Image-interpolated_Spacing"]) == list(SPACING)
    print("helper diagnostics", {k: result[k] for k in result if k.startswith("diagnostics_Versions_")})
    print("numerical reconciliation", features)


def test_single_large_flag_preserves_configured_method(generated_case):
    image, mask, params, voxels = generated_case
    ordinary = rc.extract_radiomics_with_conda(image, mask, params, label=1, large_roi=False)
    marked = rc.extract_radiomics_with_conda(image, mask, params, label=1, large_roi=True)
    _assert_configured_measurements(ordinary, voxels)
    _assert_configured_measurements(marked, voxels)
    assert _features(marked) == pytest.approx(_features(ordinary))
    state = marked["__effective_extractor_state__"]
    assert state == ordinary["__effective_extractor_state__"]
    assert state["settings"]["resampledPixelSpacing"] == list(SPACING)
    assert set(state["image_types"]) == {"Original", "Square"}
    assert state["features"]["glcm"] == ["JointEntropy"]


def test_batch_large_flag_preserves_features_and_effective_hash(generated_case):
    image, mask, params, voxels = generated_case
    tasks = [dict(image_path=image, mask_path=mask, params_file=params, label=1,
                  roi_name="body" if flag else "small", large_roi=flag,
                  parameter_provenance_arm="MR") for flag in (False, True)]
    results = rc.extract_radiomics_batch_with_conda(tasks, params_file=params, timeout_per_roi=60)
    assert len(results) == 2
    ordinary, marked = results
    assert [r["__status__"] for r in results] == ["success", "success"]
    for result in results:
        _assert_configured_measurements(result, voxels)
    assert _features(marked) == pytest.approx(_features(ordinary))
    assert marked["__effective_parameter_hash__"] == ordinary["__effective_parameter_hash__"]


def test_materialized_ct_arm_hashes_do_not_reduce_large_rois(generated_case):
    _, _, params, _ = generated_case
    decision = json.dumps(dict(roi_class="synthetic", map_version="fixture", map_hash="0"*64,
        map_entry_source="fixture", adjudication_status="fixture",
        primary_resegment_range_hu=[-200, 300], primary_intensity_texture_disposition="extract",
        feature_publication_policy="extract"), sort_keys=True)
    rc._materialized_ct_arm_hashes.cache_clear()
    try:
        ordinary = rc._materialized_ct_arm_hashes(params, False, decision)
        marked = rc._materialized_ct_arm_hashes(params, True, decision)
        assert set(ordinary) == {"primary_resegmented", "sensitivity_raw"}
        assert all(len(v) == 64 for v in ordinary.values())
        assert marked == ordinary
        print("effective CT arm hashes", marked)
    finally:
        rc._materialized_ct_arm_hashes.cache_clear()
