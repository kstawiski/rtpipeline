"""Synthetic preprocessing checks and the optional, public IBSI CT benchmark."""
import json
import os
import sys
import shutil
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pytest
import SimpleITK as sitk

from rtpipeline.ibsi_preprocessing import (
    SIGMA_KEY, ibsi_preprocess, preprocessing_options, resegmentation_bounds,
    spatial_preprocess,
)


def test_centered_grid_ceil_size_rotated_direction_and_input_preservation():
    a = np.arange(5*6*7, dtype=np.float32).reshape(5, 6, 7)
    image, mask = sitk.GetImageFromArray(a), sitk.GetImageFromArray(np.ones_like(a, np.uint8))
    for obj in (image, mask):
        obj.SetSpacing((1.3, 1.4, 2.2))
        obj.SetOrigin((20., -10., 35.))
        obj.SetDirection((0., -1., 0., 1., 0., 0., 0., 0., 1.))
    result, morph = spatial_preprocess(image, mask, {"resampledPixelSpacing": [2., 2., 0.], "interpolator": "sitkLinear"})
    expected_size = np.ceil(np.array(image.GetSize()) * image.GetSpacing() / (2., 2., 2.2)).astype(int)
    assert result.GetSize() == tuple(expected_size)
    old_centre = image.TransformContinuousIndexToPhysicalPoint(tuple((np.array(image.GetSize()) - 1) / 2))
    new_centre = result.TransformContinuousIndexToPhysicalPoint(tuple((np.array(result.GetSize()) - 1) / 2))
    assert new_centre == pytest.approx(old_centre)
    assert morph.GetOrigin() == result.GetOrigin()
    assert result.GetDirection() == image.GetDirection()
    np.testing.assert_array_equal(sitk.GetArrayFromImage(image), a)


def test_linear_mask_includes_half_voxel_and_hu_rounds_to_nearest_even():
    a = np.broadcast_to(np.arange(4, dtype=np.float32), (4, 4, 4)).copy()
    m = np.zeros(a.shape, np.uint8); m[:, :, 0] = 7
    image, mask = sitk.GetImageFromArray(a), sitk.GetImageFromArray(m)
    processed, morph = spatial_preprocess(image, mask, {
        "label": 7, "resampledPixelSpacing": [2., 2., 2.], "interpolator": "sitkLinear"})
    np.testing.assert_array_equal(sitk.GetArrayFromImage(processed)[0, 0], [0., 2.])
    np.testing.assert_array_equal(sitk.GetArrayFromImage(morph)[0, 0], [1, 0])


def test_sigma_uses_unresegmented_population_and_intersects_absolute_range():
    values = np.array([-900.] + [0.] * 100 + [1000.])
    lower, upper = resegmentation_bounds(values, {"resegmentRange": [-1000., 400.], SIGMA_KEY: 3.})
    assert lower == pytest.approx(max(-1000., values.mean() - 3*values.std(ddof=0)))
    assert upper == pytest.approx(min(400., values.mean() + 3*values.std(ddof=0)))
    image = sitk.GetImageFromArray(values.astype(np.float32).reshape(1, 1, -1))
    mask = sitk.GetImageFromArray(np.ones((1, 1, len(values)), np.uint8))
    _, morph, intensity = ibsi_preprocess(image, mask, {"resegmentRange": [-1000., 400.], SIGMA_KEY: 3.})
    assert sitk.GetArrayFromImage(morph).sum() == 102
    assert sitk.GetArrayFromImage(intensity).sum() == 100


@pytest.mark.parametrize("options", [
    {"preprocessing": "typo"}, {"preprocessing": "ibsi", "resegment_sigma": -1},
    {"preprocessing": "ibsi", "resegment_sigma": float("nan")},
    {"preprocessing": "native", "resegment_sigma": 3},
    {"preprocessing": "ibsi", "resegmentSigma": 3},
])
def test_invalid_preprocessing_options_fail_explicitly(options):
    with pytest.raises(ValueError):
        preprocessing_options({"radiomics": options})


@pytest.mark.parametrize("settings", [{"normalize": True}, {"resegmentShape": True},
                                           {"resampledPixelSpacing": [2., -1., 2.]}])
def test_conflicting_ct_settings_are_rejected(settings):
    image = sitk.GetImageFromArray(np.zeros((4, 4, 4), np.float32))
    mask = sitk.GetImageFromArray(np.ones((4, 4, 4), np.uint8))
    with pytest.raises(ValueError):
        spatial_preprocess(image, mask, settings)


def test_numpy2_helper_route_rejects_ibsi_instead_of_delegating(tmp_path, monkeypatch):
    import rtpipeline.radiomics as radiomics
    params = tmp_path / "ibsi.yaml"
    params.write_text("radiomics:\n  preprocessing: ibsi\n")
    from types import SimpleNamespace
    monkeypatch.setattr(np, "__version__", "2.0.0")
    with pytest.raises(ValueError, match="native CT extractor"):
        radiomics._extractor(SimpleNamespace(radiomics_params_file=params))


@pytest.fixture(scope="module")
def numerical_receipt():
    executable = shutil.which(os.environ.get("RTPIPELINE_RADIOMICS_PYTHON", sys.executable))
    reason = "set RTPIPELINE_RADIOMICS_PYTHON to a NumPy 1.x/PyRadiomics environment"
    if executable is None:
        pytest.skip(reason)
    probe = subprocess.run(
        [executable, "-I", "-B", "-c",
         "import numpy, radiomics; assert int(numpy.__version__.split('.')[0]) < 2"],
        capture_output=True, text=True, timeout=30,
    )
    if probe.returncode:
        pytest.skip(reason)
    repo = Path(__file__).resolve().parents[1]
    public = repo / "validation/ibsi/data/public_sources_20261027"
    with tempfile.TemporaryDirectory(prefix="f6-ibsi-", dir=os.environ.get("TMPDIR")) as directory:
        command = [str(executable), "-I", "-B", str(Path(__file__).with_name("validate_f6_ibsi.py")),
                   "--output", directory]
        if not public.is_dir():
            command.append("--synthetic-only")
        result = subprocess.run(command, capture_output=True, text=True, timeout=240,
                                env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
        assert result.returncode == 0, result.stdout + result.stderr
        yield json.loads((Path(directory) / "receipt.json").read_text())


def test_default_extraction_and_packaged_option_on_synthetic_ct(numerical_receipt):
    assert numerical_receipt["default"]["features"] == 1223
    assert numerical_receipt["synthetic"]["synthetic_option_and_qc"] == "passed"


def test_public_benchmark_a_b_d_e_and_diagnostics(numerical_receipt):
    if "ibsi" not in numerical_receipt:
        pytest.skip("public IBSI phantom is not available")
    ibsi = numerical_receipt["ibsi"]
    assert ibsi["preprocessing_within_tolerance"] == ibsi["preprocessing_diagnostics"] == 300
    assert ibsi["feature_outputs"] == 735
    for c in "ABDE":
        assert ibsi["configurations"][c]["bit_identical_to_benchmark"]
        assert ibsi["configurations"][c]["within_tolerance"] == 140
        assert ibsi["configurations"][c]["computed_comparison_pairs"] == 141
