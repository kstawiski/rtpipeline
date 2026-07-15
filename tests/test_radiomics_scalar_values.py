import numpy as np

from rtpipeline.radiomics_robustness import _coerce_scalar_feature_value


def test_accepts_pyradiomics_zero_dimensional_numeric_array():
    value = np.asarray(3.25)
    assert _coerce_scalar_feature_value("original_firstorder_Mean", value) == 3.25


def test_accepts_single_element_numeric_array():
    value = np.asarray([7.5])
    assert _coerce_scalar_feature_value("original_shape_MeshVolume", value) == 7.5


def test_rejects_diagnostics_and_non_scalar_arrays():
    assert _coerce_scalar_feature_value("diagnostics_Mask-original_Size", np.asarray(2)) is None
    assert _coerce_scalar_feature_value("original_firstorder_Mean", np.asarray([1, 2])) is None
    assert _coerce_scalar_feature_value("original_firstorder_Mean", np.asarray("text")) is None
