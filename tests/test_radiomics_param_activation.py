import pytest

from rtpipeline.radiomics import _apply_params_to_extractor


class _FakeExtractor:
    def __init__(self):
        self.settings = {}
        self.featureClassNames = ["shape", "firstorder", "glcm", "ngtdm", "shape2D"]
        self.enabledImagetypes = {}
        self.enabledFeatures = {}

    def _setTolerance(self):
        return None


def test_parameter_loader_does_not_enable_omitted_feature_classes():
    extractor = _FakeExtractor()
    _apply_params_to_extractor(
        extractor,
        {
            "imageType": {"Original": {}},
            "featureClass": {
                "shape": ["MeshVolume"],
                "firstorder": ["Mean"],
            },
        },
    )
    assert extractor.enabledFeatures == {
        "shape": ["MeshVolume"],
        "firstorder": ["Mean"],
    }


def test_parameter_loader_rejects_unknown_feature_class():
    extractor = _FakeExtractor()
    with pytest.raises(ValueError, match="Unknown PyRadiomics feature classes"):
        _apply_params_to_extractor(
            extractor,
            {"featureClass": {"not_a_feature_class": ["anything"]}},
        )
