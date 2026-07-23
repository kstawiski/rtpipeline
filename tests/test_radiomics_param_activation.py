import pytest

from rtpipeline.radiomics import _apply_params_to_extractor, _get_params_file


class _FakeExtractor:
    def __init__(self):
        self.settings = {}
        self.featureClassNames = ["shape", "firstorder", "glcm", "ngtdm", "shape2D"]
        self.enabledImagetypes = {}
        self.enabledFeatures = {}

    def _setTolerance(self):
        return None


def test_default_params_do_not_write_to_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    params_path = _get_params_file(None)

    assert params_path is not None
    assert params_path.is_file()
    assert params_path.name == "radiomics_params.yaml"
    assert not (tmp_path / "radiomics_params.yaml").exists()


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
