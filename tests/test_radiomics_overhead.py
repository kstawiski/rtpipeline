"""Shared-load admission, fallback, and bounded diagnostic memo tests."""
import numpy as np
import pytest

from rtpipeline import radiomics_ct_contract as contract


def test_load_settings_preserve_pre_crop_admission():
    settings = {"preCrop": True, "minimumROISize": 64, "minimumROIDimensions": 2}
    assert contract._load_settings(settings) == settings
    assert contract._load_settings({**settings, "preCrop": False}) == {}
    resampled = {**settings, "interpolator": "sitkBSpline", "resampledPixelSpacing": [1, 1, 1]}
    assert contract._load_settings(resampled) == {
        "interpolator": "sitkBSpline", "resampledPixelSpacing": [1, 1, 1]}


def test_load_settings_remove_only_post_load_resegmentation():
    settings = {"label": 2, "normalize": True, "padDistance": 8,
                "resegmentRange": [-1000, 400], "resegmentMode": "absolute",
                "resegmentShape": False}
    assert contract._load_settings(settings) == {"label": 2, "normalize": True, "padDistance": 8}
    assert "resegmentRange" in settings


def test_fake_factory_falls_back():
    class Fake:
        settings = {}
    fake = Fake()
    assert not contract._share_ct_load(object(), object(), [fake])
    assert not hasattr(fake, "loadImage")


def inputs():
    sitk = pytest.importorskip("SimpleITK")
    pytest.importorskip("radiomics")
    image = sitk.GetImageFromArray(np.arange(12**3, dtype=np.int16).reshape((12,)*3))
    array = np.zeros((12,)*3, dtype=np.uint8)
    array[3:8, 3:8, 3:8] = 1
    return image, sitk.GetImageFromArray(array)


def test_memo_is_bounded_and_content_sensitive():
    image, _ = inputs()
    import SimpleITK as sitk
    contract._IMAGE_DIAGNOSTICS.clear()
    first = contract._original_image_diagnostics(image)
    assert first == contract._original_image_diagnostics(sitk.Image(image))
    image[0, 0, 0] = 3000
    assert first != contract._original_image_diagnostics(image)
    for value in range(8):
        image[0, 0, 0] = value
        contract._original_image_diagnostics(image)
        assert len(contract._IMAGE_DIAGNOSTICS) <= 3
    assert all(not isinstance(v, sitk.Image)
               for entry in contract._IMAGE_DIAGNOSTICS.values() for v in entry.values())
    moved = sitk.Image(image)
    moved.SetSpacing((2., 2., 2.))
    assert contract._original_image_diagnostics(moved) != contract._original_image_diagnostics(image)


def test_unknown_loader_version_and_mismatched_settings_fall_back(monkeypatch):
    image, mask = inputs()
    import radiomics
    from radiomics.featureextractor import RadiomicsFeatureExtractor as Extractor
    left, right = Extractor(), Extractor()
    right.settings["padDistance"] = 42
    assert not contract._share_ct_load(image, mask, [left, right])
    assert left.loadImage is Extractor.loadImage
    right.settings = dict(left.settings)
    right.loadImage = lambda *args, **kwargs: (image, mask)
    assert not contract._share_ct_load(image, mask, [left, right])
    monkeypatch.setattr(radiomics, "__version__", "v9.0.0")
    assert not contract._share_ct_load(image, mask, [left])


def test_shared_qc_load_and_changed_call_fallback():
    image, mask = inputs()
    from radiomics.featureextractor import RadiomicsFeatureExtractor as Extractor
    extractors = contract.build_ct_extractors(
        lambda: Extractor(resampledPixelSpacing=[1, 1, 1], interpolator="sitkBSpline"),
        (-1000, 400))
    before = dict(contract._OVERHEAD_COUNTS)
    scope = contract._share_ct_load(image, mask, extractors)
    assert scope is not None
    for extractor in extractors:
        contract.resampled_mask_qc(image, mask, extractor, None)
    assert contract._OVERHEAD_COUNTS["shared_loads"] == before["shared_loads"] + 1
    assert contract._OVERHEAD_COUNTS["load_hits"] == before["load_hits"] + 2
    extractor = extractors[0]
    extractor.loadImage(image, mask, None, **{**extractor.settings, "padDistance": 9})
    assert contract._OVERHEAD_COUNTS["fallbacks"] == before["fallbacks"] + 1


def test_scope_releases_buffers_without_collecting_extractor_cycles():
    import weakref
    image, mask = inputs()
    from radiomics.featureextractor import RadiomicsFeatureExtractor as Extractor
    extractor = Extractor()
    extractor.cycle = extractor
    scope = contract._share_ct_load(image, mask, [extractor])
    reference = weakref.ref(scope)
    scope.load(extractor.__class__.loadImage, image, mask, None, extractor.settings)
    loaded_reference = weakref.ref(scope.loaded[1])
    del scope
    assert reference() is None
    assert loaded_reference() is None
    # A retained extractor is still usable through its original loader.
    assert extractor.loadImage(image, mask, None, **extractor.settings)[0] is image


def test_load_settings_match_types_and_float_bits():
    same = contract._same_load_settings
    assert same({"label": 1, "resampledPixelSpacing": [1., 1., 1.]},
                {"label": 1, "resampledPixelSpacing": [1., 1., 1.]})
    assert not same({"label": 1}, {"label": True})
    assert not same({"padDistance": 5}, {"padDistance": 5.0})
    assert not same({"normalizeScale": 0.0}, {"normalizeScale": -0.0})
    assert not same({"resampledPixelSpacing": [1., 1., 1.]},
                    {"resampledPixelSpacing": (1., 1., 1.)})
    assert not same({"normalizeScale": float("nan")}, {"normalizeScale": float("nan")})
    assert same({"label": 1, "resegmentRange": [-1000, 400]}, {"label": 1})
