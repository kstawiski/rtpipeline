"""Bit-exact comparisons with commit 5de543a, including cache path assertions."""
import os

import numpy as np
import pytest
import SimpleITK as sitk

from rtpipeline import radiomics_robustness as rr
from rtpipeline import radiomics_parallel as rp
from robustness_performance_fixture import baseline, synthetic, assert_image, task_context


@pytest.fixture(scope='module')
def old():
    threads = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    try:
        yield baseline()
    finally:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads)


@pytest.mark.parametrize('name', ['small', 'large', 'border', 'thin'])
def test_all_81_conditions_and_task_files(old, tmp_path, name, monkeypatch):
    import hashlib
    shape = (200, 512, 512) if os.environ.get("RTPIPELINE_PERFORMANCE_FULL_SIZE") == "1" else (48, 80, 80)
    image, masks = synthetic(shape)
    config = rr.PerturbationConfig()
    expected_masks, expected_images = old.generate_ntcv_perturbations(masks[name], image, config, name)
    noise_cache = {}
    actual_masks, actual_images = rr.generate_ntcv_perturbations(masks[name], image, config, name, _noise_cache=noise_cache)
    assert list(actual_masks) == list(expected_masks)
    assert list(actual_images) == list(expected_images)
    assert len(actual_masks) == 81
    cfg, course, identity = task_context(tmp_path)
    expected_tasks = {}
    expected_files = {}
    for key in expected_masks:
        assert_image(actual_images[key], expected_images[key])
        if isinstance(expected_masks[key], rr.GeometricNonmeasurement):
            assert actual_masks[key] == expected_masks[key]
            continue
        assert_image(actual_masks[key], expected_masks[key])
        assert rr._perturbed_mask_identity(actual_masks[key]) == old._perturbed_mask_identity(expected_masks[key])
        _, params = old._prepare_radiomics_task(expected_images[key], expected_masks[key], cfg,
                                              'AutoRTS_total', 'urinary_bladder', course, tmp_path, False,
                                              'synthetic-run', identity)
        expected_tasks[key] = params
    for path in tmp_path.glob('*.nrrd'):
        expected_files[path.name] = (hashlib.sha256(path.read_bytes()).hexdigest(), sitk.ReadImage(str(path)))
        path.unlink()
    cache = {}
    calls = []
    digest = rr._perturbed_mask_identity
    monkeypatch.setattr(rr, '_perturbed_mask_identity', lambda m: (calls.append(id(m)), digest(m))[1])
    for key, expected in expected_tasks.items():
        _, actual = rp._prepare_radiomics_task(actual_images[key], actual_masks[key], cfg,
                                             'AutoRTS_total', 'urinary_bladder', course, tmp_path, False,
                                             'synthetic-run', identity, _cache=cache)
        assert actual == expected
    assert len(cache['images']) == 3
    assert len(calls) == len({id(actual_masks[key]) for key in expected_tasks})
    assert len(calls) <= 27
    assert {p.name for p in tmp_path.glob('*.nrrd')} == set(expected_files)
    for filename, (digest, expected) in expected_files.items():
        path = tmp_path/filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        assert_image(sitk.ReadImage(str(path)), expected)
    # The next ROI reuses the exact same noise objects, without another draw.
    def forbidden(*args, **kwargs):
        raise AssertionError('noise regenerated')
    monkeypatch.setattr(rr, 'add_noise_to_image', forbidden)
    _, reused = rr.generate_ntcv_perturbations(masks['large'], image, config, 'large', _noise_cache=noise_cache)
    assert all(reused[key] is actual_images[key] for key in reused)


@pytest.mark.parametrize('spacing', [(0.98, 0.98, 3.), (1., 1., 1.), (0.71, 1.37, 2.53)])
@pytest.mark.parametrize('kind', ['tiny', 'border', 'large', 'ties', 'full'])
@pytest.mark.parametrize('tau', [-0.15, 0.15, 3.0])
def test_volume_exact_and_distance_map(old, spacing, kind, tau, monkeypatch):
    a = np.zeros((31, 43, 47), np.uint8)
    if kind == 'tiny': a[13:15, 20:22, 24:27] = 1
    elif kind == 'border': a[8:22, 13:29, :12] = 1
    elif kind == 'large': a[3:28, 4:38, 5:41] = 1
    elif kind == 'ties': a[12:18, 16:24, 20:28] = 1
    else: a[:] = 1
    mask = sitk.GetImageFromArray(a); mask.SetSpacing(spacing)
    expected = old.volume_adapt_mask(mask, tau)
    full_distance = sitk.SignedMaurerDistanceMap(mask, insideIsPositive=True, squaredDistance=False, useImageSpacing=True)
    calls = []
    real = sitk.SignedMaurerDistanceMap
    def checked(binary, **kwargs):
        result = real(binary, **kwargs)
        x,y,z = result.GetSize()
        assert np.array_equal(sitk.GetArrayViewFromImage(result), sitk.GetArrayViewFromImage(full_distance)[:z,:y,:x])
        calls.append(result.GetSize())
        return result
    monkeypatch.setattr(sitk, 'SignedMaurerDistanceMap', checked)
    actual = rr.volume_adapt_mask(mask, tau)
    if expected is None: assert actual is None
    else: assert_image(actual, expected)
    if kind == 'ties' and tau == -0.15:
        assert calls and all(size != mask.GetSize() for size in calls)


def test_margin_enlargement_and_full_volume_fallback(old, monkeypatch):
    a = np.zeros((40, 60, 80), np.uint8)
    a[:3, :3, :3] = 1
    a[7:10, 56:59, 76:79] = 1
    mask = sitk.GetImageFromArray(a); mask.SetSpacing((0.98, 0.98, 3.))
    # Request all available voxels: no bounded candidate region can suffice.
    tau = a.size / int(a.sum()) - 1
    expected = old.volume_adapt_mask(mask, tau)
    calls = []
    real = sitk.SignedMaurerDistanceMap
    def record(binary, **kwargs):
        calls.append(binary.GetSize())
        return real(binary, **kwargs)
    monkeypatch.setattr(sitk, 'SignedMaurerDistanceMap', record)
    assert_image(rr.volume_adapt_mask(mask, tau), expected)
    assert len(calls) > 1
    assert calls[0] != mask.GetSize()
    assert calls[-1] == mask.GetSize()


def test_shifted_crop_is_not_bit_exact():
    a = np.zeros((80, 100, 100), np.uint8); a[30:45, 43:63, 38:59] = 1
    mask = sitk.GetImageFromArray(a); mask.SetSpacing((0.98, 0.98, 3.))
    def distance(image):
        return sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(image, insideIsPositive=True, squaredDistance=False, useImageSpacing=True))
    full = distance(mask)
    assert not np.array_equal(distance(mask[35:65, 40:67, 27:49]), full[27:49, 40:67, 35:65])
    assert np.array_equal(distance(mask[:65, :67, :49]), full[:49, :67, :65])


def test_custom_invariants_are_computed_once(old, tmp_path, monkeypatch):
    from rtpipeline import radiomics_ct_contract as contract
    image, masks = synthetic()
    cfg, course, identity = task_context(tmp_path)
    identity['segmentation_source'] = 'Custom'
    custom = tmp_path/'custom.yaml'
    custom.write_text('custom_structures:\n  - name: urinary_bladder\n    operation: union\n    source_structures: [urinary_bladder]\n')
    cfg.custom_structures_config = custom
    args = (image, masks['large'], cfg, 'Custom', 'urinary_bladder', course, tmp_path, False, 'synthetic-run', identity)
    expected = old._prepare_radiomics_task(*args)
    calls = {}
    for name in ('classify_ct_roi', 'load_custom_structure_provenance', 'configured_parameter_hash'):
        original = getattr(contract, name)
        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] = calls.get(_name, 0) + 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(contract, name, counted)
    cache = {}
    for _ in range(81):
        assert rp._prepare_radiomics_task(*args, _cache=cache) == expected
    assert calls == {'classify_ct_roi': 1, 'load_custom_structure_provenance': 1,
                     'configured_parameter_hash': 2}
    assert cache['images'][id(image)][0] is image
    assert cache['masks'][id(masks['large'])][0] is masks['large']


@pytest.mark.parametrize('seed', range(5))
def test_randomized_masks_at_realistic_indices(old, seed):
    rng = np.random.default_rng(seed)
    a = np.zeros((48, 512, 512), np.uint8)
    z,y,x = (int(v) for v in rng.integers((5, 200, 200), (30, 460, 460)))
    a[z:z+8, y:y+15, x:x+17] = rng.integers(0, 2, (8,15,17), dtype=np.uint8)
    mask = sitk.GetImageFromArray(a); mask.SetSpacing((0.98, 0.98, 3.))
    for tau in (-0.15, 0.15, 1.0):
        assert_image(rr.volume_adapt_mask(mask, tau), old.volume_adapt_mask(mask, tau))
    for offset in (0.1, 2.0, 9.0):
        assert_image(rr.randomize_contour(mask, offset, np.random.default_rng(seed)),
                     old.randomize_contour(mask, offset, np.random.default_rng(seed)))
