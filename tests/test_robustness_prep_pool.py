"""Per-ROI robustness preparation in a worker pool equals the serial loop.

``_prepare_robustness_roi`` streams one ROI's NTCV grid and writes its task
files; ``_prepare_robustness_rois`` runs it for several ROIs in this process or
in fork/spawn workers. For every ROI the perturbation identifiers, geometric
non-measurements and task descriptors (including content-addressed file
names and mask identities) must equal what the pre-change loop produced with
``generate_ntcv_perturbations`` and ``_prepare_radiomics_task``. Failures must
surface as the serial loop raised them: the first failing ROI in order.
"""
from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np
import pytest
import SimpleITK as sitk

from rtpipeline import radiomics_parallel as rp
from rtpipeline import radiomics_robustness as rr
from robustness_performance_fixture import synthetic, task_context


def _masks():
    image, masks = synthetic((24, 48, 48))
    arrays = {name: sitk.GetArrayFromImage(mask).astype(bool) for name, mask in masks.items()}
    edge = np.zeros_like(arrays["large"])
    edge[:3, 10:20, 10:20] = True  # touches the first slice: a clipped translation
    arrays["edge"] = edge
    return image, arrays


def _requests(tmp_path, image, arrays, perturbation, *, identity_for=None):
    config, course, identity = task_context(tmp_path)
    temp = tmp_path / "tasks"
    temp.mkdir(exist_ok=True)
    cache = {"images": {}}
    noise_levels = rr._ntcv_factors(perturbation)[3]
    image_paths = {
        index: str(rp._radiomics_task_image_path(noisy, temp, cache))
        for index, (_, noisy) in enumerate(rr._ntcv_noise_images(image, noise_levels, {}))
    }
    geometry = (image.GetSize(), image.GetSpacing(), image.GetDirection(), image.GetOrigin())
    requests = []
    for name, array in arrays.items():
        source_identity = dict(identity, roi_original_name="urinary_bladder")
        if identity_for is not None:
            source_identity = identity_for(name, source_identity)
        requests.append({
            "roi_name": "urinary_bladder", "source": "AutoRTS_total", "mask": array,
            "ct_geometry": geometry, "perturbation": perturbation, "image_paths": image_paths,
            "config": config, "course_dir": str(course), "temp_dir": str(temp),
            "run_identifier": "synthetic-run", "source_identity": source_identity,
            "itk_threads": sitk.ProcessObject.GetGlobalDefaultNumberOfThreads(),
            "label": name,
        })
    return requests, config, course, identity, temp


def _serial_reference(image, array, perturbation, config, course, identity, temp):
    """The pre-change per-ROI loop body of robustness_for_course."""
    from rtpipeline.radiomics import _mask_from_array_like

    mask_img = _mask_from_array_like(image, array)
    masks, images = rr.generate_ntcv_perturbations(
        mask_img, image, perturbation, "urinary_bladder", _noise_cache={})
    ids = list(masks)
    nonmeasurements = [(k, v) for k, v in masks.items() if isinstance(v, rr.GeometricNonmeasurement)]
    cache = {"images": {}}
    tasks = []
    for key, mask in masks.items():
        if isinstance(mask, rr.GeometricNonmeasurement):
            continue
        task_file, params = rp._prepare_radiomics_task(
            images[key], mask, config, "AutoRTS_total", "urinary_bladder", course, temp, False,
            "synthetic-run", dict(identity, roi_original_name="urinary_bladder"), _cache=cache)
        tasks.append((key, task_file, params))
    return {"perturbation_ids": ids, "nonmeasurements": nonmeasurements, "tasks": tasks}


@pytest.fixture(autouse=True)
def _single_itk_thread():
    threads = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    yield
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads)


@pytest.mark.parametrize("workers,method", [(1, None), (3, "fork"), (2, "spawn")])
def test_pooled_preparation_equals_the_serial_loop(tmp_path, workers, method):
    image, arrays = _masks()
    perturbation = rr.PerturbationConfig()
    requests, config, course, identity, temp = _requests(tmp_path, image, arrays, perturbation)
    context = mp.get_context(method) if method else None
    results = rr._prepare_robustness_rois(requests, workers, context)
    assert len(results) == len(arrays)
    for (name, array), result in zip(arrays.items(), results):
        expected = _serial_reference(image, array, perturbation, config, course, identity, temp)
        assert result == expected, name
        for _, task_file, params in result["tasks"]:
            written = sitk.ReadImage(params["mask_path"])
            assert rr._perturbed_mask_identity(written) == params["perturbed_mask_identity"]
    assert any(r["nonmeasurements"] for r in results)  # the edge ROI's clipped shift
    assert not list(temp.glob(".*"))  # no partial mask files remain


def test_legacy_volume_only_grid_equals_the_serial_loop(tmp_path):
    image, arrays = _masks()
    perturbation = rr.PerturbationConfig(max_translation_mm=0.0, n_random_contour_realizations=0,
                                         noise_levels=[0.0])
    requests, config, course, identity, temp = _requests(tmp_path, image, arrays, perturbation)
    original = rp._radiomics_task_image_path(image, temp, {"images": {}})
    for request in requests:
        request["image_paths"] = {"original": str(original)}
    results = rr._prepare_robustness_rois(requests, 2, mp.get_context("fork"))
    from rtpipeline.radiomics import _mask_from_array_like
    for (name, array), result in zip(arrays.items(), results):
        masks = rr.generate_perturbed_masks(_mask_from_array_like(image, array),
                                            perturbation.small_volume_changes, "urinary_bladder")
        assert result["perturbation_ids"] == list(masks)
        cache = {"images": {}}
        expected = [
            (key, *rp._prepare_radiomics_task(
                image, mask, config, "AutoRTS_total", "urinary_bladder", course, temp, False,
                "synthetic-run", dict(identity, roi_original_name="urinary_bladder"), _cache=cache))
            for key, mask in masks.items() if not isinstance(mask, rr.GeometricNonmeasurement)
        ]
        assert result["tasks"] == expected, name


@pytest.mark.parametrize("workers,method", [(1, None), (4, "fork")])
def test_first_failing_roi_in_order_is_raised(tmp_path, workers, method):
    image, arrays = _masks()
    names = list(arrays)

    def identity_for(name, identity):
        if name in (names[1], names[3]):
            return dict(identity, course_id=f"wrong-{name}")
        return identity

    requests, *_ = _requests(tmp_path, image, arrays, rr.PerturbationConfig(),
                             identity_for=identity_for)
    context = mp.get_context(method) if method else None
    with pytest.raises(RuntimeError, match=f"failed to prepare robustness task.*wrong-{names[1]}"):
        rr._prepare_robustness_rois(requests, workers, context)


def test_a_dead_preparation_worker_fails_the_course(tmp_path, monkeypatch):
    image, arrays = _masks()
    requests, *_ = _requests(tmp_path, image, arrays, rr.PerturbationConfig())
    real = rr._prepare_robustness_roi

    def dying(request):
        if request["label"] == "border":
            os._exit(9)
        return real(request)

    monkeypatch.setattr(rr, "_prepare_robustness_roi", dying)
    with pytest.raises(RuntimeError, match="preparation worker .* died"):
        rr._prepare_robustness_rois(requests, 2, mp.get_context("fork"))


def test_prepared_image_path_equals_written_image(tmp_path):
    image, arrays = _masks()
    config, course, identity = task_context(tmp_path)
    mask = sitk.GetImageFromArray(arrays["large"].astype(np.uint8))
    mask.CopyInformation(image)
    args = (mask, config, "AutoRTS_total", "urinary_bladder", course, tmp_path, False,
            "synthetic-run", identity)
    written = rp._prepare_radiomics_task(image, *args)
    path = rp._radiomics_task_image_path(image, tmp_path)
    assert rp._prepare_radiomics_task(None, *args, _image_path=path) == written
    with pytest.raises(RuntimeError, match="image is missing"):
        rp._prepare_radiomics_task(None, *args, _image_path=tmp_path / "absent.nrrd")


def test_packaged_params_copy_is_replaced_atomically(tmp_path):
    """Preparation workers hash this copy while other workers rewrite it."""
    from rtpipeline.config import PipelineConfig
    from rtpipeline.radiomics import _get_params_file

    config = PipelineConfig(tmp_path, tmp_path, tmp_path / "logs")
    first = _get_params_file(config, "CT")
    content = first.read_bytes()
    assert content
    second = _get_params_file(config, "CT")
    assert second == first and second.read_bytes() == content
    assert sorted(p.name for p in first.parent.iterdir()) == [first.name]
