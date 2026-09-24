"""Standalone bit-exact CT/robustness comparison with commit 2369fbb.

Run with the PyRadiomics interpreter, PYTHONNOUSERSITE=1 and
PYTHONDONTWRITEBYTECODE=1. Scratch stays inside the worktree. No pytest needed.
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
from contextlib import contextmanager
import importlib.util
import json
import logging
import math
from pathlib import Path
import resource
import struct
import subprocess
import sys
import tempfile
import time
import threading

import numpy as np
import SimpleITK as sitk
from pydicom.uid import generate_uid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from radiomics import featureextractor
from rtpipeline import radiomics_ct_contract as current
from rtpipeline import radiomics_parallel as parallel
from rtpipeline import robustness_watchdog as watchdog
from rtpipeline.config import PipelineConfig


def exact(a, b, location="rows"):
    assert type(a) is type(b), (location, type(a), type(b))
    if isinstance(a, dict):
        assert list(a) == list(b), (location, "key order", list(a), list(b))
        for key in a:
            exact(a[key], b[key], location + "." + str(key))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), (location, "length")
        for index, (left, right) in enumerate(zip(a, b)):
            exact(left, right, location + f"[{index}]")
    elif isinstance(a, (float, np.floating)):
        if math.isnan(a):
            assert math.isnan(b), location
        elif isinstance(a, np.floating):
            assert a.tobytes() == b.tobytes(), (location, a, b)
        else:
            assert struct.pack("!d", a) == struct.pack("!d", b), (location, a, b)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape, location
        assert a.tobytes() == b.tobytes(), location
    else:
        assert a == b, (location, a, b)


def baseline_module(name, scratch):
    source = subprocess.check_output(
        ["git", "show", f"2369fbb:rtpipeline/{name}.py"], cwd=ROOT)
    path = scratch / (name + ".py")
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("rtpipeline._baseline_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def contract_module(module):
    name = "rtpipeline.radiomics_ct_contract"
    previous = sys.modules[name]
    sys.modules[name] = module
    try:
        yield
    finally:
        sys.modules[name] = previous


def make_image(array):
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.98, 0.98, 3.0))
    image.SetOrigin((-243.7, 18.25, -301.2))
    return image


def synthetic_cases():
    shape = (150, 512, 512)
    rng = np.random.default_rng(713)
    pixels = rng.integers(-100, 101, size=shape, dtype=np.int16)
    pixels[15:35, 30:70, 30:70] = 800
    image = make_image(pixels)
    del pixels
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    def ellipsoid(center, radii):
        cz, cy, cx = center
        rz, ry, rx = radii
        return (((z-cz)/rz)**2 + ((y-cy)/ry)**2 + ((x-cx)/rx)**2 <= 1)
    specs = [
        ("small_5ml", (75, 256, 256), (4, 10, 10)),
        ("medium", (75, 256, 256), (10, 28, 28)),
        ("large_1500ml", (75, 256, 256), (18, 85, 82)),
        ("border", (75, 256, 0), (5, 12, 12)),
        ("resegment_empty", (25, 50, 50), (3, 7, 7)),
        ("partial_resegment", (25, 70, 50), (3, 10, 10)),
    ]
    for name, center, radii in specs:
        yield name, image, make_image(ellipsoid(center, radii).astype(np.uint8)), False
    mask = np.zeros(shape, dtype=np.uint8)
    mask[72:78, 240:260, 250:253] = 1
    mask[75, 245:267, 251:268] = 1
    yield "thin_irregular", image, make_image(mask), False
    mask[:] = 0
    mask[75, 256, 256:258] = 1
    yield "below_minimum", image, make_image(mask), False
    small = make_image(ellipsoid((75, 256, 256), (4, 10, 10)).astype(np.uint8))
    yield "fallback_settings", image, small, True
    # Three distinct noise images, each reused with two different masks in this
    # same process. Content-keyed diagnostics also survive robustness file reads.
    other = make_image(ellipsoid((80, 220, 280), (4, 10, 10)).astype(np.uint8))
    from rtpipeline.radiomics_robustness import add_noise_to_image
    for noise in range(3):
        noisy = add_noise_to_image(image, float(noise + 1), rng)
        yield f"noise{noise}_a", noisy, small, False
        yield f"noise{noise}_b", noisy, other, False


def check_cache_and_scope():
    """Native checks also run where pytest is deliberately unavailable."""
    import weakref
    import radiomics
    from radiomics.generalinfo import GeneralInfo
    current._IMAGE_DIAGNOSTICS.clear()
    checked = 0
    for dtype in (np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32,
                  np.uint64, np.int64, np.float32, np.float64):
        image = sitk.GetImageFromArray(np.arange(120).reshape(4, 5, 6).astype(dtype))
        for variant in range(4):
            if variant == 1:
                image.SetMetaData("synthetic", "changed")
            elif variant == 2:
                image[0, 0, 0] = 1
            elif variant == 3:
                image.SetSpacing((0.7, 0.9, 3.0))
            info = GeneralInfo()
            info.addImageElements(image)
            expected = {k: v for k, v in info.getGeneralInfo().items()
                        if k.startswith("diagnostics_Image-original_")}
            exact(expected, current._original_image_diagnostics(image), "image memo")
            assert len(current._IMAGE_DIAGNOSTICS) <= 3
            checked += 1
    image = sitk.GetImageFromArray(np.arange(12**3, dtype=np.int16).reshape((12,)*3))
    array = np.zeros((12,)*3, dtype=np.uint8)
    array[3:8, 3:8, 3:8] = 1
    mask = sitk.GetImageFromArray(array)
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.cycle = extractor
    scope = current._share_ct_load(image, mask, [extractor])
    reference = weakref.ref(scope)
    extractor.loadImage(image, mask, None, **extractor.settings)
    buffer = weakref.ref(scope.loaded[1])
    del scope
    assert reference() is None and buffer() is None
    assert extractor.loadImage(image, mask, None, **extractor.settings)[0] is image
    left = featureextractor.RadiomicsFeatureExtractor()
    right = featureextractor.RadiomicsFeatureExtractor()
    right.settings["padDistance"] = 42
    assert current._share_ct_load(image, mask, [left, right]) is None
    right.settings = dict(left.settings)
    right.loadImage = lambda *args, **kwargs: (image, mask)
    assert current._share_ct_load(image, mask, [left, right]) is None
    version = radiomics.__version__
    try:
        radiomics.__version__ = "unsupported-test-version"
        assert current._share_ct_load(image, mask, [left]) is None
    finally:
        radiomics.__version__ = version
    right = featureextractor.RadiomicsFeatureExtractor()
    right.settings["padDistance"] = float(left.settings["padDistance"])
    assert current._share_ct_load(image, mask, [left, right]) is None
    right.settings = dict(left.settings)
    left.settings["normalizeScale"] = 0.0
    right.settings["normalizeScale"] = -0.0
    assert current._share_ct_load(image, mask, [left, right]) is None
    current._IMAGE_DIAGNOSTICS.clear()
    for key in current._OVERHEAD_COUNTS:
        current._OVERHEAD_COUNTS[key] = 0
    print(json.dumps({"native_diagnostic_checks": checked, "memo_bound": 3,
                      "scope_release": True, "fallback_checks": 5}), flush=True)


class PeakRSS:
    """Sample resident pages during each call; ru_maxrss remains process-wide."""
    def __enter__(self):
        self.stop = threading.Event()
        self.peak = 0
        def sample():
            while not self.stop.is_set():
                with open("/proc/self/status") as stream:
                    for line in stream:
                        if line.startswith("VmRSS:"):
                            self.peak = max(self.peak, int(line.split()[1]) / 1024)
                            break
                self.stop.wait(0.02)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="store_true", help="Print cumulative extraction profiles")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--masks", type=Path, nargs="+")
    parser.add_argument("--cases", nargs="+", help="Optional synthetic case subset")
    args = parser.parse_args()
    if bool(args.image) != bool(args.masks):
        parser.error("--image and --masks must be supplied together")
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    check_cache_and_scope()
    params = ROOT / "rtpipeline/radiomics_params.yaml"
    decision = current.classify_ct_roi("AutoRTS_total", "urinary_bladder")
    with tempfile.TemporaryDirectory(prefix=".equivalence-", dir=ROOT) as directory:
        scratch = Path(directory)
        old = baseline_module("radiomics_ct_contract", scratch)
        old_parallel = baseline_module("radiomics_parallel", scratch)
        config = PipelineConfig(scratch, scratch, scratch)
        config.radiomics_params_file = params
        course = scratch / "synthetic_patient" / "synthetic_course"
        course.mkdir(parents=True)
        identity = dict(patient_id="synthetic_patient", course_id="synthetic_course",
                        series_uid=generate_uid(entropy_srcs=["rtpipeline-overhead-synthetic-series"]),
                        segmentation_source="AutoRTS_total", mask_identity="synthetic_mask",
                        roi_original_name="urinary_bladder", stable_roi_identifier="synthetic_roi")
        cases = synthetic_cases()
        if args.image:
            image = sitk.ReadImage(str(args.image))
            cases = ((f"input_{i}", image, sitk.ReadImage(str(path)), False)
                     for i, path in enumerate(args.masks))
        records = []
        from radiomics.glcm import RadiomicsGLCM
        original_mcc = RadiomicsGLCM.getMCCFeatureValue
        for name, image, mask, fallback in cases:
            if args.cases and name not in args.cases:
                continue
            calls = 0
            def factory():
                nonlocal calls
                ext = featureextractor.RadiomicsFeatureExtractor(str(params))
                calls += 1
                if fallback and calls % 3 == 1:
                    ext.settings["padDistance"] = 6
                return ext
            kwargs = dict(factory=factory, decision=decision, common_metadata=identity,
                          run_identifier="synthetic_equivalence", code_revision="2369fbb",
                          native_voxel_count=int(np.count_nonzero(sitk.GetArrayViewFromImage(mask))),
                          required=False)
            for route in ("radiomics", "robustness"):
                RadiomicsGLCM.getMCCFeatureValue = original_mcc
                if route == "robustness":
                    # Use the real baseline task-file preparer; no invented task schema.
                    with contract_module(old):
                        task = old_parallel._prepare_radiomics_task(
                            image, mask, config, "AutoRTS_total", "urinary_bladder",
                            course, scratch, False, source_identity=identity,
                            run_identifier="synthetic_equivalence")
                    task[1]["code_revision"] = "2369fbb"
                    task[1]["extra_metadata"] = {"perturbation_id": name}
                timings = []
                outputs = []
                peaks = []
                before = dict(current._OVERHEAD_COUNTS)
                for module, worker in ((old, old_parallel), (current, parallel)):
                    calls = 0
                    print(f"START {name} {route} {'old' if module is old else 'new'}", file=sys.stderr, flush=True)
                    profile = cProfile.Profile() if args.profile else None
                    if profile is not None:
                        profile.enable()
                    start = time.perf_counter()
                    with PeakRSS() as peak, contract_module(module):
                        if route == "radiomics":
                            output = module.extract_ct_roi_arms(image, mask, **kwargs)
                        else:
                            # Exercise the actual watchdog-wrapped loader too.
                            watchdog._sender = lambda stage: None
                            try:
                                output = worker._isolated_radiomics_extraction(task)
                            finally:
                                watchdog._sender = None
                    timings.append(time.perf_counter() - start)
                    if profile is not None:
                        profile.disable()
                        print(f"PROFILE {name} {route} {'old' if module is old else 'new'}", flush=True)
                        pstats.Stats(profile).sort_stats("cumulative").print_stats(25)
                    outputs.append(output)
                    peaks.append(round(peak.peak, 1))
                exact(*outputs, location=f"{name}/{route}")
                counts = {k: current._OVERHEAD_COUNTS[k]-before[k] for k in before}
                assert counts["fallbacks"] if fallback and route == "radiomics" else counts["shared_loads"], counts
                record = dict(case=name, route=route, exact=True, old_s=round(timings[0], 3),
                              new_s=round(timings[1], 3),
                              old_peak_rss_mib=peaks[0], new_peak_rss_mib=peaks[1],
                              peak_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 1),
                              fast_paths=counts, memo_entries=len(current._IMAGE_DIAGNOSTICS))
                records.append(record)
                print(json.dumps(record), flush=True)
            # Keep scratch disk bounded across masks/noise images.
            for path in scratch.glob("*.nrrd"):
                path.unlink()
        print(json.dumps({"passed_comparisons": len(records), "counters": current._OVERHEAD_COUNTS,
                          "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}), flush=True)


if __name__ == "__main__":
    main()
