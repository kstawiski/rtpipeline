"""Public IBSI benchmark adapter. Run with the pinned radiomics Python -I -B.

The existing validation scripts and data remain read-only. Their extraction
settings, feature mapping, workbook comparison and diagnostic comparison are
used directly. This adapter routes original CT/mask inputs through the packaged
option and compares every feature with the retained ibsi_preprocessed outputs.
"""
import argparse
import hashlib
import importlib.metadata
import json
import logging
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import SimpleITK as sitk
import yaml

from rtpipeline.radiomics import _extractor, _apply_params_to_extractor
from rtpipeline.ibsi_preprocessing import ibsi_preprocess, SIGMA_KEY
from radiomics.featureextractor import RadiomicsFeatureExtractor

sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
logging.getLogger("radiomics").setLevel(logging.ERROR)


def parameters(configuration, base=None):
    p = {"setting": dict(base or {}), "imageType": {"Original": {}},
         "featureClass": {k: [] for k in ("shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm")},
         "radiomics": {"preprocessing": "ibsi"}}
    s = p["setting"]
    if configuration != "A":
        s.update(resampledPixelSpacing=[2., 2., 0. if configuration == "B" else 2.],
                 interpolator="sitkBSpline" if configuration == "E" else "sitkLinear")
    if configuration in "ABC":
        s["resegmentRange"] = [-500. if configuration in "AB" else -1000., 400.]
    if configuration == "D":
        s.update(resegmentRange=[-3., 3.], resegmentMode="sigma")
    if configuration == "E":
        s["resegmentRange"] = [-1000., 400.]
        p["radiomics"]["resegment_sigma"] = 3.
    return p


def fingerprint(result):
    def plain(v):
        if isinstance(v, np.ndarray):
            return {"dtype": str(v.dtype), "shape": list(v.shape), "bytes": v.tobytes().hex()}
        if isinstance(v, np.generic):
            return {"dtype": str(v.dtype), "bytes": v.tobytes().hex()}
        if isinstance(v, dict):
            return {k: plain(w) for k, w in v.items()}
        if isinstance(v, (tuple, list)):
            return [plain(w) for w in v]
        return v
    return hashlib.sha256(json.dumps(plain(result), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check_default():
    z, y, x = np.indices((20, 21, 22))
    a = (-900 + 9*x + 7*y + 13*z).astype(np.float32)
    m = np.zeros(a.shape, np.uint8); m[3:17, 3:18, 3:19] = 1
    image, mask = sitk.GetImageFromArray(a), sitk.GetImageFromArray(m)
    for i in (image, mask):
        i.SetSpacing((1.2, 1.3, 1.7))
    default = _extractor(None)
    assert type(default) is RadiomicsFeatureExtractor
    params = yaml.safe_load((REPO / "rtpipeline/radiomics_params.yaml").read_text())
    baseline = RadiomicsFeatureExtractor()
    _apply_params_to_extractor(baseline, params)
    observed, expected = default.execute(image, mask), baseline.execute(image, mask)
    digest = fingerprint(observed)
    assert digest == fingerprint(expected)
    # Reference fingerprint for the pinned public/synthetic configuration.
    before = "4174115f0c5c677b17b56ea525606cbda57639c999035618ada18b69bcc08fd7"
    versions = {k: importlib.metadata.version(k) for k in ("numpy", "pyradiomics", "SimpleITK")}
    if versions == {"numpy": "1.26.4", "pyradiomics": "3.0.1", "SimpleITK": "2.5.3"}:
        assert digest == before, (digest, before)
    return dict(sha256=digest, reference_sha256=before, versions=versions, entries=len(observed),
                features=sum(k.startswith(("original_", "log-", "wavelet-")) for k in observed),
                parameter_sha256=hashlib.sha256((REPO / "rtpipeline/radiomics_params.yaml").read_bytes()).hexdigest())


def check_synthetic(folder):
    from rtpipeline.radiomics_ct_contract import build_ct_extractors, resampled_mask_qc, effective_parameter_hash
    from types import MethodType
    z, y, x = np.indices((10, 12, 14))
    a = (-450 + x*40 + y*5 + z*11).astype(np.float32)
    a[5, 5, 5] = 1800
    m = np.zeros(a.shape, np.uint8); m[2:8, 2:10, 2:12] = 7
    image, mask = sitk.GetImageFromArray(a), sitk.GetImageFromArray(m)
    for obj in (image, mask):
        obj.SetSpacing((1.3, 1.4, 2.1))
    p = parameters("E", {"label": 7, "minimumROISize": 5, "normalize": False, "binCount": 32})
    p["featureClass"] = {"shape": ["VoxelVolume"], "firstorder": ["Mean", "Maximum"]}
    path = folder / "synthetic.yaml"
    path.write_text(yaml.safe_dump(p))
    factory = lambda: _extractor(SimpleNamespace(radiomics_params_file=path))
    ext = factory()
    original_settings = dict(ext.settings)
    processed, morph, intensity = ibsi_preprocess(image, mask, ext.settings)
    expected = float(sitk.GetArrayFromImage(processed)[sitk.GetArrayFromImage(intensity) > 0].mean(dtype=np.float64))
    raw_compute = ext.computeFeatures
    calls = []

    def guarded(self, *args, **kwargs):
        calls.append(set(self.enabledFeatures))
        return raw_compute(*args, **kwargs)

    ext.computeFeatures = MethodType(guarded, ext)
    observed = ext.execute(image, mask)
    assert calls == [{"firstorder"}]
    assert float(observed["original_firstorder_Mean"]) == expected
    volume = int(sitk.GetArrayFromImage(morph).sum()) * np.prod(morph.GetSpacing())
    assert float(observed["original_shape_VoxelVolume"]) == volume
    assert ext.settings == original_settings
    assert set(ext.enabledFeatures) == {"shape", "firstorder"}
    qc = resampled_mask_qc(image, mask, ext, None)
    assert qc["morphologic_resampled_voxel_count"] == int(sitk.GetArrayFromImage(morph).sum())
    assert qc["resegment_after_count"] == int(sitk.GetArrayFromImage(intensity).sum())
    assert qc["morphologic_resampled_voxel_count"] == sum(qc[k] for k in (
        "resegment_after_count", "resegment_below_lower_count", "resegment_above_upper_count", "resegment_nonfinite_count"))
    shape, raw, primary = build_ct_extractors(factory, (-500., 400.))
    assert shape.execute(image, mask)["original_shape_VoxelVolume"] == observed["original_shape_VoxelVolume"]
    assert "original_shape_VoxelVolume" not in raw.execute(image, mask)
    raw_qc = resampled_mask_qc(image, mask, raw, None)
    assert raw_qc["resegment_after_count"] == raw_qc["morphologic_resampled_voxel_count"]
    primary_qc = resampled_mask_qc(image, mask, primary, (-500., 400.))
    primary_intensity = ibsi_preprocess(image, mask, primary.settings)[2]
    assert primary_qc["resegment_after_count"] == int(sitk.GetArrayFromImage(primary_intensity).sum())
    native = RadiomicsFeatureExtractor()
    native.settings.update(ext.settings); native.settings.pop("rtpipelinePreprocessing")
    native.enabledImagetypes, native.enabledFeatures = ext.enabledImagetypes, ext.enabledFeatures
    assert effective_parameter_hash(native, arm="test", window=None) != effective_parameter_hash(ext, arm="test", window=None)
    return {"synthetic_option_and_qc": "passed", "morphology_voxels": qc["morphologic_resampled_voxel_count"],
            "intensity_voxels": qc["resegment_after_count"]}


def benchmark(folder):
    sys.path.insert(0, str(REPO / "validation/ibsi/scripts"))
    import run_validation as v
    import check_preprocessing as diagnostics
    from feature_map import load_mapping
    image, mask = v.read_phantom("ct_radiomics")
    refs = v.references()
    mapping, _, _ = load_mapping(v.SOURCE, refs["digital"])
    retained_receipt = json.loads((v.OUT / "run_receipt.json").read_text())
    for source in retained_receipt["source_files"]:
        assert hashlib.sha256((v.ROOT / source["path"]).read_bytes()).hexdigest() == source["sha256"]
    for relative, expected_hash in retained_receipt["raw_output_hashes"].items():
        assert hashlib.sha256((v.ROOT / relative).read_bytes()).hexdigest() == expected_hash
    for relative in ("scripts/run_validation.py", "scripts/feature_map.py", "scripts/check_preprocessing.py"):
        assert hashlib.sha256((v.ROOT / relative).read_bytes()).hexdigest() == retained_receipt["scripts_sha256"][relative]
    preprocessing_rows, feature_rows, comparison_rows = [], [], []
    summary = {}
    for c in "ABCDE":
        p = parameters(c)
        settings = dict(p["setting"])
        if c == "E": settings[SIGMA_KEY] = 3.
        actual = ibsi_preprocess(image, mask, settings)
        expected = v.ibsi_preprocess(image, mask, c)
        for label, a, b in zip(("image", "morphology", "intensity"), actual, expected):
            assert a.GetSize() == b.GetSize() and a.GetSpacing() == b.GetSpacing()
            assert a.GetOrigin() == b.GetOrigin() and a.GetDirection() == b.GetDirection()
            assert np.array_equal(sitk.GetArrayFromImage(a), sitk.GetArrayFromImage(b)), (c, label)
        processed, morph, intensity = actual
        stages = {"init_img": diagnostics.image_values(image), "interp_img": diagnostics.image_values(processed),
                  "init_roi": diagnostics.roi_values(image, mask, mask),
                  "interp_roi": diagnostics.roi_values(processed, morph, morph),
                  "reseg_roi": diagnostics.roi_values(processed, morph, intensity)}
        for ref in refs[c]:
            if not ref["family"].startswith("Diagnostics"): continue
            tag = ref["tag"]
            stage = next(k for k in stages if tag.endswith("_" + k))
            value = stages[stage][tag[:-len(stage)-1]]
            diff = v.workbook_difference(ref["reference"], value)
            assert diff <= ref["tolerance"], (c, tag, value)
            preprocessing_rows.append(dict(configuration=c, tag=tag, computed=value,
                                           reference=ref["reference"], tolerance=ref["tolerance"]))
        runs = []
        for merged in (False, True):
            force2d = c in "AB"
            mode = ("2.5D" if force2d else "3D") + (", merged" if merged else ", direction-merged" if force2d else ", averaged")
            p = parameters(c, v.settings(c, False, force2d, merged))
            if merged:
                p["featureClass"] = {k: [] for k in ("glcm", "glrlm")}
            name = f"ibsi_preprocessed_{c}_{mode.replace(', ', '_')}"
            path = folder / f"{name}.yaml"
            path.write_text(yaml.safe_dump(p, sort_keys=False))
            ext = _extractor(SimpleNamespace(radiomics_params_file=path))
            result = ext.execute(image, mask)
            values = {k: float(value) for k, value in result.items() if k.startswith("original_")}
            baseline = json.loads((v.OUT / "raw" / f"{name}.json").read_text())["values"]
            assert values.keys() == baseline.keys(), (c, mode, values.keys() ^ baseline.keys())
            for key, value in values.items():
                assert struct.pack("!d", value) == struct.pack("!d", baseline[key]), (c, mode, key, value, baseline[key])
                feature_rows.append(dict(configuration=c, mode=mode, feature=key, value=value,
                                         benchmark_value=baseline[key], bit_identical=True))
            runs.append({"mode": mode, "values": values, "config_path": str(path.relative_to(folder))})
        rows = [v.row_result(ref, mapping, runs, "packaged_ibsi", c) for ref in refs[c]
                if not ref["family"].startswith("Diagnostics")]
        comparison_rows.extend(rows)
        within = sum(r["status"] == "within tolerance" for r in rows)
        outside = sum(r["status"] == "outside tolerance" for r in rows)
        summary[c] = dict(within_tolerance=within, computed_comparison_pairs=within+outside,
                          outside_tolerance=outside, feature_outputs=sum(len(r["values"]) for r in runs),
                          bit_identical_to_benchmark=True)
        print(c, summary[c], flush=True)
    v.write_csv(folder / "preprocessing_diagnostics.csv", preprocessing_rows)
    v.write_csv(folder / "feature_equivalence.csv", feature_rows)
    v.write_csv(folder / "feature_conformance.csv", comparison_rows)
    return dict(configurations=summary, preprocessing_diagnostics=len(preprocessing_rows),
                preprocessing_within_tolerance=len(preprocessing_rows),
                feature_outputs=len(feature_rows), all_feature_outputs_bit_identical=True,
                public_source_hashes_verified=len(retained_receipt["source_files"]),
                retained_raw_hashes_verified=len(retained_receipt["raw_output_hashes"]),
                reference_receipt_sha256=hashlib.sha256((v.OUT / "run_receipt.json").read_bytes()).hexdigest(),
                public_input_sha256={r["path"]: r["sha256"] for r in retained_receipt["source_files"]},
                benchmark_scripts_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in (REPO / "validation/ibsi/scripts").glob("*.py")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {"completed_utc": None, "python": "RTPIPELINE_RADIOMICS_PYTHON", "default": check_default(),
               "synthetic": check_synthetic(args.output)}
    if not args.synthetic_only:
        receipt["ibsi"] = benchmark(args.output)
    receipt["source_sha256"] = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        REPO / "rtpipeline/eval/geometry.py", REPO / "rtpipeline/radiomics.py",
        REPO / "rtpipeline/radiomics_ct_contract.py", REPO / "rtpipeline/ibsi_preprocessing.py",
        Path(__file__))}
    receipt["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
