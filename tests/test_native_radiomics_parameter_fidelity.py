"""ROI size or name must never change the configured native/isolated CT method.

Two mutation sites are bound here: ``rtpipeline.radiomics._extractor_large_roi``
(native CT effective-hash and measurement factories, parallel worker init and
``_extract_one``) and the ``_factory`` closure inside
``rtpipeline.radiomics_parallel._isolated_radiomics_extraction`` (robustness).

Mocked controls prove the wrappers do not mutate a configured extractor. The
actual-helper test runs real PyRadiomics 3.0.1 on a tiny generated NRRD pair in
the NumPy 1.x conda helper environment through a pytest-free probe and
reconciles the numbers independently. No clinical data, service, or pipeline
launch is used. Disclosed mock: ``_apply_thread_limit`` is neutralised in the
mocked tests so they do not rewrite the host thread environment.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

import rtpipeline
import rtpipeline.radiomics as native
import rtpipeline.radiomics_conda as rc
import rtpipeline.radiomics_ct_contract as ct_contract
import rtpipeline.radiomics_parallel as parallel
import rtpipeline.radiomics_robustness_outcomes as outcomes

SPACING = (1.25, 1.5, 2.5)
WINDOW = (0.0, 1000.0)
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
HELPER = Path(__file__).with_name("native_radiomics_parameter_fidelity_helper.py")
CONDA_ENV = "rtpipeline-radiomics"
_CONDA_EXE_AVAILABLE = bool(shutil.which(rc.CONDA_EXE) or Path(rc.CONDA_EXE).is_file())
requires_radiomics_env = pytest.mark.skipif(
    not _CONDA_EXE_AVAILABLE,
    reason=f"conda-compatible executable {rc.CONDA_EXE!r} is not available",
)


class _ConfiguredExtractor:
    """Records every method-changing call a wrapper could make."""

    def __init__(self, params_file=None):
        self.params_file = params_file
        self.settings = {"resampledPixelSpacing": list(SPACING), "binWidth": 25, "label": 1}
        self.enabledImagetypes = {"Original": {}, "Square": {}}
        self.enabledFeatures = {"firstorder": ["Mean"], "shape": ["VoxelVolume"], "glcm": ["JointEntropy"]}
        self.calls = []

    def disableAllImageTypes(self):
        self.calls.append("disableAllImageTypes"); self.enabledImagetypes = {}

    def enableImageTypeByName(self, name, *_a, **_k):
        self.calls.append(f"enableImageTypeByName:{name}"); self.enabledImagetypes[name] = {}

    def disableAllFeatures(self):
        self.calls.append("disableAllFeatures"); self.enabledFeatures = {}

    def enableFeatureClassByName(self, name, *_a, **_k):
        self.calls.append(f"enableFeatureClassByName:{name}"); self.enabledFeatures[name] = []

    def state(self):
        return (dict(self.settings), dict(self.enabledImagetypes),
                {k: list(v) for k, v in self.enabledFeatures.items()})


def _expect_configured(extractor):
    assert extractor.calls == [], f"wrapper mutated the configured method: {extractor.calls}"
    assert extractor.settings["resampledPixelSpacing"] == list(SPACING)
    assert set(extractor.enabledImagetypes) == {"Original", "Square"}
    assert set(extractor.enabledFeatures) == {"firstorder", "shape", "glcm"}


# --------------------------------------------------------------------------
# mocked controls: native compatibility wrapper
# --------------------------------------------------------------------------

def test_native_large_roi_wrapper_returns_configured_extractor_unchanged(monkeypatch):
    built = []

    def fake_extractor(config, modality="CT", normalize_override=None):
        built.append((config, modality)); ext = _ConfiguredExtractor(); return ext

    monkeypatch.setattr(native, "_extractor", fake_extractor)
    config = object()
    ordinary = native._extractor(config, "CT")
    marked = native._extractor_large_roi(config, "CT")
    _expect_configured(marked)
    assert marked.state() == ordinary.state()
    assert built == [(config, "CT"), (config, "CT")]


def test_native_large_roi_wrapper_preserves_unavailable_extractor(monkeypatch):
    monkeypatch.setattr(native, "_extractor", lambda *_a, **_k: None)
    assert native._extractor_large_roi(object(), "CT") is None


# --------------------------------------------------------------------------
# mocked controls: isolated robustness factory
# --------------------------------------------------------------------------

def _install_fake_pyradiomics(monkeypatch):
    package = types.ModuleType("radiomics")
    featureextractor = types.ModuleType("radiomics.featureextractor")
    featureextractor.RadiomicsFeatureExtractor = _ConfiguredExtractor
    glcm = types.ModuleType("radiomics.glcm")

    class RadiomicsGLCM:  # MCC contraction installs onto this attribute only.
        getMCCFeatureValue = None

    glcm.RadiomicsGLCM = RadiomicsGLCM
    package.featureextractor = featureextractor
    package.glcm = glcm
    for name, module in (("radiomics", package), ("radiomics.featureextractor", featureextractor),
                         ("radiomics.glcm", glcm)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(parallel, "_apply_thread_limit", lambda *_a, **_k: None)


def _task_params(roi_name, large_roi, params_file="/synthetic/params.yaml"):
    decision = dict(roi_class="synthetic", map_version="fixture", map_hash="0" * 64,
                    map_entry_source="fixture", adjudication_status="fixture",
                    primary_resegment_range_hu=list(WINDOW),
                    primary_intensity_texture_disposition="extract",
                    feature_publication_policy="extract")
    return {
        "image_path": "/synthetic/img.nrrd", "mask_path": "/synthetic/mask.nrrd",
        "segmentation_source": "Synthetic", "roi_name": roi_name, "patient_id": "P0",
        "course_id": "C0", "series_uid": "1.2.3", "mask_identity": "sha256:0",
        "roi_original_name": roi_name, "stable_roi_identifier": "fixture",
        "large_roi": large_roi, "params_file": params_file, "dual_arm_ct": True,
        "roi_class_decision": decision, "run_identifier": "run-fixture",
        "code_revision": "code-fixture", "native_voxel_count": 100,
        "configured_parameter_hashes": {"primary_resegmented": "a" * 64, "sensitivity_raw": "b" * 64},
        "measurement_type": "robustness", "perturbed_mask_identity": "sha256:1",
        "extra_metadata": {"perturbation_id": "baseline"},
    }


@pytest.mark.parametrize("roi_name,large_roi", [("small", False), ("small", True), ("BODY", True)])
def test_isolated_factory_keeps_configured_method_for_any_flag_or_name(monkeypatch, roi_name, large_roi):
    _install_fake_pyradiomics(monkeypatch)
    captured = {}

    def fake_extract(image, mask, *, factory, decision, **kwargs):
        captured["factory"] = factory
        return [{"extraction_arm": "primary_resegmented", "extraction_status": "success"},
                {"extraction_arm": "sensitivity_raw", "extraction_status": "success"}]

    monkeypatch.setattr(ct_contract, "extract_ct_roi_arms", fake_extract)
    result = parallel._isolated_radiomics_extraction(("/synthetic/mask.nrrd", _task_params(roi_name, large_roi)))
    assert "__records__" in result and len(result["__records__"]) == 2
    first, second = captured["factory"](), captured["factory"]()
    _expect_configured(first)
    assert first.params_file == "/synthetic/params.yaml"
    assert first.state() == second.state()


def test_isolated_factory_without_params_file_uses_library_defaults_unpruned(monkeypatch):
    _install_fake_pyradiomics(monkeypatch)
    captured = {}
    monkeypatch.setattr(ct_contract, "extract_ct_roi_arms",
                        lambda image, mask, *, factory, **k: captured.setdefault("factory", factory) and [])
    parallel._isolated_radiomics_extraction(("/synthetic/mask.nrrd", _task_params("BODY", True, params_file=None)))
    extractor = captured["factory"]()
    assert extractor.params_file is None
    _expect_configured(extractor)


def test_isolated_large_flag_preserves_geometric_nonmeasurement_semantics(monkeypatch):
    """A resource/geometry rejection still becomes a non-measurement with the unchanged factory."""
    _install_fake_pyradiomics(monkeypatch)
    seen = {}
    monkeypatch.setattr(ct_contract, "extract_ct_roi_arms", lambda image, mask, *, factory, **k: [])
    monkeypatch.setattr(outcomes, "returned_geometry_nonmeasurement",
                        lambda records, image, mask, factory: seen.setdefault("factory", factory) and
                        types.SimpleNamespace(reason_code="degenerate_after_resampling", evidence={"count": 0}))
    result = parallel._isolated_radiomics_extraction(("/synthetic/mask.nrrd", _task_params("BODY", True)))
    rows = result["__nonmeasurement_rows__"]
    assert [row["extraction_arm"] for row in rows] == ["primary_resegmented", "sensitivity_raw"]
    assert {row["robustness_status"] for row in rows} == {"geometrically_impossible"}
    assert all(np.isnan(row["value"]) for row in rows)
    _expect_configured(seen["factory"]())


# --------------------------------------------------------------------------
# actual helper: real PyRadiomics numbers in the NumPy 1.x environment
# --------------------------------------------------------------------------

@pytest.fixture
def generated_case(tmp_path):
    z, y, x = np.indices((10, 12, 12))
    array = (100 + 7 * z + 3 * y + 11 * x).astype(np.float32)
    mask = np.zeros(array.shape, dtype=np.uint8)
    mask[2:6, 3:8, 4:9] = 1
    paths = {}
    for name, data in (("image", array), ("mask", mask)):
        image = sitk.GetImageFromArray(data)
        image.SetSpacing(SPACING)
        paths[name] = tmp_path / f"{name}.nrrd"
        sitk.WriteImage(image, str(paths[name]))
    params = tmp_path / "configured.yaml"
    params.write_text(PARAMETERS, encoding="utf-8")
    return paths["image"], paths["mask"], params, array[mask == 1]


def _expected_feature_names():
    names = {f"{image}_{family}_{feature}" for image in ("original", "square")
             for family, feats in (("firstorder", ("Mean", "Minimum", "Maximum", "Range")),
                                   ("glcm", ("JointEntropy",))) for feature in feats}
    names.add("original_shape_VoxelVolume")
    return names


def _reconcile(features, voxels, *, full_shape_class=False):
    """Independent reconciliation. The CT contract's separate morphological
    shape-only arm enables the whole shape class by design; the isolated rows
    therefore carry every shape feature, while the intensity/texture set must be
    exactly the configured Original+Square firstorder/GLCM selection."""
    observed = set(features)
    if full_shape_class:
        assert {n for n in observed if "_shape_" not in n} == {
            n for n in _expected_feature_names() if "_shape_" not in n}, sorted(observed)
        assert "original_shape_VoxelVolume" in observed
        assert not any(n.startswith("square_shape_") for n in observed)
    else:
        assert observed == _expected_feature_names(), sorted(observed)
    assert all(v is not None and np.isfinite(v) for v in features.values())
    assert features["original_firstorder_Mean"] == pytest.approx(float(voxels.mean()))
    assert features["original_firstorder_Minimum"] == pytest.approx(float(voxels.min()))
    assert features["original_firstorder_Maximum"] == pytest.approx(float(voxels.max()))
    assert features["original_firstorder_Range"] == pytest.approx(float(np.ptp(voxels)))
    assert features["original_shape_VoxelVolume"] == pytest.approx(len(voxels) * float(np.prod(SPACING)))
    assert features["original_glcm_JointEntropy"] > 0
    assert features["square_firstorder_Mean"] != pytest.approx(features["original_firstorder_Mean"])


@requires_radiomics_env
def test_actual_helper_native_and_isolated_factories_keep_configured_method(generated_case, tmp_path, monkeypatch):
    image, mask, params, voxels = generated_case
    spec = tmp_path / "spec.json"
    out = tmp_path / "report.json"
    spec.write_text(json.dumps({
        "image": str(image), "mask": str(mask), "params": str(params), "out": str(out),
        "window": list(WINDOW), "voxel_count": int(len(voxels)),
    }), encoding="utf-8")
    for key in rc._THREAD_ENV_VARS:
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("RTPIPELINE_RADIOMICS_THREAD_LIMIT", "1")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", str(Path(rtpipeline.__file__).resolve().parents[1]))
    command = [rc.CONDA_EXE, "run", "-n", CONDA_ENV, "python", "-c",
               HELPER.read_text(encoding="utf-8"), str(spec)]
    completed = subprocess.run(command, cwd=str(tmp_path), env=rc._conda_subprocess_env(),
                               capture_output=True, text=True, timeout=600)
    print("helper stdout:", completed.stdout[-2000:])
    print("helper stderr:", completed.stderr[-4000:])
    assert completed.returncode == 0, completed.stderr[-4000:]
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["versions"]["numpy"].startswith("1.")
    assert report["versions"]["pyradiomics"].lstrip("v") == "3.0.1"
    print("helper versions", report["versions"])

    ordinary, marked = report["native"]["ordinary"], report["native"]["large"]
    assert marked["state"] == ordinary["state"]
    assert marked["state"]["settings"]["resampledPixelSpacing"] == list(SPACING)
    assert marked["state"]["image_types"] == ["Original", "Square"]
    assert marked["state"]["features"]["glcm"] == ["JointEntropy"]
    assert marked["interpolated_spacing"] == list(SPACING)
    _reconcile(ordinary["features"], voxels)
    _reconcile(marked["features"], voxels)
    assert marked["features"] == pytest.approx(ordinary["features"])
    assert set(marked["effective_hashes"]) == {"primary_resegmented", "sensitivity_raw"}
    assert marked["effective_hashes"] == ordinary["effective_hashes"]
    print("native numerical reconciliation", marked["features"])
    print("native effective arm hashes", marked["effective_hashes"])

    isolated = report["isolated"]
    assert set(isolated) == {"small|large_roi=False", "small|large_roi=True",
                             "BODY|large_roi=False", "BODY|large_roi=True"}
    baseline = isolated["small|large_roi=False"]
    assert baseline["nonmeasurement"] is False
    for key, entry in isolated.items():
        assert entry["nonmeasurement"] is False, key
        assert set(entry["arms"]) == {"primary_resegmented", "sensitivity_raw"}, key
        for arm, row in entry["arms"].items():
            assert row["extraction_status"] == "success", (key, arm)
            assert row["intensity_texture_disposition"] == "success", (key, arm)
            _reconcile(row["features"], voxels, full_shape_class=True)
            assert row["features"] == pytest.approx(baseline["arms"][arm]["features"]), (key, arm)
            assert row["effective_parameter_hash"] == baseline["arms"][arm]["effective_parameter_hash"], (key, arm)
            assert len(row["effective_parameter_hash"]) == 64
    # The configured-hash schema already ignores the runtime flag; keep that invariant.
    for arm in ("primary_resegmented", "sensitivity_raw"):
        assert {entry["arms"][arm]["configured_parameter_hash"] for entry in isolated.values()} == {
            baseline["arms"][arm]["configured_parameter_hash"]}
    print("isolated effective hashes", {k: {a: r["effective_parameter_hash"] for a, r in v["arms"].items()}
                                        for k, v in isolated.items()})
