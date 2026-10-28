"""Rasterizer contract; also executable without pytest in the radiomics runtime."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from rtpipeline import radiomics_rasterizer as rasterizer
from rtpipeline.multiprocessing_utils import multiprocessing_subprocess_env
from rtpipeline.radiomics_conda import _conda_subprocess_env, _radiomics_env_prefix
from rtpipeline.radiomics_ct_contract import (
    RoiClassDecision, _add_feature_completeness_metadata, _rasterizer_columns,
    disposition_rows_for_arms, extract_ct_roi_arms,
)
from rtpipeline.snakemake_delegate import runtime_environment
from rtpipeline.stage_completion import execution_environment_fingerprint
from scripts.check_radiomics_environment import dependency_errors

ROOT = Path(__file__).resolve().parents[1]


def fake_cv2(version="4.8.1", pixels=23, shifted=False):
    def fill(canvas, points, value):
        np.testing.assert_array_equal(points, [[[8, 9], [8, 13], [14, 5]]])
        # Independently specified 4.8.1 occupied x coordinates, keyed by y.
        rows = {5: [14], 6: [12, 13], 7: [11, 12, 13],
                8: [9, 10, 11, 12], 9: [8, 9, 10, 11],
                10: [8, 9, 10], 11: [8, 9, 10], 12: [8, 9], 13: [8]}
        for y, xs in rows.items():
            canvas[y, xs] = value
        if pixels != 23:
            canvas[7, 13] = 0
            canvas[8, 12] = 0
        if shifted:
            canvas[0, 0], canvas[5, 14] = value, 0
        return canvas
    return SimpleNamespace(__version__=version, __file__="/env/cv2/__init__.py", fillPoly=fill)


def pinned_version(name):
    return {"opencv-python-headless": "4.8.1.78", "rt-utils": "1.2.7"}.get(name)


def decision():
    return RoiClassDecision(
        roi_class="soft_tissue_organ", primary_resegment_range_hu=(-150., 250.),
        primary_intensity_texture_disposition="allow", feature_publication_policy="publish_both_arms",
        map_version="v1", map_hash="0" * 64, map_entry_source="canonical",
        adjudication_status="governed",
    )


class RasterizerContractTests(unittest.TestCase):
    def setUp(self):
        numpy_version = patch.object(rasterizer.np, "__version__", "1.26.4")
        numpy_version.start()
        self.addCleanup(numpy_version.stop)
        self.versions = patch.object(rasterizer, "_dist_version", side_effect=pinned_version)
        self.versions.start()
        self.addCleanup(self.versions.stop)

    def test_accepts_481_reference_mask(self):
        info = rasterizer.require_radiomics_rasterizer(cv2_module=fake_cv2())
        self.assertTrue(info["pinned_convention"])
        self.assertEqual(info["triangle_pixels"], 23)
        self.assertEqual(info["triangle_hex"], rasterizer.EXPECTED_TRIANGLE_HEX)
        self.assertEqual(info["opencv_path"], "/env/cv2/__init__.py")

    def test_rejects_wrong_version_or_fill(self):
        for version, pixels, shifted in (("4.11.0", 21, False), ("4.11.0", 23, False),
                                         ("4.8.1", 21, False), ("4.8.1", 23, True)):
            with self.subTest(version=version, pixels=pixels, shifted=shifted):
                with self.assertRaisesRegex(rasterizer.RadiomicsRasterizerError, "exact 23-pixel"):
                    rasterizer.require_radiomics_rasterizer(
                        cv2_module=fake_cv2(version, pixels, shifted))

    def test_rejects_competing_providers(self):
        for provider in rasterizer.OPENCV_PROVIDERS[1:]:
            with self.subTest(provider=provider):
                with patch.object(rasterizer, "_dist_version", side_effect=lambda name: (
                    "4.8.1.78" if name == provider else pinned_version(name))):
                    with self.assertRaisesRegex(rasterizer.RadiomicsRasterizerError, "single"):
                        rasterizer.require_radiomics_rasterizer(cv2_module=fake_cv2())

    def test_rejects_rt_utils_distribution_mismatch(self):
        for version in (None, "1.2.6", "1.3.0"):
            with self.subTest(version=version):
                with patch.object(rasterizer, "_dist_version", side_effect=lambda name: (
                    version if name == "rt-utils" else pinned_version(name))):
                    with self.assertRaisesRegex(rasterizer.RadiomicsRasterizerError, "rt-utils==1.2.7"):
                        rasterizer.require_radiomics_rasterizer(cv2_module=fake_cv2())

    def test_repair_commands_name_all_providers_and_no_deps(self):
        with patch.object(rasterizer, "_dist_version", return_value=None):
            with self.assertRaises(rasterizer.RadiomicsRasterizerError) as caught:
                rasterizer.require_radiomics_rasterizer(cv2_module=fake_cv2())
        message = str(caught.exception)
        self.assertIn("python -m pip uninstall -y opencv-python opencv-python-headless "
                      "opencv-contrib-python opencv-contrib-python-headless", message)
        self.assertIn("python -m pip install --no-deps opencv-python-headless==4.8.1.78", message)

    def test_rejects_missing_provider_and_import_failure(self):
        with patch.object(rasterizer, "_dist_version", return_value=None):
            with self.assertRaises(rasterizer.RadiomicsRasterizerError):
                rasterizer.require_radiomics_rasterizer(cv2_module=fake_cv2())
        broken = fake_cv2()
        broken.fillPoly = None
        with self.assertRaisesRegex(rasterizer.RadiomicsRasterizerError, "TypeError"):
            rasterizer.require_radiomics_rasterizer(cv2_module=broken)

    def test_fingerprint_ignores_binary_path_but_changes_with_fill(self):
        first = fake_cv2()
        with patch.dict(sys.modules, {"cv2": first}):
            original = execution_environment_fingerprint()
            first.__file__ = "/other/cv2/__init__.py"
            self.assertEqual(original, execution_environment_fingerprint())
        with patch.dict(sys.modules, {"cv2": fake_cv2(version="4.11.0")}):
            self.assertNotEqual(original, execution_environment_fingerprint())
        with patch.dict(sys.modules, {"cv2": fake_cv2(pixels=21)}):
            self.assertNotEqual(original, execution_environment_fingerprint())

    def test_guard_precedes_serial_and_parallel_reader_creation(self):
        from rtpipeline import radiomics, radiomics_parallel
        inventory = SimpleNamespace(named_rois=[SimpleNamespace(
            name="CTV", structural_code=None, roi_number=1)])
        with tempfile.TemporaryDirectory() as directory:
            rs = Path(directory) / "RS.dcm"
            rs.touch()
            with patch.object(rasterizer, "require_radiomics_rasterizer",
                              side_effect=rasterizer.RadiomicsRasterizerError("wrong rasterizer")):
                with patch("rt_utils.RTStructBuilder.create_from") as create:
                    with self.assertRaisesRegex(RuntimeError, "wrong rasterizer"):
                        radiomics._rtstruct_masks(Path(directory), rs,
                                                  structural_inventory=inventory, best_effort=True, failure_outcomes=[])
                    create.assert_not_called()
                with patch.dict(radiomics_parallel._WORKER_STATE,
                                {"builders": {}, "ct_dir": directory}, clear=True):
                    with patch("rtpipeline.rtstruct_geometry.create_scoped_rtstruct") as create:
                        with self.assertRaisesRegex(RuntimeError, "wrong rasterizer"):
                            radiomics_parallel._get_builder(rs)
                        create.assert_not_called()


class EnvironmentAndProvenanceTests(unittest.TestCase):
    def test_launchers_force_nousersite_preserve_pythonpath_and_socket_hook(self):
        with patch.dict(os.environ, {"PYTHONNOUSERSITE": "0", "PYTHONPATH": "/custom/lib"}):
            environments = (
                multiprocessing_subprocess_env(), _conda_subprocess_env(),
                runtime_environment(SimpleNamespace(root_dir="/repo", python_bin="/env/bin")),
            )
        for env in environments:
            self.assertEqual(env["PYTHONNOUSERSITE"], "1")
            paths = env["PYTHONPATH"].split(os.pathsep)
            self.assertTrue(paths[0].endswith("_mp_startup"))
            self.assertIn("/custom/lib", paths)
        self.assertIn("/repo", environments[-1]["PYTHONPATH"].split(os.pathsep))
        self.assertEqual(multiprocessing_subprocess_env({"PYTHONNOUSERSITE": "0"})["PYTHONNOUSERSITE"], "1")

    def test_segmentation_and_custom_model_launchers_isolate_overrides(self):
        from rtpipeline.segmentation import _run, _run_vec
        from rtpipeline.segmenters.totalsegmentator import run_totalsegmentator_prediction
        custom_env = {"PYTHONNOUSERSITE": "0", "PYTHONPATH": "/model/lib"}
        with patch("rtpipeline.segmentation.subprocess.run") as run:
            self.assertTrue(_run_vec(["python", "-c", "pass"], env=custom_env))
            self.assertEqual(run.call_args.kwargs["env"]["PYTHONNOUSERSITE"], "1")
            self.assertIn("/model/lib", run.call_args.kwargs["env"]["PYTHONPATH"])
            self.assertTrue(_run("python -c pass", env=custom_env))
            self.assertEqual(run.call_args.kwargs["env"]["PYTHONNOUSERSITE"], "1")
        with tempfile.TemporaryDirectory() as directory:
            observed = []
            def runner(command, *, env):
                observed.append(env)
                return True
            run_totalsegmentator_prediction(
                input_path=Path(directory) / "ct.nii.gz", output_dir=Path(directory),
                task="total", env=custom_env, runner=runner,
            )
        self.assertEqual(observed[0]["PYTHONNOUSERSITE"], "1")
        self.assertEqual(custom_env["PYTHONNOUSERSITE"], "0")

    def test_doctor_python_probe_isolates_user_site(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from rtpipeline.cli import _doctor
        with patch.dict(os.environ, {"PYTHONNOUSERSITE": "0"}):
            with patch("subprocess.check_output", return_value=b"test diagnostics") as run:
                with redirect_stdout(StringIO()):
                    self.assertEqual(_doctor([]), 0)
        python_call = next(call for call in run.call_args_list if call.args[0][0] == "python")
        self.assertEqual(python_call.kwargs["env"]["PYTHONNOUSERSITE"], "1")
        self.assertTrue(python_call.kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0].endswith("_mp_startup"))

    def test_dependency_check_waives_only_known_headless_substitution(self):
        known = "rt-utils 1.2.7 requires opencv-python, which is not installed.\n"
        result = lambda code, out, err="": subprocess.CompletedProcess([], code, out, err)
        self.assertEqual(dependency_errors(result(1, known)), [])
        self.assertEqual(dependency_errors(result(0, "No broken requirements found.")), [])
        for candidate in (result(1, known + "rt-utils 1.2.7 requires dataclasses, which is not installed."),
                          result(2, known), result(1, ""), result(1, known, "pip failed")):
            self.assertTrue(dependency_errors(candidate))

    def test_environment_check_prints_guard_repair_commands(self):
        from contextlib import redirect_stderr
        from io import StringIO
        from scripts import check_radiomics_environment as checker
        with patch.object(rasterizer, "_dist_version", return_value=None):
            try:
                rasterizer.require_radiomics_rasterizer(cv2_module=fake_cv2())
            except rasterizer.RadiomicsRasterizerError as exc:
                error = exc
        output = StringIO()
        with patch.object(checker, "require_radiomics_rasterizer", side_effect=error):
            with redirect_stderr(output):
                self.assertEqual(checker.main(), 1)
        self.assertIn("python -m pip uninstall -y opencv-python opencv-python-headless "
                      "opencv-contrib-python opencv-contrib-python-headless\n", output.getvalue())
        self.assertIn("python -m pip install --no-deps opencv-python-headless==4.8.1.78\n",
                      output.getvalue())

    def test_recorded_columns_survive_preparer_feature_runtime_boundary(self):
        from rtpipeline.radiomics_schema import RADIOMICS_TEXT_COLUMNS
        cols = _rasterizer_columns()
        self.assertTrue(set(cols).issubset(RADIOMICS_TEXT_COLUMNS))
        self.assertEqual(cols["rasterizer_backend"], "rt-utils/cv2.fillPoly")
        for key in ("rasterizer_opencv_version", "rasterizer_opencv_path", "rasterizer_rt_utils_version"):
            self.assertIn(key, cols)
        supplied = {**cols, "rasterizer_opencv_path": "/mask/preparer/cv2/__init__.py"}
        args = dict(decision=decision(), run_identifier="run-1", code_revision="rev-1",
                    native_voxel_count=100, required=False,
                    configured_parameter_hashes={"primary_resegmented": "c" * 64, "sensitivity_raw": "d" * 64})
        rows = disposition_rows_for_arms(
            supplied, disposition="declared_skip", detail="test", failure_kind="declared_ineligible",
            effective_hashes={"primary_resegmented": "a" * 64, "sensitivity_raw": "b" * 64}, **args)
        table = _add_feature_completeness_metadata(pd.DataFrame(rows))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "radiomics_ct.parquet"
            table.to_parquet(output, index=False)
            saved = pd.read_parquet(output)
        for key, value in supplied.items():
            self.assertEqual(saved[key].tolist(), [value, value])
        with patch("rtpipeline.radiomics_ct_contract._extract_ct_roi_arms_impl",
                   return_value=[dict(cols), dict(cols)]):
            extracted = extract_ct_roi_arms(None, None, factory=None, common_metadata=supplied, **args)
        for row in extracted:
            self.assertEqual(row["rasterizer_opencv_path"], supplied["rasterizer_opencv_path"])

    def test_robustness_sidecar_and_receipt_preserve_measurement_runtime(self):
        from rtpipeline import radiomics_robustness as rr
        from rtpipeline.robustness_completion import (
            RobustnessCompletionError, _validate_payload, robustness_completion_payload,
        )
        from rtpipeline.stage_completion import execution_environment_fingerprint
        with tempfile.TemporaryDirectory() as directory:
            course = Path(directory) / "P1" / "C1"
            course.mkdir(parents=True)
            output = course / "radiomics_robustness_ct.parquet"
            output.write_bytes(b"test measurement bytes")
            path = rr._write_robustness_source_dispositions(
                course, run_identifier="run-1", rows=[], source_bindings=[],
                effective_configuration={}, code_identity=rr._capture_robustness_code_identity(),
                output_path=output, measured_output=output,
            )
            source = json.loads(path.read_text())
            self.assertEqual(source["rasterizer"], rasterizer.active_radiomics_rasterizer())
            self.assertEqual(source["environment_fingerprint"], execution_environment_fingerprint())
            source["rasterizer"]["opencv_path"] = "/measurement/runtime/cv2/__init__.py"
            source["environment_fingerprint"] = "a" * 64
            path.write_text(json.dumps(source))
            receipt = robustness_completion_payload(
                course, patient_id="P1", course_id="C1", run_identifier="run-1",
                measurement_outcome="measured", output_name=output.name,
                dispositions_path=path, measured_output=output, source_disposition_count=0,
                effective_configuration_sha256=source["effective_configuration_sha256"],
            )
            self.assertEqual(receipt["rasterizer"], source["rasterizer"])
            self.assertEqual(receipt["environment_fingerprint"], "a" * 64)
            sentinel = course / ".radiomics_robustness_done"
            _validate_payload(receipt, sentinel_path=sentinel, course_dir=course)
            receipt["rasterizer"]["opencv_path"] = "/wrong/cv2/__init__.py"
            with self.assertRaisesRegex(RobustnessCompletionError, "rasterizer differs"):
                _validate_payload(receipt, sentinel_path=sentinel, course_dir=course)

    def test_missing_historical_provenance_is_not_invented_by_publisher(self):
        table = _add_feature_completeness_metadata(pd.DataFrame([{}]))
        for key in _rasterizer_columns():
            self.assertEqual(table.iloc[0][key], "unrecorded")

    def test_installed_radiomics_process_isolates_user_site_and_enforces_contract(self):
        prefix = _radiomics_env_prefix()
        if prefix is None:
            self.skipTest("radiomics conda environment (rtpipeline-radiomics) is not installed")
        python = Path(prefix) / "bin" / "python"
        if not python.is_file():
            self.skipTest(f"radiomics Python is absent: {python}")
        script = '''import json, os, site
from rtpipeline.radiomics_rasterizer import active_radiomics_rasterizer, require_radiomics_rasterizer
info = active_radiomics_rasterizer()
try:
    require_radiomics_rasterizer()
    rejected = False
except RuntimeError:
    rejected = True
print(json.dumps(dict(info=info, rejected=rejected, user_site=site.ENABLE_USER_SITE,
                     nousersite=os.environ.get("PYTHONNOUSERSITE"))))
'''
        env = _conda_subprocess_env()
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
        result = subprocess.run([str(python), "-c", script], env=env, check=True,
                                capture_output=True, text=True, timeout=60)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["user_site"])
        self.assertEqual(payload["nousersite"], "1")
        info = payload["info"]
        self.assertTrue(info["opencv_path"])
        supported = (info["pinned_convention"]
                     and info["rt_utils_version"] == "1.2.7"
                     and info["providers"] == {"opencv-python-headless": "4.8.1.78"})
        self.assertEqual(payload["rejected"], not supported)


if __name__ == "__main__":
    unittest.main(verbosity=2)
