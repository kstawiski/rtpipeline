"""Synthetic real-Snakefile organize/custom/DVH interruption and binding checks.

No model inference, network, clinical fixtures, container or cohort is used.
All runtime source copies, caches and generated DICOMs live below tmp_path.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pandas as pd
import pydicom
import pytest
import yaml
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid

from synthetic_ct_series import write_ct_series
from synthetic_rt_fixtures import make_struct, make_plan, make_dose, make_record

ROOT = Path(__file__).resolve().parents[1]


def synthetic_course(root):
    """One fully linked, multi-plane, three-ROI course with two targets."""
    root.mkdir(parents=True)
    paths = write_ct_series(root / "images", [float(i * 5) for i in range(12)], signed=False,
                            rows=24, columns=24)
    images = []
    first = pydicom.dcmread(paths[0])
    for path in paths:
        ds = pydicom.dcmread(path)
        ds.ImagePositionPatient = [-10., -10., float(ds.ImagePositionPatient[2])]
        ds.PixelSpacing = [2., 2.]
        ds.SliceThickness = 5.
        ds.save_as(path, enforce_file_format=True)
        ref = Dataset()
        ref.ReferencedSOPClassUID, ref.ReferencedSOPInstanceUID = ds.SOPClassUID, ds.SOPInstanceUID
        images.append(ref)
    common = dict(study_uid=str(first.StudyInstanceUID), frame_uid=str(first.FrameOfReferenceUID),
                  patient_id="SYNTHETIC")
    struct_uid, plan_uid, dose_uid = [generate_uid() for _ in range(3)]
    path = make_struct(root / "struct.dcm", struct_uid, roi_names=("PTV1", "PTV2", "Organ"), **common)
    rs = pydicom.dcmread(path)
    ref = Dataset()
    ref.SeriesInstanceUID = first.SeriesInstanceUID
    ref.ContourImageSequence = images
    rs.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence = [ref]
    for index, roi in enumerate(rs.ROIContourSequence):
        template = roi.ContourSequence[0]
        roi.ContourSequence = []
        x0, x1 = (-4., 8.) if index != 1 else (10., 22.)
        for zindex in (1, 2, 3):
            contour = copy.deepcopy(template)
            z = float(zindex * 5)
            contour.ContourData = [x0, -2., z, x1, -2., z, x1, 18., z, x0, 18., z]
            contour.ContourImageSequence = [images[zindex]]
            roi.ContourSequence.append(contour)
    rs.save_as(path, enforce_file_format=True)
    make_plan(root / "plan.dcm", plan_uid, struct_uid=struct_uid, fractions=3, rx_gy=6., **common)
    path = make_dose(root / "dose.dcm", dose_uid, plan_uid=plan_uid, with_pixels=True,
                     dose_gy=6., shape=(12, 24, 24), **common)
    dose = pydicom.dcmread(path)
    dose.PixelSpacing = [2., 2.]
    dose.save_as(path, enforce_file_format=True)
    for index in range(3):
        make_record(root / f"record{index}.dcm", plan_uid, patient_id="SYNTHETIC",
                    fraction_number=index + 1, date="20240102")


def source_copy(root):
    root.mkdir()
    for name in ("rtpipeline", "workflow", "envs"):
        shutil.copytree(ROOT / name, root / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("Snakefile", "config.yaml"):
        shutil.copy2(ROOT / name, root / name)
    return root


def instrument(root, runtime):
    root.mkdir()
    # This hook only observes real sampler calls and pauses at publication seams.
    # The parent test sends SIGKILL to its own new process group after the file
    # handshake; no guessed timing determines the crash boundary.
    (root / "sitecustomize.py").write_text('''
import os, sys, json, time, hashlib

def guard(event, args):
    # Refuse network access; allow local AF_UNIX sockets, which multiprocessing's
    # forkserver and resource tracker use to start process-pool workers.
    if event == "socket.getaddrinfo":
        raise RuntimeError("synthetic tests forbid network")
    if event in {"socket.connect", "socket.connect_ex", "socket.bind"}:
        import socket as _socket
        if getattr(args[0], "family", None) != _socket.AF_UNIX:
            raise RuntimeError("synthetic tests forbid network")
sys.addaudithook(guard)
if sys.prefix == os.environ.get("WS515_RUNTIME_PREFIX"):
    for path in json.loads(os.environ["WS515_RUNTIME_SITES"]):
        if path not in sys.path: sys.path.append(path)
    import rtpipeline.dvh_generation as gen
    def pause(name):
        if name == os.environ.get("WS515_KILL_POINT"):
            path = os.environ["WS515_HANDSHAKE"]
            with open(path, "w") as f:
                f.write(name); f.flush(); os.fsync(f.fileno())
            while True: time.sleep(.1)
    gen._failpoint = pause
    import rtpipeline.dvh as dvh
    original = dvh.dvhcalc.get_dvh
    def counted(*args, **kwargs):
        path = os.environ["WS515_CALLS"]
        with open(path, "a") as f:
            key = hashlib.sha256((str(getattr(args[0], "SOPInstanceUID", args[0])) + ":" +
                                  str(args[2] if len(args)>2 else kwargs.get("roi"))).encode()).hexdigest()
            f.write(json.dumps({"key": key})+"\\n")
            f.flush(); os.fsync(f.fileno())
        return original(*args, **kwargs)
    dvh.dvhcalc.get_dvh = counted
''')
    return root


class Production:
    def __init__(self, root):
        self.root = root
        self.source = source_copy(root / "source")
        self.runtime = Path(sys.executable)
        self.snakemake = os.environ.get("RTPIPELINE_TEST_SNAKEMAKE") or shutil.which("snakemake")
        if not self.snakemake:
            pytest.skip("installed Snakemake executable required")
        self.output = root / "Output"
        self.config = root / "synthetic.yaml"
        self.custom = root / "custom.yaml"
        self.custom.write_text(yaml.safe_dump({"custom_structures": [{"name": "Target_union",
            "operation": "union", "source_structures": ["PTV1", "PTV2"]}]}))
        synthetic_course(root / "Input")
        self.config.write_text(yaml.safe_dump(dict(dicom_root=str(root / "Input"),
            output_dir=str(self.output), logs_dir=str(root / "Logs"),
            environments={"main": str(self.runtime.parent.parent)}, max_workers=1,
            segmentation={"enabled": False, "device": "cpu"}, custom_models={"enabled": False},
            rtsegmentator={"enabled": False}, ct_cropping={"enabled": False, "use_cropped_for_dvh": False},
            radiomics={"enabled": False}, radiomics_robustness={"enabled": False},
            custom_structures=str(self.custom), campaign_mode=False)))
        self.cores = 1
        self.calls = root / "calls.jsonl"
        self.handshake = root / "handshake"
        hook = instrument(root / "instrumentation", self.runtime)
        self.env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "RTPIPELINE_CODE_REVISION": "synthetic-test",
                    "PYTHONPATH": os.pathsep.join((str(hook), str(self.source))),
                    "PATH": os.pathsep.join((str(self.runtime.parent),
                        "/home/konrad/micromamba/envs/rtpipeline/bin", os.environ["PATH"])),
                    "WS515_RUNTIME_PREFIX": str(self.runtime.parent.parent),
                    "WS515_RUNTIME_SITES": json.dumps([p for p in sys.path if p.endswith("site-packages") or (p and (Path(p) / "xlsxwriter").is_dir())]),
                    "WS515_HANDSHAKE": str(self.handshake), "WS515_CALLS": str(self.calls),
                    "XDG_CACHE_HOME": str(root / "cache"), "MPLCONFIGDIR": str(root / "mpl"),
                    "NUMBA_CACHE_DIR": str(root / "numba"), "TMPDIR": str(root / "tmp"),
                    "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1"}
        (root / "tmp").mkdir()

    def command(self, target):
        return [str(self.snakemake), "--snakefile", str(self.source / "Snakefile"),
                "--directory", str(self.source), "--configfile", str(self.config),
                "--cores", str(self.cores), "--rerun-incomplete", str(target)]

    def run(self, target, *, kill=None):
        self.handshake.unlink(missing_ok=True)
        env = dict(self.env, WS515_KILL_POINT=kill or "")
        log = self.root / ("snake-" + (kill or "run") + ".log")
        with log.open("a") as stream:
            process = subprocess.Popen(self.command(target), env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                if kill:
                    deadline = time.monotonic() + 180
                    while not self.handshake.exists() and process.poll() is None and time.monotonic() < deadline:
                        time.sleep(.05)
                    if not self.handshake.exists():
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        pytest.fail(log.read_text()[-14000:])
                    os.killpg(process.pid, signal.SIGKILL)
                    assert process.wait(timeout=30) == -signal.SIGKILL
                    # Every descendant shares this task-owned process group. SIGKILL
                    # has been delivered before restarting; wait for live members.
                    deadline = time.monotonic() + 10
                    while time.monotonic() < deadline:
                        members = subprocess.run(["ps", "-eo", "pgid=,stat="], capture_output=True, text=True).stdout
                        if not any(line.split()[0] == str(process.pid) and not line.split()[1].startswith("Z")
                                   for line in members.splitlines() if line.split()):
                            break
                        time.sleep(.05)
                    else:
                        pytest.fail("owned descendants survived SIGKILL")
                    locks = self.source / ".snakemake/locks"
                    if locks.exists():
                        for path in locks.iterdir(): path.unlink()
                else:
                    assert process.wait(timeout=getattr(self, "run_timeout", 240)) == 0, log.read_text()[-18000:]
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)
        return log

    def organize(self):
        self.run(self.output / "_COURSES/manifest.json")
        manifest = json.loads((self.output / "_COURSES/manifest.json").read_text())
        courses = manifest["courses"]
        assert len(courses) == 1, manifest
        self.course = Path(courses[0]["path"])
        return self.course

    def count(self):
        return len(self.calls.read_text().splitlines()) if self.calls.exists() else 0


def snapshot(course):
    from rtpipeline.dvh_generation import ARTIFACTS, current
    marker = current(course)
    artifacts = {name: (course / name).read_bytes() for name in (*ARTIFACTS, ".dvh_done")
                 if (course / name).is_file()}
    artifacts["ledger.json"] = (course / ".dvh/current/ledger.json").read_bytes()
    return artifacts, marker


@pytest.mark.integration
@pytest.mark.requires_snakemake
def test_real_production_interrupt_resume(tmp_path):
    pipeline = Production(tmp_path)
    course = pipeline.organize()
    # Store the identical organized starting point, preserving every input byte.
    clean = tmp_path / "organized"
    shutil.copytree(pipeline.output, clean, symlinks=True)
    target = course / ".dvh_done"
    pipeline.run(target)
    baseline, marker = snapshot(course)
    control_artifacts = tmp_path / "control-artifacts"
    for name, raw in baseline.items():
        path = control_artifacts / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    frame = pd.read_parquet(course / "dvh_metrics.parquet")
    assert {"PTV1", "PTV2", "Organ", "Target_union"} <= set(frame.ROI_Name)
    assert frame.DmeanGy.notna().sum() >= 4
    baseline_calls = pipeline.count()
    assert baseline_calls > 0
    evidence = {"control_calls": baseline_calls, "rows": len(frame), "artifacts": {
        name: hashlib.sha256(raw).hexdigest() for name, raw in baseline.items()}, "legs": {}}
    count = pipeline.count()
    pipeline.run(target)
    assert pipeline.count() == count
    assert snapshot(course)[0] == baseline
    for point in ("staging", "published_before_commit", "committed_before_activation", "committed_before_ledger"):
        shutil.rmtree(pipeline.output)
        shutil.copytree(clean, pipeline.output, symlinks=True)
        # Snakemake metadata belongs to the same fixture; incomplete jobs are
        # resumed with --rerun-incomplete and no forced scientific execution.
        before = pipeline.count()
        pipeline.run(target, kill=point)
        after_kill = pipeline.count()
        pipeline.run(target)
        after_resume = pipeline.count()
        observed, resumed_marker = snapshot(course)
        assert observed == baseline, {name: (hashlib.sha256(baseline[name]).hexdigest(),
            hashlib.sha256(observed[name]).hexdigest()) for name in baseline if baseline[name] != observed.get(name)}
        assert resumed_marker == marker
        records = list((pipeline.output / "_campaign_ledger/records").glob("*__dvh.json"))
        assert len(records) == 1
        ledger = json.loads(records[0].read_text())
        assert ledger["generation_id"] == marker["generation_id"]
        assert ledger["attempt_state"] == "committed"
        from rtpipeline.durable_progress import Binding, ProgressJournal, ResultCache
        from rtpipeline.dvh_generation import verify
        journal = [entry["body"] for entry in ProgressJournal.read(course / ".dvh/progress.jsonl").records]
        keys = [entry["item_key"] for entry in journal if entry["event"] == "roi_committed"]
        assert len(keys) == len(set(keys)) == baseline_calls
        loaded = ResultCache(course / ".dvh/roi-cache").load_valid(Binding(**resumed_marker["binding"]))
        assert set(loaded.entries) == set(keys) and loaded.report["valid"] == baseline_calls
        generations = list((course / ".dvh/generations").glob("*/commit.json"))
        for committed in generations:
            verify(committed.parent, Binding(**resumed_marker["binding"]))
        expected_discard = {"staging": "DVH_INCOMPLETE_STAGING",
                            "published_before_commit": "DVH_INCOMPLETE_GENERATION"}.get(point)
        if expected_discard:
            assert any(entry.get("reason_code") == expected_discard for entry in journal)
        if point == "committed_before_activation":
            assert any(entry["event"] == "recovered_commit" for entry in journal)
        sample_keys = [json.loads(line)["key"] for line in pipeline.calls.read_text().splitlines()[before:after_resume]]
        assert len(sample_keys) == len(set(sample_keys)) == baseline_calls
        assert json.loads((course / ".dvh/attempt.json").read_text())["state"] == "committed"
        attempts = list((pipeline.output / "_campaign_ledger/attempts").glob("*__dvh.json"))
        assert len(attempts) == 1
        assert json.loads(attempts[0].read_text())["state"] == "committed"
        if point == "committed_before_ledger":
            assert after_resume == after_kill
        elif point == "staging":
            assert after_kill - before == 1
            assert after_resume - after_kill == baseline_calls - 1
        else:
            assert after_resume == after_kill
        pipeline.run(target)
        assert pipeline.count() == after_resume
        assert snapshot(course)[0] == baseline
        evidence["legs"][point] = dict(killed_calls=after_kill-before,
            resumed_calls=after_resume-after_kill, byte_equal=len(baseline), ledger_records=len(records),
            roi_cache_commits=len(keys), unique_sampler_keys=len(set(sample_keys)),
            verified_generations=len(generations))
    count = pipeline.count()
    scientific_mtimes = {name: (course / name).stat().st_mtime_ns
                         for name in baseline if name != "ledger.json"}
    config = yaml.safe_load(pipeline.config.read_text())
    config["max_workers"] = 2
    pipeline.config.write_text(yaml.safe_dump(config))
    pipeline.cores = 2
    pipeline.run(target)
    assert pipeline.count() == count
    assert snapshot(course)[0] == baseline
    assert {name: (course / name).stat().st_mtime_ns for name in scientific_mtimes} == scientific_mtimes
    evidence["worker_budget_only"] = {"additional_sampler_calls": 0, "byte_equal": len(baseline),
                                      "scientific_mtimes_unchanged": True}
    evidence["source_sha256"] = {name: hashlib.sha256((pipeline.source / name).read_bytes()).hexdigest()
        for name in ("Snakefile", "rtpipeline/dvh.py", "rtpipeline/dvh_generation.py",
                     "rtpipeline/custom_structures_rtstruct.py", "rtpipeline/stage_completion.py",
                     "rtpipeline/cli.py", "rtpipeline/workflow_delegate.py",
                     "workflow/scripts/campaign_ledger.py", "workflow/scripts/run_course_stage.py",
                     "rtpipeline/durable_progress.py")}
    path = os.environ.get("WS515_EVIDENCE")
    if path:
        Path(path).write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")


@pytest.mark.integration
@pytest.mark.requires_snakemake
def test_real_snakefile_reruns_legacy_upstream_and_binds_fresh_receipt(tmp_path):
    pipeline = Production(tmp_path)
    course = pipeline.organize()
    target = course / ".dvh_done"
    pipeline.run(target)
    legacy = {"status": "ok", "stage": "organize"}
    receipt = course / ".organized"
    receipt.write_text(json.dumps(legacy) + "\n")
    log = pipeline.run(target)
    assert "Re-running organize for invalid manifest course completion" in log.read_text()
    fresh = json.loads(receipt.read_text())
    assert fresh != legacy
    from rtpipeline.stage_completion import validate_stage_completion_sentinel
    validated = validate_stage_completion_sentinel(receipt, expected_stage="organize",
        expected_patient=course.parent.name, expected_course=course.name)
    assert validated["content_closure_sha256"] == fresh["content_closure_sha256"]
    from rtpipeline.dvh_generation import current, input_manifest
    effective = json.loads((course / ".dvh/current/.dvh_done").read_text())
    # Validate in the actual workflow runtime: its preflight uses the evaluated
    # rule parameters, effective configuration, and the newly published receipt.
    before = pipeline.count()
    pipeline.run(target)
    assert pipeline.count() == before
    bound = next(item for item in input_manifest(course) if item.get("path") == ".organized")
    assert bound["generation"]["content_closure_sha256"] == fresh["content_closure_sha256"]
    assert bound["generation"]["code_identity_sha256"] == fresh["code_identity_sha256"]
    assert current(course)["binding"] == effective["binding"]
    assert not (course / ".dvh/upstream-recompute.json").exists()


@pytest.mark.integration
@pytest.mark.requires_snakemake
def test_real_snakefile_routes_legacy_stage_before_first_dvh(tmp_path):
    pipeline = Production(tmp_path)
    course = pipeline.organize()
    pipeline.run(course / ".crop_ct_done")
    assert not (course / ".dvh").exists()
    receipt = course / ".segmentation_done"
    receipt.write_text('{"status":"disabled"}\n')
    pipeline.run(course / ".dvh_done")
    from rtpipeline.stage_completion import validate_stage_completion_sentinel
    fresh = validate_stage_completion_sentinel(receipt, expected_stage="segmentation",
        expected_patient=course.parent.name, expected_course=course.name)
    assert fresh["status"] == "disabled"
    from rtpipeline.dvh_generation import current, input_manifest
    bound = next(item for item in input_manifest(course) if item.get("path") == ".segmentation_done")
    assert bound["generation"]["code_identity_sha256"] == fresh["code_identity_sha256"]
    assert current(course)["binding"] == json.loads((course / ".dvh_done").read_text())["binding"]
    assert not (course / ".dvh/upstream-recompute.json").exists()
    count = pipeline.count()
    pipeline.run(course / ".dvh_done")
    assert pipeline.count() == count
