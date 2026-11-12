"""Synthetic bound-item and real-Snakefile Phase B crash/reuse evidence."""
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import hashlib
import subprocess
import tarfile
import zipfile
import re
import xml.etree.ElementTree as ET
import sys

import pandas as pd
import pytest
import yaml

from rtpipeline import durable_progress as dp
from rtpipeline import extraction_progress as ep
from test_dvh_durable_resume import Production


def test_item_binding_readback_locality_and_workers(tmp_path):
    code = tmp_path / "code.py"
    code.write_text("# exact deciding bytes\n")
    def item(name, *, source="a", parameter=1, workers=1):
        return ep.BoundItem(tmp_path / "items", name,
            {"parameter": parameter, "max_workers": workers}, {"source": source},
            environment={"numpy": "synthetic"}, code={"code.py": code})
    item("a").commit({"value": 1.0, "missing": float("nan")})
    item("b").commit({"value": 2.0})
    assert item("a", workers=4).load()["value"] == 1.0
    assert item("a", source="changed").load() is None
    assert item("a", parameter=2).load() is None
    assert item("b").load() == {"value": 2.0}
    code.write_text("# changed deciding bytes\n")
    assert item("a").load() is None
    assert item("b").load() is None


def test_failed_persistence_never_acknowledges_item(tmp_path, monkeypatch):
    item = ep.BoundItem(tmp_path / "items", "a", {}, {}, environment={})
    def fail(*args, **kwargs):
        raise dp.DurableProgressError(dp.WRITE_FAILED)
    monkeypatch.setattr(dp, "_atomic_write", fail)
    with pytest.raises(dp.DurableProgressError, match=dp.WRITE_FAILED):
        item.commit({"value": 1})
    assert item.load() is None


def test_localized_invalidation_and_custom_worker_budget(tmp_path):
    from rtpipeline.config_dependencies import (materialize_stage_dependency,
        invalidate_unbound_outputs, scientific_custom_config_sha256)
    dependency = materialize_stage_dependency(tmp_path / "dependencies", "radiomics", {"parameter": 1})
    digest = json.loads(dependency.read_text())["sha256"]
    valid, stale = tmp_path / "valid.done", tmp_path / "stale.done"
    valid.write_text(json.dumps({"configuration_dependency_sha256": digest}))
    stale.write_text(json.dumps({"configuration_dependency_sha256": "old"}))
    before = dependency.stat().st_mtime_ns, valid.read_bytes(), valid.stat().st_mtime_ns
    assert invalidate_unbound_outputs(dependency, [valid, stale],
        binding_field="configuration_dependency_sha256") == 1
    assert not stale.exists()
    assert before == (dependency.stat().st_mtime_ns, valid.read_bytes(), valid.stat().st_mtime_ns)
    custom = tmp_path / "custom.yaml"
    custom.write_text("custom_structures: []\nmax_workers: 1\n")
    first = scientific_custom_config_sha256(custom)
    custom.write_text("custom_structures: []\nmax_workers: 4\n")
    assert scientific_custom_config_sha256(custom) == first


def test_condition_adapter_queues_only_missing_and_preserves_success(tmp_path, monkeypatch):
    from rtpipeline import robustness_watchdog
    items = [ep.BoundItem(tmp_path / "conditions", str(i), {}, {}, environment={}) for i in range(3)]
    items[0].commit({"value": 0})
    submitted = []
    class Results:
        def __init__(self, context, function, tasks, workers, **kwargs):
            submitted.append([task[1]["index"] for task in tasks])
            self.results = iter([{"__task_index__": 1, "value": 2},
                {"__task_index__": 0, "__technical_failure__": {"reason_code": "synthetic"}}])
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def next(self, timeout): return next(self.results)
    monkeypatch.setattr(robustness_watchdog, "SupervisedResults", Results)
    monkeypatch.setattr(ep, "task_item", lambda task: items[task["index"]])
    tasks = [("mask", {"index": i}) for i in range(3)]
    with ep.ResumableResults(None, None, tasks, 2) as results:
        values = [results.next(timeout=1) for _ in range(3)]
        assert [value["__task_index__"] for value in values] == [0, 2, 1]
        for value in values:
            results.commit(value["__task_index__"], value)
    assert submitted == [[1, 2]]
    assert items[1].load() is None
    assert items[2].load() == {"value": 2}
    ep.ResumableResults(None, None, tasks, 1)
    assert submitted[-1] == [1]


def test_real_native_arm_cache_survives_later_arm_failure(tmp_path):
    """Real NumPy 1/PyRadiomics extraction; no polygon rasterization is needed."""
    from rtpipeline.radiomics_conda import _conda_subprocess_env
    prefix = Path(os.environ.get("RTPIPELINE_RADIOMICS_ENV_PREFIX",
        "/home/konrad/rtpipeline_release/b29a324/envs/rtpipeline-radiomics"))
    if not (prefix / "bin/python").is_file():
        pytest.skip("dedicated extraction runtime required")
    code = r'''
import json,sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
from rtpipeline.radiomics_ct_contract import (extract_ct_roi_arms, classify_ct_roi,
    configured_parameter_hash, CT_EXTRACTION_ARMS, PRIMARY_ARM, SENSITIVITY_ARM)
from rtpipeline.durable_progress import ResultCache, Binding
root=Path(sys.argv[1])
params=root/'parameters.yaml'
params.write_text('setting:\n  minimumROISize: 8\n  minimumROIDimensions: 2\n  binWidth: 25\nimageType:\n  Original: {}\nfeatureClass:\n  firstorder: [Mean]\n  shape: [VoxelVolume]\n'.replace('\\n','\n'))
image=sitk.GetImageFromArray(np.arange(12**3,dtype=np.float32).reshape(12,12,12)/8-400)
array=np.zeros((12,12,12),np.uint8);array[3:9,3:9,3:9]=1
mask=sitk.GetImageFromArray(array)
decision=classify_ct_roi('Manual','PTV')
metadata=dict(patient_id='SYNTHETIC',course_id='C1',series_uid='synthetic-series',
 segmentation_source='Manual',mask_identity='synthetic-mask',stable_roi_identifier='synthetic-roi',
 roi_original_name='PTV',roi_name='PTV',modality='CT',rasterizer_backend='nifti/no-polygon-fill',
 source_content_sha256='a'*64)
calls={'shape':0,SENSITIVITY_ARM:0,PRIMARY_ARM:0};state={'fail':True}
original=featureextractor.RadiomicsFeatureExtractor.execute
def execute(self,*args,**kwargs):
 arm='shape' if set(self.enabledFeatures)=={'shape'} else (PRIMARY_ARM if self.settings.get('resegmentRange') else SENSITIVITY_ARM)
 calls[arm]+=1
 if arm==PRIMARY_ARM and state['fail']: raise RuntimeError('synthetic later-arm failure')
 return original(self,*args,**kwargs)
featureextractor.RadiomicsFeatureExtractor.execute=execute
hashes={arm:configured_parameter_hash(params,arm=arm,window=decision.primary_resegment_range_hu if arm==PRIMARY_ARM else None,large_roi=False) for arm in CT_EXTRACTION_ARMS}
def run(workers=1):
 return extract_ct_roi_arms(image,mask,factory=lambda:featureextractor.RadiomicsFeatureExtractor(str(params)),
  decision=decision,common_metadata=metadata,run_identifier='synthetic-run',code_revision='synthetic-code',
  native_voxel_count=int(array.sum()),required=True,configured_parameter_hashes=hashes,
  progress_root=root/'progress',progress_context={'max_workers':workers})
try: run()
except RuntimeError as error: assert str(error)=='synthetic later-arm failure'
else: raise AssertionError('required arm failure was swallowed')
state['fail']=False
rows=run()
assert calls=={'shape':1,SENSITIVITY_ARM:1,PRIMARY_ARM:2},calls
assert len(rows)==2 and all(row['extraction_status']=='success' for row in rows)
run(workers=4)
assert calls=={'shape':1,SENSITIVITY_ARM:1,PRIMARY_ARM:2},calls
metadata['source_content_sha256']='b'*64
run()
assert calls=={'shape':2,SENSITIVITY_ARM:2,PRIMARY_ARM:3},calls
for path in (root/'progress').rglob('commit.json'):
 marker=json.loads(path.read_text())
 assert marker['item_key'] in ResultCache(path.parent.parent).load_valid(Binding(**marker['binding'])).entries
print(json.dumps(calls,sort_keys=True))
'''
    env = _conda_subprocess_env()
    env["LD_LIBRARY_PATH"] = str(prefix / "lib")
    completed = subprocess.run([str(prefix / "bin/python"), "-c", code, str(tmp_path)],
        capture_output=True, text=True, env=env, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_checkpoint_failure_never_marks_completed(tmp_path, monkeypatch):
    from rtpipeline.radiomics_conda import RadiomicsCheckpoint
    checkpoint = RadiomicsCheckpoint(tmp_path / "checkpoint.parquet")
    def fail(*args, **kwargs):
        raise OSError("synthetic full disk")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    with pytest.raises(dp.DurableProgressError, match="RADIOMICS_CHECKPOINT_WRITE_FAILED"):
        checkpoint.add_result({"roi_name": "PTV1", "original_firstorder_Mean": 1.0})
    assert checkpoint.get_completed_count() == 0


def test_missing_source_never_falls_back_to_unbound_checkpoint(tmp_path, monkeypatch):
    from rtpipeline import radiomics_conda as conda
    from rtpipeline.radiomics_ct_contract import CT_EXTRACTION_ARMS
    checkpoint = tmp_path / "legacy.parquet"
    checkpoint.write_bytes(b"prior unbound evidence must be preserved")
    def forbidden(*args, **kwargs):
        raise AssertionError("missing sources must fail before extraction or environment probing")
    monkeypatch.setattr(conda, "check_radiomics_env", forbidden)
    with pytest.raises(conda.RadiomicsCourseExtractionError, match="source image or mask is missing"):
        conda.process_radiomics_batch([{"image_path": str(tmp_path / "missing.nrrd"),
            "mask_path": str(tmp_path / "missing-mask.nrrd"), "roi_name": "PTV", "dual_arm_ct": True,
            "metadata": {"patient_id": "SYNTHETIC", "course_id": "C1", "series_uid": "synthetic-series",
                "segmentation_source": "Manual", "mask_identity": "synthetic-mask",
                "roi_original_name": "PTV", "stable_roi_identifier": "synthetic-roi"},
            "configured_parameter_hashes": {arm: "synthetic-parameters" for arm in CT_EXTRACTION_ARMS}}],
            tmp_path / "radiomics_ct.xlsx", checkpoint_path=checkpoint, enable_heartbeat=False)
    assert checkpoint.read_bytes() == b"prior unbound evidence must be preserved"


def test_stream_ack_follows_durable_commit_and_failure_propagates(tmp_path):
    from rtpipeline.radiomics_conda import _stream_batch
    code = '''import sys,json
for index in range(3):
 print("RTP_ROI\\t"+json.dumps({"__task_index__":index,"value":index}),flush=True)
 assert sys.stdin.readline().strip()=="ACK "+str(index)
'''
    items = []
    def accept(index, value):
        item = ep.BoundItem(tmp_path / "items", str(index), {}, {}, environment={})
        item.commit(value)
        items.append(item)
    results = _stream_batch([sys.executable, "-c", code], 30, accept)
    assert len(results) == len(items) == 3
    assert [item.load()["value"] for item in items] == [0, 1, 2]
    def fail(index, value):
        raise dp.DurableProgressError("SYNTHETIC_PERSISTENCE_FAILURE")
    with pytest.raises(dp.DurableProgressError, match="SYNTHETIC_PERSISTENCE_FAILURE"):
        _stream_batch([sys.executable, "-c", code], 30, fail)


def test_aggregate_generation_readback_and_snapshot_failure(tmp_path):
    from rtpipeline.aggregate_generation import commit, read_back
    artifact, receipt = tmp_path / "table.parquet", tmp_path / "aggregate.commit.json"
    artifact.write_bytes(b"synthetic table bytes")
    commit({artifact.name: artifact}, receipt, inputs=["frozen"])
    assert read_back(receipt)["inputs"] == ["frozen"]
    artifact.write_bytes(b"corrupt")
    with pytest.raises(dp.DurableProgressError, match="AGGREGATE_PUBLICATION_READBACK_FAILED"):
        read_back(receipt)
    def changed():
        raise RuntimeError("snapshot changed")
    with pytest.raises(RuntimeError, match="snapshot changed"):
        commit({artifact.name: artifact}, receipt, verify=changed)
    assert not receipt.exists()


def test_aggregate_preflight_revalidates_imported_code(tmp_path, monkeypatch):
    from rtpipeline import aggregate_generation as generation
    output = tmp_path / "Output"
    results = output / "_RESULTS"
    results.mkdir(parents=True)
    code = tmp_path / "deciding-code.py"
    code.write_text("# exact imported code A\n")
    monkeypatch.setattr(generation, "standard_code_manifest", lambda: {"code.py": code})
    monkeypatch.setattr(generation, "aggregate_environment", lambda: {})
    artifact = results / "table.parquet"
    artifact.write_bytes(b"synthetic output")
    receipt = results / "aggregate_results.commit.json"
    generation.commit({artifact.name: artifact}, receipt, inputs={
        "snapshots": [], "code": {"code.py": dp.file_sha256(code)}, "environment": {}})
    assert generation.preflight(output) == {"invalidated": []}
    before = artifact.read_bytes()
    code.write_text("# exact imported code B\n")
    assert generation.preflight(output) == {"invalidated": [receipt.name]}
    assert not receipt.exists() and artifact.read_bytes() == before


def read_all_commits(output):
    """Independently read every cache transaction and immutable generation."""
    from rtpipeline.aggregate_generation import read_back
    from rtpipeline.dvh_generation import verify
    counts = Counter()
    for marker_path in output.rglob("commit.json"):
        marker = json.loads(marker_path.read_text())
        if "payload_sha256" in marker:
            assert not marker["sealed"]  # These fixtures contain no sealed data.
            cache = dp.ResultCache(marker_path.parent.parent)
            loaded = cache.load_valid(dp.Binding(**marker["binding"]))
            assert marker["item_key"] in loaded.entries
            assert loaded.report["corrupt"] == loaded.report["conflict"] == 0
            counts["item_transactions"] += 1
        elif marker.get("schema") == "rtpipeline-dvh-generation-v1":
            verify(marker_path.parent)
            counts["dvh_generations"] += 1
        elif marker.get("schema") == "rtpipeline-aggregate-generation-v1":
            for name, digest in marker["artifacts"].items():
                assert dp.file_sha256(marker_path.parent / name) == digest
            counts["aggregate_generations"] += 1
    for receipt in output.rglob("*.commit.json"):
        read_back(receipt)
        counts["aggregate_publications"] += 1
    return dict(counts)


class PhaseBProduction(Production):
    run_timeout = 480
    def __init__(self, root):
        super().__init__(root)
        self.radiomics = Path(os.environ.get("RTPIPELINE_RADIOMICS_ENV_PREFIX",
            "/home/konrad/rtpipeline_release/b29a324/envs/rtpipeline-radiomics"))
        if not (self.radiomics / "bin/python").is_file():
            pytest.skip("dedicated radiomics interpreter required")
        params = root / "parameters.yaml"
        params.write_text(yaml.safe_dump({"setting": {"resampledPixelSpacing": [2., 2., 5.],
            "interpolator": "sitkBSpline", "minimumROISize": 8, "minimumROIDimensions": 2,
            "binWidth": 25, "normalize": False, "additionalInfo": True, "correctMask": True},
            "imageType": {"Original": {}}, "featureClass": {
                "firstorder": ["Mean", "Variance"], "shape": ["VoxelVolume"]}}))
        # Both operands and the union are mapped target concepts.
        self.custom.write_text(yaml.safe_dump({"custom_structures": [{"name": "PTV1+PTV2",
            "operation": "union", "source_structures": ["PTV1", "PTV2"]}]}))
        config = yaml.safe_load(self.config.read_text())
        config["environments"]["radiomics"] = str(self.radiomics)
        config["radiomics"] = {"enabled": True, "params_file": str(params), "sequential": True}
        config["radiomics_robustness"] = {"enabled": True, "modes": ["segmentation_perturbation"],
            "segmentation_perturbation": {"apply_to_structures": ["PTV1", "PTV1+PTV2"],
                "small_volume_changes": [-.15, 0., .15], "max_translation_mm": 0.,
                "n_random_contour_realizations": 0, "noise_levels": [0]}}
        self.config.write_text(yaml.safe_dump(config))
        self.env["RTPIPELINE_RADIOMICS_ENV_PREFIX"] = str(self.radiomics)
        self.env["MKL_THREADING_LAYER"] = "GNU"
        self.env["LD_LIBRARY_PATH"] = str(self.runtime.parent.parent / "lib")
        # The installed prefix has a stale second OpenCV distribution record.
        # Use a private package view containing the actual pinned headless
        # provider; no installed package, binary, metadata or guard is edited.
        sites = root / "radiomics-sites"
        sites.mkdir()
        installed = next((self.radiomics / "lib").glob("python*/site-packages"))
        for path in installed.iterdir():
            if path.name.startswith("opencv_python-") and path.name.endswith(".dist-info"):
                continue
            if path.name != "__pycache__":
                (sites / path.name).symlink_to(path, target_is_directory=path.is_dir())
        self.env["WS516_RAD_SITES"] = str(sites)
        hook = root / "instrumentation/sitecustomize.py"
        with hook.open("a") as stream:
            stream.write('''
# Phase B: only task-owned files and interpreters are observed.
if sys.prefix in {os.environ.get("WS515_RUNTIME_PREFIX"), os.environ.get("RTPIPELINE_RADIOMICS_ENV_PREFIX")}:
    if sys.prefix == os.environ.get("RTPIPELINE_RADIOMICS_ENV_PREFIX"):
        sys.path[:] = [os.environ["WS516_RAD_SITES"] if p.startswith(sys.prefix+"/lib/python") and p.endswith("site-packages") else p for p in sys.path]
    import inspect
    import rtpipeline.extraction_progress as ep
    import rtpipeline.radiomics_ct_contract as ct
    ct.new_run_identifier = lambda: "synthetic-phase-b-run"
    def event(value):
        with open(os.environ["WS516_EVENTS"], "a") as f:
            f.write(json.dumps(value)+"\\n"); f.flush(); os.fsync(f.fileno())
    def pause_b(name):
        if name == os.environ.get("WS515_KILL_POINT"):
            with open(os.environ["WS515_HANDSHAKE"], "w") as f:
                f.write(name); f.flush(); os.fsync(f.fileno())
            while True: time.sleep(.1)
    extract = ct.extract_ct_roi_arms
    active_stage = None
    def counted_extract(*args, **kwargs):
        global active_stage
        caller = inspect.currentframe().f_back.f_locals
        task = caller.get("task_params") or caller.get("task")
        if isinstance(task, dict) and task.get("__progress_root__"):
            item = ep.task_item(task)
            stage = "robustness" if "robustness" in task["__progress_root__"] else "radiomics"
            active_stage = stage
            event({"event":"extract", "stage":stage, "key":item.key})
            if stage == "robustness" and os.environ.get("WS515_KILL_POINT") == "robustness_later":
                counts = [json.loads(line) for line in open(os.environ["WS516_EVENTS"]) if line.strip()]
                if sum(e["event"]=="extract" and e["stage"]==stage for e in counts) >= 3:
                    while sum(e["event"]=="commit" and e["stage"]==stage for e in counts) < 2:
                        time.sleep(.01)
                        counts = [json.loads(line) for line in open(os.environ["WS516_EVENTS"]) if line.strip()]
        return extract(*args, **kwargs)
    ct.extract_ct_roi_arms = counted_extract
    if sys.prefix == os.environ.get("RTPIPELINE_RADIOMICS_ENV_PREFIX"):
        from radiomics import featureextractor
        original_execute = featureextractor.RadiomicsFeatureExtractor.execute
        def observed_execute(self, *args, **kwargs):
            value = original_execute(self, *args, **kwargs)
            if active_stage == "robustness" and os.environ.get("WS515_KILL_POINT") == "robustness_later":
                counts = [json.loads(line) for line in open(os.environ["WS516_EVENTS"]) if line.strip()]
                if sum(e["event"]=="commit" and e["stage"]=="robustness" for e in counts) >= 2:
                    pause_b("robustness_later")
            return value
        featureextractor.RadiomicsFeatureExtractor.execute = observed_execute
    import rtpipeline.radiomics_robustness as rr
    summarize = rr.summarize_feature_stability
    def counted_summary(group, *args, **kwargs):
        columns = [c for c in ("structure", "segmentation_source", "extraction_arm", "feature_name") if c in group]
        key = ep.digest([group[c].iloc[0] for c in columns])
        event({"event":"extract", "stage":"reduction", "key":key})
        return summarize(group, *args, **kwargs)
    rr.summarize_feature_stability = counted_summary
    import rtpipeline.radiomics as nominal
    nominal._extractor = lambda *args, **kwargs: None
    import rtpipeline.utils as utils
    adaptive = utils.run_tasks_with_adaptive_workers
    def fixture_workers(label, *args, **kwargs):
        if label == "QC": kwargs["use_processes"] = False
        return adaptive(label, *args, **kwargs)
    utils.run_tasks_with_adaptive_workers = fixture_workers
    original_commit = ep.BoundItem.commit
    def committed(self, value):
        result = original_commit(self, value)
        root = str(self.cache.root)
        if "/arms/" in root:
            return result
        stage = "robustness" if ".robustness-progress" in root else (
            "reduction" if "/reductions/" in root else "radiomics")
        event({"event":"commit", "stage":stage,"key":self.key})
        counts = [json.loads(line) for line in open(os.environ["WS516_EVENTS"])
                  if line.strip()]
        n = sum(e["event"]=="commit" and e["stage"]==stage for e in counts)
        if stage=="radiomics" and n==2: pause_b("conda_stream")
        if stage=="reduction" and n==1: pause_b("aggregate_reduction")
        return result
    ep.BoundItem.commit = committed
''')
        self.events = root / "events.jsonl"
        self.env["WS516_EVENTS"] = str(self.events)

    def event_rows(self):
        return [json.loads(line) for line in self.events.read_text().splitlines()] if self.events.exists() else []


@pytest.mark.integration
@pytest.mark.requires_snakemake
def test_real_phase_b_three_sigkill_legs(tmp_path):
    pipeline = PhaseBProduction(tmp_path)
    course = pipeline.organize()
    clean = tmp_path / "organized"
    shutil.copytree(pipeline.output, clean, symlinks=True)
    target = pipeline.output / "_RESULTS/radiomics_robustness_summary.commit.json"
    pipeline.run(target)
    control = pd.read_parquet(course / "radiomics_ct.parquet")
    control_bytes = (course / "radiomics_ct.parquet").read_bytes()
    robustness = pd.read_parquet(course / "radiomics_robustness_ct.parquet")
    robustness_bytes = (course / "radiomics_robustness_ct.parquet").read_bytes()
    raw = (pipeline.output / "_RESULTS/radiomics_robustness_summary_raw_values.parquet").read_bytes()
    shutil.rmtree(pipeline.output)
    shutil.copytree(clean, pipeline.output, symlinks=True)
    pipeline.events.unlink()
    previous_committed, previous_counts = set(), Counter()
    for leg in ("conda_stream", "robustness_later", "aggregate_reduction"):
        pipeline.run(target, kill=leg)
        committed = {(e["stage"], e["key"]) for e in pipeline.event_rows() if e["event"] == "commit"}
        before = Counter((e["stage"], e["key"]) for e in pipeline.event_rows() if e["event"] == "extract")
        for key in previous_committed:
            if key[0] in {"radiomics", "robustness", "reduction"}:
                assert before[key] == previous_counts[key], key
        previous_committed, previous_counts = committed, before
        read_all_commits(pipeline.output)
        # Proceed to the next deterministic kill boundary on the same workflow.
        # The final resume below checks all committed extraction keys together.
        pipeline.env["WS516_COMMITTED"] = json.dumps(sorted(committed))
    pipeline.run(target)
    after = Counter((e["stage"], e["key"]) for e in pipeline.event_rows() if e["event"] == "extract")
    for key in committed:
        if key[0] in {"radiomics", "robustness", "reduction"}:
            assert after[key] == before[key], key
    # Run identifiers are attempt provenance; compare every other published cell.
    for expected, path in ((control, course / "radiomics_ct.parquet"),
                           (robustness, course / "radiomics_robustness_ct.parquet")):
        actual = pd.read_parquet(path)
        columns = [c for c in expected if c != "run_identifier"]
        pd.testing.assert_frame_equal(expected[columns], actual[columns], check_exact=True)
    actual_raw = pd.read_parquet(pipeline.output / "_RESULTS/radiomics_robustness_summary_raw_values.parquet")
    from io import BytesIO
    expected_raw = pd.read_parquet(BytesIO(raw))
    columns = [c for c in expected_raw if c != "run_identifier"]
    pd.testing.assert_frame_equal(expected_raw[columns], actual_raw[columns], check_exact=True)
    assert (course / "radiomics_ct.parquet").read_bytes() == control_bytes
    assert (course / "radiomics_robustness_ct.parquet").read_bytes() == robustness_bytes
    assert (pipeline.output / "_RESULTS/radiomics_robustness_summary_raw_values.parquet").read_bytes() == raw
    pipeline.run(pipeline.output / "_RESULTS/aggregate_results.commit.json")
    calls = Counter((e["stage"], e["key"]) for e in pipeline.event_rows() if e["event"] == "extract")
    config = yaml.safe_load(pipeline.config.read_text())
    config["max_workers"] = 2
    pipeline.config.write_text(yaml.safe_dump(config))
    custom = yaml.safe_load(pipeline.custom.read_text())
    custom["max_workers"] = 2
    pipeline.custom.write_text(yaml.safe_dump(custom))
    pipeline.cores = 2
    log = pipeline.run("all")
    assert "Nothing to be done" in log.read_text()
    assert calls == Counter((e["stage"], e["key"]) for e in pipeline.event_rows() if e["event"] == "extract")
    readback = read_all_commits(pipeline.output)
    destination = os.environ.get("WS516_EVIDENCE_DIR")
    if destination:
        Path(destination, "interrupt-resume-evidence.json").write_text(json.dumps({
            "legs": ["conda_stream", "robustness_later", "aggregate_reduction"],
            "repeated_committed_extractions": 0, "worker_only_scientific_jobs": 0,
            "byte_equal_parquet": {"radiomics_ct.parquet": hashlib.sha256(control_bytes).hexdigest(),
                "radiomics_robustness_ct.parquet": hashlib.sha256(robustness_bytes).hexdigest(),
                "raw_values.parquet": hashlib.sha256(raw).hexdigest()}, "readback": readback}, indent=2) + "\n")


@pytest.mark.integration
@pytest.mark.requires_snakemake
def test_uninterrupted_scientific_bytes_match_base(tmp_path):
    """Same source bytes, paths, parameters and runtimes against actual 3f94c63."""
    from io import BytesIO
    from test_dvh_durable_resume import ROOT
    pipeline = PhaseBProduction(tmp_path)
    course = pipeline.organize()
    clean = tmp_path / "organized"
    shutil.copytree(pipeline.output, clean, symlinks=True)
    pipeline.run("all")
    candidate = {p.relative_to(pipeline.output).as_posix(): p.read_bytes()
                 for p in pipeline.output.rglob("*.parquet")
                 if not any(part.startswith(".") for part in p.relative_to(pipeline.output).parts)}
    candidate_books = {p.relative_to(pipeline.output).as_posix(): p.read_bytes()
                 for p in pipeline.output.rglob("*.xlsx")
                 if not any(part.startswith(".") for part in p.relative_to(pipeline.output).parts)}
    for name in ("rtpipeline", "workflow", "envs"):
        shutil.rmtree(pipeline.source / name)
    archive = subprocess.run(["git", "archive", "3f94c63", "rtpipeline", "workflow", "envs", "Snakefile", "config.yaml"],
        cwd=ROOT, check=True, capture_output=True).stdout
    with tarfile.open(fileobj=BytesIO(archive)) as source:
        source.extractall(pipeline.source, filter="data")
    # The base imports dicompyler-core while validating main-built custom
    # structures in the extraction interpreter. Supply the already-installed
    # pure Python package in this task's package view, without installation.
    sites = Path(pipeline.env["WS516_RAD_SITES"])
    module = Path(sys.executable).parent.parent / ("lib/python" + ".".join(map(str, sys.version_info[:2]))) / "site-packages/dicompylercore"
    if not (sites / "dicompylercore").exists() and module.is_dir():
        (sites / "dicompylercore").symlink_to(module, target_is_directory=True)
    hook = tmp_path / "baseline-hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text('''
import os,sys
def guard(event,args):
    if event in {"socket.connect","socket.connect_ex","socket.bind","socket.getaddrinfo"}:
        raise RuntimeError("synthetic tests forbid network")
sys.addaudithook(guard)
if sys.prefix == os.environ.get("RTPIPELINE_RADIOMICS_ENV_PREFIX"):
    sys.path[:] = [os.environ["WS516_RAD_SITES"] if p.startswith(sys.prefix+"/lib/python") and p.endswith("site-packages") else p for p in sys.path]
if sys.prefix in {os.environ.get("WS515_RUNTIME_PREFIX"), os.environ.get("RTPIPELINE_RADIOMICS_ENV_PREFIX")}:
    import rtpipeline.radiomics_ct_contract as ct
    ct.new_run_identifier = lambda: "synthetic-phase-b-run"
    import rtpipeline.radiomics as nominal
    nominal._extractor = lambda *args, **kwargs: None
    import rtpipeline.utils as utils
    adaptive = utils.run_tasks_with_adaptive_workers
    def fixture_workers(label,*args,**kwargs):
        if label=="QC": kwargs["use_processes"]=False
        return adaptive(label,*args,**kwargs)
    utils.run_tasks_with_adaptive_workers = fixture_workers
''')
    pipeline.env["PYTHONPATH"] = os.pathsep.join((str(hook), str(pipeline.source)))
    shutil.rmtree(pipeline.output)
    shutil.copytree(clean, pipeline.output, symlinks=True)
    pipeline.run("all")
    evidence = {"base": "3f94c631d44d9452b0299341fa8fb01e93d1fd3c", "parquet": {}, "workbooks": {}}
    for name, expected in candidate.items():
        actual = (pipeline.output / name).read_bytes()
        if actual != expected:
            pd.testing.assert_frame_equal(pd.read_parquet(BytesIO(expected)),
                                          pd.read_parquet(BytesIO(actual)), check_exact=True)
        evidence["parquet"][name] = {"byte_equal": actual == expected,
            "candidate_sha256": hashlib.sha256(expected).hexdigest(),
            "base_sha256": hashlib.sha256(actual).hexdigest()}
        assert actual == expected, name
    for name, expected in candidate_books.items():
        actual = (pipeline.output / name).read_bytes()
        provenance_columns = {}
        with pd.ExcelFile(BytesIO(expected)) as first, pd.ExcelFile(BytesIO(actual)) as second:
            assert first.sheet_names == second.sheet_names
            for sheet in first.sheet_names:
                left, right = first.parse(sheet), second.parse(sheet)
                # These columns explicitly report producing code/receipt hashes;
                # changed implementation bytes necessarily change those hashes.
                provenance = [c for c in left if c in {"source_dispositions_sha256"}]
                provenance_columns[first.sheet_names.index(sheet) + 1] = {
                    left.columns.get_loc(c) + 1 for c in provenance}
                pd.testing.assert_frame_equal(left.drop(columns=provenance),
                    right.drop(columns=provenance), check_exact=True)
        with zipfile.ZipFile(BytesIO(expected)) as left_zip, zipfile.ZipFile(BytesIO(actual)) as right_zip:
            assert set(left_zip.namelist()) == set(right_zip.namelist())
            changed = []
            for part in left_zip.namelist():
                left, right = left_zip.read(part), right_zip.read(part)
                if left == right:
                    continue
                changed.append(part)
                if part == "docProps/core.xml":
                    def normalize(raw):
                        tree = ET.fromstring(raw)
                        for element in list(tree):
                            if element.tag.rsplit("}", 1)[-1] in {"created", "modified"}:
                                tree.remove(element)
                        return ET.tostring(tree)
                else:
                    match = re.fullmatch(r"xl/worksheets/sheet(\d+)\.xml", part)
                    assert match and provenance_columns[int(match[1])], (name, part)
                    columns = provenance_columns[int(match[1])]
                    def normalize(raw):
                        tree = ET.fromstring(raw)
                        for cell in tree.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c"):
                            address = cell.attrib.get("r", "")
                            letters = re.match(r"([A-Z]+)(\d+)", address)
                            if not letters or int(letters[2]) == 1:
                                continue
                            index = 0
                            for letter in letters[1]: index = index * 26 + ord(letter) - 64
                            if index in columns:
                                for element in cell.iter():
                                    if element.text: element.text = "CODE_DISPOSITION_HASH"
                        return ET.tostring(tree)
                assert normalize(left) == normalize(right), (name, part)
        evidence["workbooks"][name] = {"all_scientific_cells_exact": True,
            "file_byte_equal": actual == expected, "changed_zip_parts": changed,
            "differences_limited_to_packaging_time_and_declared_code_hash": True}
    destination = os.environ.get("WS516_EVIDENCE_DIR")
    if destination:
        Path(destination, "base-byte-equality.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
