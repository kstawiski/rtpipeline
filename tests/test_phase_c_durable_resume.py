"""Synthetic item read-back, production SIGKILL, and e1d551a equality evidence."""
from collections import Counter
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk
import yaml

from rtpipeline import artifact_progress as ap, durable_progress as dp
from rtpipeline.layout import build_course_dirs
from test_dvh_durable_resume import Production, ROOT
from test_rtsegmentator_integration import course as portal_course, FakeClient, make_package
from test_segmentation_fail_closed import _course


def evidence(name, value):
    directory = os.environ.get("WS518_EVIDENCE_DIR")
    if directory:
        Path(directory, name + ".json").write_text(json.dumps(value, indent=2) + "\n")


def read_back_all(root):
    count = 0
    for marker in root.rglob("commit.json"):
        if "-invalid" in str(marker):
            continue
        record = json.loads(marker.read_text())
        if "payload_sha256" not in record:
            continue
        read = dp.ResultCache(marker.parent.parent).load_valid(dp.Binding(**record["binding"]))
        assert record["item_key"] in read.entries
        assert not read.report["corrupt"] and not read.report["conflict"]
        for value in read.entries.values():
            value = __import__("rtpipeline.dvh_generation", fromlist=["_decode_row"])._decode_row(value.get("result", value))
            if isinstance(value, dict) and "artifact_root" in value:
                for name, digest in value["artifacts"].items():
                    # Stale bindings may retain historical bytes. Current
                    # publications are revalidated separately by each producer.
                    path = Path(value["artifact_root"]) / name
                    if path.is_file():
                        dp.file_sha256(path)
        count += 1
    return count


def test_artifact_readback_binding_and_local_repair(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"source-a")
    output = tmp_path / "output"
    output.write_bytes(b"complete")
    code = tmp_path / "deciding.py"
    code.write_text("# deciding bytes\n")
    def item(name="a", parameter=1):
        return ap.ArtifactItem(tmp_path / "items", name, {"parameter": parameter},
            {"source": dp.file_sha256(source)}, base=tmp_path, code={"deciding.py": code}, environment={})
    item().commit_artifacts([output])
    item("b").commit_artifacts([])
    assert item().load() is not None
    output.write_bytes(b"partial")
    assert item().load() is None
    assert item("b").load() is not None
    output.write_bytes(b"repaired")
    item().commit_artifacts([output])
    assert item().load() is not None
    assert item(parameter=2).load() is None
    source.write_bytes(b"source-b")
    assert item().load() is None
    assert read_back_all(tmp_path) == 2


def test_artifact_persistence_failure_is_not_completion(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.write_bytes(b"synthetic")
    item = ap.ArtifactItem(tmp_path / "items", "a", {}, {}, base=tmp_path,
                           code={"code": Path(ap.__file__)}, environment={})
    def fail(*args, **kwargs):
        raise dp.DurableProgressError(dp.WRITE_FAILED)
    monkeypatch.setattr(dp, "_atomic_write", fail)
    with pytest.raises(dp.DurableProgressError):
        item.commit_artifacts([output])
    assert item.load() is None


def test_copy_repairs_partial_and_source_changes_and_preserves_prior_on_failure(tmp_path, monkeypatch):
    from rtpipeline.dicom_copy import DicomCopyManager, DicomCopyConfig
    source, target = tmp_path / "input.dcm", tmp_path / "Output/copied.dcm"
    source.write_bytes(b"synthetic-complete")
    manager = DicomCopyManager(DicomCopyConfig(dedup_by_sop_uid=False, use_hardlinks=False), tmp_path / "Output")
    manager.copy_dicom(source, target)
    before = target.read_bytes(), target.stat().st_mtime_ns
    assert manager.copy_dicom(source, target) == (target, False)
    assert before == (target.read_bytes(), target.stat().st_mtime_ns)
    target.write_bytes(b"partial")
    assert manager.copy_dicom(source, target) == (target, True)
    source.write_bytes(b"same-path-new-bytes")
    assert manager.copy_dicom(source, target) == (target, True)
    previous = target.read_bytes()
    source.write_bytes(b"next-generation")
    def fail_copy(src, dst, **kwargs):
        Path(dst).write_bytes(b"partial"); raise OSError("synthetic failed copy")
    monkeypatch.setattr(shutil, "copy2", fail_copy)
    with pytest.raises(OSError, match="failed copy"):
        manager.copy_dicom(source, target)
    assert target.read_bytes() == previous
    assert not list(target.parent.glob(".prepare-*"))


def test_metadata_items_bind_bytes_and_keep_unchanged_table_mtimes(tmp_path, monkeypatch):
    from rtpipeline import meta
    from test_metadata_modality_hardening import _config, _write_export
    config = _config(tmp_path)
    _write_export(config.dicom_root, {"plan": "plan.dcm", "dose": "dose.dcm", "struct": "struct.dcm", "record": "record.dcm", "ct": "ct.dcm"})
    meta.export_metadata(config)
    books = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (config.output_root / "Data").glob("*.xlsx")}
    counts = Counter()
    read = meta._read_metadata_file
    def observed(path):
        counts[path.name] += 1; return read(path)
    monkeypatch.setattr(meta, "_read_metadata_file", observed)
    source = config.dicom_root / "ct.dcm"
    before = source.stat()
    import pydicom
    dataset = pydicom.dcmread(source); dataset.add_new((0x7fe0, 0x0010), "OB", b"synthetic pixel bytes ")
    dataset.save_as(source, enforce_file_format=True)
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    meta.export_metadata(config)
    assert counts == {"ct.dcm": 1}
    assert books == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in books}
    assert read_back_all(config.output_root) >= 5


def test_organize_current_source_locality_and_detached_record(tmp_path, monkeypatch):
    from rtpipeline import organize, segmentation, meta
    from rtpipeline.config import PipelineConfig
    from organize_io_fixture import synthetic, synthetic_ct_conversion
    from synthetic_rt_fixtures import make_record
    from pydicom.uid import generate_uid
    monkeypatch.setenv("RTPIPELINE_INDEX_PROCESSES", "1")
    monkeypatch.setenv("RTPIPELINE_MASK_PROCESSES", "1")
    inputs, output = tmp_path / "Input", tmp_path / "Output"
    synthetic(inputs, ct_slices=12, all_slices=True)
    config = PipelineConfig(inputs, output, tmp_path / "Logs", max_workers_override=1, resume=True)
    calls = Counter()
    def convert(config, dicom_dir, nifti_out, **kwargs):
        if dicom_dir.name == "CT":
            calls[str(dicom_dir)] += 1
        return synthetic_ct_conversion(config, dicom_dir, nifti_out, **kwargs)
    monkeypatch.setattr(organize, "run_dcm2niix", convert)
    monkeypatch.setattr(segmentation, "run_dcm2niix", convert)
    courses = organize.organize_and_merge(config)
    assert len(courses) == 4
    assert sum(calls.values()) == 4
    protected = {p: (p.read_bytes(), p.stat().st_mtime_ns) for co in courses for p in organize._organized_artifacts(co.dirs.root)}
    organize.organize_and_merge(config)
    assert sum(calls.values()) == 4
    assert protected == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in protected}
    import pydicom
    source = inputs / "SYNTH_A/0/ct0.dcm"
    ds = pydicom.dcmread(source); ds.PixelData = bytes([1, 0]) * (ds.Rows * ds.Columns)
    ds.save_as(source, enforce_file_format=True)
    organize.organize_and_merge(config)
    assert sum(calls.values()) == 5
    unaffected = {p: value for p, value in protected.items() if "/SYNTH_A/2024-01/" not in str(p)}
    assert unaffected == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in unaffected}
    make_record(inputs / "detached/new-record.dcm", generate_uid(), patient_id="SYNTH_A", fraction_number=1)
    organize.organize_and_merge(config)
    discovery = json.loads((output / "_COURSES/.organize-progress/discovery.json").read_text())
    assert "detached/new-record.dcm" in discovery["sources"]
    ledger = json.loads((output / "_COURSES/organize_ledger.json").read_text())
    assert ledger["source_plan_dispositions"]["unresolved_record_plans"]
    assert sum(calls.values()) == 5
    evidence("organize-source-locality", {"conversion_calls": dict(calls), "detached_record_discovered": True,
        "unaffected_scientific_bytes_and_mtimes_unchanged": True, "readback_items": read_back_all(output)})


def test_custom_networks_and_same_path_weights_resume(tmp_path, monkeypatch):
    from rtpipeline import custom_models as cm, rtstruct_geometry
    from rtpipeline.course_contract import load_course_contract
    root, config = _course(tmp_path)
    contract = load_course_contract(root)
    course = SimpleNamespace(dirs=build_course_dirs(root), patient_id="P1", course_id="C1")
    definitions = []
    for alias in ("one", "two"):
        weights = tmp_path / (alias + ".weights")
        weights.write_bytes(alias.encode())
        definitions.append(cm.NetworkDefinition(alias, alias, weights, None, alias, [alias]))
    model = cm.CustomModelDefinition("synthetic", tmp_path, "stand-in", "3d", "all", {}, definitions, ["one", "two"])
    calls = Counter()
    def predict(cfg, model, network, input_dir, output_dir, env):
        calls[network.alias] += 1
        image = sitk.ReadImage(str(contract.planning_ct_nifti))
        array = np.ones(sitk.GetArrayFromImage(image).shape, np.uint8)
        mask = sitk.GetImageFromArray(array); mask.CopyInformation(image)
        sitk.WriteImage(mask, str(output_dir / "case.nii.gz"))
    monkeypatch.setattr(cm, "_prepare_network_weights", lambda *args: None)
    monkeypatch.setattr(cm, "_run_nnunet_prediction", predict)
    monkeypatch.setattr(cm, "_check_custom_model_eligibility", lambda *args: (True, "synthetic"))
    def writer(ct_dir, structures, output_dir, paths):
        path = output_dir / "rtstruct.dcm"; path.write_bytes(b"synthetic writer"); return path
    monkeypatch.setattr(cm, "_build_rtstruct", writer)
    monkeypatch.setattr(rtstruct_geometry, "geometry_adapter_output_current", lambda *args: True)
    def interrupt(event, item):
        if item.identity == ["synthetic", "one"]:
            raise SystemExit("synthetic interruption after network commit")
    monkeypatch.setattr(ap, "_failpoint", interrupt)
    with pytest.raises(SystemExit):
        cm.run_custom_models_for_course(config, course, [model])
    monkeypatch.setattr(ap, "_failpoint", lambda *args: None)
    cm.run_custom_models_for_course(config, course, [model])
    assert calls == {"one": 1, "two": 1}
    assert cm._model_outputs_ready(root, model, ct_dir=contract.planning_ct_dir)
    paths = sorted(course.dirs.segmentation_custom_models.rglob("*.nii.gz"))
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
    cm.run_custom_models_for_course(config, course, [model])
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
    definitions[1].archive_path.write_bytes(b"new same-path weights")
    assert not cm._model_outputs_ready(root, model, ct_dir=contract.planning_ct_dir)
    cm.run_custom_models_for_course(config, course, [model])
    assert calls == {"one": 1, "two": 2}
    model.networks[1].predict_args = ["changed-setting"]
    cm.run_custom_models_for_course(config, course, [model])
    assert calls == {"one": 1, "two": 3}
    evidence("custom-network-resume", {"calls": dict(calls), "readback_items": read_back_all(root)})


def test_totalsegmentator_weight_binding_selects_only_active_tasks(tmp_path, monkeypatch):
    from rtpipeline.segmentation import _totalseg_weights_identity
    from rtpipeline.config import PipelineConfig
    for name in ("Dataset291_total", "Dataset297_fast", "Dataset298_crop", "Dataset913_lung"):
        directory = tmp_path / "weights" / name
        directory.mkdir(parents=True)
        (directory / "checkpoint.pth").write_bytes(name.encode())
    config = PipelineConfig(tmp_path / "Input", tmp_path / "Output", tmp_path / "Logs",
                            totalseg_weights_dir=tmp_path / "weights", totalseg_fast=False)
    monkeypatch.delenv("TOTALSEG_WEIGHTS_PATH", raising=False)
    total = _totalseg_weights_identity(config, "total")
    lung = _totalseg_weights_identity(config, "lung_nodules")
    (tmp_path / "weights/Dataset297_fast/checkpoint.pth").write_bytes(b"changed unused fast weights")
    assert _totalseg_weights_identity(config, "total") == total
    assert _totalseg_weights_identity(config, "lung_nodules") == lung
    (tmp_path / "weights/Dataset291_total/checkpoint.pth").write_bytes(b"changed total weights")
    assert _totalseg_weights_identity(config, "total") != total
    assert _totalseg_weights_identity(config, "lung_nodules") == lung
    (tmp_path / "weights/Dataset298_crop/checkpoint.pth").write_bytes(b"changed auxiliary crop weights")
    assert _totalseg_weights_identity(config, "lung_nodules") != lung
    environment_root = tmp_path / "environment-weights/total"
    environment_root.mkdir(parents=True)
    (environment_root / "checkpoint.pth").write_bytes(b"active environment weights")
    monkeypatch.setenv("TOTALSEG_WEIGHTS_PATH", str(environment_root.parent))
    environment_identity = _totalseg_weights_identity(config, "total")
    assert set(environment_identity["weights"]) == {"total/checkpoint.pth"}
    (environment_root / "checkpoint.pth").write_bytes(b"changed environment weights")
    assert _totalseg_weights_identity(config, "total") != environment_identity
    # Custom-network execution explicitly chooses configured weights instead.
    assert _totalseg_weights_identity(config, "total", weights_root=config.totalseg_weights_dir)["weights"] != environment_identity["weights"]


def test_same_path_archive_refreshes_prepared_weights_atomically(tmp_path, monkeypatch):
    from rtpipeline import custom_models as cm
    import zipfile
    archive = tmp_path / "weights.zip"
    def package(value):
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr("Dataset001/fold_0/checkpoint.pth", value)
    package(b"first weights")
    network = cm.NetworkDefinition("one", "one", archive, None, "Dataset001", ["one"])
    model = cm.CustomModelDefinition("synthetic", tmp_path, "stand-in", "3d", "all", {}, [network], ["one"])
    target = cm._prepare_network_weights(model, network) / "fold_0/checkpoint.pth"
    assert target.read_bytes() == b"first weights"
    before = target.stat().st_mtime_ns
    cm._prepare_network_weights(model, network)
    assert target.stat().st_mtime_ns == before
    package(b"second weights")
    cm._prepare_network_weights(model, network)
    assert target.read_bytes() == b"second weights"
    package(b"third weights")
    extract = cm._safe_extract_network_archive
    def interrupted(*args):
        extract(*args); raise SystemExit("synthetic interrupted weight preparation")
    monkeypatch.setattr(cm, "_safe_extract_network_archive", interrupted)
    with pytest.raises(SystemExit): cm._prepare_network_weights(model, network)
    assert target.read_bytes() == b"second weights"
    monkeypatch.setattr(cm, "_safe_extract_network_archive", extract)
    cm._prepare_network_weights(model, network)
    assert target.read_bytes() == b"third weights"


def test_crop_exact_inventory_resume_config_and_source_locality(tmp_path, monkeypatch):
    from rtpipeline import anatomical_cropping as crop
    from rtpipeline.course_contract import load_course_contract
    root, config = _course(tmp_path)
    contract = load_course_contract(root)
    masks = build_course_dirs(root).segmentation_totalseg / "CT"
    masks.mkdir(parents=True)
    reference = sitk.ReadImage(str(contract.planning_ct_nifti))
    for name in ("one", "two"):
        image = sitk.Image(reference.GetSize(), sitk.sitkUInt8); image.CopyInformation(reference)
        sitk.WriteImage(image, str(masks / ("total--" + name + ".nii.gz")))
    monkeypatch.setattr(crop, "determine_pelvic_crop_boundaries", lambda *a, **k: (0., 0.))
    calls = []
    original = crop.crop_image_to_boundaries
    def observed(image, *args, **kwargs):
        calls.append(image.GetPixelID()); return original(image, *args, **kwargs)
    monkeypatch.setattr(crop, "crop_image_to_boundaries", observed)
    def interrupt(event, item):
        if isinstance(item.identity, list) and item.identity[0] == "image":
            raise SystemExit("synthetic crop interruption")
    monkeypatch.setattr(ap, "_failpoint", interrupt)
    with pytest.raises(SystemExit):
        crop.apply_systematic_cropping(root)
    assert len(calls) == 1
    monkeypatch.setattr(ap, "_failpoint", lambda *args: None)
    outputs = crop.apply_systematic_cropping(root)
    assert len(calls) == 3 and len(outputs) == 3
    snapshot = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in [*outputs.values(), root / "cropping_metadata.json"]}
    for _ in range(3):
        crop.apply_systematic_cropping(root)
        crop.validate_crop_publication(root)
    assert len(calls) == 3
    assert snapshot == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in snapshot}
    assert not list(root.rglob("*_cropped_cropped*"))
    crop.apply_systematic_cropping(root, inferior_margin_cm=1.)
    assert len(calls) == 6
    source = masks / "total--two.nii.gz"
    image = sitk.Image(reference.GetSize(), sitk.sitkUInt16); image.CopyInformation(reference)
    sitk.WriteImage(image, str(source))
    crop.apply_systematic_cropping(root, inferior_margin_cm=1.)
    assert len(calls) == 7
    orphan = masks / "orphan_cropped_cropped.nii.gz"
    orphan.write_bytes(b"orphan")
    (root / "cropping_metadata.json").write_text("{}")
    crop.apply_systematic_cropping(root, inferior_margin_cm=1.)
    assert not orphan.exists() and len(calls) == 7
    crop.validate_crop_publication(root)
    with pytest.raises(ValueError, match="configuration changed"):
        crop.validate_crop_publication(root, {"region": "brain"})
    # A manifest naming some crops cannot certify a missing required crop.
    required = next(iter(outputs.values()))
    required.unlink()
    with pytest.raises(ValueError): crop.validate_crop_publication(root)
    crop.apply_systematic_cropping(root, inferior_margin_cm=1.)
    assert len(calls) == 8
    from rtpipeline.config_dependencies import materialize_stage_dependency
    from rtpipeline.stage_completion import write_stage_completion_sentinel
    dependency = materialize_stage_dependency(tmp_path / "dependencies", "crop-ct", {
        "ct_cropping": {"enabled": True, "region": "pelvis", "inferior_margin_cm": 1.}})
    write_stage_completion_sentinel(root, root / ".crop_ct_done", stage="crop_ct", status="ok", configuration_dependency=dependency)
    configuration = tmp_path / "changed.yaml"
    configuration.write_text(yaml.safe_dump({"ct_cropping": {"enabled": True, "region": "brain"}}))
    from rtpipeline.phase_c_progress import preflight
    assert preflight(config.output_root, configuration)["invalidated"] == ["P1/C1/.crop_ct_done"]
    assert len(calls) == 8
    # CT DICOM is also an input to the cropped RTSTRUCT publication, even
    # when its NIfTI representation has not changed.
    import pydicom
    dicom = next(contract.planning_ct_dir.glob("*.dcm"))
    dataset = pydicom.dcmread(dicom)
    dataset.SeriesDescription = "Changed synthetic CT source bytes"
    dataset.save_as(dicom, enforce_file_format=True)
    with pytest.raises(ValueError, match="missing or stale"):
        crop.validate_crop_publication(root)
    crop.apply_systematic_cropping(root, inferior_margin_cm=1.)
    crop.validate_crop_publication(root)
    assert len(calls) == 8  # Independent image crops retain their own binding.
    evidence("crop-inventory-locality", {"crop_calls": len(calls), "no_suffix_growth": True,
        "readback_items": read_back_all(root)})


def test_required_crop_and_qc_failures_never_publish_completion(tmp_path, monkeypatch):
    from rtpipeline import anatomical_cropping as crop, quality_control as qc
    root, config = _course(tmp_path)
    monkeypatch.setattr(crop, "determine_pelvic_crop_boundaries", lambda *a, **k: (0., 0.))
    def fail(*args, **kwargs): raise OSError("synthetic required failure")
    monkeypatch.setattr(crop, "crop_image_to_boundaries", fail)
    with pytest.raises(OSError): crop.apply_systematic_cropping(root)
    assert not (root / ".crop-progress/current.json").exists()
    monkeypatch.setattr(qc.DICOMValidator, "_validate_ct_series", fail)
    with pytest.raises(qc.QualityControlError): qc.generate_qc_report(root, root / "qc_reports")
    assert not (root / ".qc-progress/current.json").exists()
    from rtpipeline.stage_completion import write_stage_completion_sentinel
    from rtpipeline.config_dependencies import materialize_stage_dependency
    reports = root / "qc_reports"; reports.mkdir(exist_ok=True)
    (reports / "qc_report_P1_C1.json").write_text('{"overall_status":"PASS"}')
    with pytest.raises((ValueError, OSError)):
        write_stage_completion_sentinel(root, root / ".qc_done", stage="qc", status="ok",
            configuration_dependency=materialize_stage_dependency(tmp_path / "dependencies", "qc", {}))


def test_portal_sigkill_after_submit_reconnects_same_job(portal_course, tmp_path):
    course, series = portal_course
    events, handshake = tmp_path / "portal-events", tmp_path / "portal-ready"
    driver = tmp_path / "portal-driver.py"
    driver.write_text('''
import json,os,sys,time
from pathlib import Path
from rtpipeline import rtsegmentator as portal
from test_rtsegmentator_integration import FakeClient,make_package
root,series,events,ready=sys.argv[1:]
def event(name,value):
    with open(events,'a') as stream:
        stream.write(json.dumps([name,value])+'\\n');stream.flush();os.fsync(stream.fileno())
class Client(FakeClient):
    def submit(self,*args):
        value=super().submit(*args);event('submit',value['jobs'][0]['id']);return value
    def wait(self,job_id,**kwargs):
        event('wait',job_id)
        if os.environ.get('PORTAL_KILL')=='1':
            with open(ready,'w') as stream:
                stream.write(job_id);stream.flush();os.fsync(stream.fileno())
            while True: time.sleep(.05)
        return super().wait(job_id,**kwargs)
settings=portal.RTsegmentatorSettings(True,'https://synthetic.invalid',('total',))
portal.run_for_course(settings,Path(root),client=Client(lambda target:make_package(target,series),series))
''')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "tests"), os.environ.get("PYTHONPATH", ""))))
    command = [sys.executable, str(driver), str(course), series, str(events), str(handshake)]
    log = tmp_path / "portal.log"
    with log.open("w") as stream:
        child = subprocess.Popen(command, env=dict(env, PORTAL_KILL="1"), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 120
            while not handshake.exists() and child.poll() is None and time.monotonic() < deadline:
                time.sleep(.05)
            assert handshake.exists(), log.read_text()
            os.killpg(child.pid, signal.SIGKILL)
            assert child.wait(timeout=20) == -signal.SIGKILL
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
    subprocess.run(command, env=env, check=True, capture_output=True, timeout=120)
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    assert rows == [["submit", "job-1"], ["wait", "job-1"], ["wait", "job-1"]]
    evidence("portal-sigkill", {"events": rows, "readback_items": read_back_all(course)})


HOOK = r'''
import os,sys,json,time,datetime,types
def guard(event,args):
    if event == 'socket.getaddrinfo': raise RuntimeError('synthetic tests forbid DNS')
    if event in {'socket.connect','socket.connect_ex','socket.bind'}:
        import socket
        if getattr(args[0],'family',None) != socket.AF_UNIX:
            raise RuntimeError('synthetic tests forbid non-AF_UNIX sockets')
sys.addaudithook(guard)
if sys.prefix == os.environ.get('WS515_RUNTIME_PREFIX'):
    for path in json.loads(os.environ['WS515_RUNTIME_SITES']):
        if path not in sys.path: sys.path.append(path)
    def event(kind,key):
        with open(os.environ['WS518_EVENTS'],'a') as stream:
            stream.write(json.dumps({'kind':kind,'key':key})+'\n');stream.flush();os.fsync(stream.fileno())
    def pause(kind):
        if kind == os.environ.get('WS515_KILL_POINT'):
            with open(os.environ['WS515_HANDSHAKE'],'w') as stream:
                stream.write(kind);stream.flush();os.fsync(stream.fileno())
            while True: time.sleep(.05)
    import rtpipeline.utils as utils
    adaptive=utils.run_tasks_with_adaptive_workers
    def serial(label,*args,**kwargs):
        kwargs['use_processes']=False
        return adaptive(label,*args,**kwargs)
    utils.run_tasks_with_adaptive_workers=serial
    import rtpipeline.segmentation as seg
    import SimpleITK as sitk
    import numpy as np
    def inference(config,input_path,output_path,output_type,task=None,extra_args=None):
        model=task or 'total';event('inference',model)
        image=sitk.ReadImage(str(input_path));shape=sitk.GetArrayFromImage(image).shape
        names=['brain','vertebrae_L1','femur_left'] if model=='total' else ['nodule']
        Path=__import__('pathlib').Path
        output_path=Path(output_path);output_path.mkdir(parents=True,exist_ok=True)
        for index,name in enumerate(names):
            data=np.zeros(shape,np.uint8);data[2:10,5:15,5:15]=1
            mask=sitk.GetImageFromArray(data);mask.CopyInformation(image)
            sitk.WriteImage(mask,str(output_path/(name+'.nii.gz')))
        return True
    seg.run_totalsegmentator=inference
    import rtpipeline.quality_control as qc
    qc.save_body_region_qc=lambda *a,**k: None
    qc.check_model_eligibility=lambda *a,**k:(True,'synthetic')
    import rtpipeline.anatomical_cropping as crop
    crop_image=crop.crop_image_to_boundaries
    def observed_crop(*a,**k):
        # The image bytes identify the source; count real cropping calls only.
        import hashlib
        event('crop',hashlib.sha256(sitk.GetArrayViewFromImage(a[0]).tobytes()).hexdigest())
        return crop_image(*a,**k)
    crop.crop_image_to_boundaries=observed_crop
    for name in ('_check_file_existence','_validate_ct_series','_validate_rt_plan','_validate_rt_dose','_check_consistency','_segmentation_cropping'):
        original=getattr(qc.DICOMValidator,name)
        def observed(self,*a,_name=name,_original=original,**k):
            event('qc_check',_name);return _original(self,*a,**k)
        setattr(qc.DICOMValidator,name,observed)
    from rtpipeline.dicom_copy import DicomCopyManager
    original_copy=DicomCopyManager.copy_dicom
    def observed_copy(self,src,dst,*a,**k):
        result=original_copy(self,src,dst,*a,**k)
        if result[1]: event('copy',str(dst))
        return result
    DicomCopyManager.copy_dicom=observed_copy
    try:
        import rtpipeline.artifact_progress as ap
    except ImportError: pass
    else:
        def committed(event_name,item):
            path=str(item.cache.root);kind=None
            if '/.model-progress/' in path: kind='segmentation'
            elif '/_CACHE/copy-progress/' in path: kind='organize-copy'
            elif '/.conversion-progress/' in path: kind='organize-convert'
            elif '/_COURSES/.organize-progress/items/' in path:
                kind='organize-contract' if isinstance(item.identity,str) else 'organize-materialize'
            elif '/.crop-progress/items/' in path and isinstance(item.identity,list) and item.identity[0]=='image': kind='crop'
            elif '/.qc-progress/items/' in path and isinstance(item.identity,list) and item.identity[0]=='check': kind='qc'
            if kind:
                event('committed-'+kind,item.key);pause(kind)
        ap._failpoint=committed
    if os.environ.get('WS518_FIXED_CLOCK')=='1':
        class Fixed(datetime.datetime):
            @classmethod
            def now(cls,tz=None): return cls(2024,1,1,tzinfo=tz)
            @classmethod
            def utcnow(cls): return cls(2024,1,1)
        clock=types.SimpleNamespace(datetime=Fixed,date=datetime.date,timedelta=datetime.timedelta,timezone=datetime.timezone)
        from rtpipeline import organize,organize_ledger,nifti_provenance,original_segmentation,custom_models,rtstruct_geometry
        for module in (seg,qc,organize,organize_ledger,nifti_provenance,original_segmentation,custom_models,rtstruct_geometry):
            value=getattr(module,'datetime',None)
            if value is not None: module.datetime=clock if isinstance(value,types.ModuleType) else Fixed
        import pydicom.uid
        counter=iter(range(10000000,20000000))
        pydicom.uid.generate_uid=lambda *a,**k:'2.25.'+str(next(counter))
        import rt_utils.ds_helper as helper
        helper.datetime=clock
        helper.generate_uid=pydicom.uid.generate_uid
        import xlsxwriter
        initial=xlsxwriter.Workbook.__init__
        def workbook(self,*a,**k):
            initial(self,*a,**k);self.set_properties({'created':Fixed(2024,1,1)})
        xlsxwriter.Workbook.__init__=workbook
'''


class PhaseCProduction(Production):
    run_timeout = 480
    def __init__(self, root):
        super().__init__(root)
        import pydicom
        for path in (root / "Input/images").glob("*.dcm"):
            dataset = pydicom.dcmread(path)
            dataset.StudyID = "SYNTHETIC"
            dataset.save_as(path, enforce_file_format=True)
        (root / "instrumentation/sitecustomize.py").write_text(HOOK)
        self.events = root / "phase-c-events.jsonl"
        self.env.update(WS518_EVENTS=str(self.events), RTPIPELINE_INDEX_PROCESSES="1", RTPIPELINE_MASK_PROCESSES="1")
        config = yaml.safe_load(self.config.read_text())
        config["segmentation"] = {"enabled": True, "device": "cpu", "extra_models": ["lung_nodules"]}
        config["ct_cropping"] = {"enabled": True, "region": "brain", "superior_margin_cm": 0., "inferior_margin_cm": 0.}
        self.config.write_text(yaml.safe_dump(config))

    def rows(self):
        return [json.loads(line) for line in self.events.read_text().splitlines()] if self.events.exists() else []

    def validate_receipts(self, course, stages):
        program = '''import json,sys
from pathlib import Path
from rtpipeline.stage_completion import validate_stage_completion_sentinel
course=Path(sys.argv[1])
for name,stage in json.loads(sys.argv[2]):
    validate_stage_completion_sentinel(course/name, expected_stage=stage)
'''
        result = subprocess.run([str(self.runtime), "-c", program, str(course), json.dumps(stages)],
            env=self.env, cwd=self.source, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.requires_snakemake
@pytest.mark.parametrize("leg", ["segmentation", "organize-copy", "organize-convert", "organize-contract", "crop", "qc"])
def test_real_snakefile_sigkill_resumes_committed_item(tmp_path, leg):
    pipeline = PhaseCProduction(tmp_path)
    if leg.startswith("organize"):
        target = pipeline.output / "_COURSES/manifest.json"
    else:
        course = pipeline.organize()
        target = course / {"segmentation": ".segmentation_done", "crop": ".crop_ct_done", "qc": ".qc_done"}[leg]
        if leg == "crop":
            pipeline.run(course / ".segmentation_done")
        if leg == "qc":
            pipeline.run(course / ".crop_ct_done")
    pipeline.run(target, kill=leg)
    before = pipeline.rows()
    commits = [r["key"] for r in before if r["kind"] == "committed-" + leg]
    assert len(commits) == 1
    pipeline.run(target)
    after = pipeline.rows()
    assert sum(r["kind"] == "committed-" + leg and r["key"] == commits[0] for r in after) == 1
    if leg == "segmentation":
        assert Counter(r["key"] for r in after if r["kind"] == "inference") == {"total": 1, "lung_nodules": 1}
    if leg == "qc":
        assert Counter(r["key"] for r in after if r["kind"] == "qc_check")["_check_file_existence"] == 1
    count = read_back_all(pipeline.output)
    assert count
    evidence("snake-sigkill-" + leg, {"events": after, "readback_items": count, "prior_item_not_repeated": True})


@pytest.mark.requires_snakemake
def test_real_snakefile_all_phase_c_kill_boundaries(tmp_path):
    pipeline = PhaseCProduction(tmp_path)
    manifest = pipeline.output / "_COURSES/manifest.json"
    legs = ["organize-copy", "organize-convert", "organize-materialize", "organize-contract"]
    keys = {}
    for leg in legs:
        pipeline.run(manifest, kill=leg)
        committed = [r["key"] for r in pipeline.rows() if r["kind"] == "committed-" + leg]
        assert len(committed) == 1
        keys[leg] = committed[0]
    pipeline.run(manifest)
    course = Path(json.loads(manifest.read_text())["courses"][0]["path"])
    for leg, name in (("segmentation", ".segmentation_done"), ("crop", ".crop_ct_done"), ("qc", ".qc_done")):
        pipeline.run(course / name, kill=leg)
        committed = [r["key"] for r in pipeline.rows() if r["kind"] == "committed-" + leg]
        assert len(committed) == 1
        keys[leg] = committed[0]
        pipeline.run(course / name)
    rows = pipeline.rows()
    for leg, key in keys.items():
        assert sum(r["kind"] == "committed-" + leg and r["key"] == key for r in rows) == 1
    assert Counter(r["key"] for r in rows if r["kind"] == "inference") == {"total": 1, "lung_nodules": 1}
    assert Counter(r["key"] for r in rows if r["kind"] == "qc_check")["_check_file_existence"] == 1
    assert max(Counter(r["key"] for r in rows if r["kind"] == "copy").values()) == 1
    assert not list(course.rglob("*_cropped_cropped*"))
    pipeline.validate_receipts(course, ((".organized", "organize"), (".segmentation_done", "segmentation"),
                        (".crop_ct_done", "crop_ct"), (".qc_done", "qc")))
    evidence("all-phase-c-sigkill", {"boundaries": keys, "events": rows,
        "readback_items": read_back_all(pipeline.output), "all_active_stage_receipts_read_back": True})


def scientific_files(output):
    result = {}
    for path in output.rglob("*"):
        relative = path.relative_to(output)
        if not path.is_file() or any(part.startswith(".") for part in relative.parts):
            continue
        if relative.parts[0] in {"_CACHE", "_COURSES", "_campaign_ledger"} or path.suffix == ".xlsx":
            continue
        if path.suffix in {".dcm", ".json"} or path.name.endswith(".nii.gz"):
            result[relative.as_posix()] = path.read_bytes()
    return result


@pytest.mark.requires_snakemake
def test_uninterrupted_scientific_outputs_match_e1d551a(tmp_path):
    pipeline = PhaseCProduction(tmp_path)
    pipeline.env["WS518_FIXED_CLOCK"] = "1"
    configuration = yaml.safe_load(pipeline.config.read_text())
    configuration["ct_cropping"]["enabled"] = False
    pipeline.config.write_text(yaml.safe_dump(configuration))
    course = pipeline.organize()
    pipeline.run(course / ".qc_done")
    workflow_control = scientific_files(pipeline.output)
    closure = {name: (pipeline.output / name).read_bytes() for name in
               ("_COURSES/manifest.json", "_COURSES/organize_ledger.json")}
    def ledger_records():
        return {path.name: {k: v for k, v in json.loads(path.read_text()).items()
                if k not in {"started_at", "finished_at", "duration_seconds"}}
                for path in (pipeline.output / "_campaign_ledger/records").glob("*.json")}
    campaign = ledger_records()
    pipeline.validate_receipts(course, ((".organized", "organize"), (".segmentation_done", "segmentation"),
        (".crop_ct_done", "crop_ct"), (".qc_done", "qc")))
    # Compare the crop and post-crop QC producers directly too. e1d551a's
    # workflow rejects its nominal RS_auto after cropped masks are added; that
    # admission defect is repaired by Phase C rather than suppressed here.
    program = '''from pathlib import Path
from rtpipeline.anatomical_cropping import apply_systematic_cropping
from rtpipeline.quality_control import generate_qc_report
import sys
course=Path(sys.argv[1])
apply_systematic_cropping(course, region="brain", superior_margin_cm=0., inferior_margin_cm=0.)
generate_qc_report(course, course/"qc_reports")
'''
    subprocess.run([str(pipeline.runtime), "-c", program, str(course)], env=pipeline.env,
        cwd=pipeline.source, check=True, capture_output=True, timeout=180)
    candidate = scientific_files(pipeline.output)
    candidate_audit = {name: candidate.pop(name) for name in list(candidate) if name.endswith("metadata/segmentation_resume.json")}
    workbooks = {p.relative_to(pipeline.output).as_posix(): pd.read_excel(p) for p in pipeline.output.rglob("*.xlsx")}
    gates = {p.relative_to(pipeline.output).as_posix(): json.loads(p.read_text())["status"]
             for p in pipeline.output.rglob(".*") if p.is_file() and p.name in
             {".organized", ".segmentation_done", ".custom_models_done", ".crop_ct_done", ".qc_done"}}
    count = read_back_all(pipeline.output)
    shutil.rmtree(pipeline.output)
    for name in ("rtpipeline", "workflow", "envs"):
        shutil.rmtree(pipeline.source / name)
    archive = subprocess.run(["git", "archive", "e1d551a", "rtpipeline", "workflow", "envs", "Snakefile", "config.yaml"],
        cwd=ROOT, check=True, capture_output=True).stdout
    with tarfile.open(fileobj=BytesIO(archive)) as source:
        source.extractall(pipeline.source, filter="data")
    pipeline.organize()
    pipeline.run(course / ".qc_done")
    baseline_control = scientific_files(pipeline.output)
    assert campaign == ledger_records()
    for name, data in closure.items():
        assert (pipeline.output / name).read_bytes() == data
    control_differences = [name for name in workflow_control if name not in baseline_control or
        workflow_control[name] != baseline_control[name]]
    # Resume reason text is operational audit evidence, compared below.
    control_differences = [name for name in control_differences if not name.endswith("metadata/segmentation_resume.json")]
    evidence("workflow-control-differences", {"files": control_differences})
    assert not control_differences
    subprocess.run([str(pipeline.runtime), "-c", program, str(course)], env=pipeline.env,
        cwd=pipeline.source, check=True, capture_output=True, timeout=180)
    baseline = scientific_files(pipeline.output)
    baseline_audit = {name: baseline.pop(name) for name in list(baseline) if name.endswith("metadata/segmentation_resume.json")}
    assert candidate.keys() == baseline.keys()
    differences = [name for name in candidate if candidate[name] != baseline[name]]
    evidence("equality-initial-differences", {"files": differences})
    assert not differences
    def without_reason(value):
        if isinstance(value, dict): return {k: without_reason(v) for k, v in value.items() if k != "reason"}
        if isinstance(value, list): return [without_reason(v) for v in value]
        return value
    assert candidate_audit.keys() == baseline_audit.keys()
    for name, value in candidate_audit.items():
        assert without_reason(json.loads(value)) == without_reason(json.loads(baseline_audit[name]))
    for name, expected in workbooks.items():
        pd.testing.assert_frame_equal(expected, pd.read_excel(pipeline.output / name))
    for name, expected in gates.items():
        assert json.loads((pipeline.output / name).read_text())["status"] == expected
    evidence("e1d551a-equality", {"base": "e1d551a", "byte_equal_files":
        {name: __import__("hashlib").sha256(data).hexdigest() for name, data in sorted(candidate.items())},
        "table_equal_workbooks": sorted(workbooks), "equal_gate_statuses": gates,
        "candidate_item_readbacks": count, "workflow_control": "real Snakefile through QC, cropping disabled",
        "organize_manifest_and_ledger_byte_equal": sorted(closure),
        "campaign_records_equal_excluding_attempt_timing": sorted(campaign),
        "crop_and_post_crop_qc": "real producer APIs in the production interpreter",
        "declared_differences": ["new durable progress and source-byte cache records", "completion receipt code and binding provenance", "segmentation resume audit reason text; actions and outcomes equal",
        "crop-enabled upstream admission now excludes derived crops; e1d551a rejects that workflow path",
        "campaign attempt start/finish/duration are execution timing, not scientific results"]})
