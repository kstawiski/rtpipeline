"""Real native MR helpers, generated DICOM/voxel masks, SIGKILL and base bytes."""
from collections import Counter
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tarfile
import time

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
import yaml

from test_dvh_durable_resume import ROOT
from test_phase_b_durable_resume import PhaseBProduction, read_all_commits


MR_HOOK = r'''
import os,sys,json,time,hashlib,datetime

def guard(event,args):
    if event in {'socket.connect','socket.connect_ex','socket.bind','socket.getaddrinfo'}:
        raise RuntimeError('synthetic tests forbid network')
sys.addaudithook(guard)
if sys.prefix == os.environ.get('RTPIPELINE_RADIOMICS_ENV_PREFIX'):
    sys.path[:] = [os.environ['WS516_RAD_SITES'] if p.startswith(sys.prefix+'/lib/python') and p.endswith('site-packages') else p for p in sys.path]
    import xlsxwriter
    init = xlsxwriter.Workbook.__init__
    def fixed_workbook(self,*args,**kwargs):
        init(self,*args,**kwargs)
        self.set_properties({'created':datetime.datetime(2024,1,1)})
    xlsxwriter.Workbook.__init__ = fixed_workbook
    import rtpipeline.radiomics as nominal
    nominal.new_run_identifier = lambda: 'synthetic-mr-run'
    from radiomics import featureextractor
    execute = featureextractor.RadiomicsFeatureExtractor.execute
    def event(value):
        with open(os.environ['WS516_EVENTS'],'a') as stream:
            stream.write(json.dumps(value)+'\n');stream.flush();os.fsync(stream.fileno())
    def observed(self,image,mask,*args,**kwargs):
        import SimpleITK as sitk
        event({'event':'mr_extract','key':hashlib.sha256(sitk.GetArrayFromImage(mask).tobytes()).hexdigest()})
        return execute(self,image,mask,*args,**kwargs)
    featureextractor.RadiomicsFeatureExtractor.execute = observed
    try:
        import rtpipeline.extraction_progress as ep
    except ImportError:  # The actual base predates bound extraction items.
        pass
    else:
        commit = ep.BoundItem.commit
        def committed(self,value):
            result = commit(self,value)
            event({'event':'mr_commit','key':self.key})
            if os.environ.get('WS516_MR_KILL') == '1':
                with open(os.environ['WS515_HANDSHAKE'],'w') as stream:
                    stream.write(self.key);stream.flush();os.fsync(stream.fileno())
                while True: time.sleep(.1)
            return result
        ep.BoundItem.commit = committed
'''

MR_RUN = r'''
import json,sys
from pathlib import Path
from types import SimpleNamespace
from rtpipeline.config import PipelineConfig
from rtpipeline.radiomics import radiomics_for_course_mr,radiomics_for_mr_series,MRSeries
root=Path(sys.argv[1]); mode=sys.argv[2]
info=json.loads((root/'mr.json').read_text())
config=PipelineConfig(dicom_root=root/'Input',output_root=root/'Output',logs_root=root/'Logs',
 max_workers_override=1,radiomics_params_file_mr=root/'parameters.yaml',resume=True)
if mode=='course':
 output=radiomics_for_course_mr(config,SimpleNamespace(dirs=__import__('rtpipeline.layout',fromlist=['build_course_dirs']).build_course_dirs(Path(info['course'])),patient_id='SYNTHETIC',course_id='C1'))
else:
 output=radiomics_for_mr_series(config,MRSeries('SYNTHETIC',info['series_uid'],Path(info['dicom'])))
assert output is not None
print(output)
'''


def mr_fixture(root, mode):
    pipeline = PhaseBProduction(root)
    # Native extraction runs in the installed NumPy 1/PyRadiomics interpreter.
    (root / 'instrumentation/sitecustomize.py').write_text(MR_HOOK)
    pipeline.env['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
    course = pipeline.output / 'SYNTHETIC/C1'
    series_root = course / 'MR/mr_series'
    dicom = series_root / 'DICOM'
    shutil.copytree(root / 'Input/images', dicom)
    uid = None
    for path in sorted(dicom.glob('*.dcm')):
        ds = pydicom.dcmread(path)
        ds.Modality = 'MR'
        ds.SOPClassUID = pydicom.uid.MRImageStorage
        ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
        ds.SeriesDescription = 'Synthetic T2'
        ds.save_as(path, enforce_file_format=True)
        uid = str(ds.SeriesInstanceUID)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(dicom), uid))
    image = reader.Execute()
    nifti = series_root / 'NIFTI/image.nii.gz'
    nifti.parent.mkdir()
    sitk.WriteImage(image, str(nifti))
    (nifti.parent / 'image.metadata.json').write_text(json.dumps({'modality':'MR',
        'nifti_path':str(nifti),'source_directory':str(dicom),'series_instance_uid':uid}))
    seg = series_root / 'Segmentation_TotalSegmentator'
    seg.mkdir()
    labels = np.zeros(tuple(reversed(image.GetSize())), np.uint8)
    for index, name in enumerate(('liver','kidney','spleen'), 1):
        mask = np.zeros_like(labels)
        mask[2:6, 3:9, 2+index*4:5+index*4] = 1
        labels[mask != 0] = index
        volume = sitk.GetImageFromArray(mask);volume.CopyInformation(image)
        sitk.WriteImage(volume, str(seg / f'total_mr--{name}.nii.gz'))
    output = pipeline.output / 'SYNTHETIC' / f'MR_{uid}'
    if mode == 'series_manual':
        rs = root / 'Input/SYNTHETIC/RS.dcm'
        rs.parent.mkdir()
        ds = pydicom.dcmread(root / 'Input/struct.dcm')
        ds.FrameOfReferenceUID = pydicom.dcmread(next(dicom.glob('*.dcm'))).FrameOfReferenceUID
        for roi in ds.ROIContourSequence:
            for contour in roi.ContourSequence:
                for ref in contour.ContourImageSequence:
                    ref.ReferencedSOPClassUID = pydicom.uid.MRImageStorage
        ds.save_as(rs, enforce_file_format=True)
    elif mode == 'series_auto':
        auto = output / 'TotalSegmentator_total_mr_NIFTI'
        auto.mkdir(parents=True)
        volume = sitk.GetImageFromArray(labels);volume.CopyInformation(image)
        sitk.WriteImage(volume, str(auto / 'mr_total_multilabel.nii.gz'))
        (auto / 'mr_total_segmentations.json').write_text(json.dumps({'liver':1,'kidney':2,'spleen':3}))
    (root / 'mr.json').write_text(json.dumps({'course':str(course),'series_uid':uid,'dicom':str(dicom)}))
    pipeline.env['WS516_EVENTS'] = str(root / 'mr-events.jsonl')
    if mode == 'course':
        output = course / 'MR/radiomics_mr.xlsx'
    else:
        output = output / 'radiomics_features_MR.xlsx'
    return pipeline, output


def run_mr(pipeline, mode, *, kill=False):
    pipeline.handshake.unlink(missing_ok=True)
    env = dict(pipeline.env, WS516_MR_KILL='1' if kill else '')
    command = [str(pipeline.radiomics/'bin/python'), '-c', MR_RUN, str(pipeline.root), mode]
    log = pipeline.root / 'mr.log'
    with log.open('a') as stream:
        child = subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            if kill:
                deadline = time.monotonic() + 120
                while not pipeline.handshake.exists() and child.poll() is None and time.monotonic() < deadline:
                    time.sleep(.05)
                assert pipeline.handshake.exists(), log.read_text()[-12000:]
                os.killpg(child.pid,signal.SIGKILL)
                assert child.wait(timeout=30) == -signal.SIGKILL
            else:
                assert child.wait(timeout=180) == 0, log.read_text()[-15000:]
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=30)


def mr_events(pipeline):
    path=Path(pipeline.env['WS516_EVENTS'])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.integration
@pytest.mark.parametrize('mode', ['course','series_manual','series_auto'])
def test_native_mr_roi_sigkill_resume_and_base_bytes(tmp_path, mode):
    pipeline, output = mr_fixture(tmp_path, mode)
    clean = tmp_path / 'mr-clean'
    shutil.copytree(pipeline.output,clean)
    run_mr(pipeline,mode)
    candidate = {p.name:p.read_bytes() for p in (output,output.with_suffix('.parquet'))}
    control_calls = Counter(e['key'] for e in mr_events(pipeline) if e['event']=='mr_extract')
    assert sum(control_calls.values()) == 3
    shutil.rmtree(pipeline.output);shutil.copytree(clean,pipeline.output)
    Path(pipeline.env['WS516_EVENTS']).unlink()
    run_mr(pipeline,mode,kill=True)
    interrupted_calls = Counter(e['key'] for e in mr_events(pipeline) if e['event']=='mr_extract')
    assert sum(interrupted_calls.values()) == 1
    assert len([e for e in mr_events(pipeline) if e['event']=='mr_commit']) == 1
    read_all_commits(pipeline.output)
    run_mr(pipeline,mode)
    assert Counter(e['key'] for e in mr_events(pipeline) if e['event']=='mr_extract') == control_calls
    for path in (output,output.with_suffix('.parquet')):
        assert path.read_bytes() == candidate[path.name]
    before=output.with_suffix('.parquet').stat().st_mtime_ns
    run_mr(pipeline,mode)
    assert Counter(e['key'] for e in mr_events(pipeline) if e['event']=='mr_extract') == control_calls
    if mode != 'course':
        assert output.with_suffix('.parquet').stat().st_mtime_ns == before
    readback=read_all_commits(pipeline.output)
    # The same source paths, source bytes, parameters and runtime against base.
    for name in ('rtpipeline','workflow','envs'):
        shutil.rmtree(pipeline.source/name)
    archive=subprocess.run(['git','archive','3f94c63','rtpipeline','workflow','envs','Snakefile','config.yaml'],
        cwd=ROOT,check=True,capture_output=True).stdout
    with tarfile.open(fileobj=BytesIO(archive)) as source:
        source.extractall(pipeline.source,filter='data')
    shutil.rmtree(pipeline.output);shutil.copytree(clean,pipeline.output)
    run_mr(pipeline,mode)
    evidence={'mode':mode,'base':'3f94c63','interrupted_committed_rois':1,
        'control_extractor_calls':sum(control_calls.values()),'resume_extractor_calls':2,
        'repeated_committed_calls':0,'readback':readback,'outputs':{}}
    for path in (output,output.with_suffix('.parquet')):
        actual=path.read_bytes()
        assert actual == candidate[path.name], path
        evidence['outputs'][path.name]={'base_byte_equal':True,'sha256':hashlib.sha256(actual).hexdigest()}
    destination=os.environ.get('WS516_EVIDENCE_DIR')
    if destination:
        Path(destination,f'mr-{mode}-evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
        shutil.copyfile(tmp_path/'mr.log',Path(destination,f'mr-{mode}.log'))
