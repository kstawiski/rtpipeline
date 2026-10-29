"""Synthetic cache corruption and interrupted publication regressions."""
from pathlib import Path
import copy
import gzip
import hashlib
import json
import os
import shutil
import nibabel as nib
import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
from rtpipeline import segmentation as seg, ct_series_hygiene as hygiene
from rtpipeline.config import PipelineConfig
from rtpipeline.course_contract import CourseContractError
from synthetic_ct_series import write_ct_series, uniform_z_positions

BINARY = os.environ.get('RTPIPELINE_TEST_DCM2NIIX') or shutil.which('dcm2niix')

@pytest.fixture(autouse=True)
def require_converter():
    assert BINARY and Path(BINARY).is_file(), 'real dcm2niix is required for publication checks'

def config(tmp):
    return PipelineConfig(dicom_root=tmp/'input', output_root=tmp/'out',
                          logs_root=tmp/'logs', dcm2niix_cmd=BINARY)


def change(path, **attrs):
    ds = pydicom.dcmread(path)
    for key, value in attrs.items():
        setattr(ds, key, value)
    ds.save_as(path, enforce_file_format=True)


def uid(path):
    return str(pydicom.dcmread(path, stop_before_pixels=True).SOPInstanceUID)


def references(path, frame, contour, *, extra=None):
    first = pydicom.dcmread((frame or contour)[0], stop_before_pixels=True)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0'*128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = 'RTSTRUCT'
    ds.StudyInstanceUID = first.StudyInstanceUID
    ds.SeriesInstanceUID = generate_uid()
    def items(paths, additions=()):
        out = []
        for u in [uid(p) for p in paths] + list(additions):
            item = Dataset()
            item.ReferencedSOPClassUID = first.SOPClassUID
            item.ReferencedSOPInstanceUID = u
            out.append(item)
        return Sequence(out)
    series = Dataset()
    series.SeriesInstanceUID = first.SeriesInstanceUID
    series.ContourImageSequence = items(frame, extra or [])
    study = Dataset()
    study.RTReferencedSeriesSequence = Sequence([series])
    ref = Dataset()
    ref.FrameOfReferenceUID = first.FrameOfReferenceUID
    ref.RTReferencedStudySequence = Sequence([study])
    ds.ReferencedFrameOfReferenceSequence = Sequence([ref])
    c = Dataset()
    c.ContourImageSequence = items(contour)
    roi = Dataset()
    roi.ContourSequence = Sequence([c])
    ds.ROIContourSequence = Sequence([roi])
    ds.save_as(path, enforce_file_format=True)
    return path


def mixed(tmp, *, orient=None):
    paths = write_ct_series(tmp/'ct', uniform_z_positions(), signed=False)
    for p in paths:
        attrs = {'AcquisitionNumber': 1}
        if orient is not None:
            attrs['ImageOrientationPatient'] = orient
        if p in paths[3:6]:
            attrs['ImageType'] = ['DERIVED', 'SECONDARY', 'AXIAL']
        change(p, **attrs)
    rs = references(tmp/'RS.dcm', paths, paths[1:8])
    return paths, rs


def duplicate(paths, *, second_acq=True):
    ds = pydicom.dcmread(paths[4])
    ds.SOPInstanceUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.InstanceNumber = 90
    if second_acq:
        ds.AcquisitionNumber = 2
    data = ds.pixel_array.copy()
    data[2, 3] += 7
    ds.PixelData = data.tobytes()
    dest = paths[0].parent/'duplicate.dcm'
    ds.save_as(dest, enforce_file_format=True)
    return dest


def stacks(tmp):
    paths = write_ct_series(tmp/'ct', uniform_z_positions(), signed=False)
    other = []
    for i, p in enumerate(paths):
        change(p, ScanOptions='HELIX', ConvolutionKernel='SHARP',
               ReconstructionDiameter=300., WindowCenter=40., Exposure=200)
        ds = pydicom.dcmread(p)
        ds.SOPInstanceUID = generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.InstanceNumber = i+101
        ds.ConvolutionKernel = 'SMOOTH'
        ds.ReconstructionDiameter = 280.
        ds.WindowCenter = 50.
        ds.Exposure = 180
        data = ds.pixel_array.copy()
        data[2, 3] += 19
        ds.PixelData = data.tobytes()
        dest = tmp/'ct'/f'second_{i:03d}.dcm'
        ds.save_as(dest, enforce_file_format=True)
        other.append(dest)
    rs = references(tmp/'RS.dcm', paths+other, paths[1:6]+other[3:5])
    with pytest.raises(hygiene.CTSeriesHygieneError) as error:
        hygiene.select_ct_instances(tmp/'ct', rs)
    assert error.value.reason_code == 'ct_hygiene_references_span_reconstruction_stacks'
    candidates = error.value.evidence['candidate_stacks']
    sharp = next(c['stack_signature'] for c in candidates if c['ConvolutionKernel']=='SHARP')
    smooth = next(c['stack_signature'] for c in candidates if c['ConvolutionKernel']=='SMOOTH')
    for p in paths+other:
        assert uid(p) not in json.dumps(candidates)
    return paths, other, rs, sharp, smooth


def override(signature):
    return {'adjudication_record_id':'SYNTH-WS210-1',
            'reason':'Synthetic adjudication for independent verification',
            'stack_signature':copy.deepcopy(signature)}


def sidecar(out):
    return json.loads(out.with_name(out.name[:-7]+'.metadata.json').read_text())


def contract_validate(tmp, ct, rs, out):
    from rtpipeline.course_contract import CourseContract, _validate_nifti_provenance
    meta = sidecar(out)
    provenance = {k:meta[k] for k in ('series_instance_uid','sop_hash','geometry',
                                   'nifti_geometry','nifti_sha256','nifti_conversion')}
    provenance['sidecar_path'] = str(out.with_name(out.name[:-7]+'.metadata.json').relative_to(tmp))
    c = CourseContract(tmp, tmp/'metadata.json',
                       {'authoritative_rtstruct':{'path':str(rs.relative_to(tmp))}})
    _validate_nifti_provenance(c, {'nifti_provenance':provenance}, ct, out, meta['series_instance_uid'])


def save(tmp,name,data):
    (tmp/(name+'_evidence.json')).write_text(json.dumps(data,indent=2,default=str))


def make(tmp,pattern):
    ov=None
    if pattern=='D':
        paths,other,rs,sharp,smooth=stacks(tmp);ov=override(sharp)
    elif pattern=='E':paths,rs=mixed(tmp)
    else:
        paths=write_ct_series(tmp/'ct',uniform_z_positions(),signed=False)
        for p in paths:change(p,AcquisitionNumber=1)
        if pattern=='B':duplicate(paths)
        else:
            ds=pydicom.dcmread(paths[2]);ds.SOPInstanceUID=pydicom.uid.generate_uid();ds.file_meta.MediaStorageSOPInstanceUID=ds.SOPInstanceUID
            ds.InstanceNumber=88;ds.AcquisitionNumber=2;ds.ImagePositionPatient=[-50.,-40.,12.]
            ds.save_as(tmp/'ct/stray.dcm',enforce_file_format=True)
        rs=references(tmp/'RS.dcm',paths,paths)
    return paths,rs,ov


def convert(tmp,rs,ov):
    return seg._ensure_ct_nifti(config(tmp),tmp/'ct',tmp/'nifti',authoritative_rtstruct=rs,stack_override=ov,require_hygiene=True)


def break_volume(out,fault):
    if fault=='truncated':
        out.write_bytes(gzip.compress(gzip.decompress(out.read_bytes())[:352],mtime=0))
    else:
        image=nib.load(out);data=image.get_fdata().copy();data[2,3,4]=np.nan if fault=='nan' else data[2,3,4]+1
        altered=nib.Nifti1Image(data,image.affine,image.header)
        altered.set_data_dtype(np.float64)
        altered.set_sform(image.get_sform(),int(image.header['sform_code']));altered.set_qform(image.get_qform(),int(image.header['qform_code']))
        nib.save(altered,out)


@pytest.mark.parametrize('pattern',list('ABD'))
def test_cached_finite_corruption_must_not_be_rebound_as_valid(tmp_path,monkeypatch,pattern):
    paths,rs,ov=make(tmp_path,pattern);out=convert(tmp_path,rs,ov);assert out
    before=sidecar(out);expected=nib.load(out).get_fdata();break_volume(out,'finite')
    corrupt_hash=hashlib.sha256(out.read_bytes()).hexdigest()
    with pytest.raises(CourseContractError,match='content hash'):
        contract_validate(tmp_path,tmp_path/'ct',rs,out)
    calls=[];real=seg.run_dcm2niix
    def counted(*args,**kw):calls.append(str(args[1]));return real(*args,**kw)
    monkeypatch.setattr(seg,'run_dcm2niix',counted)
    result=convert(tmp_path,rs,ov);after=sidecar(result)
    accepted=True
    try:contract_validate(tmp_path,tmp_path/'ct',rs,result)
    except CourseContractError:accepted=False
    save(tmp_path,'finite_cache',{'pattern':pattern,'old_hash':before['nifti_sha256'],'corrupt_hash':corrupt_hash,'rebound_hash':after['nifti_sha256'],'converter_calls':calls,'changed_voxels':int(np.count_nonzero(nib.load(result).get_fdata()!=expected)),'contract_accepts_after_reuse':accepted})
    assert accepted
    assert calls, 'finite cache differs from recorded hash but is reused and its hash is silently rebound'
    assert np.array_equal(nib.load(result).get_fdata(),expected)


def test_interrupted_override_publication_does_not_reuse_wrong_stack(tmp_path,monkeypatch):
    paths,other,rs,sharp,smooth=stacks(tmp_path)
    source_before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (tmp_path/'ct').iterdir()}
    sharp_override=override(sharp);smooth_override=override(smooth)
    out=convert(tmp_path,rs,sharp_override);assert out
    sharp_voxels=nib.load(out).get_fdata();old=sidecar(out)
    original_write=Path.write_text
    with monkeypatch.context() as temporary:
        def fail_sidecar(path,*args,**kwargs):
            if path.name.endswith('.metadata.json'):
                raise OSError('synthetic storage error after new volume publication')
            return original_write(path,*args,**kwargs)
        temporary.setattr(Path,'write_text',fail_sidecar)
        with pytest.raises(OSError,match='synthetic storage error'):
            convert(tmp_path,rs,smooth_override)
    assert sidecar(out)==old
    interrupted_hash=hashlib.sha256(out.read_bytes()).hexdigest()
    assert interrupted_hash!=old['nifti_sha256']
    assert not np.array_equal(nib.load(out).get_fdata(),sharp_voxels)
    with pytest.raises(CourseContractError, match='content hash'):
        contract_validate(tmp_path,tmp_path/'ct',rs,out)
    calls=[];real=seg.run_dcm2niix
    def counted(*args,**kwargs):calls.append(str(args[1]));return real(*args,**kwargs)
    monkeypatch.setattr(seg,'run_dcm2niix',counted)
    result=convert(tmp_path,rs,sharp_override)
    accepted=True
    try:contract_validate(tmp_path,tmp_path/'ct',rs,result)
    except Exception:accepted=False
    after=sidecar(result)
    save(tmp_path,'interrupted_override',{'old_sharp_hash':old['nifti_sha256'],'interrupted_smooth_hash':interrupted_hash,'published_hash':after['nifti_sha256'],'recorded_kernel':after['nifti_conversion']['stack_override']['stack_signature']['ConvolutionKernel'],'changed_voxels_from_requested_stack':int(np.count_nonzero(nib.load(result).get_fdata()!=sharp_voxels)),'converter_calls_after_interruption':calls,'contract_accepts':accepted,'source_bytes_unchanged':source_before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (tmp_path/'ct').iterdir()}})
    assert calls
    assert accepted
    assert source_before == {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (tmp_path/'ct').iterdir()}
    assert np.array_equal(nib.load(result).get_fdata(),sharp_voxels), 'cached SMOOTH voxels are accepted and hash-bound as requested SHARP after an interrupted sidecar write'


@pytest.mark.parametrize('pattern', list('ABD'))
def test_missing_publication_hash_regenerates(tmp_path, monkeypatch, pattern):
    _, rs, ov = make(tmp_path, pattern)
    out = convert(tmp_path, rs, ov)
    meta = sidecar(out)
    del meta['nifti_sha256']
    out.with_name(out.name[:-7] + '.metadata.json').write_text(json.dumps(meta))
    calls = []
    original = seg.run_dcm2niix
    def counted(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)
    monkeypatch.setattr(seg, 'run_dcm2niix', counted)
    result = convert(tmp_path, rs, ov)
    assert result and calls
    contract_validate(tmp_path, tmp_path/'ct', rs, result)


def test_provenance_writer_refuses_hash_rebinding(tmp_path):
    from rtpipeline.nifti_provenance import annotate
    _, rs, ov = make(tmp_path, 'D')
    out = convert(tmp_path, rs, ov)
    meta = sidecar(out)
    original = copy.deepcopy(meta)
    break_volume(out, 'finite')
    with pytest.raises(ValueError, match='content hash'):
        annotate(meta, out, tmp_path/'ct', regenerated=False, existing_sidecar=original)
    assert meta == original


def test_reused_provenance_retains_verified_publication_digest(tmp_path, monkeypatch):
    from rtpipeline import nifti_provenance
    volume = tmp_path/'cached.nii.gz'
    volume.write_bytes(b'synthetic bytes for the digest predicate')
    digest = hashlib.sha256(volume.read_bytes()).hexdigest()
    previous = {'nifti_conversion':{'method':hygiene.METHOD},
                'nifti_sha256':digest, 'generated_at':'original',
                'nifti_generated_at':'original'}
    def unexpected_rehash(path):
        raise AssertionError('reuse must preserve the verified recorded digest')
    monkeypatch.setattr(Path, 'read_bytes', unexpected_rehash)
    monkeypatch.setattr(nifti_provenance, 'nifti_geometry', lambda path: {})
    result = nifti_provenance.annotate({}, volume, tmp_path, regenerated=False,
                                       existing_sidecar=previous)
    assert result['nifti_sha256'] == digest
    assert result['generated_at'] == result['nifti_generated_at'] == 'original'
