from dataclasses import replace
from types import SimpleNamespace
import json
import numpy as np
import pytest
import SimpleITK as sitk
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from rtpipeline import radiomics_ct_contract as c, radiomics_parallel as p
from rtpipeline.radiomics_source_inventory import source_observations, source_ledger_rows
from test_radiomics_ct_dual_arm import _FakeExtractor, _qc, _versions


def task(name, code=None):
    return p._RoiTask('Manual', '/does-not-exist/RS.dcm', name, '/p/c', 'ct', 'rs',
                      'rtstruct_roi_number:1', c.classify_ct_roi('Manual', name),
                      'run', 'rev', {a:'configured' for a in c.CT_EXTRACTION_ARMS},
                      {a:'effective' for a in c.CT_EXTRACTION_ARMS}, False, code)


@pytest.mark.parametrize('name', ['obszar', 'podkladka'])
def test_inventory_policy_precedes_mask_and_resource_guard(monkeypatch, name):
    monkeypatch.setattr(p, '_WORKER_STATE', {})
    monkeypatch.setattr(p, '_get_builder', lambda *a: pytest.fail('mask construction started'))
    t = task(name)
    assert t.decision.feature_publication_policy == c.FEATURE_POLICY_INVENTORY_ONLY
    rows = p._extract_one(t)
    assert len(rows) == 2
    assert all(r['extraction_status'] == t.decision.primary_intensity_texture_disposition for r in rows)
    assert all(r['native_mask_voxel_count'] is None for r in rows)


@pytest.mark.parametrize('count', [50, 57, 64])
def test_resampled_minimum_before_any_feature_execute(monkeypatch, count):
    monkeypatch.setattr(c, 'resampled_mask_qc', lambda *a: {**_qc(), 'morphologic_resampled_voxel_count': count})
    monkeypatch.setattr(c, '_runtime_versions', _versions)
    class NoExecute(_FakeExtractor):
        def execute(self, *a):
            pytest.fail('features started before grid admission')
    rows = c.extract_ct_roi_arms(object(), object(), factory=NoExecute,
        decision=c.classify_ct_roi('AutoRTS_total', 'kidney_left'), common_metadata={},
        run_identifier='run', code_revision='rev', native_voxel_count=100, required=False)
    assert len(rows) == 2
    assert all(r['extraction_status'] == 'below_minimum_voxels' for r in rows)
    assert all(r['native_mask_voxel_count'] == 100 and r['resampled_mask_voxel_count'] == count for r in rows)
    assert all(r['extraction_failure_kind'] == 'resampled_degenerate_mask' for r in rows)


def test_resampled_dimensions_checked_before_execute(monkeypatch):
    monkeypatch.setattr(c, 'resampled_mask_qc', lambda *a: {**_qc(), 'observed_roi_dimensions_before_resegmentation': 1})
    monkeypatch.setattr(c, '_runtime_versions', _versions)
    rows = c.extract_ct_roi_arms(object(), object(), factory=_FakeExtractor,
        decision=c.classify_ct_roi('Manual', 'PTV'), common_metadata={},
        run_identifier='run', code_revision='rev', native_voxel_count=100, required=False)
    assert all(r['extraction_status'] == 'below_minimum_dimensions' for r in rows)


@pytest.mark.parametrize('code,status', [('ROI_DECLARED_EMPTY_CONTOUR_SEQUENCE','declared_empty'),
    ('ROI_NONVOLUMETRIC_POINT','non_volumetric'), ('ROI_CONTOUR_UNPARSEABLE','malformed_source_roi')])
def test_source_terminal_before_mask(monkeypatch, code, status):
    monkeypatch.setattr(p, '_WORKER_STATE', {})
    rows = p._extract_one(task('PTV', code))
    assert len(rows) == 2
    assert all(r['extraction_status'] == status and r['roi_structural_code'] == code for r in rows)


def dicom_inventory(path, empty=482, points=3):
    meta=FileMetaDataset(); meta.TransferSyntaxUID=ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID='1.2.840.10008.5.1.4.1.1.481.3'; meta.MediaStorageSOPInstanceUID=generate_uid()
    ds=FileDataset(str(path), {}, file_meta=meta, preamble=b'\0'*128)
    ds.SOPInstanceUID=meta.MediaStorageSOPInstanceUID
    ds.StructureSetROISequence=[]; ds.ROIContourSequence=[]
    for i in range(empty+points):
        roi=Dataset(); roi.ROINumber=i+1; roi.ROIName=f'roi_{i+1}'; ds.StructureSetROISequence.append(roi)
        item=Dataset(); item.ReferencedROINumber=i+1; item.ContourSequence=[]
        if i>=empty:
            contour=Dataset(); contour.ContourGeometricType='POINT'; contour.NumberOfContourPoints=1
            contour.ContourData=[1.,2.,3.]; item.ContourSequence=[contour]
        ds.ROIContourSequence.append(item)
    ds.save_as(path, enforce_file_format=True)
    return ds


def test_482_empty_and_3_point_source_universe_exact(tmp_path, monkeypatch):
    path=tmp_path/'RS.dcm'; dicom_inventory(path)
    observations=source_observations(path)
    assert len(observations)==485
    assert sum(x.structural_code=='ROI_DECLARED_EMPTY_CONTOUR_SEQUENCE' for x in observations)==482
    assert sum(x.structural_code=='ROI_NONVOLUMETRIC_POINT' for x in observations)==3
    tasks=[replace(task(x.name,x.structural_code),rs_path=str(path), stable_roi_identifier=f'rtstruct_roi_number:{x.roi_number}') for x in observations]
    monkeypatch.setattr(p,'_WORKER_STATE',{})
    rows=[r for t in tasks for r in p._extract_one(t)]
    ledger=source_ledger_rows([path],tasks,rows)
    source={(str(path),i) for i in range(485)}
    observed={(r['source_path'],r['declaration_ordinal']) for r in ledger}
    assert source-observed==observed-source==set()
    assert all(len(r['arm_dispositions'])==2 for r in ledger)
    assert sum(r['disposition']=='declared_empty' for r in ledger)==482
    assert sum(r['disposition']=='non_volumetric' for r in ledger)==3


def test_same_name_different_sources_are_not_collapsed(tmp_path):
    a=tmp_path/'RS.dcm'; b=tmp_path/'RS_auto.dcm'
    dicom_inventory(a,1,0); dicom_inventory(b,1,0)
    ledger=source_ledger_rows([a,b],[],[])
    assert len(ledger)==2 and len({r['source_path'] for r in ledger})==2


def test_native_small_mask_retains_degenerate_code(monkeypatch):
    image=sitk.GetImageFromArray(np.zeros((8,8,8)))
    mask=np.zeros((8,8,8),np.uint8); mask[1:3,1:3,1:3]=1
    monkeypatch.setattr(p,'_WORKER_STATE',{'img':image,'extractor':_FakeExtractor(),'min_voxels':64})
    monkeypatch.setattr(p,'_get_builder',lambda *a: SimpleNamespace(get_roi_mask_by_name=lambda *a:mask))
    rows=p._extract_one(task('PTV'))
    assert all(r['extraction_failure_kind']=='degenerate_mask' and r['native_mask_voxel_count']==8 for r in rows)
