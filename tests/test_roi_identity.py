"""ROI identity regressions on real synthetic DICOM, without reader stand-ins."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from rt_utils import RTStructBuilder, image_helper

from robustness_prep_fixture import write_ct, write_rtstruct
from rtpipeline import organize, radiomics as rm
from rtpipeline.rtstruct_geometry import rtstruct_roi_mask, create_scoped_rtstruct
from rtpipeline.original_segmentation import quarantine_stale_original_exports

# Evidence runs can require the immutable baseline package explicitly.
import os
_expected_source = os.environ.get("ROI_IDENTITY_EXPECTED_SOURCE_ROOT")
if _expected_source:
    assert Path(organize.__file__).resolve().is_relative_to(Path(_expected_source).resolve())
    print("ROI identity production source:", organize.__file__)

@pytest.fixture
def source(tmp_path):
    ct=tmp_path/'DICOM/CT'
    write_ct(ct,rows=24,columns=24,slices=5,spacing=(1,1,1))
    first=np.zeros((24,24,5),bool);first[2:6,3:8,1:4]=True
    second=np.zeros_like(first);second[11:17,12:19,1:4]=True
    rs=tmp_path/'RS.dcm';write_rtstruct(ct,rs,[('Organ',first),('Organ',second)])
    reader=sitk.ImageSeriesReader();reader.SetFileNames([str(p) for p in sorted(ct.glob('*.dcm'))]);image=reader.Execute()
    ref=tmp_path/'ct.nii.gz';sitk.WriteImage(image,str(ref))
    from rtpipeline.course_contract import _ct_provenance
    sidecar={**_ct_provenance(ct),'nifti_geometry':organize.nifti_provenance.nifti_geometry(ref),'nifti_sha256':hashlib.sha256(ref.read_bytes()).hexdigest()}
    (tmp_path/'ct.metadata.json').write_text(json.dumps(sidecar))
    return ct,rs,ref,first,second

def names(rs,values):
    ds=pydicom.dcmread(rs)
    for roi,value in zip(ds.StructureSetROISequence,values):roi.ROIName=value
    ds.save_as(rs)

def builder(source):
    ct,rs,*_=source
    return RTStructBuilder.create_from(str(ct),str(rs))

def test_number_selects_each_duplicate_mask(source):
    b=builder(source)
    for number,expected in enumerate(source[3:],1):
        assert np.array_equal(rtstruct_roi_mask(b,roi_number=number),expected)

def test_ambiguous_name_is_explicitly_refused(source):
    with pytest.raises(ValueError,match='RTSTRUCT_AMBIGUOUS_ROI_NAME'):
        rtstruct_roi_mask(builder(source),'Organ')

def test_name_number_disagreement_is_refused(source):
    names(source[1],['First','Second'])
    with pytest.raises(ValueError,match='ROI_NAME_NUMBER_MISMATCH'):
        rtstruct_roi_mask(builder(source),'First',roi_number=2)

def test_indexed_reader_never_resolves_a_duplicate_to_first(source):
    with pytest.raises(ValueError,match='RTSTRUCT_AMBIGUOUS_ROI_NAME'):
        rm._rt_utils_indexed_roi_mask(builder(source),'Organ',retain=True)

@pytest.mark.parametrize('values',[['Organ','Organ'],['Organ',' Organ']])
@pytest.mark.requires_radiomics_env
@pytest.mark.skipif(int(np.__version__.split(".")[0]) >= 2, reason="requires the native NumPy 1.x radiomics interpreter")
def test_name_keyed_collectors_record_every_ambiguous_declaration(source,values):
    names(source[1],values)
    from rtpipeline.radiomics_rasterizer import require_radiomics_rasterizer
    # This test must be run in the supplied radiomics runtime, with its real pin.
    require_radiomics_rasterizer()
    sink=[]
    with pytest.raises(rm.RadiomicsCourseExtractionError,match='RTSTRUCT_AMBIGUOUS_ROI_NAME'):
        rm._rtstruct_masks(source[0],source[1],best_effort=True,failure_outcomes=sink)
    assert {str(x['roi_number']) for x in sink}=={'1','2'}
    assert all(x['structural_code']=='RTSTRUCT_AMBIGUOUS_ROI_NAME' for x in sink)


def export(source,root):
    return organize._export_original_segmentation_from_paths(rs_path=source[1],primary_nifti=source[2],dicom_ct_dir=source[0],segmentation_original_dir=root,log_root=root.parent,overwrite=True)

def test_export_uses_numbers_and_disjoint_filenames(source,tmp_path):
    root=tmp_path/'Segmentation_Original';manifest=export(source,root)
    entries=manifest['structures'];assert [e['roi_number'] for e in entries]==[1,2]
    assert len({e['mask'].casefold() for e in entries})==2
    assert [int(sitk.GetArrayFromImage(sitk.ReadImage(str(root/e['mask']))).sum()) for e in entries]==[60,126]

def test_export_stem_collision_cannot_overwrite(source,tmp_path):
    ct,rs,*_=source
    b=builder(source);b.ds.StructureSetROISequence[0].ROIName='A B';b.ds.StructureSetROISequence[1].ROIName='A_B'
    b.add_roi(mask=source[3]|source[4],name='A_B_2');b.save(str(rs))
    root=tmp_path/'Segmentation_Original';manifest=export(source,root)
    entries=manifest['structures'];assert len({e['mask'].casefold() for e in entries})==3
    assert [int(sitk.GetArrayFromImage(sitk.ReadImage(str(root/e['mask']))).sum()) for e in entries]==[60,126,186]

def test_alias_collision_records_nonmeasurement(source):
    from rtpipeline.custom_structures import CustomStructureProcessor,CustomStructureConfig
    processor=CustomStructureProcessor()
    config=CustomStructureConfig(name='union',source_structures=['ORGAN'],operation='union')
    assert processor.process_custom_structure(config,{'Organ':source[3],'organ':source[4]}) is None
    outcome=processor.structure_outcomes['union']
    assert outcome['status']=='ambiguous_source'
    assert outcome['reason_code']=='CUSTOM_SOURCE_AMBIGUOUS_NAME'
    assert set(outcome['ambiguous_sources']['ORGAN'])=={'Organ','organ'}

def test_unique_exact_source_wins_even_if_aliases_collide(source):
    from rtpipeline.custom_structures import CustomStructureProcessor,CustomStructureConfig
    processor=CustomStructureProcessor()
    config=CustomStructureConfig(name='union',source_structures=['organ'],operation='union')
    assert np.array_equal(processor.process_custom_structure(config,{'Organ':source[3],'organ':source[4]}),source[4])

def legacy(root,ref,rs):
    old=root/'ct';old.mkdir(parents=True)
    (old/'Organ.nii.gz').write_bytes(b'synthetic-legacy-mask')
    (old/'metadata.json').write_text(json.dumps({'model':'manual','source_rtstruct':str(rs),'source_nifti':str(ref),'structures':[{'roi_name':'Organ','mask':'ct/Organ.nii.gz'}]}))
    return old

def test_quarantine_moves_unverifiable_other_producers_and_loose_files(source,tmp_path):
    root=tmp_path/'Segmentation_Original';old=legacy(root,source[2],source[1])
    other=root/'external';other.mkdir();payload=b'external synthetic mask'
    (other/'mask.nii.gz').write_bytes(payload)
    (other/'metadata.json').write_text(json.dumps({'schema':'foreign-v1','model':'manual'}))
    loose=root/'foreign.nii.gz';loose.write_bytes(payload)
    quarantine_stale_original_exports(root)
    assert not old.exists()
    assert not other.exists() and not loose.exists()
    quarantined = root.parent / 'Segmentation_Quarantine'
    assert any(path.read_bytes() == payload for path in quarantined.rglob('mask.nii.gz'))
    assert any(path.read_bytes() == payload for path in quarantined.rglob('export') if path.is_file())

def test_unverifiable_target_is_quarantined_before_export(source,tmp_path):
    root=tmp_path/'Segmentation_Original';target=root/'ct';target.mkdir(parents=True)
    (target/'metadata.json').write_text('{"producer":"other"}')
    marker=target/'other.nii.gz';marker.write_bytes(b'keep-exact-bytes')
    assert export(source,root)['structures']
    assert not marker.exists()
    assert any(path.read_bytes() == b'keep-exact-bytes'
               for path in (root.parent / 'Segmentation_Quarantine').rglob('other.nii.gz'))

def test_standalone_repairs_legacy_sidecar_without_mutating_input(source,tmp_path):
    from rtpipeline.cli import main
    old=source[2].with_name('ct.metadata.json');old.write_text('{"series_instance_uid":"legacy"}')
    original=old.read_bytes();root=tmp_path/'repaired'
    assert main(['re-export-original','--rtstruct',str(source[1]),'--ct-series',str(source[0]),'--reference-nifti',str(source[2]),'--output-root',str(root)])==0
    assert old.read_bytes()==original
    manifest=json.loads(next((root/'Segmentation_Original').glob('*/metadata.json')).read_text())
    assert [r['roi_number'] for r in manifest['structures']]==[1,2]
    from rtpipeline.original_segmentation import require_current_original_export
    require_current_original_export(root/'Segmentation_Original/ct')

def test_unique_masks_match_the_original_rasterizer_byte_for_byte(source):
    names(source[1],['First','Second']);b=builder(source)
    for i,name in enumerate(b.get_roi_names(),1):
        expected=b.get_roi_mask_by_name(name)
        assert rtstruct_roi_mask(b,roi_number=i).tobytes()==expected.tobytes()

def test_duplicate_roi_numbers_are_refused(source):
    ds=pydicom.dcmread(source[1]);ds.StructureSetROISequence[1].ROINumber=1;ds.save_as(source[1])
    with pytest.raises(ValueError,match='RTSTRUCT_DUPLICATE_ROI_NUMBER'):
        rtstruct_roi_mask(builder(source),'Organ')

def test_scoped_reader_preserves_distinct_stored_names_by_number(source):
    names(source[1],['Organ',' Organ'])
    reader=create_scoped_rtstruct(source[0],source[1])
    for number,expected in enumerate(source[3:],1):
        assert np.array_equal(reader.get_roi_mask_by_number(number),expected)
    with pytest.raises(ValueError, match='RTSTRUCT_AMBIGUOUS_ROI_NAME'):
        reader.get_roi_mask_by_name(' Organ')

def test_qc_cropping_is_bound_to_the_second_roi_number(source,tmp_path):
    from rtpipeline.quality_control import DICOMValidator
    b=RTStructBuilder.create_new(str(source[0]));b.add_roi(mask=source[3],name='Organ')
    edge=np.zeros_like(source[3]);edge[0:5,12:18,1:4]=1;b.add_roi(mask=edge,name='Organ');b.save(str(source[1]))
    validator=object.__new__(DICOMValidator)
    validator.course_dir=tmp_path;validator.ct_dir=source[0];validator.rs_path=source[1];validator.rs_auto_path=tmp_path/'absent.dcm'
    result=validator._segmentation_cropping()
    assert len(result['structures'])==1
    assert result['structures'][0]['roi_number']==2
    assert 'y_min' in result['structures'][0]['cropped_sides']

def test_roi_fixer_records_identity_refusal_without_changing_source(source,tmp_path):
    from rtpipeline.roi_fixer import fix_rtstruct_rois
    previous=source[1].read_bytes();target=tmp_path/'fixed.dcm'
    summary=fix_rtstruct_rois(source[0],source[1],output_path=target)
    assert summary.identity_error=='RTSTRUCT_AMBIGUOUS_ROI_NAME'
    assert source[1].read_bytes()==previous
    assert not target.exists()

def test_axial_review_keeps_both_duplicate_overlays(source,tmp_path,monkeypatch):
    from types import SimpleNamespace
    pytest.importorskip("plotly", reason="viewer is exercised in the supplied main runtime")
    from rtpipeline import visualize
    monkeypatch.setattr(visualize,'load_course_contract',lambda _:SimpleNamespace(planning_ct_dir=source[0],authoritative_rtstruct_path=source[1]))
    output=visualize.generate_axial_review(tmp_path)
    html=output.read_text()
    assert 'Organ [ROI 1]' in html and 'Organ [ROI 2]' in html

def test_rectangular_unique_number_reader_keeps_the_geometry_adapter(tmp_path):
    from rt_utils import ds_helper
    from rtpipeline.rtstruct_geometry import rtstruct_contours_to_mask
    ct=tmp_path/'CT';write_ct(ct,rows=24,columns=32,slices=5,spacing=(1.3,2.1,1.0))
    m=np.zeros((24,32,5),bool);m[3:8,4:12,1:4]=1
    rs=tmp_path/'rect.dcm';write_rtstruct(ct,rs,[('Unique',m)])
    b=RTStructBuilder.create_from(str(ct),str(rs))
    expected=rtstruct_contours_to_mask(b.series_data,ds_helper.get_contour_sequence_by_roi_number(b.ds,1))
    assert rtstruct_roi_mask(b,roi_number=1).tobytes()==expected.tobytes()

def test_quarantine_moves_a_mixed_unverifiable_export_directory(source,tmp_path):
    root=tmp_path/'Segmentation_Original';old=legacy(root,source[2],source[1])
    (old/'external.nii.gz').write_bytes(b'unclaimed')
    quarantine_stale_original_exports(root)
    assert not old.exists()
    assert any(path.read_bytes() == b'unclaimed'
               for path in (root.parent / 'Segmentation_Quarantine').rglob('external.nii.gz'))

def test_standalone_wrong_grid_is_refused_before_touching_output(source,tmp_path):
    from rtpipeline.cli import main
    ref=sitk.ReadImage(str(source[2]));ref.SetOrigin((100.,100.,100.));sitk.WriteImage(ref,str(source[2]))
    root=tmp_path/'out'
    assert main(['re-export-original','--rtstruct',str(source[1]),'--ct-series',str(source[0]),'--reference-nifti',str(source[2]),'--output-root',str(root)])==2
    assert not root.exists()

def test_configured_label_globs_resolve_colliding_producer_exports(source, tmp_path):
    from rtpipeline.segmenters.postprocessing import _resolve_reference_paths
    names(source[1], ['GTV-1vis-1', 'GTV-1vis-1'])
    root = tmp_path / 'Segmentation_Original'
    manifest = export(source, root)
    expected = [root / f'ct/GTV-1vis-1__roi-{number}.nii.gz' for number in (1, 2)]
    assert [root / entry['mask'] for entry in manifest['structures']] == expected
    with pytest.raises(ValueError, match='RTSTRUCT_AMBIGUOUS_ROI_NAME'):
        _resolve_reference_paths(root / 'ct/GTV-1vis-1.nii.gz')
    assert _resolve_reference_paths(root / 'ct/GTV-1vis-*.nii.gz') == expected
    assert _resolve_reference_paths(root / 'ct/*__roi-*.nii.gz') == expected
    for path in expected:
        assert _resolve_reference_paths(path) == [path]


def test_stage_validation_records_foreign_exports_as_stale(tmp_path):
    from rtpipeline.stage_completion import _stale_original_exports,_validate_stage_specific_outcome,stage_definition
    other=tmp_path/'Segmentation_Original/other';other.mkdir(parents=True)
    (other/'metadata.json').write_text('{"schema":"foreign-producer-v1"}')
    (other/'mask.nii.gz').write_bytes(b'foreign-synthetic')
    assert _stale_original_exports(tmp_path,stage_definition('organize'))[0]['status']=='stale'
    with pytest.raises(ValueError, match='stale'):
        _validate_stage_specific_outcome(tmp_path,stage_definition('segmentation'),'ok',[])
    assert (other/'mask.nii.gz').exists()

@pytest.mark.requires_radiomics_env
@pytest.mark.skipif(int(np.__version__.split(".")[0]) >= 2, reason="requires the native NumPy 1.x radiomics interpreter")
@pytest.mark.parametrize('missing_sop_uid', [False, True])
def test_conda_preflight_records_stripped_name_ambiguity_before_extraction(source,tmp_path,missing_sop_uid,monkeypatch):
    from rtpipeline.radiomics_rasterizer import require_radiomics_rasterizer
    require_radiomics_rasterizer()
    from course_contract_test_utils import write_minimal_course_contract
    from rtpipeline.course_contract import _ct_provenance
    from rtpipeline.config import PipelineConfig
    from rtpipeline import radiomics_conda
    names(source[1],['Organ',' Organ'])
    contract_path=write_minimal_course_contract(tmp_path,authoritative_rtstruct=source[1],planning_ct_nifti=source[2])
    sidecar=source[2].with_name('ct.metadata.json');meta=json.loads(sidecar.read_text());meta.update(_ct_provenance(source[0]));sidecar.write_text(json.dumps(meta))
    payload=json.loads(contract_path.read_text());payload['course_contract']['planning_ct']['nifti_provenance'].update(meta);contract_path.write_text(json.dumps(payload))
    config=PipelineConfig(dicom_root=source[0],output_root=tmp_path,logs_root=tmp_path/'logs')
    if missing_sop_uid:
        # Authoritative sources require a SOP UID at the course-contract boundary.
        # A CustomModel source reaches this preflight through its output inventory.
        model=tmp_path/'Segmentation_CustomModels/no_sop';model.mkdir(parents=True)
        dataset=pydicom.dcmread(source[1]);del dataset.SOPInstanceUID;dataset.save_as(model/'rtstruct.dcm')
        names(source[1],['First','Second'])
        monkeypatch.setattr(radiomics_conda,'validate_custom_model_output_inventory',lambda *_:{'no_sop':['Organ']})
        monkeypatch.setattr(radiomics_conda,'list_custom_model_outputs',lambda *_:[('no_sop',model)])
    with pytest.raises(rm.RadiomicsCourseExtractionError,match='RTSTRUCT_AMBIGUOUS_ROI_NAME'):
        radiomics_conda.radiomics_for_course(tmp_path,config,max_workers=1)
    ledger=json.loads((tmp_path/'metadata/radiomics_ct_roi_ledger.json').read_text())
    assert {row['roi_number'] for row in ledger['course_roi']}=={'1','2'}
    assert ledger['course'][0]['reason_code']=='RTSTRUCT_AMBIGUOUS_ROI_NAME'
    assert ledger['course'][0]['technical_exclusion'] is False
    if missing_sop_uid:
        assert all(row['rtstruct_sop_instance_uid']=='' for row in ledger['course_roi'])
    assert not (tmp_path/'metadata/radiomics_ct_identity_refusals.json').exists()
    assert not (tmp_path/'radiomics_ct.parquet').exists()

def test_quarantine_preserves_unlisted_bytes_outside_active_exports(source,tmp_path):
    root=tmp_path/'Segmentation_Original';manifest=export(source,root)
    owned=root/Path(manifest['structures'][0]['mask']).parent
    foreign=owned/'foreign-producer.nii.gz';foreign.write_bytes(b'foreign-synthetic')
    quarantine_stale_original_exports(root)
    assert not foreign.exists()
    assert any(path.read_bytes() == b'foreign-synthetic'
               for path in (root.parent / 'Segmentation_Quarantine').rglob('foreign-producer.nii.gz'))
    assert not any((root/entry['mask']).exists() for entry in manifest['structures'])


def test_numbered_file_and_roi_label_reference_collision_is_refused(source, tmp_path):
    from rtpipeline.segmenters.postprocessing import _resolve_reference_paths
    reader = builder(source)
    reader.add_roi(mask=source[3] | source[4], name='Organ__roi-1')
    reader.save(str(source[1]))
    root = tmp_path / 'Segmentation_Original'
    manifest = export(source, root)
    assert [entry['mask'] for entry in manifest['structures']] == [
        'ct/Organ__roi-1.nii.gz', 'ct/Organ__roi-2.nii.gz', 'ct/Organ__roi-1__roi-3.nii.gz']
    with pytest.raises(ValueError, match='RTSTRUCT_AMBIGUOUS_ROI_NAME'):
        _resolve_reference_paths(root / manifest['structures'][0]['mask'])


@pytest.mark.parametrize('pattern', ['r*.nii.gz', '*roi*.nii.gz', 'roi*.nii.gz'])
def test_broad_label_glob_keeps_label_selection_with_number_suffix(source, tmp_path, pattern):
    from rtpipeline.segmenters.postprocessing import _resolve_reference_paths
    names(source[1], ['rectum', 'rectum'])
    root = tmp_path / 'Segmentation_Original'
    manifest = export(source, root)
    expected = [root / entry['mask'] for entry in manifest['structures']] if pattern == 'r*.nii.gz' else []
    assert _resolve_reference_paths(root / 'ct' / pattern) == expected


def test_standalone_non_dicom_source_returns_refusal_json(source, tmp_path, capsys):
    from rtpipeline.cli import main
    source[1].write_bytes(b'non-DICOM synthetic source')
    output = tmp_path / 'out'
    assert main(['re-export-original', '--rtstruct', str(source[1]), '--ct-series', str(source[0]),
                 '--reference-nifti', str(source[2]), '--output-root', str(output)]) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'refused'
    assert not output.exists()


@pytest.mark.parametrize('field,value', [('roi_name', 'Foreign'), ('roi_number', 3)])
def test_numbered_export_checks_source_number_name_binding(source, tmp_path, field, value):
    from rtpipeline.original_segmentation import require_current_original_export, require_current_original_mask
    root = tmp_path / 'Segmentation_Original'
    manifest = export(source, root)
    manifest['structures'][0][field] = value
    (root / 'ct/metadata.json').write_text(json.dumps(manifest))
    for consumer, path in [(require_current_original_export, root / 'ct'),
                           (require_current_original_mask, root / manifest['structures'][0]['mask'])]:
        with pytest.raises(ValueError, match='ROI_NAME_NUMBER_MISMATCH'):
            consumer(path)


def _configure_identity_course(source, course):
    from course_contract_test_utils import write_minimal_course_contract
    from rtpipeline.course_contract import _ct_provenance
    from rtpipeline.config import PipelineConfig
    contract_path = write_minimal_course_contract(
        course, authoritative_rtstruct=source[1], planning_ct_nifti=source[2])
    sidecar = source[2].with_name('ct.metadata.json')
    meta = json.loads(sidecar.read_text())
    meta.update(_ct_provenance(source[0]))
    sidecar.write_text(json.dumps(meta))
    payload = json.loads(contract_path.read_text())
    payload['course_contract']['planning_ct']['nifti_provenance'].update(meta)
    contract_path.write_text(json.dumps(payload))
    config = PipelineConfig(source[0], course, course / 'logs', max_workers_override=1)
    params = course / 'radiomics.yaml'
    params.write_text('imageType:\n  Original: {}\nfeatureClass:\n  firstorder: [Mean]\n  shape: [VoxelVolume]\nsetting:\n  minimumROIDimensions: 2\n')
    config.radiomics_params_file = params
    config.radiomics_min_voxels = 1
    return config


@pytest.mark.parametrize('backend', ['direct', 'conda'])
@pytest.mark.requires_radiomics_env
@pytest.mark.skipif(int(np.__version__.split('.')[0]) >= 2, reason='requires the native NumPy 1.x radiomics interpreter')
def test_course_identity_refusal_withholds_unambiguous_roi_then_clears_on_success(source, tmp_path, backend):
    from rtpipeline import radiomics_conda
    from rtpipeline.radiomics_rasterizer import require_radiomics_rasterizer
    require_radiomics_rasterizer()
    reader = builder(source)
    reader.add_roi(mask=source[3] | source[4], name='PTV')
    reader.save(str(source[1]))
    config = _configure_identity_course(source, tmp_path)
    obsolete = tmp_path / 'metadata/radiomics_ct_identity_refusals.json'
    obsolete.write_text('[{"stale": true}]')
    output = tmp_path / 'radiomics_ct.parquet'
    output.write_bytes(b'stale synthetic publication')
    call = (lambda: rm.radiomics_for_course(config, tmp_path)) if backend == 'direct' else (
        lambda: radiomics_conda.radiomics_for_course(tmp_path, config, max_workers=1))
    with pytest.raises(rm.RadiomicsCourseExtractionError, match='RTSTRUCT_AMBIGUOUS_ROI_NAME'):
        call()
    assert not output.exists() and not output.with_suffix('.xlsx').exists()
    assert not obsolete.exists()
    for ledger_name in ['radiomics_ct_roi_ledger.json', 'radiomics_roi_ledger.json']:
        ledger = json.loads((tmp_path / 'metadata' / ledger_name).read_text())
        outcome = ledger['course'][0]
        assert outcome['reason_code'] == 'RTSTRUCT_AMBIGUOUS_ROI_NAME'
        assert outcome['identity_refusal'] is True and outcome['technical_exclusion'] is False
        refusals = ledger['course_roi']
        assert [(row['roi_number'], row['roi_name']) for row in refusals] == [('1', 'Organ'), ('2', 'Organ')]
        assert all(row['reason_code'] == 'RTSTRUCT_AMBIGUOUS_ROI_NAME' for row in refusals)
        assert all(row['segmentation_source'] == 'Manual' and row['source_path'] == str(source[1]) for row in refusals)
    counts = json.loads((tmp_path / 'metadata/radiomics_ct_denominators.json').read_text())['COURSE_ROI']['CT:Organ']
    assert counts['excluded_identity'] == 2 and counts['excluded_technical'] == 0
    # A prior backend version may have kept this refusal outside the ledger.
    obsolete.write_text(json.dumps(refusals))
    # Correct the same source and rerun the same course through the same entrypoint.
    names(source[1], ['First', 'Second', 'PTV'])
    assert call() is not None
    assert not obsolete.exists()
    publication = rm.read_authoritative_ct_publication(output)
    assert set(publication.roi_original_name) == {'First', 'Second', 'PTV'}
    assert set(publication.extraction_status) == {'success'}
    outcome = json.loads((tmp_path / 'metadata/radiomics_ct_roi_ledger.json').read_text())['course'][0]
    assert outcome['extracted'] is True and not outcome.get('identity_refusal', False)


@pytest.mark.parametrize('read_error', ['invalid_dicom', 'os_error'])
@pytest.mark.requires_radiomics_env
@pytest.mark.skipif(int(np.__version__.split('.')[0]) >= 2, reason='requires the native NumPy 1.x radiomics interpreter')
def test_conda_preflight_preserves_governed_unreadable_source_outcome(source, tmp_path, monkeypatch, read_error):
    from rtpipeline import radiomics_conda
    from rtpipeline.radiomics_rasterizer import require_radiomics_rasterizer
    require_radiomics_rasterizer()
    names(source[1], ['First', 'Second'])
    config = _configure_identity_course(source, tmp_path)
    model = tmp_path / 'Segmentation_CustomModels/broken'
    model.mkdir(parents=True)
    broken = model / 'rtstruct.dcm'
    broken.write_bytes(b'unreadable synthetic custom model source')
    monkeypatch.setattr(radiomics_conda, 'validate_custom_model_output_inventory', lambda *_: {'broken': ['PTV']})
    monkeypatch.setattr(radiomics_conda, 'list_custom_model_outputs', lambda *_: [('broken', model)])
    if read_error == 'os_error':
        original_read = pydicom.dcmread
        def fail_broken(path, *args, **kwargs):
            if Path(path) == broken:
                raise OSError('synthetic source read error')
            return original_read(path, *args, **kwargs)
        monkeypatch.setattr(pydicom, 'dcmread', fail_broken)
    output = tmp_path / 'radiomics_ct.parquet'
    output.write_bytes(b'stale synthetic publication')
    with pytest.raises(rm.RadiomicsCourseExtractionError, match='Failed to construct RTSTRUCT reader for CustomModel:broken'):
        radiomics_conda.radiomics_for_course(tmp_path, config, max_workers=1)
    assert not output.exists()
