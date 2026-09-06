"""Cross-repair acceptance gates, using synthetic DICOM only."""
from dataclasses import replace
import pytest
from pydicom.dataset import Dataset
from rtpipeline import radiomics_ct_contract as contract, radiomics_parallel as parallel
from rtpipeline.rt_details import is_target_volume_candidate, is_target_volume_name
from rtpipeline.roi_requiredness import inspect_rtstruct
from rtpipeline.radiomics_source_inventory import source_observations, source_ledger_rows
from test_radiomics_confirmed_defects import dicom_inventory, task


@pytest.mark.parametrize('kind,points', [
    ('POINT', [1., 2., 3.]),
    ('OPEN_NONPLANAR', [1., 2., 3., 4., 5., 7.]),
])
@pytest.mark.parametrize('name', ['PTV', 'obszar'])
def test_geometry_precedes_policy_skip_builder_and_resource(tmp_path, monkeypatch, kind, points, name):
    path = tmp_path / 'RS.dcm'
    ds = dicom_inventory(path, empty=0, points=1)
    ds.StructureSetROISequence[0].ROIName = name
    contour = ds.ROIContourSequence[0].ContourSequence[0]
    contour.ContourGeometricType = kind
    contour.ContourData = points
    contour.NumberOfContourPoints = len(points) // 3
    ds.save_as(path, enforce_file_format=True)
    t = replace(task(name), rs_path=str(path), mask_identity=str(ds.SOPInstanceUID))
    monkeypatch.setattr(parallel, '_WORKER_STATE', {'skip_rois': {parallel._norm(name)}})
    monkeypatch.setattr(parallel, '_get_builder', lambda *a: pytest.fail('non-volume reached builder'))
    monkeypatch.setattr(parallel, 'estimate_resampled_bounding_box', lambda *a, **k: pytest.fail('non-volume reached resource guard'))
    rows = parallel._extract_one(t)
    assert len(rows) == 2
    assert {r['extraction_status'] for r in rows} == {'nonvolumetric_nonmeasurement'}
    assert {r['roi_structural_code'] for r in rows} == {'ROI_NONVOLUMETRIC_' + kind}
    assert {r['mask_identity'] for r in rows} == {str(ds.SOPInstanceUID)}
    assert {r['stable_roi_identifier'] for r in rows} == {'rtstruct_roi_number:1'}
    assert all(r['native_mask_voxel_count'] is None for r in rows)
    import pandas as pd
    # A correct extractor return must also survive the real publication gate.
    contract.validate_ct_publication(pd.DataFrame(rows), fail_on_unclassified_required=False)
    ledger = source_ledger_rows([path], [t], rows)
    assert ledger[0]['reason_code'] == 'ROI_NONVOLUMETRIC_' + kind
    assert len(ledger[0]['arm_dispositions']) == 2


@pytest.mark.parametrize('kind,points,code', [
    ('CLOSED_PLANAR', [0., 0., 0., 1., 1., 0., 2., 2., 0.], 'ROI_CONTOUR_UNPARSEABLE'),
    ('CLOSED_PLANAR', [0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 1.], 'ROI_CONTOUR_UNPARSEABLE'),
    ('OPEN_NONPLANAR', [0., 0., 0., 0., 0., 0.], 'ROI_CONTOUR_UNPARSEABLE'),
    ('OPEN_NONPLANAR', [0., 0., 0., 1., 1., 1.], 'ROI_NONVOLUMETRIC_OPEN_NONPLANAR'),
])
def test_inventory_and_extraction_share_strict_geometry(tmp_path, kind, points, code):
    path = tmp_path / 'RS.dcm'
    ds = dicom_inventory(path, empty=0, points=1)
    contour = ds.ROIContourSequence[0].ContourSequence[0]
    contour.ContourGeometricType = kind
    contour.ContourData = points
    contour.NumberOfContourPoints = len(points) // 3
    assert source_observations(path, ds)[0].structural_code == code
    assert inspect_rtstruct(path, ds).named_rois[0].structural_code == code


@pytest.mark.parametrize('name,candidate,roi_class', [
    ('PTV1-jelita', True, 'unresolved_mixed'),
    ('Pecherz - PTV', False, 'planning_helper'),
    ('PTV2 - PTV1', True, 'planning_helper'),
    ('2cm od PTV2', True, 'planning_helper'),
    ('GTVn 2', True, 'target'),
])
def test_screening_candidacy_cannot_override_governed_feature_class(name, candidate, roi_class):
    assert is_target_volume_candidate(name) is candidate
    assert is_target_volume_name(name) is candidate
    decision = contract.classify_ct_roi('Manual', name)
    assert decision.roi_class == roi_class
    if roi_class == 'planning_helper':
        assert decision.feature_publication_policy == contract.FEATURE_POLICY_INVENTORY_ONLY
    elif roi_class == 'unresolved_mixed':
        assert decision.adjudication_status == 'operator_adjudication_required'
        assert decision.primary_resegment_range_hu is None
        assert decision.primary_intensity_texture_disposition == 'unclassified_roi'


@pytest.mark.parametrize('route', ['native_worker', 'serial', 'conda'])
def test_returned_grid_admission_remains_typed_robustness(tmp_path, monkeypatch, route):
    import os
    import subprocess
    from pathlib import Path
    try:
        from radiomics import featureextractor
    except ImportError:
        interpreter = '/home/konrad/micromamba/envs/rtpipeline-radiomics/bin/python'
        node = str(Path(__file__)) + '::test_returned_grid_admission_remains_typed_robustness[' + route + ']'
        result = subprocess.run([interpreter, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
            '--basetemp=' + str(tmp_path/'native'), node], capture_output=True, text=True, timeout=120,
            env=dict(os.environ, LD_LIBRARY_PATH='/home/konrad/micromamba/envs/rtpipeline-radiomics/lib'))
        assert result.returncode == 0, result.stdout + result.stderr
        return
    import numpy as np
    import pandas as pd
    import SimpleITK as sitk
    from rtpipeline.config import PipelineConfig
    from rtpipeline import radiomics_robustness as rr
    course = tmp_path/'P'/'C'; course.mkdir(parents=True)
    temp = tmp_path/'inputs'; temp.mkdir()
    params = tmp_path/'params.yaml'
    params.write_text('imageType:\n  Original: {}\nfeatureClass:\n  firstorder: []\n  shape: []\nsetting:\n  minimumROISize: 64\n  minimumROIDimensions: 2\n')
    a = np.zeros((8,8,8), np.uint8); a[2:4,2:4,2:4] = 1
    mask = sitk.GetImageFromArray(a)
    image = sitk.GetImageFromArray(a.astype(np.float32))
    identity = dict(patient_id='P', course_id='C', series_uid='1.2.3', segmentation_source='AutoRTS_total',
        roi_original_name='urinary_bladder', mask_identity='source-mask', stable_roi_identifier='roi-1')
    cfg = PipelineConfig(tmp_path,tmp_path,tmp_path); cfg.radiomics_params_file = params
    if route == 'native_worker':
        mask_path, params_task = parallel._prepare_radiomics_task(image, mask, cfg, 'AutoRTS_total', 'urinary_bladder', course, temp, False, source_identity=identity)
        params_task['extra_metadata'] = {'perturbation_id':'ntcv_v0'}
        result = parallel._isolated_radiomics_extraction_with_retry((mask_path, params_task))
        assert '__nonmeasurement_rows__' in result
        frame = pd.DataFrame(rr._feature_rows_from_worker_result(result))
    else:
        if route == 'conda':
            from rtpipeline import radiomics as rm, radiomics_conda as rc
            monkeypatch.setattr(rm, '_extractor', lambda *a: None)
            monkeypatch.setattr(rc, 'check_radiomics_env', lambda: True)
            real_run = subprocess.run
            def direct_interpreter(argv, **kwargs):
                if len(argv) > 6 and argv[1:3] == ['run', '-n']:
                    # Replace only environment-manager discovery. Execute the
                    # actual generated batch script and parse its real output.
                    argv = ['/home/konrad/micromamba/envs/rtpipeline-radiomics/bin/python', '-B', *argv[5:]]
                    kwargs['env'] = dict(kwargs['env'], LD_LIBRARY_PATH='/home/konrad/micromamba/envs/rtpipeline-radiomics/lib')
                return real_run(argv, **kwargs)
            monkeypatch.setattr(rc.subprocess, 'run', direct_interpreter)
        frame = rr.extract_features_for_masks(image, {'ntcv_v0':mask}, cfg, structure_name='urinary_bladder',
            patient_id='P', course_id='C', segmentation_source='AutoRTS_total',
            source_identity=rr.RobustnessRoiIdentity.from_mapping(identity))
    assert len(frame) == 2
    assert set(frame.robustness_status) == {'geometrically_impossible'}
    assert set(frame.reason_code) == {'resampled_mask_below_minimum_voxels'}
    assert set(frame.extraction_arm) == set(contract.CT_EXTRACTION_ARMS)
    assert frame.value.isna().all() and frame.feature_name.isna().all()


def test_returned_grid_adapter_rejects_unproven_or_mixed_results(monkeypatch):
    from rtpipeline import radiomics_robustness_outcomes as outcomes
    rows = [{'extraction_arm':arm, 'extraction_status':'below_minimum_voxels',
             'extraction_failure_kind':'resampled_degenerate_mask'} for arm in contract.CT_EXTRACTION_ARMS]
    monkeypatch.setattr(outcomes, 'extraction_nonmeasurement', lambda *a: None)
    with pytest.raises(outcomes.GeometricAdmissionContractError, match='reproducible evidence'):
        outcomes.returned_geometry_nonmeasurement(rows, None, None, None)
    with pytest.raises(outcomes.GeometricAdmissionContractError, match='inconsistent paired'):
        outcomes.returned_geometry_nonmeasurement(rows[:1], None, None, None)
    technical = [dict(r, extraction_failure_kind='resource_limit', extraction_status='failed') for r in rows]
    assert outcomes.returned_geometry_nonmeasurement(technical, None, None, None) is None


def test_memory_family_guard_preserves_completed_work_observation(monkeypatch):
    import numpy as np
    import SimpleITK as sitk
    from rtpipeline import robustness_watchdog as wd, radiomics_memory as memory
    events = []
    monkeypatch.setattr(wd, '_sender', events.append)
    class Extractor:
        settings = {}
        enabledFeatures = {'firstorder': [], 'glcm': []}
        def loadImage(self, *args): return args
        def computeShape(self, *args): return {}
        def computeFeatures(self, image, mask, imageTypeName, **kwargs):
            return {next(iter(self.enabledFeatures)): 1.0}
    ext = wd.observed_extractor(Extractor())
    metadata = {'resource_guard_reason_code': 'ROI_RESOURCE_BBOX_ADMITTED',
        'resource_guard_native_image_voxels': 64, 'resource_guard_crop_voxels_bound': 64,
        'resource_guard_predicted_peak_bytes': 0}
    memory.install_texture_budget(ext, metadata)
    image = sitk.GetImageFromArray(np.ones((4,4,4), np.float32))
    mask = sitk.GetImageFromArray(np.ones((4,4,4), np.uint8))
    assert ext.computeFeatures(image, mask, 'original') == {'firstorder': 1.0, 'glcm': 1.0}
    assert events == ['computeFeatures:original', 'computeFeatures:original']
    assert ext.enabledFeatures == {'firstorder': [], 'glcm': []}
