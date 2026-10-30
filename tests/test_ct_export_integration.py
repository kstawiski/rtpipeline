"""Planning-stack selection and physical original-export lifecycle."""
import hashlib
import json

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from rtpipeline import organize, segmentation
from rtpipeline.original_segmentation import require_current_original_mask
from rtpipeline.radiomics_rasterizer import opencv_runtime_provenance
from rtpipeline.rtstruct_geometry import image_array_for_rtstruct
from test_cache_publication_integrity import make, convert, sidecar, config, uid
from test_nifti_nonuniform_fallback import dcm2niix  # noqa: F401


def clinical_roi(rs, paths, pattern):
    ds = pydicom.dcmread(rs)
    roi = Dataset()
    roi.ROINumber = 1
    roi.ROIName = 'Asymmetric'
    roi.ReferencedFrameOfReferenceUID = pydicom.dcmread(paths[0]).FrameOfReferenceUID
    ds.StructureSetROISequence = Sequence([roi])
    item = Dataset()
    item.ReferencedROINumber = 1
    item.ROIDisplayColor = [255, 0, 0]
    item.ContourSequence = Sequence([])
    for index in range(3, 7):
        image = pydicom.dcmread(paths[index])
        # D references both complete reconstructions. The selected stack must
        # preserve the polygons whose source references name its counterpart.
        if pattern == 'D' and index % 2:
            image = pydicom.dcmread(paths[0].parent / f'second_{index:03d}.dcm')
        ref = Dataset()
        ref.ReferencedSOPClassUID = image.SOPClassUID
        ref.ReferencedSOPInstanceUID = image.SOPInstanceUID
        contour = Dataset()
        contour.ContourGeometricType = 'CLOSED_PLANAR'
        contour.NumberOfContourPoints = 4
        z = float(image.ImagePositionPatient[2])
        contour.ContourData = [-46., -37., z, -42., -37., z,
                               -42., -34., z, -46., -34., z]
        contour.ContourImageSequence = Sequence([ref])
        item.ContourSequence.append(contour)
    ds.ROIContourSequence = Sequence([item])
    observation = Dataset()
    observation.ObservationNumber = 1
    observation.ReferencedROINumber = 1
    observation.RTROIInterpretedType = 'ORGAN'
    observation.ROIInterpreter = ''
    ds.RTROIObservationsSequence = Sequence([observation])
    ds.save_as(rs, enforce_file_format=True)


def export(tmp, rs, out):
    return organize._export_original_segmentation_from_paths(
        rs_path=rs, primary_nifti=out, dicom_ct_dir=tmp/'ct',
        segmentation_original_dir=tmp/'Segmentation_Original',
        log_root=tmp, overwrite=False)


@pytest.mark.parametrize('pattern', list('ABDE'))
def test_selected_stack_exports_exact_physical_roi(tmp_path, dcm2niix, pattern):
    paths, rs, ov = make(tmp_path, pattern)
    clinical_roi(rs, paths, pattern)
    source = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (tmp_path/'ct').iterdir()}
    out = convert(tmp_path, rs, ov)
    manifest = export(tmp_path, rs, out)
    assert manifest and manifest['geometry_status'] == 'exact_physical_mapping'
    assert manifest['opencv_runtime'] == opencv_runtime_provenance()
    mask = tmp_path/'Segmentation_Original'/manifest['structures'][0]['mask']
    require_current_original_mask(mask)
    image = sitk.ReadImage(str(mask))
    slices = [pydicom.dcmread(p) for p in paths]
    expected = np.zeros((16, 20, 12), dtype=bool)
    expected[3:7, 4:9, 3:7] = True
    np.testing.assert_array_equal(image_array_for_rtstruct(image, slices) > 0, expected)
    assert source == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (tmp_path/'ct').iterdir()}
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in mask.parent.iterdir()}
    assert export(tmp_path, rs, out) == manifest
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in mask.parent.iterdir()}


@pytest.mark.parametrize('change', ['publication', 'governance', 'uncommitted_bytes'])
def test_ct_change_quarantines_old_export(tmp_path, dcm2niix, change):
    paths, rs, ov = make(tmp_path, 'D')
    clinical_roi(rs, paths, 'D')
    out = convert(tmp_path, rs, ov)
    manifest = export(tmp_path, rs, out)
    mask = tmp_path/'Segmentation_Original'/manifest['structures'][0]['mask']
    old = mask.read_bytes()
    if change == 'uncommitted_bytes':
        out.write_bytes(out.read_bytes() + b'uncommitted')
    elif change == 'governance':
        ov['reason'] = 'Independent synthetic adjudication of the same stack'
        assert convert(tmp_path, rs, ov) == out
    else:
        assert segmentation._ensure_ct_nifti(config(tmp_path), tmp_path/'ct', tmp_path/'nifti',
            force=True, authoritative_rtstruct=rs, stack_override=ov, require_hygiene=True) == out
    with pytest.raises(ValueError, match='stale'):
        require_current_original_mask(mask)
    result = export(tmp_path, rs, out)
    quarantined = list((tmp_path/'Segmentation_Quarantine').rglob('Asymmetric.nii.gz'))
    assert len(quarantined) == 1 and quarantined[0].read_bytes() == old
    if change == 'uncommitted_bytes':
        assert result is None and not mask.exists()
    else:
        assert result and mask.exists()
        require_current_original_mask(mask)
        provenance = json.loads((mask.parent/'provenance.json').read_text())
        assert provenance['source_nifti_conversion'] == sidecar(out)['nifti_conversion']
        assert provenance['source_nifti_generated_at'] == sidecar(out)['nifti_generated_at']


def test_governed_export_refuses_shifted_counterpart(tmp_path, dcm2niix):
    paths, rs, ov = make(tmp_path, 'D')
    clinical_roi(rs, paths, 'D')
    for path in (tmp_path/'ct').glob('second_*.dcm'):
        ds = pydicom.dcmread(path)
        ds.ImagePositionPatient[0] += .1
        ds.save_as(path, enforce_file_format=True)
    out = convert(tmp_path, rs, ov)
    assert out is not None
    assert export(tmp_path, rs, out) is None
    assert not list((tmp_path/'Segmentation_Original').rglob('*.nii*'))
