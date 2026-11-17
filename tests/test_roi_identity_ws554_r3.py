"""Released export compatibility and refusal of ambiguous legacy inventories."""
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pydicom
import pytest

from legacy_source import legacy_show
from test_roi_identity_ws554 import source, names, builder, export
from rtpipeline import organize
from rtpipeline.original_segmentation import (
    original_roi_filenames, require_current_original_export, require_current_original_mask,
)


def _legacy_module(relative, name):
    module = types.ModuleType('rtpipeline.' + name)
    module.__file__ = str(Path(organize.__file__).with_name(relative))
    sys.modules[module.__name__] = module
    exec(compile(legacy_show('4ad1c53', 'rtpipeline/' + relative), module.__file__, 'exec'), module.__dict__)
    return module


def test_collision_free_export_is_byte_identical_to_303(source, tmp_path):
    names(source[1], ['First', 'Second'])
    fixture = Path(__file__).with_name('ws554_303_export_fixture.py.txt').read_bytes()
    assert hashlib.sha256(fixture).hexdigest() == '186ea63c985b30b30b361471bfdb30f05c41ee7e6663d22b75d04e16eb45ad5b'
    namespace = dict(vars(organize))
    exec(compile(fixture, 'ws554_303_export_fixture.py.txt', 'exec'), namespace)
    root = tmp_path / 'Segmentation_Original'
    args = dict(rs_path=source[1], primary_nifti=source[2], dicom_ct_dir=source[0],
                segmentation_original_dir=root, log_root=tmp_path, overwrite=True)
    old = namespace['_export_original_segmentation_from_paths'](**args)
    before = {p.name: p.read_bytes() for p in (root / 'ct').iterdir()}
    assert old['schema'] == 'rtpipeline-original-segmentation-v2'
    assert all('roi_number' not in entry for entry in old['structures'])
    assert export(source, root) == old
    assert {p.name: p.read_bytes() for p in (root / 'ct').iterdir()} == before
    require_current_original_export(root / 'ct')
    # Released manifests remain reusable without quarantine or touching bytes.
    args['overwrite'] = False
    mtimes = {p.name: p.stat().st_mtime_ns for p in (root / 'ct').iterdir()}
    assert organize._export_original_segmentation_from_paths(**args) == old
    assert {p.name: p.stat().st_mtime_ns for p in (root / 'ct').iterdir()} == mtimes


def test_numbered_masks_match_pinned_geometry_on_unique_names(source):
    from rtpipeline.rtstruct_geometry import rtstruct_roi_mask
    names(source[1], ['First', 'Second'])
    old = _legacy_module('rtstruct_geometry.py', '_ws554_pinned_geometry')
    reader = builder(source)
    for number, name in enumerate(reader.get_roi_names(), 1):
        assert old.rtstruct_roi_mask(reader, name).tobytes() == rtstruct_roi_mask(reader, name, roi_number=number).tobytes()


def test_custom_outputs_match_pinned_processor_without_ambiguity(source):
    from rtpipeline.custom_structures import CustomStructureProcessor, CustomStructureConfig
    old = _legacy_module('custom_structures.py', '_ws554_pinned_custom')
    available = {'First': source[3], 'Second': np.roll(source[3], 1, axis=0) | source[4]}
    for operation in ['union', 'intersection', 'subtract', 'xor']:
        current, historical = CustomStructureProcessor(), old.CustomStructureProcessor()
        config = dict(name='Derived', source_structures=['FIRST', 'Second'], operation=operation)
        actual = current.process_custom_structure(CustomStructureConfig(**config), available)
        expected = historical.process_custom_structure(old.CustomStructureConfig(**config), available)
        assert actual.tobytes() == expected.tobytes()
        assert current.structure_outcomes == historical.structure_outcomes


@pytest.mark.parametrize('labels', [
    ['A B', 'A_B', 'A_B_2'], ['Organ', 'organ'], ['Organ', ' Organ'],
    ['A B', 'A_B', 'A_B__roi-1'],
])
def test_only_complete_collision_groups_are_numbered(labels):
    entries = list(enumerate(labels + ['Unrelated'], 1))
    filenames, colliding = original_roi_filenames(entries)
    assert len({name.casefold() for name in filenames}) == len(filenames)
    assert colliding == set(range(len(labels)))
    assert filenames[-1] == 'Unrelated.nii.gz'
    assert all(f'__roi-{index + 1}' in filenames[index] for index in colliding)


def _old_ambiguous_export(source, tmp_path, mode):
    if mode == 'second_order':
        reader = builder(source)
        reader.ds.StructureSetROISequence[0].ROIName = 'A B'
        reader.ds.StructureSetROISequence[1].ROIName = 'A_B'
        reader.add_roi(mask=source[3] | source[4], name='A_B_2')
        reader.save(str(source[1]))
    root = tmp_path / 'Segmentation_Original'
    manifest = export(source, root)
    manifest['schema'] = 'rtpipeline-original-segmentation-v2'
    for entry in manifest['structures']:
        entry.pop('roi_number')
    if mode == 'second_order':
        # The released writer could bind both entries to the overwritten file.
        manifest['structures'][1].update({key: manifest['structures'][2][key]
                                         for key in ['mask', 'sha256', 'size_bytes']})
    (root / 'ct/metadata.json').write_text(json.dumps(manifest))
    return root, manifest


@pytest.mark.parametrize('mode', ['duplicate', 'second_order'])
@pytest.mark.parametrize('consumer', ['export', 'mask', 'anchor', 'dice', 'robustness', 'qc', 'reuse'])
def test_303_ambiguous_manifest_is_refused_by_consumers(source, tmp_path, mode, consumer):
    from rtpipeline.segmenters.postprocessing import derive_anchor_from_reference, _resolve_reference_paths
    from rtpipeline.quality_control import _qc_inputs
    from rtpipeline.radiomics_robustness import _robustness_voxel_target_inventory
    root, manifest = _old_ambiguous_export(source, tmp_path, mode)
    mask = root / manifest['structures'][0]['mask']
    if consumer == 'reuse':
        valid, reason = organize._original_masks_reusable(root / 'ct', root, manifest,
            organize._original_segmentation_provenance(source[1], source[2]))
        assert not valid and ('ROI' in reason or 'uniquely' in reason)
        return
    calls = {
        'export': lambda: require_current_original_export(root / 'ct'),
        'mask': lambda: require_current_original_mask(mask),
        'anchor': lambda: derive_anchor_from_reference(mask),
        'dice': lambda: _resolve_reference_paths(mask),
        'robustness': lambda: _robustness_voxel_target_inventory([
            {'source_path': str(mask), 'segmentation_source': 'Manual', 'roi_name': manifest['structures'][0]['roi_name']}]),
        'qc': lambda: _qc_inputs(tmp_path),
    }
    with pytest.raises((ValueError, RuntimeError), match='(ROI|uniquely)'):
        calls[consumer]()


@pytest.mark.parametrize('mode', ['duplicate', 'second_order'])
def test_reexport_quarantines_ambiguous_303_manifest_and_uses_source_numbers(source, tmp_path, mode):
    from rtpipeline.reexport_original import reexport_original
    root, manifest = _old_ambiguous_export(source, tmp_path, mode)
    result = reexport_original(rtstruct=source[1], ct_series=source[0], reference_nifti=source[2], output_root=tmp_path)
    assert result['status'] == 'exported'
    receipts = list((tmp_path / 'Segmentation_Quarantine').glob('*/record.json'))
    assert receipts and all('stale' in json.loads(p.read_text())['reason'] for p in receipts)
    current = json.loads((root / 'ct/metadata.json').read_text())
    assert [entry['roi_number'] for entry in current['structures']] == list(range(1, len(current['structures']) + 1))
    require_current_original_export(root / 'ct')


@pytest.mark.parametrize('mode', ['empty', 'read_failed'])
def test_unique_omissions_preserve_303_export_bytes_and_record_reason(source, tmp_path, mode):
    from test_roi_identity_ws554_r2 import _omit
    names(source[1], ['First', 'Second'])
    _omit(source, mode)
    namespace = dict(vars(organize))
    exec(compile(Path(__file__).with_name('ws554_303_export_fixture.py.txt').read_bytes(),
                 'ws554_303_export_fixture.py.txt', 'exec'), namespace)
    root = tmp_path / 'Segmentation_Original'
    old = namespace['_export_original_segmentation_from_paths'](
        rs_path=source[1], primary_nifti=source[2], dicom_ct_dir=source[0],
        segmentation_original_dir=root, log_root=tmp_path, overwrite=True)
    before = {p.name: p.read_bytes() for p in (root / 'ct').iterdir()}
    assert export(source, root) == old
    assert {p.name: p.read_bytes() for p in (root / 'ct').iterdir()} == before
    omissions = json.loads((tmp_path / 'metadata/original_segmentation_omissions/ct.json').read_text())
    assert omissions['skipped_structures'] == [{'roi_number': 2, 'roi_name': 'Second',
        'reason_code': 'ROI_MASK_EMPTY' if mode == 'empty' else 'ROI_MASK_READ_FAILED'}]


def test_missing_roi_numbers_use_only_unique_names():
    from types import SimpleNamespace
    from rtpipeline.rtstruct_geometry import rtstruct_roi_mask
    mask = np.ones((2, 2, 2), dtype=bool)
    reader = SimpleNamespace(ds=SimpleNamespace(StructureSetROISequence=[SimpleNamespace(ROIName='Unique')]),
                             get_roi_mask_by_name=lambda name: mask)
    assert rtstruct_roi_mask(reader, 'Unique') is mask
    reader.ds.StructureSetROISequence.append(SimpleNamespace(ROIName='Unique'))
    with pytest.raises(ValueError, match='RTSTRUCT_AMBIGUOUS_ROI_NAME'):
        rtstruct_roi_mask(reader, 'Unique')
