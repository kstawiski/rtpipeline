"""RF18/RF15: robustness reads a strictly unparseable brachytherapy CTV as main radiomics does."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from rt_utils import image_helper

from rtpipeline import radiomics, rtstruct_geometry as geometry
from rtpipeline.radiomics_robustness import BRACHYTHERAPY_CTV_PATTERNS, robustness_selection_matches, with_brachytherapy_ctv_patterns
from rtpipeline.roi_requiredness import Requiredness
from test_quantized_contours import fixture


@pytest.fixture(autouse=True)
def synthetic_shape(monkeypatch):
    monkeypatch.setattr(image_helper, 'create_empty_series_mask',
                        lambda series: np.zeros((64, 64, len(series)), dtype=bool))
    import rtpipeline.rtstruct_identity as identity
    monkeypatch.setattr(identity, 'require_rtstruct_identity', lambda path: 'synthetic-sop-uid')


def _hr_ctv(ds):
    ds.StructureSetROISequence[0].ROIName = 'HR-CTV'
    return ds


def _read(monkeypatch, ds, images, *, rescue, later_marker=False):
    monkeypatch.setattr(radiomics, '_scoped_rtstruct_readers',
                        lambda series, rs: (geometry.ScopedRTStruct(ds, images), None))
    contours = list(ds.ROIContourSequence[0].ContourSequence)
    strict = geometry.roi_geometry_code(contours, allow_quantization=False)
    named_rois = [SimpleNamespace(name='HR-CTV', structural_code=strict, roi_number=1)]
    if later_marker:
        # Recorded from the inventory after HR-CTV, before any mask is read.
        named_rois.append(SimpleNamespace(
            name='Marker', structural_code='ROI_NONVOLUMETRIC_POINT', roi_number=2))
    inventory = SimpleNamespace(
        named_rois=named_rois,
        structural_codes=tuple(o.structural_code for o in named_rois if o.structural_code))
    outcomes = []
    masks = radiomics._rtstruct_masks(
        Path('ct'), Path('RS.dcm'), failure_outcomes=outcomes, tolerate_unselected=True,
        requiredness_by_roi={'HR-CTV': Requiredness.ANALYSIS_REQUIRED},
        contourless_required_is_absence=True, unmeasurable_required_is_disposition=True,
        structural_inventory=inventory, scoped_reader=True, quantized_scope_rescue=rescue)
    return strict, masks, outcomes


def test_brachytherapy_selection_reaches_hr_ctv():
    selection = with_brachytherapy_ctv_patterns(['GTV*', 'CTV*', 'PTV*'])
    assert set(BRACHYTHERAPY_CTV_PATTERNS) <= set(selection)
    assert robustness_selection_matches('HR-CTV', selection)
    assert not robustness_selection_matches('HR-CTV', ['GTV*', 'CTV*', 'PTV*'])


@pytest.mark.usefixtures("rasterizer_guard_stub")
def test_scope_only_quantization_was_already_read_and_still_is(monkeypatch):
    # Strict inventory accepts it; only the CT-plane scope needs RF15. Unaffected by the flag.
    for rescue in (False, True):
        ds, images = fixture(0.1)
        strict, masks, outcomes = _read(monkeypatch, _hr_ctv(ds), images, rescue=rescue)
        assert strict is None and outcomes == [] and masks['HR-CTV'].any()


@pytest.mark.usefixtures("rasterizer_guard_stub")
def test_quantized_hr_ctv_is_read_through_projection(monkeypatch):
    # The production class: the strict inventory itself refuses the contours.
    ds, images = fixture(0.25)
    strict, masks, outcomes = _read(monkeypatch, _hr_ctv(ds), images, rescue=True)
    assert strict in radiomics._QUANTIZATION_CANDIDATE_CODES
    assert outcomes == []
    expected = geometry.ScopedRTStruct(ds, images).get_roi_mask_by_name('HR-CTV')
    assert masks['HR-CTV'] is not None and masks['HR-CTV'].any()
    assert masks['HR-CTV'].tobytes() == expected.tobytes()


def test_without_rescue_the_inventory_disposition_is_unchanged(monkeypatch):
    ds, images = fixture(0.25)
    strict, masks, outcomes = _read(monkeypatch, _hr_ctv(ds), images, rescue=False)
    assert 'HR-CTV' not in masks
    assert [o['status'] for o in outcomes] == ['structural_nonmeasurement']
    assert outcomes[0]['failure_kind'] == 'unmeasurable_source_contour'
    assert outcomes[0]['structural_code'] == strict


@pytest.mark.usefixtures("rasterizer_guard_stub")
def test_unresolvable_contour_keeps_the_identical_disposition(monkeypatch):
    def broken():
        ds, images = fixture(0., rounded=False)
        contour = ds.ROIContourSequence[0].ContourSequence[0]
        points = np.array(contour.ContourData).reshape(-1, 3)
        points[2, 2] += 0.08  # fitted-plane residual beyond the RF15 bound
        contour.ContourData = points.ravel().tolist()
        return _hr_ctv(ds), images
    _, masks_off, off = _read(monkeypatch, *broken(), rescue=False)
    strict, masks_on, on = _read(monkeypatch, *broken(), rescue=True)
    assert strict in radiomics._QUANTIZATION_CANDIDATE_CODES
    assert 'HR-CTV' not in masks_on and 'HR-CTV' not in masks_off
    assert on == off and len(on) == 1


def _unresolvable():
    ds, images = fixture(0., rounded=False)
    contour = ds.ROIContourSequence[0].ContourSequence[0]
    points = np.array(contour.ContourData).reshape(-1, 3)
    points[2, 2] += 0.08  # fitted-plane residual beyond the RF15 bound
    contour.ContourData = points.ravel().tolist()
    return _hr_ctv(ds), images


@pytest.mark.usefixtures("rasterizer_guard_stub")
def test_unresolved_disposition_keeps_its_place_among_other_rows(monkeypatch):
    # The deferred row must not move behind rows recorded later from the
    # inventory: published dispositions are hashed in their recorded order.
    _, _, off = _read(monkeypatch, *_unresolvable(), rescue=False, later_marker=True)
    _, _, on = _read(monkeypatch, *_unresolvable(), rescue=True, later_marker=True)
    assert [o['roi_name'] for o in off] == ['HR-CTV', 'Marker']
    assert on == off


@pytest.mark.usefixtures("rasterizer_guard_stub")
def test_resolved_roi_withdraws_only_its_own_row(monkeypatch):
    ds, images = fixture(0.25)
    _, _, off = _read(monkeypatch, _hr_ctv(ds), images, rescue=False, later_marker=True)
    ds, images = fixture(0.25)
    _, masks, on = _read(monkeypatch, _hr_ctv(ds), images, rescue=True, later_marker=True)
    assert [o['roi_name'] for o in off] == ['HR-CTV', 'Marker']
    assert on == off[1:] and masks['HR-CTV'].any()


def test_robustness_opts_in_for_the_source_rtstruct():
    import inspect
    from rtpipeline import radiomics_robustness
    source = inspect.getsource(radiomics_robustness)
    call = source[source.index('source_masks = _rtstruct_masks('):]
    call = call[:call.index('_record_source_dispositions(source_binding, source_sink)')]
    assert 'quantized_scope_rescue=True' in call and 'scoped_reader=True' in call
