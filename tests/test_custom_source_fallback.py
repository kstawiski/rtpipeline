"""Fallback completion of custom sources from the non-base RTSTRUCT.

On 2026-09-22 one Kopernik course lost all radiomics because two clinician
iliac-vessel contours were partially unparseable while the automatic
same-name contours were clean: the base-scope veto aborted the whole custom
build. A source whose base geometry verdict says unusable may now be
completed from the non-base RTSTRUCT, with the segment origin recorded.
Read failures, declared-empty sources and truly-missing sources keep their
existing behavior: no completion, and the veto still fires when a source is
unavailable everywhere.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from rt_utils import RTStructBuilder

from test_science_batch_c import _build_real_rtstruct
from test_custom_structures_d16_auto_harvest import (
    _make_partially_unparseable,
    _patch_contract,
)
from rtpipeline import custom_structures_rtstruct as custom


def _vessel_course(tmp_path, monkeypatch, *, break_manual_b=True, auto_has_b=True,
                   break_auto_b=False, base_manual=True):
    """Manual + auto RTSTRUCTs sharing VES_A/VES_B; B broken per flags."""
    ct = tmp_path / "ct"
    ct.mkdir()
    rt_manual = _build_real_rtstruct(ct, n_slices=4, side=12)
    mask_a = np.zeros((12, 12, 4), dtype=bool); mask_a[2:5, 2:5, 1:3] = True
    mask_b = np.zeros((12, 12, 4), dtype=bool); mask_b[2:5, 7:10, 1:3] = True
    rt_manual.add_roi(mask=mask_a, name="VES_A")
    rt_manual.add_roi(mask=mask_b, name="VES_B")
    if break_manual_b:
        _make_partially_unparseable(rt_manual, "VES_B")
    rs_manual = tmp_path / "RS_manual.dcm"
    rt_manual.save(str(rs_manual))

    rt_auto = RTStructBuilder.create_new(dicom_series_path=str(ct))
    rt_auto.add_roi(mask=mask_a, name="VES_A")
    if auto_has_b:
        rt_auto.add_roi(mask=mask_b, name="VES_B")
        if break_auto_b:
            _make_partially_unparseable(rt_auto, "VES_B")
    rs_auto = tmp_path / "RS_auto.dcm"
    rt_auto.ds.save_as(str(rs_auto))

    series_uid = str(rt_manual.series_data[0].SeriesInstanceUID)
    _patch_contract(tmp_path, ct, rs_manual if base_manual else None,
                    series_uid, monkeypatch)
    config = tmp_path / "custom.yaml"
    config.write_text(
        "custom_structures:\n"
        "  - name: test_vess\n"
        "    source_structures: [VES_A, VES_B]\n"
        "    operation: union\n",
        encoding="utf-8",
    )
    return rs_manual, rs_auto, config


def _meta(tmp_path):
    return json.loads((tmp_path / "metadata" / "rs_custom_meta.json").read_text())


def test_unparseable_manual_source_completed_from_clean_auto(tmp_path, monkeypatch):
    rs_manual, rs_auto, config = _vessel_course(tmp_path, monkeypatch)
    out = custom._create_custom_structures_rtstruct_unlocked(
        tmp_path, config, rs_manual, rs_auto
    )
    assert out is not None and out.exists()
    published = RTStructBuilder.create_from(
        str(tmp_path / "ct"), str(out)
    ).get_roi_names()
    assert "test_vess" in published
    assert "test_vess__partial" not in published
    meta = _meta(tmp_path)
    outcome = meta["custom_structure_outcomes"]["test_vess"]
    assert outcome["status"] == "generated"
    assert outcome["fallback_segment_origins"]["VES_B"]["origin"] == "auto"
    assert outcome["fallback_segment_origins"]["VES_B"]["base_code"] == \
        "ROI_CONTOUR_PARTIALLY_UNPARSEABLE"
    assert meta["unread_source_rois"]["VES_B"]["label"].startswith("base:")


def test_unparseable_auto_source_completed_from_clean_manual(tmp_path, monkeypatch):
    rs_manual, rs_auto, config = _vessel_course(
        tmp_path, monkeypatch, break_manual_b=False, break_auto_b=True,
        base_manual=False,
    )
    out = custom._create_custom_structures_rtstruct_unlocked(
        tmp_path, config, rs_manual, rs_auto
    )
    assert out is not None and out.exists()
    meta = _meta(tmp_path)
    outcome = meta["custom_structure_outcomes"]["test_vess"]
    assert outcome["status"] == "generated"
    assert outcome["fallback_segment_origins"]["VES_B"]["origin"] == "manual"


def test_source_unusable_everywhere_still_aborts(tmp_path, monkeypatch):
    rs_manual, rs_auto, config = _vessel_course(
        tmp_path, monkeypatch, auto_has_b=False
    )
    with pytest.raises(Exception) as excinfo:
        custom._create_custom_structures_rtstruct_unlocked(
            tmp_path, config, rs_manual, rs_auto
        )
    assert "CUSTOM_SOURCE_GEOMETRY_UNRESOLVED" in str(excinfo.value)
    assert "VES_B" in str(excinfo.value)


def test_read_failure_is_never_completed_from_auto(tmp_path, monkeypatch):
    """An empty rasterisation is a read failure, not a geometry verdict: the
    manual authority rule keeps it unavailable even with a clean auto contour."""
    from rtpipeline import rtstruct_geometry
    rs_manual, rs_auto, config = _vessel_course(
        tmp_path, monkeypatch, break_manual_b=False
    )
    original = rtstruct_geometry.ScopedRTStruct.get_roi_mask_by_name

    def patched(self, name):
        if name == "VES_B":
            return np.zeros((12, 12, 4), dtype=bool)
        return original(self, name)

    monkeypatch.setattr(
        rtstruct_geometry.ScopedRTStruct, "get_roi_mask_by_name", patched
    )
    out = custom._create_custom_structures_rtstruct_unlocked(
        tmp_path, config, rs_manual, rs_auto
    )
    assert out is not None and out.exists()
    meta = _meta(tmp_path)
    outcome = meta["custom_structure_outcomes"]["test_vess"]
    assert "fallback_segment_origins" not in outcome
    published = RTStructBuilder.create_from(
        str(tmp_path / "ct"), str(out)
    ).get_roi_names()
    assert "test_vess" not in published
    assert "test_vess__partial" in published
