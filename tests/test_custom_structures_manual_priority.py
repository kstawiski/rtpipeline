"""Synthetic source-priority checks: manual names are not automatic fallbacks."""
import json
from pathlib import Path

import numpy as np
import pytest
from rt_utils import RTStructBuilder

from rtpipeline import custom_structures_rtstruct as custom
from rtpipeline import rtstruct_geometry
from test_custom_structures_d16_auto_harvest import _manual_only_course


@pytest.mark.parametrize("failure", [None, "empty", "exception"])
def test_manual_priority_does_not_block_distinct_auto_roi(tmp_path, monkeypatch, failure):
    assert tmp_path.resolve().is_relative_to(Path("/tmp"))
    course = _manual_only_course(tmp_path, monkeypatch)
    manual_mask = RTStructBuilder.create_from(
        str(course.ct), str(course.rs_manual)
    ).get_roi_mask_by_name("PTV")
    auto = RTStructBuilder.create_new(dicom_series_path=str(course.ct))
    auto.add_roi(mask=course.automatic, name="PTV")
    distinct = np.zeros_like(course.automatic)
    distinct[7:10, 7:10, 1:3] = True
    auto.add_roi(mask=distinct, name="distinct_auto")
    auto_path = tmp_path / "RS_auto.dcm"
    auto.save(str(auto_path))
    # Verify the conflicting source is healthy before observing which ROIs the
    # actual integration reads; a bad automatic ROI cannot explain rejection.
    assert RTStructBuilder.create_from(str(course.ct), str(auto_path)).get_roi_mask_by_name("PTV").any()
    course.config.write_text(
        "custom_structures:\n"
        "  - name: ptv_union\n"
        "    source_structures: [PTV]\n"
        "    operation: union\n"
        "  - name: auto_union\n"
        "    source_structures: [distinct_auto]\n"
        "    operation: union\n",
        encoding="utf-8",
    )
    original_factory = rtstruct_geometry.create_scoped_rtstruct
    auto_reads = []

    def factory(ct_dir, source, *args, **kwargs):
        reader = original_factory(ct_dir, source, *args, **kwargs)
        original_read = reader.get_roi_mask_by_name

        def read_roi(name):
            if Path(source) == auto_path:
                auto_reads.append(name)
            if Path(source) == course.rs_manual and name == "PTV" and failure:
                if failure == "exception":
                    raise RuntimeError("synthetic manual reader failure")
                return np.zeros_like(original_read(name))
            return original_read(name)

        reader.get_roi_mask_by_name = read_roi
        return reader

    monkeypatch.setattr(rtstruct_geometry, "create_scoped_rtstruct", factory)
    out = custom._create_custom_structures_rtstruct_unlocked(
        tmp_path, course.config, course.rs_manual, auto_path
    )
    assert out is not None and out.exists()
    metadata = json.loads((tmp_path / "metadata" / "rs_custom_meta.json").read_text())
    published = RTStructBuilder.create_from(str(course.ct), str(out))
    assert metadata["custom_structure_outcomes"]["auto_union"]["status"] == "generated"
    assert np.array_equal(published.get_roi_mask_by_name("auto_union"), distinct)
    assert auto_reads == ["distinct_auto"], "manual names are reserved even when unread"
    ptv = metadata["custom_structure_outcomes"]["ptv_union"]
    if failure:
        assert ptv["status"] == "source_unavailable"
        assert ptv["available_sources"] == []
        unread = metadata["unread_source_rois"]["PTV"]
        assert unread["label"] == "base:manual"
        assert unread["code"] == ("ROI_MASK_READ_FAILED" if failure == "exception" else "ROI_RASTERISED_EMPTY")
        assert "ptv_union" not in published.get_roi_names()
    else:
        assert ptv["status"] == "generated"
        assert "PTV" not in metadata["unread_source_rois"]
        assert np.array_equal(published.get_roi_mask_by_name("ptv_union"), manual_mask)
