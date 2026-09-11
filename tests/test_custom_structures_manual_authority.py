"""Synthetic integration checks for failed manual versus healthy automatic ROIs."""
import json
from pathlib import Path

import numpy as np
import pytest
from rt_utils import RTStructBuilder

from rtpipeline import custom_structures_rtstruct as custom
from rtpipeline import rtstruct_geometry
from test_custom_structures_d16_auto_harvest import _manual_only_course


@pytest.mark.parametrize("failure", ["empty", "exception"])
def test_failed_manual_roi_is_not_replaced_by_same_name_auto_rtstruct(tmp_path, monkeypatch, failure):
    course = _manual_only_course(tmp_path, monkeypatch)
    auto = RTStructBuilder.create_new(dicom_series_path=str(course.ct))
    auto.add_roi(mask=course.automatic, name="PTV")
    auto_path = tmp_path / "RS_auto.dcm"
    auto.save(str(auto_path))
    original_factory = rtstruct_geometry.create_scoped_rtstruct
    failed_reads = []
    auto_reads = []

    def source_scoped_factory(ct_dir, source, *args, **kwargs):
        reader = original_factory(ct_dir, source, *args, **kwargs)
        original_read = reader.get_roi_mask_by_name

        def read_roi(name):
            if name == "PTV" and Path(source) == course.rs_manual:
                failed_reads.append(name)
                if failure == "exception":
                    raise RuntimeError("synthetic manual-only rasterizer failure")
                return np.zeros_like(original_read(name))
            result = original_read(name)
            if name == "PTV" and Path(source) == auto_path:
                auto_reads.append(name)
                assert result.any(), "automatic source must be genuinely nonempty"
            return result

        reader.get_roi_mask_by_name = read_roi
        return reader

    monkeypatch.setattr(rtstruct_geometry, "create_scoped_rtstruct", source_scoped_factory)
    output = custom._create_custom_structures_rtstruct_unlocked(
        tmp_path, course.config, course.rs_manual, auto_path
    )
    assert failed_reads, "the authoritative manual read failure must be exercised"
    assert output is not None
    metadata = json.loads((tmp_path / "metadata" / "rs_custom_meta.json").read_text())
    assert metadata["custom_structure_outcomes"]["ptv_union"]["status"] == "source_unavailable"
    assert metadata["unread_source_rois"]["PTV"]["label"].startswith("base:")
    published = RTStructBuilder.create_from(str(course.ct), str(output))
    assert "ptv_union" not in published.get_roi_names()
    assert "ptv_union__partial" not in published.get_roi_names()
