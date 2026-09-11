"""Manager checks for aggregation against real generated RTSTRUCT source bytes.

Course contract/catalog and feature extraction are mocked by the inspected
fixture. CT and RTSTRUCT files, rasterization, sidecar admission and aggregation
are real. This is not clinical data, a full NTCV grid, or executed-code proof.
"""
import json

import pandas as pd
import pydicom
import pytest

from rtpipeline import radiomics_robustness as rr
from test_robustness_nonmeasurements import _real_mixed_course


@pytest.mark.parametrize("change", ["rtstruct_source", "deciding_code"])
def test_aggregation_rejects_changed_bound_source_or_code(tmp_path, monkeypatch, change):
    course, config, robustness, source = _real_mixed_course(tmp_path, monkeypatch)
    stub_features = rr.extract_features_for_masks

    def run_bound_features(*args, **kwargs):
        frame = stub_features(*args, **kwargs)
        frame["run_identifier"] = kwargs["run_identifier"]
        return frame

    monkeypatch.setattr(rr, "extract_features_for_masks", run_bound_features)
    table = rr.robustness_for_course(config, robustness, course)
    assert table is not None
    sidecar = json.loads(rr.robustness_source_dispositions_path(course).read_text())
    assert sidecar["source_bindings"]
    assert sidecar["source_bindings"][0]["source_path"] == str(source)
    assert any(row["roi_name"] == "Marker1" for row in sidecar["rows"])

    output = tmp_path / "_RESULTS" / "summary.xlsx"
    raw = output.with_name("summary_raw_values.parquet")
    rr.aggregate_robustness_results([table], output, robustness)
    assert output.is_file() and raw.is_file()
    measured = pd.read_parquet(raw)
    assert set(measured["structure"]) == {"ROI"}
    assert set(measured["robustness_status"]) == {"measured"}

    if change == "rtstruct_source":
        dataset = pydicom.dcmread(source)
        sop_uid = str(dataset.SOPInstanceUID)
        dataset.StructureSetLabel = "CHANGED"
        dataset.save_as(source)
        assert str(pydicom.dcmread(source).SOPInstanceUID) == sop_uid
    else:
        original_digest = rr._file_sha256

        def changed_code_digest(path):
            if path.name == "radiomics_robustness_outcomes.py":
                return "0" * 64
            return original_digest(path)

        monkeypatch.setattr(rr, "_file_sha256", changed_code_digest)

    with pytest.raises(ValueError, match="content changed|produced by different code"):
        rr.aggregate_robustness_results([table], output, robustness)
    assert not output.exists()
    assert not raw.exists()
