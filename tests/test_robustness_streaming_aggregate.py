"""Bounded aggregation equivalence and admission regressions; synthetic only."""
from dataclasses import replace
from functools import wraps

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from rtpipeline import radiomics_robustness as rr
from rtpipeline import radiomics_robustness_aggregate as ra
import test_robustness_aggregation_admission as admission
import test_robustness_cohort_snapshot_contract as snapshots


def _equal_outputs(old, new):
    with pd.ExcelFile(old) as first, pd.ExcelFile(new) as second:
        assert first.sheet_names == second.sheet_names
        for sheet in first.sheet_names:
            pd.testing.assert_frame_equal(first.parse(sheet), second.parse(sheet), check_exact=True)
    old_raw = rr.robustness_cohort_output_paths(old)[1]
    new_raw = rr.robustness_cohort_output_paths(new)[1]
    assert pq.read_schema(old_raw).equals(pq.read_schema(new_raw), check_metadata=True)
    pd.testing.assert_frame_equal(pd.read_parquet(old_raw), pd.read_parquet(new_raw), check_exact=True)


@pytest.mark.parametrize("kinds", [("measured", "measured", "source_only"), ("source_only",)])
def test_manifest_equivalence(tmp_path, monkeypatch, kinds):
    _, _, cohort, config, courses = snapshots._fixture(tmp_path, monkeypatch, kinds)
    old, new = tmp_path / "old.xlsx", tmp_path / "new.xlsx"
    rr.aggregate_robustness_cohort(courses, old, config, cohort=cohort)
    refs = [ra.robustness_course_reference(c.course_dir, patient_id=c.patient_id,
                                          course_id=c.course_id) for c in courses]
    ra.aggregate_robustness_cohort(refs, new, config, cohort=cohort)
    _equal_outputs(old, new)


@pytest.mark.parametrize("variant", ["normal", "constant", "nan", "not_evaluable", "optional_columns", "roi_name"])
def test_metric_and_raw_equivalence(tmp_path, monkeypatch, variant):
    cohort = admission._build_cohort(tmp_path, monkeypatch, patients=("P1", "P2", "P3"))
    admitted = [rr._admit_robustness_aggregation_input(p, cohort.rob) for p in cohort.tables]
    # Admission itself is tested below without substitution. Here certified
    # frames are varied to exercise numerical and schema edge cases directly.
    frames = {}
    for index, entry in enumerate(admitted):
        frame = entry.frame.copy()
        frame = pd.concat([frame.assign(segmentation_source=source, structure=structure)
                           for source, structure in [("Manual", "ROI"), ("Auto", "ROI"),
                                                     ("Auto", "Other")]], ignore_index=True)
        if variant == "constant":
            frame["value"] = 42.0
        elif variant == "nan":
            frame.loc[frame.index[0], "value"] = np.nan
        elif variant == "not_evaluable":
            frame["reason_code"] = None
            if index == 0:
                frame.loc[0, ["value", "robustness_status", "reason_code"]] = [
                    np.nan, rr.ROBUSTNESS_FEATURE_NOT_EVALUABLE_STATUS, "synthetic_gap"]
        elif variant == "optional_columns" and index == 1:
            frame["optional_integer"] = 7
            frame["reason_code"] = "example"
        elif variant == "roi_name":
            frame = frame.rename(columns={"structure": "roi_name"})
        frames[entry.path] = replace(entry, frame=frame)
    monkeypatch.setattr(rr, "_admit_robustness_aggregation_input", lambda p, cfg: frames[p])
    old, new = tmp_path / "old.xlsx", tmp_path / "new.xlsx"
    rr.aggregate_robustness_results(cohort.tables, old, cohort.rob)
    ra.aggregate_robustness_results(cohort.tables, new, cohort.rob)
    _equal_outputs(old, new)


# Reuse the existing, independently specified admission and mutation contracts
# against the streaming entry points as well as their original legacy tests.
@pytest.fixture
def streaming(monkeypatch):
    monkeypatch.setattr(rr, "aggregate_robustness_results", ra.aggregate_robustness_results)
    monkeypatch.setattr(rr, "aggregate_robustness_cohort", ra.aggregate_robustness_cohort)


def _streaming_contract(test):
    @wraps(test)
    def wrapped(*args, **kwargs):
        return test(*args, **kwargs)
    return pytest.mark.usefixtures("streaming")(wrapped)


for _module in (admission, snapshots):
    for _name, _test in vars(_module).items():
        if _name.startswith("test_"):
            globals()["test_streaming_" + _name[5:]] = _streaming_contract(_test)


@pytest.mark.parametrize("failure", ["technical_failure", "geometrically_impossible", "duplicate", "missing_grid", "missing_feature"])
def test_metric_refusals_match_legacy(tmp_path, monkeypatch, failure):
    cohort = admission._build_cohort(tmp_path, monkeypatch)
    entries = [rr._admit_robustness_aggregation_input(p, cohort.rob) for p in cohort.tables]
    frame = entries[0].frame.copy()
    if failure in {"technical_failure", "geometrically_impossible"}:
        frame.loc[0, "robustness_status"] = failure
    elif failure == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif failure == "missing_grid":
        frame = frame.iloc[1:]
    else:
        extra = frame.copy()
        extra["feature_name"] = "additional_feature"
        frame = pd.concat([frame, extra], ignore_index=True)
    entries[0] = replace(entries[0], frame=frame)
    supplied = {entry.path: entry for entry in entries}
    monkeypatch.setattr(rr, "_admit_robustness_aggregation_input", lambda p, cfg: supplied[p])
    for module in (rr, ra):
        output, raw = admission._summary_paths(tmp_path)
        admission._plant_stale_outputs(output, raw)
        with pytest.raises((RuntimeError, ValueError)):
            module.aggregate_robustness_results(cohort.tables, output, cohort.rob)
        admission._assert_no_nominal_outputs(output, raw)


def test_unresolved_quarantine_refused(tmp_path, monkeypatch):
    _, _, cohort, config, courses = snapshots._fixture(tmp_path, monkeypatch)
    cohort["technical_quarantine_count"] = 1
    output, raw = admission._summary_paths(tmp_path)
    admission._plant_stale_outputs(output, raw)
    with pytest.raises(RuntimeError, match="unresolved technical quarantine"):
        ra.aggregate_robustness_cohort(courses, output, config, cohort=cohort)
    admission._assert_no_nominal_outputs(output, raw)


def test_reference_path_releases_each_admitted_frame(tmp_path, monkeypatch):
    import gc
    import weakref

    _, _, cohort, config, courses = snapshots._fixture(
        tmp_path, monkeypatch, ("measured", "measured", "source_only")
    )
    references = [ra.robustness_course_reference(c.course_dir, patient_id=c.patient_id,
                                                course_id=c.course_id) for c in courses]
    del courses
    admit = rr.admit_robustness_cohort_course
    previous = None
    calls = 0

    def one_live_frame(*args, **kwargs):
        nonlocal previous, calls
        gc.collect()
        assert previous is None or previous() is None, "previous course frame survived admission"
        result = admit(*args, **kwargs)
        previous = weakref.ref(result.frame) if result.frame is not None else None
        calls += 1
        return result

    monkeypatch.setattr(rr, "admit_robustness_cohort_course", one_live_frame)
    ra.aggregate_robustness_cohort(references, tmp_path / "summary.xlsx", config, cohort=cohort)
    assert calls == 2 * len(references)  # Initial admission and pre-publication recheck.
