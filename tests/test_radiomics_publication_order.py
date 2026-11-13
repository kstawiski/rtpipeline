"""Radiomics publication order must not depend on worker scheduling or resume.

Parallel backends finish ROIs in whatever order the scheduler allows, and a
resume mixes reused and fresh rows. The published table is the order a serial
run produces: ROI enumeration order, then extraction-arm order.
"""
from __future__ import annotations

import random
from concurrent.futures import Future
from typing import Any, cast

import pandas as pd
import pytest
from pydicom.uid import generate_uid

import rtpipeline.radiomics as rad
import rtpipeline.radiomics_parallel as radpar
from rtpipeline.radiomics_ct_contract import (
    CT_EXTRACTION_ARMS,
    PRIMARY_ARM,
    SENSITIVITY_ARM,
    base_identity_key,
    order_rows_by_enumeration,
)
from course_contract_test_utils import write_synthetic_rtstruct
from test_b4_all_series_radiomics import _Cfg, _write_valid_ct

pytestmark = pytest.mark.usefixtures("rasterizer_guard_stub")


def _row(roi: str, arm: str, **extra: Any) -> dict[str, Any]:
    return {"patient_id": "P1", "course_id": "C1", "series_uid": "1.2.3",
            "segmentation_source": "Manual", "mask_identity": "m",
            "roi_original_name": roi, "stable_roi_identifier": f"rtstruct_roi_number:{roi}",
            "extraction_arm": arm, **extra}


def test_rows_follow_enumeration_then_arm_order_for_any_arrival_order():
    rois = ["PTV", "CTV", "Bladder", "Rectum"]
    expected = [_row(roi, arm) for roi in rois for arm in CT_EXTRACTION_ARMS]
    enumeration = [[base_identity_key(_row(roi, ""))] for roi in rois]
    generator = random.Random(20261113)
    for _ in range(25):
        arrived = list(expected)
        generator.shuffle(arrived)
        assert order_rows_by_enumeration(arrived, base_identity_key, enumeration) == expected


def test_reused_rows_take_their_enumerated_place_not_the_end():
    # A resume appends reused rows after fresh ones; publication must not.
    fresh = [_row("Rectum", PRIMARY_ARM), _row("Rectum", SENSITIVITY_ARM)]
    reused = [_row("PTV", SENSITIVITY_ARM), _row("PTV", PRIMARY_ARM)]
    enumeration = [[base_identity_key(_row(roi, ""))] for roi in ("PTV", "Rectum")]
    ordered = order_rows_by_enumeration(fresh + reused, base_identity_key, enumeration)
    assert [(row["roi_original_name"], row["extraction_arm"]) for row in ordered] == [
        ("PTV", PRIMARY_ARM), ("PTV", SENSITIVITY_ARM),
        ("Rectum", PRIMARY_ARM), ("Rectum", SENSITIVITY_ARM)]


def test_unknown_rows_stay_last_for_the_identity_check_to_refuse():
    enumeration = [[base_identity_key(_row("PTV", ""))]]
    stray = _row("Unlisted", PRIMARY_ARM)
    ordered = order_rows_by_enumeration(
        [stray, _row("PTV", PRIMARY_ARM)], base_identity_key, enumeration)
    assert ordered[-1] is stray


class _OrderedExecutor:
    """Completes every submitted task at once; ``reverse`` flips arrival order."""

    def __init__(self, reverse: bool):
        self.reverse = reverse
        self.submitted: list[Future] = []

    def submit(self, _function, task):
        future = Future()
        future.set_result(radpar._status_records(task, "success", "test success"))
        self.submitted.append(future)
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        return None


def _publish(tmp_path, monkeypatch, *, reverse: bool, name: str):
    src = tmp_path / f"series-{name}"
    src.mkdir()
    series_uid = generate_uid()
    _write_valid_ct(src / "slice.dcm", series_uid=series_uid, study_uid=generate_uid())
    rs = write_synthetic_rtstruct(tmp_path / f"rs-{name}.dcm", referenced_series_uid=series_uid,
                                  roi_names=("PTV1", "CTV1", "GTV1", "Bladder", "Rectum"))
    course = tmp_path / f"root-{name}" / ".all_series_radiomics" / "P1" / "series"
    assert rad._materialize_temp_course_tree(course, src, rs) is True
    monkeypatch.setattr(rad, "_load_series_image", lambda *_a, **_k: object())
    monkeypatch.setattr(rad, "_extractor", lambda *_a, **_k: object())
    monkeypatch.setattr(radpar, "effective_parameter_hashes_for_arms", lambda *_a, **_k: {
        PRIMARY_ARM: "effective-primary", SENSITIVITY_ARM: "effective-sensitivity"})
    monkeypatch.setattr(radpar, "validate_custom_model_output_inventory", lambda *_a, **_k: {})
    monkeypatch.setattr(radpar, "list_custom_model_outputs", lambda *_a, **_k: [])
    monkeypatch.setattr(radpar, "_calculate_optimal_workers", lambda: 8)
    executor = _OrderedExecutor(reverse)
    monkeypatch.setattr(radpar, "ProcessPoolExecutor", lambda **_k: executor)

    def arrival(futures, timeout=None, return_when=None):
        # Hand back every submitted future at once, in submission or reverse order.
        done = [future for future in executor.submitted if future in futures]
        return (list(reversed(done)) if reverse else done), set()

    monkeypatch.setattr(radpar, "wait", arrival)
    outcome = radpar.parallel_radiomics_for_course(
        cast(Any, _Cfg(tmp_path)), course, max_workers=8, allow_all_series_temp=True)
    assert len(executor.submitted) == 5
    frame = pd.read_parquet(outcome.output_path.with_suffix(".parquet"), engine="pyarrow")
    return frame


def test_native_publication_is_independent_of_completion_order(tmp_path, monkeypatch):
    forward = _publish(tmp_path, monkeypatch, reverse=False, name="forward")
    backward = _publish(tmp_path, monkeypatch, reverse=True, name="backward")
    identity = ["segmentation_source", "roi_original_name", "extraction_arm"]
    assert len(forward) == 10
    assert (list(forward[identity].itertuples(index=False, name=None))
            == list(backward[identity].itertuples(index=False, name=None)))
    # Arms of one ROI stay adjacent and in contract order.
    arms = forward["extraction_arm"].tolist()
    assert arms == [PRIMARY_ARM, SENSITIVITY_ARM] * 5
