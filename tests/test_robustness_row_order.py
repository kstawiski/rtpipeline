"""Published robustness rows follow ROBUSTNESS_TABLE_ROW_ORDER, whatever their input order."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rtpipeline import radiomics_robustness as rr


def _table(rng):
    rows = []
    for source in ("Manual", "AutoRTS_total"):
        for roi in ("GTV1", "PTV1"):
            for pid in ("ntcv_v0", "ntcv_n10_v0", "ntcv_c1_v-15"):
                for arm in ("primary_resegmented", "sensitivity_raw"):
                    for feature in ("original_shape_VoxelVolume", "original_glcm_MCC", None):
                        rows.append({
                            "segmentation_source": source, "mask_identity": f"m-{roi}",
                            "stable_roi_identifier": f"s-{roi}", "structure": roi,
                            "perturbation_id": pid, "extraction_arm": arm,
                            "measurement_type": rr.ROBUSTNESS_MEASUREMENT_TYPE,
                            "feature_name": feature,
                            "value": np.nan if feature is None else float(rng.normal()),
                            "run_identifier": "run",
                        })
    return pd.DataFrame(rows)


def test_any_input_order_gives_the_same_rows_in_the_same_order():
    rng = np.random.default_rng(0)
    table = _table(rng)
    reference = rr._order_robustness_rows(table)
    for seed in range(5):
        shuffled = table.sample(frac=1.0, random_state=seed)
        pd.testing.assert_frame_equal(rr._order_robustness_rows(shuffled), reference,
                                      check_exact=True)
    # Same rows, values and dtypes; only the order differs.
    key = lambda f: f.astype(str).agg("|".join, axis=1).sort_values().tolist()
    assert key(reference) == key(table)
    assert list(reference.dtypes) == list(table.dtypes)
    assert list(reference.index) == list(range(len(table)))


def test_order_is_lexicographic_by_the_documented_key_with_missing_last():
    table = _table(np.random.default_rng(1))
    ordered = rr._order_robustness_rows(table)
    columns = list(rr.ROBUSTNESS_TABLE_ROW_ORDER)
    keys = [tuple((value is None or value != value, "" if value is None or value != value else str(value))
                  for value in row) for row in ordered[columns].itertuples(index=False)]
    assert keys == sorted(keys)
    assert ordered.feature_name.isna().to_numpy()[2::3].all()


def test_rows_equal_on_the_key_are_ordered_by_content():
    table = pd.DataFrame({
        "segmentation_source": ["Manual"] * 3, "structure": ["GTV1"] * 3,
        "perturbation_id": ["p"] * 3, "feature_name": ["f"] * 3,
        "value": [3.0, 1.0, 2.0],
    })
    first = rr._order_robustness_rows(table)
    second = rr._order_robustness_rows(table.iloc[::-1])
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert sorted(first.value) == [1.0, 2.0, 3.0]


def test_a_table_without_key_columns_is_returned_unchanged():
    table = pd.DataFrame({"value": [2.0, 1.0]}, index=[5, 3])
    pd.testing.assert_frame_equal(rr._order_robustness_rows(table),
                                  table.reset_index(drop=True))
