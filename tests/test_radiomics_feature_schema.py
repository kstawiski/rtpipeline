import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import rtpipeline.radiomics_conda as radiomics_conda
from rtpipeline.radiomics_ct_contract import _scalarize
from rtpipeline.radiomics_schema import (
    RadiomicsFeatureTypeError,
    assert_radiomics_arrow_schema,
    coerce_radiomic_feature_value,
    normalize_radiomics_result,
    write_radiomics_feature_table_atomic,
)


@pytest.mark.parametrize(
    ("modality", "workbook_name"),
    [("CT", "radiomics_ct.xlsx"), ("MR", "radiomics_mr.xlsx")],
)
def test_ct_and_mr_feature_tables_have_numeric_arrow_features(
    tmp_path, modality, workbook_name
):
    frame = pd.DataFrame(
        [
            {
                "modality": modality,
                "patient_id": "P001",
                "roi_name": "bladder",
                "diagnostics_Versions_PyRadiomics": "3.0.1",
                "diagnostics_Configuration_EnabledImageTypes": {"Original": {}},
                "original_shape_MeshVolume": np.array(12.5),
                "original_shape_Maximum2DDiameterColumn": np.array([8.25]),
                "original_firstorder_Mean": "4.5",
            }
        ]
    )

    workbook = tmp_path / workbook_name
    parquet = write_radiomics_feature_table_atomic(frame, workbook)
    assert workbook.exists()
    assert parquet.exists()

    schema = pq.read_schema(parquet)
    for name in (
        "original_shape_MeshVolume",
        "original_shape_Maximum2DDiameterColumn",
        "original_firstorder_Mean",
    ):
        assert pa.types.is_floating(schema.field(name).type)
    string_columns = {
        field.name
        for field in schema
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
    }
    assert string_columns == {
        "modality",
        "patient_id",
        "roi_name",
        "diagnostics_Versions_PyRadiomics",
        "diagnostics_Configuration_EnabledImageTypes",
    }
    written = pd.read_parquet(parquet)
    assert written.loc[0, "original_shape_MeshVolume"] == 12.5
    assert written.loc[0, "original_shape_Maximum2DDiameterColumn"] == 8.25
    assert written.loc[0, "original_firstorder_Mean"] == 4.5
    assert written.loc[0, "diagnostics_Versions_PyRadiomics"] == "3.0.1"


def test_pyradiomics_zero_dimensional_arrays_are_numeric_not_text():
    result = normalize_radiomics_result(
        {
            "original_shape_MeshVolume": np.array(10.125),
            "original_shape_Sphericity": np.float64(0.75),
            "diagnostics_Image-original_Hash": "abc123",
            "diagnostics_Configuration_Settings": {"binWidth": 25},
        }
    )

    assert result["original_shape_MeshVolume"] == 10.125
    assert isinstance(result["original_shape_MeshVolume"], float)
    assert result["original_shape_Sphericity"] == 0.75
    assert isinstance(result["original_shape_Sphericity"], float)
    assert result["diagnostics_Image-original_Hash"] == "abc123"
    assert result["diagnostics_Configuration_Settings"] == '{"binWidth": 25}'


def test_native_and_parallel_ct_scalarizer_uses_numeric_contract():
    result = _scalarize(
        {
            "original_shape_Maximum3DDiameter": np.array(7.25),
            "diagnostics_Versions_PyRadiomics": "3.0.1",
        }
    )

    assert result["original_shape_Maximum3DDiameter"] == 7.25
    assert isinstance(result["original_shape_Maximum3DDiameter"], float)
    assert result["diagnostics_Versions_PyRadiomics"] == "3.0.1"


def test_conda_parent_boundary_coerces_numeric_text_and_writes_numeric_arrow(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        radiomics_conda, "check_radiomics_env", lambda *args, **kwargs: True
    )
    monkeypatch.setenv("RTPIPELINE_RADIOMICS_BATCH", "0")
    monkeypatch.setattr(
        radiomics_conda,
        "extract_radiomics_with_conda",
        lambda *args, **kwargs: {
            "original_shape_Sphericity": "0.875",
            "diagnostics_Versions_PyRadiomics": "3.0.1",
        },
    )
    workbook = tmp_path / "radiomics_mr.xlsx"
    tasks = [
        {
            "image_path": "image.nrrd",
            "mask_path": "mask.nrrd",
            "roi_name": "bladder",
            "cleanup": False,
            "metadata": {
                "modality": "MR",
                "patient_id": "P001",
                "series_uid": "1.2.3",
                "segmentation_source": "Manual",
            },
        }
    ]

    result = radiomics_conda.process_radiomics_batch(
        tasks,
        workbook,
        sequential=True,
        enable_heartbeat=False,
    )

    assert result == workbook
    schema = pq.read_schema(workbook.with_suffix(".parquet"))
    assert pa.types.is_floating(schema.field("original_shape_Sphericity").type)
    assert pd.read_parquet(workbook.with_suffix(".parquet")).loc[
        0, "original_shape_Sphericity"
    ] == 0.875


def test_conda_non_numeric_feature_fails_course_without_publication(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        radiomics_conda, "check_radiomics_env", lambda *args, **kwargs: True
    )
    monkeypatch.setenv("RTPIPELINE_RADIOMICS_BATCH", "0")
    monkeypatch.setattr(
        radiomics_conda,
        "extract_radiomics_with_conda",
        lambda *args, **kwargs: {
            "original_shape_Sphericity": "not-a-number",
        },
    )
    workbook = tmp_path / "radiomics_mr.xlsx"
    workbook.write_text("stale", encoding="utf-8")
    workbook.with_suffix(".parquet").write_text("stale", encoding="utf-8")
    tasks = [
        {
            "image_path": "image.nrrd",
            "mask_path": "mask.nrrd",
            "roi_name": "bladder",
            "cleanup": False,
            "required": True,
            "metadata": {
                "modality": "MR",
                "segmentation_source": "Manual",
            },
        }
    ]

    with pytest.raises(RadiomicsFeatureTypeError, match="non-numeric text"):
        radiomics_conda.process_radiomics_batch(
            tasks,
            workbook,
            sequential=True,
            enable_heartbeat=False,
        )
    assert not workbook.exists()
    assert not workbook.with_suffix(".parquet").exists()


@pytest.mark.parametrize("value", ["not-a-number", "", True, np.array([1.0, 2.0])])
def test_invalid_feature_values_fail_closed(value):
    with pytest.raises(RadiomicsFeatureTypeError):
        coerce_radiomic_feature_value("original_shape_MeshVolume", value)


def test_invalid_feature_table_is_not_published(tmp_path):
    workbook = tmp_path / "radiomics_mr.xlsx"
    frame = pd.DataFrame(
        [
            {
                "modality": "MR",
                "patient_id": "P001",
                "roi_name": "bladder",
                "original_shape_MeshVolume": "not-a-number",
            }
        ]
    )

    with pytest.raises(RadiomicsFeatureTypeError, match="non-numeric text"):
        write_radiomics_feature_table_atomic(frame, workbook)
    assert not workbook.exists()
    assert not workbook.with_suffix(".parquet").exists()


def test_arrow_schema_assertion_rejects_string_feature_column(tmp_path):
    parquet = tmp_path / "bad.parquet"
    pd.DataFrame(
        {
            "modality": ["MR"],
            "original_shape_MeshVolume": ["12.5"],
        }
    ).to_parquet(parquet, index=False)

    with pytest.raises(RadiomicsFeatureTypeError, match="non-numeric feature"):
        assert_radiomics_arrow_schema(
            parquet, expected_string_columns={"modality"}
        )


def test_writer_rejects_undeclared_string_metadata(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "modality": "CT",
                "roi_name": "bladder",
                "unexpected_comment": "silently-added text",
                "original_firstorder_Mean": 1.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="Unexpected string-valued radiomics column"):
        write_radiomics_feature_table_atomic(frame, tmp_path / "radiomics_ct.xlsx")
