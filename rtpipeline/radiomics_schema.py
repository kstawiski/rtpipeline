from __future__ import annotations

import json
import numbers
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


RADIOMIC_FEATURE_MARKERS = (
    "_firstorder_",
    "_shape_",
    "_shape2D_",
    "_glcm_",
    "_glrlm_",
    "_glszm_",
    "_gldm_",
    "_ngtdm_",
)

# String-valued publication metadata are explicit. Any other string column is a
# schema error unless it is a PyRadiomics diagnostics column.
RADIOMICS_TEXT_COLUMNS = frozenset(
    {
        "modality",
        "patient_id",
        "course_id",
        "course_key",
        "series_uid",
        "study_uid",
        "image_class",
        "image_modality",
        "contrast_phase",
        "body_regions",
        "segmentation_source",
        "roi_name",
        "roi_original_name",
        "course_dir",
        "series_dir",
        "nifti_path",
        "mask_path_source",
        "mask_identity",
        "rtstruct_sop_instance_uid",
        "stable_roi_identifier",
        "extraction_arm",
        "roi_class",
        "roi_map_version",
        "roi_map_hash",
        "roi_map_entry_source",
        "roi_class_adjudication_status",
        "effective_parameter_hash",
        "configured_parameter_hash",
        "code_revision",
        "run_identifier",
        "pyradiomics_version",
        "simpleitk_version",
        "numpy_version",
        "shape_disposition",
        "intensity_texture_disposition",
        "extraction_status",
        "extraction_status_detail",
        "extraction_failure_kind",
        "radiomics_course_status",
        "radiomics_roi_counts_by_source",
        "acq_provenance_status",
        "acq_provenance_detail",
        "acq_series_instance_uid",
        "acq_manufacturer",
        "acq_model",
        "acq_series_description",
        "acq_kernel",
        "acq_scale_class",
        "acq_contrast_agent",
        "radiomics_cohort_provenance_schema",
        "radiomics_denominator_source_sha256",
        "radiomics_cohort_exclusions_json",
        # Fields the radiomics resource guard writes on the rows it bounds.
        # These are provenance for an ROI that was deliberately not extracted:
        # the disposition code, and the measured/estimated extents that justify
        # it. Omitting roi_structural_code failed 13 live courses; omitting the
        # two shape fields then failed 3 more. The regression derives this set
        # from the guard's own payload so a newly added field cannot slip
        # through the allowlist the same way a third time.
        "roi_structural_code",
        "native_mask_bbox_shape",
        "estimated_resampled_bbox_shape",
    }
)


class RadiomicsFeatureTypeError(ValueError):
    """A feature-named value cannot be represented as one numeric scalar."""


def is_radiomic_feature_column(name: Any) -> bool:
    text = str(name)
    return not text.startswith("diagnostics_") and any(
        marker in text for marker in RADIOMIC_FEATURE_MARKERS
    )


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (dict, list, set, tuple)):
        return False
    if isinstance(value, np.ndarray) and value.ndim > 0:
        return False
    try:
        import pandas as pd

        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)):
            return bool(missing)
    except (TypeError, ValueError):
        pass
    return False


def coerce_radiomic_feature_value(name: Any, value: Any) -> float | None:
    """Return one numeric feature scalar without changing its numeric value.

    PyRadiomics 3.0.1 returns most features as zero-dimensional ndarrays, while
    a small shape subset is returned as NumPy floating scalars. JSON transports
    can additionally return numeric text. All are accepted. Non-scalar arrays,
    booleans, blank text, and non-numeric text fail closed.
    """

    feature_name = str(name)
    if not is_radiomic_feature_column(feature_name):
        raise RadiomicsFeatureTypeError(
            f"{feature_name!r} is not a recognized radiomic feature column"
        )
    if _is_missing_scalar(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        raise RadiomicsFeatureTypeError(
            f"Radiomic feature {feature_name!r} is boolean, not numeric"
        )
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise RadiomicsFeatureTypeError(
                f"Radiomic feature {feature_name!r} is not scalar: "
                f"shape={value.shape!r}, dtype={value.dtype!s}"
            )
        return coerce_radiomic_feature_value(feature_name, value.reshape(()).item())
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise RadiomicsFeatureTypeError(
                f"Radiomic feature {feature_name!r} is not scalar: "
                f"{type(value).__name__} length={len(value)}"
            )
        return coerce_radiomic_feature_value(feature_name, value[0])
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            raise RadiomicsFeatureTypeError(
                f"Radiomic feature {feature_name!r} is blank text"
            )
        try:
            return float(candidate)
        except ValueError as exc:
            raise RadiomicsFeatureTypeError(
                f"Radiomic feature {feature_name!r} is non-numeric text: {value!r}"
            ) from exc
    if isinstance(value, numbers.Real):
        return float(value)
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RadiomicsFeatureTypeError(
            f"Radiomic feature {feature_name!r} cannot be coerced from "
            f"{type(value).__name__}: {value!r}"
        ) from exc
    return converted


def _normalize_diagnostic_value(value: Any) -> Any:
    if _is_missing_scalar(value):
        return None
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _normalize_diagnostic_value(value.item())
        return json.dumps(value.tolist(), default=str, sort_keys=True)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return json.dumps(sorted(value, key=str), default=str, sort_keys=True)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str, sort_keys=True)
    if isinstance(value, (str, bool, numbers.Real)):
        return value
    return str(value)


def normalize_radiomics_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize extractor output by column role, not by value appearance."""

    normalized: dict[str, Any] = {}
    for key, value in result.items():
        name = str(key)
        if is_radiomic_feature_column(name):
            normalized[name] = coerce_radiomic_feature_value(name, value)
        elif name.startswith("diagnostics_"):
            normalized[name] = _normalize_diagnostic_value(value)
        else:
            normalized[name] = value
    return normalized


def _normalize_text_value(value: Any) -> str | None:
    if _is_missing_scalar(value):
        return None
    return str(value)


def normalize_radiomics_dataframe(dataframe: Any) -> Any:
    """Normalize all feature, diagnostics, and text metadata columns."""

    output = dataframe.copy()
    for column in output.columns:
        name = str(column)
        if is_radiomic_feature_column(name):
            output[column] = output[column].map(
                lambda value, feature=name: coerce_radiomic_feature_value(feature, value)
            ).astype("float64")
        elif name.startswith("diagnostics_"):
            output[column] = output[column].map(_normalize_diagnostic_value)
        elif name in RADIOMICS_TEXT_COLUMNS:
            output[column] = output[column].map(_normalize_text_value)
    return output


def expected_radiomics_string_columns(dataframe: Any) -> set[str]:
    """Return the exact permitted Arrow string fields for this table."""

    expected: set[str] = set()
    for column in dataframe.columns:
        name = str(column)
        values = [value for value in dataframe[column].tolist() if not _is_missing_scalar(value)]
        string_values = [value for value in values if isinstance(value, str)]
        if not string_values:
            continue
        if is_radiomic_feature_column(name):
            raise RadiomicsFeatureTypeError(
                f"Radiomic feature column {name!r} still contains string values"
            )
        if name.startswith("diagnostics_") or name in RADIOMICS_TEXT_COLUMNS:
            expected.add(name)
            continue
        raise ValueError(
            f"Unexpected string-valued radiomics column {name!r}; "
            "only declared identifier/provenance fields and diagnostics may be strings"
        )
    return expected


def assert_radiomics_arrow_schema(
    parquet_path: Path,
    *,
    expected_string_columns: set[str] | None = None,
) -> None:
    """Assert numeric feature fields and the exact Arrow string-field contract."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_path = Path(parquet_path)
    schema = pq.read_schema(parquet_path)
    bad_features = {
        field.name: str(field.type)
        for field in schema
        if is_radiomic_feature_column(field.name)
        and not (
            pa.types.is_integer(field.type)
            or pa.types.is_floating(field.type)
            or pa.types.is_decimal(field.type)
        )
    }
    if bad_features:
        detail = ", ".join(
            f"{name}={arrow_type}" for name, arrow_type in sorted(bad_features.items())
        )
        raise RadiomicsFeatureTypeError(
            f"Radiomics Parquet has non-numeric feature column(s): {detail}"
        )

    actual_strings = {
        field.name
        for field in schema
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
    }
    if expected_string_columns is None:
        table = pq.read_table(parquet_path)
        expected_string_columns = expected_radiomics_string_columns(table.to_pandas())
    if actual_strings != expected_string_columns:
        unexpected = sorted(actual_strings - expected_string_columns)
        missing = sorted(expected_string_columns - actual_strings)
        raise ValueError(
            "Radiomics Parquet string-column contract failed: "
            f"unexpected={unexpected}, missing={missing}"
        )


def write_radiomics_feature_table_atomic(dataframe: Any, workbook_path: Path) -> Path:
    """Write Excel and authoritative Parquet only after Arrow schema validation."""

    import pandas as pd

    workbook_path = Path(workbook_path)
    parquet_path = workbook_path.with_suffix(".parquet")
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    publish_df = normalize_radiomics_dataframe(dataframe)
    expected_strings = expected_radiomics_string_columns(publish_df)

    fd, parquet_tmp_name = tempfile.mkstemp(
        prefix=f".{parquet_path.name}.", suffix=".tmp.parquet", dir=parquet_path.parent
    )
    os.close(fd)
    parquet_tmp = Path(parquet_tmp_name)
    fd, workbook_tmp_name = tempfile.mkstemp(
        prefix=f".{workbook_path.name}.", suffix=".tmp.xlsx", dir=workbook_path.parent
    )
    os.close(fd)
    workbook_tmp = Path(workbook_tmp_name)
    try:
        publish_df.to_parquet(parquet_tmp, index=False, engine="pyarrow")
        assert_radiomics_arrow_schema(
            parquet_tmp, expected_string_columns=expected_strings
        )
        publish_df.to_excel(workbook_tmp, index=False)
        pd.read_excel(workbook_tmp, engine="openpyxl")
        os.replace(parquet_tmp, parquet_path)
        os.replace(workbook_tmp, workbook_path)
    except Exception:
        # Destinations are untouched until both temporary files validate. Preserve a
        # prior published pair if creation or validation of the replacement fails.
        raise
    finally:
        parquet_tmp.unlink(missing_ok=True)
        workbook_tmp.unlink(missing_ok=True)
    return parquet_path
