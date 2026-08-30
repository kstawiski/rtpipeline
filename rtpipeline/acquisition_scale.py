"""Acquisition-scale provenance for CT radiomics rows.

A cohort can mix CT reconstructions whose intensity scales differ by an order of
magnitude. Fixed bin-size discretisation then produces different grey-level
counts even when extraction settings are identical. The descriptor defined here
makes that acquisition property visible without changing any image, mask,
resampling, binning, feature class, or extracted feature value.

The descriptor reads every DICOM instance in the contracted planning-CT series.
It does not sample. ``acq_observed_hu_min`` and ``acq_observed_hu_max`` are the
minimum and maximum decoded values across all instances after applying each
instance's rescale slope and intercept. ``acq_representable_hu_min`` and
``acq_representable_hu_max`` are the extrema implied by each instance's slope,
intercept, BitsStored, and PixelRepresentation. The legacy
``acq_effective_hu_min`` and ``acq_effective_hu_max`` fields remain aliases of
the observed full-series range.

Descriptor construction is fail-soft. It always returns the complete schema and
uses ``acq_provenance_status`` plus ``acq_scale_class='unknown'`` when the
contracted series cannot be characterised. Publication is fail-closed. A CT row
cannot be written unless a complete descriptor object is supplied.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

STANDARD_HU_MAX = 3071.0
STANDARD_HU_MIN = -1024.0
EXTENDED_HU_MAX = 4000.0

_SCALE_CLASSES = {"standard", "intermediate", "extended", "unknown"}
_IMAR_PATTERN = re.compile(r"(?<![A-Za-z0-9])i?mar(?![A-Za-z0-9])", re.IGNORECASE)

ACQUISITION_DESCRIPTOR_FIELDS = (
    "acq_provenance_status",
    "acq_provenance_detail",
    "acq_series_instance_uid",
    "acq_dicom_instance_count",
    "acq_manufacturer",
    "acq_model",
    "acq_series_description",
    "acq_kernel",
    "acq_kvp",
    "acq_slice_thickness",
    "acq_rescale_slope",
    "acq_rescale_intercept",
    "acq_bits_stored",
    "acq_pixel_representation",
    "acq_representable_hu_min",
    "acq_representable_hu_max",
    "acq_observed_hu_min",
    "acq_observed_hu_max",
    "acq_effective_hu_min",
    "acq_effective_hu_max",
    "acq_scale_class",
    "acq_imar_present",
    "acq_contrast_present",
    "acq_contrast_agent",
)


class AcquisitionDescriptorError(RuntimeError):
    """A CT publication omitted or malformed its acquisition descriptor."""


def classify_scale(hu_min: Optional[float], hu_max: Optional[float]) -> str:
    """Return 'standard', 'intermediate', 'extended' or 'unknown'."""
    if hu_min is None or hu_max is None:
        return "unknown"
    if hu_max > EXTENDED_HU_MAX:
        return "extended"
    if hu_max > STANDARD_HU_MAX or hu_min < STANDARD_HU_MIN - 100:
        return "intermediate"
    return "standard"


def _empty_descriptor(
    status: str,
    *,
    series_instance_uid: Optional[str] = None,
    detail: Optional[str] = None,
    instance_count: int = 0,
) -> Dict[str, Any]:
    return {
        "acq_provenance_status": status,
        "acq_provenance_detail": detail,
        "acq_series_instance_uid": series_instance_uid or None,
        "acq_dicom_instance_count": int(instance_count),
        "acq_manufacturer": None,
        "acq_model": None,
        "acq_series_description": None,
        "acq_kernel": None,
        "acq_kvp": None,
        "acq_slice_thickness": None,
        "acq_rescale_slope": None,
        "acq_rescale_intercept": None,
        "acq_bits_stored": None,
        "acq_pixel_representation": None,
        "acq_representable_hu_min": None,
        "acq_representable_hu_max": None,
        "acq_observed_hu_min": None,
        "acq_observed_hu_max": None,
        "acq_effective_hu_min": None,
        "acq_effective_hu_max": None,
        "acq_scale_class": "unknown",
        "acq_imar_present": None,
        "acq_contrast_present": None,
        "acq_contrast_agent": None,
    }


def _text(dataset: Any, name: str) -> Optional[str]:
    value = getattr(dataset, name, None)
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def _number(dataset: Any, name: str, default: Optional[float] = None) -> Optional[float]:
    value = getattr(dataset, name, None)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(dataset: Any, name: str) -> Optional[int]:
    value = getattr(dataset, name, None)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _uniform(values: Sequence[Any]) -> Any:
    """Return one series-wide value only when every instance agrees."""
    if not values or any(value is None for value in values):
        return None
    first = values[0]
    return first if all(value == first for value in values[1:]) else None


def _unreadable(
    series_instance_uid: Optional[str],
    detail: str,
    *,
    instance_count: int = 0,
) -> Dict[str, Any]:
    logger.debug("Could not describe planning-CT acquisition scale. %s", detail)
    return _empty_descriptor(
        "series_unreadable",
        series_instance_uid=series_instance_uid,
        detail=detail,
        instance_count=instance_count,
    )


def describe_planning_ct(
    ct_dir: Path,
    *,
    series_instance_uid: Optional[str] = None,
) -> Dict[str, Any]:
    """Describe one planning-CT series without sampling or raising.

    Every readable DICOM instance directly in ``ct_dir`` must be a CT object from
    ``series_instance_uid`` when that UID is supplied. Extensionless DICOM
    instances are included. Every matching instance's pixel data are decoded.
    Any unreadable identified instance or identity mismatch returns an unknown
    scale with an explicit status rather than a partial range.

    The observed range is calculated across all decoded pixels after per-instance
    slope and intercept application. The representable range is calculated from
    the stored-value endpoints implied by BitsStored and PixelRepresentation for
    every instance. A missing mapping field produces
    ``mapping_metadata_missing`` and null representable bounds, while retaining
    a valid full observed range when pixel decoding succeeded.
    """
    expected_uid = str(series_instance_uid or "").strip() or None
    try:
        import numpy as np
        import pydicom
    except Exception as exc:  # pragma: no cover - import environment issue
        return _empty_descriptor(
            "dependency_unavailable",
            series_instance_uid=expected_uid,
            detail=str(exc),
        )

    try:
        root = Path(ct_dir)
        candidates = sorted(path for path in root.iterdir() if path.is_file())
        if not candidates:
            return _unreadable(expected_uid, f"no files found in {root}")

        instances: list[tuple[Path, Any]] = []
        actual_uids: set[str] = set()
        for path in candidates:
            try:
                dataset = pydicom.dcmread(str(path), force=True)
            except Exception as exc:
                if path.suffix.lower() != ".dcm":
                    continue
                return _unreadable(
                    expected_uid,
                    f"DICOM instance is unreadable at {path}. {exc}",
                    instance_count=len(instances),
                )
            modality = str(getattr(dataset, "Modality", "") or "").strip().upper()
            actual_uid = str(
                getattr(dataset, "SeriesInstanceUID", "") or ""
            ).strip()
            if not modality and not actual_uid:
                if path.suffix.lower() != ".dcm":
                    continue
                return _unreadable(
                    expected_uid,
                    f"DICOM instance has no CT identity at {path}",
                    instance_count=len(instances),
                )
            if modality != "CT" or not actual_uid:
                return _empty_descriptor(
                    "series_mismatch",
                    series_instance_uid=expected_uid,
                    detail=(
                        f"DICOM instance {path} is not an identified CT object. "
                        f"Modality={modality!r}, SeriesInstanceUID={actual_uid!r}"
                    ),
                    instance_count=len(instances),
                )
            if expected_uid is not None and actual_uid != expected_uid:
                return _empty_descriptor(
                    "series_mismatch",
                    series_instance_uid=expected_uid,
                    detail=(
                        f"DICOM instance {path} has SeriesInstanceUID {actual_uid!r}, "
                        f"not contracted UID {expected_uid!r}"
                    ),
                    instance_count=len(instances),
                )
            actual_uids.add(actual_uid)
            instances.append((path, dataset))

        if not instances:
            return _unreadable(expected_uid, f"no readable CT instances found in {root}")

        files = [path for path, _dataset in instances]
        datasets = [dataset for _path, dataset in instances]

        if len(actual_uids) != 1:
            return _empty_descriptor(
                "series_mismatch",
                series_instance_uid=expected_uid,
                detail=f"planning CT contains multiple series UIDs {sorted(actual_uids)!r}",
                instance_count=len(datasets),
            )
        actual_uid = next(iter(actual_uids))
        if expected_uid is None:
            expected_uid = actual_uid

        slopes: list[Optional[float]] = []
        intercepts: list[Optional[float]] = []
        bits_values: list[Optional[int]] = []
        representation_values: list[Optional[int]] = []
        observed_mins: list[float] = []
        observed_maxs: list[float] = []
        representable_mins: list[float] = []
        representable_maxs: list[float] = []
        rescale_complete = True
        storage_mapping_complete = True

        for path, dataset in zip(files, datasets):
            slope = _number(dataset, "RescaleSlope")
            intercept = _number(dataset, "RescaleIntercept")
            valid_rescale = (
                slope is not None
                and intercept is not None
                and bool(np.isfinite(slope))
                and bool(np.isfinite(intercept))
                and slope != 0
            )
            slopes.append(slope)
            intercepts.append(intercept)
            try:
                pixels = np.asarray(dataset.pixel_array, dtype="float64")
                if pixels.size == 0:
                    raise ValueError("pixel array is empty")
            except Exception as exc:
                return _unreadable(
                    expected_uid,
                    f"pixel data are unreadable in {path}. {exc}",
                    instance_count=len(datasets),
                )
            if valid_rescale:
                assert slope is not None and intercept is not None
                mapped = pixels * slope + intercept
                observed_mins.append(float(np.min(mapped)))
                observed_maxs.append(float(np.max(mapped)))
            else:
                rescale_complete = False

            bits = _integer(dataset, "BitsStored")
            representation = _integer(dataset, "PixelRepresentation")
            bits_values.append(bits)
            representation_values.append(representation)
            if bits is None or bits < 1 or bits > 64 or representation not in (0, 1):
                storage_mapping_complete = False
                continue
            if not valid_rescale:
                continue
            assert slope is not None and intercept is not None
            if representation == 0:
                stored_min, stored_max = 0, (1 << bits) - 1
            else:
                stored_min = -(1 << (bits - 1))
                stored_max = (1 << (bits - 1)) - 1
            endpoint_a = stored_min * slope + intercept
            endpoint_b = stored_max * slope + intercept
            representable_mins.append(float(min(endpoint_a, endpoint_b)))
            representable_maxs.append(float(max(endpoint_a, endpoint_b)))

        mapping_complete = rescale_complete and storage_mapping_complete
        observed_min = min(observed_mins) if rescale_complete else None
        observed_max = max(observed_maxs) if rescale_complete else None
        status = "ok" if mapping_complete else "mapping_metadata_missing"
        detail = None
        if not mapping_complete:
            missing_groups = []
            if not rescale_complete:
                missing_groups.append("RescaleSlope or RescaleIntercept")
            if not storage_mapping_complete:
                missing_groups.append("BitsStored or PixelRepresentation")
            detail = (
                " or ".join(missing_groups)
                + " is missing or invalid for at least one contracted DICOM instance"
            )

        descriptions = [_text(dataset, "SeriesDescription") for dataset in datasets]
        kernels = [_text(dataset, "ConvolutionKernel") for dataset in datasets]
        imar_text = [value for value in descriptions + kernels if value is not None]
        imar_present: Optional[bool]
        if any(_IMAR_PATTERN.search(value) is not None for value in imar_text):
            imar_present = True
        elif all(
            description is not None or kernel is not None
            for description, kernel in zip(descriptions, kernels)
        ):
            imar_present = False
        else:
            imar_present = None

        contrast_tags_present = ["ContrastBolusAgent" in dataset for dataset in datasets]
        contrast_values = [_text(dataset, "ContrastBolusAgent") for dataset in datasets]
        nonempty_agents = list(
            dict.fromkeys(value for value in contrast_values if value is not None)
        )
        if nonempty_agents:
            contrast_present: Optional[bool] = True
            contrast_agent: Optional[str] = "; ".join(nonempty_agents)
        elif all(contrast_tags_present):
            contrast_present = False
            contrast_agent = None
        else:
            contrast_present = None
            contrast_agent = None

        descriptor = _empty_descriptor(
            status,
            series_instance_uid=expected_uid,
            detail=detail,
            instance_count=len(datasets),
        )
        descriptor.update(
            {
                "acq_manufacturer": _uniform(
                    [_text(dataset, "Manufacturer") for dataset in datasets]
                ),
                "acq_model": _uniform(
                    [_text(dataset, "ManufacturerModelName") for dataset in datasets]
                ),
                "acq_series_description": _uniform(descriptions),
                "acq_kernel": _uniform(kernels),
                "acq_kvp": _uniform(
                    [_number(dataset, "KVP") for dataset in datasets]
                ),
                "acq_slice_thickness": _uniform(
                    [_number(dataset, "SliceThickness") for dataset in datasets]
                ),
                "acq_rescale_slope": _uniform(slopes),
                "acq_rescale_intercept": _uniform(intercepts),
                "acq_bits_stored": _uniform(bits_values),
                "acq_pixel_representation": _uniform(representation_values),
                "acq_representable_hu_min": (
                    min(representable_mins) if mapping_complete else None
                ),
                "acq_representable_hu_max": (
                    max(representable_maxs) if mapping_complete else None
                ),
                "acq_observed_hu_min": observed_min,
                "acq_observed_hu_max": observed_max,
                "acq_effective_hu_min": observed_min,
                "acq_effective_hu_max": observed_max,
                "acq_scale_class": classify_scale(observed_min, observed_max),
                "acq_imar_present": imar_present,
                "acq_contrast_present": contrast_present,
                "acq_contrast_agent": contrast_agent,
            }
        )
        return descriptor
    except Exception as exc:
        return _unreadable(expected_uid, f"unexpected descriptor failure for {ct_dir}. {exc}")


def describe_contract_planning_ct(contract: Any) -> Dict[str, Any]:
    """Describe exactly the planning CT named by a loaded course contract.

    The helper never falls back to ``course/DICOM/CT``. Both
    ``planning_ct.dicom_dir`` and ``planning_ct.series_instance_uid`` must be
    declared. Contract access failures are represented in the returned status.
    """
    if contract is None:
        return _empty_descriptor(
            "contract_missing",
            detail="no course contract was supplied",
        )
    try:
        planning_ct = contract.planning_ct
        if not isinstance(planning_ct, Mapping):
            return _empty_descriptor(
                "contract_invalid",
                detail="planning_ct is not an object",
            )
        series_uid = str(planning_ct.get("series_instance_uid") or "").strip()
        dicom_dir_value = str(planning_ct.get("dicom_dir") or "").strip()
        if not series_uid or not dicom_dir_value:
            return _empty_descriptor(
                "contract_missing",
                series_instance_uid=series_uid or None,
                detail=(
                    "planning_ct.series_instance_uid and planning_ct.dicom_dir are "
                    "both required"
                ),
            )
        ct_dir = contract.planning_ct_dir
        if ct_dir is None:
            return _empty_descriptor(
                "contract_missing",
                series_instance_uid=series_uid,
                detail="planning_ct.dicom_dir did not resolve to a directory",
            )
        return describe_planning_ct(
            Path(ct_dir),
            series_instance_uid=series_uid,
        )
    except Exception as exc:
        return _empty_descriptor(
            "contract_invalid",
            detail=str(exc),
        )


def _validated_descriptor(descriptor: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(descriptor, Mapping):
        raise AcquisitionDescriptorError(
            "a complete acquisition descriptor is required for CT publication"
        )
    missing = [field for field in ACQUISITION_DESCRIPTOR_FIELDS if field not in descriptor]
    if missing:
        raise AcquisitionDescriptorError(
            "acquisition descriptor is missing required field(s). " + ", ".join(missing)
        )
    status = str(descriptor.get("acq_provenance_status") or "").strip()
    if not status:
        raise AcquisitionDescriptorError(
            "acquisition descriptor requires a nonempty acq_provenance_status"
        )
    scale_class = str(descriptor.get("acq_scale_class") or "").strip()
    if scale_class not in _SCALE_CLASSES:
        raise AcquisitionDescriptorError(
            f"acquisition descriptor has invalid acq_scale_class {scale_class!r}"
        )
    return {field: descriptor[field] for field in ACQUISITION_DESCRIPTOR_FIELDS}


def attach_acquisition_descriptor(
    rows: Sequence[MutableMapping[str, Any]],
    descriptor: Optional[Mapping[str, Any]],
) -> None:
    """Attach one validated descriptor to every CT row.

    Rows without a modality are CT by historical convention. MR and PET rows are
    left unchanged. Supplying no descriptor for at least one CT row is a hard
    publication error. A complete descriptor whose status is not ``ok`` remains
    valid because it explicitly exposes unknown or partial provenance.
    """
    ct_rows = [
        row
        for row in rows
        if str(row.get("modality", "CT") or "CT").strip().upper() == "CT"
    ]
    if not ct_rows:
        return
    validated = _validated_descriptor(descriptor)
    for row in ct_rows:
        row.update(validated)


def validate_acquisition_descriptor_table(
    table: Any,
    *,
    expected_descriptor: Optional[Mapping[str, Any]] = None,
    expected_series_instance_uid: Optional[str] = None,
    expected_series_uid_column: Optional[str] = None,
) -> None:
    """Reject persisted CT rows with absent, mixed, or stale provenance."""
    columns = set(getattr(table, "columns", []))
    missing = [field for field in ACQUISITION_DESCRIPTOR_FIELDS if field not in columns]
    if missing:
        raise AcquisitionDescriptorError(
            "persisted CT radiomics table is missing acquisition descriptor field(s). "
            + ", ".join(missing)
        )
    if expected_series_uid_column and expected_series_uid_column not in columns:
        raise AcquisitionDescriptorError(
            "persisted CT radiomics table is missing expected series UID column "
            f"{expected_series_uid_column!r}"
        )
    if expected_descriptor is not None and expected_series_uid_column is not None:
        raise AcquisitionDescriptorError(
            "descriptor validation cannot use a single expected descriptor and a "
            "per-row series UID column together"
        )

    try:
        records = list(table.to_dict(orient="records"))
    except Exception as exc:
        raise AcquisitionDescriptorError(
            f"persisted CT radiomics table cannot be inspected. {exc}"
        ) from exc
    if not records:
        raise AcquisitionDescriptorError("persisted CT radiomics table has no rows")

    def _normalise(value: Any) -> Any:
        try:
            if value is None or bool(value != value):
                return None
        except (TypeError, ValueError):
            if value is None:
                return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip()
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value).strip()

    validated_expected = (
        _validated_descriptor(expected_descriptor)
        if expected_descriptor is not None
        else None
    )
    expected_uid = str(expected_series_instance_uid or "").strip()
    descriptors_by_uid: Dict[str, set[tuple[Any, ...]]] = {}
    for row_number, record in enumerate(records, 1):
        status = _normalise(record.get("acq_provenance_status"))
        if not isinstance(status, str) or not status:
            raise AcquisitionDescriptorError(
                "persisted CT radiomics table has a missing "
                f"acq_provenance_status at row {row_number}"
            )
        scale_class = _normalise(record.get("acq_scale_class"))
        if scale_class not in _SCALE_CLASSES:
            raise AcquisitionDescriptorError(
                "persisted CT radiomics table has invalid acq_scale_class "
                f"{scale_class!r} at row {row_number}"
            )
        series_uid = _normalise(record.get("acq_series_instance_uid"))
        if not isinstance(series_uid, str) or not series_uid:
            raise AcquisitionDescriptorError(
                "persisted CT radiomics table has a missing "
                f"acq_series_instance_uid at row {row_number}"
            )
        if expected_uid and series_uid != expected_uid:
            raise AcquisitionDescriptorError(
                "persisted CT radiomics table describes SeriesInstanceUID "
                f"{series_uid!r}, not contracted UID {expected_uid!r}"
            )
        if expected_series_uid_column is not None:
            row_expected_uid = _normalise(record.get(expected_series_uid_column))
            if not isinstance(row_expected_uid, str) or not row_expected_uid:
                raise AcquisitionDescriptorError(
                    "persisted CT radiomics table has a missing "
                    f"{expected_series_uid_column} at row {row_number}"
                )
            if series_uid != row_expected_uid:
                raise AcquisitionDescriptorError(
                    "persisted CT radiomics row describes SeriesInstanceUID "
                    f"{series_uid!r}, not row UID {row_expected_uid!r}"
                )

        descriptor_values = tuple(
            _normalise(record.get(field)) for field in ACQUISITION_DESCRIPTOR_FIELDS
        )
        descriptors_by_uid.setdefault(series_uid, set()).add(descriptor_values)
        if validated_expected is not None:
            expected_values = tuple(
                _normalise(validated_expected[field])
                for field in ACQUISITION_DESCRIPTOR_FIELDS
            )
            if descriptor_values != expected_values:
                changed_fields = [
                    field
                    for field, actual, expected in zip(
                        ACQUISITION_DESCRIPTOR_FIELDS,
                        descriptor_values,
                        expected_values,
                    )
                    if actual != expected
                ]
                raise AcquisitionDescriptorError(
                    "persisted CT radiomics descriptor does not match the current "
                    "contracted series for field(s). " + ", ".join(changed_fields)
                )

    mixed_uids = sorted(
        uid for uid, descriptors in descriptors_by_uid.items() if len(descriptors) != 1
    )
    if mixed_uids:
        raise AcquisitionDescriptorError(
            "persisted CT radiomics table has mixed descriptors within "
            f"SeriesInstanceUID value(s) {mixed_uids!r}"
        )
