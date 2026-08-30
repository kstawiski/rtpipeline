from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import yaml


PRIMARY_ARM = "primary_resegmented"
SENSITIVITY_ARM = "sensitivity_raw"
CT_EXTRACTION_ARMS = (PRIMARY_ARM, SENSITIVITY_ARM)
SHAPE_FEATURE_MARKERS = ("_shape_", "_shape2D_")
INTENSITY_TEXTURE_FEATURE_MARKERS = (
    "_firstorder_",
    "_glcm_",
    "_glrlm_",
    "_glszm_",
    "_gldm_",
    "_ngtdm_",
)
PUBLICATION_KEY_COLUMNS = (
    "patient_id",
    "course_id",
    "series_uid",
    "segmentation_source",
    "mask_identity",
    "roi_original_name",
    "stable_roi_identifier",
    "extraction_arm",
)
BASE_IDENTITY_COLUMNS = PUBLICATION_KEY_COLUMNS[:-1]
_COUNT_COLUMNS = (
    "morphologic_resampled_voxel_count",
    "resegment_after_count",
    "resegment_below_lower_count",
    "resegment_above_upper_count",
    "resegment_nonfinite_count",
)
_PRIMARY_WINDOWS: dict[str, tuple[float, float]] = {
    "target": (-1000.0, 400.0),
    "hollow_pelvic_organ": (-1000.0, 400.0),
    "solid_soft_tissue_neural": (-500.0, 400.0),
}
_NONAPPLICABLE_DISPOSITIONS = {
    "bone": "not_applicable_bone",
    "positioning_support": "not_primary_analysis_anatomy",
    "vessel": "not_applicable_pending_vessel_adjudication",
    "unresolved_mixed": "unclassified_roi",
}
_COMPLETE_DISPOSITIONS = {
    "success",
    "below_minimum_voxels",
    "below_minimum_dimensions",
    "not_applicable_bone",
    "not_primary_analysis_anatomy",
    "not_applicable_pending_vessel_adjudication",
    "unclassified_roi",
    "declared_skip",
    "failed",
}


@dataclass(frozen=True)
class RoiClassDecision:
    roi_class: str
    map_version: str
    map_hash: str
    map_entry_source: str
    adjudication_status: str
    primary_resegment_range_hu: Optional[tuple[float, float]]
    primary_intensity_texture_disposition: str


@lru_cache(maxsize=2)
def load_roi_class_map(path_text: Optional[str] = None) -> tuple[dict[str, Any], str]:
    if path_text:
        path = Path(path_text)
        raw = path.read_bytes()
    else:
        resource = importlib_resources.files("rtpipeline").joinpath("roi_class_map_v1.yaml")
        raw = resource.read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("CT ROI class map must be a mapping")
    if int(data.get("schema_version", 0)) != 1:
        raise ValueError("unsupported CT ROI class map schema")
    version = str(data.get("map_version") or "").strip()
    if not version:
        raise ValueError("CT ROI class map lacks map_version")
    total = data.get("totalsegmentator")
    canonical_names = total.get("canonical_names") if isinstance(total, dict) else None
    expected_canonical_hash = (
        str(total.get("canonical_names_hash") or "").strip()
        if isinstance(total, dict)
        else ""
    )
    if not isinstance(canonical_names, dict) or not expected_canonical_hash:
        raise ValueError("CT ROI class map lacks a hash-pinned TotalSegmentator vocabulary")
    canonical_names_text = json.dumps(
        canonical_names,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    observed_canonical_hash = hashlib.sha256(
        canonical_names_text.encode("utf-8")
    ).hexdigest()
    if observed_canonical_hash != expected_canonical_hash:
        raise ValueError("CT ROI class map TotalSegmentator vocabulary hash is stale")
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return data, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def roi_class_map_identity(path: Optional[Path] = None) -> tuple[str, str]:
    data, digest = load_roi_class_map(str(Path(path).resolve()) if path else None)
    return str(data["map_version"]), digest


def _strip_partial(name: str) -> str:
    text = str(name)
    return text[:-9] if text.endswith("__partial") else text


def load_custom_structure_provenance(path: Optional[Path]) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    candidate = Path(path)
    raw = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    items = raw.get("custom_structures") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        raise ValueError(f"custom structure configuration lacks custom_structures: {candidate}")
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"custom structure entry must be a mapping: {candidate}")
        name = str(item.get("name") or "").strip()
        operation = str(item.get("operation") or "").strip()
        sources = item.get("source_structures")
        if not name or operation not in {"union", "intersection", "subtract", "xor"}:
            raise ValueError(f"custom structure provenance is incomplete for {name!r}: {candidate}")
        if not isinstance(sources, list) or not sources or not all(
            isinstance(source, str) and source.strip() for source in sources
        ):
            raise ValueError(f"custom structure base structures are incomplete for {name!r}: {candidate}")
        out[name] = {
            "operation": operation,
            "source_structures": [str(source).strip() for source in sources],
            "margin": item.get("margin"),
        }
    return out


def _decision(
    roi_class: str,
    *,
    data: Mapping[str, Any],
    digest: str,
    source: str,
    status: str,
) -> RoiClassDecision:
    window = _PRIMARY_WINDOWS.get(roi_class)
    disposition = "success" if window is not None else _NONAPPLICABLE_DISPOSITIONS.get(
        roi_class, "unclassified_roi"
    )
    return RoiClassDecision(
        roi_class=roi_class,
        map_version=str(data["map_version"]),
        map_hash=digest,
        map_entry_source=source,
        adjudication_status=status,
        primary_resegment_range_hu=window,
        primary_intensity_texture_disposition=disposition,
    )


def classify_ct_roi(
    segmentation_source: str,
    roi_original_name: str,
    *,
    custom_provenance: Optional[Mapping[str, Mapping[str, Any]]] = None,
    map_path: Optional[Path] = None,
) -> RoiClassDecision:
    data, digest = load_roi_class_map(str(Path(map_path).resolve()) if map_path else None)
    source = str(segmentation_source).strip()
    name = str(roi_original_name)
    base_name = _strip_partial(name)
    source_folded = source.casefold()

    if source_folded.startswith(("autorts_total", "autots_total")):
        entries = data["totalsegmentator"]["canonical_names"]
        entry = entries.get(base_name)
        if isinstance(entry, dict):
            return _decision(
                str(entry["roi_class"]),
                data=data,
                digest=digest,
                source=(
                    f"TotalSegmentator:{data['totalsegmentator']['model']}:"
                    f"{data['totalsegmentator']['version']}"
                ),
                status="versioned_canonical_name",
            )
        return _decision(
            "unresolved_mixed",
            data=data,
            digest=digest,
            source="TotalSegmentator:unrecognized_name",
            status="operator_adjudication_required",
        )

    if source_folded == "custom":
        derived_entries = data.get("derived_crosswalk", {})
        derived_entry = derived_entries.get(base_name)
        provenance_map = custom_provenance or {}
        provenance = provenance_map.get(base_name)
        if isinstance(derived_entry, dict) and bool(derived_entry.get("require_provenance")):
            if not isinstance(provenance, Mapping):
                return _decision(
                    "unresolved_mixed",
                    data=data,
                    digest=digest,
                    source="derived_crosswalk:missing_provenance",
                    status="operator_adjudication_required",
                )
            operation = str(provenance.get("operation") or "")
            bases = provenance.get("source_structures")
            if operation not in {"union", "intersection", "subtract", "xor"} or not isinstance(
                bases, Sequence
            ) or isinstance(bases, (str, bytes)) or not bases:
                return _decision(
                    "unresolved_mixed",
                    data=data,
                    digest=digest,
                    source="derived_crosswalk:invalid_provenance",
                    status="operator_adjudication_required",
                )

            def _recorded_base_class(base: str, seen: set[str]) -> Optional[str]:
                base = _strip_partial(str(base))
                if base in seen:
                    return None
                canonical = data["totalsegmentator"]["canonical_names"].get(base)
                if isinstance(canonical, Mapping):
                    return str(canonical.get("roi_class") or "") or None
                manual = data.get("manual_custom_crosswalk", {}).get(base)
                if isinstance(manual, Mapping):
                    return str(manual.get("roi_class") or "") or None
                nested = derived_entries.get(base)
                nested_provenance = provenance_map.get(base)
                if not isinstance(nested, Mapping) or not isinstance(nested_provenance, Mapping):
                    return None
                nested_operation = str(nested_provenance.get("operation") or "")
                nested_bases = nested_provenance.get("source_structures")
                if nested_operation not in {"union", "intersection", "subtract", "xor"} or not isinstance(
                    nested_bases, Sequence
                ) or isinstance(nested_bases, (str, bytes)) or not nested_bases:
                    return None
                nested_classes = [
                    _recorded_base_class(str(item), seen | {base})
                    for item in nested_bases
                ]
                if any(value is None for value in nested_classes):
                    return None
                inherited = nested_classes[0]
                if nested_operation != "subtract" and any(
                    value != inherited for value in nested_classes[1:]
                ):
                    return None
                if inherited != str(nested.get("roi_class") or ""):
                    return None
                return inherited

            base_classes = [
                _recorded_base_class(str(base), {base_name}) for base in bases
            ]
            if any(value is None for value in base_classes):
                return _decision(
                    "unresolved_mixed",
                    data=data,
                    digest=digest,
                    source="derived_crosswalk:unclassifiable_recorded_base",
                    status="operator_adjudication_required",
                )
            inherited_class = base_classes[0]
            if operation != "subtract" and any(
                value != inherited_class for value in base_classes[1:]
            ):
                return _decision(
                    "unresolved_mixed",
                    data=data,
                    digest=digest,
                    source="derived_crosswalk:mixed_recorded_base_classes",
                    status="operator_adjudication_required",
                )
            if inherited_class != str(derived_entry.get("roi_class") or ""):
                return _decision(
                    "unresolved_mixed",
                    data=data,
                    digest=digest,
                    source="derived_crosswalk:declared_class_disagrees_with_bases",
                    status="operator_adjudication_required",
                )
            return _decision(
                str(inherited_class),
                data=data,
                digest=digest,
                source="derived_crosswalk:recorded_operation_and_classified_bases",
                status="approved_by_binding_spec",
            )

    entry = data.get("manual_custom_crosswalk", {}).get(base_name)
    if isinstance(entry, dict):
        return _decision(
            str(entry["roi_class"]),
            data=data,
            digest=digest,
            source="manual_custom_crosswalk:exact_name",
            status=str(entry.get("adjudication_status") or "approved"),
        )
    return _decision(
        "unresolved_mixed",
        data=data,
        digest=digest,
        source="manual_custom_crosswalk:unlisted_exact_name",
        status="operator_adjudication_required",
    )


def rtstruct_roi_identities(path: Path) -> dict[str, tuple[str, str]]:
    import pydicom

    dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    sop_uid = str(getattr(dataset, "SOPInstanceUID", "") or "").strip()
    if not sop_uid:
        raise ValueError(f"RTSTRUCT has no SOPInstanceUID: {path}")
    identities: dict[str, tuple[str, str]] = {}
    for item in getattr(dataset, "StructureSetROISequence", []) or []:
        name = str(getattr(item, "ROIName", "") or "").strip()
        roi_number = str(getattr(item, "ROINumber", "") or "").strip()
        if not name or not roi_number:
            raise ValueError(f"RTSTRUCT has a blank ROI name or number: {path}")
        if name in identities:
            raise ValueError(f"RTSTRUCT has duplicate ROI name {name!r}: {path}")
        identities[name] = (sop_uid, roi_number)
    if not identities:
        raise ValueError(f"RTSTRUCT contains no stable ROI identities: {path}")
    return identities


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_rtstruct_roi_identity(path: Path, roi_name: str) -> tuple[str, str]:
    """Return DICOM identities, or content/name identities for legacy inputs."""
    try:
        identities = rtstruct_roi_identities(path)
        return identities[str(roi_name)]
    except (KeyError, OSError, ValueError):
        content_identity = f"sha256:{file_sha256(path)}"
        name_digest = hashlib.sha256(str(roi_name).encode("utf-8")).hexdigest()
        return content_identity, f"roi-name-sha256:{name_digest}"


def configured_parameter_hash(
    params_file: Optional[Path],
    *,
    arm: str,
    window: Optional[tuple[float, float]],
    large_roi: bool,
) -> str:
    raw: Any
    if params_file is None:
        raw = {"source": "pyradiomics-defaults"}
    else:
        raw = yaml.safe_load(Path(params_file).read_text(encoding="utf-8"))
    payload = {
        "configured_parameters": raw,
        "arm": arm,
        "window": list(window) if window is not None else None,
        # Runtime large-ROI overrides are represented by effective_parameter_hash.
        # The configured hash identifies the immutable source configuration so it
        # can be recomputed before mask materialization and compared on resume.
        "shape_contract": "separate_unresegmented_morphological_mask_v1",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def current_code_revision() -> str:
    configured = os.environ.get("RTPIPELINE_CODE_REVISION", "").strip()
    if configured:
        return configured
    try:
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        revision = result.stdout.strip()
        return revision or "unknown"
    except Exception:
        return "unknown"


def new_run_identifier() -> str:
    return str(uuid.uuid4())


def effective_parameter_hash(extractor: Any, *, arm: str, window: Optional[tuple[float, float]]) -> str:
    payload = {
        "arm": arm,
        "window": list(window) if window is not None else None,
        "settings": dict(getattr(extractor, "settings", {}) or {}),
        "image_types": dict(getattr(extractor, "enabledImagetypes", {}) or {}),
        "features": dict(getattr(extractor, "enabledFeatures", {}) or {}),
        "shape_contract": "separate_unresegmented_morphological_mask_v1",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _disable_shape(extractor: Any) -> Any:
    enabled = dict(getattr(extractor, "enabledFeatures", {}) or {})
    enabled.pop("shape", None)
    enabled.pop("shape2D", None)
    extractor.enabledFeatures = enabled
    extractor.settings.pop("resegmentRange", None)
    extractor.settings["resegmentShape"] = False
    return extractor


def _configure_shape_only(extractor: Any) -> Any:
    extractor.settings.pop("resegmentRange", None)
    extractor.settings["resegmentShape"] = False
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("shape")
    return extractor


def build_ct_extractors(
    factory: Callable[[], Any],
    window: Optional[tuple[float, float]],
) -> tuple[Any, Any, Optional[Any]]:
    shape_extractor = _configure_shape_only(factory())
    raw_extractor = _disable_shape(factory())
    primary_extractor = None
    if window is not None:
        primary_extractor = _disable_shape(factory())
        primary_extractor.settings["resegmentRange"] = [float(window[0]), float(window[1])]
        primary_extractor.settings["resegmentMode"] = "absolute"
        primary_extractor.settings["resegmentShape"] = False
    return shape_extractor, raw_extractor, primary_extractor


def _scalarize(result: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in result.items():
        try:
            if hasattr(value, "item"):
                output[str(key)] = value.item()
            elif hasattr(value, "tolist"):
                output[str(key)] = value.tolist()
            elif isinstance(value, (str, int, float, bool)) or value is None:
                output[str(key)] = value
            else:
                output[str(key)] = str(value)
        except Exception:
            output[str(key)] = str(value)
    return output


def _feature_subset(result: Mapping[str, Any], markers: tuple[str, ...]) -> dict[str, Any]:
    return {str(key): value for key, value in result.items() if any(marker in str(key) for marker in markers)}


def _component_qc(mask: np.ndarray) -> tuple[int, int]:
    foreground = np.asarray(mask, dtype=bool)
    if not foreground.any():
        return 0, 0
    from scipy import ndimage

    labels, count = ndimage.label(foreground, structure=np.ones((3, 3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())
    largest = int(sizes[1:].max()) if sizes.size > 1 else 0
    return int(count), largest


def _observed_dimensions(mask: np.ndarray) -> int:
    coordinates = np.where(np.asarray(mask, dtype=bool))
    if not coordinates or coordinates[0].size == 0:
        return 0
    return int(sum((int(axis.max()) - int(axis.min()) + 1) > 1 for axis in coordinates))


def resampled_mask_qc(
    image: Any,
    mask: Any,
    extractor: Any,
    window: Optional[tuple[float, float]],
) -> dict[str, Any]:
    from radiomics import imageoperations
    import SimpleITK as sitk

    settings = dict(extractor.settings)
    settings.pop("resegmentRange", None)
    loaded_image, loaded_mask = extractor.loadImage(image, mask, None, **settings)
    _, corrected_mask = imageoperations.checkMask(loaded_image, loaded_mask, **settings)
    if corrected_mask is not None:
        loaded_mask = corrected_mask
    label = int(settings.get("label", 1))
    image_array = sitk.GetArrayFromImage(loaded_image)
    morphologic = sitk.GetArrayFromImage(loaded_mask) == label
    morphologic_count = int(morphologic.sum())
    values = np.asarray(image_array[morphologic])
    finite = np.isfinite(values)
    nonfinite_count = int((~finite).sum())
    if window is None:
        below_count = 0
        above_count = 0
        retained_values = finite
    else:
        lower, upper = window
        below_count = int((finite & (values < lower)).sum())
        above_count = int((finite & (values > upper)).sum())
        retained_values = finite & (values >= lower) & (values <= upper)
    after_count = int(retained_values.sum())
    after_mask = np.zeros_like(morphologic, dtype=bool)
    after_mask[morphologic] = retained_values
    before_components, before_largest = _component_qc(morphologic)
    after_components, after_largest = _component_qc(after_mask)
    if after_count + below_count + above_count + nonfinite_count != morphologic_count:
        raise AssertionError(
            "resegmentation count identity failed: "
            f"{after_count}+{below_count}+{above_count}+{nonfinite_count}!={morphologic_count}"
        )
    return {
        "morphologic_resampled_voxel_count": morphologic_count,
        "resegment_after_count": after_count,
        "resegment_below_lower_count": below_count,
        "resegment_above_upper_count": above_count,
        "resegment_nonfinite_count": nonfinite_count,
        "components_26_before": before_components,
        "components_26_after": after_components,
        "largest_component_voxel_count_after": after_largest,
        "resegment_retained_fraction": (
            float(after_count / morphologic_count) if morphologic_count else None
        ),
        "largest_component_retained_fraction": (
            float(after_largest / before_largest) if before_largest else None
        ),
        "largest_component_fraction_after": (
            float(after_largest / after_count) if after_count else None
        ),
        "component_count_increased": bool(after_components > before_components),
        "observed_roi_dimensions_after_resegmentation": _observed_dimensions(after_mask),
        "largest_component_voxel_count_before": before_largest,
    }


def _runtime_versions() -> dict[str, str]:
    import SimpleITK as sitk
    import radiomics

    return {
        "pyradiomics_version": str(getattr(radiomics, "__version__", "unknown")),
        "simpleitk_version": str(sitk.Version_VersionString()),
        "numpy_version": str(np.__version__),
    }


def _arm_metadata(
    *,
    arm: str,
    decision: RoiClassDecision,
    extractor: Any,
    run_identifier: str,
    code_revision: str,
    native_voxel_count: int,
    required: bool,
    configured_parameter_hash_value: str,
) -> dict[str, Any]:
    window = decision.primary_resegment_range_hu if arm == PRIMARY_ARM else None
    return {
        "extraction_arm": arm,
        "effective_resegment_lower_hu": window[0] if window is not None else None,
        "effective_resegment_upper_hu": window[1] if window is not None else None,
        "roi_class": decision.roi_class,
        "roi_map_version": decision.map_version,
        "roi_map_hash": decision.map_hash,
        "roi_map_entry_source": decision.map_entry_source,
        "roi_class_adjudication_status": decision.adjudication_status,
        "effective_parameter_hash": effective_parameter_hash(extractor, arm=arm, window=window),
        "configured_parameter_hash": configured_parameter_hash_value,
        "code_revision": code_revision,
        "run_identifier": run_identifier,
        "native_mask_voxel_count": int(native_voxel_count),
        "roi_required": bool(required),
        **_runtime_versions(),
    }


def extract_ct_roi_arms(
    image: Any,
    mask: Any,
    *,
    factory: Callable[[], Any],
    decision: RoiClassDecision,
    common_metadata: Mapping[str, Any],
    run_identifier: str,
    code_revision: str,
    native_voxel_count: int,
    required: bool,
    configured_parameter_hashes: Optional[Mapping[str, str]] = None,
) -> list[dict[str, Any]]:
    shape_extractor, raw_extractor, primary_extractor = build_ct_extractors(
        factory, decision.primary_resegment_range_hu
    )
    shape_result = _scalarize(shape_extractor.execute(image, mask))
    shape_features = _feature_subset(shape_result, SHAPE_FEATURE_MARKERS)
    if not shape_features:
        raise RuntimeError("separate shape-only extractor returned no shape features")

    raw_result = _scalarize(raw_extractor.execute(image, mask))
    raw_result.update(shape_features)
    raw_qc = resampled_mask_qc(image, mask, raw_extractor, None)
    sensitivity = dict(raw_result)
    sensitivity.update(common_metadata)
    sensitivity.update(
        _arm_metadata(
            arm=SENSITIVITY_ARM,
            decision=decision,
            extractor=raw_extractor,
            run_identifier=run_identifier,
            code_revision=code_revision,
            native_voxel_count=native_voxel_count,
            required=required,
            configured_parameter_hash_value=str(
                (configured_parameter_hashes or {}).get(SENSITIVITY_ARM, "unavailable")
            ),
        )
    )
    sensitivity.update(raw_qc)
    sensitivity.update(
        {
            "shape_disposition": "success",
            "intensity_texture_disposition": "success",
            "extraction_status": "success",
        }
    )

    primary_qc_extractor = primary_extractor or raw_extractor
    primary_qc = resampled_mask_qc(
        image, mask, primary_qc_extractor, decision.primary_resegment_range_hu
    )
    primary: dict[str, Any] = dict(shape_features)
    primary.update(common_metadata)
    primary.update(
        _arm_metadata(
            arm=PRIMARY_ARM,
            decision=decision,
            extractor=primary_qc_extractor,
            run_identifier=run_identifier,
            code_revision=code_revision,
            native_voxel_count=native_voxel_count,
            required=required,
            configured_parameter_hash_value=str(
                (configured_parameter_hashes or {}).get(PRIMARY_ARM, "unavailable")
            ),
        )
    )
    primary.update(primary_qc)
    primary["shape_disposition"] = "success"
    disposition = decision.primary_intensity_texture_disposition
    if primary_extractor is not None:
        minimum_size = primary_extractor.settings.get("minimumROISize")
        minimum_dimensions = int(primary_extractor.settings.get("minimumROIDimensions", 2))
        after_count = int(primary_qc["resegment_after_count"])
        after_dimensions = int(primary_qc["observed_roi_dimensions_after_resegmentation"])
        if minimum_size is not None and after_count <= int(minimum_size):
            disposition = "below_minimum_voxels"
        elif after_dimensions < minimum_dimensions:
            disposition = "below_minimum_dimensions"
        else:
            primary_result = _scalarize(primary_extractor.execute(image, mask))
            primary_result = {
                key: value
                for key, value in primary_result.items()
                if not any(marker in key for marker in SHAPE_FEATURE_MARKERS)
            }
            primary_result.update(shape_features)
            primary = {**primary_result, **primary}
            disposition = "success"
    primary["intensity_texture_disposition"] = disposition
    primary["extraction_status"] = "success"

    rows = [primary, sensitivity]
    assert_paired_shape_identity(rows)
    return rows


def disposition_rows_for_arms(
    common_metadata: Mapping[str, Any],
    *,
    decision: RoiClassDecision,
    disposition: str,
    detail: str,
    failure_kind: str,
    run_identifier: str,
    code_revision: str,
    native_voxel_count: Optional[int],
    required: bool,
    effective_hashes: Optional[Mapping[str, str]] = None,
    configured_parameter_hashes: Optional[Mapping[str, str]] = None,
) -> list[dict[str, Any]]:
    versions = {
        "pyradiomics_version": "unavailable",
        "simpleitk_version": "unavailable",
        "numpy_version": str(np.__version__),
    }
    rows: list[dict[str, Any]] = []
    for arm in CT_EXTRACTION_ARMS:
        window = decision.primary_resegment_range_hu if arm == PRIMARY_ARM else None
        rows.append(
            {
                **common_metadata,
                "extraction_arm": arm,
                "effective_resegment_lower_hu": window[0] if window else None,
                "effective_resegment_upper_hu": window[1] if window else None,
                "roi_class": decision.roi_class,
                "roi_map_version": decision.map_version,
                "roi_map_hash": decision.map_hash,
                "roi_map_entry_source": decision.map_entry_source,
                "roi_class_adjudication_status": decision.adjudication_status,
                "effective_parameter_hash": str((effective_hashes or {}).get(arm, "unavailable")),
                "configured_parameter_hash": str(
                    (configured_parameter_hashes or {}).get(arm, "unavailable")
                ),
                "code_revision": code_revision,
                "run_identifier": run_identifier,
                "native_mask_voxel_count": native_voxel_count,
                "roi_required": bool(required),
                "shape_disposition": disposition,
                "intensity_texture_disposition": disposition,
                "extraction_status": disposition,
                "extraction_status_detail": detail,
                "extraction_failure_kind": failure_kind,
                "morphologic_resampled_voxel_count": None,
                "resegment_after_count": None,
                "resegment_below_lower_count": None,
                "resegment_above_upper_count": None,
                "resegment_nonfinite_count": None,
                "components_26_before": None,
                "components_26_after": None,
                "largest_component_voxel_count_before": None,
                "largest_component_voxel_count_after": None,
                "resegment_retained_fraction": None,
                "largest_component_retained_fraction": None,
                "largest_component_fraction_after": None,
                "component_count_increased": None,
                "observed_roi_dimensions_after_resegmentation": None,
                **versions,
            }
        )
    return rows


def publication_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(column) if row.get(column) is not None else "").strip() for column in PUBLICATION_KEY_COLUMNS)


def base_identity_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return publication_key(row)[:-1]


def expected_publication_keys(base_rows: Iterable[Mapping[str, Any]]) -> set[tuple[str, ...]]:
    out: set[tuple[str, ...]] = set()
    for row in base_rows:
        base = tuple(str(row.get(column) if row.get(column) is not None else "").strip() for column in BASE_IDENTITY_COLUMNS)
        if any(not value for value in base):
            raise ValueError("expected CT identity has a blank publication-key field")
        for arm in CT_EXTRACTION_ARMS:
            out.add((*base, arm))
    return out


def _is_missing(value: Any) -> bool:
    try:
        import pandas as pd

        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)):
            return bool(missing)
    except Exception:
        pass
    return value is None


def _row_bool(value: Any) -> bool:
    if _is_missing(value):
        return False
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes"}
    return bool(value)


def assert_paired_shape_identity(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 2 or {str(row.get("extraction_arm")) for row in rows} != set(CT_EXTRACTION_ARMS):
        raise ValueError("paired shape assertion requires exactly the two governed CT arms")
    shape_columns = {
        str(column)
        for row in rows
        for column in row
        if any(marker in str(column) for marker in SHAPE_FEATURE_MARKERS)
    }
    for column in shape_columns:
        left, right = rows[0].get(column), rows[1].get(column)
        if _is_missing(left) and _is_missing(right):
            continue
        if _is_missing(left) != _is_missing(right):
            raise ValueError(f"paired CT shape feature {column} is missing in one extraction arm")
        try:
            equal = bool(np.isclose(float(left), float(right), rtol=0.0, atol=0.0, equal_nan=True))
        except Exception:
            equal = left == right
        if not equal:
            raise ValueError(f"paired CT shape feature differs across arms: {column}: {left!r} != {right!r}")


def validate_ct_publication(
    dataframe: Any,
    *,
    expected_keys: Optional[set[tuple[str, ...]]] = None,
    fail_on_unclassified_required: bool = True,
) -> set[tuple[str, ...]]:
    import pandas as pd

    if dataframe is None or dataframe.empty:
        raise ValueError("CT radiomics publication is empty")
    required_columns = set(PUBLICATION_KEY_COLUMNS) | {
        "shape_disposition",
        "intensity_texture_disposition",
        "effective_resegment_lower_hu",
        "effective_resegment_upper_hu",
        "roi_class",
        "roi_map_version",
        "roi_map_hash",
        "roi_class_adjudication_status",
        "effective_parameter_hash",
        "configured_parameter_hash",
        "code_revision",
        "pyradiomics_version",
        "simpleitk_version",
        "numpy_version",
        "run_identifier",
        "roi_required",
        *_COUNT_COLUMNS,
        "components_26_before",
        "components_26_after",
        "largest_component_retained_fraction",
        "component_count_increased",
    }
    missing_columns = sorted(required_columns - {str(column) for column in dataframe.columns})
    if missing_columns:
        raise ValueError("CT radiomics publication lacks required columns: " + ", ".join(missing_columns))

    records = dataframe.to_dict("records")
    keys = [publication_key(record) for record in records]
    if any(any(not field for field in key) for key in keys):
        raise ValueError("CT radiomics publication has a blank publication-key field")
    if len(keys) != len(set(keys)):
        raise ValueError("CT radiomics publication has duplicate full identities including extraction_arm")
    key_set = set(keys)
    if expected_keys is not None and key_set != expected_keys:
        raise ValueError(
            "CT radiomics publication identity set is incomplete or stale "
            f"(expected {len(expected_keys)}, found {len(key_set)})"
        )

    current_version, current_hash = roi_class_map_identity()
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for record in records:
        arm = str(record["extraction_arm"])
        if arm not in CT_EXTRACTION_ARMS:
            raise ValueError(f"invalid CT extraction_arm: {arm!r}")
        if str(record.get("roi_map_version")) != current_version or str(record.get("roi_map_hash")) != current_hash:
            raise ValueError("CT radiomics publication uses a stale ROI class map identity")
        for column in (
            "effective_parameter_hash",
            "configured_parameter_hash",
            "code_revision",
            "pyradiomics_version",
            "simpleitk_version",
            "numpy_version",
            "run_identifier",
        ):
            if _is_missing(record.get(column)) or not str(record.get(column)).strip():
                raise ValueError(f"CT radiomics row has blank {column}")
        if (
            fail_on_unclassified_required
            and _row_bool(record.get("roi_required"))
            and str(record.get("roi_class_adjudication_status"))
            == "operator_adjudication_required"
        ):
            raise ValueError(
                "required ROI is unclassified; operator adjudication is required before publication: "
                f"{record.get('segmentation_source')}/{record.get('roi_original_name')}"
            )
        disposition = str(record.get("intensity_texture_disposition") or "")
        if disposition not in _COMPLETE_DISPOSITIONS:
            raise ValueError(f"CT radiomics row has incomplete disposition {disposition!r}")
        if all(not _is_missing(record.get(column)) for column in _COUNT_COLUMNS):
            values = [int(record[column]) for column in _COUNT_COLUMNS]
            morphologic, after, below, above, nonfinite = values
            if after + below + above + nonfinite != morphologic:
                raise ValueError(
                    "resegmentation count identity failed in publication row: "
                    f"{after}+{below}+{above}+{nonfinite}!={morphologic}"
                )
        if arm == SENSITIVITY_ARM:
            if not _is_missing(record.get("effective_resegment_lower_hu")) or not _is_missing(
                record.get("effective_resegment_upper_hu")
            ):
                raise ValueError("sensitivity_raw row must not carry an effective resegmentation window")
        else:
            expected_window = _PRIMARY_WINDOWS.get(str(record.get("roi_class")))
            lower = record.get("effective_resegment_lower_hu")
            upper = record.get("effective_resegment_upper_hu")
            if expected_window is None:
                if not _is_missing(lower) or not _is_missing(upper):
                    raise ValueError("non-applicable primary row carries a resegmentation window")
            elif _is_missing(lower) or _is_missing(upper) or (
                float(lower), float(upper)
            ) != expected_window:
                raise ValueError("primary row carries the wrong class-governed resegmentation window")
            if disposition != "success":
                feature_columns = [
                    column
                    for column in dataframe.columns
                    if any(marker in str(column) for marker in INTENSITY_TEXTURE_FEATURE_MARKERS)
                ]
                if any(not _is_missing(record.get(column)) for column in feature_columns):
                    raise ValueError(
                        "primary disposition row contains intensity or texture feature values"
                    )
        grouped.setdefault(base_identity_key(record), []).append(record)

    for base, pair in grouped.items():
        if len(pair) != 2 or {str(row["extraction_arm"]) for row in pair} != set(CT_EXTRACTION_ARMS):
            raise ValueError(f"CT base identity does not have exactly both extraction arms: {base}")
        if len({str(row.get("run_identifier")) for row in pair}) != 1:
            raise ValueError(f"paired CT arms do not share one run_identifier: {base}")
        assert_paired_shape_identity(pair)
    return key_set


def _jsonify_nested_columns(dataframe: Any) -> Any:
    if dataframe.empty:
        return dataframe
    output = dataframe.copy()
    for column in output.columns:
        if output[column].map(lambda value: isinstance(value, (dict, list, set, tuple))).any():
            output[column] = output[column].map(
                lambda value: json.dumps(value, default=str, sort_keys=True)
                if isinstance(value, (dict, list, set, tuple))
                else value
            )
    return output


def write_ct_publication_atomic(
    dataframe: Any,
    workbook_path: Path,
    *,
    expected_keys: Optional[set[tuple[str, ...]]] = None,
) -> Path:
    import pandas as pd

    workbook_path = Path(workbook_path)
    parquet_path = workbook_path.with_suffix(".parquet")
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    publish_df = _jsonify_nested_columns(dataframe)
    validate_ct_publication(publish_df, expected_keys=expected_keys)

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
        parquet_check = pd.read_parquet(parquet_tmp, engine="pyarrow")
        validate_ct_publication(parquet_check, expected_keys=expected_keys)
        publish_df.to_excel(workbook_tmp, index=False)
        workbook_check = pd.read_excel(workbook_tmp, engine="openpyxl")
        validate_ct_publication(workbook_check, expected_keys=expected_keys)
        os.replace(parquet_tmp, parquet_path)
        os.replace(workbook_tmp, workbook_path)
    except Exception:
        parquet_path.unlink(missing_ok=True)
        workbook_path.unlink(missing_ok=True)
        raise
    finally:
        parquet_tmp.unlink(missing_ok=True)
        workbook_tmp.unlink(missing_ok=True)
    return parquet_path


def read_authoritative_ct_publication(path: Path) -> Any:
    import pandas as pd

    candidate = Path(path)
    parquet_path = candidate if candidate.suffix == ".parquet" else candidate.with_suffix(".parquet")
    dataframe = pd.read_parquet(parquet_path, engine="pyarrow")
    validate_ct_publication(dataframe)
    return dataframe


def sentinel_payload(dataframe: Any, parquet_path: Path) -> dict[str, Any]:
    keys = sorted("\x1f".join(key) for key in validate_ct_publication(dataframe))
    key_digest = hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()
    return {
        "status": "ok",
        "schema": "rtpipeline-radiomics-completion-v1",
        "authoritative_parquet": str(Path(parquet_path).name),
        "row_count": int(len(dataframe)),
        "identity_set_sha256": key_digest,
        "roi_map_version": str(dataframe["roi_map_version"].iloc[0]),
        "roi_map_hash": str(dataframe["roi_map_hash"].iloc[0]),
        "run_identifiers": sorted({str(value) for value in dataframe["run_identifier"].tolist()}),
        "effective_parameter_hashes": sorted(
            {str(value) for value in dataframe["effective_parameter_hash"].tolist()}
        ),
        "configured_parameter_hashes": sorted(
            {str(value) for value in dataframe["configured_parameter_hash"].tolist()}
        ),
        "code_revisions": sorted(
            {str(value) for value in dataframe["code_revision"].tolist()}
        ),
    }


def validate_completion_sentinel(
    course_dir: Path, sentinel_path: Optional[Path] = None
) -> dict[str, Any]:
    course_dir = Path(course_dir)
    target = Path(sentinel_path) if sentinel_path else course_dir / ".radiomics_done"
    try:
        observed = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"governed radiomics completion sentinel is unreadable: {target}") from exc
    if not isinstance(observed, dict):
        raise ValueError("governed radiomics completion sentinel must be a JSON object")
    parquet_path = course_dir / "radiomics_ct.parquet"
    dataframe = read_authoritative_ct_publication(parquet_path)
    expected = sentinel_payload(dataframe, parquet_path)
    for field in (
        "status",
        "schema",
        "authoritative_parquet",
        "row_count",
        "identity_set_sha256",
        "roi_map_version",
        "roi_map_hash",
        "effective_parameter_hashes",
        "configured_parameter_hashes",
        "code_revisions",
    ):
        if observed.get(field) != expected.get(field):
            raise ValueError(
                f"governed radiomics completion sentinel is stale for {field}"
            )
    return observed


def write_completion_sentinel(course_dir: Path, sentinel_path: Optional[Path] = None) -> Path:
    course_dir = Path(course_dir)
    parquet_path = course_dir / "radiomics_ct.parquet"
    dataframe = read_authoritative_ct_publication(parquet_path)
    payload = sentinel_payload(dataframe, parquet_path)
    target = Path(sentinel_path) if sentinel_path else course_dir / ".radiomics_done"
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_path, target)
    finally:
        tmp_path.unlink(missing_ok=True)
    return target
