"""Explicit geometric non-measurements for the robustness Cartesian grid.

Only measured geometry can establish impossibility. Exceptions, timeouts,
resource rejection and malformed worker results are never non-measurements.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from typing import Any, Mapping

import numpy as np
import SimpleITK as sitk


GEOMETRIC_REASON_CODES = frozenset({
    "translation_outside_image", "volume_below_generation_minimum",
    "volume_above_image_capacity", "volume_change_not_representable",
    "empty_perturbed_mask", "resampled_mask_below_minimum_voxels",
    "resampled_mask_below_minimum_dimensions",
})


@dataclass(frozen=True)
class GeometricNonmeasurement:
    reason_code: str
    evidence: Mapping[str, Any]

    def __post_init__(self):
        if self.reason_code not in GEOMETRIC_REASON_CODES or not self.evidence:
            raise ValueError("unrecognized or unjustified geometric non-measurement")
        e = self.evidence
        try:
            if self.reason_code == "translation_outside_image":
                valid = 0 <= e["translated_voxels"] < e["original_voxels"]
            elif self.reason_code == "volume_below_generation_minimum":
                valid = min(e["original_voxels"], e["target_voxels"]) < e["minimum_voxels"]
            elif self.reason_code == "volume_above_image_capacity":
                valid = e["target_voxels"] > e["image_voxels"]
            elif self.reason_code == "volume_change_not_representable":
                valid = e["target_voxels"] == e["original_voxels"] and e["tau"] != 0
            elif self.reason_code == "empty_perturbed_mask":
                valid = e.get("resampled_voxels", e.get("native_voxels")) == 0
            elif self.reason_code == "resampled_mask_below_minimum_voxels":
                valid = (e["resampled_voxels"] <= 1 or (
                    e["minimumROISize"] is not None
                    and e["resampled_voxels"] <= e["minimumROISize"]))
            else:
                valid = e["observed_dimensions"] < e["minimumROIDimensions"]
        except (KeyError, TypeError):
            valid = False
        if not valid:
            raise ValueError("geometric evidence does not establish the stated reason")


class GeometricNotExtractable(RuntimeError):
    def __init__(self, outcome: GeometricNonmeasurement):
        self.outcome = outcome
        super().__init__(f"{outcome.reason_code}: {dict(outcome.evidence)}")


def volume_nonmeasurement(mask, tau):
    """Explain a failed exact-count adaptation, never an arbitrary failure."""
    n = int(np.count_nonzero(sitk.GetArrayViewFromImage(mask)))
    minimum = max(10, int(np.ceil(10.0 / np.prod(mask.GetSpacing()))))
    target = int(round(n * (1.0 + tau)))
    evidence = dict(original_voxels=n, target_voxels=target,
                    minimum_voxels=minimum, image_voxels=int(np.prod(mask.GetSize())), tau=tau)
    if n < minimum or target < minimum:
        return GeometricNonmeasurement("volume_below_generation_minimum", evidence)
    if target > evidence["image_voxels"]:
        return GeometricNonmeasurement("volume_above_image_capacity", evidence)
    if target == n:
        return GeometricNonmeasurement("volume_change_not_representable", evidence)
    raise RuntimeError(f"unexplained volume adaptation failure: {evidence}")


def extraction_nonmeasurement(image, mask, factory):
    """Check the exact extraction grid against its configured geometric limits.

    Keep the original preprocessing settings. Failure to load, align, or resample
    propagates as a technical error. Do not infer geometry from exception strings.
    """
    extractor = factory()
    settings = dict(extractor.settings)
    settings.pop("resegmentRange", None)
    native_mask = sitk.ReadImage(str(mask)) if not isinstance(mask, sitk.Image) else mask
    native_count = int(np.count_nonzero(sitk.GetArrayViewFromImage(native_mask)))
    if native_count == 0:
        return GeometricNonmeasurement("empty_perturbed_mask", {"native_voxels": 0})
    _, loaded_mask = extractor.loadImage(image, mask, None, **settings)
    binary = sitk.GetArrayFromImage(loaded_mask) == int(settings.get("label", 1))
    count = int(binary.sum())
    coords = np.where(binary)
    dimensions = sum(int(axis.max()) > int(axis.min()) for axis in coords) if count else 0
    minimum = settings.get("minimumROISize")
    min_dimensions = int(settings.get("minimumROIDimensions", 2))
    evidence = dict(native_voxels=native_count, resampled_voxels=count,
                    minimumROISize=minimum, observed_dimensions=dimensions,
                    minimumROIDimensions=min_dimensions,
                    resampled_spacing=list(loaded_mask.GetSpacing()))
    if count == 0:
        return GeometricNonmeasurement("empty_perturbed_mask", evidence)
    if count == 1 or (minimum is not None and count <= int(minimum)):
        return GeometricNonmeasurement("resampled_mask_below_minimum_voxels", evidence)
    if dimensions < min_dimensions:
        return GeometricNonmeasurement("resampled_mask_below_minimum_dimensions", evidence)
    return None


class GeometricAdmissionContractError(RuntimeError):
    """A contradictory returned admission result is a technical failure."""


def returned_geometry_nonmeasurement(records, image, mask, factory):
    """Adapt explicit CT grid-admission rows without relying on exceptions.

    The CT extractor may return non-measurement rows instead of throwing. Only
    paired geometric admission failures qualify, and their geometry is measured
    again under the unchanged extractor settings. Resource/technical errors are
    never converted by this adapter.
    """
    from .radiomics_ct_contract import CT_EXTRACTION_ARMS
    geometric = [r for r in records if r.get("extraction_failure_kind") == "resampled_degenerate_mask"]
    if not geometric:
        return None
    if len(records) != 2 or len(geometric) != 2 or {
        r.get("extraction_arm") for r in records
    } != set(CT_EXTRACTION_ARMS) or any(
        r.get("extraction_status") not in {"below_minimum_voxels", "below_minimum_dimensions"}
        for r in records
    ):
        raise GeometricAdmissionContractError("inconsistent paired robustness geometric admission rows")
    outcome = extraction_nonmeasurement(image, mask, factory)
    if outcome is None:
        raise GeometricAdmissionContractError("returned CT geometric admission failure lacks reproducible evidence")
    return outcome


def nonmeasurement_rows(outcome, identity, perturbation_id, run_identifier, *, mask_identity=None):
    from .radiomics_ct_contract import CT_EXTRACTION_ARMS
    from .radiomics_robustness import ROBUSTNESS_MEASUREMENT_TYPE
    return [{
        **dict(identity), "structure": identity["roi_original_name"],
        "modality": "CT", "perturbation_id": perturbation_id,
        "extraction_arm": arm, "measurement_type": ROBUSTNESS_MEASUREMENT_TYPE,
        "perturbed_mask_identity": mask_identity,
        "run_identifier": run_identifier, "robustness_status": "geometrically_impossible",
        "reason_code": outcome.reason_code,
        "geometry_evidence": json.dumps(dict(outcome.evidence), sort_keys=True),
        "feature_name": None, "value": np.nan,
    } for arm in CT_EXTRACTION_ARMS]
