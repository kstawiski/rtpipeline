"""Acquisition-scale provenance for CT radiomics rows.

A cohort can mix CT reconstructions whose intensity scales differ by an order of
magnitude: a standard scan spans roughly [-1000, 3071] HU while a Siemens
extended-scale reconstruction (typically produced with iMAR for metal artefact
reduction) reaches beyond 8000 HU. Under fixed bin-size discretisation those
scans discretise into very different numbers of grey levels, so first-order and
texture features are not directly comparable between them.

The feature table previously carried PyRadiomics diagnostics but no descriptor of
the acquisition scale, so that confounder was invisible to downstream analysis.
These fields make it explicit and auditable. They are descriptive only: nothing
here changes an extracted feature value.

The scale class is derived from the effective HU mapping, never from
RescaleIntercept alone, because slope, pixel representation and bit depth all
contribute (an intercept of -10240 or +31768 can still yield a normal range).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

STANDARD_HU_MAX = 3071.0
STANDARD_HU_MIN = -1024.0
EXTENDED_HU_MAX = 4000.0


def classify_scale(hu_min: Optional[float], hu_max: Optional[float]) -> str:
    """Return 'standard', 'intermediate', 'extended' or 'unknown'."""
    if hu_min is None or hu_max is None:
        return "unknown"
    if hu_max > EXTENDED_HU_MAX:
        return "extended"
    if hu_max > STANDARD_HU_MAX or hu_min < STANDARD_HU_MIN - 100:
        return "intermediate"
    return "standard"


def describe_planning_ct(ct_dir: Path) -> Dict[str, Any]:
    """Summarise the acquisition scale of a planning-CT series.

    Every field fails soft: an unreadable or absent header yields ``None`` and a
    scale class of ``unknown`` rather than raising, because this descriptor must
    never be able to fail an extraction.
    """
    out: Dict[str, Any] = {
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
        "acq_effective_hu_min": None,
        "acq_effective_hu_max": None,
        "acq_scale_class": "unknown",
        "acq_imar_present": None,
        "acq_contrast_agent": None,
    }
    try:
        import numpy as np
        import pydicom
    except Exception:  # pragma: no cover - import environment issue
        return out
    try:
        files = sorted(Path(ct_dir).rglob("*.dcm"))
        if not files:
            return out
        mid = files[len(files) // 2]
        ds = pydicom.dcmread(str(mid), force=True)
        out["acq_manufacturer"] = str(getattr(ds, "Manufacturer", "") or "") or None
        out["acq_model"] = str(getattr(ds, "ManufacturerModelName", "") or "") or None
        desc = str(getattr(ds, "SeriesDescription", "") or "") or None
        out["acq_series_description"] = desc
        kern = getattr(ds, "ConvolutionKernel", None)
        out["acq_kernel"] = str(kern) if kern not in (None, "") else None
        for tag, key in (("KVP", "acq_kvp"), ("SliceThickness", "acq_slice_thickness")):
            v = getattr(ds, tag, None)
            out[key] = float(v) if v not in (None, "") else None
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        icpt = float(getattr(ds, "RescaleIntercept", 0) or 0)
        out["acq_rescale_slope"] = slope
        out["acq_rescale_intercept"] = icpt
        out["acq_bits_stored"] = int(getattr(ds, "BitsStored", 0) or 0) or None
        out["acq_pixel_representation"] = int(getattr(ds, "PixelRepresentation", 0) or 0)
        arr = ds.pixel_array.astype("float64") * slope + icpt
        out["acq_effective_hu_min"] = float(np.min(arr))
        out["acq_effective_hu_max"] = float(np.max(arr))
        out["acq_scale_class"] = classify_scale(
            out["acq_effective_hu_min"], out["acq_effective_hu_max"]
        )
        blob = " ".join(filter(None, [desc, out["acq_kernel"]])).lower()
        out["acq_imar_present"] = ("imar" in blob) or ("mar" == blob.strip()) or None
        agent = getattr(ds, "ContrastBolusAgent", None)
        out["acq_contrast_agent"] = str(agent) if agent not in (None, "") else None
    except Exception as exc:
        logger.debug("Could not describe planning-CT acquisition scale for %s: %s", ct_dir, exc)
    return out
