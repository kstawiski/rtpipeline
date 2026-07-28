from __future__ import annotations

import csv
import datetime
import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import SimpleITK as sitk


VISCERAL_FAT_PROXY_NOTE = "TotalSegmentator torso_fat is used as a VAT proxy, not a direct visceral-fat label."

# CT Hounsfield-unit windows for L3 body composition (sliceOmatic / Alberta
# protocol; Mourtzakis 2008, Martin 2013 JCO, Aubrey 2014). Areas and muscle
# radiodensity are computed only over voxels of the TotalSegmentator label that
# also fall inside these ranges, so partial-volume / mis-labelled voxels (e.g.
# air or fat inside the muscle label) are excluded and the metrics are
# comparable to the published threshold-based literature.
MUSCLE_HU_RANGE: tuple[float, float] = (-29.0, 150.0)
VISCERAL_FAT_HU_RANGE: tuple[float, float] = (-150.0, -50.0)
SUBCUTANEOUS_FAT_HU_RANGE: tuple[float, float] = (-190.0, -30.0)


def _find_mask(segmentation_dir: Path, model: str, name: str) -> Path | None:
    patterns = (
        f"{model}--{name}.nii.gz",
        f"{model}--{name}.nii",
        f"*--{model}--{name}.nii.gz",
        f"*--{model}--{name}.nii",
    )
    for pattern in patterns:
        matches = sorted(segmentation_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _read_mask(path: Path, reference: sitk.Image) -> np.ndarray:
    image = sitk.ReadImage(str(path))
    if image.GetSize() != reference.GetSize() or not all(
        np.allclose(getter(image), getter(reference), atol=1e-3)
        for getter in (sitk.Image.GetSpacing, sitk.Image.GetOrigin, sitk.Image.GetDirection)
    ):
        raise ValueError(f"Mask physical geometry does not match CT for {path}")
    arr = sitk.GetArrayFromImage(image)
    return arr > 0


def _first_dicom_patient_size_m(dicom_dir: Path) -> tuple[float | None, str | None]:
    for dcm_path in sorted(dicom_dir.glob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        raw = getattr(ds, "PatientSize", None)
        if raw in (None, ""):
            return None, "DICOM PatientSize absent"
        try:
            height = float(raw)
        except (TypeError, ValueError):
            return None, "DICOM PatientSize invalid"
        if height <= 0:
            return None, "DICOM PatientSize invalid"
        if height > 3.0:
            height = height / 100.0
        return height, None
    return None, "DICOM PatientSize absent"


def _middle_occupied_slice(mask: np.ndarray) -> tuple[int | None, int]:
    occupied = np.flatnonzero(mask.any(axis=(1, 2)))
    if occupied.size == 0:
        return None, 0
    return int(occupied[len(occupied) // 2]), int(occupied.size)


def _windowed_slice(
    ct_array: np.ndarray, mask: np.ndarray, z_index: int, hu_range: tuple[float, float]
) -> np.ndarray:
    """Boolean slice = label voxels that also fall inside the HU window."""
    lo, hi = hu_range
    ct_slice = ct_array[z_index]
    return mask[z_index] & (ct_slice >= lo) & (ct_slice <= hi)


def _area_cm2(selected_slice: np.ndarray, pixel_area_cm2: float) -> float:
    return float(np.count_nonzero(selected_slice) * pixel_area_cm2)


def _mean_hu(ct_array: np.ndarray, selected_slice: np.ndarray, z_index: int) -> float | None:
    values = ct_array[z_index][selected_slice]
    if values.size == 0:
        return None
    return float(np.mean(values))


def compute_body_composition(
    *,
    ct_nifti: Path,
    segmentation_dir: Path,
    dicom_dir: Path,
    patient_id: str,
    series_uid: str,
    image_class: str,
) -> dict[str, Any]:
    ct_image = sitk.ReadImage(str(ct_nifti))
    ct_array = sitk.GetArrayFromImage(ct_image)
    spacing = tuple(float(x) for x in ct_image.GetSpacing())
    if len(spacing) < 2:
        raise ValueError(f"CT spacing has fewer than 2 dimensions: {spacing}")
    pixel_area_cm2 = (spacing[0] * spacing[1]) / 100.0

    l3_path = _find_mask(Path(segmentation_dir), "total", "vertebrae_L3")
    base: dict[str, Any] = {
        "patient_id": str(patient_id),
        "series_uid": str(series_uid),
        "image_class": str(image_class),
        "source_nifti": str(ct_nifti),
        "segmentation_dir": str(segmentation_dir),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "l3_selection": {
            "method": "middle occupied axial slice of TotalSegmentator total--vertebrae_L3 mask",
            "mask": str(l3_path) if l3_path else None,
            "slice_index": None,
            "occupied_slices": 0,
        },
        "pixel_spacing_mm": [spacing[0], spacing[1]],
        "hu_windows": {
            "skeletal_muscle_hu": list(MUSCLE_HU_RANGE),
            "visceral_fat_hu": list(VISCERAL_FAT_HU_RANGE),
            "subcutaneous_fat_hu": list(SUBCUTANEOUS_FAT_HU_RANGE),
            "note": "Areas and muscle radiodensity use TotalSegmentator label voxels intersected with these HU windows.",
        },
        "metrics": {
            "skeletal_muscle_area_cm2": None,
            "skeletal_muscle_radiodensity_hu": None,
            "visceral_fat_area_cm2": None,
            "visceral_fat_proxy": "torso_fat",
            "visceral_fat_proxy_note": VISCERAL_FAT_PROXY_NOTE,
            "subcutaneous_fat_area_cm2": None,
            "smi_cm2_m2": None,
            "smi_missing_reason": None,
        },
        "status": "ok",
        "missing_masks": [],
    }
    height_m, height_missing_reason = _first_dicom_patient_size_m(Path(dicom_dir))

    if l3_path is None:
        base["status"] = "l3_mask_missing"
        base["missing_masks"].append("total--vertebrae_L3")
        base["metrics"]["smi_missing_reason"] = height_missing_reason or "L3 vertebrae mask missing"
        return base

    l3_mask = _read_mask(l3_path, ct_image)
    z_index, occupied_slices = _middle_occupied_slice(l3_mask)
    base["l3_selection"]["occupied_slices"] = occupied_slices
    if z_index is None:
        base["status"] = "l3_mask_empty"
        base["metrics"]["smi_missing_reason"] = height_missing_reason or "L3 vertebrae mask empty"
        return base
    base["l3_selection"]["slice_index"] = z_index

    tissue_masks = {
        "skeletal_muscle": _find_mask(Path(segmentation_dir), "tissue_types", "skeletal_muscle"),
        "torso_fat": _find_mask(Path(segmentation_dir), "tissue_types", "torso_fat"),
        "subcutaneous_fat": _find_mask(Path(segmentation_dir), "tissue_types", "subcutaneous_fat"),
    }
    missing = [f"tissue_types--{name}" for name, path in tissue_masks.items() if path is None]
    if missing:
        base["status"] = "tissue_mask_missing"
        base["missing_masks"].extend(missing)

    if tissue_masks["skeletal_muscle"] is not None:
        muscle = _read_mask(tissue_masks["skeletal_muscle"], ct_image)
        muscle_sel = _windowed_slice(ct_array, muscle, z_index, MUSCLE_HU_RANGE)
        muscle_area = _area_cm2(muscle_sel, pixel_area_cm2)
        base["metrics"]["skeletal_muscle_area_cm2"] = muscle_area
        base["metrics"]["skeletal_muscle_radiodensity_hu"] = _mean_hu(ct_array, muscle_sel, z_index)
        if height_m is not None:
            base["metrics"]["smi_cm2_m2"] = muscle_area / (height_m * height_m)

    if tissue_masks["torso_fat"] is not None:
        torso_fat = _read_mask(tissue_masks["torso_fat"], ct_image)
        torso_sel = _windowed_slice(ct_array, torso_fat, z_index, VISCERAL_FAT_HU_RANGE)
        base["metrics"]["visceral_fat_area_cm2"] = _area_cm2(torso_sel, pixel_area_cm2)

    if tissue_masks["subcutaneous_fat"] is not None:
        subq = _read_mask(tissue_masks["subcutaneous_fat"], ct_image)
        subq_sel = _windowed_slice(ct_array, subq, z_index, SUBCUTANEOUS_FAT_HU_RANGE)
        base["metrics"]["subcutaneous_fat_area_cm2"] = _area_cm2(subq_sel, pixel_area_cm2)

    if height_missing_reason is not None:
        base["metrics"]["smi_missing_reason"] = height_missing_reason
    elif base["metrics"]["skeletal_muscle_area_cm2"] is None:
        base["metrics"]["smi_missing_reason"] = "skeletal muscle mask missing"

    return base


def write_series_body_composition(
    *,
    ct_nifti: Path,
    segmentation_dir: Path,
    dicom_dir: Path,
    patient_id: str,
    series_uid: str,
    image_class: str,
) -> Path:
    result = compute_body_composition(
        ct_nifti=Path(ct_nifti),
        segmentation_dir=Path(segmentation_dir),
        dicom_dir=Path(dicom_dir),
        patient_id=patient_id,
        series_uid=series_uid,
        image_class=image_class,
    )
    out_path = Path(segmentation_dir) / "body_composition.json"
    tmp_path = out_path.parent / f".body_composition.{os.getpid()}.{uuid.uuid4().hex}.json.tmp"
    try:
        tmp_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(out_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return out_path


def _csv_row_from_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    l3 = payload.get("l3_selection", {}) if isinstance(payload.get("l3_selection"), dict) else {}
    windows = payload.get("hu_windows", {}) if isinstance(payload.get("hu_windows"), dict) else {}
    return {
        "patient_id": payload.get("patient_id"),
        "series_uid": payload.get("series_uid"),
        "image_class": payload.get("image_class"),
        "status": payload.get("status"),
        "l3_slice_index": l3.get("slice_index"),
        "l3_selection_method": l3.get("method"),
        "skeletal_muscle_area_cm2": metrics.get("skeletal_muscle_area_cm2"),
        "skeletal_muscle_radiodensity_hu": metrics.get("skeletal_muscle_radiodensity_hu"),
        "visceral_fat_area_cm2": metrics.get("visceral_fat_area_cm2"),
        "visceral_fat_proxy": metrics.get("visceral_fat_proxy"),
        "visceral_fat_proxy_note": metrics.get("visceral_fat_proxy_note"),
        "subcutaneous_fat_area_cm2": metrics.get("subcutaneous_fat_area_cm2"),
        "smi_cm2_m2": metrics.get("smi_cm2_m2"),
        "smi_missing_reason": metrics.get("smi_missing_reason"),
        "skeletal_muscle_hu_window": windows.get("skeletal_muscle_hu"),
        "visceral_fat_hu_window": windows.get("visceral_fat_hu"),
        "subcutaneous_fat_hu_window": windows.get("subcutaneous_fat_hu"),
        "json_path": str(path),
    }


def write_body_composition_csv(output_root: Path) -> Path | None:
    output_root = Path(output_root)
    rows: list[dict[str, Any]] = []
    for json_path in sorted(output_root.rglob("body_composition.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(_csv_row_from_json(json_path, payload))
    if not rows:
        return None

    data_dir = output_root / "Data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "body_composition.csv"
    fieldnames = list(rows[0].keys())
    rows.sort(key=lambda row: (str(row.get("patient_id") or ""), str(row.get("series_uid") or "")))
    # Unique temp name per writer + process so that if this aggregator is ever
    # invoked concurrently, each writer lands a *complete* file via the atomic
    # replace (no shared-tmp interleaving / FileNotFoundError on rename).
    tmp_path = data_dir / f".body_composition.{os.getpid()}.{uuid.uuid4().hex}.csv.tmp"
    try:
        with tmp_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        tmp_path.replace(out_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return out_path
