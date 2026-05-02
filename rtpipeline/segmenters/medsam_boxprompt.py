from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk


logger = logging.getLogger(__name__)


def _resample_to_reference(img: sitk.Image, reference: sitk.Image) -> sitk.Image:
    if (
        img.GetSize() == reference.GetSize()
        and img.GetSpacing() == reference.GetSpacing()
        and img.GetOrigin() == reference.GetOrigin()
        and img.GetDirection() == reference.GetDirection()
    ):
        return img
    return sitk.Resample(
        img,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        img.GetPixelID(),
    )


def _ct_slice_to_rgb(slice_arr: np.ndarray) -> np.ndarray:
    # Phase 2 sensitivity question: this soft-tissue window may understate
    # lung-parenchyma contrast for thoracic tumors, but Phase 1d keeps it fixed.
    windowed = np.clip(slice_arr.astype(np.float32), -160.0, 240.0)
    denom = float(windowed.max() - windowed.min())
    if denom <= 0:
        scaled = np.zeros(windowed.shape, dtype=np.uint8)
    else:
        scaled = ((windowed - windowed.min()) / denom * 255.0).astype(np.uint8)
    return np.repeat(scaled[..., None], 3, axis=-1)


def _slice_bbox(mask_slice: np.ndarray, margin: int) -> np.ndarray | None:
    coords = np.argwhere(mask_slice > 0)
    if coords.size == 0:
        return None
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    h, w = mask_slice.shape
    x0 = max(0, int(x0) - margin)
    y0 = max(0, int(y0) - margin)
    x1 = min(w - 1, int(x1) + margin)
    y1 = min(h - 1, int(y1) + margin)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def _empty_mask(shape: tuple[int, ...], reference: sitk.Image) -> sitk.Image:
    out_img = sitk.GetImageFromArray(np.zeros(shape, dtype=np.uint8))
    out_img.CopyInformation(reference)
    return out_img


def _write_mask_and_metadata(
    output_dir: Path,
    output_name: str,
    mask_img: sitk.Image,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(mask_img, str(output_dir / f"{output_name}.nii.gz"))
    (output_dir / f"{output_name}.metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def _prompt_from_anchor_bbox(
    anchor: dict[str, Any] | None,
    reference_shape: tuple[int, int, int],
) -> np.ndarray:
    prompt = np.zeros(reference_shape, dtype=bool)
    if not anchor:
        return prompt
    bbox = anchor.get("bbox_zyx")
    if not bbox or len(bbox) != 2:
        return prompt
    mins = np.asarray(bbox[0], dtype=int)
    maxs = np.asarray(bbox[1], dtype=int)
    if mins.shape != (3,) or maxs.shape != (3,):
        return prompt
    mins = np.maximum(mins, 0)
    maxs = np.minimum(maxs, np.asarray(reference_shape, dtype=int) - 1)
    if np.any(maxs < mins):
        return prompt
    z0, y0, x0 = [int(v) for v in mins]
    z1, y1, x1 = [int(v) for v in maxs]
    prompt[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1] = True
    return prompt


def run_medsam_boxprompt_prediction(
    *,
    input_path: Path,
    output_dir: Path,
    checkpoint_path: Path,
    source_dir: Path | None,
    prompt_mask_path: Path | None = None,
    prompt_source: str | None = None,
    anchor: dict[str, Any] | None = None,
    output_name: str = "lung_tumor",
    model_type: str = "vit_b",
    device: str = "cuda",
    bbox_margin_voxels: int = 5,
) -> dict[str, Any]:
    """Run MedSAM ViT-B on CT slices using deterministic 2D bboxes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ct_img = sitk.ReadImage(str(input_path))
    ct_arr = sitk.GetArrayFromImage(ct_img)
    source = (prompt_source or "").strip()
    metadata: dict[str, Any] = {
        "status": "started",
        "prompt_source": source or "mask",
        "bbox_margin_voxels": int(bbox_margin_voxels),
    }

    if source == "anchor_bbox":
        prompt_arr = _prompt_from_anchor_bbox(anchor, ct_arr.shape)
        metadata["anchor_centroid_zyx"] = (
            [float(v) for v in anchor.get("centroid_zyx", [])]
            if anchor
            else None
        )
        metadata["anchor_bbox_zyx"] = anchor.get("bbox_zyx") if anchor else None
    else:
        if prompt_mask_path is None or not Path(prompt_mask_path).exists():
            logger.warning("MedSAM skipped: prompt mask is unavailable (%s)", prompt_mask_path)
            metadata.update({"status": "skipped_no_prompt", "reason": "missing_prompt_mask"})
            return _write_mask_and_metadata(
                output_dir,
                output_name,
                _empty_mask(ct_arr.shape, ct_img),
                metadata,
            )
        prompt_img = _resample_to_reference(sitk.ReadImage(str(prompt_mask_path)), ct_img)
        prompt_arr = sitk.GetArrayFromImage(prompt_img) > 0

    if not np.any(prompt_arr):
        logger.warning("MedSAM skipped: prompt is empty for %s", input_path)
        metadata.update({"status": "skipped_no_prompt", "reason": "empty_prompt"})
        return _write_mask_and_metadata(
            output_dir,
            output_name,
            _empty_mask(ct_arr.shape, ct_img),
            metadata,
        )

    if source_dir is not None:
        sys.path.insert(0, str(source_dir))

    import torch
    from segment_anything import SamPredictor, sam_model_registry

    device_name = "cuda" if str(device).lower() in {"gpu", "cuda"} else "cpu"

    medsam = sam_model_registry[str(model_type)](checkpoint=str(checkpoint_path))
    medsam.to(device_name)
    medsam.eval()
    predictor = SamPredictor(medsam)

    out = np.zeros(prompt_arr.shape, dtype=np.uint8)
    z_indices = np.where(prompt_arr.reshape(prompt_arr.shape[0], -1).any(axis=1))[0]
    with torch.no_grad():
        for z in z_indices:
            bbox = _slice_bbox(prompt_arr[z], bbox_margin_voxels)
            if bbox is None:
                continue
            predictor.set_image(_ct_slice_to_rgb(ct_arr[z]))
            masks, _, _ = predictor.predict(box=bbox, multimask_output=False)
            predicted = masks[0].astype(bool)

            x0, y0, x1, y1 = bbox.astype(int)
            roi = np.zeros_like(predicted, dtype=bool)
            roi[y0 : y1 + 1, x0 : x1 + 1] = True
            out[z] = (predicted & roi).astype(np.uint8)

    out_img = sitk.GetImageFromArray(out)
    out_img.CopyInformation(ct_img)
    metadata.update(
        {
            "status": "completed",
            "n_prompt_slices": int(len(z_indices)),
            "output_voxels": int(out.sum()),
        }
    )
    return _write_mask_and_metadata(output_dir, output_name, out_img, metadata)
