from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


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


def run_medsam_boxprompt_prediction(
    *,
    input_path: Path,
    output_dir: Path,
    checkpoint_path: Path,
    source_dir: Path | None,
    prompt_mask_path: Path,
    output_name: str = "lung_tumor",
    model_type: str = "vit_b",
    device: str = "cuda",
    bbox_margin_voxels: int = 5,
) -> None:
    """Run MedSAM ViT-B on CT slices using deterministic bboxes from a prompt mask."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if source_dir is not None:
        sys.path.insert(0, str(source_dir))

    import torch
    from segment_anything import SamPredictor, sam_model_registry

    device_name = "cuda" if str(device).lower() in {"gpu", "cuda"} else "cpu"
    ct_img = sitk.ReadImage(str(input_path))
    prompt_img = _resample_to_reference(sitk.ReadImage(str(prompt_mask_path)), ct_img)
    ct_arr = sitk.GetArrayFromImage(ct_img)
    prompt_arr = sitk.GetArrayFromImage(prompt_img) > 0

    if not np.any(prompt_arr):
        raise RuntimeError(f"Prompt mask is empty: {prompt_mask_path}")

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
    sitk.WriteImage(out_img, str(output_dir / f"{output_name}.nii.gz"))
