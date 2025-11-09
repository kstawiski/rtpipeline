"""Systematic anatomical cropping for consistent volume analysis.

This module provides functionality to crop CT images and segmentations to
consistent anatomical boundaries defined by TotalSegmentator landmarks.

This ensures that percentage-based DVH metrics (V95%, V20Gy, etc.) have
consistent volume denominators across patients, making them meaningful for
cross-patient comparison and statistical analysis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import SimpleITK as sitk
import numpy as np
import pydicom

logger = logging.getLogger(__name__)


def extract_vertebrae_boundaries(course_dir: Path) -> Dict[str, Dict[str, float]]:
    """
    Extract superior/inferior z-coordinates for vertebrae from TotalSegmentator.

    Args:
        course_dir: Course directory containing Segmentation_TotalSegmentator

    Returns:
        Dictionary mapping vertebra names to {'superior': z, 'inferior': z, 'center': z}
        Coordinates are in mm (patient coordinate system, superior is +z)

    Raises:
        ValueError: If TotalSegmentator output directory not found
    """
    from .layout import build_course_dirs

    course_dirs = build_course_dirs(course_dir)
    seg_root = course_dirs.segmentation_totalseg

    if not seg_root.exists():
        raise ValueError(f"TotalSegmentator output not found: {seg_root}")

    # Find the actual segmentation directory (e.g., seg_root/<ct_name>/)
    candidate_dirs = sorted(p for p in seg_root.iterdir() if p.is_dir())
    if not candidate_dirs:
        raise ValueError(f"No segmentation directories found in: {seg_root}")

    seg_dir = candidate_dirs[0]  # Use first (typically only one)

    landmarks = {}

    # Find vertebrae masks
    vertebrae_pattern = ["vertebrae_L1", "vertebrae_L5", "vertebrae_T12", "vertebrae_C7"]

    for vertebra in vertebrae_pattern:
        # Look for masks with model prefix (e.g., "total--vertebrae_L1.nii.gz")
        mask_candidates = list(seg_dir.glob(f"*--{vertebra}.nii.gz"))
        if not mask_candidates:
            # Try without prefix
            mask_candidates = list(seg_dir.glob(f"{vertebra}.nii.gz"))

        if not mask_candidates:
            logger.debug(f"Vertebra mask not found: {vertebra}")
            continue

        mask_path = mask_candidates[0]

        try:
            # Read mask
            mask_img = sitk.ReadImage(str(mask_path))
            mask_array = sitk.GetArrayFromImage(mask_img)  # [z, y, x]

            if not mask_array.any():
                logger.debug(f"Vertebra mask is empty: {vertebra}")
                continue

            # Find superior and inferior extent in physical coordinates
            z_indices = np.where(mask_array.any(axis=(1, 2)))[0]

            if len(z_indices) == 0:
                continue

            # Convert indices to physical coordinates
            origin = mask_img.GetOrigin()
            spacing = mask_img.GetSpacing()

            # In DICOM/SimpleITK, superior is typically higher z value
            # But check the direction to be sure
            direction = mask_img.GetDirection()
            z_direction = direction[8]  # The [2,2] element of 3x3 direction matrix

            superior_idx = z_indices.max()
            inferior_idx = z_indices.min()

            # Convert to physical coordinates
            superior_z = origin[2] + superior_idx * spacing[2]
            inferior_z = origin[2] + inferior_idx * spacing[2]

            # If z_direction is negative, flip superior/inferior
            if z_direction < 0:
                superior_z, inferior_z = inferior_z, superior_z

            landmarks[vertebra] = {
                'superior': float(superior_z),
                'inferior': float(inferior_z),
                'center': float((superior_z + inferior_z) / 2)
            }

            logger.debug(f"Extracted {vertebra}: superior={superior_z:.1f}mm, inferior={inferior_z:.1f}mm")

        except Exception as e:
            logger.warning(f"Failed to process vertebra mask {vertebra}: {e}")
            continue

    return landmarks


def extract_femur_boundaries(course_dir: Path) -> Dict[str, Dict[str, float]]:
    """
    Extract femoral head boundaries from TotalSegmentator.

    Args:
        course_dir: Course directory

    Returns:
        Dictionary mapping 'femur_left'/'femur_right' to {'superior': z, 'inferior': z}
    """
    from .layout import build_course_dirs

    course_dirs = build_course_dirs(course_dir)
    seg_root = course_dirs.segmentation_totalseg

    if not seg_root.exists():
        raise ValueError(f"TotalSegmentator output not found: {seg_root}")

    candidate_dirs = sorted(p for p in seg_root.iterdir() if p.is_dir())
    if not candidate_dirs:
        raise ValueError(f"No segmentation directories found in: {seg_root}")

    seg_dir = candidate_dirs[0]

    landmarks = {}

    for side in ['left', 'right']:
        # Look for femur masks with model prefix
        mask_candidates = list(seg_dir.glob(f"*--femur_{side}.nii.gz"))
        if not mask_candidates:
            mask_candidates = list(seg_dir.glob(f"femur_{side}.nii.gz"))

        if not mask_candidates:
            logger.debug(f"Femur mask not found: femur_{side}")
            continue

        mask_path = mask_candidates[0]

        try:
            mask_img = sitk.ReadImage(str(mask_path))
            mask_array = sitk.GetArrayFromImage(mask_img)

            z_indices = np.where(mask_array.any(axis=(1, 2)))[0]
            if len(z_indices) == 0:
                continue

            origin = mask_img.GetOrigin()
            spacing = mask_img.GetSpacing()
            direction = mask_img.GetDirection()
            z_direction = direction[8]

            superior_idx = z_indices.max()
            inferior_idx = z_indices.min()

            superior_z = origin[2] + superior_idx * spacing[2]
            inferior_z = origin[2] + inferior_idx * spacing[2]

            if z_direction < 0:
                superior_z, inferior_z = inferior_z, superior_z

            landmarks[f'femur_{side}'] = {
                'superior': float(superior_z),
                'inferior': float(inferior_z)
            }

            logger.debug(f"Extracted femur_{side}: superior={superior_z:.1f}mm, inferior={inferior_z:.1f}mm")

        except Exception as e:
            logger.warning(f"Failed to process femur mask femur_{side}: {e}")
            continue

    return landmarks


def determine_pelvic_crop_boundaries(
    course_dir: Path,
    inferior_margin_cm: float = 10.0
) -> Tuple[float, float]:
    """
    Determine superior and inferior boundaries for pelvic cropping.

    Uses L1 vertebra superior edge as superior boundary and femoral heads
    inferior edge + margin as inferior boundary.

    Args:
        course_dir: Course directory
        inferior_margin_cm: Margin below femoral heads in cm (default: 10 cm)

    Returns:
        Tuple of (superior_z, inferior_z) in mm (patient coordinates)

    Raises:
        ValueError: If required landmarks (L1, femurs) are not found
    """
    vertebrae = extract_vertebrae_boundaries(course_dir)
    femurs = extract_femur_boundaries(course_dir)

    # Superior boundary: top of L1
    if 'vertebrae_L1' not in vertebrae:
        raise ValueError(
            "L1 vertebra not found in TotalSegmentator output - "
            "required for pelvic cropping. Available vertebrae: "
            f"{list(vertebrae.keys())}"
        )

    superior_z = vertebrae['vertebrae_L1']['superior']

    # Inferior boundary: margin below bottom of femoral heads
    femur_inferiors = [
        femurs[key]['inferior']
        for key in femurs
        if 'inferior' in femurs[key]
    ]

    if not femur_inferiors:
        raise ValueError(
            "Femoral heads not found in TotalSegmentator output - "
            "required for pelvic cropping. Available structures: "
            f"{list(femurs.keys())}"
        )

    # Use the most inferior point of both femurs
    femur_inferior = min(femur_inferiors)
    inferior_z = femur_inferior - (inferior_margin_cm * 10)  # cm to mm

    logger.info(
        f"Pelvic crop boundaries: superior={superior_z:.1f}mm (L1), "
        f"inferior={inferior_z:.1f}mm (femur-{inferior_margin_cm}cm)"
    )

    return superior_z, inferior_z


def crop_image_to_boundaries(
    image: sitk.Image,
    superior_z: float,
    inferior_z: float
) -> sitk.Image:
    """
    Crop image (CT or mask) to specified z-boundaries.

    Args:
        image: SimpleITK image to crop
        superior_z: Superior boundary in patient coords (mm)
        inferior_z: Inferior boundary in patient coords (mm)

    Returns:
        Cropped SimpleITK image with updated origin
    """
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    size = image.GetSize()
    direction = image.GetDirection()

    # Convert physical coordinates to indices
    # z_physical = origin[2] + index * spacing[2]
    # index = (z_physical - origin[2]) / spacing[2]

    superior_idx = int(round((superior_z - origin[2]) / spacing[2]))
    inferior_idx = int(round((inferior_z - origin[2]) / spacing[2]))

    # Check if requested boundaries are within CT extent
    requested_superior_idx = int(round((superior_z - origin[2]) / spacing[2]))
    requested_inferior_idx = int(round((inferior_z - origin[2]) / spacing[2]))

    # Clamp to valid range
    clamped_superior_idx = max(0, min(size[2] - 1, requested_superior_idx))
    clamped_inferior_idx = max(0, min(size[2] - 1, requested_inferior_idx))

    # Warn if clamping occurred
    if clamped_superior_idx != requested_superior_idx:
        actual_superior_z = origin[2] + clamped_superior_idx * spacing[2]
        logger.warning(
            f"Superior boundary ({superior_z:.1f}mm) exceeds CT extent. "
            f"Clamped to {actual_superior_z:.1f}mm (CT boundary)"
        )
        superior_idx = clamped_superior_idx
    else:
        superior_idx = requested_superior_idx

    if clamped_inferior_idx != requested_inferior_idx:
        actual_inferior_z = origin[2] + clamped_inferior_idx * spacing[2]
        logger.warning(
            f"Inferior boundary ({inferior_z:.1f}mm) exceeds CT extent. "
            f"Clamped to {actual_inferior_z:.1f}mm (CT boundary). "
            f"This may affect cross-patient consistency."
        )
        inferior_idx = clamped_inferior_idx
    else:
        inferior_idx = requested_inferior_idx

    # Ensure superior > inferior in index space (depends on direction)
    # If superior_z > inferior_z in physical space but direction[8] < 0,
    # then superior_idx < inferior_idx in index space
    z_direction = direction[8]
    if z_direction > 0:
        # Normal orientation: higher z = higher index
        if superior_idx < inferior_idx:
            superior_idx, inferior_idx = inferior_idx, superior_idx
    else:
        # Flipped orientation: higher z = lower index
        if inferior_idx < superior_idx:
            superior_idx, inferior_idx = inferior_idx, superior_idx

    # Extract region using ExtractImageFilter
    extract_filter = sitk.ExtractImageFilter()
    extract_size = [size[0], size[1], abs(superior_idx - inferior_idx) + 1]
    extract_index = [0, 0, min(inferior_idx, superior_idx)]

    extract_filter.SetSize(extract_size)
    extract_filter.SetIndex(extract_index)

    cropped = extract_filter.Execute(image)

    logger.debug(
        f"Cropped image: {size} -> {cropped.GetSize()} "
        f"(indices: {extract_index[2]} to {extract_index[2] + extract_size[2] - 1})"
    )

    return cropped


def apply_systematic_cropping(
    course_dir: Path,
    region: str = "pelvis",
    output_suffix: str = "_cropped",
    keep_original: bool = True,
    inferior_margin_cm: float = 10.0
) -> Dict[str, Path]:
    """
    Apply systematic anatomical cropping to CT and all segmentations.

    Creates cropped versions of CT NIfTI and all segmentation masks based on
    anatomical landmarks. Saves cropping metadata for downstream analysis.

    Args:
        course_dir: Course directory
        region: Anatomical region ("pelvis" supported currently)
        output_suffix: Suffix for cropped files (default: "_cropped")
        keep_original: Whether to keep original files (default: True)
        inferior_margin_cm: Margin below femurs for pelvic region (default: 10 cm)

    Returns:
        Dictionary of cropped file paths: {'ct': Path, 'mask_name': Path, ...}

    Raises:
        ValueError: If unsupported region or landmarks not found
    """
    from .layout import build_course_dirs

    course_dirs = build_course_dirs(course_dir)

    # Determine boundaries based on region
    if region == "pelvis":
        superior_z, inferior_z = determine_pelvic_crop_boundaries(
            course_dir,
            inferior_margin_cm=inferior_margin_cm
        )
    else:
        raise ValueError(f"Unsupported region: {region}. Currently only 'pelvis' is supported.")

    logger.info(
        f"Cropping {course_dir.name} to {region} region: "
        f"superior={superior_z:.1f}mm, inferior={inferior_z:.1f}mm"
    )

    cropped_files = {}

    # Crop CT NIfTI
    ct_nifti = course_dirs.nifti / "ct.nii.gz"
    if not ct_nifti.exists():
        # Try to find any CT NIfTI file
        ct_candidates = list(course_dirs.nifti.glob("*.nii.gz"))
        if ct_candidates:
            ct_nifti = ct_candidates[0]
        else:
            logger.warning(f"No CT NIfTI found in {course_dirs.nifti}")
            ct_nifti = None

    if ct_nifti and ct_nifti.exists():
        try:
            ct_img = sitk.ReadImage(str(ct_nifti))
            ct_cropped = crop_image_to_boundaries(ct_img, superior_z, inferior_z)

            ct_cropped_path = ct_nifti.parent / f"{ct_nifti.stem.replace('.nii', '')}{output_suffix}.nii.gz"
            sitk.WriteImage(ct_cropped, str(ct_cropped_path))
            cropped_files['ct'] = ct_cropped_path

            logger.info(f"Cropped CT: {ct_img.GetSize()} → {ct_cropped.GetSize()}")
        except Exception as e:
            logger.error(f"Failed to crop CT: {e}")

    # Crop all segmentation masks
    seg_root = course_dirs.segmentation_totalseg
    if seg_root.exists():
        candidate_dirs = sorted(p for p in seg_root.iterdir() if p.is_dir())
        if candidate_dirs:
            seg_dir = candidate_dirs[0]

            for mask_path in seg_dir.glob("*.nii.gz"):
                try:
                    mask_img = sitk.ReadImage(str(mask_path))
                    mask_cropped = crop_image_to_boundaries(mask_img, superior_z, inferior_z)

                    # Create cropped filename
                    mask_stem = mask_path.stem.replace('.nii', '')
                    mask_cropped_path = mask_path.parent / f"{mask_stem}{output_suffix}.nii.gz"
                    sitk.WriteImage(mask_cropped, str(mask_cropped_path))

                    cropped_files[mask_path.stem] = mask_cropped_path
                    logger.debug(f"Cropped mask: {mask_path.name}")

                except Exception as e:
                    logger.warning(f"Failed to crop mask {mask_path.name}: {e}")
                    continue

    # Save cropping metadata
    metadata = {
        'region': region,
        'superior_z_mm': superior_z,
        'inferior_z_mm': inferior_z,
        'inferior_margin_cm': inferior_margin_cm,
        'cropped_files': {k: str(v) for k, v in cropped_files.items()},
        'original_kept': keep_original
    }

    metadata_path = course_dir / "cropping_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved cropping metadata: {metadata_path}")
    logger.info(f"Cropped {len(cropped_files)} NIfTI files total")

    # Create cropped RTSTRUCT from cropped masks
    rs_auto_cropped = _create_rtstruct_from_cropped_masks(
        course_dir,
        course_dirs,
        output_suffix
    )

    if rs_auto_cropped:
        logger.info(f"Created cropped RTSTRUCT: {rs_auto_cropped}")
        metadata['rs_auto_cropped'] = str(rs_auto_cropped)

        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    return cropped_files


def _create_rtstruct_from_cropped_masks(
    course_dir: Path,
    course_dirs,
    output_suffix: str
) -> Optional[Path]:
    """
    Create RTSTRUCT file from cropped NIfTI masks.

    This creates RS_auto_cropped.dcm from the systematically cropped masks,
    ensuring that all structure volumes are consistently defined across patients.

    Args:
        course_dir: Course directory
        course_dirs: Course directory structure
        output_suffix: Suffix used for cropped files

    Returns:
        Path to created RTSTRUCT or None
    """
    try:
        from rt_utils import RTStructBuilder
    except ImportError:
        logger.warning("rt_utils not available; skipping cropped RTSTRUCT creation")
        return None

    # Check if CT DICOM exists
    if not course_dirs.dicom_ct.exists():
        logger.warning("CT DICOM not found; cannot create cropped RTSTRUCT")
        return None

    # Find cropped masks directory
    seg_root = course_dirs.segmentation_totalseg
    if not seg_root.exists():
        logger.warning("Segmentation directory not found")
        return None

    candidate_dirs = sorted(p for p in seg_root.iterdir() if p.is_dir())
    if not candidate_dirs:
        logger.warning("No segmentation directories found")
        return None

    seg_dir = candidate_dirs[0]

    # Find cropped masks
    cropped_masks = list(seg_dir.glob(f"*{output_suffix}.nii.gz"))
    if not cropped_masks:
        logger.warning("No cropped masks found")
        return None

    # Load CT image for geometry reference
    try:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(course_dirs.dicom_ct))
        if not series_ids:
            logger.warning("No DICOM series found in CT directory")
            return None
        files = reader.GetGDCMSeriesFileNames(str(course_dirs.dicom_ct), series_ids[0])
        reader.SetFileNames(files)
        ct_img = reader.Execute()
    except Exception as e:
        logger.error(f"Failed to load CT DICOM: {e}")
        return None

    # Create new RTSTRUCT
    try:
        rtstruct = RTStructBuilder.create_new(dicom_series_path=str(course_dirs.dicom_ct))
        rtstruct.ds.SeriesDescription = "Auto-segmented (Systematically Cropped)"
    except Exception as e:
        logger.error(f"Failed to create RTStructBuilder: {e}")
        return None

    # Add each cropped mask as an ROI
    added_count = 0
    for mask_path in cropped_masks:
        # Extract ROI name (remove suffix and prefixes)
        roi_name = mask_path.stem.replace('.nii', '').replace(output_suffix, '')

        # Remove model prefix if present (e.g., "total--" or "<basename>--total--")
        if '--' in roi_name:
            parts = roi_name.split('--')
            roi_name = parts[-1]  # Take last part after all prefixes

        try:
            # Read cropped mask
            mask_img = sitk.ReadImage(str(mask_path))
            mask_array = sitk.GetArrayFromImage(mask_img)  # [z, y, x]

            if not np.any(mask_array):
                continue  # Skip empty masks

            # Convert to [y, x, z] for rt-utils
            mask_array = np.moveaxis(mask_array, 0, -1)

            rtstruct.add_roi(mask=mask_array > 0, name=roi_name)
            added_count += 1
            logger.debug(f"Added cropped ROI: {roi_name}")

        except Exception as e:
            logger.warning(f"Failed to add ROI {roi_name}: {e}")
            continue

    if added_count == 0:
        logger.warning("No ROIs added to cropped RTSTRUCT")
        return None

    # Save cropped RTSTRUCT
    output_path = course_dir / "RS_auto_cropped.dcm"
    try:
        rtstruct.save(str(output_path))
        logger.info(f"Created RS_auto_cropped.dcm with {added_count} ROIs")
        return output_path
    except Exception as e:
        logger.error(f"Failed to save cropped RTSTRUCT: {e}")
        return None
