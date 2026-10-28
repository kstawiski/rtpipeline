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
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional

import SimpleITK as sitk
import numpy as np
from .course_contract import load_course_contract

logger = logging.getLogger(__name__)


def _ct_z_extent_mm(course_dir: Path) -> Optional[Tuple[float, float]]:
    """Return (superior_z_mm, inferior_z_mm) for the course CT NIfTI if available.

    This is used as a robust fallback when landmark masks (e.g. L1) are not present
    because the scan does not cover them. Coordinates are in patient space (mm).
    """

    contract = load_course_contract(course_dir)
    ct_nifti = contract.planning_ct_nifti
    if ct_nifti is None:
        return None

    try:
        img = sitk.ReadImage(str(ct_nifti))
        size = img.GetSize()  # (x, y, z)
        if len(size) < 3 or size[2] < 1:
            return None
        from itertools import product

        z = [img.TransformIndexToPhysicalPoint(tuple(index))[2]
             for index in product(*[(0, int(n) - 1) for n in size])]
        superior, inferior = float(max(z)), float(min(z))
        return superior, inferior
    except Exception as exc:
        logger.debug("Failed reading CT extent for %s: %s", course_dir, exc)
        return None


def _mask_z_extent(image: sitk.Image, array: np.ndarray) -> tuple[float, float]:
    """Superior/inferior voxel centers in LPS, including oblique axes."""
    indices = np.argwhere(array > 0)[:, ::-1]
    z_basis = np.asarray(image.GetDirection()).reshape(3, 3)[2] * image.GetSpacing()
    positions = indices @ z_basis + image.GetOrigin()[2]
    return float(positions.max()), float(positions.min())


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

            superior_z, inferior_z = _mask_z_extent(mask_img, mask_array)

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

            if not mask_array.any():
                continue
            superior_z, inferior_z = _mask_z_extent(mask_img, mask_array)

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

    Notes:
        Some prostate planning CTs do not extend cranially to L1. In that case we
        fall back to the **CT superior extent** (i.e., do not crop superiorly),
        while still cropping inferiorly based on femoral heads when available.
    """
    try:
        vertebrae = extract_vertebrae_boundaries(course_dir)
    except Exception as exc:
        logger.warning("Pelvic cropping: unable to read vertebrae landmarks for %s: %s", course_dir, exc)
        vertebrae = {}

    try:
        femurs = extract_femur_boundaries(course_dir)
    except Exception as exc:
        logger.warning("Pelvic cropping: unable to read femur landmarks for %s: %s", course_dir, exc)
        femurs = {}

    ct_extent = _ct_z_extent_mm(course_dir)
    fallback_notes: list[str] = []

    # Superior boundary: top of L1
    if 'vertebrae_L1' in vertebrae:
        superior_z = vertebrae['vertebrae_L1']['superior']
        superior_source = "vertebrae_L1"
    else:
        if ct_extent is not None:
            superior_z = ct_extent[0]
            superior_source = "ct_superior_extent"
            fallback_notes.append(f"missing_vertebrae_L1 (available={sorted(vertebrae.keys())}); using CT superior extent")
            logger.warning(
                "Pelvic cropping: L1 missing for %s (available=%s); using CT superior extent (%.1fmm)",
                course_dir,
                sorted(vertebrae.keys()),
                superior_z,
            )
        elif vertebrae:
            # Best-effort: use the most superior vertebra we have (still better than failing).
            best_name, best = max(vertebrae.items(), key=lambda kv: kv[1].get("superior", float("-inf")))
            superior_z = float(best.get("superior", 0.0))
            superior_source = f"{best_name}_fallback"
            fallback_notes.append(f"missing_vertebrae_L1; using {best_name} superior as fallback")
            logger.warning(
                "Pelvic cropping: L1 missing for %s; using %s superior (%.1fmm) as fallback",
                course_dir,
                best_name,
                superior_z,
            )
        else:
            # Last resort: avoid crashing; downstream clamping will keep bounds sane if CT is readable.
            superior_z = 0.0
            superior_source = "unknown_fallback"
            fallback_notes.append("missing_vertebrae_L1 and CT extent unavailable; using 0.0mm fallback")
            logger.warning("Pelvic cropping: L1 missing and CT extent unavailable for %s; using 0.0mm fallback", course_dir)

    # Inferior boundary: margin below bottom of femoral heads
    femur_inferiors = [
        femurs[key]['inferior']
        for key in femurs
        if 'inferior' in femurs[key]
    ]

    if femur_inferiors:
        # Use the most inferior point of both femurs
        femur_inferior = min(femur_inferiors)
        inferior_z = femur_inferior - (inferior_margin_cm * 10)  # cm to mm
        inferior_source = f"femur_inferior_minus_{inferior_margin_cm}cm"
    else:
        if ct_extent is not None:
            inferior_z = ct_extent[1]
            inferior_source = "ct_inferior_extent"
            fallback_notes.append(f"missing_femurs (available={sorted(femurs.keys())}); using CT inferior extent")
            logger.warning(
                "Pelvic cropping: femurs missing for %s (available=%s); using CT inferior extent (%.1fmm)",
                course_dir,
                sorted(femurs.keys()),
                inferior_z,
            )
        elif vertebrae:
            inferior_z = float(min(v.get("inferior", 0.0) for v in vertebrae.values()))
            inferior_source = "vertebra_inferior_fallback"
            fallback_notes.append("missing_femurs and CT extent unavailable; using inferior of available vertebrae")
            logger.warning(
                "Pelvic cropping: femurs missing and CT extent unavailable for %s; using vertebra inferior fallback (%.1fmm)",
                course_dir,
                inferior_z,
            )
        else:
            inferior_z = 0.0
            inferior_source = "unknown_fallback"
            fallback_notes.append("missing_femurs and CT extent unavailable; using 0.0mm fallback")
            logger.warning("Pelvic cropping: femurs missing and CT extent unavailable for %s; using 0.0mm fallback", course_dir)

    if superior_z <= inferior_z:
        if ct_extent is not None:
            logger.warning(
                "Pelvic cropping: invalid boundaries for %s (superior=%.1f <= inferior=%.1f); using CT extent fallback",
                course_dir,
                superior_z,
                inferior_z,
            )
            superior_z, inferior_z = ct_extent
            superior_source = "ct_extent_fallback"
            inferior_source = "ct_extent_fallback"
            fallback_notes.append("invalid_boundary_order; using CT extent fallback")
        else:
            logger.warning(
                "Pelvic cropping: invalid boundaries for %s (superior=%.1f <= inferior=%.1f); swapping",
                course_dir,
                superior_z,
                inferior_z,
            )
            superior_z, inferior_z = inferior_z, superior_z
            fallback_notes.append("invalid_boundary_order; swapped")

    logger.info(
        "Pelvic crop boundaries for %s: superior=%.1fmm (%s), inferior=%.1fmm (%s)",
        course_dir.name,
        superior_z,
        superior_source,
        inferior_z,
        inferior_source,
    )

    return superior_z, inferior_z


def extract_organ_boundaries(course_dir: Path, organ_patterns: list) -> Dict[str, Dict[str, float]]:
    """
    Extract superior/inferior z-coordinates for organs from TotalSegmentator.

    Args:
        course_dir: Course directory containing Segmentation_TotalSegmentator
        organ_patterns: List of organ name patterns to search for

    Returns:
        Dictionary mapping organ names to {'superior': z, 'inferior': z}
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

    for organ in organ_patterns:
        # Look for masks with model prefix (e.g., "total--lung_upper_lobe_left.nii.gz")
        mask_candidates = list(seg_dir.glob(f"*--{organ}.nii.gz"))
        if not mask_candidates:
            # Try without prefix
            mask_candidates = list(seg_dir.glob(f"{organ}.nii.gz"))

        if not mask_candidates:
            logger.debug(f"Organ mask not found: {organ}")
            continue

        mask_path = mask_candidates[0]

        try:
            mask_img = sitk.ReadImage(str(mask_path))
            mask_array = sitk.GetArrayFromImage(mask_img)

            if not mask_array.any():
                logger.debug(f"Organ mask is empty: {organ}")
                continue

            if not mask_array.any():
                continue
            superior_z, inferior_z = _mask_z_extent(mask_img, mask_array)

            landmarks[organ] = {
                'superior': float(superior_z),
                'inferior': float(inferior_z),
                'center': float((superior_z + inferior_z) / 2)
            }

            logger.debug(f"Extracted {organ}: superior={superior_z:.1f}mm, inferior={inferior_z:.1f}mm")

        except Exception as e:
            logger.warning(f"Failed to process organ mask {organ}: {e}")
            continue

    return landmarks


def determine_thorax_crop_boundaries(
    course_dir: Path,
    superior_margin_cm: float = 2.0,
    inferior_margin_cm: float = 2.0
) -> Tuple[float, float]:
    """
    Determine superior and inferior boundaries for thoracic cropping.

    Uses C7 vertebra (or lung apex) as superior boundary and L1 vertebra
    (or diaphragm/liver superior edge) as inferior boundary.

    Args:
        course_dir: Course directory
        superior_margin_cm: Margin above C7/lungs in cm (default: 2 cm)
        inferior_margin_cm: Margin below L1/diaphragm in cm (default: 2 cm)

    Returns:
        Tuple of (superior_z, inferior_z) in mm (patient coordinates)

    Raises:
        ValueError: If required landmarks are not found
    """
    vertebrae = extract_vertebrae_boundaries(course_dir)

    # Try to get lung boundaries for reference
    lung_patterns = ['lung_upper_lobe_left', 'lung_upper_lobe_right',
                     'lung_lower_lobe_left', 'lung_lower_lobe_right']
    organs = extract_organ_boundaries(course_dir, lung_patterns + ['heart', 'liver'])

    # Superior boundary: C7 vertebra or lung apex
    superior_z = None
    if 'vertebrae_C7' in vertebrae:
        superior_z = vertebrae['vertebrae_C7']['superior'] + (superior_margin_cm * 10)
        logger.info(f"Using C7 superior edge + {superior_margin_cm}cm for superior boundary")
    else:
        # Fallback to lung apex
        lung_superiors = [
            organs[key]['superior']
            for key in organs
            if 'lung_upper' in key and 'superior' in organs[key]
        ]
        if lung_superiors:
            superior_z = max(lung_superiors) + (superior_margin_cm * 10)
            logger.info(f"Using lung apex + {superior_margin_cm}cm for superior boundary")

    if superior_z is None:
        raise ValueError(
            "Neither C7 vertebra nor lung segmentations found - "
            "required for thorax cropping. Available structures: "
            f"vertebrae={list(vertebrae.keys())}, organs={list(organs.keys())}"
        )

    # Inferior boundary: L1 vertebra or liver/diaphragm
    inferior_z = None
    if 'vertebrae_L1' in vertebrae:
        inferior_z = vertebrae['vertebrae_L1']['inferior'] - (inferior_margin_cm * 10)
        logger.info(f"Using L1 inferior edge - {inferior_margin_cm}cm for inferior boundary")
    elif 'liver' in organs:
        inferior_z = organs['liver']['superior'] - (inferior_margin_cm * 10)
        logger.info(f"Using liver superior edge - {inferior_margin_cm}cm for inferior boundary")
    else:
        # Fallback to lung base
        lung_inferiors = [
            organs[key]['inferior']
            for key in organs
            if 'lung_lower' in key and 'inferior' in organs[key]
        ]
        if lung_inferiors:
            inferior_z = min(lung_inferiors) - (inferior_margin_cm * 10)
            logger.info(f"Using lung base - {inferior_margin_cm}cm for inferior boundary")

    if inferior_z is None:
        raise ValueError(
            "Neither L1 vertebra, liver, nor lung segmentations found - "
            "required for thorax cropping"
        )

    logger.info(
        f"Thorax crop boundaries: superior={superior_z:.1f}mm, "
        f"inferior={inferior_z:.1f}mm (span={(superior_z-inferior_z)/10:.1f}cm)"
    )

    return superior_z, inferior_z


def determine_abdomen_crop_boundaries(
    course_dir: Path,
    superior_margin_cm: float = 2.0,
    inferior_margin_cm: float = 2.0
) -> Tuple[float, float]:
    """
    Determine superior and inferior boundaries for abdominal cropping.

    Uses T12/L1 vertebra (diaphragm level) as superior boundary and
    L5 vertebra (pelvic inlet) as inferior boundary.

    Args:
        course_dir: Course directory
        superior_margin_cm: Margin above T12/L1 in cm (default: 2 cm)
        inferior_margin_cm: Margin below L5 in cm (default: 2 cm)

    Returns:
        Tuple of (superior_z, inferior_z) in mm (patient coordinates)

    Raises:
        ValueError: If required landmarks are not found
    """
    vertebrae = extract_vertebrae_boundaries(course_dir)

    # Get abdominal organ boundaries for validation
    organ_patterns = ['liver', 'spleen', 'kidney_left', 'kidney_right', 'stomach']
    organs = extract_organ_boundaries(course_dir, organ_patterns)

    # Superior boundary: T12 or L1 vertebra (diaphragm level)
    superior_z = None
    if 'vertebrae_T12' in vertebrae:
        superior_z = vertebrae['vertebrae_T12']['superior'] + (superior_margin_cm * 10)
        logger.info(f"Using T12 superior edge + {superior_margin_cm}cm for superior boundary")
    elif 'vertebrae_L1' in vertebrae:
        superior_z = vertebrae['vertebrae_L1']['superior'] + (superior_margin_cm * 10)
        logger.info(f"Using L1 superior edge + {superior_margin_cm}cm for superior boundary")
    elif 'liver' in organs:
        superior_z = organs['liver']['superior'] + (superior_margin_cm * 10)
        logger.info(f"Using liver superior edge + {superior_margin_cm}cm for superior boundary")

    if superior_z is None:
        raise ValueError(
            "Neither T12/L1 vertebrae nor liver found - "
            "required for abdomen cropping. Available vertebrae: "
            f"{list(vertebrae.keys())}, organs: {list(organs.keys())}"
        )

    # Inferior boundary: L5 vertebra (pelvic inlet)
    if 'vertebrae_L5' not in vertebrae:
        raise ValueError(
            "L5 vertebra not found in TotalSegmentator output - "
            "required for abdomen cropping. Available vertebrae: "
            f"{list(vertebrae.keys())}"
        )

    inferior_z = vertebrae['vertebrae_L5']['inferior'] - (inferior_margin_cm * 10)

    logger.info(
        f"Abdomen crop boundaries: superior={superior_z:.1f}mm, "
        f"inferior={inferior_z:.1f}mm (span={(superior_z-inferior_z)/10:.1f}cm)"
    )

    return superior_z, inferior_z


def determine_head_neck_crop_boundaries(
    course_dir: Path,
    superior_margin_cm: float = 2.0,
    inferior_margin_cm: float = 2.0
) -> Tuple[float, float]:
    """
    Determine superior and inferior boundaries for head & neck cropping.

    Uses brain/skull superior edge as superior boundary and C7/T1 vertebra
    or clavicles as inferior boundary.

    Args:
        course_dir: Course directory
        superior_margin_cm: Margin above skull/brain in cm (default: 2 cm)
        inferior_margin_cm: Margin below C7/clavicles in cm (default: 2 cm)

    Returns:
        Tuple of (superior_z, inferior_z) in mm (patient coordinates)

    Raises:
        ValueError: If required landmarks are not found
    """
    vertebrae = extract_vertebrae_boundaries(course_dir)

    # Get head/neck structures
    organ_patterns = ['brain', 'skull', 'mandible', 'clavicle_left', 'clavicle_right']
    organs = extract_organ_boundaries(course_dir, organ_patterns)

    # Superior boundary: skull or brain apex
    superior_z = None
    if 'brain' in organs:
        superior_z = organs['brain']['superior'] + (superior_margin_cm * 10)
        logger.info(f"Using brain apex + {superior_margin_cm}cm for superior boundary")
    elif 'skull' in organs:
        superior_z = organs['skull']['superior'] + (superior_margin_cm * 10)
        logger.info(f"Using skull apex + {superior_margin_cm}cm for superior boundary")

    if superior_z is None:
        raise ValueError(
            "Neither brain nor skull segmentations found - "
            "required for head & neck cropping. Available structures: "
            f"{list(organs.keys())}"
        )

    # Inferior boundary: C7/T1 vertebra or clavicles
    inferior_z = None
    if 'vertebrae_C7' in vertebrae:
        inferior_z = vertebrae['vertebrae_C7']['inferior'] - (inferior_margin_cm * 10)
        logger.info(f"Using C7 inferior edge - {inferior_margin_cm}cm for inferior boundary")
    else:
        # Fallback to clavicles
        clavicle_inferiors = [
            organs[key]['inferior']
            for key in organs
            if 'clavicle' in key and 'inferior' in organs[key]
        ]
        if clavicle_inferiors:
            inferior_z = min(clavicle_inferiors) - (inferior_margin_cm * 10)
            logger.info(f"Using clavicles inferior edge - {inferior_margin_cm}cm for inferior boundary")

    if inferior_z is None:
        raise ValueError(
            "Neither C7 vertebra nor clavicles found - "
            "required for head & neck cropping. Available structures: "
            f"vertebrae={list(vertebrae.keys())}, organs={list(organs.keys())}"
        )

    logger.info(
        f"Head & neck crop boundaries: superior={superior_z:.1f}mm, "
        f"inferior={inferior_z:.1f}mm (span={(superior_z-inferior_z)/10:.1f}cm)"
    )

    return superior_z, inferior_z


def determine_brain_crop_boundaries(
    course_dir: Path,
    superior_margin_cm: float = 1.0,
    inferior_margin_cm: float = 1.0
) -> Tuple[float, float]:
    """
    Determine superior and inferior boundaries for brain cropping.

    Uses brain segmentation boundaries with minimal margins.

    Args:
        course_dir: Course directory
        superior_margin_cm: Margin above brain apex in cm (default: 1 cm)
        inferior_margin_cm: Margin below brain base in cm (default: 1 cm)

    Returns:
        Tuple of (superior_z, inferior_z) in mm (patient coordinates)

    Raises:
        ValueError: If brain segmentation not found
    """
    organ_patterns = ['brain']
    organs = extract_organ_boundaries(course_dir, organ_patterns)

    if 'brain' not in organs:
        raise ValueError(
            "Brain segmentation not found in TotalSegmentator output - "
            "required for brain cropping. Available structures: "
            f"{list(organs.keys())}"
        )

    superior_z = organs['brain']['superior'] + (superior_margin_cm * 10)
    inferior_z = organs['brain']['inferior'] - (inferior_margin_cm * 10)

    logger.info(
        f"Brain crop boundaries: superior={superior_z:.1f}mm, "
        f"inferior={inferior_z:.1f}mm (span={(superior_z-inferior_z)/10:.1f}cm)"
    )

    return superior_z, inferior_z


def crop_image_to_boundaries(
    image: sitk.Image,
    superior_z: float,
    inferior_z: float,
    warn_once: Optional[Dict[str, bool]] = None
) -> Tuple[sitk.Image, Dict[str, object]]:
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

    z_row = np.asarray(direction).reshape(3, 3)[2]
    axes = np.flatnonzero(np.abs(z_row) > 1e-6)
    if len(axes) != 1 or not np.isclose(abs(z_row[axes[0]]), 1.0, atol=1e-6):
        raise ValueError("Physical superior/inferior cropping requires an axis-aligned grid")
    axis = int(axes[0])
    step = spacing[axis] * z_row[axis]
    requested_superior_idx = int(round((superior_z - origin[2]) / step))
    requested_inferior_idx = int(round((inferior_z - origin[2]) / step))

    # Clamp to valid range
    clamped_superior_idx = max(0, min(size[axis] - 1, requested_superior_idx))
    clamped_inferior_idx = max(0, min(size[axis] - 1, requested_inferior_idx))

    clamp_info: Dict[str, float | bool | Dict[str, float]] = {
        'requested_superior_mm': float(superior_z),
        'requested_inferior_mm': float(inferior_z),
        'applied_superior_mm': None,
        'applied_inferior_mm': None,
        'superior_clamped': bool(clamped_superior_idx != requested_superior_idx),
        'inferior_clamped': bool(clamped_inferior_idx != requested_inferior_idx),
        'ct_extent_mm': None,
    }

    z_extent = [origin[2], origin[2] + (size[axis] - 1) * step]
    clamp_info['ct_extent_mm'] = {
        'min': float(min(z_extent)),
        'max': float(max(z_extent)),
    }

    def _warn_once(key: str, message: str) -> None:
        if warn_once is None:
            logger.warning(message)
            return
        already_logged = warn_once.get(key)
        if already_logged:
            return
        warn_once[key] = True
        logger.warning(message)

    # Warn if clamping occurred
    if clamped_superior_idx != requested_superior_idx:
        actual_superior_z = origin[2] + clamped_superior_idx * step
        _warn_once(
            'superior',
            f"Superior boundary ({superior_z:.1f}mm) exceeds CT extent. "
            f"Clamped to {actual_superior_z:.1f}mm (CT boundary)",
        )
        superior_idx = clamped_superior_idx
    else:
        superior_idx = requested_superior_idx

    if clamped_inferior_idx != requested_inferior_idx:
        actual_inferior_z = origin[2] + clamped_inferior_idx * step
        _warn_once(
            'inferior',
            f"Inferior boundary ({inferior_z:.1f}mm) exceeds CT extent. "
            f"Clamped to {actual_inferior_z:.1f}mm (CT boundary). "
            f"This may affect cross-patient consistency.",
        )
        inferior_idx = clamped_inferior_idx
    else:
        inferior_idx = requested_inferior_idx

    # Ensure superior > inferior in index space (depends on direction)
    # If superior_z > inferior_z in physical space but direction[8] < 0,
    # then superior_idx < inferior_idx in index space
    z_direction = step
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
    extract_size = list(size)
    extract_size[axis] = abs(superior_idx - inferior_idx) + 1
    extract_index = [0, 0, 0]
    extract_index[axis] = min(inferior_idx, superior_idx)

    extract_filter.SetSize(extract_size)
    extract_filter.SetIndex(extract_index)

    cropped = extract_filter.Execute(image)

    logger.debug(
        f"Cropped image: {size} -> {cropped.GetSize()} "
        f"(indices: {extract_index[axis]} to {extract_index[axis] + extract_size[axis] - 1})"
    )

    clamp_info['applied_superior_mm'] = float(origin[2] + superior_idx * step)
    clamp_info['applied_inferior_mm'] = float(origin[2] + inferior_idx * step)

    return cropped, clamp_info  # type: ignore[return-value]


def apply_systematic_cropping(
    course_dir: Path,
    region: str = "pelvis",
    output_suffix: str = "_cropped",
    keep_original: bool = True,
    inferior_margin_cm: Optional[float] = None,
    superior_margin_cm: Optional[float] = None
) -> Dict[str, Path]:
    """
    Apply systematic anatomical cropping to CT and all segmentations.

    Creates cropped versions of CT NIfTI and all segmentation masks based on
    anatomical landmarks. Saves cropping metadata for downstream analysis.
    The uncropped CT/mask NIfTI files are always kept: they are shared inputs
    reused by other pipeline stages (radiomics fallback, QC, resumed
    segmentation), so this function never deletes them.

    Args:
        course_dir: Course directory
        region: Anatomical region - one of:
            - "pelvis": L1 vertebra to femoral heads + margin
            - "thorax": C7/lung apex to L1/diaphragm
            - "abdomen": T12/L1 to L5 vertebra
            - "head_neck": brain/skull apex to C7/clavicles
            - "brain": brain boundaries with minimal margins
        output_suffix: Suffix for cropped files (default: "_cropped")
        keep_original: Deprecated, ignored. Uncropped CT/mask NIfTI files are always
            kept (see above); this flag never controlled that. Retained only for
            backward compatibility with existing callers.
        inferior_margin_cm: Margin below inferior landmark in cm. If None, uses the
            region-specific default (10 cm for pelvis, 2 cm for thorax/abdomen/head_neck,
            1 cm for brain).
        superior_margin_cm: Margin above superior landmark in cm. If None, uses the
            region-specific default (unused for pelvis, 2 cm for thorax/abdomen/head_neck,
            1 cm for brain).

    Returns:
        Dictionary of cropped file paths: {'ct': Path, 'mask_name': Path, ...}

    Raises:
        ValueError: If unsupported region or landmarks not found
    """
    contract = load_course_contract(course_dir)

    if keep_original is not True:
        warnings.warn(
            "apply_systematic_cropping(keep_original=...) is deprecated and ignored; "
            "uncropped CT/mask NIfTI files are always kept.",
            DeprecationWarning,
            stacklevel=2,
        )

    from .layout import build_course_dirs

    course_dirs = build_course_dirs(course_dir)

    # Determine boundaries based on region
    supported_regions = {
        "pelvis": determine_pelvic_crop_boundaries,
        "thorax": determine_thorax_crop_boundaries,
        "abdomen": determine_abdomen_crop_boundaries,
        "head_neck": determine_head_neck_crop_boundaries,
        "brain": determine_brain_crop_boundaries
    }

    if region not in supported_regions:
        raise ValueError(
            f"Unsupported region: {region}. "
            f"Supported regions: {', '.join(supported_regions.keys())}"
        )

    # Resolve None (caller omitted) to the region-specific margin default; an
    # explicitly passed value (even one equal to a default) is used as-is.
    if region == "pelvis":
        # Pelvis superior boundary comes from the L1 vertebra edge; no superior margin applies.
        effective_superior_margin_cm = superior_margin_cm
        effective_inferior_margin_cm = inferior_margin_cm if inferior_margin_cm is not None else 10.0
        superior_z, inferior_z = determine_pelvic_crop_boundaries(
            course_dir,
            inferior_margin_cm=effective_inferior_margin_cm
        )
    elif region == "brain":
        # Brain uses minimal margins (defaults: 1 cm)
        effective_superior_margin_cm = superior_margin_cm if superior_margin_cm is not None else 1.0
        effective_inferior_margin_cm = inferior_margin_cm if inferior_margin_cm is not None else 1.0
        superior_z, inferior_z = determine_brain_crop_boundaries(
            course_dir,
            superior_margin_cm=effective_superior_margin_cm,
            inferior_margin_cm=effective_inferior_margin_cm
        )
    else:
        # Thorax, abdomen, head_neck use symmetric margins (default: 2 cm)
        effective_superior_margin_cm = superior_margin_cm if superior_margin_cm is not None else 2.0
        effective_inferior_margin_cm = inferior_margin_cm if inferior_margin_cm is not None else 2.0
        boundary_func = supported_regions[region]
        superior_z, inferior_z = boundary_func(
            course_dir,
            superior_margin_cm=effective_superior_margin_cm,
            inferior_margin_cm=effective_inferior_margin_cm
        )

    logger.info(
        f"Cropping {course_dir.name} to {region} region: "
        f"superior={superior_z:.1f}mm, inferior={inferior_z:.1f}mm"
    )

    cropped_files = {}
    warn_once = {"superior": False, "inferior": False}
    clamp_summary: Dict[str, object] = {
        'requested_superior_mm': float(superior_z),
        'requested_inferior_mm': float(inferior_z),
        'applied_superior_mm': None,
        'applied_inferior_mm': None,
        'superior_clamped': False,
        'inferior_clamped': False,
        'ct_extent_mm': None,
    }

    # Crop the planning CT selected by organize.
    ct_nifti = contract.planning_ct_nifti

    if ct_nifti is not None:
        try:
            ct_img = sitk.ReadImage(str(ct_nifti))
            ct_cropped, clamp_info = crop_image_to_boundaries(
                ct_img,
                superior_z,
                inferior_z,
                warn_once=warn_once,
            )
            clamp_summary.update(clamp_info)

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
                    mask_cropped, _ = crop_image_to_boundaries(
                        mask_img,
                        superior_z,
                        inferior_z,
                        warn_once=warn_once,
                    )

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
        'superior_margin_cm': effective_superior_margin_cm,
        'inferior_margin_cm': effective_inferior_margin_cm,
        'cropped_files': {k: str(v) for k, v in cropped_files.items()},
        'crop_boundaries_mm': clamp_summary,
        'clamped_axes': [axis for axis in ("superior", "inferior") if clamp_summary.get(f"{axis}_clamped")],
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

    The cropped masks are embedded back into full CT geometry before adding to
    the RTSTRUCT, with zeros padding the non-cropped regions. This ensures the
    RTSTRUCT contours are geometrically consistent with the referenced CT series.

    Args:
        course_dir: Course directory
        course_dirs: Course directory structure
        output_suffix: Suffix used for cropped files

    Returns:
        Path to created RTSTRUCT or None
    """
    contract = load_course_contract(course_dir)
    ct_dir = contract.planning_ct_dir
    try:
        from rt_utils import RTStructBuilder
    except ImportError:
        logger.warning("rt_utils not available; skipping cropped RTSTRUCT creation")
        return None
    from .rtstruct_geometry import image_array_for_rtstruct, place_added_rois_on_planes

    # Check if CT DICOM exists
    if ct_dir is None:
        logger.warning("Course contract has no planning CT; cannot create cropped RTSTRUCT")
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

    # Filename order keeps ROI numbers and display colours stable across filesystems.
    cropped_masks = sorted(seg_dir.glob(f"*{output_suffix}.nii.gz"), key=lambda path: path.name)
    if not cropped_masks:
        logger.warning("No cropped masks found")
        return None

    # Load CT image for geometry reference
    try:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(ct_dir))
        if not series_ids:
            logger.warning("No DICOM series found in CT directory")
            return None
        files = reader.GetGDCMSeriesFileNames(str(ct_dir), series_ids[0])
        reader.SetFileNames(files)
        ct_img = reader.Execute()
        ct_size = ct_img.GetSize()  # (x, y, z)
        ct_origin = ct_img.GetOrigin()
        ct_spacing = ct_img.GetSpacing()
        logger.debug(f"Original CT geometry: size={ct_size}, origin={ct_origin}, spacing={ct_spacing}")
    except Exception as e:
        logger.error(f"Failed to load CT DICOM: {e}")
        return None

    # Create new RTSTRUCT
    try:
        rtstruct = RTStructBuilder.create_new(dicom_series_path=str(ct_dir))
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
                continue
            # The mask retains its full physical geometry, including the crop
            # origin and any NIfTI axis flips. Sample the actual DICOM planes.
            full_mask = image_array_for_rtstruct(mask_img, rtstruct.series_data) > 0
            if not full_mask.any():
                continue

            rtstruct.add_roi(mask=full_mask, name=roi_name)
            added_count += 1


        except Exception as e:
            logger.warning(f"Failed to add ROI {roi_name}: {e}")
            continue

    if added_count == 0:
        logger.warning("No ROIs added to cropped RTSTRUCT")
        return None

    # rt-utils places mask slices on a uniform grid; never publish contours off
    # the planes of the CT images they reference.
    try:
        place_added_rois_on_planes(rtstruct.ds, rtstruct.series_data)
    except ValueError as e:
        logger.error(f"Not saving cropped RTSTRUCT: {e}")
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
