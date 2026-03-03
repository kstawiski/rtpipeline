"""
Radiomics robustness analysis module.

Implements an **RTpipeline-adapted perturbation framework** inspired by
Zwanenburg et al. 2019 (Sci Rep) for feature stability assessment:

- NTCV perturbation chain (Noise + Translation + Contour + Volume)
- Rotation sensitivity analysis (optional, for Zwanenburg N/T/R/V/C parity)
- Mask perturbations (erosion/dilation with volume adaptation)
- Image noise injection (Gaussian noise in HU)
- Rigid translations (±2-4 mm shifts)
- Contour randomization (morphological boundary perturbation)
- ICC (Intraclass Correlation Coefficient) computation
- CoV (Coefficient of Variation) and QCD metrics
- Redundancy pruning (Spearman correlation clustering)
- Determinants-of-instability analysis (mixed-effects regression)
- Multi-axis robustness evaluation (segmentation perturbation, segmentation method, scan-rescan)

Note: The default NTCV chain omits **Rotation (R)** and uses **morphological**
contour perturbation rather than supervoxel-based randomization. These
differences from the original Zwanenburg 2019 N/T/R/V/C framework are
intentional simplifications documented in the manuscript. Rotation sensitivity
analysis is available via `generate_rotation_sensitivity_perturbations()`.

Based on 2023-2025 radiomics stability research:
- Zwanenburg et al. 2019 (Sci Rep): NTCV perturbation chains with ICC >0.75
- Lo Iacono et al. 2024 (SpringerLink): volume adaptation for stability
- Poirot et al. 2022 (Sci Rep): multi-method ICC with Pingouin
- Traverso et al. 2024: cross-extractor reproducibility
- OPC RobustDB 2025 (PMID: 41367878): feature stability atlas with pruning
- Modern best practice: 30-60 perturbations per ROI for comprehensive stability testing
- Conservative clinical thresholds: ICC >0.90 and CoV <10%
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk

from .config import PipelineConfig
from .layout import build_course_dirs

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass
class PerturbationConfig:
    """Configuration for mask perturbation (RTpipeline-adapted NTCV chain: Noise + Translation + Contour + Volume).

    Note: The original Zwanenburg 2019 framework uses N/T/R/V/C including Rotation.
    Rotation is available as a separate sensitivity analysis via rotation_angles.
    """
    apply_to_structures: List[str] = field(default_factory=lambda: ["GTV", "CTV", "PTV", "BLADDER", "RECTUM"])
    small_volume_changes: List[float] = field(default_factory=lambda: [-0.15, 0.0, 0.15])
    large_volume_changes: List[float] = field(default_factory=lambda: [-0.30, 0.0, 0.30])
    n_random_contour_realizations: int = 0
    max_translation_mm: float = 0.0
    contour_randomization_mm: float = 0.0  # C8 fix: independent contour noise (0 = auto from max_translation_mm/2 for backwards compat)
    noise_levels: List[float] = field(default_factory=lambda: [0.0])  # Gaussian noise std dev in HU
    intensity: str = "standard"  # "mild", "standard", "aggressive" - controls perturbation count
    rotation_angles: List[float] = field(default_factory=list)  # Rotation sensitivity: e.g., [1, -1, 3, -3] degrees


@dataclass
class ICCConfig:
    """Configuration for ICC computation."""
    implementation: Literal["pingouin", "manual"] = "pingouin"
    icc_type: Literal["ICC1", "ICC2", "ICC3"] = "ICC3"
    ci: bool = True


@dataclass
class MetricsConfig:
    """Configuration for robustness metrics."""
    icc: ICCConfig = field(default_factory=ICCConfig)
    cov_enabled: bool = True
    qcd_enabled: bool = True


@dataclass
class RobustnessThresholds:
    """Thresholds for classifying feature robustness."""
    icc_robust: float = 0.90
    icc_acceptable: float = 0.75
    cov_robust_pct: float = 10.0
    cov_acceptable_pct: float = 20.0


@dataclass
class RobustnessConfig:
    """Main configuration for radiomics robustness analysis."""
    enabled: bool = True
    modes: List[str] = field(default_factory=lambda: ["segmentation_perturbation"])
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    thresholds: RobustnessThresholds = field(default_factory=RobustnessThresholds)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobustnessConfig":
        """Load configuration from dictionary (parsed YAML)."""
        if not data:
            return cls()

        enabled = data.get("enabled", False)
        modes = data.get("modes", ["segmentation_perturbation"])
        if isinstance(modes, str):
            modes = [modes]

        # Perturbation config
        pert_data = data.get("segmentation_perturbation", {})
        perturbation = PerturbationConfig(
            apply_to_structures=pert_data.get("apply_to_structures", ["GTV", "CTV", "PTV", "BLADDER", "RECTUM"]),
            small_volume_changes=pert_data.get("small_volume_changes", [-0.15, 0.0, 0.15]),
            large_volume_changes=pert_data.get("large_volume_changes", [-0.30, 0.0, 0.30]),
            n_random_contour_realizations=pert_data.get("n_random_contour_realizations", 0),
            max_translation_mm=pert_data.get("max_translation_mm", 0.0),
            contour_randomization_mm=pert_data.get("contour_randomization_mm", 0.0),
            noise_levels=pert_data.get("noise_levels", [0.0]),
            intensity=pert_data.get("intensity", "standard"),
        )

        # Metrics config
        metrics_data = data.get("metrics", {})
        icc_data = metrics_data.get("icc", {})
        icc_config = ICCConfig(
            implementation=icc_data.get("implementation", "pingouin"),
            icc_type=icc_data.get("icc_type", "ICC3"),
            ci=icc_data.get("ci", True),
        )
        metrics = MetricsConfig(
            icc=icc_config,
            cov_enabled=metrics_data.get("cov", {}).get("enabled", True),
            qcd_enabled=metrics_data.get("qcd", {}).get("enabled", True),
        )

        # Thresholds
        thresh_data = data.get("thresholds", {})
        thresholds = RobustnessThresholds(
            icc_robust=thresh_data.get("icc", {}).get("robust", 0.90),
            icc_acceptable=thresh_data.get("icc", {}).get("acceptable", 0.75),
            cov_robust_pct=thresh_data.get("cov", {}).get("robust_pct", 10.0),
            cov_acceptable_pct=thresh_data.get("cov", {}).get("acceptable_pct", 20.0),
        )

        return cls(
            enabled=enabled,
            modes=modes,
            perturbation=perturbation,
            metrics=metrics,
            thresholds=thresholds,
        )


def volume_adapt_mask(mask: sitk.Image, tau: float, max_iterations: int = 20) -> Optional[sitk.Image]:
    """
    Apply IBSI-style volume adaptation to mask via iterative erosion/dilation.

    Uses spacing-aware structuring elements to ensure isotropic physical-space
    morphology on anisotropic voxel grids (e.g., 1x1x3 mm clinical CTs).

    Args:
        mask: Binary SimpleITK image
        tau: Target volume change ratio (e.g., 0.15 for +15%, -0.15 for -15%)
        max_iterations: Maximum number of erosion/dilation iterations

    Returns:
        Perturbed mask or None if unsuccessful
    """
    if abs(tau) < 1e-6:
        return mask

    arr = sitk.GetArrayFromImage(mask).astype(bool)
    original_volume = arr.sum()

    # C2 fix: spacing-aware minimum volume threshold (at least 0.01 cm³ = 10 mm³)
    spacing = mask.GetSpacing()  # (x, y, z) in mm
    voxel_vol_mm3 = float(np.prod(spacing))
    min_voxels = max(10, int(np.ceil(10.0 / voxel_vol_mm3)))  # at least 10 mm³

    if original_volume < min_voxels:
        logger.debug("Mask too small for volume adaptation (volume=%d voxels, min=%d)", original_volume, min_voxels)
        return None

    target_volume = int(original_volume * (1.0 + tau))
    # Determine operation (erosion or dilation)
    is_erosion = tau < 0
    operation = sitk.BinaryErode if is_erosion else sitk.BinaryDilate

    # C2 fix: compute spacing-aware structuring element radii
    # Target ~1mm physical radius per iteration; scale inversely by voxel spacing
    target_radius_mm = 1.0
    kernel_radius = [max(1, int(round(target_radius_mm / s))) for s in spacing]
    # SimpleITK uses (x, y, z) ordering for kernel radius
    logger.debug("Volume adaptation kernel radius (voxels): %s for spacing %s mm", kernel_radius, spacing)

    # Iterative approach - track all candidates including overshoot
    current_mask = mask
    current_volume = original_volume
    best_mask = None  # Start with None to ensure we return a perturbed mask
    best_diff = float('inf')
    last_valid_perturbed = None  # Track last valid perturbed mask even if it overshoots

    for iteration in range(1, max_iterations + 1):
        try:
            perturbed = operation(current_mask, kernel_radius, sitk.sitkBall)
            perturbed_arr = sitk.GetArrayFromImage(perturbed).astype(bool)
            perturbed_volume = perturbed_arr.sum()

            if perturbed_volume == current_volume:
                logger.debug(
                    "Iteration %d produced no volume change during volume adaptation",
                    iteration,
                )
                break

            # Always track the last valid perturbed mask (non-empty)
            if perturbed_volume >= 5:
                last_valid_perturbed = perturbed

            diff = abs(perturbed_volume - target_volume)
            if diff < best_diff:
                best_diff = diff
                best_mask = perturbed

            current_mask = perturbed
            current_volume = perturbed_volume

            if perturbed_volume == 0 and is_erosion:
                logger.debug("Mask fully eroded during volume adaptation")
                break

            # Stop if we reached target or overshot
            if is_erosion and perturbed_volume <= target_volume:
                # Even if we overshot, use this perturbed mask if it's closer to target
                # than any previous candidate (or if we have no candidate yet)
                if best_mask is None or diff <= best_diff:
                    best_mask = perturbed
                break
            elif not is_erosion and perturbed_volume >= target_volume:
                if best_mask is None or diff <= best_diff:
                    best_mask = perturbed
                break
        except Exception as e:
            logger.debug("Volume adaptation failed at iteration %d: %s", iteration, e)
            break

    # If we never found a best_mask (shouldn't happen), use last valid perturbed
    if best_mask is None:
        best_mask = last_valid_perturbed

    # Final fallback: if still None, we couldn't create any perturbation
    if best_mask is None:
        logger.debug("Could not generate any valid perturbation for tau=%.3f", tau)
        return None

    # Check if result is reasonable
    best_arr = sitk.GetArrayFromImage(best_mask).astype(bool)
    best_volume = best_arr.sum()

    if best_volume < min_voxels:
        logger.debug("Perturbed mask too small (volume=%d voxels, min=%d)", best_volume, min_voxels)
        return None

    # Ensure we're returning a DIFFERENT mask from original
    if best_volume == original_volume:
        logger.debug("Perturbation produced no volume change, using last valid perturbed")
        if last_valid_perturbed is not None:
            best_mask = last_valid_perturbed
            best_arr = sitk.GetArrayFromImage(best_mask).astype(bool)
            best_volume = best_arr.sum()

    achieved_tau = (best_volume - original_volume) / original_volume
    logger.debug("Volume adaptation: target τ=%.3f, achieved τ=%.3f (volume %d→%d)",
                 tau, achieved_tau, original_volume, best_volume)

    return best_mask


def translate_mask(mask: sitk.Image, translation_mm: Tuple[float, float, float]) -> sitk.Image:
    """
    Apply rigid translation to mask.
    
    Args:
        mask: Binary SimpleITK image
        translation_mm: Translation vector in mm (x, y, z)
        
    Returns:
        Translated mask
    """
    transform = sitk.TranslationTransform(3, translation_mm)
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetReferenceImage(mask)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    
    return resampler.Execute(mask)


def randomize_contour(
    mask: sitk.Image,
    randomization_mm: float,
    rng: Optional[np.random.Generator] = None,
) -> sitk.Image:
    """
    Apply boundary randomization to simulate inter-observer variability.

    Applies morphological opening or closing with spacing-aware structuring elements.

    Args:
        mask: Binary SimpleITK image
        randomization_mm: Maximum boundary displacement in mm
        rng: Optional numpy Generator for reproducible randomization (avoids global state)

    Returns:
        Mask with randomized contour
    """
    if rng is None:
        rng = np.random.default_rng()

    spacing = mask.GetSpacing()
    # C2 fix: spacing-aware kernel radii instead of isotropic voxel radius
    kernel_radius = [max(1, int(round(randomization_mm / s))) for s in spacing]

    # Randomly choose erosion or dilation
    if rng.random() > 0.5:
        # Slight erosion followed by dilation (morphological opening)
        temp = sitk.BinaryErode(mask, kernel_radius, sitk.sitkBall)
        result = sitk.BinaryDilate(temp, kernel_radius, sitk.sitkBall)
    else:
        # Slight dilation followed by erosion (morphological closing)
        temp = sitk.BinaryDilate(mask, kernel_radius, sitk.sitkBall)
        result = sitk.BinaryErode(temp, kernel_radius, sitk.sitkBall)

    return result


def add_noise_to_image(
    image: sitk.Image,
    noise_std_hu: float,
    rng: Optional[np.random.Generator] = None,
) -> sitk.Image:
    """
    Add Gaussian noise to image for image-based perturbation testing.

    Args:
        image: CT/MR image
        noise_std_hu: Standard deviation of Gaussian noise in HU
        rng: Optional numpy Generator for reproducible noise (avoids global state)

    Returns:
        Noisy image
    """
    if noise_std_hu <= 0:
        return image

    if rng is None:
        rng = np.random.default_rng()

    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    noise = rng.normal(0, noise_std_hu, arr.shape).astype(np.float32)
    noisy_arr = arr + noise
    
    noisy_img = sitk.GetImageFromArray(noisy_arr)
    noisy_img.CopyInformation(image)
    
    return noisy_img


def generate_perturbed_masks(
    original_mask: sitk.Image,
    volume_changes: List[float],
    structure_name: str,
) -> Dict[str, sitk.Image]:
    """
    Generate multiple perturbed versions of a mask via volume adaptation.

    Args:
        original_mask: Original binary mask
        volume_changes: List of target volume change ratios (e.g., [-0.15, 0, 0.15])
        structure_name: Structure name for logging

    Returns:
        Dictionary mapping perturbation_id to perturbed mask
    """
    perturbed = {}

    for tau in volume_changes:
        pert_id = f"tau{tau:+.2f}".replace(".", "p").replace("+", "plus").replace("-", "minus")

        if abs(tau) < 1e-6:
            # Original mask
            perturbed[pert_id] = original_mask
        else:
            pert_mask = volume_adapt_mask(original_mask, tau)
            if pert_mask is not None:
                perturbed[pert_id] = pert_mask
            else:
                logger.warning("Failed to generate perturbation %s for %s", pert_id, structure_name)

    return perturbed


def generate_ntcv_perturbations(
    original_mask: sitk.Image,
    original_image: sitk.Image,
    config: PerturbationConfig,
    structure_name: str,
) -> Tuple[Dict[str, sitk.Image], Dict[str, sitk.Image]]:
    """
    Generate NTCV (Noise + Translation + Contour + Volume) perturbation chain.

    This implements systematic perturbation chains following Zwanenburg et al. 2019
    (Scientific Reports) and IBSI-2 guidelines for comprehensive feature stability
    assessment. The NTCV order is standardized and should not be changed.

    Perturbation Order (IBSI-recommended):
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1. **N (Noise)**: Image intensity noise (Gaussian) - applied first to simulate
       acquisition variability. Applied to the IMAGE, not the mask.

    2. **T (Translation)**: Geometric shifts - simulates inter-observer ROI
       placement variability. Applied BEFORE contour randomization so that
       boundary perturbations are applied to the shifted ROI.

    3. **C (Contour)**: Boundary randomization - simulates segmentation
       uncertainty at ROI edges. Applied AFTER translation to perturb the
       already-shifted boundary.

    4. **V (Volume)**: Erosion/dilation - simulates systematic over/under-
       segmentation. Applied LAST as it represents the final morphological
       adjustment to the perturbed contour.

    Why This Order Matters:
    ~~~~~~~~~~~~~~~~~~~~~~~
    The NTCV order ensures proper propagation of uncertainty sources:
    - Noise affects feature values but not geometry
    - Translation affects geometric position before boundary uncertainty
    - Contour randomization adds edge uncertainty to translated position
    - Volume adaptation is the final morphological adjustment

    Reversing or randomizing the order would produce non-comparable perturbation
    sets and potentially invalid ICC estimates. For example, applying volume
    adaptation before translation would create a different systematic bias.

    References:
        Zwanenburg et al. (2019). The Image Biomarker Standardization Initiative:
        Standardized Quantitative Radiomics for High-Throughput Image-based
        Phenotyping. Radiology, 295(2), 328-338.

    Args:
        original_mask: Original binary mask
        original_image: CT/MR image (for noise perturbations)
        config: Perturbation configuration
        structure_name: Structure name for logging

    Returns:
        Tuple of (perturbed_masks_dict, perturbed_images_dict)
        Both dictionaries map perturbation_id to SimpleITK images.
        The perturbation_id encodes the full chain: "ntcv_n{noise}_t{x}_{y}_{z}_c{idx}_v{pct}"
    """
    perturbed_masks = {}
    perturbed_images = {}
    
    # Determine perturbation count based on intensity level
    if config.intensity == "mild":
        # Minimal testing: ~10-15 perturbations
        volume_changes = config.small_volume_changes[:2] if len(config.small_volume_changes) > 2 else config.small_volume_changes
        translation_steps = 1 if config.max_translation_mm > 0 else 0
        contour_realizations = min(1, config.n_random_contour_realizations)
        noise_levels = config.noise_levels[:1] if len(config.noise_levels) > 1 else config.noise_levels
    elif config.intensity == "aggressive":
        # Comprehensive testing: 30-60 perturbations (research-grade)
        volume_changes = config.large_volume_changes + config.small_volume_changes
        translation_steps = 2 if config.max_translation_mm > 0 else 0
        contour_realizations = config.n_random_contour_realizations
        noise_levels = config.noise_levels
    else:  # standard
        # Balanced testing: 15-30 perturbations
        volume_changes = config.small_volume_changes
        translation_steps = 1 if config.max_translation_mm > 0 else 0
        contour_realizations = config.n_random_contour_realizations
        noise_levels = config.noise_levels
    
    pert_count = 0
    # C9 fix: master RNG for reproducible but isolated random state
    master_rng = np.random.Generator(np.random.PCG64(42))

    # Generate perturbations using combinatorial approach
    for noise_std in noise_levels:
        # N: Noise perturbation (image-based)
        if noise_std > 0:
            noise_rng = np.random.Generator(np.random.PCG64(master_rng.integers(2**63)))
            noisy_image = add_noise_to_image(original_image, noise_std, rng=noise_rng)
            noise_suffix = f"_n{int(noise_std)}"
        else:
            noisy_image = original_image
            noise_suffix = ""
        
        # T: Translation perturbations (geometric)
        translation_vectors = [(0, 0, 0)]  # Always include no-translation
        if config.max_translation_mm > 0 and translation_steps > 0:
            # Generate translations in different directions
            max_t = config.max_translation_mm
            if translation_steps == 1:
                # Single direction (superior-inferior)
                translation_vectors.extend([(0, 0, max_t), (0, 0, -max_t)])
            else:
                # Multiple directions (x, y, z)
                translation_vectors.extend([
                    (max_t, 0, 0), (-max_t, 0, 0),
                    (0, max_t, 0), (0, -max_t, 0),
                    (0, 0, max_t), (0, 0, -max_t),
                ])
        
        for trans_vec in translation_vectors:
            # Apply translation
            if any(abs(t) > 1e-3 for t in trans_vec):
                translated_mask = translate_mask(original_mask, trans_vec)
                trans_suffix = f"_t{int(trans_vec[0])}_{int(trans_vec[1])}_{int(trans_vec[2])}"
            else:
                translated_mask = original_mask
                trans_suffix = ""
            
            # C: Contour randomization (boundary noise)
            contour_variants = [translated_mask]  # Always include original contour
            if config.n_random_contour_realizations > 0 and contour_realizations > 0:
                # C9 fix: use isolated RNG instead of polluting global np.random state
                rng = np.random.Generator(np.random.PCG64(42 + pert_count))
                # C8 fix: use independent contour_randomization_mm config;
                # fall back to max_translation_mm/2 for backwards compatibility
                contour_noise_mm = config.contour_randomization_mm if config.contour_randomization_mm > 0 else config.max_translation_mm / 2
                for c_idx in range(contour_realizations):
                    try:
                        randomized = randomize_contour(translated_mask, contour_noise_mm, rng=rng)
                        contour_variants.append(randomized)
                    except Exception as e:
                        logger.debug("Contour randomization failed for %s: %s", structure_name, e)
            
            for c_idx, contour_mask in enumerate(contour_variants):
                contour_suffix = f"_c{c_idx}" if c_idx > 0 else ""
                
                # V: Volume adaptation (erosion/dilation)
                for tau in volume_changes:
                    if abs(tau) < 1e-6:
                        final_mask = contour_mask
                        vol_suffix = "_v0"
                    else:
                        final_mask = volume_adapt_mask(contour_mask, tau)
                        if final_mask is None:
                            logger.debug("Volume adaptation failed for %s (tau=%.2f)", structure_name, tau)
                            continue
                        vol_suffix = f"_v{int(tau*100):+03d}"
                    
                    # Create unique perturbation ID
                    pert_id = f"ntcv{noise_suffix}{trans_suffix}{contour_suffix}{vol_suffix}"
                    
                    # Store results
                    perturbed_masks[pert_id] = final_mask
                    perturbed_images[pert_id] = noisy_image
                    pert_count += 1
    
    logger.info("Generated %d NTCV perturbations for %s (intensity=%s)",
                pert_count, structure_name, config.intensity)
    
    return perturbed_masks, perturbed_images


# ============================================================================
# Radiomics Feature Extraction
# ============================================================================

def extract_features_for_masks(
    image: sitk.Image,
    masks: Dict[str, sitk.Image],
    config: PipelineConfig,
    modality: str = "CT",
    structure_name: str = "",
    patient_id: str = "",
    course_id: str = "",
    perturbed_images: Optional[Dict[str, sitk.Image]] = None,
) -> pd.DataFrame:
    """
    Extract radiomics features for multiple mask variants.

    Args:
        image: CT/MR image (base image)
        masks: Dictionary of {perturbation_id: mask}
        config: Pipeline configuration
        modality: "CT" or "MR"
        structure_name: ROI name
        patient_id: Patient identifier
        course_id: Course identifier
        perturbed_images: Optional dictionary of {perturbation_id: perturbed_image} for noise perturbations

    Returns:
        Tidy DataFrame with columns [patient_id, course_id, structure, perturbation_id, feature_name, value]
    """
    from .radiomics import _extractor

    rows = []

    for pert_id, mask in masks.items():
        try:
            ext = _extractor(config, modality)
            if ext is None:
                logger.debug("No radiomics extractor available for %s/%s", structure_name, pert_id)
                continue

            # Use perturbed image if available (for noise perturbations)
            current_image = perturbed_images.get(pert_id, image) if perturbed_images else image

            # Execute PyRadiomics
            result = ext.execute(current_image, mask)

            # Convert to flat dictionary
            for key, value in result.items():
                if isinstance(value, (int, float, np.floating, np.integer)):
                    rows.append({
                        "patient_id": patient_id,
                        "course_id": course_id,
                        "modality": modality,
                        "structure": structure_name,
                        "perturbation_id": pert_id,
                        "feature_name": str(key),
                        "value": float(value),
                    })
        except Exception as e:
            logger.debug("Feature extraction failed for %s/%s: %s", structure_name, pert_id, e)

    return pd.DataFrame(rows)


# ============================================================================
# Robustness Metrics Computation
# ============================================================================

def compute_icc_pingouin(
    df: pd.DataFrame,
    icc_config: ICCConfig,
) -> Dict[str, float]:
    """
    Compute ICC using Pingouin library.

    Args:
        df: DataFrame with columns ['subject', 'rater', 'value']
        icc_config: ICC configuration

    Returns:
        Dictionary with keys: icc, icc_ci95_low, icc_ci95_high
    """
    try:
        import pingouin as pg
    except ImportError:
        logger.warning("Pingouin not available; install with: pip install pingouin")
        return {"icc": np.nan, "icc_ci95_low": np.nan, "icc_ci95_high": np.nan}

    try:
        # --- C1 fix: detect unbalanced data before ICC computation ---
        # Pingouin's nan_policy="omit" performs listwise deletion, dropping entire
        # subjects when ANY rater value is missing. This silently biases ICC upward
        # (surviving subjects are easier cases). We detect and report this explicitly.
        n_raters_expected = df["rater"].nunique()
        subject_counts = df.groupby("subject")["rater"].nunique()
        complete_subjects = (subject_counts == n_raters_expected).sum()
        total_subjects = len(subject_counts)
        incomplete_subjects = total_subjects - complete_subjects
        drop_pct = (incomplete_subjects / total_subjects * 100) if total_subjects > 0 else 0

        if incomplete_subjects > 0:
            logger.warning(
                "ICC: %d/%d subjects (%.1f%%) have missing perturbations and will be "
                "dropped by listwise deletion. ICC may be biased upward.",
                incomplete_subjects, total_subjects, drop_pct,
            )
            if drop_pct > 10:
                logger.warning(
                    "ICC: >10%% subjects dropped — consider using ICC2 (two-way random) "
                    "or mixed-effects models for unbalanced designs."
                )

        # Use only complete cases to make the listwise deletion explicit
        # (rather than relying on Pingouin's silent nan_policy="omit")
        if incomplete_subjects > 0:
            complete_subject_ids = subject_counts[subject_counts == n_raters_expected].index
            df_balanced = df[df["subject"].isin(complete_subject_ids)].copy()
            if df_balanced["subject"].nunique() < 2:
                logger.warning("ICC: fewer than 2 complete subjects remain after dropping incomplete cases")
                return {
                    "icc": np.nan, "icc_ci95_low": np.nan, "icc_ci95_high": np.nan,
                    "n_subjects_dropped": int(incomplete_subjects),
                    "n_subjects_complete": int(complete_subjects),
                }
        else:
            df_balanced = df

        icc_res = pg.intraclass_corr(
            data=df_balanced,
            targets="subject",
            raters="rater",
            ratings="value",
            nan_policy="raise"  # Fail loudly if data is still unbalanced
        )

        # Select appropriate ICC type
        filtered = icc_res.loc[icc_res["Type"] == icc_config.icc_type]
        if filtered.empty:
            logger.debug("ICC type %s not found in results", icc_config.icc_type)
            return {"icc": np.nan, "icc_ci95_low": np.nan, "icc_ci95_high": np.nan}
        row = filtered.iloc[0]

        result = {
            "icc": float(row["ICC"]),
            "n_subjects_dropped": int(incomplete_subjects),
            "n_subjects_complete": int(complete_subjects),
        }

        if icc_config.ci:
            ci = row["CI95%"]
            result["icc_ci95_low"] = float(ci[0])
            result["icc_ci95_high"] = float(ci[1])
        else:
            result["icc_ci95_low"] = np.nan
            result["icc_ci95_high"] = np.nan

        return result
    except Exception as e:
        logger.debug("ICC computation failed: %s", e)
        return {"icc": np.nan, "icc_ci95_low": np.nan, "icc_ci95_high": np.nan}


def compute_cov(values: np.ndarray) -> float:
    """Compute Coefficient of Variation (CoV) in percent."""
    if len(values) < 2:
        return np.nan

    mean_val = np.mean(values)
    if abs(mean_val) < 1e-10:
        return np.nan

    std_val = np.std(values, ddof=1)
    return float((std_val / abs(mean_val)) * 100.0)


def compute_qcd(values: np.ndarray) -> float:
    """Compute Quartile Coefficient of Dispersion (QCD)."""
    if len(values) < 4:
        return np.nan

    q1, q3 = np.percentile(values, [25, 75])

    if abs(q1 + q3) < 1e-10:
        return np.nan

    return float((q3 - q1) / (q3 + q1))


def summarize_feature_stability(
    df_long: pd.DataFrame,
    config: RobustnessConfig,
    group_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute per-feature robustness metrics (ICC, CoV, QCD) and classify robustness.

    Args:
        df_long: Tidy DataFrame with columns [patient_id, course_id, structure, segmentation_source,
                 perturbation_id, feature_name, value]
        config: Robustness configuration

    Returns:
        DataFrame with one row per (structure, segmentation_source, feature_name) containing metrics and robustness label
    """
    rows = []

    if group_columns is None:
        if "segmentation_source" in df_long.columns:
            group_columns = ["structure", "segmentation_source", "feature_name"]
        else:
            group_columns = ["structure", "feature_name"]

    for group_key, group in df_long.groupby(group_columns):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row_data = {col: val for col, val in zip(group_columns, key_values)}

        values = group["value"].to_numpy(dtype=float)

        if len(values) < 2:
            cov = np.nan
            qcd = np.nan
        else:
            cov = compute_cov(values) if config.metrics.cov_enabled else np.nan
            qcd = compute_qcd(values) if config.metrics.qcd_enabled else np.nan

        # Build unique subject identifier including course_id to avoid collapsing
        # different courses of the same patient into one subject for ICC computation.
        # Subject = patient_id + course_id + structure + segmentation_source
        # This ensures each course/timepoint is treated as a distinct subject instance.
        icc_df = group[["patient_id", "perturbation_id", "value"]].copy()
        icc_df.rename(columns={"perturbation_id": "rater"}, inplace=True)

        # Start with patient_id
        subject_parts = [group["patient_id"].astype(str)]

        # Include course_id if available (critical for multi-course/longitudinal data)
        if "course_id" in group.columns:
            subject_parts.append(group["course_id"].astype(str))

        # Include structure (already fixed by grouping, but ensures uniqueness across groups)
        if "structure" in df_long.columns:
            subject_parts.append(group["structure"].astype(str))

        # Include segmentation_source
        if "segmentation_source" in df_long.columns:
            subject_parts.append(group.get("segmentation_source", "unknown").astype(str))

        # Combine all parts with underscore separator
        icc_df["subject"] = subject_parts[0]
        for part in subject_parts[1:]:
            icc_df["subject"] = icc_df["subject"] + "_" + part

        n_subjects = icc_df["subject"].nunique()
        n_raters = icc_df["rater"].nunique()

        if n_subjects < 2 or n_raters < 2:
            icc_info = {"icc": np.nan, "icc_ci95_low": np.nan, "icc_ci95_high": np.nan}
        else:
            icc_info = compute_icc_pingouin(icc_df, config.metrics.icc)

        icc = icc_info["icc"]
        icc_ci_low = icc_info.get("icc_ci95_low", np.nan)
        icc_ci_high = icc_info.get("icc_ci95_high", np.nan)

        icc_for_threshold = icc_ci_low if not np.isnan(icc_ci_low) else icc

        robust_icc = icc_for_threshold >= config.thresholds.icc_robust
        acceptable_icc = icc_for_threshold >= config.thresholds.icc_acceptable

        robust_cov = cov <= config.thresholds.cov_robust_pct if not np.isnan(cov) else False
        acceptable_cov = cov <= config.thresholds.cov_acceptable_pct if not np.isnan(cov) else False

        if robust_icc and robust_cov:
            robustness_label = "robust"
        elif acceptable_icc and acceptable_cov:
            robustness_label = "acceptable"
        else:
            robustness_label = "poor"

        row_data.update({
            "feature_name": row_data.get("feature_name", group["feature_name"].iloc[0]),
            "n_subjects": n_subjects,
            "n_courses": group["course_id"].nunique() if "course_id" in group.columns else np.nan,
            "n_perturbations": n_raters,
            "icc": icc,
            "icc_ci95_low": icc_ci_low,
            "icc_ci95_high": icc_ci_high,
            "cov_pct": cov,
            "qcd": qcd,
            "robustness_label": robustness_label,
            "pass_seg_perturb": robustness_label in ["robust", "acceptable"],
        })

        rows.append(row_data)

    return pd.DataFrame(rows)


# ============================================================================
# Rotation Perturbation (v1.2 — sensitivity analysis)
# ============================================================================

def rotate_image_and_mask(
    image: sitk.Image,
    mask: sitk.Image,
    angle_degrees: float,
    axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Apply rigid rotation to both image and mask around the volume center.

    This implements the **R** (Rotation) perturbation absent from the default
    RTpipeline NTCV chain. The original Zwanenburg 2019 framework (N/T/R/V/C)
    included rotation; RTpipeline's adapted framework omits it by default.
    This function enables rotation sensitivity analyses to quantify the impact
    of that omission.

    Args:
        image: CT/MR SimpleITK image
        mask: Binary SimpleITK mask (same geometry as image)
        angle_degrees: Rotation angle in degrees
        axis: Rotation axis as (x, y, z) unit vector.
              Default (0,0,1) = axial rotation (around superior-inferior axis).

    Returns:
        Tuple of (rotated_image, rotated_mask)
    """
    angle_rad = np.deg2rad(angle_degrees)

    # Compute center of rotation (volume center in physical coordinates)
    size = image.GetSize()
    center_index = [s / 2.0 for s in size]
    center_physical = image.TransformContinuousIndexToPhysicalPoint(center_index)

    # Create Euler3D transform (rotation around center)
    transform = sitk.Euler3DTransform()
    transform.SetCenter(center_physical)

    # Set rotation angles based on axis
    ax = np.array(axis, dtype=float)
    ax = ax / (np.linalg.norm(ax) + 1e-12)
    # Euler3D uses rotations around x, y, z axes
    transform.SetRotation(
        float(ax[0] * angle_rad),
        float(ax[1] * angle_rad),
        float(ax[2] * angle_rad),
    )

    # Resample image with linear interpolation
    rotated_image = sitk.Resample(
        image,
        image,  # reference
        transform,
        sitk.sitkLinear,
        float(sitk.GetArrayViewFromImage(image).min()),  # default pixel value
    )

    # Resample mask with nearest neighbor (preserve binary)
    rotated_mask = sitk.Resample(
        mask,
        mask,
        transform,
        sitk.sitkNearestNeighbor,
        0,
    )

    return rotated_image, rotated_mask


def generate_rotation_sensitivity_perturbations(
    original_mask: sitk.Image,
    original_image: sitk.Image,
    rotation_angles: Optional[List[float]] = None,
    base_config: Optional[PerturbationConfig] = None,
    structure_name: str = "",
) -> Tuple[Dict[str, sitk.Image], Dict[str, sitk.Image]]:
    """
    Generate rotation-augmented perturbations for sensitivity analysis.

    Produces perturbation sets with and without rotation, enabling direct
    comparison of ICC values to quantify the impact of omitting rotation
    from the default NTCV chain.

    Default rotation angles: [+1, -1, +3, -3] degrees (axial plane).

    Args:
        original_mask: Original binary mask
        original_image: CT/MR image
        rotation_angles: List of rotation angles in degrees. Default: [1, -1, 3, -3]
        base_config: Optional base perturbation config (for combining with NTCV)
        structure_name: Structure name for logging

    Returns:
        Tuple of (perturbed_masks_dict, perturbed_images_dict)
    """
    if rotation_angles is None:
        rotation_angles = [1.0, -1.0, 3.0, -3.0]

    perturbed_masks = {"rot_original": original_mask}
    perturbed_images = {"rot_original": original_image}

    for angle in rotation_angles:
        try:
            rot_img, rot_mask = rotate_image_and_mask(
                original_image, original_mask, angle
            )
            pert_id = f"rot_{angle:+.1f}deg".replace(".", "p").replace("+", "plus").replace("-", "minus")
            perturbed_masks[pert_id] = rot_mask
            perturbed_images[pert_id] = rot_img
        except Exception as e:
            logger.warning(
                "Rotation perturbation failed for %s at %.1f°: %s",
                structure_name, angle, e,
            )

    logger.info(
        "Generated %d rotation perturbations for %s (angles: %s)",
        len(perturbed_masks) - 1, structure_name, rotation_angles,
    )

    return perturbed_masks, perturbed_images


# ============================================================================
# Redundancy Pruning (v1.2 — non-redundant robust feature panels)
# ============================================================================

def prune_redundant_features(
    stability_df: pd.DataFrame,
    feature_values_df: pd.DataFrame,
    correlation_threshold: float = 0.90,
    robustness_label_col: str = "robustness_label",
    icc_col: str = "icc",
    feature_col: str = "feature_name",
    structure_col: str = "structure",
) -> pd.DataFrame:
    """
    Prune redundant features using Spearman correlation clustering.

    For each structure, clusters robust features by Spearman |r| > threshold,
    then selects the representative with the highest mean ICC from each cluster.
    Reports both the full robust set and the non-redundant subset.

    This addresses the concern that wavelet/filter-expanded feature families
    can inflate robustness counts without adding independent information
    (METRICS, PMID: 38228979).

    Args:
        stability_df: Output from summarize_feature_stability() with ICC/robustness labels.
        feature_values_df: Wide-format DataFrame with features as columns, patients as rows.
            Must contain 'structure' and 'patient_id' columns plus feature value columns.
        correlation_threshold: Spearman |r| above which features are considered redundant.
            Default: 0.90 (conservative, per IBSI recommendations).
        robustness_label_col: Column name for robustness classification.
        icc_col: Column name for ICC values.
        feature_col: Column name for feature names.
        structure_col: Column name for structure names.

    Returns:
        stability_df with additional columns:
        - 'redundancy_cluster': cluster ID (integer) within each structure
        - 'is_cluster_representative': True if this feature is the cluster representative
        - 'cluster_size': number of features in the cluster
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    result_rows = []

    for structure, struct_stability in stability_df.groupby(structure_col):
        # Filter to robust/acceptable features only
        robust_mask = struct_stability[robustness_label_col].isin(["robust", "acceptable"])
        robust_features = struct_stability.loc[robust_mask, feature_col].tolist()

        if len(robust_features) < 2:
            # No clustering needed
            for _, row in struct_stability.iterrows():
                row_copy = row.to_dict()
                row_copy["redundancy_cluster"] = 0
                row_copy["is_cluster_representative"] = row[feature_col] in robust_features
                row_copy["cluster_size"] = 1
                result_rows.append(row_copy)
            continue

        # Get feature values for this structure
        struct_values = feature_values_df[
            feature_values_df[structure_col] == structure
        ] if structure_col in feature_values_df.columns else feature_values_df

        # Build correlation matrix for robust features
        available_features = [f for f in robust_features if f in struct_values.columns]
        if len(available_features) < 2:
            for _, row in struct_stability.iterrows():
                row_copy = row.to_dict()
                row_copy["redundancy_cluster"] = 0
                row_copy["is_cluster_representative"] = row[feature_col] in robust_features
                row_copy["cluster_size"] = 1
                result_rows.append(row_copy)
            continue

        corr_matrix = struct_values[available_features].corr(method="spearman").abs()

        # Convert to distance matrix and cluster
        distance_matrix = 1.0 - corr_matrix.values
        np.fill_diagonal(distance_matrix, 0)
        # Ensure symmetry and non-negativity
        distance_matrix = np.maximum(distance_matrix, 0)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2

        try:
            condensed = squareform(distance_matrix)
            Z = linkage(condensed, method="complete")
            clusters = fcluster(Z, t=1.0 - correlation_threshold, criterion="distance")
        except Exception as e:
            logger.warning("Clustering failed for %s: %s", structure, e)
            clusters = np.arange(len(available_features))

        # Map features to clusters
        feature_to_cluster = dict(zip(available_features, clusters))

        # Select representatives (highest ICC per cluster)
        icc_lookup = dict(
            zip(
                struct_stability[feature_col],
                struct_stability[icc_col],
            )
        )

        cluster_representatives = {}
        cluster_sizes = {}
        for cluster_id in set(clusters):
            cluster_features = [f for f, c in feature_to_cluster.items() if c == cluster_id]
            cluster_sizes[cluster_id] = len(cluster_features)
            # Pick feature with highest ICC
            best_feature = max(cluster_features, key=lambda f: icc_lookup.get(f, -1))
            cluster_representatives[cluster_id] = best_feature

        # Build output
        for _, row in struct_stability.iterrows():
            row_copy = row.to_dict()
            fname = row[feature_col]
            if fname in feature_to_cluster:
                cid = feature_to_cluster[fname]
                row_copy["redundancy_cluster"] = int(cid)
                row_copy["is_cluster_representative"] = (cluster_representatives.get(cid) == fname)
                row_copy["cluster_size"] = cluster_sizes.get(cid, 1)
            else:
                # Non-robust feature
                row_copy["redundancy_cluster"] = -1
                row_copy["is_cluster_representative"] = False
                row_copy["cluster_size"] = 0
            result_rows.append(row_copy)

    return pd.DataFrame(result_rows)


# ============================================================================
# Determinants of Feature Instability (v1.2 — exploratory mixed-effects)
# ============================================================================

def model_instability_determinants(
    stability_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    icc_col: str = "icc",
    structure_col: str = "structure",
    feature_col: str = "feature_name",
) -> Dict[str, Any]:
    """
    Fit exploratory mixed-effects model to identify determinants of feature instability.

    Model: ICC ~ ROI_volume + surface_volume_ratio + manufacturer + slice_thickness + feature_family + (1|patient)

    This answers: "which ROIs are radiomics-ready on planning CT, and which are not?"
    Bounded to pre-specified predictors only (no data-driven variable selection).

    Args:
        stability_df: Output from summarize_feature_stability() with per-(structure, feature) ICC.
            Must contain 'icc', 'structure', 'feature_name' columns.
        metadata_df: Patient/acquisition metadata with columns:
            - 'patient_id': patient identifier
            - 'manufacturer' (optional): CT manufacturer
            - 'slice_thickness' (optional): slice thickness in mm
            - 'pixel_spacing' (optional): pixel spacing in mm
            - 'roi_volume_cc' (optional): ROI volume in cc
            - 'surface_volume_ratio' (optional): surface/volume ratio
        icc_col: Column name for ICC values.
        structure_col: Column name for structure names.
        feature_col: Column name for feature names.

    Returns:
        Dictionary with:
        - 'model_summary': string summary of the mixed-effects model
        - 'fixed_effects': DataFrame of fixed effect coefficients
        - 'significant_predictors': list of significant predictors (p < 0.05)
        - 'radiomics_ready_rois': list of ROIs with mean ICC > 0.75
        - 'high_risk_rois': list of ROIs with mean ICC < 0.50
        - 'feature_family_effects': DataFrame of per-family mean ICC
    """
    results: Dict[str, Any] = {}

    # --- Feature family extraction ---
    def _extract_family(feature_name: str) -> str:
        """Extract feature family from PyRadiomics feature name."""
        parts = feature_name.split("_")
        if len(parts) >= 2:
            # e.g., "original_glcm_Autocorrelation" → "glcm"
            # or "wavelet-LLH_firstorder_Mean" → "firstorder"
            for i, part in enumerate(parts):
                if part.lower() in (
                    "shape", "shape2d", "firstorder",
                    "glcm", "glrlm", "glszm", "gldm", "ngtdm",
                ):
                    return part.lower()
        return "unknown"

    df = stability_df.copy()
    df["feature_family"] = df[feature_col].apply(_extract_family)

    # --- Per-structure summary ---
    structure_icc = df.groupby(structure_col)[icc_col].agg(["mean", "median", "std", "count"])
    structure_icc.columns = ["mean_icc", "median_icc", "std_icc", "n_features"]
    results["structure_summary"] = structure_icc.reset_index()

    results["radiomics_ready_rois"] = structure_icc[
        structure_icc["mean_icc"] >= 0.75
    ].index.tolist()

    results["high_risk_rois"] = structure_icc[
        structure_icc["mean_icc"] < 0.50
    ].index.tolist()

    # --- Per-feature-family summary ---
    family_icc = df.groupby("feature_family")[icc_col].agg(["mean", "median", "std", "count"])
    family_icc.columns = ["mean_icc", "median_icc", "std_icc", "n_features"]
    results["feature_family_effects"] = family_icc.reset_index()

    # --- Mixed-effects model (if statsmodels available) ---
    try:
        import statsmodels.formula.api as smf

        # Merge with metadata if available
        if metadata_df is not None and not metadata_df.empty:
            # Try to merge on patient_id if both have it
            if "patient_id" in df.columns and "patient_id" in metadata_df.columns:
                df = df.merge(metadata_df, on="patient_id", how="left", suffixes=("", "_meta"))

        # Build formula from available predictors
        predictors = []
        for col in ["feature_family", structure_col]:
            if col in df.columns and df[col].nunique() > 1:
                predictors.append(f"C({col})")
        for col in ["manufacturer"]:
            if col in df.columns and df[col].nunique() > 1:
                predictors.append(f"C({col})")
        for col in ["slice_thickness", "pixel_spacing", "roi_volume_cc", "surface_volume_ratio"]:
            if col in df.columns and df[col].notna().sum() > 10:
                predictors.append(col)

        if predictors and "patient_id" in df.columns:
            formula = f"{icc_col} ~ " + " + ".join(predictors)
            # Drop NaN ICC values
            df_model = df.dropna(subset=[icc_col])

            if df_model["patient_id"].nunique() > 2:
                try:
                    model = smf.mixedlm(
                        formula, df_model, groups=df_model["patient_id"]
                    )
                    fit = model.fit(reml=True)
                    results["model_summary"] = str(fit.summary())
                    results["fixed_effects"] = fit.summary().tables[1] if hasattr(fit.summary(), "tables") else str(fit.params)
                    results["significant_predictors"] = [
                        name for name, pval in fit.pvalues.items()
                        if pval < 0.05 and name != "Intercept"
                    ]
                except Exception as e:
                    logger.warning("Mixed-effects model fitting failed: %s", e)
                    results["model_summary"] = f"Model fitting failed: {e}"
                    results["significant_predictors"] = []
            else:
                results["model_summary"] = "Insufficient patient groups for mixed-effects model"
                results["significant_predictors"] = []
        else:
            results["model_summary"] = "Insufficient predictors for mixed-effects model"
            results["significant_predictors"] = []

    except ImportError:
        logger.info("statsmodels not available; skipping mixed-effects model")
        results["model_summary"] = "statsmodels not installed"
        results["significant_predictors"] = []

    return results


# ============================================================================
# Main Workflow Functions
# ============================================================================

def robustness_for_course(
    config: PipelineConfig,
    rob_config: RobustnessConfig,
    course_dir: Path,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Run radiomics robustness analysis for a single course.

    Collects masks from multiple sources:
    - TotalSegmentator (RS_auto.dcm)
    - Custom structures (RS_custom.dcm)
    - Custom models (Segmentation_{model_name}/rtstruct.dcm)

    Args:
        config: Pipeline configuration
        rob_config: Robustness configuration
        course_dir: Course directory path

    Returns:
        Path to output parquet file or None
    """
    if not rob_config.enabled:
        logger.info("Radiomics robustness disabled; skipping")
        return None

    if "segmentation_perturbation" not in rob_config.modes:
        logger.info("Segmentation perturbation mode not enabled; skipping")
        return None

    logger.info("Running radiomics robustness analysis for %s", course_dir.name)

    course_dirs = build_course_dirs(course_dir)
    if output_path is None:
        output_path = course_dir / "radiomics_robustness_ct.parquet"

    # Load CT image
    from .radiomics import _load_series_image
    ct_image = _load_series_image(course_dirs.dicom_ct)
    if ct_image is None:
        logger.warning("No CT image found for robustness analysis in %s", course_dir)
        return None

    # ========================================================================
    # Collect masks from all segmentation sources
    # ========================================================================
    from .radiomics import _rtstruct_masks
    from .custom_models import list_custom_model_outputs
    from fnmatch import fnmatch

    # Dictionary: {(roi_name, source): mask_array}
    all_masks: Dict[Tuple[str, str], np.ndarray] = {}

    # 1. TotalSegmentator (RS_auto.dcm)
    rs_auto = course_dir / "RS_auto.dcm"
    if rs_auto.exists():
        ts_masks = _rtstruct_masks(course_dirs.dicom_ct, rs_auto)
        for roi_name, mask_array in ts_masks.items():
            all_masks[(roi_name, "AutoRTS_total")] = mask_array
        logger.info("Loaded %d structures from TotalSegmentator", len(ts_masks))
    else:
        logger.info("RS_auto.dcm not found")

    # 2. Custom structures (RS_custom.dcm)
    rs_custom = course_dir / "RS_custom.dcm"
    if rs_custom.exists():
        custom_masks = _rtstruct_masks(course_dirs.dicom_ct, rs_custom)
        for roi_name, mask_array in custom_masks.items():
            all_masks[(roi_name, "Custom")] = mask_array
        logger.info("Loaded %d structures from custom structures", len(custom_masks))
    else:
        logger.debug("RS_custom.dcm not found")

    # 3. Custom models
    try:
        for model_name, model_dir in list_custom_model_outputs(course_dir):
            rs_model = model_dir / "rtstruct.dcm"
            if rs_model.exists():
                model_masks = _rtstruct_masks(course_dirs.dicom_ct, rs_model)
                source_label = f"CustomModel:{model_name}"
                for roi_name, mask_array in model_masks.items():
                    all_masks[(roi_name, source_label)] = mask_array
                logger.info("Loaded %d structures from custom model '%s'", len(model_masks), model_name)
    except Exception as e:
        logger.debug("Failed to load custom model outputs: %s", e)

    if not all_masks:
        logger.warning("No masks found from any source for %s", course_dir)
        return None

    logger.info("Total masks collected: %d from %d unique sources",
                len(all_masks), len(set(source for _, source in all_masks.keys())))

    # ========================================================================
    # Filter structures based on configuration patterns
    # ========================================================================
    selected_structures: List[Tuple[str, str]] = []  # [(roi_name, source), ...]

    for (roi_name, source), mask_array in all_masks.items():
        for pattern in rob_config.perturbation.apply_to_structures:
            if fnmatch(roi_name.upper(), pattern.upper()):
                selected_structures.append((roi_name, source))
                break

    if not selected_structures:
        logger.info("No structures matched robustness patterns; skipping %s", course_dir)
        return None

    logger.info("Selected %d structure(s) for robustness analysis:", len(selected_structures))
    for roi_name, source in selected_structures:
        logger.info("  - %s (from %s)", roi_name, source)

    # ========================================================================
    # Extract features for each structure with perturbations
    # ========================================================================
    from .radiomics import _mask_from_array_like
    
    # Import parallel processing helpers
    try:
        from .radiomics_parallel import (
            _prepare_radiomics_task,
            _isolated_radiomics_extraction_with_retry,
            _calculate_optimal_workers,
            _apply_thread_limit,
            _resolve_thread_limit
        )
        has_parallel = True
    except ImportError:
        logger.warning("Parallel radiomics helpers not available; falling back to sequential processing")
        has_parallel = False

    disable_parallel = os.environ.get("RTPIPELINE_DISABLE_PARALLEL_RADIOMICS", "0").strip().lower()
    if disable_parallel not in {"", "0", "false", "no"}:
        logger.info("Parallel radiomics disabled via RTPIPELINE_DISABLE_PARALLEL_RADIOMICS=%s", disable_parallel)
        has_parallel = False
    all_features = []
    
    # Prepare tasks for parallel execution
    tasks = []
    temp_dir = None
    
    if has_parallel:
        temp_dir = Path(tempfile.mkdtemp(prefix='rob_radiomics_'))
        # Apply thread limit to main process
        _apply_thread_limit(_resolve_thread_limit(getattr(config, 'radiomics_thread_limit', None)))
    
    try:
        for roi_name, source in selected_structures:
            mask_array = all_masks[(roi_name, source)]
            mask_img = _mask_from_array_like(ct_image, mask_array)

            # Check if NTCV mode is enabled (any perturbation beyond volume is configured)
            use_ntcv = (
                rob_config.perturbation.max_translation_mm > 0 or
                rob_config.perturbation.n_random_contour_realizations > 0 or
                any(n > 0 for n in rob_config.perturbation.noise_levels)
            )

            if use_ntcv:
                # Generate NTCV perturbation chain
                perturbed_masks, perturbed_images = generate_ntcv_perturbations(
                    mask_img,
                    ct_image,
                    rob_config.perturbation,
                    roi_name,
                )
            else:
                # Legacy mode: volume-only perturbations
                # C19 fix: honor intensity config — use large_volume_changes for aggressive
                intensity = rob_config.perturbation.intensity
                if intensity == "aggressive":
                    volume_changes = rob_config.perturbation.large_volume_changes
                else:
                    volume_changes = rob_config.perturbation.small_volume_changes
                perturbed_masks = generate_perturbed_masks(
                    mask_img,
                    volume_changes,
                    roi_name,
                )
                perturbed_images = None

            if len(perturbed_masks) < 2:
                logger.warning("Insufficient perturbations for %s (%s); skipping", roi_name, source)
                continue

            if has_parallel:
                # Prepare parallel tasks
                for pert_id, mask in perturbed_masks.items():
                    # Use perturbed image if available
                    current_image = perturbed_images.get(pert_id, ct_image) if perturbed_images else ct_image
                    
                    try:
                        # We treat each perturbation as a "structure" for the parallel worker
                        task_file, task_params = _prepare_radiomics_task(
                            current_image, mask, config, source, roi_name, course_dir, temp_dir, False
                        )
                        # Add perturbation-specific metadata to extra_metadata
                        task_params['extra_metadata'] = {'perturbation_id': pert_id}
                        tasks.append((task_file, task_params))
                    except Exception as e:
                        logger.warning("Failed to prepare task for %s/%s/%s: %s", source, roi_name, pert_id, e)
            else:
                # Sequential fallback
                features_df = extract_features_for_masks(
                    ct_image,
                    perturbed_masks,
                    config,
                    modality="CT",
                    structure_name=roi_name,
                    patient_id=course_dir.parent.name,
                    course_id=course_dir.name,
                    perturbed_images=perturbed_images,
                )
                if not features_df.empty:
                    features_df["segmentation_source"] = source
                    all_features.append(features_df)

        # Execute parallel tasks
        if has_parallel and tasks:
            max_workers = _calculate_optimal_workers()
            # Respect global worker limit if set
            try:
                config_workers = int(getattr(config, 'effective_workers')())
                max_workers = min(max_workers, config_workers)
            except Exception:
                pass
            
            max_workers = max(1, min(max_workers, len(tasks)))
            logger.info("Processing %d robustness perturbations with %d workers", len(tasks), max_workers)

            ctx = get_context('spawn')

            # Timeout configuration for watchdog
            course_timeout = int(os.environ.get("RTPIPELINE_ROBUSTNESS_COURSE_TIMEOUT", "3600"))  # 1 hour default
            progress_timeout = int(os.environ.get("RTPIPELINE_ROBUSTNESS_PROGRESS_TIMEOUT", "300"))  # 5 min default

            with ctx.Pool(max_workers) as pool:
                completed_count = 0
                total_count = len(tasks)
                start_time = time.time()
                last_progress_time = time.time()
                timed_out = False

                # Use imap_unordered with watchdog timeout
                results_iter = pool.imap_unordered(_isolated_radiomics_extraction_with_retry, tasks)

                while completed_count < total_count and not timed_out:
                    try:
                        # Poll for results with a short timeout
                        result = results_iter.next(timeout=10)  # 10 second poll interval
                        completed_count += 1
                        last_progress_time = time.time()

                        if completed_count % 10 == 0 or completed_count == total_count:
                            elapsed = time.time() - start_time
                            rate = completed_count / elapsed if elapsed > 0 else 0
                            eta = (total_count - completed_count) / rate if rate > 0 else 0
                            logger.info("Robustness progress: %d/%d (%.1f%%), ETA: %.1fs",
                                       completed_count, total_count,
                                       100 * completed_count / total_count, eta)
                    except StopIteration:
                        # All results processed
                        break
                    except TimeoutError:
                        # Check watchdog conditions
                        elapsed = time.time() - start_time
                        no_progress_time = time.time() - last_progress_time

                        if elapsed > course_timeout:
                            logger.error(
                                "Robustness analysis exceeded course timeout (%ds), terminating pool",
                                course_timeout
                            )
                            pool.terminate()
                            timed_out = True
                        elif no_progress_time > progress_timeout:
                            logger.error(
                                "No progress for %ds (threshold: %ds), likely hung worker - terminating pool",
                                int(no_progress_time), progress_timeout
                            )
                            pool.terminate()
                            timed_out = True
                        else:
                            # Continue waiting
                            continue
                    except Exception as iter_err:
                        logger.warning("Error fetching result: %s", iter_err)
                        continue

                    # Process result if we got one

                    if result:
                        # Convert wide result dict (from worker) to long format DataFrame rows
                        meta_keys = {
                            'modality', 'segmentation_source', 'roi_name', 'roi_original_name',
                            'course_dir', 'patient_id', 'course_id', 'structure_cropped',
                            'perturbation_id'
                        }
                        
                        # Extract metadata
                        metadata = {k: result[k] for k in meta_keys if k in result}
                        
                        # Convert features
                        rows = []
                        for k, v in result.items():
                            if k in meta_keys: continue
                            # Skip diagnostic info if present
                            if k.startswith('diagnostics_'): continue 
                            
                            if isinstance(v, (int, float, np.floating, np.integer)):
                                row = metadata.copy()
                                row['feature_name'] = str(k)
                                row['value'] = float(v)
                                rows.append(row)
                        
                        if rows:
                            all_features.append(pd.DataFrame(rows))
                        else:
                            logger.debug(
                                "Robustness worker returned metadata-only result for %s/%s (perturbation %s)",
                                metadata.get('segmentation_source'),
                                metadata.get('roi_name'),
                                metadata.get('perturbation_id'),
                            )

    except Exception as e:
        logger.error("Robustness analysis failed: %s", e)
        import traceback
        logger.error(traceback.format_exc())
    finally:
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.debug("Failed to clean up temp dir %s: %s", temp_dir, e)

    if not all_features:
        logger.warning("No features extracted for robustness analysis in %s", course_dir)
        return None

    # Combine all features
    combined_df = pd.concat(all_features, ignore_index=True)

    # Save raw feature values for aggregation stage
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_parquet(output_path, index=False)
        logger.info(
            "Saved robustness feature values to %s (%d rows, %d unique perturbations)",
            output_path,
            len(combined_df),
            combined_df["perturbation_id"].nunique(),
        )
        return output_path
    except Exception as e:
        logger.warning("Failed to save robustness results: %s", e)
        return None


def aggregate_robustness_results(
    input_parquets: List[Path],
    output_excel: Path,
    rob_config: RobustnessConfig,
) -> None:
    """
    Aggregate per-course robustness results into cohort-level summary.

    Args:
        input_parquets: List of per-course parquet files
        output_excel: Output Excel file path
        rob_config: Robustness configuration
    """
    logger.info("Aggregating robustness results from %d courses", len(input_parquets))

    all_dfs = []
    for parquet_path in input_parquets:
        if not parquet_path.exists():
            continue
        try:
            df = pd.read_parquet(parquet_path)
            all_dfs.append(df)
        except Exception as e:
            logger.warning("Failed to read %s: %s", parquet_path, e)

    if not all_dfs:
        logger.warning("No robustness results to aggregate")
        # Create empty Excel
        output_excel.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_excel(output_excel, index=False)
        return

    combined_raw = pd.concat(all_dfs, ignore_index=True)

    # Normalize column names
    if "roi_name" in combined_raw.columns and "structure" not in combined_raw.columns:
        combined_raw.rename(columns={"roi_name": "structure"}, inplace=True)

    per_structure_summary = summarize_feature_stability(combined_raw, rob_config)
    global_summary = summarize_feature_stability(combined_raw, rob_config, group_columns=["feature_name"])

    per_source_summary: Optional[pd.DataFrame] = None
    if "segmentation_source" in combined_raw.columns:
        per_source_summary = summarize_feature_stability(
            combined_raw,
            rob_config,
            group_columns=["segmentation_source", "feature_name"],
        )

    if global_summary.empty:
        robust_features = pd.DataFrame(columns=global_summary.columns)
        acceptable_features = pd.DataFrame(columns=global_summary.columns)
    else:
        robust_features = global_summary[global_summary["robustness_label"] == "robust"]
        acceptable_features = global_summary[global_summary["pass_seg_perturb"]]

    if per_source_summary is not None and not per_source_summary.empty:
        robust_per_source = per_source_summary[per_source_summary["robustness_label"] == "robust"]
    else:
        robust_per_source = None

    output_excel.parent.mkdir(parents=True, exist_ok=True)

    # Save raw values to parquet (no row limit, unlike Excel's 1,048,576)
    raw_parquet_path = output_excel.parent / (output_excel.stem + "_raw_values.parquet")
    try:
        combined_raw.to_parquet(raw_parquet_path, index=False)
        logger.info(
            "Saved raw robustness values to %s (%d rows)",
            raw_parquet_path,
            len(combined_raw),
        )
    except Exception as e:
        logger.error("Failed to write raw values parquet: %s", e)
        raise

    try:
        with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
            global_summary.to_excel(writer, sheet_name="global_summary", index=False)

            if per_source_summary is not None:
                per_source_summary.to_excel(writer, sheet_name="per_source_summary", index=False)

            per_structure_summary.to_excel(writer, sheet_name="per_structure_source", index=False)

            robust_features.to_excel(writer, sheet_name="robust_features", index=False)
            acceptable_features.to_excel(writer, sheet_name="acceptable_features", index=False)

            if robust_per_source is not None:
                robust_per_source.to_excel(writer, sheet_name="robust_features_per_source", index=False)

            # Note: Raw values saved to parquet file (see log above)
            # Excel has 1,048,576 row limit; large datasets exceed this

        logger.info(
            "Saved aggregated robustness results to %s (features=%d, structures=%d)",
            output_excel,
            len(global_summary),
            len(per_structure_summary),
        )
        if not global_summary.empty:
            logger.info(
                "Global summary: %d robust, %d acceptable",
                len(robust_features),
                len(acceptable_features),
            )
        if per_source_summary is not None:
            logger.info(
                "Per-source summary: %d combinations",
                len(per_source_summary),
            )
    except Exception as e:
        logger.error("Failed to write aggregated results: %s", e)
        raise
