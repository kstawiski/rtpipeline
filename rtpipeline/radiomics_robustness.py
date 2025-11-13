"""
Radiomics robustness analysis module.

Implements IBSI-compliant feature stability assessment via:
- Mask perturbations (erosion/dilation with volume adaptation)
- ICC (Intraclass Correlation Coefficient) computation
- CoV (Coefficient of Variation) and QCD metrics
- Multi-axis robustness evaluation (segmentation perturbation, segmentation method, scan-rescan)

References:
- Zwanenburg et al. 2019 (Sci Rep): image perturbation chains with ICC
- Lo Iacono et al. 2024 (SpringerLink): volume adaptation for stability
- Poirot et al. 2022 (Sci Rep): multi-method ICC with Pingouin
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
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
    """Configuration for mask perturbation (volume adaptation)."""
    apply_to_structures: List[str] = field(default_factory=lambda: ["GTV", "CTV", "PTV", "BLADDER", "RECTUM"])
    small_volume_changes: List[float] = field(default_factory=lambda: [-0.15, 0.0, 0.15])
    large_volume_changes: List[float] = field(default_factory=lambda: [-0.30, 0.0, 0.30])
    n_random_contour_realizations: int = 0
    max_translation_mm: float = 0.0


@dataclass
class ICCConfig:
    """Configuration for ICC computation."""
    implementation: Literal["pingouin", "manual"] = "pingouin"
    icc_type: Literal["ICC1", "ICC2", "ICC3"] = "ICC2"
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
    enabled: bool = False
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
        )

        # Metrics config
        metrics_data = data.get("metrics", {})
        icc_data = metrics_data.get("icc", {})
        icc_config = ICCConfig(
            implementation=icc_data.get("implementation", "pingouin"),
            icc_type=icc_data.get("icc_type", "ICC2"),
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


# ============================================================================
# Mask Perturbation Functions
# ============================================================================

def _get_ball_radius_vox(spacing_mm: Tuple[float, float, float], target_thickness_mm: float) -> int:
    """Compute ball kernel radius in voxels from physical thickness."""
    avg_spacing = float(np.mean(spacing_mm))
    return max(1, int(round(target_thickness_mm / avg_spacing)))


def volume_adapt_mask(mask: sitk.Image, tau: float, max_iterations: int = 20) -> Optional[sitk.Image]:
    """
    Apply IBSI-style volume adaptation to mask via iterative erosion/dilation.

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

    if original_volume < 10:
        logger.debug("Mask too small for volume adaptation (volume=%d)", original_volume)
        return None

    target_volume = int(original_volume * (1.0 + tau))
    spacing = mask.GetSpacing()

    # Determine operation (erosion or dilation)
    is_erosion = tau < 0
    operation = sitk.BinaryErode if is_erosion else sitk.BinaryDilate

    # Iterative approach
    current_mask = mask
    best_mask = mask
    best_diff = abs(original_volume - target_volume)

    for radius in range(1, max_iterations + 1):
        try:
            perturbed = operation(current_mask, [radius] * 3, sitk.sitkBall)
            perturbed_arr = sitk.GetArrayFromImage(perturbed).astype(bool)
            perturbed_volume = perturbed_arr.sum()

            diff = abs(perturbed_volume - target_volume)
            if diff < best_diff:
                best_diff = diff
                best_mask = perturbed

            # Stop if we reached target or overshot
            if is_erosion and perturbed_volume <= target_volume:
                break
            elif not is_erosion and perturbed_volume >= target_volume:
                break
        except Exception as e:
            logger.debug("Volume adaptation failed at radius %d: %s", radius, e)
            break

    # Check if result is reasonable
    best_arr = sitk.GetArrayFromImage(best_mask).astype(bool)
    best_volume = best_arr.sum()

    if best_volume < 5:
        logger.debug("Perturbed mask too small (volume=%d)", best_volume)
        return None

    achieved_tau = (best_volume - original_volume) / original_volume
    logger.debug("Volume adaptation: target τ=%.3f, achieved τ=%.3f (volume %d→%d)",
                 tau, achieved_tau, original_volume, best_volume)

    return best_mask


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
) -> pd.DataFrame:
    """
    Extract radiomics features for multiple mask variants.

    Args:
        image: CT/MR image
        masks: Dictionary of {perturbation_id: mask}
        config: Pipeline configuration
        modality: "CT" or "MR"
        structure_name: ROI name
        patient_id: Patient identifier
        course_id: Course identifier

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

            # Execute PyRadiomics
            result = ext.execute(image, mask)

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
        icc_res = pg.intraclass_corr(
            data=df,
            targets="subject",
            raters="rater",
            ratings="value"
        )

        # Select appropriate ICC type
        row = icc_res.loc[icc_res["Type"] == icc_config.icc_type].iloc[0]

        result = {"icc": float(row["ICC"])}

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
) -> pd.DataFrame:
    """
    Compute per-feature robustness metrics (ICC, CoV, QCD) and classify robustness.

    Args:
        df_long: Tidy DataFrame with columns [patient_id, course_id, structure, perturbation_id, feature_name, value]
        config: Robustness configuration

    Returns:
        DataFrame with one row per (structure, feature_name) containing metrics and robustness label
    """
    rows = []

    for (structure, feature_name), group in df_long.groupby(["structure", "feature_name"]):
        values = group["value"].to_numpy(dtype=float)

        # Skip if insufficient data
        if len(values) < 2:
            continue

        # Compute CoV
        cov = compute_cov(values) if config.metrics.cov_enabled else np.nan

        # Compute QCD
        qcd = compute_qcd(values) if config.metrics.qcd_enabled else np.nan

        # Compute ICC
        icc_df = group[["patient_id", "perturbation_id", "value"]].copy()
        icc_df.columns = ["subject", "rater", "value"]

        # Create unique subject ID (for multi-patient analysis, use patient_id directly)
        icc_df["subject"] = icc_df["subject"].astype(str) + "_" + structure

        icc_info = compute_icc_pingouin(icc_df, config.metrics.icc)
        icc = icc_info["icc"]
        icc_ci_low = icc_info.get("icc_ci95_low", np.nan)
        icc_ci_high = icc_info.get("icc_ci95_high", np.nan)

        # Classification logic
        # Use lower CI bound if available (conservative), else use point estimate
        icc_for_threshold = icc_ci_low if not np.isnan(icc_ci_low) else icc

        robust_icc = icc_for_threshold >= config.thresholds.icc_robust
        acceptable_icc = icc_for_threshold >= config.thresholds.icc_acceptable

        robust_cov = cov <= config.thresholds.cov_robust_pct if not np.isnan(cov) else False
        acceptable_cov = cov <= config.thresholds.cov_acceptable_pct if not np.isnan(cov) else False

        # Combined robustness label
        if robust_icc and robust_cov:
            robustness_label = "robust"
        elif acceptable_icc and acceptable_cov:
            robustness_label = "acceptable"
        else:
            robustness_label = "poor"

        rows.append({
            "structure": structure,
            "feature_name": feature_name,
            "n_perturbations": len(values),
            "icc": icc,
            "icc_ci95_low": icc_ci_low,
            "icc_ci95_high": icc_ci_high,
            "cov_pct": cov,
            "qcd": qcd,
            "robustness_label": robustness_label,
            "pass_seg_perturb": robustness_label in ["robust", "acceptable"],
        })

    return pd.DataFrame(rows)


# ============================================================================
# Main Workflow Functions
# ============================================================================

def robustness_for_course(
    config: PipelineConfig,
    rob_config: RobustnessConfig,
    course_dir: Path,
) -> Optional[Path]:
    """
    Run radiomics robustness analysis for a single course.

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
    output_path = course_dir / "radiomics_robustness_ct.parquet"

    # Load CT image
    from .radiomics import _load_series_image
    ct_image = _load_series_image(course_dirs.dicom_ct)
    if ct_image is None:
        logger.warning("No CT image found for robustness analysis in %s", course_dir)
        return None

    # Load masks from RS_auto.dcm (TotalSegmentator)
    from .radiomics import _rtstruct_masks
    rs_auto = course_dir / "RS_auto.dcm"
    if not rs_auto.exists():
        logger.warning("RS_auto.dcm not found; skipping robustness analysis for %s", course_dir)
        return None

    masks = _rtstruct_masks(course_dirs.dicom_ct, rs_auto)
    if not masks:
        logger.warning("No masks found in RS_auto.dcm for %s", course_dir)
        return None

    # Filter structures based on configuration
    from fnmatch import fnmatch
    selected_structures = []
    for roi_name in masks.keys():
        for pattern in rob_config.perturbation.apply_to_structures:
            if fnmatch(roi_name.upper(), pattern.upper()):
                selected_structures.append(roi_name)
                break

    if not selected_structures:
        logger.info("No structures matched robustness patterns; skipping %s", course_dir)
        return None

    logger.info("Selected %d structures for robustness analysis: %s",
                len(selected_structures), ", ".join(selected_structures))

    # Convert masks to SimpleITK images
    from .radiomics import _mask_from_array_like
    all_features = []

    for roi_name in selected_structures:
        mask_array = masks[roi_name]
        mask_img = _mask_from_array_like(ct_image, mask_array)

        # Generate perturbed masks
        perturbed_masks = generate_perturbed_masks(
            mask_img,
            rob_config.perturbation.small_volume_changes,
            roi_name,
        )

        if len(perturbed_masks) < 2:
            logger.warning("Insufficient perturbations for %s; skipping", roi_name)
            continue

        # Extract features for all perturbations
        features_df = extract_features_for_masks(
            ct_image,
            perturbed_masks,
            config,
            modality="CT",
            structure_name=roi_name,
            patient_id=course_dir.parent.name,
            course_id=course_dir.name,
        )

        if not features_df.empty:
            all_features.append(features_df)

    if not all_features:
        logger.warning("No features extracted for robustness analysis in %s", course_dir)
        return None

    # Combine all features
    combined_df = pd.concat(all_features, ignore_index=True)

    # Compute robustness metrics
    summary_df = summarize_feature_stability(combined_df, rob_config)

    # Save results
    try:
        summary_df.to_parquet(output_path, index=False)
        logger.info("Saved robustness results to %s (%d features)", output_path, len(summary_df))
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
        pd.DataFrame().to_excel(output_excel, index=False)
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    # Compute global statistics (average across all courses/structures)
    global_summary = combined.groupby("feature_name").agg({
        "icc": "mean",
        "icc_ci95_low": "mean",
        "icc_ci95_high": "mean",
        "cov_pct": "mean",
        "qcd": "mean",
        "n_perturbations": "mean",
    }).reset_index()

    # Determine global robustness label
    def _classify_global(row):
        icc = row["icc"]
        cov = row["cov_pct"]

        robust_icc = icc >= rob_config.thresholds.icc_robust
        acceptable_icc = icc >= rob_config.thresholds.icc_acceptable
        robust_cov = cov <= rob_config.thresholds.cov_robust_pct if not np.isnan(cov) else False
        acceptable_cov = cov <= rob_config.thresholds.cov_acceptable_pct if not np.isnan(cov) else False

        if robust_icc and robust_cov:
            return "robust"
        elif acceptable_icc and acceptable_cov:
            return "acceptable"
        else:
            return "poor"

    global_summary["robustness_label"] = global_summary.apply(_classify_global, axis=1)
    global_summary["pass_seg_perturb"] = global_summary["robustness_label"].isin(["robust", "acceptable"])

    # Write to Excel with multiple sheets
    try:
        with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
            # Sheet 1: Global feature summary (averaged across all structures/courses)
            global_summary.to_excel(writer, sheet_name="global_summary", index=False)

            # Sheet 2: Per-structure summary
            combined.to_excel(writer, sheet_name="per_structure", index=False)

            # Sheet 3: Robust features only (global)
            robust_features = global_summary[global_summary["robustness_label"] == "robust"]
            robust_features.to_excel(writer, sheet_name="robust_features", index=False)

            # Sheet 4: Acceptable or better features
            acceptable_features = global_summary[global_summary["pass_seg_perturb"]]
            acceptable_features.to_excel(writer, sheet_name="acceptable_features", index=False)

        logger.info("Saved aggregated robustness results to %s", output_excel)
        logger.info("Global summary: %d total features, %d robust, %d acceptable",
                    len(global_summary),
                    len(robust_features),
                    len(acceptable_features))
    except Exception as e:
        logger.error("Failed to write aggregated results: %s", e)
        raise
