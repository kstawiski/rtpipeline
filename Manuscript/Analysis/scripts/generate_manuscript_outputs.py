#!/usr/bin/env python3
"""
RTpipeline Manuscript - Analysis Script
Generates tables and figures according to Manuscript/Plan specifications.

This script implements the statistical analysis plan from:
  - Manuscript/Plan/TABLES_PLAN.md
  - Manuscript/Plan/FIGURES_PLAN.md
  - Manuscript/Plan/STATISTICAL_ANALYSIS_PLAN.md

Author: RTpipeline Manuscript Team
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Iterator

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pingouin as pg
from statsmodels.stats.multitest import multipletests

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Random seed for reproducibility (per Statistical Analysis Plan Section 11)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# =============================================================================
# Configuration
# =============================================================================

# Allow override via command-line argument or environment variable
# Usage: python generate_manuscript_outputs.py /path/to/Output
# Or: DATA_ROOT=/path/to/Output python generate_manuscript_outputs.py
if len(sys.argv) > 1:
    OUTPUT_DIR = Path(sys.argv[1])
elif os.environ.get("DATA_ROOT"):
    OUTPUT_DIR = Path(os.environ["DATA_ROOT"])
else:
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    OUTPUT_DIR = PROJECT_ROOT / "Output"

RESULTS_DIR = OUTPUT_DIR / "_RESULTS"
ANALYSIS_OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
TABLES_DIR = ANALYSIS_OUTPUT_DIR / "tables"
FIGURES_DIR = ANALYSIS_OUTPUT_DIR / "figures"

# Robustness thresholds (from plan)
ICC_ROBUST_THRESHOLD = 0.90
ICC_ACCEPTABLE_THRESHOLD = 0.75
COV_ROBUST_THRESHOLD = 10.0  # percent

# Feature class mappings
FEATURE_CLASS_MAP = {
    'shape': 'Shape',
    'firstorder': 'First-Order',
    'glcm': 'GLCM',
    'glrlm': 'GLRLM',
    'glszm': 'GLSZM',
    'gldm': 'GLDM',
    'ngtdm': 'NGTDM',
}

# Color scheme - colorblind-safe (Okabe-Ito palette for categorical)
# Based on consensus from gpt-5.1-codex and gemini-3-pro-preview
COLORS = {
    'processing': '#0072B2',   # Blue
    'outputs': '#009E73',      # Bluish green
    'robustness': '#E69F00',   # Orange
    'warning': '#D55E00',      # Vermilion
    'neutral': '#999999',      # Gray
    'robust': '#009E73',       # Bluish green (colorblind-safe)
    'acceptable': '#E69F00',   # Orange (colorblind-safe)
    'poor': '#CC79A7',         # Reddish purple (colorblind-safe, not red)
}

# Journal formatting constants (CMPB standards)
FIGURE_WIDTH_DOUBLE = 7.5      # inches (190mm double column)
FIGURE_WIDTH_SINGLE = 3.5      # inches (90mm single column)
FIGURE_DPI_PRINT = 300         # minimum for publication
FIGURE_DPI_SCREEN = 150        # for previews
FONT_SIZE_LABEL = 9            # pt, minimum for readability
FONT_SIZE_TITLE = 11           # pt
FONT_SIZE_TICK = 8             # pt
FONT_FAMILY = 'Arial'          # standard journal font

# Perturbation ID to human-readable label mapping
PERTURBATION_LABELS = {
    'tauminus0p15': 'Volume −15%',
    'tauplus0p00': 'Baseline',
    'tauplus0p15': 'Volume +15%',
    'baseline': 'Baseline',
    'noise_10': 'Noise σ=10 HU',
    'noise_20': 'Noise σ=20 HU',
    'translate_2mm': 'Shift ±2mm',
    'translate_4mm': 'Shift ±4mm',
}

# CoV cap for visualization (prevents outlier compression)
COV_DISPLAY_CAP = 100.0  # percent

# =============================================================================
# Structure Filtering and Name Normalization
# Based on consensus from GPT-5.1 and Gemini-3-Pro (both 9/10 confidence)
# =============================================================================

# PRIMARY structures for clinical analysis (N≥50 patients required)
# - Manual targets: CTV1, CTV2, GTVp (radiomics for outcome prediction)
# - AI OARs: urinary_bladder, colon, pelvic_bones (radiomics for toxicity)
PRIMARY_STRUCTURES = {
    # Manual target volumes
    'CTV1', 'CTV2', 'GTVp',
    # AI-segmented OARs (high clinical relevance)
    'urinary_bladder', 'colon', 'pelvic_bones',
}

# SECONDARY structures for technical validation (sanity checks)
# - Used to verify feature stability in consistent AI contours
SECONDARY_STRUCTURES = {
    'iliac_vess',  # Combined vessels - good for testing tubular structures
}

# Patterns to EXCLUDE (technical/derived, not biologically meaningful)
EXCLUDE_PATTERNS = [
    r'PTV\d?\s*-\s*PTV',      # Difference structures like "PTV2 - PTV1"
    r'ptv\d?\s*-\s*ptv',      # Lowercase variants
    r'_partial$',              # Partial structures
    r'__partial$',             # Double underscore partial
    r'_margin',                # Margin structures
    r'margin',                 # Any margin
    r'minus',                  # Subtraction structures
    r'ring',                   # Ring structures
    r'expand',                 # Expanded structures
    r'_mm$',                   # Morphological variants (e.g., _3mm)
    r'\d+mm$',                 # Numeric mm variants
    r'_NEW$',                  # Modified/new versions
    r'PACHWINA',               # Polish naming variants (groin regions)
]

def normalize_structure_name(name: str) -> str:
    """
    Normalize structure names to canonical form.
    Based on consensus: merge naming variants to boost N above 50 threshold.

    Rules (per GPT-5.1 and Gemini consensus):
    - CTV1 / "CTV 1" / "ctv1" → CTV1
    - CTV2 / "CTV 2" / "ctv2" → CTV2
    - GTVp / "GTVp MR" / "GTVp mr" / "GTVp CT" → GTVp (merge all primary tumor GTVs)
    - GTVn → GTVn (keep nodal GTV separate)
    - iliac_vess / iliac_vessels / iliac_artery_* / iliac_vena_* / iliac_vein_* → iliac_vess
    - pelvic_bones / pelvic_bones_3mm → pelvic_bones
    """
    import re

    if pd.isna(name):
        return name

    name = str(name).strip()

    # Fix common typos
    name = name.replace('illiac', 'iliac')
    name = name.replace('Illiac', 'iliac')

    # === Case normalization for target structures ===
    # "Colon" → "colon" (PRIMARY_STRUCTURES uses lowercase)
    if name.lower() == 'colon':
        return 'colon'
    if name.lower() == 'urinary_bladder':
        return 'urinary_bladder'

    # === CTV normalization ===
    # "CTV 1", "CTV1", "ctv1", "CTV-1", "CTV_1" → CTV1
    ctv_match = re.match(r'^[Cc][Tt][Vv]\s*[-_]?\s*(\d+)$', name)
    if ctv_match:
        return f"CTV{ctv_match.group(1)}"

    # === GTVp normalization (merge all primary tumor GTV variants) ===
    # "GTVp", "GTVp MR", "GTVp mr", "GTVp CT", "GTVp MRI", "GTVp1" → GTVp
    gtvp_match = re.match(r'^[Gg][Tt][Vv][Pp]\d*\s*(MR|mr|CT|ct|MRI|mri)?.*$', name, re.IGNORECASE)
    if gtvp_match:
        return "GTVp"

    # "GTV MR", "GTV_MR", "GTV mr" → GTVp (assuming these are primary tumor)
    gtvmr_match = re.match(r'^[Gg][Tt][Vv]\s*[-_]?\s*(MR|mr|MRI|mri|CT|ct).*$', name, re.IGNORECASE)
    if gtvmr_match:
        return "GTVp"

    # Plain "GTV" without qualifier → GTVp
    if re.match(r'^[Gg][Tt][Vv]$', name):
        return "GTVp"

    # === GTVn normalization (nodal GTV - keep separate) ===
    if re.match(r'^[Gg][Tt][Vv][Nn].*$', name, re.IGNORECASE):
        return "GTVn"

    # === PTV normalization (for optional secondary analysis) ===
    ptv_match = re.match(r'^[Pp][Tt][Vv]\s*[-_]?\s*(\d+)$', name)
    if ptv_match:
        return f"PTV{ptv_match.group(1)}"

    # === Iliac vessel normalization ===
    # Merge all iliac vessel variants: iliac_vess, iliac_vessels, iliac_artery_*, iliac_vena_*, iliac_vein_*
    if re.match(r'^iliac[_\s]*(vess(els?)?|arter(y|ies)|vein|vena)([_\s]*(left|right|l|r))?$', name, re.IGNORECASE):
        return "iliac_vess"

    # === Pelvic bones normalization (remove margin suffixes) ===
    pelvic_match = re.match(r'^pelvic[_\s]*bones?([_\s]*\d+mm)?$', name, re.IGNORECASE)
    if pelvic_match:
        return "pelvic_bones"

    return name


def is_target_structure(name: str) -> bool:
    """
    Check if structure should be included in analysis.
    Returns True for target structures, False for technical/excluded ones.

    Based on GPT-5.1 and Gemini-3-Pro consensus (both 9/10 confidence):
    - PRIMARY: CTV1, CTV2, GTVp, urinary_bladder, colon, pelvic_bones
    - SECONDARY: iliac_vess (for technical validation)

    Logic order:
    1. Normalize the name first (this handles variants like pelvic_bones_3mm -> pelvic_bones)
    2. If normalized name is in PRIMARY or SECONDARY, INCLUDE it (bypass exclusion patterns)
    3. Otherwise, check exclusion patterns on original name
    """
    import re

    if pd.isna(name):
        return False

    name = str(name).strip()

    # FIRST: Normalize the name (this converts pelvic_bones_3mm -> pelvic_bones, iliac_artery_* -> iliac_vess)
    normalized = normalize_structure_name(name)

    # If normalized name is a target structure, INCLUDE it (bypass exclusion patterns)
    # This allows pelvic_bones_3mm (normalized to pelvic_bones) to be included
    if normalized in PRIMARY_STRUCTURES or normalized in SECONDARY_STRUCTURES:
        # Still exclude __partial variants (they're truly different structures)
        if '__partial' in name.lower():
            return False
        return True

    # For non-target structures, apply exclusion patterns
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, name, re.IGNORECASE):
            return False

    return False


def filter_target_structures(df: pd.DataFrame, structure_col: str = 'structure') -> pd.DataFrame:
    """
    Filter DataFrame to include only target structures.
    Also normalizes structure names.
    """
    if df.empty or structure_col not in df.columns:
        return df

    # Normalize names
    df = df.copy()
    df[structure_col] = df[structure_col].apply(normalize_structure_name)

    # Filter to target structures
    mask = df[structure_col].apply(is_target_structure)
    filtered = df[mask].copy()

    print(f"  Filtered structures: {df[structure_col].nunique()} -> {filtered[structure_col].nunique()}")
    if not filtered.empty:
        print(f"  Included: {sorted(filtered[structure_col].unique())}")

    return filtered


def setup_directories():
    """Create output directories if they don't exist."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directories created: {ANALYSIS_OUTPUT_DIR}")


def classify_feature(feature_name: str) -> str:
    """Classify feature into its class (Shape, First-Order, Texture, etc.)."""
    feature_lower = feature_name.lower()

    # Remove filter prefixes (log-sigma, wavelet, etc.)
    if 'original_' in feature_lower:
        feature_lower = feature_lower.split('original_')[1]
    elif 'log-sigma' in feature_lower or 'wavelet' in feature_lower:
        # Extract feature type after filter prefix
        parts = feature_lower.split('_')
        for part in parts:
            if part in FEATURE_CLASS_MAP:
                return FEATURE_CLASS_MAP[part]
        # Default for filtered features
        if 'firstorder' in feature_lower:
            return 'First-Order (Filtered)'
        elif any(tex in feature_lower for tex in ['glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']):
            return 'Texture (Filtered)'

    # Match known classes
    for key, value in FEATURE_CLASS_MAP.items():
        if key in feature_lower:
            return value

    return 'Other'


def get_robustness_label(icc: float, cov_pct: float) -> str:
    """
    Classify feature robustness according to plan criteria.
    Robust: ICC >= 0.90 AND CoV <= 10%
    Acceptable: ICC >= 0.75
    Poor: ICC < 0.75
    """
    if pd.isna(icc) or pd.isna(cov_pct):
        return 'poor'

    if icc >= ICC_ROBUST_THRESHOLD and cov_pct <= COV_ROBUST_THRESHOLD:
        return 'robust'
    elif icc >= ICC_ACCEPTABLE_THRESHOLD:
        return 'acceptable'
    else:
        return 'poor'


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_dvh_metrics() -> pd.DataFrame:
    """Load aggregated DVH metrics."""
    path = RESULTS_DIR / "dvh_metrics.xlsx"
    if not path.exists():
        print(f"Warning: {path} not found")
        return pd.DataFrame()

    df = pd.read_excel(path)
    print(f"Loaded DVH metrics: {df.shape}")
    return df


def load_radiomics_ct() -> pd.DataFrame:
    """Load aggregated CT radiomics features."""
    path = RESULTS_DIR / "radiomics_ct.xlsx"
    if not path.exists():
        print(f"Warning: {path} not found")
        return pd.DataFrame()

    df = pd.read_excel(path)
    print(f"Loaded radiomics CT: {df.shape}")
    return df


def load_robustness_summary() -> pd.DataFrame:
    """Load robustness summary data."""
    path = RESULTS_DIR / "radiomics_robustness_summary.xlsx"
    if not path.exists():
        print(f"Warning: {path} not found")
        return pd.DataFrame()

    df = pd.read_excel(path)
    print(f"Loaded robustness summary: {df.shape}")
    return df


def load_case_metadata() -> pd.DataFrame:
    """Load case metadata."""
    path = RESULTS_DIR / "case_metadata.xlsx"
    if not path.exists():
        print(f"Warning: {path} not found")
        return pd.DataFrame()

    df = pd.read_excel(path)
    print(f"Loaded case metadata: {df.shape}")
    return df


def load_per_patient_robustness() -> pd.DataFrame:
    """
    Load and combine robustness data from all patient directories.
    Returns long-format DataFrame with perturbation results.
    """
    dfs = []

    for patient_dir in OUTPUT_DIR.iterdir():
        if not patient_dir.is_dir() or patient_dir.name.startswith('_'):
            continue

        # Find course directories
        for course_dir in patient_dir.iterdir():
            if not course_dir.is_dir():
                continue

            parquet_path = course_dir / "radiomics_robustness_ct.parquet"
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                df['patient_id'] = patient_dir.name
                df['course_id'] = course_dir.name
                dfs.append(df)

    if not dfs:
        print("Warning: No per-patient robustness data found")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    # Normalize column names: 'roi_name' -> 'structure' for consistency
    if 'roi_name' in combined.columns and 'structure' not in combined.columns:
        combined = combined.rename(columns={'roi_name': 'structure'})

    print(f"Loaded per-patient robustness data: {combined.shape}")
    return combined


# =============================================================================
# ICC Computation (Per Statistical Analysis Plan)
# =============================================================================

def compute_icc_per_structure_feature(
    full_df: pd.DataFrame,
    value_col: str = 'value',
    structure_col: str = 'roi_name'
) -> Iterator[dict]:
    """
    Compute ICC(3,1) per (structure, feature) tuple.

    According to the statistical analysis plan:
    - Subject = patient_id (not patient + perturbation)
    - Rater = perturbation_id
    - ICC(3,1) for two-way mixed model

    Requires ≥ 2 subjects (patients) and ≥ 2 raters (perturbations).
    Note: ICC with only 2 subjects has limited reliability; ≥3 recommended.
    """
    if full_df.empty:
        return

    required_cols = [structure_col, 'feature_name', 'patient_id', 'perturbation_id', value_col]
    missing = [c for c in required_cols if c not in full_df.columns]
    if missing:
        print(f"Warning: Missing columns for ICC computation: {missing}")
        return

    for (structure, feature_name), group_df in full_df.groupby([structure_col, 'feature_name']):
        n_subjects = group_df['patient_id'].nunique()
        n_raters = group_df['perturbation_id'].nunique()

        # Need at least 2 subjects and 2 raters for ICC
        if n_subjects < 2 or n_raters < 2:
            yield {
                'structure': structure,
                'feature_name': feature_name,
                'n_subjects': n_subjects,
                'n_raters': n_raters,
                'icc': np.nan,
                'icc_ci95_low': np.nan,
                'icc_ci95_high': np.nan,
                'error': 'insufficient_data'
            }
            continue

        try:
            # Prepare data for pingouin ICC
            icc_data = group_df[['patient_id', 'perturbation_id', value_col]].copy()
            icc_data.columns = ['targets', 'raters', 'ratings']

            # Remove any duplicates (take mean if multiple values)
            icc_data = icc_data.groupby(['targets', 'raters'])['ratings'].mean().reset_index()

            # Compute ICC(3,1) - two-way mixed, single measures
            icc_result = pg.intraclass_corr(
                data=icc_data,
                targets='targets',
                raters='raters',
                ratings='ratings'
            )

            # Extract ICC(3,1) - "ICC3" in pingouin
            icc3_row = icc_result[icc_result['Type'] == 'ICC3']

            if len(icc3_row) > 0:
                icc_value = icc3_row['ICC'].values[0]
                ci95 = icc3_row['CI95%'].values[0]

                yield {
                    'structure': structure,
                    'feature_name': feature_name,
                    'n_subjects': n_subjects,
                    'n_raters': n_raters,
                    'icc': icc_value,
                    'icc_ci95_low': ci95[0] if isinstance(ci95, (list, tuple, np.ndarray)) else np.nan,
                    'icc_ci95_high': ci95[1] if isinstance(ci95, (list, tuple, np.ndarray)) else np.nan,
                    'error': None
                }
            else:
                yield {
                    'structure': structure,
                    'feature_name': feature_name,
                    'n_subjects': n_subjects,
                    'n_raters': n_raters,
                    'icc': np.nan,
                    'icc_ci95_low': np.nan,
                    'icc_ci95_high': np.nan,
                    'error': 'icc3_not_found'
                }

        except Exception as e:
            yield {
                'structure': structure,
                'feature_name': feature_name,
                'n_subjects': n_subjects,
                'n_raters': n_raters,
                'icc': np.nan,
                'icc_ci95_low': np.nan,
                'icc_ci95_high': np.nan,
                'error': str(e)
            }


def compute_cov_per_structure_feature(
    full_df: pd.DataFrame,
    value_col: str = 'value',
    structure_col: str = 'roi_name'
) -> pd.DataFrame:
    """
    Compute Coefficient of Variation per (structure, feature) tuple.

    Returns both:
    - cov_pct: Standard CoV = (SD / |mean|) * 100 [%]
    - qcov_pct: Quartile CoV = (IQR / |median|) * 100 [%] - robust alternative

    QCoV is preferred when mean is near zero (values cross zero, e.g., HU values).
    Per Gemini-3-pro consensus review: standard CoV is mathematically invalid for
    features with near-zero means. QCoV provides a robust alternative.
    """
    if full_df.empty:
        return pd.DataFrame()

    cov_results = []

    for (structure, feature_name), group_df in full_df.groupby([structure_col, 'feature_name']):
        values = group_df[value_col].dropna()

        if len(values) < 2:
            cov_results.append({
                'structure': structure,
                'feature_name': feature_name,
                'cov_pct': np.nan,
                'qcov_pct': np.nan,
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                'iqr': np.nan,
                'n': len(values),
                'cov_reliable': False
            })
            continue

        mean_val = values.mean()
        median_val = values.median()
        std_val = values.std()
        q1, q3 = values.quantile([0.25, 0.75])
        iqr_val = q3 - q1

        # Standard CoV - avoid division by near-zero mean
        # Flag as unreliable if |mean| < 0.1 * std (mean is dominated by noise)
        cov_reliable = abs(mean_val) >= 0.1 * std_val if std_val > 0 else True
        if abs(mean_val) < 1e-10:
            cov_pct = np.nan
        else:
            cov_pct = (std_val / abs(mean_val)) * 100

        # Quartile CoV (QCoV) - robust alternative using IQR/median
        if abs(median_val) < 1e-10:
            qcov_pct = np.nan
        else:
            qcov_pct = (iqr_val / abs(median_val)) * 100

        cov_results.append({
            'structure': structure,
            'feature_name': feature_name,
            'cov_pct': cov_pct,
            'qcov_pct': qcov_pct,
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
            'iqr': iqr_val,
            'n': len(values),
            'cov_reliable': cov_reliable
        })

    return pd.DataFrame(cov_results)


# =============================================================================
# Table Generation Functions
# =============================================================================

def generate_table2_demographics(metadata_df: pd.DataFrame, dvh_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Table 2: Example Dataset Characteristics.
    Part A: Patient Demographics
    Part B: Technical Parameters
    """
    print("\nGenerating Table 2: Dataset Characteristics...")

    # Part A: Patient demographics from DVH (one row per patient)
    if not dvh_df.empty:
        patient_info = dvh_df.groupby('patient_id').first().reset_index()

        demographics = pd.DataFrame({
            'Patient ID': patient_info['patient_id'].values,
            'N Structures': dvh_df.groupby('patient_id').size().values,
        })

        # Add prescription dose if available
        if 'total_prescription_gy' in dvh_df.columns:
            demographics['Prescription (Gy)'] = patient_info['total_prescription_gy'].values
    else:
        demographics = pd.DataFrame()

    # Part B: Technical parameters from metadata
    tech_params = {}
    if not metadata_df.empty:
        if 'ct_manufacturer' in metadata_df.columns:
            tech_params['CT Manufacturer'] = metadata_df['ct_manufacturer'].dropna().unique()
        if 'ct_slice_thickness' in metadata_df.columns:
            thicknesses = metadata_df['ct_slice_thickness'].dropna()
            if len(thicknesses) > 0:
                tech_params['Slice Thickness (mm)'] = f"{thicknesses.min():.1f} - {thicknesses.max():.1f}"
        if 'ct_pixel_spacing' in metadata_df.columns:
            tech_params['Pixel Spacing'] = metadata_df['ct_pixel_spacing'].iloc[0] if len(metadata_df) > 0 else 'N/A'

    # Save demographics
    demographics.to_excel(TABLES_DIR / "table2_demographics.xlsx", index=False)

    # Save technical parameters
    tech_df = pd.DataFrame([tech_params])
    tech_df.to_excel(TABLES_DIR / "table2_technical_params.xlsx", index=False)

    print(f"  Saved Table 2 with {len(demographics)} patients")
    return demographics


def generate_table4_dvh_summary(dvh_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Table 4: DVH Metrics Summary.
    Structure-wise summary with statistical aggregation.
    """
    print("\nGenerating Table 4: DVH Metrics Summary...")

    if dvh_df.empty:
        print("  Warning: No DVH data available")
        return pd.DataFrame()

    # Key structures to include
    key_structures = ['CTV', 'PTV', 'GTV', 'RECTUM', 'BLADDER', 'FEMUR', 'BOWEL', 'BODY', 'prostate', 'urinary_bladder']

    # Columns to summarize
    metric_cols = ['DmeanGy', 'D95Gy', 'D2Gy', 'Volume (cm³)']
    available_metrics = [c for c in metric_cols if c in dvh_df.columns]

    # Check which structure column exists
    structure_col = 'ROI_Name' if 'ROI_Name' in dvh_df.columns else 'structure'

    summary_rows = []
    for structure_pattern in key_structures:
        # Find matching structures
        mask = dvh_df[structure_col].str.contains(structure_pattern, case=False, na=False)
        struct_df = dvh_df[mask]

        if len(struct_df) == 0:
            continue

        row = {'Structure': structure_pattern, 'N': len(struct_df)}

        for metric in available_metrics:
            values = struct_df[metric].dropna()
            if len(values) > 0:
                row[f'{metric} (mean±SD)'] = f"{values.mean():.2f}±{values.std():.2f}"
            else:
                row[f'{metric} (mean±SD)'] = 'N/A'

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_excel(TABLES_DIR / "table4_dvh_summary.xlsx", index=False)

    print(f"  Saved Table 4 with {len(summary_df)} structures")
    return summary_df


def generate_table5_feature_classes(radiomics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Table 5: Radiomics Feature Classes.
    Summarize features by class with counts and examples.
    """
    print("\nGenerating Table 5: Feature Classes...")

    # Get original features only (not filtered)
    feature_cols = [c for c in radiomics_df.columns if c.startswith('original_')]

    # Classify features
    feature_classes = {}
    for feat in feature_cols:
        feat_class = classify_feature(feat)
        if feat_class not in feature_classes:
            feature_classes[feat_class] = []
        feature_classes[feat_class].append(feat)

    # Build summary table
    rows = []
    for feat_class in ['Shape', 'First-Order', 'GLCM', 'GLRLM', 'GLSZM', 'GLDM', 'NGTDM']:
        if feat_class in feature_classes:
            features = feature_classes[feat_class]
            examples = [f.split('_')[-1] for f in features[:3]]
            rows.append({
                'Feature Class': feat_class,
                'N Features': len(features),
                'Examples': ', '.join(examples),
                'PyRadiomics Module': feat_class.lower().replace('-', '')
            })

    # Add total
    rows.append({
        'Feature Class': 'Total',
        'N Features': len(feature_cols),
        'Examples': '',
        'PyRadiomics Module': ''
    })

    summary_df = pd.DataFrame(rows)
    summary_df.to_excel(TABLES_DIR / "table5_feature_classes.xlsx", index=False)

    print(f"  Saved Table 5 with {len(rows)-1} feature classes, {len(feature_cols)} total features")
    return summary_df


def generate_table6_robustness(robustness_df: pd.DataFrame, perturbation_df: pd.DataFrame) -> dict:
    """
    Generate Table 6: Robustness Analysis Results (KEY TABLE).

    Panel A: Overall Robustness Summary
    Panel B: Robustness by Structure
    Panel C: Top 10 Most Robust Features
    Panel D: Top 10 Least Robust Features
    """
    print("\nGenerating Table 6: Robustness Analysis Results...")

    results = {}

    # If we have per-patient perturbation data, recompute ICC
    if not perturbation_df.empty:
        print("  Computing ICC from perturbation data...")

        # Compute ICC per (structure, feature)
        icc_results = list(compute_icc_per_structure_feature(
            perturbation_df,
            value_col='value',
            structure_col='structure'
        ))
        icc_df = pd.DataFrame(icc_results)

        # Compute CoV per (structure, feature)
        cov_df = compute_cov_per_structure_feature(
            perturbation_df,
            value_col='value',
            structure_col='structure'
        )

        # Merge ICC and CoV
        print(f"  ICC results: {len(icc_df)} rows")
        print(f"  CoV results: {len(cov_df)} rows")

        if not icc_df.empty and not cov_df.empty:
            rob_computed = icc_df.merge(
                cov_df[['structure', 'feature_name', 'cov_pct']],
                on=['structure', 'feature_name'],
                how='left'
            )

            # Add feature class
            rob_computed['feature_class'] = rob_computed['feature_name'].apply(classify_feature)

            # Add robustness label
            rob_computed['robustness_label'] = rob_computed.apply(
                lambda row: get_robustness_label(row['icc'], row['cov_pct']), axis=1
            )

            robustness_df = rob_computed
            print(f"  Merged results: {len(robustness_df)} rows with columns: {list(robustness_df.columns)}")
        else:
            print("  WARNING: ICC or CoV computation returned empty results - using per-feature summary instead")
            print(f"  Original robustness_df columns: {list(robustness_df.columns)}")

    # Use existing robustness data if no perturbation data
    if robustness_df.empty or 'icc' not in robustness_df.columns:
        print("  Warning: No valid robustness data available")
        # Create empty placeholder tables
        results['panel_a'] = pd.DataFrame({'Metric': ['No data available']})
        results['panel_b'] = pd.DataFrame()
        results['panel_c'] = pd.DataFrame()
        results['panel_d'] = pd.DataFrame()
        return results

    # Add feature class if not present
    if 'feature_class' not in robustness_df.columns:
        robustness_df['feature_class'] = robustness_df['feature_name'].apply(classify_feature)

    # Add robustness label if not present
    if 'robustness_label' not in robustness_df.columns:
        robustness_df['robustness_label'] = robustness_df.apply(
            lambda row: get_robustness_label(row.get('icc', np.nan), row.get('cov_pct', np.nan)),
            axis=1
        )

    # Panel A: Overall Robustness Summary
    valid_icc = robustness_df[~robustness_df['icc'].isna()]

    def compute_class_summary(df):
        if len(df) == 0:
            return {'N': 0, 'Mean ICC': np.nan, 'Median ICC': np.nan,
                    '% Robust': np.nan, '% Acceptable': np.nan, 'Mean CoV (%)': np.nan}
        return {
            'N': len(df),
            'Mean ICC': df['icc'].mean(),
            'Median ICC': df['icc'].median(),
            '% Robust': (df['robustness_label'] == 'robust').mean() * 100,
            '% Acceptable': (df['robustness_label'].isin(['robust', 'acceptable'])).mean() * 100,
            'Mean CoV (%)': df['cov_pct'].mean() if 'cov_pct' in df.columns else np.nan
        }

    panel_a_data = {
        'Metric': ['N features', 'Mean ICC', 'Median ICC',
                   '% Robust (ICC≥0.90, CoV≤10%)', '% Acceptable (ICC≥0.75)', 'Mean CoV (%)']
    }

    # All features
    all_summary = compute_class_summary(valid_icc)
    panel_a_data['All Features'] = [
        all_summary['N'],
        f"{all_summary['Mean ICC']:.3f}" if not pd.isna(all_summary['Mean ICC']) else 'N/A',
        f"{all_summary['Median ICC']:.3f}" if not pd.isna(all_summary['Median ICC']) else 'N/A',
        f"{all_summary['% Robust']:.1f}%" if not pd.isna(all_summary['% Robust']) else 'N/A',
        f"{all_summary['% Acceptable']:.1f}%" if not pd.isna(all_summary['% Acceptable']) else 'N/A',
        f"{all_summary['Mean CoV (%)']:.1f}" if not pd.isna(all_summary['Mean CoV (%)']) else 'N/A'
    ]

    # By feature class - include all classes from FEATURE_CLASS_MAP
    all_classes = list(FEATURE_CLASS_MAP.values()) + ['Texture (Filtered)', 'First-Order (Filtered)', 'Other']
    for feat_class in all_classes:
        class_df = valid_icc[valid_icc['feature_class'] == feat_class]
        if len(class_df) == 0:
            continue  # Skip empty classes
        class_summary = compute_class_summary(class_df)
        panel_a_data[feat_class] = [
            class_summary['N'],
            f"{class_summary['Mean ICC']:.3f}" if not pd.isna(class_summary['Mean ICC']) else 'N/A',
            f"{class_summary['Median ICC']:.3f}" if not pd.isna(class_summary['Median ICC']) else 'N/A',
            f"{class_summary['% Robust']:.1f}%" if not pd.isna(class_summary['% Robust']) else 'N/A',
            f"{class_summary['% Acceptable']:.1f}%" if not pd.isna(class_summary['% Acceptable']) else 'N/A',
            f"{class_summary['Mean CoV (%)']:.1f}" if not pd.isna(class_summary['Mean CoV (%)']) else 'N/A'
        ]

    results['panel_a'] = pd.DataFrame(panel_a_data)

    # Panel B: Robustness by Structure (filtered to target structures, with normalization)
    # Use full robustness_df if valid_icc is empty (e.g., when ICC cannot be computed due to insufficient patients)
    data_for_panels = valid_icc if len(valid_icc) > 0 else robustness_df
    if 'structure' in data_for_panels.columns and len(data_for_panels) > 0:
        # Filter to target structures and add normalized column
        target_icc = data_for_panels[data_for_panels['structure'].apply(is_target_structure)].copy()
        if len(target_icc) > 0:
            target_icc['structure_normalized'] = target_icc['structure'].apply(normalize_structure_name)
        else:
            print("  Warning: No target structures found in data")
            results['panel_b'] = pd.DataFrame()
            results['panel_c'] = pd.DataFrame()
            results['panel_d'] = pd.DataFrame()
            # Save empty panels
            results['panel_a'].to_excel(TABLES_DIR / "table6_panel_a_overall.xlsx", index=False)
            results['panel_b'].to_excel(TABLES_DIR / "table6_panel_b_structure.xlsx", index=False)
            results['panel_c'].to_excel(TABLES_DIR / "table6_panel_c_top_robust.xlsx", index=False)
            results['panel_d'].to_excel(TABLES_DIR / "table6_panel_d_least_robust.xlsx", index=False)
            return results

        structure_summary = []
        for norm_structure in sorted(target_icc['structure_normalized'].unique()):
            struct_df = target_icc[target_icc['structure_normalized'] == norm_structure]
            struct_summary = compute_class_summary(struct_df)
            structure_summary.append({
                'Structure': norm_structure,
                'N Features': struct_summary['N'],
                'Mean ICC': f"{struct_summary['Mean ICC']:.3f}" if not pd.isna(struct_summary['Mean ICC']) else 'N/A',
                '% Robust': f"{struct_summary['% Robust']:.1f}%" if not pd.isna(struct_summary['% Robust']) else 'N/A',
                'Mean CoV (%)': f"{struct_summary['Mean CoV (%)']:.1f}" if not pd.isna(struct_summary['Mean CoV (%)']) else 'N/A'
            })
        results['panel_b'] = pd.DataFrame(structure_summary)
        print(f"  Panel B: {len(structure_summary)} target structures (filtered from {valid_icc['structure'].nunique()} total)")
    else:
        results['panel_b'] = pd.DataFrame()

    # Panel C: Top 10 Most Robust Features
    sorted_by_icc = valid_icc.sort_values('icc', ascending=False)
    top_10 = sorted_by_icc.head(10).copy()
    top_10['Rank'] = range(1, len(top_10) + 1)
    results['panel_c'] = top_10[['Rank', 'feature_name', 'feature_class', 'icc', 'cov_pct']].copy()
    results['panel_c'].columns = ['Rank', 'Feature Name', 'Class', 'ICC', 'CoV (%)']

    # Panel D: Top 10 Least Robust Features (with negative ICC warning)
    bottom_10 = sorted_by_icc.tail(10).iloc[::-1].copy()
    bottom_10['Rank'] = range(1, len(bottom_10) + 1)

    # Flag negative ICC values (per Gemini-3-pro: indicates broken pipeline or methodological issue)
    negative_icc_count = (valid_icc['icc'] < 0).sum()
    if negative_icc_count > 0:
        print(f"  WARNING: {negative_icc_count} features have negative ICC values (indicates measurement artifacts)")
        # Add warning flag to Panel D features
        bottom_10['Warning'] = bottom_10['icc'].apply(
            lambda x: 'NEGATIVE ICC' if x < 0 else ''
        )
        results['panel_d'] = bottom_10[['Rank', 'feature_name', 'feature_class', 'icc', 'cov_pct', 'Warning']].copy()
        results['panel_d'].columns = ['Rank', 'Feature Name', 'Class', 'ICC', 'CoV (%)', 'Warning']
    else:
        results['panel_d'] = bottom_10[['Rank', 'feature_name', 'feature_class', 'icc', 'cov_pct']].copy()
        results['panel_d'].columns = ['Rank', 'Feature Name', 'Class', 'ICC', 'CoV (%)']

    # Save all panels
    with pd.ExcelWriter(TABLES_DIR / "table6_robustness.xlsx") as writer:
        results['panel_a'].to_excel(writer, sheet_name='Panel_A_Overall', index=False)
        if not results['panel_b'].empty:
            results['panel_b'].to_excel(writer, sheet_name='Panel_B_ByStructure', index=False)
        results['panel_c'].to_excel(writer, sheet_name='Panel_C_Top10Robust', index=False)
        results['panel_d'].to_excel(writer, sheet_name='Panel_D_Top10Poor', index=False)

    print(f"  Saved Table 6 with {len(valid_icc)} features analyzed")
    return results


def generate_table8_perturbation_params() -> pd.DataFrame:
    """
    Generate Table 8: Perturbation Parameter Settings.
    Documents NTCV parameters used in robustness analysis.
    """
    print("\nGenerating Table 8: Perturbation Parameters...")

    # Parameters from TABLES_PLAN.md
    params = [
        {'Perturbation Type': 'Noise (N)', 'Parameter': 'Distribution', 'Value': 'Gaussian', 'Rationale': 'Scanner noise model'},
        {'Perturbation Type': '', 'Parameter': 'σ (HU)', 'Value': '10, 20 (0 = baseline)', 'Rationale': 'Typical CT noise level (Zwanenburg 2019)'},
        {'Perturbation Type': '', 'Parameter': 'N samples', 'Value': '3', 'Rationale': 'Computational balance'},
        {'Perturbation Type': 'Translation (T)', 'Parameter': 'Direction', 'Value': 'Axial plane (x, y)', 'Rationale': 'Patient positioning uncertainty'},
        {'Perturbation Type': '', 'Parameter': 'Distance (mm)', 'Value': '±2, ±4', 'Rationale': 'Setup margins (clinical)'},
        {'Perturbation Type': '', 'Parameter': 'N samples', 'Value': '4', 'Rationale': ''},
        {'Perturbation Type': 'Contour (C)', 'Parameter': 'Method', 'Value': 'Morphological operations', 'Rationale': 'Inter-observer variation'},
        {'Perturbation Type': '', 'Parameter': 'Kernel size (mm)', 'Value': '1-2', 'Rationale': 'Typical contouring uncertainty'},
        {'Perturbation Type': '', 'Parameter': 'N samples', 'Value': '3', 'Rationale': ''},
        {'Perturbation Type': 'Volume (V)', 'Parameter': 'Method', 'Value': 'Morphological erosion/dilation', 'Rationale': 'Segmentation uncertainty'},
        {'Perturbation Type': '', 'Parameter': 'Volume change', 'Value': '±15%', 'Rationale': 'Clinical variability range'},
        {'Perturbation Type': '', 'Parameter': 'N samples', 'Value': '3', 'Rationale': ''},
        {'Perturbation Type': 'Combined (NTCV-full)', 'Parameter': 'Application', 'Value': 'Simultaneous (compound)', 'Rationale': 'Tests stability under combined uncertainty'},
        {'Perturbation Type': '', 'Parameter': 'Total perturbations', 'Value': 'N×T×C×V', 'Rationale': 'Full factorial design'},
        {'Perturbation Type': '', 'Parameter': 'N samples', 'Value': '108 (3×4×3×3)', 'Rationale': ''},
    ]

    params_df = pd.DataFrame(params)
    params_df.to_excel(TABLES_DIR / "table8_perturbation_params.xlsx", index=False)

    print(f"  Saved Table 8")
    return params_df


# =============================================================================
# Figure Generation Functions
# =============================================================================

def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': [FONT_FAMILY, 'DejaVu Sans', 'Helvetica'],
        'font.size': FONT_SIZE_LABEL,
        'axes.titlesize': FONT_SIZE_TITLE,
        'axes.labelsize': FONT_SIZE_LABEL,
        'xtick.labelsize': FONT_SIZE_TICK,
        'ytick.labelsize': FONT_SIZE_TICK,
        'legend.fontsize': FONT_SIZE_TICK,
        'figure.dpi': FIGURE_DPI_SCREEN,
        'savefig.dpi': FIGURE_DPI_PRINT,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.5,
    })


def truncate_feature_name(name: str, max_len: int = 30) -> str:
    """Truncate and clean feature names for display."""
    # Remove common prefixes
    name = name.replace('original_', '').replace('wavelet-', 'W:').replace('log-sigma-', 'L:')
    if len(name) > max_len:
        return name[:max_len-2] + '...'
    return name


def generate_figure4_robustness(robustness_df: pd.DataFrame, perturbation_df: pd.DataFrame):
    """
    Generate Figure 4: Robustness Analysis Summary (KEY FIGURE).
    Publication-ready version based on consensus from gpt-5.1-codex and gemini-3-pro-preview.

    Panel A: ICC Distribution Histogram with shaded threshold regions
    Panel B: Horizontal stacked bar chart by Structure (for 30+ structures)
    Panel C: Feature Robustness Heatmap with truncated names
    Panel D: Hexbin density plot for ICC vs CoV (handles 10k+ points)
    """
    print("\nGenerating Figure 4: Robustness Analysis (publication-ready)...")

    # Setup publication style
    setup_publication_style()

    # Prepare data
    if not perturbation_df.empty:
        # Recompute ICC if we have perturbation data
        icc_results = list(compute_icc_per_structure_feature(
            perturbation_df, value_col='value', structure_col='structure'
        ))
        cov_df = compute_cov_per_structure_feature(
            perturbation_df, value_col='value', structure_col='structure'
        )

        rob_df = pd.DataFrame(icc_results)
        if not cov_df.empty:
            rob_df = rob_df.merge(
                cov_df[['structure', 'feature_name', 'cov_pct']],
                on=['structure', 'feature_name'],
                how='left'
            )
        rob_df['feature_class'] = rob_df['feature_name'].apply(classify_feature)
        rob_df['robustness_label'] = rob_df.apply(
            lambda row: get_robustness_label(row.get('icc', np.nan), row.get('cov_pct', np.nan)),
            axis=1
        )
    else:
        rob_df = robustness_df.copy()
        if 'feature_class' not in rob_df.columns:
            rob_df['feature_class'] = rob_df['feature_name'].apply(classify_feature)
        if 'robustness_label' not in rob_df.columns:
            rob_df['robustness_label'] = 'poor'

    # Filter to valid ICC values
    valid_df = rob_df[~rob_df['icc'].isna()].copy()

    if len(valid_df) == 0:
        print("  Warning: No valid ICC values for figure generation")
        # Create placeholder figure
        fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH_DOUBLE, FIGURE_WIDTH_DOUBLE * 0.85))
        for ax in axes.flat:
            ax.text(0.5, 0.5, 'Insufficient data\n(need ≥2 patients)',
                   ha='center', va='center', fontsize=FONT_SIZE_TITLE, transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        fig.suptitle('Figure 4: Robustness Analysis Summary\n(Placeholder - requires multi-patient data)',
                    fontsize=FONT_SIZE_TITLE, fontweight='bold')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig4_robustness_placeholder.pdf", dpi=FIGURE_DPI_PRINT, bbox_inches='tight')
        plt.savefig(FIGURES_DIR / "fig4_robustness_placeholder.png", dpi=FIGURE_DPI_PRINT, bbox_inches='tight')
        plt.close()
        return

    # Create figure with GridSpec for better layout control
    fig = plt.figure(figsize=(FIGURE_WIDTH_DOUBLE, FIGURE_WIDTH_DOUBLE * 0.9))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # =========================================================================
    # Panel A: ICC Distribution Histogram with shaded threshold regions
    # =========================================================================
    ax = fig.add_subplot(gs[0, 0])

    # Add shaded regions for classification zones
    ax.axvspan(0, ICC_ACCEPTABLE_THRESHOLD, alpha=0.15, color=COLORS['poor'], label='Poor zone')
    ax.axvspan(ICC_ACCEPTABLE_THRESHOLD, ICC_ROBUST_THRESHOLD, alpha=0.15, color=COLORS['acceptable'], label='Acceptable zone')
    ax.axvspan(ICC_ROBUST_THRESHOLD, 1.0, alpha=0.15, color=COLORS['robust'], label='Robust zone')

    # Histogram
    ax.hist(valid_df['icc'].dropna(), bins=30, edgecolor='white', alpha=0.85, color=COLORS['processing'])

    # Threshold lines with annotations
    ax.axvline(ICC_ROBUST_THRESHOLD, color=COLORS['robust'], linestyle='-', linewidth=2)
    ax.axvline(ICC_ACCEPTABLE_THRESHOLD, color=COLORS['acceptable'], linestyle='-', linewidth=2)

    # Add text annotations for thresholds
    ymax = ax.get_ylim()[1]
    ax.text(ICC_ROBUST_THRESHOLD + 0.02, ymax * 0.95, f'ICC={ICC_ROBUST_THRESHOLD}',
           fontsize=FONT_SIZE_TICK, color=COLORS['robust'], fontweight='bold', va='top')
    ax.text(ICC_ACCEPTABLE_THRESHOLD + 0.02, ymax * 0.85, f'ICC={ICC_ACCEPTABLE_THRESHOLD}',
           fontsize=FONT_SIZE_TICK, color=COLORS['acceptable'], fontweight='bold', va='top')

    ax.set_xlabel('ICC Value')
    ax.set_ylabel('Number of Features')
    ax.set_title('A. ICC Distribution', fontweight='bold')
    ax.set_xlim(0, 1)

    # =========================================================================
    # Panel B: Horizontal stacked bar chart by Structure
    # =========================================================================
    ax = fig.add_subplot(gs[0, 1])
    if 'structure' in valid_df.columns:
        structure_counts = valid_df.groupby(['structure', 'robustness_label']).size().unstack(fill_value=0)
        structure_pct = structure_counts.div(structure_counts.sum(axis=1), axis=0) * 100

        # Sort by % robust for better visualization
        if 'robust' in structure_pct.columns:
            structure_pct = structure_pct.sort_values('robust', ascending=True)

        labels_order = ['poor', 'acceptable', 'robust']
        colors_order = [COLORS['poor'], COLORS['acceptable'], COLORS['robust']]

        # Use horizontal bars for many structures
        available_labels = [l for l in labels_order if l in structure_pct.columns]
        if available_labels:
            structure_pct[available_labels].plot(
                kind='barh', stacked=True, ax=ax,
                color=[colors_order[labels_order.index(l)] for l in available_labels],
                edgecolor='white', linewidth=0.5
            )

        ax.set_xlabel('Percentage of Features (%)')
        ax.set_ylabel('')  # Structure names on y-axis
        ax.set_title('B. Robustness by Structure', fontweight='bold')
        ax.legend(title='Classification', loc='lower right', framealpha=0.9)

        # Limit structure labels if too many
        if len(structure_pct) > 15:
            ax.tick_params(axis='y', labelsize=6)
    else:
        ax.text(0.5, 0.5, 'Structure data not available', ha='center', va='center', transform=ax.transAxes)

    # =========================================================================
    # Panel C: Feature Robustness Heatmap with truncated names
    # =========================================================================
    ax = fig.add_subplot(gs[1, 0])
    if 'structure' in valid_df.columns and len(valid_df['structure'].unique()) > 1:
        pivot = valid_df.pivot_table(index='feature_name', columns='structure', values='icc')

        # Limit to top 15 features for readability
        if len(pivot) > 15:
            mean_icc = pivot.mean(axis=1).sort_values(ascending=False)
            pivot = pivot.loc[mean_icc.head(15).index]

        # Truncate feature names
        pivot.index = [truncate_feature_name(f, 25) for f in pivot.index]

        # Use viridis colormap (colorblind-safe)
        sns.heatmap(pivot, cmap='viridis', ax=ax, vmin=0, vmax=1,
                   cbar_kws={'label': 'ICC', 'shrink': 0.8},
                   annot=False, linewidths=0.5, linecolor='white')
        ax.set_title('C. ICC Heatmap (Top 15 Features)', fontweight='bold')
        ax.set_xlabel('Structure')
        ax.set_ylabel('')

        # Rotate structure labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=6)
        plt.setp(ax.get_yticklabels(), fontsize=7)
    else:
        ax.text(0.5, 0.5, 'Insufficient structure variation', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('C. ICC Heatmap', fontweight='bold')

    # =========================================================================
    # Panel D: Hexbin density plot for ICC vs CoV (handles 10k+ points)
    # =========================================================================
    ax = fig.add_subplot(gs[1, 1])
    if 'cov_pct' in valid_df.columns:
        scatter_df = valid_df.dropna(subset=['icc', 'cov_pct']).copy()

        # Cap CoV for display (per consensus: prevents outlier compression)
        scatter_df['cov_display'] = scatter_df['cov_pct'].clip(upper=COV_DISPLAY_CAP)
        n_capped = (scatter_df['cov_pct'] > COV_DISPLAY_CAP).sum()

        # Use hexbin for dense data (10k+ points)
        if len(scatter_df) > 500:
            hb = ax.hexbin(scatter_df['icc'], scatter_df['cov_display'],
                          gridsize=40, cmap='viridis', mincnt=1,
                          extent=[0, 1, 0, COV_DISPLAY_CAP])
            cb = plt.colorbar(hb, ax=ax, shrink=0.8)
            cb.set_label('Count', fontsize=FONT_SIZE_TICK)
        else:
            ax.scatter(scatter_df['icc'], scatter_df['cov_display'],
                      alpha=0.5, s=15, c=COLORS['processing'])

        # Add threshold lines
        ax.axvline(ICC_ROBUST_THRESHOLD, color=COLORS['robust'], linestyle='-', linewidth=2)
        ax.axhline(COV_ROBUST_THRESHOLD, color=COLORS['robust'], linestyle='-', linewidth=2)

        # Shade robust quadrant
        ax.fill_between([ICC_ROBUST_THRESHOLD, 1], [0, 0],
                       [COV_ROBUST_THRESHOLD, COV_ROBUST_THRESHOLD],
                       color=COLORS['robust'], alpha=0.2, zorder=0)

        # Add "Robust" label in quadrant
        ax.text(0.95, COV_ROBUST_THRESHOLD/2, 'ROBUST',
               fontsize=FONT_SIZE_TICK, fontweight='bold', color=COLORS['robust'],
               ha='right', va='center', alpha=0.7)

        ax.set_xlabel('ICC')
        ax.set_ylabel(f'CoV (%) [capped at {COV_DISPLAY_CAP}%]')
        ax.set_title('D. ICC vs CoV (Density)', fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, COV_DISPLAY_CAP)

        # Annotate capped points if any
        if n_capped > 0:
            ax.text(0.02, 0.98, f'{n_capped:,} points above {COV_DISPLAY_CAP}% not shown',
                   transform=ax.transAxes, fontsize=7, va='top', style='italic')
    else:
        ax.text(0.5, 0.5, 'CoV data not available', ha='center', va='center', transform=ax.transAxes)

    plt.savefig(FIGURES_DIR / "fig4_robustness.pdf", dpi=FIGURE_DPI_PRINT, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "fig4_robustness.png", dpi=FIGURE_DPI_PRINT, bbox_inches='tight')
    plt.close()

    print(f"  Saved Figure 4 with {len(valid_df)} feature-structure combinations")


def generate_figure5_perturbation_analysis(perturbation_df: pd.DataFrame):
    """
    Generate Figure 5: Perturbation Contribution Analysis.
    Publication-ready version based on consensus from gpt-5.1-codex and gemini-3-pro-preview.

    Improvements:
    - Human-readable perturbation labels (not cryptic IDs)
    - CoV capped at 100% to prevent outlier compression
    - Box plots with jittered points for better distribution visualization
    - Proper font sizing and journal dimensions
    """
    print("\nGenerating Figure 5: Perturbation Analysis (publication-ready)...")

    # Setup publication style
    setup_publication_style()

    if perturbation_df.empty:
        print("  Warning: No perturbation data available")
        return

    # Group by perturbation_id and compute statistics
    perturbation_stats = []
    for pert_id in perturbation_df['perturbation_id'].unique():
        pert_df = perturbation_df[perturbation_df['perturbation_id'] == pert_id]

        # Compute CoV for each feature
        for feature in pert_df['feature_name'].unique():
            feat_df = pert_df[pert_df['feature_name'] == feature]
            values = feat_df['value'].dropna()

            if len(values) > 1 and abs(values.mean()) > 1e-10:
                cov = (values.std() / abs(values.mean())) * 100
                perturbation_stats.append({
                    'perturbation_id': pert_id,
                    'feature_name': feature,
                    'cov_pct': cov,
                    'mean': values.mean(),
                    'std': values.std()
                })

    if not perturbation_stats:
        print("  Warning: Could not compute perturbation statistics")
        return

    stats_df = pd.DataFrame(perturbation_stats)

    # Map perturbation IDs to human-readable labels
    stats_df['perturbation_label'] = stats_df['perturbation_id'].map(
        lambda x: PERTURBATION_LABELS.get(x, x)
    )

    # Cap CoV for display (prevents outlier compression per consensus)
    stats_df['cov_display'] = stats_df['cov_pct'].clip(upper=COV_DISPLAY_CAP)
    n_capped = (stats_df['cov_pct'] > COV_DISPLAY_CAP).sum()
    total_points = len(stats_df)

    # Create figure with single column width
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_DOUBLE * 0.7, FIGURE_WIDTH_DOUBLE * 0.5))

    # Sort by median CoV for better visualization
    order = stats_df.groupby('perturbation_label')['cov_display'].median().sort_values().index.tolist()

    # Box plot with strip plot overlay (better than violin for showing outliers)
    box_color = COLORS['processing']

    # Draw box plot
    sns.boxplot(data=stats_df, x='perturbation_label', y='cov_display', ax=ax,
               order=order, color=box_color, width=0.5, linewidth=1.5,
               fliersize=0)  # Hide outliers, we'll show with strip

    # Add jittered points (subsample if too many)
    if len(stats_df) > 1000:
        sample_df = stats_df.sample(n=min(500, len(stats_df)), random_state=RANDOM_SEED)
    else:
        sample_df = stats_df

    sns.stripplot(data=sample_df, x='perturbation_label', y='cov_display', ax=ax,
                 order=order, color=COLORS['neutral'], alpha=0.3, size=2, jitter=0.2)

    # Add robust threshold line
    ax.axhline(COV_ROBUST_THRESHOLD, color=COLORS['robust'], linestyle='-', linewidth=2,
              label=f'Robust threshold (CoV ≤ {COV_ROBUST_THRESHOLD}%)', zorder=5)

    # Shade robust zone
    ax.axhspan(0, COV_ROBUST_THRESHOLD, alpha=0.1, color=COLORS['robust'], zorder=0)

    ax.set_xlabel('Perturbation Type')
    ax.set_ylabel(f'Coefficient of Variation (%) [capped at {COV_DISPLAY_CAP}%]')
    ax.set_title('Figure 5: Feature Stability by Perturbation Type', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)

    # Rotate x labels if needed
    if len(order) > 4:
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    ax.set_ylim(0, COV_DISPLAY_CAP)

    # Add annotation about capped points
    if n_capped > 0:
        pct_capped = n_capped / total_points * 100
        ax.text(0.02, 0.98, f'{n_capped:,} points ({pct_capped:.1f}%) above {COV_DISPLAY_CAP}% capped',
               transform=ax.transAxes, fontsize=7, va='top', style='italic')

    # Add summary statistics annotation
    median_cov = stats_df['cov_pct'].median()
    robust_pct = (stats_df['cov_pct'] <= COV_ROBUST_THRESHOLD).mean() * 100
    ax.text(0.98, 0.98, f'Median CoV: {median_cov:.1f}%\n{robust_pct:.1f}% features robust',
           transform=ax.transAxes, fontsize=7, va='top', ha='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_perturbation.pdf", dpi=FIGURE_DPI_PRINT, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "fig5_perturbation.png", dpi=FIGURE_DPI_PRINT, bbox_inches='tight')
    plt.close()

    print(f"  Saved Figure 5 with {len(stats_df)} perturbation-feature combinations")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    print("=" * 60)
    print("RTpipeline Manuscript - Analysis Script")
    print("=" * 60)

    # Setup
    setup_directories()

    # Load data
    print("\n--- Loading Data ---")
    dvh_df = load_dvh_metrics()
    radiomics_df = load_radiomics_ct()
    robustness_df = load_robustness_summary()
    metadata_df = load_case_metadata()
    perturbation_df = load_per_patient_robustness()

    # Filter to target structures (consensus: CTV1, CTV2, GTVp, urinary_bladder, colon, pelvic_bones + iliac_vess for validation)
    print("\n--- Filtering Target Structures ---")
    if not perturbation_df.empty and 'structure' in perturbation_df.columns:
        print(f"  Before filtering: {perturbation_df['structure'].nunique()} structures")
        perturbation_df = filter_target_structures(perturbation_df, structure_col='structure')
        print(f"  After filtering: {perturbation_df['structure'].nunique()} structures")
        if perturbation_df.empty:
            print("  WARNING: No target structures found in perturbation data!")
        else:
            print(f"  Structures included: {sorted(perturbation_df['structure'].unique())}")

    if not robustness_df.empty and 'structure' in robustness_df.columns:
        robustness_df = filter_target_structures(robustness_df, structure_col='structure')

    # Generate Tables
    print("\n--- Generating Tables ---")
    generate_table2_demographics(metadata_df, dvh_df)
    generate_table4_dvh_summary(dvh_df)
    generate_table5_feature_classes(radiomics_df)
    generate_table6_robustness(robustness_df, perturbation_df)
    generate_table8_perturbation_params()

    # Generate Figures
    print("\n--- Generating Figures ---")
    generate_figure4_robustness(robustness_df, perturbation_df)
    generate_figure5_perturbation_analysis(perturbation_df)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Tables saved to: {TABLES_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("=" * 60)

    # Summary statistics
    print("\n--- Data Summary ---")
    print(f"Patients processed: {dvh_df['patient_id'].nunique() if not dvh_df.empty else 0}")
    print(f"Structures with DVH: {dvh_df['ROI_Name'].nunique() if not dvh_df.empty and 'ROI_Name' in dvh_df.columns else 0}")
    print(f"Radiomics features: {len([c for c in radiomics_df.columns if c.startswith('original_')]) if not radiomics_df.empty else 0}")

    if not perturbation_df.empty:
        print(f"Perturbation samples: {len(perturbation_df)}")
        print(f"Perturbation types: {perturbation_df['perturbation_id'].nunique()}")

    # Note about data requirements
    valid_icc_count = len(robustness_df[~robustness_df['icc'].isna()]) if not robustness_df.empty and 'icc' in robustness_df.columns else 0
    if valid_icc_count == 0:
        print("\n⚠️  NOTE: ICC values are NaN - need ≥2 patients with perturbation data for valid ICC computation")
        print("    Run the pipeline on multiple patients to generate complete robustness analysis")


if __name__ == "__main__":
    main()
