#!/usr/bin/env python3
"""Reconcile the NSCLC benchmark and emit reviewer-facing current-data summaries.

This step intentionally focuses on the frozen manuscript package and only uses
already materialized analysis outputs. It does not regenerate raw radiomics.

Outputs
-------
- analysis/tables/nsclc_benchmark_denominator_flow_2026-04-23.csv
- analysis/tables/nsclc_io_within_patient_summary.csv
- analysis/tables/nsclc_io_patient_collapsed_family_support.csv
- analysis/tables/family_balanced_nsclc_summary_2026-04-21.csv
- analysis/tables/shared_feature_bridge_vs_full.csv
- analysis/tables/cluster_aware_nsclc_inference_2026-04-21.csv
- analysis/tables/nsclc_benchmark_geometry_radiomics_bridge_2026-04-23.csv
- analysis/tables/nsclc_benchmark_geometry_radiomics_bridge_patient_2026-04-23.csv
- analysis/tables/nsclc_thoracic_whitelist_2026-04-23.csv
- analysis/tables/nsclc_thoracic_whitelist_family_summary_2026-04-23.csv
- analysis/tables/nsclc_normalized_mad_sensitivity_2026-04-23.csv
- analysis/tables/qc_mechanism_summary_2026-04-23.csv

Key policy
----------
- The direct human-versus-AI benchmark is restricted to the true paired set:
  patient-feature rows with both human and AI dispersion present.
- The physical local benchmark package contains 21 case folders. One case
  (`interobs19`) remains human-only for radiomics because the frozen package
  does not preserve recoverable tumor AI masks / AI feature cache for that case.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.stats import binomtest, spearmanr, wilcoxon

PROJECT_ROOT = Path("/umed-projekty/rtpipeline")
ANALYSIS_DIR = PROJECT_ROOT / "manuscript" / "analysis"
DATA_DIR = ANALYSIS_DIR / "data"
TABLES_DIR = ANALYSIS_DIR / "tables"
NSCLC_DATASET_ROOT = Path("/umed-projekty/DICOMRT-datasets/NSCLC_Interobserver/output")

FAMILY_ORDER = ["shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
SEED = 20260423
BOOTSTRAP_ITERS = 10000


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def c4(n: int) -> float:
    if n <= 1:
        return float("nan")
    return math.sqrt(2.0 / (n - 1)) * gamma(n / 2.0) / gamma((n - 1) / 2.0)


def percentile_ci(values: list[float] | np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    return (
        float(np.quantile(arr, alpha / 2.0)),
        float(np.quantile(arr, 1.0 - alpha / 2.0)),
    )


def safe_wilcoxon(values: pd.Series | np.ndarray) -> tuple[float, float]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or np.allclose(arr, 0.0):
        return (0.0, 1.0)
    stat, p_value = wilcoxon(arr, zero_method="wilcox", alternative="two-sided")
    return (float(stat), float(p_value))


def sign_support(values: pd.Series | np.ndarray) -> tuple[int, int, int, float]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    positive = int(np.sum(arr > 0))
    negative = int(np.sum(arr < 0))
    zero = int(np.sum(np.isclose(arr, 0.0)))
    if positive + negative == 0:
        p_value = 1.0
    else:
        p_value = float(binomtest(positive, positive + negative, 0.5, alternative="greater").pvalue)
    return positive, negative, zero, p_value


def patient_cluster_bootstrap_median(
    df: pd.DataFrame,
    value_col: str,
    patient_col: str = "patient_id",
    n_boot: int = BOOTSTRAP_ITERS,
    seed: int = SEED,
) -> tuple[float, float]:
    working = df.loc[df[value_col].notna(), [patient_col, value_col]].copy()
    working[patient_col] = working[patient_col].astype(str)
    patient_arrays = [
        sub[value_col].to_numpy(dtype=float)
        for _, sub in working.groupby(patient_col, sort=True)
    ]
    if not patient_arrays:
        return (float("nan"), float("nan"))
    n_patients = len(patient_arrays)
    rng = np.random.default_rng(seed)
    boot_values = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sampled = rng.integers(0, n_patients, size=n_patients)
        boot_values[idx] = float(np.median(np.concatenate([patient_arrays[i] for i in sampled])))
    return percentile_ci(boot_values)


def summarize_pairwise(
    df: pd.DataFrame,
    delta_col: str = "cov_delta_pct",
    human_col: str = "human_cov_pct",
    ai_col: str = "ai_cov_pct",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family in FAMILY_ORDER:
        sub = df.loc[df["family"] == family].copy()
        stat, p_value = safe_wilcoxon(sub[delta_col])
        rows.append(
            {
                "feature_family": family,
                "n_features": int(sub["feature_name"].nunique()),
                "n_patient_feature_pairs": int(len(sub)),
                "median_human_cov_pct": float(sub[human_col].median()),
                "median_ai_cov_pct": float(sub[ai_col].median()),
                "median_diff_human_minus_ai": float(sub[delta_col].median()),
                "wilcoxon_stat": stat,
                "wilcoxon_p": p_value,
                "wilcoxon_direction": "human_higher" if float(sub[delta_col].median()) > 0 else "ai_higher",
            }
        )
    return pd.DataFrame(rows)


def summarize_patient_collapsed_family(
    df: pd.DataFrame,
    delta_col: str = "cov_delta_pct",
    human_col: str = "human_cov_pct",
    ai_col: str = "ai_cov_pct",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_family = (
        df.groupby(["patient_id", "family"], as_index=False, observed=False)
        .agg(
            patient_median_delta=(delta_col, "median"),
            patient_median_human=(human_col, "median"),
            patient_median_ai=(ai_col, "median"),
        )
    )
    rows: list[dict[str, object]] = []
    for family in FAMILY_ORDER:
        sub = patient_family.loc[patient_family["family"] == family].copy()
        pos, neg, zero, p_value = sign_support(sub["patient_median_delta"])
        rows.append(
            {
                "family": family,
                "n_patients": int(sub["patient_id"].nunique()),
                "median_patient_median_delta": float(sub["patient_median_delta"].median()),
                "pct_patients_ai_lower": float((sub["patient_median_delta"] > 0).mean() * 100.0),
                "sign_test_p_greater": p_value,
                "n_positive": pos,
                "n_negative": neg,
                "n_zero": zero,
            }
        )
    return pd.DataFrame(rows), patient_family


def summarize_family_balanced(patient_family: pd.DataFrame, analysis_set: str, notes: str) -> dict[str, object]:
    per_patient = (
        patient_family.groupby("patient_id", as_index=False)
        .agg(
            balanced_human=("patient_median_human", "mean"),
            balanced_ai=("patient_median_ai", "mean"),
            balanced_delta=("patient_median_delta", "mean"),
            n_families=("family", "nunique"),
        )
        .sort_values("patient_id")
    )
    stat, p_value = safe_wilcoxon(per_patient["balanced_delta"])
    return {
        "analysis_set": analysis_set,
        "n_patients": int(per_patient["patient_id"].nunique()),
        "n_families": int(per_patient["n_families"].max()),
        "median_human_cov_pct": float(per_patient["balanced_human"].median()),
        "median_ai_cov_pct": float(per_patient["balanced_ai"].median()),
        "median_balanced_delta_pct": float(per_patient["balanced_delta"].median()),
        "pct_patients_ai_favoring": float((per_patient["balanced_delta"] > 0).mean() * 100.0),
        "wilcoxon_p": p_value,
        "delta_retention_vs_full": np.nan,
        "notes": notes,
    }


def bootstrap_current_and_c4(df: pd.DataFrame) -> dict[str, float]:
    c4_human = c4(5)
    c4_ai = c4(3)
    human_corr = df["human_cov_pct"] / c4_human
    ai_corr = df["ai_cov_pct"] / c4_ai
    corrected_delta = human_corr - ai_corr
    corrected_df = df.copy()
    corrected_df["c4_delta_pct"] = corrected_delta
    current_ci = patient_cluster_bootstrap_median(df, "cov_delta_pct")
    corrected_ci = patient_cluster_bootstrap_median(corrected_df, "c4_delta_pct")
    return {
        "current_human_cov_pct": float(df["human_cov_pct"].median()),
        "current_ai_cov_pct": float(df["ai_cov_pct"].median()),
        "current_delta_pct": float(df["cov_delta_pct"].median()),
        "current_ci_low": current_ci[0],
        "current_ci_high": current_ci[1],
        "c4_human_cov_pct": float(human_corr.median()),
        "c4_ai_cov_pct": float(ai_corr.median()),
        "c4_delta_pct": float((human_corr - ai_corr).median()),
        "c4_ci_low": corrected_ci[0],
        "c4_ci_high": corrected_ci[1],
    }


def normalized_mad_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["human_nmads_pct"] = np.where(
        np.abs(out["human_median"]) > 0,
        100.0 * out["human_med_abs_dev"] / np.abs(out["human_median"]),
        np.nan,
    )
    out["ai_nmads_pct"] = np.where(
        np.abs(out["ai_median"]) > 0,
        100.0 * out["ai_med_abs_dev"] / np.abs(out["ai_median"]),
        np.nan,
    )
    out["nmad_delta_pct"] = out["human_nmads_pct"] - out["ai_nmads_pct"]
    out = out.loc[out["human_nmads_pct"].notna() & out["ai_nmads_pct"].notna()].copy()
    return out


def main() -> None:
    deviations = pd.read_parquet(DATA_DIR / "nsclc_io_within_patient_deviations.parquet").copy()
    overlap = pd.read_parquet(DATA_DIR / "shared_feature_bridge_nsclc_io.parquet").copy()
    icc = pd.read_parquet(DATA_DIR / "icc_results.parquet").copy()
    icc_qc = pd.read_parquet(DATA_DIR / "icc_results_qc_pass.parquet").copy()
    geometry_pairwise = pd.read_csv(TABLES_DIR / "nsclc_benchmark_geometry_pairwise_2026-04-21.csv")
    qc_flags = pd.read_csv(TABLES_DIR / "qc_pass_flag_summary.csv")
    qc_compare = pd.read_csv(TABLES_DIR / "qc_pass_sensitivity_comparison.csv")
    qc_rank = pd.read_csv(TABLES_DIR / "qc_pass_rank_stability.csv")
    scanner = pd.read_csv(TABLES_DIR / "scanner_demographics.csv")

    paired = deviations.loc[deviations["human_cov_pct"].notna() & deviations["ai_cov_pct"].notna()].copy()
    paired["family"] = pd.Categorical(paired["family"], categories=FAMILY_ORDER, ordered=True)
    paired["cov_delta_pct"] = paired["human_cov_pct"] - paired["ai_cov_pct"]
    paired_patients = sorted(paired["patient_id"].astype(str).unique())
    human_only_patients = sorted(
        set(deviations["patient_id"].astype(str).unique()) - set(paired_patients)
    )

    overlap_paired = overlap.loc[
        overlap["human_cov_pct"].notna() & overlap["ai_cov_pct"].notna() & overlap["patient_id"].isin(paired_patients)
    ].copy()
    overlap_paired["family"] = pd.Categorical(overlap_paired["family"], categories=FAMILY_ORDER, ordered=True)
    overlap_paired["cov_delta_pct"] = overlap_paired["human_cov_pct"] - overlap_paired["ai_cov_pct"]

    within_patient_summary = summarize_pairwise(paired)
    atomic_write_csv(within_patient_summary, TABLES_DIR / "nsclc_io_within_patient_summary.csv")

    patient_support, patient_family = summarize_patient_collapsed_family(paired)
    atomic_write_csv(patient_support, TABLES_DIR / "nsclc_io_patient_collapsed_family_support.csv")

    overlap_patient_support, overlap_patient_family = summarize_patient_collapsed_family(overlap_paired)

    family_balanced_rows = [
        summarize_family_balanced(
            patient_family,
            analysis_set="full",
            notes="Patient-collapsed equal-family mean delta within each patient on the true 20-patient paired benchmark.",
        ),
        summarize_family_balanced(
            overlap_patient_family,
            analysis_set="overlap_107",
            notes="Bridge-restricted patient-collapsed equal-family mean delta within each patient on the true 20-patient paired benchmark; retains all 7 families.",
        ),
    ]
    family_balanced_rows[0]["delta_retention_vs_full"] = 1.0
    family_balanced_rows[1]["delta_retention_vs_full"] = (
        family_balanced_rows[1]["median_balanced_delta_pct"] / family_balanced_rows[0]["median_balanced_delta_pct"]
        if family_balanced_rows[0]["median_balanced_delta_pct"] != 0
        else np.nan
    )
    family_balanced = pd.DataFrame(family_balanced_rows)
    atomic_write_csv(family_balanced, TABLES_DIR / "family_balanced_nsclc_summary_2026-04-21.csv")

    full_boot = bootstrap_current_and_c4(paired)
    overlap_boot = bootstrap_current_and_c4(overlap_paired)

    comparison_rows: list[dict[str, object]] = []
    all_families = ["overall"] + FAMILY_ORDER
    for family in all_families:
        full_sub = paired if family == "overall" else paired.loc[paired["family"] == family]
        overlap_sub = overlap_paired if family == "overall" else overlap_paired.loc[overlap_paired["family"] == family]
        full_stat, full_p = safe_wilcoxon(full_sub["cov_delta_pct"])
        overlap_stat, overlap_p = safe_wilcoxon(overlap_sub["cov_delta_pct"])
        if family == "overall":
            full_ci = (full_boot["current_ci_low"], full_boot["current_ci_high"])
            overlap_ci = (overlap_boot["current_ci_low"], overlap_boot["current_ci_high"])
        else:
            full_ci = patient_cluster_bootstrap_median(full_sub, "cov_delta_pct")
            overlap_ci = patient_cluster_bootstrap_median(overlap_sub, "cov_delta_pct")
        full_delta = float(full_sub["cov_delta_pct"].median())
        overlap_delta = float(overlap_sub["cov_delta_pct"].median())
        comparison_rows.append(
            {
                "analysis_set_full": "full",
                "family": family,
                "n_pairs_full": int(len(full_sub)),
                "n_subjects_full": int(full_sub["patient_id"].nunique()),
                "n_features_full": int(full_sub["feature_name"].nunique()),
                "human_median_cov_pct_full": float(full_sub["human_cov_pct"].median()),
                "ai_median_cov_pct_full": float(full_sub["ai_cov_pct"].median()),
                "median_paired_delta_pct_full": full_delta,
                "pct_pairs_ai_lower_full": float((full_sub["cov_delta_pct"] > 0).mean() * 100.0),
                "wilcoxon_stat_full": full_stat,
                "wilcoxon_p_full": full_p,
                "ci_low_full": full_ci[0],
                "ci_high_full": full_ci[1],
                "analysis_set_overlap": "overlap_107",
                "n_pairs_overlap": int(len(overlap_sub)),
                "n_subjects_overlap": int(overlap_sub["patient_id"].nunique()),
                "n_features_overlap": int(overlap_sub["feature_name"].nunique()),
                "human_median_cov_pct_overlap": float(overlap_sub["human_cov_pct"].median()),
                "ai_median_cov_pct_overlap": float(overlap_sub["ai_cov_pct"].median()),
                "median_paired_delta_pct_overlap": overlap_delta,
                "pct_pairs_ai_lower_overlap": float((overlap_sub["cov_delta_pct"] > 0).mean() * 100.0),
                "wilcoxon_stat_overlap": overlap_stat,
                "wilcoxon_p_overlap": overlap_p,
                "ci_low_overlap": overlap_ci[0],
                "ci_high_overlap": overlap_ci[1],
                "effect_size_retention_fraction": overlap_delta / full_delta if full_delta != 0 else np.nan,
                "effect_size_retention_pct": (overlap_delta / full_delta * 100.0) if full_delta != 0 else np.nan,
                "delta_change_pct_points": overlap_delta - full_delta,
                "direction_consistent": bool(np.sign(overlap_delta) == np.sign(full_delta) or np.isclose(overlap_delta, 0.0)),
                "ai_advantage_persists_overlap": bool(overlap_delta > 0 and overlap_ci[1] > 0),
            }
        )
    bridge_vs_full = pd.DataFrame(comparison_rows)
    atomic_write_csv(bridge_vs_full, TABLES_DIR / "shared_feature_bridge_vs_full.csv")

    cluster_rows: list[dict[str, object]] = [
        {
            "component": "overall_cluster_bootstrap",
            "analysis_set": "full_1223",
            "family": "overall",
            "n_subjects": int(paired["patient_id"].nunique()),
            "n_features": int(paired["feature_name"].nunique()),
            "current_human_cov_pct": full_boot["current_human_cov_pct"],
            "current_ai_cov_pct": full_boot["current_ai_cov_pct"],
            "current_delta_pct": full_boot["current_delta_pct"],
            "current_ci_low": full_boot["current_ci_low"],
            "current_ci_high": full_boot["current_ci_high"],
            "current_direction_support": "current paired patient-cluster bootstrap CI excludes 0",
            "c4_human_cov_pct": full_boot["c4_human_cov_pct"],
            "c4_ai_cov_pct": full_boot["c4_ai_cov_pct"],
            "c4_delta_pct": full_boot["c4_delta_pct"],
            "c4_ci_low": full_boot["c4_ci_low"],
            "c4_ci_high": full_boot["c4_ci_high"],
            "c4_direction_support": "10k patient bootstrap on the true paired benchmark",
            "source_paths": (
                f"{TABLES_DIR / 'shared_feature_bridge_vs_full.csv'};"
                f"{DATA_DIR / 'nsclc_io_within_patient_deviations.parquet'}"
            ),
        },
        {
            "component": "overall_cluster_bootstrap",
            "analysis_set": "overlap_107",
            "family": "overall",
            "n_subjects": int(overlap_paired["patient_id"].nunique()),
            "n_features": int(overlap_paired["feature_name"].nunique()),
            "current_human_cov_pct": overlap_boot["current_human_cov_pct"],
            "current_ai_cov_pct": overlap_boot["current_ai_cov_pct"],
            "current_delta_pct": overlap_boot["current_delta_pct"],
            "current_ci_low": overlap_boot["current_ci_low"],
            "current_ci_high": overlap_boot["current_ci_high"],
            "current_direction_support": "current paired patient-cluster bootstrap CI excludes 0",
            "c4_human_cov_pct": overlap_boot["c4_human_cov_pct"],
            "c4_ai_cov_pct": overlap_boot["c4_ai_cov_pct"],
            "c4_delta_pct": overlap_boot["c4_delta_pct"],
            "c4_ci_low": overlap_boot["c4_ci_low"],
            "c4_ci_high": overlap_boot["c4_ci_high"],
            "c4_direction_support": "10k patient bootstrap on the true paired overlap benchmark",
            "source_paths": (
                f"{TABLES_DIR / 'shared_feature_bridge_vs_full.csv'};"
                f"{DATA_DIR / 'shared_feature_bridge_nsclc_io.parquet'}"
            ),
        },
    ]

    c4_human = c4(5)
    c4_ai = c4(3)
    patient_family_c4 = patient_family.copy()
    patient_family_c4["patient_median_delta_c4"] = (
        patient_family_c4["patient_median_human"] / c4_human
        - patient_family_c4["patient_median_ai"] / c4_ai
    )
    full_medians = within_patient_summary.set_index("feature_family")
    for row in patient_support.to_dict(orient="records"):
        family = row["family"]
        family_c4 = patient_family_c4.loc[patient_family_c4["family"] == family, "patient_median_delta_c4"]
        c4_pos, c4_neg, c4_zero, c4_p = sign_support(family_c4)
        cluster_rows.append(
            {
                "component": "patient_collapsed_sign",
                "analysis_set": "full_1223",
                "family": family,
                "n_subjects": int(row["n_patients"]),
                "n_features": int(full_medians.loc[family, "n_features"]),
                "current_human_cov_pct": float(full_medians.loc[family, "median_human_cov_pct"]),
                "current_ai_cov_pct": float(full_medians.loc[family, "median_ai_cov_pct"]),
                "current_delta_pct": float(row["median_patient_median_delta"]),
                "current_ci_low": np.nan,
                "current_ci_high": np.nan,
                "current_direction_support": (
                    f"{row['n_positive']}/{row['n_patients']} positive; "
                    f"{row['n_negative']} negative; {row['n_zero']} zero; "
                    f"one-sided sign p={row['sign_test_p_greater']:.3g}"
                ),
                "c4_human_cov_pct": np.nan,
                "c4_ai_cov_pct": np.nan,
                "c4_delta_pct": float(pd.Series(family_c4).median()),
                "c4_ci_low": np.nan,
                "c4_ci_high": np.nan,
                "c4_direction_support": (
                    f"{c4_pos}/{row['n_patients']} positive; {c4_neg} negative; {c4_zero} zero; "
                    f"one-sided sign p={c4_p:.3g}"
                ),
                "source_paths": (
                    f"{TABLES_DIR / 'nsclc_io_patient_collapsed_family_support.csv'};"
                    f"{TABLES_DIR / 'nsclc_io_within_patient_summary.csv'};"
                    f"{DATA_DIR / 'nsclc_io_within_patient_deviations.parquet'}"
                ),
            }
        )

    triplet_summary = pd.read_csv(TABLES_DIR / "nsclc_io_triplet_matched_summary.csv")
    triplet_overall = triplet_summary.loc[triplet_summary["scope"] == "overall"].iloc[0]
    cluster_rows.append(
        {
            "component": "triplet_matched",
            "analysis_set": "matched_20",
            "family": "overall",
            "n_subjects": int(triplet_overall["shared_patients"]),
            "n_features": int(paired["feature_name"].nunique()),
            "current_human_cov_pct": float(triplet_overall["median_human_cov_pct"]),
            "current_ai_cov_pct": float(triplet_overall["median_ai_cov_pct"]),
            "current_delta_pct": float(triplet_overall["median_triplet_delta_pct"]),
            "current_ci_low": np.nan,
            "current_ci_high": np.nan,
            "current_direction_support": (
                f"{int(triplet_overall['triplets_with_positive_delta'])}/{int(triplet_overall['human_triplets_total'])} positive triplets; "
                f"delta range {triplet_overall['min_triplet_delta_pct']:.4f} to {triplet_overall['max_triplet_delta_pct']:.4f}; "
                f"median AI-lower fraction {triplet_overall['median_ai_lower_fraction_pct']:.1f}%"
            ),
            "c4_human_cov_pct": np.nan,
            "c4_ai_cov_pct": np.nan,
            "c4_delta_pct": np.nan,
            "c4_ci_low": np.nan,
            "c4_ci_high": np.nan,
            "c4_direction_support": np.nan,
            "source_paths": (
                f"{TABLES_DIR / 'nsclc_io_triplet_matched_summary.csv'};"
                f"{DATA_DIR / 'nsclc_io_triplet_matched_sensitivity.parquet'}"
            ),
        }
    )
    cluster_aware = pd.DataFrame(cluster_rows)
    atomic_write_csv(cluster_aware, TABLES_DIR / "cluster_aware_nsclc_inference_2026-04-21.csv")

    geometry_sub = geometry_pairwise.loc[
        (geometry_pairwise["scope"] == "benchmark_matched_5v3_inferred_auto123")
        & (geometry_pairwise["patient_id"].isin(paired_patients))
    ].copy()
    geom_patient = (
        geometry_sub.groupby(["patient_id", "arm"], as_index=False)
        .agg(
            patient_median_dice=("dice", "median"),
            patient_median_volume_ratio=("volume_ratio_max_over_min", "median"),
            pair_count=("dice", "size"),
        )
        .pivot(index="patient_id", columns="arm", values=["patient_median_dice", "patient_median_volume_ratio", "pair_count"])
    )
    geom_patient.columns = [
        f"{metric}_{arm}" for metric, arm in geom_patient.columns.to_flat_index()
    ]
    geom_patient = geom_patient.reset_index()
    patient_radiomics = (
        paired.groupby("patient_id", as_index=False)
        .agg(
            median_human_cov_pct=("human_cov_pct", "median"),
            median_ai_cov_pct=("ai_cov_pct", "median"),
            median_delta_pct=("cov_delta_pct", "median"),
            pct_pairs_ai_lower=("cov_delta_pct", lambda s: float((s > 0).mean() * 100.0)),
        )
    )
    geom_bridge_patient = geom_patient.merge(patient_radiomics, on="patient_id", how="inner")
    geom_bridge_patient["delta_dice_ai_minus_human"] = (
        geom_bridge_patient["patient_median_dice_auto"] - geom_bridge_patient["patient_median_dice_human"]
    )
    geom_bridge_patient["delta_volume_ratio_human_minus_ai"] = (
        geom_bridge_patient["patient_median_volume_ratio_human"] - geom_bridge_patient["patient_median_volume_ratio_auto"]
    )
    atomic_write_csv(
        geom_bridge_patient.sort_values("patient_id"),
        TABLES_DIR / "nsclc_benchmark_geometry_radiomics_bridge_patient_2026-04-23.csv",
    )

    dice_rho = spearmanr(
        geom_bridge_patient["median_delta_pct"],
        geom_bridge_patient["delta_dice_ai_minus_human"],
        nan_policy="omit",
    )
    vr_rho = spearmanr(
        geom_bridge_patient["median_delta_pct"],
        geom_bridge_patient["delta_volume_ratio_human_minus_ai"],
        nan_policy="omit",
    )
    geometry_summary = pd.DataFrame(
        [
            {
                "comparison": "human_internal_geometry_paired20",
                "patients": int(geom_bridge_patient["patient_id"].nunique()),
                "median_dice": float(geom_bridge_patient["patient_median_dice_human"].median()),
                "median_volume_ratio": float(geom_bridge_patient["patient_median_volume_ratio_human"].median()),
                "spearman_rho_vs_radiomic_benefit": np.nan,
                "spearman_p_vs_radiomic_benefit": np.nan,
            },
            {
                "comparison": "ai_internal_geometry_paired20",
                "patients": int(geom_bridge_patient["patient_id"].nunique()),
                "median_dice": float(geom_bridge_patient["patient_median_dice_auto"].median()),
                "median_volume_ratio": float(geom_bridge_patient["patient_median_volume_ratio_auto"].median()),
                "spearman_rho_vs_radiomic_benefit": np.nan,
                "spearman_p_vs_radiomic_benefit": np.nan,
            },
            {
                "comparison": "radiomics_vs_delta_dice",
                "patients": int(geom_bridge_patient["patient_id"].nunique()),
                "median_dice": np.nan,
                "median_volume_ratio": np.nan,
                "spearman_rho_vs_radiomic_benefit": float(dice_rho.statistic),
                "spearman_p_vs_radiomic_benefit": float(dice_rho.pvalue),
            },
            {
                "comparison": "radiomics_vs_delta_volume_ratio",
                "patients": int(geom_bridge_patient["patient_id"].nunique()),
                "median_dice": np.nan,
                "median_volume_ratio": np.nan,
                "spearman_rho_vs_radiomic_benefit": float(vr_rho.statistic),
                "spearman_p_vs_radiomic_benefit": float(vr_rho.pvalue),
            },
        ]
    )
    atomic_write_csv(geometry_summary, TABLES_DIR / "nsclc_benchmark_geometry_radiomics_bridge_2026-04-23.csv")

    overlap_feature_stats = (
        overlap_paired.assign(family=overlap_paired["family"].astype(str))
        .groupby(["feature_name", "family"], as_index=False, observed=True)
        .agg(
            benchmark_human_cov_pct=("human_cov_pct", "median"),
            benchmark_ai_cov_pct=("ai_cov_pct", "median"),
            benchmark_delta_pct=("cov_delta_pct", "median"),
            benchmark_ai_lower_pct=("cov_delta_pct", lambda s: float((s > 0).mean() * 100.0)),
        )
    )
    thorax_feature_stats = (
        icc.loc[icc["body_region"] == "Thorax"]
        .groupby(["feature_name", "feature_family"], as_index=False)
        .agg(
            thorax_median_icc=("icc", "median"),
            thorax_median_cov_pct=("cov_percent", "median"),
            thorax_n_rows=("icc", "size"),
            thorax_n_cohorts=("cohort", "nunique"),
        )
        .rename(columns={"feature_family": "family"})
    )
    whitelist = overlap_feature_stats.merge(thorax_feature_stats, on=["feature_name", "family"], how="inner")
    whitelist["benchmark_ai_favorable"] = whitelist["benchmark_delta_pct"] > 0
    whitelist["benchmark_ai_low_cov"] = whitelist["benchmark_ai_cov_pct"] <= 10.0
    whitelist["thorax_robust"] = (whitelist["thorax_median_icc"] >= 0.90) & (whitelist["thorax_median_cov_pct"] <= 10.0)
    whitelist["deployment_ready_shortlist"] = (
        whitelist["benchmark_ai_favorable"] & whitelist["benchmark_ai_low_cov"] & whitelist["thorax_robust"]
    )
    whitelist = whitelist.sort_values(
        ["deployment_ready_shortlist", "family", "thorax_median_icc", "benchmark_delta_pct"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)
    atomic_write_csv(whitelist, TABLES_DIR / "nsclc_thoracic_whitelist_2026-04-23.csv")

    whitelist_family = (
        whitelist.groupby("family", as_index=False)
        .agg(
            overlap_features=("feature_name", "nunique"),
            shortlisted_features=("deployment_ready_shortlist", lambda s: int(np.sum(s))),
        )
        .assign(
            shortlist_fraction=lambda df: df["shortlisted_features"] / df["overlap_features"],
            shortlist_fraction_pct=lambda df: df["shortlist_fraction"] * 100.0,
        )
        .sort_values("family")
    )
    atomic_write_csv(whitelist_family, TABLES_DIR / "nsclc_thoracic_whitelist_family_summary_2026-04-23.csv")

    nmad_rows: list[dict[str, object]] = []
    for analysis_set, sub in [("full_1223", paired), ("overlap_107", overlap_paired)]:
        nmad = normalized_mad_frame(sub)
        ci_low, ci_high = patient_cluster_bootstrap_median(nmad, "nmad_delta_pct")
        nmad_rows.append(
            {
                "analysis_set": analysis_set,
                "patient_feature_pairs": int(len(nmad)),
                "median_human_normalized_mad_pct": float(nmad["human_nmads_pct"].median()),
                "median_ai_normalized_mad_pct": float(nmad["ai_nmads_pct"].median()),
                "median_delta_pct": float(nmad["nmad_delta_pct"].median()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "pct_pairs_ai_lower": float((nmad["nmad_delta_pct"] > 0).mean() * 100.0),
                "notes": "Normalized MAD uses MAD / |median| * 100 as a robust dispersion sensitivity.",
            }
        )
    atomic_write_csv(pd.DataFrame(nmad_rows), TABLES_DIR / "nsclc_normalized_mad_sensitivity_2026-04-23.csv")

    local_case_dirs = sorted(NSCLC_DATASET_ROOT.glob("interobs*/2019-02"))
    denominator_flow = pd.DataFrame(
        [
            {
                "stage": "TCIA public source cases",
                "n_cases": 22,
                "scope": "benchmark",
                "notes": "Published NSCLC-RADIOMICS-INTEROBSERVER1 source cohort size.",
            },
            {
                "stage": "Cases used in Kothari et al benchmark",
                "n_cases": 20,
                "scope": "benchmark",
                "notes": "Original publication benchmark after missing contour exclusions.",
            },
            {
                "stage": "Local benchmark folders present on disk",
                "n_cases": len(local_case_dirs),
                "scope": "local_package",
                "notes": "Count of local 2019-02 benchmark folders under /output.",
            },
            {
                "stage": "Local folders with preserved human contour geometry",
                "n_cases": len(local_case_dirs),
                "scope": "local_package",
                "notes": "All 21 local folders preserve 5 human GTV masks for geometry/provenance audit.",
            },
            {
                "stage": "True paired radiomics benchmark cases",
                "n_cases": len(paired_patients),
                "scope": "paired_radiomics",
                "notes": "Requires both human and AI within-patient radiomic dispersion rows.",
            },
            {
                "stage": "Replicate-matched triplet sensitivity cases",
                "n_cases": 20,
                "scope": "paired_radiomics",
                "notes": "Same 20 paired cases on the raw-feature intersection.",
            },
            {
                "stage": "Human-only local cases outside paired benchmark",
                "n_cases": len(human_only_patients),
                "scope": "exclusions",
                "notes": "interobs19 remains outside the paired benchmark because the frozen AI tumor cache / archived auto masks are not recoverable.",
            },
        ]
    )
    atomic_write_csv(denominator_flow, TABLES_DIR / "nsclc_benchmark_denominator_flow_2026-04-23.csv")

    thorax_scanner = scanner.loc[scanner["body_region"] == "Thorax", ["cohort", "manufacturer_distribution", "most_common_kernel", "slice_thickness_mm", "n_unique_kernels"]].copy()
    thorax_scanner["cohort"] = thorax_scanner["cohort"].replace(
        {"NSCLC_Interobserver": "NSCLC_RADIOMICS_INTEROBSERVER1"}
    )
    qc_mechanism = (
        qc_compare.loc[qc_compare["summary_level"] == "body_region", [
            "body_region",
            "feature_family",
            "median_cov_percent_all_data",
            "median_cov_percent_qc_pass",
            "robust_rate_pct_all_data",
            "robust_rate_pct_qc_pass",
            "row_retention_pct",
        ]]
        .merge(qc_rank, on="body_region", how="left")
        .assign(
            qc_cov_worsened=lambda df: df["median_cov_percent_qc_pass"] > df["median_cov_percent_all_data"],
            qc_robust_rate_worsened=lambda df: df["robust_rate_pct_qc_pass"] < df["robust_rate_pct_all_data"],
        )
        .sort_values(["body_region", "feature_family"])
    )
    atomic_write_csv(qc_mechanism, TABLES_DIR / "qc_mechanism_summary_2026-04-23.csv")

    # Also emit a compact thoracic scanner-context table because the discussion uses it.
    atomic_write_csv(thorax_scanner, TABLES_DIR / "thorax_scanner_context_2026-04-23.csv")


if __name__ == "__main__":
    main()
