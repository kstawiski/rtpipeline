#!/usr/bin/env python3
"""Merge Immunodozymetria + PlucaRCHT into one manuscript-facing Lung cancer cohort.

This is a manuscript packaging step. It updates the local repo analysis mirrors that
drive the submission figures/tables without touching the upstream raw stores.

Policy
------
- The merged cohort is named ``Lung cancer`` in all regenerated outputs.
- Feature-level all-data and QC-pass ICC rows are collapsed with ``n_subjects``-
  weighted averages within each shared ``(feature_name, roi_name)`` cell.
- Human-facing denominators use the reviewed deduplicated cohort ledger:
    2 exact cross-cohort lung patient-course overlaps
    -> 194 analysis-ready patients
    -> 211 course-level perturbation inputs
    -> 212 metadata-inventory entries
"""

from __future__ import annotations

import math
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


PROJECT_ROOT = Path("/umed-projekty/rtpipeline")
ANALYSIS_DIR = PROJECT_ROOT / "manuscript" / "analysis"
DATA_DIR = ANALYSIS_DIR / "data"
TABLES_DIR = ANALYSIS_DIR / "tables"

ALL_ICC_PATH = DATA_DIR / "icc_results.parquet"
QC_ICC_PATH = DATA_DIR / "icc_results_qc_pass.parquet"

OLD_LUNG = ["Immunodozymetria", "PlucaRCHT"]
MERGED_LUNG = "Lung cancer"

ICC_ROBUST = 0.90
COV_ROBUST = 10.0
ICC_ACCEPTABLE = 0.75
COV_ACCEPTABLE = 20.0

FAMILY_ORDER = ["shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
REGION_ORDER = ["Brain", "Pelvis", "Thorax"]

SCALE_INVARIANT_BASES = {
    "compactness1",
    "compactness2",
    "elongation",
    "flatness",
    "sphericaldisproportion",
    "sphericity",
}
SCALE_DEPENDENT_BASES = {
    "leastaxislength",
    "majoraxislength",
    "maximum3ddiameter",
    "meshvolume",
    "minoraxislength",
    "surfacearea",
    "voxelvolume",
}


def classify_icc_cov(icc: float, cov_pct: float) -> str:
    if not np.isfinite(icc) or not np.isfinite(cov_pct):
        return "Dropped"
    if icc >= ICC_ROBUST and cov_pct <= COV_ROBUST:
        return "Robust"
    if icc >= ICC_ACCEPTABLE and cov_pct <= COV_ACCEPTABLE:
        return "Acceptable"
    return "Poor"


def parse_count_distribution(text: str) -> Counter:
    counter: Counter[str] = Counter()
    if not isinstance(text, str) or not text.strip():
        return counter
    for part in text.split(";"):
        token = part.strip()
        if not token or "(" not in token or not token.endswith(")"):
            continue
        label, count = token.rsplit("(", 1)
        label = label.strip()
        try:
            n = int(count[:-1])
        except ValueError:
            continue
        counter[label] += n
    return counter


def format_count_distribution(counter: Counter) -> str:
    parts = [f"{label} ({counter[label]})" for label in sorted(counter) if counter[label] > 0]
    return "; ".join(parts)


def parse_summary_range(text: str) -> tuple[float | None, float | None, float | None]:
    if not isinstance(text, str) or not text.strip() or text == "N/A":
        return (None, None, None)
    try:
        med_str, rng = text.split(" [", 1)
        lo_str, hi_str = rng[:-1].split("-", 1)
        return (float(med_str), float(lo_str), float(hi_str))
    except Exception:
        return (None, None, None)


def format_summary_range(median: float | None, low: float | None, high: float | None, digits: int) -> str:
    if median is None or low is None or high is None:
        return "N/A"
    fmt = f"{{:.{digits}f}}"
    return f"{fmt.format(median)} [{fmt.format(low)}-{fmt.format(high)}]"


def weighted_range_merge(values: list[str], weights: list[float], digits: int) -> str:
    parsed = [parse_summary_range(v) for v in values]
    valid = [(med, lo, hi, w) for (med, lo, hi), w in zip(parsed, weights, strict=False) if med is not None]
    if not valid:
        return "N/A"
    total_w = sum(w for _, _, _, w in valid)
    med = sum(m * w for m, _, _, w in valid) / total_w
    low = min(lo for _, lo, _, _ in valid)
    high = max(hi for _, _, hi, _ in valid)
    return format_summary_range(med, low, high, digits)


def feature_base_name(feature_name: str) -> str:
    lower = str(feature_name).lower()
    marker = "_shape_"
    if marker in lower:
        return lower.split(marker, 1)[1]
    return lower.rsplit("_", 1)[-1]


def classify_shape_scale(feature_name: str) -> str:
    base = feature_base_name(feature_name)
    if base in SCALE_INVARIANT_BASES:
        return "scale_invariant"
    if base in SCALE_DEPENDENT_BASES or base.startswith("maximum2ddiameter"):
        return "scale_dependent"
    return "unclassified"


def percentile_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    return (
        float(np.quantile(arr, alpha / 2.0)),
        float(np.quantile(arr, 1.0 - alpha / 2.0)),
    )


def bootstrap_median_se(values: np.ndarray, n_boot: int = 500, seed: int = 42) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        draws[idx] = np.median(rng.choice(values, size=values.size, replace=True))
    return float(np.std(draws, ddof=1))


def merge_icc_points(df: pd.DataFrame) -> pd.DataFrame:
    lung_mask = df["cohort"].isin(OLD_LUNG)
    non_lung = df.loc[~lung_mask].copy()
    lung = df.loc[lung_mask].copy()

    group_cols = ["feature_name", "roi_name", "body_region", "feature_family", "image_type"]
    rows: list[dict[str, object]] = []
    numeric_weighted = [
        col
        for col in ["icc", "icc_ci_low", "icc_ci_high", "cov_percent", "qcd", "mean_value", "sd_value"]
        if col in df.columns
    ]

    for key, sub in lung.groupby(group_cols, sort=False):
        weights = pd.to_numeric(sub["n_subjects"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        weights = np.where(weights > 0, weights, 1.0)
        row = {col: value for col, value in zip(group_cols, key, strict=False)}
        row["cohort"] = MERGED_LUNG
        row["n_subjects"] = int(round(pd.to_numeric(sub["n_subjects"], errors="coerce").fillna(0).sum()))
        row["n_raters"] = int(pd.to_numeric(sub["n_raters"], errors="coerce").max())
        for col in numeric_weighted:
            vals = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(vals)
            row[col] = float(np.average(vals[mask], weights=weights[mask])) if mask.any() else float("nan")
        row["classification"] = classify_icc_cov(row["icc"], row["cov_percent"])
        rows.append(row)

    merged = pd.concat([non_lung, pd.DataFrame(rows)], ignore_index=True)
    merged = merged.sort_values(["body_region", "cohort", "roi_name", "feature_name"]).reset_index(drop=True)
    return merged


def summarise_points(points: pd.DataFrame, analysis_set: str) -> pd.DataFrame:
    df = points.copy()
    df["analysis_set"] = analysis_set
    df["classification"] = df["classification"].fillna("Dropped")
    df["is_robust"] = df["classification"].eq("Robust")
    df["is_acceptable"] = df["classification"].eq("Acceptable")
    df["is_poor"] = df["classification"].eq("Poor")
    df["is_dropped"] = df["classification"].eq("Dropped")
    df["has_metric"] = df["icc"].notna() & df["cov_percent"].notna()

    rows: list[dict[str, object]] = []
    for (cohort, body_region, feature_family), sub in df.groupby(["cohort", "body_region", "feature_family"], sort=False):
        rows.append(
            {
                "summary_level": "cohort",
                "analysis_set": analysis_set,
                "cohort": cohort,
                "body_region": body_region,
                "feature_family": feature_family,
                "row_count_total": int(len(sub)),
                "row_count_nonmissing": int(sub["has_metric"].sum()),
                "roi_count": int(sub["roi_name"].nunique()),
                "feature_count": int(sub["feature_name"].nunique()),
                "median_icc": float(sub["icc"].median(skipna=True)),
                "median_cov_percent": float(sub["cov_percent"].median(skipna=True)),
                "robust_rate_pct": float(100.0 * sub["is_robust"].mean()),
                "acceptable_rate_pct": float(100.0 * sub["is_acceptable"].mean()),
                "poor_rate_pct": float(100.0 * sub["is_poor"].mean()),
                "dropped_rate_pct": float(100.0 * sub["is_dropped"].mean()),
            }
        )
    cohort_summary = pd.DataFrame(rows)

    agg_rows: list[dict[str, object]] = []
    for (body_region, feature_family), sub in df.groupby(["body_region", "feature_family"], sort=False):
        agg_rows.append(
            {
                "summary_level": "body_region",
                "analysis_set": analysis_set,
                "cohort": np.nan,
                "body_region": body_region,
                "feature_family": feature_family,
                "row_count_total": int(len(sub)),
                "row_count_nonmissing": int(sub["has_metric"].sum()),
                "roi_count": int(sub["roi_name"].nunique()),
                "feature_count": int(sub["feature_name"].nunique()),
                "median_icc": float(sub["icc"].median(skipna=True)),
                "median_cov_percent": float(sub["cov_percent"].median(skipna=True)),
                "robust_rate_pct": float(100.0 * sub["is_robust"].mean()),
                "acceptable_rate_pct": float(100.0 * sub["is_acceptable"].mean()),
                "poor_rate_pct": float(100.0 * sub["is_poor"].mean()),
                "dropped_rate_pct": float(100.0 * sub["is_dropped"].mean()),
            }
        )
    for feature_family, sub in df.groupby("feature_family", sort=False):
        agg_rows.append(
            {
                "summary_level": "overall",
                "analysis_set": analysis_set,
                "cohort": np.nan,
                "body_region": "Overall",
                "feature_family": feature_family,
                "row_count_total": int(len(sub)),
                "row_count_nonmissing": int(sub["has_metric"].sum()),
                "roi_count": int(sub["roi_name"].nunique()),
                "feature_count": int(sub["feature_name"].nunique()),
                "median_icc": float(sub["icc"].median(skipna=True)),
                "median_cov_percent": float(sub["cov_percent"].median(skipna=True)),
                "robust_rate_pct": float(100.0 * sub["is_robust"].mean()),
                "acceptable_rate_pct": float(100.0 * sub["is_acceptable"].mean()),
                "poor_rate_pct": float(100.0 * sub["is_poor"].mean()),
                "dropped_rate_pct": float(100.0 * sub["is_dropped"].mean()),
            }
        )
    return pd.concat([cohort_summary, pd.DataFrame(agg_rows)], ignore_index=True)


def build_comparison_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    keep = summary_df.loc[summary_df["summary_level"].isin(["body_region", "overall"])].copy()
    all_df = keep.loc[keep["analysis_set"] == "all_data"].copy()
    qc_df = keep.loc[keep["analysis_set"] == "qc_pass"].copy()
    join_cols = ["summary_level", "body_region", "feature_family"]
    merged = all_df.merge(qc_df, on=join_cols, how="outer", suffixes=("_all_data", "_qc_pass"))
    for metric in [
        "median_icc",
        "median_cov_percent",
        "robust_rate_pct",
        "acceptable_rate_pct",
        "poor_rate_pct",
        "dropped_rate_pct",
        "row_count_nonmissing",
        "row_count_total",
    ]:
        merged[f"delta_{metric}"] = merged[f"{metric}_qc_pass"] - merged[f"{metric}_all_data"]
    merged["row_retention_pct"] = (
        100.0
        * merged["row_count_nonmissing_qc_pass"]
        / merged["row_count_nonmissing_all_data"].replace(0, np.nan)
    )
    return merged


def build_rank_stability(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keep = summary_df.loc[summary_df["summary_level"] == "body_region"].copy()
    for body_region in REGION_ORDER:
        base = keep.loc[(keep["body_region"] == body_region) & (keep["analysis_set"] == "all_data")]
        qc = keep.loc[(keep["body_region"] == body_region) & (keep["analysis_set"] == "qc_pass")]
        merged = base.merge(qc, on="feature_family", suffixes=("_all_data", "_qc_pass"))
        if len(merged) >= 2:
            rho_icc = float(sp_stats.spearmanr(merged["median_icc_all_data"], merged["median_icc_qc_pass"]).statistic)
            rho_robust = float(
                sp_stats.spearmanr(
                    merged["robust_rate_pct_all_data"],
                    merged["robust_rate_pct_qc_pass"],
                ).statistic
            )
        else:
            rho_icc = float("nan")
            rho_robust = float("nan")
        rows.append(
            {
                "body_region": body_region,
                "spearman_rho_median_icc": rho_icc,
                "spearman_rho_robust_rate_pct": rho_robust,
                "n_families_compared": int(len(merged)),
            }
        )
    return pd.DataFrame(rows)


def compute_cohort_effects(icc_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (region, family, cohort), grp in icc_df.groupby(["body_region", "feature_family", "cohort"], sort=False):
        vals = grp["icc"].dropna().to_numpy(dtype=float)
        if vals.size < 3:
            continue
        med_icc = float(np.median(vals))
        se = bootstrap_median_se(vals, n_boot=500, seed=42)
        rows.append(
            {
                "body_region": region,
                "feature_family": family,
                "cohort": cohort,
                "n_rows": int(vals.size),
                "n_subjects": float(pd.to_numeric(grp["n_subjects"], errors="coerce").median()),
                "median_icc": med_icc,
                "bootstrap_se": se,
                "ci_lower": med_icc - 1.96 * se if np.isfinite(se) else np.nan,
                "ci_upper": med_icc + 1.96 * se if np.isfinite(se) else np.nan,
                "re_weight": np.nan,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["body_region", "feature_family", "cohort"]).reset_index(drop=True)


def dersimonian_laird(thetas: np.ndarray, variances: np.ndarray) -> dict[str, object]:
    k = len(thetas)
    if k < 2:
        nan = float("nan")
        return {
            "Q": nan,
            "df": nan,
            "p_Q": nan,
            "I2": nan,
            "tau2": nan,
            "pooled": nan,
            "pooled_se": nan,
            "pooled_ci": (nan, nan),
            "weights": np.full(k, nan),
        }
    variances = np.maximum(np.asarray(variances, dtype=float), 1e-12)
    thetas = np.asarray(thetas, dtype=float)
    w_fe = 1.0 / variances
    theta_bar_fe = np.sum(w_fe * thetas) / np.sum(w_fe)
    Q = float(np.sum(w_fe * (thetas - theta_bar_fe) ** 2))
    df = k - 1
    p_Q = float(1.0 - sp_stats.chi2.cdf(Q, df=df))
    I2 = float(max(0.0, (Q - df) / Q)) if Q > 0 else 0.0
    c = float(np.sum(w_fe) - np.sum(w_fe**2) / np.sum(w_fe))
    tau2 = float(max(0.0, (Q - df) / c)) if c > 0 else 0.0
    w_re = 1.0 / (variances + tau2)
    pooled = float(np.sum(w_re * thetas) / np.sum(w_re))
    pooled_se = float(np.sqrt(1.0 / np.sum(w_re)))
    return {
        "Q": Q,
        "df": df,
        "p_Q": p_Q,
        "I2": I2,
        "tau2": tau2,
        "pooled": pooled,
        "pooled_se": pooled_se,
        "pooled_ci": (pooled - 1.96 * pooled_se, pooled + 1.96 * pooled_se),
        "weights": w_re,
    }


def compute_heterogeneity_summary(cohort_effects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    effects = cohort_effects.copy()
    for (region, family), grp in effects.groupby(["body_region", "feature_family"], sort=False):
        grp_valid = grp.dropna(subset=["median_icc", "bootstrap_se"])
        if len(grp_valid) < 2:
            continue
        meta = dersimonian_laird(grp_valid["median_icc"].to_numpy(), np.square(grp_valid["bootstrap_se"].to_numpy()))
        effects.loc[grp_valid.index, "re_weight"] = meta["weights"]
        if np.isnan(meta["I2"]):
            hetero_class = "insufficient_k"
        elif meta["I2"] < 0.50:
            hetero_class = "low"
        elif meta["I2"] < 0.75:
            hetero_class = "moderate"
        else:
            hetero_class = "high"
        rows.append(
            {
                "body_region": region,
                "feature_family": family,
                "k_cohorts": int(len(grp_valid)),
                "Q": round(meta["Q"], 4),
                "df": int(meta["df"]),
                "p_Q": round(meta["p_Q"], 4),
                "I2": round(meta["I2"], 4),
                "tau2": round(meta["tau2"], 6),
                "pooled_icc": round(meta["pooled"], 4),
                "pooled_se": round(meta["pooled_se"], 4),
                "pooled_ci_lower": round(meta["pooled_ci"][0], 4),
                "pooled_ci_upper": round(meta["pooled_ci"][1], 4),
                "heterogeneity_class": hetero_class,
            }
        )
    return (
        effects.sort_values(["body_region", "feature_family", "cohort"]).reset_index(drop=True),
        pd.DataFrame(rows).sort_values(["body_region", "feature_family"]).reset_index(drop=True),
    )


def get_feature_family(fname: str) -> str:
    for fam in FAMILY_ORDER:
        if f"_{fam}_" in fname.lower():
            return fam
    return "unknown"


def get_image_type(fname: str) -> str:
    if fname.startswith("original_"):
        return "original"
    if fname.startswith("log-sigma-"):
        return "log-sigma"
    if fname.startswith("wavelet-"):
        return fname.split("_")[0]
    return "other"


def build_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["feature_name", "body_region"], as_index=False)
        .agg(
            median_icc=("icc", "median"),
            median_cov=("cov_percent", "median"),
            n_cohorts=("cohort", "nunique"),
        )
    )
    fam_map = df.groupby("feature_name")["feature_family"].first().to_dict()
    it_map = df.groupby("feature_name")["image_type"].first().to_dict()
    agg["feature_family"] = agg["feature_name"].map(fam_map)
    agg["image_type"] = agg["feature_name"].map(it_map)
    agg["is_robust"] = (agg["median_icc"] >= ICC_ROBUST) & (agg["median_cov"] <= COV_ROBUST)
    return agg.sort_values(["body_region", "median_icc"], ascending=[True, False]).reset_index(drop=True)


def jaccard(set_a: set[str], set_b: set[str]) -> float:
    union = len(set_a | set_b)
    return float("nan") if union == 0 else len(set_a & set_b) / union


def sorensen(set_a: set[str], set_b: set[str]) -> float:
    denom = len(set_a) + len(set_b)
    return float("nan") if denom == 0 else 2 * len(set_a & set_b) / denom


def top_k_features(hier_df: pd.DataFrame, k: int) -> dict[tuple[str, str], set[str]]:
    result: dict[tuple[str, str], set[str]] = {}
    for key, grp in hier_df.groupby(["feature_family", "body_region"], sort=False):
        result[key] = set(grp.nlargest(k, "median_icc")["feature_name"].tolist())
    return result


def build_loco_outputs(icc_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref_hier = build_hierarchy(icc_df)
    ref_idx = ref_hier.set_index(["feature_name", "body_region"])["median_icc"]
    ref_top20 = top_k_features(ref_hier, 20)
    ref_top50 = top_k_features(ref_hier, 50)
    metric_rows: list[dict[str, object]] = []
    fragile_rows: list[dict[str, object]] = []

    for held_out in sorted(icc_df["cohort"].unique().tolist()):
        sub_df = icc_df.loc[icc_df["cohort"] != held_out].copy()
        loco_hier = build_hierarchy(sub_df)
        loco_idx = loco_hier.set_index(["feature_name", "body_region"])["median_icc"]
        common_idx = ref_idx.index.intersection(loco_idx.index)
        if len(common_idx) >= 10:
            ref_vals = ref_idx.loc[common_idx].to_numpy(dtype=float)
            loco_vals = loco_idx.loc[common_idx].to_numpy(dtype=float)
            rho, rho_p = sp_stats.spearmanr(ref_vals, loco_vals)
            tau, tau_p = sp_stats.kendalltau(ref_vals, loco_vals)
        else:
            rho = rho_p = tau = tau_p = float("nan")

        merged = ref_hier[["feature_name", "body_region", "median_icc", "median_cov", "is_robust", "feature_family"]].merge(
            loco_hier[["feature_name", "body_region", "median_icc", "median_cov", "is_robust"]].rename(
                columns={"median_icc": "loco_icc", "median_cov": "loco_cov", "is_robust": "loco_robust"}
            ),
            on=["feature_name", "body_region"],
            how="inner",
        )
        newly_robust = int(((~merged["is_robust"]) & merged["loco_robust"]).sum())
        newly_non_robust = int((merged["is_robust"] & (~merged["loco_robust"])).sum())

        def avg_overlap(ref_map: dict[tuple[str, str], set[str]], k: int) -> tuple[float, float]:
            loco_map = top_k_features(loco_hier, k)
            j_vals = []
            s_vals = []
            for key, ref_set in ref_map.items():
                loco_set = loco_map.get(key, set())
                j_vals.append(jaccard(ref_set, loco_set))
                s_vals.append(sorensen(ref_set, loco_set))
            return (float(np.nanmean(j_vals)), float(np.nanmean(s_vals)))

        j20, s20 = avg_overlap(ref_top20, 20)
        j50, s50 = avg_overlap(ref_top50, 50)

        metric_rows.append(
            {
                "held_out_cohort": held_out,
                "n_remaining_cohorts": int(sub_df["cohort"].nunique()),
                "n_common_features": int(len(common_idx)),
                "spearman_rho": float(rho),
                "spearman_p": float(rho_p),
                "kendall_tau": float(tau),
                "kendall_p": float(tau_p),
                "jaccard_top20": j20,
                "sorensen_top20": s20,
                "jaccard_top50": j50,
                "sorensen_top50": s50,
                "newly_robust": newly_robust,
                "newly_non_robust": newly_non_robust,
                "n_features_merged": int(len(merged)),
                "material_instability": bool(np.isfinite(rho) and rho < 0.95),
            }
        )

        loco_top20 = top_k_features(loco_hier, 20)
        for (fam, region), ref_set in ref_top20.items():
            dropped = ref_set - loco_top20.get((fam, region), set())
            for fname in dropped:
                ref_row = ref_hier.loc[(ref_hier["feature_name"] == fname) & (ref_hier["body_region"] == region)]
                loco_row = loco_hier.loc[(loco_hier["feature_name"] == fname) & (loco_hier["body_region"] == region)]
                ref_icc = float(ref_row["median_icc"].iloc[0]) if not ref_row.empty else np.nan
                loco_icc = float(loco_row["median_icc"].iloc[0]) if not loco_row.empty else np.nan
                fragile_rows.append(
                    {
                        "held_out_cohort": held_out,
                        "body_region": region,
                        "feature_family": fam,
                        "feature_name": fname,
                        "ref_median_icc": ref_icc,
                        "loco_median_icc": loco_icc,
                        "icc_drop": ref_icc - loco_icc if np.isfinite(loco_icc) else np.nan,
                    }
                )

    df = pd.DataFrame(metric_rows).sort_values("held_out_cohort").reset_index(drop=True)
    numeric_cols = [
        "spearman_rho",
        "kendall_tau",
        "jaccard_top20",
        "sorensen_top20",
        "jaccard_top50",
        "sorensen_top50",
        "newly_robust",
        "newly_non_robust",
    ]
    mean_row = {"held_out_cohort": "SUMMARY_mean", "material_instability": bool(df["material_instability"].any())}
    mean_row.update({c: df[c].mean() for c in numeric_cols})
    sd_row = {"held_out_cohort": "SUMMARY_sd", "material_instability": False}
    sd_row.update({c: df[c].std(ddof=1) if len(df) > 1 else np.nan for c in numeric_cols})
    for col in ["n_remaining_cohorts", "n_common_features", "n_features_merged", "spearman_p", "kendall_p"]:
        mean_row[col] = np.nan
        sd_row[col] = np.nan
    loco_table = pd.concat([df, pd.DataFrame([mean_row, sd_row])], ignore_index=True)

    fragile = pd.DataFrame(fragile_rows)
    if not fragile.empty:
        fragile = fragile.sort_values(["held_out_cohort", "body_region", "icc_drop"], ascending=[True, True, False])
        fragile = fragile.groupby(["held_out_cohort", "body_region"], as_index=False, sort=False).head(5).reset_index(drop=True)
    return loco_table, fragile


def build_shape_outputs(icc_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shape_df = icc_df.loc[icc_df["feature_family"].eq("shape")].copy()
    shape_df["feature_base_name"] = shape_df["feature_name"].map(feature_base_name)
    shape_df["scale_class"] = shape_df["feature_name"].map(classify_shape_scale)

    cohort_roi = (
        shape_df.groupby(["cohort", "body_region", "roi_name", "scale_class"], as_index=False)
        .agg(
            median_icc=("icc", "median"),
            median_cov_pct=("cov_percent", "median"),
            n_features=("feature_name", "nunique"),
        )
    )
    cohort_roi["summary_level"] = "cohort_roi"
    cohort_roi["n_cohort_roi_groups"] = np.nan
    cohort_roi["ci_low_icc"] = np.nan
    cohort_roi["ci_high_icc"] = np.nan
    cohort_roi["ci_low_cov_pct"] = np.nan
    cohort_roi["ci_high_cov_pct"] = np.nan

    def bootstrap_summary(scope_df: pd.DataFrame, summary_level: str, body_region: str) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for scale_class, sub in scope_df.groupby("scale_class", sort=False):
            icc_vals = sub["median_icc"].to_numpy(dtype=float)
            cov_vals = sub["median_cov_pct"].to_numpy(dtype=float)
            rng = np.random.default_rng(2604)
            icc_boot = np.empty(1000, dtype=float)
            cov_boot = np.empty(1000, dtype=float)
            for idx in range(1000):
                sel = rng.integers(0, len(sub), size=len(sub))
                icc_boot[idx] = float(np.median(icc_vals[sel]))
                cov_boot[idx] = float(np.median(cov_vals[sel]))
            icc_ci = percentile_ci(icc_boot)
            cov_ci = percentile_ci(cov_boot)
            rows.append(
                {
                    "summary_level": summary_level,
                    "cohort": np.nan,
                    "body_region": body_region,
                    "roi_name": np.nan,
                    "scale_class": scale_class,
                    "median_icc": float(np.median(icc_vals)),
                    "median_cov_pct": float(np.median(cov_vals)),
                    "n_features": np.nan,
                    "n_cohort_roi_groups": int(len(sub)),
                    "ci_low_icc": icc_ci[0],
                    "ci_high_icc": icc_ci[1],
                    "ci_low_cov_pct": cov_ci[0],
                    "ci_high_cov_pct": cov_ci[1],
                }
            )
        return rows

    summary_rows: list[dict[str, object]] = []
    for region in REGION_ORDER:
        region_df = cohort_roi.loc[cohort_roi["body_region"] == region]
        summary_rows.extend(bootstrap_summary(region_df, "body_region", region))
    summary_rows.extend(bootstrap_summary(cohort_roi, "overall", "Overall"))

    summary_df = pd.concat(
        [
            cohort_roi[
                [
                    "summary_level",
                    "cohort",
                    "body_region",
                    "roi_name",
                    "scale_class",
                    "median_icc",
                    "median_cov_pct",
                    "n_features",
                    "n_cohort_roi_groups",
                    "ci_low_icc",
                    "ci_high_icc",
                    "ci_low_cov_pct",
                    "ci_high_cov_pct",
                ]
            ],
            pd.DataFrame(summary_rows),
        ],
        ignore_index=True,
    )

    pairwise_rows: list[dict[str, object]] = []
    for scope in ["Brain", "Pelvis", "Thorax", "Overall"]:
        scope_df = cohort_roi if scope == "Overall" else cohort_roi.loc[cohort_roi["body_region"] == scope]
        for metric, value_col in [("median_icc", "median_icc"), ("median_cov_pct", "median_cov_pct")]:
            pivot = scope_df.pivot_table(
                index=["cohort", "roi_name"],
                columns="scale_class",
                values=value_col,
                aggfunc="first",
            )
            pivot = pivot.dropna(subset=["scale_invariant", "scale_dependent"], how="any")
            diffs = pivot["scale_invariant"] - pivot["scale_dependent"]
            if diffs.empty or np.allclose(diffs.to_numpy(dtype=float), 0.0):
                stat = 0.0
                p_val = 1.0
            else:
                stat, p_val = sp_stats.wilcoxon(diffs.to_numpy(dtype=float), zero_method="wilcox", alternative="two-sided")
            pairwise_rows.append(
                {
                    "comparison_scope": scope,
                    "metric": metric,
                    "n_pairs": f"{len(pivot)} (cohort, roi_name)",
                    "pair_unit": "cluster_median_paired_wilcoxon",
                    "test_label": "cluster_median_paired_wilcoxon",
                    "median_scale_invariant": float(pivot["scale_invariant"].median()),
                    "median_scale_dependent": float(pivot["scale_dependent"].median()),
                    "median_paired_delta": float(diffs.median()),
                    "wilcoxon_stat": float(stat),
                    "wilcoxon_p": float(p_val),
                    "note": "Paired Wilcoxon is performed on (cohort, roi_name)-level class medians. A one-to-one feature-level pairing between scale classes is not well-defined.",
                }
            )
    pairwise_df = pd.DataFrame(pairwise_rows)
    return shape_df, summary_df, pairwise_df


def update_scanner_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    scanner = pd.read_csv(TABLES_DIR / "scanner_demographics.csv")
    keep = scanner.loc[~scanner["cohort"].isin(OLD_LUNG)].copy()
    lung = scanner.loc[scanner["cohort"].isin(OLD_LUNG)].copy()

    counter = Counter()
    for text in lung["manufacturer_distribution"]:
        counter.update(parse_count_distribution(text))
    if counter.get("Siemens", 0) >= 2:
        counter["Siemens"] -= 2

    merged_row = {
        "cohort": MERGED_LUNG,
        "body_region": "Thorax",
        "n_total": int(lung["n_total"].sum() - 2),
        "n_with_scanner_info": int(lung["n_with_scanner_info"].sum() - 2),
        "manufacturer_distribution": format_count_distribution(counter),
        "most_common_model": "Sensation Open",
        "kvp_values": "0/120",
        "slice_thickness_mm": weighted_range_merge(lung["slice_thickness_mm"].tolist(), lung["n_total"].tolist(), 1),
        "pixel_spacing_mm": weighted_range_merge(lung["pixel_spacing_mm"].tolist(), lung["n_total"].tolist(), 3),
        "fov_mm": weighted_range_merge(lung["fov_mm"].tolist(), lung["n_total"].tolist(), 0),
        "most_common_kernel": "B31f",
        "n_unique_kernels": 10,
    }
    scanner_out = pd.concat([keep, pd.DataFrame([merged_row])], ignore_index=True)
    scanner_out = scanner_out.sort_values(["body_region", "cohort"]).reset_index(drop=True)

    thorax = pd.read_csv(TABLES_DIR / "thorax_scanner_context_2026-04-23.csv")
    thorax_keep = thorax.loc[~thorax["cohort"].isin(OLD_LUNG)].copy()
    thorax_row = pd.DataFrame(
        [
            {
                "cohort": MERGED_LUNG,
                "manufacturer_distribution": merged_row["manufacturer_distribution"],
                "most_common_kernel": merged_row["most_common_kernel"],
                "slice_thickness_mm": merged_row["slice_thickness_mm"],
                "n_unique_kernels": merged_row["n_unique_kernels"],
            }
        ]
    )
    thorax_out = pd.concat([thorax_keep, thorax_row], ignore_index=True)
    thorax_out = thorax_out.sort_values("cohort").reset_index(drop=True)
    return scanner_out, thorax_out


def update_qc_flag_summary() -> pd.DataFrame:
    path = TABLES_DIR / "qc_pass_flag_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    qc = pd.read_csv(path)
    keep = qc.loc[~qc["cohort"].isin(OLD_LUNG) & ~qc["cohort"].eq("Overall")].copy()
    lung = qc.loc[qc["cohort"].isin(OLD_LUNG)].copy()
    merged_row = {
        "cohort": MERGED_LUNG,
        "body_region": "Thorax",
        "n_structure_instances": int(lung["n_structure_instances"].sum()),
        "n_nonempty": int(lung["n_nonempty"].sum()),
        "n_cropped": int(lung["n_cropped"].sum()),
        "n_volume_outlier": int(lung["n_volume_outlier"].sum()),
        "n_bilateral_asymmetry": int(lung["n_bilateral_asymmetry"].sum()),
        "n_qc_fail": int(lung["n_qc_fail"].sum()),
        "n_qc_pass": int(lung["n_qc_pass"].sum()),
    }
    merged_row["qc_pass_rate_pct"] = 100.0 * merged_row["n_qc_pass"] / max(merged_row["n_structure_instances"], 1)
    overall_row = qc.loc[qc["cohort"].eq("Overall")].copy()
    out = pd.concat([keep, pd.DataFrame([merged_row]), overall_row], ignore_index=True)
    out.to_csv(path, index=False)
    return out


def update_robustness_summary() -> None:
    path = DATA_DIR / "robustness_by_cohort_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    keep = df.loc[~df["cohort"].isin(OLD_LUNG)].copy()
    lung = df.loc[df["cohort"].isin(OLD_LUNG)].copy()
    merged = pd.DataFrame(
        [
            {
                "cohort": MERGED_LUNG,
                "rows": int(lung["rows"].sum()),
                "size_gb": float(lung["size_gb"].sum()),
                "fails": int(lung["fails"].sum()),
            }
        ]
    )
    out = pd.concat([keep, merged], ignore_index=True).sort_values("cohort")
    out.to_csv(path, index=False)


def run_reml_sensitivity() -> None:
    script = PROJECT_ROOT / "manuscript" / "scripts" / "66_forest_reml_hksj_sensitivity.R"
    subprocess.run(["Rscript", str(script)], check=True)


def main() -> None:
    all_icc = pd.read_parquet(ALL_ICC_PATH)
    qc_icc = pd.read_parquet(QC_ICC_PATH)

    all_icc_merged = merge_icc_points(all_icc)
    qc_icc_merged = merge_icc_points(qc_icc)

    all_icc_merged.to_parquet(ALL_ICC_PATH, index=False)
    qc_icc_merged.to_parquet(QC_ICC_PATH, index=False)

    summary_all = summarise_points(all_icc_merged, "all_data")
    summary_qc = summarise_points(qc_icc_merged, "qc_pass")
    summary_df = pd.concat([summary_all, summary_qc], ignore_index=True)
    summary_df.to_csv(TABLES_DIR / "qc_pass_sensitivity_cohort_summary.csv", index=False)
    build_comparison_table(summary_df).to_csv(TABLES_DIR / "qc_pass_sensitivity_comparison.csv", index=False)
    build_rank_stability(summary_df).to_csv(TABLES_DIR / "qc_pass_rank_stability.csv", index=False)

    cohort_effects = compute_cohort_effects(all_icc_merged)
    cohort_effects, hetero = compute_heterogeneity_summary(cohort_effects)
    cohort_effects.to_csv(TABLES_DIR / "forest_cohort_heterogeneity.csv", index=False)
    hetero.to_csv(TABLES_DIR / "forest_heterogeneity_summary.csv", index=False)

    loco_table, fragile = build_loco_outputs(all_icc_merged)
    loco_table.to_csv(TABLES_DIR / "loco_hierarchy_stability.csv", index=False)
    fragile.to_csv(TABLES_DIR / "loco_top_features_overlap.csv", index=False)

    shape_df, shape_summary, shape_pairwise = build_shape_outputs(all_icc_merged)
    shape_df.to_parquet(DATA_DIR / "shape_icc_stratified.parquet", index=False)
    shape_summary.to_csv(TABLES_DIR / "shape_icc_stratified_summary.csv", index=False)
    shape_pairwise.to_csv(TABLES_DIR / "shape_icc_stratified_pairwise.csv", index=False)

    scanner_df, thorax_df = update_scanner_tables()
    scanner_df.to_csv(TABLES_DIR / "scanner_demographics.csv", index=False)
    thorax_df.to_csv(TABLES_DIR / "thorax_scanner_context_2026-04-23.csv", index=False)

    update_qc_flag_summary()
    update_robustness_summary()
    run_reml_sensitivity()


if __name__ == "__main__":
    main()
