#!/usr/bin/env python3
"""
Step 46: Pooled Polish structure-role/size synthesis from existing #20 outputs.

This lane consumes the synced `human_vs_ai_v2` cohort tables and produces
manuscript-ready pooled summaries without rerunning the underlying heavy
radiomics computation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BODY_REGION_BY_COHORT = {
    "Prostata": "Pelvis",
    "Odbytnice": "Pelvis",
    "Immunodozymetria": "Thorax",
    "PlucaRCHT": "Thorax",
    "Hipokampy": "Brain",
    "GBM": "Brain",
}

ROLE_ORDER = ["oar_large", "oar_small", "unknown"]
ROLE_LABELS = {
    "oar_large": "Large OAR",
    "oar_small": "Small OAR",
    "unknown": "Unknown / unlabeled",
}
REGION_ORDER = ["Pelvis", "Thorax", "Brain"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date-stamp", default="2026-04-21")
    return parser.parse_args()


def manuscript_root() -> Path:
    return Path(__file__).resolve().parents[1]


def flatten_geom_columns(columns: pd.MultiIndex) -> list[str]:
    flat: list[str] = []
    unnamed_seen = 0
    for top, bottom in columns:
        top = str(top)
        bottom = str(bottom)
        if top.startswith("Unnamed:"):
            flat.append("cohort" if unnamed_seen == 0 else "structure_role")
            unnamed_seen += 1
            continue
        flat.append(f"{top}_{bottom}")
    return flat


def read_geom_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=[0, 1])
    df.columns = flatten_geom_columns(df.columns)
    numeric_cols = [col for col in df.columns if col not in {"cohort", "structure_role"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values.loc[mask], weights=weights.loc[mask]))


def format_range(values: pd.Series, digits: int = 4) -> str:
    clean = values.dropna()
    if clean.empty:
        return ""
    return f"{clean.min():.{digits}f} to {clean.max():.{digits}f}"


def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_frames: list[pd.DataFrame] = []
    feature_frames: list[pd.DataFrame] = []
    geom_frames: list[pd.DataFrame] = []

    for cohort in BODY_REGION_BY_COHORT:
        summary_path = input_dir / f"polish_subgroup_summary_by_role_{cohort}.csv"
        feature_path = input_dir / f"polish_subgroup_by_role_{cohort}.csv"
        geom_path = input_dir / f"polish_subgroup_geom_by_role_{cohort}.csv"

        if not summary_path.exists() or not feature_path.exists() or not geom_path.exists():
            missing = [str(p) for p in [summary_path, feature_path, geom_path] if not p.exists()]
            raise FileNotFoundError(f"Missing synced #20 inputs for {cohort}: {missing}")

        summary_df = pd.read_csv(summary_path)
        feature_df = pd.read_csv(feature_path)
        geom_df = read_geom_table(geom_path)

        for df in (summary_df, feature_df, geom_df):
            df["body_region"] = BODY_REGION_BY_COHORT[cohort]
            df["role_label"] = df["structure_role"].map(ROLE_LABELS).fillna(df["structure_role"])

        summary_frames.append(summary_df)
        feature_frames.append(feature_df)
        geom_frames.append(geom_df)

    summary_all = pd.concat(summary_frames, ignore_index=True)
    feature_all = pd.concat(feature_frames, ignore_index=True)
    geom_all = pd.concat(geom_frames, ignore_index=True)
    return summary_all, feature_all, geom_all


def build_by_cohort_table(
    summary_all: pd.DataFrame, feature_all: pd.DataFrame, geom_all: pd.DataFrame
) -> pd.DataFrame:
    patient_counts = (
        feature_all.groupby(["cohort", "structure_role"], as_index=False)["n_patients"]
        .max()
        .rename(columns={"n_patients": "n_patients_pairs"})
    )

    feature_metrics = (
        feature_all.groupby(["cohort", "body_region", "structure_role", "role_label"], as_index=False)
        .agg(
            pooled_feature_median_ccc=("ccc", "median"),
            pooled_feature_median_icc31=("icc31_consistency", "median"),
            pct_features_ccc_ge_0_75=("ccc", lambda s: 100.0 * float((s >= 0.75).mean())),
            pct_features_ccc_ge_0_90=("ccc", lambda s: 100.0 * float((s >= 0.90).mean())),
            pct_features_icc31_ge_0_75=(
                "icc31_consistency",
                lambda s: 100.0 * float((s >= 0.75).mean()),
            ),
            pct_features_icc31_ge_0_90=(
                "icc31_consistency",
                lambda s: 100.0 * float((s >= 0.90).mean()),
            ),
        )
    )

    by_cohort = summary_all.merge(
        patient_counts,
        on=["cohort", "structure_role"],
        how="left",
    ).merge(
        feature_metrics,
        on=["cohort", "body_region", "structure_role", "role_label"],
        how="left",
    ).merge(
        geom_all,
        on=["cohort", "body_region", "structure_role", "role_label"],
        how="left",
    )

    ordered_cols = [
        "cohort",
        "body_region",
        "structure_role",
        "role_label",
        "n_patients_pairs",
        "n_features",
        "median_ccc",
        "median_icc31",
        "pct_features_ccc_ge_0_75",
        "pct_features_ccc_ge_0_90",
        "pct_features_icc31_ge_0_75",
        "pct_features_icc31_ge_0_90",
        "geo_dice_mean",
        "geo_dice_median",
        "geo_dice_count",
        "geo_hd95_mm_mean",
        "geo_hd95_mm_median",
        "geo_msd_mm_mean",
        "geo_msd_mm_median",
        "geo_vol_ratio_ai_over_human_mean",
        "geo_vol_ratio_ai_over_human_median",
        "mean_ba_bias",
    ]
    by_cohort = by_cohort.loc[:, ordered_cols].copy()
    by_cohort["body_region"] = pd.Categorical(by_cohort["body_region"], REGION_ORDER, ordered=True)
    by_cohort["structure_role"] = pd.Categorical(by_cohort["structure_role"], ROLE_ORDER, ordered=True)
    by_cohort = by_cohort.sort_values(["body_region", "cohort", "structure_role"]).reset_index(drop=True)
    return by_cohort


def summarise_group(
    key_values: dict[str, str],
    cohort_subset: pd.DataFrame,
    feature_subset: pd.DataFrame,
    geom_subset: pd.DataFrame,
) -> dict[str, object]:
    row: dict[str, object] = {}
    row.update(key_values)
    row["role_label"] = ROLE_LABELS.get(key_values["structure_role"], key_values["structure_role"])
    row["n_cohorts"] = int(cohort_subset["cohort"].nunique())
    row["cohorts"] = "|".join(sorted(cohort_subset["cohort"].astype(str).unique()))
    if "body_region" not in key_values:
        row["body_regions_present"] = "|".join(sorted(cohort_subset["body_region"].astype(str).unique()))
    row["total_patient_pairs"] = int(cohort_subset["n_patients_pairs"].sum())
    row["patient_pairs_range"] = format_range(cohort_subset["n_patients_pairs"], digits=0)
    row["total_feature_rows"] = int(len(feature_subset))
    row["feature_rows_per_cohort"] = int(feature_subset.groupby("cohort").size().median())
    row["median_of_cohort_median_ccc"] = float(cohort_subset["median_ccc"].median())
    row["cohort_median_ccc_range"] = format_range(cohort_subset["median_ccc"])
    row["pooled_feature_median_ccc"] = float(feature_subset["ccc"].median())
    row["pct_features_ccc_ge_0_75"] = 100.0 * float((feature_subset["ccc"] >= 0.75).mean())
    row["pct_features_ccc_ge_0_90"] = 100.0 * float((feature_subset["ccc"] >= 0.90).mean())
    row["median_of_cohort_median_icc31"] = float(cohort_subset["median_icc31"].median())
    row["cohort_median_icc31_range"] = format_range(cohort_subset["median_icc31"])
    row["pooled_feature_median_icc31"] = float(feature_subset["icc31_consistency"].median())
    row["pct_features_icc31_ge_0_75"] = 100.0 * float((feature_subset["icc31_consistency"] >= 0.75).mean())
    row["pct_features_icc31_ge_0_90"] = 100.0 * float((feature_subset["icc31_consistency"] >= 0.90).mean())
    row["geom_pair_count_total"] = int(geom_subset["geo_dice_count"].fillna(0).sum())
    row["weighted_mean_geo_dice"] = weighted_mean(geom_subset["geo_dice_mean"], geom_subset["geo_dice_count"])
    row["median_of_cohort_geo_dice_median"] = float(geom_subset["geo_dice_median"].median())
    row["weighted_mean_geo_hd95_mm"] = weighted_mean(
        geom_subset["geo_hd95_mm_mean"], geom_subset["geo_hd95_mm_count"]
    )
    row["median_of_cohort_geo_hd95_mm_median"] = float(geom_subset["geo_hd95_mm_median"].median())
    row["weighted_mean_geo_msd_mm"] = weighted_mean(
        geom_subset["geo_msd_mm_mean"], geom_subset["geo_msd_mm_count"]
    )
    row["median_of_cohort_geo_msd_mm_median"] = float(geom_subset["geo_msd_mm_median"].median())
    row["weighted_mean_vol_ratio_ai_over_human"] = weighted_mean(
        geom_subset["geo_vol_ratio_ai_over_human_mean"],
        geom_subset["geo_vol_ratio_ai_over_human_count"],
    )
    row["median_of_cohort_vol_ratio_median"] = float(
        geom_subset["geo_vol_ratio_ai_over_human_median"].median()
    )
    return row


def build_pooled_tables(
    by_cohort: pd.DataFrame, feature_all: pd.DataFrame, geom_all: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []

    for structure_role in ROLE_ORDER:
        cohort_subset = by_cohort.loc[by_cohort["structure_role"] == structure_role].copy()
        if cohort_subset.empty:
            continue
        feature_subset = feature_all.loc[feature_all["structure_role"] == structure_role].copy()
        geom_subset = geom_all.loc[geom_all["structure_role"] == structure_role].copy()
        pooled_rows.append(
            summarise_group(
                {"structure_role": structure_role},
                cohort_subset,
                feature_subset,
                geom_subset,
            )
        )

    for body_region in REGION_ORDER:
        for structure_role in ROLE_ORDER:
            cohort_subset = by_cohort.loc[
                (by_cohort["body_region"] == body_region) & (by_cohort["structure_role"] == structure_role)
            ].copy()
            if cohort_subset.empty:
                continue
            feature_subset = feature_all.loc[
                (feature_all["body_region"] == body_region)
                & (feature_all["structure_role"] == structure_role)
            ].copy()
            geom_subset = geom_all.loc[
                (geom_all["body_region"] == body_region) & (geom_all["structure_role"] == structure_role)
            ].copy()
            region_rows.append(
                summarise_group(
                    {"body_region": body_region, "structure_role": structure_role},
                    cohort_subset,
                    feature_subset,
                    geom_subset,
                )
            )

    pooled_df = pd.DataFrame(pooled_rows)
    region_df = pd.DataFrame(region_rows)
    if not pooled_df.empty:
        pooled_df["structure_role"] = pd.Categorical(pooled_df["structure_role"], ROLE_ORDER, ordered=True)
        pooled_df = pooled_df.sort_values("structure_role").reset_index(drop=True)
    if not region_df.empty:
        region_df["body_region"] = pd.Categorical(region_df["body_region"], REGION_ORDER, ordered=True)
        region_df["structure_role"] = pd.Categorical(region_df["structure_role"], ROLE_ORDER, ordered=True)
        region_df = region_df.sort_values(["body_region", "structure_role"]).reset_index(drop=True)
    return pooled_df, region_df


def memo_text(
    date_stamp: str,
    input_dir: Path,
    script_path: Path,
    out_by_cohort: Path,
    out_pooled: Path,
    out_region: Path,
    by_cohort: pd.DataFrame,
    pooled: pd.DataFrame,
    region: pd.DataFrame,
) -> str:
    large = pooled.loc[pooled["structure_role"] == "oar_large"].iloc[0]
    small = pooled.loc[pooled["structure_role"] == "oar_small"].iloc[0]
    unknown = pooled.loc[pooled["structure_role"] == "unknown"]
    unknown_line = ""
    if not unknown.empty:
        unk = unknown.iloc[0]
        unknown_line = (
            f"- `Unknown / unlabeled` roles appear in `{unk['n_cohorts']}` thoracic cohorts only "
            f"with pooled median CCC `{unk['pooled_feature_median_ccc']:.4f}` and weighted mean Dice "
            f"`{unk['weighted_mean_geo_dice']:.4f}`. I retained these rows as `unknown` rather than "
            "relabeling them as targets without a validated crosswalk in the synced package.\n"
        )

    brain_large = region.loc[
        (region["body_region"] == "Brain") & (region["structure_role"] == "oar_large")
    ].iloc[0]
    pelvis_large = region.loc[
        (region["body_region"] == "Pelvis") & (region["structure_role"] == "oar_large")
    ].iloc[0]
    thorax_large = region.loc[
        (region["body_region"] == "Thorax") & (region["structure_role"] == "oar_large")
    ].iloc[0]

    return f"""# 46 Polish Role/Size Synthesis

Project root: `/umed-projekty/rtpipeline`
Date: `{date_stamp}`

## Scope

This delegated lane pooled the already-synced `#20` Polish human-vs-AI package into a manuscript-ready structure-role/size synthesis. I used the existing `human_vs_ai_v2` CSV tables first and did not rerun heavy radiomics or segmentation compute.

## Inputs Used

- `{input_dir / "polish_subgroup_summary_by_role_Prostata.csv"}`
- `{input_dir / "polish_subgroup_summary_by_role_Odbytnice.csv"}`
- `{input_dir / "polish_subgroup_summary_by_role_PlucaRCHT.csv"}`
- `{input_dir / "polish_subgroup_summary_by_role_Immunodozymetria.csv"}`
- `{input_dir / "polish_subgroup_summary_by_role_GBM.csv"}`
- `{input_dir / "polish_subgroup_summary_by_role_Hipokampy.csv"}`
- `{input_dir / "polish_subgroup_by_role_*.csv"}`
- `{input_dir / "polish_subgroup_geom_by_role_*.csv"}`
- `{script_path}`

## Outputs Written

- `{out_by_cohort}`
- `{out_pooled}`
- `{out_region}`

## Key Findings

- `Large OAR` rows are consistently more stable than `Small OAR` rows across all available cohorts. Pooled feature-level median CCC is `{large['pooled_feature_median_ccc']:.4f}` for large OARs versus `{small['pooled_feature_median_ccc']:.4f}` for small OARs; pooled feature-level median ICC(3,1) is `{large['pooled_feature_median_icc31']:.4f}` versus `{small['pooled_feature_median_icc31']:.4f}`.
- The geometry signal shows the same size gradient. Count-weighted mean Dice is `{large['weighted_mean_geo_dice']:.4f}` for large OARs versus `{small['weighted_mean_geo_dice']:.4f}` for small OARs, while count-weighted mean HD95 is `{large['weighted_mean_geo_hd95_mm']:.2f}` mm versus `{small['weighted_mean_geo_hd95_mm']:.2f}` mm.
- Across cohorts, large-OAR cohort-median CCC spans `{large['cohort_median_ccc_range']}`, whereas small-OAR cohort-median CCC spans `{small['cohort_median_ccc_range']}`. This gives a clean manuscript-ready size effect even before invoking more detailed family-level decomposition.
- Regionally, pelvis and thorax large-OAR performance remain strong: pooled large-OAR median CCC is `{pelvis_large['pooled_feature_median_ccc']:.4f}` in pelvis and `{thorax_large['pooled_feature_median_ccc']:.4f}` in thorax. Brain large-OAR performance is much weaker at `{brain_large['pooled_feature_median_ccc']:.4f}`, which is directionally consistent with the previously documented head-task / empty-mask limitations for GBM and Hipokampy.
{unknown_line.rstrip()}

## Interpretation

The existing `#20` package already supports a concise cross-cohort statement that human-vs-AI radiomic agreement degrades as structures get smaller, with the most defensible pooled contrast being `large OAR` versus `small OAR`. That synthesis is strongest in pelvis and thorax and should be treated as attenuated in brain cohorts pending the separate head-task remediation lane.

## Limitations

- The synthesis intentionally stays at the synced role labels: `oar_large`, `oar_small`, and `unknown`.
- `mean_ba_bias` was retained in the cohort output for completeness but is not emphasized because feature-scale units differ across families.
- Geometry pooling is based on the exported cohort-role summary tables, so cross-cohort Dice / distance values are count-weighted summaries rather than patient-level re-estimates.

## Recommended Use

Use `{out_pooled.name}` for the headline pooled role/size table, `{out_region.name}` for anatomy-qualified phrasing, and `{out_by_cohort.name}` when the manuscript needs cohort-level support rows or reviewer-facing traceability.
"""


def main() -> None:
    args = parse_args()
    mroot = manuscript_root()
    input_dir = mroot / "analysis" / "human_vs_ai_v2" / "tables"
    output_dir = mroot / "analysis" / "tables"
    review_dir = mroot / "review" / "delegated"
    output_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    out_by_cohort = output_dir / f"46_polish_role_size_synthesis_by_cohort_{args.date_stamp}.csv"
    out_pooled = output_dir / f"46_polish_role_size_synthesis_pooled_{args.date_stamp}.csv"
    out_region = output_dir / f"46_polish_role_size_synthesis_by_region_{args.date_stamp}.csv"
    out_memo = review_dir / f"46_polish_role_size_synthesis_{args.date_stamp}.md"

    summary_all, feature_all, geom_all = load_inputs(input_dir)
    by_cohort = build_by_cohort_table(summary_all, feature_all, geom_all)
    pooled, region = build_pooled_tables(by_cohort, feature_all, geom_all)

    by_cohort.to_csv(out_by_cohort, index=False)
    pooled.to_csv(out_pooled, index=False)
    region.to_csv(out_region, index=False)
    out_memo.write_text(
        memo_text(
            args.date_stamp,
            input_dir,
            Path(__file__).resolve(),
            out_by_cohort.resolve(),
            out_pooled.resolve(),
            out_region.resolve(),
            by_cohort,
            pooled,
            region,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
