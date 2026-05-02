#!/usr/bin/env /home/kgs24/miniforge3/envs/radiomics_sge/bin/python
"""
Step 40: Null-feature control for contour-insensitive first-order features.

PURPOSE
-------
Test whether the Human-vs-AI CoV difference remains near zero for features that
are expected to be comparatively insensitive to contour boundary placement.

The null feature set is derived from the IBSI phantom validation table using:
  * config == "IBSI-compliant"
  * first-order features only
  * pct_deviation < 0.5

The NSCLC interobserver within-patient deviations parquet is then used to
compute paired Human-minus-AI CoV differences for each null feature.

INPUTS
------
  /home/kgs24/rtpipeline_manuscript/analysis/data/nsclc_io_within_patient_deviations.parquet
  /home/kgs24/rtpipeline_manuscript/analysis/tables/ibsi_validation.csv

NOTES ON IDENTIFIERS
--------------------
The deviations parquet currently exposes `patient_id` but not `course_id`, so
`subject = patient_id` is used when `course_id` is absent.  This is recorded in
the manifest.

BOOTSTRAP DESIGN
----------------
Level 1: resample patient clusters with replacement.
Level 2: within each sampled patient, resample rows within `feature_name`.

For a single feature the level-2 resample is degenerate because each
patient-feature group contains one within-patient CoV summary; the uncertainty
is therefore driven by the patient-cluster step.

OUTPUTS
-------
  data/null_feature_control.parquet
      Per-null-feature paired delta summaries, Wilcoxon results, and bootstrap
      confidence intervals.

  tables/null_feature_control_summary.csv
      Group summary for null vs non-null features plus their contrast.

  figures/figure_null_feature.{png,pdf}
      Violin + jitter plot of feature-level median paired deltas.

  logs/40_manifest_<JOBID>.json
      Provenance, checksums, and null-set definition notes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ANALYSIS_DIR = Path("/home/kgs24/rtpipeline_manuscript/analysis")
DATA_DIR = ANALYSIS_DIR / "data"
TABLES_DIR = ANALYSIS_DIR / "tables"
FIGURES_DIR = ANALYSIS_DIR / "figures"
LOGS_DIR = ANALYSIS_DIR / "logs"

DEVIATIONS_PARQUET = DATA_DIR / "nsclc_io_within_patient_deviations.parquet"
IBSI_TABLE = TABLES_DIR / "ibsi_validation.csv"

OUT_PARQUET = DATA_DIR / "null_feature_control.parquet"
OUT_SUMMARY = TABLES_DIR / "null_feature_control_summary.csv"
OUT_FIG_PNG = FIGURES_DIR / "figure_null_feature.png"
OUT_FIG_PDF = FIGURES_DIR / "figure_null_feature.pdf"

SEED = 2604
N_BOOT_DEFAULT = 1000
FAMILY_ORDER = ["shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]


class ContextFilter(logging.Filter):
    def __init__(self, host: str, job_id: str):
        super().__init__()
        self.host = host
        self.job_id = job_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.host = self.host
        record.job_id = self.job_id
        return True


def configure_logging(job_id: str) -> logging.Logger:
    logger = logging.getLogger("script40")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [host=%(host)s job=%(job_id)s] %(message)s"
        )
    )
    handler.addFilter(ContextFilter(socket.gethostname(), job_id))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def parse_cohorts(raw_value: str) -> list[str]:
    return [token.strip() for token in raw_value.split(",") if token.strip()]


def build_subject_column(df: pd.DataFrame, logger: logging.Logger) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    if "subject" in out.columns:
        return out, "subject"
    if {"patient_id", "course_id"}.issubset(out.columns):
        out["subject"] = out["patient_id"].astype(str) + "_" + out["course_id"].astype(str)
        return out, "patient_id_plus_course_id"
    if "patient_id" in out.columns:
        logger.info(
            "Input parquet has no course_id column; using patient_id as subject identifier."
        )
        out["subject"] = out["patient_id"].astype(str)
        return out, "patient_id_only_fallback"
    raise RuntimeError("Input parquet is missing both subject and patient_id columns.")


def load_deviations(logger: logging.Logger) -> tuple[pd.DataFrame, dict[str, object]]:
    if not DEVIATIONS_PARQUET.exists():
        raise FileNotFoundError(f"Missing input parquet: {DEVIATIONS_PARQUET}")
    df = pd.read_parquet(DEVIATIONS_PARQUET)
    required = {"feature_name", "family", "human_cov_pct", "ai_cov_pct"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Deviation parquet missing required columns: {sorted(missing)}")
    df, subject_mode = build_subject_column(df, logger)
    df["family"] = pd.Categorical(df["family"], categories=FAMILY_ORDER, ordered=True)
    df["paired_delta_cov_pct"] = df["human_cov_pct"] - df["ai_cov_pct"]
    meta = {
        "subject_mode": subject_mode,
        "n_subjects": int(df["subject"].nunique()),
        "n_rows": int(len(df)),
        "n_features": int(df["feature_name"].nunique()),
    }
    return df, meta


def derive_null_feature_set(logger: logging.Logger) -> tuple[pd.DataFrame, dict[str, object]]:
    if not IBSI_TABLE.exists():
        raise FileNotFoundError(f"Missing IBSI table: {IBSI_TABLE}")
    ibsi = pd.read_csv(IBSI_TABLE)
    required = {"config", "pyradiomics_feature", "pct_deviation"}
    missing = required.difference(ibsi.columns)
    if missing:
        raise RuntimeError(f"IBSI table missing required columns: {sorted(missing)}")
    ibsi["family_guess"] = ibsi["pyradiomics_feature"].astype(str).str.split("_").str[1].str.lower()
    filtered = ibsi.loc[
        (ibsi["config"].astype(str) == "IBSI-compliant")
        & (ibsi["family_guess"] == "firstorder")
        & (ibsi["pct_deviation"].astype(float) < 0.5)
    ].copy()
    filtered = (
        filtered.sort_values(["pyradiomics_feature", "pct_deviation"])
        .drop_duplicates(subset=["pyradiomics_feature"], keep="first")
        .rename(columns={"pyradiomics_feature": "feature_name", "pct_deviation": "ibsi_pct_deviation"})
    )
    logger.info(
        "Derived IBSI null-feature set: %d features.",
        filtered["feature_name"].nunique(),
    )
    return filtered[["feature_name", "ibsi_pct_deviation"]], {
        "null_feature_count": int(filtered["feature_name"].nunique()),
        "config_filter": "IBSI-compliant",
        "pct_deviation_threshold": 0.5,
        "family_filter": "firstorder",
    }


def resample_subject_clusters(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    subject_ids = df["subject"].drop_duplicates().to_numpy()
    sampled_subjects = rng.choice(subject_ids, size=len(subject_ids), replace=True)
    subject_blocks: list[pd.DataFrame] = []
    for draw_index, subject_id in enumerate(sampled_subjects):
        subject_block = df.loc[df["subject"] == subject_id].copy()
        sampled_blocks = []
        for _, group in subject_block.groupby("feature_name", sort=False):
            take = rng.integers(0, len(group), size=len(group))
            sampled_blocks.append(group.iloc[take].copy())
        sampled_subject = pd.concat(sampled_blocks, ignore_index=True)
        sampled_subject["subject"] = f"{subject_id}__boot{draw_index}"
        subject_blocks.append(sampled_subject)
    return pd.concat(subject_blocks, ignore_index=True)


def percentile_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    low = float(np.nanpercentile(values, 100.0 * alpha / 2.0))
    high = float(np.nanpercentile(values, 100.0 * (1.0 - alpha / 2.0)))
    return low, high


def wilcoxon_with_guard(deltas: np.ndarray) -> tuple[float, float]:
    clean = deltas[np.isfinite(deltas)]
    if clean.size == 0:
        return (float("nan"), float("nan"))
    if np.allclose(clean, 0.0):
        return (0.0, 1.0)
    stat, p_value = sp_stats.wilcoxon(clean, zero_method="wilcox", alternative="two-sided")
    return float(stat), float(p_value)


def cohens_dz(deltas: np.ndarray) -> float:
    clean = deltas[np.isfinite(deltas)]
    if clean.size < 2:
        return float("nan")
    sd = np.std(clean, ddof=1)
    if np.isclose(sd, 0.0):
        return 0.0
    return float(np.mean(clean) / sd)


def compute_feature_delta_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["feature_name", "family"], observed=True, sort=False)
        .agg(
            n_subjects=("subject", "nunique"),
            human_median_cov_pct=("human_cov_pct", "median"),
            ai_median_cov_pct=("ai_cov_pct", "median"),
            median_paired_delta_pct=("paired_delta_cov_pct", "median"),
            q25_paired_delta_pct=("paired_delta_cov_pct", lambda values: np.nanpercentile(values, 25)),
            q75_paired_delta_pct=("paired_delta_cov_pct", lambda values: np.nanpercentile(values, 75)),
        )
        .reset_index()
    )


def bootstrap_feature_ci(feature_df: pd.DataFrame, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        boot_df = resample_subject_clusters(feature_df, rng)
        values.append(float(np.nanmedian(boot_df["paired_delta_cov_pct"].to_numpy(dtype=float))))
    return percentile_ci(np.asarray(values, dtype=float))


def build_null_feature_results(
    raw_deviations: pd.DataFrame,
    null_features: pd.DataFrame,
    n_boot: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    null_feature_names = set(null_features["feature_name"].astype(str))
    intersecting = raw_deviations.loc[raw_deviations["feature_name"].isin(null_feature_names)].copy()
    rows: list[dict[str, object]] = []
    for feature_name in sorted(null_feature_names):
        feature_df = intersecting.loc[intersecting["feature_name"] == feature_name].copy()
        if feature_df.empty:
            continue
        deltas = feature_df["paired_delta_cov_pct"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_feature_ci(feature_df, n_boot=n_boot, seed=SEED)
        stat, p_value = wilcoxon_with_guard(deltas)
        rows.append(
            {
                "feature_name": feature_name,
                "family": str(feature_df["family"].astype(str).iloc[0]),
                "n_subjects": int(feature_df["subject"].nunique()),
                "human_median_cov_pct": float(feature_df["human_cov_pct"].median()),
                "ai_median_cov_pct": float(feature_df["ai_cov_pct"].median()),
                "median_paired_delta_pct": float(np.nanmedian(deltas)),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "wilcoxon_stat": stat,
                "wilcoxon_p": p_value,
                "effect_size_dz": cohens_dz(deltas),
            }
        )
    result_df = pd.DataFrame(rows).merge(null_features, on="feature_name", how="left")
    if result_df.empty:
        raise RuntimeError("No null features overlapped the NSCLC deviation parquet.")
    logger.info(
        "Computed null-feature results for %d features.",
        result_df["feature_name"].nunique(),
    )
    feature_delta_df = compute_feature_delta_table(raw_deviations)
    feature_delta_df["group"] = np.where(
        feature_delta_df["feature_name"].isin(null_feature_names), "null", "non_null"
    )
    return result_df, feature_delta_df


def bootstrap_group_summary(
    raw_deviations: pd.DataFrame,
    null_feature_names: set[str],
    n_boot: int,
    seed: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for iteration in range(n_boot):
        boot_df = resample_subject_clusters(raw_deviations, rng)
        feature_delta_df = compute_feature_delta_table(boot_df)
        feature_delta_df["group"] = np.where(
            feature_delta_df["feature_name"].isin(null_feature_names), "null", "non_null"
        )
        medians = {}
        for group_name in ("null", "non_null"):
            sub = feature_delta_df.loc[feature_delta_df["group"] == group_name]
            if sub.empty:
                continue
            group_median = float(sub["median_paired_delta_pct"].median())
            records.append(
                {
                    "bootstrap_iteration": iteration,
                    "group": group_name,
                    "group_median_delta_pct": group_median,
                }
            )
            medians[group_name] = group_median
        if {"null", "non_null"}.issubset(medians):
            records.append(
                {
                    "bootstrap_iteration": iteration,
                    "group": "contrast_null_minus_non_null",
                    "group_median_delta_pct": medians["null"] - medians["non_null"],
                }
            )
        if (iteration + 1) % 100 == 0 or iteration == n_boot - 1:
            logger.info("Bootstrap %d/%d complete.", iteration + 1, n_boot)
    boot_df = pd.DataFrame(records)
    rows: list[dict[str, object]] = []
    for group_name, sub in boot_df.groupby("group", sort=False):
        ci_low, ci_high = percentile_ci(sub["group_median_delta_pct"].to_numpy(dtype=float))
        rows.append({"group": group_name, "ci_low": ci_low, "ci_high": ci_high})
    return pd.DataFrame(rows)


def build_group_summary(
    feature_delta_df: pd.DataFrame,
    null_result_df: pd.DataFrame,
    ci_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    null_feature_names = set(null_result_df["feature_name"].astype(str))
    null_feature_medians = feature_delta_df.loc[
        feature_delta_df["feature_name"].isin(null_feature_names), "median_paired_delta_pct"
    ].to_numpy(dtype=float)
    non_null_feature_medians = feature_delta_df.loc[
        ~feature_delta_df["feature_name"].isin(null_feature_names), "median_paired_delta_pct"
    ].to_numpy(dtype=float)

    for group_name, values in (
        ("null", null_feature_medians),
        ("non_null", non_null_feature_medians),
    ):
        rows.append(
            {
                "group": group_name,
                "n_features": int(len(values)),
                "median_feature_delta_pct": float(np.nanmedian(values)),
                "q25_feature_delta_pct": float(np.nanpercentile(values, 25)),
                "q75_feature_delta_pct": float(np.nanpercentile(values, 75)),
                "pct_features_ai_lower": float(100.0 * np.mean(values > 0.0)),
                "mannwhitney_p_vs_null": float("nan"),
            }
        )

    _, mannwhitney_p = sp_stats.mannwhitneyu(
        null_feature_medians,
        non_null_feature_medians,
        alternative="two-sided",
    )
    rows.append(
        {
            "group": "contrast_null_minus_non_null",
            "n_features": int(len(null_feature_medians)),
            "median_feature_delta_pct": float(np.nanmedian(null_feature_medians) - np.nanmedian(non_null_feature_medians)),
            "q25_feature_delta_pct": float("nan"),
            "q75_feature_delta_pct": float("nan"),
            "pct_features_ai_lower": float("nan"),
            "mannwhitney_p_vs_null": float(mannwhitney_p),
        }
    )
    summary_df = pd.DataFrame(rows).merge(ci_df, on="group", how="left")
    summary_df.loc[summary_df["group"].isin(["null", "non_null"]), "mannwhitney_p_vs_null"] = float(mannwhitney_p)
    return summary_df


def plot_feature_delta_distributions(feature_delta_df: pd.DataFrame) -> None:
    plot_df = feature_delta_df.loc[feature_delta_df["group"].isin(["null", "non_null"])].copy()
    order = ["null", "non_null"]
    colors = {"null": "#4c78a8", "non_null": "#bab0ac"}
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    positions = np.arange(len(order))
    violin_values = [
        plot_df.loc[plot_df["group"] == group_name, "median_paired_delta_pct"].to_numpy(dtype=float)
        for group_name in order
    ]
    violins = ax.violinplot(
        violin_values,
        positions=positions,
        showmeans=False,
        showmedians=True,
        widths=0.8,
    )
    for body, group_name in zip(violins["bodies"], order):
        body.set_facecolor(colors[group_name])
        body.set_edgecolor("black")
        body.set_alpha(0.75)
    for part_name in ("cbars", "cmins", "cmaxes", "cmedians"):
        violins[part_name].set_color("black")
        violins[part_name].set_linewidth(1.0)

    rng = np.random.default_rng(SEED)
    for idx, group_name in enumerate(order):
        values = plot_df.loc[plot_df["group"] == group_name, "median_paired_delta_pct"].to_numpy(dtype=float)
        jitter = rng.normal(loc=0.0, scale=0.045, size=len(values))
        ax.scatter(
            np.full(len(values), positions[idx]) + jitter,
            values,
            s=14,
            color="black",
            alpha=0.35,
        )

    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(["Null features", "Non-null features"])
    ax.set_ylabel("Feature-level median paired delta (Human CoV - AI CoV)")
    ax.set_title("Null-feature control: paired delta distributions")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(OUT_FIG_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG_PDF, bbox_inches="tight")
    plt.close(fig)


def write_manifest(
    *,
    job_id: str,
    argv: list[str],
    runtime_seconds: float,
    bootstrap_iterations: int,
    cohorts_argument: list[str],
    input_meta: dict[str, object],
    null_meta: dict[str, object],
    smoke_test_details: dict[str, object] | None,
) -> Path:
    manifest_path = LOGS_DIR / f"40_manifest_{job_id}.json"
    manifest = {
        "script_path": str(Path(__file__).resolve()),
        "argv": argv,
        "host": socket.gethostname(),
        "job_id": job_id,
        "project_path": str(Path(__file__).resolve().parents[1]),
        "bootstrap_seed": SEED,
        "bootstrap_iterations": bootstrap_iterations,
        "cohorts_argument": cohorts_argument,
        "input_file_checksums": {
            str(DEVIATIONS_PARQUET): sha256_file(DEVIATIONS_PARQUET),
            str(IBSI_TABLE): sha256_file(IBSI_TABLE),
        },
        "output_file_paths": [
            str(OUT_PARQUET),
            str(OUT_SUMMARY),
            str(OUT_FIG_PNG),
            str(OUT_FIG_PDF),
        ],
        "output_file_checksums": {
            str(OUT_PARQUET): sha256_file(OUT_PARQUET),
            str(OUT_SUMMARY): sha256_file(OUT_SUMMARY),
            str(OUT_FIG_PNG): sha256_file(OUT_FIG_PNG),
            str(OUT_FIG_PDF): sha256_file(OUT_FIG_PDF),
        },
        "runtime_seconds": runtime_seconds,
        "git_sha": resolve_git_sha(),
        "input_meta": input_meta,
        "null_meta": null_meta,
        "smoke_test_details": smoke_test_details,
        "method_notes": {
            "null_definition": "IBSI-compliant first-order features with phantom pct_deviation < 0.5.",
            "delta_definition": "paired_delta_cov_pct = human_cov_pct - ai_cov_pct",
            "contrast_definition": "contrast_null_minus_non_null = median(null feature medians) - median(non-null feature medians)",
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="Restrict the null set to 5 features and the non-null set to 40 features; cap bootstrap iterations at 50.")
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=N_BOOT_DEFAULT,
        help="Number of patient-clustered bootstrap iterations.",
    )
    parser.add_argument(
        "--cohorts",
        default="NSCLC_Interobserver",
        help="Accepted for interface compatibility; recorded in the manifest.",
    )
    parser.add_argument(
        "--job-id",
        default=os.environ.get("JOB_ID", "local"),
        help="Job identifier used in logs and manifest output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cohorts_argument = parse_cohorts(args.cohorts)
    for directory in (DATA_DIR, TABLES_DIR, FIGURES_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    logger = configure_logging(args.job_id)
    n_boot = args.bootstrap_iterations
    if args.smoke_test and n_boot > 50:
        logger.info(
            "Smoke test requested: reducing bootstrap iterations from %d to 50.",
            n_boot,
        )
        n_boot = 50

    t0 = time.time()
    logger.info("Starting step 40 null-feature control.")
    raw_deviations, input_meta = load_deviations(logger)
    null_features, null_meta = derive_null_feature_set(logger)

    smoke_test_details: dict[str, object] | None = None
    if args.smoke_test:
        null_keep = sorted(null_features["feature_name"].astype(str))[:5]
        non_null_keep = sorted(
            set(raw_deviations["feature_name"].astype(str)).difference(null_keep)
        )[:40]
        keep_features = set(null_keep).union(non_null_keep)
        raw_deviations = raw_deviations.loc[
            raw_deviations["feature_name"].isin(keep_features)
        ].copy()
        null_features = null_features.loc[null_features["feature_name"].isin(null_keep)].copy()
        smoke_test_details = {
            "null_features_kept": null_keep,
            "non_null_features_kept": non_null_keep,
        }
        logger.info(
            "Smoke test enabled: retained %d null features and %d non-null features.",
            len(null_keep),
            len(non_null_keep),
        )

    null_result_df, feature_delta_df = build_null_feature_results(
        raw_deviations, null_features, n_boot=n_boot, logger=logger
    )
    ci_df = bootstrap_group_summary(
        raw_deviations,
        set(null_features["feature_name"].astype(str)),
        n_boot=n_boot,
        seed=SEED,
        logger=logger,
    )
    summary_df = build_group_summary(feature_delta_df, null_result_df, ci_df)

    null_result_df.to_parquet(OUT_PARQUET, index=False)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    plot_feature_delta_distributions(feature_delta_df)

    manifest_path = write_manifest(
        job_id=args.job_id,
        argv=sys.argv,
        runtime_seconds=time.time() - t0,
        bootstrap_iterations=n_boot,
        cohorts_argument=cohorts_argument,
        input_meta=input_meta,
        null_meta=null_meta,
        smoke_test_details=smoke_test_details,
    )

    logger.info("Wrote %s", OUT_PARQUET)
    logger.info("Wrote %s", OUT_SUMMARY)
    logger.info("Wrote %s and %s", OUT_FIG_PNG, OUT_FIG_PDF)
    logger.info("Wrote %s", manifest_path)
    logger.info(
        "Step 40 completed successfully with %d null features.",
        null_result_df["feature_name"].nunique(),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - defensive exit path
        print(f"[FATAL] {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(2)
