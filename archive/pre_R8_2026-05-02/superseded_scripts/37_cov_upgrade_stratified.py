#!/usr/bin/env /home/kgs24/miniforge3/envs/radiomics_sge/bin/python
"""
Step 37: Feature-class upgrade stratified by baseline human CoV.

PURPOSE
-------
Test whether the contouring advantage is concentrated in the feature quartiles
where specialist human contouring shows the highest within-patient CoV.

The script:
  1. derives the locked public-cohort feature universe from `icc_results.parquet`,
  2. computes feature-level median human and AI CoV across NSCLC interobserver
     subjects,
  3. bins features into quartiles of baseline human CoV,
  4. assigns Robust / Acceptable / Poor classes using the requested NTCV-style
     thresholds, and
  5. summarises improvement transitions by quartile and feature family.

INPUTS
------
Required:
  /home/kgs24/rtpipeline_manuscript/analysis/data/nsclc_io_within_patient_deviations.parquet
  /home/kgs24/rtpipeline_manuscript/analysis/data/icc_results.parquet

Required for arm-specific Human-vs-AI class assignment:
  /home/kgs24/rtpipeline_manuscript/analysis/data/nsclc_io_human_ai_contour_comparison.parquet

WHY THE EXTRA NSCLC ICC FILE IS REQUIRED
----------------------------------------
`icc_results.parquet` is the cohort-wide robustness table used to derive the
locked 107-feature public intersection, but it does not contain separate
Human-vs-AI ICC columns.  Human-vs-AI class assignment therefore uses the
NSCLC-specific comparison parquet when present.  If that file is absent and no
equivalent arm-specific ICC columns exist, the script exits with a diagnostic.

NOTES ON IDENTIFIERS
--------------------
The current NSCLC deviations parquet exposes `patient_id` but not `course_id`.
This script therefore uses `subject = patient_id` when `course_id` is absent.

CLASS RULES
-----------
Robust:
  ICC >= 0.90 and CoV <= 10%

Acceptable:
  0.75 <= ICC < 0.90 and CoV <= 20%

Poor:
  all other combinations

OUTPUTS
-------
  data/feature_class_transitions.parquet
      Feature-level class assignments, quartiles, and transition labels.

  tables/class_transitions_by_quartile.csv
      Long-format quartile × family × transition summary with bootstrap CIs.

  figures/figure_quartile_upgrade.{png,pdf}
      Stacked bar chart of transition rates by quartile for each family.

  logs/37_manifest_<JOBID>.json
      Provenance, checksums, parameters, and feature-universe notes.
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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ANALYSIS_DIR = Path("/home/kgs24/rtpipeline_manuscript/analysis")
DATA_DIR = ANALYSIS_DIR / "data"
TABLES_DIR = ANALYSIS_DIR / "tables"
FIGURES_DIR = ANALYSIS_DIR / "figures"
LOGS_DIR = ANALYSIS_DIR / "logs"

DEVIATIONS_PARQUET = DATA_DIR / "nsclc_io_within_patient_deviations.parquet"
ICC_RESULTS_PARQUET = DATA_DIR / "icc_results.parquet"
NSCLC_HUMAN_AI_ICC_PARQUET = DATA_DIR / "nsclc_io_human_ai_contour_comparison.parquet"

OUT_PARQUET = DATA_DIR / "feature_class_transitions.parquet"
OUT_SUMMARY = TABLES_DIR / "class_transitions_by_quartile.csv"
OUT_FIG_PNG = FIGURES_DIR / "figure_quartile_upgrade.png"
OUT_FIG_PDF = FIGURES_DIR / "figure_quartile_upgrade.pdf"

SEED = 2604
N_BOOT_DEFAULT = 1000
FAMILY_ORDER = ["shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
QUARTILE_ORDER = ["Q1", "Q2", "Q3", "Q4"]
TRANSITION_ORDER = [
    "Poor->Acceptable",
    "Poor->Robust",
    "Acceptable->Robust",
    "No improvement / other",
]
DEFAULT_COHORTS = [
    "GBM",
    "Hipokampy",
    "Immunodozymetria",
    "LCTSC",
    "NSCLC_Interobserver",
    "NSCLC_Radiomics",
    "Odbytnice",
    "PlucaRCHT",
    "Prostata",
    "RIDER",
]


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
    logger = logging.getLogger("script37")
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
        raise RuntimeError(
            f"Deviation parquet missing required columns: {sorted(missing)}"
        )
    df, subject_mode = build_subject_column(df, logger)
    df["family"] = pd.Categorical(df["family"], categories=FAMILY_ORDER, ordered=True)
    meta = {
        "subject_mode": subject_mode,
        "n_subjects": int(df["subject"].nunique()),
        "n_rows": int(len(df)),
        "n_features": int(df["feature_name"].nunique()),
    }
    return df, meta


def derive_locked_feature_set(
    requested_cohorts: list[str], logger: logging.Logger
) -> tuple[set[str], dict[str, object]]:
    if not ICC_RESULTS_PARQUET.exists():
        raise FileNotFoundError(f"Missing ICC results parquet: {ICC_RESULTS_PARQUET}")
    icc_df = pd.read_parquet(ICC_RESULTS_PARQUET, columns=["feature_name", "cohort"])
    available_cohorts = sorted(icc_df["cohort"].astype(str).unique().tolist())
    missing = sorted(set(requested_cohorts).difference(available_cohorts))
    if missing:
        raise RuntimeError(
            f"Requested cohorts missing from icc_results.parquet: {missing}"
        )
    filtered = icc_df.loc[icc_df["cohort"].isin(requested_cohorts)].copy()
    counts = filtered.groupby("feature_name")["cohort"].nunique()
    locked = set(counts[counts == len(requested_cohorts)].index.astype(str))
    logger.info(
        "Derived locked feature set across %d cohorts: %d features.",
        len(requested_cohorts),
        len(locked),
    )
    return locked, {
        "requested_cohorts": requested_cohorts,
        "available_cohorts": available_cohorts,
        "locked_feature_count": int(len(locked)),
    }


def load_arm_specific_icc(
    locked_features: set[str], logger: logging.Logger
) -> tuple[pd.DataFrame, dict[str, object]]:
    meta: dict[str, object] = {}
    if NSCLC_HUMAN_AI_ICC_PARQUET.exists():
        icc_df = pd.read_parquet(NSCLC_HUMAN_AI_ICC_PARQUET)
        required = {
            "feature_name",
            "human_icc",
            "ai_icc",
            "feature_family",
        }
        missing = required.difference(icc_df.columns)
        if missing:
            raise RuntimeError(
                f"{NSCLC_HUMAN_AI_ICC_PARQUET} missing columns: {sorted(missing)}"
            )
        meta["icc_source"] = str(NSCLC_HUMAN_AI_ICC_PARQUET)
    else:
        icc_df = pd.read_parquet(ICC_RESULTS_PARQUET)
        if {"human_icc", "ai_icc", "feature_family"}.issubset(icc_df.columns):
            meta["icc_source"] = str(ICC_RESULTS_PARQUET)
        else:
            raise RuntimeError(
                "Human-vs-AI class assignment requires arm-specific ICC columns. "
                f"{NSCLC_HUMAN_AI_ICC_PARQUET} is absent and "
                f"{ICC_RESULTS_PARQUET} does not expose human_icc/ai_icc."
            )

    out = icc_df.rename(columns={"feature_family": "icc_family"}).copy()
    out["feature_name"] = out["feature_name"].astype(str)
    before = int(out["feature_name"].nunique())
    out = out.loc[out["feature_name"].isin(locked_features)].copy()
    available = set(out["feature_name"])
    missing_locked = sorted(locked_features.difference(available))
    if missing_locked:
        logger.warning(
            "Locked feature set contains %d features without arm-specific ICCs; "
            "dropping them. Example: %s",
            len(missing_locked),
            ", ".join(missing_locked[:3]),
        )
    out = out[["feature_name", "human_icc", "ai_icc", "icc_family"]].drop_duplicates()
    meta["locked_feature_count_before_arm_join"] = before
    meta["available_arm_specific_features"] = int(out["feature_name"].nunique())
    meta["missing_locked_features"] = missing_locked
    return out, meta


def compute_feature_covariates(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["feature_name", "family"], observed=True, sort=False)
        .agg(
            n_subjects=("subject", "nunique"),
            human_median_cov_pct=("human_cov_pct", "median"),
            ai_median_cov_pct=("ai_cov_pct", "median"),
        )
        .reset_index()
    )
    out["median_delta_cov_pct"] = out["human_median_cov_pct"] - out["ai_median_cov_pct"]
    return out


def classify_feature(icc_value: float, cov_value: float) -> str:
    if pd.notna(icc_value) and pd.notna(cov_value):
        if icc_value >= 0.90 and cov_value <= 10.0:
            return "Robust"
        if 0.75 <= icc_value < 0.90 and cov_value <= 20.0:
            return "Acceptable"
    return "Poor"


def transition_label(human_class: str, ai_class: str) -> str:
    if human_class == "Poor" and ai_class == "Acceptable":
        return "Poor->Acceptable"
    if human_class == "Poor" and ai_class == "Robust":
        return "Poor->Robust"
    if human_class == "Acceptable" and ai_class == "Robust":
        return "Acceptable->Robust"
    return "No improvement / other"


def assign_quartiles(feature_df: pd.DataFrame) -> pd.DataFrame:
    out = feature_df.copy()
    ranks = out["human_median_cov_pct"].rank(method="first")
    out["quartile"] = pd.qcut(ranks, q=4, labels=QUARTILE_ORDER)
    return out


def build_transition_table(
    raw_deviations: pd.DataFrame,
    arm_icc_df: pd.DataFrame,
    allowed_features: set[str],
) -> pd.DataFrame:
    filtered = raw_deviations.loc[raw_deviations["feature_name"].isin(allowed_features)].copy()
    feature_cov = compute_feature_covariates(filtered)
    merged = feature_cov.merge(arm_icc_df, on="feature_name", how="inner")
    if "family" in merged.columns and "icc_family" in merged.columns:
        mismatch = merged.loc[
            merged["icc_family"].notna()
            & (merged["family"].astype(str) != merged["icc_family"].astype(str))
        ]
        if not mismatch.empty:
            raise RuntimeError("Family mismatch between deviations and arm-specific ICC table.")
    merged = assign_quartiles(merged)
    merged["human_class"] = [
        classify_feature(icc_value, cov_value)
        for icc_value, cov_value in zip(
            merged["human_icc"].to_numpy(dtype=float),
            merged["human_median_cov_pct"].to_numpy(dtype=float),
        )
    ]
    merged["ai_class"] = [
        classify_feature(icc_value, cov_value)
        for icc_value, cov_value in zip(
            merged["ai_icc"].to_numpy(dtype=float),
            merged["ai_median_cov_pct"].to_numpy(dtype=float),
        )
    ]
    merged["transition"] = [
        transition_label(human_class, ai_class)
        for human_class, ai_class in zip(
            merged["human_class"].astype(str),
            merged["ai_class"].astype(str),
        )
    ]
    class_rank = {"Poor": 0, "Acceptable": 1, "Robust": 2}
    merged["class_delta"] = (
        merged["ai_class"].map(class_rank) - merged["human_class"].map(class_rank)
    )
    merged["quartile"] = pd.Categorical(
        merged["quartile"], categories=QUARTILE_ORDER, ordered=True
    )
    return merged


def summarise_transitions(feature_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for quartile in QUARTILE_ORDER:
        quartile_block = feature_df.loc[feature_df["quartile"] == quartile]
        for family in FAMILY_ORDER:
            sub = quartile_block.loc[quartile_block["family"] == family]
            if sub.empty:
                continue
            n_features = int(len(sub))
            med_human = float(sub["human_median_cov_pct"].median())
            med_ai = float(sub["ai_median_cov_pct"].median())
            med_delta = float(sub["median_delta_cov_pct"].median())
            for transition in TRANSITION_ORDER:
                n_transition = int((sub["transition"] == transition).sum())
                rows.append(
                    {
                        "quartile": quartile,
                        "family": family,
                        "transition": transition,
                        "n_features": n_features,
                        "n_transition": n_transition,
                        "transition_rate_pct": 100.0 * n_transition / max(n_features, 1),
                        "median_human_cov_pct": med_human,
                        "median_ai_cov_pct": med_ai,
                        "median_delta_cov_pct": med_delta,
                    }
                )
    return pd.DataFrame(rows)


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


def bootstrap_transition_cis(
    raw_deviations: pd.DataFrame,
    arm_icc_df: pd.DataFrame,
    allowed_features: set[str],
    n_boot: int,
    seed: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[pd.DataFrame] = []
    for iteration in range(n_boot):
        boot_df = resample_subject_clusters(raw_deviations, rng)
        transition_df = build_transition_table(boot_df, arm_icc_df, allowed_features)
        summary = summarise_transitions(transition_df)
        summary["bootstrap_iteration"] = iteration
        records.append(
            summary[
                [
                    "quartile",
                    "family",
                    "transition",
                    "transition_rate_pct",
                    "bootstrap_iteration",
                ]
            ]
        )
        if (iteration + 1) % 100 == 0 or iteration == n_boot - 1:
            logger.info("Bootstrap %d/%d complete.", iteration + 1, n_boot)
    boot_df = pd.concat(records, ignore_index=True)
    ci_rows: list[dict[str, object]] = []
    for (quartile, family, transition), sub in boot_df.groupby(
        ["quartile", "family", "transition"], sort=False
    ):
        ci_low, ci_high = percentile_ci(sub["transition_rate_pct"].to_numpy(dtype=float))
        ci_rows.append(
            {
                "quartile": quartile,
                "family": family,
                "transition": transition,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return pd.DataFrame(ci_rows)


def plot_transition_summary(summary_df: pd.DataFrame) -> None:
    colors = {
        "Poor->Acceptable": "#f4a261",
        "Poor->Robust": "#2a9d8f",
        "Acceptable->Robust": "#457b9d",
        "No improvement / other": "#c0c7d1",
    }
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=True)
    axis_list = axes.ravel()
    x = np.arange(len(QUARTILE_ORDER))

    for axis, family in zip(axis_list, FAMILY_ORDER + ["__unused__"]):
        if family == "__unused__":
            axis.axis("off")
            continue
        sub = summary_df.loc[summary_df["family"] == family].copy()
        pivot = (
            sub.pivot_table(
                index="quartile",
                columns="transition",
                values="transition_rate_pct",
                aggfunc="first",
            )
            .reindex(QUARTILE_ORDER)
            .fillna(0.0)
        )
        bottom = np.zeros(len(QUARTILE_ORDER))
        for transition in TRANSITION_ORDER:
            values = pivot[transition].to_numpy(dtype=float) if transition in pivot else np.zeros(len(QUARTILE_ORDER))
            axis.bar(
                x,
                values,
                bottom=bottom,
                color=colors[transition],
                width=0.72,
                label=transition,
            )
            bottom += values
        axis.set_title(family)
        axis.set_xticks(x)
        axis.set_xticklabels(QUARTILE_ORDER)
        axis.set_ylim(0, 100)
        axis.grid(axis="y", linestyle=":", alpha=0.35)

    axes[0, 0].set_ylabel("Transition rate (%)")
    axes[1, 0].set_ylabel("Transition rate (%)")
    handles, labels = axis_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Feature-class transitions by baseline human CoV quartile", y=1.02)
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
    requested_cohorts: list[str],
    input_meta: dict[str, object],
    locked_meta: dict[str, object],
    icc_meta: dict[str, object],
    feature_count_used: int,
    smoke_test_family: str | None,
) -> Path:
    manifest_path = LOGS_DIR / f"37_manifest_{job_id}.json"
    checksums = {
        str(DEVIATIONS_PARQUET): sha256_file(DEVIATIONS_PARQUET),
        str(ICC_RESULTS_PARQUET): sha256_file(ICC_RESULTS_PARQUET),
    }
    icc_source = icc_meta.get("icc_source")
    if icc_source:
        checksums[str(icc_source)] = sha256_file(Path(icc_source))
    manifest = {
        "script_path": str(Path(__file__).resolve()),
        "argv": argv,
        "host": socket.gethostname(),
        "job_id": job_id,
        "project_path": str(Path(__file__).resolve().parents[1]),
        "bootstrap_seed": SEED,
        "bootstrap_iterations": bootstrap_iterations,
        "requested_cohorts": requested_cohorts,
        "input_file_checksums": checksums,
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
        "locked_feature_meta": locked_meta,
        "icc_meta": icc_meta,
        "feature_count_used": feature_count_used,
        "smoke_test_family": smoke_test_family,
        "classification_thresholds": {
            "robust": "ICC >= 0.90 and CoV <= 10%",
            "acceptable": "0.75 <= ICC < 0.90 and CoV <= 20%",
            "poor": "all other combinations",
        },
        "method_notes": {
            "locked_feature_set": "Derived from icc_results.parquet cohort coverage; default 10-cohort intersection is 107 features.",
            "arm_specific_icc": "Human-vs-AI class assignment uses the NSCLC arm-specific ICC parquet because icc_results.parquet lacks separate human/ai columns.",
            "bootstrap_scope": "Bootstrap resamples patient-level CoV summaries; ICC values remain fixed because only feature-level arm-specific ICC estimates are available.",
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="Restrict the analysis to the GLSZM family and cap bootstrap iterations at 50.")
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=N_BOOT_DEFAULT,
        help="Number of patient-clustered bootstrap iterations.",
    )
    parser.add_argument(
        "--cohorts",
        default=",".join(DEFAULT_COHORTS),
        help="Comma-separated cohort list used to derive the locked feature universe.",
    )
    parser.add_argument(
        "--job-id",
        default=os.environ.get("JOB_ID", "local"),
        help="Job identifier used in logs and manifest output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    requested_cohorts = parse_cohorts(args.cohorts)
    for directory in (DATA_DIR, TABLES_DIR, FIGURES_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    logger = configure_logging(args.job_id)
    n_boot = args.bootstrap_iterations
    smoke_test_family: str | None = None
    if args.smoke_test and n_boot > 50:
        logger.info(
            "Smoke test requested: reducing bootstrap iterations from %d to 50.",
            n_boot,
        )
        n_boot = 50

    t0 = time.time()
    logger.info("Starting step 37 stratified class-upgrade analysis.")
    raw_deviations, input_meta = load_deviations(logger)
    locked_features, locked_meta = derive_locked_feature_set(requested_cohorts, logger)
    arm_icc_df, icc_meta = load_arm_specific_icc(locked_features, logger)
    allowed_features = set(arm_icc_df["feature_name"].astype(str))

    if args.smoke_test:
        smoke_test_family = "glszm"
        glszm_features = {
            feature_name
            for feature_name, family in zip(
                arm_icc_df["feature_name"].astype(str),
                arm_icc_df["icc_family"].astype(str),
            )
            if family == smoke_test_family
        }
        if not glszm_features:
            smoke_test_family = str(arm_icc_df["icc_family"].astype(str).iloc[0])
            glszm_features = {
                feature_name
                for feature_name, family in zip(
                    arm_icc_df["feature_name"].astype(str),
                    arm_icc_df["icc_family"].astype(str),
                )
                if family == smoke_test_family
            }
        allowed_features = allowed_features.intersection(glszm_features)
        logger.info(
            "Smoke test enabled: restricted analysis to family=%s (%d features).",
            smoke_test_family,
            len(allowed_features),
        )

    working_deviations = raw_deviations.loc[
        raw_deviations["feature_name"].isin(allowed_features)
    ].copy()
    logger.info(
        "Transition analysis will use %d rows across %d features after feature-universe filtering.",
        len(working_deviations),
        working_deviations["feature_name"].nunique(),
    )

    transition_df = build_transition_table(working_deviations, arm_icc_df, allowed_features)
    summary_df = summarise_transitions(transition_df)
    ci_df = bootstrap_transition_cis(
        working_deviations,
        arm_icc_df,
        allowed_features,
        n_boot=n_boot,
        seed=SEED,
        logger=logger,
    )
    summary_df = summary_df.merge(
        ci_df, on=["quartile", "family", "transition"], how="left"
    )

    transition_df.to_parquet(OUT_PARQUET, index=False)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    plot_transition_summary(summary_df)

    manifest_path = write_manifest(
        job_id=args.job_id,
        argv=sys.argv,
        runtime_seconds=time.time() - t0,
        bootstrap_iterations=n_boot,
        requested_cohorts=requested_cohorts,
        input_meta=input_meta,
        locked_meta=locked_meta,
        icc_meta=icc_meta,
        feature_count_used=int(transition_df["feature_name"].nunique()),
        smoke_test_family=smoke_test_family,
    )

    logger.info("Wrote %s", OUT_PARQUET)
    logger.info("Wrote %s", OUT_SUMMARY)
    logger.info("Wrote %s and %s", OUT_FIG_PNG, OUT_FIG_PDF)
    logger.info("Wrote %s", manifest_path)
    logger.info(
        "Step 37 completed successfully with %d transition features.",
        transition_df["feature_name"].nunique(),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - defensive exit path
        print(f"[FATAL] {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(2)
