#!/usr/bin/env /home/kgs24/miniforge3/envs/radiomics_sge/bin/python
"""
Step 51: Outcome-anchored robust feature whitelist.

PURPOSE
-------
Intersect the post-rebuild robustness classes with the fixed step-41 outcome
retention outputs to identify features that are both reproducible and
clinically nontrivial.

Current production scope is explicitly limited by the fixed step-41 analysis:
  - Prostata toxicity (G2+ lymphopenia) only.

INPUTS
------
Preferred BeeGFS inputs:
  /home/kgs24/rtpipeline_manuscript/analysis/data/prognostic_retention_feature_effects_fixed.parquet
  /home/kgs24/rtpipeline_manuscript/analysis/tables/prognostic_retention_panel_cv_folds_fixed.csv

Local mirrors used when present:
  <project>/analysis/data/prognostic_retention_feature_effects_fixed.parquet
  <project>/analysis/tables/prognostic_retention_panel_cv_folds_fixed.csv

OUTPUTS
-------
  analysis/data/outcome_anchored_whitelist_annotated.parquet
      Full feature table with q-values, CV support counts, and whitelist flags.

  analysis/tables/outcome_anchored_robust_whitelist.csv
      Filtered robust-feature shortlist for the current production endpoint.

  analysis/tables/outcome_anchored_whitelist_family_summary.csv
      Family-level counts and yield fractions.

  analysis/figures/figure_outcome_anchored_whitelist.{png,pdf}
      Top robust features ranked by combined evidence.

  analysis/logs/51_manifest_<JOBID>.json
      Provenance, scope notes, and counts.
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

FAMILY_ORDER = ["shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
PRODUCTION_LEG = "prostata_toxicity_g2plus"


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
    logger = logging.getLogger("script51")
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


def resolve_analysis_dir() -> Path:
    candidates = [
        Path("/home/kgs24/rtpipeline_manuscript/analysis"),
        Path("/umed-projekty/rtpipeline/manuscript/analysis"),
        Path(__file__).resolve().parents[1] / "analysis",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


ANALYSIS_DIR = resolve_analysis_dir()
DATA_DIR = ANALYSIS_DIR / "data"
TABLES_DIR = ANALYSIS_DIR / "tables"
FIGURES_DIR = ANALYSIS_DIR / "figures"
LOGS_DIR = ANALYSIS_DIR / "logs"

FEATURE_EFFECTS_CANDIDATES = [
    DATA_DIR / "prognostic_retention_feature_effects_fixed.parquet",
    Path("/home/kgs24/rtpipeline_manuscript/analysis/data/prognostic_retention_feature_effects_fixed.parquet"),
]
PANEL_FOLDS_CANDIDATES = [
    TABLES_DIR / "prognostic_retention_panel_cv_folds_fixed.csv",
    Path("/home/kgs24/rtpipeline_manuscript/analysis/tables/prognostic_retention_panel_cv_folds_fixed.csv"),
]

OUT_ANNOTATED = DATA_DIR / "outcome_anchored_whitelist_annotated.parquet"
OUT_WHITELIST = TABLES_DIR / "outcome_anchored_robust_whitelist.csv"
OUT_FAMILY = TABLES_DIR / "outcome_anchored_whitelist_family_summary.csv"
OUT_FIG_PNG = FIGURES_DIR / "figure_outcome_anchored_whitelist.png"
OUT_FIG_PDF = FIGURES_DIR / "figure_outcome_anchored_whitelist.pdf"


def ensure_output_dirs() -> None:
    for path in (DATA_DIR, TABLES_DIR, FIGURES_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


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


def resolve_first_existing(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No existing file among candidates: {candidates}")


def bh_fdr(p_values: pd.Series) -> pd.Series:
    clean = pd.to_numeric(p_values, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=clean.index, dtype=float)
    valid = clean.notna()
    if not valid.any():
        return out
    ordered = clean.loc[valid].sort_values()
    n = len(ordered)
    adjusted = ordered * n / np.arange(1, n + 1)
    adjusted = adjusted[::-1].cummin()[::-1].clip(upper=1.0)
    out.loc[adjusted.index] = adjusted
    return out


def load_feature_effects(logger: logging.Logger) -> tuple[pd.DataFrame, dict[str, object]]:
    path = resolve_first_existing(FEATURE_EFFECTS_CANDIDATES)
    df = pd.read_parquet(path)
    required = {
        "feature_name",
        "feature_family",
        "robustness_class",
        "p_value",
        "neg_log10_p",
        "leg",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Feature-effects parquet missing columns: {sorted(missing)}")
    logger.info("Using feature-effects parquet: %s", path)
    df = df.copy()
    df["feature_family"] = df["feature_family"].astype(str).str.lower()
    df["robustness_class"] = df["robustness_class"].astype(str)
    df["leg"] = df["leg"].astype(str)
    return df, {
        "feature_effects_path": str(path),
        "feature_effects_sha256": sha256_file(path),
        "feature_effect_rows": int(len(df)),
        "feature_effect_unique_features": int(df["feature_name"].nunique()),
    }


def load_panel_folds(logger: logging.Logger) -> tuple[pd.DataFrame, dict[str, object]]:
    path = resolve_first_existing(PANEL_FOLDS_CANDIDATES)
    df = pd.read_csv(path)
    required = {"robustness_class", "selected_features", "fold_id", "leg"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Panel-fold CSV missing columns: {sorted(missing)}")
    logger.info("Using panel-fold CSV: %s", path)
    return df, {
        "panel_folds_path": str(path),
        "panel_folds_sha256": sha256_file(path),
        "panel_fold_rows": int(len(df)),
    }


def compute_selection_support(panel_folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in panel_folds.iterrows():
        features_blob = str(row.get("selected_features", ""))
        for feature_name in [token for token in features_blob.split("|") if token]:
            rows.append(
                {
                    "leg": str(row["leg"]),
                    "robustness_class": str(row["robustness_class"]),
                    "feature_name": feature_name,
                    "selected_in_fold": 1,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["leg", "robustness_class", "feature_name", "cv_selected_folds", "cv_selection_rate"]
        )
    support = pd.DataFrame(rows)
    folds_by_class = (
        panel_folds.groupby(["leg", "robustness_class"], as_index=False)["fold_id"]
        .nunique()
        .rename(columns={"fold_id": "n_folds_in_class"})
    )
    support = (
        support.groupby(["leg", "robustness_class", "feature_name"], as_index=False)["selected_in_fold"]
        .sum()
        .rename(columns={"selected_in_fold": "cv_selected_folds"})
        .merge(folds_by_class, on=["leg", "robustness_class"], how="left")
    )
    support["cv_selection_rate"] = (
        support["cv_selected_folds"] / support["n_folds_in_class"].replace(0, np.nan)
    )
    return support


def annotate_features(
    feature_df: pd.DataFrame,
    support_df: pd.DataFrame,
    nominal_p: float,
    fdr_q: float,
) -> pd.DataFrame:
    out = feature_df.copy()
    out["p_value"] = pd.to_numeric(out["p_value"], errors="coerce")
    out["neg_log10_p"] = pd.to_numeric(out["neg_log10_p"], errors="coerce")
    out = out.loc[out["leg"] == PRODUCTION_LEG].copy()
    if out.empty:
        raise RuntimeError(f"No rows found for production leg {PRODUCTION_LEG}.")
    out["fdr_q"] = bh_fdr(out["p_value"])
    out = out.merge(
        support_df,
        on=["leg", "robustness_class", "feature_name"],
        how="left",
    )
    out["cv_selected_folds"] = out["cv_selected_folds"].fillna(0).astype(int)
    out["cv_selection_rate"] = out["cv_selection_rate"].fillna(0.0)
    out["is_robust"] = out["robustness_class"].eq("Robust")
    out["passes_nominal"] = out["p_value"] < nominal_p
    out["passes_fdr"] = out["fdr_q"] < fdr_q
    out["cv_supported"] = out["cv_selected_folds"] > 0
    out["whitelist_nominal"] = out["is_robust"] & out["passes_nominal"]
    out["whitelist_fdr"] = out["is_robust"] & out["passes_fdr"]
    out["whitelist_panel_supported"] = out["whitelist_nominal"] & out["cv_supported"]
    out["combined_evidence_score"] = (
        out["neg_log10_p"].fillna(0.0) + out["cv_selection_rate"].fillna(0.0)
    )
    out = out.sort_values(
        ["whitelist_panel_supported", "whitelist_nominal", "combined_evidence_score"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    return out


def build_family_summary(annotated: pd.DataFrame) -> pd.DataFrame:
    grouped = annotated.groupby("feature_family", dropna=False)
    rows: list[dict[str, object]] = []
    for family_name, block in grouped:
        n_total = int(len(block))
        n_robust = int(block["is_robust"].sum())
        n_nominal = int(block["whitelist_nominal"].sum())
        n_fdr = int(block["whitelist_fdr"].sum())
        n_panel = int(block["whitelist_panel_supported"].sum())
        rows.append(
            {
                "feature_family": family_name,
                "n_total_features": n_total,
                "n_robust_features": n_robust,
                "n_whitelist_nominal": n_nominal,
                "n_whitelist_fdr": n_fdr,
                "n_whitelist_panel_supported": n_panel,
                "robust_fraction_pct": (100.0 * n_robust / n_total) if n_total else np.nan,
                "whitelist_nominal_fraction_of_robust_pct": (100.0 * n_nominal / n_robust) if n_robust else np.nan,
                "whitelist_panel_fraction_of_robust_pct": (100.0 * n_panel / n_robust) if n_robust else np.nan,
                "top_feature": block.iloc[0]["feature_name"] if n_total else None,
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["feature_family"] = pd.Categorical(
            summary["feature_family"], categories=FAMILY_ORDER, ordered=True
        )
        summary = summary.sort_values("feature_family").reset_index(drop=True)
    return summary


def build_whitelist_table(annotated: pd.DataFrame) -> pd.DataFrame:
    keep = annotated.loc[
        annotated["whitelist_nominal"] | annotated["whitelist_fdr"] | annotated["whitelist_panel_supported"]
    ].copy()
    cols = [
        "feature_name",
        "feature_family",
        "robustness_class",
        "p_value",
        "fdr_q",
        "neg_log10_p",
        "log_or",
        "or",
        "or_ci_lo",
        "or_ci_hi",
        "cv_selected_folds",
        "cv_selection_rate",
        "whitelist_nominal",
        "whitelist_fdr",
        "whitelist_panel_supported",
        "combined_evidence_score",
    ]
    existing = [col for col in cols if col in keep.columns]
    return keep.loc[:, existing].copy()


def build_figure(whitelist: pd.DataFrame) -> None:
    top = whitelist.head(15).copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    if not top.empty:
        plot_top = top.iloc[::-1].copy()
        axes[0].barh(plot_top["feature_name"], plot_top["neg_log10_p"], color="#1f77b4")
        axes[0].set_title("Top robust features by -log10(p)")
        axes[0].set_xlabel("-log10(p)")

        axes[1].barh(plot_top["feature_name"], plot_top["cv_selection_rate"], color="#2a9d8f")
        axes[1].set_title("Nested-CV support")
        axes[1].set_xlabel("Selection rate")
        axes[1].set_xlim(0.0, 1.0)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "No whitelist features", ha="center", va="center")
            ax.set_axis_off()

    fig.suptitle("Step 51: Outcome-anchored robust whitelist")
    fig.tight_layout()
    fig.savefig(OUT_FIG_PNG, dpi=300)
    fig.savefig(OUT_FIG_PDF)
    plt.close(fig)


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", default=os.environ.get("JOB_ID", "manual"))
    parser.add_argument("--nominal-p-threshold", type=float, default=0.05)
    parser.add_argument("--fdr-q-threshold", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    ensure_output_dirs()
    args = parse_args()
    logger = configure_logging(args.job_id)
    manifest_path = LOGS_DIR / f"51_manifest_{args.job_id}.json"
    manifest: dict[str, object] = {
        "job_id": args.job_id,
        "hostname": socket.gethostname(),
        "started_at_epoch": time.time(),
        "analysis_dir": str(ANALYSIS_DIR),
        "script_path": str(Path(__file__).resolve()),
        "git_sha": resolve_git_sha(),
        "production_scope": PRODUCTION_LEG,
        "scope_note": "Current whitelist is limited to the fixed step-41 Prostata toxicity production leg; Odbytnice is not yet included.",
        "thresholds": {
            "nominal_p": float(args.nominal_p_threshold),
            "fdr_q": float(args.fdr_q_threshold),
        },
    }

    try:
        feature_df, feature_meta = load_feature_effects(logger)
        panel_df, panel_meta = load_panel_folds(logger)
        support_df = compute_selection_support(panel_df)
        annotated = annotate_features(
            feature_df=feature_df,
            support_df=support_df,
            nominal_p=float(args.nominal_p_threshold),
            fdr_q=float(args.fdr_q_threshold),
        )
        whitelist = build_whitelist_table(annotated)
        family_summary = build_family_summary(annotated.loc[annotated["is_robust"]].copy())

        annotated.to_parquet(OUT_ANNOTATED, index=False)
        whitelist.to_csv(OUT_WHITELIST, index=False)
        family_summary.to_csv(OUT_FAMILY, index=False)
        build_figure(whitelist=whitelist)

        manifest.update(feature_meta)
        manifest.update(panel_meta)
        manifest["support_rows"] = int(len(support_df))
        manifest["result_counts"] = {
            "annotated_rows": int(len(annotated)),
            "robust_rows": int(annotated["is_robust"].sum()),
            "whitelist_nominal_rows": int(annotated["whitelist_nominal"].sum()),
            "whitelist_fdr_rows": int(annotated["whitelist_fdr"].sum()),
            "whitelist_panel_supported_rows": int(annotated["whitelist_panel_supported"].sum()),
        }
        manifest["outputs"] = {
            "annotated_parquet": str(OUT_ANNOTATED),
            "whitelist_csv": str(OUT_WHITELIST),
            "family_summary_csv": str(OUT_FAMILY),
            "figure_png": str(OUT_FIG_PNG),
            "figure_pdf": str(OUT_FIG_PDF),
        }
        manifest["completed"] = True
        manifest["ended_at_epoch"] = time.time()
        write_manifest(manifest_path, manifest)
        logger.info("Step 51 complete. Manifest: %s", manifest_path)
    except Exception as exc:
        manifest["completed"] = False
        manifest["error"] = str(exc)
        manifest["traceback"] = traceback.format_exc()
        manifest["ended_at_epoch"] = time.time()
        write_manifest(manifest_path, manifest)
        logger.error("Step 51 failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
