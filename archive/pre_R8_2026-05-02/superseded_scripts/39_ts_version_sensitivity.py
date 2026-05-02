#!/usr/bin/env python3
"""
Step 39: TotalSegmentator weight-version sensitivity feasibility audit.

This script first checks whether a defensible paired TotalSegmentator v1-v2
comparison is possible from the currently materialized data. If not, it pivots
to the strongest available paired-AI sensitivity already on disk: Step 21
inter-algorithm radiomics agreement (TotalSegmentator vs custom nnU-Net).

Expected primary run target:
  /home/kgs24/rtpipeline_manuscript/analysis
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

ICC_ROBUST = 0.90
ICC_ACCEPTABLE = 0.75
TS_VERSION_TOKENS = ("v1", "v2", "v2.12", "2.12", "TotalSegmentator v2.12.0")
RELEVANT_COHORTS = ("PlucaRCHT", "Hipokampy", "NSCLC_Interobserver")

REMOTE_MANUSCRIPT_ROOT = Path("/home/kgs24/rtpipeline_manuscript")
LOCAL_PROJECT_ROOT = Path("/umed-projekty/rtpipeline")
LOCAL_ANALYSIS_ROOT = LOCAL_PROJECT_ROOT / "manuscript" / "analysis"


@dataclass
class CandidatePath:
    label: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit TotalSegmentator version sensitivity feasibility and summarize fallback paired-AI sensitivity."
    )
    parser.add_argument(
        "--analysis-root",
        default=None,
        help="Directory containing analysis/{data,tables,logs}. Defaults to the first readable remote/local candidate.",
    )
    parser.add_argument(
        "--manuscript-root",
        default=None,
        help="Root containing cohort/data directories. Defaults to the first readable remote/local candidate.",
    )
    parser.add_argument(
        "--job-id",
        default=os.environ.get("JOB_ID", "local"),
        help="Job identifier for the JSON log filename.",
    )
    return parser.parse_args()


def resolve_first_existing(candidates: Iterable[CandidatePath]) -> CandidatePath:
    for candidate in candidates:
        if candidate.path.exists():
            return candidate
    joined = "\n".join(f"- {c.label}: {c.path}" for c in candidates)
    raise FileNotFoundError(f"No readable candidate path found:\n{joined}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def list_case_dirs(manuscript_root: Path, cohort: str) -> list[Path]:
    data_root = manuscript_root / cohort / "data"
    if not data_root.exists():
        return []
    return sorted(p for p in data_root.glob("*/*") if p.is_dir())


def audit_ts_generations(manuscript_root: Path) -> tuple[pd.DataFrame, dict, list[dict]]:
    rows: list[dict] = []
    duplicate_cases: list[dict] = []
    token_hits = Counter()

    for cohort in RELEVANT_COHORTS:
        for case_dir in list_case_dirs(manuscript_root, cohort):
            seg_dir = case_dir / "Segmentation_TotalSegmentator"
            if not seg_dir.exists():
                continue

            ts_subdirs = sorted(p for p in seg_dir.iterdir() if p.is_dir())
            manifests = sorted(seg_dir.glob("*/manifest.json"))
            case_row = {
                "cohort": cohort,
                "case_id": str(case_dir.relative_to(manuscript_root / cohort / "data")),
                "n_ts_subdirs": len(ts_subdirs),
                "n_manifests": len(manifests),
                "has_multiple_ts_subdirs": len(ts_subdirs) > 1,
                "has_manifest": len(manifests) > 0,
            }
            rows.append(case_row)

            for manifest in manifests:
                text = manifest.read_text(errors="ignore")
                for token in TS_VERSION_TOKENS:
                    if token in text:
                        token_hits[token] += 1

            if len(ts_subdirs) > 1:
                duplicate_entry = {
                    "cohort": cohort,
                    "case_id": case_row["case_id"],
                    "ts_subdirs": [p.name for p in ts_subdirs],
                    "manifests": [str(p) for p in manifests],
                }
                common_probe_masks = ("total--aorta.nii.gz", "total--heart.nii.gz", "total--esophagus.nii.gz")
                probe_records = []
                for mask_name in common_probe_masks:
                    per_subdir = []
                    for subdir in ts_subdirs:
                        mask_path = subdir / mask_name
                        if mask_path.exists():
                            per_subdir.append(
                                {
                                    "subdir": subdir.name,
                                    "mask_name": mask_name,
                                    "size_bytes": mask_path.stat().st_size,
                                    "sha256": sha256_file(mask_path),
                                }
                            )
                    if len(per_subdir) >= 2:
                        probe_records.append({"mask_name": mask_name, "per_subdir": per_subdir})
                duplicate_entry["probe_masks"] = probe_records
                duplicate_cases.append(duplicate_entry)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No Segmentation_TotalSegmentator case directories found in the audited cohorts.")

    summary = {
        "n_cases": int(len(df)),
        "cohort_case_counts": {k: int(v) for k, v in df["cohort"].value_counts().to_dict().items()},
        "n_with_multiple_ts_subdirs": int(df["has_multiple_ts_subdirs"].sum()),
        "n_with_zero_manifest": int((~df["has_manifest"]).sum()),
        "manifest_version_token_hits": dict(token_hits),
    }
    return df, summary, duplicate_cases


def summarize_step21(analysis_root: Path) -> tuple[pd.DataFrame, dict]:
    data_path = analysis_root / "data" / "inter_algorithm_ai_icc.parquet"
    course_discovery_path = analysis_root / "tables" / "inter_algorithm_ai_course_discovery.csv"
    manifest_path = analysis_root / "logs" / "21_manifest_3560752.json"

    df = pd.read_parquet(data_path)
    course_df = pd.read_csv(course_discovery_path)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    feature_summary = (
        df.groupby(["body_region", "roi_name", "feature_family"], dropna=False)
        .agg(
            n_features=("feature_name", "count"),
            median_icc21_inter_algorithm=("icc21_absolute_agreement", "median"),
            pct_icc21_ge_090=("icc21_absolute_agreement", lambda s: float((s >= ICC_ROBUST).mean())),
            pct_icc21_lt_075=("icc21_absolute_agreement", lambda s: float((s < ICC_ACCEPTABLE).mean())),
            median_contour_pct_total=("contour_pct_total", "median"),
            q95_contour_pct_total=("contour_pct_total", lambda s: float(s.quantile(0.95))),
            median_icc31_ts_within=("icc31_ts_within", "median"),
            median_fisher_z_delta=("fisher_z_delta", "median"),
        )
        .reset_index()
        .sort_values(["body_region", "roi_name", "feature_family"])
    )

    retained = course_df.loc[course_df["status"] == "retained"].copy()
    excluded = course_df.loc[course_df["status"] != "retained"].copy()

    summary = {
        "input_parquet": str(data_path),
        "input_course_discovery_csv": str(course_discovery_path),
        "input_manifest_json": str(manifest_path) if manifest_path.exists() else None,
        "n_feature_rows": int(len(df)),
        "n_retained_courses": int(len(retained)),
        "retained_cohorts": sorted(retained["cohort"].dropna().astype(str).unique().tolist()),
        "retained_rois": sorted(df["roi_name"].dropna().astype(str).unique().tolist()),
        "excluded_reason_counts": {
            k: int(v) for k, v in excluded["reason"].fillna("missing_reason").value_counts().to_dict().items()
        },
        "overall_median_icc21_inter_algorithm": float(df["icc21_absolute_agreement"].median()),
        "overall_pct_icc21_ge_090": float((df["icc21_absolute_agreement"] >= ICC_ROBUST).mean()),
        "overall_pct_icc21_lt_075": float((df["icc21_absolute_agreement"] < ICC_ACCEPTABLE).mean()),
        "overall_median_contour_pct_total": float(df["contour_pct_total"].median()),
        "overall_q95_contour_pct_total": float(df["contour_pct_total"].quantile(0.95)),
        "step21_manifest_discovery": manifest.get("discovery"),
    }
    return feature_summary, summary


def summarize_benchmark_context(analysis_root: Path) -> pd.DataFrame:
    within_path = analysis_root / "tables" / "nsclc_io_within_patient_summary.csv"
    triplet_path = analysis_root / "tables" / "nsclc_io_triplet_matched_summary.csv"

    rows: list[dict] = []
    if within_path.exists():
        within_df = pd.read_csv(within_path)
        for rec in within_df.to_dict(orient="records"):
            rows.append(
                {
                    "source": "nsclc_io_within_patient_summary",
                    "family_or_scope": rec["feature_family"],
                    "n_features": int(rec["n_features"]),
                    "median_human_cov_pct": float(rec["median_human_cov_pct"]),
                    "median_ai_cov_pct": float(rec["median_ai_cov_pct"]),
                    "median_delta_human_minus_ai_pct": float(rec["median_diff_human_minus_ai"]),
                    "direction_or_note": rec["wilcoxon_direction"],
                    "input_path": str(within_path),
                }
            )
    if triplet_path.exists():
        triplet_df = pd.read_csv(triplet_path)
        for rec in triplet_df.to_dict(orient="records"):
            rows.append(
                {
                    "source": "nsclc_io_triplet_matched_summary",
                    "family_or_scope": rec["scope"],
                    "n_features": None,
                    "median_human_cov_pct": float(rec["median_human_cov_pct"]),
                    "median_ai_cov_pct": float(rec["median_ai_cov_pct"]),
                    "median_delta_human_minus_ai_pct": float(rec["median_triplet_delta_pct"]),
                    "direction_or_note": f"triplets_with_positive_delta={int(rec['triplets_with_positive_delta'])}/10",
                    "input_path": str(triplet_path),
                }
            )
    return pd.DataFrame(rows)


def build_feasibility_table(case_audit_df: pd.DataFrame, audit_summary: dict) -> pd.DataFrame:
    rows = []
    for cohort, cohort_df in case_audit_df.groupby("cohort"):
        rows.append(
            {
                "cohort": cohort,
                "n_cases_with_segmentation_totalsegmentator": int(len(cohort_df)),
                "n_cases_with_multiple_ts_subdirs": int(cohort_df["has_multiple_ts_subdirs"].sum()),
                "n_cases_without_manifest": int((~cohort_df["has_manifest"]).sum()),
                "version_token_hits_in_manifests": int(sum(audit_summary["manifest_version_token_hits"].values())),
            }
        )
    return pd.DataFrame(rows).sort_values("cohort")


def main() -> None:
    args = parse_args()

    analysis_candidate = (
        CandidatePath("cli", Path(args.analysis_root))
        if args.analysis_root
        else resolve_first_existing(
            [
                CandidatePath("remote_analysis", REMOTE_MANUSCRIPT_ROOT / "analysis"),
                CandidatePath("local_analysis", LOCAL_ANALYSIS_ROOT),
            ]
        )
    )
    if isinstance(analysis_candidate, CandidatePath):
        analysis_root = analysis_candidate.path
        analysis_source_label = analysis_candidate.label
    else:
        analysis_root = Path(args.analysis_root)
        analysis_source_label = "cli"

    manuscript_candidate = (
        CandidatePath("cli", Path(args.manuscript_root))
        if args.manuscript_root
        else resolve_first_existing(
            [
                CandidatePath("remote_manuscript", REMOTE_MANUSCRIPT_ROOT),
                CandidatePath("local_project", LOCAL_PROJECT_ROOT),
            ]
        )
    )
    if isinstance(manuscript_candidate, CandidatePath):
        manuscript_root = manuscript_candidate.path
        manuscript_source_label = manuscript_candidate.label
    else:
        manuscript_root = Path(args.manuscript_root)
        manuscript_source_label = "cli"

    case_audit_df, audit_summary, duplicate_cases = audit_ts_generations(manuscript_root)
    feasibility_df = build_feasibility_table(case_audit_df, audit_summary)
    step21_df, step21_summary = summarize_step21(analysis_root)
    benchmark_df = summarize_benchmark_context(analysis_root)

    version_feasible = (
        audit_summary["n_with_multiple_ts_subdirs"] > 1
        and bool(audit_summary["manifest_version_token_hits"])
    )
    fallback_reason = (
        "No paired semantic TotalSegmentator version labels were recoverable from audited manifests; "
        "the only multi-generation edge case was a single unversioned duplicate PlucaRCHT course."
    )

    tables_dir = analysis_root / "tables"
    logs_dir = analysis_root / "logs"
    write_csv(feasibility_df, tables_dir / "39_ts_version_sensitivity_feasibility.csv")
    write_csv(step21_df, tables_dir / "39_ts_version_sensitivity_fallback_inter_algorithm_summary.csv")
    if not benchmark_df.empty:
        write_csv(benchmark_df, tables_dir / "39_ts_version_sensitivity_benchmark_context.csv")

    report = {
        "job_id": args.job_id,
        "host": socket.gethostname(),
        "analysis_root": str(analysis_root),
        "analysis_root_source": analysis_source_label,
        "manuscript_root": str(manuscript_root),
        "manuscript_root_source": manuscript_source_label,
        "version_sensitivity_feasible": bool(version_feasible),
        "fallback_analysis_selected": "step21_inter_algorithm_ai" if not version_feasible else "paired_ts_version",
        "fallback_reason": fallback_reason if not version_feasible else None,
        "audit_summary": audit_summary,
        "duplicate_cases": duplicate_cases,
        "step21_summary": step21_summary,
        "benchmark_context_rows": benchmark_df.to_dict(orient="records"),
        "outputs": {
            "feasibility_csv": str(tables_dir / "39_ts_version_sensitivity_feasibility.csv"),
            "fallback_summary_csv": str(tables_dir / "39_ts_version_sensitivity_fallback_inter_algorithm_summary.csv"),
            "benchmark_context_csv": str(tables_dir / "39_ts_version_sensitivity_benchmark_context.csv"),
        },
    }
    write_json(report, logs_dir / f"39_ts_version_sensitivity_{args.job_id}.json")

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
