#!/usr/bin/env python3
"""Audit NSCLC benchmark mask provenance and compute tumor geometry summaries.

This script is intentionally scoped to the reviewer-critical benchmark package:
`/umed-projekty/DICOMRT-datasets/NSCLC_Interobserver/output`.

Outputs:
- manuscript/analysis/tables/nsclc_benchmark_geometry_pairwise_2026-04-21.csv
- manuscript/analysis/tables/nsclc_benchmark_geometry_summary_2026-04-21.csv
- manuscript/analysis/tables/nsclc_benchmark_provenance_audit_2026-04-21.csv

Key design choices:
- Human masks are the five `GTV-1vis-*` masks preserved under `Segmentation_Original`.
- Packaged auto masks are the preserved `GTV-1auto-*` masks under the same manifest.
- The benchmark-matched geometry scope uses `GTV-1auto-1..3` because the archived
  raw-feature benchmark parquet retains a 3-rater AI arm but does not encode the
  original mask filenames. The `auto123` mapping is therefore an explicit,
  documented reconstruction choice rather than a hidden assumption.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

DATASET_ROOT = Path("/umed-projekty/DICOMRT-datasets/NSCLC_Interobserver/output")
PROJECT_ROOT = Path("/umed-projekty/rtpipeline")
TABLES_DIR = PROJECT_ROOT / "manuscript" / "analysis" / "tables"

OUT_PAIRWISE = TABLES_DIR / "nsclc_benchmark_geometry_pairwise_2026-04-21.csv"
OUT_SUMMARY = TABLES_DIR / "nsclc_benchmark_geometry_summary_2026-04-21.csv"
OUT_PROVENANCE = TABLES_DIR / "nsclc_benchmark_provenance_audit_2026-04-21.csv"


@dataclass(frozen=True)
class MaskInfo:
    patient_id: str
    roi_name: str
    path: Path


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def find_case_dirs() -> list[Path]:
    return sorted(path for path in DATASET_ROOT.glob("interobs*/2019-02") if path.is_dir())


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_mask(mask_path: Path) -> tuple[np.ndarray, float]:
    img = nib.load(str(mask_path))
    data = np.asarray(img.get_fdata(), dtype=np.float32) > 0
    zooms = img.header.get_zooms()[:3]
    voxel_volume_mm3 = math.prod(float(v) for v in zooms)
    volume_cc = float(data.sum() * voxel_volume_mm3 / 1000.0)
    return data, volume_cc


def dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    denom = mask_a.sum() + mask_b.sum()
    if denom == 0:
        return float("nan")
    inter = np.logical_and(mask_a, mask_b).sum()
    return float((2.0 * inter) / denom)


def volume_ratio(v1: float, v2: float) -> float:
    if not np.isfinite(v1) or not np.isfinite(v2) or v1 <= 0 or v2 <= 0:
        return float("nan")
    return float(max(v1, v2) / min(v1, v2))


def resolve_original_mask_dir(case_dir: Path, original_meta: dict) -> Path:
    rel_mask = Path(original_meta["structures"][0]["mask"])
    return (case_dir / "Segmentation_Original" / rel_mask.parent).resolve()


def collect_case_payload(case_dir: Path) -> tuple[dict[str, object], dict[str, list[MaskInfo]]]:
    patient_id = case_dir.parent.name
    meta_dir = case_dir / "metadata"
    case_meta = load_json(meta_dir / "case_metadata.json")
    original_meta = load_json(case_dir / "Segmentation_Original" / "CT_5_20190218" / "metadata.json")
    original_mask_dir = resolve_original_mask_dir(case_dir, original_meta)
    totalseg_manifest_paths = [Path(p) for p in case_meta.get("segmentation_totalseg_manifests", [])]
    ts_masks: list[str] = []
    for manifest_path in totalseg_manifest_paths:
        manifest = load_json(manifest_path)
        for model in manifest.get("models", []):
            ts_masks.extend(model.get("masks", []))

    mask_groups = {"human_vis": [], "auto_full": [], "auto123": []}
    for entry in original_meta.get("structures", []):
        roi_name = str(entry["roi_name"])
        mask_path = (case_dir / "Segmentation_Original" / entry["mask"]).resolve()
        info = MaskInfo(patient_id=patient_id, roi_name=roi_name, path=mask_path)
        if roi_name.startswith("GTV-1vis-"):
            mask_groups["human_vis"].append(info)
        elif roi_name.startswith("GTV-1auto-"):
            mask_groups["auto_full"].append(info)
            suffix = roi_name.rsplit("-", 1)[-1]
            if suffix in {"1", "2", "3"}:
                mask_groups["auto123"].append(info)

    for key in mask_groups:
        mask_groups[key] = sorted(mask_groups[key], key=lambda item: item.roi_name)

    provenance = {
        "patient_id": patient_id,
        "case_dir": str(case_dir),
        "rs_path": str(case_meta.get("rs_path", "")),
        "rs_auto_path": str(case_meta.get("rs_auto_path", "")),
        "segmentation_original_dir": str(case_meta.get("segmentation_original_dir", "")),
        "segmentation_totalseg_manifests": "|".join(str(p) for p in totalseg_manifest_paths),
        "human_vis_count": len(mask_groups["human_vis"]),
        "auto_full_count": len(mask_groups["auto_full"]),
        "auto123_count": len(mask_groups["auto123"]),
        "original_manifest_model": str(original_meta.get("model", "")),
        "original_manifest_source_rtstruct": str(original_meta.get("source_rtstruct", "")),
        "totalseg_manifest_count": len(totalseg_manifest_paths),
        "totalseg_has_gtv_mask": any("gtv" in name.lower() for name in ts_masks),
        "totalseg_mask_count": len(ts_masks),
    }
    return provenance, mask_groups


def pairwise_geometry(patient_id: str, arm: str, scope: str, masks: list[MaskInfo]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cache = {mask.path: load_mask(mask.path) for mask in masks}
    for first, second in itertools.combinations(masks, 2):
        mask_a, vol_a_cc = cache[first.path]
        mask_b, vol_b_cc = cache[second.path]
        rows.append(
            {
                "patient_id": patient_id,
                "scope": scope,
                "arm": arm,
                "mask_a": first.roi_name,
                "mask_b": second.roi_name,
                "mask_a_path": str(first.path),
                "mask_b_path": str(second.path),
                "dice": dice(mask_a, mask_b),
                "volume_ratio_max_over_min": volume_ratio(vol_a_cc, vol_b_cc),
                "volume_a_cc": vol_a_cc,
                "volume_b_cc": vol_b_cc,
            }
        )
    return rows


def summarize_geometry(pairwise: pd.DataFrame) -> pd.DataFrame:
    patient_level = (
        pairwise.groupby(["scope", "arm", "patient_id"], as_index=False)
        .agg(
            patient_median_dice=("dice", "median"),
            patient_median_volume_ratio=("volume_ratio_max_over_min", "median"),
            pair_count=("dice", "size"),
        )
    )
    summary = (
        patient_level.groupby(["scope", "arm"], as_index=False)
        .agg(
            patients=("patient_id", "nunique"),
            patient_pairs=("pair_count", "sum"),
            median_of_patient_median_dice=("patient_median_dice", "median"),
            dice_q25=("patient_median_dice", lambda s: float(s.quantile(0.25))),
            dice_q75=("patient_median_dice", lambda s: float(s.quantile(0.75))),
            median_of_patient_median_volume_ratio=("patient_median_volume_ratio", "median"),
            volume_ratio_q25=("patient_median_volume_ratio", lambda s: float(s.quantile(0.25))),
            volume_ratio_q75=("patient_median_volume_ratio", lambda s: float(s.quantile(0.75))),
        )
    )
    return summary.sort_values(["scope", "arm"]).reset_index(drop=True)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    provenance_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []

    for case_dir in find_case_dirs():
        provenance, mask_groups = collect_case_payload(case_dir)
        provenance_rows.append(provenance)

        patient_id = provenance["patient_id"]
        human_masks = mask_groups["human_vis"]
        auto_masks_full = mask_groups["auto_full"]
        auto_masks_123 = mask_groups["auto123"]

        if len(human_masks) >= 2:
            pairwise_rows.extend(pairwise_geometry(patient_id, "human", "package_5v5", human_masks))
            pairwise_rows.extend(pairwise_geometry(patient_id, "human", "benchmark_matched_5v3_inferred_auto123", human_masks))
        if len(auto_masks_full) >= 2:
            pairwise_rows.extend(pairwise_geometry(patient_id, "auto", "package_5v5", auto_masks_full))
        if len(auto_masks_123) >= 2:
            pairwise_rows.extend(pairwise_geometry(patient_id, "auto", "benchmark_matched_5v3_inferred_auto123", auto_masks_123))

    provenance_df = pd.DataFrame(provenance_rows).sort_values("patient_id").reset_index(drop=True)
    pairwise_df = pd.DataFrame(pairwise_rows).sort_values(["scope", "arm", "patient_id", "mask_a", "mask_b"]).reset_index(drop=True)
    summary_df = summarize_geometry(pairwise_df)

    atomic_write_csv(provenance_df, OUT_PROVENANCE)
    atomic_write_csv(pairwise_df, OUT_PAIRWISE)
    atomic_write_csv(summary_df, OUT_SUMMARY)


if __name__ == "__main__":
    main()
