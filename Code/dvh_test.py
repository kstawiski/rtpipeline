#!/usr/bin/env python3
"""Ad hoc DVH comparison utility for manual and auto-generated RTSTRUCTs.

This script is intentionally kept outside the packaged library because it is a
local debugging / exploratory utility. It must not execute any file-system
access at import time, otherwise generic ``pytest`` discovery breaks on
machines that do not have the author's local DICOM paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from dicompylercore import dvhcalc


def load_ct_images(ct_dir: Path) -> list[pydicom.dataset.FileDataset]:
    """Load CT DICOM files from a directory and return them sorted by slice."""
    ct_files = []
    for path in sorted(ct_dir.iterdir()):
        if path.suffix.lower() != ".dcm":
            continue
        ds = pydicom.dcmread(path)
        ct_files.append(ds)
    ct_files.sort(key=lambda ds: int(getattr(ds, "InstanceNumber", 0)))
    return ct_files


def dose_at_fraction(bins, cumulative, fraction):
    """Interpolate the dose at which cumulative volume reaches a fraction."""
    v_total = cumulative[0]
    target = fraction * v_total
    idx = np.where(cumulative <= target)[0]
    if idx.size == 0:
        return bins[-1]
    i = idx[0]
    if i == 0:
        return bins[0]
    x0, x1 = bins[i - 1], bins[i]
    y0, y1 = cumulative[i - 1], cumulative[i]
    if y1 == y0:
        return x0
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)


def get_volume_at_threshold(bins, cumulative, threshold):
    """Estimate the volume receiving at least ``threshold`` Gy."""
    v_total = cumulative[0]
    if threshold <= bins[0]:
        return v_total
    if threshold >= bins[-1]:
        return 0.0
    idx = np.where(bins >= threshold)[0]
    if idx.size == 0:
        return 0.0
    i = idx[0]
    if i == 0:
        return v_total
    x0, x1 = bins[i - 1], bins[i]
    y0, y1 = cumulative[i - 1], cumulative[i]
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (threshold - x0) / (x1 - x0)


def compute_dvh_metrics(abs_dvh, rx_dose):
    """Compute absolute, relative, and Vx DVH metrics."""
    bins_abs = abs_dvh.bincenters
    cum_abs = abs_dvh.counts
    if cum_abs.size == 0 or cum_abs[0] == 0:
        return None

    v_total = cum_abs[0]
    dmean_gy = abs_dvh.mean
    dmax_gy = abs_dvh.max
    dmin_gy = abs_dvh.min
    d95_gy = dose_at_fraction(bins_abs, cum_abs, 0.95)
    d98_gy = dose_at_fraction(bins_abs, cum_abs, 0.98)
    d2_gy = dose_at_fraction(bins_abs, cum_abs, 0.02)
    d50_gy = dose_at_fraction(bins_abs, cum_abs, 0.50)
    hi_abs = (d2_gy - d98_gy) / d50_gy if d50_gy != 0 else np.nan
    spread_gy = dmax_gy - dmin_gy

    v_abs = {}
    v_rel = {}
    for x in range(1, 61):
        vol = get_volume_at_threshold(bins_abs, cum_abs, x)
        v_abs[f"V{x}Gy (cm³)"] = vol
        v_rel[f"V{x}Gy (%)"] = (vol / v_total) * 100

    integral_dose = dmean_gy * v_total

    rel_dvh = abs_dvh.relative_dose(rx_dose)
    bins_rel = rel_dvh.bincenters
    cum_rel = rel_dvh.counts
    dmean_pct = rel_dvh.mean
    dmax_pct = rel_dvh.max
    dmin_pct = rel_dvh.min
    d95_pct = dose_at_fraction(bins_rel, cum_rel, 0.95)
    d98_pct = dose_at_fraction(bins_rel, cum_rel, 0.98)
    d2_pct = dose_at_fraction(bins_rel, cum_rel, 0.02)
    d50_pct = dose_at_fraction(bins_rel, cum_rel, 0.50)
    hi_pct = (d2_pct - d98_pct) / d50_pct if d50_pct != 0 else np.nan
    spread_pct = dmax_pct - dmin_pct

    metrics = {
        "DmeanGy": dmean_gy,
        "DmaxGy": dmax_gy,
        "DminGy": dmin_gy,
        "D95Gy": d95_gy,
        "D98Gy": d98_gy,
        "D2Gy": d2_gy,
        "D50Gy": d50_gy,
        "HI": hi_abs,
        "SpreadGy": spread_gy,
        "Dmean%": dmean_pct,
        "Dmax%": dmax_pct,
        "Dmin%": dmin_pct,
        "D95%": d95_pct,
        "D98%": d98_pct,
        "D2%": d2_pct,
        "D50%": d50_pct,
        "HI%": hi_pct,
        "Spread%": spread_pct,
        "Volume (cm³)": v_total,
        "IntegralDose_Gycm3": integral_dose,
    }
    metrics.update(v_abs)
    metrics.update(v_rel)
    return metrics


def process_rtstruct(rtstruct, rtstruct_file: Path, rtdose, segmentation_source: str, rx_dose_estimated: float):
    """Compute DVH metrics for every ROI in an RTSTRUCT."""
    results = []
    for roi in rtstruct.StructureSetROISequence:
        roi_number = roi.ROINumber
        roi_name = roi.ROIName
        print(f"Processing ROI: {roi_name} (Number: {roi_number}) in {segmentation_source}")
        try:
            abs_dvh = dvhcalc.get_dvh(rtstruct, rtdose, roi_number)
            if abs_dvh.volume == 0:
                print(f"  Skipping ROI {roi_name} (zero volume).")
                continue
            metrics = compute_dvh_metrics(abs_dvh, rx_dose_estimated)
            if metrics is None:
                print(f"  Skipping ROI {roi_name} (no valid DVH data).")
                continue
            metrics.update(
                {
                    "RTStruct": str(rtstruct_file),
                    "ROI_Number": roi_number,
                    "ROI_Name": roi_name,
                    "PrescribedDose_Gy": rx_dose_estimated,
                    "Segmentation_Source": segmentation_source,
                }
            )
            results.append(metrics)
        except Exception as ex:
            print(f"  Error processing ROI {roi_name} in {segmentation_source}: {ex}")
            continue
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DVH metrics across two RTSTRUCTs.")
    parser.add_argument("--rtplan", required=True, type=Path, help="Path to RTPLAN (RP) DICOM file")
    parser.add_argument("--rtdose", required=True, type=Path, help="Path to RTDOSE (RD) DICOM file")
    parser.add_argument("--manual-rtstruct", required=True, type=Path, help="Path to manual/original RTSTRUCT file")
    parser.add_argument("--ct-dir", required=True, type=Path, help="Directory containing CT DICOM slices")
    parser.add_argument(
        "--total-rtstruct",
        type=Path,
        default=None,
        help="Path to TotalSegmentator RTSTRUCT file (default: <ct-dir>_total_dicom/segmentations.dcm)",
    )
    parser.add_argument("--output-excel", type=Path, default=Path("dvh_metrics.xlsx"), help="Output XLSX path")
    parser.add_argument(
        "--default-rx-dose",
        type=float,
        default=50.0,
        help="Fallback prescribed dose in Gy if no CTV1 ROI is found",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rtstruct_total_path = args.total_rtstruct or Path(f"{args.ct_dir}_total_dicom/segmentations.dcm")

    print("Loading common DICOM files...")
    rtplan = pydicom.dcmread(args.rtplan)
    rtdose = pydicom.dcmread(args.rtdose)
    ct_images = load_ct_images(args.ct_dir)
    print(f"Loaded {len(ct_images)} CT image(s).")

    print("\nLoading RT structure sets...")
    rtstruct_manual = pydicom.dcmread(args.manual_rtstruct)
    rtstruct_total = pydicom.dcmread(rtstruct_total_path)

    rx_dose_estimated = None
    for roi in rtstruct_manual.StructureSetROISequence:
        roi_name = roi.ROIName
        if "ctv1" not in roi_name.replace(" ", "").lower():
            continue
        try:
            abs_dvh_ctv = dvhcalc.get_dvh(rtstruct_manual, rtdose, roi.ROINumber)
            if abs_dvh_ctv.volume > 0:
                bins_ctv = abs_dvh_ctv.bincenters
                cum_ctv = abs_dvh_ctv.counts
                rx_dose_estimated = dose_at_fraction(bins_ctv, cum_ctv, 0.95)
                print(f"Estimated prescribed dose from {roi_name}: {rx_dose_estimated:.2f} Gy")
                break
        except Exception as ex:
            print(f"Error processing ROI {roi_name} for rx dose estimation: {ex}")
            continue

    if rx_dose_estimated is None:
        rx_dose_estimated = args.default_rx_dose
        print(f"No CTV1 found in manual segmentation; using default rx_dose: {rx_dose_estimated:.2f} Gy")

    print("\nProcessing manual RTSTRUCT...")
    results_manual = process_rtstruct(
        rtstruct_manual,
        args.manual_rtstruct,
        rtdose,
        "Manual",
        rx_dose_estimated,
    )
    print("\nProcessing TotalSegmentator RTSTRUCT...")
    results_total = process_rtstruct(
        rtstruct_total,
        rtstruct_total_path,
        rtdose,
        "TotalSegmentator",
        rx_dose_estimated,
    )
    all_results = results_manual + results_total

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_excel(args.output_excel, index=False)
        print(f"\nDVH metrics saved to {args.output_excel}")
        return 0

    print("\nNo DVH metrics calculated (all structures empty or errors encountered).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
