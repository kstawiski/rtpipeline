from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pydicom
from dicompylercore import dvhcalc

logger = logging.getLogger(__name__)


def _dose_at_fraction(bins, cumulative, fraction: float) -> float:
    V_total = cumulative[0]
    target = fraction * V_total
    idx = np.where(cumulative <= target)[0]
    if idx.size == 0:
        return float(bins[-1])
    i = int(idx[0])
    if i == 0:
        return float(bins[0])
    x0, x1 = float(bins[i - 1]), float(bins[i])
    y0, y1 = float(cumulative[i - 1]), float(cumulative[i])
    if y1 == y0:
        return x0
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)


def _get_volume_at_threshold(bins, cumulative, threshold: float) -> float:
    V_total = cumulative[0]
    if threshold <= bins[0]:
        return float(V_total)
    if threshold >= bins[-1]:
        return 0.0
    idx = np.where(bins >= threshold)[0]
    if idx.size == 0:
        return 0.0
    i = int(idx[0])
    if i == 0:
        return float(V_total)
    x0, x1 = float(bins[i - 1]), float(bins[i])
    y0, y1 = float(cumulative[i - 1]), float(cumulative[i])
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (threshold - x0) / (x1 - x0)


def _compute_metrics(abs_dvh, rx_dose: float) -> Optional[Dict[str, float]]:
    bins_abs = abs_dvh.bincenters
    cum_abs = abs_dvh.counts
    if cum_abs.size == 0 or cum_abs[0] == 0:
        return None
    V_total = float(cum_abs[0])
    DmeanGy = float(abs_dvh.mean)
    DmaxGy = float(abs_dvh.max)
    DminGy = float(abs_dvh.min)
    D95Gy = _dose_at_fraction(bins_abs, cum_abs, 0.95)
    D98Gy = _dose_at_fraction(bins_abs, cum_abs, 0.98)
    D2Gy = _dose_at_fraction(bins_abs, cum_abs, 0.02)
    D50Gy = _dose_at_fraction(bins_abs, cum_abs, 0.50)
    HI_abs = (D2Gy - D98Gy) / D50Gy if D50Gy != 0 else float("nan")
    SpreadGy = DmaxGy - DminGy

    V_abs: Dict[str, float] = {}
    V_rel: Dict[str, float] = {}
    for x in range(1, 61):
        vol = _get_volume_at_threshold(bins_abs, cum_abs, float(x))
        V_abs[f"V{x}Gy (cm³)"] = float(vol)
        V_rel[f"V{x}Gy (%)"] = float((vol / V_total) * 100.0)

    IntegralDose = DmeanGy * V_total

    # Additional clinically useful metrics
    # Hottest small volumes (absolute), commonly used for OARs
    D1ccGy = None
    D0_1ccGy = None
    try:
        if V_total > 0:
            if V_total >= 1.0:
                D1ccGy = _dose_at_fraction(bins_abs, cum_abs, 1.0 / V_total)
            if V_total >= 0.1:
                D0_1ccGy = _dose_at_fraction(bins_abs, cum_abs, 0.1 / V_total)
    except Exception:
        pass

    # Coverage at 95% and 100% of Rx dose
    V95Rx_cc = None
    V95Rx_pct = None
    V100Rx_cc = None
    V100Rx_pct = None
    try:
        if rx_dose and rx_dose > 0:
            V95Rx_cc = _get_volume_at_threshold(bins_abs, cum_abs, 0.95 * rx_dose)
            V100Rx_cc = _get_volume_at_threshold(bins_abs, cum_abs, 1.00 * rx_dose)
            V95Rx_pct = (V95Rx_cc / V_total) * 100.0 if V_total > 0 else None
            V100Rx_pct = (V100Rx_cc / V_total) * 100.0 if V_total > 0 else None
    except Exception:
        pass

    rel_dvh = abs_dvh.relative_dose(float(rx_dose))
    bins_rel = rel_dvh.bincenters
    cum_rel = rel_dvh.counts
    Dmean_pct = float(rel_dvh.mean)
    Dmax_pct = float(rel_dvh.max)
    Dmin_pct = float(rel_dvh.min)
    D95_pct = _dose_at_fraction(bins_rel, cum_rel, 0.95)
    D98_pct = _dose_at_fraction(bins_rel, cum_rel, 0.98)
    D2_pct = _dose_at_fraction(bins_rel, cum_rel, 0.02)
    D50_pct = _dose_at_fraction(bins_rel, cum_rel, 0.50)
    HI_pct = (D2_pct - D98_pct) / D50_pct if D50_pct != 0 else float("nan")
    Spread_pct = Dmax_pct - Dmin_pct

    metrics: Dict[str, float] = {
        "DmeanGy": DmeanGy,
        "DmaxGy": DmaxGy,
        "DminGy": DminGy,
        "D95Gy": D95Gy,
        "D98Gy": D98Gy,
        "D2Gy": D2Gy,
        "D50Gy": D50Gy,
        "HI": HI_abs,
        "SpreadGy": SpreadGy,
        "D1ccGy": D1ccGy,
        "D0.1ccGy": D0_1ccGy,
        "Dmean%": Dmean_pct,
        "Dmax%": Dmax_pct,
        "Dmin%": Dmin_pct,
        "D95%": D95_pct,
        "D98%": D98_pct,
        "D2%": D2_pct,
        "D50%": D50_pct,
        "HI%": HI_pct,
        "Spread%": Spread_pct,
        "Volume (cm³)": V_total,
        "IntegralDose_Gycm3": float(IntegralDose),
        "V95%Rx (cm³)": V95Rx_cc,
        "V95%Rx (%)": V95Rx_pct,
        "V100%Rx (cm³)": V100Rx_cc,
        "V100%Rx (%)": V100Rx_pct,
    }
    metrics.update(V_abs)
    metrics.update(V_rel)
    return metrics


def _estimate_rx_from_ctv1(rtstruct: pydicom.dataset.FileDataset, rtdose: pydicom.dataset.FileDataset) -> float:
    default_rx = 50.0
    try:
        for roi in rtstruct.StructureSetROISequence:
            name = str(getattr(roi, "ROIName", "") or "")
            if "ctv1" in name.replace(" ", "").lower():
                abs_dvh_ctv = dvhcalc.get_dvh(rtstruct, rtdose, roi.ROINumber)
                if abs_dvh_ctv.volume > 0:
                    bins_ctv = abs_dvh_ctv.bincenters
                    cum_ctv = abs_dvh_ctv.counts
                    return float(_dose_at_fraction(bins_ctv, cum_ctv, 0.95))
    except Exception as e:
        logger.debug("RX estimate failed: %s", e)
    return default_rx


def dvh_for_course(course_dir: Path) -> Optional[Path]:
    rp = course_dir / "RP.dcm"
    rd = course_dir / "RD.dcm"
    rs_manual = course_dir / "RS.dcm"
    rs_auto = course_dir / "RS_auto.dcm"
    if not rd.exists() or not rp.exists():
        logger.warning("Missing RP/RD in %s; skipping DVH", course_dir)
        return None
    try:
        rtdose = pydicom.dcmread(str(rd))
        rtplan = pydicom.dcmread(str(rp))
    except Exception as e:
        logger.error("Failed to read RP/RD: %s", e)
        return None

    results: List[Dict] = []

    def process_struct(rs_path: Path, source: str, rx_dose: float) -> None:
        try:
            rtstruct = pydicom.dcmread(str(rs_path))
        except Exception as e:
            logger.warning("Cannot read RTSTRUCT %s: %s", rs_path, e)
            return
        for roi in getattr(rtstruct, "StructureSetROISequence", []):
            try:
                abs_dvh = dvhcalc.get_dvh(rtstruct, rtdose, roi.ROINumber)
                if abs_dvh.volume == 0:
                    continue
                m = _compute_metrics(abs_dvh, rx_dose)
                if m is None:
                    continue
                m.update({
                    "ROI_Number": int(roi.ROINumber),
                    "ROI_Name": str(getattr(roi, "ROIName", "")),
                    "Segmentation_Source": source,
                })
                results.append(m)
            except Exception as e:
                logger.debug("DVH failed for ROI %s: %s", getattr(roi, "ROIName", ""), e)

    # Estimate rx from manual if present; else default
    rx_est = 50.0
    if rs_manual.exists():
        try:
            rx_est = _estimate_rx_from_ctv1(pydicom.dcmread(str(rs_manual)), rtdose)
        except Exception:
            pass

    if rs_manual.exists():
        process_struct(rs_manual, "Manual", rx_est)
    if rs_auto.exists():
        process_struct(rs_auto, "AutoRTS", rx_est)

    if not results:
        logger.info("No DVH results for %s", course_dir)
        return None

    df = pd.DataFrame(results)
    out_xlsx = course_dir / "dvh_metrics.xlsx"
    df.to_excel(out_xlsx, index=False)
    return out_xlsx
