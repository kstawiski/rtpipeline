!pip install --upgrade numpy pydicom==1.4.2

import os
import pydicom
import numpy as np
import pandas as pd
from dicompylercore import dvhcalc

# -----------------------------------
# Example file paths for debugging:
rtplan_path           = "../Data_Organized/408175/RP.dcm"      # RT Plan (RP)
rtdose_path           = "../Data_Organized/408175/RD.dcm"       # RT Dose (RD)
rtstruct_manual_path  = "../Data_Organized/408175/RS.dcm"  # Original (manual) RTSTRUCT
ct_dir                = "../DICOM_OrganizedCT/408175/series_3"          # CT images directory
# TotalSegmentator segmentation file:
rtstruct_total_path   = f"{ct_dir}_total_dicom/segmentations.dcm"
# -----------------------------------

default_rx_dose = 50.0  # default prescribed dose (Gy) if estimation fails

def load_ct_images(ct_dir):
    """Load CT DICOM files from a directory and return a sorted list of datasets."""
    ct_files = []
    for f in os.listdir(ct_dir):
        if f.lower().endswith('.dcm'):
            ds = pydicom.dcmread(os.path.join(ct_dir, f))
            ct_files.append(ds)
    ct_files.sort(key=lambda ds: int(getattr(ds, 'InstanceNumber', 0)))
    return ct_files

def dose_at_fraction(bins, cumulative, fraction):
    """
    Interpolate the dose at which the cumulative volume equals 'fraction' * total volume.
    bins: dose levels.
    cumulative: cumulative volume (assumed monotonically decreasing).
    fraction: desired fraction (e.g. 0.95 for D95).
    """
    V_total = cumulative[0]
    target = fraction * V_total
    idx = np.where(cumulative <= target)[0]
    if idx.size == 0:
        return bins[-1]
    i = idx[0]
    if i == 0:
        return bins[0]
    x0, x1 = bins[i-1], bins[i]
    y0, y1 = cumulative[i-1], cumulative[i]
    if y1 == y0:
        return x0
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)

def get_volume_at_threshold(bins, cumulative, threshold):
    """
    Estimate the volume (cm³) receiving at least 'threshold' Gy from a cumulative DVH.
    Uses linear interpolation between bins.
    """
    V_total = cumulative[0]
    if threshold <= bins[0]:
        return V_total
    if threshold >= bins[-1]:
        return 0.0
    idx = np.where(bins >= threshold)[0]
    if idx.size == 0:
        return 0.0
    i = idx[0]
    if i == 0:
        return V_total
    x0, x1 = bins[i-1], bins[i]
    y0, y1 = cumulative[i-1], cumulative[i]
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (threshold - x0) / (x1 - x0)

def safe_attr(obj, attr):
    """Safely get an attribute from a DVH object; return np.nan on failure."""
    try:
        return float(getattr(obj, attr))
    except Exception:
        return np.nan

def compute_dvh_metrics(abs_dvh, rx_dose):
    """
    Compute DVH metrics from the absolute DVH and its corresponding relative DVH.
    
    Returns a dictionary with:
      -- Absolute metrics (in Gy): 
           DmeanGy, DmaxGy, DminGy, D95Gy, D98Gy, D2Gy, D50Gy,
           HI (absolute; = (D2Gy - D98Gy) / D50Gy), and SpreadGy.
      -- Relative metrics (expressed as % of rx_dose): 
           Dmean%, Dmax%, Dmin%, D95%, D98%, D2%, D50%,
           HI%, and Spread%.
      -- Volume (cm³) and IntegralDose (Gy*cm³).
      -- V metrics: for x = 1 to 60, the volume receiving at least x Gy,
           reported as "VxGy (cm³)" and "VxGy (%)".
    """
    # Absolute DVH data
    bins_abs = abs_dvh.bincenters
    cum_abs  = abs_dvh.counts
    if cum_abs.size == 0 or cum_abs[0] == 0:
        return None
    V_total = cum_abs[0]
    DmeanGy = abs_dvh.mean
    DmaxGy  = abs_dvh.max
    DminGy  = abs_dvh.min
    D95Gy   = dose_at_fraction(bins_abs, cum_abs, 0.95)
    D98Gy   = dose_at_fraction(bins_abs, cum_abs, 0.98)
    D2Gy    = dose_at_fraction(bins_abs, cum_abs, 0.02)
    D50Gy   = dose_at_fraction(bins_abs, cum_abs, 0.50)
    HI_abs  = (D2Gy - D98Gy) / D50Gy if D50Gy != 0 else np.nan
    SpreadGy = DmaxGy - DminGy

    # V metrics (absolute): volume (cm³) receiving at least x Gy for x=1..60
    V_abs = {}
    V_rel = {}
    for x in range(1, 61):
        vol = get_volume_at_threshold(bins_abs, cum_abs, x)
        V_abs[f"V{x}Gy (cm³)"] = vol
        V_rel[f"V{x}Gy (%)"] = (vol / V_total) * 100

    IntegralDose = DmeanGy * V_total  # Gy*cm³

    # Relative DVH processing using the rx_dose estimated from manual segmentation.
    rel_dvh = abs_dvh.relative_dose(rx_dose)
    bins_rel = rel_dvh.bincenters
    cum_rel  = rel_dvh.counts
    Dmean_pct = rel_dvh.mean
    Dmax_pct  = rel_dvh.max
    Dmin_pct  = rel_dvh.min
    D95_pct   = dose_at_fraction(bins_rel, cum_rel, 0.95)
    D98_pct   = dose_at_fraction(bins_rel, cum_rel, 0.98)
    D2_pct    = dose_at_fraction(bins_rel, cum_rel, 0.02)
    D50_pct   = dose_at_fraction(bins_rel, cum_rel, 0.50)
    HI_pct    = (D2_pct - D98_pct) / D50_pct if D50_pct != 0 else np.nan
    Spread_pct = Dmax_pct - Dmin_pct

    metrics = {
        "DmeanGy": DmeanGy,
        "DmaxGy": DmaxGy,
        "DminGy": DminGy,
        "D95Gy": D95Gy,
        "D98Gy": D98Gy,
        "D2Gy": D2Gy,
        "D50Gy": D50Gy,
        "HI": HI_abs,
        "SpreadGy": SpreadGy,
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
        "IntegralDose_Gycm3": IntegralDose,
    }
    metrics.update(V_abs)
    metrics.update(V_rel)
    return metrics

def process_rtstruct(rtstruct, segmentation_source, rx_dose_estimated):
    """
    Process an RT structure dataset and compute DVH metrics for each ROI.
    Uses the provided rx_dose_estimated (from the manual segmentation).
    Returns a list of dictionaries containing metrics and reference info.
    """
    results = []
    # Determine the actual file path based on the segmentation source
    rtstruct_file = rtstruct_manual_path if segmentation_source == "Manual" else rtstruct_total_path
    
    for roi in rtstruct.StructureSetROISequence:
        roi_number = roi.ROINumber
        roi_name   = roi.ROIName
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
            metrics.update({
                "RTPlan": rtplan_path,
                "RTDose": rtdose_path,
                "RTStruct": rtstruct_file,
                "CT_Dir": ct_dir,
                "ROI_Number": roi_number,
                "ROI_Name": roi_name,
                "PrescribedDose_Gy": rx_dose_estimated,
                "Segmentation_Source": segmentation_source
            })
            results.append(metrics)
        except Exception as ex:
            print(f"  Error processing ROI {roi_name} in {segmentation_source}: {ex}")
            continue
    return results

# ------------------------------
# Load common DICOM datasets (RT plan, RT dose, CT images)
# ------------------------------
print("Loading common DICOM files...")
try:
    rtplan = pydicom.dcmread(rtplan_path)
    rtdose = pydicom.dcmread(rtdose_path)
except Exception as e:
    print("Error loading RT plan or dose files:", e)
    raise

ct_images = load_ct_images(ct_dir)
print(f"Loaded {len(ct_images)} CT image(s).")

# ------------------------------
# Load both RT structure sets
# ------------------------------
print("\nLoading RT structure sets...")
try:
    rtstruct_manual = pydicom.dcmread(rtstruct_manual_path)
except Exception as e:
    print("Error loading manual RTSTRUCT file:", e)
    raise

try:
    rtstruct_total = pydicom.dcmread(rtstruct_total_path)
except Exception as e:
    print("Error loading TotalSegmentator RTSTRUCT file:", e)
    raise

# ------------------------------
# Estimate prescribed dose from original segmentation ("CTV1" or "CTV 1")
# ------------------------------
rx_dose_estimated = None
for roi in rtstruct_manual.StructureSetROISequence:
    roi_name = roi.ROIName
    if "ctv1" in roi_name.replace(" ", "").lower():
        try:
            abs_dvh_ctv = dvhcalc.get_dvh(rtstruct_manual, rtdose, roi.ROINumber)
            if abs_dvh_ctv.volume > 0:
                bins_ctv = abs_dvh_ctv.bincenters
                cum_ctv  = abs_dvh_ctv.counts
                rx_dose_estimated = dose_at_fraction(bins_ctv, cum_ctv, 0.95)
                print(f"Estimated prescribed dose from {roi_name}: {rx_dose_estimated:.2f} Gy")
                break
        except Exception as ex:
            print(f"Error processing ROI {roi_name} for rx dose estimation: {ex}")
            continue
if rx_dose_estimated is None:
    rx_dose_estimated = default_rx_dose
    print(f"No CTV1 found in manual segmentation; using default rx_dose: {rx_dose_estimated:.2f} Gy")

# ------------------------------
# Process both segmentation sets using the same rx_dose_estimated
# ------------------------------
print("\nProcessing manual RTSTRUCT...")
results_manual = process_rtstruct(rtstruct_manual, "Manual", rx_dose_estimated)
print("\nProcessing TotalSegmentator RTSTRUCT...")
results_total = process_rtstruct(rtstruct_total, "TotalSegmentator", rx_dose_estimated)
all_results = results_manual + results_total

# ------------------------------
# Save combined results to Excel
# ------------------------------
if all_results:
    df = pd.DataFrame(all_results)
    output_excel = "dvh_metrics.xlsx"
    df.to_excel(output_excel, index=False)
    print(f"\nDVH metrics saved to {output_excel}")
else:
    print("\nNo DVH metrics calculated (all structures empty or errors encountered).")


import os
import pydicom
import numpy as np
import pandas as pd
from dicompylercore import dvhcalc

# -----------------------------------
# Example file paths for debugging:
rtplan_path           = "../DICOM/172543/RP.172543.odbytnica.dcm"      # RT Plan (RP)
rtdose_path           = "../DICOM/172543/RD.172543.odbytnica.dcm"       # RT Dose (RD)
rtstruct_manual_path  = "../DICOM/172543/RS.172543.Auto__Miednica.0002.dcm"  # Original (manual) RTSTRUCT
ct_dir                = "../DICOM_OrganizedCT/172543/series_5"          # CT images directory
# TotalSegmentator segmentation file:
rtstruct_total_path   = f"{ct_dir}_total_dicom/segmentations.dcm"
# -----------------------------------

default_rx_dose = 50.0  # default prescribed dose (Gy) if estimation fails

def load_ct_images(ct_dir):
    """Load CT DICOM files from a directory and return a sorted list of datasets."""
    ct_files = []
    for f in os.listdir(ct_dir):
        if f.lower().endswith('.dcm'):
            ds = pydicom.dcmread(os.path.join(ct_dir, f))
            ct_files.append(ds)
    ct_files.sort(key=lambda ds: int(getattr(ds, 'InstanceNumber', 0)))
    return ct_files

def dose_at_fraction(bins, cumulative, fraction):
    """
    Interpolate the dose at which the cumulative volume equals 'fraction' * total volume.
    bins: dose levels.
    cumulative: cumulative volume (assumed monotonically decreasing).
    fraction: desired fraction (e.g. 0.95 for D95).
    """
    V_total = cumulative[0]
    target = fraction * V_total
    idx = np.where(cumulative <= target)[0]
    if idx.size == 0:
        return bins[-1]
    i = idx[0]
    if i == 0:
        return bins[0]
    x0, x1 = bins[i-1], bins[i]
    y0, y1 = cumulative[i-1], cumulative[i]
    if y1 == y0:
        return x0
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)

def get_volume_at_threshold(bins, cumulative, threshold):
    """
    Estimate the volume (cm³) receiving at least 'threshold' Gy from a cumulative DVH.
    Uses linear interpolation between bins.
    """
    V_total = cumulative[0]
    if threshold <= bins[0]:
        return V_total
    if threshold >= bins[-1]:
        return 0.0
    idx = np.where(bins >= threshold)[0]
    if idx.size == 0:
        return 0.0
    i = idx[0]
    if i == 0:
        return V_total
    x0, x1 = bins[i-1], bins[i]
    y0, y1 = cumulative[i-1], cumulative[i]
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (threshold - x0) / (x1 - x0)

def safe_attr(obj, attr):
    """Safely get an attribute from a DVH object; return np.nan on failure."""
    try:
        return float(getattr(obj, attr))
    except Exception:
        return np.nan

def compute_dvh_metrics(abs_dvh, rx_dose):
    """
    Compute DVH metrics from the absolute DVH and its corresponding relative DVH.
    
    Returns a dictionary with:
      -- Absolute metrics (in Gy): 
           DmeanGy, DmaxGy, DminGy, D95Gy, D98Gy, D2Gy, D50Gy,
           HI (absolute; = (D2Gy - D98Gy) / D50Gy), and SpreadGy.
      -- Relative metrics (expressed as % of rx_dose): 
           Dmean%, Dmax%, Dmin%, D95%, D98%, D2%, D50%,
           HI%, and Spread%.
      -- Volume (cm³) and IntegralDose (Gy*cm³).
      -- V metrics: for x = 1 to 60, the volume receiving at least x Gy,
           reported as "VxGy (cm³)" and "VxGy (%)".
    """
    # Absolute DVH data
    bins_abs = abs_dvh.bincenters
    cum_abs  = abs_dvh.counts
    if cum_abs.size == 0 or cum_abs[0] == 0:
        return None
    V_total = cum_abs[0]
    DmeanGy = abs_dvh.mean
    DmaxGy  = abs_dvh.max
    DminGy  = abs_dvh.min
    D95Gy   = dose_at_fraction(bins_abs, cum_abs, 0.95)
    D98Gy   = dose_at_fraction(bins_abs, cum_abs, 0.98)
    D2Gy    = dose_at_fraction(bins_abs, cum_abs, 0.02)
    D50Gy   = dose_at_fraction(bins_abs, cum_abs, 0.50)
    HI_abs  = (D2Gy - D98Gy) / D50Gy if D50Gy != 0 else np.nan
    SpreadGy = DmaxGy - DminGy

    # V metrics (absolute): volume (cm³) receiving at least x Gy for x=1..60
    V_abs = {}
    V_rel = {}
    for x in range(1, 61):
        vol = get_volume_at_threshold(bins_abs, cum_abs, x)
        V_abs[f"V{x}Gy (cm³)"] = vol
        V_rel[f"V{x}Gy (%)"] = (vol / V_total) * 100

    IntegralDose = DmeanGy * V_total  # Gy*cm³

    # Relative DVH processing using the rx_dose estimated from manual segmentation.
    rel_dvh = abs_dvh.relative_dose(rx_dose)
    bins_rel = rel_dvh.bincenters
    cum_rel  = rel_dvh.counts
    Dmean_pct = rel_dvh.mean
    Dmax_pct  = rel_dvh.max
    Dmin_pct  = rel_dvh.min
    D95_pct   = dose_at_fraction(bins_rel, cum_rel, 0.95)
    D98_pct   = dose_at_fraction(bins_rel, cum_rel, 0.98)
    D2_pct    = dose_at_fraction(bins_rel, cum_rel, 0.02)
    D50_pct   = dose_at_fraction(bins_rel, cum_rel, 0.50)
    HI_pct    = (D2_pct - D98_pct) / D50_pct if D50_pct != 0 else np.nan
    Spread_pct = Dmax_pct - Dmin_pct

    metrics = {
        "DmeanGy": DmeanGy,
        "DmaxGy": DmaxGy,
        "DminGy": DminGy,
        "D95Gy": D95Gy,
        "D98Gy": D98Gy,
        "D2Gy": D2Gy,
        "D50Gy": D50Gy,
        "HI": HI_abs,
        "SpreadGy": SpreadGy,
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
        "IntegralDose_Gycm3": IntegralDose,
    }
    metrics.update(V_abs)
    metrics.update(V_rel)
    return metrics

def process_rtstruct(rtstruct, segmentation_source, rx_dose_estimated):
    """
    Process an RT structure dataset and compute DVH metrics for each ROI.
    Uses the provided rx_dose_estimated (from the manual segmentation).
    Returns a list of dictionaries containing metrics and reference info.
    """
    results = []
    # Determine the actual file path based on the segmentation source
    rtstruct_file = rtstruct_manual_path if segmentation_source == "Manual" else rtstruct_total_path
    
    for roi in rtstruct.StructureSetROISequence:
        roi_number = roi.ROINumber
        roi_name   = roi.ROIName
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
            metrics.update({
                "RTPlan": rtplan_path,
                "RTDose": rtdose_path,
                "RTStruct": rtstruct_file,
                "CT_Dir": ct_dir,
                "ROI_Number": roi_number,
                "ROI_Name": roi_name,
                "PrescribedDose_Gy": rx_dose_estimated,
                "Segmentation_Source": segmentation_source
            })
            results.append(metrics)
        except Exception as ex:
            print(f"  Error processing ROI {roi_name} in {segmentation_source}: {ex}")
            continue
    return results

# ------------------------------
# Load common DICOM datasets (RT plan, RT dose, CT images)
# ------------------------------
print("Loading common DICOM files...")
try:
    rtplan = pydicom.dcmread(rtplan_path)
    rtdose = pydicom.dcmread(rtdose_path)
except Exception as e:
    print("Error loading RT plan or dose files:", e)
    raise

ct_images = load_ct_images(ct_dir)
print(f"Loaded {len(ct_images)} CT image(s).")

# ------------------------------
# Load both RT structure sets
# ------------------------------
print("\nLoading RT structure sets...")
try:
    rtstruct_manual = pydicom.dcmread(rtstruct_manual_path)
except Exception as e:
    print("Error loading manual RTSTRUCT file:", e)
    raise

try:
    rtstruct_total = pydicom.dcmread(rtstruct_total_path)
except Exception as e:
    print("Error loading TotalSegmentator RTSTRUCT file:", e)
    raise

# ------------------------------
# Estimate prescribed dose from original segmentation ("CTV1" or "CTV 1")
# ------------------------------
rx_dose_estimated = None
for roi in rtstruct_manual.StructureSetROISequence:
    roi_name = roi.ROIName
    if "ctv1" in roi_name.replace(" ", "").lower():
        try:
            abs_dvh_ctv = dvhcalc.get_dvh(rtstruct_manual, rtdose, roi.ROINumber)
            if abs_dvh_ctv.volume > 0:
                bins_ctv = abs_dvh_ctv.bincenters
                cum_ctv  = abs_dvh_ctv.counts
                rx_dose_estimated = dose_at_fraction(bins_ctv, cum_ctv, 0.95)
                print(f"Estimated prescribed dose from {roi_name}: {rx_dose_estimated:.2f} Gy")
                break
        except Exception as ex:
            print(f"Error processing ROI {roi_name} for rx dose estimation: {ex}")
            continue
if rx_dose_estimated is None:
    rx_dose_estimated = default_rx_dose
    print(f"No CTV1 found in manual segmentation; using default rx_dose: {rx_dose_estimated:.2f} Gy")

# ------------------------------
# Process both segmentation sets using the same rx_dose_estimated
# ------------------------------
print("\nProcessing manual RTSTRUCT...")
results_manual = process_rtstruct(rtstruct_manual, "Manual", rx_dose_estimated)
print("\nProcessing TotalSegmentator RTSTRUCT...")
results_total = process_rtstruct(rtstruct_total, "TotalSegmentator", rx_dose_estimated)
all_results = results_manual + results_total

# ------------------------------
# Save combined results to Excel
# ------------------------------
if all_results:
    df = pd.DataFrame(all_results)
    output_excel = "dvh_metrics.xlsx"
    df.to_excel(output_excel, index=False)
    print(f"\nDVH metrics saved to {output_excel}")
else:
    print("\nNo DVH metrics calculated (all structures empty or errors encountered).")




import os
import pydicom
import numpy as np
import pandas as pd
from dicompylercore import dvhcalc

# -----------------------------
# Helper functions
# -----------------------------
def load_ct_images(ct_dir):
    """Load CT DICOM files from a directory and return a sorted list of datasets."""
    ct_files = []
    for f in os.listdir(ct_dir):
        if f.lower().endswith('.dcm'):
            ds = pydicom.dcmread(os.path.join(ct_dir, f))
            ct_files.append(ds)
    ct_files.sort(key=lambda ds: int(getattr(ds, 'InstanceNumber', 0)))
    return ct_files

def dose_at_fraction(bins, cumulative, fraction):
    """
    Interpolate the dose at which the cumulative volume equals 'fraction' * total volume.
    bins: dose levels.
    cumulative: cumulative volume (assumed monotonically decreasing).
    fraction: desired fraction (e.g., 0.95 for D95).
    """
    V_total = cumulative[0]
    target = fraction * V_total
    idx = np.where(cumulative <= target)[0]
    if idx.size == 0:
        return bins[-1]
    i = idx[0]
    if i == 0:
        return bins[0]
    x0, x1 = bins[i-1], bins[i]
    y0, y1 = cumulative[i-1], cumulative[i]
    if y1 == y0:
        return x0
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)

def get_volume_at_threshold(bins, cumulative, threshold):
    """
    Estimate the volume (cm³) receiving at least 'threshold' Gy from a cumulative DVH.
    Uses linear interpolation between bins.
    """
    V_total = cumulative[0]
    if threshold <= bins[0]:
        return V_total
    if threshold >= bins[-1]:
        return 0.0
    idx = np.where(bins >= threshold)[0]
    if idx.size == 0:
        return 0.0
    i = idx[0]
    if i == 0:
        return V_total
    x0, x1 = bins[i-1], bins[i]
    y0, y1 = cumulative[i-1], cumulative[i]
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (threshold - x0) / (x1 - x0)

def safe_attr(obj, attr):
    """Safely get an attribute from a DVH object; return np.nan on failure."""
    try:
        return float(getattr(obj, attr))
    except Exception:
        return np.nan

def compute_dvh_metrics(abs_dvh, rx_dose):
    """
    Compute DVH metrics from the absolute DVH and its corresponding relative DVH.
    
    Returns a dictionary with the following labels:
      -- Absolute metrics (in Gy): 
           DmeanGy, DmaxGy, DminGy, D95Gy, D98Gy, D2Gy, D50Gy,
           HI (absolute; = (D2Gy - D98Gy) / D50Gy), SpreadGy.
      -- Relative metrics (expressed as % of rx_dose): 
           Dmean%, Dmax%, Dmin%, D95%, D98%, D2%, D50%,
           HI%, Spread%.
      -- Volume (cm³) and IntegralDose_Gycm3.
      -- V metrics for thresholds 1 to 60 Gy:
           For each x, "VxGy (cm³)" and "VxGy (%)".
    """
    # Absolute DVH
    bins_abs = abs_dvh.bincenters
    cum_abs  = abs_dvh.counts
    if cum_abs.size == 0 or cum_abs[0] == 0:
        return None
    V_total = cum_abs[0]
    DmeanGy = abs_dvh.mean
    DmaxGy  = abs_dvh.max
    DminGy  = abs_dvh.min
    D95Gy   = dose_at_fraction(bins_abs, cum_abs, 0.95)
    D98Gy   = dose_at_fraction(bins_abs, cum_abs, 0.98)
    D2Gy    = dose_at_fraction(bins_abs, cum_abs, 0.02)
    D50Gy   = dose_at_fraction(bins_abs, cum_abs, 0.50)
    HI_abs  = (D2Gy - D98Gy) / D50Gy if D50Gy != 0 else np.nan
    SpreadGy = DmaxGy - DminGy

    # V metrics for absolute DVH
    V_abs = {}
    V_rel = {}
    for x in range(1, 61):
        vol = get_volume_at_threshold(bins_abs, cum_abs, x)
        V_abs[f"V{x}Gy (cm³)"] = vol
        V_rel[f"V{x}Gy (%)"] = (vol / V_total) * 100

    IntegralDose = DmeanGy * V_total  # Gy*cm³

    # Relative DVH (using rx_dose from manual segmentation)
    rel_dvh = abs_dvh.relative_dose(rx_dose)
    bins_rel = rel_dvh.bincenters
    cum_rel  = rel_dvh.counts
    Dmean_pct = rel_dvh.mean
    Dmax_pct  = rel_dvh.max
    Dmin_pct  = rel_dvh.min
    D95_pct   = dose_at_fraction(bins_rel, cum_rel, 0.95)
    D98_pct   = dose_at_fraction(bins_rel, cum_rel, 0.98)
    D2_pct    = dose_at_fraction(bins_rel, cum_rel, 0.02)
    D50_pct   = dose_at_fraction(bins_rel, cum_rel, 0.50)
    HI_pct    = (D2_pct - D98_pct) / D50_pct if D50_pct != 0 else np.nan
    Spread_pct = Dmax_pct - Dmin_pct

    metrics = {
        "DmeanGy": DmeanGy,
        "DmaxGy": DmaxGy,
        "DminGy": DminGy,
        "D95Gy": D95Gy,
        "D98Gy": D98Gy,
        "D2Gy": D2Gy,
        "D50Gy": D50Gy,
        "HI": HI_abs,
        "SpreadGy": SpreadGy,
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
        "IntegralDose_Gycm3": IntegralDose,
    }
    metrics.update(V_abs)
    metrics.update(V_rel)
    return metrics

def process_rtstruct(rtstruct, segmentation_source, rx_dose_estimated, rtstruct_file):
    """
    Process an RT structure dataset and compute DVH metrics for each ROI.
    Uses the provided rx_dose_estimated (from the manual segmentation).
    The rtstruct_file parameter holds the actual file path for the RTSTRUCT.
    Returns a list of dictionaries containing metrics and reference info.
    """
    results = []
    for roi in rtstruct.StructureSetROISequence:
        roi_number = roi.ROINumber
        roi_name   = roi.ROIName
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
            metrics.update({
                "RTPlan": rtplan_path,
                "RTDose": rtdose_path,
                "RTStruct": rtstruct_file,
                "CT_Dir": ct_dir,
                "ROI_Number": roi_number,
                "ROI_Name": roi_name,
                "PrescribedDose_Gy": rx_dose_estimated,
                "Segmentation_Source": segmentation_source
            })
            results.append(metrics)
        except Exception as ex:
            print(f"  Error processing ROI {roi_name} in {segmentation_source}: {ex}")
            continue
    return results

# -----------------------------
# Main processing loop for all samples
# -----------------------------
# Read the metadata Excel file (update the path if needed)
metadata_df = pd.read_excel("../Data/metadata_checked20250302_withImages.xlsx")

all_results = []
# Iterate over each sample (row)
for idx, row in metadata_df.iterrows():
    print(f"\n--- Processing Sample {idx+1} ---")
    # Map columns to our variables
    rtplan_path = row['plans_file_path']
    rtdose_path = row['dosimetrics_file_path']
    rtstruct_manual_path = row['structures_file_path']
    ct_dir = row['organized_path']
    # Construct the TotalSegmentator RTSTRUCT file path from ct_dir
    rtstruct_total_path = f"{ct_dir}_total_dicom/segmentations.dcm"
    
    sample_id = row.get('SampleID', idx+1)  # if a SampleID column exists; otherwise use index
    
    # Load common DICOM files for this sample
    try:
        rtplan = pydicom.dcmread(rtplan_path)
        rtdose = pydicom.dcmread(rtdose_path)
    except Exception as e:
        print(f"Error loading RT plan or dose for sample {sample_id}: {e}")
        continue
    try:
        ct_images = load_ct_images(ct_dir)
        print(f"Loaded {len(ct_images)} CT image(s) for sample {sample_id}.")
    except Exception as e:
        print(f"Error loading CT images for sample {sample_id}: {e}")
        continue
    
    # Load RTSTRUCTs
    try:
        rtstruct_manual = pydicom.dcmread(rtstruct_manual_path)
    except Exception as e:
        print(f"Error loading manual RTSTRUCT for sample {sample_id}: {e}")
        continue
    try:
        rtstruct_total = pydicom.dcmread(rtstruct_total_path)
    except Exception as e:
        print(f"Error loading TotalSegmentator RTSTRUCT for sample {sample_id}: {e}")
        continue
    
    # Estimate prescribed dose from manual segmentation ("CTV1" or "CTV 1")
    rx_dose_estimated = None
    for roi in rtstruct_manual.StructureSetROISequence:
        roi_name = roi.ROIName
        if "ctv1" in roi_name.replace(" ", "").lower():
            try:
                abs_dvh_ctv = dvhcalc.get_dvh(rtstruct_manual, rtdose, roi.ROINumber)
                if abs_dvh_ctv.volume > 0:
                    bins_ctv = abs_dvh_ctv.bincenters
                    cum_ctv  = abs_dvh_ctv.counts
                    rx_dose_estimated = dose_at_fraction(bins_ctv, cum_ctv, 0.95)
                    print(f"Sample {sample_id}: Estimated prescribed dose from {roi_name}: {rx_dose_estimated:.2f} Gy")
                    break
            except Exception as ex:
                print(f"Error processing ROI {roi_name} for rx dose estimation in sample {sample_id}: {ex}")
                continue
    if rx_dose_estimated is None:
        rx_dose_estimated = default_rx_dose
        print(f"Sample {sample_id}: No CTV1 found in manual segmentation; using default rx_dose: {rx_dose_estimated:.2f} Gy")
    
    # Process both segmentation sets using the same rx_dose_estimated
    results_manual = process_rtstruct(rtstruct_manual, "Manual", rx_dose_estimated, rtstruct_manual_path)
    results_total = process_rtstruct(rtstruct_total, "TotalSegmentator", rx_dose_estimated, rtstruct_total_path)
    sample_results = results_manual + results_total
    # Add a SampleID column to each result
    for res in sample_results:
        res["SampleID"] = sample_id
    all_results.extend(sample_results)

# -----------------------------
# Save all results to a single Excel file
# -----------------------------
if all_results:
    df_all = pd.DataFrame(all_results)
    output_excel = "../Data/DVH_metrics.xlsx"
    df_all.to_excel(output_excel, index=False)
    print(f"\nAll DVH metrics saved to {output_excel}")
else:
    print("\nNo DVH metrics calculated for any samples.")


