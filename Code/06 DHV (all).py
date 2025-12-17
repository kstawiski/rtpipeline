import os
import shutil
import gzip
import pandas as pd
from tqdm import tqdm  # For the progress bar

# Set the base output directory (Modify this as needed)
BASE_OUTPUT_DIR = "/storage-lodz/Odbytnice/Data_Organized/"

# Load CSV files
rt_merge_path = "../Logs/rt_merge_results_20250304_171051.csv"
total_segmentator_path = "../Data/TotalSegmentator_processing_summary.csv"

rt_merge_df = pd.read_csv(rt_merge_path)
total_segmentator_df = pd.read_csv(total_segmentator_path)

# Standardize column names for joining
total_segmentator_df.rename(columns={"Patient_ID": "patient_id"}, inplace=True)

# Deduplicate total_segmentator_df by keeping the first occurrence per patient_id
total_segmentator_deduplicated = (
    total_segmentator_df.groupby("patient_id").first().reset_index()
)

# Perform a left join to ensure we retain all 205 patients
merged_df = rt_merge_df.merge(total_segmentator_deduplicated, on="patient_id", how="left")

def file_size_matches(src, dst):
    """Check if the destination file exists and has the same size as the source."""
    return os.path.exists(dst) and os.path.getsize(src) == os.path.getsize(dst)

# Iterate through each patient and copy files using a progress bar.
# Debugging prints have been removed to keep the output clean.
for _, row in tqdm(merged_df.iterrows(), total=merged_df.shape[0], desc="Processing patients", unit="patient"):
    patient_id = str(row["patient_id"])
    patient_dir = os.path.join(BASE_OUTPUT_DIR, patient_id)
    
    # Ensure patient directory exists
    os.makedirs(patient_dir, exist_ok=True)

    # Convert potential NaN values to empty strings
    dicom_input_dir = str(row["DICOM_Input"]) if isinstance(row["DICOM_Input"], str) else ""
    nifti_input_dir = str(row["NIfTI_Output"]) if isinstance(row["NIfTI_Output"], str) else ""
    dicom_segmentation_dir = str(row["DICOM_Segmentation_Output"]) if isinstance(row["DICOM_Segmentation_Output"], str) else ""
    nifti_segmentation_dir = str(row["NIfTI_Segmentation_Output"]) if isinstance(row["NIfTI_Segmentation_Output"], str) else ""

    # Copy DICOM files to CT_DICOM subdirectory
    dicom_output_dir = os.path.join(patient_dir, "CT_DICOM")
    if dicom_input_dir and os.path.exists(dicom_input_dir) and os.path.isdir(dicom_input_dir):
        if not os.path.exists(dicom_output_dir):
            shutil.copytree(dicom_input_dir, dicom_output_dir, dirs_exist_ok=True)

    # Copy and compress NIfTI output to ct.nii.gz
    nifti_output_file = os.path.join(patient_dir, "ct.nii.gz")
    if nifti_input_dir and os.path.exists(nifti_input_dir) and os.path.isdir(nifti_input_dir):
        nifti_files = [f for f in os.listdir(nifti_input_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]
        if nifti_files:
            nifti_file = os.path.join(nifti_input_dir, nifti_files[0])
            if not os.path.exists(nifti_output_file):
                with open(nifti_file, "rb") as f_in:
                    with gzip.open(nifti_output_file, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

    # Copy "segmentations.dcm" to "TotalSegmentator.dcm"
    segmentation_input_file = os.path.join(dicom_segmentation_dir, "segmentations.dcm")
    segmentation_output_file = os.path.join(patient_dir, "TotalSegmentator.dcm")
    if dicom_segmentation_dir and os.path.exists(segmentation_input_file) and os.path.isfile(segmentation_input_file):
        if not file_size_matches(segmentation_input_file, segmentation_output_file):
            shutil.copy(segmentation_input_file, segmentation_output_file)

    # Copy all files from NIfTI_Segmentation_Output to TotalSegmentator subdirectory
    total_segmentator_dir = os.path.join(patient_dir, "TotalSegmentator")
    if nifti_segmentation_dir and os.path.exists(nifti_segmentation_dir) and os.path.isdir(nifti_segmentation_dir):
        os.makedirs(total_segmentator_dir, exist_ok=True)
        for file in os.listdir(nifti_segmentation_dir):
            file_path = os.path.join(nifti_segmentation_dir, file)
            dest_file_path = os.path.join(total_segmentator_dir, file)
            if os.path.isfile(file_path) and not file_size_matches(file_path, dest_file_path):
                shutil.copy(file_path, dest_file_path)

# Optionally, print a final message when done.
print("File transfer process completed.")

import os
import shutil
import gzip
import pandas as pd
from tqdm import tqdm  # For the progress bar

# Set the base output directory (Modify this as needed)
BASE_OUTPUT_DIR = "/storage-lodz/Odbytnice/Data_Organized/"

# Load CSV files
rt_merge_path = "../Logs/rt_merge_results_20250304_171051.csv"
total_segmentator_path = "../Data/TotalSegmentator_processing_summary.csv"

rt_merge_df = pd.read_csv(rt_merge_path)
total_segmentator_df = pd.read_csv(total_segmentator_path)

# Standardize column names for joining
total_segmentator_df.rename(columns={"Patient_ID": "patient_id"}, inplace=True)

# Deduplicate total_segmentator_df by keeping the first occurrence per patient_id
total_segmentator_deduplicated = (
    total_segmentator_df.groupby("patient_id").first().reset_index()
)

# Perform a left join to ensure we retain all 205 patients
merged_df = rt_merge_df.merge(total_segmentator_deduplicated, on="patient_id", how="left")

merged_df.to_csv("merged_df.csv")
merged_df

import os
import pydicom
import numpy as np
import pandas as pd
from dicompylercore import dvhcalc
import traceback

# Load the patient list from CSV
#merged_df = pd.read_csv('merged_df.csv')
print(f"Loaded {len(merged_df)} patients from merged_df.csv")

# Base directories
data_org_base = "../Data_Organized/"
ct_base = "../DICOM_OrganizedCT/"
output_dir = "../Data/"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Default prescribed dose (Gy) if estimation fails
default_rx_dose = 50.0

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

def process_rtstruct(rtstruct, rtdose, segmentation_source, rx_dose_estimated, patient_id, rtplan_path, rtdose_path, rtstruct_path, ct_dir):
    """
    Process an RT structure dataset and compute DVH metrics for each ROI.
    Uses the provided rx_dose_estimated (from the manual segmentation).
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
                "patient_id": patient_id,
                "RTPlan": rtplan_path,
                "RTDose": rtdose_path,
                "RTStruct": rtstruct_path,
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

def process_patient(patient_id):
    """Process a single patient's DVH metrics and return results"""
    
    patient_id_str = str(patient_id)
    print(f"\n{'='*80}\nProcessing patient: {patient_id_str}\n{'='*80}")
    
    # Define paths for this patient
    rtplan_path = os.path.join(data_org_base, patient_id_str, "RP.dcm")
    rtdose_path = os.path.join(data_org_base, patient_id_str, "RD.dcm")
    rtstruct_manual_path = os.path.join(data_org_base, patient_id_str, "RS.dcm")
    
    # Find the CT directory series for this patient
    ct_base_patient = os.path.join(ct_base, patient_id_str)
    series_dirs = [d for d in os.listdir(ct_base_patient) if d.startswith('series_') and os.path.isdir(os.path.join(ct_base_patient, d))]
    
    if not series_dirs:
        print(f"No series directories found for patient {patient_id_str}")
        return []
    
    # Use the first series for simplicity (could be modified to select a specific series)
    ct_dir = os.path.join(ct_base_patient, series_dirs[0])
    rtstruct_total_path = os.path.join(data_org_base, patient_id_str, "TotalSegmentator.dcm")
    
    print(f"Using paths:\nRTPlan: {rtplan_path}\nRTDose: {rtdose_path}\nRT Manual Structure: {rtstruct_manual_path}\nCT Dir: {ct_dir}\nRT Total Structure: {rtstruct_total_path}")
    
    # Check if all required files exist
    for path, label in [(rtplan_path, "RT Plan"), (rtdose_path, "RT Dose"), 
                       (rtstruct_manual_path, "Manual RT Structure"),
                       (ct_dir, "CT Directory")]:
        if not os.path.exists(path):
            print(f"Missing {label} for patient {patient_id_str}: {path}")
            return []
    
    # Special check for the TotalSegmentator structure
    totalseg_exists = os.path.exists(rtstruct_total_path)
    if not totalseg_exists:
        print(f"Warning: TotalSegmentator structure not found: {rtstruct_total_path}")
    
    # ------------------------------
    # Load common DICOM datasets (RT plan, RT dose, CT images)
    # ------------------------------
    print("Loading common DICOM files...")
    try:
        rtplan = pydicom.dcmread(rtplan_path)
        rtdose = pydicom.dcmread(rtdose_path)
    except Exception as e:
        print(f"Error loading RT plan or dose files for patient {patient_id_str}:", e)
        return []

    try:
        ct_images = load_ct_images(ct_dir)
        print(f"Loaded {len(ct_images)} CT image(s) for patient {patient_id_str}.")
    except Exception as e:
        print(f"Error loading CT images for patient {patient_id_str}:", e)
        return []

    # ------------------------------
    # Load both RT structure sets
    # ------------------------------
    print("\nLoading RT structure sets...")
    try:
        rtstruct_manual = pydicom.dcmread(rtstruct_manual_path)
    except Exception as e:
        print(f"Error loading manual RTSTRUCT file for patient {patient_id_str}:", e)
        return []

    rtstruct_total = None
    if totalseg_exists:
        try:
            rtstruct_total = pydicom.dcmread(rtstruct_total_path)
        except Exception as e:
            print(f"Error loading TotalSegmentator RTSTRUCT file for patient {patient_id_str}:", e)
            # Continue with manual only if TotalSegmentator fails

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
    all_results = []
    
    print("\nProcessing manual RTSTRUCT...")
    results_manual = process_rtstruct(rtstruct_manual, rtdose, "Manual", rx_dose_estimated, 
                                     patient_id, rtplan_path, rtdose_path, rtstruct_manual_path, ct_dir)
    all_results.extend(results_manual)
    
    if rtstruct_total is not None:
        print("\nProcessing TotalSegmentator RTSTRUCT...")
        results_total = process_rtstruct(rtstruct_total, rtdose, "TotalSegmentator", rx_dose_estimated,
                                        patient_id, rtplan_path, rtdose_path, rtstruct_total_path, ct_dir)
        all_results.extend(results_total)
    
    # ------------------------------
    # Save individual patient results to Excel
    # ------------------------------
    if all_results:
        df = pd.DataFrame(all_results)
        output_excel = os.path.join(data_org_base, patient_id_str, "dvh_metrics.xlsx")
        df.to_excel(output_excel, index=False)
        print(f"\nDVH metrics for patient {patient_id_str} saved to {output_excel}")
    else:
        print(f"\nNo DVH metrics calculated for patient {patient_id_str} (all structures empty or errors encountered).")
    
    return all_results

# Main processing loop
all_patient_results = []
error_patients = []

for idx, row in merged_df.iterrows():
    patient_id = row['patient_id']
    try:
        patient_results = process_patient(patient_id)
        if patient_results:
            all_patient_results.extend(patient_results)
        else:
            error_patients.append(patient_id)
    except Exception as e:
        print(f"Error processing patient {patient_id}:")
        traceback.print_exc()
        error_patients.append(patient_id)

# Save combined results to a master Excel file
if all_patient_results:
    master_df = pd.DataFrame(all_patient_results)
    master_output = os.path.join(output_dir, "DVH_metrics.xlsx")
    master_df.to_excel(master_output, index=False)
    print(f"\nCombined DVH metrics for all patients saved to {master_output}")
    print(f"Successfully processed {len(set(master_df['patient_id']))} patients.")
else:
    print("\nNo DVH metrics calculated for any patients.")

# Report any errors
if error_patients:
    print(f"\nFailed to process {len(error_patients)} patients: {error_patients}")

