import pandas as pd

# Read the Excel metadata file containing DICOM-RT file paths
metadata_df = pd.read_excel("../Data/metadata.xlsx")

metadata_df

import os
import shutil
import copy
import logging

import pandas as pd
import pydicom
import numpy as np
from pydicom.uid import generate_uid
from scipy.ndimage import map_coordinates

# =============================================================================
# Configuration
# =============================================================================

METADATA_PATH = "../Data/metadata_checked20250302.xlsx"
OUTPUT_DIR = "/home/konrad/test2/"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# We'll accumulate final info here to produce a DataFrame
RESULTS = []

# =============================================================================
# Helper Functions
# =============================================================================

def safe_copy(src, dst):
    """
    Copy src to dst, overwriting if needed.
    """
    if not os.path.isfile(src):
        raise FileNotFoundError(f"File not found: {src}")
    if os.path.exists(dst) and not os.path.samefile(src, dst):
        os.remove(dst)
    shutil.copy2(src, dst)

def create_summed_plan(plan_files, total_dose_gy, out_plan_path):
    """
    Create a 'summed' RT Plan by taking the first plan as a template
    and updating its prescription dose. Real merges may need more logic.
    """
    if not plan_files:
        raise ValueError("No plan files to sum.")

    ds_plan = pydicom.dcmread(plan_files[0])
    if hasattr(ds_plan, "DoseReferenceSequence") and ds_plan.DoseReferenceSequence:
        ds_plan.DoseReferenceSequence[0].TargetPrescriptionDose = float(total_dose_gy)
        logger.info(f"Updated plan prescription dose to {total_dose_gy} Gy")
    else:
        logger.warning("No DoseReferenceSequence found in plan. Skipping prescription update.")

    ds_plan.SeriesDescription = "Summed Plan"
    ds_plan.SOPInstanceUID = generate_uid()
    ds_plan.SeriesInstanceUID = generate_uid()

    ds_plan.save_as(out_plan_path)
    logger.info(f"Wrote summed plan to {out_plan_path}")
    return ds_plan

def sum_doses_with_resample(dose_files, out_dose_path):
    """
    Summation of multiple RT Dose files that may differ in grid size.
    Resamples each dose onto a reference grid and accumulates them.
    The resulting dose is converted to an integer representation with a scaling factor.
    """
    if not dose_files:
        raise ValueError("No dose files to sum.")

    # 1) Use the first dose as the reference.
    ref_path = dose_files[0]
    ds_ref = pydicom.dcmread(ref_path)
    ref_array = ds_ref.pixel_array.astype(np.float32)
    rows_ref = ds_ref.Rows
    cols_ref = ds_ref.Columns
    frames_ref = ds_ref.NumberOfFrames if hasattr(ds_ref, "NumberOfFrames") else 1

    logger.info(f"Reference Dose: {ref_path}")
    logger.info(f" - shape: {ref_array.shape}, Rows={rows_ref}, Cols={cols_ref}, Frames={frames_ref}")

    origin_ref = getattr(ds_ref, "ImagePositionPatient", [0, 0, 0])
    pixel_spacing_ref = getattr(ds_ref, "PixelSpacing", [1.0, 1.0])
    offsets_ref = getattr(ds_ref, "GridFrameOffsetVector", None)
    if offsets_ref is not None and len(offsets_ref) == frames_ref:
        slice_thickness_ref = float(offsets_ref[1] - offsets_ref[0]) if frames_ref > 1 else 0.0
    else:
        slice_thickness_ref = 1.0

    # Build reference coordinate arrays (z,y,x)
    z_positions_ref = np.array([origin_ref[2] + i * slice_thickness_ref for i in range(frames_ref)])
    y_positions_ref = np.array([origin_ref[1] + r * pixel_spacing_ref[0] for r in range(rows_ref)])
    x_positions_ref = np.array([origin_ref[0] + c * pixel_spacing_ref[1] for c in range(cols_ref)])

    accumulated = ref_array.copy()

    # 2) Resample and accumulate subsequent doses.
    for dose_path in dose_files[1:]:
        ds_next = pydicom.dcmread(dose_path)
        arr_next = ds_next.pixel_array.astype(np.float32)
        rows_next = ds_next.Rows
        cols_next = ds_next.Columns
        frames_next = ds_next.NumberOfFrames if hasattr(ds_next, "NumberOfFrames") else 1

        logger.info(f"Resampling {dose_path}")
        logger.info(f" - shape: {arr_next.shape}, Rows={rows_next}, Cols={cols_next}, Frames={frames_next}")

        origin_next = getattr(ds_next, "ImagePositionPatient", [0, 0, 0])
        pixel_spacing_next = getattr(ds_next, "PixelSpacing", [1.0, 1.0])
        offsets_next = getattr(ds_next, "GridFrameOffsetVector", None)
        if offsets_next is not None and len(offsets_next) == frames_next:
            slice_thickness_next = float(offsets_next[1] - offsets_next[0]) if frames_next > 1 else 0.0
        else:
            slice_thickness_next = 1.0

        Zref, Yref, Xref = np.meshgrid(z_positions_ref, y_positions_ref, x_positions_ref, indexing='ij')
        indexZ = (Zref - origin_next[2]) / slice_thickness_next
        indexY = (Yref - origin_next[1]) / pixel_spacing_next[0]
        indexX = (Xref - origin_next[0]) / pixel_spacing_next[1]
        coords_3d = np.stack([indexZ, indexY, indexX], axis=0)

        dose_resampled_flat = map_coordinates(arr_next, coords_3d, order=1, mode='constant', cval=0.0)
        dose_resampled = dose_resampled_flat.reshape((frames_ref, rows_ref, cols_ref))
        accumulated += dose_resampled

    # Replace any NaN values with zero to avoid invalid casts.
    accumulated = np.nan_to_num(accumulated, nan=0.0)

    # 3) Convert to integer representation.
    scaling_factor = 1000.0  # Example: scaling to store dose as int (e.g., Gy*1000)
    accumulated_int = (accumulated * scaling_factor).astype(np.int32)

    # 4) Create new RT Dose dataset.
    new_ds = copy.deepcopy(ds_ref)
    new_ds.SOPInstanceUID = generate_uid()
    new_ds.SeriesInstanceUID = generate_uid()
    new_ds.SeriesDescription = "Summed Dose (Resampled)"
    new_ds.DoseSummationType = "PLAN"

    new_ds.BitsAllocated = 32
    new_ds.BitsStored = 32
    new_ds.HighBit = 31
    new_ds.PixelRepresentation = 1

    # Set DoseGridScaling to convert stored int values back to Gy.
    new_ds.DoseGridScaling = 1.0 / scaling_factor

    # Remove the following attributes if they exist; they can cause type issues on saving.
    if (0x0028, 0x0106) in new_ds:  # Smallest Image Pixel Value
        del new_ds[(0x0028, 0x0106)]
    if (0x0028, 0x0107) in new_ds:  # Largest Image Pixel Value
        del new_ds[(0x0028, 0x0107)]

    new_ds.PixelData = accumulated_int.tobytes()

    if hasattr(new_ds, "PerFrameFunctionalGroupsSequence"):
        del new_ds.PerFrameFunctionalGroupsSequence

    new_ds.save_as(out_dose_path)
    logger.info(f"Wrote summed/resampled dose to {out_dose_path}")
    return new_ds


# =============================================================================
# Main Script
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read metadata
    df = pd.read_excel(METADATA_PATH)
    logger.info(f"Loaded {len(df)} rows from {METADATA_PATH}")

    grouped = df.groupby("plans_patient_id")

    for patient_id, group in grouped:
        patient_outdir = os.path.join(OUTPUT_DIR, str(patient_id))
        os.makedirs(patient_outdir, exist_ok=True)
        logger.info(f"Processing patient {patient_id}")

        # We'll store final results about this patient in a dict
        result_entry = {
            "patient_id": patient_id,
            "final_RP": "",
            "final_RD": "",
            "final_RS": "",
            "summed_reference_dose": 0.0,
            "status": "",
        }

        # (A) Identify the first structure set in the group (lowest row).
        first_index = group.index.min()
        first_structure_path = group.loc[first_index, "structures_file_path"]
        logger.info(f"  The first structure set is: {first_structure_path}")

        # Filter the group to only keep that structure set
        group_filtered = group[group["structures_file_path"] == first_structure_path]
        if len(group_filtered) == 0:
            logger.warning("  No rows remain after filtering for first structure set. Skipping patient.")
            result_entry["status"] = "ignored_no_matching_structures"
            RESULTS.append(result_entry)
            continue

        # Distinct plan & dose files from that filtered subset
        plan_paths = group_filtered["plans_file_path"].unique().tolist()
        dose_paths = group_filtered["dosimetrics_file_path"].unique().tolist()

        # Summation of reference doses from the filtered subset
        total_ref_dose = group_filtered["plans_reference_dose"].sum()
        logger.info(f"  Found {len(plan_paths)} plan(s) and {len(dose_paths)} dose file(s) after ignoring re-plans.")
        logger.info(f"  Summed reference dose = {total_ref_dose} Gy")

        if len(plan_paths) == 1 and len(dose_paths) == 1:
            # Single-stage => just copy
            rp_dst = os.path.join(patient_outdir, "RP.dcm")
            rd_dst = os.path.join(patient_outdir, "RD.dcm")
            rs_dst = os.path.join(patient_outdir, "RS.dcm")

            safe_copy(plan_paths[0], rp_dst)
            safe_copy(dose_paths[0], rd_dst)
            safe_copy(first_structure_path, rs_dst)

            logger.info("  Single-stage: Copied RP, RD, RS")

            # Fill result entry
            result_entry["final_RP"] = rp_dst
            result_entry["final_RD"] = rd_dst
            result_entry["final_RS"] = rs_dst
            result_entry["summed_reference_dose"] = float(total_ref_dose)
            result_entry["status"] = "single-stage"

        else:
            # Multi-stage => sum
            rd_sum_path = os.path.join(patient_outdir, "RD.dcm")
            ds_rd_sum = sum_doses_with_resample(dose_paths, rd_sum_path)

            rp_sum_path = os.path.join(patient_outdir, "RP.dcm")
            ds_rp_sum = create_summed_plan(plan_paths, total_ref_dose, rp_sum_path)

            rs_dst = os.path.join(patient_outdir, "RS.dcm")
            safe_copy(first_structure_path, rs_dst)

            logger.info("  Multi-stage: Summed doses + created new plan + copied RS")

            # Fill result entry
            result_entry["final_RP"] = rp_sum_path
            result_entry["final_RD"] = rd_sum_path
            result_entry["final_RS"] = rs_dst
            result_entry["summed_reference_dose"] = float(total_ref_dose)
            result_entry["status"] = "multi-stage"

        logger.info(f"Processed patient {patient_id} -> {patient_outdir}")
        RESULTS.append(result_entry)

    logger.info("Done organizing and summing plans/doses.")

    # -------------------------------------------------------------------------
    # Create a DataFrame from RESULTS so you can check for anything fishy
    # -------------------------------------------------------------------------
    final_df = pd.DataFrame(RESULTS)
    logger.info("Final summary DataFrame:")
    logger.info("\n" + str(final_df))

    # If desired, you can save it to Excel or CSV:
    final_df.to_excel("../Data/organize_log_summary.xlsx", index=False)
    # final_df.to_csv("/home/konrad/final_summary.csv", index=False)

if __name__ == "__main__":
    main()


# Add CT path to metadata file
import pandas as pd
import os

# Load input file paths
ct_images_path = "../Data/CT_images.xlsx"
metadata_path = "../Data/metadata_checked20250302.xlsx"

# Load the Excel files
ct_images_df = pd.read_excel(ct_images_path, sheet_name=0)  # First sheet
metadata_df = pd.read_excel(metadata_path, sheet_name=0)  # First sheet

# Ensure data consistency
metadata_df['plans_CT_study'] = metadata_df['plans_CT_study'].astype(str).str.strip()
ct_images_df['CT_study'] = ct_images_df['CT_study'].astype(str).str.strip()

# Extract directory paths from organized_path
ct_images_df['organized_path'] = ct_images_df['organized_path'].apply(lambda x: os.path.dirname(str(x)))

# Create a mapping of `organized_path` using only `CT_study` and `PatientID`
organized_path_mapping = (
    ct_images_df.groupby(['CT_study', 'PatientID'])['organized_path']
    .first()  # Select the first occurrence if multiple exist
    .to_dict()
)

# Apply the mapping safely to preserve row count
metadata_df['organized_path'] = metadata_df.apply(
    lambda row: organized_path_mapping.get((row['plans_CT_study'], row['plans_patient_id']), 'MISSING_PATH'),
    axis=1
)

# Save the corrected metadata file
output_path = "../Data/metadata_checked20250302_withImages.xlsx"
metadata_df.to_excel(output_path, index=False)

print(f"Updated metadata saved to: {output_path}")

