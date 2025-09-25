import os
import shutil
import copy
import logging
import datetime

import pandas as pd
import pydicom
import numpy as np
from pydicom.uid import generate_uid
from scipy.ndimage import map_coordinates

# =============================================================================
# Configuration
# =============================================================================

METADATA_PATH = "../Data/metadata_checked20250302.xlsx"
OUTPUT_DIR = "../Data_Organized"
LOG_DIR = "../Logs/"

# Create log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging to both console and file
log_filename = os.path.join(LOG_DIR, f"rt_merge_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# We'll accumulate final info here to produce a DataFrame
RESULTS = []

# =============================================================================
# Helper Functions
# =============================================================================

def safe_copy(src, dst):
    """
    Copy src to dst, overwriting if needed.
    
    Args:
        src (str): Source file path
        dst (str): Destination file path
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not os.path.isfile(src):
            raise FileNotFoundError(f"File not found: {src}")
        if os.path.exists(dst) and not os.path.isfile(dst):
            raise IsADirectoryError(f"Destination exists and is a directory: {dst}")
        if os.path.exists(dst) and os.path.isfile(dst):
            # Only remove if they're not the same file
            if not os.path.samefile(src, dst):
                os.remove(dst)
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        logger.error(f"Error copying {src} to {dst}: {e}")
        return False

def validate_dicom(file_path, expected_modality=None):
    """
    Validate that a file is a valid DICOM with expected modality.
    
    Args:
        file_path (str): Path to DICOM file
        expected_modality (str): Expected modality (e.g., 'RTPLAN', 'RTDOSE', 'RTSTRUCT')
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        ds = pydicom.dcmread(file_path)
        if expected_modality and hasattr(ds, 'Modality') and ds.Modality != expected_modality:
            logger.warning(f"File {file_path} has modality {ds.Modality}, expected {expected_modality}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error validating DICOM {file_path}: {e}")
        return False

def create_summed_plan(plan_files, total_dose_gy, out_plan_path):
    """
    Create a 'summed' RT Plan by taking the first plan as a template
    and updating its prescription dose and other relevant fields.
    
    Args:
        plan_files (list): List of paths to RT Plan DICOM files
        total_dose_gy (float): Total prescription dose in Gy
        out_plan_path (str): Path to save the summed plan
        
    Returns:
        pydicom.dataset.FileDataset: The summed plan dataset
    """
    if not plan_files:
        raise ValueError("No plan files to sum.")
    
    # Validate all plan files
    valid_plans = []
    for plan_file in plan_files:
        if validate_dicom(plan_file, 'RTPLAN'):
            valid_plans.append(plan_file)
        else:
            logger.warning(f"Skipping invalid plan file: {plan_file}")
    
    if not valid_plans:
        raise ValueError("No valid plan files to sum.")
    
    # Read the first plan as a template
    ds_plan = pydicom.dcmread(valid_plans[0])
    
    # Generate new UIDs
    study_instance_uid = ds_plan.StudyInstanceUID  # Keep the same study
    series_instance_uid = generate_uid()
    sop_instance_uid = generate_uid()
    
    # Create a deep copy to avoid modifying the original
    new_plan = copy.deepcopy(ds_plan)
    
    # Update identification
    new_plan.SeriesInstanceUID = series_instance_uid
    new_plan.SOPInstanceUID = sop_instance_uid
    new_plan.SeriesDescription = f"Summed Plan ({len(valid_plans)} fractions)"
    
    # Update creation information
    new_plan.InstanceCreationDate = datetime.datetime.now().strftime("%Y%m%d")
    new_plan.InstanceCreationTime = datetime.datetime.now().strftime("%H%M%S")
    
    # Update prescription information
    if hasattr(new_plan, "DoseReferenceSequence") and new_plan.DoseReferenceSequence:
        for dose_ref in new_plan.DoseReferenceSequence:
            if hasattr(dose_ref, "TargetPrescriptionDose"):
                dose_ref.TargetPrescriptionDose = float(total_dose_gy)
                logger.info(f"Updated plan prescription dose to {total_dose_gy} Gy")
    else:
        logger.warning("No DoseReferenceSequence found in plan. Prescription dose not updated.")
    
    # Update fraction group sequence if it exists
    if hasattr(new_plan, "FractionGroupSequence") and new_plan.FractionGroupSequence:
        # Collect total number of fractions from all plans
        total_fractions = 0
        for plan_file in valid_plans:
            plan = pydicom.dcmread(plan_file)
            if hasattr(plan, "FractionGroupSequence") and plan.FractionGroupSequence:
                for fg in plan.FractionGroupSequence:
                    if hasattr(fg, "NumberOfFractionsPlanned"):
                        total_fractions += fg.NumberOfFractionsPlanned
        
        # Update the fraction group in the new plan
        for fg in new_plan.FractionGroupSequence:
            fg.NumberOfFractionsPlanned = total_fractions
            
            # If beam sequence exists, update meterset values proportionally
            if hasattr(fg, "ReferencedBeamSequence") and fg.ReferencedBeamSequence:
                dose_ratio = float(total_dose_gy) / float(ds_plan.DoseReferenceSequence[0].TargetPrescriptionDose)
                for beam_ref in fg.ReferencedBeamSequence:
                    if hasattr(beam_ref, "BeamMeterset"):
                        beam_ref.BeamMeterset = float(beam_ref.BeamMeterset) * dose_ratio
                        
        logger.info(f"Updated plan to {total_fractions} total fractions")
    
    # Add a private tag to indicate this is a summed plan
    private_creator_tag = pydicom.tag.Tag(0x0071, 0x0010)
    new_plan.add_new(private_creator_tag, 'LO', 'SUMMED_PLAN')
    
    private_data_tag = pydicom.tag.Tag(0x0071, 0x1000)
    source_plans = ",".join([os.path.basename(p) for p in valid_plans])
    new_plan.add_new(private_data_tag, 'LO', f"Source Plans: {source_plans}")
    
    # Add original prescription doses
    private_dose_tag = pydicom.tag.Tag(0x0071, 0x1001)
    original_doses = []
    for plan_file in valid_plans:
        plan = pydicom.dcmread(plan_file)
        if hasattr(plan, "DoseReferenceSequence") and plan.DoseReferenceSequence:
            for dose_ref in plan.DoseReferenceSequence:
                if hasattr(dose_ref, "TargetPrescriptionDose"):
                    original_doses.append(str(dose_ref.TargetPrescriptionDose))
                    break
    if original_doses:
        new_plan.add_new(private_dose_tag, 'LO', f"Original Doses (Gy): {','.join(original_doses)}")
    
    # Save the new plan
    new_plan.save_as(out_plan_path)
    logger.info(f"Wrote summed plan to {out_plan_path}")
    
    return new_plan

def sum_doses_with_resample(dose_files, out_dose_path):
    """
    Summation of multiple RT Dose files with resampling to a reference grid.
    
    Args:
        dose_files (list): List of paths to RT Dose DICOM files
        out_dose_path (str): Path to save the summed dose
        
    Returns:
        pydicom.dataset.FileDataset: The summed dose dataset
    """
    if not dose_files:
        raise ValueError("No dose files to sum.")
    
    # Validate all dose files
    valid_doses = []
    for dose_file in dose_files:
        if validate_dicom(dose_file, 'RTDOSE'):
            valid_doses.append(dose_file)
        else:
            logger.warning(f"Skipping invalid dose file: {dose_file}")
    
    if not valid_doses:
        raise ValueError("No valid dose files to sum.")
    
    # 1) Determine reference grid - use highest resolution dose as reference
    ref_idx = 0
    best_resolution = float('inf')
    
    for i, dose_path in enumerate(valid_doses):
        ds = pydicom.dcmread(dose_path)
        if not hasattr(ds, "PixelSpacing") or len(ds.PixelSpacing) < 2:
            continue
        
        # Calculate volume of a single voxel as a measure of resolution
        pixel_spacing = ds.PixelSpacing
        slice_thickness = 1.0
        
        if hasattr(ds, "GridFrameOffsetVector") and len(ds.GridFrameOffsetVector) > 1:
            slice_thickness = abs(float(ds.GridFrameOffsetVector[1] - ds.GridFrameOffsetVector[0]))
        
        voxel_volume = float(pixel_spacing[0]) * float(pixel_spacing[1]) * slice_thickness
        
        if voxel_volume < best_resolution:
            best_resolution = voxel_volume
            ref_idx = i
    
    # Use the selected reference dose
    ref_path = valid_doses[ref_idx]
    ds_ref = pydicom.dcmread(ref_path)
    ref_array = ds_ref.pixel_array.astype(np.float32)
    
    # Apply scaling to get actual dose values
    dose_scaling = float(ds_ref.DoseGridScaling) if hasattr(ds_ref, "DoseGridScaling") else 1.0
    ref_array = ref_array * dose_scaling
    
    rows_ref = ds_ref.Rows
    cols_ref = ds_ref.Columns
    frames_ref = ds_ref.NumberOfFrames if hasattr(ds_ref, "NumberOfFrames") else 1
    
    logger.info(f"Reference Dose: {ref_path}")
    logger.info(f" - shape: {ref_array.shape}, Rows={rows_ref}, Cols={cols_ref}, Frames={frames_ref}")
    logger.info(f" - selected as reference due to highest resolution")
    
    # Extract reference grid geometry
    origin_ref = list(map(float, getattr(ds_ref, "ImagePositionPatient", [0, 0, 0])))
    pixel_spacing_ref = list(map(float, getattr(ds_ref, "PixelSpacing", [1.0, 1.0])))
    offsets_ref = getattr(ds_ref, "GridFrameOffsetVector", None)
    
    if offsets_ref is not None and len(offsets_ref) == frames_ref:
        z_positions_ref = np.array([origin_ref[2] + float(offset) for offset in offsets_ref])
    else:
        slice_thickness_ref = 1.0
        z_positions_ref = np.array([origin_ref[2] + i * slice_thickness_ref for i in range(frames_ref)])
    
    y_positions_ref = np.array([origin_ref[1] + r * pixel_spacing_ref[0] for r in range(rows_ref)])
    x_positions_ref = np.array([origin_ref[0] + c * pixel_spacing_ref[1] for c in range(cols_ref)])
    
    # Initialize accumulated dose with the reference dose
    accumulated = ref_array.copy()
    
    # 2) Resample and accumulate subsequent doses
    for dose_path in valid_doses:
        # Skip the reference dose since it's already included
        if dose_path == ref_path:
            continue
        
        ds_next = pydicom.dcmread(dose_path)
        arr_next = ds_next.pixel_array.astype(np.float32)
        
        # Apply scaling to get actual dose values
        dose_scaling_next = float(ds_next.DoseGridScaling) if hasattr(ds_next, "DoseGridScaling") else 1.0
        arr_next = arr_next * dose_scaling_next
        
        rows_next = ds_next.Rows
        cols_next = ds_next.Columns
        frames_next = ds_next.NumberOfFrames if hasattr(ds_next, "NumberOfFrames") else 1
        
        logger.info(f"Resampling {dose_path}")
        logger.info(f" - shape: {arr_next.shape}, Rows={rows_next}, Cols={cols_next}, Frames={frames_next}")
        
        # Get grid geometry for this dose
        origin_next = list(map(float, getattr(ds_next, "ImagePositionPatient", [0, 0, 0])))
        pixel_spacing_next = list(map(float, getattr(ds_next, "PixelSpacing", [1.0, 1.0])))
        offsets_next = getattr(ds_next, "GridFrameOffsetVector", None)
        
        # Build z coordinate mapping
        if offsets_next is not None and len(offsets_next) == frames_next:
            z_coords_next = np.array([origin_next[2] + float(offset) for offset in offsets_next])
        else:
            slice_thickness_next = 1.0
            z_coords_next = np.array([origin_next[2] + i * slice_thickness_next for i in range(frames_next)])
        
        # Check if we need to reshape the array for 3D resampling
        if frames_next == 1:
            # Create a 3D array from a 2D slice for consistent processing
            arr_next = arr_next.reshape((1, rows_next, cols_next))
        
        # Build coordinate arrays in the next dose's frame of reference
        # We'll use these to map from reference grid to this dose's grid
        z_min, z_max = z_coords_next.min(), z_coords_next.max()
        y_min = origin_next[1]
        y_max = origin_next[1] + (rows_next - 1) * pixel_spacing_next[0]
        x_min = origin_next[0]
        x_max = origin_next[0] + (cols_next - 1) * pixel_spacing_next[1]
        
        # Create meshgrid of reference coordinates
        Z, Y, X = np.meshgrid(z_positions_ref, y_positions_ref, x_positions_ref, indexing='ij')
        
        # Convert world coordinates to indices in the dose array
        # First, clip coordinates to stay within the dose volume bounds
        Z_clipped = np.clip(Z, z_min, z_max)
        Y_clipped = np.clip(Y, y_min, y_max)
        X_clipped = np.clip(X, x_min, x_max)
        
        # Then convert to array indices
        z_indices = np.searchsorted(z_coords_next, Z_clipped) - 0.5
        y_indices = (Y_clipped - origin_next[1]) / pixel_spacing_next[0]
        x_indices = (X_clipped - origin_next[0]) / pixel_spacing_next[1]
        
        # Check that the indices are within valid range
        # Add a small buffer to avoid interpolation artifacts at edges
        valid_z = (Z >= z_min - 0.1) & (Z <= z_max + 0.1)
        valid_y = (Y >= y_min - 0.1) & (Y <= y_max + 0.1)
        valid_x = (X >= x_min - 0.1) & (X <= x_max + 0.1)
        valid_mask = valid_z & valid_y & valid_x
        
        # Prepare coordinate arrays for map_coordinates
        coords = np.stack([z_indices, y_indices, x_indices], axis=0)
        
        # Resample dose to reference grid
        resampled = map_coordinates(arr_next, coords, order=1, mode='constant', cval=0.0)
        resampled = resampled.reshape((frames_ref, rows_ref, cols_ref))
        
        # Use the valid mask to only add dose where the next dose actually covers
        resampled[~valid_mask] = 0
        
        # Add to accumulated dose
        accumulated += resampled
        
        logger.info(f" - Resampled and added to accumulated dose")
    
    # 3) Create a new RT Dose dataset based on the reference
    new_ds = copy.deepcopy(ds_ref)
    
    # Generate new UIDs but maintain study relationship
    study_instance_uid = ds_ref.StudyInstanceUID  # Keep the same study
    new_ds.SeriesInstanceUID = generate_uid()
    new_ds.SOPInstanceUID = generate_uid()
    
    # Update descriptive fields
    new_ds.SeriesDescription = f"Summed Dose ({len(valid_doses)} fractions)"
    new_ds.DoseSummationType = "PLAN"
    
    # Update creation information
    new_ds.InstanceCreationDate = datetime.datetime.now().strftime("%Y%m%d")
    new_ds.InstanceCreationTime = datetime.datetime.now().strftime("%H%M%S")
    
    # Convert accumulated dose to integer representation for storage
    # Select appropriate scaling factor based on dose range
    max_dose = np.max(accumulated)
    
    # Use scaling that gives good precision but stays within int32 range
    if max_dose > 1000:
        scaling_factor = 10.0  # For very high doses, less precision
    elif max_dose > 100:
        scaling_factor = 100.0  # Medium precision
    else:
        scaling_factor = 1000.0  # High precision for typical doses
    
    # Replace any NaN values with zero
    accumulated = np.nan_to_num(accumulated, nan=0.0)
    
    # Scale and convert to int
    accumulated_int = np.rint(accumulated * scaling_factor).astype(np.int32)
    
    # Update DICOM pixel attributes
    new_ds.BitsAllocated = 32
    new_ds.BitsStored = 32
    new_ds.HighBit = 31
    new_ds.PixelRepresentation = 1  # Signed
    new_ds.DoseGridScaling = 1.0 / scaling_factor
    
    # Remove attributes that could cause issues with storage of int32 data
    for tag in [(0x0028, 0x0106), (0x0028, 0x0107)]:
        if tag in new_ds:
            del new_ds[tag]
    
    # Store the pixel data
    new_ds.PixelData = accumulated_int.tobytes()
    
    # Remove functional groups if present (might cause issues)
    if hasattr(new_ds, "PerFrameFunctionalGroupsSequence"):
        del new_ds.PerFrameFunctionalGroupsSequence
    
    # Add private tags to document the source doses
    private_creator_tag = pydicom.tag.Tag(0x0071, 0x0010)
    new_ds.add_new(private_creator_tag, 'LO', 'SUMMED_DOSE')
    
    private_data_tag = pydicom.tag.Tag(0x0071, 0x1000)
    source_doses = ",".join([os.path.basename(d) for d in valid_doses])
    new_ds.add_new(private_data_tag, 'LO', f"Source Doses: {source_doses}")
    
    # Save the new dose
    new_ds.save_as(out_dose_path)
    logger.info(f"Wrote summed dose to {out_dose_path}")
    
    return new_ds

# =============================================================================
# Main Script
# =============================================================================

def main():
    """
    Main function to process and organize RT plan data
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Make sure logs directory exists
    os.makedirs(LOG_DIR, exist_ok=True)
    
    logger.info(f"Starting RT Plan and Dose processing")
    logger.info(f"Metadata file: {METADATA_PATH}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Read metadata
    try:
        df = pd.read_excel(METADATA_PATH)
        logger.info(f"Loaded {len(df)} rows from {METADATA_PATH}")
    except Exception as e:
        logger.error(f"Failed to load metadata file: {e}")
        return
    
    # Check required columns exist
    required_columns = [
        "plans_patient_id", 
        "plans_file_path", 
        "dosimetrics_file_path", 
        "structures_file_path",
        "plans_reference_dose"
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return
    
    # Validate all file paths exist before processing
    path_columns = ["plans_file_path", "dosimetrics_file_path", "structures_file_path"]
    df['all_files_exist'] = True
    
    for col in path_columns:
        df[f'{col}_exists'] = df[col].apply(lambda x: os.path.exists(x))
        df['all_files_exist'] &= df[f'{col}_exists']
    
    if not df['all_files_exist'].all():
        missing_files_df = df[~df['all_files_exist']]
        logger.warning(f"Some files are missing. {len(missing_files_df)} rows affected.")
        
        # Log the first few missing files
        for idx, row in missing_files_df.head(5).iterrows():
            for col in path_columns:
                if not row[f'{col}_exists']:
                    logger.warning(f"Missing file for patient {row['plans_patient_id']}: {row[col]}")
        
        # Filter out rows with missing files
        df = df[df['all_files_exist']]
        logger.info(f"Continuing with {len(df)} valid rows.")
    
    # Drop the temporary validation columns
    for col in path_columns:
        if f'{col}_exists' in df.columns:
            df = df.drop(columns=[f'{col}_exists'])
    
    if 'all_files_exist' in df.columns:
        df = df.drop(columns=['all_files_exist'])
    
    # Group by patient ID
    grouped = df.groupby("plans_patient_id")
    
    # Process each patient
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
            "errors": ""
        }
        
        try:
            # (A) Identify the first structure set in the group (by creation date if available)
            if 'structures_creation_date' in group.columns:
                # Sort by creation date if available
                group = group.sort_values('structures_creation_date')
                first_row = group.iloc[0]
            else:
                # Otherwise use the lowest row index
                first_index = group.index.min()
                first_row = group.loc[first_index]
            
            first_structure_path = first_row["structures_file_path"]
            logger.info(f"  Selected structure set: {first_structure_path}")
            
            # Check if structure set is valid
            if not validate_dicom(first_structure_path, 'RTSTRUCT'):
                logger.error(f"  Invalid structure set: {first_structure_path}")
                result_entry["status"] = "error_invalid_structure"
                result_entry["errors"] = f"Invalid structure set: {first_structure_path}"
                RESULTS.append(result_entry)
                continue
            
            # Filter group to only include rows that use this structure set
            group_filtered = group[group["structures_file_path"] == first_structure_path]
            if len(group_filtered) == 0:
                logger.warning("  No rows remain after filtering for selected structure set.")
                result_entry["status"] = "error_no_matching_structures"
                result_entry["errors"] = "No plans/doses match the selected structure set"
                RESULTS.append(result_entry)
                continue
            
            # Distinct plan & dose files from the filtered subset
            plan_paths = group_filtered["plans_file_path"].unique().tolist()
            dose_paths = group_filtered["dosimetrics_file_path"].unique().tolist()
            
            # Summation of reference doses from the filtered subset
            total_ref_dose = group_filtered["plans_reference_dose"].sum()
            logger.info(f"  Found {len(plan_paths)} plan(s) and {len(dose_paths)} dose file(s).")
            logger.info(f"  Summed reference dose = {total_ref_dose} Gy")
            
            if len(plan_paths) == 0 or len(dose_paths) == 0:
                logger.error(f"  Missing plans or doses for patient {patient_id}")
                result_entry["status"] = "error_missing_files"
                result_entry["errors"] = f"Missing plans ({len(plan_paths)}) or doses ({len(dose_paths)})"
                RESULTS.append(result_entry)
                continue
            
            # Output file paths
            rp_dst = os.path.join(patient_outdir, "RP.dcm")
            rd_dst = os.path.join(patient_outdir, "RD.dcm")
            rs_dst = os.path.join(patient_outdir, "RS.dcm")
            
            if len(plan_paths) == 1 and len(dose_paths) == 1:
                # Single-stage => just copy
                plan_copy_ok = safe_copy(plan_paths[0], rp_dst)
                dose_copy_ok = safe_copy(dose_paths[0], rd_dst)
                struct_copy_ok = safe_copy(first_structure_path, rs_dst)
                
                if plan_copy_ok and dose_copy_ok and struct_copy_ok:
                    logger.info("  Single-stage: Copied RP, RD, RS successfully")
                    
                    # Fill result entry
                    result_entry["final_RP"] = rp_dst
                    result_entry["final_RD"] = rd_dst
                    result_entry["final_RS"] = rs_dst
                    result_entry["summed_reference_dose"] = float(total_ref_dose)
                    result_entry["status"] = "single-stage"
                else:
                    logger.error("  Failed to copy one or more files")
                    result_entry["status"] = "error_copy_failed"
                    result_entry["errors"] = "File copy operation failed"
            
            else:
                # Multi-stage => sum
                try:
                    # Sum doses
                    ds_rd_sum = sum_doses_with_resample(dose_paths, rd_dst)
                    
                    # Create summed plan
                    ds_rp_sum = create_summed_plan(plan_paths, total_ref_dose, rp_dst)
                    
                    # Copy structure set
                    struct_copy_ok = safe_copy(first_structure_path, rs_dst)
                    
                    if struct_copy_ok:
                        logger.info("  Multi-stage: Successfully summed doses + created new plan + copied RS")
                        
                        # Fill result entry
                        result_entry["final_RP"] = rp_dst
                        result_entry["final_RD"] = rd_dst
                        result_entry["final_RS"] = rs_dst
                        result_entry["summed_reference_dose"] = float(total_ref_dose)
                        result_entry["status"] = "multi-stage"
                    else:
                        logger.error("  Failed to copy structure set")
                        result_entry["status"] = "error_structure_copy_failed"
                        result_entry["errors"] = "Structure set copy failed"
                
                except Exception as e:
                    logger.error(f"  Error during dose/plan summation: {e}")
                    result_entry["status"] = "error_summation_failed"
                    result_entry["errors"] = str(e)
            
            logger.info(f"Completed processing for patient {patient_id}")
        
        except Exception as e:
            logger.error(f"Unhandled error processing patient {patient_id}: {e}")
            result_entry["status"] = "error_unhandled"
            result_entry["errors"] = str(e)
        
        RESULTS.append(result_entry)
    
    logger.info("Completed organizing and summing plans/doses.")
    
    # Create a DataFrame from RESULTS
    final_df = pd.DataFrame(RESULTS)
    
    # Count results by status
    status_counts = final_df['status'].value_counts()
    logger.info(f"Results summary by status:")
    for status, count in status_counts.items():
        logger.info(f"  {status}: {count}")
    
    # Log error summary if any errors occurred
    if 'errors' in final_df.columns and final_df['errors'].notna().any():
        error_counts = final_df[final_df['errors'].notna()].groupby('status')['patient_id'].count()
        logger.info(f"Error summary by status:")
        for status, count in error_counts.items():
            logger.info(f"  {status}: {count}")
    
    # Save results to Excel and CSV
    results_xlsx = os.path.join(LOG_DIR, f"rt_merge_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    results_csv = os.path.join(LOG_DIR, f"rt_merge_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    final_df.to_excel(results_xlsx, index=False)
    final_df.to_csv(results_csv, index=False)
    
    logger.info(f"Results saved to:")
    logger.info(f"  Excel: {results_xlsx}")
    logger.info(f"  CSV: {results_csv}")

if __name__ == "__main__":
    main()

