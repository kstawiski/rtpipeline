import pandas as pd
import re

# File paths
plans_path = "../Data/plans.xlsx"
structure_sets_path = "../Data/structure_sets.xlsx"
dosimetrics_path = "../Data/dosimetrics.xlsx"

# Load data
plans_df = pd.read_excel(plans_path)
structure_sets_df = pd.read_excel(structure_sets_path)
dosimetrics_df = pd.read_excel(dosimetrics_path)

# Add prefixes to avoid column name conflicts
plans_df = plans_df.add_prefix("plans_")
structure_sets_df = structure_sets_df.add_prefix("structures_")
dosimetrics_df = dosimetrics_df.add_prefix("dosimetrics_")

# Extract core key from filenames for RP-RD merging (strict one-to-one mapping)
def extract_core_key(file_path):
    match = re.search(r'R[PD]\.(\d+)\.(.*?)\.dcm', file_path)
    return f"{match.group(1)}.{match.group(2)}" if match else None

plans_df["core_key"] = plans_df["plans_file_path"].apply(extract_core_key)
dosimetrics_df["core_key"] = dosimetrics_df["dosimetrics_file_path"].apply(extract_core_key)

# Ensure one-to-one merge between RP and RD
merged_df = plans_df.merge(
    dosimetrics_df, on="core_key", suffixes=("_plans", "_dosimetrics"), how="inner"
)

# Merge structure sets using patient_id (one-to-many relationship)
final_merged_df = merged_df.merge(
    structure_sets_df, left_on="plans_patient_id", right_on="structures_patient_id", 
    suffixes=("", "_structures"), how="left"
)

# Save final merged dataset
final_merged_df.to_excel("../Data/metadata.xlsx", index=False)


import pydicom

# Function to extract structures from RS file
def extract_structures_from_rs(rs_file_path):
    try:
        ds = pydicom.dcmread(rs_file_path)
        if "StructureSetROISequence" in ds:
            structures = {roi.ROIName for roi in ds.StructureSetROISequence}
            print(f"[RS] Extracted {len(structures)} structures from {rs_file_path}")
            return structures, ds.FrameOfReferenceUID
        print(f"[RS] No structures found in {rs_file_path}")
        return set(), ds.FrameOfReferenceUID
    except Exception as e:
        print(f"[ERROR] Failed to read RS file {rs_file_path}: {e}")
        return set(), None

# Function to extract Frame of Reference UID and doses to CTVs from RD file
def extract_frame_of_reference_and_ctv_doses(rd_file_path):
    try:
        ds = pydicom.dcmread(rd_file_path)
        frame_of_reference_uid = ds.FrameOfReferenceUID if "FrameOfReferenceUID" in ds else None

        # Extract doses assigned to CTVs
        ctv_doses = {}
        if "DVHSequence" in ds:
            for dvh in ds.DVHSequence:
                if hasattr(dvh, "DVHReferencedROISequence"):
                    for ref_roi in dvh.DVHReferencedROISequence:
                        roi_number = ref_roi.ReferencedROINumber
                        dose = dvh.DVHMaximumDose if hasattr(dvh, "DVHMaximumDose") else None
                        if dose:
                            ctv_doses[f"CTV_{roi_number}_dose"] = dose

        print(f"[RD] Extracted Frame of Reference UID: {frame_of_reference_uid} from {rd_file_path}")
        print(f"[RD] Extracted {len(ctv_doses)} CTV doses from {rd_file_path}")

        return frame_of_reference_uid, ctv_doses
    except Exception as e:
        print(f"[ERROR] Failed to read RD file {rd_file_path}: {e}")
        return None, {}

# Process each RD-RS pair in the merged dataset
matching_files = []
total_pairs = 0
matched_pairs = 0

print("\n[INFO] Checking RD-RS structure consistency using Frame of Reference UID...")

for index, row in final_merged_df.iterrows():
    rd_file = row.get("dosimetrics_file_path")
    rs_file = row.get("structures_file_path")
    
    if pd.isna(rd_file) or pd.isna(rs_file):
        print(f"[SKIP] Missing RD or RS file for row {index}. Skipping.")
        continue
    
    total_pairs += 1
    print(f"\n[PROCESSING] Comparing RD: {rd_file} with RS: {rs_file}")

    rd_for_uid, rd_ctv_doses = extract_frame_of_reference_and_ctv_doses(rd_file)
    rs_structures, rs_for_uid = extract_structures_from_rs(rs_file)
    
    # Check if Frame of Reference UID matches
    if rd_for_uid and rs_for_uid and rd_for_uid == rs_for_uid:
        print(f"[MATCH] RD and RS have matching Frame of Reference UID: {rd_for_uid}.")
        row_data = row.copy()
        row_data["available_structures"] = ", ".join(rs_structures)  # Store structures as comma-separated list
        row_data.update(rd_ctv_doses)  # Add CTV dose data dynamically
        matching_files.append(row_data)
        matched_pairs += 1
    else:
        print(f"[NO MATCH] RD and RS have different Frame of Reference UIDs (RD: {rd_for_uid}, RS: {rs_for_uid}).")

# Convert the filtered dataset to a DataFrame
filtered_df = pd.DataFrame(matching_files)

# Save and display the result
filtered_df.to_excel("../Data/metadata.xlsx", index=False)


# Filter the dataset to retain only cases where "PTV" is present in the available structures list
filtered_df_ptv = filtered_df[filtered_df["available_structures"].str.contains("PTV", case=False, na=False)]

# Save and display the filtered dataset
filtered_df_ptv.to_excel("../Data/metadata.xlsx", index=False)

# Summary output
print(f"[INFO] Filtered dataset contains {len(filtered_df_ptv)} cases with PTV structures.")

filtered_df_ptv

