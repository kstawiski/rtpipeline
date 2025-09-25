!pip install git+https://github.com/wasserth/TotalSegmentator.git
!pip install --upgrade rt_utils
!totalseg_set_license -l aca_HZ08ZGEUAM3H3U
!pip install dcm2niix

import pandas as pd
metadata = pd.read_excel("../Data/metadata_checked20250302_withImages.xlsx")
metadata

import os
import subprocess
import pandas as pd

# Load your metadata DataFrame
# Example: metadata = pd.read_csv('metadata.csv')

# Define the path to the log file that tracks completed directories
log_file = '../Logs/TotalSegmentator_processed_directories.log'

# Define the path to the summary CSV file
summary_csv = '../Data/TotalSegmentator_processing_summary.csv'

def load_processed_directories(log_file):
    """Load the list of processed directories from the log file."""
    if os.path.exists(log_file):
        with open(log_file, 'r') as file:
            processed = file.read().splitlines()
        print(f"Loaded {len(processed)} processed directories from log.")
    else:
        processed = []
        print("No log file found. Starting fresh.")
    return set(processed)

def save_processed_directory(log_file, directory):
    """Append a processed directory to the log file."""
    with open(log_file, 'a') as file:
        file.write(f"{directory}\n")
    print(f"Logged processed directory: {directory}")

def run_command(command):
    """Run a shell command and handle exceptions."""
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, shell=True, executable='/bin/bash')
        print(f"Success: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return False

def convert_dicom_to_nifti(dicom_dir, output_dir):
    """Convert DICOM series to NIfTI format using dcm2niix."""
    os.makedirs(output_dir, exist_ok=True)
    command = f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate rt && dcm2niix -z y -o {output_dir} {dicom_dir}"
    print(f"Executing DICOM to NIfTI conversion: {command}")
    return run_command(command)

def run_totalsegmentator(input_path, output_dir, output_type):
    """Run TotalSegmentator on the given input path."""
    os.makedirs(output_dir, exist_ok=True)
    command = f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate rt && TotalSegmentator -i {input_path} -o {output_dir} -ta total -ot {output_type}"
    print(f"Executing TotalSegmentator: {command}")
    return run_command(command)

def update_summary(summary_csv, patient_id, dicom_input, nifti_output, dicom_seg_output, nifti_seg_output):
    """Update the summary DataFrame with the paths for each patient."""
    new_row = {
        'Patient_ID': patient_id,
        'DICOM_Input': dicom_input,
        'NIfTI_Output': nifti_output,
        'DICOM_Segmentation_Output': dicom_seg_output,
        'NIfTI_Segmentation_Output': nifti_seg_output
    }
    
    if os.path.exists(summary_csv):
        summary_df = pd.read_csv(summary_csv)
    else:
        summary_df = pd.DataFrame(columns=['Patient_ID', 'DICOM_Input', 'NIfTI_Output', 'DICOM_Segmentation_Output', 'NIfTI_Segmentation_Output'])

    # Convert new_row to a DataFrame and concatenate
    new_row_df = pd.DataFrame([new_row])
    summary_df = pd.concat([summary_df, new_row_df], ignore_index=True)

    summary_df.to_csv(summary_csv, index=False)
    print(f"Updated summary CSV for patient: {patient_id}")
    
# Load the list of processed directories
processed_directories = load_processed_directories(log_file)

# Iterate over each patient directory in the metadata
for patient_dir in metadata['organized_path']:
    patient_id = os.path.basename(os.path.dirname(patient_dir))
    if patient_dir in processed_directories:
        print(f"Skipping already processed directory: {patient_dir}")
        continue

    print(f"Processing directory: {patient_dir}")

    # Convert DICOM to NIfTI
    nifti_output_dir = os.path.join(patient_dir, 'nifti')
    if not convert_dicom_to_nifti(patient_dir, nifti_output_dir):
        print(f"Failed to convert DICOM to NIfTI for directory: {patient_dir}. Skipping.")
        continue

    # Identify the converted NIfTI file
    nifti_files = [f for f in os.listdir(nifti_output_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
    if not nifti_files:
        print(f"No NIfTI files found in {nifti_output_dir}. Skipping NIfTI segmentation.")
        continue
    nifti_file = os.path.join(nifti_output_dir, nifti_files[0])

    # Run TotalSegmentator in DICOM mode
    dicom_output_dir = f"{patient_dir}_total_dicom"
    if not run_totalsegmentator(patient_dir, dicom_output_dir, 'dicom'):
        print(f"Failed to run TotalSegmentator in DICOM mode for directory: {patient_dir}. Skipping.")
        continue

    # Run TotalSegmentator in NIfTI mode
    nifti_seg_output_dir = f"{patient_dir}_total_nifti"
    if not run_totalsegmentator(nifti_file, nifti_seg_output_dir, 'nifti'):
        print(f"Failed to run TotalSegmentator in NIfTI mode for file: {nifti_file}. Skipping.")
        continue

    # Log the processed directory
    save_processed_directory(log_file, patient_dir)

    # Update the summary DataFrame
    update_summary(summary_csv, patient_id, patient_dir, nifti_file, dicom_output_dir, nifti_seg_output_dir)


