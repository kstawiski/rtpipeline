!pip install dicompyler-core nested-lookup openpyxl pydicom==1.4.2

from IPython.display import display
import os
import fnmatch
import logging
import pandas as pd
from tqdm import tqdm
from dicompylercore import dicomparser
from nested_lookup import nested_lookup
from typing import List, Dict, Any, Generator, Set
from pathlib import Path
import shutil
import hashlib
import json  # new import for JSON handling

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DicomSeriesOrganizer:
    """Class to handle DICOM file processing and organization operations."""
    
    def __init__(self, base_directory: str, output_base: str):
        """
        Initialize the DicomSeriesOrganizer.
        
        Args:
            base_directory (str): Root directory containing DICOM files
            output_base (str): Base directory for organized output
        """
        self.base_directory = Path(base_directory)
        self.output_base = Path(output_base)
        self.required_packages = [
            'dicompyler-core',
            'nested-lookup',
            'openpyxl',
            'pydicom'  # removed version constraint to use the latest version
        ]
        
    def get_file_hash(self, filepath: str) -> str:
        """
        Calculate MD5 hash of file to identify duplicates.
        
        Args:
            filepath (str): Path to file
            
        Returns:
            str: MD5 hash of file
        """
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    @staticmethod
    def safe_lookup(tag: str, dicom_plan: Any, index: int = 0) -> str:
        """
        Safely extract DICOM tag value.
        
        Args:
            tag (str): DICOM tag to look up
            dicom_plan: DicomParser object
            index (int): Index for nested lookup
            
        Returns:
            str: Tag value or "NA" if not found
        """
        try:
            # Convert the Dataset to JSON string and then parse into a dict.
            ds_json = json.loads(dicom_plan.ds.to_json())
            return nested_lookup(tag, ds_json)[index]['Value'][0]
        except (IndexError, KeyError, AttributeError):
            return "NA"

    def find_ct_files(self, pattern: str = 'CT*.dcm') -> Generator[str, None, None]:
        """
        Find CT DICOM files in directory and subdirectories.
        
        Args:
            pattern (str): File pattern to match
            
        Yields:
            str: Path to matching file
        """
        try:
            for root, _, files in os.walk(self.base_directory):
                for filename in fnmatch.filter(files, pattern):
                    yield os.path.join(root, filename)
        except Exception as e:
            logger.error(f"Error searching for CT files: {str(e)}")
            raise

    def get_destination_path(self, patient_id: str, series_number: str, instance_number: str) -> Path:
        """
        Generate destination path for organized file structure.
        
        Args:
            patient_id (str): Patient identifier
            series_number (str): Series number
            instance_number (str): Instance number
            
        Returns:
            Path: Destination path for file
        """
        return self.output_base / str(patient_id) / f"series_{series_number}" / f"CT_{instance_number}.dcm"

    def process_ct_files(self) -> pd.DataFrame:
        """
        Process CT files and remove duplicates based on file content.
        
        Returns:
            pd.DataFrame: DataFrame containing unique CT file information
        """
        try:
            paths = list(self.find_ct_files())
            if not paths:
                logger.warning("No CT files found")
                return pd.DataFrame()
            
            data = []
            seen_hashes: Set[str] = set()
            
            for path in tqdm(paths, desc="Processing CT files"):
                try:
                    # Calculate file hash
                    file_hash = self.get_file_hash(path)
                    
                    # Skip if we've seen this file before
                    if file_hash in seen_hashes:
                        logger.info(f"Skipping duplicate file: {path}")
                        continue
                    
                    seen_hashes.add(file_hash)
                    
                    # Process DICOM file
                    dicomrt_file = dicomparser.DicomParser(path)
                    
                    # Get DICOM tags
                    patient_id = self.safe_lookup("00100020", dicomrt_file)
                    series_number = self.safe_lookup("00200011", dicomrt_file)
                    instance_number = self.safe_lookup("00200013", dicomrt_file)
                    
                    # Generate destination path
                    dest_path = self.get_destination_path(patient_id, series_number, instance_number)
                    
                    row_info = {
                        "original_path": path,
                        "organized_path": str(dest_path),
                        "CT_series": self.safe_lookup("0020000E", dicomrt_file),
                        "CT_study": self.safe_lookup("0020000D", dicomrt_file),
                        "PatientID": patient_id,
                        "SeriesNumber": series_number,
                        "InstanceNumber": instance_number,
                        "file_hash": file_hash
                    }
                    data.append(row_info)
                    
                except Exception as e:
                    logger.error(f"Error processing file {path}: {str(e)}")
                    continue
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error in process_ct_files: {str(e)}")
            raise

    def organize_files(self, df: pd.DataFrame) -> None:
        """
        Organize DICOM files into patient/series structure.
        
        Args:
            df (pd.DataFrame): DataFrame containing CT file information
        """
        try:
            for _, row in tqdm(df.iterrows(), desc="Organizing files", total=len(df)):
                try:
                    # Get destination path
                    destination = Path(row['organized_path'])
                    
                    # Create directory structure
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy file to new location
                    shutil.copy2(row['original_path'], destination)
                    
                except Exception as e:
                    logger.error(f"Error organizing file {row['original_path']}: {str(e)}")
                    continue
                    
            logger.info("File organization complete")
            
        except Exception as e:
            logger.error(f"Error in organize_files: {str(e)}")
            raise

def main():
    """Main execution function."""
    try:
        # Initialize paths
        input_dir = "../DICOM"
        output_dir = "../DICOM_OrganizedCT"
        excel_path = "../Data/CT_images.xlsx"
        csv_path = "../Data/CT_images.csv"
        
        # Initialize organizer
        organizer = DicomSeriesOrganizer(input_dir, output_dir)
        
        # Process files and remove duplicates
        df = organizer.process_ct_files()
        
        # Save to Excel and CSV
        df.to_excel(excel_path, index=False)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(df)} unique files information to {excel_path}")
        
        # Organize files into patient/series structure
        organizer.organize_files(df)
        
    except Exception as e:
        logger.error(f"Main execution error: {str(e)}")
        raise

if __name__ == "__main__":
    main()


