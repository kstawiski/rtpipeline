import os
import logging
import fnmatch
from typing import List, Dict, Any, Generator, Set, Tuple
from pathlib import Path
from datetime import datetime
import hashlib

import pandas as pd
import numpy as np
from tqdm import tqdm
from nested_lookup import nested_lookup
from dicompylercore import dicomparser

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dicom_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DicomProcessor:
    """Class to handle DICOM file processing for radiotherapy data."""
    
    def __init__(self, input_dir: str, output_dir: str):
        """Initialize the processor with input and output directories."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.processed_files: Dict[str, Set[str]] = {
            'plans': set(),
            'fractions': set(),
            'structure_sets': set(),
            'dosimetrics': set()
        }
        self.validate_directories()
        
    def validate_directories(self) -> None:
        """Validate and create directories if needed."""
        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    @staticmethod
    def find_files(directory: str, pattern: str) -> Generator[str, None, None]:
        """Find files matching pattern in directory."""
        for root, _, files in os.walk(directory):
            for filename in fnmatch.filter(files, pattern):
                yield os.path.join(root, filename)
    
    @staticmethod
    def safe_lookup(tag: str, dicom_file: dicomparser.DicomParser, index: int = 0) -> str:
        """Safely extract DICOM tag value with enhanced error handling."""
        try:
            return str(nested_lookup(tag, dicom_file.ds.to_json_dict())[index]['Value'][0])
        except (IndexError, KeyError) as e:
            logger.debug(f"Tag {tag} not found: {str(e)}")
            return "NA"
        except Exception as e:
            logger.warning(f"Unexpected error accessing tag {tag}: {str(e)}")
            return "NA"

    def generate_file_hash(self, path: str, key_tags: List[str]) -> str:
        """Generate a unique hash for a DICOM file based on key identifying tags."""
        try:
            dicom_file = dicomparser.DicomParser(path)
            hash_input = []
            
            for tag in key_tags:
                value = self.safe_lookup(tag, dicom_file)
                hash_input.append(str(value))
            
            return hashlib.md5(''.join(hash_input).encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating hash for {path}: {str(e)}")
            return None

    def is_duplicate(self, file_type: str, file_hash: str) -> bool:
        """Check if a file has already been processed based on its hash."""
        return file_hash in self.processed_files[file_type]

    def process_file(self, path: str, tags_dict: Dict[str, str], 
                    file_type: str, key_tags: List[str]) -> Dict[str, str]:
        """Process a single DICOM file and extract specified tags."""
        try:
            # Generate hash based on key identifying tags
            file_hash = self.generate_file_hash(path, key_tags)
            if not file_hash:
                return None
                
            # Check for duplicates
            if self.is_duplicate(file_type, file_hash):
                logger.info(f"Skipping duplicate file: {path}")
                return None
                
            # Process the file
            dicom_file = dicomparser.DicomParser(path)
            result = {"file_path": path}
            
            for key, tag in tags_dict.items():
                result[key] = self.safe_lookup(tag, dicom_file)
            
            # Add the hash to processed files
            self.processed_files[file_type].add(file_hash)
            
            return result
        except Exception as e:
            logger.error(f"Error processing file {path}: {str(e)}")
            return None

    def process_files(self, file_pattern: str, tags_dict: Dict[str, str], 
                     output_name: str, key_tags: List[str]) -> pd.DataFrame:
        """Process all files matching pattern and save results."""
        paths = list(self.find_files(self.input_dir, file_pattern))
        data = []
        
        logger.info(f"Processing {len(paths)} files matching pattern {file_pattern}")
        
        for path in tqdm(paths, desc=f"Processing {file_pattern}"):
            result = self.process_file(path, tags_dict, output_name, key_tags)
            if result:
                data.append(result)
        
        df = pd.DataFrame(data)
        
        if not df.empty:
            # Add processing metadata
            df['processing_timestamp'] = datetime.now().isoformat()
            df['file_pattern'] = file_pattern
            
            # Save to both Excel and CSV
            output_base = self.output_dir / output_name
            df.to_excel(f"{output_base}.xlsx", index=False)
            df.to_csv(f"{output_base}.csv", index=False)
            
            # Log duplicate statistics
            total_files = len(paths)
            unique_files = len(data)
            duplicates = total_files - unique_files
            logger.info(f"Found {duplicates} duplicate files out of {total_files} total files for {output_name}")
        
        return df

    def process_all(self) -> Dict[str, pd.DataFrame]:
        """Process all DICOM file types and return results."""
        # Define tags for each file type
        tags = {
            "plans": {
                "plan_name": "300A0002",
                "plan_date": "300A0006",
                "reference_dose_name": "300A0016",
                "reference_dose": "300A0026",
                "approval": "300E0002",
                "CT_series": "0020000E",
                "CT_study": "0020000D",
                "patient_id": "00100020",
                "patient_dob": "00100030",
                "patient_gender": "00100040",
                "patient_pesel": "00101000"
            },
            "fractions": {
                "fraction_id": "00080018",
                "date": "30080024",
                "time": "30080025",
                "fraction_number": "30080022",
                "verification_status": "3008002C",
                "termination_status": "3008002A",
                "delivery_time": "3008003B",
                "fluence_mode": "30020052",
                "plan_id": "00081155",
                "machine": "300A00B2",
                "patient_id": "00100020"
            },
            "structure_sets": {
                "CT_series": "0020000E",
                "CT_study": "0020000D",
                "approval": "300E0002",
                "patient_id": "00100020"
            },
            "dosimetrics": {
                "CT_series": "0020000E",
                "CT_study": "0020000D",
                "plan_id": "00081155",
                "patient_id": "00100020"
            }
        }
        
        # Define key identifying tags for each file type to detect duplicates
        key_tags = {
            "plans": ["00100020", "300A0002", "300A0006"],  # patient_id, plan_name, plan_date
            "fractions": ["00100020", "00080018", "30080022"],  # patient_id, fraction_id, fraction_number
            "structure_sets": ["00100020", "0020000E", "0020000D"],  # patient_id, CT_series, CT_study
            "dosimetrics": ["00100020", "00081155", "0020000E"]  # patient_id, plan_id, CT_series
        }
        
        # Process each file type
        results = {}
        file_patterns = {
            "plans": "RP*.dcm",
            "fractions": "RT*.dcm",
            "structure_sets": "RS*.dcm",
            "dosimetrics": "RD*.dcm"
        }
        
        for name, pattern in file_patterns.items():
            logger.info(f"Processing {name}")
            results[name] = self.process_files(pattern, tags[name], name, key_tags[name])
            
        return results

def main():
    """Main execution function."""
    try:
        processor = DicomProcessor(
            input_dir = "../DICOM/",
            output_dir = "../Data/"
        )
        
        results = processor.process_all()
        
        # Basic validation and statistics
        for name, df in results.items():
            if not df.empty:
                logger.info(f"\nSummary for {name}:")
                logger.info(f"Total records: {len(df)}")
                logger.info(f"Missing values:\n{df.isna().sum()}")
                logger.info(f"Unique patients: {df['patient_id'].nunique()}")
            else:
                logger.warning(f"No valid records found for {name}")
            
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()

