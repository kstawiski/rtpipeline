"""
DICOM Validation Module
Validates uploaded files and provides suggestions for conversion
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any
import pydicom
from pydicom.errors import InvalidDicomError

logger = logging.getLogger(__name__)


class DICOMValidator:
    """Validates DICOM files and directory structure"""

    def __init__(self):
        self.valid_modalities = {'CT', 'MR', 'RTPLAN', 'RTDOSE', 'RTSTRUCT', 'REG', 'PT', 'RTIMAGE'}
        self.rt_modalities = {'RTPLAN', 'RTDOSE', 'RTSTRUCT'}

    def validate_directory(self, directory: str) -> Dict[str, Any]:
        """
        Validate a directory containing DICOM files

        Args:
            directory: Path to directory containing DICOM files

        Returns:
            Dictionary with validation results and suggestions
        """
        try:
            dir_path = Path(directory)
            if not dir_path.exists():
                return {
                    'valid': False,
                    'error': 'Directory does not exist',
                    'dicom_count': 0
                }

            # Find all potential DICOM files
            dicom_files = self._find_dicom_files(dir_path)

            if not dicom_files:
                return {
                    'valid': False,
                    'error': 'No DICOM files found',
                    'dicom_count': 0,
                    'suggestions': self._get_conversion_suggestions(dir_path)
                }

            # Validate each DICOM file
            validation_results = {
                'valid': True,
                'dicom_count': len(dicom_files),
                'files': [],
                'modalities': {},
                'patients': set(),
                'studies': set(),
                'series': set(),
                'has_ct': False,
                'has_mr': False,
                'has_rt_plan': False,
                'has_rt_dose': False,
                'has_rt_struct': False,
                'warnings': [],
                'errors': [],
                'suggestions': []
            }

            for file_path in dicom_files:
                file_result = self._validate_dicom_file(file_path)
                if file_result:
                    validation_results['files'].append(file_result)

                    # Track modalities
                    modality = file_result.get('modality', 'UNKNOWN')
                    validation_results['modalities'][modality] = \
                        validation_results['modalities'].get(modality, 0) + 1

                    # Track unique identifiers
                    if 'patient_id' in file_result:
                        validation_results['patients'].add(file_result['patient_id'])
                    if 'study_uid' in file_result:
                        validation_results['studies'].add(file_result['study_uid'])
                    if 'series_uid' in file_result:
                        validation_results['series'].add(file_result['series_uid'])

                    # Check for specific modalities
                    if modality == 'CT':
                        validation_results['has_ct'] = True
                    elif modality == 'MR':
                        validation_results['has_mr'] = True
                    elif modality == 'RTPLAN':
                        validation_results['has_rt_plan'] = True
                    elif modality == 'RTDOSE':
                        validation_results['has_rt_dose'] = True
                    elif modality == 'RTSTRUCT':
                        validation_results['has_rt_struct'] = True

            # Convert sets to counts for JSON serialization
            validation_results['patient_count'] = len(validation_results['patients'])
            validation_results['study_count'] = len(validation_results['studies'])
            validation_results['series_count'] = len(validation_results['series'])
            validation_results['patients'] = list(validation_results['patients'])
            validation_results['studies'] = list(validation_results['studies'])
            validation_results['series'] = list(validation_results['series'])

            # Generate warnings and suggestions
            self._generate_warnings_and_suggestions(validation_results)

            logger.info(f"Validated {len(dicom_files)} DICOM files in {directory}")
            logger.info(f"Found modalities: {validation_results['modalities']}")

            return validation_results

        except Exception as e:
            logger.error(f"Validation error: {str(e)}", exc_info=True)
            return {
                'valid': False,
                'error': str(e),
                'dicom_count': 0
            }

    def _find_dicom_files(self, directory: Path) -> List[Path]:
        """
        Recursively find all DICOM files in directory

        Args:
            directory: Path to search

        Returns:
            List of paths to DICOM files
        """
        dicom_files = []

        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file

                # Skip hidden files and common non-DICOM files
                if file.startswith('.'):
                    continue

                # Check for DICOM files by extension or content
                if self._is_dicom_file(file_path):
                    dicom_files.append(file_path)

        return dicom_files

    def _is_dicom_file(self, file_path: Path) -> bool:
        """
        Check if a file is a valid DICOM file

        Args:
            file_path: Path to file

        Returns:
            True if file is a valid DICOM file
        """
        # Check by extension
        if file_path.suffix.lower() in ['.dcm', '.dicom']:
            return True

        # Check by name patterns
        if file_path.name.upper() == 'DICOMDIR':
            return True

        # Check by reading file header
        try:
            # Try to read as DICOM
            logger.debug(f"Using force=True for DICOM validation on {file_path.name} - bypassing strict checks for compatibility")
            pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
            return True
        except (InvalidDicomError, Exception):
            return False

    def _validate_dicom_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Validate a single DICOM file and extract metadata

        Args:
            file_path: Path to DICOM file

        Returns:
            Dictionary with file information
        """
        try:
            ds = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)

            file_info = {
                'path': str(file_path),
                'filename': file_path.name,
                'size': file_path.stat().st_size,
                'valid': True
            }

            # Extract key DICOM tags
            if hasattr(ds, 'Modality'):
                file_info['modality'] = ds.Modality

            if hasattr(ds, 'PatientID'):
                file_info['patient_id'] = str(ds.PatientID)

            if hasattr(ds, 'PatientName'):
                file_info['patient_name'] = str(ds.PatientName)

            if hasattr(ds, 'StudyInstanceUID'):
                file_info['study_uid'] = ds.StudyInstanceUID

            if hasattr(ds, 'SeriesInstanceUID'):
                file_info['series_uid'] = ds.SeriesInstanceUID

            if hasattr(ds, 'SOPInstanceUID'):
                file_info['sop_uid'] = ds.SOPInstanceUID

            if hasattr(ds, 'StudyDescription'):
                file_info['study_description'] = str(ds.StudyDescription)

            if hasattr(ds, 'SeriesDescription'):
                file_info['series_description'] = str(ds.SeriesDescription)

            if hasattr(ds, 'InstanceNumber'):
                file_info['instance_number'] = int(ds.InstanceNumber)

            return file_info

        except Exception as e:
            logger.warning(f"Failed to validate DICOM file {file_path}: {str(e)}")
            return {
                'path': str(file_path),
                'filename': file_path.name,
                'valid': False,
                'error': str(e)
            }

    def _generate_warnings_and_suggestions(self, results: Dict[str, Any]):
        """
        Generate warnings and suggestions based on validation results

        Args:
            results: Validation results dictionary (modified in place)
        """
        warnings = []
        suggestions = []

        # Check if we have imaging data
        if not results['has_ct'] and not results['has_mr']:
            warnings.append('No CT or MR imaging data found')
            suggestions.append('rtpipeline requires CT or MR images for processing')

        # Check for RT data
        if results['has_ct'] or results['has_mr']:
            if not results['has_rt_dose'] and not results['has_rt_struct']:
                warnings.append('No RT DOSE or RT STRUCT found')
                suggestions.append(
                    'For DVH analysis, you need RTDOSE and RTSTRUCT files. '
                    'Segmentation and radiomics can run without RT data.'
                )

        # Check for multiple patients
        if results['patient_count'] > 1:
            warnings.append(f'Multiple patients found ({results["patient_count"]})')
            suggestions.append(
                'The pipeline will process all patients, but results will be organized separately'
            )

        # Check for RTSTRUCT without RTDOSE
        if results['has_rt_struct'] and not results['has_rt_dose']:
            warnings.append('RTSTRUCT found but no RTDOSE')
            suggestions.append('DVH metrics cannot be computed without RTDOSE data')

        # Check for RTDOSE without RTSTRUCT
        if results['has_rt_dose'] and not results['has_rt_struct']:
            warnings.append('RTDOSE found but no RTSTRUCT')
            suggestions.append(
                'No manual structures found. Pipeline can generate automatic segmentations '
                'using TotalSegmentator.'
            )

        results['warnings'] = warnings
        results['suggestions'] = suggestions

    def _get_conversion_suggestions(self, directory: Path) -> List[str]:
        """
        Suggest conversions for non-DICOM files

        Args:
            directory: Directory path

        Returns:
            List of conversion suggestions
        """
        suggestions = []

        # Check for common medical imaging formats
        nifti_files = list(directory.glob('**/*.nii')) + list(directory.glob('**/*.nii.gz'))
        if nifti_files:
            suggestions.append(
                'NIfTI files found. Convert to DICOM using tools like dcm2niix or ITK-SNAP'
            )

        # Check for NRRD files
        nrrd_files = list(directory.glob('**/*.nrrd'))
        if nrrd_files:
            suggestions.append(
                'NRRD files found. Convert to DICOM or NIfTI using 3D Slicer or ITK'
            )

        # Check for common image formats
        image_files = (
            list(directory.glob('**/*.png')) +
            list(directory.glob('**/*.jpg')) +
            list(directory.glob('**/*.jpeg'))
        )
        if image_files:
            suggestions.append(
                'Standard image files found. These cannot be directly processed. '
                'Medical images must be in DICOM format.'
            )

        if not suggestions:
            suggestions.append(
                'No recognizable medical imaging files found. '
                'Please ensure you have DICOM files (CT, MR, RTPLAN, RTDOSE, RTSTRUCT).'
            )

        return suggestions
