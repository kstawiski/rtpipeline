#!/usr/bin/env python3
"""
Radiomics module using conda environment isolation.
Runs PyRadiomics in a separate conda environment with NumPy 1.x for compatibility.
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk

logger = logging.getLogger(__name__)

RADIOMICS_ENV = "rtpipeline-radiomics"


def check_radiomics_env() -> bool:
    """Check if the radiomics conda environment exists and is functional."""
    try:
        result = subprocess.run(
            ["conda", "run", "-n", RADIOMICS_ENV, "python", "-c",
             "import radiomics; import numpy; print('OK')"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0 and "OK" in result.stdout
    except Exception as e:
        logger.error(f"Failed to check radiomics environment: {e}")
        return False


def extract_radiomics_with_conda(
    image_path: str,
    mask_path: str,
    params_file: Optional[str] = None,
    label: Optional[int] = None
) -> Dict[str, Any]:
    """
    Extract radiomics features using conda environment.

    Args:
        image_path: Path to image file (NRRD format)
        mask_path: Path to mask file (NRRD format)
        params_file: Optional path to radiomics parameters YAML file
        label: Optional label value for the mask

    Returns:
        Dictionary of extracted features
    """
    # Create extraction script
    extraction_script = '''
import sys
import json
import warnings
import SimpleITK as sitk
from radiomics import featureextractor
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('radiomics').setLevel(logging.ERROR)

# Read parameters from stdin
params = json.loads(sys.stdin.read())

image_path = params['image_path']
mask_path = params['mask_path']
params_file = params.get('params_file')
label = params.get('label')

# Create extractor
if params_file:
    extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
else:
    extractor = featureextractor.RadiomicsFeatureExtractor()

# Execute extraction
if label is not None:
    features = extractor.execute(image_path, mask_path, label=label)
else:
    features = extractor.execute(image_path, mask_path)

# Convert to JSON-serializable format
output = {}
for key, value in features.items():
    try:
        if hasattr(value, 'item'):  # numpy scalar
            output[key] = value.item()
        elif hasattr(value, 'tolist'):  # numpy array
            output[key] = value.tolist()
        else:
            output[key] = value
    except:
        output[key] = str(value)

# Output as JSON
print(json.dumps(output))
'''

    # Prepare input parameters
    input_params = {
        'image_path': image_path,
        'mask_path': mask_path,
        'params_file': params_file,
        'label': label
    }

    try:
        # Write parameters to temporary file to avoid stdin issues
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as param_file:
            json.dump(input_params, param_file)
            param_file_path = param_file.name

        # Modified extraction script to read from file
        extraction_script_with_file = f'''
import sys
import json
import warnings
import SimpleITK as sitk
from radiomics import featureextractor
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('radiomics').setLevel(logging.ERROR)

# Read parameters from file
with open('{param_file_path}', 'r') as f:
    params = json.load(f)

image_path = params['image_path']
mask_path = params['mask_path']
params_file = params.get('params_file')
label = params.get('label')

# Create extractor
if params_file:
    extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
else:
    extractor = featureextractor.RadiomicsFeatureExtractor()

# Execute extraction
if label is not None:
    features = extractor.execute(image_path, mask_path, label=label)
else:
    features = extractor.execute(image_path, mask_path)

# Convert to JSON-serializable format
output = {{}}
for key, value in features.items():
    try:
        if hasattr(value, 'item'):  # numpy scalar
            output[key] = value.item()
        elif hasattr(value, 'tolist'):  # numpy array
            output[key] = value.tolist()
        else:
            output[key] = value
    except:
        output[key] = str(value)

# Output as JSON
print(json.dumps(output))
'''

        # Run extraction in conda environment
        result = subprocess.run(
            ["conda", "run", "-n", RADIOMICS_ENV, "python", "-c", extraction_script_with_file],
            capture_output=True,
            text=True,
            timeout=900  # allow up to 15 minutes for large ROIs
        )

        if result.returncode != 0:
            raise RuntimeError(f"Radiomics extraction failed: {result.stderr}")

        # Parse and return features
        features = json.loads(result.stdout)

        # Clean up temporary parameter file
        try:
            os.unlink(param_file_path)
        except:
            pass

        return features

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse radiomics output: {e}")
        logger.error(f"stdout: {result.stdout[:500]}")
        # Clean up temporary parameter file
        try:
            os.unlink(param_file_path)
        except:
            pass
        raise
    except subprocess.TimeoutExpired:
        # Clean up temporary parameter file
        try:
            os.unlink(param_file_path)
        except:
            pass
        raise RuntimeError("Radiomics extraction timed out")
    except Exception as e:
        # Clean up temporary parameter file
        try:
            os.unlink(param_file_path)
        except:
            pass
        raise RuntimeError(f"Radiomics extraction failed: {e}")


def process_radiomics_batch(
    tasks: List[Tuple[str, str, str, Optional[str], Optional[int]]],
    output_path: Path,
    sequential: bool = False,
    max_workers: Optional[int] = None,
) -> Optional[Path]:
    """
    Process a batch of radiomics tasks.

    Args:
        tasks: List of (image_path, mask_path, roi_name, params_file, label) tuples
        output_path: Path to save the radiomics Excel file
        sequential: If True, process sequentially instead of in parallel
        max_workers: Optional cap on parallel worker processes

    Returns:
        Path to the output file if successful, None otherwise
    """
    if not tasks:
        logger.warning("No radiomics tasks to process")
        return None

    if not check_radiomics_env():
        logger.error(f"Radiomics conda environment '{RADIOMICS_ENV}' not found or not functional")
        logger.error(f"Please run: conda create -n {RADIOMICS_ENV} python=3.11 numpy=1.26.* pyradiomics SimpleITK -c conda-forge")
        return None

    results = []

    if sequential:
        logger.info(f"Processing {len(tasks)} radiomics tasks sequentially")
        for i, (image_path, mask_path, roi_name, params_file, label) in enumerate(tasks, 1):
            logger.info(f"Processing {i}/{len(tasks)}: {roi_name}")
            try:
                features = extract_radiomics_with_conda(
                    image_path, mask_path, params_file, label
                )
                features['roi_name'] = roi_name
                results.append(features)
            except Exception as e:
                logger.error(f"Failed to extract features for {roi_name}: {e}")
    else:
        # Use multiprocessing with proper error handling
        from concurrent.futures import ProcessPoolExecutor, as_completed

        logger.info(f"Processing {len(tasks)} radiomics tasks in parallel")

        worker_limit = max_workers if max_workers and max_workers > 0 else (os.cpu_count() or 4)
        with ProcessPoolExecutor(max_workers=min(len(tasks), worker_limit)) as executor:
            # Submit all tasks
            future_to_roi = {}
            for image_path, mask_path, roi_name, params_file, label in tasks:
                future = executor.submit(
                    extract_radiomics_with_conda,
                    image_path, mask_path, params_file, label
                )
                future_to_roi[future] = roi_name

            # Collect results as they complete
            for future in as_completed(future_to_roi):
                roi_name = future_to_roi[future]
                try:
                    features = future.result(timeout=900)
                    features['roi_name'] = roi_name
                    results.append(features)
                    logger.debug(f"Completed radiomics for {roi_name}")
                except Exception as e:
                    logger.error(f"Failed to extract features for {roi_name}: {e}")

    if not results:
        logger.warning("No radiomics features extracted")
        return None

    # Convert to DataFrame and save
    try:
        df = pd.DataFrame(results)
        df.to_excel(output_path, index=False)
        logger.info(f"Saved {len(results)} radiomics results to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to save radiomics results: {e}")
        return None


def radiomics_for_course(
    course_dir: Path,
    config: Any,
    custom_structures_config: Optional[str] = None
) -> Optional[Path]:
    """
    Extract radiomics features for a course using conda environment.

    Args:
        course_dir: Path to the course directory
        config: Pipeline configuration
        custom_structures_config: Optional path to custom structures config

    Returns:
        Path to the radiomics Excel file if successful, None otherwise
    """
    # Check for CT DICOM files
    ct_dir = course_dir / "CT_DICOM"
    if not ct_dir.exists():
        logger.warning(f"No CT_DICOM directory found in {course_dir}")
        return None

    # Check for segmentation files
    rs_auto = course_dir / "RS_auto.dcm"
    rs_custom = course_dir / "RS_custom.dcm"

    if not rs_auto.exists() and not rs_custom.exists():
        logger.warning(f"No segmentation files found in {course_dir}")
        return None

    # Use the available segmentation
    rs_file = rs_custom if rs_custom.exists() else rs_auto

    # Load CT image
    try:
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(str(ct_dir))
        reader.SetFileNames(dicom_files)
        ct_image = reader.Execute()
    except Exception as e:
        logger.error(f"Failed to load CT image: {e}")
        return None

    # Save CT image to temporary NRRD file
    with tempfile.NamedTemporaryFile(suffix='.nrrd', delete=False) as ct_file:
        sitk.WriteImage(ct_image, ct_file.name, useCompression=True)
        ct_image_path = ct_file.name

    # Extract ROIs and create tasks
    tasks = []
    try:
        # Load RTSTRUCT
        from rt_utils import RTStructBuilder
        rtstruct = RTStructBuilder.create_from(
            dicom_series_path=str(ct_dir),
            rt_struct_path=str(rs_file)
        )

        # Get all ROI names
        roi_names = rtstruct.get_roi_names()

        skip_rois = {"body", "couchsurface", "couchinterior"}

        for roi_name in roi_names:
            normalized = roi_name.replace(" ", "").lower()
            if normalized in skip_rois:
                logger.debug(f"Skipping radiomics for ROI {roi_name} (heuristic filter)")
                continue
            try:
                # Get mask for ROI
                mask = rtstruct.get_roi_mask_by_name(roi_name)
                if mask is None or not mask.any():
                    logger.debug(f"Skipping radiomics for ROI {roi_name}: mask empty")
                    continue

                # Convert to SimpleITK image
                mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8).transpose(2, 0, 1))
                mask_sitk.CopyInformation(ct_image)

                # Save mask to temporary NRRD file
                with tempfile.NamedTemporaryFile(suffix='.nrrd', delete=False) as mask_file:
                    sitk.WriteImage(mask_sitk, mask_file.name, useCompression=True)
                    mask_path = mask_file.name

                # Add task
                params_file = str(config.radiomics_params_file) if config.radiomics_params_file else None
                tasks.append((ct_image_path, mask_path, roi_name, params_file, None))

            except Exception as e:
                logger.debug(f"Could not process ROI {roi_name}: {e}")

    except Exception as e:
        logger.error(f"Failed to process RTSTRUCT: {e}")
        return None

    if not tasks:
        logger.warning(f"No valid ROIs found in {rs_file}")
        return None

    # Process radiomics
    output_path = course_dir / "Radiomics_CT.xlsx"
    sequential = os.environ.get('RTPIPELINE_RADIOMICS_SEQUENTIAL', '').lower() in ('1', 'true', 'yes')

    max_workers = None
    if getattr(config, "workers", None):
        try:
            max_workers = int(config.workers)
        except Exception:
            max_workers = None

    result = process_radiomics_batch(
        tasks,
        output_path,
        sequential=sequential,
        max_workers=max_workers,
    )

    # Clean up temporary files
    try:
        os.unlink(ct_image_path)
        for _, mask_path, _, _, _ in tasks:
            try:
                os.unlink(mask_path)
            except:
                pass
    except:
        pass

    return result
