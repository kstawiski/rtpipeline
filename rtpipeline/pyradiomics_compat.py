#!/usr/bin/env python3
"""
PyRadiomics compatibility wrapper for NumPy 2.x environments.

This module provides a compatibility layer that allows PyRadiomics to work
in environments with NumPy 2.x by creating a subprocess with NumPy 1.x.
"""

import os
import sys
import subprocess
import tempfile
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _create_radiomics_subprocess_script() -> str:
    """Create a Python script that runs PyRadiomics in a subprocess with NumPy 1.x."""
    return '''
import sys
import os
import pickle
import warnings
import tempfile
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")

def run_radiomics_extraction(input_file_path):
    """Run PyRadiomics feature extraction in subprocess."""

    # Load input parameters
    with open(input_file_path, 'rb') as f:
        params = pickle.load(f)

    image_path = params['image_path']
    mask_path = params['mask_path']
    extractor_params = params.get('extractor_params', {})

    try:
        # Import PyRadiomics in subprocess (with NumPy 1.x)
        from radiomics import featureextractor

        # Create extractor with parameters
        extractor = featureextractor.RadiomicsFeatureExtractor(**extractor_params)

        # Extract features
        features = extractor.execute(image_path, mask_path)

        # Convert numpy arrays to regular Python types for pickling
        serializable_features = {}
        for key, value in features.items():
            try:
                if hasattr(value, 'item'):  # numpy scalar
                    serializable_features[key] = value.item()
                elif hasattr(value, 'tolist'):  # numpy array
                    serializable_features[key] = value.tolist()
                else:
                    serializable_features[key] = value
            except Exception:
                serializable_features[key] = str(value)

        return {
            'success': True,
            'features': serializable_features,
            'error': None
        }

    except Exception as e:
        return {
            'success': False,
            'features': None,
            'error': str(e)
        }

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    result = run_radiomics_extraction(input_file)

    # Save result
    with open(output_file, 'wb') as f:
        pickle.dump(result, f)
'''


def extract_radiomics_features_compat(
    image_path: str,
    mask_path: str,
    extractor_params: Optional[Dict[str, Any]] = None,
    numpy1_python: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract PyRadiomics features using a subprocess with NumPy 1.x compatibility.

    Args:
        image_path: Path to the image file
        mask_path: Path to the mask file
        extractor_params: Parameters for the RadiomicsFeatureExtractor
        numpy1_python: Path to Python with NumPy 1.x (optional)

    Returns:
        Dictionary of extracted features

    Raises:
        RuntimeError: If feature extraction fails
    """
    if extractor_params is None:
        extractor_params = {}

    # Try to find a Python environment with NumPy 1.x
    if numpy1_python is None:
        # Try common conda environment locations
        candidates = [
            "python",  # Current environment first
            os.path.expanduser("~/miniconda3/envs/rtpipeline-numpy1/bin/python"),
            os.path.expanduser("~/anaconda3/envs/rtpipeline-numpy1/bin/python"),
            "/opt/conda/envs/rtpipeline-numpy1/bin/python",
        ]

        numpy1_python = None
        for candidate in candidates:
            try:
                # Test if this Python has NumPy < 2.0
                result = subprocess.run([
                    candidate, "-c",
                    "import numpy as np; exit(0 if np.__version__.startswith('1.') else 1)"
                ], capture_output=True, timeout=10)
                if result.returncode == 0:
                    numpy1_python = candidate
                    break
            except Exception:
                continue

    if numpy1_python is None:
        # Fallback: try current Python and hope for the best
        numpy1_python = sys.executable
        logger.warning(
            "Could not find Python with NumPy 1.x, using current Python. "
            "Consider creating a separate environment with NumPy 1.x for PyRadiomics."
        )

    # Create temporary files for communication
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as input_file:
        input_params = {
            'image_path': str(image_path),
            'mask_path': str(mask_path),
            'extractor_params': extractor_params
        }
        pickle.dump(input_params, input_file)
        input_file_path = input_file.name

    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as output_file:
        output_file_path = output_file.name

    # Create subprocess script
    script_content = _create_radiomics_subprocess_script()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
        script_file.write(script_content)
        script_file_path = script_file.name

    try:
        # Run PyRadiomics in subprocess
        logger.info(f"Running PyRadiomics in subprocess with {numpy1_python}")
        result = subprocess.run([
            numpy1_python, script_file_path, input_file_path, output_file_path
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout

        if result.returncode != 0:
            raise RuntimeError(f"PyRadiomics subprocess failed: {result.stderr}")

        # Load results
        with open(output_file_path, 'rb') as f:
            extraction_result = pickle.load(f)

        if not extraction_result['success']:
            raise RuntimeError(f"PyRadiomics extraction failed: {extraction_result['error']}")

        return extraction_result['features']

    finally:
        # Clean up temporary files
        for temp_path in [input_file_path, output_file_path, script_file_path]:
            try:
                os.unlink(temp_path)
            except Exception:
                pass


def create_numpy1_environment(env_name: str = "rtpipeline-numpy1") -> bool:
    """
    Create a conda environment with NumPy 1.x for PyRadiomics compatibility.

    Args:
        env_name: Name of the conda environment to create

    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if conda is available
        conda_cmd = None
        for cmd in ["mamba", "conda"]:
            try:
                subprocess.run([cmd, "--version"], capture_output=True, check=True)
                conda_cmd = cmd
                break
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue

        if conda_cmd is None:
            logger.error("Neither conda nor mamba found")
            return False

        # Create environment with NumPy 1.x and PyRadiomics
        logger.info(f"Creating {env_name} environment with NumPy 1.x...")
        subprocess.run([
            conda_cmd, "create", "-n", env_name, "-y",
            "python=3.11", "numpy=1.26.*", "scipy", "pyradiomics",
            "-c", "conda-forge", "-c", "radiomics"
        ], check=True)

        logger.info(f"✅ Created {env_name} environment successfully")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create NumPy 1.x environment: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error creating environment: {e}")
        return False


# Convenience function that mimics the original PyRadiomics API
def execute_radiomics(image_path: str, mask_path: str, **kwargs) -> Dict[str, Any]:
    """
    Execute PyRadiomics feature extraction with automatic compatibility handling.

    This function attempts to use PyRadiomics directly first, and falls back
    to subprocess execution if there are NumPy compatibility issues.
    """
    try:
        # First try direct execution (works if NumPy 1.x or compatible)
        from radiomics import featureextractor
        extractor = featureextractor.RadiomicsFeatureExtractor(**kwargs)
        return extractor.execute(image_path, mask_path)

    except ImportError as e:
        if "numpy" in str(e).lower():
            logger.info("NumPy compatibility issue detected, using subprocess method")
            return extract_radiomics_features_compat(image_path, mask_path, kwargs)
        else:
            raise
    except Exception as e:
        if "numpy" in str(e).lower() or "dtype" in str(e).lower():
            logger.info("Potential NumPy compatibility issue, trying subprocess method")
            return extract_radiomics_features_compat(image_path, mask_path, kwargs)
        else:
            raise