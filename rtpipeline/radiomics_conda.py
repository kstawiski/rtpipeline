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
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # For future type hints

import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom

from .layout import build_course_dirs
from .utils import mask_is_cropped

logger = logging.getLogger(__name__)

RADIOMICS_ENV = "rtpipeline-radiomics"


# Thread-limiting environment variables for subprocesses
_THREAD_ENV_VARS = (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'NUMBA_NUM_THREADS',
)


def _conda_subprocess_env() -> Dict[str, str]:
    """Create environment for conda subprocess with thread limits.

    CRITICAL: Each subprocess must have thread limits set to prevent
    CPU oversubscription. Without this, N workers × M threads each
    can spawn 100+ threads fighting for CPU cores.
    """
    env = os.environ.copy()
    env.setdefault("CONDA_NO_PLUGINS", "1")
    env.setdefault("CONDA_OVERRIDE_CUDA", "0")

    # Get thread limit from environment or default to 1
    # Using 1 thread per subprocess is optimal when running many parallel subprocesses
    thread_limit = os.environ.get("RTPIPELINE_RADIOMICS_THREAD_LIMIT", "1")
    try:
        thread_limit = str(max(1, int(thread_limit)))
    except (ValueError, TypeError):
        thread_limit = "1"

    # Set thread limits for all common libraries
    for var in _THREAD_ENV_VARS:
        env.setdefault(var, thread_limit)

    return env


def check_radiomics_env() -> bool:
    """Check if the radiomics conda environment exists and is functional."""
    try:
        result = subprocess.run(
            [
                "conda",
                "run",
                "-n",
                RADIOMICS_ENV,
                "python",
                "-c",
                "import radiomics; import numpy; print('OK')",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            env=_conda_subprocess_env(),
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
    except Exception:
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
    except Exception:
        output[key] = str(value)

# Output as JSON
print(json.dumps(output))
'''

        # Run extraction in conda environment
        result = subprocess.run(
            ["conda", "run", "-n", RADIOMICS_ENV, "python", "-c", extraction_script_with_file],
            capture_output=True,
            text=True,
            timeout=900,  # allow up to 15 minutes for large ROIs
            env=_conda_subprocess_env(),
        )

        if result.returncode != 0:
            raise RuntimeError(f"Radiomics extraction failed: {result.stderr}")

        # Parse and return features
        features = json.loads(result.stdout)

        # Clean up temporary parameter file
        try:
            os.unlink(param_file_path)
        except OSError:
            # Ignore errors when deleting the temporary file; not critical if removal fails.
            pass

        return features

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse radiomics output: {e}")
        logger.error(f"stdout: {result.stdout[:500]}")
        # Clean up temporary parameter file
        try:
            os.unlink(param_file_path)
        except OSError:
            # Ignore errors when deleting the temporary file; not critical if removal fails.
            pass
        raise
    except subprocess.TimeoutExpired:
        # Clean up temporary parameter file
        try:
            os.unlink(param_file_path)
        except OSError:
            # Ignore errors when deleting the temporary file; not critical if removal fails.
            pass
        raise RuntimeError("Radiomics extraction timed out")
    except Exception as e:
        # Clean up temporary parameter file
        try:
            os.unlink(param_file_path)
        except OSError:
            # Ignore errors when deleting the temporary file; not critical if removal fails.
            pass
        raise RuntimeError(f"Radiomics extraction failed: {e}")


def extract_radiomics_batch_with_conda(
    tasks: List[Dict[str, Any]],
    params_file: Optional[str] = None,
    timeout_per_roi: int = 120,
) -> List[Dict[str, Any]]:
    """
    Extract radiomics features for multiple ROIs in a SINGLE subprocess.

    This dramatically reduces overhead by loading the radiomics library once
    and processing all ROIs sequentially within that subprocess.

    Args:
        tasks: List of dicts with 'image_path', 'mask_path', optional 'label', 'roi_name'
        params_file: Optional path to radiomics parameters YAML file
        timeout_per_roi: Timeout per ROI in seconds (default 120s = 2 min)

    Returns:
        List of feature dictionaries (one per task), None entries for failed ROIs
    """
    if not tasks:
        return []

    # Calculate total timeout based on number of tasks
    total_timeout = max(300, len(tasks) * timeout_per_roi)  # At least 5 minutes

    # Create batch extraction script that processes all ROIs in one go
    batch_script = '''
import sys
import json
import warnings
import logging

# Suppress warnings before importing radiomics
warnings.filterwarnings('ignore')
logging.getLogger('radiomics').setLevel(logging.ERROR)
logging.getLogger('radiomics.featureextractor').setLevel(logging.ERROR)

from radiomics import featureextractor

# Read batch parameters from file
with open(sys.argv[1], 'r') as f:
    batch_params = json.load(f)

tasks = batch_params['tasks']
params_file = batch_params.get('params_file')

# Create extractor ONCE (this is the expensive operation we're amortizing)
if params_file:
    extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
else:
    extractor = featureextractor.RadiomicsFeatureExtractor()

# Process each task
for task in tasks:
    image_path = task['image_path']
    mask_path = task['mask_path']
    label = task.get('label')
    roi_name = task.get('roi_name', 'ROI')

    try:
        if label is not None:
            features = extractor.execute(image_path, mask_path, label=label)
        else:
            features = extractor.execute(image_path, mask_path)

        # Convert to JSON-serializable format
        output = {'__status__': 'success', '__roi_name__': roi_name}
        for key, value in features.items():
            try:
                if hasattr(value, 'item'):
                    output[key] = value.item()
                elif hasattr(value, 'tolist'):
                    output[key] = value.tolist()
                else:
                    output[key] = value
            except Exception:
                output[key] = str(value)
        print(json.dumps(output), flush=True)

    except Exception as e:
        error_msg = str(e).lower()
        if 'size of the roi is too small' in error_msg:
            print(json.dumps({'__status__': 'skipped', '__roi_name__': roi_name, '__reason__': 'ROI too small'}), flush=True)
        else:
            print(json.dumps({'__status__': 'error', '__roi_name__': roi_name, '__error__': str(e)}), flush=True)
'''

    try:
        # Write batch parameters to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as batch_file:
            json.dump({
                'tasks': [
                    {
                        'image_path': t.get('image_path'),
                        'mask_path': t.get('mask_path'),
                        'label': t.get('label'),
                        'roi_name': t.get('roi_name', 'ROI'),
                    }
                    for t in tasks
                ],
                'params_file': params_file,
            }, batch_file)
            batch_file_path = batch_file.name

        # Run batch extraction in conda environment
        result = subprocess.run(
            ["conda", "run", "-n", RADIOMICS_ENV, "python", "-c", batch_script, batch_file_path],
            capture_output=True,
            text=True,
            timeout=total_timeout,
            env=_conda_subprocess_env(),
        )

        # Parse results - one JSON per line
        results = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                results.append(parsed)
            except json.JSONDecodeError:
                logger.debug("Failed to parse batch output line: %s", line[:100])

        # Clean up
        try:
            os.unlink(batch_file_path)
        except OSError:
            pass

        if result.returncode != 0 and not results:
            logger.error("Batch radiomics failed: %s", result.stderr[:500] if result.stderr else "unknown error")
            return [None] * len(tasks)

        return results

    except subprocess.TimeoutExpired:
        logger.error("Batch radiomics timed out after %ds for %d tasks", total_timeout, len(tasks))
        try:
            os.unlink(batch_file_path)
        except (OSError, NameError):
            pass
        return [None] * len(tasks)

    except Exception as e:
        logger.error("Batch radiomics failed: %s", e)
        try:
            os.unlink(batch_file_path)
        except (OSError, NameError):
            pass
        return [None] * len(tasks)


def _write_mask_to_file(mask_array: np.ndarray, mask_path: str, ct_info: Dict[str, Any]) -> None:
    """Write a binary mask with CT geometry metadata."""

    arr = np.ascontiguousarray(mask_array.astype(np.uint8).transpose(2, 0, 1))
    img = sitk.GetImageFromArray(arr, isVector=False)

    spacing = tuple(ct_info.get('spacing', (1.0, 1.0, 1.0)))
    if len(spacing) < 3:
        spacing = tuple(list(spacing) + [1.0] * (3 - len(spacing)))
    img.SetSpacing(spacing)

    origin = ct_info.get('origin')
    if origin is not None:
        img.SetOrigin(tuple(origin))

    direction = ct_info.get('direction')
    if direction is not None:
        img.SetDirection(tuple(direction))

    sitk.WriteImage(img, mask_path, useCompression=True)


def _ensure_mask_has_three_dimensions(mask_path: str, ct_info: Dict[str, Any]) -> bool:
    """Ensure mask stored at ``mask_path`` has three dimensions.

    Returns True when the mask was rewritten.
    """

    try:
        mask_img = sitk.ReadImage(mask_path)
    except Exception as exc:
        logger.debug("Failed to read mask %s for dimension fix: %s", mask_path, exc)
        return False

    if mask_img.GetDimension() >= 3:
        return False

    arr2d = sitk.GetArrayFromImage(mask_img)
    arr3d = np.ascontiguousarray(np.expand_dims(arr2d, axis=0).astype(np.uint8))
    img3d = sitk.GetImageFromArray(arr3d, isVector=False)

    spacing = tuple(ct_info.get('spacing', (1.0, 1.0, 1.0)))
    if len(spacing) < 3:
        spacing = tuple(list(spacing) + [1.0] * (3 - len(spacing)))
    img3d.SetSpacing(spacing)

    origin = ct_info.get('origin')
    if origin is not None:
        img3d.SetOrigin(tuple(origin))

    direction = ct_info.get('direction')
    if direction is not None:
        img3d.SetDirection(tuple(direction))

    sitk.WriteImage(img3d, mask_path, useCompression=True)
    return True


def _combine_feature_record(features: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    record: Dict[str, Any] = {}
    record.update(features)
    record.update(metadata)
    return record


def process_radiomics_batch(
    tasks: List[Dict[str, Any]],
    output_path: Path,
    sequential: bool = False,
    max_workers: Optional[int] = None,
) -> Optional[Path]:
    """Process radiomics extraction tasks and persist them as an Excel sheet."""

    if not tasks:
        logger.warning("No radiomics tasks to process")
        return None

    if not check_radiomics_env():
        logger.error(
            "Radiomics conda environment '%s' not found or not functional",
            RADIOMICS_ENV,
        )
        logger.error(
            "Please run: conda create -n %s python=3.11 numpy=1.26.* pyradiomics SimpleITK -c conda-forge",
            RADIOMICS_ENV,
        )
        return None

    cleanup_paths = {
        task.get('mask_path')
        for task in tasks
        if task.get('mask_path') and task.get('cleanup', True)
    }

    def _execute(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        roi_name = task.get('roi_name', 'ROI')
        image_path = task.get('image_path')
        mask_path = task.get('mask_path')
        params_file = task.get('params_file')
        label = task.get('label')
        metadata = dict(task.get('metadata', {}))
        metadata.setdefault('roi_name', roi_name)
        metadata.setdefault('roi_original_name', roi_name)

        if not image_path or not mask_path:
            logger.error("Radiomics task is missing required paths for %s", roi_name)
            return None

        try:
            features = extract_radiomics_with_conda(image_path, mask_path, params_file, label)
        except Exception as exc:
            msg = str(exc).lower()
            if 'size of the roi is too small' in msg:
                logger.info(
                    "Skipping radiomics for ROI %s: %s",
                    roi_name,
                    str(exc).strip(),
                )
                return None
            if 'mask has too few dimensions' in msg and task.get('ct_info'):
                repaired = _ensure_mask_has_three_dimensions(mask_path, task['ct_info'])
                if repaired:
                    logger.debug("Rewrote 2D mask for ROI %s", roi_name)
                    try:
                        features = extract_radiomics_with_conda(image_path, mask_path, params_file, label)
                    except Exception as inner_exc:
                        logger.error("Radiomics retry failed for %s: %s", roi_name, inner_exc)
                        return None
                else:
                    logger.error("Unable to repair mask dimensionality for %s", roi_name)
                    return None
            else:
                logger.error("Failed to extract features for %s: %s", roi_name, exc)
                return None

        metadata.setdefault('modality', 'CT')
        return _combine_feature_record(features, metadata)

    results: List[Dict[str, Any]] = []

    def _run_sequential(seq: List[Dict[str, Any]]) -> None:
        for idx, task in enumerate(seq, 1):
            roi_name = task.get('roi_name', 'ROI')
            logger.info("Processing %d/%d: %s", idx, len(seq), roi_name)
            rec = _execute(task)
            if rec:
                results.append(rec)

    tasks_list = list(tasks)
    if max_workers and max_workers > 0:
        worker_limit = max_workers
    else:
        cpu_total = os.cpu_count() or 2
        worker_limit = max(1, cpu_total - 1)
    worker_limit = max(1, min(worker_limit, len(tasks_list)))

    # Use batch processing to reduce subprocess overhead
    # Instead of N subprocesses (each loading radiomics library),
    # we use N/batch_size subprocesses, dramatically reducing startup overhead
    use_batch_processing = os.environ.get('RTPIPELINE_RADIOMICS_BATCH', '1').lower() in ('1', 'true', 'yes')

    if sequential or len(tasks_list) == 1 or worker_limit == 1:
        if use_batch_processing:
            # Even sequential mode benefits from batch processing
            logger.info("Processing %d radiomics tasks in batch mode (sequential)", len(tasks_list))
            params_file = tasks_list[0].get('params_file') if tasks_list else None
            batch_results = extract_radiomics_batch_with_conda(tasks_list, params_file)

            for task, features in zip(tasks_list, batch_results):
                if features is None or features.get('__status__') != 'success':
                    status = features.get('__status__', 'failed') if features else 'failed'
                    roi_name = task.get('roi_name', 'ROI')
                    if status == 'skipped':
                        logger.info("Skipped radiomics for %s: %s", roi_name, features.get('__reason__', 'unknown'))
                    elif status == 'error':
                        logger.error("Radiomics failed for %s: %s", roi_name, features.get('__error__', 'unknown'))
                    continue

                # Remove internal status fields and combine with metadata
                features_clean = {k: v for k, v in features.items() if not k.startswith('__')}
                metadata = dict(task.get('metadata') or task.get('extra_metadata') or {})
                metadata.setdefault('roi_name', task.get('roi_name', 'ROI'))
                metadata.setdefault('roi_original_name', task.get('roi_name', 'ROI'))
                metadata.setdefault('modality', 'CT')
                results.append(_combine_feature_record(features_clean, metadata))
        else:
            _run_sequential(tasks_list)
    elif use_batch_processing:
        # OPTIMIZED: Split tasks into batches and process each batch in a single subprocess
        # This amortizes the ~2-5 second subprocess startup across multiple ROIs
        batch_size = max(5, len(tasks_list) // worker_limit)  # At least 5 ROIs per batch
        batches = [tasks_list[i:i + batch_size] for i in range(0, len(tasks_list), batch_size)]

        logger.info(
            "Processing %d radiomics tasks in %d batches (%d ROIs/batch) with %d workers",
            len(tasks_list),
            len(batches),
            batch_size,
            min(worker_limit, len(batches)),
        )

        from concurrent.futures import ProcessPoolExecutor, as_completed

        def _process_batch(batch: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
            """Process a batch of tasks and return (task, features) pairs."""
            params_file = batch[0].get('params_file') if batch else None
            batch_results = extract_radiomics_batch_with_conda(batch, params_file)
            return list(zip(batch, batch_results))

        # Use ProcessPoolExecutor for better parallelism (avoids GIL)
        # Each worker processes one batch (multiple ROIs) per subprocess
        with ProcessPoolExecutor(max_workers=min(worker_limit, len(batches))) as executor:
            future_to_batch = {executor.submit(_process_batch, batch): batch for batch in batches}

            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_results = future.result()
                    for task, features in batch_results:
                        if features is None or features.get('__status__') != 'success':
                            status = features.get('__status__', 'failed') if features else 'failed'
                            roi_name = task.get('roi_name', 'ROI')
                            if status == 'skipped':
                                logger.debug("Skipped radiomics for %s: %s", roi_name, features.get('__reason__', 'unknown'))
                            elif status == 'error':
                                logger.warning("Radiomics failed for %s: %s", roi_name, features.get('__error__', 'unknown'))
                            continue

                        # Remove internal status fields and combine with metadata
                        features_clean = {k: v for k, v in features.items() if not k.startswith('__')}
                        metadata = dict(task.get('metadata') or task.get('extra_metadata') or {})
                        metadata.setdefault('roi_name', task.get('roi_name', 'ROI'))
                        metadata.setdefault('roi_original_name', task.get('roi_name', 'ROI'))
                        metadata.setdefault('modality', 'CT')
                        results.append(_combine_feature_record(features_clean, metadata))
                        logger.debug("Completed radiomics for %s", task.get('roi_name', 'ROI'))
                except Exception as exc:
                    logger.error("Batch processing failed: %s", exc)
    else:
        # Legacy per-ROI processing (can be enabled via RTPIPELINE_RADIOMICS_BATCH=0)
        logger.info(
            "Processing %d radiomics tasks with up to %d worker threads (legacy mode)",
            len(tasks_list),
            worker_limit,
        )
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=worker_limit) as executor:
            future_to_task = {executor.submit(_execute, task): task for task in tasks_list}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                roi_name = task.get('roi_name', 'ROI')
                try:
                    rec = future.result()
                    if rec:
                        results.append(rec)
                        logger.debug("Completed radiomics for %s", roi_name)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error("Radiomics task crashed for %s: %s", roi_name, exc)

    if not results:
        logger.warning("No radiomics features extracted")
        for mask_path in cleanup_paths:
            try:
                if mask_path:
                    os.unlink(mask_path)
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.debug("Cleanup failed for %s: %s", mask_path, exc)
        return None

    try:
        df = pd.DataFrame(results)
        df.to_excel(output_path, index=False)
        logger.info("Saved %d radiomics rows to %s", len(df), output_path)
    except Exception as exc:
        logger.error("Failed to save radiomics results: %s", exc)
        return None
    finally:
        for mask_path in cleanup_paths:
            try:
                if mask_path:
                    os.unlink(mask_path)
            except FileNotFoundError:
                continue
            except Exception as exc:
                logger.debug("Cleanup failed for %s: %s", mask_path, exc)

    return output_path


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
    course_dirs = build_course_dirs(course_dir)

    # Check for CT DICOM files
    ct_dir = course_dirs.dicom_ct
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

    ct_info = {
        'spacing': tuple(ct_image.GetSpacing()),
        'origin': tuple(ct_image.GetOrigin()),
        'direction': tuple(ct_image.GetDirection()),
    }

    # Map ROI names to their originating source if structure mapping is available
    source_map: Dict[str, str] = {}
    mapping_path = course_dir / "metadata" / "structure_mapping.json"
    if mapping_path.exists():
        try:
            mapping_data = json.loads(mapping_path.read_text(encoding='utf-8'))
            source_config = mapping_data.get('sources', {}) or {}
            source_labels = {
                'manual': 'Manual',
                'auto': 'AutoTS',
                'custom_config': 'Custom',
            }
            for key, names in source_config.items():
                label = source_labels.get(key, str(key).title())
                for name in names or []:
                    source_map[str(name)] = label
        except Exception as exc:
            logger.debug("Failed to parse structure_mapping.json for %s: %s", course_dir, exc)

    def _norm(name: str) -> str:
        return ''.join(ch for ch in name.lower() if ch.isalnum())

    manual_rs = course_dir / "RS.dcm"
    auto_rs = course_dir / "RS_auto.dcm"
    manual_names: set[str] = set()
    auto_names: set[str] = set()

    def _load_roi_names(rs_path: Path) -> set[str]:
        if not rs_path.exists():
            return set()
        try:
            ds = pydicom.dcmread(str(rs_path), stop_before_pixels=True)
            names = {
                str(getattr(roi, "ROIName", ""))
                for roi in getattr(ds, "StructureSetROISequence", []) or []
            }
            return {name for name in names if name}
        except Exception as exc:
            logger.debug("Failed to read ROI names from %s: %s", rs_path, exc)
            return set()

    manual_names = _load_roi_names(manual_rs)
    auto_names = _load_roi_names(auto_rs)
    manual_norm = {_norm(name): "Manual" for name in manual_names}
    auto_norm = {_norm(name): "AutoTS" for name in auto_names}

    custom_config = config.custom_structures_config if getattr(config, "custom_structures_config", None) else None
    custom_norm: set[str] = set()
    if custom_config and Path(custom_config).exists():
        try:
            custom_data = json.loads(Path(custom_config).read_text())
        except json.JSONDecodeError:
            try:
                import yaml
                custom_yaml = yaml.safe_load(Path(custom_config).read_text())  # type: ignore
            except Exception:
                custom_yaml = None
            if custom_yaml and isinstance(custom_yaml, dict):
                for item in custom_yaml.get("custom_structures", []) or []:
                    name = str(item.get("name", ""))
                    if name:
                        custom_norm.add(_norm(name))
        except Exception:
            pass

    default_source = 'Merged' if rs_file == rs_custom else ('AutoTS' if rs_file == rs_auto else 'Manual')
    params_file = str(config.radiomics_params_file) if config.radiomics_params_file else None

    # Save CT image to temporary NRRD file
    with tempfile.NamedTemporaryFile(suffix='.nrrd', delete=False) as ct_file:
        sitk.WriteImage(ct_image, ct_file.name, useCompression=True)
        ct_image_path = ct_file.name

    tasks: List[Dict[str, Any]] = []
    try:
        from rt_utils import RTStructBuilder

        rtstruct = RTStructBuilder.create_from(
            dicom_series_path=str(ct_dir),
            rt_struct_path=str(rs_file)
        )

        skip_rois_default = {
            "body",
            "couchsurface",
            "couchinterior",
            "couchexterior",
            "bones",
            "entirebody",
            "entirebodyroi",
            "m1",
            "m2",
            "table",
            "support",
        }
        cfg_skip = {
            _norm(item)
            for item in getattr(config, "radiomics_skip_rois", [])
            if isinstance(item, str) and item.strip()
        }
        skip_rois = skip_rois_default | cfg_skip

        max_voxels_limit = getattr(config, "radiomics_max_voxels", None)
        if max_voxels_limit is None:
            max_voxels_limit = 15_000_000
        elif max_voxels_limit < 1:
            max_voxels_limit = 15_000_000

        min_voxels_limit = getattr(config, "radiomics_min_voxels", None)
        if min_voxels_limit is None:
            min_voxels_limit = 120
        elif min_voxels_limit < 1:
            min_voxels_limit = 1

        for roi_name in rtstruct.get_roi_names():
            norm_key = _norm(roi_name)
            if norm_key in skip_rois:
                logger.debug("Skipping radiomics for ROI %s (skip list)", roi_name)
                continue
            try:
                mask = rtstruct.get_roi_mask_by_name(roi_name)
            except Exception as exc:
                logger.debug("Failed to obtain mask for %s: %s", roi_name, exc)
                continue
            if mask is None:
                continue
            mask_bool = mask.astype(bool)
            if not mask_bool.any():
                logger.debug("Skipping radiomics for ROI %s: empty mask", roi_name)
                continue

            voxel_count = int(mask_bool.sum())
            if voxel_count < min_voxels_limit:
                logger.info(
                    "Skipping radiomics for ROI %s: %d voxels below minimum %d",
                    roi_name,
                    voxel_count,
                    min_voxels_limit,
                )
                continue
            if voxel_count > max_voxels_limit:
                logger.info(
                    "Skipping radiomics for ROI %s: %d voxels exceeds limit %d",
                    roi_name,
                    voxel_count,
                    max_voxels_limit,
                )
                continue

            try:
                with tempfile.NamedTemporaryFile(suffix='.nrrd', delete=False) as mask_file:
                    _write_mask_to_file(mask_bool, mask_file.name, ct_info)
                    mask_path = mask_file.name
            except Exception as exc:
                logger.debug("Failed to serialise mask for %s: %s", roi_name, exc)
                continue

            seg_source = source_map.get(roi_name)
            if not seg_source:
                norm = norm_key
                seg_source = manual_norm.get(norm)
                if not seg_source:
                    seg_source = auto_norm.get(norm)
                if not seg_source:
                    seg_source = 'Custom' if norm in custom_norm else default_source

            cropped_flag = mask_is_cropped(mask_bool)
            display_roi = roi_name if (not cropped_flag or roi_name.endswith("__partial")) else f"{roi_name}__partial"

            metadata = {
                'segmentation_source': seg_source,
                'course_dir': str(course_dir),
                'patient_id': course_dir.parent.name,
                'course_id': course_dir.name,
                'structure_cropped': cropped_flag,
                'roi_original_name': roi_name,
            }

            tasks.append({
                'image_path': ct_image_path,
                'mask_path': mask_path,
                'roi_name': display_roi,
                'params_file': params_file,
                'label': None,
                'metadata': metadata,
                'cleanup': True,
                'ct_info': ct_info,
            })

    except Exception as exc:
        logger.error("Failed to prepare radiomics masks for %s: %s", course_dir, exc)
        try:
            os.unlink(ct_image_path)
        except Exception:
            pass
        return None

    if not tasks:
        logger.warning("No valid ROIs found in %s", rs_file)
        try:
            os.unlink(ct_image_path)
        except Exception:
            pass
        return None

    output_path = course_dir / "radiomics_ct.xlsx"
    sequential = os.environ.get('RTPIPELINE_RADIOMICS_SEQUENTIAL', '').lower() in ('1', 'true', 'yes')

    # Determine worker count - respect Snakemake thread budget via env var or config
    max_workers = None
    worker_source = "unknown"

    # Priority 1: Environment variable (set by CLI from --max-workers or by Snakemake)
    env_workers = int(os.environ.get('RTPIPELINE_MAX_WORKERS', '0') or 0)
    if env_workers > 0:
        max_workers = env_workers
        worker_source = "RTPIPELINE_MAX_WORKERS env"

    # Priority 2: PipelineConfig.effective_workers() (respects --max-workers CLI arg)
    if max_workers is None and hasattr(config, 'effective_workers') and callable(config.effective_workers):
        try:
            max_workers = config.effective_workers()
            worker_source = "config.effective_workers()"
        except Exception:
            pass

    # Priority 3: Safe default based on task count (not cpu_count to avoid oversubscription)
    if max_workers is None:
        # When no budget is set, use conservative parallelism to avoid oversubscribing
        # With batch processing, each worker runs one subprocess containing multiple ROIs
        max_workers = min(4, len(tasks))  # Max 4 parallel batches by default
        worker_source = "default (no budget set)"
        logger.warning(
            "No worker budget set (RTPIPELINE_MAX_WORKERS env or config.effective_workers). "
            "Using conservative default of %d workers. Set RTPIPELINE_MAX_WORKERS for optimal performance.",
            max_workers
        )

    logger.info("Conda radiomics using %d workers (%s, CPU cores: %d)", max_workers, worker_source, os.cpu_count() or 0)

    result = process_radiomics_batch(
        tasks,
        output_path,
        sequential=sequential,
        max_workers=max_workers,
    )

    try:
        os.unlink(ct_image_path)
    except Exception:
        pass

    return result


def radiomics_for_course_mr(
    course_dir: Path,
    config: Any,
    params_file: Optional[Path] = None
) -> Optional[Path]:
    """
    Extract MR radiomics features using conda environment.

    Processes all MR series in the course directory that have:
    - NIFTI/ subdirectory with .nii.gz files
    - Segmentation_TotalSegmentator/ subdirectory with total_mr--*.nii.gz masks

    Args:
        course_dir: Path to course directory
        config: Pipeline configuration
        params_file: Optional path to MR radiomics parameters YAML

    Returns:
        Path to output radiomics_mr.xlsx if successful, None otherwise
    """
    course_dir = Path(course_dir)
    mr_root = course_dir / "MR"
    out_path = mr_root / "radiomics_mr.xlsx"

    if not mr_root.exists():
        logger.debug("No MR directory in %s", course_dir)
        return None

    # Get MR params file from config if not provided
    if params_file is None and config is not None:
        mr_params = getattr(config, 'mr_params_file', None)
        if mr_params:
            params_file = Path(mr_params)
            if not params_file.exists():
                # Try relative to course dir or config dir
                for base in [course_dir, Path.cwd()]:
                    candidate = base / mr_params
                    if candidate.exists():
                        params_file = candidate
                        break

    tasks: List[Dict[str, Any]] = []
    temp_files: List[Path] = []

    for series_root in sorted(p for p in mr_root.iterdir() if p.is_dir()):
        nifti_dir = series_root / "NIFTI"
        seg_dir = series_root / "Segmentation_TotalSegmentator"

        if not nifti_dir.exists() or not seg_dir.exists():
            continue

        # Find MR NIfTI file
        nifti_files = sorted(nifti_dir.glob("*.nii.gz"))
        if not nifti_files:
            continue

        mr_nifti = nifti_files[0]

        # Check metadata for modality
        meta_files = sorted(nifti_dir.glob("*.metadata.json"))
        if meta_files:
            try:
                import json
                meta = json.loads(meta_files[0].read_text(encoding='utf-8'))
                if str(meta.get('modality', '')).upper() != 'MR':
                    continue
            except Exception:
                pass

        # Convert MR NIfTI to NRRD for radiomics
        try:
            mr_img = sitk.ReadImage(str(mr_nifti))
            mr_nrrd = tempfile.NamedTemporaryFile(
                suffix=".nrrd", delete=False, prefix="mr_image_"
            )
            mr_nrrd.close()
            sitk.WriteImage(mr_img, mr_nrrd.name)
            temp_files.append(Path(mr_nrrd.name))
            mr_image_path = mr_nrrd.name
        except Exception as exc:
            logger.warning("Failed to convert MR NIfTI %s: %s", mr_nifti, exc)
            continue

        series_uid = series_root.name

        # Process each mask in Segmentation_TotalSegmentator
        for mask_path in sorted(seg_dir.glob("total_mr--*.nii.gz")):
            try:
                mask_img = sitk.ReadImage(str(mask_path))
                mask_arr = sitk.GetArrayFromImage(mask_img)

                # Skip empty masks
                if not (mask_arr > 0).any():
                    continue

                # Extract ROI name from mask filename
                mask_name = mask_path.name
                if mask_name.startswith("total_mr--"):
                    roi_name = mask_name[10:]  # Remove "total_mr--"
                else:
                    roi_name = mask_name
                if roi_name.endswith(".nii.gz"):
                    roi_name = roi_name[:-7]

                # Convert mask to NRRD
                mask_nrrd = tempfile.NamedTemporaryFile(
                    suffix=".nrrd", delete=False, prefix=f"mr_mask_{roi_name}_"
                )
                mask_nrrd.close()
                sitk.WriteImage(mask_img, mask_nrrd.name)
                temp_files.append(Path(mask_nrrd.name))

                tasks.append({
                    'image_path': mr_image_path,
                    'mask_path': mask_nrrd.name,
                    'roi_name': roi_name,
                    'params_file': str(params_file) if params_file and params_file.exists() else None,
                    'cleanup': False,  # We'll clean up temp files ourselves
                    'metadata': {  # Use 'metadata' not 'extra_metadata' for correct handling
                        'modality': 'MR',
                        'series_uid': series_uid,
                        'segmentation_source': 'AutoTS_total_mr',
                        'patient_id': course_dir.parent.name,
                        'course_id': course_dir.name,
                    }
                })
            except Exception as exc:
                logger.debug("Failed to process MR mask %s: %s", mask_path, exc)
                continue

    if not tasks:
        logger.debug("No valid MR radiomics tasks for %s", course_dir)
        # Cleanup temp files
        for tf in temp_files:
            try:
                tf.unlink()
            except Exception:
                pass
        return None

    logger.info("Processing %d MR radiomics tasks for %s", len(tasks), course_dir.name)

    # Determine worker count - same logic as CT radiomics
    max_workers = None
    env_workers = int(os.environ.get('RTPIPELINE_MAX_WORKERS', '0') or 0)
    if env_workers > 0:
        max_workers = env_workers
    elif hasattr(config, 'effective_workers') and callable(config.effective_workers):
        try:
            max_workers = config.effective_workers()
        except Exception:
            pass
    if max_workers is None:
        # Conservative default when no budget is set
        max_workers = min(4, len(tasks))

    logger.info("MR radiomics using %d workers", max_workers)

    result = process_radiomics_batch(
        tasks,
        out_path,
        sequential=False,
        max_workers=max_workers,
    )

    # Cleanup temp files
    for tf in temp_files:
        try:
            tf.unlink()
        except Exception:
            pass

    return result
