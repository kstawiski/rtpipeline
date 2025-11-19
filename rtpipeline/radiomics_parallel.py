"""
Parallel radiomics processing module with segmentation fault prevention.

This module provides robust parallel processing for radiomics feature extraction
using process-based parallelism instead of thread-based to avoid segmentation faults.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import tempfile
import traceback
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, get_context
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


# SECURITY: Restricted unpickler to prevent arbitrary code execution
class RestrictedUnpickler(pickle.Unpickler):
    """
    Restricted unpickler that only allows safe classes to prevent code execution attacks.

    Only allows:
    - NumPy types (ndarray, dtype, generic numeric types)
    - Python built-in types (dict, list, tuple, str, int, float, bool, None)
    - pathlib.Path
    """
    ALLOWED_MODULES = {
        'numpy': {'ndarray', 'dtype', 'int64', 'int32', 'float64', 'float32', 'bool_', 'generic'},
        'numpy.core.multiarray': {'_reconstruct'},
        'builtins': {'dict', 'list', 'tuple', 'str', 'int', 'float', 'bool', 'NoneType'},
        'pathlib': {'Path', 'PosixPath', 'WindowsPath'},
        'collections': {'OrderedDict'},  # Used internally by numpy
        'SimpleITK.SimpleITK': {'Image'},
        'rtpipeline.config': {'PipelineConfig'},
    }

    def find_class(self, module, name):
        """Override find_class to restrict allowed classes."""
        # Allow only specific safe modules and classes
        if module in self.ALLOWED_MODULES:
            if name in self.ALLOWED_MODULES[module]:
                return super().find_class(module, name)

        # Raise error for any other module/class
        raise pickle.UnpicklingError(
            f"Global '{module}.{name}' is forbidden for security reasons. "
            f"Only numpy arrays, built-in types, and pathlib.Path are allowed."
        )


def restricted_pickle_load(file_obj):
    """
    Safe pickle load using RestrictedUnpickler.

    Args:
        file_obj: File object opened in binary read mode

    Returns:
        Unpickled object (only if it contains safe types)

    Raises:
        pickle.UnpicklingError: If forbidden classes are encountered
    """
    return RestrictedUnpickler(file_obj).load()

# Global constants for optimization
_OPTIMAL_WORKERS_CACHE = None
_MAX_RETRIES = 3
_RETRY_DELAY = 0.5  # seconds
_THREAD_VARS = (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'NUMBA_NUM_THREADS',
)
_THREAD_LIMIT_ENV = 'RTPIPELINE_RADIOMICS_THREAD_LIMIT'
_TASK_TIMEOUT = int(os.environ.get('RTPIPELINE_RADIOMICS_TASK_TIMEOUT', '600'))  # 10 minutes per ROI


def _calculate_optimal_workers() -> int:
    """Calculate optimal number of workers (cpu_count - 1) with caching."""
    global _OPTIMAL_WORKERS_CACHE
    if _OPTIMAL_WORKERS_CACHE is not None:
        return _OPTIMAL_WORKERS_CACHE

    cpu_count = os.cpu_count() or 2
    optimal = max(1, cpu_count - 1)

    # Check if user has set a specific override
    env_workers = int(os.environ.get('RTPIPELINE_MAX_WORKERS', '0') or 0)
    if env_workers > 0:
        optimal = env_workers
        logger.info("Using %d radiomics workers from RTPIPELINE_MAX_WORKERS", optimal)
    else:
        logger.info("Calculated optimal radiomics workers: %d (CPU cores: %d, using: %d)",
                   optimal, cpu_count, optimal)

    _OPTIMAL_WORKERS_CACHE = optimal
    return optimal


def _isolated_radiomics_extraction_with_retry(task_data: Tuple[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Wrapper for radiomics extraction with retry logic."""
    temp_file_path, task_params = task_data

    for attempt in range(_MAX_RETRIES):
        try:
            result = _isolated_radiomics_extraction(task_data)
            if result is not None:
                return result
            # None result means extraction failed but no exception - don't retry
            break
        except Exception as e:
            if attempt < _MAX_RETRIES - 1:
                logger.debug("Radiomics extraction attempt %d/%d failed for %s/%s: %s. Retrying...",
                           attempt + 1, _MAX_RETRIES,
                           task_params.get('source', 'unknown'),
                           task_params.get('roi', 'unknown'), str(e))
                time.sleep(_RETRY_DELAY * (attempt + 1))  # Exponential backoff
            else:
                logger.warning("Radiomics extraction failed after %d attempts for %s/%s: %s",
                             _MAX_RETRIES,
                             task_params.get('source', 'unknown'),
                             task_params.get('roi', 'unknown'), str(e))
    return None


def _isolated_radiomics_extraction(task_data: Tuple[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Isolated radiomics extraction function for multiprocessing.

    This function runs in a separate process to avoid segmentation faults
    from pyradiomics/OpenMP interactions.

    Args:
        task_data: Tuple of (temp_file_path, task_params)

    Returns:
        Dictionary with radiomics features or None if failed
    """
    temp_file_path, task_params = task_data

    try:
        _apply_thread_limit(_resolve_thread_limit())
        # Import in process to avoid module state issues
        from .radiomics import _extractor, _mask_from_array_like
        import numpy as np

        # Load task data from temporary file (SECURITY: using restricted unpickler)
        with open(temp_file_path, 'rb') as f:
            data = restricted_pickle_load(f)

        img_array = data['img_array']
        img_info = data['img_info']
        mask = data['mask']
        config = data['config']
        source = data['source']
        roi_original = data.get('roi_original', data['roi'])
        roi_display = data.get('roi_display', roi_original)
        course_dir = Path(data['course_dir'])

        # Reconstruct SimpleITK image
        img = sitk.GetImageFromArray(img_array)
        img.SetSpacing(img_info['spacing'])
        img.SetOrigin(img_info['origin'])
        img.SetDirection(img_info['direction'])

        # Create fresh extractor instance for this process
        ext = _extractor(config, 'CT')
        if ext is None:
            logger.debug("No radiomics extractor available for %s/%s", source, roi_display)
            return None

        # Create mask image
        m_img = _mask_from_array_like(img, mask)

        # Set environment variables to control OpenMP in this process
        # This prevents threading conflicts that cause segmentation faults
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        os.environ['NUMBA_NUM_THREADS'] = '1'

        # Execute radiomics extraction
        res = ext.execute(img, m_img)

        # Convert results to serializable format
        rec = {}
        for k, v in res.items():
            try:
                rec[k] = float(v)
            except (ValueError, TypeError):
                rec[k] = str(v)
                
        patient_id = course_dir.parent.name if course_dir.parent != course_dir else course_dir.name
        rec.update({
            'modality': 'CT',
            'segmentation_source': source,
            'roi_name': roi_display,
            'roi_original_name': roi_original,
            'course_dir': str(course_dir),
            'patient_id': patient_id,
            'course_id': course_dir.name,
            'structure_cropped': bool(task_params.get('structure_cropped', False)),
        })

        # Merge any extra metadata provided in task params
        if 'extra_metadata' in task_params:
            rec.update(task_params['extra_metadata'])

        return rec

    except Exception as e:
        logger.warning("Radiomics failed for %s/%s: %s",
                    task_params.get('source', 'unknown'),
                    task_params.get('roi', 'unknown'),
                    str(e))
        # Log full traceback for debugging
        logger.debug("Full traceback: %s", traceback.format_exc())
        return None
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        except Exception:
            pass


def _prepare_radiomics_task(
    img: sitk.Image,
    mask: np.ndarray,
    config: Any,
    source: str,
    roi: str,
    course_dir: Path,
    temp_dir: Path,
    structure_cropped: bool,
) -> Tuple[str, Dict[str, Any]]:
    """
    Prepare a radiomics task for parallel processing.

    Args:
        img: SimpleITK image
        mask: Numpy mask array
        config: Pipeline configuration
        source: Source of the ROI (Manual, AutoRTS, Custom)
        roi: ROI name
        course_dir: Course directory path
        temp_dir: Temporary directory for task files

    Returns:
        Tuple of (temp_file_path, task_params)
    """
    # Extract image data and metadata
    img_array = sitk.GetArrayFromImage(img)
    img_info = {
        'spacing': img.GetSpacing(),
        'origin': img.GetOrigin(),
        'direction': img.GetDirection()
    }
    
    # Ensure mask is numpy array (fix for robustness module passing sitk.Image)
    mask_array = mask
    if isinstance(mask, sitk.Image):
        mask_array = sitk.GetArrayFromImage(mask)

    # Create temporary file for task data
    temp_fd, temp_file_path = tempfile.mkstemp(suffix='.pkl', dir=temp_dir)

    try:
        # Package data for serialization
        display_roi = roi if (not structure_cropped or roi.endswith("__partial")) else f"{roi}__partial"

        task_data = {
            'img_array': img_array,
            'img_info': img_info,
            'mask': mask_array,
            'config': config,
            'source': source,
            'roi': roi,
            'roi_display': display_roi,
            'roi_original': roi,
            'course_dir': course_dir,
            'structure_cropped': bool(structure_cropped),
        }

        # Write to temporary file
        with os.fdopen(temp_fd, 'wb') as f:
            pickle.dump(task_data, f)

        # Task parameters for tracking
        task_params = {
            'source': source,
            'roi': display_roi,
            'roi_original': roi,
            'course_dir': str(course_dir),
            'structure_cropped': bool(structure_cropped),
        }

        return temp_file_path, task_params

    except Exception:
        # Clean up on error
        try:
            os.close(temp_fd)
            os.unlink(temp_file_path)
        except Exception:
            pass
        raise


from .layout import build_course_dirs
from .utils import mask_is_cropped


def parallel_radiomics_for_course(
    config: Any,
    course_dir: Path,
    custom_structures_config: Optional[Path] = None,
    max_workers: Optional[int] = None,
    use_cropped: bool = True
) -> Optional[Path]:
    """
    Parallel radiomics processing for a single course.

    Args:
        config: Pipeline configuration
        course_dir: Course directory path
        custom_structures_config: Custom structures configuration
        max_workers: Maximum number of parallel workers

    Returns:
        Path to output file or None if failed
    """
    try:
        from .radiomics import (
            radiomics_for_course as original_radiomics_for_course,
            _extractor, _load_series_image, _rtstruct_masks
        )
    except ImportError as e:
        logger.error("Failed to import radiomics modules: %s", e)
        return None

    # Apply any configured thread limit in the parent process
    _apply_thread_limit(_resolve_thread_limit(getattr(config, 'radiomics_thread_limit', None)))

    # Resume-friendly: skip if output exists
    course_dirs = build_course_dirs(course_dir)
    out_path = course_dir / 'radiomics_ct.xlsx'
    if getattr(config, 'resume', False) and out_path.exists():
        return out_path

    # Check if pyradiomics is available
    extractor = _extractor(config, 'CT')
    if extractor is None:
        try:
            from .radiomics_conda import radiomics_for_course as conda_radiomics_for_course
        except ImportError as exc:
            logger.warning("Conda-based radiomics helper unavailable: %s", exc)
            return None
        logger.info("Delegating CT radiomics for %s to conda environment", course_dir)
        return conda_radiomics_for_course(course_dir, config, custom_structures_config)

    # Load CT image
    img = _load_series_image(course_dirs.dicom_ct)
    if img is None:
        logger.info("No CT image for radiomics in %s", course_dir)
        return None

    # Determine which auto RTSTRUCT to use (cropped or original)
    rs_auto_name = "RS_auto.dcm"
    if use_cropped:
        rs_auto_cropped = course_dir / "RS_auto_cropped.dcm"
        cropping_metadata_path = course_dir / "cropping_metadata.json"
        if rs_auto_cropped.exists() and cropping_metadata_path.exists():
            try:
                crop_meta = json.loads(cropping_metadata_path.read_text())
                logger.info(
                    "Using systematically cropped RTSTRUCT for radiomics "
                    "(region: %s, superior: %.1fmm, inferior: %.1fmm)",
                    crop_meta.get("region", "unknown"),
                    float(crop_meta.get("superior_z_mm", 0.0)),
                    float(crop_meta.get("inferior_z_mm", 0.0)),
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load cropping metadata from %s: %s",
                    cropping_metadata_path,
                    exc,
                )
            else:
                rs_auto_name = "RS_auto_cropped.dcm"

    # Collect all tasks
    tasks: List[tuple[str, str, np.ndarray, bool]] = []

    # Process standard RTSTRUCTs
    for source, rs_name in (("Manual", "RS.dcm"), ("AutoRTS_total", rs_auto_name)):
        rs_path = course_dir / rs_name
        if not rs_path.exists():
            continue
        masks = _rtstruct_masks(course_dirs.dicom_ct, rs_path)
        for roi, mask in masks.items():
            tasks.append((source, roi, mask, mask_is_cropped(mask)))

    # Process custom structures if configuration provided
    if custom_structures_config and custom_structures_config.exists():
        try:
            from .dvh import _create_custom_structures_rtstruct, _is_rs_custom_stale
            rs_custom = course_dir / "RS_custom.dcm"
            rs_manual = course_dir / "RS.dcm"
            rs_auto = course_dir / "RS_auto.dcm"

            # Check if RS_custom needs regeneration
            if _is_rs_custom_stale(rs_custom, custom_structures_config, rs_manual, rs_auto):
                logger.info("Regenerating RS_custom.dcm for radiomics in %s", course_dir.name)
                rs_custom = _create_custom_structures_rtstruct(
                    course_dir,
                    custom_structures_config,
                    rs_manual,
                    rs_auto
                )

            if rs_custom and rs_custom.exists():
                masks = _rtstruct_masks(course_dirs.dicom_ct, rs_custom)
                for roi, mask in masks.items():
                    tasks.append(("Custom", roi, mask, mask_is_cropped(mask)))
        except Exception as e:
            logger.warning("Failed to process custom structures for radiomics: %s", e)

    if not tasks:
        logger.info("No radiomics tasks for %s", course_dir)
        return None

    # Determine number of workers - use optimal calculation
    if max_workers is None:
        max_workers = _calculate_optimal_workers()
        # Still respect config if available
        try:
            config_workers = int(getattr(config, 'effective_workers')())
            max_workers = min(max_workers, config_workers)
        except Exception:
            pass

    # Don't exceed number of tasks
    max_workers = max(1, min(max_workers, len(tasks)))

    logger.info("Processing %d radiomics tasks with %d workers for %s",
                len(tasks), max_workers, course_dir.name)

    # Create temporary directory for task files
    temp_dir = Path(tempfile.mkdtemp(prefix='radiomics_'))

    try:
        # Prepare all tasks
        prepared_tasks = []
        for source, roi, mask, cropped in tasks:
            try:
                task_file, task_params = _prepare_radiomics_task(
                    img, mask, config, source, roi, course_dir, temp_dir, cropped
                )
                prepared_tasks.append((task_file, task_params))
            except Exception as e:
                logger.warning("Failed to prepare task for %s/%s: %s", source, roi, e)
                continue

        if not prepared_tasks:
            logger.warning("No valid radiomics tasks prepared for %s", course_dir)
            return None

        # Process tasks in parallel using process pool with enhanced efficiency
        results = []

        if max_workers == 1:
            # Sequential processing for debugging or single-core systems
            logger.info("Using sequential radiomics processing")
            for task_data in prepared_tasks:
                result = _isolated_radiomics_extraction_with_retry(task_data)
                if result:
                    results.append(result)
        else:
            # Parallel processing with process pool and retry logic
            logger.info("Using parallel radiomics processing with %d workers (CPU cores: %d)",
                       max_workers, os.cpu_count() or 'unknown')

            # Use spawn context to avoid issues with forked processes
            import multiprocessing
            ctx = multiprocessing.get_context('spawn')

            # Don't use initializer with spawn context as it causes pickling issues
            with ctx.Pool(max_workers) as pool:
                # Use imap_unordered for progress monitoring
                completed_count = 0
                total_count = len(prepared_tasks)
                start_time = time.time()

                try:
                    # Use imap_unordered for progress monitoring
                    for result in pool.imap_unordered(_isolated_radiomics_extraction_with_retry, prepared_tasks):
                        if result:
                            results.append(result)
                        completed_count += 1

                        # Log progress periodically
                        if completed_count % 10 == 0 or completed_count == total_count:
                            elapsed = time.time() - start_time
                            rate = completed_count / elapsed if elapsed > 0 else 0
                            eta = (total_count - completed_count) / rate if rate > 0 else 0
                            logger.info("Radiomics progress: %d/%d (%.1f%%), ETA: %.1fs",
                                       completed_count, total_count,
                                       100 * completed_count / total_count, eta)
                except Exception as e:
                    logger.error("Error during parallel radiomics processing: %s", e)
                    # Pool will be cleaned up by context manager

        logger.info("Completed %d/%d radiomics extractions for %s",
                   len(results), len(prepared_tasks), course_dir.name)

        # Save results
        if results:
            try:
                import pandas as pd
                df = pd.DataFrame(results)
                df.to_excel(out_path, index=False)
                return out_path
            except Exception as e:
                logger.error("Failed to save radiomics results: %s", e)
                return None
        else:
            logger.warning("No successful radiomics extractions for %s", course_dir)
            return None

    finally:
        # Clean up temporary directory
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.debug("Failed to clean up temp directory %s: %s", temp_dir, e)


def _worker_initializer():
    """Initialize worker process with optimal settings."""
    _apply_thread_limit(_resolve_thread_limit())

    # Set process priority to normal to avoid system slowdown
    try:
        import psutil
        psutil.Process().nice(0)  # Normal priority
    except ImportError:
        pass  # psutil not available, continue without priority adjustment


def configure_parallel_radiomics(thread_limit: Optional[int] = None):
    """Configure environment for parallel radiomics processing."""
    limit = _resolve_thread_limit(thread_limit)
    _apply_thread_limit(limit)


def enable_parallel_radiomics_processing(thread_limit: Optional[int] = None):
    """
    Enable parallel radiomics processing by setting environment variable.

    This function should be called before running the pipeline to enable
    the new parallel radiomics implementation.
    """
    os.environ['RTPIPELINE_USE_PARALLEL_RADIOMICS'] = '1'
    if thread_limit is not None and thread_limit > 0:
        os.environ[_THREAD_LIMIT_ENV] = str(int(thread_limit))
    else:
        os.environ.pop(_THREAD_LIMIT_ENV, None)
    configure_parallel_radiomics(thread_limit)

    # Pre-calculate optimal workers
    optimal_workers = _calculate_optimal_workers()
    logger.info("Enabled parallel radiomics processing with %d optimal workers", optimal_workers)


def is_parallel_radiomics_enabled() -> bool:
    """Check if parallel radiomics processing is enabled."""
    return os.environ.get('RTPIPELINE_USE_PARALLEL_RADIOMICS', '').lower() in ('1', 'true', 'yes')


def _coerce_thread_limit(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return value


def _resolve_thread_limit(explicit: Optional[int] = None) -> Optional[int]:
    if explicit is not None:
        return _coerce_thread_limit(explicit)
    env_value = os.environ.get(_THREAD_LIMIT_ENV)
    return _coerce_thread_limit(env_value)


def _apply_thread_limit(limit: Optional[int]) -> None:
    if limit is None:
        for var in _THREAD_VARS:
            os.environ.pop(var, None)
        return
    limit = max(1, int(limit))
    value = str(limit)
    for var in _THREAD_VARS:
        os.environ[var] = value
