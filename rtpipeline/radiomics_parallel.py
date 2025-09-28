"""
Parallel radiomics processing module with segmentation fault prevention.

This module provides robust parallel processing for radiomics feature extraction
using process-based parallelism instead of thread-based to avoid segmentation faults.
"""

from __future__ import annotations

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

# Global constants for optimization
_OPTIMAL_WORKERS_CACHE = None
_MAX_RETRIES = 3
_RETRY_DELAY = 0.5  # seconds


def _calculate_optimal_workers() -> int:
    """Calculate optimal number of workers (cpu_count - 1) with caching."""
    global _OPTIMAL_WORKERS_CACHE
    if _OPTIMAL_WORKERS_CACHE is not None:
        return _OPTIMAL_WORKERS_CACHE

    cpu_count = os.cpu_count() or 2
    optimal = max(1, cpu_count - 1)

    # Check if user has set a specific override
    env_workers = int(os.environ.get('RTPIPELINE_RADIOMICS_WORKERS', '0'))
    if env_workers > 0:
        optimal = env_workers
        logger.info("Using %d radiomics workers from RTPIPELINE_RADIOMICS_WORKERS", optimal)
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
        # Import in process to avoid module state issues
        from .radiomics import _extractor, _mask_from_array_like
        import numpy as np

        # Load task data from temporary file
        with open(temp_file_path, 'rb') as f:
            data = pickle.load(f)

        img_array = data['img_array']
        img_info = data['img_info']
        mask = data['mask']
        config = data['config']
        source = data['source']
        roi = data['roi']
        course_dir = data['course_dir']

        # Reconstruct SimpleITK image
        img = sitk.GetImageFromArray(img_array)
        img.SetSpacing(img_info['spacing'])
        img.SetOrigin(img_info['origin'])
        img.SetDirection(img_info['direction'])

        # Create fresh extractor instance for this process
        ext = _extractor(config, 'CT')
        if ext is None:
            logger.debug("No radiomics extractor available for %s/%s", source, roi)
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
        rec = {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in res.items()}
        rec.update({
            'modality': 'CT',
            'segmentation_source': source,
            'roi_name': roi,
            'course_dir': str(course_dir)
        })

        return rec

    except Exception as e:
        logger.debug("Radiomics failed for %s/%s: %s",
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
    temp_dir: Path
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

    # Create temporary file for task data
    temp_fd, temp_file_path = tempfile.mkstemp(suffix='.pkl', dir=temp_dir)

    try:
        # Package data for serialization
        task_data = {
            'img_array': img_array,
            'img_info': img_info,
            'mask': mask,
            'config': config,
            'source': source,
            'roi': roi,
            'course_dir': course_dir
        }

        # Write to temporary file
        with os.fdopen(temp_fd, 'wb') as f:
            pickle.dump(task_data, f)

        # Task parameters for tracking
        task_params = {
            'source': source,
            'roi': roi,
            'course_dir': str(course_dir)
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


def parallel_radiomics_for_course(
    config: Any,
    course_dir: Path,
    custom_structures_config: Optional[Path] = None,
    max_workers: Optional[int] = None
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

    # Resume-friendly: skip if output exists
    out_path = course_dir / 'radiomics_features_CT.xlsx'
    if getattr(config, 'resume', False) and out_path.exists():
        return out_path

    # Check if pyradiomics is available
    extractor = _extractor(config, 'CT')
    if extractor is None:
        return None

    # Load CT image
    img = _load_series_image(course_dir / 'CT_DICOM')
    if img is None:
        logger.info("No CT image for radiomics in %s", course_dir)
        return None

    # Collect all tasks
    tasks = []

    # Process standard RTSTRUCTs
    for source, rs_name in (("Manual", "RS.dcm"), ("AutoRTS_total", "RS_auto.dcm")):
        rs_path = course_dir / rs_name
        if not rs_path.exists():
            continue
        masks = _rtstruct_masks(course_dir / 'CT_DICOM', rs_path)
        for roi, mask in masks.items():
            tasks.append((source, roi, mask))

    # Process custom structures if configuration provided
    if custom_structures_config and custom_structures_config.exists():
        try:
            rs_custom = course_dir / "RS_custom.dcm"
            if not rs_custom.exists():
                from .dvh import _create_custom_structures_rtstruct
                rs_custom = _create_custom_structures_rtstruct(
                    course_dir,
                    custom_structures_config,
                    course_dir / "RS.dcm",
                    course_dir / "RS_auto.dcm"
                )
            if rs_custom and rs_custom.exists():
                masks = _rtstruct_masks(course_dir / 'CT_DICOM', rs_custom)
                for roi, mask in masks.items():
                    tasks.append(("Custom", roi, mask))
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
        for source, roi, mask in tasks:
            try:
                task_file, task_params = _prepare_radiomics_task(
                    img, mask, config, source, roi, course_dir, temp_dir
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
            ctx = get_context('spawn')

            # Don't use initializer with spawn context as it causes pickling issues
            with ctx.Pool(max_workers) as pool:
                # Submit all tasks with retry wrapper
                future_results = pool.map(_isolated_radiomics_extraction_with_retry, prepared_tasks)

                # Collect successful results
                for result in future_results:
                    if result:
                        results.append(result)

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
    # Configure environment for this worker process
    env_vars = {
        'OMP_NUM_THREADS': '1',           # OpenMP
        'OPENBLAS_NUM_THREADS': '1',      # OpenBLAS
        'MKL_NUM_THREADS': '1',           # Intel MKL
        'NUMEXPR_NUM_THREADS': '1',       # NumExpr
        'NUMBA_NUM_THREADS': '1',         # Numba
    }

    for var, value in env_vars.items():
        os.environ[var] = value

    # Set process priority to normal to avoid system slowdown
    try:
        import psutil
        psutil.Process().nice(0)  # Normal priority
    except ImportError:
        pass  # psutil not available, continue without priority adjustment


def configure_parallel_radiomics():
    """Configure environment for parallel radiomics processing."""
    # Set environment variables to control threading libraries
    env_vars = {
        'OMP_NUM_THREADS': '1',           # OpenMP
        'OPENBLAS_NUM_THREADS': '1',      # OpenBLAS
        'MKL_NUM_THREADS': '1',           # Intel MKL
        'NUMEXPR_NUM_THREADS': '1',       # NumExpr
        'NUMBA_NUM_THREADS': '1',         # Numba
    }

    for var, value in env_vars.items():
        if var not in os.environ:
            os.environ[var] = value
            logger.debug("Set %s=%s", var, value)


def enable_parallel_radiomics_processing():
    """
    Enable parallel radiomics processing by setting environment variable.

    This function should be called before running the pipeline to enable
    the new parallel radiomics implementation.
    """
    os.environ['RTPIPELINE_USE_PARALLEL_RADIOMICS'] = '1'
    configure_parallel_radiomics()

    # Pre-calculate optimal workers
    optimal_workers = _calculate_optimal_workers()
    logger.info("Enabled parallel radiomics processing with %d optimal workers", optimal_workers)


def is_parallel_radiomics_enabled() -> bool:
    """Check if parallel radiomics processing is enabled."""
    return os.environ.get('RTPIPELINE_USE_PARALLEL_RADIOMICS', '').lower() in ('1', 'true', 'yes')
