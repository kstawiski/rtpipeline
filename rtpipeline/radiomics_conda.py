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
import pydicom

from .layout import build_course_dirs
from .utils import mask_is_cropped

logger = logging.getLogger(__name__)

RADIOMICS_ENV = "rtpipeline-radiomics"


def _conda_subprocess_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("CONDA_NO_PLUGINS", "1")
    env.setdefault("CONDA_OVERRIDE_CUDA", "0")
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
    worker_limit = max_workers if max_workers and max_workers > 0 else (os.cpu_count() or 4)
    worker_limit = max(1, min(worker_limit, len(tasks_list)))

    if sequential or len(tasks_list) == 1 or worker_limit == 1:
        _run_sequential(tasks_list)
    else:
        logger.info(
            "Processing %d radiomics tasks with up to %d worker threads",
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

    # Determine optimal worker count (matching radiomics_parallel.py logic)
    max_workers = None

    # Check for environment variable override first
    env_workers = int(os.environ.get('RTPIPELINE_RADIOMICS_WORKERS', '0'))
    if env_workers > 0:
        max_workers = env_workers
    # Try config.effective_workers() if available
    elif hasattr(config, 'effective_workers') and callable(config.effective_workers):
        try:
            max_workers = config.effective_workers()
        except Exception:
            pass
    # Fall back to cpu_count - 1 (same as radiomics_parallel.py)
    if max_workers is None:
        cpu_count = os.cpu_count() or 4
        max_workers = max(1, cpu_count - 1)

    logger.info("Conda radiomics using %d workers (CPU cores: %d)", max_workers, os.cpu_count() or 0)

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
