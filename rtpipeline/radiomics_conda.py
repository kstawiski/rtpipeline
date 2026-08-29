#!/usr/bin/env python3
"""
Radiomics module using conda environment isolation.
Runs PyRadiomics in a separate conda environment with NumPy 1.x for compatibility.
"""

import json
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # For future type hints

import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom

from .custom_models import (
    list_custom_model_outputs,
    validate_custom_model_output_inventory,
)
from .layout import build_course_dirs, find_dcm
from .radiomics_outcomes import (
    RadiomicsCourseExtractionError,
    RadiomicsCourseOutcome,
    course_diagnostic_columns,
    extraction_status_is_nonfatal_for_required,
    invalidate_radiomics_outputs as _invalidate_radiomics_outputs,
    remove_artifact_strict as _remove_artifact_strict,
    roi_source_is_required,
    write_excel_atomic as _write_excel_atomic,
)
from .utils import mask_is_cropped, radiomics_mp_context

logger = logging.getLogger(__name__)


RADIOMICS_ENV = os.environ.get("RTPIPELINE_RADIOMICS_ENV", "rtpipeline-radiomics")
_ENV_PROBE_TIMEOUT_ENV = "RTPIPELINE_RADIOMICS_ENV_PROBE_TIMEOUT"
_DEFAULT_ENV_PROBE_TIMEOUT = 180


class RadiomicsEnvironmentProbeTimeout(RuntimeError):
    """The isolated radiomics import probe exhausted its time budget."""

    code = "RADIOMICS_ENV_PROBE_TIMEOUT"

    def __init__(self, command: List[str], timeout: int, attempts: int) -> None:
        self.command = tuple(command)
        self.timeout = int(timeout)
        self.attempts = int(attempts)
        super().__init__(
            "radiomics environment probe timed out: "
            f"command={shlex.join(command)!r}, timeout={timeout}s, attempts={attempts}"
        )

    def __reduce__(self):
        return (type(self), (list(self.command), self.timeout, self.attempts))


def _conda_executable() -> str:
    """Resolve a conda-compatible executable for isolated radiomics calls."""
    configured = os.environ.get("RTPIPELINE_CONDA_EXE")
    if configured:
        return configured
    for candidate in ("conda", "micromamba", "mamba"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return "conda"  # Preserve the historical error message when none is installed.


CONDA_EXE = _conda_executable()

# Heartbeat interval for progress logging (seconds)
HEARTBEAT_INTERVAL = 60

# Cache for a confirmed-functional radiomics conda env. The env does not change
# mid-run, but ``process_radiomics_batch`` is called once per course, so a naive
# per-course ``conda run`` probe spawns a fresh subprocess every time. Under the
# nested worker load (course-level threads each driving a ProcessPoolExecutor of
# radiomics workers) that probe's startup contends for the box and times out,
# returning False and silently skipping an otherwise-healthy course. We therefore
# cache the first SUCCESSFUL check process-wide (double-checked under a lock) so a
# transient probe timeout cannot drop courses. A False result is never cached, so
# a genuinely-not-yet-ready env can still recover on a later call.
_ENV_CHECK_LOCK = threading.Lock()
_ENV_CHECK_OK: Optional[bool] = None


def _jsonify_nested_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    """JSON-encode any column holding nested (dict/list/set/tuple) values for Parquet.

    PyRadiomics emits diagnostics fields such as
    ``diagnostics_Configuration_EnabledImageTypes`` whose value can be a nested
    dict like ``{'Original': {}}`` — an empty-child struct that PyArrow cannot
    encode ("Cannot write struct type 'Original' with no child field to
    Parquet"), so every checkpoint flush failed. We JSON-encode nested values to
    strings (rather than dropping the columns) so the checkpoint remains a
    FAITHFUL record of every completed ROI: it is the data source used to rebuild
    the per-course workbook on resume, and dropping provenance columns would make
    a resumed course's ``radiomics_ct.xlsx`` lose those columns. Scalar columns
    are left untouched.
    """
    if df.empty:
        return df
    out = df
    copied = False
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, (dict, list, set, tuple))).any():
            if not copied:
                out = df.copy()
                copied = True
            out[col] = out[col].map(
                lambda v: json.dumps(v, default=str, sort_keys=True)
                if isinstance(v, (dict, list, set, tuple))
                else v
            )
    return out


def _identity_value(obj: Dict[str, Any], name: str, fallback: str = "") -> str:
    metadata = obj.get("metadata") or obj.get("extra_metadata") or {}
    value = obj.get(name)
    try:
        missing = value is None or bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = value is None
    if missing:
        value = metadata.get(name)
    try:
        missing = value is None or bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = value is None
    if missing and fallback:
        value = obj.get(fallback)
        try:
            fallback_missing = value is None or bool(pd.isna(value))
        except (TypeError, ValueError):
            fallback_missing = value is None
        if fallback_missing:
            value = metadata.get(fallback)
    try:
        if value is None or bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        if value is None:
            return ""
    return str(value).strip()


def _roi_instance_key(obj: Dict[str, Any]) -> str:
    """Stable source/ROI/series checkpoint identity.

    Source is required because Manual, AutoRTS, Custom, and custom-model
    RTSTRUCTs can legitimately contain the same ROI name. Series remains part of
    the identity for MR, where the same source/ROI recurs across acquisitions.
    """
    source = _identity_value(obj, "segmentation_source")
    roi = _identity_value(obj, "roi_original_name", fallback="roi_name")
    series = _identity_value(obj, "series_uid")
    return f"{source}\x1f{roi}\x1f{series}"


def _validated_identity_keys(records: List[Dict[str, Any]], *, context: str) -> Set[str]:
    keys: List[str] = []
    for record in records:
        source = _identity_value(record, "segmentation_source")
        roi = _identity_value(record, "roi_original_name", fallback="roi_name")
        if not source or not roi:
            raise ValueError(
                f"{context} has blank segmentation_source/roi_original_name identity"
            )
        keys.append(_roi_instance_key(record))
    if len(keys) != len(set(keys)):
        raise ValueError(f"{context} has duplicate source/ROI/series identities")
    return set(keys)


def _strip_nii_suffix(path: Path) -> str:
    """Return a NIfTI filename stem, handling .nii.gz as one suffix."""
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def _norm_roi_key(name: str) -> str:
    return ''.join(ch for ch in str(name).lower() if ch.isalnum())


def _ct_skip_rois(config: Any) -> Set[str]:
    skip_rois_default = {
        "couchsurface",
        "couchinterior",
        "couchexterior",
        "bones",
        "m1",
        "m2",
        "table",
        "support",
    }
    cfg_skip = {
        _norm_roi_key(item)
        for item in getattr(config, "radiomics_skip_rois", [])
        if isinstance(item, str) and item.strip()
    }
    return skip_rois_default | cfg_skip


def _ct_voxel_limits(config: Any) -> Tuple[int, int]:
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
    return int(min_voxels_limit), int(max_voxels_limit)


def _select_usable_rtstruct(*paths: Path) -> Optional[Path]:
    """Return the first RTSTRUCT path that exists and has at least one ROI."""
    for path in paths:
        if not path.exists():
            continue
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
            rois = getattr(ds, "StructureSetROISequence", []) or []
            if rois:
                return path
            logger.warning("Ignoring RTSTRUCT with no ROIs for CT radiomics: %s", path)
        except Exception as exc:
            raise RadiomicsCourseExtractionError(
                f"Required RTSTRUCT is unreadable for CT radiomics: {path}: {exc}"
            ) from exc
    return None


def _ct_nifti_candidates(course_dir: Path) -> Dict[str, Path]:
    nifti_dir = course_dir / "NIFTI"
    if not nifti_dir.exists():
        return {}
    out: Dict[str, Path] = {}
    for path in sorted(nifti_dir.glob("*.nii*")):
        if not path.is_file() or path.name.startswith(".") or "_cropped" in path.name:
            continue
        out[_strip_nii_suffix(path)] = path
    return out


def _series_uid_from_nifti(nifti_path: Path, fallback: str) -> str:
    meta_path = nifti_path.with_name(f"{_strip_nii_suffix(nifti_path)}.metadata.json")
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            uid = data.get("series_instance_uid")
            if uid:
                return str(uid)
        except Exception as exc:
            logger.debug("Failed to read NIfTI metadata %s: %s", meta_path, exc)
    return fallback


def _totalseg_roi_name(mask_path: Path) -> Optional[str]:
    name = _strip_nii_suffix(mask_path)
    if "--" in name:
        _model, name = name.split("--", 1)
    if name.endswith("_cropped"):
        return None
    return name or None


class RadiomicsCheckpoint:
    """Manages checkpoint state for resumable radiomics extraction.

    Saves completed ROI results to a Parquet file so extraction can resume
    if interrupted. This is critical for large datasets where extraction
    can take hours.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        buffer_size: int = 50,
        expected_keys: Optional[Set[str]] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.buffer_size = buffer_size
        self._buffer: List[Dict[str, Any]] = []
        self._completed_keys: Set[str] = set()
        self._lock = threading.Lock()
        self._load_existing(expected_keys)

    def _load_existing(self, expected_keys: Optional[Set[str]]) -> None:
        """Reuse only a structurally valid, exact current-identity checkpoint."""
        if not self.checkpoint_path.exists():
            return
        try:
            df = pd.read_parquet(self.checkpoint_path)
            if df.empty:
                raise ValueError("checkpoint is empty")
            feature_markers = (
                "_firstorder_", "_shape_", "_shape2D_", "_glcm_",
                "_glrlm_", "_glszm_", "_gldm_", "_ngtdm_",
            )
            if not any(
                any(marker in str(column) for marker in feature_markers)
                for column in df.columns
            ):
                raise ValueError("checkpoint has no PyRadiomics feature column")
            records = df.to_dict("records")
            keys = _validated_identity_keys(records, context="checkpoint")
            if expected_keys is not None and keys != expected_keys:
                raise ValueError(
                    "checkpoint identity set does not exactly match current tasks "
                    f"(expected {len(expected_keys)}, found {len(keys)})"
                )
            self._completed_keys = keys
            logger.info("Checkpoint loaded: %d ROIs already completed", len(keys))
        except Exception as exc:
            logger.warning(
                "Rejecting stale or invalid checkpoint %s: %s",
                self.checkpoint_path,
                exc,
            )
            _remove_artifact_strict(
                self.checkpoint_path,
                context="rejecting invalid radiomics checkpoint",
            )
            self._completed_keys.clear()

    def is_completed(self, key: str) -> bool:
        """Check if a ROI instance (see ``_roi_instance_key``) has been processed."""
        return key in self._completed_keys

    def add_result(self, result: Dict[str, Any]) -> None:
        """Add a completed result and flush if buffer is full."""
        with self._lock:
            self._buffer.append(result)
            if result.get('roi_name'):
                self._completed_keys.add(_roi_instance_key(result))

            if len(self._buffer) >= self.buffer_size:
                self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Write buffered results to checkpoint file."""
        if not self._buffer:
            return

        try:
            new_df = _jsonify_nested_columns(pd.DataFrame(self._buffer))

            if self.checkpoint_path.exists():
                existing_df = pd.read_parquet(self.checkpoint_path)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                combined_df = new_df

            # Write to a temp file and atomically publish via os.replace(): writing
            # the parquet file in place would leave it truncated (and silently
            # discarded by _load_existing on the next run) if the process is
            # killed mid-write.
            tmp_path = self.checkpoint_path.parent / f".{self.checkpoint_path.name}.{os.getpid()}.tmp"
            try:
                combined_df.to_parquet(tmp_path, index=False)
                os.replace(tmp_path, self.checkpoint_path)
            finally:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            self._buffer.clear()
            logger.debug("Checkpoint flushed: %d total ROIs", len(self._completed_keys))
        except Exception as exc:
            logger.error("Failed to flush checkpoint: %s", exc)

    def flush(self) -> None:
        """Force flush any remaining buffered results."""
        with self._lock:
            self._flush_buffer()

    def get_completed_count(self) -> int:
        """Return number of completed ROI instances."""
        return len(self._completed_keys)

    def discard(self) -> None:
        """Delete a checkpoint that cannot represent a complete publication."""
        with self._lock:
            self._buffer.clear()
            self._completed_keys.clear()
            _remove_artifact_strict(
                self.checkpoint_path,
                context="discarding incomplete radiomics checkpoint",
            )

    def load_records(self) -> List[Dict[str, Any]]:
        """Return structurally valid checkpoint records without silent deduplication."""
        with self._lock:
            self._flush_buffer()  # make sure buffered rows are on disk first
            if not self.checkpoint_path.exists():
                return []
            try:
                df = pd.read_parquet(self.checkpoint_path)
            except Exception as exc:
                raise RadiomicsCourseExtractionError(
                    f"Failed to read checkpoint records {self.checkpoint_path}: {exc}"
                ) from exc
        records = df.to_dict("records")
        _validated_identity_keys(records, context="checkpoint")
        return records


class HeartbeatLogger:
    """Logs periodic progress updates during long-running operations.

    Starts a background thread that logs progress every HEARTBEAT_INTERVAL seconds.
    This provides visibility when radiomics extraction is running for hours.
    """

    def __init__(self, total_tasks: int, description: str = "Processing"):
        self.total_tasks = total_tasks
        self.description = description
        self._completed = 0
        self._failed = 0
        self._skipped = 0
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time = 0.0

    def start(self) -> None:
        """Start the heartbeat logging thread."""
        self._running = True
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the heartbeat logging thread and log final summary."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        # Log final summary
        self._log_final_summary()

    def _log_final_summary(self) -> None:
        """Log the final completion summary."""
        with self._lock:
            elapsed = time.monotonic() - self._start_time
            logger.info(
                "[Complete] %s: %d completed, %d failed, %d skipped in %.1f minutes",
                self.description,
                self._completed,
                self._failed,
                self._skipped,
                elapsed / 60,
            )

    def update(self, completed: int = 0, failed: int = 0, skipped: int = 0) -> None:
        """Update progress counters."""
        with self._lock:
            self._completed += completed
            self._failed += failed
            self._skipped += skipped

    def _heartbeat_loop(self) -> None:
        """Background thread that logs progress periodically."""
        last_log = time.monotonic()
        while self._running:
            time.sleep(1)
            now = time.monotonic()
            if now - last_log >= HEARTBEAT_INTERVAL:
                self._log_progress()
                last_log = now

    def _log_progress(self) -> None:
        """Log current progress."""
        with self._lock:
            done = self._completed + self._failed + self._skipped
            elapsed = time.monotonic() - self._start_time

            if done > 0 and elapsed > 0:
                rate = done / elapsed
                remaining = self.total_tasks - done
                eta_seconds = remaining / rate if rate > 0 else 0
                eta_str = f"{eta_seconds/60:.1f}min" if eta_seconds < 3600 else f"{eta_seconds/3600:.1f}hr"
            else:
                eta_str = "calculating..."

            logger.info(
                "[Heartbeat] %s: %d/%d done (%d completed, %d failed, %d skipped) - ETA: %s",
                self.description,
                done,
                self.total_tasks,
                self._completed,
                self._failed,
                self._skipped,
                eta_str,
            )

    def __enter__(self) -> 'HeartbeatLogger':
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()  # stop() now includes final summary logging


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


def check_radiomics_env(timeout: Optional[int] = None, retries: int = 1) -> bool:
    """Check if the radiomics conda environment exists and is functional.

    A successful result is cached process-wide: the env cannot disappear
    mid-run, and re-probing per course spawned a ``conda run`` subprocess whose
    cold-start under heavy worker load timed out spuriously, silently skipping
    healthy courses. The first check uses a generous timeout and one retry so a
    transient cold-start stall does not produce a false negative; only a True
    result is cached, so a genuinely-unready env can still recover later. The
    timeout can be set with ``RTPIPELINE_RADIOMICS_ENV_PROBE_TIMEOUT`` (or explicitly)
    and an exhausted timeout raises a typed error instead of masquerading as a
    feature-extraction failure.
    """
    global _ENV_CHECK_OK
    if _ENV_CHECK_OK:
        return True
    with _ENV_CHECK_LOCK:
        if _ENV_CHECK_OK:
            return True
        configured_timeout = timeout
        if configured_timeout is None:
            try:
                configured_timeout = int(os.environ.get(_ENV_PROBE_TIMEOUT_ENV, _DEFAULT_ENV_PROBE_TIMEOUT))
            except (TypeError, ValueError):
                configured_timeout = _DEFAULT_ENV_PROBE_TIMEOUT
        if configured_timeout <= 0:
            configured_timeout = _DEFAULT_ENV_PROBE_TIMEOUT

        command = [
            CONDA_EXE,
            "run",
            "-n",
            RADIOMICS_ENV,
            "python",
            "-c",
            "import radiomics; import numpy; print('OK')",
        ]
        attempts = max(1, int(retries) + 1)
        timeout_failures = 0
        last_err: Optional[str] = None
        for attempt in range(attempts):
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=configured_timeout,
                    env=_conda_subprocess_env(),
                )
                if result.returncode == 0 and "OK" in result.stdout:
                    _ENV_CHECK_OK = True
                    return True
                last_err = (result.stderr or "").strip() or f"returncode={result.returncode}"
            except subprocess.TimeoutExpired as e:
                timeout_failures += 1
                last_err = str(e)
                logger.warning(
                    "Radiomics env check attempt %d/%d timed out after %ds",
                    attempt + 1,
                    attempts,
                    configured_timeout,
                )
            except Exception as e:
                last_err = str(e)
                logger.warning(
                    "Radiomics env check attempt %d/%d failed: %s",
                    attempt + 1,
                    attempts,
                    e,
                )
        if timeout_failures == attempts:
            raise RadiomicsEnvironmentProbeTimeout(command, configured_timeout, attempts)
        logger.error("Failed to verify radiomics environment after %d attempts: %s", attempts, last_err)
        return False


def extract_radiomics_with_conda(
    image_path: str,
    mask_path: str,
    params_file: Optional[str] = None,
    label: Optional[int] = None,
    large_roi: bool = False,
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
        'label': label,
        'large_roi': bool(large_roi),
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
large_roi = bool(params.get('large_roi'))

# Create extractor
if params_file:
    extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
else:
    extractor = featureextractor.RadiomicsFeatureExtractor()

# Reduced settings for very large ROIs (feasibility)
if large_roi:
    try:
        extractor.disableAllImageTypes()
        extractor.enableImageTypeByName('Original')
    except Exception:
        pass
    try:
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName('firstorder')
        extractor.enableFeatureClassByName('shape')
    except Exception:
        pass
    try:
        extractor.settings['resampledPixelSpacing'] = [2.0, 2.0, 2.0]
    except Exception:
        pass

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
            [CONDA_EXE, "run", "-n", RADIOMICS_ENV, "python", "-c", extraction_script_with_file],
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
    timeout_per_roi: int = 900,
) -> List[Dict[str, Any]]:
    """
    Extract radiomics features for multiple ROIs in a SINGLE subprocess.

    This dramatically reduces overhead by loading the radiomics library once
    and processing all ROIs sequentially within that subprocess.

    Args:
        tasks: List of dicts with 'image_path', 'mask_path', optional 'label', 'roi_name'
        params_file: Optional path to radiomics parameters YAML file
        timeout_per_roi: Timeout per ROI in seconds (default 900s = 15 min)

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

# Create extractors (full + reduced for large ROIs)
if params_file:
    extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
    extractor_large = featureextractor.RadiomicsFeatureExtractor(params_file)
else:
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor_large = featureextractor.RadiomicsFeatureExtractor()

# Configure the reduced extractor for feasibility on very large masks (e.g., BODY)
try:
    extractor_large.disableAllImageTypes()
    extractor_large.enableImageTypeByName('Original')
except Exception:
    pass
try:
    extractor_large.disableAllFeatures()
    extractor_large.enableFeatureClassByName('firstorder')
    extractor_large.enableFeatureClassByName('shape')
except Exception:
    pass
try:
    extractor_large.settings['resampledPixelSpacing'] = [2.0, 2.0, 2.0]
except Exception:
    pass

# Process each task
for task in tasks:
    image_path = task['image_path']
    mask_path = task['mask_path']
    label = task.get('label')
    roi_name = task.get('roi_name', 'ROI')
    large_roi = bool(task.get('large_roi'))
    task_index = task.get('__task_index__')

    try:
        current_extractor = extractor_large if large_roi else extractor
        if label is not None:
            features = current_extractor.execute(image_path, mask_path, label=label)
        else:
            features = current_extractor.execute(image_path, mask_path)

        # Convert to JSON-serializable format
        output = {'__status__': 'success', '__roi_name__': roi_name, '__task_index__': task_index}
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
            print(json.dumps({'__status__': 'skipped', '__roi_name__': roi_name, '__task_index__': task_index, '__reason__': 'ROI too small'}), flush=True)
        else:
            print(json.dumps({'__status__': 'error', '__roi_name__': roi_name, '__task_index__': task_index, '__error__': str(e)}), flush=True)
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
                        'large_roi': bool(t.get('large_roi', False)),
                        # Unique per-task position, echoed back by the subprocess so the
                        # caller can match results by index instead of roi_name (which is
                        # not guaranteed unique within a batch) or a fragile positional zip.
                        '__task_index__': i,
                    }
                    for i, t in enumerate(tasks)
                ],
                'params_file': params_file,
            }, batch_file)
            batch_file_path = batch_file.name

        # Run batch extraction in conda environment
        result = subprocess.run(
            [CONDA_EXE, "run", "-n", RADIOMICS_ENV, "python", "-c", batch_script, batch_file_path],
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


def _process_batch(batch: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
    """Process a batch of radiomics tasks in a subprocess and return (task, features) pairs."""
    params_file = batch[0].get('params_file') if batch else None
    batch_results = extract_radiomics_batch_with_conda(batch, params_file)

    # Match results back to tasks by the unique __task_index__ the subprocess echoes
    # back (injected per-task in extract_radiomics_batch_with_conda), not by roi_name
    # or a positional zip(batch, batch_results). roi_name is not guaranteed unique
    # within a batch (this module documents that elsewhere), so name-based matching
    # can still misattribute features between same-named ROIs when a subprocess
    # output line is dropped/unparseable -- only the injected index is unambiguous.
    results_by_index: Dict[int, Dict[str, Any]] = {}
    for result in batch_results:
        if not result:
            continue
        idx = result.get('__task_index__')
        if isinstance(idx, int) and idx not in results_by_index:
            results_by_index[idx] = result

    pairs: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]] = []
    for i, task in enumerate(batch):
        roi_name = task.get('roi_name', 'ROI')
        result = results_by_index.get(i)
        if result is None:
            logger.warning("Batch radiomics: no matching result for ROI %s (task #%d); dropping", roi_name, i)
        pairs.append((task, result))
    return pairs


def process_radiomics_batch(
    tasks: List[Dict[str, Any]],
    output_path: Path,
    sequential: bool = False,
    max_workers: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    enable_heartbeat: bool = True,
    env_probe_timeout: Optional[int] = None,
) -> Optional[Path]:
    """Process radiomics extraction tasks and persist them as an Excel sheet.

    Args:
        tasks: List of radiomics task dictionaries
        output_path: Path for output Excel file
        sequential: Force sequential processing
        max_workers: Maximum parallel workers
        checkpoint_path: Optional path for checkpoint file (enables resume)
        enable_heartbeat: Whether to enable progress heartbeat logging
        env_probe_timeout: Seconds allowed for each environment import probe

    Returns:
        Path to output file if successful, None otherwise
    """

    if not tasks:
        logger.warning("No radiomics tasks to process")
        _invalidate_radiomics_outputs(Path(output_path))
        return None

    try:
        expected_keys = _validated_identity_keys(tasks, context="radiomics task inventory")
    except ValueError as exc:
        _invalidate_radiomics_outputs(Path(output_path))
        raise RadiomicsCourseExtractionError(str(exc)) from exc

    # A checkpoint is reusable only when it already represents this exact task
    # inventory. Partial, stale, blank, duplicate, missing, or extra identities are
    # deleted and the full current inventory is recomputed.
    checkpoint: Optional[RadiomicsCheckpoint] = None
    if checkpoint_path:
        checkpoint_file_existed = Path(checkpoint_path).exists()
        checkpoint = RadiomicsCheckpoint(
            checkpoint_path,
            expected_keys=expected_keys,
        )
        already_completed = checkpoint.get_completed_count()
        if checkpoint_file_existed and already_completed == 0:
            _invalidate_radiomics_outputs(Path(output_path))
        if already_completed > 0:
            logger.info("Resume mode: %d ROIs already completed, filtering tasks", already_completed)

    cleanup_paths = {
        task.get('mask_path')
        for task in tasks
        if task.get('mask_path') and task.get('cleanup', True)
    }

    # Filter out already-completed tasks if checkpointing is enabled
    original_task_count = len(tasks)
    if checkpoint:
        tasks = [t for t in tasks if not checkpoint.is_completed(_roi_instance_key(t))]
        skipped_count = original_task_count - len(tasks)
        if skipped_count > 0:
            logger.info("Skipping %d already-completed ROIs (resume mode)", skipped_count)

    # The conda env is only needed to COMPUTE features. If every ROI was already
    # completed in a prior run (tasks is now empty after checkpoint filtering), skip
    # the probe entirely and fall through to rebuild the workbook from the checkpoint
    # below. Otherwise the env must be functional before we try to extract anything.
    if tasks and not check_radiomics_env(timeout=env_probe_timeout):
        logger.error(
            "Radiomics conda environment '%s' not found or not functional",
            RADIOMICS_ENV,
        )
        logger.error(
            "Please run: conda create -n %s python=3.11 numpy=1.26.* pyradiomics SimpleITK -c conda-forge",
            RADIOMICS_ENV,
        )
        _invalidate_radiomics_outputs(Path(output_path))
        raise RadiomicsCourseExtractionError(
            f"Radiomics conda environment '{RADIOMICS_ENV}' is unavailable"
        )

    def _execute(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        roi_name = task.get('roi_name', 'ROI')
        image_path = task.get('image_path')
        mask_path = task.get('mask_path')
        params_file = task.get('params_file')
        label = task.get('label')
        metadata = dict(task.get('metadata', {}))
        metadata.setdefault('roi_name', roi_name)
        metadata.setdefault('roi_original_name', roi_name)

        if task.get("precomputed_failure"):
            return {
                "__status__": "error",
                "__error__": str(task["precomputed_failure"].get("reason", "unknown error")),
                "__extraction_status__": str(
                    task["precomputed_failure"].get("status", "failed")
                ),
                "__failure_kind__": str(
                    task["precomputed_failure"].get("failure_kind", "extraction_error")
                ),
            }

        if not image_path or not mask_path:
            logger.error("Radiomics task is missing required paths for %s", roi_name)
            return {'__status__': 'error', '__error__': 'missing required image or mask path'}

        try:
            features = extract_radiomics_with_conda(
                image_path,
                mask_path,
                params_file,
                label,
                bool(task.get("large_roi", False)),
            )
        except Exception as exc:
            msg = str(exc).lower()
            if 'size of the roi is too small' in msg:
                logger.info(
                    "Skipping radiomics for ROI %s: %s",
                    roi_name,
                    str(exc).strip(),
                )
                return {'__status__': 'skipped', '__reason__': str(exc).strip()}
            if 'mask has too few dimensions' in msg and task.get('ct_info'):
                repaired = _ensure_mask_has_three_dimensions(mask_path, task['ct_info'])
                if repaired:
                    logger.debug("Rewrote 2D mask for ROI %s", roi_name)
                    try:
                        features = extract_radiomics_with_conda(
                            image_path,
                            mask_path,
                            params_file,
                            label,
                            bool(task.get("large_roi", False)),
                        )
                    except Exception as inner_exc:
                        logger.error("Radiomics retry failed for %s: %s", roi_name, inner_exc)
                        return {'__status__': 'error', '__error__': str(inner_exc)}
                else:
                    logger.error("Unable to repair mask dimensionality for %s", roi_name)
                    return {
                        '__status__': 'error',
                        '__error__': 'mask dimensionality could not be repaired',
                    }
            else:
                logger.error("Failed to extract features for %s: %s", roi_name, exc)
                return {'__status__': 'error', '__error__': str(exc)}

        metadata.setdefault('modality', 'CT')
        return _combine_feature_record(features, metadata)

    results: List[Dict[str, Any]] = []
    failures: List[str] = []
    failure_rows: List[Dict[str, Any]] = []

    def _record_failure(
        task: Dict[str, Any],
        detail: str,
        *,
        status: str = "failed",
        failure_kind: str = "extraction_error",
    ) -> None:
        roi_name = str(task.get("roi_name", "ROI"))
        metadata = dict(task.get("metadata") or task.get("extra_metadata") or {})
        source = str(metadata.get("segmentation_source", "unknown"))
        if (
            roi_source_is_required(source)
            and not extraction_status_is_nonfatal_for_required(status)
        ):
            failures.append(f"{source}/{roi_name}: {detail}")
            return
        metadata.setdefault("modality", "CT")
        metadata.setdefault("roi_name", roi_name)
        metadata.setdefault("roi_original_name", roi_name)
        metadata.update(
            {
                "extraction_status": status,
                "extraction_status_detail": detail,
                "extraction_failure_kind": failure_kind,
            }
        )
        failure_rows.append(metadata)

    def _run_sequential(seq: List[Dict[str, Any]]) -> None:
        for idx, task in enumerate(seq, 1):
            roi_name = task.get('roi_name', 'ROI')
            logger.info("Processing %d/%d: %s", idx, len(seq), roi_name)
            rec = _execute(task)
            if rec and rec.get('__status__') == 'skipped':
                _record_failure(
                    task,
                    str(rec.get('__reason__', 'mask was too small for radiomics')),
                    status="below_minimum_voxels",
                    failure_kind="degenerate_mask",
                )
                continue
            if rec and rec.get('__status__') == 'error':
                _record_failure(
                    task,
                    str(rec.get('__error__', 'unknown error')),
                    status=str(rec.get('__extraction_status__', 'failed')),
                    failure_kind=str(rec.get('__failure_kind__', 'extraction_error')),
                )
            elif rec:
                results.append(rec)
            else:
                _record_failure(task, "returned no outcome record")

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
    if any(task.get("precomputed_failure") for task in tasks_list):
        # Synthetic failure records have no mask file and must be handled by the
        # metadata-aware executor rather than passed to the external batch CLI.
        use_batch_processing = False

    # Initialize heartbeat logger if enabled
    heartbeat: Optional[HeartbeatLogger] = None
    if enable_heartbeat and tasks_list:
        heartbeat = HeartbeatLogger(
            total_tasks=original_task_count,
            description=f"Radiomics ({Path(output_path).stem})"
        )
        heartbeat.start()
        # Account for already-skipped tasks from checkpoint
        if checkpoint:
            heartbeat.update(completed=checkpoint.get_completed_count())

    try:
        if sequential or len(tasks_list) == 1 or worker_limit == 1:
            if use_batch_processing:
                # Even sequential mode benefits from batch processing. Match results
                # back to tasks via _process_batch's __task_index__-based pairing (not a
                # positional zip(tasks_list, batch_results)): the same dropped/
                # unparseable-subprocess-line hazard that motivated the fix in
                # _process_batch applies here too, since this also runs the whole
                # task list through a single extract_radiomics_batch_with_conda() call.
                logger.info("Processing %d radiomics tasks in batch mode (sequential)", len(tasks_list))
                batch_pairs = _process_batch(tasks_list)

                for task, features in batch_pairs:
                    roi_name = task.get('roi_name', 'ROI')
                    if features is None or features.get('__status__') != 'success':
                        status = features.get('__status__', 'failed') if features else 'failed'
                        if status == 'skipped':
                            _record_failure(
                                task,
                                str((features or {}).get('__reason__', 'mask was too small for radiomics')),
                                status="below_minimum_voxels",
                                failure_kind="degenerate_mask",
                            )
                            logger.info("Recorded best-effort/required degenerate ROI %s", roi_name)
                            if heartbeat:
                                heartbeat.update(failed=1)
                        else:
                            detail = (
                                features.get('__error__', 'unknown error')
                                if features else 'returned no outcome record'
                            )
                            logger.error("Radiomics failed for %s: %s", roi_name, detail)
                            _record_failure(
                                task,
                                str(detail),
                                status=(
                                    str(features.get('__extraction_status__', 'failed'))
                                    if features else 'failed'
                                ),
                                failure_kind=(
                                    str(features.get('__failure_kind__', 'extraction_error'))
                                    if features else 'extraction_error'
                                ),
                            )
                            if heartbeat:
                                heartbeat.update(failed=1)
                        continue

                    # Remove internal status fields and combine with metadata
                    features_clean = {k: v for k, v in features.items() if not k.startswith('__')}
                    metadata = dict(task.get('metadata') or task.get('extra_metadata') or {})
                    metadata.setdefault('roi_name', roi_name)
                    metadata.setdefault('roi_original_name', roi_name)
                    metadata.setdefault('modality', 'CT')
                    record = _combine_feature_record(features_clean, metadata)
                    results.append(record)

                    # Save to checkpoint if enabled
                    if checkpoint:
                        checkpoint.add_result(record)
                    if heartbeat:
                        heartbeat.update(completed=1)
            else:
                _run_sequential(tasks_list)
        elif use_batch_processing:
            # OPTIMIZED: Split tasks into batches and process each batch in a single subprocess
            # This amortizes the ~2-5 second subprocess startup across multiple ROIs
            batch_size = max(1, len(tasks_list) // worker_limit)  # Allow fine-grained parallelism
            batches = [tasks_list[i:i + batch_size] for i in range(0, len(tasks_list), batch_size)]

            logger.info(
                "Processing %d radiomics tasks in %d batches (%d ROIs/batch) with %d workers",
                len(tasks_list),
                len(batches),
                batch_size,
                min(worker_limit, len(batches)),
            )

            from concurrent.futures import ProcessPoolExecutor, as_completed

            # Use ProcessPoolExecutor for better parallelism (avoids GIL).
            # Each worker processes one batch (multiple ROIs) per subprocess.
            # NOTE: this pool is created from WITHIN a course-level ThreadPoolExecutor
            # worker thread. The default 'fork' start method copies locked mutexes from the
            # multi-threaded parent into workers and deadlocks (workers block forever on an
            # inherited-locked SemLock); use a forkserver/spawn context instead.
            with ProcessPoolExecutor(
                max_workers=min(worker_limit, len(batches)),
                mp_context=radiomics_mp_context(),
            ) as executor:
                future_to_batch = {executor.submit(_process_batch, batch): batch for batch in batches}

                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        batch_results = future.result()
                        for task, features in batch_results:
                            roi_name = task.get('roi_name', 'ROI')
                            if features is None or features.get('__status__') != 'success':
                                status = features.get('__status__', 'failed') if features else 'failed'
                                if status == 'skipped':
                                    _record_failure(
                                        task,
                                        str((features or {}).get('__reason__', 'mask was too small for radiomics')),
                                        status="below_minimum_voxels",
                                        failure_kind="degenerate_mask",
                                    )
                                    logger.debug("Recorded best-effort/required degenerate ROI %s", roi_name)
                                    if heartbeat:
                                        heartbeat.update(failed=1)
                                else:
                                    detail = (
                                        features.get('__error__', 'unknown error')
                                        if features else 'returned no outcome record'
                                    )
                                    logger.warning("Radiomics failed for %s: %s", roi_name, detail)
                                    _record_failure(
                                        task,
                                        str(detail),
                                        status=(
                                            str(features.get('__extraction_status__', 'failed'))
                                            if features else 'failed'
                                        ),
                                        failure_kind=(
                                            str(features.get('__failure_kind__', 'extraction_error'))
                                            if features else 'extraction_error'
                                        ),
                                    )
                                    if heartbeat:
                                        heartbeat.update(failed=1)
                                continue

                            # Remove internal status fields and combine with metadata
                            features_clean = {k: v for k, v in features.items() if not k.startswith('__')}
                            metadata = dict(task.get('metadata') or task.get('extra_metadata') or {})
                            metadata.setdefault('roi_name', roi_name)
                            metadata.setdefault('roi_original_name', roi_name)
                            metadata.setdefault('modality', 'CT')
                            record = _combine_feature_record(features_clean, metadata)
                            results.append(record)

                            # Save to checkpoint if enabled
                            if checkpoint:
                                checkpoint.add_result(record)
                            if heartbeat:
                                heartbeat.update(completed=1)

                            logger.debug("Completed radiomics for %s", roi_name)
                    except Exception as exc:
                        logger.error("Batch processing failed: %s", exc)
                        for task in batch:
                            _record_failure(task, f"batch crashed: {exc}")
                        if heartbeat:
                            heartbeat.update(failed=len(batch))
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
                        if rec and rec.get('__status__') == 'skipped':
                            _record_failure(
                                task,
                                str(rec.get('__reason__', 'mask was too small for radiomics')),
                                status="below_minimum_voxels",
                                failure_kind="degenerate_mask",
                            )
                            if heartbeat:
                                heartbeat.update(failed=1)
                        elif rec and rec.get('__status__') == 'error':
                            _record_failure(
                                task,
                                str(rec.get('__error__', 'unknown error')),
                                status=str(rec.get('__extraction_status__', 'failed')),
                                failure_kind=str(rec.get('__failure_kind__', 'extraction_error')),
                            )
                            if heartbeat:
                                heartbeat.update(failed=1)
                        elif rec:
                            results.append(rec)
                            if checkpoint:
                                checkpoint.add_result(rec)
                            if heartbeat:
                                heartbeat.update(completed=1)
                            logger.debug("Completed radiomics for %s", roi_name)
                        else:
                            _record_failure(task, "returned no successful feature record")
                            if heartbeat:
                                heartbeat.update(failed=1)
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.error("Radiomics task crashed for %s: %s", roi_name, exc)
                        _record_failure(task, f"task crashed: {exc}")
                        if heartbeat:
                            heartbeat.update(failed=1)

    finally:
        # Always stop heartbeat and flush checkpoint
        if heartbeat:
            heartbeat.stop()
        if checkpoint:
            checkpoint.flush()

    if failures:
        _invalidate_radiomics_outputs(Path(output_path))
        if checkpoint is not None:
            checkpoint.discard()
        raise RadiomicsCourseExtractionError(
            "Radiomics course extraction is incomplete: " + "; ".join(failures)
        )

    # Publish only an exact identity set. A valid checkpoint is all-or-nothing;
    # partial or stale rows are never unioned into current results.
    rows: List[Dict[str, Any]] = list(results)
    if checkpoint is not None:
        try:
            done_now = {_roi_instance_key(record) for record in results}
            prior = [
                record for record in checkpoint.load_records()
                if _roi_instance_key(record) not in done_now
            ]
        except Exception:
            _invalidate_radiomics_outputs(Path(output_path))
            checkpoint.discard()
            raise
        if prior:
            logger.info(
                "Resume: reusing %d exact-current checkpoint ROI(s) for %s",
                len(prior), output_path,
            )
            rows.extend(prior)
    rows.extend(failure_rows)

    required_publication_keys = expected_keys
    try:
        publication_keys = _validated_identity_keys(
            rows,
            context="radiomics publication",
        )
    except ValueError as exc:
        _invalidate_radiomics_outputs(Path(output_path))
        if checkpoint is not None:
            checkpoint.discard()
        raise RadiomicsCourseExtractionError(str(exc)) from exc
    if publication_keys != required_publication_keys:
        _invalidate_radiomics_outputs(Path(output_path))
        if checkpoint is not None:
            checkpoint.discard()
        raise RadiomicsCourseExtractionError(
            "Radiomics publication identity set is incomplete or stale "
            f"(expected {len(required_publication_keys)}, found {len(publication_keys)})"
        )

    if not rows:
        logger.warning("No radiomics features extracted")
        for mask_path in cleanup_paths:
            try:
                if mask_path:
                    os.unlink(mask_path)
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.debug("Cleanup failed for %s: %s", mask_path, exc)
        _invalidate_radiomics_outputs(Path(output_path))
        if checkpoint is not None:
            checkpoint.discard()
        return None

    try:
        source_counts: Dict[str, Dict[str, int]] = {}
        roi_failures: List[Dict[str, str]] = []
        for row in rows:
            source = str(row.get("segmentation_source", "unknown"))
            status = row.get("extraction_status")
            try:
                if status != status:
                    status = None
            except (TypeError, ValueError):
                pass
            counts = source_counts.setdefault(source, {"attempted": 0, "extracted": 0, "failed": 0})
            counts["attempted"] += 1
            if status in (None, "success"):
                counts["extracted"] += 1
            else:
                counts["failed"] += 1
                roi_failures.append(
                    {
                        "roi_name": str(row.get("roi_original_name", row.get("roi_name", "unknown"))),
                        "source": source,
                        "status": str(status),
                        "failure_kind": str(row.get("extraction_failure_kind", "extraction_error")),
                        "reason": str(row.get("extraction_status_detail", "unknown error")),
                    }
                )
        outcome = RadiomicsCourseOutcome.extracted(
            Path(output_path),
            roi_counts=source_counts,
            roi_failures=roi_failures,
            detail=(
                f"extracted {sum(values['extracted'] for values in source_counts.values())} "
                f"of {sum(values['attempted'] for values in source_counts.values())} "
                "attempted ROIs"
            ),
        )
        diagnostics = course_diagnostic_columns(outcome)
        for row in rows:
            row.update(diagnostics)
        df = pd.DataFrame(rows)
        _write_excel_atomic(df, Path(output_path))
        _remove_artifact_strict(
            Path(output_path).with_suffix(".parquet"),
            context="invalidating stale Parquet sidecar after conda workbook publication",
        )
        logger.info("Saved %d radiomics rows to %s", len(df), output_path)
    except Exception as exc:
        _invalidate_radiomics_outputs(Path(output_path))
        if checkpoint is not None:
            checkpoint.discard()
        raise RadiomicsCourseExtractionError(
            f"Failed to save radiomics workbook {output_path}: {exc}"
        ) from exc
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


def radiomics_for_course_ct_nifti_fallback(
    course_dir: Path,
    config: Any,
) -> Optional[Path]:
    """Extract CT radiomics from TotalSegmentator NIfTI masks when no RS is usable."""
    course_dir = Path(course_dir)
    course_dirs = build_course_dirs(course_dir)
    seg_dir = course_dirs.segmentation_totalseg
    output_path = course_dir / "radiomics_ct.xlsx"

    if not seg_dir.exists():
        logger.debug("No TotalSegmentator NIfTI mask directory for CT fallback in %s", course_dir)
        _invalidate_radiomics_outputs(output_path)
        return None

    params_file = str(config.radiomics_params_file) if getattr(config, "radiomics_params_file", None) else None
    skip_rois = _ct_skip_rois(config)
    min_voxels_limit, max_voxels_limit = _ct_voxel_limits(config)
    nifti_by_stem = _ct_nifti_candidates(course_dir)

    series_dirs = [p for p in sorted(seg_dir.iterdir()) if p.is_dir()]
    if any(seg_dir.glob("*.nii*")):
        series_dirs.insert(0, seg_dir)

    tasks: List[Dict[str, Any]] = []
    preparation_failures: List[Dict[str, str]] = []
    temp_files: List[Path] = []
    image_cache: Dict[Path, Tuple[str, Tuple[float, float, float]]] = {}
    dicom_image_path: Optional[str] = None
    dicom_spacing: Optional[Tuple[float, float, float]] = None

    def _cleanup_temp_files() -> None:
        for temp_file in temp_files:
            try:
                temp_file.unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("Failed to remove temporary CT fallback file %s: %s", temp_file, exc)

    def _fail_fallback(detail: str, exc: Optional[BaseException] = None) -> None:
        _cleanup_temp_files()
        _invalidate_radiomics_outputs(output_path)
        error = RadiomicsCourseExtractionError(detail)
        if exc is not None:
            raise error from exc
        raise error

    def _record_preparation_failure(
        roi_name: str,
        detail: str,
        *,
        status: str = "failed",
        failure_kind: str = "extraction_error",
    ) -> None:
        preparation_failures.append(
            {
                "roi_name": str(roi_name),
                "reason": detail,
                "status": status,
                "failure_kind": failure_kind,
            }
        )
        logger.warning(
            "Best-effort radiomics ROI AutoTS_total_nifti_fallback/%s was not prepared: %s",
            roi_name,
            detail,
        )

    def _dicom_image() -> Optional[Tuple[str, Tuple[float, float, float]]]:
        nonlocal dicom_image_path, dicom_spacing
        if dicom_image_path is not None and dicom_spacing is not None:
            return dicom_image_path, dicom_spacing
        ct_dir = course_dirs.dicom_ct
        if not ct_dir.exists():
            return None
        try:
            reader = sitk.ImageSeriesReader()
            dicom_files = reader.GetGDCMSeriesFileNames(str(ct_dir))
            if not dicom_files:
                return None
            reader.SetFileNames(dicom_files)
            img = reader.Execute()
            tmp = tempfile.NamedTemporaryFile(suffix=".nrrd", delete=False, prefix="ct_dicom_fallback_")
            tmp.close()
            sitk.WriteImage(img, tmp.name, useCompression=True)
            temp_files.append(Path(tmp.name))
            dicom_image_path = tmp.name
            dicom_spacing = tuple(float(x) for x in img.GetSpacing())
            return dicom_image_path, dicom_spacing
        except Exception as exc:
            _fail_fallback(
                f"Failed to load DICOM CT for NIfTI-mask fallback in {course_dir}: {exc}",
                exc,
            )

    def _image_for_series(series_name: str) -> Optional[Tuple[str, str, Tuple[float, float, float], str]]:
        nifti_path = nifti_by_stem.get(series_name)
        if nifti_path is None and len(nifti_by_stem) == 1:
            nifti_path = next(iter(nifti_by_stem.values()))

        if nifti_path is not None:
            cached = image_cache.get(nifti_path)
            if cached is None:
                try:
                    img = sitk.ReadImage(str(nifti_path))
                    tmp = tempfile.NamedTemporaryFile(suffix=".nrrd", delete=False, prefix="ct_nifti_fallback_")
                    tmp.close()
                    sitk.WriteImage(img, tmp.name, useCompression=True)
                    cached = (tmp.name, tuple(float(x) for x in img.GetSpacing()))
                    image_cache[nifti_path] = cached
                    temp_files.append(Path(tmp.name))
                except Exception as exc:
                    _fail_fallback(
                        f"Failed to convert CT NIfTI {nifti_path} for radiomics fallback: {exc}",
                        exc,
                    )
            image_path, spacing = cached
            return image_path, str(nifti_path), spacing, _series_uid_from_nifti(nifti_path, series_name)

        dicom = _dicom_image()
        if dicom is None:
            return None
        image_path, spacing = dicom
        return image_path, "", spacing, series_name

    for series_root in series_dirs:
        series_name = series_root.name if series_root != seg_dir else "CT"
        image_info = _image_for_series(series_name)
        if image_info is None:
            _fail_fallback(
                f"No CT image matched TotalSegmentator mask series {series_name} in {course_dir}"
            )
        image_path, nifti_path_str, image_spacing, series_uid = image_info

        for mask_path in sorted(series_root.glob("*.nii*")):
            roi_name = _totalseg_roi_name(mask_path)
            if not roi_name:
                continue
            norm_key = _norm_roi_key(roi_name)
            if norm_key in skip_rois:
                logger.debug("Skipping CT fallback ROI %s (skip list)", roi_name)
                continue

            try:
                mask_img = sitk.ReadImage(str(mask_path))
                mask_arr = sitk.GetArrayFromImage(mask_img)
            except Exception as exc:
                _record_preparation_failure(
                    roi_name,
                    f"Failed to read CT fallback mask {mask_path}: {exc}",
                )
                continue

            mask_bool = mask_arr > 0
            if not mask_bool.any():
                _record_preparation_failure(
                    roi_name,
                    f"CT fallback ROI {roi_name} has an empty mask: {mask_path}",
                    failure_kind="degenerate_mask",
                )
                continue

            voxel_count = int(mask_bool.sum())
            if voxel_count < min_voxels_limit:
                _record_preparation_failure(
                    roi_name,
                    f"ROI contains {voxel_count} voxels; configured minimum is "
                    f"{min_voxels_limit}",
                    status="below_minimum_voxels",
                    failure_kind="degenerate_mask",
                )
                continue

            try:
                native_voxel_mm3 = float(image_spacing[0]) * float(image_spacing[1]) * float(image_spacing[2])
            except Exception:
                native_voxel_mm3 = 1.0
            physical_volume_mm3 = float(voxel_count) * max(1e-9, native_voxel_mm3)
            estimated_voxels = physical_volume_mm3
            large_roi = norm_key.startswith("body") or (estimated_voxels > float(max_voxels_limit))
            if large_roi:
                logger.info(
                    "CT fallback ROI %s is large (native=%d voxels, est@1mm=%.0f voxels, cap=%d); using reduced radiomics settings",
                    roi_name,
                    voxel_count,
                    estimated_voxels,
                    int(max_voxels_limit),
                )

            try:
                mask_nrrd = tempfile.NamedTemporaryFile(
                    suffix=".nrrd",
                    delete=False,
                    prefix=f"ct_ts_mask_{roi_name}_",
                )
                mask_nrrd.close()
                sitk.WriteImage(mask_img, mask_nrrd.name, useCompression=True)
                temp_files.append(Path(mask_nrrd.name))
            except Exception as exc:
                _record_preparation_failure(
                    roi_name,
                    f"Failed to convert CT fallback mask {mask_path}: {exc}",
                )
                continue

            cropped_flag = mask_is_cropped(mask_bool)
            display_roi = roi_name if (not cropped_flag or roi_name.endswith("__partial")) else f"{roi_name}__partial"

            tasks.append({
                "image_path": image_path,
                "mask_path": mask_nrrd.name,
                "roi_name": display_roi,
                "params_file": params_file,
                "label": None,
                "large_roi": bool(large_roi),
                "cleanup": False,
                "metadata": {
                    "modality": "CT",
                    "series_uid": series_uid,
                    "segmentation_source": "AutoTS_total_nifti_fallback",
                    "course_dir": str(course_dir),
                    "patient_id": course_dir.parent.name,
                    "course_id": course_dir.name,
                    "structure_cropped": cropped_flag,
                    "roi_original_name": roi_name,
                    "nifti_path": nifti_path_str,
                    "mask_path_source": str(mask_path),
                },
            })

    for failure in preparation_failures:
        tasks.append(
            {
                "image_path": dicom_image_path or "best-effort-preparation-failure",
                "mask_path": None,
                "roi_name": failure["roi_name"],
                "params_file": params_file,
                "label": None,
                "large_roi": False,
                "cleanup": False,
                "metadata": {
                    "modality": "CT",
                    "segmentation_source": "AutoTS_total_nifti_fallback",
                    "course_dir": str(course_dir),
                    "patient_id": course_dir.parent.name,
                    "course_id": course_dir.name,
                    "structure_cropped": False,
                    "roi_original_name": failure["roi_name"],
                },
                "precomputed_failure": failure,
            }
        )

    if not tasks:
        logger.warning("No valid TotalSegmentator NIfTI ROIs found for CT fallback in %s", course_dir)
        _cleanup_temp_files()
        _invalidate_radiomics_outputs(output_path)
        return None

    logger.info("Processing %d CT radiomics fallback tasks for %s", len(tasks), course_dir.name)

    max_workers = None
    env_workers = int(os.environ.get("RTPIPELINE_MAX_WORKERS", "0") or 0)
    if env_workers > 0:
        max_workers = env_workers
    elif hasattr(config, "effective_workers") and callable(config.effective_workers):
        try:
            max_workers = config.effective_workers()
        except Exception:
            pass
    if max_workers is None:
        max_workers = min(4, len(tasks))

    sequential = os.environ.get("RTPIPELINE_RADIOMICS_SEQUENTIAL", "").lower() in ("1", "true", "yes")
    checkpoint_path = course_dir / "metadata" / "radiomics_ct_checkpoint.parquet"

    try:
        result = process_radiomics_batch(
            tasks,
            output_path,
            sequential=sequential,
            max_workers=max_workers,
            checkpoint_path=checkpoint_path,
            enable_heartbeat=True,
            env_probe_timeout=getattr(config, "radiomics_env_probe_timeout", None),
        )
    finally:
        _cleanup_temp_files()

    if result is None:
        _invalidate_radiomics_outputs(output_path)
    return result


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
    course_dir = Path(course_dir)
    course_dirs = build_course_dirs(course_dir)
    output_path = course_dir / "radiomics_ct.xlsx"

    # Check for CT DICOM files
    ct_dir = course_dirs.dicom_ct
    has_ct_dicom = ct_dir.exists() and any(path.is_file() for path in ct_dir.rglob('*'))
    has_ct_nifti = bool(_ct_nifti_candidates(course_dir))
    if not has_ct_dicom and not has_ct_nifti:
        logger.warning(f"No CT image found in {course_dir}")
        _invalidate_radiomics_outputs(output_path)
        return None

    rs_manual = find_dcm(course_dirs.dicom_rtstruct, "RS.dcm", course_dir)
    rs_auto = course_dir / "RS_auto.dcm"
    rs_custom = course_dir / "RS_custom.dcm"

    def _norm(name: str) -> str:
        return "".join(ch for ch in str(name).lower() if ch.isalnum())

    skip_rois = _ct_skip_rois(config)
    custom_cfg_value = (
        custom_structures_config
        or getattr(config, "custom_structures_config", None)
    )
    custom_cfg: Optional[Path] = None
    desired_custom: Set[str] = set()
    if custom_cfg_value:
        custom_cfg = Path(custom_cfg_value)
        if not custom_cfg.is_file():
            _invalidate_radiomics_outputs(output_path)
            raise RadiomicsCourseExtractionError(
                f"Configured required custom structure file is missing: {custom_cfg}"
            )
        try:
            from .radiomics import _custom_roi_names_from_config

            desired_custom = {
                name for name in _custom_roi_names_from_config(custom_cfg)
                if _norm(name) not in skip_rois
            }
        except Exception as exc:
            _invalidate_radiomics_outputs(output_path)
            raise RadiomicsCourseExtractionError(
                f"Failed to read configured custom ROI identities for {course_dir}: {exc}"
            ) from exc
        try:
            from .custom_structures_rtstruct import (
                _create_custom_structures_rtstruct,
                _is_rs_custom_stale,
            )

            if desired_custom and _is_rs_custom_stale(
                rs_custom, custom_cfg, rs_manual, rs_auto
            ):
                rs_custom = _create_custom_structures_rtstruct(
                    course_dir, custom_cfg, rs_manual, rs_auto
                ) or rs_custom
        except Exception as exc:
            _invalidate_radiomics_outputs(output_path)
            raise RadiomicsCourseExtractionError(
                f"Failed to prepare configured custom RTSTRUCT for {course_dir}: {exc}"
            ) from exc

    try:
        from .radiomics import _list_roi_names_dicom

        if custom_cfg is None and not desired_custom and rs_custom.exists():
            manual_names = set(_list_roi_names_dicom(rs_manual))
            auto_names = set(_list_roi_names_dicom(rs_auto))
            custom_names = set(_list_roi_names_dicom(rs_custom))
            inferred = custom_names - (manual_names | auto_names)
            desired_custom = {
                name[:-9] if name.endswith("__partial") else name
                for name in inferred
                if _norm(name) not in skip_rois
            }

        custom_wanted: List[str] = []
        if desired_custom:
            if not rs_custom.is_file():
                raise RadiomicsCourseExtractionError(
                    f"Required custom RTSTRUCT is missing for configured ROIs in {course_dir}"
                )
            available_custom = set(_list_roi_names_dicom(rs_custom))
            missing_custom: List[str] = []
            for base in sorted(desired_custom):
                if base in available_custom:
                    custom_wanted.append(base)
                elif f"{base}__partial" in available_custom:
                    custom_wanted.append(f"{base}__partial")
                else:
                    missing_custom.append(base)
            if missing_custom:
                raise RadiomicsCourseExtractionError(
                    f"Required configured custom ROI(s) missing from {rs_custom}: "
                    + ", ".join(missing_custom)
                )

        custom_model_expected_rois = validate_custom_model_output_inventory(
            course_dir,
            getattr(config, "custom_model_names", None),
            getattr(config, "custom_models_root", None),
        )
        custom_model_outputs = list_custom_model_outputs(course_dir)
    except Exception as exc:
        _invalidate_radiomics_outputs(output_path)
        if isinstance(exc, RadiomicsCourseExtractionError):
            raise
        raise RadiomicsCourseExtractionError(
            f"Failed to enumerate required segmentation inventory for {course_dir}: {exc}"
        ) from exc

    # Each source is extracted independently. Never collapse colliding ROI names
    # from Manual, AutoRTS, Custom, or a custom model into one merged identity.
    source_specs: List[Tuple[str, Path, Optional[List[str]]]] = []
    if rs_manual.exists():
        source_specs.append(("Manual", rs_manual, None))
    if rs_auto.exists():
        source_specs.append(("AutoRTS_total", rs_auto, None))
    if custom_wanted:
        source_specs.append(("Custom", rs_custom, custom_wanted))
    for model_name, model_course_dir in custom_model_outputs:
        source_specs.append(
            (
                f"CustomModel:{model_name}",
                Path(model_course_dir) / "rtstruct.dcm",
                custom_model_expected_rois[model_name],
            )
        )

    if not source_specs:
        logger.info(
            "No current RTSTRUCT source for %s; trying CT TotalSegmentator NIfTI fallback",
            course_dir,
        )
        return radiomics_for_course_ct_nifti_fallback(course_dir, config)

    if not has_ct_dicom:
        _invalidate_radiomics_outputs(output_path)
        raise RadiomicsCourseExtractionError(
            f"RTSTRUCT radiomics requires a CT DICOM series in {course_dir}"
        )

    # Load CT image
    try:
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(str(ct_dir))
        reader.SetFileNames(dicom_files)
        ct_image = reader.Execute()
    except Exception as e:
        logger.error(f"Failed to load CT image: {e}")
        _invalidate_radiomics_outputs(output_path)
        raise RadiomicsCourseExtractionError(
            f"CT series is present but unreadable for radiomics in {course_dir}: {e}"
        ) from e

    ct_info = {
        'spacing': tuple(ct_image.GetSpacing()),
        'origin': tuple(ct_image.GetOrigin()),
        'direction': tuple(ct_image.GetDirection()),
    }

    params_file = (
        str(config.radiomics_params_file)
        if getattr(config, "radiomics_params_file", None)
        else None
    )

    # Save CT image to a temporary NRRD. Serialization is required acquisition
    # preparation and must not be reclassified as an empty course.
    try:
        with tempfile.NamedTemporaryFile(suffix='.nrrd', delete=False) as ct_file:
            ct_image_path = ct_file.name
        sitk.WriteImage(ct_image, ct_image_path, useCompression=True)
    except Exception as exc:
        try:
            Path(ct_image_path).unlink(missing_ok=True)
        except UnboundLocalError:
            pass
        _invalidate_radiomics_outputs(output_path)
        raise RadiomicsCourseExtractionError(
            f"Failed to serialize CT image for radiomics in {course_dir}: {exc}"
        ) from exc

    tasks: List[Dict[str, Any]] = []
    preparation_failures: List[Dict[str, Any]] = []
    try:
        from rt_utils import RTStructBuilder

        min_voxels_limit, max_voxels_limit = _ct_voxel_limits(config)

        for segmentation_source, rs_file, selected_rois in source_specs:
            best_effort = not roi_source_is_required(segmentation_source)

            def _record_preparation_failure(
                roi_name: str,
                reason: str,
                *,
                status: str = "failed",
                failure_kind: str = "extraction_error",
            ) -> None:
                if (
                    not best_effort
                    and not extraction_status_is_nonfatal_for_required(status)
                ):
                    raise RadiomicsCourseExtractionError(reason)
                preparation_failures.append(
                    {
                        "source": segmentation_source,
                        "roi_name": str(roi_name),
                        "status": status,
                        "failure_kind": failure_kind,
                        "reason": reason,
                    }
                )
                logger.warning(
                    "Best-effort radiomics ROI %s/%s was not prepared: %s",
                    segmentation_source,
                    roi_name,
                    reason,
                )

            try:
                rtstruct = RTStructBuilder.create_from(
                    dicom_series_path=str(ct_dir),
                    rt_struct_path=str(rs_file),
                )
                roi_names = (
                    list(selected_rois)
                    if selected_rois is not None
                    else list(rtstruct.get_roi_names())
                )
            except Exception as exc:
                if best_effort:
                    try:
                        from .radiomics import _list_roi_names_dicom

                        advertised_rois = list(_list_roi_names_dicom(rs_file))
                    except Exception:
                        advertised_rois = []
                    if advertised_rois:
                        for advertised_roi in advertised_rois:
                            _record_preparation_failure(
                                advertised_roi,
                                f"Failed to construct RTSTRUCT reader for {segmentation_source} "
                                f"at {rs_file}: {exc}",
                            )
                        continue
                raise RadiomicsCourseExtractionError(
                    f"Failed to construct RTSTRUCT reader for {segmentation_source} at {rs_file}: {exc}"
                ) from exc
            if not roi_names:
                raise RadiomicsCourseExtractionError(
                    f"Required RTSTRUCT contains no named ROIs: {segmentation_source} at {rs_file}"
                )
            if len(roi_names) != len(set(roi_names)):
                raise RadiomicsCourseExtractionError(
                    f"Required RTSTRUCT has duplicate ROI identities: {segmentation_source} at {rs_file}"
                )

            for roi_name in roi_names:
                norm_key = _norm(roi_name)
                if norm_key in skip_rois:
                    logger.debug(
                        "Skipping radiomics for %s/%s (skip list)",
                        segmentation_source,
                        roi_name,
                    )
                    continue
                try:
                    mask = rtstruct.get_roi_mask_by_name(roi_name)
                except Exception as exc:
                    _record_preparation_failure(
                        roi_name,
                        f"Expected ROI {roi_name!r} in {segmentation_source} {rs_file} "
                        f"could not be read: {exc}",
                    )
                    continue
                if mask is None:
                    _record_preparation_failure(
                        roi_name,
                        f"Expected ROI {roi_name!r} in {segmentation_source} {rs_file} "
                        "did not provide a mask",
                    )
                    continue
                try:
                    mask_bool = np.asarray(mask).astype(bool)
                except Exception as exc:
                    _record_preparation_failure(
                        roi_name,
                        f"Expected ROI {roi_name!r} in {segmentation_source} {rs_file} "
                        f"could not be converted to a mask: {exc}",
                    )
                    continue
                if not mask_bool.any():
                    _record_preparation_failure(
                        roi_name,
                        f"Expected ROI {roi_name!r} in {segmentation_source} {rs_file} "
                        "produced an empty mask",
                        failure_kind="degenerate_mask",
                    )
                    continue

                voxel_count = int(mask_bool.sum())
                if voxel_count < min_voxels_limit:
                    _record_preparation_failure(
                        roi_name,
                        f"ROI contains {voxel_count} voxels; configured minimum is "
                        f"{min_voxels_limit}",
                        status="below_minimum_voxels",
                        failure_kind="degenerate_mask",
                    )
                    continue
                spacing = tuple(ct_info.get("spacing", (1.0, 1.0, 1.0)))
                try:
                    native_voxel_mm3 = (
                        float(spacing[0]) * float(spacing[1]) * float(spacing[2])
                    )
                except Exception:
                    native_voxel_mm3 = 1.0
                physical_volume_mm3 = float(voxel_count) * max(1e-9, native_voxel_mm3)
                estimated_voxels = physical_volume_mm3
                large_roi = norm_key.startswith("body") or (
                    estimated_voxels > float(max_voxels_limit)
                )
                if large_roi:
                    logger.info(
                        "ROI %s/%s is large (native=%d voxels, est@1mm=%.0f voxels, "
                        "cap=%d); using reduced radiomics settings",
                        segmentation_source,
                        roi_name,
                        voxel_count,
                        estimated_voxels,
                        int(max_voxels_limit),
                    )

                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=".nrrd", delete=False
                    ) as mask_file:
                        _write_mask_to_file(mask_bool, mask_file.name, ct_info)
                        mask_path = mask_file.name
                except Exception as exc:
                    _record_preparation_failure(
                        roi_name,
                        f"Failed to serialize mask {segmentation_source}/{roi_name} "
                        f"from {rs_file}: {exc}",
                    )
                    continue

                cropped_flag = mask_is_cropped(mask_bool)
                display_roi = (
                    roi_name
                    if (not cropped_flag or roi_name.endswith("__partial"))
                    else f"{roi_name}__partial"
                )
                tasks.append({
                    "image_path": ct_image_path,
                    "mask_path": mask_path,
                    "roi_name": display_roi,
                    "params_file": params_file,
                    "label": None,
                    "large_roi": bool(large_roi),
                    "metadata": {
                        "segmentation_source": segmentation_source,
                        "course_dir": str(course_dir),
                        "patient_id": course_dir.parent.name,
                        "course_id": course_dir.name,
                        "structure_cropped": cropped_flag,
                        "roi_original_name": roi_name,
                    },
                    "cleanup": True,
                    "ct_info": ct_info,
                })

        for failure in preparation_failures:
            tasks.append(
                {
                    "image_path": ct_image_path,
                    "mask_path": None,
                    "roi_name": failure["roi_name"],
                    "params_file": params_file,
                    "label": None,
                    "metadata": {
                        "segmentation_source": failure["source"],
                        "course_dir": str(course_dir),
                        "patient_id": course_dir.parent.name,
                        "course_id": course_dir.name,
                        "structure_cropped": False,
                        "roi_original_name": failure["roi_name"],
                    },
                    "precomputed_failure": failure,
                    "cleanup": False,
                    "ct_info": ct_info,
                }
            )

    except Exception as exc:
        logger.error("Failed to prepare radiomics masks for %s: %s", course_dir, exc)
        Path(ct_image_path).unlink(missing_ok=True)
        for task in tasks:
            mask_path = task.get('mask_path')
            if mask_path:
                Path(mask_path).unlink(missing_ok=True)
        _invalidate_radiomics_outputs(output_path)
        if isinstance(exc, RadiomicsCourseExtractionError):
            raise
        raise RadiomicsCourseExtractionError(
            f"RTSTRUCT radiomics preparation failed for {course_dir}: {exc}"
        ) from exc

    if not tasks:
        logger.warning("No eligible ROIs found in current RTSTRUCT inventory for %s", course_dir)
        Path(ct_image_path).unlink(missing_ok=True)
        _invalidate_radiomics_outputs(output_path)
        return None

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

    # Enable checkpointing for resumable extraction
    checkpoint_path = course_dir / "metadata" / "radiomics_ct_checkpoint.parquet"

    try:
        return process_radiomics_batch(
            tasks,
            output_path,
            sequential=sequential,
            max_workers=max_workers,
            checkpoint_path=checkpoint_path,
            enable_heartbeat=True,
            env_probe_timeout=getattr(config, "radiomics_env_probe_timeout", None),
        )
    finally:
        Path(ct_image_path).unlink(missing_ok=True)


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
    configured_mr_params = getattr(config, 'radiomics_params_file_mr', None) if config is not None else None
    mr_required = configured_mr_params is not None or params_file is not None

    if params_file is None and configured_mr_params is not None:
        params_file = Path(configured_mr_params)
    if params_file is not None and not Path(params_file).exists():
        _invalidate_radiomics_outputs(out_path)
        raise RadiomicsCourseExtractionError(
            f"Configured required MR radiomics parameter path is missing: {params_file}"
        )

    if not mr_root.exists():
        logger.debug("No MR directory in %s", course_dir)
        _invalidate_radiomics_outputs(out_path)
        if mr_required:
            raise RadiomicsCourseExtractionError(
                f"MR radiomics is required but no MR directory exists for {course_dir}"
            )
        return None

    tasks: List[Dict[str, Any]] = []
    temp_files: List[Path] = []
    min_voxels_limit, _ = _ct_voxel_limits(config)

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
            _invalidate_radiomics_outputs(out_path)
            raise RadiomicsCourseExtractionError(
                f"Failed to acquire/serialize required MR image {mr_nifti}: {exc}"
            ) from exc

        series_uid = series_root.name

        # Process each mask in Segmentation_TotalSegmentator
        for mask_path in sorted(seg_dir.glob("total_mr--*.nii.gz")):
            try:
                mask_img = sitk.ReadImage(str(mask_path))
                mask_arr = sitk.GetArrayFromImage(mask_img)
            except Exception as exc:
                _invalidate_radiomics_outputs(out_path)
                raise RadiomicsCourseExtractionError(
                    f"Required MR mask is unreadable: {mask_path}: {exc}"
                ) from exc

            mask_bool = mask_arr > 0
            if not mask_bool.any():
                _invalidate_radiomics_outputs(out_path)
                raise RadiomicsCourseExtractionError(
                    f"Required MR mask is empty: {mask_path}"
                )
            if int(mask_bool.sum()) < min_voxels_limit:
                logger.info(
                    "Skipping MR mask %s: %d voxels below minimum %d",
                    mask_path,
                    int(mask_bool.sum()),
                    min_voxels_limit,
                )
                continue

            mask_name = mask_path.name
            if mask_name.startswith("total_mr--"):
                roi_name = mask_name[10:]
            else:
                roi_name = mask_name
            if roi_name.endswith(".nii.gz"):
                roi_name = roi_name[:-7]

            try:
                mask_nrrd = tempfile.NamedTemporaryFile(
                    suffix=".nrrd", delete=False, prefix=f"mr_mask_{roi_name}_"
                )
                mask_nrrd.close()
                sitk.WriteImage(mask_img, mask_nrrd.name)
                temp_files.append(Path(mask_nrrd.name))
            except Exception as exc:
                _invalidate_radiomics_outputs(out_path)
                raise RadiomicsCourseExtractionError(
                    f"Failed to serialize required MR mask {mask_path}: {exc}"
                ) from exc

            tasks.append({
                'image_path': mr_image_path,
                'mask_path': mask_nrrd.name,
                'roi_name': roi_name,
                'params_file': str(params_file) if params_file else None,
                'cleanup': False,
                'metadata': {
                    'modality': 'MR',
                    'series_uid': series_uid,
                    'segmentation_source': 'AutoTS_total_mr',
                    'patient_id': course_dir.parent.name,
                    'course_id': course_dir.name,
                }
            })

    if not tasks:
        logger.debug("No eligible MR radiomics tasks for %s", course_dir)
        for tf in temp_files:
            try:
                tf.unlink()
            except OSError as exc:
                logger.debug("Failed to clean MR temporary file %s: %s", tf, exc)
        _invalidate_radiomics_outputs(out_path)
        if mr_required:
            raise RadiomicsCourseExtractionError(
                f"MR radiomics is required but produced no eligible tasks for {course_dir}"
            )
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

    # Enable checkpointing for resumable extraction
    checkpoint_path = mr_root / "radiomics_mr_checkpoint.parquet"

    try:
        result = process_radiomics_batch(
            tasks,
            out_path,
            sequential=False,
            max_workers=max_workers,
            checkpoint_path=checkpoint_path,
            enable_heartbeat=True,
            env_probe_timeout=getattr(config, "radiomics_env_probe_timeout", None),
        )
        if result is None and mr_required:
            _invalidate_radiomics_outputs(out_path)
            raise RadiomicsCourseExtractionError(
                f"MR radiomics is required but produced no workbook for {course_dir}"
            )
        return result
    finally:
        for tf in temp_files:
            try:
                tf.unlink()
            except OSError as exc:
                logger.debug("Failed to clean MR temporary file %s: %s", tf, exc)
