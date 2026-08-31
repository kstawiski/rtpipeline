from __future__ import annotations

import datetime
import functools
import io
import json
import hashlib
import logging
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import zipfile
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any, Optional, Dict, List, Sequence

import pydicom
import numpy as np

# Modern NumPy/SciPy work fine with TotalSegmentator - no compatibility shims needed

from .nifti_provenance import annotate as annotate_nifti_provenance
from .config import PipelineConfig
from .course_contract import load_course_contract
from .inventory import TS_TASK_BY_CLASS, manual_rtstruct_bindings_from_inventory, ts_tasks_for_image_class
from .layout import build_course_dirs

logger = logging.getLogger(__name__)
_TOTALSEG_OUTPUT_TYPE_FALLBACK = {"nifti", "dicom", "dicom_rtstruct", "dicom_seg"}
SEGMENTATION_SENTINEL_NAME = ".segmentation_done"
SEGMENTATION_STATUS_RELATIVE_PATH = Path("metadata/segmentation_status.json")
_SEGMENTATION_STATUSES = {"ok", "disabled", "failed"}
_TOTALSEG_RUN_STATE = threading.local()


def _set_totalseg_failure(category: str, reason: str) -> None:
    _TOTALSEG_RUN_STATE.failure = {
        "category": str(category),
        "reason": str(reason),
    }


def _clear_totalseg_failure() -> None:
    _TOTALSEG_RUN_STATE.failure = None


def _last_totalseg_failure() -> dict[str, str] | None:
    value = getattr(_TOTALSEG_RUN_STATE, "failure", None)
    return dict(value) if isinstance(value, dict) else None

# Lazy import for QC functions to avoid circular imports
_qc_module = None

def _get_qc_functions():
    """Lazy import of quality_control module to avoid circular imports."""
    global _qc_module
    if _qc_module is None:
        from . import quality_control as qc
        _qc_module = qc
    return _qc_module


# B5 [SAFE-3]: persist contrast phase as a queryable per-series manifest field, for the
# calibrated-CT classes only (CBCT is uncalibrated; 4DCT is projection/respiratory -> both
# excluded, per plan v2.4 §4 contrast-phase footnote).
_CONTRAST_PHASE_CLASSES = frozenset({"planning_ct", "diagnostic_ct", "petct_ct"})


def _detect_contrast_phase_safe(config, nifti_path) -> Optional[str]:
    """B5: classify a calibrated-CT NIfTI's contrast phase via TotalSegmentator's
    ``totalseg_get_phase`` and return the phase string (native / arterial_early /
    arterial_late / portal_venous / unknown) for persistence in the series manifest.
    Returns None on any failure -- this QC enrichment must NEVER fail segmentation."""
    try:
        qc = _get_qc_functions()
        result = qc.detect_contrast_phase(
            Path(nifti_path), conda_activate=getattr(config, "conda_activate", None)
        )
        if isinstance(result, dict):
            phase = result.get("phase")
            return str(phase) if phase else None
    except Exception as exc:
        logger.debug("B5 contrast-phase detection failed for %s: %s", nifti_path, exc)
    return None


def _run_vec(cmd: List[str], env: Optional[dict] = None, timeout: Optional[int] = None) -> bool:
    """Execute a command using argument list (shell=False) for better security. Returns True on success.

    This is the preferred method for executing external commands as it avoids shell injection risks.

    Args:
        cmd: Command as list of arguments (e.g., ['dcm2niix', '-z', 'y', '-o', '/path'])
        env: Optional environment variables
        timeout: Optional timeout in seconds (default: 3600 for TotalSegmentator, 300 for others)

    Raises:
        RuntimeError: If command fails or times out
    """
    if not cmd:
        raise ValueError("Empty command list")

    # Default timeout based on command type
    if timeout is None:
        cmd_name = cmd[0].lower() if cmd else ""
        if 'totalsegmentator' in cmd_name:
            timeout = int(os.environ.get('TOTALSEG_TIMEOUT', '3600'))  # 1 hour default
        elif 'dcm2niix' in cmd_name:
            timeout = int(os.environ.get('DCM2NIIX_TIMEOUT', '300'))  # 5 minutes default
        else:
            timeout = 1800  # 30 minutes default for other commands

    try:
        cmd_preview = ' '.join(cmd[:4]) + ('...' if len(cmd) > 4 else '')
        logger.debug("Running command (shell=False) with timeout=%ds: %s", timeout, cmd_preview)
        is_totalseg = "totalsegmentator" in cmd[0].lower()
        if not is_totalseg:
            subprocess.run(cmd, check=True, shell=False, env=env, timeout=timeout)
            return True

        # A temporary file avoids PIPE deadlocks from verbose nnU-Net children while
        # retaining enough evidence to classify CUDA exhaustion. Replay it unchanged
        # so production logs remain the operator's primary diagnostic surface.
        with tempfile.TemporaryFile(mode="w+b") as stderr_file:
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    shell=False,
                    env=env,
                    timeout=timeout,
                    stderr=stderr_file,
                )
            except subprocess.CalledProcessError as exc:
                stderr_file.flush()
                stderr_file.seek(0)
                stderr_bytes = stderr_file.read()
                if stderr_bytes:
                    stream = getattr(sys.stderr, "buffer", None)
                    if stream is not None:
                        stream.write(stderr_bytes)
                        stream.flush()
                    else:
                        sys.stderr.write(stderr_bytes.decode("utf-8", errors="replace"))
                stderr_text = stderr_bytes.decode("utf-8", errors="replace").lower()
                oom_patterns = (
                    "cuda error: out of memory",
                    "cuda out of memory",
                    "cudamalloc failed",
                    "cublas status alloc failed",
                )
                category = (
                    "cuda_out_of_memory"
                    if any(pattern in stderr_text for pattern in oom_patterns)
                    else "nonzero_exit"
                )
                reason = f"TotalSegmentator {category} with exit code {exc.returncode}"
                _set_totalseg_failure(category, reason)
                raise RuntimeError(reason) from exc
            else:
                stderr_file.flush()
                stderr_file.seek(0)
                stderr_bytes = stderr_file.read()
                if stderr_bytes:
                    stream = getattr(sys.stderr, "buffer", None)
                    if stream is not None:
                        stream.write(stderr_bytes)
                        stream.flush()
                    else:
                        sys.stderr.write(stderr_bytes.decode("utf-8", errors="replace"))
        return True
    except subprocess.TimeoutExpired:
        cmd_preview = ' '.join(cmd[:3])
        logger.error("Command timed out after %ds: %s...", timeout, cmd_preview)
        logger.error("This usually indicates a hung process or insufficient resources.")
        raise RuntimeError(f"Command timed out: {cmd_preview}...")
    except subprocess.CalledProcessError as e:
        cmd_preview = ' '.join(cmd[:3])
        logger.error("Command failed with exit code %d: %s...", e.returncode, cmd_preview)
        raise RuntimeError(f"Command failed with exit code {e.returncode}")


def _run(cmd: str, env: Optional[dict] = None, timeout: Optional[int] = None) -> bool:
    """Execute a trusted shell command with timeout protection. Returns True on success.

    The command is passed as an explicit argument to the selected shell rather
    than through ``subprocess``'s ``shell=True`` mode. Only use this helper with
    fully trusted, internally-generated commands.
    For external tool invocation, prefer _run_vec() with argument lists.

    Args:
        cmd: Shell command to execute (must be trusted, not user-controlled)
        env: Optional environment variables
        timeout: Optional timeout in seconds (default: 3600 for TotalSegmentator, 300 for others)
    """

    # Detect shell to use
    shell = os.environ.get('SHELL', '/bin/bash')
    if not os.path.isfile(shell):
        shell = shutil.which('bash') or shutil.which('sh') or '/bin/sh'

    # Default timeout based on command type
    if timeout is None:
        if 'TotalSegmentator' in cmd:
            timeout = int(os.environ.get('TOTALSEG_TIMEOUT', '3600'))  # 1 hour default
        elif 'dcm2niix' in cmd:
            timeout = int(os.environ.get('DCM2NIIX_TIMEOUT', '300'))  # 5 minutes default
        else:
            timeout = 1800  # 30 minutes default for other commands

    try:
        logger.debug(f"Running command with timeout={timeout}s: {cmd[:100]}...")
        # Note: Don't capture output with PIPE as it causes buffer deadlock
        # when child processes (like nnUNet workers in TotalSegmentator) produce output.
        # Let output stream directly to avoid hanging.
        subprocess.run([shell, "-lc", cmd], check=True, env=env, timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after {timeout}s: {cmd[:100]}...")
        logger.error(f"This usually indicates a hung process or insufficient resources.")
        raise RuntimeError(f"Command timed out: {cmd[:100]}...")
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}: {cmd[:100]}...")
        raise RuntimeError(f"Command failed with exit code {e.returncode}")


def _prefix(config: PipelineConfig) -> str:
    return f"{config.conda_activate} && " if config.conda_activate else ""


def _pkg_zip_bytes(name: str) -> Optional[bytes]:
    """Read a bundled ZIP from rtpipeline/ext inside the installed package.
    Returns bytes or None if not present.
    """
    try:
        res = importlib_resources.files('rtpipeline').joinpath('ext', name)
        if res.is_file():
            return res.read_bytes()
    except Exception:
        pass
    return None


def _find_ext_zip_fs(name: str) -> Optional[Path]:
    """Locate an ext/ ZIP by filename on filesystem for repo/dev runs."""
    # Try CWD/ext first (common when running from repo)
    cwd = Path.cwd() / "ext" / name
    if cwd.exists():
        return cwd
    # Try repo layout relative to this file (../ext)
    here = Path(__file__).resolve()
    for parent in list(here.parents)[:4]:
        cand = parent / "ext" / name
        if cand.exists():
            return cand
    return None


def _ensure_local_dcm2niix(config: PipelineConfig) -> Optional[Path]:
    """If dcm2niix is not available, try using packaged zips under ext/.
    Extracts into logs_root/bin and returns the binary path on success.
    """
    if config.conda_activate:
        return None
    # Pick ZIP by platform
    if sys.platform.startswith("win"):
        zip_name = "dcm2niix_win.zip"
        bin_name = "dcm2niix.exe"
    elif sys.platform == "darwin":
        zip_name = "dcm2niix_mac.zip"
        bin_name = "dcm2niix"
    else:
        zip_name = "dcm2niix_lnx.zip"
        bin_name = "dcm2niix"
    data = _pkg_zip_bytes(zip_name)
    zpath = None
    if data is None:
        zpath = _find_ext_zip_fs(zip_name)
        if zpath is None:
            logger.debug("No ext zip %s found (package or FS)", zip_name)
            return None
    dest = config.logs_root / "bin"

    def _existing_binary() -> Path | None:
        """Return an already-prepared, executable binary if one is present."""
        for base, dirs, files in os.walk(dest):
            dirs[:] = [d for d in dirs if d != "__MACOSX"]
            for fn in files:
                if fn.startswith("._"):
                    continue
                if fn.lower() == bin_name.lower():
                    candidate = Path(base) / fn
                    if os.access(candidate, os.X_OK):
                        return candidate
        return None

    # Every parallel worker calls this. Extracting straight onto the target path
    # meant one worker rewrote the binary while another was executing it, which
    # POSIX refuses with ETXTBSY ("Text file busy"), and CT conversion then failed
    # for that course. Observed 2,567 times in one 154-patient cohort run.
    #
    # Fast path first, so the common case never writes at all; then a lock so only
    # one worker extracts; then re-check inside the lock, because another worker
    # may have finished while this one waited.
    ready = _existing_binary()
    if ready is not None:
        logger.debug("Using already-prepared bundled dcm2niix at %s", ready)
        return ready

    try:
        dest.mkdir(parents=True, exist_ok=True)
        dest_resolved = dest.resolve()
    except Exception as e:
        logger.error("Failed to prepare bundled dcm2niix: %s", e)
        return None

    lock_path = dest / f".{bin_name}.prepare.lock"
    try:
        from filelock import FileLock, Timeout as _LockTimeout
    except Exception:  # pragma: no cover - filelock is a declared dependency
        FileLock = None
        _LockTimeout = ()

    lock_ctx = FileLock(str(lock_path), timeout=300) if FileLock is not None else None

    try:
        if lock_ctx is not None:
            lock_ctx.acquire()
    except Exception as e:
        logger.warning("Could not lock bundled dcm2niix preparation (%s); proceeding unlocked", e)
        lock_ctx = None

    try:
        ready = _existing_binary()
        if ready is not None:
            return ready

        def _safe_extract_bundled_zip(zf: zipfile.ZipFile, dest_root: Path) -> None:
            """Extract bundled zip with path traversal protection."""
            for member in zf.infolist():
                raw_name = member.filename or ""
                # Check for path traversal
                if ".." in raw_name or raw_name.startswith("/"):
                    logger.warning("Skipping unsafe bundled zip entry: %s", raw_name)
                    continue
                # Check for symlinks
                try:
                    mode = (member.external_attr >> 16) & 0o177777
                    if stat.S_ISLNK(mode):
                        logger.warning("Skipping symlink in bundled zip: %s", raw_name)
                        continue
                except Exception:
                    pass
                # Verify target stays within dest
                target = (dest_root / raw_name).resolve()
                try:
                    target.relative_to(dest_root)
                except ValueError:
                    logger.warning("Skipping zip entry that escapes dest: %s", raw_name)
                    continue
                # Extract
                if raw_name.endswith('/'):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member, 'r') as src, open(target, 'wb') as dst:
                        import shutil as _shutil
                        _shutil.copyfileobj(src, dst)

        # Extract into a private staging directory and publish by rename, so a
        # concurrent reader never observes a half-written binary and never has
        # its running executable rewritten underneath it.
        staging = Path(tempfile.mkdtemp(prefix=".stage-", dir=str(dest_resolved)))
        staging_resolved = staging.resolve()
        try:
            if data is not None:
                with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
                    _safe_extract_bundled_zip(zf, staging_resolved)
            else:
                with zipfile.ZipFile(zpath, "r") as zf:
                    _safe_extract_bundled_zip(zf, staging_resolved)

            for base, dirs, files in os.walk(staging_resolved):
                dirs[:] = [d for d in dirs if d != "__MACOSX"]
                for fn in files:
                    if fn.startswith("._") or fn.lower() != bin_name.lower():
                        continue
                    staged = Path(base) / fn
                    try:
                        mode = os.stat(staged).st_mode
                        os.chmod(staged, mode | stat.S_IEXEC)
                    except Exception:
                        pass
                    published = dest_resolved / bin_name
                    os.replace(staged, published)
                    logger.info("Using bundled dcm2niix at %s", published)
                    return published
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        # Search for binary inside extracted tree
        candidates = []
        for base, dirs, files in os.walk(dest):
            # Filter out macOS resource fork directories
            dirs[:] = [d for d in dirs if d != "__MACOSX"]
            for fn in files:
                if fn.startswith("._"):
                    continue
                if fn.lower() == bin_name.lower():
                    candidates.append(Path(base) / fn)
        if not candidates:
            if zpath is not None:
                logger.warning("Extracted %s but did not find %s", zpath.name, bin_name)
            else:
                logger.warning("Extracted bundled data but did not find %s", bin_name)
            return None
        bin_path = candidates[0]
        # Ensure executable on POSIX
        try:
            if os.name != "nt":
                mode = os.stat(bin_path).st_mode
                os.chmod(bin_path, mode | stat.S_IEXEC)
        except Exception as exc:
            logger.debug("Failed to set executable permission on %s: %s", bin_path, exc)
        logger.info("Using bundled dcm2niix at %s", bin_path)
        return bin_path
    except Exception as e:
        logger.error("Failed to prepare bundled dcm2niix: %s", e)
        return None
    finally:
        if lock_ctx is not None:
            try:
                lock_ctx.release()
            except Exception:
                pass


def run_dcm2niix(
    config: PipelineConfig,
    dicom_dir: Path,
    nifti_out: Path,
    recursive_depth: int | None = None,
) -> Optional[Path]:
    nifti_out.mkdir(parents=True, exist_ok=True)
    # Verify command availability (when no conda prefix is used)
    local_cmd = None
    if not config.conda_activate and shutil.which(config.dcm2niix_cmd) is None:
        # Try to use bundled binary from ext/
        local = _ensure_local_dcm2niix(config)
        if local is None:
            logger.warning("dcm2niix command '%s' not found; skipping NIfTI conversion", config.dcm2niix_cmd)
            return None
        local_cmd = str(local)
    cmd_name = local_cmd or config.dcm2niix_cmd

    # Use shell=False when no conda prefix is needed (more secure)
    if not config.conda_activate:
        # Build command as argument list for _run_vec (shell=False)
        cmd_list = [cmd_name]
        if recursive_depth is not None:
            cmd_list.extend(["-d", str(max(0, int(recursive_depth)))])
        cmd_list.extend(["-z", "y", "-o", str(nifti_out), str(dicom_dir)])
        logger.info("Running dcm2niix (shell=False): %s", " ".join(cmd_list[:4]) + "...")
        try:
            ok = _run_vec(cmd_list)
        except RuntimeError:
            ok = False
    else:
        # Run the trusted activation fragment through the explicit shell helper.
        depth_args = ""
        if recursive_depth is not None:
            depth_args = f" -d {max(0, int(recursive_depth))}"
        if local_cmd:
            inner_cmd = f'{shlex.quote(cmd_name)}{depth_args} -z y -o {shlex.quote(str(nifti_out))} {shlex.quote(str(dicom_dir))}'
            cmd = f"{_prefix(config)}bash -c {shlex.quote(inner_cmd)}"
        else:
            cmd = f"{_prefix(config)}{shlex.quote(cmd_name)}{depth_args} -z y -o {shlex.quote(str(nifti_out))} {shlex.quote(str(dicom_dir))}"
        logger.info("Running dcm2niix (with conda): %s", cmd)
        try:
            ok = _run(cmd)
        except RuntimeError:
            ok = False

    if not ok:
        logger.warning("dcm2niix failed; continuing with DICOM-only segmentation")
        return None
    # pick largest nii(.gz) – deterministic selection of the primary volume
    nii_files = [fn for fn in os.listdir(nifti_out)
                 if fn.endswith(".nii") or fn.endswith(".nii.gz")]
    nii_files.sort(key=lambda fn: os.path.getsize(nifti_out / fn), reverse=True)
    if nii_files:
        return nifti_out / nii_files[0]
    return None

def _validate_totalseg_environment(config: PipelineConfig) -> bool:
    """Validate TotalSegmentator environment and dependencies."""
    if config.conda_activate:
        return True  # Assume conda environment is properly configured
    
    if not shutil.which(config.totalseg_cmd):
        logger.error("TotalSegmentator command '%s' not found in PATH", config.totalseg_cmd)
        return False
    
    # Check for numpy compatibility issue
    try:
        result = subprocess.run(
            [config.totalseg_cmd, "--help"], 
            capture_output=True, 
            timeout=30
        )
        if result.returncode != 0:
            stderr = result.stderr.decode() if result.stderr else ""
            normalized = (stderr or "").lower()
            if "np.isdtype" in normalized or "has no attribute 'isdtype'" in normalized:
                logger.error(
                    "TotalSegmentator detected NumPy < 2.0 (missing np.isdtype). "
                    "Please upgrade the environment, e.g. 'pip install \"numpy>=2.0\"'."
                )
                return False
            if "numpy" in normalized:
                logger.error("TotalSegmentator reported a NumPy-related error: %s", stderr.strip())
                return False
            logger.warning("TotalSegmentator help command failed: %s", stderr.strip())
        return True
    except subprocess.TimeoutExpired:
        logger.warning("TotalSegmentator validation timed out")
        return True  # Don't fail completely on timeout
    except Exception as e:
        logger.warning("TotalSegmentator validation failed: %s", e)
        return True  # Don't fail completely on validation error


@functools.lru_cache(maxsize=32)
def _totalseg_supported_output_types_cached(prefix: str, cmd: str) -> set[str]:
    """Inspect TotalSegmentator CLI to determine supported output types."""
    shell = os.environ.get('SHELL', '/bin/bash')
    if not os.path.isfile(shell):
        shell = shutil.which('bash') or shutil.which('sh') or '/bin/sh'
    base_command = cmd.strip() or "TotalSegmentator"
    probe = f"{prefix}{base_command} --help"
    command = [shell, "-lc", probe] if prefix else [*shlex.split(base_command), "--help"]
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
        )
    except Exception as exc:
        logger.debug("Failed to inspect TotalSegmentator CLI: %s", exc)
        return set(_TOTALSEG_OUTPUT_TYPE_FALLBACK)

    output = result.stdout or ""
    match = re.search(r"Choices:\s*(.*?)(?:\.\s|$)", output, re.IGNORECASE | re.DOTALL)
    if not match:
        match = re.search(r"-ot\s*\{\s*([^}]*)\}", output)
    if not match:
        return set(_TOTALSEG_OUTPUT_TYPE_FALLBACK)

    options = {
        opt.strip()
        for opt in re.split(r"[\s,]+", match.group(1))
        if opt.strip() and re.match(r"^[A-Za-z0-9_]+$", opt.strip())
    }
    return options or set(_TOTALSEG_OUTPUT_TYPE_FALLBACK)


def _totalseg_supported_output_types(config: PipelineConfig) -> set[str]:
    prefix = f"{config.conda_activate} && " if config.conda_activate else ""
    cmd = config.totalseg_cmd or "TotalSegmentator"
    return _totalseg_supported_output_types_cached(prefix, cmd)


def _command_with_device(command: Sequence[str], device: str) -> list[str]:
    """Return one TotalSegmentator command with exactly one device selection."""

    updated = list(command)
    device_indices = [index for index, value in enumerate(updated) if value == "-d"]
    if device_indices:
        first = device_indices[0]
        if first + 1 >= len(updated):
            updated.append(device)
        else:
            updated[first + 1] = device
        for index in reversed(device_indices[1:]):
            del updated[index : min(index + 2, len(updated))]
    else:
        updated.extend(["-d", device])
    return updated


def run_totalsegmentator(
    config: PipelineConfig,
    input_path: Path,
    output_path: Path,
    output_type: str,
    task: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> bool:
    """Run TotalSegmentator directly without compatibility wrapper."""

    _clear_totalseg_failure()

    # Use TotalSegmentator directly - modern NumPy/SciPy work fine
    supported_types = _totalseg_supported_output_types(config)
    if output_type and output_type not in supported_types:
        logger.info(
            "TotalSegmentator output_type '%s' not supported by current CLI; skipping direct export",
            output_type,
        )
        _set_totalseg_failure(
            "unsupported_output_type",
            f"TotalSegmentator does not support output type {output_type!r}",
        )
        return False

    totalseg_cmd = config.totalseg_cmd or "TotalSegmentator"
    device = getattr(config, "totalseg_device", "gpu") or "gpu"
    cmd_parts = [
        totalseg_cmd,
        "-i", str(input_path),
        "-o", str(output_path),
        "-ot", output_type,
    ]

    if task:
        cmd_parts.extend(["--task", task])

    if extra_args:
        cmd_parts.extend(str(arg) for arg in extra_args)

    if getattr(config, "totalseg_fast", False):
        cmd_parts.append("--fast")

    if getattr(config, "totalseg_roi_subset", None):
        roi_tokens = [
            token
            for token in re.split(r"[\s,]+", str(config.totalseg_roi_subset).strip())
            if token
        ]
        if roi_tokens:
            cmd_parts.extend(["--roi_subset", *roi_tokens])

    if getattr(config, "totalseg_license_key", None):
        logger.debug("Using TotalSegmentator license key from config")

    if getattr(config, "totalseg_force_split", True):
        cmd_parts.append("--force_split")

    if device:
        cmd_parts.extend(["-d", device])

    nr_thr_resamp = getattr(config, "totalseg_nr_thr_resamp", None)
    if nr_thr_resamp:
        try:
            nr_thr_resamp_int = max(1, int(nr_thr_resamp))
            cmd_parts.extend(["--nr_thr_resamp", str(nr_thr_resamp_int)])
        except (TypeError, ValueError):
            pass

    nr_thr_saving = getattr(config, "totalseg_nr_thr_saving", None)
    if nr_thr_saving:
        try:
            nr_thr_saving_int = max(1, int(nr_thr_saving))
            cmd_parts.extend(["--nr_thr_saving", str(nr_thr_saving_int)])
        except (TypeError, ValueError):
            pass

    # Build environment variables
    env = os.environ.copy()
    thread_limit = getattr(config, "segmentation_thread_limit", None)
    thread_vars = (
        'OMP_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS',
        'NUMEXPR_NUM_THREADS',
        'NUMBA_NUM_THREADS',
    )
    if thread_limit is not None:
        try:
            thread_limit_int = max(1, int(thread_limit))
        except (TypeError, ValueError):
            thread_limit_int = 1
        thread_str = str(thread_limit_int)
        for var in thread_vars:
            env[var] = thread_str

    if getattr(config, "totalseg_license_key", None):
        env.setdefault("TOTALSEG_LICENSE", str(config.totalseg_license_key))
    if getattr(config, "totalseg_weights_dir", None):
        env.setdefault("TOTALSEG_WEIGHTS_PATH", str(config.totalseg_weights_dir))

    # Constrain TotalSegmentator worker behaviour to avoid Docker spawning issues
    def _to_env_int(value: Optional[int], fallback: int) -> str:
        try:
            if value is None:
                return str(fallback)
            coerced = int(value)
            return str(coerced if coerced > 0 else fallback)
        except (TypeError, ValueError):
            return str(fallback)

    env.setdefault("TOTALSEG_NUM_PROCESSES_PREPROCESSING", _to_env_int(getattr(config, "totalseg_num_proc_pre", None), 1))
    env.setdefault("TOTALSEG_NUM_PROCESSES_SEGMENTATION_EXPORT", _to_env_int(getattr(config, "totalseg_num_proc_export", None), 1))
    env.setdefault("TOTALSEG_FORCE_TORCH_NUM_THREADS", "1")
    env.setdefault("TOTALSEG_PRELOAD_WEIGHTS", "1")
    if device:
        env.setdefault("TOTALSEG_ACCELERATOR", device if device != "gpu" else "cuda")
        env.setdefault("TOTALSEG_DEVICE", device)
    # Mirror nnU-Net expectations (helps when env variables missing)
    weights_env = env.get("TOTALSEG_WEIGHTS_PATH")
    if weights_env:
        env.setdefault("nnUNet_results", weights_env)
        env.setdefault("nnUNet_preprocessed", weights_env)
        env.setdefault("nnUNet_raw", weights_env)

    # Choose execution method based on whether conda activation is needed
    use_shell = bool(config.conda_activate)

    if use_shell:
        cmd = "{}{}".format(
            _prefix(config),
            " ".join(shlex.quote(part) for part in cmd_parts),
        )
        logger.info("Running TotalSegmentator (%s, activated shell): %s", output_type, cmd)
        try:
            ok = _run(cmd, env=env)
        except RuntimeError as exc:
            if _last_totalseg_failure() is None:
                _set_totalseg_failure(
                    "timeout" if "timed out" in str(exc).lower() else "nonzero_exit",
                    str(exc),
                )
            ok = False
    else:
        cmd_preview = " ".join(cmd_parts[:5]) + ("..." if len(cmd_parts) > 5 else "")
        logger.info("Running TotalSegmentator (%s, shell=False): %s", output_type, cmd_preview)
        try:
            ok = _run_vec(cmd_parts, env=env)
        except RuntimeError as exc:
            if _last_totalseg_failure() is None:
                _set_totalseg_failure(
                    "timeout" if "timed out" in str(exc).lower() else "nonzero_exit",
                    str(exc),
                )
            ok = False

    if not ok:
        primary_failure = _last_totalseg_failure() or {
            "category": "nonzero_exit",
            "reason": "TotalSegmentator returned an unsuccessful process result",
        }
        _set_totalseg_failure(primary_failure["category"], primary_failure["reason"])
        if getattr(config, "totalseg_allow_fallback", False):
            logger.info("Retrying TotalSegmentator with CPU-only and single-process env")
            env_retry = env.copy()
            env_retry["CUDA_VISIBLE_DEVICES"] = "-1"
            env_retry["TOTALSEG_ACCELERATOR"] = "cpu"
            env_retry["TOTALSEG_DEVICE"] = "cpu"
            cmd_parts_retry = _command_with_device(cmd_parts, "cpu")

            if use_shell:
                cmd_retry = "{}{}".format(
                    _prefix(config),
                    " ".join(shlex.quote(part) for part in cmd_parts_retry),
                )
                try:
                    ok = _run(cmd_retry, env=env_retry)
                except RuntimeError as exc:
                    _set_totalseg_failure(
                        "cpu_fallback_failed",
                        f"{primary_failure['reason']}; CPU fallback failed: {exc}",
                    )
                    ok = False
            else:
                try:
                    ok = _run_vec(cmd_parts_retry, env=env_retry)
                except RuntimeError as exc:
                    _set_totalseg_failure(
                        "cpu_fallback_failed",
                        f"{primary_failure['reason']}; CPU fallback failed: {exc}",
                    )
                    ok = False
            if ok:
                logger.warning(
                    "TotalSegmentator recovered through the operator-enabled CPU fallback"
                )
                _clear_totalseg_failure()
            elif (_last_totalseg_failure() or {}).get("category") != "cpu_fallback_failed":
                _set_totalseg_failure(
                    "cpu_fallback_failed",
                    f"{primary_failure['reason']}; CPU fallback returned an unsuccessful process result",
                )
        else:
            logger.error("TotalSegmentator failed and fallback is disabled.")
    else:
        _clear_totalseg_failure()

    return ok


def _sanitize_token(token: str) -> str:
    token = token.strip().replace(" ", "_")
    cleaned = []
    for ch in token:
        if ch.isalnum() or ch in {'.', '-', '_'}:
            cleaned.append(ch)
        else:
            cleaned.append('_')
    result = ''.join(cleaned)
    while '__' in result:
        result = result.replace('__', '_')
    return result.strip('_')[:80] or "CT"


def _derive_nifti_name(ct_dir: Path) -> str:
    try:
        first_file = next(p for p in sorted(ct_dir.iterdir()) if p.is_file())
    except StopIteration:
        return "CT"
    try:
        ds = pydicom.dcmread(str(first_file), stop_before_pixels=True)
    except Exception:
        return "CT"

    desc = (
        str(getattr(ds, "SeriesDescription", ""))
        or str(getattr(ds, "StudyDescription", ""))
        or str(getattr(ds, "BodyPartExamined", ""))
    )
    desc = _sanitize_token(desc)

    thickness_token = ""
    try:
        thickness = getattr(ds, "SliceThickness", None)
        if thickness not in (None, ""):
            thickness_token = f"{float(thickness):.1f}".rstrip("0").rstrip(".")
    except Exception:
        pass

    study_date = str(getattr(ds, "StudyDate", "") or getattr(ds, "SeriesDate", ""))
    series_uid = str(getattr(ds, "SeriesInstanceUID", ""))
    parts = []
    if desc:
        parts.append(desc)
    if thickness_token:
        parts.append(thickness_token)
    if study_date:
        parts.append(_sanitize_token(study_date))
    elif series_uid:
        parts.append(_sanitize_token(series_uid[-6:]))
    name = "_".join(part for part in parts if part)
    if not name:
        name = _sanitize_token(series_uid[-8:] if series_uid else "CT")
    return name or "CT"


def _collect_series_metadata(ct_dir: Path) -> dict:
    metadata = {
        "study_instance_uid": "",
        "series_instance_uid": "",
        "instances": [],
        "modality": "",
        "geometry": {},
    }
    for dcm_path in sorted(p for p in ct_dir.iterdir() if p.is_file()):
        try:
            ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
        except Exception:
            continue
        if not metadata["study_instance_uid"]:
            metadata["study_instance_uid"] = str(getattr(ds, "StudyInstanceUID", ""))
        if not metadata["series_instance_uid"]:
            metadata["series_instance_uid"] = str(getattr(ds, "SeriesInstanceUID", ""))
        if not metadata["modality"]:
            metadata["modality"] = str(getattr(ds, "Modality", ""))
        if not metadata["geometry"]:
            def _numbers(value):
                if value in (None, "") or not isinstance(value, (list, tuple)):
                    return None
                try:
                    return [float(item) for item in value]
                except (TypeError, ValueError):
                    return None
            metadata["geometry"] = {
                "rows": int(getattr(ds, "Rows", 0) or 0),
                "columns": int(getattr(ds, "Columns", 0) or 0),
                "pixel_spacing": _numbers(getattr(ds, "PixelSpacing", None)),
                "image_orientation_patient": _numbers(
                    getattr(ds, "ImageOrientationPatient", None)
                ),
                "slice_thickness": (
                    float(getattr(ds, "SliceThickness"))
                    if getattr(ds, "SliceThickness", None) not in (None, "")
                    else None
                ),
            }
        sop = getattr(ds, "SOPInstanceUID", None)
        if sop:
            metadata["instances"].append(str(sop))
    metadata["instance_count"] = len(metadata["instances"])
    concat = "".join(metadata["instances"])
    metadata["sop_hash"] = hashlib.sha256(concat.encode("utf-8")).hexdigest() if concat else ""
    return metadata


def _ensure_ct_nifti(
    config: PipelineConfig,
    ct_dir: Path,
    nifti_dir: Path,
    force: bool = False,
    dcm2niix_depth: int | None = None,
) -> Optional[Path]:
    nifti_dir.mkdir(parents=True, exist_ok=True)
    metadata = _collect_series_metadata(ct_dir)
    base = _derive_nifti_name(ct_dir)
    candidate = base
    suffix_counter = 1
    while True:
        target = nifti_dir / f"{candidate}.nii.gz"
        meta_path = nifti_dir / f"{candidate}.metadata.json"
        if target.exists() and meta_path.exists():
            try:
                existing = json.loads(meta_path.read_text(encoding='utf-8'))
            except Exception:
                existing = {}
            if existing.get("series_instance_uid") == metadata["series_instance_uid"]:
                base = candidate
                break
        if not target.exists():
            base = candidate
            break
        suffix_counter += 1
        candidate = f"{base}_{suffix_counter}"

    target = nifti_dir / f"{base}.nii.gz"
    existing_sidecar: dict[str, Any] = {}
    meta_path = nifti_dir / f"{base}.metadata.json"
    if meta_path.exists():
        try:
            parsed_sidecar = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(parsed_sidecar, dict):
                existing_sidecar = parsed_sidecar
        except Exception:
            existing_sidecar = {}

    regenerated = not target.exists() or force
    if regenerated:
        tmp_dir = nifti_dir / ".tmp_dcm2niix"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        generated = run_dcm2niix(config, ct_dir, tmp_dir, recursive_depth=dcm2niix_depth)
        if generated is None:
            logger.error("dcm2niix failed for %s", ct_dir)
            return None
        if target.exists():
            target.unlink()
        shutil.move(str(generated), str(target))
        # Clean up temp directory completely
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Shared with organize's related-series conversion so both sidecars carry the
    # provenance the course contract requires.
    annotate_nifti_provenance(
        metadata,
        target,
        ct_dir,
        regenerated=regenerated,
        existing_sidecar=existing_sidecar,
    )
    meta_path = nifti_dir / f"{base}.metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return target


def _strip_nifti_base(nifti_path: Path) -> str:
    name = nifti_path.name
    if name.endswith('.nii.gz'):
        return name[:-7]
    if name.endswith('.nii'):
        return name[:-4]
    return nifti_path.stem


def _clear_previous_masks(seg_dir: Path, base_name: str, model: str) -> None:
    prefixes = (f"{model}--", f"{base_name}--{model}--")
    for prefix in prefixes:
        for existing in list(seg_dir.glob(f"{prefix}*")):
            try:
                existing.unlink()
            except Exception:
                pass


@functools.lru_cache(maxsize=1)
def _totalseg_version() -> str:
    """Installed TotalSegmentator version, for per-series segmentation provenance.

    Cached because it is identical for every series in a run. Returns ``"unknown"``
    if the package metadata cannot be read, so provenance never breaks segmentation.
    """
    try:
        from importlib.metadata import version as _pkg_version
        return str(_pkg_version("TotalSegmentator"))
    except Exception:
        return "unknown"


def _write_ts_version_sidecar(dest: Path, model: str) -> None:
    """Record the TotalSegmentator version used for this series/model alongside its masks.

    Lets cohort-wide version uniformity be audited from the outputs themselves
    rather than assumed from the run environment. Fail-soft: a write error is
    logged at debug and never aborts segmentation.
    """
    try:
        prov = {
            "totalsegmentator_version": _totalseg_version(),
            "model": model,
            "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        }
        (dest / f"{model}--ts_version.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - provenance must never break seg
        logger.debug("Could not write TS version sidecar in %s: %s", dest, exc)


def _materialize_masks(source: Path, dest: Path, base_name: str, model: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    _clear_previous_masks(dest, base_name, model)

    multi_label = source / "segmentations.nii.gz"
    if multi_label.exists():
        shutil.copy2(multi_label, dest / f"{model}--multilabel.nii.gz")

    json_path = source / "segmentations.json"
    if json_path.exists():
        shutil.copy2(json_path, dest / f"{model}--segmentations.json")

    masks_root = source / "segmentations"
    if masks_root.exists():
        for mask in masks_root.glob("*.nii*"):
            dest_mask = dest / f"{model}--{mask.name}"
            shutil.copy2(mask, dest_mask)

    for mask in source.glob("*.nii*"):
        if mask.name in {"segmentations.nii", "segmentations.nii.gz"}:
            continue
        if (source / "segmentations").exists() and mask.parent == source / "segmentations":
            continue
        dest_mask = dest / f"{model}--{mask.name}"
        shutil.copy2(mask, dest_mask)

    _write_ts_version_sidecar(dest, model)


def _ensure_model_rtstruct_from_masks(
    ct_dir: Path,
    seg_dir: Path,
    base_name: str,
    model: str,
) -> Optional[Path]:
    """Materialize a missing model RTSTRUCT from validated masks only."""
    target = seg_dir / f"{base_name}--{model}.dcm"
    if target.is_file():
        return target
    try:
        from rt_utils import RTStructBuilder
        from .auto_rtstruct import (
            _geometry_compatible,
            _iter_binary_masks,
            _load_ct_image,
            _pretty_roi_name,
            _resample_to_reference,
            _unique_roi_name,
            _write_rtstruct_atomic,
        )
        import SimpleITK as sitk
    except Exception as exc:
        logger.warning("Cannot build derived RTSTRUCT from masks: %s", exc)
        return None

    try:
        ct_img = _load_ct_image(ct_dir)
        if ct_img is None:
            return None

        selected: dict[str, tuple[int, sitk.Image]] = {}
        current_prefix = f"{model}--"
        legacy_prefix = f"{base_name}--{model}--"
        for raw_name, mask_img in _iter_binary_masks(seg_dir):
            if raw_name.startswith(current_prefix):
                rank = 0
                roi = raw_name[len(current_prefix):]
            elif raw_name.startswith(legacy_prefix):
                rank = 1
                roi = raw_name[len(legacy_prefix):]
            else:
                continue
            if roi and (roi not in selected or rank < selected[roi][0]):
                selected[roi] = (rank, mask_img)
        if not selected:
            return None
        selected_images = {roi: item[1] for roi, item in selected.items()}
        if any(not _geometry_compatible(image, ct_img) for image in selected_images.values()):
            logger.warning("Cannot derive %s from masks with incompatible geometry", target)
            return None

        rtstruct = RTStructBuilder.create_new(dicom_series_path=str(ct_dir))
        used_names: set[str] = set()
        for raw_name, (_rank, mask_img) in sorted(selected.items()):
            image = _resample_to_reference(mask_img, ct_img)
            array = np.moveaxis(sitk.GetArrayFromImage(image), 0, -1)
            mask = array > 0
            if not np.any(mask):
                continue
            roi_name = _unique_roi_name(_pretty_roi_name(raw_name), used_names)
            rtstruct.add_roi(mask=mask, name=roi_name)
            used_names.add(roi_name)
        if not used_names:
            return None
        _write_rtstruct_atomic(target, rtstruct.save)
        return target if target.is_file() else None
    except Exception as exc:
        logger.warning("Failed to derive %s from masks: %s", target, exc)
        return None


def _write_manifest_atomic(path: Path, data: dict) -> None:
    """Write ``manifest.json`` atomically (unique temp file + ``os.replace``) so a process killed
    mid-write can never leave `_series_segmentation_ready` looking at a truncated/partial file.

    The temp name is PID-qualified so concurrent same-dir writers cannot collide, and any temp
    left behind by a failed write is removed in ``finally`` (the successful ``os.replace`` consumes
    the temp, so cleanup is a no-op on the happy path)."""
    tmp_path = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _series_segmentation_ready(base_dir: Path, base_name: str, model: str) -> bool:
    """Return True only if `model`'s masks are present AND `manifest.json` confirms completeness.

    Mask-file presence alone is not a reliable completion signal: a run killed mid-write
    (e.g. during `_materialize_masks`, which copies masks one at a time) can leave a partial
    set of mask files on disk. `manifest.json` is written last, after every mask for a model
    has been copied, so require it to also exist, parse as valid JSON, contain an entry for
    this `model`, and have a NON-EMPTY mask list whose every entry is present on disk. Any
    deviation (manifest missing/corrupt, no entry for this model, an empty mask list — which a
    failed rtstruct-only run records — or a recorded mask absent from disk) means the
    segmentation is incomplete and must be re-run.
    """
    mask_files = list(base_dir.glob(f"{model}--*.nii*")) or list(base_dir.glob(f"{base_name}--{model}--*.nii*"))
    if not mask_files:
        return False
    manifest_path = base_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    model_entries = manifest.get("models") if isinstance(manifest, dict) else None
    if not isinstance(model_entries, list):
        return False
    for entry in model_entries:
        if not isinstance(entry, dict) or entry.get("model") != model:
            continue
        masks = entry.get("masks") or []
        # Require a NON-EMPTY mask list: a failed run records rtstruct-only (masks==[]),
        # and an empty list would otherwise vacuously satisfy `all(...)` and forge readiness.
        if not isinstance(masks, list) or any(not isinstance(mask, str) or not mask.strip() for mask in masks):
            return False
        root = base_dir.resolve(strict=False)
        for mask in masks:
            path = (base_dir / mask).resolve(strict=False)
            try:
                path.relative_to(root)
            except ValueError:
                return False
            if not path.is_file():
                return False
        return bool(masks)
    return False


def _segmentation_resume_record_path(course_dir: Path) -> Path:
    return Path(course_dir) / "metadata" / "segmentation_resume.json"


def record_segmentation_resume_decision(
    course_dir: Path,
    decisions: dict[str, Any],
    *,
    source: Optional[dict[str, Any]] = None,
    record_path: Optional[Path] = None,
) -> Path:
    """Persist the content-based resume decision for later audit.

    The record is deliberately separate from the completion sentinel. A sentinel
    answers whether a stage published success. This record answers which outputs
    were reused or rebuilt, and which input identity supported that choice.
    """
    path = Path(record_path) if record_path is not None else _segmentation_resume_record_path(Path(course_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:
            existing = {}
    existing.setdefault("version", 1)
    existing.setdefault("course_dir", str(Path(course_dir)))
    if source:
        recorded_source = existing.get("source")
        if not isinstance(recorded_source, dict):
            recorded_source = {}
        recorded_source.update(source)
        existing["source"] = recorded_source
    stored = existing.setdefault("decisions", {})
    if not isinstance(stored, dict):
        stored = {}
        existing["decisions"] = stored
    stored.update(decisions)
    existing["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _write_manifest_atomic(path, existing)
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_geometry(image: Any) -> dict[str, tuple[float, ...] | tuple[int, ...]]:
    return {
        "size": tuple(int(value) for value in image.GetSize()),
        "spacing": tuple(float(value) for value in image.GetSpacing()),
        "origin": tuple(float(value) for value in image.GetOrigin()),
        "direction": tuple(float(value) for value in image.GetDirection()),
    }


def _same_image_geometry(first: Any, second: Any, tolerance: float = 1e-4) -> bool:
    try:
        left = _image_geometry(first)
        right = _image_geometry(second)
        if left["size"] != right["size"]:
            return False
        for key in ("spacing", "origin", "direction"):
            a = left[key]
            b = right[key]
            if len(a) != len(b) or any(abs(float(x) - float(y)) > tolerance for x, y in zip(a, b)):
                return False
        return True
    except Exception:
        return False


def _parse_timestamp(value: object) -> Optional[datetime.datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed


def _legacy_segmentation_provenance_is_safe(
    base_dir: Path,
    manifest: dict[str, Any],
    source_nifti: Path,
    planning_ct_series_uid: str,
    source_nifti_sha256: str,
    source_ct_sop_hash: Optional[str],
) -> bool:
    """Admit pre-contract manifests only when their identity chain is explicit.

    Existing cohort masks predate the course contract. Their old manifest did not
    store the CT UID, so it can be upgraded only when its source NIfTI name and
    generation order agree with the current contract and its sidecar identifies
    the same CT series. Otherwise the safe action is model recomputation.
    """
    recorded_source = manifest.get("source_nifti")
    if not isinstance(recorded_source, str) or not recorded_source.strip():
        return False
    recorded_source = recorded_source.strip()
    allowed_source_names = {
        source_nifti.name,
        str(source_nifti),
        str(source_nifti.resolve(strict=False)),
    }
    if recorded_source not in allowed_source_names and Path(recorded_source).name != source_nifti.name:
        return False

    sidecar_base = _strip_nifti_base(source_nifti)
    sidecar = source_nifti.parent / f"{sidecar_base}.metadata.json"
    try:
        sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(sidecar_data, dict):
        return False
    if str(sidecar_data.get("series_instance_uid") or "").strip() != planning_ct_series_uid:
        return False
    sidecar_hash = str(sidecar_data.get("nifti_sha256") or "").strip()
    if sidecar_hash and sidecar_hash != source_nifti_sha256:
        return False
    if source_ct_sop_hash:
        sidecar_ct_hash = str(sidecar_data.get("sop_hash") or "").strip()
        if sidecar_ct_hash and sidecar_ct_hash != source_ct_sop_hash:
            return False

    # ``generated_at`` on the metadata sidecar is the write time of the sidecar,
    # not the generation time of the NIfTI content.  It may therefore be newer
    # after organize refreshes metadata and must not revoke valid legacy masks.
    # New conversions carry ``nifti_generated_at``.  Older sidecars have no such
    # field, so their only admissible ordering evidence is the manifest mtime
    # after the current NIfTI mtime.  An uncheckable ordering fails closed.
    manifest_time = _parse_timestamp(manifest.get("generated_at"))
    nifti_generated_time = _parse_timestamp(sidecar_data.get("nifti_generated_at"))
    try:
        manifest_path = base_dir / "manifest.json"
        if nifti_generated_time is not None and manifest_time is not None:
            if manifest_time <= nifti_generated_time:
                return False
        elif nifti_generated_time is not None:
            manifest_mtime = manifest_path.stat().st_mtime
            if manifest_mtime <= nifti_generated_time.timestamp():
                return False
        else:
            if manifest_path.stat().st_mtime <= source_nifti.stat().st_mtime:
                return False
    except (OSError, ValueError, OverflowError):
        return False
    return True


def _series_masks_current(
    base_dir: Path,
    base_name: str,
    model: str,
    *,
    source_nifti: Path,
    planning_ct_series_uid: str,
    source_ct_sop_hash: Optional[str] = None,
) -> tuple[bool, str]:
    """Check completeness, source identity, and readability of a mask set.

    A complete manifest is not enough for reuse. The mask set must identify the
    contracted planning CT and every recorded NIfTI mask must be readable and
    share the current planning-NIfTI geometry. Missing provenance fails closed,
    except for the narrowly checked legacy upgrade path above.
    """
    if not _series_segmentation_ready(base_dir, base_name, model):
        return False, "mask manifest is missing, incomplete, or inconsistent"
    manifest_path = base_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return False, "mask manifest is not an object"
        entries = manifest.get("models")
        entry = next(
            item for item in entries
            if isinstance(item, dict) and item.get("model") == model
        )
        masks = entry.get("masks")
        if not isinstance(masks, list) or not masks:
            return False, "mask manifest has no mask inventory"
        source_nifti_sha256 = _file_sha256(source_nifti)
    except (OSError, StopIteration, TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, f"mask provenance could not be read: {exc}"

    recorded_uids = {
        str(manifest.get(key) or "").strip()
        for key in ("source_series_instance_uid", "planning_ct_series_instance_uid")
        if str(manifest.get(key) or "").strip()
    }
    if recorded_uids and recorded_uids != {planning_ct_series_uid}:
        return False, "mask provenance names a different planning CT series"
    recorded_nifti_hash = str(manifest.get("source_nifti_sha256") or "").strip()
    if recorded_nifti_hash and recorded_nifti_hash != source_nifti_sha256:
        return False, "mask provenance names a different planning CT NIfTI"
    recorded_ct_hash = str(manifest.get("source_ct_sop_hash") or "").strip()
    if recorded_ct_hash and source_ct_sop_hash and recorded_ct_hash != source_ct_sop_hash:
        return False, "mask provenance names a different CT instance set"
    if not recorded_uids or not recorded_nifti_hash:
        if not _legacy_segmentation_provenance_is_safe(
            base_dir,
            manifest,
            source_nifti,
            planning_ct_series_uid,
            source_nifti_sha256,
            source_ct_sop_hash,
        ):
            return False, "legacy mask provenance cannot establish correspondence to the planning CT"

    try:
        import SimpleITK as sitk

        reference = sitk.ReadImage(str(source_nifti))
        if reference.GetDimension() != 3:
            return False, "planning CT NIfTI is not three-dimensional"
        root = base_dir.resolve(strict=False)
        expected_paths: set[Path] = set()
        for mask_name in masks:
            if not isinstance(mask_name, str) or not mask_name.strip():
                return False, "mask manifest contains an invalid path"
            mask_path = (base_dir / mask_name).resolve(strict=False)
            try:
                mask_path.relative_to(root)
            except ValueError:
                return False, "mask manifest contains a path outside its segmentation directory"
            expected_paths.add(mask_path)
        actual_paths = {
            path.resolve(strict=False)
            for pattern in (f"{model}--*.nii*", f"{base_name}--{model}--*.nii*")
            for path in base_dir.glob(pattern)
            if path.is_file()
        }
        if actual_paths != expected_paths:
            extra = sorted(path.name for path in actual_paths - expected_paths)
            missing = sorted(path.name for path in expected_paths - actual_paths)
            return False, (
                "mask directory does not match its manifest inventory "
                f"(unmanifested={extra}, missing={missing})"
            )
        non_empty_masks = 0
        for mask_name in masks:
            mask_path = (base_dir / mask_name).resolve(strict=False)
            mask = sitk.ReadImage(str(mask_path))
            if mask.GetDimension() != 3 or not _same_image_geometry(mask, reference):
                return False, f"mask {mask_name} is unreadable or mismatched to the planning CT geometry"
            if bool(np.any(sitk.GetArrayViewFromImage(mask))):
                non_empty_masks += 1
        if non_empty_masks == 0:
            return False, "mask inventory contains no readable non-empty segmentation ROI"
    except Exception as exc:
        return False, f"mask geometry validation failed: {exc}"
    return True, (
        f"complete masks match the contracted planning CT and contain "
        f"{non_empty_masks} non-empty ROI mask(s)"
    )


def _segmentation_source_provenance(
    nifti_path: Path,
    planning_ct_series_uid: str,
    source_ct_sop_hash: Optional[str] = None,
) -> dict[str, Any]:
    """Build the provenance fields written into a completed mask manifest."""
    return {
        "source_nifti": str(Path(nifti_path).name),
        "source_nifti_path": str(Path(nifti_path).resolve(strict=False)),
        "source_nifti_sha256": _file_sha256(Path(nifti_path)),
        "source_series_instance_uid": str(planning_ct_series_uid or ""),
        "planning_ct_series_instance_uid": str(planning_ct_series_uid or ""),
        "source_ct_sop_hash": str(source_ct_sop_hash or ""),
    }


def _series_model_manifest_entry(base_dir: Path, base_name: str, model: str) -> dict[str, object] | None:
    named_dicom = base_dir / f"{base_name}--{model}.dcm"
    legacy_dicom = base_dir / f"{model}.dcm"
    rt_out = named_dicom if named_dicom.exists() else legacy_dicom
    masks_for_model = sorted(base_dir.glob(f"{model}--*.nii*")) or sorted(
        base_dir.glob(f"{base_name}--{model}--*.nii*")
    )
    entry: dict[str, object] = {"model": model, "rtstruct_ok": False, "rtstruct": "", "masks": []}
    if rt_out.exists() and rt_out.is_file():
        entry["rtstruct_ok"] = True
        entry["rtstruct"] = str(rt_out.relative_to(base_dir))
    if masks_for_model:
        entry["masks"] = [str(p.relative_to(base_dir)) for p in masks_for_model]
    return entry if entry["rtstruct"] or entry["masks"] else None


def _series_artifact_dirs(input_dir: Path) -> tuple[Path, Path]:
    if input_dir.name == "DICOM":
        return input_dir.parent / "NIFTI", input_dir.parent / "Segmentation_TotalSegmentator"
    return (
        input_dir.parent / "NIFTI" / input_dir.name,
        input_dir.parent / "Segmentation_TotalSegmentator" / input_dir.name,
    )


def _series_original_artifact_dir(input_dir: Path) -> Path:
    if input_dir.name == "DICOM":
        return input_dir.parent / "Segmentation_Original"
    return input_dir.parent / "Segmentation_Original" / input_dir.name


def _log_no_all_series_original(patient_id: str, row: dict) -> None:
    logger.debug(
        "all_series_original_segmentation event=no_original_available patient_id=%s "
        "series_uid=%s study_uid=%s frame_of_reference_uid=%s image_class=%s",
        patient_id,
        row.get("series_uid", ""),
        row.get("study_uid", ""),
        row.get("frame_of_reference_uid", ""),
        row.get("image_class", ""),
    )


MANIFEST_ERROR_KEY = "__manifest_error__"


def _manifest_error_summary(reason: str) -> dict:
    """Summary marking a present-but-unusable manifest as a failure.

    Distinct from ``{}``, which means no manifest exists for this patient and is
    a legitimate skip rather than an error.
    """
    return {
        MANIFEST_ERROR_KEY: {
            "attempted": 0,
            "segmented": 0,
            "failed": 1,
            "skipped": 0,
            "reason": reason,
        }
    }


def _summary_bucket(summary: dict, image_class: str) -> dict:
    return summary.setdefault(
        image_class,
        {"attempted": 0, "segmented": 0, "failed": 0, "skipped": 0},
    )


def _write_updated_series_manifest(manifest_path: Path, manifest: dict) -> None:
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _limit_fourdct_to_representative(rows: list) -> list:
    """Keep at most ONE representative 4DCT volume in ``rows``.

    The single kept 4DCT row is the first ``fourdct_ave`` (averaged reconstruction) in manifest order if
    any ave exists; otherwise the first ``fourdct_phase``. Every other 4DCT row (extra ave reconstructions
    and all non-selected phases) is dropped. Non-4DCT rows pass through unchanged.
    """
    chosen = None
    for r in rows:  # prefer the first averaged 4DCT
        if isinstance(r, dict) and str(r.get("image_class") or "") == "fourdct_ave":
            chosen = r
            break
    if chosen is None:  # else the first 4DCT phase
        for r in rows:
            if isinstance(r, dict) and str(r.get("image_class") or "") == "fourdct_phase":
                chosen = r
                break
    out: list = []
    for r in rows:
        cls = str(r.get("image_class") or "") if isinstance(r, dict) else ""
        if cls in ("fourdct_ave", "fourdct_phase"):
            if r is chosen:
                out.append(r)
        else:
            out.append(r)
    return out


def _select_all_series_rows(config: PipelineConfig, rows: list) -> list:
    """Apply the configured all-series segmentation scope to ``rows``.

    Returns the subset of manifest rows that should be segmented, honoring the optional image_class
    allow-list (``config.all_series_segment_classes``) and the one-representative-4DCT reduction
    (``config.all_series_fourdct_single_representative``). Returned rows are the same dict objects as in
    ``rows`` (the loop mutates them in place), so excluded rows remain in the manifest with their existing
    status. ``None`` allow-list preserves legacy behavior (every eligible class).
    """
    allowed = getattr(config, "all_series_segment_classes", None)
    selected: list = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if allowed is not None and str(row.get("image_class") or "") not in allowed:
            continue
        selected.append(row)
    if getattr(config, "all_series_fourdct_single_representative", False):
        selected = _limit_fourdct_to_representative(selected)
    return selected


def segment_all_series_for_patient(config: PipelineConfig, patient_id: str, *, force: bool = False) -> dict:
    """Run TotalSegmentator for every eligible materialized series in a patient manifest."""

    patient_series_root = Path(config.output_root) / str(patient_id) / "all_series"
    course_dirs = build_course_dirs(patient_series_root)
    manifest_path = course_dirs.metadata / "series_manifest.json"
    if not manifest_path.exists():
        logger.info("All-series manifest not found for patient %s at %s; skipping", patient_id, manifest_path)
        return {}

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Unable to read all-series manifest for patient %s: %s", patient_id, exc)
        # A manifest that exists but cannot be read is a failure, not an absence:
        # eligible series may be listed in it and would otherwise be skipped
        # silently while the run still reported success.
        return _manifest_error_summary("unreadable manifest")

    rows = manifest.get("series", [])
    if not isinstance(rows, list):
        logger.warning("All-series manifest for patient %s has no series list; skipping", patient_id)
        return _manifest_error_summary("manifest has no series list")

    summary: dict = {}
    segmentable_statuses = {"materialized", "segmented", "seg_failed", "seg_skipped_idempotent"}

    # Optionally restrict segmented image classes and cap 4DCT to one representative.
    # Excluded rows stay in `manifest` (written back below) with their materialized status untouched.
    seg_rows = _select_all_series_rows(config, rows)
    manual_rtstructs = manual_rtstruct_bindings_from_inventory(
        getattr(config, "inventory_db_path", None),
        patient_id,
        rows,
    )

    for row in seg_rows:
        if not isinstance(row, dict):
            continue
        task = str(row.get("ts_task") or "none")
        status = str(row.get("status") or "")
        image_class = str(row.get("image_class") or "unknown")
        series_uid = str(row.get("series_uid") or "")
        if task == "none" or status not in segmentable_statuses:
            continue

        bucket = _summary_bucket(summary, image_class)
        expected_task = TS_TASK_BY_CLASS.get(image_class)
        if task not in {"total", "total_mr"} or task != expected_task:
            reason = f"invalid ts_task {task!r} for image_class {image_class!r}"
            logger.warning("Skipping all-series segmentation for patient %s: %s", patient_id, reason)
            row["status"] = "seg_failed"
            row["segmentation_error"] = reason
            bucket["failed"] += 1
            continue

        output_dir_text = str(row.get("output_dir") or "")
        if not output_dir_text:
            row["status"] = "seg_failed"
            bucket["failed"] += 1
            continue
        input_dir = Path(output_dir_text)
        if not input_dir.exists():
            logger.warning("All-series input directory missing for patient %s: %s", patient_id, input_dir)
            row["status"] = "seg_failed"
            bucket["failed"] += 1
            continue

        models = ts_tasks_for_image_class(image_class, getattr(config, "body_composition_classes", None))

        try:
            nifti_dir, seg_root = _series_artifact_dirs(input_dir)
            nifti_path = _ensure_ct_nifti(config, input_dir, nifti_dir, force=force, dcm2niix_depth=0)
            if nifti_path is None:
                logger.warning("All-series NIfTI conversion failed for %s", input_dir)
                row["status"] = "seg_failed"
                bucket["attempted"] += 1
                bucket["failed"] += 1
                continue

            # B5: persist contrast phase (queryable) for calibrated-CT classes. QC-only,
            # runs whenever the CT NIfTI exists (even if segmentation is idempotent-skipped),
            # never blocks segmentation; idempotent unless force.
            if image_class in _CONTRAST_PHASE_CLASSES and (force or not row.get("contrast_phase")):
                phase = _detect_contrast_phase_safe(config, nifti_path)
                if phase is not None:
                    row["contrast_phase"] = phase

            base_name = _strip_nifti_base(nifti_path)
            manual_rtstruct = manual_rtstructs.get(series_uid)
            if manual_rtstruct and Path(manual_rtstruct).exists():
                try:
                    from .organize import _export_original_segmentation_from_paths

                    manual_manifest = _export_original_segmentation_from_paths(
                        rs_path=Path(manual_rtstruct),
                        primary_nifti=Path(nifti_path),
                        dicom_ct_dir=input_dir,
                        segmentation_original_dir=_series_original_artifact_dir(input_dir),
                        log_root=input_dir,
                        overwrite=force,
                    )
                    if manual_manifest:
                        row["manual_segmentation_manifest"] = str(
                            _series_original_artifact_dir(input_dir) / base_name / "metadata.json"
                        )
                except Exception as exc:
                    logger.warning(
                        "all_series_original_segmentation event=export_failed patient_id=%s "
                        "series_uid=%s rtstruct=%s error=%s",
                        patient_id,
                        series_uid,
                        manual_rtstruct,
                        exc,
                    )
            else:
                _log_no_all_series_original(patient_id, row)

            seg_root.mkdir(parents=True, exist_ok=True)
            base_dir = seg_root / base_name
            base_dir.mkdir(parents=True, exist_ok=True)
            tmp_parent = Path(config.segmentation_temp_root) if getattr(config, "segmentation_temp_root", None) else nifti_dir.parent
            try:
                tmp_parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                tmp_parent = nifti_dir.parent

            attempted_any = False
            failed_models: list[str] = []
            manifest_entries: list[dict[str, object]] = []
            resume_decisions: dict[str, Any] = {}
            try:
                series_metadata = _collect_series_metadata(input_dir)
            except Exception:
                series_metadata = {}
            source_provenance = _segmentation_source_provenance(
                nifti_path,
                series_uid,
                str(row.get("sop_hash") or series_metadata.get("sop_hash") or ""),
            )

            for model in models:
                if not force:
                    current, reason = _series_masks_current(
                        base_dir,
                        base_name,
                        model,
                        source_nifti=nifti_path,
                        planning_ct_series_uid=series_uid,
                        source_ct_sop_hash=source_provenance["source_ct_sop_hash"],
                    )
                    if current:
                        ready_entry = _series_model_manifest_entry(base_dir, base_name, model)
                        if ready_entry:
                            manifest_entries.append(ready_entry)
                        resume_decisions[model] = {
                            "action": "reused",
                            "model_run": False,
                            "artefact": "TotalSegmentator masks",
                            "reason": reason,
                        }
                        logger.info(
                            "Segmentation resume patient=%s series=%s model=%s action=reused "
                            "model_run=false reason=%s",
                            patient_id,
                            series_uid,
                            model,
                            reason,
                        )
                        continue
                    resume_decisions[model] = {
                        "action": "pending",
                        "model_run": False,
                        "artefact": "TotalSegmentator masks",
                        "reason": reason,
                    }
                else:
                    resume_decisions[model] = {
                        "action": "pending",
                        "model_run": False,
                        "artefact": "TotalSegmentator masks",
                        "reason": "forced segmentation",
                    }

                attempted_any = True
                model_extra_args = (
                    list(getattr(config, "cbct_totalseg_extra_args", []) or [])
                    if image_class == "cbct" and model == "total"
                    else None
                )
                with tempfile.TemporaryDirectory(prefix="seg_series_", dir=str(tmp_parent)) as tmp_root_str:
                    tmp_root = Path(tmp_root_str)
                    nifti_tmp = tmp_root / model / "nifti"
                    nifti_tmp.mkdir(parents=True, exist_ok=True)

                    rt_out = base_dir / f"{base_name}--{model}.dcm"
                    _clear_previous_masks(base_dir, base_name, model)
                    if rt_out.is_dir():
                        shutil.rmtree(rt_out, ignore_errors=True)
                    else:
                        rt_out.unlink(missing_ok=True)

                    ok_nifti = run_totalsegmentator(
                        config,
                        nifti_path,
                        nifti_tmp,
                        "nifti",
                        task=model,
                        extra_args=model_extra_args,
                    )

                    if ok_nifti:
                        _materialize_masks(nifti_tmp, base_dir, base_name, model)
                        _ensure_model_rtstruct_from_masks(
                            input_dir,
                            base_dir,
                            base_name,
                            model,
                        )

                    entry = _series_model_manifest_entry(base_dir, base_name, model)
                    if entry is not None and not ok_nifti:
                        entry["masks"] = []
                    if entry:
                        manifest_entries.append(entry)
                    current_masks_value = entry.get("masks", []) if entry else []
                    current_masks = current_masks_value if isinstance(current_masks_value, list) else []
                    current_run_ready = bool(current_masks) and all(
                        (base_dir / str(mask)).exists() for mask in current_masks
                    )
                    run_succeeded = bool(ok_nifti and current_run_ready)
                    decision_reason = str(resume_decisions[model].get("reason") or "")
                    resume_decisions[model] = {
                        "action": "rebuilt" if run_succeeded else "failed",
                        "model_run": True,
                        "run_succeeded": run_succeeded,
                        "artefact": "TotalSegmentator masks",
                        "reason": (
                            f"{decision_reason}; TotalSegmentator completed and published the mask inventory"
                            if run_succeeded
                            else f"{decision_reason}; TotalSegmentator was invoked but did not publish a complete mask inventory"
                        ),
                    }
                    if not run_succeeded:
                        failed_models.append(model)

            if attempted_any:
                bucket["attempted"] += 1

            if attempted_any and manifest_entries:
                series_manifest_path = base_dir / "manifest.json"
                previous_manifest: dict[str, Any] = {}
                if series_manifest_path.exists():
                    try:
                        parsed_manifest = json.loads(series_manifest_path.read_text(encoding="utf-8"))
                        if isinstance(parsed_manifest, dict):
                            previous_manifest = parsed_manifest
                    except Exception:
                        previous_manifest = {}
                previous_skipped = previous_manifest.get("skipped_models")
                merged_skipped = dict(previous_skipped) if isinstance(previous_skipped, dict) else {}
                # All-series runs do not currently add QC skips, but retain any
                # prior skips rather than erasing their audit trail.
                series_manifest = {
                    **source_provenance,
                    "source_dicom": str(input_dir),
                    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "models": manifest_entries,
                }
                if merged_skipped:
                    series_manifest["skipped_models"] = merged_skipped
                _write_manifest_atomic(series_manifest_path, series_manifest)

            if resume_decisions:
                try:
                    record_segmentation_resume_decision(
                        patient_series_root,
                        {series_uid: resume_decisions},
                        source=source_provenance,
                        record_path=base_dir / "resume_decision.json",
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not record all-series segmentation resume decision for %s: %s",
                        input_dir,
                        exc,
                    )

            if not failed_models and len(models) > 1:
                # Per-series body-composition JSON is the collision-free source of
                # truth. The global Data/body_composition.csv is aggregated once,
                # serially, at the end of the all-series stage (rtpipeline.cli) to
                # avoid the cross-process race + O(N^2) full-tree rescan that a
                # per-series rewrite caused. A derived-metric failure here must NOT
                # invalidate an otherwise-successful segmentation.
                try:
                    from .body_composition import write_series_body_composition

                    body_json = write_series_body_composition(
                        ct_nifti=nifti_path,
                        segmentation_dir=base_dir,
                        dicom_dir=input_dir,
                        patient_id=str(row.get("patient_id") or patient_id),
                        series_uid=series_uid,
                        image_class=image_class,
                    )
                    row["body_composition_json"] = str(body_json)
                except Exception as exc:
                    logger.warning(
                        "Body-composition metrics failed for patient %s series %s: %s",
                        patient_id,
                        series_uid,
                        exc,
                    )
                    row["body_composition_error"] = str(exc)

            if failed_models:
                row["status"] = "seg_failed"
                row["segmentation_error"] = "failed models: " + ",".join(failed_models)
                bucket["failed"] += 1
            elif attempted_any:
                row["status"] = "segmented"
                bucket["segmented"] += 1
            else:
                row["status"] = "seg_skipped_idempotent"
                bucket["skipped"] += 1
        except Exception as exc:
            logger.warning(
                "All-series segmentation failed for patient %s series %s: %s",
                patient_id,
                row.get("series_uid", ""),
                exc,
            )
            row["status"] = "seg_failed"
            bucket["failed"] += 1

    _write_updated_series_manifest(manifest_path, manifest)
    return summary


def _segmentation_status_base(course_dir: Path) -> dict[str, Any]:
    return {
        "version": 1,
        "course_dir": str(Path(course_dir).resolve(strict=False)),
        "assessed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "failed",
        "reasons": [],
        "evidence": {},
    }


def failed_segmentation_outcome(course_dir: Path, reason: str) -> dict[str, Any]:
    outcome = _segmentation_status_base(course_dir)
    outcome["status"] = "failed"
    outcome["reasons"] = [str(reason)]
    return outcome


def _recorded_segmentation_failures(course_dir: Path) -> list[dict[str, str]]:
    audit_path = Path(course_dir) / "metadata" / "segmentation_resume.json"
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    decisions = audit.get("decisions") if isinstance(audit, dict) else None
    if not isinstance(decisions, dict):
        return []
    failures: list[dict[str, str]] = []
    for artifact, raw in decisions.items():
        # RS_custom is produced after segmentation and may consume outputs from the
        # segmentation_custom_models rule. Letting its radiomics/DVH failure close
        # .segmentation_done creates a dependency cycle that prevents its own repair.
        # RS_custom remains fail-closed in the owning course stage and its sentinel.
        if str(artifact) == "RS_custom":
            continue
        if not isinstance(raw, dict) or raw.get("action") != "failed":
            continue
        failure = raw.get("failure")
        category = "segmentation_failure"
        detail = str(raw.get("reason") or "segmentation artifact failed")
        if isinstance(failure, dict):
            category = str(failure.get("category") or category)
            detail = str(failure.get("reason") or detail)
        failures.append(
            {
                "artifact": str(artifact),
                "category": category,
                "reason": detail,
            }
        )
    return failures


def assess_course_segmentation(course_dir: Path) -> dict[str, Any]:
    """Derive the stage outcome from contracted, readable surviving content."""

    course_dir = Path(course_dir)
    outcome = _segmentation_status_base(course_dir)
    evidence = outcome["evidence"]
    assert isinstance(evidence, dict)
    try:
        contract = load_course_contract(course_dir)
    except Exception as exc:
        outcome["reasons"] = [
            f"authoritative course contract could not be loaded: {type(exc).__name__}: {exc}"
        ]
        return outcome

    ct_dir = contract.planning_ct_dir
    nifti_path = contract.planning_ct_nifti
    planning_ct = contract.planning_ct
    planning_uid = str(planning_ct.get("series_instance_uid") or "").strip()
    evidence["planning_ct_series_instance_uid"] = planning_uid
    evidence["planning_ct_dicom_dir"] = str(ct_dir) if ct_dir is not None else None
    evidence["planning_ct_nifti"] = str(nifti_path) if nifti_path is not None else None

    if ct_dir is None or nifti_path is None:
        outcome["status"] = "disabled"
        outcome["reasons"] = [
            "nothing applicable to segment because the authoritative course contract "
            "has no planning CT DICOM and NIfTI pair"
        ]
        return outcome
    if not planning_uid:
        outcome["reasons"] = [
            "planning CT exists but the authoritative course contract has no series identity"
        ]
        return outcome
    if not nifti_path.is_file():
        outcome["reasons"] = [f"contracted planning CT NIfTI is missing: {nifti_path}"]
        return outcome

    nifti_provenance = planning_ct.get("nifti_provenance")
    source_ct_sop_hash = (
        str(nifti_provenance.get("sop_hash") or "")
        if isinstance(nifti_provenance, dict)
        else ""
    )
    base_name = _strip_nifti_base(nifti_path)
    base_dir = build_course_dirs(course_dir).segmentation_totalseg / base_name
    manifest_path = base_dir / "manifest.json"
    evidence["manifest"] = str(manifest_path)
    evidence["segmentation_directory"] = str(base_dir)

    current, current_reason = _series_masks_current(
        base_dir,
        base_name,
        "total",
        source_nifti=nifti_path,
        planning_ct_series_uid=planning_uid,
        source_ct_sop_hash=source_ct_sop_hash,
    )
    evidence["total_masks_current"] = current
    evidence["total_masks_reason"] = current_reason
    recorded_failures = _recorded_segmentation_failures(course_dir)
    evidence["recorded_failures"] = recorded_failures
    if not current:
        reasons = [f"usable TotalSegmentator masks are absent: {current_reason}"]
        reasons.extend(
            f"{item['artifact']} {item['category']}: {item['reason']}"
            for item in recorded_failures
        )
        outcome["reasons"] = reasons
        return outcome

    # Requested model failures are material stage failures even when the default
    # total model survived. Explicit QC skips remain non-failures in the manifest.
    if recorded_failures:
        outcome["reasons"] = [
            f"{item['artifact']} {item['category']}: {item['reason']}"
            for item in recorded_failures
        ]
        return outcome

    from .auto_rtstruct import (
        _derived_rtstruct_dependencies_are_current,
        _is_valid_rtstruct,
        _rtstruct_matches_planning_ct,
    )

    rs_auto = course_dir / "RS_auto.dcm"
    rs_auto_valid = _is_valid_rtstruct(rs_auto)
    rs_auto_matches = rs_auto_valid and _rtstruct_matches_planning_ct(rs_auto, planning_uid)
    rs_auto_current = bool(
        rs_auto_matches
        and _derived_rtstruct_dependencies_are_current(
            rs_auto,
            ct_dir=ct_dir,
            nifti_path=nifti_path,
            segmentation_root=build_course_dirs(course_dir).segmentation_totalseg,
        )
    )
    evidence["rs_auto"] = str(rs_auto)
    evidence["rs_auto_readable_non_empty"] = rs_auto_valid
    evidence["rs_auto_matches_planning_ct"] = rs_auto_matches
    evidence["rs_auto_current"] = rs_auto_current
    if not rs_auto_current:
        outcome["reasons"] = [
            "RS_auto.dcm could not be derived as a readable, non-empty, current RTSTRUCT "
            "bound to the contracted planning CT"
        ]
        return outcome

    outcome["status"] = "ok"
    outcome["reasons"] = [
        "contracted planning CT, complete readable non-empty masks, and current "
        "RS_auto.dcm were validated"
    ]
    return outcome


def _write_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_course_segmentation_status(
    course_dir: Path,
    outcome: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Atomically publish a content-derived stage report and legacy status sentinel."""

    course_dir = Path(course_dir)
    requested = dict(outcome) if outcome is not None else None
    requested_status = str((requested or {}).get("status") or "").strip().lower()
    # Success is never caller-asserted. Reassess contracted content even when a
    # caller supplies an apparently successful outcome. Explicit non-success
    # outcomes remain available so a caught crash can preserve its exact reason.
    final = (
        assess_course_segmentation(course_dir)
        if requested is None or requested_status == "ok"
        else requested
    )
    status = str(final.get("status") or "").strip().lower()
    if status not in _SEGMENTATION_STATUSES:
        final = failed_segmentation_outcome(
            course_dir,
            f"invalid segmentation status {status!r} was rejected",
        )
        status = "failed"
    reasons = final.get("reasons")
    if not isinstance(reasons, list) or not reasons:
        final["reasons"] = ["segmentation status had no recorded reason"]
        status = "failed"
    final["status"] = status
    final["assessed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _write_manifest_atomic(course_dir / SEGMENTATION_STATUS_RELATIVE_PATH, final)
    _write_text_atomic(course_dir / SEGMENTATION_SENTINEL_NAME, f"{status}\n")
    logger.log(
        logging.INFO if status == "ok" else logging.ERROR,
        "Segmentation stage status=%s course=%s reasons=%s",
        status,
        course_dir,
        "; ".join(str(reason) for reason in final["reasons"]),
    )
    return final


def segment_course(config: PipelineConfig, course_dir: Path, force: bool = False) -> dict:
    """Run TotalSegmentator for a course organised under the new directory layout."""

    contract = load_course_contract(course_dir)
    course_dirs = build_course_dirs(course_dir)
    course_dirs.ensure()

    results: dict[str, Any] = {"nifti": None, "dicom_seg": None, "nifti_seg_dir": None}
    ct_dir = contract.planning_ct_dir
    nifti_path = contract.planning_ct_nifti
    if ct_dir is None or nifti_path is None:
        logger.warning("Course contract has no planning CT for %s; skipping CT segmentation", course_dir)
        _segment_mr_series_for_course(config, course_dirs, course_dir, force=force)
        return results

    base_name = _strip_nifti_base(nifti_path)
    seg_root = course_dirs.segmentation_totalseg
    seg_root.mkdir(parents=True, exist_ok=True)
    base_dir = seg_root / base_name
    base_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: List[Dict[str, object]] = []

    results["nifti"] = str(nifti_path)

    models = ["total"] + [m for m in (config.extra_seg_models or []) if not m.endswith("_mr")]
    planning_ct = contract.data.get("planning_ct", {})
    nifti_provenance = planning_ct.get("nifti_provenance") if isinstance(planning_ct, dict) else None
    source_series_uid = str(planning_ct.get("series_instance_uid") or "")
    source_ct_sop_hash = (
        str(nifti_provenance.get("sop_hash") or "")
        if isinstance(nifti_provenance, dict)
        else ""
    )
    source_provenance = _segmentation_source_provenance(
        nifti_path,
        source_series_uid,
        source_ct_sop_hash,
    )

    tmp_parent = Path(config.segmentation_temp_root) if getattr(config, "segmentation_temp_root", None) else course_dir
    try:
        Path(tmp_parent).mkdir(parents=True, exist_ok=True)
    except Exception:
        tmp_parent = course_dir

    # Track skipped models due to body region QC
    skipped_models: Dict[str, str] = {}
    resume_decisions: dict[str, Any] = {}
    body_region_qc_done = False
    attempted_any = False

    with tempfile.TemporaryDirectory(prefix="seg_", dir=str(tmp_parent)) as tmp_root_str:
        tmp_root = Path(tmp_root_str)
        for model in models:
            if not force:
                current, reason = _series_masks_current(
                    base_dir,
                    base_name,
                    model,
                    source_nifti=nifti_path,
                    planning_ct_series_uid=source_series_uid,
                    source_ct_sop_hash=source_ct_sop_hash,
                )
                if current:
                    ready_entry = _series_model_manifest_entry(base_dir, base_name, model)
                    if ready_entry is not None:
                        manifest_entries.append(ready_entry)
                    resume_decisions[model] = {
                        "action": "reused",
                        "model_run": False,
                        "artefact": "TotalSegmentator masks",
                        "reason": reason,
                    }
                    logger.info(
                        "Segmentation resume course=%s model=%s action=reused model_run=false reason=%s",
                        course_dir,
                        model,
                        reason,
                    )
                    if model == "total":
                        default_named_dicom = base_dir / f"{base_name}--total.dcm"
                        default_dicom = default_named_dicom if default_named_dicom.exists() else base_dir / "total.dcm"
                        if default_dicom.exists():
                            results["dicom_seg"] = str(default_dicom)
                        else:
                            derived_dicom = _ensure_model_rtstruct_from_masks(
                                ct_dir, base_dir, base_name, model
                            )
                            decision_key = "RS_auto" if model == "total" else f"{model}_rtstruct"
                            if derived_dicom is not None:
                                results["dicom_seg"] = str(derived_dicom)
                                for entry in manifest_entries:
                                    if entry.get("model") == model:
                                        entry["rtstruct_ok"] = True
                                        entry["rtstruct"] = str(derived_dicom.relative_to(base_dir))
                                resume_decisions[decision_key] = {
                                    "action": "rebuilt",
                                    "model_run": False,
                                    "artefact": str(derived_dicom.name),
                                    "reason": "derived from current reusable masks without rerunning the model",
                                }
                            else:
                                resume_decisions[decision_key] = {
                                    "action": "failed",
                                    "model_run": False,
                                    "artefact": str(default_named_dicom.name),
                                    "reason": "current masks were reusable but the derived RTSTRUCT could not be rebuilt",
                                }
                    continue
                logger.info(
                    "Segmentation resume course=%s model=%s action=pending model_run=false reason=%s",
                    course_dir,
                    model,
                    reason,
                )
                resume_decisions[model] = {
                    "action": "pending",
                    "model_run": False,
                    "artefact": "TotalSegmentator masks",
                    "reason": reason,
                }
            else:
                resume_decisions[model] = {
                    "action": "pending",
                    "model_run": False,
                    "artefact": "TotalSegmentator masks",
                    "reason": "forced segmentation",
                }
            task_name = None if model == "total" else model

            # After "total" completes, run body region QC before extra models
            if model != "total" and not body_region_qc_done:
                body_region_qc_done = True
                try:
                    qc = _get_qc_functions()
                    model_requirements = getattr(config, "model_region_requirements", {})
                    qc.save_body_region_qc(
                        course_dir,
                        model_region_requirements=model_requirements,
                        nifti_path=nifti_path,
                        conda_activate=config.conda_activate,
                    )
                    logger.info("Body region QC completed for %s", course_dir)
                except Exception as exc:
                    logger.warning("Body region QC failed for %s: %s", course_dir, exc)

            # Check model eligibility for extra models (not "total")
            if model != "total":
                try:
                    qc = _get_qc_functions()
                    model_requirements = getattr(config, "model_region_requirements", {})
                    eligible, reason = qc.check_model_eligibility(
                        course_dir, model, model_requirements
                    )
                    if not eligible:
                        block_missing = getattr(config, "body_region_qc_block_missing", True)
                        if block_missing:
                            logger.warning(
                                "Skipping TotalSegmentator model '%s' for %s: %s",
                                model, course_dir, reason
                            )
                            skipped_models[model] = reason
                            resume_decisions[model] = {
                                "action": "skipped",
                                "model_run": False,
                                "artefact": "TotalSegmentator masks",
                                "reason": reason,
                            }
                            logger.info(
                                "Segmentation resume course=%s model=%s action=skipped "
                                "model_run=false reason=%s",
                                course_dir,
                                model,
                                reason,
                            )
                            continue
                        else:
                            logger.warning(
                                "Model '%s' may not be appropriate for %s: %s (continuing anyway)",
                                model, course_dir, reason
                            )
                except Exception as exc:
                    logger.debug("Model eligibility check failed for %s: %s", model, exc)

            model_tmp = tmp_root / model
            nifti_tmp = model_tmp / "nifti"
            attempted_any = True
            nifti_tmp.mkdir(parents=True, exist_ok=True)

            model_entry: Dict[str, object] = {"model": model, "rtstruct": "", "masks": []}
            dest_dicom = base_dir / f"{base_name}--{model}.dcm"

            # Rejected prior artifacts cannot remain visible if the replacement
            # model run fails. TotalSegmentator receives only the contracted NIfTI.
            _clear_previous_masks(base_dir, base_name, model)
            dest_dicom.unlink(missing_ok=True)
            legacy_dicom = base_dir / f"{model}.dcm"
            if legacy_dicom != dest_dicom:
                legacy_dicom.unlink(missing_ok=True)

            ok_nifti = run_totalsegmentator(
                config,
                nifti_path,
                nifti_tmp,
                "nifti",
                task=task_name,
            )

            if ok_nifti:
                _materialize_masks(nifti_tmp, base_dir, base_name, model)
                masks_for_model = sorted(base_dir.glob(f"{model}--*.nii*"))
                if masks_for_model:
                    model_entry["masks"] = [
                        str(path.relative_to(base_dir)) for path in masks_for_model
                    ]
                    derived_dicom = _ensure_model_rtstruct_from_masks(
                        ct_dir,
                        base_dir,
                        base_name,
                        model,
                    )
                    if derived_dicom is not None:
                        model_entry["rtstruct"] = str(derived_dicom.relative_to(base_dir))
                        model_entry["rtstruct_ok"] = True
                        if model == "total":
                            results["dicom_seg"] = str(derived_dicom)
            if model_entry["rtstruct"] or model_entry["masks"]:
                manifest_entries.append(model_entry)

            run_succeeded = bool(ok_nifti and model_entry["masks"])
            decision_reason = str(resume_decisions[model].get("reason") or "")
            failure_detail = None if run_succeeded else _last_totalseg_failure()
            resume_decisions[model] = {
                "action": "rebuilt" if run_succeeded else "failed",
                "model_run": True,
                "run_succeeded": run_succeeded,
                "artefact": "TotalSegmentator masks",
                "reason": (
                    f"{decision_reason}; TotalSegmentator completed and published the mask inventory"
                    if run_succeeded
                    else f"{decision_reason}; TotalSegmentator was invoked but did not publish a complete mask inventory"
                ),
            }
            if failure_detail is not None:
                resume_decisions[model]["failure"] = failure_detail
            logger.info(
                "Segmentation resume course=%s model=%s action=%s model_run=true "
                "run_succeeded=%s",
                course_dir,
                model,
                resume_decisions[model]["action"],
                run_succeeded,
            )

    default_dicom = base_dir / "total.dcm"
    if default_dicom.exists():
        results["dicom_seg"] = str(default_dicom)

    manifest_path = base_dir / "manifest.json"
    manifest_needs_update = any(
        isinstance(decision, dict) and decision.get("action") != "reused"
        for decision in resume_decisions.values()
    )
    if attempted_any or (
        manifest_entries and (manifest_needs_update or not manifest_path.exists())
    ):
        try:
            previous_manifest: dict[str, Any] = {}
            if manifest_path.exists():
                parsed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(parsed_manifest, dict):
                    previous_manifest = parsed_manifest
            previous_skipped = previous_manifest.get("skipped_models")
            merged_skipped = dict(previous_skipped) if isinstance(previous_skipped, dict) else {}
            merged_skipped.update(skipped_models)
            manifest = {
                "source_nifti": f"{base_name}.nii.gz",
                **source_provenance,
                "generated_at": (
                    datetime.datetime.now(datetime.timezone.utc).isoformat()
                    if attempted_any
                    else previous_manifest.get("generated_at")
                    or datetime.datetime.now(datetime.timezone.utc).isoformat()
                ),
                "models": manifest_entries,
            }
            if merged_skipped:
                manifest["skipped_models"] = merged_skipped
            _write_manifest_atomic(manifest_path, manifest)
        except Exception as exc:
            logger.debug("Failed to persist segmentation manifest for %s: %s", course_dir, exc)

    results["nifti_seg_dir"] = str(base_dir)
    if skipped_models:
        results["skipped_models"] = skipped_models
    if resume_decisions:
        try:
            record_segmentation_resume_decision(
                course_dir,
                resume_decisions,
                source=source_provenance,
            )
        except Exception as exc:
            logger.warning("Could not record segmentation resume decision for %s: %s", course_dir, exc)

    # ------------------------------------------------------------------
    # MR segmentation for auxiliary series in DICOM_related/
    # ------------------------------------------------------------------
    mr_models = [m for m in (config.extra_seg_models or []) if m.endswith("_mr")]
    if "total_mr" not in mr_models:
        mr_models.append("total_mr")
    mr_models = sorted({m.strip() for m in mr_models if m.strip()})

    def _detect_series_uid(dicom_root: Path) -> Optional[str]:
        for candidate in sorted(dicom_root.rglob("*.dcm")):
            try:
                ds = pydicom.dcmread(str(candidate), stop_before_pixels=True)
            except Exception:
                continue
            uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
            if uid:
                return uid
        return None

    if mr_models and course_dirs.dicom_mr.exists():
        for series_root in sorted(p for p in course_dirs.dicom_mr.iterdir() if p.is_dir()):
            dicom_dir = series_root / "DICOM"
            if dicom_dir.exists():
                source_dir = dicom_dir
            else:
                source_dir = series_root
            if not any(source_dir.glob("*.dcm")):
                continue

            if not _mr_series_is_anatomic(source_dir):
                continue  # C2: skip non-anatomic MR (DWI/ADC/DCE/localizer) before conversion

            nifti_dir = series_root / "NIFTI"
            nifti_dir.mkdir(parents=True, exist_ok=True)

            meta_files = sorted(nifti_dir.glob("*.metadata.json"))
            if not meta_files:
                tmp_out = nifti_dir / f".tmp_{series_root.name}"
                tmp_out.mkdir(parents=True, exist_ok=True)
                generated = run_dcm2niix(config, source_dir, tmp_out)
                if generated is not None:
                    target_path = nifti_dir / generated.name
                    if target_path.exists():
                        target_path.unlink()
                    shutil.move(str(generated), target_path)
                    series_uid = _detect_series_uid(source_dir)
                    metadata = {
                        "modality": "MR",
                        "nifti_path": str(target_path),
                        "source_directory": str(source_dir),
                        "series_instance_uid": series_uid or "",
                        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    }
                    meta_path = nifti_dir / f"{target_path.stem}.metadata.json"
                    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                shutil.rmtree(tmp_out, ignore_errors=True)
                meta_files = sorted(nifti_dir.glob("*.metadata.json"))

            if not meta_files:
                continue
            try:
                meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(meta.get("modality", "")).upper() != "MR":
                continue
            nifti_path = Path(meta.get("nifti_path") or "")
            if not nifti_path.exists():
                continue
            source_dir = Path(meta.get("source_directory") or source_dir)
            if not source_dir.exists():
                source_dir = series_root
            base_name_mr = _strip_nifti_base(nifti_path)
            base_dir_mr = series_root / "Segmentation_TotalSegmentator"
            base_dir_mr.mkdir(parents=True, exist_ok=True)

            def _mr_ready(model: str) -> bool:
                rt_path = base_dir_mr / f"{base_name_mr}--{model}.dcm"
                # Masks are materialized with a "<model>--" prefix (mirrors CT segmentation layout)
                return rt_path.exists() and _series_segmentation_ready(base_dir_mr, base_name_mr, model)

            if not force and all(_mr_ready(model) for model in mr_models):
                continue

            with tempfile.TemporaryDirectory(prefix="seg_mr_", dir=str(course_dir)) as tmp_root_str:
                tmp_root = Path(tmp_root_str)
                manifest_mr: List[Dict[str, object]] = []
                for model in mr_models:
                    task_name = model
                    model_tmp = tmp_root / model
                    nifti_tmp = model_tmp / "nifti"
                    nifti_tmp.mkdir(parents=True, exist_ok=True)

                    rt_out = base_dir_mr / f"{base_name_mr}--{model}.dcm"
                    _clear_previous_masks(base_dir_mr, base_name_mr, model)
                    rt_out.unlink(missing_ok=True)
                    ok_nifti = run_totalsegmentator(
                        config,
                        nifti_path,
                        nifti_tmp,
                        "nifti",
                        task=task_name,
                    )

                    entry = {"model": model, "rtstruct": "", "masks": []}
                    if ok_nifti:
                        _materialize_masks(nifti_tmp, base_dir_mr, base_name_mr, model)
                        masks_for_model = sorted(base_dir_mr.glob(f"{model}--*.nii*"))
                        if masks_for_model:
                            entry["masks"] = [
                                str(path.relative_to(base_dir_mr)) for path in masks_for_model
                            ]
                            derived = _ensure_model_rtstruct_from_masks(
                                source_dir,
                                base_dir_mr,
                                base_name_mr,
                                model,
                            )
                            if derived is not None:
                                entry["rtstruct"] = str(derived.relative_to(base_dir_mr))
                    if entry["rtstruct"] or entry["masks"]:
                        manifest_mr.append(entry)

                if manifest_mr:
                    try:
                        manifest_path = base_dir_mr / "manifest.json"
                        _write_manifest_atomic(
                            manifest_path,
                            {
                                "source_nifti": str(nifti_path.name),
                                "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                "models": manifest_mr,
                            },
                        )
                    except Exception as exc:
                        logger.debug("Failed to persist MR segmentation manifest for %s: %s", base_dir_mr, exc)

    return results


# C2: exclusion reasons that are ADEQUACY-only / unrecognized, NOT non-anatomic content.
# A series classified ("exclude", <one of these>) was SEGMENTED before C2, so it must keep
# segmenting (back-compat). In particular a <10-slice *anatomic* MR classifies
# 'sub_volumetric_lt10' and must NOT be dropped (it yields a thin but valid mask).
# Accepted side effect (non-regression): _common_image_exclusion checks n<10 BEFORE the
# description, so a <10-slice *functional*/localizer MR also reports 'sub_volumetric_lt10'
# and is segmented too — identical to pre-C2 behavior, negligible (<10 slices), and
# total_mr on such a tiny series is harmless. The functional-misroute we actually target
# (REG-referenced DWI/ADC/DCE) is volumetric (>=10 slices) -> caught as mr_functional.
_C2_BACKCOMPAT_SEGMENT_REASONS = frozenset({"sub_volumetric_lt10", "mr_unrecognized_default_deny"})


def _imagetype_to_list(raw_it) -> List[str]:
    """Normalize a DICOM ImageType value to a list of strings. Handles the pydicom
    ``MultiValue`` (a MutableSequence — NOT ``list``/``tuple``), a backslash/slash-joined
    string, a plain list/tuple, and None. (A bare ``isinstance(x,(list,tuple))`` check
    misses MultiValue and stringifies it to a malformed single token.)"""
    if raw_it is None:
        return []
    if isinstance(raw_it, str):
        return [s for s in raw_it.replace("/", "\\").split("\\") if s]
    try:
        return [str(x) for x in raw_it]
    except TypeError:
        return [s for s in str(raw_it).replace("/", "\\").split("\\") if s]


def _mr_series_is_anatomic(source_dir: Path) -> bool:
    """C2 [defense-in-depth]: classify an MR series from its DICOM headers and decide
    whether to run ``total_mr`` on it. The course-MR path routes series to ``dicom_mr``
    on a bare ``modality=='MR'`` check, so a REG-referenced functional series
    (DWI/ADC/DCE) dragged into a course would otherwise be mis-segmented by the anatomic model.

    Reads SeriesDescription/ImageType/instance-count from the first readable DICOM slice
    that carries a SeriesDescription (the dcm2niix sidecar does not carry these), builds the
    lowercase classifier-meta (incl. modality='MR' so ``classify_series`` does not
    default-deny on missing modality), and calls ``classify_series``.

    Back-compat-safe LENIENT boundary (endorsed over the strict WORK_PLAN reading by the
    code gate): SEGMENT (True) for ``mr_anatomic`` and for adequacy-only/unrecognized
    exclusions (``sub_volumetric_lt10`` / ``mr_unrecognized_default_deny`` — segmented
    before C2). SKIP (False, with a structured log) only series POSITIVELY classified as
    non-anatomic CONTENT: ``mr_functional`` (DWI/ADC/DCE) or a content exclusion
    (localizer/scout/derived/report). Any error / no-DICOM / unreadable header fails OPEN
    to segment, so the guard can never break a previously-working course.

    NB: MR MIP-by-description is intentionally NOT excluded here — ``_classify_mr`` calls
    ``_common_image_exclusion(include_mip=False)`` by design; changing that is out of C2's
    scope. Only ImageType/description localizer + derived/report content are caught.
    """
    from .modality_classifier import classify_series
    try:
        dcms = sorted(source_dir.glob("*.dcm"))
        if not dcms:
            return True  # nothing to classify here; let the existing flow proceed
        ds = None
        for cand in dcms:
            try:
                cand_ds = pydicom.dcmread(str(cand), stop_before_pixels=True, force=True)
            except Exception:
                continue
            if ds is None:
                ds = cand_ds  # first readable header, kept as a fallback
            # prefer a slice that actually carries a SeriesDescription: a corrupted or
            # empty first slice (force=True yields an empty Dataset) must not drive the
            # whole-series decision.
            if str(getattr(cand_ds, "SeriesDescription", "") or "").strip():
                ds = cand_ds
                break
        if ds is None:
            return True  # no readable header -> preserve back-compat (segment)
        image_types = _imagetype_to_list(getattr(ds, "ImageType", None))
        meta = {
            "modality": str(getattr(ds, "Modality", "") or "MR"),
            "series_description": str(getattr(ds, "SeriesDescription", "") or ""),
            "image_type": image_types,
            "image_types": image_types,
            "n_slices": len(dcms),
            "n_instances": len(dcms),
            "manufacturer": str(getattr(ds, "Manufacturer", "") or ""),
            "manufacturer_model": str(getattr(ds, "ManufacturerModelName", "") or ""),
        }
        image_class, reason = classify_series(meta)
        if image_class == "mr_anatomic":
            return True
        # adequacy-only / unrecognized exclusions were segmented pre-C2 -> keep segmenting
        # (BC-2: a <10-slice anatomic MR is 'sub_volumetric_lt10' and must not be dropped)
        if image_class == "exclude" and reason in _C2_BACKCOMPAT_SEGMENT_REASONS:
            return True
        logger.info(
            "C2: skipping non-anatomic MR series for total_mr "
            "(image_class=%s, reason=%s, desc=%r, dir=%s)",
            image_class, reason, meta["series_description"], str(source_dir),
        )
        return False
    except Exception as exc:  # never let the guard break a previously-working course
        logger.debug("C2 MR classify failed for %s: %s; proceeding (back-compat)", source_dir, exc)
        return True


def _segment_mr_series_for_course(config: PipelineConfig, course_dirs, course_dir: Path, force: bool = False) -> None:
    """Run MR TotalSegmentator even when CT/planning data are unavailable."""

    mr_models = [m for m in (config.extra_seg_models or []) if m.endswith("_mr")]
    if "total_mr" not in mr_models:
        mr_models.append("total_mr")
    mr_models = sorted({m.strip() for m in mr_models if m.strip()})
    if not mr_models or not course_dirs.dicom_mr.exists():
        return

    def _detect_series_uid(dicom_root: Path) -> Optional[str]:
        for candidate in sorted(dicom_root.rglob("*.dcm")):
            try:
                ds = pydicom.dcmread(str(candidate), stop_before_pixels=True)
            except Exception:
                continue
            uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
            if uid:
                return uid
        return None

    for series_root in sorted(p for p in course_dirs.dicom_mr.iterdir() if p.is_dir()):
        dicom_dir = series_root / "DICOM"
        if dicom_dir.exists():
            source_dir = dicom_dir
        else:
            source_dir = series_root
        if not any(source_dir.glob("*.dcm")):
            continue

        if not _mr_series_is_anatomic(source_dir):
            continue  # C2: skip non-anatomic MR (DWI/ADC/DCE/localizer) before conversion

        nifti_dir = series_root / "NIFTI"
        nifti_dir.mkdir(parents=True, exist_ok=True)

        meta_files = sorted(nifti_dir.glob("*.metadata.json"))
        if not meta_files:
            tmp_out = nifti_dir / f".tmp_{series_root.name}"
            tmp_out.mkdir(parents=True, exist_ok=True)
            generated = run_dcm2niix(config, source_dir, tmp_out)
            if generated is not None:
                target_path = nifti_dir / generated.name
                if target_path.exists():
                    target_path.unlink()
                shutil.move(str(generated), target_path)
                series_uid = _detect_series_uid(source_dir)
                metadata = {
                    "modality": "MR",
                    "nifti_path": str(target_path),
                    "source_directory": str(source_dir),
                    "series_instance_uid": series_uid or "",
                    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
                meta_path = nifti_dir / f"{target_path.stem}.metadata.json"
                meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            shutil.rmtree(tmp_out, ignore_errors=True)
            meta_files = sorted(nifti_dir.glob("*.metadata.json"))

        if not meta_files:
            continue
        try:
            meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(meta.get("modality", "")).upper() != "MR":
            continue
        nifti_path = Path(meta.get("nifti_path") or "")
        if not nifti_path.exists():
            continue
        source_dir = Path(meta.get("source_directory") or source_dir)
        if not source_dir.exists():
            source_dir = series_root
        base_name_mr = _strip_nifti_base(nifti_path)
        base_dir_mr = series_root / "Segmentation_TotalSegmentator"
        base_dir_mr.mkdir(parents=True, exist_ok=True)

        def _mr_ready(model: str) -> bool:
            rt_path = base_dir_mr / f"{base_name_mr}--{model}.dcm"
            return rt_path.exists() and _series_segmentation_ready(base_dir_mr, base_name_mr, model)

        if not force and all(_mr_ready(model) for model in mr_models):
            continue

        with tempfile.TemporaryDirectory(prefix="seg_mr_", dir=str(course_dir)) as tmp_root_str:
            tmp_root = Path(tmp_root_str)
            manifest_mr: List[Dict[str, object]] = []
            for model in mr_models:
                task_name = model
                model_tmp = tmp_root / model
                nifti_tmp = model_tmp / "nifti"
                nifti_tmp.mkdir(parents=True, exist_ok=True)

                rt_out = base_dir_mr / f"{base_name_mr}--{model}.dcm"
                _clear_previous_masks(base_dir_mr, base_name_mr, model)
                rt_out.unlink(missing_ok=True)
                ok_nifti = run_totalsegmentator(
                    config,
                    nifti_path,
                    nifti_tmp,
                    "nifti",
                    task=task_name,
                )

                entry = {"model": model, "rtstruct": "", "masks": []}
                if ok_nifti:
                    _materialize_masks(nifti_tmp, base_dir_mr, base_name_mr, model)
                    masks_for_model = sorted(base_dir_mr.glob(f"{model}--*.nii*"))
                    if masks_for_model:
                        entry["masks"] = [
                            str(path.relative_to(base_dir_mr)) for path in masks_for_model
                        ]
                        derived = _ensure_model_rtstruct_from_masks(
                            source_dir,
                            base_dir_mr,
                            base_name_mr,
                            model,
                        )
                        if derived is not None:
                            entry["rtstruct"] = str(derived.relative_to(base_dir_mr))
                if entry["rtstruct"] or entry["masks"]:
                    manifest_mr.append(entry)

            if manifest_mr:
                try:
                    manifest_path = base_dir_mr / "manifest.json"
                    _write_manifest_atomic(
                        manifest_path,
                        {
                            "source_nifti": str(nifti_path.name),
                            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "models": manifest_mr,
                        },
                    )
                except Exception as exc:
                    logger.debug("Failed to persist MR segmentation manifest for %s: %s", base_dir_mr, exc)


def _scan_mr_series(dicom_root: Path) -> list[tuple[str, str, Path]]:
    """Return list of (patient_id, series_uid, series_dir) for MR series under dicom_root."""
    found = {}
    for base, _, files in os.walk(dicom_root):
        series_uid = None
        patient_id = None
        any_mr = False
        for fn in files:
            p = Path(base) / fn
            ds = None
            try:
                import pydicom
                ds = pydicom.dcmread(str(p), stop_before_pixels=True)
            except Exception:
                continue
            if getattr(ds, "Modality", None) != "MR":
                continue
            any_mr = True
            if series_uid is None:
                try:
                    series_uid = str(getattr(ds, "SeriesInstanceUID", ""))
                    patient_id = str(getattr(ds, "PatientID", ""))
                except Exception:
                    series_uid = None
        if any_mr and series_uid:
            found[(patient_id or "", series_uid)] = Path(base)
    return [(pid, suid, pth) for (pid, suid), pth in found.items()]


def segment_extra_models_mr(config: PipelineConfig, force: bool = False) -> None:
    """Legacy helper retained for backward compatibility (no-op)."""
    logger.info("segment_extra_models_mr is deprecated; MR segmentations are handled per-course.")
