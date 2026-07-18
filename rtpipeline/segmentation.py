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
import zipfile
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Optional, Dict, List, Sequence

import pydicom

# Modern NumPy/SciPy work fine with TotalSegmentator - no compatibility shims needed

from .config import PipelineConfig
from .inventory import TS_TASK_BY_CLASS
from .layout import build_course_dirs

logger = logging.getLogger(__name__)
_TOTALSEG_OUTPUT_TYPE_FALLBACK = {"nifti", "dicom", "dicom_rtstruct", "dicom_seg"}

# Lazy import for QC functions to avoid circular imports
_qc_module = None

def _get_qc_functions():
    """Lazy import of quality_control module to avoid circular imports."""
    global _qc_module
    if _qc_module is None:
        from . import quality_control as qc
        _qc_module = qc
    return _qc_module


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
        subprocess.run(cmd, check=True, shell=False, env=env, timeout=timeout)
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
    try:
        dest.mkdir(parents=True, exist_ok=True)
        dest_resolved = dest.resolve()

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

        if data is not None:
            with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
                _safe_extract_bundled_zip(zf, dest_resolved)
        else:
            with zipfile.ZipFile(zpath, "r") as zf:
                _safe_extract_bundled_zip(zf, dest_resolved)
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


def run_totalsegmentator(
    config: PipelineConfig,
    input_path: Path,
    output_path: Path,
    output_type: str,
    task: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> bool:
    """Run TotalSegmentator directly without compatibility wrapper."""

    # Use TotalSegmentator directly - modern NumPy/SciPy work fine
    supported_types = _totalseg_supported_output_types(config)
    if output_type and output_type not in supported_types:
        logger.info(
            "TotalSegmentator output_type '%s' not supported by current CLI; skipping direct export",
            output_type,
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
        # Run the trusted activation fragment through the explicit shell helper.
        cmd = "{}{}".format(
            _prefix(config),
            " ".join(shlex.quote(part) for part in cmd_parts),
        )
        logger.info("Running TotalSegmentator (%s, activated shell): %s", output_type, cmd)
        try:
            ok = _run(cmd, env=env)
        except RuntimeError:
            ok = False
    else:
        # Use shell=False for better security (preferred path)
        cmd_preview = " ".join(cmd_parts[:5]) + ("..." if len(cmd_parts) > 5 else "")
        logger.info("Running TotalSegmentator (%s, shell=False): %s", output_type, cmd_preview)
        try:
            ok = _run_vec(cmd_parts, env=env)
        except RuntimeError:
            ok = False

    if not ok:
        if getattr(config, "totalseg_allow_fallback", False):
            logger.info("Retrying TotalSegmentator with CPU-only and single-process env")
            # Add CPU-only flag and disable multiprocessing
            env_retry = env.copy()
            env_retry['CUDA_VISIBLE_DEVICES'] = '-1'

            if use_shell:
                cmd_retry = cmd + " -d cpu"
                try:
                    ok = _run(cmd_retry, env=env_retry)
                except RuntimeError:
                    ok = False
            else:
                cmd_parts_retry = cmd_parts + ["-d", "cpu"]
                try:
                    ok = _run_vec(cmd_parts_retry, env=env_retry)
                except RuntimeError:
                    ok = False
        else:
            logger.error("TotalSegmentator failed and fallback is disabled.")

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
    if not target.exists() or force:
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

    metadata.update(
        {
            "nifti_path": str(target),
            "source_directory": str(ct_dir),
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "modality": metadata.get("modality") or "CT",
        }
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
    prefix = f"{model}--"
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
        return bool(masks) and all((base_dir / str(m)).exists() for m in masks)
    return False


def _series_artifact_dirs(input_dir: Path) -> tuple[Path, Path]:
    if input_dir.name == "DICOM":
        return input_dir.parent / "NIFTI", input_dir.parent / "Segmentation_TotalSegmentator"
    return (
        input_dir.parent / "NIFTI" / input_dir.name,
        input_dir.parent / "Segmentation_TotalSegmentator" / input_dir.name,
    )


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
        return {}

    rows = manifest.get("series", [])
    if not isinstance(rows, list):
        logger.warning("All-series manifest for patient %s has no series list; skipping", patient_id)
        return {}

    summary: dict = {}
    segmentable_statuses = {"materialized", "segmented", "seg_failed", "seg_skipped_idempotent"}

    # Optionally restrict segmented image classes and cap 4DCT to one representative.
    # Excluded rows stay in `manifest` (written back below) with their materialized status untouched.
    seg_rows = _select_all_series_rows(config, rows)

    for row in seg_rows:
        if not isinstance(row, dict):
            continue
        task = str(row.get("ts_task") or "none")
        status = str(row.get("status") or "")
        image_class = str(row.get("image_class") or "unknown")
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

        extra_args = (
            list(getattr(config, "cbct_totalseg_extra_args", []) or [])
            if image_class == "cbct"
            else None
        )

        try:
            nifti_dir, seg_root = _series_artifact_dirs(input_dir)
            nifti_path = _ensure_ct_nifti(config, input_dir, nifti_dir, force=force, dcm2niix_depth=0)
            if nifti_path is None:
                logger.warning("All-series NIfTI conversion failed for %s", input_dir)
                row["status"] = "seg_failed"
                bucket["attempted"] += 1
                bucket["failed"] += 1
                continue

            base_name = _strip_nifti_base(nifti_path)
            seg_root.mkdir(parents=True, exist_ok=True)
            base_dir = seg_root / base_name
            base_dir.mkdir(parents=True, exist_ok=True)
            model = task

            if not force and _series_segmentation_ready(base_dir, base_name, model):
                row["status"] = "seg_skipped_idempotent"
                bucket["skipped"] += 1
                continue

            bucket["attempted"] += 1
            tmp_parent = Path(config.segmentation_temp_root) if getattr(config, "segmentation_temp_root", None) else nifti_dir.parent
            try:
                tmp_parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                tmp_parent = nifti_dir.parent

            entry: Dict[str, object] = {"model": model, "rtstruct_ok": False, "rtstruct": "", "masks": []}
            with tempfile.TemporaryDirectory(prefix="seg_series_", dir=str(tmp_parent)) as tmp_root_str:
                tmp_root = Path(tmp_root_str)
                dicom_input_tmp = tmp_root / model / "dicom"
                dicom_input_tmp.mkdir(parents=True, exist_ok=True)
                for dicom_slice in sorted(input_dir.glob("*.dcm")):
                    if dicom_slice.is_file():
                        shutil.copy2(dicom_slice, dicom_input_tmp / dicom_slice.name)
                nifti_tmp = tmp_root / model / "nifti"
                nifti_tmp.mkdir(parents=True, exist_ok=True)

                rt_out = base_dir / f"{base_name}--{model}.dcm"
                if rt_out.exists():
                    if rt_out.is_dir():
                        shutil.rmtree(rt_out, ignore_errors=True)
                    else:
                        rt_out.unlink()
                ok_dicom = run_totalsegmentator(
                    config,
                    dicom_input_tmp,
                    rt_out,
                    "dicom_rtstruct",
                    task=task,
                    extra_args=extra_args,
                )
                ok_nifti = run_totalsegmentator(
                    config,
                    nifti_path,
                    nifti_tmp,
                    "nifti",
                    task=task,
                    extra_args=extra_args,
                )

                if ok_dicom and rt_out.exists() and rt_out.is_file():
                    entry["rtstruct_ok"] = True
                elif ok_dicom and rt_out.exists() and rt_out.is_dir():
                    dicom_files = sorted(rt_out.rglob("*.dcm"))
                    if dicom_files:
                        tmp_rt = base_dir / f".{base_name}--{model}.rtstruct.tmp.dcm"
                        if tmp_rt.exists():
                            tmp_rt.unlink()
                        shutil.copy2(dicom_files[0], tmp_rt)
                        shutil.rmtree(rt_out, ignore_errors=True)
                        shutil.move(str(tmp_rt), str(rt_out))
                        entry["rtstruct_ok"] = True
                if rt_out.exists():
                    entry["rtstruct"] = str(rt_out.relative_to(base_dir))

                # Capture masks ONLY on a successful current run. On a failed retry
                # (ok_nifti False) `_materialize_masks`/`_clear_previous_masks` did not run,
                # so any masks on disk are STALE from a prior partial run; recording them here
                # would forge a completion signal and skip the course forever incomplete.
                if ok_nifti:
                    _materialize_masks(nifti_tmp, base_dir, base_name, model)
                    masks_for_model = sorted(base_dir.glob(f"{model}--*.nii*"))
                    if masks_for_model:
                        entry["masks"] = [str(p.relative_to(base_dir)) for p in masks_for_model]

            manifest_entries = [entry] if entry["rtstruct"] or entry["masks"] else []
            if manifest_entries:
                series_manifest = {
                    "source_nifti": str(nifti_path.name),
                    "source_dicom": str(input_dir),
                    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "models": manifest_entries,
                }
                _write_manifest_atomic(base_dir / "manifest.json", series_manifest)

            if ok_nifti and _series_segmentation_ready(base_dir, base_name, model):
                row["status"] = "segmented"
                bucket["segmented"] += 1
            else:
                row["status"] = "seg_failed"
                bucket["failed"] += 1
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


def segment_course(config: PipelineConfig, course_dir: Path, force: bool = False) -> dict:
    """Run TotalSegmentator for a course organised under the new directory layout."""

    course_dirs = build_course_dirs(course_dir)
    course_dirs.ensure()

    results = {"nifti": None, "dicom_seg": None, "nifti_seg_dir": None}
    ct_dir = course_dirs.dicom_ct
    if not ct_dir.exists():
        logger.warning("CT DICOM not found for %s; skipping CT segmentation", course_dir)
        _segment_mr_series_for_course(config, course_dirs, course_dir, force=force)
        return results

    nifti_path = _ensure_ct_nifti(config, ct_dir, course_dirs.nifti, force=force)
    if nifti_path is None:
        _segment_mr_series_for_course(config, course_dirs, course_dir, force=force)
        return results

    base_name = _strip_nifti_base(nifti_path)
    seg_root = course_dirs.segmentation_totalseg
    seg_root.mkdir(parents=True, exist_ok=True)
    base_dir = seg_root / base_name
    base_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: List[Dict[str, object]] = []

    results["nifti"] = str(nifti_path)

    def _model_ready(model: str) -> bool:
        legacy_dicom = base_dir / f"{model}.dcm"
        named_dicom = base_dir / f"{base_name}--{model}.dcm"
        dicom_file = named_dicom if named_dicom.exists() else legacy_dicom
        return dicom_file.exists() and _series_segmentation_ready(base_dir, base_name, model)

    models = ["total"] + [m for m in (config.extra_seg_models or []) if not m.endswith("_mr")]

    if not force and all(_model_ready(model) for model in models):
        default_named_dicom = base_dir / f"{base_name}--total.dcm"
        default_dicom = default_named_dicom if default_named_dicom.exists() else base_dir / "total.dcm"
        if default_dicom.exists():
            results["dicom_seg"] = str(default_dicom)
        results["nifti_seg_dir"] = str(base_dir)
        return results

    tmp_parent = Path(config.segmentation_temp_root) if getattr(config, "segmentation_temp_root", None) else course_dir
    try:
        Path(tmp_parent).mkdir(parents=True, exist_ok=True)
    except Exception:
        tmp_parent = course_dir

    # Track skipped models due to body region QC
    skipped_models: Dict[str, str] = {}
    body_region_qc_done = False

    with tempfile.TemporaryDirectory(prefix="seg_", dir=str(tmp_parent)) as tmp_root_str:
        tmp_root = Path(tmp_root_str)
        for model in models:
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
                            continue
                        else:
                            logger.warning(
                                "Model '%s' may not be appropriate for %s: %s (continuing anyway)",
                                model, course_dir, reason
                            )
                except Exception as exc:
                    logger.debug("Model eligibility check failed for %s: %s", model, exc)

            model_tmp = tmp_root / model
            dicom_tmp = model_tmp / "dicom"
            nifti_tmp = model_tmp / "nifti"
            dicom_tmp.mkdir(parents=True, exist_ok=True)
            nifti_tmp.mkdir(parents=True, exist_ok=True)

            model_entry: Dict[str, object] = {"model": model, "rtstruct": "", "masks": []}

            dest_dicom = base_dir / f"{base_name}--{model}.dcm"
            ok_dicom = run_totalsegmentator(config, ct_dir, dicom_tmp, "dicom_rtstruct", task=task_name)
            ok_nifti = run_totalsegmentator(config, nifti_path, nifti_tmp, "nifti", task=task_name)

            if ok_dicom:
                dicom_files = sorted(dicom_tmp.glob("*.dcm"))
                if dicom_files:
                    target = dicom_files[0]
                    if dest_dicom.exists():
                        dest_dicom.unlink()
                    shutil.copy2(target, dest_dicom)
                    if model == "total":
                        results["dicom_seg"] = str(dest_dicom)
                elif dicom_tmp.exists():
                    dicom_files = sorted(dicom_tmp.rglob("*.dcm"))
                    if dicom_files:
                        if dest_dicom.exists():
                            dest_dicom.unlink()
                        shutil.copy2(dicom_files[0], dest_dicom)
                        if model == "total":
                            results["dicom_seg"] = str(dest_dicom)
            legacy_dicom = base_dir / f"{model}.dcm"
            if legacy_dicom.exists() and legacy_dicom != dest_dicom:
                try:
                    legacy_dicom.unlink()
                except Exception:
                    logger.debug("Unable to remove legacy RTSTRUCT %s", legacy_dicom)

            if dest_dicom.exists():
                model_entry["rtstruct"] = str(dest_dicom.relative_to(base_dir))

            # Capture masks ONLY on a successful current run (see all-series path): a failed
            # retry skips `_materialize_masks`, so on-disk masks would be stale.
            if ok_nifti:
                _materialize_masks(nifti_tmp, base_dir, base_name, model)
                masks_for_model = sorted(base_dir.glob(f"{model}--*.nii*"))
                if masks_for_model:
                    model_entry["masks"] = [str(p.relative_to(base_dir)) for p in masks_for_model]
            if model_entry["rtstruct"] or model_entry["masks"]:
                manifest_entries.append(model_entry)

    default_dicom = base_dir / "total.dcm"
    if default_dicom.exists():
        results["dicom_seg"] = str(default_dicom)

    if manifest_entries or skipped_models:
        try:
            manifest = {
                "source_nifti": f"{base_name}.nii.gz",
                "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "models": manifest_entries,
            }
            if skipped_models:
                manifest["skipped_models"] = skipped_models
            _write_manifest_atomic(base_dir / "manifest.json", manifest)
        except Exception as exc:
            logger.debug("Failed to persist segmentation manifest for %s: %s", course_dir, exc)

    results["nifti_seg_dir"] = str(base_dir)
    if skipped_models:
        results["skipped_models"] = skipped_models

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
                    dicom_tmp = model_tmp / "dicom"
                    nifti_tmp = model_tmp / "nifti"
                    dicom_tmp.mkdir(parents=True, exist_ok=True)
                    nifti_tmp.mkdir(parents=True, exist_ok=True)

                    rt_out = base_dir_mr / f"{base_name_mr}--{model}.dcm"
                    ok_dicom = run_totalsegmentator(config, source_dir, dicom_tmp, "dicom_rtstruct", task=task_name)
                    ok_nifti = run_totalsegmentator(config, nifti_path, nifti_tmp, "nifti", task=task_name)

                    entry = {"model": model, "rtstruct": "", "masks": []}

                    if ok_dicom:
                        dicom_files = sorted(dicom_tmp.glob("*.dcm"))
                        if dicom_files:
                            if rt_out.exists():
                                rt_out.unlink()
                            shutil.copy2(dicom_files[0], rt_out)
                    if rt_out.exists():
                        entry["rtstruct"] = str(rt_out.relative_to(base_dir_mr))

                    # Capture masks ONLY on a successful current run (see all-series path):
                    # a failed retry skips `_materialize_masks`, so on-disk masks would be stale.
                    if ok_nifti:
                        _materialize_masks(nifti_tmp, base_dir_mr, base_name_mr, model)
                        masks_for_model = sorted(base_dir_mr.glob(f"{model}--*.nii*"))
                        if masks_for_model:
                            entry["masks"] = [str(p.relative_to(base_dir_mr)) for p in masks_for_model]
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
                dicom_tmp = model_tmp / "dicom"
                nifti_tmp = model_tmp / "nifti"
                dicom_tmp.mkdir(parents=True, exist_ok=True)
                nifti_tmp.mkdir(parents=True, exist_ok=True)

                rt_out = base_dir_mr / f"{base_name_mr}--{model}.dcm"
                ok_dicom = run_totalsegmentator(config, source_dir, dicom_tmp, "dicom_rtstruct", task=task_name)
                ok_nifti = run_totalsegmentator(config, nifti_path, nifti_tmp, "nifti", task=task_name)

                entry = {"model": model, "rtstruct": "", "masks": []}

                if ok_dicom:
                    dicom_files = sorted(dicom_tmp.glob("*.dcm"))
                    if dicom_files:
                        if rt_out.exists():
                            rt_out.unlink()
                        shutil.copy2(dicom_files[0], rt_out)
                if rt_out.exists():
                    entry["rtstruct"] = str(rt_out.relative_to(base_dir_mr))

                # Capture masks ONLY on a successful current run (see all-series path): a failed
                # retry skips `_materialize_masks`, so on-disk masks would be stale.
                if ok_nifti:
                    _materialize_masks(nifti_tmp, base_dir_mr, base_name_mr, model)
                    masks_for_model = sorted(base_dir_mr.glob(f"{model}--*.nii*"))
                    if masks_for_model:
                        entry["masks"] = [str(p.relative_to(base_dir_mr)) for p in masks_for_model]
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
