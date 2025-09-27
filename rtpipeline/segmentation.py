from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import zipfile
import stat
import io
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Optional

# Modern NumPy/SciPy work fine with TotalSegmentator - no compatibility shims needed

from .config import PipelineConfig

logger = logging.getLogger(__name__)


def _run(cmd: str, env: Optional[dict] = None) -> bool:
    """Execute a command using shell. Returns True on success."""
    
    # Detect shell to use
    shell = os.environ.get('SHELL', '/bin/bash')
    if not os.path.isfile(shell):
        shell = shutil.which('bash') or shutil.which('sh') or '/bin/sh'
    
    try:
        subprocess.run(cmd, check=True, shell=True, capture_output=True, executable=shell, env=env)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {cmd}")
        stderr = e.stderr.decode() if e.stderr else "No error output"
        logger.error(f"Error: {stderr}")
        return False


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
        if data is not None:
            with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
                zf.extractall(dest)
        else:
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(dest)
        # Search for binary inside extracted tree
        candidates = []
        for base, _, files in os.walk(dest):
            for fn in files:
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
        except Exception:
            pass
        logger.info("Using bundled dcm2niix at %s", bin_path)
        return bin_path
    except Exception as e:
        logger.error("Failed to prepare bundled dcm2niix: %s", e)
        return None


def run_dcm2niix(config: PipelineConfig, dicom_dir: Path, nifti_out: Path) -> Optional[Path]:
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

    # Use bash to run dcm2niix to avoid permission issues with bundled binaries
    if local_cmd:
        cmd = f"{_prefix(config)}bash '{cmd_name}' -z y -o '{nifti_out}' '{dicom_dir}'"
    else:
        cmd = f"{_prefix(config)}{cmd_name} -z y -o '{nifti_out}' '{dicom_dir}'"
    logger.info("Running dcm2niix: %s", cmd)
    ok = _run(cmd)
    if not ok:
        logger.warning("dcm2niix failed; continuing with DICOM-only segmentation")
        return None
    # pick first nii(.gz)
    for fn in os.listdir(nifti_out):
        if fn.endswith(".nii") or fn.endswith(".nii.gz"):
            return nifti_out / fn
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
            if "np.isdtype" in stderr or "numpy" in stderr.lower():
                logger.error(
                    "TotalSegmentator has numpy compatibility issues. "
                    "Try: pip install 'numpy>=1.20,<2.0'"
                )
                return False
            logger.warning("TotalSegmentator help command failed: %s", stderr)
        return True
    except subprocess.TimeoutExpired:
        logger.warning("TotalSegmentator validation timed out")
        return True  # Don't fail completely on timeout
    except Exception as e:
        logger.warning("TotalSegmentator validation failed: %s", e)
        return True  # Don't fail completely on validation error


def run_totalsegmentator(config: PipelineConfig, input_path: Path, output_path: Path, output_type: str, task: Optional[str] = None) -> bool:
    """Run TotalSegmentator directly without compatibility wrapper."""

    # Use TotalSegmentator directly - modern NumPy/SciPy work fine
    cmd_parts = [
        "TotalSegmentator",
        "-i", str(input_path),
        "-o", str(output_path),
        "-ot", output_type
    ]

    if task:
        cmd_parts.extend(["--task", task])

    # Set environment variables for TotalSegmentator
    if hasattr(config, 'totalseg_fast') and config.totalseg_fast:
        cmd_parts.append("--fast")

    cmd = " ".join(f'"{part}"' if " " in part else part for part in cmd_parts)

    logger.info(f"Running TotalSegmentator ({output_type}): {cmd}")

    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('OPENBLAS_NUM_THREADS', '1')
    env.setdefault('MKL_NUM_THREADS', '1')
    env.setdefault('NUMEXPR_NUM_THREADS', '1')
    env.setdefault('NUMBA_NUM_THREADS', '1')

    # First attempt with default settings
    ok = _run(cmd, env=env)

    if not ok:
        logger.info("Retrying TotalSegmentator with CPU-only and single-process env")
        # Add CPU-only flag and disable multiprocessing
        cmd_retry = cmd + " -d cpu"
        env_retry = env.copy()
        env_retry.update({
            'CUDA_VISIBLE_DEVICES': '-1',
            'OMP_NUM_THREADS': '1'
        })
        ok = _run(cmd_retry, env=env_retry)

    return ok


def segment_course(config: PipelineConfig, course_dir: Path, force: bool = False) -> dict:
    """Runs dcm2niix + TotalSegmentator for a course directory that contains CT_DICOM."""
    results = {"nifti": None, "dicom_seg": None, "nifti_seg_dir": None}
    ct_dir = course_dir / "CT_DICOM"
    if not ct_dir.exists():
        logger.warning("CT_DICOM not found for %s; skipping segmentation", course_dir)
        return results

    # Check for existing outputs to support resume
    dicom_seg_dir = course_dir / "TotalSegmentator_DICOM"
    seg_file = dicom_seg_dir / "segmentations.dcm"
    nifti_seg_dir = course_dir / "TotalSegmentator_NIFTI"
    existing_dicom = seg_file.exists() or any(dicom_seg_dir.glob("*.dcm"))
    existing_nifti = nifti_seg_dir.exists() and any(nifti_seg_dir.glob("*.nii*"))
    if (existing_dicom or existing_nifti) and not force:
        logger.info("Segmentation outputs already present for %s; skipping run", course_dir)
        if seg_file.exists():
            results["dicom_seg"] = str(seg_file)
        if existing_nifti:
            results["nifti_seg_dir"] = str(nifti_seg_dir)
        # Also capture input NIfTI series if previously converted
        nifti_out_dir = course_dir / "nifti"
        if nifti_out_dir.exists():
            for fn in nifti_out_dir.iterdir():
                if fn.suffix in (".nii", ".gz") and fn.name.endswith((".nii", ".nii.gz")):
                    results["nifti"] = str(fn)
                    break
        return results

    # Convert CT to NIfTI if not yet available for NIfTI-mode segmentation
    nifti_out_dir = course_dir / "nifti"
    nii = None
    if nifti_out_dir.exists() and any(nifti_out_dir.glob("*.nii*")):
        for fn in nifti_out_dir.iterdir():
            if fn.suffix in (".nii", ".gz") and fn.name.endswith((".nii", ".nii.gz")):
                nii = fn
                break
    else:
        nii = run_dcm2niix(config, ct_dir, nifti_out_dir)
    if nii:
        results["nifti"] = str(nii)

    # Prefer NIfTI segmentation first (more robust conversion)
    if nii is not None and (force or not (nifti_seg_dir.exists() and any(nifti_seg_dir.glob("*.nii*")))):
        ok2 = run_totalsegmentator(config, nii, nifti_seg_dir, "nifti", task=None)
        if ok2:
            results["nifti_seg_dir"] = str(nifti_seg_dir)
    # Then attempt DICOM-SEG (best-effort)
    ok1 = run_totalsegmentator(config, ct_dir, dicom_seg_dir, "dicom", task=None)
    if ok1:
        seg_file = dicom_seg_dir / "segmentations.dcm"
        if seg_file.exists():
            results["dicom_seg"] = str(seg_file)
        else:
            # Some TotalSegmentator versions may produce different filenames; pick first .dcm
            dcms = [p for p in dicom_seg_dir.glob("*.dcm")]
            if dcms:
                target = dicom_seg_dir / "segmentations.dcm"
                if dcms[0] != target:
                    try:
                        dcms[0].rename(target)
                    except Exception:
                        pass
                if target.exists():
                    results["dicom_seg"] = str(target)

    # Extra models (in addition to default 'total'): run both dicom and nifti outputs
    for model in (m for m in (config.extra_seg_models or []) if not m.endswith("_mr")):
        # DICOM
        out_d = course_dir / f"TotalSegmentator_{model}_DICOM"
        need = force or not (out_d.exists() and any(out_d.glob("*.dcm")))
        if need:
            run_totalsegmentator(config, ct_dir, out_d, "dicom", task=model)
        # NIfTI
        out_n = course_dir / f"TotalSegmentator_{model}_NIFTI"
        need = nii is not None and (force or not (out_n.exists() and any(out_n.glob("*.nii*"))))
        if need and nii is not None:
            run_totalsegmentator(config, nii, out_n, "nifti", task=model)

    return results


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
    """Run extra TotalSegmentator models for MR series found in dicom_root.
    Results are stored under output_root/<patient_id>/MR_<series_uid>/TotalSegmentator_<model>_{DICOM,NIFTI}
    """
    series = _scan_mr_series(config.dicom_root)
    if not series:
        return
    # Default MR task 'total_mr' is always included; add any extra _mr models
    models_mr = set(m for m in (config.extra_seg_models or []) if m.endswith("_mr"))
    models_mr.add("total_mr")
    if not models_mr:
        return
    total = len(series) * len(models_mr)
    if total == 0:
        return
    import time as _time
    t0 = _time.time()
    done = 0
    for pid, suid, sdir in series:
        base_out = config.output_root / (pid or "unknown") / f"MR_{suid}"
        # Convert MR DICOM to NIfTI to avoid internal DICOM conversion
        mr_nifti_dir = base_out / "nifti"
        nii_mr = run_dcm2niix(config, sdir, mr_nifti_dir)
        for model in sorted(models_mr):
            # NIfTI output preferred
            out_n = base_out / f"TotalSegmentator_{model}_NIFTI"
            need_n = force or not (out_n.exists() and any(out_n.glob("*.nii*")))
            if need_n and nii_mr is not None:
                run_totalsegmentator(config, nii_mr, out_n, "nifti", task=model)
            # DICOM output best-effort
            out_d = base_out / f"TotalSegmentator_{model}_DICOM"
            need_d = force or not (out_d.exists() and any(out_d.glob("*.dcm")))
            if need_d:
                run_totalsegmentator(config, sdir, out_d, "dicom", task=model)
            done += 1
            elapsed = _time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else float('inf')
            logger.info("Segmentation (MR extra: %s): %d/%d (%.0f%%) elapsed %.0fs ETA %.0fs", model, done, total, 100*done/total, elapsed, eta)
