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

from .config import PipelineConfig

logger = logging.getLogger(__name__)


def _run(cmd: str, env: Optional[dict] = None) -> bool:
    try:
        subprocess.run(cmd, check=True, shell=True, capture_output=True, executable="/bin/bash", env=env)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Command failed: %s\n%s", cmd, e.stderr.decode(errors="ignore") if e.stderr else "")
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


def run_totalsegmentator(config: PipelineConfig, inp: Path, out_dir: Path, out_type: str, task: Optional[str] = None) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Verify command availability (when no conda prefix is used)
    if not config.conda_activate and shutil.which(config.totalseg_cmd) is None:
        logger.error("TotalSegmentator command '%s' not found; cannot run segmentation", config.totalseg_cmd)
        return False
    # Prepare environment (license if provided)
    env = os.environ.copy()
    if config.totalseg_license_key:
        env.setdefault("TOTALSEG_LICENSE", config.totalseg_license_key)
        env.setdefault("TOTALSEGMENTATOR_LICENSE", config.totalseg_license_key)
    extra_flags = []
    if config.totalseg_fast:
        extra_flags.append("--fast")
    if config.totalseg_roi_subset:
        extra_flags.append(f"--roi_subset {config.totalseg_roi_subset}")
    base = f"{_prefix(config)}{config.totalseg_cmd} -i '{inp}' -o '{out_dir}' -ot {out_type} {' '.join(extra_flags)}".strip()
    if not task:
        cmd = base
        logger.info("Running TotalSegmentator (%s): %s", out_type, cmd)
        return _run(cmd, env=env)
    # Try a few different task flags for compatibility across versions
    for flag in ("-ta", "--task", "-m"):
        cmd = f"{base} {flag} {task}"
        logger.info("Running TotalSegmentator (%s, task=%s): %s", out_type, task, cmd)
        if _run(cmd, env=env):
            return True
    logger.error("TotalSegmentator failed for task '%s' with all known flags", task)
    return False


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

    # Run DICOM-SEG (force or not present)
    # Always run default 'total' model and never modify it
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

    # Run NIfTI segmentation (force or not present)
    if nii is not None and (force or not (nifti_seg_dir.exists() and any(nifti_seg_dir.glob("*.nii*")))):
        ok2 = run_totalsegmentator(config, nii, nifti_seg_dir, "nifti", task=None)
        if ok2:
            results["nifti_seg_dir"] = str(nifti_seg_dir)

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
    if not config.extra_seg_models:
        return
    series = _scan_mr_series(config.dicom_root)
    if not series:
        return
    # Only models ending with _mr are considered for MR
    models_mr = [m for m in (config.extra_seg_models or []) if m.endswith("_mr")]
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
        for model in models_mr:
            # DICOM output
            out_d = base_out / f"TotalSegmentator_{model}_DICOM"
            if force or not (out_d.exists() and any(out_d.glob("*.dcm"))):
                run_totalsegmentator(config, sdir, out_d, "dicom", task=model)
            # NIfTI output
            out_n = base_out / f"TotalSegmentator_{model}_NIFTI"
            if force or not (out_n.exists() and any(out_n.glob("*.nii*"))):
                run_totalsegmentator(config, sdir, out_n, "nifti", task=model)
            done += 1
            elapsed = _time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else float('inf')
            logger.info("Segmentation (MR extra: %s): %d/%d (%.0f%%) elapsed %.0fs ETA %.0fs", model, done, total, 100*done/total, elapsed, eta)
