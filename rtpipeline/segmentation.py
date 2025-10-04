from __future__ import annotations

import datetime
import io
import json
import hashlib
import logging
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Optional, Dict, List

import pydicom

# Modern NumPy/SciPy work fine with TotalSegmentator - no compatibility shims needed

from .config import PipelineConfig
from .layout import build_course_dirs

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
    import shlex

    cmd_parts = [
        config.totalseg_cmd or "TotalSegmentator",
        "-i", str(input_path),
        "-o", str(output_path),
        "-ot", output_type,
    ]

    if task:
        cmd_parts.extend(["--task", task])

    if getattr(config, "totalseg_fast", False):
        cmd_parts.append("--fast")

    if getattr(config, "totalseg_roi_subset", None):
        cmd_parts.extend(["--roi_subset", str(config.totalseg_roi_subset)])

    if getattr(config, "totalseg_license_key", None):
        logger.debug("Using TotalSegmentator license key from config")

    cmd = "{}{}".format(
        _prefix(config),
        " ".join(shlex.quote(part) for part in cmd_parts),
    )

    logger.info("Running TotalSegmentator (%s): %s", output_type, cmd)

    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('OPENBLAS_NUM_THREADS', '1')
    env.setdefault('MKL_NUM_THREADS', '1')
    env.setdefault('NUMEXPR_NUM_THREADS', '1')
    env.setdefault('NUMBA_NUM_THREADS', '1')

    if getattr(config, "totalseg_license_key", None):
        env.setdefault("TOTALSEG_LICENSE", str(config.totalseg_license_key))
    if getattr(config, "totalseg_weights_dir", None):
        env.setdefault("TOTALSEG_WEIGHTS_PATH", str(config.totalseg_weights_dir))

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
    metadata["sop_hash"] = hashlib.sha1(concat.encode("utf-8")).hexdigest() if concat else ""
    return metadata


def _ensure_ct_nifti(
    config: PipelineConfig,
    ct_dir: Path,
    nifti_dir: Path,
    force: bool = False,
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
        generated = run_dcm2niix(config, ct_dir, tmp_dir)
        if generated is None:
            logger.error("dcm2niix failed for %s", ct_dir)
            return None
        if target.exists():
            target.unlink()
        shutil.move(str(generated), str(target))
        for leftover in tmp_dir.iterdir():
            if leftover.is_file():
                leftover.unlink()
        tmp_dir.rmdir()

    metadata.update(
        {
            "nifti_path": str(target),
            "source_directory": str(ct_dir),
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
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

def segment_course(config: PipelineConfig, course_dir: Path, force: bool = False) -> dict:
    """Run TotalSegmentator for a course organised under the new directory layout."""

    course_dirs = build_course_dirs(course_dir)
    course_dirs.ensure()

    results = {"nifti": None, "dicom_seg": None, "nifti_seg_dir": None}
    ct_dir = course_dirs.dicom_ct
    if not ct_dir.exists():
        logger.warning("CT DICOM not found for %s; skipping segmentation", course_dir)
        return results

    nifti_path = _ensure_ct_nifti(config, ct_dir, course_dirs.nifti, force=force)
    if nifti_path is None:
        return results

    base_name = _strip_nifti_base(nifti_path)
    seg_root = course_dirs.segmentation_totalseg
    seg_root.mkdir(parents=True, exist_ok=True)
    base_dir = seg_root / base_name
    base_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: List[Dict[str, object]] = []

    results["nifti"] = str(nifti_path)

    def _model_ready(model: str) -> bool:
        dicom_file = base_dir / f"{model}.dcm"
        mask_files = list(base_dir.glob(f"{model}--*.nii*"))
        return dicom_file.exists() and len(mask_files) > 0

    models = ["total"] + [m for m in (config.extra_seg_models or []) if not m.endswith("_mr")]

    if not force and all(_model_ready(model) for model in models):
        default_dicom = base_dir / f"total.dcm"
        if default_dicom.exists():
            results["dicom_seg"] = str(default_dicom)
        results["nifti_seg_dir"] = str(base_dir)
        return results

    with tempfile.TemporaryDirectory(prefix="seg_", dir=str(course_dir)) as tmp_root_str:
        tmp_root = Path(tmp_root_str)
        for model in models:
            task_name = None if model == "total" else model
            model_tmp = tmp_root / model
            dicom_tmp = model_tmp / "dicom"
            nifti_tmp = model_tmp / "nifti"
            dicom_tmp.mkdir(parents=True, exist_ok=True)
            nifti_tmp.mkdir(parents=True, exist_ok=True)

            model_entry: Dict[str, object] = {"model": model, "rtstruct": "", "masks": []}

            dest_dicom = base_dir / f"{model}.dcm"
            ok_dicom = run_totalsegmentator(config, ct_dir, dicom_tmp, "dicom", task=task_name)
            ok_nifti = run_totalsegmentator(config, nifti_path, nifti_tmp, "nifti", task=task_name)

            if ok_dicom:
                dicom_files = sorted(dicom_tmp.glob("*.dcm"))
                if dicom_files:
                    if dest_dicom.exists():
                        dest_dicom.unlink()
                    shutil.copy2(dicom_files[0], dest_dicom)
                    if model == "total":
                        results["dicom_seg"] = str(dest_dicom)

            if dest_dicom.exists():
                model_entry["rtstruct"] = str(dest_dicom.relative_to(base_dir))

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

    if manifest_entries:
        try:
            manifest = {
                "source_nifti": f"{base_name}.nii.gz",
                "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
                "models": manifest_entries,
            }
            (base_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("Failed to persist segmentation manifest for %s: %s", course_dir, exc)

    results["nifti_seg_dir"] = str(base_dir)

    # ------------------------------------------------------------------
    # MR segmentation for auxiliary series in DICOM_related/
    # ------------------------------------------------------------------
    mr_models = [m for m in (config.extra_seg_models or []) if m.endswith("_mr")]
    if "total_mr" not in mr_models:
        mr_models.append("total_mr")
    mr_models = sorted({m.strip() for m in mr_models if m.strip()})

    if mr_models:
        for meta_file in sorted(course_dirs.nifti.glob("*.metadata.json")):
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            modality = str(meta.get("modality", "")).upper()
            if modality != "MR":
                continue
            nifti_str = meta.get("nifti_path")
            source_str = meta.get("source_directory")
            if not isinstance(nifti_str, str) or not nifti_str:
                continue
            nifti_path = Path(nifti_str)
            if not nifti_path.exists():
                continue
            source_dir = Path(source_str) if isinstance(source_str, str) else None
            base_name_mr = _strip_nifti_base(nifti_path)
            base_dir_mr = seg_root / base_name_mr
            base_dir_mr.mkdir(parents=True, exist_ok=True)

            def _mr_ready(model: str) -> bool:
                rt_path = base_dir_mr / f"{base_name_mr}--{model}.dcm"
                mask_paths = list(base_dir_mr.glob(f"{base_name_mr}--{model}--*.nii*"))
                return rt_path.exists() and len(mask_paths) > 0

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
                    ok_dicom = False
                    if source_dir and source_dir.exists():
                        ok_dicom = run_totalsegmentator(config, source_dir, dicom_tmp, "dicom", task=task_name)
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

                    if ok_nifti:
                        _materialize_masks(nifti_tmp, base_dir_mr, base_name_mr, model)

                    masks_for_model = sorted(base_dir_mr.glob(f"{base_name_mr}--{model}--*.nii*"))
                    if masks_for_model:
                        entry["masks"] = [str(p.relative_to(base_dir_mr)) for p in masks_for_model]
                    if entry["rtstruct"] or entry["masks"]:
                        manifest_mr.append(entry)

                if manifest_mr:
                    try:
                        manifest_path = base_dir_mr / "manifest.json"
                        manifest_path.write_text(
                            json.dumps(
                                {
                                    "source_nifti": str(nifti_path.name),
                                    "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
                                    "models": manifest_mr,
                                },
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                    except Exception as exc:
                        logger.debug("Failed to persist MR segmentation manifest for %s: %s", base_dir_mr, exc)

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
    """Legacy helper retained for backward compatibility (no-op)."""
    logger.info("segment_extra_models_mr is deprecated; MR segmentations are handled per-course.")
