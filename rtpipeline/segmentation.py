from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

from .config import PipelineConfig

logger = logging.getLogger(__name__)


def _run(cmd: str) -> bool:
    try:
        subprocess.run(cmd, check=True, shell=True, capture_output=True, executable="/bin/bash")
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Command failed: %s\n%s", cmd, e.stderr.decode(errors="ignore") if e.stderr else "")
        return False


def _prefix(config: PipelineConfig) -> str:
    return f"{config.conda_activate} && " if config.conda_activate else ""


def run_dcm2niix(config: PipelineConfig, dicom_dir: Path, nifti_out: Path) -> Optional[Path]:
    nifti_out.mkdir(parents=True, exist_ok=True)
    cmd = f"{_prefix(config)}{config.dcm2niix_cmd} -z y -o '{nifti_out}' '{dicom_dir}'"
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


def run_totalsegmentator(config: PipelineConfig, inp: Path, out_dir: Path, out_type: str) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = f"{_prefix(config)}{config.totalseg_cmd} -i '{inp}' -o '{out_dir}' -ta total -ot {out_type}"
    logger.info("Running TotalSegmentator (%s): %s", out_type, cmd)
    return _run(cmd)


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
    ok1 = run_totalsegmentator(config, ct_dir, dicom_seg_dir, "dicom")
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
        ok2 = run_totalsegmentator(config, nii, nifti_seg_dir, "nifti")
        if ok2:
            results["nifti_seg_dir"] = str(nifti_seg_dir)
    
    return results
