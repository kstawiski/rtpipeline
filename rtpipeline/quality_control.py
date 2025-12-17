"""Quality control and validation functions for DICOM-RT processing pipeline."""

from __future__ import annotations
import logging
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pydicom
import numpy as np
import pandas as pd
from datetime import datetime

from .layout import build_course_dirs
from .utils import mask_is_cropped

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

logger = logging.getLogger(__name__)


# =============================================================================
# Body Region Detection Configuration
# =============================================================================
# Based on consensus from GPT-5.2 and Gemini-3-pro-preview:
# - Vertebrae as primary anchors (most deterministic)
# - Multi-anchor combinations for robustness
# - Low presence thresholds to detect partial coverage (not validate anatomy)
# - Additive boolean flags (whole-body = all True)

BODY_REGION_ANCHORS = {
    "HEAD_NECK": {
        "vertebrae": ["vertebrae_C1", "vertebrae_C2", "vertebrae_C3", "vertebrae_C4",
                      "vertebrae_C5", "vertebrae_C6", "vertebrae_C7"],
        "organs": ["brain", "skull"],
        "min_vertebrae": 1,  # At least one C-spine vertebra
        "organ_thresholds_ml": {"brain": 50.0, "skull": 50.0},  # mL, low to detect partial
    },
    "THORAX": {
        "vertebrae": ["vertebrae_T1", "vertebrae_T2", "vertebrae_T3", "vertebrae_T4",
                      "vertebrae_T5", "vertebrae_T6", "vertebrae_T7", "vertebrae_T8",
                      "vertebrae_T9", "vertebrae_T10", "vertebrae_T11", "vertebrae_T12"],
        "organs": ["lung_upper_lobe_left", "lung_lower_lobe_left",
                   "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
                   "heart"],
        "min_vertebrae": 3,  # At least 3 T-spine vertebrae
        "organ_thresholds_ml": {"lungs_combined": 200.0, "heart": 50.0},
    },
    "ABDOMEN": {
        "vertebrae": ["vertebrae_L1", "vertebrae_L2", "vertebrae_L3",
                      "vertebrae_L4", "vertebrae_L5"],
        "organs": ["liver", "spleen", "kidney_left", "kidney_right"],
        "min_vertebrae": 2,  # At least 2 L-spine vertebrae
        "organ_thresholds_ml": {"liver": 100.0, "spleen": 30.0, "kidney_combined": 40.0},
    },
    "PELVIS": {
        "vertebrae": ["sacrum", "vertebrae_S1"],
        "organs": ["hip_left", "hip_right", "femur_left", "femur_right", "urinary_bladder"],
        "min_vertebrae": 1,  # Sacrum present
        "organ_thresholds_ml": {"hip_combined": 30.0, "urinary_bladder": 10.0},
    },
}

# Default model requirements for gating (fallback when no config provided)
# Production code should use config.model_region_requirements instead
_DEFAULT_MODEL_REGION_REQUIREMENTS = {
    "heartchambers_highres": {"required_regions": ["THORAX"], "min_confidence": 0.5},
    "coronary_arteries": {"required_regions": ["THORAX"], "min_confidence": 0.6},
    "head_neck_oar": {"required_regions": ["HEAD_NECK"], "min_confidence": 0.5},
    "total": {"required_regions": []},
    "total_mr": {"required_regions": []},
}


class QualityControlError(Exception):
    """Exception raised when quality control checks fail."""
    pass


class DICOMValidator:
    """Validates DICOM files for consistency and integrity."""

    def __init__(self, course_dir: Path, skip_structure_cropping: bool = False):
        """Initialize validator.

        Args:
            course_dir: Path to course directory
            skip_structure_cropping: If True, skip the expensive structure cropping check
                                     (significantly faster but won't detect boundary issues)
        """
        self.course_dir = course_dir
        self.dirs = build_course_dirs(course_dir)
        self.ct_dir = self.dirs.dicom_ct
        self.rp_path = course_dir / "RP.dcm"
        self.rd_path = course_dir / "RD.dcm"
        self.rs_path = course_dir / "RS.dcm"
        self.rs_auto_path = course_dir / "RS_auto.dcm"
        self.skip_structure_cropping = skip_structure_cropping

    def validate_all(self) -> Dict[str, Any]:
        """Run all validation checks and return summary."""
        results = {
            "patient_id": self.course_dir.parent.name,
            "course_id": self.course_dir.name,
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }

        try:
            # Basic file existence
            results["checks"]["file_existence"] = self._check_file_existence()

            # DICOM header validation
            if results["checks"]["file_existence"]["ct_exists"]:
                results["checks"]["ct_validation"] = self._validate_ct_series()

            if results["checks"]["file_existence"]["rp_exists"]:
                results["checks"]["rp_validation"] = self._validate_rt_plan()

            if results["checks"]["file_existence"]["rd_exists"]:
                results["checks"]["rd_validation"] = self._validate_rt_dose()

            # Cross-modality consistency
            results["checks"]["consistency"] = self._check_consistency()

            # Segmentation cropping detection (can be skipped for performance)
            if not self.skip_structure_cropping:
                results["checks"]["structure_cropping"] = self._segmentation_cropping()
            else:
                results["checks"]["structure_cropping"] = {"status": "SKIP", "reason": "skipped by configuration"}

            # Calculate overall status
            results["overall_status"] = self._calculate_overall_status(results["checks"])

        except Exception as e:
            logger.error(f"Quality control failed for {self.course_dir}: {e}")
            results["error"] = str(e)
            results["overall_status"] = "ERROR"

        return results

    def _check_file_existence(self) -> Dict[str, bool]:
        """Check if required files exist."""
        return {
            "ct_exists": self.ct_dir.exists() and any(self.ct_dir.glob("*.dcm")),
            "rp_exists": self.rp_path.exists(),
            "rd_exists": self.rd_path.exists(),
            "rs_exists": self.rs_path.exists(),
            "rs_auto_exists": self.rs_auto_path.exists()
        }

    def _validate_ct_series(self) -> Dict[str, Any]:
        """Validate CT series for consistency."""
        ct_files = sorted(self.ct_dir.glob("*.dcm"))
        if not ct_files:
            return {"status": "FAIL", "error": "No DICOM files found in CT directory"}

        try:
            first_ds = pydicom.dcmread(ct_files[0])
            validation = {
                "status": "PASS",
                "num_slices": len(ct_files),
                "modality": getattr(first_ds, 'Modality', 'Unknown'),
                "slice_thickness": getattr(first_ds, 'SliceThickness', None),
                "pixel_spacing": getattr(first_ds, 'PixelSpacing', None),
                "frame_of_reference_uid": getattr(first_ds, 'FrameOfReferenceUID', None),
                "issues": []
            }

            # Check if this is actually a CT
            if validation["modality"] != "CT":
                validation["issues"].append(f"Expected CT modality, got {validation['modality']}")

            # Check slice thickness consistency
            slice_thicknesses = []
            frame_refs = []

            for dcm_file in ct_files[:min(10, len(ct_files))]:  # Sample first 10 files
                ds = pydicom.dcmread(dcm_file)
                if hasattr(ds, 'SliceThickness'):
                    slice_thicknesses.append(float(ds.SliceThickness))
                if hasattr(ds, 'FrameOfReferenceUID'):
                    frame_refs.append(ds.FrameOfReferenceUID)

            if slice_thicknesses and len(set(slice_thicknesses)) > 1:
                validation["issues"].append(f"Inconsistent slice thickness: {set(slice_thicknesses)}")

            if frame_refs and len(set(frame_refs)) > 1:
                validation["issues"].append(f"Multiple frame of reference UIDs: {len(set(frame_refs))}")

            if validation["issues"]:
                validation["status"] = "WARNING"

        except Exception as e:
            validation = {"status": "FAIL", "error": str(e)}

        return validation

    def _validate_rt_plan(self) -> Dict[str, Any]:
        """Validate RTPLAN file."""
        try:
            ds = pydicom.dcmread(self.rp_path)
            validation = {
                "status": "PASS",
                "sop_class_uid": getattr(ds, 'SOPClassUID', None),
                "modality": getattr(ds, 'Modality', None),
                "frame_of_reference_uid": getattr(ds, 'FrameOfReferenceUID', None),
                "issues": []
            }

            # Check SOP Class UID for RTPLAN
            expected_rtplan_uids = [
                "1.2.840.10008.5.1.4.1.1.481.5",  # Standard DICOM RTPLAN
                "1.2.246.352.70.1.70"  # Vendor-specific RTPLAN (commonly used)
            ]
            if validation["sop_class_uid"] not in expected_rtplan_uids:
                validation["issues"].append(f"Unexpected SOP Class UID for RTPLAN: {validation['sop_class_uid']}")

            if validation["modality"] != "RTPLAN":
                validation["issues"].append(f"Expected RTPLAN modality, got {validation['modality']}")

            if validation["issues"]:
                validation["status"] = "WARNING"

        except Exception as e:
            validation = {"status": "FAIL", "error": str(e)}

        return validation

    def _validate_rt_dose(self) -> Dict[str, Any]:
        """Validate RTDOSE file."""
        try:
            ds = pydicom.dcmread(self.rd_path)
            validation = {
                "status": "PASS",
                "sop_class_uid": getattr(ds, 'SOPClassUID', None),
                "modality": getattr(ds, 'Modality', None),
                "frame_of_reference_uid": getattr(ds, 'FrameOfReferenceUID', None),
                "dose_grid_scaling": getattr(ds, 'DoseGridScaling', None),
                "dose_units": getattr(ds, 'DoseUnits', None),
                "dose_type": getattr(ds, 'DoseType', None),
                "dose_summation_type": getattr(ds, 'DoseSummationType', None),
                "issues": []
            }

            # Check SOP Class UID for RTDOSE
            expected_rtdose_uid = "1.2.840.10008.5.1.4.1.1.481.2"
            if validation["sop_class_uid"] != expected_rtdose_uid:
                validation["issues"].append(f"Unexpected SOP Class UID for RTDOSE: {validation['sop_class_uid']}")

            if validation["modality"] != "RTDOSE":
                validation["issues"].append(f"Expected RTDOSE modality, got {validation['modality']}")

            if validation["dose_grid_scaling"] is None:
                validation["issues"].append("Missing DoseGridScaling")

            # Dose units/type sanity checks (header-only; do not force PixelData read)
            dose_units = str(validation.get("dose_units") or "").strip().upper()
            if dose_units and dose_units not in {"GY", "CGY"}:
                validation["issues"].append(f"Unexpected DoseUnits: {validation.get('dose_units')}")
            if not dose_units:
                validation["issues"].append("Missing DoseUnits")

            dose_type = str(validation.get("dose_type") or "").strip().upper()
            if dose_type and dose_type not in {"PHYSICAL"}:
                # Many TPS export PHYSICAL; flag anything else for review.
                validation["issues"].append(f"Unexpected DoseType: {validation.get('dose_type')}")

            dose_sum = str(validation.get("dose_summation_type") or "").strip().upper()
            if not dose_sum:
                validation["issues"].append("Missing DoseSummationType")

            if validation["issues"]:
                validation["status"] = "WARNING"

        except Exception as e:
            validation = {"status": "FAIL", "error": str(e)}

        return validation

    def _check_consistency(self) -> Dict[str, Any]:
        """Check consistency across different DICOM files."""
        consistency = {
            "status": "PASS",
            "frame_of_reference_consistency": True,
            "issues": []
        }

        try:
            frame_refs = []

            # Collect frame of reference UIDs
            if self.ct_dir.exists():
                ct_files = list(self.ct_dir.glob("*.dcm"))
                if ct_files:
                    ct_ds = pydicom.dcmread(ct_files[0])
                    if hasattr(ct_ds, 'FrameOfReferenceUID'):
                        frame_refs.append(("CT", ct_ds.FrameOfReferenceUID))

            for file_path, modality in [(self.rp_path, "RTPLAN"), (self.rd_path, "RTDOSE")]:
                if file_path.exists():
                    ds = pydicom.dcmread(file_path)
                    if hasattr(ds, 'FrameOfReferenceUID'):
                        frame_refs.append((modality, ds.FrameOfReferenceUID))

            # Check if all frame of reference UIDs are the same
            if frame_refs:
                unique_refs = set(ref[1] for ref in frame_refs)
                if len(unique_refs) > 1:
                    consistency["frame_of_reference_consistency"] = False
                    consistency["issues"].append(f"Inconsistent Frame of Reference UIDs: {dict(frame_refs)}")
                    consistency["status"] = "WARNING"

        except Exception as e:
            consistency["status"] = "FAIL"
            consistency["error"] = str(e)

        return consistency

    def _segmentation_cropping(self) -> Dict[str, Any]:
        """Check for cropped structures with optimized processing.

        This method checks if any ROI masks touch the image boundaries,
        which indicates the structure may have been cropped by the CT scan field of view.

        Notes
        -----
        - Systematic CT cropping in this pipeline is **axial-only (z)**. Therefore, masks
          touching the in-plane boundaries (x/y) are stronger evidence of problematic
          acquisition truncation (true FOV crop) and are escalated to WARNING.
        - z-only boundary touches are common for pelvic RT planning CTs with limited
          cranio-caudal coverage and are reported as INFO by default.
        """
        info = {
            "status": "PASS",
            "structures": [],
        }
        cropping_metadata: Dict[str, Any] = {}
        crop_meta_path = self.course_dir / "cropping_metadata.json"
        if crop_meta_path.exists():
            try:
                cropping_metadata = json.loads(crop_meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.debug("Unable to read cropping metadata for %s: %s", self.course_dir, exc)
                cropping_metadata = {}
        try:
            from rt_utils import RTStructBuilder
        except Exception:
            info["status"] = "SKIP"
            return info

        if not self.ct_dir.exists():
            info["status"] = "SKIP"
            return info

        # Quick check: verify CT files exist before processing
        ct_files = list(self.ct_dir.glob("*.dcm"))
        if not ct_files:
            info["status"] = "SKIP"
            return info
        ct_num_slices = len(ct_files)

        def _axis_name_map(z_axis: int | None) -> dict[int, str]:
            # rt_utils masks are typically either [z, y, x] or [y, x, z]. We identify the
            # z axis by matching the CT slice count; if ambiguous, we fall back to axisN.
            if z_axis == 0:
                return {0: "z", 1: "y", 2: "x"}
            if z_axis == 2:
                return {0: "y", 1: "x", 2: "z"}
            if z_axis == 1:
                return {0: "y", 1: "z", 2: "x"}
            return {0: "axis0", 1: "axis1", 2: "axis2"}

        def _cropping_sides(mask: np.ndarray) -> tuple[list[str], bool]:
            """Return (cropped_sides, inplane_cropped) for a binary mask."""
            arr = np.asarray(mask).astype(bool)
            if arr.ndim != 3 or not arr.any():
                return [], False

            z_axis: int | None = None
            candidates = [i for i, dim in enumerate(arr.shape) if dim == ct_num_slices]
            if len(candidates) == 1:
                z_axis = candidates[0]

            names = _axis_name_map(z_axis)
            touched_axes: set[int] = set()
            sides: list[str] = []
            for axis in range(arr.ndim):
                slicer = [slice(None)] * arr.ndim
                slicer[axis] = 0
                if arr[tuple(slicer)].any():
                    touched_axes.add(axis)
                    sides.append(f"{names[axis]}_min")
                slicer[axis] = arr.shape[axis] - 1
                if arr[tuple(slicer)].any():
                    touched_axes.add(axis)
                    sides.append(f"{names[axis]}_max")

            if not sides:
                return [], False

            if z_axis is None:
                # Conservative: if we cannot infer z axis, treat any crop as potentially in-plane.
                return sides, True
            inplane_cropped = any(axis != z_axis for axis in touched_axes)
            return sides, inplane_cropped

        def _evaluate(rs_path: Path, source: str) -> None:
            """Evaluate cropping for a single structure set."""
            if not rs_path.exists():
                return
            try:
                builder = RTStructBuilder.create_from(
                    dicom_series_path=str(self.ct_dir),
                    rt_struct_path=str(rs_path),
                )
            except Exception as exc:
                info.setdefault("errors", []).append({"source": source, "error": str(exc)})
                return

            # Process ROIs with early exit optimizations
            roi_names = builder.get_roi_names()
            for roi_name in roi_names:
                try:
                    mask = builder.get_roi_mask_by_name(roi_name)
                except Exception:
                    # Skip ROIs that fail to load
                    continue
                if mask is None:
                    continue
                # Optimization: skip empty masks early (avoids boundary checking)
                if not np.any(mask):
                    continue
                if not mask_is_cropped(mask):
                    continue
                cropped_sides, inplane_cropped = _cropping_sides(mask)
                if not cropped_sides:
                    continue

                roi_norm = str(roi_name).strip().lower()
                ignore_for_status = "couch" in roi_norm

                entry: Dict[str, Any] = {
                    "source": source,
                    "roi_name": roi_name,
                    "cropped_sides": cropped_sides,
                    "inplane_cropped": bool(inplane_cropped),
                    "ignore_for_status": bool(ignore_for_status),
                    # Expected if cropping occurs only in z (limited scan range or systematic cropping).
                    "expected_due_to_ct_crop": not bool(inplane_cropped),
                    "cropping_metadata_present": bool(cropping_metadata),
                }
                if cropping_metadata:
                    entry["crop_region"] = cropping_metadata.get("region")
                    entry["clamped_axes"] = cropping_metadata.get("clamped_axes", [])
                info["structures"].append(entry)

        # Evaluate both structure sets (manual and auto)
        _evaluate(self.rs_path, "manual")
        _evaluate(self.rs_auto_path, "auto")

        if info["structures"]:
            # Escalate only when there is evidence of in-plane truncation (true FOV crop),
            # ignoring couch-related helper ROIs.
            unexpected = [
                entry
                for entry in info["structures"]
                if entry.get("inplane_cropped") and not entry.get("ignore_for_status")
            ]
            info["status"] = "WARNING" if unexpected else "INFO"
        return info

    def _calculate_overall_status(self, checks: Dict[str, Any]) -> str:
        """Calculate overall validation status."""
        statuses = []
        for check_name, check_result in checks.items():
            if isinstance(check_result, dict) and "status" in check_result:
                statuses.append(check_result["status"])

        if "FAIL" in statuses or "ERROR" in statuses:
            return "FAIL"
        elif "WARNING" in statuses:
            return "WARNING"
        else:
            return "PASS"


def calculate_file_checksum(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def validate_segmentation_volumes(course_dir: Path) -> Dict[str, Any]:
    """Validate segmentation volumes for outliers."""
    validation = {
        "status": "PASS",
        "volumes": {},
        "issues": []
    }

    try:
        rs_auto_path = course_dir / "RS_auto.dcm"
        if rs_auto_path.exists():
            # This would need rt_utils or similar to read structure volumes
            # Placeholder for volume validation logic
            validation["status"] = "PASS"
    except Exception as e:
        validation["status"] = "FAIL"
        validation["error"] = str(e)

    return validation


def generate_qc_report(course_dir: Path, output_dir: Path, skip_structure_cropping: bool = False) -> Path:
    """Generate a comprehensive QC report for a course.

    Args:
        course_dir: Path to the course directory
        output_dir: Path where QC report will be saved
        skip_structure_cropping: If True, skip expensive structure cropping check for better performance

    Returns:
        Path to the generated QC report
    """
    validator = DICOMValidator(course_dir, skip_structure_cropping=skip_structure_cropping)
    qc_results = validator.validate_all()

    # Add segmentation volume validation
    qc_results["segmentation_validation"] = validate_segmentation_volumes(course_dir)

    # Save report
    report_path = output_dir / f"qc_report_{course_dir.parent.name}_{course_dir.name}.json"
    with open(report_path, 'w') as f:
        json.dump(qc_results, f, indent=2, default=str)

    logger.info(f"QC report saved to {report_path}")
    return report_path


def check_pipeline_requirements() -> Dict[str, Any]:
    """Check if all required software and dependencies are available."""
    requirements = {
        "timestamp": datetime.now().isoformat(),
        "checks": {}
    }

    try:
        import subprocess

        # Check TotalSegmentator
        try:
            result = subprocess.run(["TotalSegmentator", "--version"],
                                  capture_output=True, text=True, timeout=10)
            requirements["checks"]["totalsegmentator"] = {
                "available": result.returncode == 0,
                "version": result.stdout.strip() if result.returncode == 0 else None
            }
        except (subprocess.TimeoutExpired, FileNotFoundError):
            requirements["checks"]["totalsegmentator"] = {"available": False}

        # Check GPU availability
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
            requirements["checks"]["gpu"] = {"available": result.returncode == 0}
        except (subprocess.TimeoutExpired, FileNotFoundError):
            requirements["checks"]["gpu"] = {"available": False}

        # Check Python packages
        packages = ["pydicom", "SimpleITK", "pyradiomics", "pandas", "numpy"]
        for package in packages:
            try:
                __import__(package)
                requirements["checks"][package] = {"available": True}
            except ImportError:
                requirements["checks"][package] = {"available": False}

    except Exception as e:
        requirements["error"] = str(e)

    return requirements


# =============================================================================
# Body Region Detection QC
# =============================================================================

def _compute_mask_volume_ml(nifti_path: Path) -> Tuple[float, int]:
    """Compute volume in mL for a NIfTI mask file.

    Returns:
        Tuple of (volume_ml, voxel_count)
    """
    if sitk is None:
        raise ImportError("SimpleITK is required for body region detection")

    img = sitk.ReadImage(str(nifti_path))
    arr = sitk.GetArrayFromImage(img)
    voxel_count = int(np.sum(arr > 0))

    if voxel_count == 0:
        return 0.0, 0

    spacing = img.GetSpacing()  # (x, y, z) in mm
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    volume_ml = (voxel_count * voxel_volume_mm3) / 1000.0  # mm³ to mL

    return volume_ml, voxel_count


def _run_totalseg_tool(
    tool_name: str,
    input_path: Path,
    output_path: Path,
    conda_activate: Optional[str] = None,
    timeout: int = 300,
) -> Tuple[bool, Optional[Dict]]:
    """Run a TotalSegmentator auxiliary tool (totalseg_get_phase, totalseg_get_modality).

    Args:
        tool_name: Command name (e.g., 'totalseg_get_phase', 'totalseg_get_modality')
        input_path: Path to input NIfTI file
        output_path: Path to output JSON file
        conda_activate: Optional conda activation command
        timeout: Timeout in seconds

    Returns:
        Tuple of (success: bool, result_dict: Optional[Dict])
    """
    import subprocess
    import shlex
    import shutil

    # Build command
    cmd_parts = [tool_name, "-i", str(input_path), "-o", str(output_path)]

    # Check if tool is available
    if not conda_activate and not shutil.which(tool_name):
        logger.debug("%s not found in PATH", tool_name)
        return False, None

    # Build full command with optional conda activation
    if conda_activate:
        cmd = f"{conda_activate} && {' '.join(shlex.quote(p) for p in cmd_parts)}"
        shell = True
    else:
        cmd = cmd_parts
        shell = False

    try:
        logger.debug("Running %s: %s", tool_name, cmd if shell else " ".join(cmd_parts))
        result = subprocess.run(
            cmd,
            shell=shell,
            capture_output=True,
            timeout=timeout,
            env=os.environ.copy(),
        )

        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
            logger.debug("%s failed: %s", tool_name, stderr[:500])
            return False, None

        # Read output JSON
        if output_path.exists():
            try:
                with open(output_path) as f:
                    return True, json.load(f)
            except Exception as e:
                logger.debug("Failed to read %s output: %s", tool_name, e)
                return False, None

        return False, None

    except subprocess.TimeoutExpired:
        logger.warning("%s timed out after %ds", tool_name, timeout)
        return False, None
    except Exception as e:
        logger.debug("%s failed: %s", tool_name, e)
        return False, None


def detect_contrast_phase(
    nifti_path: Path,
    output_dir: Optional[Path] = None,
    conda_activate: Optional[str] = None,
) -> Dict[str, Any]:
    """Detect contrast phase of a CT image using TotalSegmentator's totalseg_get_phase.

    Args:
        nifti_path: Path to input CT NIfTI file
        output_dir: Optional directory for output JSON (uses temp if not provided)
        conda_activate: Optional conda activation command

    Returns:
        Dict with phase classification results:
        - phase: str (native, arterial_early, arterial_late, portal_venous, or unknown)
        - probabilities: Dict[str, float] for each phase
        - pi_time: Optional predicted post-injection time
        - status: success/error/unavailable
    """
    import tempfile

    result = {
        "phase": "unknown",
        "probabilities": {},
        "pi_time": None,
        "status": "unavailable",
        "tool": "totalseg_get_phase",
    }

    if not nifti_path.exists():
        result["status"] = "error"
        result["error"] = "Input NIfTI file not found"
        return result

    # Determine output path
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_json = output_dir / "contrast_phase.json"
    else:
        # Use temporary file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_json = Path(f.name)

    try:
        success, phase_data = _run_totalseg_tool(
            "totalseg_get_phase",
            nifti_path,
            output_json,
            conda_activate=conda_activate,
        )

        if success and phase_data:
            result["status"] = "success"
            # Parse TotalSegmentator phase output format
            # Expected: {"phase": "portal_venous", "probabilities": {...}, "pi_time": 60.5}
            result["phase"] = phase_data.get("phase", "unknown")
            result["probabilities"] = phase_data.get("probabilities", {})
            result["pi_time"] = phase_data.get("pi_time")
            # Also capture raw output for debugging
            result["raw"] = phase_data
        else:
            result["status"] = "unavailable"

    finally:
        # Cleanup temp file if used
        if not output_dir and output_json.exists():
            try:
                output_json.unlink()
            except Exception:
                pass

    return result


def detect_modality(
    nifti_path: Path,
    output_dir: Optional[Path] = None,
    conda_activate: Optional[str] = None,
) -> Dict[str, Any]:
    """Detect image modality (CT/MR) using TotalSegmentator's totalseg_get_modality.

    Args:
        nifti_path: Path to input NIfTI file
        output_dir: Optional directory for output JSON (uses temp if not provided)
        conda_activate: Optional conda activation command

    Returns:
        Dict with modality classification results:
        - modality: str (CT, MR, or unknown)
        - confidence: Optional[float]
        - status: success/error/unavailable
    """
    import tempfile

    result = {
        "modality": "unknown",
        "confidence": None,
        "status": "unavailable",
        "tool": "totalseg_get_modality",
    }

    if not nifti_path.exists():
        result["status"] = "error"
        result["error"] = "Input NIfTI file not found"
        return result

    # Determine output path
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_json = output_dir / "modality.json"
    else:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_json = Path(f.name)

    try:
        success, modality_data = _run_totalseg_tool(
            "totalseg_get_modality",
            nifti_path,
            output_json,
            conda_activate=conda_activate,
        )

        if success and modality_data:
            result["status"] = "success"
            # Parse TotalSegmentator modality output format
            # Expected: {"modality": "CT", "confidence": 0.98}
            result["modality"] = modality_data.get("modality", "unknown")
            result["confidence"] = modality_data.get("confidence")
            result["raw"] = modality_data
        else:
            result["status"] = "unavailable"

    finally:
        if not output_dir and output_json.exists():
            try:
                output_json.unlink()
            except Exception:
                pass

    return result


def _find_totalseg_masks(totalseg_dir: Path) -> Dict[str, Path]:
    """Find all TotalSegmentator mask files and map structure names to paths.

    Handles the naming convention: {task}--{structure_name}.nii.gz
    Excludes cropped versions.
    """
    masks = {}
    if not totalseg_dir.exists():
        return masks

    for task_dir in totalseg_dir.iterdir():
        if not task_dir.is_dir():
            continue
        for nii_file in task_dir.glob("*.nii.gz"):
            filename = nii_file.name
            # Skip cropped versions
            if "_cropped" in filename:
                continue
            # Parse {task}--{structure}.nii.gz
            if "--" in filename:
                structure_name = filename.split("--", 1)[1].replace(".nii.gz", "")
                # Prefer 'total' task for common structures
                if structure_name not in masks or filename.startswith("total--"):
                    masks[structure_name] = nii_file

    return masks


def detect_body_regions(
    course_dir: Path,
    model_region_requirements: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Any]:
    """Detect which body regions are present in a CT scan using TotalSegmentator outputs.

    Uses vertebrae as primary anchors (most deterministic) with organ volumes as
    secondary evidence. Returns additive boolean flags (whole-body CT = all True).

    Args:
        course_dir: Path to course directory containing Segmentation_TotalSegmentator
        model_region_requirements: Dict mapping model names to their region requirements.
            Each entry should have "required_regions" (list) and optionally "min_confidence" (float).
            If None, uses _DEFAULT_MODEL_REGION_REQUIREMENTS.

    Returns:
        Dict with body_regions, evidence, model_eligibility, and warnings
    """
    if model_region_requirements is None:
        model_region_requirements = _DEFAULT_MODEL_REGION_REQUIREMENTS
    result = {
        "qc_version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "course_path": str(course_dir),
        "body_regions": {
            "CONTAINS_HEAD_NECK": False,
            "CONTAINS_THORAX": False,
            "CONTAINS_ABDOMEN": False,
            "CONTAINS_PELVIS": False,
        },
        "confidence": {
            "HEAD_NECK": 0.0,
            "THORAX": 0.0,
            "ABDOMEN": 0.0,
            "PELVIS": 0.0,
        },
        "evidence": {
            "HEAD_NECK": {"vertebrae_found": [], "organ_volumes_ml": {}},
            "THORAX": {"vertebrae_found": [], "organ_volumes_ml": {}},
            "ABDOMEN": {"vertebrae_found": [], "organ_volumes_ml": {}},
            "PELVIS": {"vertebrae_found": [], "organ_volumes_ml": {}},
        },
        "model_eligibility": {},
        "warnings": [],
        "status": "PASS",
    }

    dirs = build_course_dirs(course_dir)
    totalseg_dir = dirs.segmentation_totalseg

    if not totalseg_dir.exists():
        result["status"] = "SKIP"
        result["warnings"].append("TotalSegmentator output directory not found")
        return result

    try:
        masks = _find_totalseg_masks(totalseg_dir)
    except Exception as e:
        result["status"] = "ERROR"
        result["warnings"].append(f"Failed to scan TotalSegmentator masks: {e}")
        return result

    if not masks:
        result["status"] = "SKIP"
        result["warnings"].append("No TotalSegmentator masks found")
        return result

    # Build set of relevant structures from BODY_REGION_ANCHORS (optimization)
    # Only compute volumes for structures we actually use as anchors
    relevant_structures: set[str] = set()
    for region_cfg in BODY_REGION_ANCHORS.values():
        relevant_structures.update(region_cfg["vertebrae"])
        relevant_structures.update(region_cfg["organs"])

    # Compute volumes only for relevant anchor structures
    volumes_ml: Dict[str, float] = {}
    for structure_name, nii_path in masks.items():
        if structure_name not in relevant_structures:
            continue  # Skip non-anchor structures (ribs, etc.)
        try:
            vol_ml, _ = _compute_mask_volume_ml(nii_path)
            if vol_ml > 0:
                volumes_ml[structure_name] = vol_ml
        except Exception as e:
            logger.debug(f"Failed to compute volume for {structure_name}: {e}")

    # Detect each body region
    for region_name, config in BODY_REGION_ANCHORS.items():
        region_key = f"CONTAINS_{region_name}"
        evidence = result["evidence"][region_name]

        # Check vertebrae (primary anchors)
        vertebrae_found = []
        for vert in config["vertebrae"]:
            if vert in volumes_ml and volumes_ml[vert] > 1.0:  # > 1mL to filter noise
                vertebrae_found.append(vert)
        evidence["vertebrae_found"] = vertebrae_found
        vertebrae_present = len(vertebrae_found) >= config["min_vertebrae"]

        # Check organs (secondary anchors)
        organ_present = False
        organ_volumes = {}

        for organ, threshold in config["organ_thresholds_ml"].items():
            if organ == "lungs_combined":
                # Sum all lung lobes
                lung_vol = sum(volumes_ml.get(lobe, 0) for lobe in [
                    "lung_upper_lobe_left", "lung_lower_lobe_left",
                    "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"
                ])
                organ_volumes["lungs_combined"] = lung_vol
                if lung_vol >= threshold:
                    organ_present = True
            elif organ == "kidney_combined":
                kidney_vol = volumes_ml.get("kidney_left", 0) + volumes_ml.get("kidney_right", 0)
                organ_volumes["kidney_combined"] = kidney_vol
                if kidney_vol >= threshold:
                    organ_present = True
            elif organ == "hip_combined":
                hip_vol = volumes_ml.get("hip_left", 0) + volumes_ml.get("hip_right", 0)
                organ_volumes["hip_combined"] = hip_vol
                if hip_vol >= threshold:
                    organ_present = True
            else:
                if organ in volumes_ml:
                    organ_volumes[organ] = volumes_ml[organ]
                    if volumes_ml[organ] >= threshold:
                        organ_present = True

        evidence["organ_volumes_ml"] = organ_volumes

        # Region is present if vertebrae OR organs pass threshold
        region_detected = vertebrae_present or organ_present
        result["body_regions"][region_key] = region_detected

        # Calculate confidence score (0-1)
        # Weight vertebrae more heavily (0.7) than organs (0.3)
        vert_score = min(len(vertebrae_found) / max(config["min_vertebrae"], 1), 1.0) * 0.7
        organ_score = 0.3 if organ_present else 0.0
        confidence = min(vert_score + organ_score, 1.0)
        result["confidence"][region_name] = round(confidence, 2)

    # Determine model eligibility
    for model_name, requirements in model_region_requirements.items():
        required_regions = requirements.get("required_regions", [])
        min_confidence = requirements.get("min_confidence", 0.5)

        if not required_regions:
            # No requirements - always eligible
            result["model_eligibility"][model_name] = {
                "eligible": True,
                "reason": "No region requirements"
            }
        else:
            # Check all required regions
            eligible = True
            missing_regions = []
            low_confidence_regions = []

            for region in required_regions:
                region_key = f"CONTAINS_{region}"
                if not result["body_regions"].get(region_key, False):
                    eligible = False
                    missing_regions.append(region)
                elif result["confidence"].get(region, 0) < min_confidence:
                    low_confidence_regions.append(region)

            if missing_regions:
                result["model_eligibility"][model_name] = {
                    "eligible": False,
                    "reason": f"Missing regions: {', '.join(missing_regions)}",
                    "block": True
                }
            elif low_confidence_regions:
                result["model_eligibility"][model_name] = {
                    "eligible": True,
                    "reason": f"Low confidence for: {', '.join(low_confidence_regions)}",
                    "warning": True
                }
            else:
                result["model_eligibility"][model_name] = {
                    "eligible": True,
                    "reason": f"All required regions present with sufficient confidence"
                }

    # Add warnings for edge cases
    regions_present = [k.replace("CONTAINS_", "") for k, v in result["body_regions"].items() if v]
    if len(regions_present) == 0:
        result["warnings"].append("No body regions detected - check TotalSegmentator output")
        result["status"] = "WARNING"
    elif len(regions_present) == 4:
        result["warnings"].append("Whole-body CT detected - all regions present")

    # Add warnings for any ineligible models (dynamically)
    for model_name, elig_info in result["model_eligibility"].items():
        if not elig_info.get("eligible", True):
            result["warnings"].append(
                f"Model '{model_name}' NOT eligible: {elig_info.get('reason', 'unknown')}"
            )
            result["status"] = "WARNING"

    return result


def _find_ct_nifti(course_dir: Path) -> Optional[Path]:
    """Find the CT NIfTI file for a course.

    Looks in standard locations (in priority order):
    1. NIFTI directory (primary location for dcm2niix output)
    2. Segmentation_TotalSegmentator subdirectories
    """
    dirs = build_course_dirs(course_dir)

    # Primary location: NIFTI directory (where dcm2niix outputs CT)
    nifti_dir = dirs.nifti
    if nifti_dir.exists():
        # Look for any .nii.gz file that's not a mask/segmentation
        candidates = []
        for nii in sorted(nifti_dir.glob("*.nii.gz")):
            # Skip obvious segmentation outputs
            if any(skip in nii.name.lower() for skip in ["mask", "seg", "label", "cropped"]):
                continue
            candidates.append(nii)

        if candidates:
            if len(candidates) > 1:
                logger.warning(
                    "Multiple CT NIfTI candidates found in %s: %s. Using first: %s",
                    nifti_dir, [c.name for c in candidates], candidates[0].name
                )
            return candidates[0]

    # Fallback: Segmentation_TotalSegmentator directory
    totalseg_dir = dirs.segmentation_totalseg
    if totalseg_dir.exists():
        # Check common locations
        candidates = [
            totalseg_dir / "ct.nii.gz",
            totalseg_dir / "total" / "ct.nii.gz",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Fall back to searching for ct*.nii.gz
        for nii in totalseg_dir.glob("**/ct*.nii.gz"):
            if "cropped" not in nii.name.lower():
                return nii

    return None


def save_body_region_qc(
    course_dir: Path,
    output_filename: str = "body_regions.json",
    model_region_requirements: Optional[Dict[str, Dict]] = None,
    nifti_path: Optional[Path] = None,
    conda_activate: Optional[str] = None,
) -> Path:
    """Run body region detection and save results to QC directory.

    Also runs contrast phase and modality detection if the CT NIfTI is available.

    Args:
        course_dir: Path to course directory
        output_filename: Name of output JSON file
        model_region_requirements: Dict mapping model names to their region requirements.
            If None, uses _DEFAULT_MODEL_REGION_REQUIREMENTS.
        nifti_path: Path to CT NIfTI file. If None, auto-detects from course directory.
        conda_activate: Optional conda activation command for TotalSegmentator tools.

    Returns:
        Path to saved QC file
    """
    dirs = build_course_dirs(course_dir)
    qc_dir = dirs.qc_reports
    qc_dir.mkdir(parents=True, exist_ok=True)

    results = detect_body_regions(course_dir, model_region_requirements)

    # Find CT NIfTI for phase/modality detection
    if nifti_path is None:
        nifti_path = _find_ct_nifti(course_dir)

    if nifti_path and nifti_path.exists():
        # Run contrast phase detection
        try:
            phase_result = detect_contrast_phase(
                nifti_path,
                output_dir=qc_dir,
                conda_activate=conda_activate,
            )
            results["contrast_phase"] = phase_result
        except Exception as e:
            logger.warning(f"Contrast phase detection failed: {e}")
            results["contrast_phase"] = {
                "phase": "unknown",
                "status": "error",
                "error": str(e),
            }

        # Run modality detection
        try:
            modality_result = detect_modality(
                nifti_path,
                output_dir=qc_dir,
                conda_activate=conda_activate,
            )
            results["image_modality"] = modality_result
        except Exception as e:
            logger.warning(f"Modality detection failed: {e}")
            results["image_modality"] = {
                "modality": "unknown",
                "status": "error",
                "error": str(e),
            }
    else:
        results["contrast_phase"] = {"phase": "unknown", "status": "unavailable", "reason": "CT NIfTI not found"}
        results["image_modality"] = {"modality": "unknown", "status": "unavailable", "reason": "CT NIfTI not found"}

    output_path = qc_dir / output_filename
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Body region QC saved to {output_path}")
    return output_path


def check_model_eligibility(
    course_dir: Path,
    model_name: str,
    model_region_requirements: Optional[Dict[str, Dict]] = None,
) -> Tuple[bool, str]:
    """Check if a specific model is eligible to run on this course.

    Loads existing body_regions.json if available, otherwise runs detection.

    Args:
        course_dir: Path to course directory
        model_name: Name of the model to check (e.g., "cardiac_STOPSTORM")
        model_region_requirements: Dict mapping model names to their region requirements.
            If None, uses _DEFAULT_MODEL_REGION_REQUIREMENTS.

    Returns:
        Tuple of (eligible: bool, reason: str)
    """
    if model_region_requirements is None:
        model_region_requirements = _DEFAULT_MODEL_REGION_REQUIREMENTS

    dirs = build_course_dirs(course_dir)
    qc_file = dirs.qc_reports / "body_regions.json"

    if qc_file.exists():
        try:
            with open(qc_file) as f:
                results = json.load(f)
        except Exception:
            results = detect_body_regions(course_dir, model_region_requirements)
    else:
        results = detect_body_regions(course_dir, model_region_requirements)

    # Check if model is in the provided requirements
    if model_name in model_region_requirements:
        requirements = model_region_requirements[model_name]
        required_regions = requirements.get("required_regions", [])
        min_confidence = requirements.get("min_confidence", 0.5)

        if not required_regions:
            return True, "No region requirements for this model"

        # Check each required region
        missing_regions = []
        low_confidence_regions = []

        for region in required_regions:
            region_key = f"CONTAINS_{region}"
            if not results.get("body_regions", {}).get(region_key, False):
                missing_regions.append(region)
            elif results.get("confidence", {}).get(region, 0) < min_confidence:
                low_confidence_regions.append(region)

        if missing_regions:
            return False, f"Missing regions: {', '.join(missing_regions)}"
        elif low_confidence_regions:
            return True, f"Low confidence for: {', '.join(low_confidence_regions)} (warning)"
        else:
            return True, "All required regions present with sufficient confidence"

    # Model not in requirements - check if it's in the cached results
    model_elig = results.get("model_eligibility", {}).get(model_name, {})
    if model_elig:
        return model_elig.get("eligible", True), model_elig.get("reason", "Unknown")

    # Model not in any requirements list - allow by default
    return True, "No specific requirements defined for this model"
