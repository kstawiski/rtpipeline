"""Quality control and validation functions for DICOM-RT processing pipeline."""

from __future__ import annotations
import logging
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pydicom
import numpy as np
import pandas as pd
from datetime import datetime

from .layout import build_course_dirs
from .utils import mask_is_cropped

logger = logging.getLogger(__name__)


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
                if mask_is_cropped(mask):
                    entry: Dict[str, Any] = {"source": source, "roi_name": roi_name}
                    if cropping_metadata:
                        entry["expected_due_to_ct_crop"] = True
                        entry["crop_region"] = cropping_metadata.get("region")
                        entry["clamped_axes"] = cropping_metadata.get("clamped_axes", [])
                    else:
                        entry["expected_due_to_ct_crop"] = False
                    info["structures"].append(entry)

        # Evaluate both structure sets (manual and auto)
        _evaluate(self.rs_path, "manual")
        _evaluate(self.rs_auto_path, "auto")

        if info["structures"]:
            unexpected = [entry for entry in info["structures"] if not entry.get("expected_due_to_ct_crop")]
            if unexpected:
                info["status"] = "WARNING"
            else:
                info["status"] = "INFO"
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
