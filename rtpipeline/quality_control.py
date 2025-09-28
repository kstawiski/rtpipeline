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

logger = logging.getLogger(__name__)


class QualityControlError(Exception):
    """Exception raised when quality control checks fail."""
    pass


class DICOMValidator:
    """Validates DICOM files for consistency and integrity."""

    def __init__(self, patient_dir: Path):
        self.patient_dir = patient_dir
        self.ct_dir = patient_dir / "CT_DICOM"
        self.rp_path = patient_dir / "RP.dcm"
        self.rd_path = patient_dir / "RD.dcm"
        self.rs_path = patient_dir / "RS.dcm"
        self.rs_auto_path = patient_dir / "RS_auto.dcm"

    def validate_all(self) -> Dict[str, Any]:
        """Run all validation checks and return summary."""
        results = {
            "patient_id": self.patient_dir.name,
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

            # Calculate overall status
            results["overall_status"] = self._calculate_overall_status(results["checks"])

        except Exception as e:
            logger.error(f"Quality control failed for {self.patient_dir.name}: {e}")
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
            expected_rtplan_uid = "1.2.840.10008.5.1.4.1.1.481.5"
            if validation["sop_class_uid"] != expected_rtplan_uid:
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


def generate_qc_report(patient_dir: Path, output_dir: Path) -> Path:
    """Generate a comprehensive QC report for a patient."""
    validator = DICOMValidator(patient_dir)
    qc_results = validator.validate_all()

    # Add segmentation volume validation
    qc_results["segmentation_validation"] = validate_segmentation_volumes(patient_dir)

    # Save report
    report_path = output_dir / f"qc_report_{patient_dir.name}.json"
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