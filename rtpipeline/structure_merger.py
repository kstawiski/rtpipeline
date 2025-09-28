"""Structure merger for combining manual, automated, and custom DICOM-RT structures."""

from __future__ import annotations
import logging
import json
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import pydicom
import numpy as np
from datetime import datetime
from scipy import ndimage
from skimage import morphology

from .dvh import _create_custom_structures_rtstruct

logger = logging.getLogger(__name__)


class StructureMerger:
    """Merges multiple DICOM-RT structure sources with priority rules and naming conventions."""

    def __init__(self, patient_dir: Path):
        self.patient_dir = patient_dir
        self.rs_manual_path = patient_dir / "RS.dcm"
        self.rs_auto_path = patient_dir / "RS_auto.dcm"
        self.rs_custom_path = patient_dir / "RS_custom.dcm"

        # Priority rules from clinical recommendations
        self.priority_rules = {
            # Target volumes - manual takes priority
            "ptv": "manual",
            "ctv": "manual",
            "gtv": "manual",
            "target": "manual",
            "tumor": "manual",
            "planning": "manual",

            # Organs at risk - automated takes priority
            "bladder": "auto",
            "rectum": "auto",
            "femur": "auto",
            "bowel": "auto",
            "kidney": "auto",
            "liver": "auto",
            "spinal": "auto",
            "cord": "auto",
            "brain": "auto",
            "heart": "auto",
            "lung": "auto",
            "prostate": "auto",
            "uterus": "auto",
            "ovary": "auto",
        }

    def load_structures_from_dicom(self, dicom_path: Path, suffix: str) -> Dict[str, Dict]:
        """Load structures from DICOM file and add suffix to names."""
        if not dicom_path.exists():
            logger.info(f"DICOM file not found: {dicom_path}")
            return {}

        try:
            ds = pydicom.dcmread(dicom_path)
            structures = {}

            if hasattr(ds, 'StructureSetROISequence'):
                for roi_seq in ds.StructureSetROISequence:
                    roi_name = str(roi_seq.ROIName).strip()
                    roi_number = int(roi_seq.ROINumber)

                    # Standardize naming (capitalize first letter, handle spaces)
                    standardized_name = self._standardize_name(roi_name)
                    final_name = f"{standardized_name}_{suffix}"

                    structures[final_name] = {
                        "original_name": roi_name,
                        "roi_number": roi_number,
                        "standardized_name": standardized_name,
                        "suffix": suffix,
                        "source_file": str(dicom_path)
                    }

            logger.info(f"Loaded {len(structures)} structures from {dicom_path} with suffix '{suffix}'")
            return structures

        except Exception as e:
            logger.error(f"Error loading structures from {dicom_path}: {e}")
            return {}

    def _standardize_name(self, name: str) -> str:
        """Standardize structure names for consistent processing."""
        # Remove extra spaces and standardize capitalization
        standardized = " ".join(name.split()).title()

        # Handle common variations
        replacements = {
            "Bladder": "Bladder",
            "bladder": "Bladder",
            "BLADDER": "Bladder",
            "Rectum": "Rectum",
            "rectum": "Rectum",
            "RECTUM": "Rectum",
            "Femur": "Femur",
            "femur": "Femur",
            "FEMUR": "Femur",
            "Femoral_Head": "Femur_Head",
            "femoral_head": "Femur_Head",
            "FemoralHead": "Femur_Head",
        }

        return replacements.get(standardized, standardized)

    def determine_priority(self, structure_name: str) -> str:
        """Determine priority (manual/auto) for a given structure."""
        name_lower = structure_name.lower()

        for keyword, priority in self.priority_rules.items():
            if keyword in name_lower:
                return priority

        # Default to manual for unknown structures
        return "manual"

    def resolve_conflicts(self, all_structures: Dict[str, Dict]) -> Dict[str, Dict]:
        """Resolve naming conflicts using priority rules."""
        resolved_structures = {}
        conflict_groups = {}

        # Group structures by base name (without suffix)
        for full_name, info in all_structures.items():
            base_name = info["standardized_name"]
            if base_name not in conflict_groups:
                conflict_groups[base_name] = []
            conflict_groups[base_name].append((full_name, info))

        # Resolve conflicts for each group
        for base_name, candidates in conflict_groups.items():
            if len(candidates) == 1:
                # No conflict - keep as is
                full_name, info = candidates[0]
                resolved_structures[full_name] = info
            else:
                # Conflict - apply priority rules
                priority = self.determine_priority(base_name)
                logger.info(f"Resolving conflict for '{base_name}' - priority: {priority}")

                selected = None
                alternatives = []

                for full_name, info in candidates:
                    if info["suffix"] == priority:
                        selected = (full_name, info)
                    else:
                        alternatives.append((full_name, info))

                if selected:
                    full_name, info = selected
                    # Remove suffix for final name since conflict is resolved
                    final_name = base_name
                    info["final_name"] = final_name
                    info["alternatives"] = [alt[0] for alt in alternatives]
                    resolved_structures[final_name] = info
                    logger.info(f"Selected '{full_name}' as '{final_name}' (discarded: {[alt[0] for alt in alternatives]})")
                else:
                    # No preferred suffix found, keep first one
                    full_name, info = candidates[0]
                    final_name = base_name
                    info["final_name"] = final_name
                    info["alternatives"] = [alt[0] for alt in candidates[1:]]
                    resolved_structures[final_name] = info
                    logger.warning(f"No {priority} version found for '{base_name}', using '{full_name}'")

        return resolved_structures

    def create_merged_rtstruct(self, structures: Dict[str, Dict], custom_config_path: Optional[Path] = None) -> Path:
        """Create a new DICOM-RT structure file with merged structures."""

        # Start with manual structures as base if available
        if self.rs_manual_path.exists():
            base_ds = pydicom.dcmread(self.rs_manual_path)
            logger.info(f"Using manual structures as base: {self.rs_manual_path}")
        elif self.rs_auto_path.exists():
            base_ds = pydicom.dcmread(self.rs_auto_path)
            logger.info(f"Using auto structures as base: {self.rs_auto_path}")
        else:
            raise RuntimeError("No base structure file found (RS.dcm or RS_auto.dcm)")

        # Create new structure set
        new_ds = pydicom.dcmread(self.rs_auto_path)  # Use auto as template for now

        # Update metadata
        new_ds.StructureSetLabel = "Merged_Structures"
        new_ds.StructureSetName = "Combined Manual + Auto + Custom Structures"
        new_ds.StructureSetDescription = f"Generated by rtpipeline on {datetime.now().isoformat()}"

        # For now, just copy the auto structures and add metadata about merging
        # Full DICOM structure merging requires complex geometric operations
        # This is a placeholder for the structure merging logic

        # Save the merged file
        output_path = self.rs_custom_path
        new_ds.save_as(output_path)

        # Save structure mapping information
        mapping_path = self.patient_dir / "structure_mapping.json"
        mapping_info = {
            "timestamp": datetime.now().isoformat(),
            "structures": structures,
            "priority_rules": self.priority_rules,
            "output_file": str(output_path)
        }

        with open(mapping_path, 'w') as f:
            json.dump(mapping_info, f, indent=2, default=str)

        logger.info(f"Merged structure file saved to: {output_path}")
        logger.info(f"Structure mapping saved to: {mapping_path}")

        return output_path

    def generate_comparison_report(self, structures: Dict[str, Dict]) -> Path:
        """Generate a comparison report for structure volumes and overlaps."""
        report_data = {
            "patient_id": self.patient_dir.name,
            "timestamp": datetime.now().isoformat(),
            "structure_comparison": [],
            "summary": {
                "total_structures": len(structures),
                "manual_count": sum(1 for s in structures.values() if s["suffix"] == "manual"),
                "auto_count": sum(1 for s in structures.values() if s["suffix"] == "auto"),
                "conflicts_resolved": sum(1 for s in structures.values() if "alternatives" in s)
            }
        }

        for name, info in structures.items():
            structure_info = {
                "final_name": name,
                "original_name": info["original_name"],
                "source": info["suffix"],
                "source_file": info["source_file"],
                "priority_applied": self.determine_priority(name),
                "alternatives": info.get("alternatives", [])
            }
            report_data["structure_comparison"].append(structure_info)

        report_path = self.patient_dir / "structure_comparison_report.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Structure comparison report saved to: {report_path}")
        return report_path

    def load_custom_structure_config(self, config_path: Path) -> Dict[str, Any]:
        """Load custom structure configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('custom_structures', [])
        except Exception as e:
            logger.error(f"Failed to load custom structure config from {config_path}: {e}")
            return []

    def get_structure_mask(self, structure_name: str, rtstruct_ds: pydicom.Dataset) -> Optional[np.ndarray]:
        """Extract binary mask for a structure from RT Structure dataset."""
        # Find structure by name (case-insensitive partial match)
        target_seq = None
        structure_name_lower = structure_name.lower()

        for roi_seq in rtstruct_ds.StructureSetROISequence:
            roi_name = roi_seq.ROIName.lower()
            # Check for exact match or partial match with common variations
            if (roi_name == structure_name_lower or
                structure_name_lower in roi_name or
                roi_name.replace('_', '').replace(' ', '') == structure_name_lower.replace('_', '').replace(' ', '')):
                target_seq = roi_seq
                break

        if target_seq is None:
            logger.warning(f"Structure '{structure_name}' not found in RT Structure")
            return None

        # For this implementation, we'll return a placeholder
        # In practice, you'd need to extract the actual contour data and rasterize it
        # This would require the referenced CT series and proper coordinate transformation
        logger.info(f"Found structure '{structure_name}' (ROI Number: {target_seq.ROINumber})")
        return None  # Placeholder - actual mask extraction would go here

    def apply_margin_to_mask(self, mask: np.ndarray, margin_mm: float, pixel_spacing: List[float]) -> np.ndarray:
        """Apply margin expansion to a binary mask."""
        if margin_mm <= 0:
            return mask

        # Convert margin from mm to pixels
        margin_pixels = [margin_mm / spacing for spacing in pixel_spacing]

        # Create structuring element for dilation
        # Use spherical structuring element for 3D
        if len(mask.shape) == 3:
            # 3D case
            radius = max(margin_pixels)
            structuring_element = morphology.ball(int(np.ceil(radius)))
        else:
            # 2D case
            radius = max(margin_pixels[:2])
            structuring_element = morphology.disk(int(np.ceil(radius)))

        # Apply binary dilation
        expanded_mask = ndimage.binary_dilation(mask, structure=structuring_element)
        return expanded_mask.astype(mask.dtype)

    def perform_boolean_operation(self, operation: str, masks: List[np.ndarray]) -> Optional[np.ndarray]:
        """Perform boolean operation on list of masks."""
        if not masks:
            return None

        result = masks[0].copy()

        for mask in masks[1:]:
            if operation == "union":
                result = np.logical_or(result, mask)
            elif operation == "intersection":
                result = np.logical_and(result, mask)
            elif operation == "subtract":
                result = np.logical_and(result, np.logical_not(mask))
            else:
                logger.error(f"Unknown boolean operation: {operation}")
                return None

        return result.astype(np.uint8)

    def _resolve_dependencies(self, custom_config: List[Dict]) -> List[Dict]:
        """Resolve dependencies and return custom structures in correct creation order."""
        # Build dependency graph
        dependency_graph = {}
        struct_configs = {cfg['name']: cfg for cfg in custom_config}

        for cfg in custom_config:
            struct_name = cfg['name']
            dependencies = []

            # Check if any source structure is a custom structure
            for source in cfg.get('source_structures', []):
                # Check if source refers to another custom structure
                if source in struct_configs:
                    dependencies.append(source)

            dependency_graph[struct_name] = dependencies

        # Topological sort
        sorted_order = []
        visited = set()
        temp_mark = set()

        def visit(node):
            if node in temp_mark:
                logger.warning(f"Circular dependency detected involving {node}")
                return False
            if node not in visited:
                temp_mark.add(node)
                for dep in dependency_graph.get(node, []):
                    if not visit(dep):
                        return False
                temp_mark.remove(node)
                visited.add(node)
                sorted_order.append(node)
            return True

        # Visit all nodes
        for node in dependency_graph:
            if node not in visited:
                if not visit(node):
                    logger.error("Failed to resolve dependencies due to circular references")
                    return custom_config  # Return original order if circular dependency

        # Return configs in dependency order
        ordered_configs = []
        for name in sorted_order:
            if name in struct_configs:
                ordered_configs.append(struct_configs[name])

        logger.info(f"Resolved dependency order: {sorted_order}")
        return ordered_configs

    def create_custom_structures(self, custom_config: List[Dict], all_structures: Dict[str, Dict]) -> Dict[str, Dict]:
        """Create custom structures based on configuration."""
        custom_structures = {}

        # Resolve dependencies first
        ordered_config = self._resolve_dependencies(custom_config)

        # Load RT Structure datasets for mask operations
        try:
            manual_ds = pydicom.dcmread(self.rs_manual_path) if self.rs_manual_path.exists() else None
            auto_ds = pydicom.dcmread(self.rs_auto_path) if self.rs_auto_path.exists() else None
        except Exception as e:
            logger.error(f"Failed to load RT Structure files for custom structure creation: {e}")
            return custom_structures

        # Keep track of all available structures (initial + created)
        available_structures = dict(all_structures)

        for custom_struct in ordered_config:
            struct_name = custom_struct.get('name')
            operation = custom_struct.get('operation', 'union')
            source_structures = custom_struct.get('source_structures', [])
            margin = custom_struct.get('margin', 0)
            description = custom_struct.get('description', '')

            logger.info(f"Creating custom structure: {struct_name}")

            # For now, we'll create a placeholder structure entry
            # Full implementation would require proper mask operations
            source_masks = []
            missing_sources = []
            resolved_sources = []

            for source_name in source_structures:
                found = False

                # First check if it's a custom structure we already created
                custom_key = f"{source_name}_custom"
                if custom_key in available_structures:
                    found = True
                    resolved_sources.append(custom_key)
                    logger.info(f"Found custom source structure: {source_name} -> {custom_key}")
                else:
                    # Try to find source structure in existing structures
                    source_name_clean = source_name.lower().replace('_', '').replace(' ', '').replace('.nii', '')

                    for existing_name in available_structures.keys():
                        existing_clean = existing_name.lower().replace('_', '').replace(' ', '').replace('.nii', '')

                        # Check multiple matching patterns
                        if (source_name.lower() == existing_name.lower() or
                            source_name.lower() in existing_name.lower() or
                            existing_name.lower() in source_name.lower() or
                            source_name_clean == existing_clean or
                            source_name_clean in existing_clean or
                            existing_clean in source_name_clean):

                            found = True
                            resolved_sources.append(existing_name)
                            logger.info(f"Found source structure: {source_name} -> {existing_name}")
                            break

                if not found:
                    missing_sources.append(source_name)
                    logger.warning(f"Source structure not found: {source_name}")

            if missing_sources:
                logger.warning(f"Skipping custom structure '{struct_name}' due to missing sources: {missing_sources}")
                continue

            # Create custom structure entry
            custom_key = f"{struct_name}_custom"
            custom_structures[custom_key] = {
                "original_name": struct_name,
                "suffix": "custom",
                "source_file": "custom_config",
                "alternatives": [],
                "operation": operation,
                "sources": resolved_sources,  # Use resolved source names
                "margin": margin,
                "description": description
            }

            # Add to available structures for subsequent custom structures
            available_structures[custom_key] = custom_structures[custom_key]

            logger.info(f"Created custom structure: {custom_key} from sources: {resolved_sources}")

        return custom_structures

    def merge_all_structures(self, custom_config_path: Optional[Path] = None) -> Tuple[Path, Path]:
        """Combine manual, auto and custom structures into RS_custom and emit a report."""

        logger.info("Building merged RTSTRUCT for %s", self.patient_dir)

        try:
            merged_file = _create_custom_structures_rtstruct(
                self.patient_dir,
                custom_config_path,
                self.rs_manual_path if self.rs_manual_path.exists() else None,
                self.rs_auto_path if self.rs_auto_path.exists() else None,
            )
        except Exception as exc:
            logger.error("Custom structure generation failed: %s", exc)
            merged_file = None

        if not merged_file or not Path(merged_file).exists():
            logger.warning("Falling back to existing RTSTRUCT due to custom merge failure")
            fallback = None
            if self.rs_auto_path.exists():
                fallback = self.rs_auto_path
            elif self.rs_manual_path.exists():
                fallback = self.rs_manual_path
            if not fallback:
                raise RuntimeError("Neither manual nor auto RTSTRUCT available for fallback")
            shutil.copy2(fallback, self.rs_custom_path)
            merged_path = self.rs_custom_path
            fallback_used = True
        else:
            merged_path = Path(merged_file)
            fallback_used = False

        # Derive source labels for reporting
        def _roi_names(path: Path) -> Set[str]:
            if not path or not path.exists():
                return set()
            try:
                ds = pydicom.dcmread(str(path), stop_before_pixels=True)
                return {
                    str(getattr(roi, "ROIName", "")).strip()
                    for roi in getattr(ds, "StructureSetROISequence", [])
                }
            except Exception as exc:  # pragma: no cover - safety
                logger.debug("Failed reading ROI names from %s: %s", path, exc)
                return set()

        manual_names = _roi_names(self.rs_manual_path)
        auto_names = _roi_names(self.rs_auto_path)

        custom_names: Set[str] = set()
        if custom_config_path and Path(custom_config_path).exists():
            try:
                cfg = yaml.safe_load(Path(custom_config_path).read_text()) or {}
                for item in cfg.get("custom_structures", []):
                    name = item.get("name")
                    if name:
                        custom_names.add(str(name))
            except Exception as exc:
                logger.warning("Failed to parse custom structure config %s: %s", custom_config_path, exc)

        try:
            ds_final = pydicom.dcmread(str(merged_path), stop_before_pixels=True)
            final_rois = getattr(ds_final, "StructureSetROISequence", []) or []
        except Exception as exc:
            logger.error("Unable to read merged RTSTRUCT for reporting: %s", exc)
            final_rois = []

        comparison: List[Dict[str, Any]] = []
        for roi in final_rois:
            name = str(getattr(roi, "ROIName", "")).strip()
            sources = []
            if name in custom_names:
                sources.append("custom")
            if name in manual_names:
                sources.append("manual")
            if name in auto_names and "auto" not in sources:
                sources.append("auto")
            if not sources:
                sources.append("unknown")
            comparison.append({
                "final_name": name,
                "sources": sources,
            })

        report_data = {
            "patient_dir": str(self.patient_dir),
            "timestamp": datetime.now().isoformat(),
            "fallback_used": fallback_used,
            "structure_comparison": comparison,
        }

        report_path = self.patient_dir / "structure_comparison_report.json"
        report_path.write_text(json.dumps(report_data, indent=2))

        mapping_path = self.patient_dir / "structure_mapping.json"
        mapping_info = {
            "timestamp": datetime.now().isoformat(),
            "output_file": str(merged_path.resolve()),
            "sources": {
                "manual": sorted(manual_names),
                "auto": sorted(auto_names),
                "custom_config": sorted(custom_names),
            },
        }
        mapping_path.write_text(json.dumps(mapping_info, indent=2))

        return merged_path, report_path


def merge_patient_structures(patient_dir: Path, custom_config_path: Optional[Path] = None) -> Tuple[Path, Path]:
    """Convenience function to merge structures for a single patient."""
    merger = StructureMerger(patient_dir)
    return merger.merge_all_structures(custom_config_path)
