from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
import SimpleITK as sitk

logger = logging.getLogger(__name__)


@dataclass
class MarginConfig:
    """Configuration for margin operations on structures."""
    anterior_mm: float = 0
    posterior_mm: float = 0
    left_mm: float = 0
    right_mm: float = 0
    superior_mm: float = 0
    inferior_mm: float = 0
    uniform_mm: Optional[float] = None

    def __post_init__(self):
        if self.uniform_mm is not None:
            self.anterior_mm = self.uniform_mm
            self.posterior_mm = self.uniform_mm
            self.left_mm = self.uniform_mm
            self.right_mm = self.uniform_mm
            self.superior_mm = self.uniform_mm
            self.inferior_mm = self.uniform_mm


@dataclass
class CustomStructureConfig:
    """Configuration for a custom structure created from boolean operations."""
    name: str
    operation: str  # union, intersection, subtract, xor
    source_structures: List[str]
    margin: Optional[MarginConfig] = None
    description: Optional[str] = None


class CustomStructureProcessor:
    """Process custom structures with boolean operations and margins."""

    def __init__(self, spacing: tuple[float, float, float] = None):
        """
        Initialize the custom structure processor.

        Args:
            spacing: Voxel spacing in mm (x, y, z). Required for margin operations.
        """
        self.spacing = spacing
        self.custom_configs: List[CustomStructureConfig] = []
        self.partial_structures: dict[str, list[str]] = {}

    def load_config(self, config_path: Union[str, Path]) -> None:
        """
        Load custom structure configurations from YAML file.

        Args:
            config_path: Path to YAML configuration file
        """
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Custom structure config not found: {config_path}")
            return

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if not config or 'custom_structures' not in config:
            logger.warning("No custom_structures section found in config")
            return

        for struct_config in config['custom_structures']:
            margin_config = None
            if 'margin' in struct_config:
                margin_data = struct_config['margin']
                if isinstance(margin_data, (int, float)):
                    margin_config = MarginConfig(uniform_mm=float(margin_data))
                else:
                    margin_config = MarginConfig(**margin_data)

            custom_struct = CustomStructureConfig(
                name=struct_config['name'],
                operation=struct_config.get('operation', 'union'),
                source_structures=struct_config['source_structures'],
                margin=margin_config,
                description=struct_config.get('description')
            )
            self.custom_configs.append(custom_struct)

        logger.info(f"Loaded {len(self.custom_configs)} custom structure configurations")

    def union(self, masks: List[np.ndarray]) -> np.ndarray:
        """Perform union (OR) operation on multiple masks."""
        if not masks:
            raise ValueError("No masks provided for union operation")

        result = masks[0].copy()
        for mask in masks[1:]:
            result = np.logical_or(result, mask)
        return result.astype(np.uint8)

    def intersection(self, masks: List[np.ndarray]) -> np.ndarray:
        """Perform intersection (AND) operation on multiple masks."""
        if not masks:
            raise ValueError("No masks provided for intersection operation")

        result = masks[0].copy()
        for mask in masks[1:]:
            result = np.logical_and(result, mask)
        return result.astype(np.uint8)

    def subtract(self, masks: List[np.ndarray]) -> np.ndarray:
        """Subtract masks[1:] from masks[0]."""
        if len(masks) < 2:
            raise ValueError("Subtract operation requires at least 2 masks")

        result = masks[0].copy()
        for mask in masks[1:]:
            result = np.logical_and(result, np.logical_not(mask))
        return result.astype(np.uint8)

    def xor(self, masks: List[np.ndarray]) -> np.ndarray:
        """Perform XOR operation on multiple masks."""
        if not masks:
            raise ValueError("No masks provided for XOR operation")

        result = masks[0].copy()
        for mask in masks[1:]:
            result = np.logical_xor(result, mask)
        return result.astype(np.uint8)

    def apply_margin(self, mask: np.ndarray, margin: MarginConfig) -> np.ndarray:
        """
        Apply margin (expansion/contraction) to a mask.

        Args:
            mask: Input binary mask
            margin: Margin configuration

        Returns:
            Modified mask with margin applied
        """
        if self.spacing is None:
            logger.warning("Spacing not set, cannot apply margin accurately")
            return mask

        # Convert margin from mm to voxels
        x_spacing, y_spacing, z_spacing = self.spacing

        # Create anisotropic structuring element based on margins
        margin_voxels = [
            int(np.ceil(max(margin.left_mm, margin.right_mm) / x_spacing)),
            int(np.ceil(max(margin.anterior_mm, margin.posterior_mm) / y_spacing)),
            int(np.ceil(max(margin.superior_mm, margin.inferior_mm) / z_spacing))
        ]

        if all(m == 0 for m in margin_voxels):
            return mask

        # Apply dilation for positive margins
        if any(m > 0 for m in margin_voxels):
            struct_elem = generate_binary_structure(3, 1)

            # Dilate based on maximum margin
            max_iter = max(margin_voxels)
            result = binary_dilation(mask, structure=struct_elem, iterations=max_iter)
        else:
            result = mask.copy()

        # Handle asymmetric margins if needed (more complex implementation)
        # For now, using uniform expansion based on maximum margin

        return result.astype(np.uint8)

    @staticmethod
    def _normalize_name(name: str) -> str:
        return "".join(ch.lower() for ch in name if ch.isalnum())

    def process_custom_structure(
        self,
        config: CustomStructureConfig,
        available_masks: Dict[str, np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Process a single custom structure configuration.

        Args:
            config: Custom structure configuration
            available_masks: Dictionary of available masks by name

        Returns:
            Processed mask or None if source structures not available
        """
        # Build lookup table with normalized keys for resilient matching
        normalization_map: Dict[str, str] = {
            self._normalize_name(key): key for key in available_masks.keys()
        }

        missing_sources: list[str] = []
        source_masks: list[np.ndarray] = []
        found_mappings: Dict[str, str] = {}  # requested name -> actual name

        for source_name in config.source_structures:
            actual_key = None

            # Try exact match first
            if source_name in available_masks:
                actual_key = source_name
            # Try normalized match
            elif self._normalize_name(source_name) in normalization_map:
                actual_key = normalization_map[self._normalize_name(source_name)]

            if actual_key is None or actual_key not in available_masks:
                missing_sources.append(source_name)
                # Log available alternatives that might be close
                normalized_requested = self._normalize_name(source_name)
                similar = [k for k in available_masks.keys()
                          if normalized_requested in self._normalize_name(k)
                          or self._normalize_name(k) in normalized_requested]
                if similar:
                    logger.debug(
                        "Custom structure '%s': source '%s' not found. Similar available: %s",
                        config.name, source_name, ", ".join(similar[:3])
                    )
                continue

            mask = available_masks[actual_key]
            if mask is None or not np.any(mask):
                missing_sources.append(source_name)
                logger.debug(
                    "Custom structure '%s': source '%s' exists but is empty",
                    config.name, source_name
                )
                continue

            source_masks.append(mask)
            found_mappings[source_name] = actual_key

        if not source_masks:
            logger.warning(
                "No usable source structures found for custom structure '%s' (requested: %s). Available structures: %s",
                config.name,
                ", ".join(config.source_structures),
                ", ".join(sorted(list(available_masks.keys())[:10]))  # Show first 10
            )
            self.partial_structures.pop(config.name, None)
            return None

        if missing_sources:
            logger.warning(
                "Custom structure '%s': missing %d/%d source structures: %s (found: %s)",
                config.name,
                len(missing_sources),
                len(config.source_structures),
                ", ".join(sorted({src for src in missing_sources})),
                ", ".join(f"{req}->{act}" for req, act in found_mappings.items())
            )
            self.partial_structures[config.name] = sorted({src for src in missing_sources})
        else:
            logger.info(
                "Custom structure '%s': successfully matched all %d source structures",
                config.name,
                len(source_masks)
            )
            self.partial_structures.pop(config.name, None)

        # Apply boolean operation
        if config.operation == 'union':
            result = self.union(source_masks)
        elif config.operation == 'intersection':
            result = self.intersection(source_masks)
        elif config.operation == 'subtract':
            result = self.subtract(source_masks)
        elif config.operation == 'xor':
            result = self.xor(source_masks)
        else:
            logger.error(f"Unknown operation: {config.operation}")
            return None

        # Validate structure size before margin
        voxel_count_pre = int(np.sum(result > 0))
        if voxel_count_pre == 0:
            logger.warning(
                "Custom structure '%s': %s operation resulted in empty structure. "
                "Check if source structures overlap correctly.",
                config.name, config.operation
            )
            return None
        elif voxel_count_pre < 10:
            logger.warning(
                "Custom structure '%s': only %d voxels after %s operation. "
                "This structure may be too small for meaningful analysis.",
                config.name, voxel_count_pre, config.operation
            )

        # Apply margin if specified
        if config.margin:
            result = self.apply_margin(result, config.margin)
            voxel_count_post = int(np.sum(result > 0))
            if voxel_count_post == 0:
                logger.warning(
                    "Custom structure '%s': margin operation resulted in empty structure",
                    config.name
                )
                return None
            logger.debug(
                "Custom structure '%s': margin changed voxel count from %d to %d",
                config.name, voxel_count_pre, voxel_count_post
            )

        # Calculate and log final structure size
        final_voxel_count = int(np.sum(result > 0))
        volume_ml = None
        if self.spacing:
            voxel_volume_mm3 = self.spacing[0] * self.spacing[1] * self.spacing[2]
            volume_ml = (final_voxel_count * voxel_volume_mm3) / 1000.0
            logger.info(
                "Created custom structure '%s': %s of %d sources, %d voxels (%.2f mL)",
                config.name, config.operation, len(source_masks),
                final_voxel_count, volume_ml
            )
        else:
            logger.info(
                "Created custom structure '%s': %s of %d sources, %d voxels",
                config.name, config.operation, len(source_masks), final_voxel_count
            )

        return result

    def process_all_custom_structures(
        self,
        available_masks: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Process all configured custom structures.

        Args:
            available_masks: Dictionary of available masks by name

        Returns:
            Dictionary of custom structure masks
        """
        custom_masks = {}

        for config in self.custom_configs:
            mask = self.process_custom_structure(config, available_masks)
            if mask is not None:
                custom_masks[config.name] = mask
                # Add to available masks for use in subsequent operations
                available_masks[config.name] = mask

        return custom_masks
