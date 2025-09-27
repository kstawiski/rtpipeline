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
        # Check if all source structures are available
        source_masks = []
        for source_name in config.source_structures:
            if source_name not in available_masks:
                logger.warning(f"Source structure '{source_name}' not found for custom structure '{config.name}'")
                return None
            source_masks.append(available_masks[source_name])

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

        # Apply margin if specified
        if config.margin:
            result = self.apply_margin(result, config.margin)

        logger.info(f"Created custom structure '{config.name}' using {config.operation} of {config.source_structures}")

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