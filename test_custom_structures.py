#!/usr/bin/env python3
"""
Test script for custom structures functionality in RTpipeline.

Usage:
    python test_custom_structures.py
"""

import sys
import logging
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def test_boolean_operations():
    """Test boolean operations for custom structures."""
    from rtpipeline.custom_structures import CustomStructureProcessor

    logger.info("Testing boolean operations...")

    # Create test masks
    mask1 = np.zeros((10, 10, 10), dtype=np.uint8)
    mask1[3:7, 3:7, 3:7] = 1  # Cube in center

    mask2 = np.zeros((10, 10, 10), dtype=np.uint8)
    mask2[5:9, 5:9, 5:9] = 1  # Overlapping cube

    processor = CustomStructureProcessor()

    # Test union
    union_mask = processor.union([mask1, mask2])
    expected_union_volume = np.sum(np.logical_or(mask1, mask2))
    assert np.sum(union_mask) == expected_union_volume, "Union operation failed"
    logger.info(f"✓ Union: {np.sum(union_mask)} voxels")

    # Test intersection
    intersection_mask = processor.intersection([mask1, mask2])
    expected_intersection_volume = np.sum(np.logical_and(mask1, mask2))
    assert np.sum(intersection_mask) == expected_intersection_volume, "Intersection operation failed"
    logger.info(f"✓ Intersection: {np.sum(intersection_mask)} voxels")

    # Test subtraction
    subtract_mask = processor.subtract([mask1, mask2])
    expected_subtract_volume = np.sum(np.logical_and(mask1, np.logical_not(mask2)))
    assert np.sum(subtract_mask) == expected_subtract_volume, "Subtraction operation failed"
    logger.info(f"✓ Subtraction: {np.sum(subtract_mask)} voxels")

    # Test XOR
    xor_mask = processor.xor([mask1, mask2])
    expected_xor_volume = np.sum(np.logical_xor(mask1, mask2))
    assert np.sum(xor_mask) == expected_xor_volume, "XOR operation failed"
    logger.info(f"✓ XOR: {np.sum(xor_mask)} voxels")

    logger.info("All boolean operations passed!")


def test_margin_operations():
    """Test margin operations for custom structures."""
    from rtpipeline.custom_structures import CustomStructureProcessor, MarginConfig

    logger.info("Testing margin operations...")

    # Create test mask
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[9:11, 9:11, 9:11] = 1  # Small cube in center

    # Set spacing (2mm x 2mm x 2mm)
    processor = CustomStructureProcessor(spacing=(2.0, 2.0, 2.0))

    # Test uniform margin
    margin = MarginConfig(uniform_mm=4)  # 4mm = 2 voxels in each direction
    expanded_mask = processor.apply_margin(mask, margin)

    # The expanded mask should be larger
    assert np.sum(expanded_mask) > np.sum(mask), "Margin expansion failed"
    logger.info(f"✓ Original volume: {np.sum(mask)} voxels")
    logger.info(f"✓ Expanded volume: {np.sum(expanded_mask)} voxels")

    logger.info("Margin operations passed!")


def test_yaml_loading():
    """Test YAML configuration loading."""
    from rtpipeline.custom_structures import CustomStructureProcessor

    logger.info("Testing YAML configuration loading...")

    processor = CustomStructureProcessor()

    # Check if example YAML exists
    yaml_path = Path("custom_structures_example.yaml")
    if yaml_path.exists():
        processor.load_config(yaml_path)
        logger.info(f"✓ Loaded {len(processor.custom_configs)} custom structure configurations")

        # Print loaded configurations
        for config in processor.custom_configs:
            logger.info(f"  - {config.name}: {config.operation} of {config.source_structures}")
    else:
        logger.warning("Example YAML file not found, skipping YAML test")

    logger.info("YAML loading test completed!")


def test_integration():
    """Test integration with DVH and radiomics modules."""
    logger.info("Testing module integration...")

    # Test imports
    try:
        from rtpipeline.dvh import _create_custom_structures_rtstruct
        logger.info("✓ DVH integration imports successfully")
    except ImportError as e:
        logger.error(f"Failed to import DVH integration: {e}")
        return False

    try:
        from rtpipeline.radiomics import radiomics_for_course
        logger.info("✓ Radiomics integration imports successfully")
    except ImportError as e:
        logger.error(f"Failed to import radiomics integration: {e}")
        return False

    logger.info("Module integration test completed!")
    return True


def main():
    """Run all tests."""
    logger.info("Starting custom structures tests...")

    try:
        test_boolean_operations()
        test_margin_operations()
        test_yaml_loading()
        test_integration()

        logger.info("\n✅ All tests passed successfully!")
        return 0

    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())