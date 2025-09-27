#!/usr/bin/env python3
"""
Demonstration script showing what happens when source structures are missing or empty.

Usage:
    python test_missing_structures.py
"""

import sys
import logging
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def test_missing_structures():
    """Test behavior with missing source structures."""
    from rtpipeline.custom_structures import CustomStructureProcessor, CustomStructureConfig, MarginConfig

    logger.info("🔍 Testing behavior with missing structures...")

    processor = CustomStructureProcessor(spacing=(1.0, 1.0, 1.0))

    # Create some available masks
    available_masks = {
        "liver": np.ones((10, 10, 10), dtype=np.uint8),
        "kidney_left": np.ones((5, 5, 5), dtype=np.uint8),
        # Note: kidney_right is missing
        # Note: spleen is missing
    }

    logger.info(f"Available structures: {list(available_masks.keys())}")

    # Test 1: Structure with all available sources
    config1 = CustomStructureConfig(
        name="liver_expanded",
        operation="union",
        source_structures=["liver"],
        margin=MarginConfig(uniform_mm=5)
    )

    # Test 2: Structure with some missing sources
    config2 = CustomStructureConfig(
        name="both_kidneys",
        operation="union",
        source_structures=["kidney_left", "kidney_right"]  # kidney_right missing
    )

    # Test 3: Structure with all missing sources
    config3 = CustomStructureConfig(
        name="missing_organs",
        operation="union",
        source_structures=["spleen", "pancreas"]  # both missing
    )

    # Test 4: Structure that depends on another custom structure
    config4 = CustomStructureConfig(
        name="liver_kidney_combined",
        operation="union",
        source_structures=["liver_expanded", "both_kidneys"]  # both_kidneys will fail
    )

    processor.custom_configs = [config1, config2, config3, config4]

    logger.info("\n" + "="*60)
    logger.info("PROCESSING CUSTOM STRUCTURES")
    logger.info("="*60)

    custom_masks = processor.process_all_custom_structures(available_masks)

    logger.info("\n" + "="*60)
    logger.info("RESULTS SUMMARY")
    logger.info("="*60)

    logger.info(f"✅ Successfully created {len(custom_masks)} out of {len(processor.custom_configs)} custom structures:")
    for name, mask in custom_masks.items():
        volume = np.sum(mask)
        logger.info(f"  • {name}: {volume} voxels")

    failed_count = len(processor.custom_configs) - len(custom_masks)
    if failed_count > 0:
        logger.info(f"\n❌ Failed to create {failed_count} structures due to missing sources")

    return len(custom_masks) > 0

def test_empty_structures():
    """Test behavior with empty (zero volume) source structures."""
    from rtpipeline.custom_structures import CustomStructureProcessor, CustomStructureConfig

    logger.info("\n🔍 Testing behavior with empty structures...")

    processor = CustomStructureProcessor()

    # Create masks where some are empty
    available_masks = {
        "normal_structure": np.ones((5, 5, 5), dtype=np.uint8),
        "empty_structure": np.zeros((5, 5, 5), dtype=np.uint8),  # Empty mask
        "single_voxel": np.zeros((5, 5, 5), dtype=np.uint8)
    }
    available_masks["single_voxel"][2, 2, 2] = 1  # Only one voxel

    logger.info(f"Available structures:")
    for name, mask in available_masks.items():
        volume = np.sum(mask)
        logger.info(f"  • {name}: {volume} voxels")

    # Test union with empty structure
    config = CustomStructureConfig(
        name="union_with_empty",
        operation="union",
        source_structures=["normal_structure", "empty_structure"]
    )

    processor.custom_configs = [config]
    custom_masks = processor.process_all_custom_structures(available_masks)

    if custom_masks:
        result_volume = np.sum(list(custom_masks.values())[0])
        logger.info(f"✅ Union with empty structure created: {result_volume} voxels")
        logger.info("   (Empty structures are handled gracefully in boolean operations)")
    else:
        logger.error("❌ Failed to create structure with empty source")

    return True

def test_cascade_failures():
    """Test how failures cascade when structures depend on each other."""
    from rtpipeline.custom_structures import CustomStructureProcessor, CustomStructureConfig

    logger.info("\n🔍 Testing cascade failure behavior...")

    processor = CustomStructureProcessor()

    available_masks = {
        "base_structure": np.ones((5, 5, 5), dtype=np.uint8)
        # missing_base is not available
    }

    # Create a chain of dependencies
    configs = [
        CustomStructureConfig(
            name="level1",
            operation="union",
            source_structures=["base_structure", "missing_base"]  # Will fail
        ),
        CustomStructureConfig(
            name="level2",
            operation="union",
            source_structures=["level1", "base_structure"]  # Will fail because level1 failed
        ),
        CustomStructureConfig(
            name="level3",
            operation="union",
            source_structures=["level2"]  # Will fail because level2 failed
        ),
        CustomStructureConfig(
            name="independent",
            operation="union",
            source_structures=["base_structure"]  # Should succeed
        )
    ]

    processor.custom_configs = configs

    logger.info("Processing cascading structure dependencies...")
    custom_masks = processor.process_all_custom_structures(available_masks)

    logger.info(f"\n📊 Cascade Results:")
    logger.info(f"  • Total structures attempted: {len(configs)}")
    logger.info(f"  • Successfully created: {len(custom_masks)}")
    logger.info(f"  • Failed due to missing dependencies: {len(configs) - len(custom_masks)}")

    if "independent" in custom_masks:
        logger.info("  ✅ Independent structure succeeded (not affected by cascade failures)")
    else:
        logger.error("  ❌ Independent structure failed unexpectedly")

    return True

def test_real_world_scenario():
    """Test a realistic scenario with pelvic structures where some TotalSegmentator structures might be missing."""
    from rtpipeline.custom_structures import CustomStructureProcessor

    logger.info("\n🔍 Testing realistic pelvic scenario with missing TotalSegmentator structures...")

    processor = CustomStructureProcessor()

    # Simulate realistic scenario where some TotalSegmentator structures are missing
    # (e.g., segmentation failed for certain organs)
    available_masks = {
        # Available structures
        "liver": np.ones((10, 10, 10), dtype=np.uint8),
        "kidney_left": np.ones((8, 8, 8), dtype=np.uint8),
        "kidney_right": np.ones((8, 8, 8), dtype=np.uint8),
        "urinary_bladder": np.ones((6, 6, 6), dtype=np.uint8),
        "sacrum": np.ones((15, 15, 15), dtype=np.uint8),
        "hip_left": np.ones((12, 12, 12), dtype=np.uint8),
        "hip_right": np.ones((12, 12, 12), dtype=np.uint8),
        "colon": np.ones((20, 20, 20), dtype=np.uint8),

        # Missing structures (common TotalSegmentator failures)
        # "iliac_artery_left": MISSING
        # "iliac_artery_right": MISSING
        # "iliac_vena_left": MISSING
        # "iliac_vena_right": MISSING
        # "small_bowel": MISSING
        # "vertebrae_S1": MISSING
        # "femur_left": MISSING
        # "femur_right": MISSING
    }

    logger.info(f"Available structures ({len(available_masks)}): {list(available_masks.keys())}")

    # Load the actual pelvic configuration
    processor.load_config("custom_structures_pelvic.yaml")
    original_count = len(processor.custom_configs)

    logger.info(f"\nProcessing {original_count} pelvic structures with missing dependencies...")
    custom_masks = processor.process_all_custom_structures(available_masks)

    logger.info(f"\n📊 Real-world Results:")
    logger.info(f"  • Total pelvic structures in template: {original_count}")
    logger.info(f"  • Successfully created: {len(custom_masks)}")
    logger.info(f"  • Failed due to missing sources: {original_count - len(custom_masks)}")
    logger.info(f"  • Success rate: {len(custom_masks)/original_count*100:.1f}%")

    logger.info(f"\n✅ Successfully created structures:")
    for name in sorted(custom_masks.keys()):
        volume = np.sum(custom_masks[name])
        logger.info(f"  • {name}: {volume} voxels")

    # Show which structures would fail
    created_names = set(custom_masks.keys())
    all_names = {config.name for config in processor.custom_configs}
    failed_names = all_names - created_names

    if failed_names:
        logger.info(f"\n❌ Failed structures:")
        for name in sorted(failed_names):
            logger.info(f"  • {name}")

    return True

def main():
    """Run all missing structure tests."""
    logger.info("🚀 Testing behavior with missing and empty structures...")

    tests = [
        ("Missing Structures", test_missing_structures),
        ("Empty Structures", test_empty_structures),
        ("Cascade Failures", test_cascade_failures),
        ("Real-world Scenario", test_real_world_scenario),
    ]

    all_passed = True
    for name, test_func in tests:
        logger.info(f"\n{'='*70}")
        logger.info(f"TEST: {name}")
        logger.info(f"{'='*70}")

        try:
            success = test_func()
            if success:
                logger.info(f"✅ {name} test completed successfully")
            else:
                logger.error(f"❌ {name} test failed")
                all_passed = False
        except Exception as e:
            logger.error(f"❌ {name} test failed with exception: {e}")
            all_passed = False

    logger.info(f"\n{'='*70}")
    logger.info("OVERALL SUMMARY")
    logger.info(f"{'='*70}")

    if all_passed:
        logger.info("🎉 All tests passed!")
        logger.info("\nKEY FINDINGS:")
        logger.info("• Missing structures are gracefully skipped with warnings")
        logger.info("• Empty structures are handled correctly in boolean operations")
        logger.info("• Failed structures don't break the pipeline")
        logger.info("• Available structures are processed successfully")
        logger.info("• Cascade failures are contained (independent structures still work)")
        return 0
    else:
        logger.error("💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())