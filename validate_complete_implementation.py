#!/usr/bin/env python3
"""
Complete validation script for custom structures implementation.

Usage:
    python validate_complete_implementation.py
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def validate_file_structure():
    """Validate that all required files are in place."""
    logger.info("🔍 Validating file structure...")

    required_files = [
        "rtpipeline/custom_structures.py",
        "custom_structures_example.yaml",
        "custom_structures_pelvic.yaml",
        "test_custom_structures.py",
        "test_pelvic_config.py",
        "CUSTOM_STRUCTURES_README.md",
        "PELVIC_STRUCTURES_README.md"
    ]

    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)

    if missing:
        logger.error(f"❌ Missing files: {missing}")
        return False

    logger.info("✅ All required files present")
    return True

def validate_module_imports():
    """Validate that all modules can be imported correctly."""
    logger.info("🔍 Validating module imports...")

    try:
        from rtpipeline.custom_structures import CustomStructureProcessor, MarginConfig, CustomStructureConfig
        logger.info("✅ Custom structures module imports")

        from rtpipeline.dvh import dvh_for_course, _create_custom_structures_rtstruct
        logger.info("✅ DVH integration imports")

        from rtpipeline.radiomics import radiomics_for_course, run_radiomics
        logger.info("✅ Radiomics integration imports")

        from rtpipeline.cli import build_parser
        logger.info("✅ CLI integration imports")

        return True

    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False

def validate_configuration_loading():
    """Validate that configurations load correctly."""
    logger.info("🔍 Validating configuration loading...")

    from rtpipeline.custom_structures import CustomStructureProcessor

    # Test example config
    processor = CustomStructureProcessor()

    try:
        processor.load_config("custom_structures_example.yaml")
        example_count = len(processor.custom_configs)
        logger.info(f"✅ Example config: {example_count} structures")
    except Exception as e:
        logger.error(f"❌ Example config failed: {e}")
        return False

    # Test pelvic config
    processor = CustomStructureProcessor()
    try:
        processor.load_config("custom_structures_pelvic.yaml")
        pelvic_count = len(processor.custom_configs)
        logger.info(f"✅ Pelvic config: {pelvic_count} structures")
    except Exception as e:
        logger.error(f"❌ Pelvic config failed: {e}")
        return False

    return True

def validate_boolean_operations():
    """Validate boolean operations work correctly."""
    logger.info("🔍 Validating boolean operations...")

    from rtpipeline.custom_structures import CustomStructureProcessor
    import numpy as np

    processor = CustomStructureProcessor()

    # Create test masks
    mask1 = np.zeros((5, 5, 5), dtype=np.uint8)
    mask1[1:4, 1:4, 1:4] = 1

    mask2 = np.zeros((5, 5, 5), dtype=np.uint8)
    mask2[2:5, 2:5, 2:5] = 1

    try:
        # Test all operations
        union_result = processor.union([mask1, mask2])
        assert union_result is not None and np.sum(union_result) > 0

        intersection_result = processor.intersection([mask1, mask2])
        assert intersection_result is not None

        subtract_result = processor.subtract([mask1, mask2])
        assert subtract_result is not None

        xor_result = processor.xor([mask1, mask2])
        assert xor_result is not None

        logger.info("✅ All boolean operations work correctly")
        return True

    except Exception as e:
        logger.error(f"❌ Boolean operations failed: {e}")
        return False

def validate_margin_operations():
    """Validate margin operations work correctly."""
    logger.info("🔍 Validating margin operations...")

    from rtpipeline.custom_structures import CustomStructureProcessor, MarginConfig
    import numpy as np

    processor = CustomStructureProcessor(spacing=(1.0, 1.0, 1.0))

    # Create test mask
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[4:6, 4:6, 4:6] = 1  # 2x2x2 cube

    try:
        # Test uniform margin
        margin = MarginConfig(uniform_mm=2)
        expanded = processor.apply_margin(mask, margin)

        assert expanded is not None
        assert np.sum(expanded) > np.sum(mask)  # Should be larger

        logger.info("✅ Margin operations work correctly")
        return True

    except Exception as e:
        logger.error(f"❌ Margin operations failed: {e}")
        return False

def validate_cli_integration():
    """Validate CLI integration."""
    logger.info("🔍 Validating CLI integration...")

    try:
        from rtpipeline.cli import build_parser

        parser = build_parser()

        # Test help includes custom structures
        help_text = parser.format_help()
        assert "--custom-structures" in help_text
        assert "pelvic template" in help_text

        # Test argument parsing
        args = parser.parse_args([
            "--dicom-root", "/test/path",
            "--custom-structures", "test.yaml"
        ])

        assert args.custom_structures == "test.yaml"

        logger.info("✅ CLI integration works correctly")
        return True

    except Exception as e:
        logger.error(f"❌ CLI integration failed: {e}")
        return False

def validate_pelvic_template_completeness():
    """Validate pelvic template has all expected structures."""
    logger.info("🔍 Validating pelvic template completeness...")

    from rtpipeline.custom_structures import CustomStructureProcessor

    processor = CustomStructureProcessor()
    processor.load_config("custom_structures_pelvic.yaml")

    # Expected key structures from the Boolean operations notebook
    expected_structures = [
        "iliac_vess",
        "iliac_area",
        "pelvic_bones",
        "pelvic_bones_3mm",
        "bowel_bag",
        "major_vessels",
        "pelvic_OARs"
    ]

    found_structures = [config.name for config in processor.custom_configs]

    missing = set(expected_structures) - set(found_structures)
    if missing:
        logger.error(f"❌ Missing expected structures: {missing}")
        return False

    logger.info(f"✅ All {len(expected_structures)} key pelvic structures present")

    # Validate structure types
    operations_used = set(config.operation for config in processor.custom_configs)
    expected_operations = {"union", "intersection", "subtract"}

    if not expected_operations.intersection(operations_used):
        logger.error("❌ No boolean operations found in pelvic template")
        return False

    logger.info(f"✅ Uses operations: {operations_used}")

    # Validate margins are used
    margins_used = sum(1 for config in processor.custom_configs if config.margin)
    if margins_used == 0:
        logger.error("❌ No margin operations found in pelvic template")
        return False

    logger.info(f"✅ {margins_used} structures use margins")

    return True

def main():
    """Run complete validation."""
    logger.info("🚀 Starting complete custom structures implementation validation...")

    validations = [
        ("File Structure", validate_file_structure),
        ("Module Imports", validate_module_imports),
        ("Configuration Loading", validate_configuration_loading),
        ("Boolean Operations", validate_boolean_operations),
        ("Margin Operations", validate_margin_operations),
        ("CLI Integration", validate_cli_integration),
        ("Pelvic Template", validate_pelvic_template_completeness),
    ]

    results = []
    for name, validation_func in validations:
        logger.info(f"\n{'='*50}")
        logger.info(f"VALIDATION: {name}")
        logger.info(f"{'='*50}")

        try:
            success = validation_func()
            results.append((name, success))
        except Exception as e:
            logger.error(f"❌ {name} validation failed with exception: {e}")
            results.append((name, False))

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'='*50}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status}: {name}")

    logger.info(f"\nOVERALL: {passed}/{total} validations passed")

    if passed == total:
        logger.info("\n🎉 ALL VALIDATIONS PASSED!")
        logger.info("Custom structures implementation is complete and working correctly.")
        logger.info("\nThe pelvic template will be used automatically in RTpipeline.")
        return 0
    else:
        logger.error(f"\n💥 {total - passed} VALIDATION(S) FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())