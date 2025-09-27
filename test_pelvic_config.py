#!/usr/bin/env python3
"""
Test script for pelvic custom structures configuration.

Usage:
    python test_pelvic_config.py
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def test_pelvic_config():
    """Test loading of the pelvic custom structures configuration."""
    from rtpipeline.custom_structures import CustomStructureProcessor

    logger.info("Testing pelvic custom structures configuration...")

    # Load the pelvic config
    config_path = Path("custom_structures_pelvic.yaml")
    if not config_path.exists():
        logger.error("Pelvic config file not found: %s", config_path)
        return False

    processor = CustomStructureProcessor()
    processor.load_config(config_path)

    logger.info(f"✓ Loaded {len(processor.custom_configs)} pelvic structure configurations")

    # Group structures by category
    categories = {
        "Vascular": [],
        "Bone": [],
        "Muscle": [],
        "Bowel": [],
        "OAR": [],
        "Other": []
    }

    for config in processor.custom_configs:
        name = config.name.lower()
        if any(word in name for word in ["vessel", "vena", "artery", "vascular"]):
            categories["Vascular"].append(config)
        elif any(word in name for word in ["bone", "spine", "femur", "pelvic_bones", "lumbar"]):
            categories["Bone"].append(config)
        elif any(word in name for word in ["muscle", "gluteus", "iliopsoas"]):
            categories["Muscle"].append(config)
        elif any(word in name for word in ["bowel", "colon", "duodenum"]):
            categories["Bowel"].append(config)
        elif any(word in name for word in ["oar", "kidney", "bladder"]):
            categories["OAR"].append(config)
        else:
            categories["Other"].append(config)

    # Print categorized structures
    for category, structures in categories.items():
        if structures:
            logger.info(f"\n{category} Structures ({len(structures)}):")
            for structure in structures:
                margin_info = ""
                if structure.margin:
                    if hasattr(structure.margin, 'uniform_mm') and structure.margin.uniform_mm:
                        margin_info = f" +{structure.margin.uniform_mm}mm"
                    else:
                        margin_info = " +margin"

                logger.info(f"  • {structure.name}: {structure.operation} of {structure.source_structures}{margin_info}")

    # Test specific important structures
    important_structures = [
        "iliac_vess", "iliac_area", "pelvic_bones", "pelvic_bones_3mm",
        "bowel_bag", "major_vessels", "pelvic_OARs"
    ]

    found_important = []
    for config in processor.custom_configs:
        if config.name in important_structures:
            found_important.append(config.name)

    logger.info(f"\n✓ Found {len(found_important)}/{len(important_structures)} important structures:")
    for name in found_important:
        logger.info(f"  • {name}")

    missing = set(important_structures) - set(found_important)
    if missing:
        logger.warning(f"Missing important structures: {missing}")

    return True

def main():
    """Run the test."""
    logger.info("Starting pelvic configuration test...")

    try:
        success = test_pelvic_config()
        if success:
            logger.info("\n✅ Pelvic configuration test passed!")
            return 0
        else:
            logger.error("\n❌ Pelvic configuration test failed!")
            return 1

    except Exception as e:
        logger.error(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())