#!/usr/bin/env python3
"""
Test script to validate rtpipeline fixes and environment setup.
"""

import sys
import importlib
import subprocess
from pathlib import Path

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    
    try:
        import rtpipeline
        print("✓ rtpipeline package imports successfully")
    except ImportError as e:
        print(f"✗ rtpipeline import failed: {e}")
        return False
    
    # Test main modules
    modules = ['cli', 'config', 'organize', 'dvh', 'segmentation', 'visualize']
    for module in modules:
        try:
            importlib.import_module(f'rtpipeline.{module}')
            print(f"✓ rtpipeline.{module}")
        except ImportError as e:
            print(f"✗ rtpipeline.{module}: {e}")
    
    return True

def test_dependencies():
    """Test that all dependencies are available."""
    print("\nTesting dependencies...")
    
    required = [
        'numpy', 'pandas', 'scipy', 'matplotlib', 
        'pydicom', 'SimpleITK', 'rt_utils'
    ]
    
    optional = [
        'radiomics', 'nbformat', 'IPython'
    ]
    
    missing_required = []
    missing_optional = []
    
    for dep in required:
        try:
            importlib.import_module(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"✗ {dep} (REQUIRED)")
            missing_required.append(dep)
    
    for dep in optional:
        try:
            importlib.import_module(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"⚠ {dep} (optional)")
            missing_optional.append(dep)
    
    return len(missing_required) == 0, missing_required, missing_optional

def test_numpy_version():
    """Check numpy version compatibility."""
    print("\nTesting numpy version...")
    
    try:
        import numpy as np
        version = np.__version__
        major = int(version.split('.')[0])
        
        print(f"NumPy version: {version}")
        
        if major >= 2:
            print("✓ NumPy 2.x detected with legacy compatibility support")
            # Test if our legacy shims work
            try:
                from rtpipeline.numpy_legacy_compat import verify_compatibility
                if verify_compatibility():
                    print("✓ NumPy legacy compatibility shims working")
                    return True
                else:
                    print("⚠ NumPy legacy compatibility shims failed")
                    return False
            except ImportError:
                print("⚠ NumPy legacy compatibility shims not available")
                return False
        else:
            print("✓ NumPy 1.x version - direct compatibility")
            return True
    except Exception as e:
        print(f"✗ NumPy check failed: {e}")
        return False

def test_external_tools():
    """Test external command-line tools."""
    print("\nTesting external tools...")
    
    tools = {
        'dcm2niix': 'DCM to NIfTI conversion (optional)',
        'TotalSegmentator': 'Auto segmentation (optional if --no-segmentation)'
    }
    
    available = {}
    
    for tool, desc in tools.items():
        try:
            result = subprocess.run([tool, '--help'], 
                                  capture_output=True, 
                                  timeout=10)
            if result.returncode == 0:
                print(f"✓ {tool}: {desc}")
                available[tool] = True
            else:
                # Check for specific errors
                stderr = result.stderr.decode() if result.stderr else ""
                if "np.isdtype" in stderr:
                    print(f"✗ {tool}: NumPy compatibility issue")
                    print("  Fix: pip install 'numpy>=1.20,<2.0'")
                else:
                    print(f"⚠ {tool}: Available but returned error")
                available[tool] = False
        except FileNotFoundError:
            print(f"⚠ {tool}: Not found in PATH - {desc}")
            available[tool] = False
        except subprocess.TimeoutExpired:
            print(f"⚠ {tool}: Command timeout")
            available[tool] = False
        except Exception as e:
            print(f"⚠ {tool}: Error testing - {e}")
            available[tool] = False
    
    return available

def test_basic_functionality():
    """Test basic rtpipeline functionality."""
    print("\nTesting basic functionality...")
    
    try:
        from rtpipeline.cli import build_parser
        parser = build_parser()
        print("✓ CLI parser builds successfully")
    except Exception as e:
        print(f"✗ CLI parser failed: {e}")
        return False
    
    try:
        from rtpipeline.config import PipelineConfig
        from pathlib import Path
        config = PipelineConfig(
            dicom_root=Path("."),
            output_root=Path("."),
            logs_root=Path(".")
        )
        print("✓ Config object creates successfully")
    except Exception as e:
        print(f"✗ Config creation failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("=== rtpipeline Environment Test ===")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()
    
    # Run tests
    import_ok = test_imports()
    deps_ok, missing_req, missing_opt = test_dependencies()
    numpy_ok = test_numpy_version()
    tools_available = test_external_tools()
    basic_ok = test_basic_functionality()
    
    print("\n=== Summary ===")
    
    if import_ok and deps_ok and basic_ok:
        print("✓ Core rtpipeline functionality should work")
    else:
        print("✗ Core issues detected - rtpipeline may not work properly")
    
    if missing_req:
        print(f"✗ Missing required dependencies: {', '.join(missing_req)}")
        print("  Install with: pip install " + " ".join(missing_req))
    
    if missing_opt:
        print(f"⚠ Missing optional dependencies: {', '.join(missing_opt)}")
        print("  Some features may be disabled")
    
    if not numpy_ok:
        print("⚠ NumPy version may cause segmentation issues")
    
    segmentation_ok = tools_available.get('TotalSegmentator', False) and numpy_ok
    if not segmentation_ok:
        print("⚠ Segmentation may not work - consider --no-segmentation")
    else:
        print("✓ Segmentation should work")
    
    if not tools_available.get('dcm2niix', False):
        print("⚠ dcm2niix not available - NIfTI conversion will be skipped")
    
    print("\nFor setup help, see:")
    print("- ./setup_environment.sh (automated setup)")
    print("- TROUBLESHOOTING.md (manual fixes)")
    print("- rtpipeline doctor (detailed environment check)")
    
    # Return exit code
    return 0 if (import_ok and deps_ok and basic_ok) else 1

if __name__ == "__main__":
    sys.exit(main())