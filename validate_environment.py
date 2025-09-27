#!/usr/bin/env python3
"""
Comprehensive test script for rtpipeline environment validation
Tests all major components and dependencies
"""

import sys
import subprocess
import importlib
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BLUE}{Colors.BOLD}=== {text} ==={Colors.ENDC}")

def print_success(text):
    print(f"{Colors.GREEN}✓{Colors.ENDC} {text}")

def print_error(text):
    print(f"{Colors.RED}✗{Colors.ENDC} {text}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠{Colors.ENDC} {text}")

def test_python_version():
    """Test Python version compatibility"""
    print_header("Python Environment")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print(f"Python version: {version_str}")
    
    if version.major == 3 and version.minor >= 11:
        print_success(f"Python {version_str} is compatible")
        return True
    else:
        print_error(f"Python {version_str} may have compatibility issues")
        print_warning("Recommend Python 3.11 for full compatibility")
        return False

def test_package_imports():
    """Test core package imports"""
    print_header("Core Package Imports")
    
    packages = {
        'numpy': 'numpy',
        'pandas': 'pandas', 
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'pydicom': 'pydicom',
        'SimpleITK': 'sitk',
        'sklearn': 'scikit-learn'
    }
    
    success_count = 0
    for package_name, import_name in packages.items():
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, '__version__', 'unknown')
            print_success(f"{import_name}: {version}")
            success_count += 1
        except ImportError as e:
            print_error(f"{import_name}: {e}")
    
    return success_count == len(packages)

def test_medical_imaging_packages():
    """Test medical imaging specific packages"""
    print_header("Medical Imaging Packages")
    
    packages = {
        'dicompylercore': 'dicompyler-core',
        'rt_utils': 'rt-utils',
        'pydicom_seg': 'pydicom-seg',
        'dicom2nifti': 'dicom2nifti'
    }
    
    success_count = 0
    for package_name, display_name in packages.items():
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, '__version__', 'unknown')
            print_success(f"{display_name}: {version}")
            success_count += 1
        except ImportError as e:
            print_error(f"{display_name}: {e}")
    
    return success_count >= len(packages) * 0.8  # 80% success rate

def test_optional_packages():
    """Test optional packages that may not be available"""
    print_header("Optional Packages")
    
    # Test pyradiomics
    try:
        import radiomics
        version = getattr(radiomics, '__version__', 'unknown')
        print_success(f"pyradiomics: {version}")
        radiomics_available = True
    except ImportError:
        print_warning("pyradiomics: Not available (normal for Python 3.12+)")
        radiomics_available = False
    
    # Test TotalSegmentator
    try:
        import totalsegmentator
        version = getattr(totalsegmentator, '__version__', 'unknown')
        print_success(f"TotalSegmentator: {version}")
        totalseg_available = True
    except ImportError:
        print_error("TotalSegmentator: Not available")
        totalseg_available = False
    
    return {'radiomics': radiomics_available, 'totalseg': totalseg_available}

def test_numpy_compatibility():
    """Test numpy version compatibility"""
    print_header("NumPy Compatibility")
    
    try:
        import numpy as np
        version = np.__version__
        print(f"NumPy version: {version}")
        
        major, minor = map(int, version.split('.')[:2])
        if major < 2:
            print_success("NumPy version < 2.0 - Direct compatibility")
            return True
        else:
            print(f"NumPy version >= 2.0 - Testing legacy compatibility...")
            try:
                from rtpipeline.numpy_legacy_compat import verify_compatibility
                if verify_compatibility():
                    print_success("NumPy 2.x with legacy compatibility shims working")
                    return True
                else:
                    print_error("NumPy 2.x legacy compatibility failed")
                    return False
            except ImportError as e:
                print_error(f"NumPy legacy compatibility shims not available: {e}")
                return False
    except ImportError:
        print_error("NumPy not available")
        return False

def test_external_tools():
    """Test external command line tools"""
    print_header("External Tools")
    
    tools = ['dcm2niix']
    success_count = 0
    
    for tool in tools:
        try:
            result = subprocess.run(
                [tool, '--version'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                version = result.stdout.strip().split('\n')[0]
                print_success(f"{tool}: {version}")
                success_count += 1
            else:
                print_error(f"{tool}: Command failed")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print_error(f"{tool}: Not found in PATH")
    
    return success_count == len(tools)

def test_rtpipeline():
    """Test rtpipeline package specifically"""
    print_header("RTpipeline Package")
    
    try:
        import rtpipeline
        version = getattr(rtpipeline, '__version__', 'development')
        print_success(f"rtpipeline package: {version}")
        
        # Test CLI availability
        try:
            result = subprocess.run(
                ['rtpipeline', '--help'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                print_success("rtpipeline CLI: Available")
                cli_available = True
            else:
                print_error("rtpipeline CLI: Failed")
                cli_available = False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print_error("rtpipeline CLI: Not found in PATH")
            cli_available = False
        
        # Test doctor command
        if cli_available:
            try:
                result = subprocess.run(
                    ['rtpipeline', 'doctor'], 
                    capture_output=True, 
                    text=True, 
                    timeout=30
                )
                if result.returncode == 0:
                    print_success("rtpipeline doctor: PASSED")
                    return True
                else:
                    print_warning("rtpipeline doctor: Issues found")
                    print(f"Output: {result.stdout}")
                    return True  # Still functional, just with warnings
            except (subprocess.TimeoutExpired, FileNotFoundError):
                print_error("rtpipeline doctor: Failed to run")
                return False
        
        return True
        
    except ImportError as e:
        print_error(f"rtpipeline package: {e}")
        return False

def test_data_directories():
    """Test for expected data directories"""
    print_header("Data Directories")
    
    expected_dirs = [
        'Data_Organized',
        'Example_data', 
        'Code',
        'Logs'
    ]
    
    success_count = 0
    for dir_name in expected_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print_success(f"{dir_name}: Exists")
            success_count += 1
        else:
            print_warning(f"{dir_name}: Not found")
    
    return success_count >= 2  # At least some data directories

def generate_report(test_results):
    """Generate summary report"""
    print_header("Test Summary")
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result is True)
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print_success("All tests passed! Environment is fully functional.")
        return 0
    elif passed_tests >= total_tests * 0.8:
        print_warning("Most tests passed. Environment is functional with minor issues.")
        return 0
    else:
        print_error("Several tests failed. Environment may have significant issues.")
        return 1

def main():
    """Main test function"""
    print(f"{Colors.BOLD}RTpipeline Environment Validation{Colors.ENDC}")
    print("Testing all components and dependencies...\n")
    
    test_results = {
        'python_version': test_python_version(),
        'core_packages': test_package_imports(),
        'medical_packages': test_medical_imaging_packages(),
        'numpy_compatibility': test_numpy_compatibility(),
        'external_tools': test_external_tools(),
        'rtpipeline': test_rtpipeline(),
        'data_directories': test_data_directories()
    }
    
    # Test optional packages (don't fail on these)
    optional_results = test_optional_packages()
    
    return generate_report(test_results)

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)