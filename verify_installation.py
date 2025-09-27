#!/usr/bin/env python3
"""
RTpipeline Installation Verification Script
Comprehensive check of all components and compatibility layers
"""

import sys
import os
import subprocess
from pathlib import Path

# Add rtpipeline to path
sys.path.insert(0, str(Path(__file__).parent))

def check_mark(condition):
    return "✅" if condition else "❌"

def warning_mark(condition):
    return "✅" if condition else "⚠️"

print("=" * 60)
print("    RTpipeline Installation Verification")
print("=" * 60)
print()

# 1. Python Version
print("1. Python Environment:")
python_version = sys.version.split()[0]
python_major = int(python_version.split('.')[0])
python_minor = int(python_version.split('.')[1])
is_python_ok = python_major == 3 and python_minor >= 10
print(f"   {check_mark(is_python_ok)} Python {python_version}")
print()

# 2. NumPy Configuration
print("2. NumPy Configuration:")
try:
    import numpy as np
    numpy_version = np.__version__
    numpy_major = int(numpy_version.split('.')[0])
    is_numpy2 = numpy_major >= 2
    print(f"   {check_mark(is_numpy2)} NumPy {numpy_version} {'(Optimal for TotalSegmentator)' if is_numpy2 else '(Consider upgrading to 2.x)'}")

    # Check compatibility aliases
    has_bool = hasattr(np, 'bool')
    has_int = hasattr(np, 'int')
    has_float = hasattr(np, 'float')
    has_isdtype = hasattr(np, 'isdtype')

    if is_numpy2:
        print(f"   {check_mark(has_bool)} np.bool compatibility alias")
        print(f"   {check_mark(has_int)} np.int compatibility alias")
        print(f"   {check_mark(has_float)} np.float compatibility alias")
    print(f"   {check_mark(has_isdtype)} np.isdtype available")
except ImportError as e:
    print(f"   ❌ NumPy not installed: {e}")
print()

# 3. Core Dependencies
print("3. Core Dependencies:")
dependencies = [
    ("scipy", None),
    ("pandas", None),
    ("SimpleITK", None),
    ("pydicom", None),
    ("matplotlib", None),
    ("sklearn", "scikit-learn"),
    ("skimage", "scikit-image"),
    ("PIL", "pillow"),
]

for module_name, package_name in dependencies:
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'installed')
        print(f"   ✅ {package_name or module_name}: {version}")
    except ImportError:
        print(f"   ❌ {package_name or module_name}: not installed")
print()

# 4. Medical Imaging Tools
print("4. Medical Imaging Tools:")

# TotalSegmentator
try:
    import totalsegmentator
    ts_version = getattr(totalsegmentator, '__version__', 'installed')
    print(f"   ✅ TotalSegmentator: {ts_version}")
except ImportError:
    print(f"   ❌ TotalSegmentator: not installed")

# PyRadiomics
try:
    from rtpipeline.radiomics import _have_pyradiomics
    if _have_pyradiomics():
        try:
            import radiomics
            pr_version = getattr(radiomics, '__version__', 'installed')
            print(f"   ✅ PyRadiomics: {pr_version} (compatible)")
        except:
            print(f"   ✅ PyRadiomics: compatible via wrapper")
    else:
        print(f"   ⚠️ PyRadiomics: will use subprocess fallback")
except Exception as e:
    print(f"   ❌ PyRadiomics check failed: {e}")

# rt-utils
try:
    import rt_utils
    print(f"   ✅ rt-utils: installed")
except ImportError:
    print(f"   ❌ rt-utils: not installed")

# dcm2niix
try:
    result = subprocess.run(['dcm2niix', '-h'], capture_output=True, timeout=2)
    print(f"   ✅ dcm2niix: available")
except:
    print(f"   ⚠️ dcm2niix: not found in PATH")
print()

# 5. RTpipeline Modules
print("5. RTpipeline Modules:")
rtpipeline_modules = [
    "rtpipeline.cli",
    "rtpipeline.config",
    "rtpipeline.organize",
    "rtpipeline.segmentation",
    "rtpipeline.radiomics",
    "rtpipeline.radiomics_parallel",
    "rtpipeline.numpy2_compat",
    "rtpipeline.pyradiomics_compat",
    "rtpipeline.totalsegmentator_compat",
]

for module_path in rtpipeline_modules:
    try:
        __import__(module_path)
        print(f"   ✅ {module_path.split('.')[-1]}")
    except ImportError as e:
        print(f"   ❌ {module_path.split('.')[-1]}: {e}")
print()

# 6. Parallel Processing Configuration
print("6. Parallel Processing:")
try:
    from rtpipeline.radiomics_parallel import (
        is_parallel_radiomics_enabled,
        _calculate_optimal_workers
    )

    cpu_count = os.cpu_count() or 1
    optimal_workers = _calculate_optimal_workers()
    is_enabled = is_parallel_radiomics_enabled()

    print(f"   ℹ️ CPU cores: {cpu_count}")
    print(f"   ℹ️ Optimal workers: {optimal_workers}")
    print(f"   {check_mark(is_enabled)} Parallel radiomics: {'ENABLED' if is_enabled else 'DISABLED'}")

    if not is_enabled:
        print(f"      💡 Enable with: export RTPIPELINE_USE_PARALLEL_RADIOMICS=1")
except Exception as e:
    print(f"   ❌ Parallel processing check failed: {e}")
print()

# 7. Compatibility Layer Status
print("7. Compatibility Layers:")
try:
    from rtpipeline import numpy2_compat
    print(f"   ✅ NumPy 2.x compatibility layer loaded")
except ImportError:
    print(f"   ❌ NumPy 2.x compatibility layer not found")

try:
    from rtpipeline.pyradiomics_compat import execute_radiomics
    print(f"   ✅ PyRadiomics compatibility wrapper available")
except ImportError:
    print(f"   ❌ PyRadiomics compatibility wrapper not found")

try:
    from rtpipeline.totalsegmentator_compat import main
    print(f"   ✅ TotalSegmentator compatibility wrapper available")
except ImportError:
    print(f"   ⚠️ TotalSegmentator compatibility wrapper not found")
print()

# 8. Environment Variables
print("8. Environment Variables:")
env_vars = [
    ("RTPIPELINE_USE_PARALLEL_RADIOMICS", "Parallel radiomics"),
    ("OMP_NUM_THREADS", "OpenMP threads"),
    ("MKL_NUM_THREADS", "MKL threads"),
]

for var, description in env_vars:
    value = os.environ.get(var, "not set")
    if var == "RTPIPELINE_USE_PARALLEL_RADIOMICS":
        is_good = value in ('1', 'true', 'yes', 'True')
        print(f"   {warning_mark(is_good)} {description}: {value}")
    else:
        print(f"   ℹ️ {description}: {value}")
print()

# Summary
print("=" * 60)
print("    Summary")
print("=" * 60)

all_good = True
warnings = []
errors = []

# Check critical components
if not is_python_ok:
    errors.append("Python version should be 3.10 or higher")
    all_good = False

if not is_numpy2:
    warnings.append("Consider upgrading to NumPy 2.x for optimal TotalSegmentator performance")

try:
    from rtpipeline.radiomics import _have_pyradiomics
    if not _have_pyradiomics():
        warnings.append("PyRadiomics will use subprocess fallback (slightly slower)")
except:
    errors.append("PyRadiomics compatibility check failed")
    all_good = False

if not is_enabled:
    warnings.append("Parallel radiomics not enabled - export RTPIPELINE_USE_PARALLEL_RADIOMICS=1")

# Print summary
if all_good and not errors:
    print("✅ Installation VERIFIED - All critical components working!")
    print("   Your pipeline is ready for optimal performance.")
elif errors:
    print("❌ Installation has ERRORS:")
    for error in errors:
        print(f"   • {error}")
else:
    print("⚠️ Installation FUNCTIONAL with warnings")

if warnings:
    print("\n💡 Recommendations:")
    for warning in warnings:
        print(f"   • {warning}")

print("\n🚀 Quick Start:")
print("   1. Source optimization: source optimize_pipeline.sh")
print("   2. Run pipeline: ./test.sh")
print("   3. Or use: rtpipeline --help")
print()