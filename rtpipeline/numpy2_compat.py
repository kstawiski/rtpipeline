"""
Enhanced NumPy 2.x compatibility layer for RTpipeline.

This module ensures compatibility between:
- TotalSegmentator (works best with NumPy 2.x)
- PyRadiomics (requires NumPy 1.x or special handling)
- Other components that may have varying NumPy requirements
"""

import numpy as np
import sys
import os
import warnings
from typing import Any, Union

def get_numpy_version() -> tuple:
    """Get NumPy version as tuple."""
    return tuple(map(int, np.__version__.split('.')[:2]))

def patch_for_pyradiomics():
    """Apply patches needed for PyRadiomics compatibility with NumPy 2.x."""
    numpy_version = get_numpy_version()

    if numpy_version >= (2, 0):
        # PyRadiomics expects some deprecated NumPy APIs
        # Create compatibility aliases

        # np.bool was removed in NumPy 2.0
        if not hasattr(np, 'bool'):
            np.bool = np.bool_

        # np.int was removed in NumPy 2.0
        if not hasattr(np, 'int'):
            np.int = np.int_

        # np.float was removed in NumPy 2.0
        if not hasattr(np, 'float'):
            np.float = np.float64

        # np.complex was removed in NumPy 2.0
        if not hasattr(np, 'complex'):
            np.complex = np.complex128

        # np.object was removed in NumPy 2.0
        if not hasattr(np, 'object'):
            np.object = object

        # np.str was removed in NumPy 2.0
        if not hasattr(np, 'str'):
            np.str = str

        print(f"Applied PyRadiomics compatibility patches for NumPy {np.__version__}")

def patch_scipy_compatibility():
    """Apply patches for SciPy compatibility with NumPy 2.x."""
    try:
        # Some SciPy versions have issues with NumPy 2.x
        # Ensure proper dtype handling
        original_asarray = np.asarray

        def patched_asarray(a, dtype=None, order=None, *, like=None):
            """Patched asarray that handles edge cases."""
            try:
                result = original_asarray(a, dtype=dtype, order=order)
                return result
            except Exception:
                # Fallback for problematic inputs
                if hasattr(a, '__array__'):
                    return a.__array__()
                return original_asarray(a, dtype=dtype, order=order)

        np.asarray = patched_asarray

    except Exception as e:
        warnings.warn(f"Could not apply SciPy compatibility patches: {e}")

def configure_environment():
    """Configure environment for optimal compatibility."""
    # Suppress warnings that may confuse users
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="radiomics")
    warnings.filterwarnings("ignore", category=FutureWarning, module="radiomics")
    warnings.filterwarnings("ignore", message=".*numpy.dtype size changed.*")
    warnings.filterwarnings("ignore", message=".*numpy.ufunc size changed.*")

    # Set threading to prevent conflicts in parallel processing
    env_vars = {
        'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS', '1'),
        'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS', '1'),
    }

    for var, value in env_vars.items():
        os.environ[var] = value

def check_pyradiomics_compatibility():
    """Check if PyRadiomics can be imported with current NumPy."""
    try:
        import radiomics
        return True
    except ImportError as e:
        if "numpy" in str(e).lower():
            print(f"PyRadiomics incompatible with NumPy {np.__version__}")
            print("Will use subprocess-based extraction via pyradiomics_compat module")
            return False
        raise

def apply_all_patches():
    """Apply all compatibility patches based on environment."""
    numpy_version = get_numpy_version()

    print(f"Configuring compatibility for NumPy {np.__version__}")

    # Configure environment first
    configure_environment()

    # Apply patches based on NumPy version
    if numpy_version >= (2, 0):
        patch_for_pyradiomics()
        patch_scipy_compatibility()

    # Import base compatibility module for additional patches
    try:
        from . import numpy_compat
        numpy_compat.apply_numpy_compatibility()
    except ImportError:
        pass

    return True

# Auto-apply patches when imported
if __name__ != "__main__":
    apply_all_patches()