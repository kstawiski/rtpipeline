"""
NumPy 2.0 compatibility shim for rtpipeline
Provides missing numpy.isdtype function for NumPy 1.x to maintain TotalSegmentator compatibility
"""

import numpy as np
import sys
from typing import Any, Union

def _numpy_isdtype_shim(dtype: Any, kind: Union[str, type, tuple] = None) -> bool:
    """
    Compatibility shim for numpy.isdtype (introduced in NumPy 2.0)
    
    This function provides basic dtype checking functionality for NumPy 1.x
    to maintain compatibility with packages that expect NumPy 2.0 features.
    
    Args:
        dtype: The dtype to check
        kind: The kind of dtype to check against (optional)
        
    Returns:
        bool: Whether the dtype matches the specified kind
    """
    try:
        # Convert to numpy dtype if not already
        if hasattr(np, 'dtype'):
            if not isinstance(dtype, np.dtype):
                dtype = np.dtype(dtype)
        
        # If no kind specified, just return True if it's a valid dtype
        if kind is None:
            return True
            
        # Handle string kinds (basic implementation)
        if isinstance(kind, str):
            dtype_kind = getattr(dtype, 'kind', '')
            if kind in ['bool', 'b']:
                return dtype_kind == 'b'
            elif kind in ['int', 'i', 'u']:
                return dtype_kind in ['i', 'u']
            elif kind in ['float', 'f']:
                return dtype_kind == 'f'
            elif kind in ['complex', 'c']:
                return dtype_kind == 'c'
            elif kind in ['str', 'U', 'S']:
                return dtype_kind in ['U', 'S']
            else:
                return dtype_kind == kind
                
        # Handle type kinds
        if isinstance(kind, type):
            return np.issubdtype(dtype, kind)
            
        # Handle tuple of kinds
        if isinstance(kind, tuple):
            return any(_numpy_isdtype_shim(dtype, k) for k in kind)
            
        return False
        
    except Exception:
        # Fallback: if anything goes wrong, return False
        return False

def patch_numpy_bool():
    """
    Monkey-patch numpy to add 'bool' alias if it doesn't exist (NumPy < 2.0)
    In NumPy 2.0, np.bool was deprecated and replaced with np.bool_
    """
    numpy_version = tuple(map(int, np.__version__.split('.')[:2]))
    
    # Only patch if NumPy < 2.0 and bool doesn't exist as an attribute
    if numpy_version < (2, 0):
        # Suppress FutureWarning when checking for np.bool existence
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            has_bool = hasattr(np, 'bool')
        
        if not has_bool:
            # Create bool alias pointing to bool_
            np.bool = np.bool_
            print(f"Applied NumPy bool compatibility alias for NumPy {np.__version__}")

def patch_numpy_isdtype():
    """
    Monkey-patch numpy to add isdtype function if it doesn't exist (NumPy < 2.0)
    """
    numpy_version = tuple(map(int, np.__version__.split('.')[:2]))
    
    # Only patch if NumPy < 2.0 and isdtype doesn't exist
    if numpy_version < (2, 0) and not hasattr(np, 'isdtype'):
        np.isdtype = _numpy_isdtype_shim
        print(f"Applied NumPy isdtype compatibility shim for NumPy {np.__version__}")
    
def apply_numpy_compatibility():
    """
    Apply all necessary NumPy compatibility patches
    """
    try:
        patch_numpy_bool()
        patch_numpy_isdtype()
        return True
    except Exception as e:
        print(f"Warning: Failed to apply NumPy compatibility patches: {e}")
        return False

# Auto-apply patches when this module is imported
if __name__ != "__main__":
    apply_numpy_compatibility()