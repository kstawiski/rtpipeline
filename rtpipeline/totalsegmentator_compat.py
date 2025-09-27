#!/usr/bin/env python3
"""
Wrapper for TotalSegmentator with NumPy 2.x compatibility shims for legacy libraries.

Features:
- Uses NumPy 2.x as the base (modern, fast, compatible with newer libraries)
- Provides backward compatibility shims for legacy libraries that expect NumPy 1.x behavior
- Forces a headless Matplotlib backend to avoid GUI requirements
- Enhanced error handling and logging for debugging DICOM-SEG issues
"""

import sys
import os
import types
import warnings
from importlib.machinery import ModuleSpec
try:
    from importlib.abc import Loader as _LoaderABC  # py3.11+
except Exception:
    _LoaderABC = object  # fallback

class _DummyLoader(_LoaderABC):  # pragma: no cover
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        return None

# Add the rtpipeline directory to Python path
rtpipeline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, rtpipeline_dir)

# Force headless plotting
os.environ.setdefault("MPLBACKEND", "Agg")

# Apply comprehensive warning suppression for compatibility
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*numpy.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*numpy.*")

# Apply NumPy 2.x with legacy compatibility shims for older libraries
try:
    # Import our new legacy compatibility module
    from rtpipeline.numpy_legacy_compat import apply_all_legacy_compatibility, verify_compatibility
    apply_all_legacy_compatibility()
    verify_compatibility()
except ImportError as e:
    print(f"Warning: Could not apply NumPy legacy compatibility shims: {e}")
    # Fallback to basic NumPy import
    import numpy as np
    print(f"Using NumPy {np.__version__} without compatibility shims")

# Provide a minimal seaborn stub to satisfy nnUNet logger imports without
# triggering heavy SciPy imports in environments where SciPy may be problematic.
if 'seaborn' not in sys.modules:
    sb = types.ModuleType('seaborn')
    # Commonly used API in nnUNet logger
    def _noop(*args, **kwargs):
        return None
    sb.set = _noop
    sb.set_theme = _noop
    sb.set_context = _noop
    sb.set_style = _noop
    sb.axes_style = lambda *a, **k: {}
    # Expose a couple of attributes to look like seaborn
    sb.__version__ = '0.0'
    # Mark as package-like and provide a spec
    sb.__path__ = []  # type: ignore[attr-defined]
    sb.__spec__ = ModuleSpec('seaborn', _DummyLoader(), is_package=True)  # type: ignore[attr-defined]
    sys.modules['seaborn'] = sb

# Note: TotalSegmentator actually needs the real scipy, so we don't stub it
# The scipy stub has been removed to avoid conflicts

# Provide a minimal scikit-learn stub to avoid importing SciPy during inference
if 'sklearn' not in sys.modules:
    sk = types.ModuleType('sklearn')
    sk.__version__ = '0.0'
    sk.__path__ = []  # type: ignore[attr-defined]
    sk.__spec__ = ModuleSpec('sklearn', _DummyLoader(), is_package=True)  # type: ignore[attr-defined]
    # Submodule: model_selection with KFold placeholder (unused during inference)
    ms = types.ModuleType('sklearn.model_selection')
    ms.__spec__ = ModuleSpec('sklearn.model_selection', _DummyLoader(), is_package=False)  # type: ignore[attr-defined]
    class KFold:  # pragma: no cover
        def __init__(self, *a, **k): pass
        def split(self, *a, **k): return iter(())
    ms.KFold = KFold
    # Submodule shells required by sklearn import chain in some versions
    utils = types.ModuleType('sklearn.utils')
    utils.__spec__ = ModuleSpec('sklearn.utils', _DummyLoader(), is_package=True)  # type: ignore[attr-defined]
    utils_fixes = types.ModuleType('sklearn.utils.fixes')
    utils_fixes.__spec__ = ModuleSpec('sklearn.utils.fixes', _DummyLoader(), is_package=False)  # type: ignore[attr-defined]
    def parse_version(v):  # pragma: no cover
        return v
    utils_fixes.parse_version = parse_version
    utils_meta = types.ModuleType('sklearn.utils._metadata_requests')
    utils_meta.__spec__ = ModuleSpec('sklearn.utils._metadata_requests', _DummyLoader(), is_package=False)  # type: ignore[attr-defined]
    utils_chunk = types.ModuleType('sklearn.utils._chunking')
    utils_chunk.__spec__ = ModuleSpec('sklearn.utils._chunking', _DummyLoader(), is_package=False)  # type: ignore[attr-defined]
    utils_param = types.ModuleType('sklearn.utils._param_validation')
    utils_param.__spec__ = ModuleSpec('sklearn.utils._param_validation', _DummyLoader(), is_package=False)  # type: ignore[attr-defined]
    utils_valid = types.ModuleType('sklearn.utils.validation')
    utils_valid.__spec__ = ModuleSpec('sklearn.utils.validation', _DummyLoader(), is_package=False)  # type: ignore[attr-defined]
    utils_arr = types.ModuleType('sklearn.utils._array_api')
    utils_arr.__spec__ = ModuleSpec('sklearn.utils._array_api', _DummyLoader(), is_package=False)  # type: ignore[attr-defined]
    # minimal names referenced during import; no runtime use expected
    def _noop(*a, **k): return None
    utils_param.Interval = object
    utils_param.validate_params = _noop
    utils_valid._is_arraylike_not_scalar = lambda x: True
    utils_arr._asarray_with_order = lambda *a, **k: a[0] if a else None
    utils_arr._is_numpy_namespace = lambda *a, **k: True
    utils_arr.get_namespace = lambda *a, **k: None
    # Register modules
    sys.modules['sklearn'] = sk
    sys.modules['sklearn.model_selection'] = ms
    sys.modules['sklearn.utils'] = utils
    sys.modules['sklearn.utils.fixes'] = utils_fixes
    sys.modules['sklearn.utils._metadata_requests'] = utils_meta
    sys.modules['sklearn.utils._chunking'] = utils_chunk
    sys.modules['sklearn.utils._param_validation'] = utils_param
    sys.modules['sklearn.utils.validation'] = utils_valid
    sys.modules['sklearn.utils._array_api'] = utils_arr

# Now run the actual TotalSegmentator
if __name__ == "__main__":
    import sys
    try:
        # Print command line args for debugging
        print("TotalSegmentator args:", sys.argv[1:])
        
        # Suppress additional warnings during execution
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            
            from totalsegmentator.bin.TotalSegmentator import main
            main()
            
    except ImportError as e:
        print(f"Error: TotalSegmentator not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"TotalSegmentator execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
