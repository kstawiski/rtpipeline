#!/usr/bin/env python3
"""
Test TotalSegmentator without compatibility stubs
"""
import sys
import os

# Apply numpy compatibility patches before importing anything else
try:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        from rtpipeline.numpy_compat import apply_numpy_compatibility
        apply_numpy_compatibility()
except ImportError:
    pass

# Force headless plotting
os.environ.setdefault("MPLBACKEND", "Agg")

if __name__ == "__main__":
    try:
        from totalsegmentator.bin.TotalSegmentator import main
        print("Successfully imported TotalSegmentator main")
        main()
    except ImportError as e:
        print(f"Error: TotalSegmentator import failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error running TotalSegmentator: {e}")
        sys.exit(1)