#!/usr/bin/env python3
"""
Wrapper script for TotalSegmentator that applies NumPy compatibility patches.
This ensures TotalSegmentator works with NumPy < 2.0 by providing missing functions.
"""

import sys
import os

# Add the rtpipeline directory to Python path
rtpipeline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, rtpipeline_dir)

# Apply numpy compatibility patches before importing anything else
try:
    from rtpipeline.numpy_compat import apply_numpy_compatibility
    apply_numpy_compatibility()
except ImportError:
    pass

# Now run the actual TotalSegmentator
if __name__ == "__main__":
    try:
        from totalsegmentator.bin.TotalSegmentator import main
        main()
    except ImportError:
        print("Error: TotalSegmentator not found. Please install it with: pip install TotalSegmentator")
        sys.exit(1)