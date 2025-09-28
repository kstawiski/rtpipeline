# RTpipeline package initialization
# Version 2.0: Dual environment approach
# - Main environment: NumPy 2.x for TotalSegmentator
# - Radiomics environment: NumPy 1.x for PyRadiomics (via conda run)

__version__ = "2.0.0"

__all__ = [
    "config",
    "ct",
    "rt_details",
    "metadata",
    "organize",
    "segmentation",
    "dvh",
    "visualize",
    "custom_structures",
    "radiomics",
    "radiomics_conda",  # New conda-based radiomics module
]

