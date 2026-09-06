"""Exact PyRadiomics MCC contraction using contiguous, per-angle BLAS operands.

Preserve epsilon inside the denominator. Factoring it out or replacing Q
with a symmetric/SVD approximation changes the configured measurement.
Only summation order changes. Each completed angle is genuine progress.
"""
import numpy as np


def mcc_feature(self):
    from .robustness_watchdog import report_progress
    p = self.P_glcm
    px, py, eps = (self.coefficients[k] for k in ('px', 'py', 'eps'))
    if p.shape[1] < 2:
        return 1
    eigenvalues = []
    for voxel in range(p.shape[0]):
        angles = []
        for angle in range(p.shape[3]):
            # The native layout has angle as the innermost dimension. Passing
            # that strided view to matmul can fall back to cubic scalar loops.
            matrix = np.ascontiguousarray(p[voxel, :, :, angle])
            denominator = (px[voxel, :, 0, angle, None]
                           * py[voxel, 0, :, angle][None, :] + eps)
            weighted = np.ascontiguousarray(matrix / denominator)
            q = weighted @ matrix.T
            angles.append(np.linalg.eigvals(q))
            report_progress(f"MCC_angle_complete:{voxel}:{angle}")
        eigenvalues.append(angles)
    values = np.asarray(eigenvalues)
    values.sort()
    return np.nanmean(np.sqrt(values[:, :, -2]), 1).real


def install_mcc_contraction():
    # Only invoked in robustness extraction workers, not on module import.
    from radiomics.glcm import RadiomicsGLCM
    if RadiomicsGLCM.getMCCFeatureValue is not mcc_feature:
        RadiomicsGLCM.getMCCFeatureValue = mcc_feature
