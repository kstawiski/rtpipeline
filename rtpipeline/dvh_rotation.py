"""Bounded patient-coordinate sampling for slightly rotated RTDOSE grids.

Exact supported axial orientations bypass this module's sampler. Other grids
must be orthonormal (absolute Gram-matrix error <= 1e-6) and within 1 degree of
one of the eight signed axial orientations supported by dicompyler-core.
Frame positions follow the declared monotone Grid Frame Offset Vector.

The destination is a canonical patient XYZ lattice with the original axis
spacings and a node bounding box enclosing every original node. Dose is sampled
with trilinear interpolation in the ORIGINAL grid, without integer requantizing.
Support is the closed original node box, not the enlarged destination box and
not dose > 0. Outside samples are masked, never interpreted as measured zero.

For axial contours, destination XY planes are sampled at the actual contour Z.
Tilted contours use a native-plane mask: a local orthonormal XY lattice is
mapped back into patient coordinates before sampling the original dose. This
keeps each original contour point in its physical plane, with no flattening,
contour snapping or second dose interpolation. Parallel contour planes must be
within 1 degree of axial. A least-squares common normal handles rounded DICOM
coordinates, with each point projected by at most 0.01 mm (recorded per ROI).
Plane grouping tolerance is 0.001 mm; larger projection errors fail closed. Each contour plane represents the slab bounded by adjacent plane
midpoints; the endpoints extend by half their adjacent interval. This equals
dicompyler's slice thickness for uniform planes and does not assign a tiny
near-duplicate interval to every slice. Rasterization, 1 cGy histograms and
metric extraction remain those of dicompyler-core. This is point-dose interpolation, not an energy-conserving
volume integration. ``max_node_displacement_mm`` is the maximum displacement
that snapping the original orientation about its origin WOULD have caused; no
such snapping is performed.
"""
from __future__ import annotations

import copy
from itertools import product

import numpy as np
from pydicom.dataset import Dataset
from scipy.ndimage import map_coordinates

MAX_ROTATION_DEGREES = 1.0
ORTHONORMAL_TOLERANCE = 1e-6
MAX_CONTOUR_PROJECTION_MM = 0.01
METHOD = "rotated_dose_resampled_to_axis_aligned_grid"


def _axial_bases():
    for swap, sx, sy in product((False, True), (-1, 1), (-1, 1)):
        x = np.array([0., sx, 0.]) if swap else np.array([sx, 0., 0.])
        y = np.array([sy, 0., 0.]) if swap else np.array([0., sy, 0.])
        yield np.column_stack((x, y, np.cross(x, y)))


class DoseOrientationError(ValueError):
    """An identifier-free refusal suitable for course QC."""

    def __init__(self, reason, rotation_angle_degrees=None):
        super().__init__(reason)
        self.rotation_angle_degrees = rotation_angle_degrees


class _FloatDose(Dataset):
    """In-memory dicompyler adapter; never serialized as a derived RTDOSE."""

    @property
    def pixel_array(self):
        return self._resampled_pixels


def prepare_rotated_dose(dose, *, source_grids=()):
    """Return None for exact axial grids, else a validated physical sampler."""
    try:
        orientation = np.asarray(dose.ImageOrientationPatient, dtype=float)
        if orientation.shape != (6,) or not np.isfinite(orientation).all():
            raise ValueError
    except (AttributeError, TypeError, ValueError) as exc:
        raise DoseOrientationError("rotated_dose_invalid_orientation") from exc
    bases = list(_axial_bases())
    if any(np.array_equal(orientation, b[:, :2].T.ravel()) for b in bases):
        return None
    basis = np.column_stack((orientation[:3], orientation[3:],
                             np.cross(orientation[:3], orientation[3:])))
    if not np.allclose(basis.T @ basis, np.eye(3), atol=ORTHONORMAL_TOLERANCE, rtol=0):
        raise DoseOrientationError("rotated_dose_non_orthonormal_orientation")
    angles = [float(np.degrees(np.arccos(np.clip((np.trace(b.T @ basis)-1)/2, -1, 1))))
              for b in bases]
    nearest = bases[int(np.argmin(angles))]
    angle = min(angles)
    if angle > MAX_ROTATION_DEGREES + 1e-9:
        raise DoseOrientationError("rotated_dose_rotation_exceeds_1_degree", angle)
    return RotatedDoseGrid(dose, basis, nearest, angle, source_grids=source_grids)


class DoseGridCoordinates:
    """Physical node coordinates, independent of the stored dose values."""

    def __init__(self, dose):
        from .organize import _validated_dose_grid_geometry
        geometry = _validated_dose_grid_geometry(dose, "RTDOSE")
        first_offset = geometry["offsets"][0]
        self.origin = geometry["origin"] + first_offset * geometry["slice_cos"]
        self.spacing = geometry["spacing"][::-1]
        self.basis = np.column_stack((geometry["row_cos"], geometry["col_cos"], geometry["slice_cos"]))
        self.inverse_basis = np.linalg.inv(self.basis)
        self.offsets = geometry["offsets"] - first_offset
        self.shape = np.array([geometry["cols"], geometry["rows"], geometry["frames"]])
        self.frame_uid = geometry["frame_of_reference"]

    def index(self, points, *, half_voxel_edges=False):
        local = (np.asarray(points) - self.origin) @ self.inverse_basis.T
        xy = local[..., :2] / self.spacing
        z = local[..., 2]
        lower_z, upper_z = self.offsets[[0, -1]]
        tolerance = 1e-9
        if half_voxel_edges:
            # Preserve the NIfTI resampler's half-open edge-voxel support.
            if len(self.offsets) > 1:
                lower_z -= 0.5 * (self.offsets[1] - self.offsets[0])
                upper_z += 0.5 * (self.offsets[-1] - self.offsets[-2])
                z_supported = (z >= lower_z) & (z < upper_z)
            else:
                z_supported = abs(z-lower_z) <= tolerance
            xy_supported = ((xy >= -0.5) & (xy < self.shape[:2]-0.5)).all(axis=-1)
        else:
            z_supported = (z >= lower_z-tolerance) & (z <= upper_z+tolerance)
            xy_supported = ((xy >= -tolerance) & (xy <= self.shape[:2]-1+tolerance)).all(axis=-1)
        support = xy_supported & z_supported
        frames = np.interp(z, self.offsets, np.arange(len(self.offsets), dtype=float))
        index = np.concatenate((xy, frames[..., None]), axis=-1)
        index = np.clip(index, 0., self.shape-1)
        # Do not treat roundoff at a node as dependence on an adjacent node.
        rounded = np.rint(index)
        return np.where(abs(index-rounded) <= tolerance, rounded, index), support


class PatientDoseGrid(DoseGridCoordinates):
    """Interpolate floating-point stored values on the original dose geometry."""

    def __init__(self, dose, *, source_grids=()):
        super().__init__(dose)
        self.original = dose
        self.source_grids = tuple(source_grids)
        if any(grid.frame_uid != self.frame_uid for grid in self.source_grids):
            raise ValueError("COMPOSITE_DOSE_SOURCE_FRAME_MISMATCH")
        values = dose.pixel_array
        if self.shape[2] == 1 and values.ndim == 2:
            values = values[None, ...]
        if values.shape != tuple(self.shape[::-1]):
            raise ValueError("RTDOSE pixel shape disagrees with the grid geometry")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Physical RTDOSE samples must be finite and nonnegative")
        self.values = values.astype(np.float64)
        self.node_support = self._composite_node_support() if self.source_grids else None
        raw_offsets = np.atleast_1d(getattr(dose, "GridFrameOffsetVector", [0.]))
        if len(raw_offsets) > 1 and float(raw_offsets[0]) > float(raw_offsets[-1]):
            self.values = self.values[::-1]

    def _composite_node_support(self):
        x, y = np.meshgrid(np.arange(self.shape[0]), np.arange(self.shape[1]))
        xy = self.origin + x[..., None] * self.spacing[0] * self.basis[:, 0]
        xy = xy + y[..., None] * self.spacing[1] * self.basis[:, 1]
        support = np.ones(tuple(self.shape[::-1]), dtype=bool)
        for frame, offset in enumerate(self.offsets):
            points = xy + offset * self.basis[:, 2]
            for source in self.source_grids:
                support[frame] &= source.index(points)[1]
        return support

    def support_at(self, points, *, half_voxel_edges=False):
        index, support = self.index(points, half_voxel_edges=half_voxel_edges and self.node_support is None)
        if self.node_support is not None:
            # A physical sum requires all contributions at every interpolation
            # node, absent independent proof of a zero contribution.
            low, high = np.floor(index).astype(int), np.ceil(index).astype(int)
            for side in product((False, True), repeat=3):
                x, y, z = np.moveaxis(np.where(side, high, low), -1, 0)
                support &= self.node_support[z, y, x]
        return index, support

    def sample(self, points, *, half_voxel_edges=False):
        index, support = self.support_at(points, half_voxel_edges=half_voxel_edges)
        sampled = map_coordinates(self.values, index[..., ::-1].reshape(-1, 3).T,
                                  order=1, mode="nearest", prefilter=False).reshape(support.shape)
        return np.ma.array(sampled, mask=~support)

    def sample_image(self, reference, *, half_voxel_edges=False):
        """Return interpolated Gy and independent support at CT voxel centres."""
        size = reference.GetSize()
        spacing = np.asarray(reference.GetSpacing())
        direction = np.asarray(reference.GetDirection()).reshape(3, 3)
        origin = np.asarray(reference.GetOrigin())
        xx, yy = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
        values = np.empty(size[::-1], dtype=float)
        support = np.empty(size[::-1], dtype=bool)
        for z in range(size[2]):
            points = np.stack((xx, yy, np.full_like(xx, z)), axis=-1)*spacing
            sampled = self.sample(points @ direction.T + origin, half_voxel_edges=half_voxel_edges)
            values[z] = sampled.filled(0.) * float(self.original.DoseGridScaling)
            support[z] = ~np.ma.getmaskarray(sampled)
        return values, support


class RotatedDoseGrid(PatientDoseGrid):
    def __init__(self, dose, basis, nearest, angle, *, source_grids=()):
        try:
            super().__init__(dose, source_grids=source_grids)
            if (self.shape < 2).any():
                raise ValueError
        except (AttributeError, TypeError, ValueError, IndexError) as exc:
            raise DoseOrientationError("rotated_dose_invalid_grid") from exc
        self.step = np.r_[self.spacing, (self.offsets[-1]-self.offsets[0])/(len(self.offsets)-1)]
        corners = np.array(list(product(*[(0., float(n-1)) for n in self.shape]))) * self.step
        physical = self.origin + corners @ basis.T
        self.lower = physical.min(axis=0)
        self.upper = physical.max(axis=0)
        self.target_spacing = np.abs(nearest) @ np.abs(self.step)
        self.target_shape = np.ceil((self.upper-self.lower)/self.target_spacing).astype(int) + 1
        self.x = self.lower[0] + np.arange(self.target_shape[0])*self.target_spacing[0]
        self.y = self.lower[1] + np.arange(self.target_shape[1])*self.target_spacing[1]
        self.provenance = {
            "method": METHOD, "original_orientation": [float(v) for v in dose.ImageOrientationPatient],
            "rotation_angle_degrees": angle, "rotation_tolerance_degrees": MAX_ROTATION_DEGREES,
            "orthonormal_tolerance": ORTHONORMAL_TOLERANCE,
            "interpolation": "trilinear_in_original_patient_geometry",
            "outside_original_grid": "not_covered_masked",
            "contour_plane_sampling": "native_planar_mask_mapped_to_patient_coordinates",
            "contour_max_projection_tolerance_mm": MAX_CONTOUR_PROJECTION_MM,
            "contour_plane_grouping_tolerance_mm": 0.001,
            "contour_plane_fit": "least_squares_common_normal_from_centred_contour_points",
            "contour_tilt_tolerance_degrees": 1.0,
            "contour_volume_integration": "adjacent_plane_midpoint_slabs_half_interval_end_extension",
            "max_node_displacement_mm": float(np.linalg.norm(corners @ (basis-nearest).T, axis=1).max()),
            "max_node_displacement_definition": "displacement_if_orientation_were_snapped_about_original_origin",
            "original_shape_xyz": self.shape.tolist(),
            "original_origin_xyz_mm": [float(v) for v in dose.ImagePositionPatient],
            "original_step_column_row_frame_mm": self.step.tolist(),
            "original_frame_offsets_mm": [float(v) for v in dose.GridFrameOffsetVector],
            "nifti_sampling": "direct_trilinear_at_ct_voxel_centres",
            "resampled_shape_xyz": self.target_shape.tolist(),
            "resampled_spacing_xyz_mm": self.target_spacing.tolist(),
            "resampled_origin_xyz_mm": self.lower.tolist(),
        }
        # Geometry adapter only: plane sampling below does not allocate a second
        # full 3D volume or quantize interpolated values back into integer DICOM.
        self.dataset = _FloatDose(copy.deepcopy(dose))
        self.dataset.file_meta = copy.deepcopy(dose.file_meta)
        self.dataset.Rows = int(self.target_shape[1])
        self.dataset.Columns = int(self.target_shape[0])
        self.dataset.NumberOfFrames = int(self.target_shape[2])
        self.dataset.ImageOrientationPatient = [1., 0., 0., 0., 1., 0.]
        self.dataset.ImagePositionPatient = self.lower.tolist()
        self.dataset.PixelSpacing = self.target_spacing[[1, 0]].tolist()
        self.dataset.GridFrameOffsetVector = (np.arange(self.target_shape[2])*self.target_spacing[2]).tolist()
        self.dataset._resampled_pixels = self.values

    def plane(self, z):
        xx, yy = np.meshgrid(self.x, self.y)
        return self.sample(np.stack((xx, yy, np.full_like(xx, float(z))), axis=-1))

    def get_dvh(self, structure, roi, calculate_full_volume=True):
        # Both settings sample only supported points. The caller suppresses
        # whole-ROI values for incomplete coverage and labels covered metrics.
        # Import after dvh.py's compatibility initialization for pydicom >= 3.
        from dicompylercore import dicomparser
        from .dvh_support import contour_slab_widths, weighted_plane_dvh
        sampler = self

        class PatientGridParser(dicomparser.DicomParser):
            def GetDoseGrid(self, z=0, threshold=0.5):
                return sampler.plane(z)

            def GetDoseData(self):
                data = self.GetImageData()
                data.update(doseunits=sampler.original.DoseUnits,
                            dosetype=getattr(sampler.original, "DoseType", "PHYSICAL"),
                            dosegridscaling=float(sampler.original.DoseGridScaling),
                            dosemax=float(sampler.values.max()),
                            lut=self.GetPatientToPixelLUT(), x_lut_index=0)
                return data

        rtss = dicomparser.DicomParser(structure)
        item = rtss.GetStructures()[roi]
        # Read the original triples, not dicompyler's rounded Z keys. Some TPS
        # rotate the RTSTRUCT together with RTDOSE even when CT is axial.
        contours = [contour for group in rtss.GetStructureCoordinates(roi).values()
                    for contour in group]
        polygons = []
        for contour in contours:
            points = np.asarray(contour["data"], dtype=float)
            if contour["type"] not in {"CLOSED_PLANAR", "CLOSEDPLANAR_XOR"}:
                raise DoseOrientationError("rotated_dose_unsupported_contour_geometry")
            if len(np.unique(points, axis=0)) < 3:
                continue
            if points.shape[1:] != (3,) or not np.isfinite(points).all():
                raise DoseOrientationError("rotated_dose_invalid_contour_coordinates")
            polygons.append(points)
        if not polygons:
            raise DoseOrientationError("rotated_dose_insufficient_contour_planes_for_volume")
        scatter = np.zeros((3, 3))
        for points in polygons:
            centred = points-points.mean(axis=0)
            scatter += centred.T @ centred
        eigenvalues, axes = np.linalg.eigh(scatter)
        if eigenvalues[1] <= 1e-18:
            raise DoseOrientationError("rotated_dose_degenerate_contour_plane")
        normal = axes[:, 0]
        if normal[2] < 0:
            normal = -normal
        tilt = float(np.degrees(np.arccos(np.clip(normal[2], -1, 1))))
        if tilt > MAX_ROTATION_DEGREES + 1e-8:
            raise DoseOrientationError("rotated_dose_contour_tilt_exceeds_1_degree")
        u = np.array([1., 0., 0.])-normal*normal[0]
        u /= np.linalg.norm(u)
        frame = np.column_stack((u, np.cross(normal, u), normal))
        if np.allclose(frame, np.eye(3), atol=1e-12, rtol=0):
            frame = np.eye(3)
        axial = np.array_equal(frame, np.eye(3))
        local_polygons = []
        for points in polygons:
            local = points @ frame
            z = float(np.mean(local[:, 2]))
            if np.max(np.abs(local[:, 2]-z)) > MAX_CONTOUR_PROJECTION_MM:
                raise DoseOrientationError("rotated_dose_nonparallel_or_nonplanar_contours")
            local_polygons.append((z, local))
        planes = {}
        max_projection = 0.
        for z, local in sorted(local_polygons, key=lambda item: item[0]):
            if planes and abs(z-next(reversed(planes))) <= .001:
                z = next(reversed(planes))
            max_projection = max(max_projection, float(np.max(np.abs(local[:, 2]-z))))
            if max_projection > MAX_CONTOUR_PROJECTION_MM:
                raise DoseOrientationError("rotated_dose_nonparallel_or_nonplanar_contours")
            planes.setdefault(z, []).append({"data": local.tolist(), "type": "CLOSED_PLANAR",
                                            "num_points": len(local)})
        zs = np.array(sorted(planes), dtype=float)
        if len(zs) < 2:
            raise DoseOrientationError("rotated_dose_insufficient_contour_planes_for_volume")
        weights = contour_slab_widths(zs)
        parser = PatientGridParser(self.dataset)
        dd, image = parser.GetDoseData(), parser.GetImageData()
        if axial:
            x, y = self.x, self.y
        else:
            corners = np.array(list(product(*[(0., float(n-1)) for n in self.shape]))) * self.step
            local_corners = (self.origin + corners @ self.basis.T) @ frame
            lower, upper = local_corners.min(axis=0), local_corners.max(axis=0)
            shape = np.ceil((upper-lower)/self.target_spacing).astype(int)+1
            x = lower[0]+np.arange(shape[0])*self.target_spacing[0]
            y = lower[1]+np.arange(shape[1])*self.target_spacing[1]
        dd.update(lut=(x, y), rows=len(y), columns=len(x))
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack((xx.ravel(), yy.ravel()))
        def sampled_planes():
            for z, weight in zip(zs, weights):
                doseplane = (parser.GetDoseGrid(z) if axial else self.sample(
                    np.stack((xx, yy, np.full_like(xx, z)), axis=-1) @ frame.T))
                yield planes[z], doseplane, weight

        result = weighted_plane_dvh(item["name"], sampled_planes(), dd, image, points)
        result.notes.update(contour_plane_tilt_degrees=tilt, contour_max_projection_mm=max_projection)
        return result

    def coverage(self, structure, roi, *, automatic_source=None):
        from .dvh_support import roi_nonvolume_coverage
        disposition = roi_nonvolume_coverage(structure, roi, automatic_source=automatic_source)
        if disposition is not None:
            return disposition
        from .dvh import (_dose_grid_contour_coordinates, _polygon_intersects_box)
        contours, unresolved = _dose_grid_contour_coordinates(structure, roi)
        status = "coverage_unresolved"
        reason = "Missing, malformed or unsupported ROI contour geometry."
        if contours and not unresolved:
            lower, upper = np.zeros(3), self.shape-1
            inside, intersects = True, False
            for contour in contours:
                q = (contour-self.origin) @ self.inverse_basis.T / self.step
                contained = bool(((q >= -1e-9) & (q <= upper+1e-9)).all())
                inside &= contained
                intersects |= contained or _polygon_intersects_box(q, lower, upper, 1e-9)
            status = "fully_covered" if inside else "partial_grid" if intersects else "outside_grid"
            reason = "Original rotated node-box containment; outside samples are masked."
        return {"status": status, "fraction": 1. if status == "fully_covered" else
                0. if status == "outside_grid" else None,
                "method": "original_rotated_grid_contour_polygon_containment",
                "reason": reason}


def add_brachy_metrics(histogram, metrics):
    """Add requested absolute endpoints only on the new rotated-grid path."""
    if metrics is None:
        return None
    from .dvh import _bounded_dose_at_fraction, _sample_endpoint
    metrics["dose_grid_contour_plane_tilt_degrees"] = histogram.notes["contour_plane_tilt_degrees"]
    metrics["dose_grid_contour_max_projection_mm"] = histogram.notes["contour_max_projection_mm"]
    if metrics["dose_summary_basis"] == "unbinned_subcentigray":
        doses, cumulative = histogram.notes['_unbinned_dose_summary']()["distribution"]
        metrics["D90Gy"] = _sample_endpoint(doses, cumulative, .90*cumulative[0])
    else:
        metrics["D90Gy"] = _bounded_dose_at_fraction(
            histogram.bincenters, histogram.counts, .90, histogram.min, histogram.max)
    return metrics


def add_brachy_metrics_from_values(values, voxel_volume, max_dose, metrics):
    if metrics is None or not values.size:
        return metrics
    from .dvh import _bounded_dose_at_fraction
    count = max(int(np.ceil(max_dose/.01))+1, 2)
    edges = np.linspace(0, count*.01, count+1)
    hist, _ = np.histogram(values, bins=edges)
    cumulative = np.cumsum(hist[::-1])[::-1] * voxel_volume
    centers = (edges[:-1]+edges[1:])/2
    metrics["D90Gy"] = _bounded_dose_at_fraction(centers, cumulative, .9, values.min(), values.max())
    return metrics
