# Auto-RTSTRUCT geometry diagnosis and repair

The automated radiotherapy structure set (auto-RTSTRUCT) geometry gate rejected the TotalSegmentator masks available in the rebuilt Kopernik cohort because it compared image-frame metadata rather than physical voxel-grid equivalence. The repair accepts the ordinary NIfTI/DICOM axis convention change while preserving fail-closed rejection of a different scan, spacing, voxel grid, or physical extent.

In the read-only cohort snapshot, the repaired gate accepted the masks available from 2 of 122 courses. The baseline gate rejected both courses. The other 120 courses had an existing `Segmentation_TotalSegmentator` directory but no binary mask files, so their mask acceptance could not be observed directly. A planning-NIfTI geometry audit matched 122 of 122 courses and found 0 of 122 accepted by the baseline comparison and 122 of 122 accepted by the repaired comparison. The latter result is an inference about masks derived from those primary NIfTIs, not evidence that new masks were generated.

## Measured case

The reproduced course was `/home/konrad/rtpipeline_campaign/kopernik_bladder_v3/Output/441642/2021-05`. Its `DICOM/CT` directory contained 181 files from exactly 1 series. The series description was `MIEDNICA 3.0 B31f`.

The selected TotalSegmentator directory contained 117 per-region-of-interest (ROI) NIfTI masks. The example mask was `total--adrenal_gland_left.nii.gz`. No DICOM-SEG file was used for this comparison.

SimpleITK reported the following planning computed tomography (CT) geometry.

- Size `(512, 512, 181)`.
- Spacing `(0.9765625, 0.9765625, 3.0)` millimetres.
- Origin `(-249.51171875, -448.51171875, -1042.5)` millimetres.
- Direction `(1, 0, 0, 0, 1, 0, 0, 0, 1)`.

SimpleITK reported the following geometry for the NIfTI mask.

- Size `(512, 512, 181)`.
- Spacing `(0.9765625, 0.9765625, 3.0)` millimetres.
- Origin `(-249.51171875, 50.51171875, -1042.5)` millimetres.
- Direction `(1, 0, 0, 0, -1, 0, 0, 0, 1)`.

The Y-origin difference was 499.0234375 millimetres. The CT therefore runs from approximately -448.5117 to 50.5117 millimetres along Y. The NIfTI mask runs through the same interval in the opposite Y direction because 511 voxel steps separate its two Y endpoints.

The mask and the course primary NIfTI had identical shape, voxel zooms, and affine according to nibabel. This confirms that the disagreement was between the NIfTI-frame mask and the DICOM-frame CT, rather than between the mask and its source NIfTI.

The baseline `_geometry_compatible` implementation required equal direction matrices and origins within 2.0 millimetres. It therefore returned `False` for this mask. The fallback then logged that 117 of 117 masks did not share the planning CT physical space and skipped `RS_auto`.

## Repaired equivalence test

The repair replaces raw direction and origin equality with a physical voxel-grid comparison. It applies to both the combined `seg_img` guard and the per-ROI binary-mask fallback.

First, the gate requires three-dimensional images, finite positive spacing, identical total voxel count, and a matching size under a possible axis reorder. These checks reject missing or malformed geometry and altered grids before resampling.

Second, it forms the three physical step vectors from each direction matrix and its spacing. It matches the candidate vectors to the CT vectors under every signed axis permutation. A signed permutation represents an axis reorder or a direction-sign change, including the NIfTI/DICOM Y-axis convention in the reproduced case. An arbitrary rotation does not match this representation.

Third, it computes the physical bounding box from all eight voxel-grid corners. Both lower and upper corners must agree within the existing `tol_mm` value of 2.0 millimetres. The repair did not widen this tolerance. The extent check rejects a translated or partially overlapping volume even when its array shape is identical.

This combination is stricter than comparing an axis-aligned extent alone. The physical step-vector check requires the same lattice orientation up to axis order and sign. The size and spacing checks bind each voxel count to the corresponding physical step. The extent check then binds the lattice to the same physical location.

A different series with the same array shape remains incompatible when its spacing, step vectors, or physical extent differs. An oblique direction matrix also remains incompatible unless it is the same CT orientation expressed by a signed axis permutation. If geometry cannot be read or any check raises an exception, `_geometry_compatible` returns `False`.

The per-ROI fallback remains whole-build fail-closed. It evaluates every readable binary mask before resampling. If any one mask is incompatible, the fallback list is cleared and no ROI from that fallback is materialised. The `seg_img` path returns without creating `RS_auto` when its combined image fails the same gate.

## Regression tests

The synthetic SimpleITK tests cover the repaired contract without patient data.

- A negated Y direction with the correspondingly shifted origin is accepted.
- A pure signed axis reorder with paired size and spacing is accepted.
- A 50 millimetre translation is rejected.
- Different spacing is rejected.
- A different voxel count is rejected.
- A 10-degree oblique rotation is rejected.
- The combined `seg_img` path accepts the NIfTI convention.
- The binary-mask fallback accepts masks in that convention only when all masks pass.
- The binary-mask fallback refuses to create an output when one mask is incompatible.

Against baseline commit `7d45561c4e58ca1fd6716f31f7490bfbbfb7d9d0`, the delivered focused test file returned 4 failures, 32 passes, and 6 warnings. The failures were `test_geometry_nifti_y_axis_convention_is_compatible`, `test_geometry_signed_axis_permutation_is_compatible`, `test_build_geometry_net_accepts_nifti_axis_convention`, and `test_build_binary_fallback_succeeds_when_all_compatible`. On the repaired source, the same file returned 36 passes and 6 warnings.

Full-suite validation ran from `/umed-projekty/rtpipeline` with Python 3.11.14. It used the short writable temporary directory `/umed-projekty/rtpipeline/.t` so multiprocessing could create AF_UNIX sockets without exceeding the operating system path limit. The exact test command was `/home/konrad/micromamba/envs/rtpipeline/bin/python -m pytest -q tests`. It exited with code 0 and returned 919 passes, 1 skip, and 877 warnings in 143.35 seconds.

The validated SHA-256 hashes were `8543f3e672c7de094ff7ce760c9baac7066ba4f69f4ff75faa3d02a4df0dee46` for `rtpipeline/auto_rtstruct.py` and `69147e1393be3eb0626d87d1c5b12f58877f9e6bfd705ff4fb44c4a37cb61043` for `tests/test_auto_rtstruct_planning_ct_pin.py`. These match the reviewed artifact hashes in the repair packet.

## Cohort-scale calculation

The audit examined `/home/konrad/rtpipeline_campaign/kopernik_bladder_v3/Output` without writing to it or launching a pipeline. It discovered 122 two-level course directories containing both `DICOM/CT` and `Segmentation_TotalSegmentator`.

Only 2 courses had selected directories containing binary TotalSegmentator masks. They contributed 234 mask files, with 117 masks per course. The baseline gate rejected all 2 courses. The repaired gate accepted all 2 courses. Thus, among courses with observed binary masks, the before and after counts were 2 rejected and 0 accepted, followed by 0 rejected and 2 accepted.

The remaining 120 courses had no binary mask files available in their TotalSegmentator directories. They are not counted as accepted or rejected masks. This missing artifact is why the direct mask denominator is 2 rather than 122.

The audit matched each course's planning NIfTI to its CT series using the NIfTI metadata record when available, or a documented description, date, and instance-count fallback for 2 courses whose stored metadata used a different DICOM source directory. The baseline geometry logic rejected all 122 planning-NIfTI geometries. The repaired gate accepted all 122. This supports the expected cohort-wide convention effect if TotalSegmentator masks are regenerated from those primary NIfTIs, but it does not substitute for observing those absent masks.

The complete per-course results and calculation metadata are in `docs/diagnosis-auto-rtstruct-geometry-evidence.json`. That record stores the read-only source root, discovery rule, baseline commit, source hashes, matching method, aggregate counts, and per-course outcomes. The audit implementation is `analysis/diagnose_auto_rtstruct_geometry.py`.

## Interpretation and next action

The available evidence supports a pre-resampling frame-convention defect, not evidence that the masks came from a different scan. The repair addresses that defect without deleting the geometry gate or silently resampling across an unverified frame.

The local regression suite and the read-only cohort audit support the repaired behavior. The audit cannot establish output generation for the 120 courses without masks. Pipeline execution was intentionally not launched, and the evidence directories were not modified. A stopped pipeline should be rerun only under the campaign's existing operational controls, followed by verification that `RS_auto` and the configured custom structures materialise before radiomics.

## Evidence sources

- Source code at `rtpipeline/auto_rtstruct.py` in workspace `/umed-projekty/rtpipeline`.
- Regression tests at `tests/test_auto_rtstruct_planning_ct_pin.py`.
- Baseline commit `7d45561c4e58ca1fd6716f31f7490bfbbfb7d9d0` on branch `fix/radiomics-fail-closed-20260808`.
- Reproduced course under `/home/konrad/rtpipeline_campaign/kopernik_bladder_v3/Output/441642/2021-05`.
- Read-only cohort root under `/home/konrad/rtpipeline_campaign/kopernik_bladder_v3/Output`.
- Baseline rejection log at `/home/konrad/rtpipeline_campaign/kopernik_bladder_v3/Logs/segmentation/441642_2021-05.log`.
- Calculation record at `docs/diagnosis-auto-rtstruct-geometry-evidence.json`.
