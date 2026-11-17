"""Re-export clinical contours from explicit plaintext sources and a reference grid."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path


def reexport_original(*, rtstruct: Path, ct_series: Path, reference_nifti: Path,
                      output_root: Path, overwrite: bool = False) -> dict:
    """Validate explicit source identity/grid and write a current owned export.

    An old input sidecar is neither trusted nor modified. The output carries a
    separately derived reference sidecar bound to the explicit CT series and
    the supplied NIfTI bytes. No source discovery or decryption is performed.
    """
    import numpy as np
    import SimpleITK as sitk
    from rt_utils import RTStructBuilder
    from .course_contract import _ct_provenance
    from .nifti_provenance import nifti_geometry
    from .organize import _export_original_segmentation_from_paths, _strip_nifti_base
    from .original_segmentation import require_current_original_export
    from .rtstruct_geometry import rtstruct_roi_entries, rtstruct_mask_to_image
    from .rtstruct_identity import require_rtstruct_identity

    rtstruct, ct_series, reference_nifti, output_root = (
        Path(path).resolve() for path in (rtstruct, ct_series, reference_nifti, output_root)
    )
    if any(path.name.endswith('.age') for path in (rtstruct, reference_nifti)):
        raise ValueError('REEXPORT_SEALED_SOURCE: supply explicit readable plaintext paths')
    if not rtstruct.is_file() or not reference_nifti.is_file() or not ct_series.is_dir():
        raise ValueError('REEXPORT_SOURCE_UNAVAILABLE: RTSTRUCT, CT series and reference NIfTI are required')
    if any(path.name.endswith('.age') or path.resolve().name.endswith('.age')
           for path in ct_series.rglob('*') if path.is_file()):
        raise ValueError('REEXPORT_SEALED_SOURCE: CT directory contains sealed inputs')
    require_rtstruct_identity(rtstruct)
    ct_metadata = _ct_provenance(ct_series)
    builder = RTStructBuilder.create_from(str(ct_series), str(rtstruct))
    rtstruct_roi_entries(builder)
    reference = sitk.ReadImage(str(reference_nifti))
    shape = (int(builder.series_data[0].Rows), int(builder.series_data[0].Columns),
             len(builder.series_data))
    # Use the production physical mapping; reject interpolation, clipping or an
    # unrelated reference before touching an existing export.
    rtstruct_mask_to_image(np.zeros(shape, dtype=bool), builder.series_data, reference)
    target = output_root / 'Segmentation_Original' / _strip_nifti_base(reference_nifti)
    metadata = {
        **ct_metadata,
        'nifti_geometry': nifti_geometry(reference_nifti),
        'nifti_sha256': hashlib.sha256(reference_nifti.read_bytes()).hexdigest(),
        'nifti_path': str(reference_nifti),
        'source_directory': str(ct_series),
        'modality': 'CT',
        'provenance_method': 'explicit-ct-reference-grid-verification',
    }
    identity = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    sidecar = output_root / 'metadata' / 'original_reexport' / f'{identity}.json'
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    manifest = _export_original_segmentation_from_paths(
        rs_path=rtstruct, primary_nifti=reference_nifti, dicom_ct_dir=ct_series,
        segmentation_original_dir=output_root / 'Segmentation_Original',
        log_root=output_root, overwrite=overwrite, nifti_sidecar_path=sidecar,
    )
    if manifest is None:
        raise ValueError('REEXPORT_NO_MASKS: no nonempty export on the verified reference grid')
    require_current_original_export(target)
    return {'status': 'exported', 'manifest': str(target / 'metadata.json'),
            'schema': manifest['schema'], 'structures': len(manifest['structures'])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='rtpipeline re-export-original',
        description='Re-export ROI-number-bound masks from explicit RTSTRUCT, CT series and reference NIfTI paths.',
    )
    parser.add_argument('--rtstruct', type=Path, required=True)
    parser.add_argument('--ct-series', type=Path, required=True)
    parser.add_argument('--reference-nifti', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true', help='Rebuild an export even when its inputs are current.')
    args = parser.parse_args(argv)
    try:
        result = reexport_original(rtstruct=args.rtstruct, ct_series=args.ct_series,
                                   reference_nifti=args.reference_nifti, output_root=args.output_root,
                                   overwrite=args.overwrite)
    except (OSError, ValueError, RuntimeError) as exc:
        logging.getLogger(__name__).error('Original ROI re-export refused: %s', exc)
        print(json.dumps({'status': 'refused', 'reason': str(exc)}))
        return 2
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
