"""Compare every explicit-preprocessing diagnostic with the IBSI workbook."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import SimpleITK as sitk
from run_validation import ROOT, OUT, references, read_phantom, ibsi_preprocess, describe, workbook_difference, write_csv


def image_values(image):
    a = sitk.GetArrayFromImage(image)
    out = {f'img_dim_{axis}': image.GetSize()[i] for i, axis in enumerate('xyz')}
    out.update({f'vox_dim_{axis}': image.GetSpacing()[i] for i, axis in enumerate('xyz')})
    out.update(mean_int=float(a.mean(dtype=np.float64)), min_int=float(a.min()), max_int=float(a.max()))
    return out


def roi_values(image, morph, intensity):
    out = {}
    for prefix, mask in [('int_mask', intensity), ('morph_mask', morph)]:
        d = describe(image, mask)
        out.update({f'{prefix}_dim_{axis}': d['size_xyz'][i] for i, axis in enumerate('xyz')})
        out.update({f'{prefix}_bb_dim_{axis}': d['bbox_size_xyz'][i] for i, axis in enumerate('xyz')})
        out.update({f'{prefix}_vox_count': d['voxel_count'], f'{prefix}_mean_int': d['roi_mean'],
                    f'{prefix}_min_int': d['roi_min'], f'{prefix}_max_int': d['roi_max']})
    return out


def main():
    image, mask = read_phantom('ct_radiomics')
    refs = references()
    rows = []
    for configuration in 'ABCDE':
        processed, morph, intensity = ibsi_preprocess(image, mask, configuration)
        stages = {'init_img': image_values(image), 'interp_img': image_values(processed),
                  'init_roi': roi_values(image, mask, mask), 'interp_roi': roi_values(processed, morph, morph),
                  'reseg_roi': roi_values(processed, morph, intensity)}
        for ref in refs[configuration]:
            if not ref['family'].startswith('Diagnostics'):
                continue
            tag = ref['tag']
            stage = next(k for k in stages if tag.endswith('_' + k))
            value = stages[stage][tag[:-len(stage)-1]]
            diff = workbook_difference(ref['reference'], value)
            rows.append({'configuration': configuration, 'stage': stage, 'ibsi_tag': tag,
                         'reference': ref['reference'], 'tolerance': ref['tolerance'], 'computed': value,
                         'absolute_error': abs(value-ref['reference']), 'workbook_difference': diff,
                         'status': 'within tolerance' if diff <= ref['tolerance'] else 'outside tolerance'})
    write_csv(OUT / 'preprocessing_diagnostics.csv', rows)
    failures = [r for r in rows if r['status'] == 'outside tolerance']
    print(f'IBSI preprocessing diagnostics: {len(rows)-len(failures)}/{len(rows)} within tolerance')
    for row in failures:
        print(row)


if __name__ == '__main__':
    main()
