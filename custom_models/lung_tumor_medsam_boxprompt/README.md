# lung_tumor_medsam_boxprompt

MedSAM ViT-B foundation-model comparator for R8 thoracic tumor-candidate segmentation.

- Source code: https://github.com/bowang-lab/MedSAM
- Weights: Zenodo record `10.5281/zenodo.10689643`, file `medsam_vit_b.pth`, md5 `3bb6db55bd0c9ca30b61248bca72f8d6`
- License: MedSAM code is Apache-2.0; Zenodo model weights are CC-BY-4.0.
- Architecture: Segment Anything ViT-B medical foundation model.
- Structures: `lung_tumor`
- Deterministic prompt rule: use the existing `lung_tumor_totalseg_lung_nodules/lung_nodules.nii.gz` mask; for every axial slice with prompt voxels, derive a 2D bounding box with a 3-voxel margin; run MedSAM with `multimask_output=False`; clip each slice to the prompt box; confine the 3D mask to the TotalSegmentator bilateral lung mask; keep the largest connected component above 0.5 cm3.
- Training data: MedSAM was trained on broad medical image-mask data across modalities and cancer types. The public sources do not document explicit NSCLC-Interobserver, NSCLC-Radiomics, or RIDER Lung CT overlap, so leakage remains UNKNOWN until separately audited.
- Status: R8 dry-run foundation-model comparator only; not a clinical GTV.

Download:

```bash
custom_models/download_weights.sh lung_tumor_medsam_boxprompt
```
