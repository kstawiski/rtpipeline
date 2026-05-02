# lung_tumor_pancancer_lung

Recent nnU-Net v2 ResEnc lung-cancer model from `KS987/PanCancerSeg-Specialized-weights`.

- Source: https://huggingface.co/KS987/PanCancerSeg-Specialized-weights
- Weights: `Dataset105_Lung/.../checkpoint_best.pth`
- License: Apache-2.0 per Hugging Face model card
- Structures: `lung_tumor` (binary tumor/primary lesion mask)
- Training data: CVPR 2026 FLARE Task 1 Pan-cancer Segmentation dataset; overlap with NSCLC-Interobserver/RIDER was not documented in the model card and must be rechecked before manuscript claims.
- Status: R8 dry-run comparator; output must pass downstream geometry and plausibility gates.

Download:

```bash
custom_models/download_weights.sh lung_tumor_pancancer_lung
```
