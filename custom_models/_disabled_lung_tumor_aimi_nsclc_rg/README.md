# _disabled_lung_tumor_aimi_nsclc_rg

BAMF Health AIMI lung CT pretrained nnU-Net v1 model from Zenodo.

- Source: https://zenodo.org/records/8290169
- Weights: `Task775_CT_NSCLC_RG.zip`, md5 `132c34158aab8e09464d65470d8638b0`
- Architecture: nnU-Net v1 `3d_fullres`
- Structures: `lung_tumor`
- Training data: NSCLC-Radiomics and NSCLC-Radiogenomics. This is a serious leakage risk for any analysis involving NSCLC-Radiomics-derived claims and must be flagged in Phase 3.
- Status: DISABLED for R8 Phase 1c. This model is not discovered by default because the directory is underscore-prefixed. It was dropped from the active set because the project `rtpipeline` env lacks the nnU-Net v1 `nnUNet_predict` runtime and because the training data explicitly include NSCLC-Radiomics/NSCLC-Radiogenomics, creating HIGH leakage risk for NSCLC benchmark analyses.

Disabled download target:

```bash
custom_models/download_weights.sh lung_tumor_aimi_nsclc_rg_disabled
```
