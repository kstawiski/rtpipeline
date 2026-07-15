# _disabled_lung_tumor_aimi_nsclc_rg

BAMF Health AIMI lung CT pretrained nnU-Net v1 model from Zenodo.

- Source: https://zenodo.org/records/8290169
- Weights: `Task775_CT_NSCLC_RG.zip`, md5 `132c34158aab8e09464d65470d8638b0`
- Architecture: nnU-Net v1 `3d_fullres`
- Structures: `lung_tumor`
- Training data: NSCLC-Radiomics and NSCLC-Radiogenomics. This is a serious leakage risk for evaluations involving either source collection.
- Status: DISABLED. This model is not discovered by default because the directory is underscore-prefixed. It is excluded because the `rtpipeline` environment lacks the required nnU-Net v1 `nnUNet_predict` runtime and because its training data overlap the named evaluation collections.

Disabled download target:

```bash
custom_models/download_weights.sh lung_tumor_aimi_nsclc_rg_disabled
```
