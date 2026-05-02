# lung_tumor_aimi_nsclc_rg

BAMF Health AIMI lung CT pretrained nnU-Net v1 model from Zenodo.

- Source: https://zenodo.org/records/8290169
- Weights: `Task775_CT_NSCLC_RG.zip`, md5 `132c34158aab8e09464d65470d8638b0`
- Architecture: nnU-Net v1 `3d_fullres`
- Structures: `lung_tumor`
- Training data: NSCLC-Radiomics and NSCLC-Radiogenomics. This is a serious leakage risk for any analysis involving NSCLC-Radiomics-derived claims and must be flagged in Phase 3.
- Status: integrated but requires an nnU-Net v1 runtime; the project `rtpipeline` env currently provides nnU-Net v2 only.

Download:

```bash
custom_models/download_weights.sh lung_tumor_aimi_nsclc_rg
```
