# lung_tumor_totalseg_lung_nodules

TotalSegmentator `lung_nodules` task integrated as an rtpipeline custom model.

- Source: https://github.com/wasserth/TotalSegmentator
- Weights: downloaded by `totalseg_download_weights -t lung_nodules`
- License: TotalSegmentator code is Apache-2.0; confirm the downloaded task weights metadata before external redistribution.
- Structures: `lung_nodules`
- Status: public deterministic command-line model; R8 dry-run comparator only, not a clinical GTV.

Download:

```bash
custom_models/download_weights.sh lung_tumor_totalseg_lung_nodules
```
