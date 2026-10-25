# Citing RTpipeline

Cite the RTpipeline version you used and the DOI archived for its release tag.
[`CITATION.cff`](https://github.com/kstawiski/rtpipeline/blob/main/CITATION.cff) holds
the machine-readable metadata; GitHub's "Cite this repository" button formats it.

```bibtex
@software{rtpipeline_3_0_0,
  title = {RTpipeline: Automated Radiotherapy DICOM Processing Pipeline},
  author = {Stawiski, Konrad},
  version = {3.0.0},
  url = {https://github.com/kstawiski/rtpipeline},
  year = {2026}
}
```

When the bytes you ran differ from the release tag, for example a later commit or a
locally modified configuration, report the git commit or the container image digest as
well. The [reproducibility guide](user_guide/reproducibility.md) has methods templates
and lists the configuration files to deposit with a paper.

Also cite the components your analysis used:

| Component | When |
| --- | --- |
| TotalSegmentator (Wasserthal et al., *Radiology: Artificial Intelligence* 2023) | Automatic structures from TotalSegmentator |
| PyRadiomics (van Griethuysen et al., *Cancer Research* 2017) | Radiomic features |
| IBSI (Zwanenburg et al., *Radiology* 2020) | Feature definitions and settings |
| Zwanenburg et al., *Scientific Reports* 2019 | NTCV perturbation robustness |
| RTsegmentator and each segmentation model | Structures imported from an RTsegmentator portal; the model reference is in each `Segmentation_CustomModels/rtseg_<model>/manifest.json` |
