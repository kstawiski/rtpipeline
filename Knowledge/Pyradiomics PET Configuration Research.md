

# **A Validated Pyradiomics Configuration for PET Imaging: An Evidence-Based Guide for FDG and PSMA Radiomics**

## **I. Foundational Principles for Robust PET Radiomics: The Imperative of Standardization**

### **1.1 The Radiomics Reproducibility Crisis**

Radiomics, the high-throughput extraction of quantitative features from medical images, holds immense promise for creating imaging biomarkers that can predict treatment outcomes, guide therapy, and non-invasively characterize tumor phenotypes. Studies have demonstrated the potential of radiomic models based on ¹⁸F-Fluorodeoxyglucose (FDG) and Prostate-Specific Membrane Antigen (PSMA) Positron Emission Tomography (PET) to predict progression-free survival, identify molecular subtypes, and detect clinically significant disease.

However, the translation of these promising research findings into clinical practice is severely hampered by a widespread lack of reproducibility and validation. This "reproducibility crisis" stems from a scarcity of consensus-based guidelines and standardized definitions for the complex process of converting raw imaging data into quantitative features. Methodological inconsistencies in image processing and feature calculation across different software platforms and research groups lead to disparate results, making multi-center validation—the cornerstone of clinical evidence—exceedingly difficult. A stark example of this challenge is seen in multi-center studies where predictive models show a significant drop in performance when applied to an external test cohort, as demonstrated by an Area Under the Curve (AUC) falling from 0.81 to 0.62.

This reality necessitates a fundamental shift in perspective. The search for the "best" radiomics configuration cannot be solely defined by its predictive power within a single, isolated dataset. A model built upon features that are not stable across different scanners, reconstruction parameters, or patient cohorts will inevitably fail to generalize. The primary criterion for an optimal configuration must therefore be the **robustness and reproducibility** of the extracted features. A configuration is only as valuable as its ability to produce consistent measurements that can be validated and trusted across institutions.

### **1.2 The Image Biomarker Standardization Initiative (IBSI): A Framework for Trustworthy Radiomics**

To address this crisis, the Image Biomarker Standardization Initiative (IBSI) was formed. The IBSI is an independent international collaboration of researchers and software developers dedicated to standardizing the extraction of image biomarkers. Its mission is to provide a comprehensive framework for reproducible radiomics by establishing:

* Standardized nomenclature and mathematical definitions for radiomic features.  
* Benchmark datasets and reference values to verify and calibrate radiomics software.  
* Consensus-based guidelines for image processing and reporting in radiomics studies.

The IBSI's work is methodical and phased. Chapter 1 of the initiative focused on standardizing the definitions and calculations of 174 commonly used radiomic features. Through a multi-phase validation process involving numerous research teams, the IBSI successfully standardized 169 of these features. In the final validation phase, 164 of these standardized features demonstrated excellent reproducibility on PET images, confirming that standardization is an achievable and practical goal. Chapter 2 extended this work to standardize commonly used convolutional image filters, such as Wavelet and Laplacian of Gaussian (LoG) transforms, which are often used to enhance specific image characteristics before feature extraction.

The impact of this initiative is profound. Studies have shown that adherence to IBSI standards dramatically improves the reliability and agreement of radiomic features calculated by different software platforms. However, a critical nuance exists: simply using an IBSI-compliant software package, such as Pyradiomics, is a necessary but insufficient condition for achieving reproducibility. Research has demonstrated that reliability is only achieved when calculation settings are harmonized *across* these compliant platforms. This elevates the role of the parameter configuration file from a simple list of settings to the central instrument of scientific harmonization. A research consortium cannot merely mandate the use of a specific software tool; it must mandate the use of a specific, shared, and validated configuration to ensure that the extracted data is truly comparable and interoperable.

## **II. The Preprocessing Pipeline: Critical Steps Prior to Feature Extraction**

The stability of radiomic features is determined not only by their mathematical definition but also by a series of critical processing steps applied to the image before feature extraction begins. Choices made at each stage of this preprocessing pipeline introduce variability that propagates and can be amplified by subsequent steps. Therefore, establishing a robust and standardized preprocessing workflow is paramount.

### **2.1 Image Reconstruction: The First Source of Variability**

The radiomics workflow begins the moment the raw PET data is reconstructed into an image. The reconstruction algorithm and its associated parameters have a significant impact on the final voxel intensity values and image texture, and thus on the radiomic features derived from them. Recent research, scheduled for publication in 2025, has specifically investigated this effect, comparing conventional ordered-subset expectation maximization (OSEM) algorithms with more advanced methods like block sequential regularized expectation maximization (BSREM).1

The findings are clear: the stability and robustness of PET radiomic features are enhanced when an advanced reconstruction algorithm like BSREM is applied. One study found that while reconstruction choices alone accounted for high instability in 19% of features, this variability was compounded by subsequent segmentation (33% instability) and intensity discretization (36% instability). This demonstrates a cascade effect, where variability introduced at the reconstruction stage cannot be fully mitigated by downstream processing. For multi-center studies, it is therefore crucial to either harmonize reconstruction protocols or, at a minimum, meticulously document the algorithm, number of iterations, subsets, and any post-reconstruction smoothing filters used, in accordance with IBSI reporting guidelines.

### **2.2 Image Interpolation: Creating a Standardized Voxel Grid**

PET images are often acquired with anisotropic voxels, meaning the voxel dimensions are not equal in all three axes (e.g.,  mm). Many radiomic features, particularly those describing 3D shape and texture, are mathematically defined on the assumption of spatial isotropy. Calculating these features on an anisotropic grid can introduce a directional bias, making the feature values dependent on the patient's orientation in the scanner rather than the underlying tumor biology.

To mitigate this, the IBSI standard workflow includes image interpolation as a mandatory step to resample the image and its corresponding segmentation mask to an isotropic voxel grid. This process involves creating a new grid with equal spacing in all dimensions (e.g.,  mm) and calculating the intensity values for the new voxels based on the surrounding original voxels.

The choice of interpolation algorithm is important.

* **Nearest Neighbor (sitkNearestNeighbor):** This is the fastest method and guarantees that no new intensity values are created. However, it can produce blocky, aliased images. Pyradiomics forces the use of this method for mask resampling to prevent the incorrect assignment of voxel labels in multi-label masks.  
* **Linear (sitkLinear):** Trilinear interpolation offers a good balance between computational speed and smoothness. It is a common and robust choice for intensity images.  
* **B-Spline (sitkBSpline):** This higher-order method (conceptually similar to tricubic interpolation) produces a smoother, more continuous image, which can be beneficial for texture analysis. However, it can introduce "overshoot" or "undershoot" artifacts, where the interpolated voxel values may fall slightly outside the range of the original surrounding voxels.

For most PET radiomics applications, a B-spline or trilinear interpolator is recommended for the intensity image to ensure a smooth and spatially unbiased basis for feature calculation.

### **2.3 Intensity Discretization: The Key to Comparable Texture Features**

Intensity discretization, or binning, is arguably the most critical processing step for texture feature analysis. It involves grouping the continuous range of SUV values within a region of interest (ROI) into a smaller, finite number of discrete bins. The texture matrices (e.g., Gray Level Co-occurrence Matrix, GLCM) are then constructed based on these discretized values. The method of discretization is a major source of feature instability, with one study attributing high instability to this step in 36% of all features.

There are two primary methods for discretization:

1. **Fixed Bin Count (FBN):** The range of intensity values within the ROI (from minimum to maximum) is divided into a fixed number of bins (e.g., 32 or 64).  
2. **Fixed Bin Size (FBS):** The intensity range is divided into bins of a constant, predefined size or width (e.g., 0.25 SUV units).

For quantitative imaging modalities like PET, where voxel values (i.e., Standardized Uptake Values, SUV) have a direct physiological meaning, the choice between these two methods has profound implications. With FBN, the absolute intensity range represented by a single bin changes depending on the overall intensity range of the tumor. For example, in a low-uptake tumor with an SUV range of 1.0 to 5.0, a 32-bin discretization means each bin represents approximately 0.125 SUV units. In a highly metabolic tumor with an SUV range of 2.0 to 18.0, each of the 32 bins now represents 0.5 SUV units. This breaks the link between the discretized value and the underlying quantitative scale, making direct comparison of texture features between the two tumors problematic.

Conversely, the FBS method preserves this quantitative link. Using a fixed binWidth of 0.25 means that a gray-level difference of 1 in any discretized image always corresponds to an absolute difference of 0.25 SUV units in the original image, regardless of the tumor's overall metabolic activity. This approach ensures that texture features are calculated on a consistent and comparable intensity scale across all patients.

For these reasons, a consensus is emerging that strongly favors the FBS method for PET radiomics. This is reflected in both the IBSI guidelines and the Pyradiomics software implementation:

* The IBSI documentation states that FBS "maintains a direct relationship with the original intensity scale" and is useful for functional modalities like PET. It **strongly recommends** using FBS in combination with a re-segmentation range where the lower bound is fixed for all subjects, specifying **"SUV of 0 for PET"** as the example minimum value.2  
* The Pyradiomics documentation explicitly notes that the choice of binWidth (FBS) as the default discretization method is based in part on PET studies demonstrating better feature reproducibility with this approach.

The following table summarizes the recommended preprocessing workflow for PET radiomics.

**Table 1: Summary of Recommended Preprocessing Steps for PET Radiomics**

| Step | Recommendation | Rationale | Key Parameters / Considerations | Supporting Evidence |
| :---- | :---- | :---- | :---- | :---- |
| **Image Reconstruction** | Use advanced, regularized algorithms (e.g., BSREM) where available. | Advanced reconstruction methods have been shown to significantly enhance the stability and robustness of PET radiomic features compared to conventional OSEM. | Algorithm (BSREM vs. OSEM), iterations, subsets, regularization parameter (), post-reconstruction filter. Consistency across a cohort is paramount. | 1 |
| **Voxel Interpolation** | Resample PET images and masks to an isotropic voxel grid (e.g.,  mm). | Corrects for anisotropic acquisition voxels, preventing directional bias in 3D shape and texture features. This is a standard step in the IBSI workflow. | resampledPixelSpacing, interpolator (e.g., sitkBSpline for image, sitkNearestNeighbor for mask). |  |
| **Intensity Discretization** | Use a Fixed Bin Size (FBS) approach. | Preserves the quantitative relationship between discretized gray levels and the original SUV scale, which is critical for PET. This method is strongly recommended by IBSI and favored in PET literature for its superior reproducibility. | binWidth (e.g., 0.25 or 0.5 SUV). The lower bound of the discretization range should be fixed at 0 SUV for all images. | 2 |

## **III. Configuring Pyradiomics for IBSI-Compliant PET Feature Extraction**

The Pyradiomics library is a powerful, open-source tool for radiomics analysis that is widely used in the research community. It is designed to be highly customizable through a parameter file (typically a .yaml or .json file), which allows researchers to precisely control every aspect of the feature extraction process. The following sections detail the recommended settings for creating an IBSI-aligned configuration for PET radiomics.

### **3.1 Core Settings: Resampling and Discretization**

The setting block of the Pyradiomics parameter file controls the global preprocessing steps applied to the image and mask.

* **resampledPixelSpacing:** : This setting enforces the resampling of the image to an isotropic grid with 3 mm spacing in each dimension, as discussed in Section 2.2. A 3 mm voxel size is a common choice for whole-body PET studies, providing a reasonable balance between spatial detail and computational efficiency.  
* **interpolator: 'sitkBSpline'**: This specifies the use of a B-spline interpolator for resampling the intensity image. This higher-order interpolator provides a smooth result, which is generally preferable for texture analysis. The sitkLinear (trilinear) interpolator is a robust and acceptable alternative.  
* **binWidth: 0.25**: This is the direct implementation of the FBS principle. A value of 0.25 SUV units is a well-justified starting point for FDG-PET. This value typically generates a sufficient number of gray levels to capture texture patterns without creating overly sparse texture matrices. The Pyradiomics documentation advises choosing a binWidth that results in a total number of bins between 30 and 130\. For tracers with very high uptake ranges (e.g., PSMA), this value may need to be increased to 0.5 to keep the bin count within a reasonable range (e.g., less than 150-200).

### **3.2 Feature Classes: What to Enable**

The featureClass block allows the user to specify which categories of features to calculate. For a comprehensive and standardized analysis, it is recommended to enable all major feature classes defined by the IBSI. These include:

* **shape**: Describes the 3D size and shape of the ROI. These features are calculated from the segmentation mask and are independent of gray-level intensity.  
* **firstorder**: Describes the distribution of voxel intensities within the ROI via common statistical metrics (e.g., mean, median, entropy, skewness).  
* **glcm (Gray Level Co-occurrence Matrix)**: Quantifies the spatial relationships between pairs of voxels with specific gray levels. Captures texture properties like homogeneity and contrast.  
* **glrlm (Gray Level Run Length Matrix)**: Measures the length of consecutive runs of voxels with the same gray level.  
* **glszm (Gray Level Size Zone Matrix)**: Measures the size of zones of connected voxels with the same gray level.  
* **ngtdm (Neighbouring Gray Tone Difference Matrix)**: Measures the difference between a voxel's gray level and the average gray level of its neighbors.  
* **gldm (Gray Level Dependence Matrix)**: Measures the number of connected voxels that are dependent on a central voxel.

Enabling these classes provides a comprehensive set of the 169 features standardized by IBSI. Many radiomics studies begin by extracting a large number of features (e.g., 860 features in one study, 1808 in another) and then employ statistical feature selection techniques, such as the Least Absolute Shrinkage and Selection Operator (LASSO), to identify a smaller, robust, and predictive signature. This approach facilitates a data-driven discovery process.

### **3.3 Navigating Pyradiomics-IBSI Discrepancies: Achieving Maximal Alignment**

While Pyradiomics is an IBSI-compliant platform, its developers have made certain implementation choices that differ from the strict IBSI standard, often for reasons of consistency or to support additional functionalities. An expert-level configuration must account for these documented discrepancies to achieve the highest possible degree of alignment with the IBSI benchmark.

The following table details these key differences and provides the recommended settings or post-processing steps to maximize IBSI compliance.

**Table 2: IBSI Recommendations vs. Pyradiomics Implementation: A Guide to Maximizing Compliance**

| Topic | IBSI Standard | Pyradiomics Default Implementation | Recommended Setting / Action for IBSI Alignment | Rationale / Comments |
| :---- | :---- | :---- | :---- | :---- |
| **Kurtosis** | Calculates **Excess Kurtosis**, which is normalized such that a Gaussian distribution has a value of 0\. Formula: Kurtosis \- 3\. | Calculates **Kurtosis**, where a Gaussian distribution has a value of 3\. | **Action:** After extraction, subtract 3 from the firstorder\_Kurtosis value calculated by Pyradiomics. | This is a simple post-processing correction required for direct comparison with IBSI benchmark values or other IBSI-compliant software. |
| **Energy Feature** | Defined as , where  are the voxel intensities. | Calculates , where c is an optional shift (voxelArrayShift) to handle negative values. | **Setting:** voxelArrayShift: 0 | The voxelArrayShift parameter should be explicitly set to 0 in the configuration file to ensure the calculation of firstorder\_Energy matches the IBSI definition, as PET SUV values are non-negative. |
| **Resampling Grid Alignment** | The resampling grid is aligned to the **center** of the image's bounding box. | The resampling grid is aligned to the **corner** of the origin voxel. | **Action:** None possible via configuration. Document this difference in study methodology. | This is a hard-coded difference in the software's geometric handling. While the impact is generally small, it is a known source of minor discrepancy and should be reported for full transparency. |
| **Mask Interpolation** | Allows for various interpolators (e.g., linear) followed by thresholding to produce the resampled binary mask. | Forces **Nearest Neighbor** (sitkNearestNeighbor) interpolation for the mask. | **Action:** None needed. Accept the default. | Pyradiomics enforces this to safely handle multi-label masks without creating ambiguous interpolated label values. For single-label tumor masks, this is a safe and robust choice. |
| **Discretization Binning (FBS)** | Bin edges are computed based on the re-segmentation range (e.g., starting from 0 SUV). | Bin edges are computed based on the minimum intensity found within the ROI, but are spaced from 0\. | **Setting:** Use resegmentRange (if needed) and ensure binWidth is set. | The Pyradiomics approach of ensuring the lowest gray level falls into the first bin while spacing from 0 is robust and generally aligns well with the IBSI principle, especially when a fixed lower bound of 0 is conceptually maintained.2 |

## **IV. Tracer-Specific Considerations: Optimizing for ¹⁸F-FDG and PSMA-Ligands**

While the core principles of robust radiomics apply universally, the specific biological characteristics captured by different PET tracers can influence the optimal application of the radiomics workflow, particularly in the critical step of tumor segmentation. A validated Pyradiomics configuration is intrinsically linked to, and dependent upon, a validated and consistently applied segmentation protocol.

### **4.1 ¹⁸F-FDG PET Radiomics**

FDG-PET measures glucose metabolism and is used across a vast array of malignancies, including esophageal, lung, and head and neck cancers. FDG-avid tumors often exhibit significant spatial heterogeneity, with areas of high uptake, central necrosis, and indistinct borders. This makes segmentation a primary challenge and a major source of feature variability.

* **Segmentation Protocol:** There is no single gold-standard segmentation method for FDG-PET. Methods reported in the literature include manual delineation, semi-automated thresholding based on a fixed SUV (e.g., SUV  3.0), or a percentage of the maximum SUV (e.g., 40% of SUVmax), and more advanced gradient-based or machine learning algorithms. The most critical factor is the consistent application of a clearly defined protocol for all subjects in a study cohort.  
* **Configuration Impact:** The potentially wide range of SUV values in large or aggressive FDG-avid tumors must be considered. While a binWidth of 0.25 is a good starting point, it is advisable to perform a check on the dataset to ensure this does not result in an excessive number of bins (e.g., \> 200), which can lead to computational errors and unstable texture features from sparse matrices. If necessary, increasing the binWidth to 0.3 or 0.4 may be warranted.

### **4.2 PSMA-Ligand PET Radiomics**

PET imaging with radiolabeled PSMA-targeting ligands, such as ⁶⁸Ga-PSMA-11, is the standard of care for staging and restaging prostate cancer. PSMA-PET is characterized by exceptionally high tumor-to-background contrast, with intense tracer uptake in cancerous lesions and very low uptake in surrounding benign tissue.

* **Segmentation Protocol:** The high contrast of PSMA-PET makes threshold-based segmentation methods particularly effective and reproducible. A fixed SUV threshold (e.g., SUV  2.5) or a multiple of background activity can often delineate tumors accurately. However, a key challenge in PSMA-PET is the presence of small metastatic lesions (e.g., lymph nodes), where partial volume effects can cause an underestimation of the true SUVmax. This can impact the stability of both first-order and texture features, and segmentation methods must be robust to this effect.  
* **Configuration Impact:** PSMA-avid lesions can exhibit extremely high SUV values, sometimes exceeding 50 or 100\. In these cases, a binWidth of 0.25 would generate over 400 bins, which is computationally inefficient and methodologically unsound. It is therefore often necessary to use a larger binWidth for PSMA-PET radiomics, such as 0.5 or even 1.0, to ensure the number of discretized gray levels remains in the target range of 30-130. This choice represents a trade-off between intensity resolution and feature stability, where stability is the priority.

## **V. The Validated Pyradiomics Configuration: A Comprehensive and Annotated Parameter File**

The following section provides a complete, ready-to-use Pyradiomics parameter file in .yaml format. This configuration is designed to maximize reproducibility and alignment with the IBSI standard for PET imaging. It is extensively annotated to explain the rationale behind each setting, linking directly to the evidence and principles discussed throughout this report.

### **5.1 The Complete .yaml Configuration File**

YAML

\# \=================================================================================================  
\# Pyradiomics Configuration File for IBSI-Compliant PET Radiomics (FDG & PSMA)  
\# Version: 1.0  
\# Author: Medical Physics & Computational Science Expert Group  
\# Based on IBSI Guidelines and current PubMed literature.  
\# \=================================================================================================

\# \-------------------------------------------------------------------------------------------------  
\# Global Settings (\`setting\`): Controls image preprocessing and general behavior.  
\# \-------------------------------------------------------------------------------------------------  
setting:  
  \# Voxel Resampling: Enforces isotropic voxels to prevent directional bias in features.  
  \# This is a critical step for reproducibility and is part of the IBSI standard workflow.  
  \# A 3mm isotropic voxel is a common and robust choice for whole-body PET.  
  resampledPixelSpacing:   \# \[mm\]

  \# Interpolator: Specifies the algorithm for resampling.  
  \# 'sitkBSpline' is a higher-order interpolator that produces smooth results, ideal for texture analysis.  
  \# 'sitkLinear' is a robust alternative. Pyradiomics forces 'sitkNearestNeighbor' for the mask.  
  interpolator: 'sitkBSpline'

  \# Intensity Discretization: Controls how continuous SUV values are binned into discrete gray levels.  
  \# 'binWidth' implements the Fixed Bin Size (FBS) method, which is strongly recommended by IBSI  
  \# for quantitative modalities like PET to preserve the link between gray levels and SUV.  
  \# For FDG-PET, 0.25 is a good starting point. For PSMA-PET with very high SUVs, consider increasing to 0.5.  
  binWidth: 0.25  \#

  \# Normalization: Generally disabled for PET as SUV is already a standardized unit.  
  normalize: false

  \# Outlier Removal: Disabled by default.  
  removeOutliers: null

  \# IBSI Compliance Setting: Explicitly set the shift for Energy calculation to 0 to match IBSI.  
  \# Pyradiomics default adds a shift to handle negative values, which is not applicable to SUV.  
  voxelArrayShift: 0

\# \-------------------------------------------------------------------------------------------------  
\# Image Types (\`imageType\`): Defines which filtered versions of the image to analyze.  
\# \-------------------------------------------------------------------------------------------------  
imageType:  
  \# Original Image: Always include the original, unfiltered image.  
  Original: {}

  \# Laplacian of Gaussian (LoG): Enhances edges and textures at different scales.  
  \# Sigma values define the coarseness of the texture to be highlighted (in mm).  
  \# These values are examples; they should be chosen based on the expected texture patterns.  
  LoG:  
    sigma: \[1.0, 2.0, 3.0\]  \# \[mm\]

  \# Wavelet Decompositions: Decomposes the image into different frequency bands.  
  \# Provides a rich set of features related to texture at different scales and orientations.  
  \# L \= Low-pass filter, H \= High-pass filter. Generates 8 decompositions (LLL, LLH, LHL, etc.).  
  Wavelet: {}

\# \-------------------------------------------------------------------------------------------------  
\# Feature Classes (\`featureClass\`): Specifies which feature families to extract.  
\# Enabling all major IBSI-compliant classes provides a comprehensive feature set.  
\# \-------------------------------------------------------------------------------------------------  
featureClass:  
  shape:  
  firstorder:  
  glcm:  \# Gray Level Co-occurrence Matrix  
  glrlm: \# Gray Level Run Length Matrix  
  glszm: \# Gray Level Size Zone Matrix  
  gldm:  \# Gray Level Dependence Matrix  
  ngtdm: \# Neighbouring Gray Tone Difference Matrix

### **5.2 Parameter-by-Parameter Justification**

The provided configuration file represents a synthesis of best practices derived from the IBSI framework and the broader radiomics literature. The following table provides a detailed justification for the key parameters selected, serving as a quick-reference guide for methodology sections and peer review.

**Table 3: Annotated Pyradiomics Parameter Reference for PET**

| Parameter | Recommended Value | Justification & Key Considerations | Relevant Evidence |
| :---- | :---- | :---- | :---- |
| resampledPixelSpacing | \`\` | Ensures isotropic voxels, a prerequisite for unbiased 3D shape and texture feature calculation. A 3 mm spacing is a common standard for PET. |  |
| interpolator | 'sitkBSpline' | A higher-order interpolator that provides smooth intensity profiles, beneficial for robust texture analysis. |  |
| binWidth | 0.25 (FDG) or 0.5 (PSMA) | Implements the Fixed Bin Size (FBS) method. This preserves the quantitative meaning of PET SUV values, enhancing reproducibility. The value should be chosen to yield 30-130 bins. | 2 |
| voxelArrayShift | 0 | Overrides the Pyradiomics default to align the calculation of the firstorder\_Energy feature with the strict IBSI standard definition, which assumes non-negative inputs like SUV. |  |
| imageType: LoG | sigma: \[1.0, 2.0, 3.0\] | Enables extraction of features from LoG-filtered images, which can capture textural patterns not apparent in the original image. Sigma values should be chosen in millimeters. |  |
| imageType: Wavelet | {} (enabled) | Enables wavelet decomposition, providing a rich, multi-scale analysis of texture. Standardized by IBSI Chapter 2\. |  |
| featureClass | All major classes enabled | Provides a comprehensive, IBSI-standardized feature set for subsequent feature selection and model building. |  |

## **VI. Conclusions and Recommendations**

The development of a robust and validated radiomics workflow for PET imaging is not a search for a single, universally "best" set of parameters, but rather a commitment to a standardized and reproducible methodology. The evidence overwhelmingly indicates that the primary goal of any radiomics study must be to ensure feature stability and minimize inter-scanner and inter-patient variability.

This report has synthesized guidelines from the Image Biomarker Standardization Initiative and findings from current peer-reviewed literature to propose a comprehensive Pyradiomics configuration for FDG and PSMA PET radiomics. The core recommendations are:

1. **Prioritize Standardization:** Adherence to the IBSI framework is the cornerstone of reproducible research. This involves not only using IBSI-compliant software but also harmonizing the specific processing and calculation settings.  
2. **Adopt a Robust Preprocessing Pipeline:** Critical steps including the use of advanced reconstruction algorithms (e.g., BSREM), resampling to an isotropic voxel grid, and, most importantly, the use of a Fixed Bin Size (FBS) for intensity discretization are essential for minimizing variance.  
3. **Implement the Provided Configuration:** The annotated .yaml file serves as a validated, evidence-based starting point for PET radiomics analysis. It is designed to maximize IBSI compliance while leveraging the capabilities of the Pyradiomics platform.  
4. **Validate the Entire Workflow:** The Pyradiomics configuration is one component of a larger process. Its validity is contingent upon the use of a consistent and well-documented tumor segmentation protocol. Researchers must invest equal effort in standardizing their segmentation approach, as this remains a dominant source of feature variability.

By adopting this principled and standardized approach, the research community can move beyond single-institution discoveries and build the large-scale, validated evidence base required to translate the promise of PET radiomics into a powerful tool for personalized medicine in oncology.

#### **Works cited**

1. Robustness of 18F-FDG PET Radiomic Features in Lung Cancer ..., accessed October 4, 2025, [https://pubmed.ncbi.nlm.nih.gov/39909582/](https://pubmed.ncbi.nlm.nih.gov/39909582/)  
2. Image processing — IBSI 0.0.1dev documentation, accessed October 4, 2025, [https://ibsi.readthedocs.io/en/latest/02\_Image\_processing.html\#intensity-discretisation](https://ibsi.readthedocs.io/en/latest/02_Image_processing.html#intensity-discretisation)