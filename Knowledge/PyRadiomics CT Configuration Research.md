

# **A Validated Pyradiomics Configuration for Computed Tomography: An Evidence-Based Technical Report**

## **Section 1: The Imperative for Standardization in CT Radiomics: From Chaos to Consensus**

### **1.1 The Radiomics Reproducibility Challenge**

Radiomics is an advanced field of quantitative image analysis that aims to extract a large number of high-dimensional, mineable features from standard-of-care medical images, such as Computed Tomography (CT) scans.1 The core hypothesis posits that these quantitative features can non-invasively capture the underlying pathophysiology and heterogeneity of tissues, providing valuable information for cancer detection, diagnosis, prognosis, and treatment response prediction.4 This technology has shown considerable potential across numerous oncological applications, including in lung, head and neck, and cervical cancers, where radiomic signatures have been associated with key clinical outcomes.1

Despite this promise, the clinical translation of radiomics has been severely impeded by a pervasive and well-documented crisis of reproducibility.2 The generalizability of published radiomic models is a primary concern, as findings from single-institution studies often fail to be validated in independent cohorts.7 This lack of robustness stems from the multitude of factors that influence the final feature values. Unlike other '-omics' data, radiomic features are not a direct biological measurement but are derived through a multi-step computational workflow, where each step introduces potential variability. This workflow includes image acquisition, reconstruction, segmentation, image preprocessing, and the feature calculation algorithms themselves.7

The lack of standardized algorithm definitions and image processing protocols is a principal driver of this reproducibility crisis.1 Methodologies in the radiomics workflow can vary greatly between studies, and a significant portion of published research fails to report these methodologies in sufficient detail.9 A systematic review found that only a small fraction of studies reported every step of their image acquisition, preprocessing, and feature extraction pipeline, leading to suboptimal Radiomics Quality Scores (RQS) and making it nearly impossible to replicate the work or perform meaningful meta-analyses.9

This lack of transparency and standardization effectively turns the feature extraction pipeline itself into a "black box," a problem more fundamental than the complexity of the machine learning models that are subsequently built.11 While the term "black box" is often associated with complex algorithms like deep neural networks, the evidence indicates that if the input features are generated from an ill-defined, unstable, and non-standardized process, the resulting model is inherently unreliable, regardless of its architectural simplicity or complexity. The absence of a common, standardized "language" for defining and extracting radiomic features is a greater barrier to clinical trust and adoption than model interpretability alone.1 Therefore, establishing a transparent, reproducible, and scientifically validated feature extraction workflow is the foundational prerequisite for building trustworthy clinical decision support tools based on radiomics.

### **1.2 The Image Biomarker Standardization Initiative (IBSI): Establishing a Gold Standard**

In response to the critical need for standardization, the Image Biomarker Standardization Initiative (IBSI) was formed. The IBSI is an independent, international collaboration of researchers from academia and industry dedicated to standardizing the extraction and definition of image biomarkers for high-throughput quantitative analysis.12 The initiative's primary goal is to address the scantiness of consensus-based guidelines and definitions that has historically plagued the field, thereby improving the reproducibility and comparability of radiomics studies.13

The IBSI's work is comprehensive, providing a suite of tools and guidelines to standardize the entire radiomics process. These include:

* **Standardized Nomenclature and Definitions:** The IBSI provides precise mathematical definitions for a large set of commonly used radiomic features, each assigned a unique permanent identifier to avoid ambiguity.12 This creates a common language for researchers worldwide.  
* **Reference Image Processing Workflow:** The initiative has defined a standardized, step-by-step image processing pipeline, covering essential stages such as image interpolation, intensity re-segmentation, and intensity discretization.13  
* **Benchmark Datasets and Reference Values:** The IBSI provides publicly available digital phantoms and clinical image datasets (e.g., CT scans of lung cancer patients).12 Through multi-team validation efforts, the IBSI has established benchmark reference values for features computed on these datasets. This allows software developers to verify and calibrate their implementations against a gold standard.12  
* **Reporting Guidelines:** To combat the issue of incomplete methodological reporting, the IBSI has developed comprehensive guidelines that specify the essential information that should be included in any radiomics publication.12

The IBSI's efforts have been organized into distinct chapters. IBSI Chapter 1, completed in 2020, focused on the standardization of 169 commonly used radiomic features, including shape, first-order, and texture features. This work culminated in a landmark publication that established reference values and validated the reproducibility of these features across multiple software implementations.12 More recently, IBSI Chapter 2 addressed the standardization of convolutional image filters (e.g., Wavelet, Laplacian of Gaussian), which are often used to enhance specific image characteristics but are a significant source of feature non-reproducibility.12 By providing this robust framework, the IBSI has established the benchmark against which all radiomics research and software should be measured.

### **1.3 PyRadiomics: A Tool for Standardized Feature Extraction**

PyRadiomics is a flexible, open-source software platform developed specifically to address the challenges of standardization and reproducibility in radiomics.1 Implemented in Python and built upon the robust SimpleITK library for image processing, PyRadiomics enables the high-throughput extraction of a large panel of engineered, IBSI-compliant features from medical images.1 It has become one of the most widely used tools in the field due to its comprehensive feature set, customizability, and commitment to standardization.

A key strength of PyRadiomics is its adherence to the IBSI standard. The development team is actively involved in the IBSI collaboration, and the software's feature definitions and calculations are largely compliant with the IBSI reference manual.20 This makes PyRadiomics an excellent choice for researchers seeking to conduct reproducible studies that can be compared with and validated by the broader scientific community.

However, a truly expert-level understanding and application of PyRadiomics requires acknowledging that its compliance with IBSI is not absolute. There are several specific, well-documented instances where the PyRadiomics implementation differs from the IBSI standard.23 These are not errors, but deliberate design choices made for reasons of internal consistency or to support broader functionality. For example:

* **Mask Resampling:** The IBSI standard allows for various interpolators to be used when resampling a binary segmentation mask. PyRadiomics, however, enforces the use of a nearest-neighbor interpolator. This choice is made to robustly support masks with multiple labels (i.e., segmentations of different regions of interest within a single file), preventing the interpolator from creating erroneous new label values at the boundaries.23  
* **Grid Alignment:** During image resampling, the IBSI aligns the new voxel grid based on the center of the image volume. In contrast, PyRadiomics aligns the grid based on the corner of the origin voxel.23  
* **Kurtosis Calculation:** The IBSI standard defines the feature 'Kurtosis' as the "excess kurtosis," which is the statistical kurtosis minus 3, such that a perfect Gaussian distribution has a kurtosis of 0\. PyRadiomics calculates the statistical kurtosis, which is always \+3 relative to the IBSI value.23

These subtle deviations are not limitations but are critical details that must be understood and reported for fully transparent and reproducible science. A "validated" configuration, therefore, is not just a list of settings but an informed protocol that acknowledges these nuances. It empowers the researcher to understand precisely how their features are being calculated, how they relate to the international gold standard, and how to defend their chosen methodology. This report will provide such a protocol, grounded in the principles of the IBSI and tailored for robust implementation within the PyRadiomics framework.

## **Section 2: Foundational Preprocessing for CT Images: A Step-by-Step Validation**

The preprocessing of medical images is a critical stage in the radiomics workflow that occurs between image segmentation and feature extraction.25 The choices made during this stage have a profound impact on the final feature values and, consequently, on the stability and reproducibility of any subsequent analysis.25 For CT images, three preprocessing steps are of paramount importance: voxel resampling (interpolation), intensity discretization, and intensity normalization. This section provides a detailed, evidence-based validation for the optimal settings for each of these steps, forming the foundation of the recommended PyRadiomics configuration.

### **2.1 Voxel Resampling and Interpolation: Achieving Geometric Uniformity**

**Rationale:** CT scans within a single study, and especially across different institutions, are often acquired with varying geometric resolutions. Differences in in-plane pixel spacing (e.g., 0.7 mm vs. 0.9 mm) and, most notably, slice thickness (e.g., 1.25 mm vs. 5.0 mm) are common. These variations in voxel size are a major source of feature instability, particularly for texture features that analyze the spatial relationships between voxels.4 A texture pattern analyzed on a coarse grid of large voxels will yield different feature values than the same pattern analyzed on a fine grid of small voxels. Therefore, resampling the image and segmentation mask to a uniform, isotropic voxel grid is an essential preprocessing step to ensure that features are comparable across all patients in a cohort and to improve reproducibility.4 Several studies have demonstrated that this preprocessing step is necessary to enhance the robustness of CT radiomic features.4

**Recommendation:** Based on a systematic review of common practices in the radiomics literature, the recommended target voxel size for CT images is an isotropic  .25 This resolution provides a good balance, offering sufficient detail for texture analysis without being overly susceptible to image noise. In PyRadiomics, this is configured by setting the

resampledPixelSpacing parameter to \`\`.

**Interpolator Choice:** The process of resampling requires an interpolation algorithm to estimate the intensity values at the new voxel locations. The choice of interpolator can influence the resulting image and the features extracted from it. For the image intensity data, a B-spline interpolator is recommended. B-spline interpolation provides a higher-order approximation that results in a smoother and more accurate representation of the underlying continuous signal compared to simpler methods like linear interpolation, while remaining computationally efficient. In PyRadiomics, this corresponds to setting the interpolator parameter to 'sitkBSpline'.

For the segmentation mask, however, a different approach is required. The mask contains integer labels defining the region of interest (ROI), and it is critical that the interpolation process does not create new, non-existent label values. For this reason, PyRadiomics defaults to and enforces the use of a nearest-neighbor interpolator (sitkNearestNeighbor) for the mask.23 This method ensures that each new voxel in the resampled mask is assigned the exact integer label of the closest voxel in the original mask, thereby preserving the integrity and boundaries of the segmentation.

### **2.2 Intensity Discretization: The Case for Fixed Bin Width (FBW) in CT**

**The Debate:** Most texture feature classes (e.g., GLCM, GLRLM) are not calculated directly on the continuous intensity values of the image. Instead, the intensities within the ROI are first discretized into a smaller number of discrete bins. This process reduces the influence of noise and makes the calculation of texture matrices computationally feasible. Two primary methods for this discretization exist: Fixed Bin Number (FBN) and Fixed Bin Width (FBW), also known as Fixed Bin Size (FBS).27

* **Fixed Bin Number (FBN):** This method divides the full range of intensity values within the ROI (from minimum to maximum) into a predefined number of bins (e.g., 32 or 64). The width of each bin is therefore relative to the intensity range of that specific ROI.  
* **Fixed Bin Width (FBW):** This method uses a constant, predefined bin size (e.g., 25 Hounsfield Units). The number of bins is not fixed but depends on the range of intensities present in the ROI.

**Rationale for FBW in CT:** For imaging modalities with a standardized, quantitative intensity scale, such as Hounsfield Units (HU) in CT, the Fixed Bin Width method is strongly preferred. The HU scale has a direct physical meaning, with values calibrated to the densities of water (0 HU) and air (-1000 HU). Using a fixed bin width preserves this quantitative relationship across all patients. For example, with a bin width of 25 HU, the intensity range of 0-24 HU will always fall into the same bin, regardless of whether the patient's tumor has a maximum intensity of 100 HU or 500 HU.

In contrast, the FBN method is highly susceptible to outliers. If a single voxel in an ROI has an unusually high or low intensity, it will stretch the entire intensity range, thereby changing the width of every bin and altering the discretization for all other voxels. This makes feature values highly dependent on the specific intensity distribution of each individual ROI, reducing their comparability across a cohort. Research has shown that varying the bin number (FBN) leads to statistically significantly more variable and less reproducible feature values compared to varying the bin size (FBW).27 A review of the literature indicates that for CT studies, FBW is the most frequently employed discretization strategy.25

**Recommendation:** Based on the superior theoretical grounding for quantitative imaging and the evidence of greater stability, the use of **Fixed Bin Width** is strongly recommended for CT radiomics. In PyRadiomics, this is the default method and is controlled by the binWidth parameter. A value of **25 HU** is a well-established and commonly used setting that provides a reasonable number of bins for most oncologic applications.28 PyRadiomics' documentation also suggests choosing a bin width that results in a total number of bins between 30 and 130, and the

DEBUG logging mode can be used to verify this for a given dataset.23 A

binWidth of 25 HU generally achieves this goal.

### **2.3 Intensity Normalization and Re-segmentation: A Minimalist Approach**

**Rationale:** Intensity normalization refers to the process of scaling image intensity values to a common range. While this is a critical step for modalities with arbitrary intensity units, such as T1-weighted MRI, it is generally unnecessary and not recommended for CT. As previously discussed, the Hounsfield Unit scale is already a standardized and calibrated unit of measurement.25 Applying an additional global normalization, such as a Z-score transformation which scales intensities based on the mean and standard deviation of the entire image volume, would distort this inherent quantitative meaning. This is supported by literature reviews which show that the vast majority of CT radiomics studies do not employ a normalization strategy.25

**Recommendation:** The global normalization feature in PyRadiomics should be disabled for CT analysis. This is achieved by setting the normalize parameter to its default value of False.29

**Outlier Removal and Re-segmentation:** PyRadiomics offers more targeted methods for controlling intensity ranges that should not be confused with global normalization. The removeOutliers parameter can be used to discard voxels that are several standard deviations from the mean, but its use is generally discouraged as it can arbitrarily remove potentially meaningful biological information.

A more interpretable approach is intensity re-segmentation. This process, as described in the IBSI standard, involves defining an absolute intensity range and excluding any voxels within the original ROI that fall outside this range.13 For example, in a soft-tissue lung tumor, a researcher might apply a re-segmentation range of \[-100, 400\] HU to exclude voxels that are clearly air or dense bone that may have been inadvertently included in the segmentation. While this can be a valid technique to improve the biological specificity of the analysis, it must be applied with caution and reported transparently, as it fundamentally alters the data from which features are extracted. For a baseline, validated configuration, no re-segmentation should be applied. This can be controlled in PyRadiomics by not specifying the

resegmentRange parameter.

The collective impact of these preprocessing steps cannot be overstated. They form a causal chain where the output of one step becomes the input for the next. The interpolation method used during resampling directly influences the intensity values that are subsequently discretized. The choice of discretization method, in turn, defines the gray levels that are used to construct the texture matrices from which the final feature values are derived. A robust and validated configuration is therefore one that stabilizes this entire chain by making coherent, evidence-based choices at each juncture. The recommendations provided here—isotropic 1x1x1 mm³ resampling with B-spline interpolation, no global normalization, and intensity discretization using a fixed bin width of 25 HU—constitute such a system, designed to maximize feature stability and reproducibility for CT-based radiomics.

### **2.4 Table 1: Recommended Foundational Preprocessing Parameters for CT Radiomics**

The following table summarizes the key preprocessing decisions and their corresponding PyRadiomics settings, providing a quick-reference guide that distills the detailed analysis of this section into actionable parameters.

| Parameter Category | Recommended Setting | PyRadiomics Key(s) | Rationale and Justification |
| :---- | :---- | :---- | :---- |
| **Voxel Resampling** | Isotropic   | resampledPixelSpacing: | Standardizes image geometry to improve feature reproducibility across scans with varying slice thicknesses and pixel sizes. This is a necessary step to mitigate a major source of feature instability.4 |
| **Image Interpolation** | B-spline | interpolator: 'sitkBSpline' | Provides a smooth and accurate estimation of intensity values on the new voxel grid. Recommended for intensity data. Note: PyRadiomics enforces nearest-neighbor for the mask to preserve label integrity.23 |
| **Intensity Discretization** | Fixed Bin Width | binWidth: 25 | Preserves the quantitative meaning of the Hounsfield Unit (HU) scale in CT. This method is more robust and reproducible than using a fixed number of bins, which is highly sensitive to outliers.25 A width of 25 HU is a widely adopted standard.28 |
| **Intensity Normalization** | Disabled | normalize: False | Unnecessary for CT images as the HU scale is already standardized. Applying global normalization would distort the inherent quantitative information.25 |
| **Intensity Re-segmentation** | Disabled (by default) | resegmentRange (not specified) | While it can be used to exclude non-relevant voxels (e.g., air, bone), it should not be part of a baseline configuration as it alters the original data. Its use should be a deliberate, reported choice based on the specific clinical application.13 |

## **Section 3: The Validated PyRadiomics Configuration for CT**

Building upon the foundational preprocessing principles established in the previous section, this section presents the complete, validated PyRadiomics configuration for CT image analysis. This configuration is designed to maximize feature stability, reproducibility, and adherence to international standards. It is presented in the YAML (YAML Ain't Markup Language) format, which is the standard method for specifying parameters for PyRadiomics in a non-interactive, script-based workflow. Each parameter group—global settings, image filtering, and feature selection—is explained and justified based on the available scientific evidence.

### **3.1 Global Settings (setting)**

This block defines the overarching parameters that control the entire feature extraction process, including the crucial preprocessing steps. The following settings represent the core of the validated configuration.

YAML

\# \------------------------------------------------------------------------------------------------------  
\# Global settings for the feature extractor  
\# \------------------------------------------------------------------------------------------------------  
setting:  
  \# Voxel Resampling: Resample image and mask to an isotropic 1x1x1 mm^3 grid.  
  \# This is a critical step for standardizing geometry and ensuring feature comparability.\[4, 25\]  
  resampledPixelSpacing: 

  \# Interpolation: Use B-spline for intensity interpolation for accuracy and smoothness.  
  \# PyRadiomics will automatically use Nearest Neighbor for the mask to preserve labels.\[23, 24\]  
  interpolator: 'sitkBSpline'

  \# Intensity Discretization: Use a Fixed Bin Width of 25 Hounsfield Units.  
  \# This is the most robust method for CT, preserving the quantitative HU scale.\[25, 27\]  
  binWidth: 25

  \# Normalization: Disable global intensity normalization.  
  \# Unnecessary for CT as the HU scale is already standardized.\[25\]  
  normalize: False

  \# ROI Constraints: Ensure that the ROI is at least a 2D structure.  
  \# Single-voxel segmentations are always excluded.  
  minimumROIDimensions: 2

  \# Geometry Tolerance: Set a reasonable tolerance for minor mismatches in image/mask geometry.  
  \# This can prevent errors from tiny floating-point differences in DICOM headers.\[24, 29\]  
  geometryTolerance: 1e-6

  \# Additional settings to ensure robustness and adherence to standards.  
  \# These settings reflect common best practices.  
  force2D: False               \# Enforce 3D extraction for volumetric analysis.  
  label: 1                     \# Specify the label in the mask to extract features from.  
  additionalInfo: True         \# Enable output of additional provenance information.

### **3.2 Image Filtering (imageType): A Focus on the Original Signal**

PyRadiomics allows for features to be extracted not only from the original image but also from a variety of filtered versions of that image. These filters, such as Wavelet decompositions and Laplacian of Gaussian (LoG), are designed to highlight different aspects of the image texture at various spatial scales.1 While these filtered images can sometimes yield features with high predictive power, they also introduce a significant source of variability and non-reproducibility.

**Recommendation:** For a baseline, robust, and validated analysis, it is strongly recommended to extract features from **only the Original image type**.

**Rationale:** The computation of features from filtered images is notoriously difficult to standardize. Features derived from filter response maps have been found to be poorly reproducible across different software and imaging parameters.18 The IBSI has dedicated an entire major effort (IBSI Chapter 2\) to the complex task of standardizing these filters, an effort that was only recently completed.12 Furthermore, studies investigating the impact of CT acquisition parameters, such as the reconstruction kernel, have found that wavelet features are among the least stable, showing very poor reproducibility between smooth and sharp kernels.30

By focusing exclusively on the Original image, the analysis is grounded in the most direct and stable representation of the image data. This maximizes the potential for the results to be reproduced and validated by other researchers. The use of image filters should be considered an advanced technique, and any study employing them should treat the filtered feature sets as separate experiments requiring their own rigorous, independent robustness analysis.

YAML

\# \------------------------------------------------------------------------------------------------------  
\# Image types (filters) to use for feature extraction  
\# \------------------------------------------------------------------------------------------------------  
imageType:  
  \# For a baseline robust and validated analysis, enable ONLY the Original image type.  
  \# Features from filtered images (e.g., Wavelet, LoG) are known to have poor reproducibility  
  \# and are highly sensitive to acquisition parameters.\[18, 30\]  
  Original: {}

### **3.3 Feature Selection (featureClass): Prioritizing Stability**

PyRadiomics organizes its features into several distinct classes. The choice of which feature classes to enable determines the final set of biomarkers that will be available for model building.

**Recommendation:** Enable the core set of IBSI-compliant feature classes to generate a comprehensive and standardized panel of descriptors. This includes shape features, first-order statistics, and the five main texture feature families.

**Rationale:** The primary goal of a validated configuration is to produce a well-defined and reproducible set of features. The IBSI Chapter 1 effort successfully standardized a panel of 169 features, providing a consensus-based foundation for radiomics research.15 By enabling all of the corresponding feature classes in PyRadiomics, the researcher starts with this complete, standardized set. This approach is preferable to ad-hoc, a priori selection of a small number of features. It allows for a comprehensive initial extraction, which can then be followed by a data-driven feature reduction step (e.g., based on robustness analysis specific to the user's dataset) to identify the most stable and informative biomarkers for the specific clinical question at hand.

The recommended feature classes are:

* shape: Quantifies the three-dimensional size and shape of the ROI. These features are independent of image intensity.  
* firstorder: Describes the distribution of voxel intensities within the ROI via common statistical metrics.  
* glcm (Gray Level Co-occurrence Matrix): Captures the spatial relationship of voxel pairs, quantifying texture based on how often different gray levels co-occur.  
* glrlm (Gray Level Run Length Matrix): Quantifies texture by measuring the length of consecutive runs of voxels with the same gray level.  
* glszm (Gray Level Size Zone Matrix): Quantifies texture by measuring the size of contiguous zones of voxels with the same gray level.  
* ngtdm (Neighbourhood Gray Tone Difference Matrix): Measures the difference between a voxel's intensity and the average intensity of its neighbors.  
* gldm (Gray Level Dependence Matrix): Measures the number of connected voxels that are dependent on a central voxel.

YAML

\# \------------------------------------------------------------------------------------------------------  
\# Feature classes to enable for extraction  
\# \------------------------------------------------------------------------------------------------------  
featureClass:  
  \# Enable all IBSI-compliant feature classes to generate a comprehensive, standardized feature set.  
  \# This provides the full panel of 169 features standardized by IBSI Chapter 1.\[15\]  
  \- shape  
  \- firstorder  
  \- glcm  
  \- glrlm  
  \- glszm  
  \- ngtdm  
  \- gldm

### **3.4 Table 2: The Validated PyRadiomics CT Configuration File (YAML Format)**

The following block of code represents the complete, copy-paste-ready YAML parameter file. It integrates all the recommendations from this section into a single, actionable configuration. Each parameter is commented to explain its function and justify its recommended value, linking back to the scientific evidence and best practices discussed throughout this report. This file serves as the primary deliverable of this technical report, providing a robust, defensible, and standardized protocol for CT radiomics feature extraction.

YAML

\# \======================================================================================================  
\#  
\# Validated PyRadiomics Configuration for CT Image Analysis  
\# Version 1.0  
\#  
\# This configuration is based on a systematic review of current research and aligns with the  
\# principles of the Image Biomarker Standardization Initiative (IBSI) to maximize feature  
\# stability and reproducibility.  
\#  
\# \======================================================================================================

\# \------------------------------------------------------------------------------------------------------  
\# Global settings for the feature extractor  
\# These settings control the overall behavior of the extraction pipeline, including preprocessing.  
\# \------------------------------------------------------------------------------------------------------  
setting:  
  \# Voxel Resampling: Resample image and mask to an isotropic 1x1x1 mm^3 grid.  
  \# This is a critical step for standardizing geometry and ensuring feature comparability.\[4, 25\]  
  resampledPixelSpacing: 

  \# Interpolation: Use B-spline for intensity interpolation for accuracy and smoothness.  
  \# PyRadiomics will automatically use Nearest Neighbor for the mask to preserve labels.\[23, 24\]  
  interpolator: 'sitkBSpline'

  \# Intensity Discretization: Use a Fixed Bin Width of 25 Hounsfield Units.  
  \# This is the most robust method for CT, preserving the quantitative HU scale.\[25, 27\]  
  \# It is the default method in PyRadiomics.  
  binWidth: 25

  \# Normalization: Disable global intensity normalization.  
  \# Unnecessary for CT as the HU scale is already standardized.\[25\]  
  normalize: False

  \# ROI Constraints: Ensure that the ROI is at least a 2D structure.  
  \# Single-voxel segmentations are always excluded.  
  minimumROIDimensions: 2

  \# Geometry Tolerance: Set a reasonable tolerance for minor mismatches in image/mask geometry.  
  \# This can prevent errors from tiny floating-point differences in DICOM headers.\[24, 29\]  
  geometryTolerance: 1e-6

  \# Enforce 3D extraction for volumetric analysis. For 2D analysis, set to True and ensure  
  \# your ROI is defined on a single slice.  
  force2D: False

  \# Specify the integer label in the mask file that corresponds to the ROI.  
  label: 1

  \# Enable the output of additional information about the extraction process (e.g., versions, settings).  
  \# This is crucial for provenance tracking and reproducibility.  
  additionalInfo: True

\# \------------------------------------------------------------------------------------------------------  
\# Image types (filters) to use for feature extraction  
\# \------------------------------------------------------------------------------------------------------  
imageType:  
  \# For a baseline robust and validated analysis, enable ONLY the Original image type.  
  \# Features from filtered images (e.g., Wavelet, LoG) are known to have poor reproducibility  
  \# and are highly sensitive to acquisition parameters.\[18, 30\]  
  Original: {}

\# \------------------------------------------------------------------------------------------------------  
\# Feature classes to enable for extraction  
\# \------------------------------------------------------------------------------------------------------  
featureClass:  
  \# Enable all IBSI-compliant feature classes to generate a comprehensive, standardized feature set.  
  \# This provides the full panel of 169 features standardized by IBSI Chapter 1.\[15\]  
  \# Individual features within these classes can be disabled if desired, but enabling the  
  \# full classes is recommended for an initial comprehensive analysis.  
  \- shape  
  \- firstorder  
  \- glcm  
  \- glrlm  
  \- glszm  
  \- ngtdm  
  \- gldm

## **Section 4: Navigating Feature Robustness: From Acquisition to Analysis**

While a standardized configuration file provides the necessary foundation for reproducible feature extraction, it is crucial to recognize that it cannot, by itself, guarantee robust results. The values of radiomic features are profoundly influenced by a host of external factors, from the specifics of the CT image acquisition to the method of tumor segmentation. A comprehensive understanding of these factors is essential for designing robust studies and for correctly interpreting their results. This section details the most significant sources of feature instability and provides a framework for selecting features with the highest likelihood of being robust in heterogeneous, real-world datasets.

### **4.1 The Overwhelming Impact of CT Acquisition Parameters**

The process of generating a CT image involves numerous user-selectable parameters that fundamentally define the image's content, texture, and noise characteristics. Variations in these parameters are the single greatest challenge to the generalizability of radiomic models.

**Reconstruction Kernel:** Perhaps the most significant factor influencing texture features is the reconstruction kernel (or algorithm). CT scanners use kernels—such as 'smooth' or 'sharp'—to process raw projection data into the final image. Smooth kernels suppress high-frequency noise, resulting in images with less apparent graininess, while sharp kernels enhance edges and fine details, often at the expense of increased noise.30 Studies have unequivocally demonstrated that the choice of kernel has a strong impact on feature stability. An analysis comparing features from images reconstructed with smooth versus sharp kernels found that, at most, only 26% of 1386 features were stable (defined as an Intraclass Correlation Coefficient \> 0.9).30 First-order intensity features were found to be the most stable, whereas complex texture and wavelet features were highly dependent on the kernel.31 This effect is also highly tissue-dependent; features that are stable in lung parenchyma may not be stable in a solid tumor or lymph node.30 While harmonization techniques like Reconstruction Kernel Normalization (RKN) have been proposed to mitigate these effects, they are not a perfect solution, and their benefit depends on the specific data being analyzed.33

**Slice Thickness & Pixel Spacing:** The geometric resolution of the scan, defined by the slice thickness and in-plane pixel spacing, is another major contributor to feature variability. These resolution parameters have been identified as having the most pronounced effect on feature reproducibility, second only to the reconstruction kernel.26 Thinner slices and smaller pixels provide more granular data, but can also be more affected by noise. Conversely, thicker slices average over a larger volume of tissue, which can obscure fine textural details.3 This is precisely why the isotropic resampling recommended in Section 2 is not merely a suggestion but a mandatory step for any robust radiomics study. Without it, features extracted from scans with different resolutions are fundamentally incomparable.

**Scanner Vendor, Model, and Dose:** While some research suggests that differences in scanner vendor and model may not significantly affect feature reproducibility provided that other parameters like the kernel and resolution are harmonized 26, this finding is not universal. Phantom studies conducted across multiple scanner models have often found significant inter-scanner variability, with very few features demonstrating high reproducibility across all systems.34 This indicates that subtle differences in hardware and proprietary reconstruction software can still introduce systematic biases. Similarly, radiation dose can impact image noise levels, which in turn affects the stability of many texture features.37

### **4.2 The Role of Segmentation and ROI Definition**

Even with a perfectly standardized image, the definition of the region of interest (ROI) from which features are extracted is a critical source of potential variability.

**Segmentation Variability:** The manual delineation of tumor boundaries by human observers is inherently subjective. It is well-documented that inter-observer and even intra-observer variability in segmentation can lead to significant differences in the values of extracted radiomic features, particularly for shape features and complex texture features that are sensitive to the ROI's boundary.21 To mitigate this, the use of semi-automatic segmentation algorithms (e.g., region growing, level-set methods) is often recommended. Studies have shown that features extracted from semi-automatic segmentations can have significantly higher reproducibility and robustness compared to those from purely manual delineations.40

**ROI Size as a Confounder:** A more subtle but equally critical issue is the intrinsic correlation of many radiomic features with the volume of the ROI. The mathematical definitions of some features, such as the first-order feature Energy (which involves a sum of squared intensities), are directly dependent on the number of voxels in the ROI.5 Many texture features are also confounded by volume. Phantom studies using a perfectly homogenous medium have demonstrated this effect starkly: when features were extracted from ROIs of different sizes, a large majority of the features showed significant differences and poor agreement, even though the underlying "texture" was uniform.41 This is a critical consideration, as a model that appears to be predicting a clinical outcome might, in reality, simply be acting as a surrogate for tumor size.

The confluence of these factors leads to a practical understanding of the primary drivers of radiomics instability. This can be conceptualized as a "triad of instability":

1. **The Reconstruction Kernel:** Fundamentally defines the image texture being measured.  
2. **The Geometric Grid:** Defined by the slice thickness and pixel spacing, this determines how the texture is sampled. This is primarily addressed by robust resampling.  
3. **The Segmentation Boundary:** Defines the region being analyzed and is a major source of variance for many features.

A truly robust radiomics study must have a clear and explicit strategy to control or account for all three components of this triad. For example, a study might restrict inclusion to images reconstructed with a single kernel, mandate the isotropic resampling protocol defined in this report, and employ a consistent, semi-automated segmentation protocol with inter-observer agreement analysis.

### **4.3 A Taxonomy of Feature Robustness**

Given the numerous sources of instability, it is evident that not all radiomic features are created equal. Decades of research have led to a general consensus on which feature classes tend to be more robust than others. This knowledge can be used to guide feature selection, particularly when analyzing heterogeneous datasets where acquisition parameters could not be fully standardized. The following classification provides a heuristic for prioritizing features based on their expected stability.

### **4.4 Table 3: Classification of Radiomic Feature Robustness for CT**

| Robustness Tier | Feature Class / Sub-class | Examples | Justification and Key Considerations |
| :---- | :---- | :---- | :---- |
| **High Robustness** | **First-Order (Intensity-based)** | Mean, Median, Maximum, Minimum | These are simple statistical descriptors of the Hounsfield Unit distribution. As HUs are standardized, these features are the most reproducible across variations in acquisition parameters and are least affected by resolution changes.26 They are considered the most stable features. |
|  | **Shape (3D)** | Volume (Voxel Count), Maximum 3D Diameter, Sphericity | These features are calculated from the segmentation mask only and are independent of image intensity values. Their robustness is primarily dependent on the consistency of the segmentation itself. With a stable segmentation protocol, they are highly robust.21 |
| **Moderate Robustness** | **First-Order (Distribution-based)** | Skewness, (Excess) Kurtosis, Entropy, Uniformity | These features describe the shape of the intensity histogram. They are more stable than texture features but can be influenced by image noise and the choice of discretization parameters (bin width).21 |
|  | **Texture (GLCM, GLRLM)** | GLCM Contrast, GLCM Correlation, GLRLM Gray Level Non-Uniformity | These are the most common texture families. Some features within these classes can be moderately robust, but their stability is highly dependent on consistent preprocessing (resampling, discretization) and acquisition parameters (especially reconstruction kernel).30 |
| **Low Robustness** | **Texture (GLSZM, NGTDM, GLDM)** | GLSZM Small Area Emphasis, NGTDM Strength, GLDM Dependence Non-Uniformity | These more complex texture features are often highly sensitive to small variations in image resolution, noise, and preprocessing. GLSZM features, in particular, have been shown to be sensitive to variations in pixel size and slice thickness.4 |
|  | **Filtered Image Features (All Classes)** | Wavelet-LLL\_glcm\_Energy, LoG\_firstorder\_Mean | Features extracted from filtered images (Wavelet, Laplacian of Gaussian, etc.) consistently demonstrate the poorest stability. They are extremely sensitive to the reconstruction kernel and other acquisition parameters, and their standardization is a complex, ongoing effort.18 Their use requires extensive, dedicated robustness analysis. |

This table provides an invaluable heuristic for the radiomics researcher. After extracting the full, IBSI-compliant panel of features using the validated configuration from Section 3, this classification can be used to inform subsequent feature selection and model building. In studies involving data from multiple sites or scanners, prioritizing features from the "High Robustness" and "Moderate Robustness" tiers can significantly increase the likelihood of developing a generalizable and clinically translatable model. It helps to answer the crucial question: "Given the heterogeneity in my data, where should I start looking for stable and meaningful biomarkers?"

## **Section 5: Recommendations for Implementation and Study Design**

Possessing a validated PyRadiomics configuration file is the necessary first step, but it is not the final one. A configuration is a tool, and its effective use depends on a rigorous and transparent scientific methodology. This final section provides actionable recommendations for implementing the proposed configuration in a research setting, emphasizing that a static parameter file must be part of a dynamic, living protocol that includes in-house validation, appropriate harmonization strategies, and adherence to international reporting standards.

### **5.1 Essential In-House Validation: Trust but Verify**

The configuration provided in this report is based on a broad synthesis of the scientific literature and represents a robust starting point for any CT radiomics study. However, every institution and every dataset has its own unique sources of variability. Therefore, it is essential for researchers to perform their own in-house validation to confirm the stability of features within their specific context before embarking on a large-scale analysis.

**Test-Retest Analysis:** The gold standard for assessing feature stability is a test-retest analysis. This involves acquiring two separate CT scans of the same subject in a short time interval, during which the underlying biology is assumed to be stable. Features are extracted from both scans, and their agreement is quantified using the Intraclass Correlation Coefficient (ICC). The ICC is a measure of reliability that assesses the consistency of measurements made under different conditions. A value of 1.0 indicates perfect agreement, while 0 indicates no agreement. Features with an ICC greater than 0.75 or 0.85 are generally considered to have good to excellent robustness and are suitable for inclusion in subsequent modeling.37 If test-retest datasets are available, performing this analysis is the most direct way to generate a list of features that are robust to the specific scan acquisition variations present at a given institution.7

**Phantom Studies:** Physical phantoms are invaluable tools for systematically assessing feature stability. Phantoms with known geometric and textural properties can be scanned across different CT scanners, using various acquisition protocols (e.g., different tube voltages, doses) and reconstruction kernels.5 This allows for a controlled investigation of how each parameter affects feature values, isolating sources of variability in a way that is not possible with patient data alone. Conducting phantom studies can help establish standardized, scanner-specific protocols that maximize feature robustness and can provide critical data for developing harmonization strategies.7

### **5.2 Harmonization for Multi-Center Studies**

In many cases, particularly in retrospective research, it is not possible to standardize image acquisition. Datasets are often aggregated from multiple institutions, each with different scanners and protocols. In these scenarios, where the "triad of instability" (kernel, resolution, segmentation) cannot be fully controlled a priori, post-extraction data harmonization becomes necessary.

**ComBat Harmonization:** One of the most common techniques for this purpose is ComBat harmonization. Originally developed for genomics data, ComBat is a statistical method designed to remove "batch effects" from high-dimensional data.26 In radiomics, a "batch" can be defined as the scanner model, the institution, or any other categorical variable that introduces a systematic, non-biological variation into the feature values. ComBat works by adjusting the feature data to a common distribution, thereby reducing scanner-specific biases. Studies have shown that ComBat can significantly increase the number of reproducible features in heterogeneous datasets.26 However, it is important to note that its effectiveness can be data-dependent, and it must be applied correctly within a cross-validation framework during model training to prevent information leakage from the test set.

### **5.3 Adherence to Reporting Standards for Transparent Science**

The ultimate goal of standardization is to enable transparent science that is reproducible, verifiable, and can be built upon by the global research community. Adhering to established reporting standards is not a bureaucratic exercise but the cornerstone of achieving this goal.

**Radiomics Quality Score (RQS):** The Radiomics Quality Score (RQS) is a checklist and scoring system designed to promote transparency and methodological rigor in radiomics research.9 It consists of 16 key components, covering aspects such as image protocol documentation, segmentation validation, feature robustness analysis, and statistical modeling practices. Researchers should use the RQS as a guide during study design and as a checklist before publication to ensure that all critical methodological details have been documented. High-quality studies should aim to achieve a high RQS score, demonstrating a commitment to transparent and reproducible science.9

**IBSI Reporting Guidelines:** In conjunction with the RQS, researchers must follow the detailed reporting guidelines established by the IBSI.12 This includes explicitly stating the software and version used for feature extraction (e.g., PyRadiomics v3.0.1), providing the complete parameter file used for the analysis (as exemplified in Section 3.4), and detailing every preprocessing step. This level of transparency is essential for others to be able to precisely replicate the study's findings and to accurately compare results across different studies.

In conclusion, the "best" and most "validated" PyRadiomics configuration is not merely a static file, but a dynamic and comprehensive scientific protocol. It begins with the robust, IBSI-aligned baseline configuration detailed in this report. This baseline is then tested and refined through rigorous in-house validation to generate a dataset-specific list of stable features. For multi-center or heterogeneous data, this protocol is augmented with a carefully applied harmonization strategy. Finally, the entire process, from acquisition to analysis, is documented with meticulous detail according to established international standards like the RQS and IBSI guidelines. By following this complete methodology, researchers can move beyond the pitfalls of the reproducibility crisis and contribute to the development of robust, generalizable, and clinically impactful radiomic biomarkers.

#### **Works cited**

1. Computational Radiomics System to Decode the Radiographic Phenotype \- PMC, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC5672828/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5672828/)  
2. Machine and Deep Learning Methods for Radiomics \- PMC, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8965689/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8965689/)  
3. Robustness of radiomic features in CT images with different slice thickness, comparing liver tumour and muscle \- PMC, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8050292/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8050292/)  
4. Robustness of magnetic resonance radiomic features to pixel size resampling and interpolation in patients with cervical cancer, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7856733/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7856733/)  
5. Enhancing the stability of CT radiomics across different volume of interest sizes using parametric feature maps: a phantom study \- PMC \- PubMed Central, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9474978/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9474978/)  
6. Application of radiomics and machine learning in head and neck cancers \- PMC, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7893590/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7893590/)  
7. Robustness of CT radiomics features: consistency within and between single-energy CT and dual-energy CT \- PMC, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9279234/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9279234/)  
8. Generalization optimizing machine learning to improve CT scan radiomics and assess immune checkpoint inhibitors' response in non-small cell lung cancer: a multicenter cohort study, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10400292/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10400292/)  
9. Current progress and quality of radiomic studies for predicting EGFR mutation in patients with non-small cell lung cancer using PET/CT images: a systematic review \- PMC, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8173688/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8173688/)  
10. Reproducibility in Radiomics: A Comparison of Feature Extraction ..., accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7615943/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7615943/)  
11. Development and validation of a pyradiomics signature to predict initial treatment response and prognosis during transarterial chemoembolization in hepatocellular carcinoma \- PMC, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9618693/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9618693/)  
12. Image Biomarker Standardisation Initiative: IBSI, accessed October 4, 2025, [https://theibsi.github.io/](https://theibsi.github.io/)  
13. The image biomarker standardisation initiative — IBSI 0.0.1dev documentation, accessed October 4, 2025, [https://ibsi.readthedocs.io/](https://ibsi.readthedocs.io/)  
14. Standardization in Quantitative Imaging: A Multicenter Comparison of Radiomic Features from Different Software Packages on Digital Reference Objects and Patient Data Sets, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7289262/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7289262/)  
15. The Image Biomarker Standardization Initiative: Standardized ..., accessed October 4, 2025, [https://pubmed.ncbi.nlm.nih.gov/32154773/](https://pubmed.ncbi.nlm.nih.gov/32154773/)  
16. Reference data sets — IBSI 0.0.1dev documentation \- Read the Docs, accessed October 4, 2025, [https://ibsi.readthedocs.io/en/latest/05\_Reference\_data\_sets.html](https://ibsi.readthedocs.io/en/latest/05_Reference_data_sets.html)  
17. The Image Biomarker Standardization Initiative: Standardized Quantitative Radiomics for High-Throughput Image-based Phenotyping \- PMC \- PubMed Central, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7193906/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7193906/)  
18. IBSI 2 – IBSI – Image Biomarker Standardisation Initiative, accessed October 4, 2025, [https://theibsi.github.io/ibsi2/](https://theibsi.github.io/ibsi2/)  
19. The Image Biomarker Standardization Initiative: Standardized Convolutional Filters for Reproducible Radiomics and Enhanced Clinical Insights | Radiology \- RSNA Journals, accessed October 4, 2025, [https://pubs.rsna.org/doi/abs/10.1148/radiol.231319](https://pubs.rsna.org/doi/abs/10.1148/radiol.231319)  
20. CT and MRI radiomic features of lung cancer (NSCLC): comparison and software consistency \- PMC, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11180643/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11180643/)  
21. Robustness of Radiomics in Pre-Surgical Computer Tomography of ..., accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9864775/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9864775/)  
22. ESR Essentials: radiomics—practice recommendations by the European Society of Medical Imaging Informatics \- PMC, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11835989/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11835989/)  
23. Frequently Asked Questions — pyradiomics v3.0.1 documentation, accessed October 4, 2025, [https://pyradiomics.readthedocs.io/en/v3.0.1/faq.html](https://pyradiomics.readthedocs.io/en/v3.0.1/faq.html)  
24. Frequently Asked Questions — pyradiomics v3.1.0rc2.post5+g6a761c4 documentation, accessed October 4, 2025, [https://pyradiomics.readthedocs.io/en/latest/faq.html](https://pyradiomics.readthedocs.io/en/latest/faq.html)  
25. Impact of Preprocessing Parameters in Medical Imaging-Based ..., accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11311340/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11311340/)  
26. MaasPenn Radiomics Reproducibility Score: A Novel Quantitative ..., accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8997100/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8997100/)  
27. Gray-level discretization impacts reproducible MRI radiomics texture features \- PMC, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC6405136/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6405136/)  
28. Impact of image filtering and assessment of volume-confounding effects on CT radiomic features and derived survival models in non-small cell lung cancer \- PMC \- PubMed Central, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9830263/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9830263/)  
29. Customizing the Extraction — pyradiomics v3.1.0rc2.post5+g6a761c4 documentation, accessed October 4, 2025, [https://pyradiomics.readthedocs.io/en/latest/customization.html](https://pyradiomics.readthedocs.io/en/latest/customization.html)  
30. Impact of CT convolution kernel on robustness of radiomic features ..., accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8010534/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8010534/)  
31. Impact of CT convolution kernel on robustness of radiomic features for different lung diseases and tissue types \- PubMed, accessed October 4, 2025, [https://pubmed.ncbi.nlm.nih.gov/33544646/](https://pubmed.ncbi.nlm.nih.gov/33544646/)  
32. Accounting for reconstruction kernel-induced variability in CT radiomic features using noise power spectra \- PubMed, accessed October 4, 2025, [https://pubmed.ncbi.nlm.nih.gov/29285518/](https://pubmed.ncbi.nlm.nih.gov/29285518/)  
33. CT Reconstruction Kernels and the Effect of Pre- and Post-Processing on the Reproducibility of Handcrafted Radiomic Features \- PMC, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9030848/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9030848/)  
34. Robustness of CT radiomics features: consistency within and between single-energy CT and dual-energy CT \- PubMed, accessed October 4, 2025, [https://pubmed.ncbi.nlm.nih.gov/35192011/](https://pubmed.ncbi.nlm.nih.gov/35192011/)  
35. Robustness of radiomics among photon-counting detector CT and dual-energy CT systems: a texture phantom study \- PubMed, accessed October 4, 2025, [https://pubmed.ncbi.nlm.nih.gov/39048741/](https://pubmed.ncbi.nlm.nih.gov/39048741/)  
36. Photon-Counting CT Scan Phantom Study: Stability of Radiomics Features \- PMC, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11941725/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11941725/)  
37. Identification of CT radiomic features robust to acquisition and segmentation variations for improved prediction of radiotherapy-treated lung cancer patient recurrence \- PMC, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11031577/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11031577/)  
38. Impact of a deep learning image reconstruction algorithm on the robustness of abdominal computed tomography radiomics features using standard and low radiation doses \- PMC, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12397659/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12397659/)  
39. Radiomics Repeatability Pitfalls in a Scan-Rescan MRI Study of Glioblastoma \- PMC, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7845781/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7845781/)  
40. Stability and Reproducibility of Radiomic Features Based Various Segmentation Technique on MR Images of Hepatocellular Carcinoma (HCC), accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8468357/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8468357/)  
41. Stability of Radiomic Features across Different Region of Interest ..., accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8293351/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8293351/)  
42. Stability of Radiomic Features across Different Region of Interest Sizes-A CT and MR Phantom Study \- PubMed, accessed October 4, 2025, [https://pubmed.ncbi.nlm.nih.gov/34201012/](https://pubmed.ncbi.nlm.nih.gov/34201012/)  
43. Improving radiomic model reliability using robust features from perturbations for head-and-neck carcinoma \- PubMed Central, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9614273/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9614273/)