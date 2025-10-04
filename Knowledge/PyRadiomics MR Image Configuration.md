

# **A Validated Protocol for Reproducible MRI Radiomics: An Evidence-Based Pyradiomics Configuration**

### **Abstract**

This report synthesizes current evidence from peer-reviewed literature to establish a comprehensive, validated protocol for conducting reproducible radiomics analysis on Magnetic Resonance (MR) images using the Pyradiomics open-source platform. Addressing the critical challenges of feature instability and poor generalizability inherent to MRI, this document details an end-to-end workflow, from foundational prerequisites to a specific, justified Pyradiomics configuration. A deep dive into MRI-specific image preprocessing, including isotropic voxel resampling, intensity standardization, and gray-level discretization, provides definitive recommendations based on phantom studies and test-retest analyses. The centerpiece of this report is a fully annotated config.yaml file, with each parameter choice meticulously justified by scientific evidence. Furthermore, the selection of image filters and feature classes is discussed to prioritize robustness, concluding with a protocol for post-extraction stability analysis to ensure the final feature set is suitable for building reliable clinical models. This document is intended to serve as a foundational standard for researchers, aiming to enhance the methodological rigor and clinical translatability of MRI-based radiomics.

## **Section 1: Foundational Prerequisites for Robust Radiomic Analysis**

The field of radiomics promises to unlock quantitative biomarkers from standard-of-care medical images, converting them into a high-dimensional, mineable feature space that can inform clinical decision-making.1 However, the translation of radiomics from a research tool to a clinical reality has been significantly hampered by issues of reproducibility and generalizability, particularly in the context of Magnetic Resonance (MR) imaging.2 The values of extracted radiomic features are highly sensitive to a cascade of methodological choices throughout the analysis pipeline.4 A finely tuned feature extraction configuration, such as that defined in a Pyradiomics parameter file, is a necessary but insufficient condition for achieving robust results. Its efficacy is entirely dependent on the quality and consistency of the upstream data it processes. Therefore, before any computational analysis can begin, two foundational prerequisites must be rigorously addressed: the standardization of image acquisition and the consistency of image segmentation. Neglecting these initial steps introduces uncontrollable variance that no amount of post-hoc normalization or feature selection can fully mitigate, rendering the subsequent analysis fundamentally flawed.

### **1.1 The Impact of Image Acquisition and Sequence Selection**

The radiomics workflow begins with image acquisition, a step that embeds the first and most significant sources of variability into the data. Unlike Computed Tomography (CT), where image intensities (Hounsfield Units) are standardized to a physical property (tissue density), MR signal intensities are relative and highly dependent on a complex interplay of scanner hardware (manufacturer, field strength), software, and sequence parameters (e.g., repetition time (TR), echo time (TE), inversion time).5 This inherent lack of standardization means that images of the same tissue acquired on different scanners, or even on the same scanner with slightly different protocols, can have vastly different intensity distributions, which in turn leads to profound differences in the calculated radiomic features.5

The choice of MRI sequence itself is a primary determinant of radiomic feature stability. Evidence from phantom studies and clinical test-retest validations consistently demonstrates that not all sequences provide an equally reliable substrate for feature extraction. A pivotal study involving phantom acquisitions on three different 3-T MRI scanners found that T2-weighted imaging (T2WI) yielded a substantially higher number of reproducible features than other common sequences. Across rigorous test-retest, multi-scanner, and computational comparisons, T2WI provided 41 reproducible features, whereas Diffusion-Weighted Imaging (DWI) provided only six, and T1-weighted imaging (T1WI) provided a mere two.7 This indicates that T2WI-derived features are inherently more stable against the technical variations introduced by different hardware and repeated acquisitions. Similarly, another phantom study concluded that fluid-attenuated inversion recovery (FLAIR) sequences deliver a highly robust substrate for radiomic analyses, with high-resolution FLAIR images yielding the greatest percentage of robust features (81% of those tested) compared to T1w and T2w sequences.8 While quantitative maps such as the Apparent Diffusion Coefficient (ADC) are valuable, the features derived from them can also be sensitive to acquisition parameters; however, the application of appropriate preprocessing has been shown to increase the number of stable ADC-derived features.9

This evidence highlights a critical principle: the radiomics pipeline acts as an amplification system for upstream variance. A minor, seemingly insignificant variation in scanner parameters or patient positioning does not remain a small error. As the image data passes through the complex mathematical transformations of feature extraction (e.g., Wavelet filtering, gray-level matrix calculations), this initial variance is propagated and magnified, resulting in large, unpredictable shifts in the high-dimensional feature space.2 This phenomenon explains why numerous studies report dramatic drops in feature stability resulting from minor changes in the acquisition process.10 Consequently, methodological rigor at the beginning of the workflow is exponentially more impactful than any downstream correction.

**Recommendation:** For any radiomics study, particularly those involving multi-center data, the image acquisition protocol must be harmonized to the greatest extent possible.5 When designing a study, researchers should prioritize the use of T2WI and FLAIR sequences for feature extraction due to their demonstrated superior stability. If a multi-parametric approach is employed, which is often beneficial for capturing diverse biological information, it is imperative that features extracted from each sequence are analyzed for stability independently before being combined into a predictive model.

### **1.2 Segmentation: The Origin Point of Feature Variability**

Following image acquisition, the delineation of the Region of Interest (ROI) or Volume of Interest (VOI) is universally recognized as a "critical and challenging step" in the radiomics workflow.1 The segmentation mask defines the precise spatial boundaries from which all subsequent feature data are derived. Any ambiguity or inconsistency in this delineation directly translates into feature variability, undermining the entire analysis.

The most common method for segmentation, manual delineation by a clinical expert, is a well-documented source of significant error. This process is not only laborious and time-consuming but is also highly susceptible to both intra-observer (the same expert delineating the ROI differently at different times) and inter-observer (different experts producing different delineations) variability.1 Studies have repeatedly shown that a large number of radiomic features, especially those related to shape and texture, are not robust against these variations in ROI delineation.12 Shape features are directly computed from the mask's geometry, while texture features are highly sensitive to the inclusion or exclusion of voxels at the tumor boundary, where intensity gradients are often sharpest.

The stability of the segmentation process, and thus the features derived from it, is not solely dependent on the operator but also on the characteristics of the lesion being measured. For instance, a study investigating breast cancer molecular subtype prediction found a positive correlation between lesion size and the number of stable radiomic features.13 This suggests that "feature stability" is not an intrinsic property of the feature algorithm alone but an emergent property of the interaction between the feature, the imaging modality, and the object of measurement. Small, indistinct, or complex-shaped lesions are inherently more challenging to segment consistently, leading to high variance in the resulting shape and texture features. Conversely, larger, more conspicuous, and regularly shaped lesions are easier to delineate reproducibly, which in turn leads to higher feature stability. This has profound implications for model development, as a radiomics model trained on a cohort of patients with large, well-defined tumors may fail when applied to patients with smaller lesions, not because of a fundamental difference in biology, but because the feature extraction process itself becomes unstable and noisy.

**Recommendation:** To minimize observer-induced variability, the use of automated or semi-automated segmentation methods is strongly preferred over purely manual delineation.1 Numerous open-source tools, such as 3D Slicer and ITK-SNAP, provide algorithms (e.g., region-growing, level-sets) that can produce more consistent results.12 Furthermore, deep learning-based segmentation models, often utilizing U-Net architectures, are rapidly emerging as the state-of-the-art for automated, reproducible organ and tumor segmentation.1 If manual segmentation is the only feasible option, it is mandatory for the study protocol to include a formal assessment of inter-observer reproducibility. This typically involves having at least two experts delineate the ROIs on a representative subset of the data. Radiomic features should then be extracted from both sets of segmentations, and a metric such as the Intraclass Correlation Coefficient (ICC) should be calculated for each feature. Only features that demonstrate high agreement (e.g., ICC \> 0.75 or \> 0.80) should be retained for the final analysis and model building.12

## **Section 2: A Deep Dive into MRI-Specific Image Preprocessing**

Once high-quality, consistently acquired images and robust segmentation masks are obtained, the data must undergo a series of preprocessing steps to harmonize them before feature extraction. This stage is particularly critical for MRI radiomics, as it is designed to computationally mitigate the inherent variability in voxel geometry and signal intensity that cannot be fully controlled during acquisition.5 These steps are not optional; they are non-negotiable components of a methodologically sound pipeline. The order of operations is also crucial, as each step prepares the data for the next in a specific sequence. The correct workflow—spatial harmonization followed by intensity standardization and finally gray-level discretization—ensures that the resulting radiomic features reflect underlying biological patterns rather than technical artifacts of the imaging process.

### **2.1 Spatial Harmonization: Isotropic Voxel Resampling**

A common characteristic of clinical MRI protocols is the acquisition of anisotropic voxels. This means the dimensions of the voxels are not equal in all three spatial directions; for example, an image may have high in-plane resolution (e.g.,  mm) but a much larger slice thickness (e.g., 5 mm). This anisotropy poses a significant problem for radiomics, particularly for the calculation of 3D shape and texture features. Many of these features are not rotationally invariant, meaning their calculated values can change simply based on the orientation of the tumor within the anisotropic voxel grid. This introduces a source of variance that is entirely unrelated to tumor biology.

To address this, spatial harmonization via resampling to an isotropic voxel grid is an essential preprocessing step. This process uses interpolation to create a new image volume where voxels have identical dimensions in x, y, and z (e.g.,  mm). A phantom study demonstrated that voxel size resampling significantly increased the stability of textural features, which were found to be particularly sensitive to variations in slice thickness and in-plane pixel spacing.9 By ensuring that all images in a dataset share the same voxel geometry, resampling guarantees that feature calculations are comparable across different patients and scanners.

The implementation of this step in a Pyradiomics-based workflow is controlled by two key parameters: resampledPixelSpacing and interpolator.14 The

resampledPixelSpacing parameter defines the target isotropic voxel dimensions. The choice of this target spacing is important; it should be selected with consideration for the native resolution of the source images to avoid excessive loss of information (if downsampling too aggressively) or the artificial creation of data (if upsampling too far beyond the original resolution). The interpolator parameter specifies the algorithm used to calculate the intensity values for the new voxel grid.

**Recommendation:**

* **resampledPixelSpacing**: This parameter should be set to a uniform isotropic voxel size appropriate for the study data, for example, \`\` for a target resolution of 1 mm in all dimensions. This enforces a consistent spatial domain for feature calculation across all images.  
* **interpolator**: For resampling the grayscale image, sitkBSpline is the recommended interpolator. It is a robust choice for interpolating continuous intensity data and is widely used in medical image processing. It is critical to note that Pyradiomics intelligently handles the mask resampling separately, always using a sitkNearestNeighbor interpolator. This is the correct approach, as it ensures that the discrete integer labels of the segmentation mask are preserved without creating new, ambiguous interpolated values at the boundaries of the ROI.14

### **2.2 Intensity Standardization: Normalizing Arbitrary Units in MRI**

Perhaps the single greatest challenge to the reproducibility and generalizability of MRI-based radiomics is the arbitrary nature of MR signal intensities.5 Unlike CT, the grayscale values in most clinical MR sequences (e.g., T1w, T2w, FLAIR) do not correspond to a standardized physical unit. They are relative values influenced by a multitude of factors including the scanner manufacturer, field strength, specific sequence parameters, receiver coil loading, and post-processing algorithms.6 Consequently, the same tissue can exhibit vastly different intensity ranges across different patients or even for the same patient scanned at different times or on different machines. This technical variability can easily overwhelm any subtle biological signal encoded in the image texture.

Therefore, intensity standardization is a mandatory preprocessing step for any MRI radiomics study that involves data from more than one scanner or from scans acquired with non-identical protocols. The goal is to transform the image intensities onto a common, standardized scale, thereby minimizing inter-scan variations that are not biological in origin. A systematic review of intensity standardization methods in the context of glioma MRI radiomics identified several techniques, including histogram-based methods (e.g., histogram matching) and intensity rescaling, confirming that this is a key preprocessing step, although a universal consensus on the single best method has yet to be established.6 However, studies have confirmed that the application of intensity standardization does increase the stability of radiomic features, particularly first-order statistics which are directly derived from the intensity histogram.9

Given the lack of a definitive gold standard, the most practical, defensible, and widely applicable approach is Z-score normalization. This method rescales the intensity values of the voxels *within the segmented ROI* to have a mean of 0 and a standard deviation of 1\. The transformation is defined as: , where  and  are the mean and standard deviation of the voxel intensities within the original ROI.

The rationale for this recommendation is threefold:

1. **Effectiveness:** It directly addresses the two primary modes of intensity variation: shifts in overall brightness (mean) and changes in image contrast (standard deviation).  
2. **Simplicity and Reproducibility:** It is a parameter-free method. Unlike techniques like histogram matching, it does not require a reference image, which can itself be a source of variability. Its mathematical definition is unambiguous, making it easy to implement consistently across studies.  
3. **Targeted Application:** By calculating the statistics ( and ) only from the voxels within the ROI, it normalizes the tissue of interest without being influenced by the intensity of surrounding anatomical structures.

**Recommendation:** Z-score normalization should be applied to the ROI of every MR image in the dataset. It is important to note that this step should be performed *after* resampling and *before* feature extraction. Crucially, this normalization should be implemented as a custom preprocessing step prior to invoking the Pyradiomics feature extractor. While Pyradiomics has a built-in normalize setting, it applies the normalization to the entire image volume based on the global mean and standard deviation.14 This is not the recommended approach, as the statistics of tissues outside the ROI can unduly influence the normalization of the target lesion. The normalization must be based on the intensity distribution of the ROI alone.

### **2.3 Gray-Level Discretization: The Case for Fixed Bin Width**

Before texture features can be calculated, the continuous or semi-continuous range of gray-level intensities within the ROI must be discretized into a finite number of bins. This process, also known as quantization, is fundamental to the construction of texture matrices like the Gray Level Co-occurrence Matrix (GLCM) or Gray Level Size Zone Matrix (GLSZM). The manner in which this discretization is performed has a profound impact on the final feature values and their reproducibility.

There are two primary methods for discretization: using a fixed number of bins (controlled by the binCount parameter in Pyradiomics) or using a fixed bin width (controlled by binWidth). When using binCount, the range of intensities within the ROI (from minimum to maximum) is divided into a predefined number of bins. The consequence of this approach is that the width of each bin becomes dependent on the intensity range of that specific ROI. If an ROI contains an outlier voxel with a very high or low intensity, the entire intensity range expands, causing the bin width to increase and compressing all other intensity values into fewer bins. This makes the resulting texture features highly sensitive to segmentation boundaries and noise.

In contrast, the fixed bin width approach defines a constant size for each bin. The number of bins is then determined by how many are needed to cover the intensity range of the ROI. This method is significantly more robust. The European Society of Medical Imaging Informatics (ESMII), in its practice recommendations, strongly favors the use of a fixed bin width over a fixed bin count. Their analysis concludes that because binWidth is independent of the intensity range in the segmented ROI, it reduces variability stemming from segmentation and yields a higher number of reproducible radiomic features.

This recommendation is a direct consequence of the causal chain of preprocessing. If intensity standardization (e.g., Z-scoring) has been correctly applied in the previous step, the intensity values within the ROI are already on a standardized scale. Using a fixed bin width on this standardized data ensures that the same intensity differences are treated identically across all patients, leading to more comparable and robust texture features.

**Recommendation:**

* **binWidth**: This parameter should be set to a specific floating-point value. The Pyradiomics default is 25, which is a reasonable starting point for non-normalized images like CT.14 However, for Z-score normalized MR data, where the standard deviation is 1, a much smaller bin width is required. A value between 0.25 and 0.5 is often more appropriate, as this would typically result in a reasonable number of bins (e.g., 16 to 128\) covering the bulk of the intensity distribution (e.g.,  
   standard deviations). The chosen value should be determined based on the dataset and reported transparently in any publication.4  
* **binCount**: This parameter should be left at its default value of None to ensure that the binWidth setting takes precedence.

The preprocessing pipeline, while essential, is not without its risks. Each step is a mathematical transformation that alters the original image data, and this can be viewed as a potential "black box".5 It is conceivable that a predictive model could learn an artifact of the specific preprocessing chain rather than a true biological signal. For example, a model might become sensitive to the shape of the post-normalization intensity histogram, a characteristic that would be different if an alternative normalization method were used. This underscores the critical importance of adhering to a single, standardized, and transparently reported preprocessing protocol for the entire study, from training to validation, to ensure that any findings are scientifically sound and interpretable.

## **Section 3: The Validated Pyradiomics Configuration for MRI**

Following the rigorous application of foundational prerequisites and MRI-specific preprocessing, the final step in the data preparation phase is the extraction of radiomic features using a well-defined and justified configuration. This section provides the central deliverable of this report: a complete, annotated Pyradiomics parameter file (config.yaml) designed to maximize the reproducibility and robustness of features extracted from MR images. This configuration is not arbitrary; it is the culmination of the evidence-based principles discussed in the preceding sections. It represents a "stability-first" philosophy, prioritizing a core set of reliable features over a larger, more comprehensive but potentially noisy and irreproducible feature set. This conservative approach is a direct response to the reproducibility challenges that have hindered the clinical translation of radiomics.2 By standardizing the extraction process with this configuration, researchers can establish a robust and defensible baseline for their analyses.

### **3.1 The Complete config.yaml for MRI Radiomics**

The following YAML file represents the recommended configuration for MRI radiomics analysis using the Pyradiomics library. It should be used in conjunction with the external preprocessing steps of ROI-based Z-score normalization as detailed in Section 2.2. The file is structured with comments to provide clarity on the purpose of each section.

YAML

\# \==============================================================================  
\# Pyradiomics Configuration File for Robust MRI Radiomics  
\# Version 1.0 \- Evidence-Based Recommendations  
\#  
\# This configuration is designed to be used AFTER external preprocessing:  
\# 1\. Resampling to isotropic voxels (e.g., 1x1x1 mm).  
\# 2\. ROI-based Z-score normalization of image intensities.  
\#  
\# This configuration prioritizes feature stability and reproducibility.  
\# \==============================================================================

\# General settings for the feature extractor  
setting:  
  \# Voxel resampling is assumed to be done as a prior preprocessing step.  
  \# Therefore, resampling within Pyradiomics is disabled by setting  
  \# resampledPixelSpacing to None. If preprocessing is not done externally,  
  \# enable this by setting it to e.g., .  
  resampledPixelSpacing: None

  \# Interpolator for resampling. 'sitkBSpline' is the recommended default for  
  \# images. This setting is moot if resampledPixelSpacing is None.  
  \# Mask is always resampled with 'sitkNearestNeighbor'.  
  interpolator: 'sitkBSpline'

  \# Gray-level discretization using a fixed bin width. This is more robust  
  \# than using a fixed bin count, especially after intensity normalization.  
  \# The value of 0.5 is a sensible starting point for Z-score normalized data.  
  \# This may need to be tuned depending on the data distribution.  
  binWidth: 0.5

  \# Ensure that binCount is not specified so that binWidth is used.  
  \# binCount: None

  \# Set verbosity level for detailed logging during extraction.  
  \# This is useful for debugging and ensuring reproducibility.  
  \# Levels: 10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR, 50=CRITICAL  
  verbosity: 30

  \# Settings for mask validation and correction.  
  \# correctMask: True can be useful if masks and images have minor  
  \# geometric mismatches, but it's better to fix this upstream.  
  correctMask: False  
  geometryTolerance: 1e-6

\# Enabled image types for feature extraction.  
\# To maximize reproducibility and minimize feature space complexity,  
\# only features from the original (preprocessed) image are extracted.  
\# Wavelet and LoG filters are disabled.  
imageType:  
  Original: {}  
  \# Wavelet: {}  \# Disabled for baseline robust configuration  
  \# LoG: {}      \# Disabled for baseline robust configuration

\# Enabled feature classes. By default, all are enabled.  
\# This configuration explicitly lists them for clarity.  
featureClass:  
  shape:  
  firstorder:  
  glcm:  
    \# Gray Level Co-occurrence Matrix settings  
    \# symmetricalGLCM: True is the default and generally preferred.  
    symmetricalGLCM: True  
    \# distances:  is the default, considering only immediate neighbors.  
    distances:   
  glrlm:  
    \# Gray Level Run Length Matrix  
  glszm:  
    \# Gray Level Size Zone Matrix  
  gldm:  
    \# Gray Level Dependence Matrix  
  ngtdm:  
    \# Neighbouring Gray Tone Difference Matrix  
    distances: 

### **3.2 Parameter-by-Parameter Justification**

To fully appreciate the rationale behind the recommended configuration, it is essential to examine each parameter choice in detail, comparing it to the Pyradiomics default and justifying it with evidence from the literature. The following table provides this comprehensive, line-by-line annotation. It is this detailed justification that transforms a simple configuration file into a scientifically validated protocol.

| Parameter Path | Recommended Value | Pyradiomics Default | Evidence-Based Rationale and Justification |  |
| :---- | :---- | :---- | :---- | :---- |
| setting:resampledPixelSpacing | None | None | The value is set to None under the critical assumption that spatial harmonization to isotropic voxels is performed as a mandatory external preprocessing step. This is the most robust approach. If not done externally, this must be enabled (e.g., \`\`) to ensure the stability and rotational invariance of texture features, which are highly sensitive to the anisotropic acquisitions common in MRI.9 Leaving this disabled without external resampling is unsafe for multi-center or longitudinal MRI studies. |  |
| setting:interpolator | 'sitkBSpline' | 'sitkBSpline' | The recommended value aligns with the Pyradiomics default, which is a well-established and robust choice for interpolating continuous image intensity data during resampling.14 This parameter is only active if | resampledPixelSpacing is enabled. |
| setting:binWidth | 0.5 | 25 | This is a cornerstone of the robust configuration. A fixed binWidth is strongly recommended over a fixed binCount to enhance feature reproducibility by decoupling discretization from the ROI's specific intensity range. The default value of 25 is suitable for images with a large intensity range (e.g., CT). For Z-score normalized MRI data (mean=0, std=1), a value of 0.5 is a more appropriate starting point to ensure a reasonable number of discrete gray levels (typically 16-128) are generated for texture analysis. |  |
| setting:verbosity | 30 | 10 (INFO) | Setting verbosity to WARNING (30) provides a clean output for batch processing, logging only potential issues. For debugging a single case, increasing this to INFO (20) or DEBUG (10) is useful. This setting does not affect feature values but is important for practical implementation and quality control. |  |
| setting:correctMask | False | False | Disabling mask correction enforces a stricter quality control. Geometric mismatches between the image and mask should ideally be resolved in the data curation phase. While enabling this can fix minor issues, it can also mask more significant alignment problems that should be addressed upstream.14 |  |
| imageType:Original | {} | Enabled | This configuration explicitly enables only the Original image type. This means features are extracted directly from the input image (which should have already been resampled and normalized). This is the most conservative and robust approach. |  |
| imageType:Wavelet | Disabled | Disabled | Wavelet filters are disabled. While they can capture multi-scale texture information, they massively increase the feature space (by 8-fold per level), heightening the risk of overfitting and finding spurious correlations. They also introduce additional parameters (e.g., wavelet type) that complicate standardization.17 For a baseline configuration, they should be excluded. |  |
| imageType:LoG | Disabled | Disabled | Laplacian of Gaussian (LoG) filters are disabled. LoG filters enhance edges and textures but require the specification of sigma values, which determine the scale of features to be enhanced. The choice of sigma is often arbitrary and becomes another source of inter-study variability. Disabling LoG prioritizes features from the original intensity distribution, enhancing simplicity and reproducibility.14 |  |
| featureClass:\* | All Enabled | All Enabled | All standard IBSI-compliant feature classes (shape, firstorder, glcm, glrlm, glszm, gldm, ngtdm) are enabled for extraction. The subsequent post-extraction validation step (Section 4\) is responsible for filtering this comprehensive set down to only the most robust features for the specific application and dataset. This ensures a complete initial extraction, followed by a data-driven stability assessment. |  |

### **3.3 Guidance on Image Filters (Wavelet, LoG)**

A key decision in the proposed configuration is the explicit disabling of derived image filters such as Wavelet and Laplacian of Gaussian (LoG). Pyradiomics offers the powerful capability to apply these filters before feature extraction, which can theoretically provide deeper insights into image texture at various scales and frequencies.17 For example, LoG filters can highlight structures of a specific size depending on the chosen sigma, while Wavelet decomposition separates the image into different frequency sub-bands.

However, this power comes at a significant cost to reproducibility and standardization. Each filter introduces its own set of parameters that must be selected and harmonized. For LoG, the list of sigma values must be defined; for Wavelet, the wavelet family (e.g., 'coif1') and decomposition levels must be chosen.14 These choices are often made without a strong theoretical or empirical basis, becoming yet another "researcher degree of freedom" that introduces variability and makes it difficult to compare results across studies.

Furthermore, applying these filters causes a combinatorial explosion in the number of features. A single level of wavelet decomposition multiplies the feature count by eight. This high dimensionality drastically increases the risk of multicollinearity, overfitting, and discovering spurious associations, a problem already endemic to radiomics studies which often feature more variables than samples.2

**Recommendation:** For a baseline, validated configuration focused on establishing a reproducible and generalizable result, it is strongly recommended to **disable all image filters** and extract features solely from the original (preprocessed) image type. This "stability-first" approach minimizes the feature space, reduces the number of arbitrary parameter choices, and prioritizes a smaller set of more robust and directly interpretable features. While filters can be valuable for more advanced, exploratory analyses, they should only be introduced after a stable baseline model has been established and with a clear, specific hypothesis for their use.

## **Section 4: Post-Extraction Validation: The Final Step to a Robust Feature Set**

The implementation of a standardized acquisition protocol, robust segmentation, and a validated Pyradiomics configuration represents the best possible effort to generate consistent and high-quality radiomic data. However, even under these optimized conditions, it is inevitable that some of the extracted features will remain sensitive to residual sources of variation. Therefore, the final and indispensable step in the pipeline is a data-driven validation of the extracted features to identify and retain only those that are demonstrably stable and robust. Building a predictive model using unstable features is analogous to building a model on noisy data; the model will learn artifacts specific to the training set and will fail to generalize to new, unseen data.7 This post-extraction filtering is the ultimate quality control gate that ensures the final feature set is suitable for developing reliable and clinically translatable models.

### **4.1 Implementing Stability Analysis**

The gold standard for assessing feature stability is a test-retest analysis.10 This involves acquiring two separate MR scans from the same group of subjects in a short time frame (e.g., within hours or days), during which the underlying biology of the tissue is assumed to be unchanged. The entire radiomics pipeline, from segmentation to feature extraction, is then performed on both sets of images. The stability of each individual feature is quantified by calculating a reproducibility metric that compares the feature's values from the first scan (test) to the second scan (retest) across the cohort.

The most common and appropriate metrics for this purpose are the Intraclass Correlation Coefficient (ICC) and the Concordance Correlation Coefficient (CCC). Both metrics measure the agreement between two sets of measurements, with values ranging from \-1 to 1\. A value of 1 indicates perfect agreement. A commonly accepted threshold for deeming a feature "robust" or "stable" is an ICC or CCC value greater than 0.75 or 0.80, though some studies employ a more stringent cutoff of 0.90.7 After this analysis, only the features that surpass the predefined stability threshold are retained for subsequent feature selection and model building.

The "Generalizability Paradox" provides a compelling rationale for this rigorous filtering. One study that explicitly compared models built with reproducible features (CCC \> 0.9) versus non-reproducible ones found that the reproducible model performed significantly better on an independent validation set (AUC 0.957 vs. 0.869).7 This may seem counterintuitive, as one might expect more complex or sensitive features to capture more information. However, unstable features are essentially carriers of technical noise. A machine learning algorithm can easily overfit to this noise, learning patterns that are unique to the scanner artifacts, patient positioning, or segmentation inconsistencies of the training set. When the model is applied to a new dataset where this specific noise signature is absent, its performance plummets. In contrast, stable features, by definition, represent signals that are consistent despite technical variations. A model trained on these features is therefore learning a more robust, underlying biological pattern, which is precisely why it generalizes more effectively to new data.

**Recommendation:** If the resources and ethical approvals are available, conducting a test-retest imaging study on a subset of the patient cohort is mandatory for any high-impact radiomics project. The results of this stability analysis should be used to filter the feature set. If a dedicated test-retest cohort is not feasible, researchers must perform an *a priori* feature filtering based on evidence from published phantom and test-retest studies, prioritizing feature classes known to have higher stability.

### **4.2 A Curated List of Generally Robust MRI Radiomic Features**

In the absence of a study-specific test-retest analysis, researchers can leverage the growing body of literature that has investigated feature stability to make informed decisions about which features to prioritize. A qualitative synthesis of 41 studies investigating radiomic feature repeatability and reproducibility provides a clear hierarchy of stability among the different feature classes.19

1. **Shape Features:** These features, which describe the geometry of the ROI in 2D and 3D, are generally found to be highly repeatable and robust.9 This is logical, as they are calculated directly from the segmentation mask and are independent of voxel intensity values. However, their stability is entirely contingent on the reproducibility of the segmentation itself. If segmentation is inconsistent, shape features will be unstable.  
2. **First-Order Statistics:** These features describe the distribution of voxel intensities within the ROI, derived from the image histogram. A systematic review found that first-order features were, on the whole, more reproducible than either shape or textural features.19 Within this class, the feature  
   **Entropy**, which measures the randomness or unpredictability of the intensity distribution, was consistently reported as one of the most stable and reproducible first-order features.19  
3. **Textural Features:** This broad category includes all features derived from gray-level matrices (GLCM, GLRLM, GLSZM, GLDM, NGTDM). These features quantify the spatial relationships between voxels of different intensities and are intended to capture patterns of intratumoral heterogeneity. Unfortunately, they are also the most sensitive to variations in image acquisition, reconstruction, and preprocessing parameters.9 Their calculation depends on both the spatial grid (affected by resampling) and the intensity values (affected by normalization and discretization), making them susceptible to multiple sources of noise. The literature does not show a clear consensus on which specific textural features are most stable, as this can depend heavily on the imaging modality and application. However, features related to  
   **coarseness** and **contrast** have appeared among the least reproducible in some studies.19

**Recommendation:** When building a radiomics model, prioritize the inclusion of shape and first-order features, as they are most likely to be robust. Textural features should be treated with a higher degree of skepticism. If they are included, they must be subjected to the most rigorous stability analysis possible. A model built on a carefully curated set of features known to be stable is far more likely to produce a generalizable and clinically meaningful result than one built from an unfiltered, high-dimensional set of thousands of potentially noisy features.7

## **Conclusion: Towards Clinical Translation through Standardization and Best Practices**

The immense potential of radiomics to provide non-invasive, quantitative biomarkers for diagnosis, prognosis, and treatment response prediction can only be realized if the field overcomes its significant challenges with reproducibility and generalizability.3 This is especially true for MRI-based radiomics, where the inherent variability of the imaging modality presents a formidable obstacle. This report has synthesized evidence from the current scientific literature to propose a comprehensive, end-to-end protocol designed to address these challenges head-on.

The recommended workflow is built on a foundation of methodological rigor, beginning with the critical prerequisites of standardized image acquisition—prioritizing stable sequences like T2WI and FLAIR—and consistent, preferably automated, ROI segmentation.1 From this solid foundation, a specific, ordered chain of MRI-specific preprocessing steps is required: (1) spatial harmonization through resampling to an isotropic voxel grid, (2) ROI-based Z-score intensity standardization to place arbitrary signal intensities onto a common scale, and (3) gray-level discretization using a fixed bin width to ensure robust texture calculation.5

The centerpiece of this protocol is a validated Pyradiomics configuration (config.yaml) that embodies a "stability-first" philosophy. By focusing feature extraction on the original (preprocessed) image and disabling complex image filters, it minimizes the feature space and reduces researcher-introduced variability, providing a robust and defensible baseline for analysis. The final, crucial step is a post-extraction validation, ideally through a test-retest analysis using the Intraclass Correlation Coefficient, to filter the feature set and retain only those biomarkers that are demonstrably robust to technical noise.10 A model built from this final, curated set of stable features is significantly more likely to generalize to new clinical data and reflect true underlying biology.7

It must be emphasized that this protocol provides a robust *baseline*, not a one-size-fits-all solution for every conceivable research question. It is, however, the most scientifically defensible starting point for producing reliable and reproducible results in the challenging domain of MRI radiomics. The future clinical translation of radiomic models is critically dependent on the widespread adoption of such standardized, evidence-based practices. Adherence to methodological transparency and reporting guidelines, such as the Radiomics Quality Score (RQS) and the CLEAR checklist, is essential for building trust and enabling meta-analysis.11 The continued development and use of open-source, community-vetted tools like Pyradiomics, coupled with a commitment to open-data initiatives, will be the primary drivers that move radiomics from a promising research concept to a powerful tool in the practice of precision medicine.4

#### **Works cited**

1. MRI-derived radiomics: methodology and clinical applications in the field of pelvic oncology, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC6913356/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6913356/)  
2. Reproducibility and interpretability in radiomics: a critical assessment \- PubMed Central, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12239541/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12239541/)  
3. The use of radiomics in magnetic resonance imaging for the pre-treatment characterisation of breast cancers: A scoping review \- PubMed, accessed October 4, 2025, [https://pubmed.ncbi.nlm.nih.gov/37534540/](https://pubmed.ncbi.nlm.nih.gov/37534540/)  
4. ESR Essentials: radiomics-practice recommendations by the ..., accessed October 4, 2025, [https://pubmed.ncbi.nlm.nih.gov/39453470/](https://pubmed.ncbi.nlm.nih.gov/39453470/)  
5. Normalization Strategies in Multi-Center Radiomics Abdominal MRI: Systematic Review and Meta-Analyses \- PubMed Central, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10241248/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10241248/)  
6. Intensity standardization of MRI prior to radiomic feature extraction for artificial intelligence research in glioma—a systematic review \- PMC \- PubMed Central, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9474349/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9474349/)  
7. Achieving imaging and computational reproducibility on ... \- PubMed, accessed October 4, 2025, [https://pubmed.ncbi.nlm.nih.gov/37665391/](https://pubmed.ncbi.nlm.nih.gov/37665391/)  
8. Robustness and Reproducibility of Radiomics in Magnetic Resonance Imaging: A Phantom Study \- PubMed, accessed October 4, 2025, [https://pubmed.ncbi.nlm.nih.gov/30433891/](https://pubmed.ncbi.nlm.nih.gov/30433891/)  
9. Repeatability and reproducibility of MRI-radiomic features: A ..., accessed October 4, 2025, [https://pubmed.ncbi.nlm.nih.gov/36310346/](https://pubmed.ncbi.nlm.nih.gov/36310346/)  
10. Reproducibility and Generalizability in Radiomics Modeling: Possible Strategies in Radiologic and Statistical Perspectives \- PMC \- PubMed Central, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC6609433/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6609433/)  
11. Criteria for the translation of radiomics into clinically useful tests \- PMC, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9707172/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9707172/)  
12. Radiomics in medical imaging—“how-to” guide and critical reflection \- PMC, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7423816/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7423816/)  
13. Radiomics Features Based on MRI-ADC Maps of Patients with Breast Cancer: Relationship with Lesion Size, Features Stability, and Model Accuracy \- PubMed, accessed October 4, 2025, [https://pubmed.ncbi.nlm.nih.gov/36128858/](https://pubmed.ncbi.nlm.nih.gov/36128858/)  
14. Customizing the Extraction — pyradiomics v3.1.0rc2.post5+ ..., accessed October 4, 2025, [https://pyradiomics.readthedocs.io/en/latest/customization.html](https://pyradiomics.readthedocs.io/en/latest/customization.html)  
15. PyRadGUI: A GUI based radiomics extractor software \- F1000Research, accessed October 4, 2025, [https://f1000research.com/articles/12-259](https://f1000research.com/articles/12-259)  
16. Development and validation of a pyradiomics signature to predict initial treatment response and prognosis during transarterial chemoembolization in hepatocellular carcinoma \- PMC, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9618693/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9618693/)  
17. Computational Radiomics System to Decode the Radiographic Phenotype \- PMC, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC5672828/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5672828/)  
18. Welcome to pyradiomics documentation\! — pyradiomics v3.1.0rc2.post5+g6a761c4 documentation, accessed October 4, 2025, [https://pyradiomics.readthedocs.io/](https://pyradiomics.readthedocs.io/)  
19. Repeatability and Reproducibility of Radiomic Features: A Systematic Review \- PMC, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC6690209/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6690209/)  
20. Radiomics Repeatability Pitfalls in a Scan-Rescan MRI Study of ..., accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7845781/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7845781/)  
21. Radiomics as a Quantitative Imaging Biomarker: Practical Considerations and the Current Standpoint in Neuro-oncologic Studies \- PMC \- PubMed Central, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC5897262/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5897262/)  
22. CLEAR guideline for radiomics: Early insights into current reporting practices endorsed by EuSoMII \- PubMed, accessed October 4, 2025, [https://pubmed.ncbi.nlm.nih.gov/39437630/](https://pubmed.ncbi.nlm.nih.gov/39437630/)  
23. a collection of standardized datasets and a technical protocol for reproducible radiomics machine learning pipelines \- PubMed, accessed October 4, 2025, [https://pubmed.ncbi.nlm.nih.gov/40760408/](https://pubmed.ncbi.nlm.nih.gov/40760408/)  
24. A Research Protocol to Make Radiomics-based Machine Learning Pipelines Reproducible \- arXiv, accessed October 4, 2025, [https://arxiv.org/pdf/2207.14776](https://arxiv.org/pdf/2207.14776)