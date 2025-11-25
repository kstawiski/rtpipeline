

# **CT Acquisition Parameters for Radiomic Analysis: A Comprehensive Guide**

## **The Foundation of Reproducibility in Radiomics: From Technical Variation to Clinical Translation**

### **Defining the Challenge: The Reproducibility Crisis in Quantitative Imaging**

Radiomics, the high-throughput extraction of quantitative features from medical images, holds immense promise for revolutionizing personalized medicine by providing non-invasive biomarkers for diagnosis, prognosis, and treatment response prediction.1 However, the translation of these promising research findings into routine clinical practice has been significantly impeded by a fundamental challenge: a lack of reproducibility.3 The very strength of radiomics—its ability to capture subtle, high-dimensional information beyond human visual perception—is also its greatest vulnerability. These quantitative features are exquisitely sensitive to variations in the entire imaging pipeline, from the initial image acquisition and reconstruction to subsequent processing and analysis steps.6

The core of this "reproducibility crisis" lies in the technical variability inherent in medical imaging. Image acquisition protocols and reconstruction parameters can vary widely, not only between different institutions but also within a single institution over time or across different scanners.6 This variability introduces non-biological fluctuations into the radiomic feature values, acting as a powerful confounding factor that can obscure or mimic true biological signals. Numerous studies have established that the effect of image acquisition and reconstruction settings on feature reproducibility is often the most significant source of variation, frequently exceeding the impact of inter-observer segmentation differences or minor variations in patient positioning during test-retest scans.11

This technical variability poses an existential threat to the validity of radiomic models. Without meticulous documentation, harmonization, and accounting for these parameters, a radiomic model risks being overfit not to the underlying tumor biology, but to the specific technical "fingerprint" of the scanner, software, and protocol used for the training data.12 Such a model would demonstrate excellent performance on internal data but would inevitably fail when applied to data from another institution or even a different scanner at the same institution, rendering it clinically non-generalizable and ultimately useless. Therefore, a rigorous and transparent approach to reporting and managing acquisition and reconstruction parameters is not merely a matter of good scientific practice; it is the absolute prerequisite for building robust, reliable, and clinically translatable radiomic tools. The evolution of community standards reflects this understanding, with a clear shift in focus from pure feature "discovery" to the establishment of methodological "rigor and reproducibility".15 Early frameworks like the Radiomics Quality Score (RQS) provided a foundational assessment of overall study design, often emphasizing high-level concepts like prospective validation or biological correlation.15 However, newer, more granular guidelines like the CheckList for EvaluAtion of Radiomics Research (CLEAR) and the METhodological RadiomICs Score (METRICS) place a much stronger emphasis on the transparent reporting of specific technical details, including image processing, segmentation methods, and adherence to standardization initiatives.5 This evolution signals a maturation of the field, recognizing that claims of clinical significance are scientifically unsound without a foundation of technical and methodological transparency.

### **Establishing a Lexicon: Stability, Repeatability, and Reproducibility**

To navigate the complexities of radiomic validation, it is essential to establish a clear and consistent lexicon for the key concepts of measurement integrity. While sometimes used interchangeably in broader contexts, in the field of quantitative imaging, these terms have distinct meanings that are critical for designing and interpreting studies.

**Repeatability** refers to the consistency of radiomic feature measurements under conditions that are as identical as possible. This is typically assessed in "test-retest" or "coffee break" style studies, where a patient undergoes two scans on the same scanner with the same protocol within a very short time frame (e.g., 15-20 minutes).11 The goal of repeatability studies is to measure the variability introduced by factors that cannot be perfectly controlled, such as minor changes in patient positioning, respiration, cardiac motion, and the inherent stochastic noise of the imaging system. High repeatability is the most basic requirement for a quantitative biomarker; if a feature is not stable under nearly identical conditions, it cannot be reliable under more variable ones.13

**Reproducibility**, the primary focus of this report, addresses a broader and more challenging question: the consistency of feature measurements under changed conditions.13 This encompasses variability introduced by using different CT scanners (even from the same manufacturer), different imaging protocols, different reconstruction algorithms, or acquiring data at different institutions in a multi-center trial.8 A feature that is reproducible is one whose value remains consistent despite these significant technical variations. Achieving high reproducibility is the central challenge for the clinical translation of radiomics, as any clinically useful biomarker must be applicable across the diverse range of equipment and protocols used in real-world healthcare settings.13

**Robustness** or **Stability** is an intrinsic property of a radiomic feature that describes its inherent sensitivity or insensitivity to variations in a specific imaging parameter.7 For example, a feature might be described as "robust to changes in tube current" if its value does not change significantly when only the mAs is varied. Conversely, a feature may be "non-robust to changes in slice thickness." Identifying which features are robust against which parameters is a critical step in developing reliable radiomic signatures. Models can be built using only features known to be robust, or alternatively, non-robust features can be used if their variability is addressed through harmonization techniques.

### **The Clinical Imperative: Why Technical Rigor is a Prerequisite for Translation**

The meticulous reporting and management of CT acquisition parameters is not an academic exercise in pedantry; it is a clinical imperative driven by a stark reality. Multiple phantom and patient studies have demonstrated that the magnitude of feature variability introduced by different scanners and protocols can be comparable to, or even exceed, the true biological variability observed between different patients' tumors.19 This finding has profound and troubling implications. It suggests that a poorly controlled radiomic study may inadvertently build a predictive model that is highly effective at differentiating a Siemens scanner from a GE scanner, rather than differentiating an aggressive tumor from an indolent one.12

When the technical "noise" is as loud as the biological "signal," any statistical association discovered is at high risk of being spurious. A model trained on such confounded data will fail when tested on an external dataset, not because the underlying biological hypothesis is wrong, but because the model learned to rely on technical artifacts that are not present in the new data. This is a primary reason why many promising single-center radiomic studies fail to generalize, hindering the field's progress and delaying the delivery of potentially valuable tools to clinicians and patients.3

The ultimate goal of radiomics is to build a bridge between standard-of-care medical imaging and the practice of personalized medicine.1 This bridge must be built on a foundation of trust, and that trust can only be earned through rigorous, transparent, and reproducible science. By understanding, controlling for, and meticulously reporting the technical parameters that influence feature values, researchers can ensure that their findings reflect true patient biology, thereby constructing a stable and reliable bridge capable of supporting the weight of clinical decision-making. Failure to address this foundational issue relegates radiomics to a collection of interesting but clinically isolated and ultimately irrelevant findings, preventing it from achieving its transformative potential.

## **Core Acquisition Parameters and Their Radiomic Impact**

The journey from an X-ray beam to a quantitative radiomic feature is a multi-stage process where decisions made at the earliest stage—image acquisition—have cascading and often irreversible effects on the final output. Understanding the physical principles behind each acquisition parameter is crucial for appreciating its impact on image quality and, subsequently, on the stability of radiomic features.

### **Radiation Flux Parameters: The Balance of Dose and Image Quality**

The quantity and energy of the X-ray photons used to create the image are fundamental determinants of image quality, directly influencing noise and contrast. These parameters are controlled primarily by the tube voltage and tube current.

#### **Tube Voltage (Kilovoltage Peak \- kVp)**

* **Physical Principles:** The kilovoltage peak () is the peak electrical potential applied across the X-ray tube. This voltage accelerates electrons from the cathode to the anode, and their subsequent deceleration at the anode target produces X-ray photons. The  determines the maximum energy (and thus the energy spectrum) of the photons in the X-ray beam.22 A higher  
   results in a "harder" beam with higher average photon energy. This increased energy enhances the beam's ability to penetrate the patient. From a physics perspective, higher energy photons are less likely to interact with tissue via the photoelectric effect, which is highly dependent on atomic number and is the primary source of subject contrast. Instead, Compton scatter, which is less dependent on tissue type, becomes the dominant interaction. This shift fundamentally alters the attenuation properties of different tissues as measured by the scanner.22  
* **Impact on Image Quality:** The primary effects of altering  are on image contrast and noise. Because higher  reduces the relative contribution of the photoelectric effect, it leads to lower intrinsic contrast between different soft tissues.22 Simultaneously, a higher  
   increases the number of photons reaching the detector for a given tube current, which can reduce image noise. However, this comes at the cost of increased radiation dose, as the dose increases approximately with the power of  in CT.22 Modern scanners often use automated protocols that adjust  
   to compensate for changes in  to maintain a target image quality or dose level. Changes in  also directly alter the measured CT numbers (Hounsfield Units, HU) of tissues, particularly those containing high atomic number elements like iodine or calcium, due to the energy-dependent nature of X-ray attenuation.24  
* **Impact on Radiomic Features:** As  directly influences the fundamental voxel intensity values (HU), it has a significant impact on all radiomic features derived from the intensity distribution. This includes first-order statistics (e.g., mean, skewness, kurtosis) and all categories of texture features (e.g., GLCM, GLRLM, NGTDM), which quantify the spatial arrangement of these intensities.26 While some phantom studies have suggested that intra-patient variability due to voltage changes may be less than inter-patient variability, indicating some features may retain prognostic power 29, the consensus is that  
   is a critical confounding variable that must be reported and accounted for.12 The effect is particularly pronounced in contrast-enhanced studies, where the HU value of iodinated tissue is highly sensitive to the  
   setting.

#### **Tube Current-Time Product (mAs)**

* **Physical Principles:** The tube current-time product (), which is the product of the tube current (in milliamperes, mA) and the exposure time (in seconds, s), directly controls the total number of X-ray photons produced by the tube for a given rotation.30 Unlike  
  , which controls the energy of the photons,  controls their quantity. A higher  means a greater photon flux is directed at the patient and, consequently, more photons reach the detector.  
* **Impact on Image Quality:** The primary and most direct effect of  is on image noise, specifically quantum mottle. Image noise is inversely proportional to the square root of the number of detected photons. Therefore, increasing the  increases the photon count and reduces the level of noise in the image, leading to a higher signal-to-noise ratio (SNR).30 This relationship is the basis for radiation dose optimization; clinicians aim to use the lowest  
   possible that still produces an image of sufficient diagnostic quality (the ALARA principle).24 Many modern CT scanners employ Automatic Exposure Control (AEC) systems, which automatically modulate the tube current (mA) based on the patient's size and attenuation profile to maintain a consistent level of image noise or quality across different body regions and patients.26  
* **Impact on Radiomic Features:** The influence of  on radiomic features is one of the most nuanced and debated topics in the literature. The evidence is mixed, and the impact appears to be highly context-dependent.  
  * Several studies, particularly those using phantoms or operating at very low dose levels, report a strong influence of tube current on feature reproducibility. As  is reduced, image noise increases, which can be misinterpreted as biological texture by feature extraction algorithms, leading to instability.6  
  * However, a crucial distinction has emerged from studies that compared materials of different intrinsic textures. The impact of \-induced noise is far more pronounced on features extracted from *homogeneous* materials (e.g., a uniform water or acrylic phantom) than from *heterogeneous*, tissue-like materials (e.g., cork, rubber particles, or actual tumors).29 In a perfectly uniform region, any variation in pixel intensity is due to noise, making texture features highly sensitive to changes in  
    . In a biologically heterogeneous tumor, the intrinsic texture variation arising from the underlying pathophysiology is often much greater in magnitude than the variations introduced by quantum noise. In this scenario, the biological "signal" can overwhelm the technical "noise," making many texture features surprisingly robust to changes in  within a typical clinical range.  
  * Further complicating the picture, other studies have found no clear correlation or influence of exposure on feature values, although some of these investigations used a relatively narrow range of  values, which may not have been sufficient to induce significant changes.20

### **Geometric Parameters: Defining the Voxel**

The voxel (volume element) is the fundamental unit of a 3D medical image. Its dimensions, determined by geometric acquisition and reconstruction parameters, define the spatial grid upon which the tumor's biology is sampled. Mismatches in these dimensions between scans can lead to profound differences in feature values.

#### **Reconstructed Slice Thickness**

* **Physical Principles:** This parameter defines the thickness of the reconstructed 2D image slice along the z-axis (the patient's longitudinal axis). It is a primary determinant of spatial resolution in the through-plane dimension.  
* **Impact on Image Quality:** Slice thickness involves a critical trade-off between spatial resolution and image noise. Thicker slices (e.g., 5 mm) average the signal from a larger volume of tissue. This volume averaging reduces the apparent image noise but also leads to the "partial volume effect," where the distinct signals from small structures or tissue boundaries are blurred together within a single voxel. This loss of detail can obscure fine anatomical or pathological features.33 Conversely, thinner slices (e.g., 1-2 mm) minimize partial volume effects and provide superior spatial resolution along the z-axis, allowing for more detailed visualization and segmentation of complex structures. However, because each voxel represents a smaller volume, fewer photons contribute to its signal, which can result in higher apparent image noise.34  
* **Impact on Radiomic Features:** Across the literature, reconstructed slice thickness is consistently identified as one of the most disruptive and influential parameters in the entire radiomics pipeline.19 Its impact is pervasive across nearly all feature families.  
  * **Shape and First-Order Features:** These feature categories are generally considered the most robust to variations in slice thickness. Shape features (e.g., Volume, Sphericity) are less affected as long as the segmentation is accurate, and first-order statistics (e.g., Mean, Skewness), which are calculated from the histogram of all voxel intensities in the volume, are less dependent on the specific spatial arrangement of those voxels.33  
  * **Texture Features:** These features are highly sensitive to slice thickness. Texture features are designed to quantify the spatial relationships between voxels. The partial volume averaging effect inherent in thicker slices acts as a low-pass spatial filter, effectively smoothing the image and erasing the fine-grained heterogeneity that many texture features are designed to measure. Consequently, features that capture fine textural patterns, such as those from the Neighborhood Grey Tone Difference Matrix (NGTDM), often show negligible reproducibility when compared across different slice thicknesses.36  
  * **Model Performance:** The choice of slice thickness also impacts the performance of the final radiomic model. Studies have shown that models built using features extracted from thin-slice images tend to be more stable and achieve better diagnostic or prognostic performance, likely because they capture more detailed biological information.34 This highlights that changing the slice thickness is not merely a technical adjustment but a fundamental alteration of the biological scale being investigated. A model trained on 1 mm slices might be learning a biomarker of micro-architectural disarray, while a model trained on 5 mm slices can only learn biomarkers related to larger-scale phenomena like necrosis. This underscores why resampling images to a common, isotropic voxel size is a critical and non-negotiable pre-processing step in any multi-protocol radiomics study.20

### **Spatial Resolution Parameters: Defining the Pixel**

#### **Pixel Spacing, Field of View (FOV), and Matrix Size**

* **Physical Principles:** These three parameters are inextricably linked and together define the in-plane (x-y) spatial resolution of the CT image. The reconstruction Field of View (FOV) is the diameter of the circular area that is reconstructed into the final image. The matrix size is the number of pixels that make up the image grid (e.g., 512x512). The pixel spacing (or pixel size) is calculated by dividing the FOV by the matrix size.38 For example, a 400 mm FOV reconstructed on a 512x512 matrix results in a pixel spacing of approximately 0.78 mm.  
* **Impact on Image Quality:** Pixel spacing directly determines the smallest object or detail that can be resolved in the x-y plane. A smaller pixel size, achieved either by using a smaller FOV for a targeted reconstruction or by using a larger matrix, provides higher spatial resolution and allows for more detailed visualization of anatomical structures.41 This is the principle behind High-Resolution CT (HRCT) of the chest, which often uses a targeted FOV to achieve very small pixel sizes for evaluating fine lung parenchymal details.  
* **Impact on Radiomic Features:** Consistent pixel spacing is of paramount importance for the reproducibility of radiomic features, especially texture features. Texture features are mathematical descriptors that inherently combine spatial information (the distance and orientation between pixels) with intensity information (the grey levels of those pixels).42 If the pixel size changes, the physical distance corresponding to a "neighboring" pixel also changes, which fundamentally alters the calculation of nearly all texture features.  
  * Studies have shown that variations in pixel size introduce large intra-patient variability, affecting the vast majority of radiomic features.41 One investigation found that 87% of all features (including shape, first-order, and texture) varied significantly across three different pixel size settings (0.8 mm, 0.4 mm, and 0.18 mm) derived from the same raw data.41  
  * The impact is pervasive across all feature classes. Shape features can be affected if the segmentation boundary is defined with greater or lesser precision due to resolution changes. First-order features can be subtly affected by changes in partial volume effects at the lesion edge. Texture features are the most profoundly affected, as their definitions are spatially dependent.  
  * Importantly, the choice of pixel spacing can also influence the clinical utility of the resulting radiomic signature. A study on pulmonary ground-glass nodules found that radiomic signatures built from HRCT images with a smaller pixel size performed significantly better at predicting invasiveness than signatures from images with a larger, standard pixel size.41 This suggests that higher spatial resolution allows the extraction of more diagnostically relevant information.

### **Intravenous Contrast Administration: A Dynamic Variable**

* **Physical Principles:** Iodinated contrast media are water-soluble compounds containing iodine atoms, which have a high atomic number (). Due to the photoelectric effect, high-Z elements strongly attenuate X-rays, particularly at lower photon energies. When injected intravenously, these agents circulate through the bloodstream and diffuse into the interstitial space of well-perfused tissues, temporarily increasing their density and making them appear brighter on CT images.43  
* **Impact on Image Quality:** The use of contrast is fundamental to many diagnostic CT protocols. It allows for the opacification of blood vessels (CT angiography), the enhancement of solid organs (e.g., liver, kidneys, pancreas), and the characterization of lesions based on their vascularity and enhancement patterns (e.g., washout in hepatocellular carcinoma).45 The timing of the scan relative to the injection is critical, with different "phases" (e.g., non-contrast, arterial, portal venous, delayed) designed to highlight different physiological processes.  
* **Impact on Radiomic Features:** The administration of intravenous contrast introduces a dynamic, time-dependent variable that has a profound effect on radiomic features. The impact is not simply a binary "contrast vs. no contrast" effect; it is a continuous function of time as the contrast agent perfuses into and washes out of the tissue of interest.  
  * A substantial number of radiomic features are highly dependent on the timing of contrast agent administration. A study using dynamic contrast-enhanced CT (dceCT) found that in highly perfused prostate tumors, nearly 39% of features showed significant alterations depending on the contrast phase. In less perfused lung tumors, the effect was still significant, with over 10% of features showing time-dependent changes.46  
  * First-order features, which directly measure the distribution of voxel intensities, are particularly sensitive. Features related to the mean, median, and overall energy of the signal were found to be among the most strongly affected by contrast influx.46  
  * This temporal dependency has direct consequences for the performance of radiomic models. The same study showed that the accuracy of a model designed to classify healthy versus tumor tissue was highly dependent on the contrast phase from which the features were extracted.46 This implies that if a study mixes images from different contrast phases without accounting for this variability, the resulting model may be unreliable.  
  * Therefore, for any radiomics study using contrast-enhanced CT, it is insufficient to simply state that contrast was used. For the results to be reproducible, the full injection and timing protocol must be reported, including the type and volume of contrast agent, the injection rate, and the specific scan delay time or the method used for bolus tracking.

## **The Pervasive Influence of Image Reconstruction**

Image reconstruction is the computational process that transforms the raw projection data (the sinogram) collected by the CT detectors into the cross-sectional images that are viewed and analyzed. The choices made during this step, specifically the algorithm and the filtering kernel, fundamentally shape the texture and noise characteristics of the final image, with profound consequences for radiomics.

### **Reconstruction Algorithms: Filtered Back Projection (FBP) vs. Iterative Reconstruction (IR)**

* **Technical Principles:** For decades, Filtered Back Projection (FBP) was the standard reconstruction algorithm in clinical CT. FBP is a computationally efficient analytical method that involves applying a high-pass filter to the projection data to correct for blurring and then "back-projecting" the filtered data to form the image. While fast, a key limitation of FBP is its amplification of noise, particularly in low-dose scans.47

  In recent years, FBP has been largely superseded by Iterative Reconstruction (IR) algorithms. IR is a more complex, model-based approach. It starts with an initial image estimate, simulates the forward projection process to see what the raw data would have looked like, compares this to the actual measured raw data, and then iteratively updates the image to minimize the difference. Modern IR algorithms also incorporate sophisticated statistical models of the imaging physics and noise properties, allowing them to preferentially reduce noise while preserving edge detail. Vendor-specific examples include Adaptive Statistical Iterative Reconstruction (ASiR) from GE Healthcare, which is mentioned in several studies.6  
* **Impact on Radiomic Features:** The choice of reconstruction algorithm has a strong influence on radiomic feature reproducibility because it directly alters the noise texture of the image. Studies that have systematically varied the level of iterative reconstruction (e.g., from 0% ASiR, which is equivalent to FBP, to 100% ASiR) have demonstrated a significant and progressive impact on the values of many radiomic features.6 IR algorithms change the magnitude and spatial characteristics of image noise compared to FBP. This can alter the output of texture feature calculations, which are designed to quantify these very patterns. Therefore, mixing images reconstructed with FBP and IR in the same dataset without harmonization is a major source of variability and can compromise the validity of a radiomic study. The specific IR algorithm name, its version, and the "strength" or "level" setting used must be reported.

### **Convolution Kernels: The Architect of Image Texture**

* **Technical Principles:** The convolution kernel, also known as the reconstruction kernel or filter, is a mathematical filter applied to the raw projection data during the reconstruction process. Its purpose is to modify the spatial frequency content of the image to achieve a desired balance between spatial resolution (sharpness) and noise.47 There is an inescapable trade-off:  
  * **Sharp (or "Hard") Kernels:** These are high-pass filters (e.g., "Lung," "Bone," "Edge" kernels). They enhance high spatial frequencies, which correspond to edges and fine details in the image. This results in images with higher spatial resolution, where small structures like lung bronchioles or trabecular bone are more clearly delineated. However, this amplification of high frequencies also amplifies high-frequency noise, making the images appear grainier or more "noisy".47  
  * **Smooth (or "Soft") Kernels:** These are low-pass filters (e.g., "Standard," "Soft Tissue" kernels). They suppress high spatial frequencies, which has the effect of reducing image noise and producing a smoother-looking image. The downside is a reduction in spatial resolution, causing fine edges and details to become blurred.47  
* **Impact on Radiomic Features:** The choice of reconstruction kernel is consistently cited as one of the most powerful and disruptive variables in the entire radiomics workflow.26 Its impact is so profound because it acts as an irreversible "texture-defining filter" at the earliest stage of image formation. The kernel's function is to fundamentally alter the image texture; therefore, it is unsurprising that radiomic features designed to quantify texture are extremely sensitive to its selection.  
  * Studies comparing features extracted from images reconstructed with sharp versus soft kernels from the same raw data show a dramatic loss of reproducibility. One study on pulmonary nodules found that while inter-reader agreement for features using the same kernel was high (Concordance Correlation Coefficient, CCC \= 0.92), the CCC between different kernels plummeted to 0.38. The percentage of reproducible features (CCC ≥ 0.85) dropped from 84.3% to just 15.2% when the kernel was changed.49  
  * Texture features and wavelet-based features (which also analyze spatial frequency content) are the most severely affected. In the aforementioned study, the CCC for texture features dropped from 0.88 to 0.61, and for wavelet features from 0.92 to 0.35, simply by changing the kernel.49  
  * Interestingly, the magnitude of the kernel's effect can depend on the intrinsic heterogeneity of the tissue being analyzed. One study found that the kernel had a greater impact on features extracted from the relatively homogeneous pancreas than on features from more heterogeneous pancreatic tumors. This suggests that in regions with strong inherent biological texture, that texture can partially resist the modifying effect of the kernel, whereas in homogeneous regions, the kernel's effect on noise and subtle texture becomes the dominant feature.50  
  * Given its overwhelming influence, the specific name of the reconstruction kernel (e.g., Siemens B30f, GE Standard) is one of the single most important parameters to report. In multi-center or multi-protocol studies, data should either be acquired with a harmonized kernel, or advanced harmonization techniques, such as deep learning-based image conversion, must be employed to mitigate this powerful source of variability.48

## **Systemic and Confounding Variables**

Beyond the user-selectable parameters on the CT console, several systemic variables related to the hardware and software environment can introduce significant variability into radiomic features. These factors are often harder to control, particularly in retrospective studies, but their impact must be acknowledged and, where possible, accounted for.

### **Inter-Scanner and Inter-Manufacturer Variability**

A central challenge in conducting large-scale, multi-center radiomics studies is the variability that arises from using different CT scanners.12 Even when primary acquisition parameters like

, slice thickness, and reconstruction kernel are nominally matched, images acquired on different scanner models, especially from different manufacturers, will not be identical. This "batch effect" is a well-documented phenomenon that can severely compromise the generalizability of radiomic models.19

This variability stems from fundamental design differences that are unique to each manufacturer and model line:

* **X-ray Tube and Detector Technology:** Differences in anode materials, focal spot sizes, detector materials (e.g., solid-state vs. gas), and detector geometry all influence the raw signal that is measured.  
* **Proprietary Reconstruction and Correction Algorithms:** Each manufacturer develops and implements its own proprietary software for image reconstruction, as well as for corrections like beam hardening and scatter reduction. An iterative reconstruction algorithm from one vendor is not the same as that from another, and even a "standard" FBP kernel from one manufacturer will have different frequency response characteristics than a "standard" kernel from a competitor.48  
* **Default Image Processing:** Scanners often apply additional, sometimes non-transparent, post-reconstruction filtering or smoothing to improve the visual appearance of images for radiologists. These processes can alter the quantitative data that radiomics relies on.

The cumulative effect of these differences is significant. Studies have shown that radiomic features can cluster by scanner manufacturer, indicating a systemic technical signature that can be stronger than the biological signal of interest.19 The number of features deemed robust and reproducible drops precipitously when data from multiple scanners are combined compared to data from a single scanner.12 This underscores the critical importance of reporting the scanner manufacturer and model for every image in a dataset. In the analysis phase, this information should be used to investigate for potential batch effects and to apply appropriate feature-level harmonization techniques like ComBat if necessary.19

### **The Hidden Variable: Software Versions and Implementation Dependencies**

A more subtle but increasingly recognized source of variability is the software itself—both the software running on the CT scanner and the software used to extract the radiomic features.

* **CT Console Software Version:** Manufacturers periodically update the operating software on their scanners. These updates can include changes to reconstruction algorithms, corrections, or default settings that may alter the quantitative characteristics of the resulting images. While this information can be difficult to obtain retrospectively, it is a potential confounding variable. One comprehensive study identified "Software Version" as a parameter whose impact on radiomic features could be quantitatively ranked, highlighting its importance.26  
* **Feature Extraction Software and Version:** The radiomics community has made significant strides in standardizing feature definitions, most notably through the Image Biomarker Standardization Initiative (IBSI).52 However, the actual implementation of these feature definitions in software packages can still vary. Minor differences in coding, mathematical libraries, or default parameter settings can lead to different feature values being calculated from the exact same image. Therefore, it is absolutely essential to report the name of the software package used for feature extraction (e.g., PyRadiomics, IBEX, CERR) and, critically, its specific version number.15 Adherence to IBSI standards is a key indicator of methodological quality, as it ensures that the calculated features conform to a community-vetted consensus, but even IBSI-compliant software can have version-to-version changes. Reporting the version number allows for exact replication of the feature extraction process.

## **Strategies for Ensuring Radiomic Integrity Across Heterogeneous Data**

Given the profound impact of technical variability, researchers must employ specific strategies to ensure the integrity and reliability of their findings, especially when dealing with data from multiple sources. These strategies can be broadly categorized into prospective harmonization, retrospective harmonization, and the use of phantoms for quality assurance.

### **Prospective Harmonization: The Gold Standard**

The most effective way to minimize technical variability is to prevent it from occurring in the first place. Prospective harmonization involves the development and strict enforcement of a standardized imaging protocol for all subjects in a study, particularly in the context of prospective clinical trials.8 This is the gold standard approach for ensuring data consistency.

A harmonized protocol goes beyond simply matching a few key parameters. It requires a detailed specification of the entire acquisition and reconstruction chain, including:

* **Scanner Requirements:** Defining acceptable scanner models (e.g., 64-slice or greater from specific vendors).  
* **Acquisition Parameters:** Fixing the , defining the AEC settings (e.g., target noise index), specifying the rotation time and pitch.  
* **Reconstruction Parameters:** Mandating a specific reconstruction algorithm (e.g., FBP or a particular IR algorithm at a set strength) and, most importantly, a specific reconstruction kernel name. Slice thickness and reconstruction increment must also be fixed.  
* **Contrast Administration:** Standardizing the type of contrast agent, concentration, volume (often weight-based), injection rate, and the exact timing of the scan phase(s).

Professional societies in related fields, such as the European Society for Therapeutic Radiology and Oncology (ESTRO) and the European Association of Nuclear Medicine (EANM), have established detailed guidelines for protocol standardization in radiation therapy planning and PET imaging, respectively.8 These frameworks serve as excellent models for the radiomics community. While achieving perfect harmonization can be challenging in multi-vendor environments, this approach maximally reduces technical noise, allowing for the most direct and reliable assessment of biological signals.

### **Retrospective Harmonization: Correcting for Inevitable Variability**

In many cases, particularly in discovery research, radiomics studies rely on retrospectively collected data from routine clinical care. In this common scenario, prospective harmonization is impossible, and data are inherently heterogeneous. Therefore, retrospective harmonization techniques are essential to mitigate the impact of technical variability before or after feature extraction.

* **Image-Level Harmonization:** These methods aim to modify the images themselves to make them more quantitatively similar.  
  * **Voxel Size Resampling:** This is the most common and essential image-level step. Since voxel dimensions (slice thickness and pixel spacing) have a massive impact on texture features, all images in a dataset must be resampled to a common, isotropic voxel size (e.g.,  mm) using a consistent interpolation algorithm (e.g., trilinear, B-spline) before feature extraction. This ensures that the spatial grid for feature calculation is consistent across all subjects.20  
  * **Advanced Methods:** More sophisticated techniques are emerging to address other sources of variability. For instance, deep learning models, such as Cycle-Consistent Generative Adversarial Networks (CycleGANs), have been successfully used to learn the transformation between images reconstructed with different kernels. These models can convert an image from a "sharp" kernel appearance to a "soft" kernel appearance, significantly improving the reproducibility of texture features between the two.48  
* **Feature-Level Harmonization:** These methods are applied after feature extraction to statistically adjust the feature values and remove unwanted variation from known sources, often referred to as "batch effects."  
  * **ComBat:** The most widely used feature-level harmonization technique in radiomics is ComBat, which was originally developed for correcting batch effects in genomics microarray data.51 ComBat uses empirical Bayes methods to model and adjust for batch effects (e.g., scanner model, institution) in the feature data, while attempting to preserve the biological variation of interest.13 It has been shown to be effective in reducing scanner-related variability in CT, PET, and MRI radiomic features.51 However, its application requires careful consideration. A critical caveat is that improved statistical harmonization does not always guarantee improved predictive performance of the final model.51 If the batch variable (e.g., scanner) is confounded with a true biological variable (e.g., patient population), ComBat may inadvertently remove the biological signal along with the technical noise. Therefore, the ultimate validation of any harmonization method is its impact on model performance in a completely independent test set.

### **The Essential Role of Phantom Studies**

Physical and digital phantoms are indispensable tools for the technical validation and quality assurance (QA) of a radiomics workflow.10 Phantoms are objects with known physical properties (e.g., size, shape, density, texture) that are scanned to assess the performance and stability of an imaging system.

Their role in radiomics is multifaceted:

* **Isolating Parameter Effects:** By scanning a phantom repeatedly while systematically varying a single acquisition or reconstruction parameter (e.g., scanning at multiple  levels while keeping all other parameters constant), researchers can precisely quantify the effect of that parameter on radiomic feature values. This allows for the identification of robust and non-robust features in a controlled environment where the underlying "biology" is static.6  
* **Multi-Scanner Comparisons:** Circulating the same phantom among multiple institutions and scanning it on different scanners is a powerful method for quantifying inter-scanner variability and assessing the feasibility of a multi-center study.19  
* **Software Validation and Calibration:** Phantoms, particularly digital phantoms with mathematically defined properties, are used to validate feature extraction software. The Image Biomarker Standardization Initiative (IBSI) used both digital and physical phantoms to establish consensus-based reference values for hundreds of features. Research groups can now use these same phantoms and reference values to test and calibrate their own software, ensuring it is IBSI-compliant and calculates features correctly.52

## **A Framework for Reporting: Adhering to Community Standards**

To combat the reproducibility crisis and foster a culture of transparency, the radiomics community has developed several key initiatives and tools to guide researchers in reporting their methods. Adherence to these standards is increasingly seen as a hallmark of high-quality, reliable research. A comprehensive report on a radiomics study should align with the principles and specific items outlined by these frameworks.

### **The Image Biomarker Standardization Initiative (IBSI): Standardizing the Language of Radiomics**

The IBSI is a foundational international collaboration aimed at standardizing the entire radiomics pipeline, from image processing to feature definition and calculation.52 Its primary goal is to ensure that when different software packages analyze the same image, they produce the same feature values. Adherence to IBSI standards is a critical step in separating variability caused by feature calculation from variability caused by image acquisition.15

IBSI provides detailed, mathematically precise definitions for 169 commonly used features and a standardized image processing workflow (e.g., specifying interpolation and discretization methods). Furthermore, it offers reporting guidelines to ensure that researchers provide all necessary information for others to replicate their work. For CT acquisition parameters, the IBSI reporting guidelines explicitly recommend documenting the following, among other items 59:

* **Tube Voltage:** The peak kilovoltage () of the X-ray source.  
* **Tube Current:** The tube current in milliamperes () or the tube current-time product ().

While the IBSI's direct recommendations for acquisition parameters are concise, its broader framework emphasizes the need for complete transparency in the entire process leading up to feature calculation.

### **CLEAR (CheckList for EvaluAtion of Radiomics research): A Guide for Authors and Reviewers**

The CLEAR checklist is a comprehensive, 58-item reporting guideline designed specifically for radiomics research. It serves as a step-by-step guide for authors to ensure they have included all essential methodological details in their manuscripts, and for reviewers to systematically assess the quality and completeness of a study.1

For CT acquisition and reconstruction, several CLEAR items are directly relevant:

* **Item \#16: Imaging protocol (i.e., image acquisition and processing):** This is the central item that requires a detailed description of how the images were obtained and prepared.  
* **Item \#20 (CT-specific details):** The official explanation accompanying the CLEAR checklist provides specific guidance for CT, mandating the reporting of:  
  * **Scanner Manufacturer and Model**  
  * **Tube Voltage ()**  
  * **Tube Current () or AEC settings**  
  * **Pitch**  
  * **Reconstructed Slice Thickness**  
  * **Beam Collimation**  
  * **Reconstruction Details:** This must include the **reconstruction algorithm** (e.g., FBP, ASiR) and the specific **convolution kernel** used.5  
* **Related Pre-processing Items:** The checklist also includes separate items for crucial pre-processing steps that are often linked to acquisition parameters, such as **Item \#21 (Image pre-processing details)**, **Item \#22 (Resampling method and its parameters)**, and **Item \#23 (Discretization method and its parameters)**.4

Using and submitting a completed CLEAR checklist as a supplementary file is a best practice that signals a commitment to transparency and reproducibility.

### **RQS and METRICS: Scoring the Methodological Quality**

Beyond checklists, the community has developed scoring systems to quantitatively assess the methodological quality of radiomics studies.

* **Radiomics Quality Score (RQS):** Introduced in 2017, the RQS was a pioneering tool consisting of 16 items, with a total possible score ranging from \-8 to \+36.15 It evaluates the entire research lifecycle, from study design to clinical validation. One of its key components is an item related to  
  **"Image protocol quality"**.17 While foundational, systematic reviews applying the RQS have often revealed low overall quality scores in the published literature, highlighting widespread methodological shortcomings.60  
* **METhodological RadiomICs Score (METRICS):** METRICS is a newer, more detailed 30-item scoring tool developed through a rigorous international Delphi consensus process.15 It places a stronger emphasis on granular methodological details, technical transparency, and reproducibility than the RQS.16  
  * The key item related to acquisition is **Item \#6: Imaging protocol with acquisition parameters**. This item specifically assesses whether the image acquisition protocol is reported with sufficient clarity to ensure the method's replicability.15

The state of reporting in the field remains a concern. Studies have shown that the voluntary use of these quality tools in publications is rare, and even when they are used, the self-reported scores may be inflated or inaccurate.62 This highlights the ongoing need for researchers, reviewers, and journal editors to champion and enforce these standards to elevate the quality of radiomics research.

The following table provides a consolidated view, mapping the critical CT parameters to the reporting requirements of these major community standards.

**Table 1: Navigating Reporting Standards for CT Image Acquisition and Reconstruction**

| Parameter | IBSI Recommendation | CLEAR Item(s) | METRICS Item(s) | Rationale for Reporting |
| :---- | :---- | :---- | :---- | :---- |
| **Scanner Manufacturer & Model** | Implied; essential for context | \#20 | Implied in \#6 | Different vendors/models have unique hardware and software, creating systemic batch effects. |
| **Tube Voltage ()** | Report peak kilo voltage | \#20 | \#6 | Affects photon energy, image contrast, and HU values, especially with contrast agents. |
| **Tube Current-Time Product ()** | Report tube current () | \#20 | \#6 | Controls photon quantity and image noise. Report specific  or AEC settings (e.g., Noise Index). |
| **Reconstructed Slice Thickness** | Report slice thickness | \#20 | \#6 | Profoundly impacts spatial resolution (z-axis) and partial volume effects, destabilizing texture features. |
| **Pixel Spacing / FOV / Matrix** | Report pixel spacing | \#20 | \#6 | Defines in-plane spatial resolution. Critical for all spatially dependent features (texture, shape). |
| **Reconstruction Algorithm** | Report algorithm type | \#20 | \#6 | FBP and different IR algorithms have distinct noise properties that strongly influence texture. Report name and strength level (e.g., ASiR 50%). |
| **Reconstruction Kernel** | Report kernel name | \#20 | \#6 | One of the most disruptive parameters. Fundamentally defines image sharpness and texture. Report specific vendor name (e.g., B50f sharp). |
| **Contrast Agent & Timing** | Report agent and phase | \#20 | \#6 | The presence, type, and timing of contrast injection dramatically alter HU values in a time-dependent manner. Report phase (e.g., portal venous) and delay. |
| **Feature Extraction Software** | Adherence to IBSI standard | \#25, \#28 | \#12 | Different software versions or implementations can yield different feature values from the same image. Report name and version number. |

## **Synthesis and Actionable Recommendations**

### **Summary of Key Findings and Best Practices**

The integrity of any radiomic study is fundamentally dependent on the quality and consistency of the underlying imaging data. This comprehensive review demonstrates that variations in CT acquisition and reconstruction parameters are a primary source of non-biological variability in radiomic features, posing a significant threat to the reproducibility and clinical translation of research findings.

The evidence consistently points to a hierarchy of influence among these parameters. **Reconstructed slice thickness**, **in-plane pixel spacing** (determined by FOV and matrix size), and the **reconstruction kernel** are unequivocally the most disruptive factors. They fundamentally alter the spatial scale and texture of the image data, causing profound instability in the majority of radiomic features, particularly those in the texture and wavelet families. Consequently, the most critical best practices for any multi-protocol or multi-center study are:

1. **Meticulously report** these three parameters for all images.  
2. **Employ retrospective harmonization**, at a minimum by resampling all images to a common isotropic voxel size to address slice thickness and pixel spacing variability. For variations in reconstruction kernels, more advanced methods like deep learning-based image conversion should be considered.

The effects of **tube voltage ()** and **intravenous contrast timing** are also highly significant, as they directly manipulate the voxel intensity (HU) values that form the basis of all first-order and texture features. Their protocols must be reported in detail. The influence of **tube current-time product ()** is more nuanced; while it is a major determinant of image noise, its impact on radiomic features appears to be greater in homogeneous phantoms than in heterogeneous tumors, where intrinsic biological texture may dominate. Nonetheless, it remains a key parameter to report, especially the settings of any Automatic Exposure Control system used.

To ensure research is transparent, reproducible, and of high methodological quality, the adoption of a standardized reporting framework is no longer optional but essential. All researchers should use the **CLEAR checklist** as a guide during manuscript preparation and include the completed checklist as supplementary material. Furthermore, striving to meet the quality criteria outlined in the **METRICS** scoring system will help ensure the methodological rigor required for the field to advance. Finally, all feature extraction should be performed using **IBSI-compliant software**, with the specific software name and version number clearly stated.

### **Comprehensive Summary Table of CT Parameters**

The following table synthesizes the findings of this report into a practical, quick-reference guide for researchers planning studies or preparing manuscripts. It links each critical parameter to its physical effect, its impact on radiomic features, and the essential reporting recommendation.

**Table 2: Summary of Critical CT Parameters and Their Impact on Radiomic Features**

| Parameter | Primary Effect on Image Quality | Radiomic Feature Classes Most Affected | Summary of Impact & Reporting Recommendation |
| :---- | :---- | :---- | :---- |
| **Tube Voltage ()** | Determines photon energy, subject contrast, and HU values. | First-Order, Texture | Significant impact on all intensity-based features. Crucial in contrast-enhanced scans. **Recommendation:** Report the specific  (e.g., 120 kVp). |
| **Tube Current-Time ()** | Controls photon quantity and quantum noise. | Texture (especially in homogeneous regions) | Impact is context-dependent, more pronounced in homogeneous tissues than in heterogeneous tumors. **Recommendation:** Report  or AEC settings (e.g., Noise Index, quality reference ). |
| **Slice Thickness** | Determines z-axis resolution; major source of partial volume effects. | Texture, Shape | Highly disruptive. Thicker slices smooth texture and reduce reproducibility. **Recommendation:** Report thickness (e.g., 3 mm). Resample to a common isotropic voxel size for analysis. |
| **Pixel Spacing** | Determines in-plane (x-y) spatial resolution. | Texture, Shape | Highly disruptive. Inconsistent spacing alters the spatial scale of texture calculations. **Recommendation:** Report pixel spacing (e.g.,  mm). Resample to a common voxel size for analysis. |
| **Reconstruction Algorithm** | Affects noise magnitude and texture (FBP vs. IR). | Texture | Significant impact. IR changes noise properties compared to FBP. **Recommendation:** Report algorithm name and settings (e.g., ASiR 50%). |
| **Reconstruction Kernel** | Controls sharpness vs. noise trade-off; fundamentally defines image texture. | Texture, Wavelet | Profound impact; one of the most disruptive variables. **Recommendation:** Report specific vendor kernel name (e.g., Siemens B30f, GE Standard). Harmonization is critical for multi-kernel studies. |
| **IV Contrast & Timing** | Increases tissue attenuation in a time-dependent manner. | First-Order, Texture | Dramatic impact on HU values. Feature values are highly dependent on the scan phase. **Recommendation:** Report use of contrast, agent type, and timing protocol (e.g., Portal Venous Phase, 70s delay). |
| **Scanner Manufacturer/Model** | Systemic differences in hardware and proprietary software. | All | Creates significant "batch effects" that can confound biological signals. **Recommendation:** Report manufacturer and model (e.g., Siemens Somatom Definition Flash). Account for this in statistical analysis. |

### **Critical Parameters and DICOM Tags**

#### **1\. Reconstruction Parameters**

* **Reconstruction Kernel/Filter** \- Most impactful on texture features. Sharp/bone kernels vs. soft tissue kernels dramatically affect radiomics.  
  * **DICOM:** (0018,1210) ConvolutionKernel  
  * **Also check:** (0008,103E) SeriesDescription  
* **Reconstruction Algorithm** \- FBP vs. iterative reconstruction (ASIR, SAFIRE, iDose, ADMIRE) and strength levels.  
  * **DICOM:** (0018,9315) ReconstructionMethod (often unpopulated)  
  * **Alternative:** Parse (0008,103E) SeriesDescription or manufacturer private tags  
* **Slice Thickness** \- Typically 1-5mm; affects resolution and noise.  
  * **DICOM:** (0018,0050) SliceThickness  
* **Slice Increment/Spacing** \- Determines overlap or gaps between slices.  
  * **DICOM:** (0018,0088) SpacingBetweenSlices  
  * **Calculate from:** (0020,1041) SliceLocation (consecutive slices)

#### **2\. Scan Acquisition Parameters**

* **Tube Voltage (kVp)** \- Usually 100-140 kVp; affects contrast and noise.  
  * **DICOM:** (0018,0060) KVP  
* **Tube Current (mAs)** \- Affects radiation dose and image noise.  
  * **DICOM:** (0018,1151) XRayTubeCurrent (mA), (0018,1152) Exposure (mAs), (0018,1153) ExposureInµAs  
* **Pitch** \- For helical/spiral CT acquisition.  
  * **DICOM:** (0018,9311) SpiralPitchFactor, (0018,9240) SingleCollimationWidth  
* **Rotation Time** \- Scanner gantry rotation period.  
  * **DICOM:** (0018,9305) RevolutionTime, (0018,1150) ExposureTime  
* **Matrix Size** \- Typically 512×512 for CT.  
  * **DICOM:** (0028,0010) Rows, (0028,0011) Columns  
* **Field of View (FOV)** \- Determines in-plane resolution.  
  * **DICOM:** Calculate from: (0028,0030) PixelSpacing × matrix dimensions. Sometimes: (0018,1100) ReconstructionDiameter  
* **Pixel/Voxel Spacing** \- Critical for feature calculation.  
  * **DICOM:** (0028,0030) PixelSpacing \[row spacing, column spacing\] in mm  
  * **Z-spacing:** Calculate from (0020,0032) ImagePositionPatient or use SliceThickness

#### **3\. Contrast Protocol (if applicable)**

* **Contrast Agent Details** \- Type, concentration, injection parameters.  
  * **DICOM:** (0018,0010) ContrastBolusAgent, (0018,1040) ContrastBolusRoute, (0018,1041) ContrastBolusVolume, (0018,1042) ContrastBolusStartTime, (0018,1043) ContrastBolusStopTime, (0018,1044) ContrastBolusTotalDose, (0018,1046) ContrastFlowRate  
* **Scan Phase/Timing** \- Arterial, portal venous, delayed, or non-contrast.  
  * **DICOM:** Often in (0008,103E) SeriesDescription

#### **4\. Scanner and Image Information**

* **Scanner Identification**  
  * **DICOM:** (0008,0070) Manufacturer, (0008,1090) ManufacturerModelName, (0018,1020) SoftwareVersions  
* **Image Type** \- Original vs. derived images.  
  * **DICOM:** (0008,0008) ImageType

### **Implementation in Python**

Python

import pydicom  
import numpy as np

def extract\_ct\_parameters(dicom\_path):  
    ds \= pydicom.dcmread(dicom\_path)  
      
    params \= {  
        \# Reconstruction  
        'kernel': getattr(ds, 'ConvolutionKernel', 'Unknown'),  
        'slice\_thickness': getattr(ds, 'SliceThickness', None),  
        'pixel\_spacing': getattr(ds, 'PixelSpacing', None),  
          
        \# Acquisition  
        'kvp': getattr(ds, 'KVP', None),  
        'mas': getattr(ds, 'Exposure', None),  
        'manufacturer': getattr(ds, 'Manufacturer', 'Unknown'),  
        'model': getattr(ds, 'ManufacturerModelName', 'Unknown'),  
          
        \# Matrix  
        'rows': ds.Rows,  
        'columns': ds.Columns,  
          
        \# Contrast  
        'contrast\_agent': getattr(ds, 'ContrastBolusAgent', 'None'),  
        'series\_description': getattr(ds, 'SeriesDescription', '')  
    }  
      
    return params

### **Final Pre-Submission Checklist for Radiomics Researchers**

Before submitting a manuscript based on CT radiomics, researchers should perform a final quality check by asking the following questions:

1. **Protocol Transparency:** Have I clearly described the imaging protocol for every cohort (training, validation, testing)?  
2. **Scanner Details:** Have I reported the manufacturer and model for all CT scanners used in the study?  
3. **Core Parameters:** Have I explicitly stated the ,  **(or AEC settings)**, **reconstructed slice thickness**, and **reconstruction kernel** used?  
4. **Spatial Resolution:** Have I reported the **pixel spacing**, or the **FOV and matrix size** from which it can be derived?  
5. **Voxel Resampling:** If images had variable slice thickness or pixel spacing, have I described the **isotropic voxel size** I resampled to and the **interpolation algorithm** used?  
6. **Contrast Enhancement:** If contrast-enhanced scans were used, have I detailed the **contrast administration protocol**, including the **scan phase and timing**?  
7. **Software and Standards:** Have I stated the **name and version number** of the software used for feature extraction, and have I confirmed its **IBSI compliance**?  
8. **Reporting Guidelines:** Have I completed a **CLEAR checklist** and included it as supplementary material to demonstrate adherence to community reporting standards?

By systematically addressing these points, researchers can significantly enhance the quality, transparency, and reproducibility of their work, thereby contributing more effectively to the ultimate goal of translating radiomics into a powerful tool for clinical decision support.

#### **Works cited**

1. Evaluating Radiomics Research Reporting Assessment Tools to Improve Quality and Generalizability \- AI Blog \- ESR | European Society of Radiology %, accessed October 4, 2025, [https://www.myesr.org/ai-blog/evaluating-radiomics-research-reporting-assessment-tools-to-improve-quality-and-generalizability/](https://www.myesr.org/ai-blog/evaluating-radiomics-research-reporting-assessment-tools-to-improve-quality-and-generalizability/)  
2. Radiomics: The bridge between medical imaging and personalized medicine | Request PDF \- ResearchGate, accessed October 4, 2025, [https://www.researchgate.net/publication/320198316\_Radiomics\_The\_bridge\_between\_medical\_imaging\_and\_personalized\_medicine](https://www.researchgate.net/publication/320198316_Radiomics_The_bridge_between_medical_imaging_and_personalized_medicine)  
3. Responsible Radiomics Research for Faster Clinical Translation ..., accessed October 4, 2025, [https://jnm.snmjournals.org/content/59/2/189](https://jnm.snmjournals.org/content/59/2/189)  
4. (PDF) CheckList for EvaluAtion of Radiomics research (CLEAR): a step-by-step reporting guideline for authors and reviewers endorsed by ESR and EuSoMII \- ResearchGate, accessed October 4, 2025, [https://www.researchgate.net/publication/370544223\_CheckList\_for\_EvaluAtion\_of\_Radiomics\_research\_CLEAR\_a\_step-by-step\_reporting\_guideline\_for\_authors\_and\_reviewers\_endorsed\_by\_ESR\_and\_EuSoMII](https://www.researchgate.net/publication/370544223_CheckList_for_EvaluAtion_of_Radiomics_research_CLEAR_a_step-by-step_reporting_guideline_for_authors_and_reviewers_endorsed_by_ESR_and_EuSoMII)  
5. CheckList for EvaluAtion of Radiomics research (CLEAR): a step-by-step reporting guideline for authors and reviewers endorsed by ESR and EuSoMII, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10160267/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10160267/)  
6. Influence of CT acquisition and reconstruction parameters on ..., accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC5812985/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5812985/)  
7. The impact of the variation of imaging parameters on the robustness of Computed Tomography radiomic features: A review \- PubMed, accessed October 4, 2025, [https://pubmed.ncbi.nlm.nih.gov/33930766/](https://pubmed.ncbi.nlm.nih.gov/33930766/)  
8. Making Radiomics More Reproducible across Scanner and Imaging ..., accessed October 4, 2025, [https://www.mdpi.com/2075-4426/11/9/842](https://www.mdpi.com/2075-4426/11/9/842)  
9. Influence of CT acquisition and reconstruction parameters on radiomic feature reproducibility \- PubMed, accessed October 4, 2025, [https://pubmed.ncbi.nlm.nih.gov/29487877/](https://pubmed.ncbi.nlm.nih.gov/29487877/)  
10. (PDF) Influence of CT acquisition and reconstruction parameters on radiomic feature reproducibility \- ResearchGate, accessed October 4, 2025, [https://www.researchgate.net/publication/323201457\_Influence\_of\_CT\_acquisition\_and\_reconstruction\_parameters\_on\_radiomic\_feature\_reproducibility](https://www.researchgate.net/publication/323201457_Influence_of_CT_acquisition_and_reconstruction_parameters_on_radiomic_feature_reproducibility)  
11. Stability and reproducibility of computed tomography radiomic features extracted from peritumoral regions of lung cancer lesions \- PMC, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC6842054/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6842054/)  
12. The impact of the variation of imaging factors on the robustness of Computed Tomography Radiomic Features: A review | medRxiv, accessed October 4, 2025, [https://www.medrxiv.org/content/10.1101/2020.07.09.20137240.full](https://www.medrxiv.org/content/10.1101/2020.07.09.20137240.full)  
13. Stability of Radiomic Models and Strategies to Enhance Reproducibility \- ResearchGate, accessed October 4, 2025, [https://www.researchgate.net/publication/378188772\_Stability\_of\_Radiomic\_Models\_and\_Strategies\_to\_Enhance\_Reproducibility](https://www.researchgate.net/publication/378188772_Stability_of_Radiomic_Models_and_Strategies_to_Enhance_Reproducibility)  
14. Mastering CT-based radiomic research in lung cancer: a practical guide from study design to critical appraisal | British Journal of Radiology | Oxford Academic, accessed October 4, 2025, [https://academic.oup.com/bjr/article/98/1169/653/8084852](https://academic.oup.com/bjr/article/98/1169/653/8084852)  
15. Exploring radiomics research quality scoring tools: a comparative analysis of METRICS and RQS \- Diagnostic and Interventional Radiology, accessed October 4, 2025, [https://dirjournal.org/articles/exploring-radiomics-research-quality-scoring-tools-a-comparative-analysis-of-metrics-and-rqs/dir.2024.242793](https://dirjournal.org/articles/exploring-radiomics-research-quality-scoring-tools-a-comparative-analysis-of-metrics-and-rqs/dir.2024.242793)  
16. Exploring radiomics research quality scoring tools: a comparative analysis of METRICS and RQS \- Diagnostic and Interventional Radiology, accessed October 4, 2025, [https://www.dirjournal.org/pdf/beb8919b-f013-4ea1-b1c8-40332e840fe1/articles/dir.2024.242793/366-369.pdf](https://www.dirjournal.org/pdf/beb8919b-f013-4ea1-b1c8-40332e840fe1/articles/dir.2024.242793/366-369.pdf)  
17. Exploring radiomics research quality scoring tools: a comparative analysis of METRICS and RQS \- PMC, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11589524/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11589524/)  
18. METhodological RadiomICs Score (METRICS): a quality scoring tool for radiomics research endorsed by EuSoMII \- National Institutes of Health (NIH) |, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10792137/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10792137/)  
19. Measuring CT scanner variability of radiomics features \- PMC, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC4598251/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4598251/)  
20. Influence of gray level discretization on radiomic feature stability for ..., accessed October 4, 2025, [https://www.tandfonline.com/doi/full/10.1080/0284186X.2017.1351624](https://www.tandfonline.com/doi/full/10.1080/0284186X.2017.1351624)  
21. Incorporating radiomics into clinical trials \- UCL Discovery, accessed October 4, 2025, [https://discovery.ucl.ac.uk/10121409/1/Fournier2021\_Article\_IncorporatingRadiomicsIntoClin.pdf](https://discovery.ucl.ac.uk/10121409/1/Fournier2021_Article_IncorporatingRadiomicsIntoClin.pdf)  
22. Kilovoltage peak | Radiology Reference Article | Radiopaedia.org, accessed October 4, 2025, [https://radiopaedia.org/articles/kilovoltage-peak](https://radiopaedia.org/articles/kilovoltage-peak)  
23. Why does patient dose increase with higher tube potential (kVp) in CT while the opposite is true for radiography? \- Quora, accessed October 4, 2025, [https://www.quora.com/Why-does-patient-dose-increase-with-higher-tube-potential-kVp-in-CT-while-the-opposite-is-true-for-radiography](https://www.quora.com/Why-does-patient-dose-increase-with-higher-tube-potential-kVp-in-CT-while-the-opposite-is-true-for-radiography)  
24. Impact of Tube Voltage on Radiation Dose (CTDI) and Image Quality at Chest CT Examination \- Semantic Scholar, accessed October 4, 2025, [https://pdfs.semanticscholar.org/dda2/6997da75602eb75973a616bc409c13c33439.pdf](https://pdfs.semanticscholar.org/dda2/6997da75602eb75973a616bc409c13c33439.pdf)  
25. The Impacts of Vertical Off-Centring, Localiser Direction, Phantom Positioning and Tube Voltage on CT Number Accuracy: An Experimental Study \- MDPI, accessed October 4, 2025, [https://www.mdpi.com/2313-433X/8/7/175](https://www.mdpi.com/2313-433X/8/7/175)  
26. Rank acquisition impact on radiomics estimation (AcquIRE) in chest CT imaging: A retrospective multi-site, multi-use-case study \- PMC, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10872259/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10872259/)  
27. Effect of CT image acquisition parameters on diagnostic performance of radiomics in predicting malignancy of pulmonary nodules of different sizes \- PMC, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9136691/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9136691/)  
28. Evaluation of radiomics feature stability in abdominal monoenergetic photon counting CT reconstructions \- PMC, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9665022/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9665022/)  
29. CT Texture Analysis Challenges: Influence of Acquisition and ..., accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7277097/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7277097/)  
30. Milliampere-seconds (mAs) | Radiology Reference Article | Radiopaedia.org, accessed October 4, 2025, [https://radiopaedia.org/articles/milliampere-seconds-mas](https://radiopaedia.org/articles/milliampere-seconds-mas)  
31. Effect of tube current on computed tomography radiomic features \- Johns Hopkins University, accessed October 4, 2025, [https://pure.johnshopkins.edu/en/publications/effect-of-tube-current-on-computed-tomography-radiomic-features](https://pure.johnshopkins.edu/en/publications/effect-of-tube-current-on-computed-tomography-radiomic-features)  
32. Effect of tube current on computed tomography radiomic features \- PubMed, accessed October 4, 2025, [https://pubmed.ncbi.nlm.nih.gov/29403060/](https://pubmed.ncbi.nlm.nih.gov/29403060/)  
33. (PDF) Impact of slice thickness on reproducibility of CT radiomic ..., accessed October 4, 2025, [https://www.researchgate.net/publication/376144731\_Impact\_of\_slice\_thickness\_on\_reproducibility\_of\_CT\_radiomic\_features\_of\_lung\_tumors](https://www.researchgate.net/publication/376144731_Impact_of_slice_thickness_on_reproducibility_of_CT_radiomic_features_of_lung_tumors)  
34. Effects of slice thickness on CT radiomics features and models for staging liver fibrosis caused by chronic liver disease \- ResearchGate, accessed October 4, 2025, [https://www.researchgate.net/publication/360457244\_Effects\_of\_slice\_thickness\_on\_CT\_radiomics\_features\_and\_models\_for\_staging\_liver\_fibrosis\_caused\_by\_chronic\_liver\_disease](https://www.researchgate.net/publication/360457244_Effects_of_slice_thickness_on_CT_radiomics_features_and_models_for_staging_liver_fibrosis_caused_by_chronic_liver_disease)  
35. Reproducibility of CT Radiomic Features within the Same Patient: Influence of Radiation Dose and CT Reconstruction Settings | Radiology \- RSNA Journals, accessed October 4, 2025, [https://pubs.rsna.org/doi/abs/10.1148/radiol.2019190928](https://pubs.rsna.org/doi/abs/10.1148/radiol.2019190928)  
36. Impact of slice thickness on reproducibility of CT radiomic features of lung tumors, accessed October 4, 2025, [https://impressions.manipal.edu/open-access-archive/8637/](https://impressions.manipal.edu/open-access-archive/8637/)  
37. Impact of slice thickness on reproducibility of CT radiomic features of lung tumors-Bohrium, accessed October 4, 2025, [https://www.bohrium.com/paper-details/impact-of-slice-thickness-on-reproducibility-of-ct-radiomic-features-of-lung-tumors/949257148175483089-28326](https://www.bohrium.com/paper-details/impact-of-slice-thickness-on-reproducibility-of-ct-radiomic-features-of-lung-tumors/949257148175483089-28326)  
38. Intrinsic dependencies of CT radiomic features on voxel size and number of gray levels, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC5462462/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5462462/)  
39. Validation of A Method to Compensate Multicenter Effects Affecting CT Radiomics, accessed October 4, 2025, [https://pubs.rsna.org/doi/pdf/10.1148/radiol.2019182023](https://pubs.rsna.org/doi/pdf/10.1148/radiol.2019182023)  
40. Intrinsic dependencies of CT radiomic features on voxel size and number of gray levels | Request PDF \- ResearchGate, accessed October 4, 2025, [https://www.researchgate.net/publication/312639643\_Intrinsic\_dependencies\_of\_CT\_radiomic\_features\_on\_voxel\_size\_and\_number\_of\_gray\_levels](https://www.researchgate.net/publication/312639643_Intrinsic_dependencies_of_CT_radiomic_features_on_voxel_size_and_number_of_gray_levels)  
41. Dependence of radiomic features on pixel size affects the diagnostic ..., accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7934291/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7934291/)  
42. Harmonizing the pixel size in retrospective computed tomography radiomics studies | PLOS One \- Research journals, accessed October 4, 2025, [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0178524](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0178524)  
43. CT and X-ray Contrast Guidelines \- Radiology | UCSF, accessed October 4, 2025, [https://radiology.ucsf.edu/patient-care/patient-safety/contrast/iodinated](https://radiology.ucsf.edu/patient-care/patient-safety/contrast/iodinated)  
44. Patient Safety \- Contrast Material \- Radiologyinfo.org, accessed October 4, 2025, [https://www.radiologyinfo.org/en/info/safety-contrast](https://www.radiologyinfo.org/en/info/safety-contrast)  
45. The role of iodinated contrast media in computed tomography structured Reporting and Data Systems (RADS): a narrative review, accessed October 4, 2025, [https://qims.amegroups.org/article/view/116543/html](https://qims.amegroups.org/article/view/116543/html)  
46. Contrast Agent Dynamics Determine Radiomics Profiles in ..., accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11049400/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11049400/)  
47. Impact of CT convolution kernel on robustness of radiomic features ..., accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8010534/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8010534/)  
48. Inter-vendor harmonization of CT reconstruction kernels using unpaired image translation \- arXiv, accessed October 4, 2025, [https://arxiv.org/pdf/2309.12953](https://arxiv.org/pdf/2309.12953)  
49. Deep Learning–based Image Conversion of CT Reconstruction Kernels Improves Radiomics Reproducibility for Pulmonary Nodules or Masses | Radiology \- RSNA Journals, accessed October 4, 2025, [https://pubs.rsna.org/doi/abs/10.1148/radiol.2019181960](https://pubs.rsna.org/doi/abs/10.1148/radiol.2019181960)  
50. Can CT Image Reconstruction Parameters Impact the Predictive Value of Radiomics Features in Grading Pancreatic Neuroendocrine Neoplasms? \- MDPI, accessed October 4, 2025, [https://www.mdpi.com/2306-5354/12/1/80](https://www.mdpi.com/2306-5354/12/1/80)  
51. Improved generalized ComBat methods for harmonization of radiomic features, accessed October 4, 2025, [https://www.researchgate.net/publication/365210711\_Improved\_generalized\_ComBat\_methods\_for\_harmonization\_of\_radiomic\_features](https://www.researchgate.net/publication/365210711_Improved_generalized_ComBat_methods_for_harmonization_of_radiomic_features)  
52. Image Biomarker Standardisation Initiative: IBSI, accessed October 4, 2025, [https://theibsi.github.io/](https://theibsi.github.io/)  
53. The Image Biomarker Standardization Initiative: Standardized Quantitative Radiomics for High-Throughput Image-based Phenotyping \- PubMed, accessed October 4, 2025, [https://pubmed.ncbi.nlm.nih.gov/32154773/](https://pubmed.ncbi.nlm.nih.gov/32154773/)  
54. Incorporating radiomics into clinical trials: expert consensus endorsed by the European Society of Radiology on considerations for data-driven compared to biologically driven quantitative biomarkers \- PubMed Central, accessed October 4, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8270834/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8270834/)  
55. Enhancing Radiomics Reproducibility: Deep Learning-Based Harmonization of Abdominal Computed Tomography (CT) Images \- MDPI, accessed October 4, 2025, [https://www.mdpi.com/2306-5354/11/12/1212](https://www.mdpi.com/2306-5354/11/12/1212)  
56. Influence of feature calculating parameters on the reproducibility of CT radiomic features: a thoracic phantom study \- Quantitative Imaging in Medicine and Surgery, accessed October 4, 2025, [https://qims.amegroups.org/article/view/47513/html](https://qims.amegroups.org/article/view/47513/html)  
57. The image biomarker standardisation initiative — IBSI 0.0.1dev documentation, accessed October 4, 2025, [https://ibsi.readthedocs.io/](https://ibsi.readthedocs.io/)  
58. \[1612.07003\] Image biomarker standardisation initiative \- arXiv, accessed October 4, 2025, [https://arxiv.org/abs/1612.07003](https://arxiv.org/abs/1612.07003)  
59. Radiomics reporting guidelines and nomenclature — IBSI 0.0.1dev ..., accessed October 4, 2025, [https://ibsi.readthedocs.io/en/latest/04\_Radiomics\_reporting\_guidelines\_and\_nomenclature.html](https://ibsi.readthedocs.io/en/latest/04_Radiomics_reporting_guidelines_and_nomenclature.html)  
60. Evaluating the impact of the Radiomics Quality Score: a systematic review and meta-analysis. \- UroToday, accessed October 4, 2025, [https://www.urotoday.com/recent-abstracts/urologic-oncology/prostate-cancer/157563-evaluating-the-impact-of-the-radiomics-quality-score-a-systematic-review-and-meta-analysis.html](https://www.urotoday.com/recent-abstracts/urologic-oncology/prostate-cancer/157563-evaluating-the-impact-of-the-radiomics-quality-score-a-systematic-review-and-meta-analysis.html)  
61. a quality scoring tool for radiomics research endorsed by EuSoMII, accessed October 4, 2025, [https://portal.findresearcher.sdu.dk/files/253607514/Open\_Access\_Version.pdf](https://portal.findresearcher.sdu.dk/files/253607514/Open_Access_Version.pdf)  
62. Self-reported checklists and quality scoring tools in radiomics: a meta-research \- PubMed, accessed October 4, 2025, [https://pubmed.ncbi.nlm.nih.gov/38180530/](https://pubmed.ncbi.nlm.nih.gov/38180530/)