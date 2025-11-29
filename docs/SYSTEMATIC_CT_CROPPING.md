# Systematic CT Cropping

## Overview

Percentage-based DVH metrics (like V95%, V20Gy) and volume-sensitive radiomics features are heavily influenced by the total scan volume. Variations in the scan field-of-view (e.g., one patient scanned from L3 to mid-thigh, another from T12 to knees) introduce noise that can obscure biological signals.

**Systematic CT Cropping** standardizes this variable. It uses anatomical landmarks (vertebrae, joints, organs) automatically detected by TotalSegmentator to crop every CT scan and its corresponding segmentations to consistent physical boundaries.

### The Problem

Consider two patients receiving the same absolute dose to the same absolute volume of the bladder:
*   **Patient A (Long Scan):** Total body volume in scan = 18,000 cm³. V20Gy = 500 / 18,000 = **2.8%**
*   **Patient B (Short Scan):** Total body volume in scan = 15,000 cm³. V20Gy = 500 / 15,000 = **3.3%**

Even though the treatment is identical, the metrics suggest a significant difference solely due to the scan length.

### The Solution

By cropping both scans to the same anatomical landmarks (e.g., "Superior: Top of L1", "Inferior: 10cm below femoral heads"), we substantially reduce field-of-view-related variation in the denominator:
*   **Patient A (Cropped):** Volume ≈ 12,000 cm³. V20Gy = **4.2%**
*   **Patient B (Cropped):** Volume ≈ 11,500 cm³. V20Gy = **4.3%**

This makes cohort-level comparison of percentage metrics more interpretable and statistically defensible, although residual anatomical differences (body habitus, prior surgery, segmentation variability) still need to be considered.

---

## Supported Anatomical Regions

The pipeline supports five standard cropping templates defined by stable bony landmarks.

### 1. Pelvis (Default)
*   **Target:** Prostate, Rectum, Bladder, Gyn
*   **Superior Boundary:** Superior aspect of **L1 vertebra**.
*   **Inferior Boundary:** 10 cm below the **Femoral Heads** (most inferior point).
*   **Code:** `region="pelvis"`

### 2. Thorax
*   **Target:** Lung, Esophagus, Heart, Breast
*   **Superior Boundary:** Superior aspect of **C7 vertebra** (or Lung apex + 2cm).
*   **Inferior Boundary:** Inferior aspect of **L1 vertebra** (or Diaphragm/Liver dome - 2cm).
*   **Code:** `region="thorax"`

### 3. Abdomen
*   **Target:** Liver, Pancreas, Kidney, Stomach
*   **Superior Boundary:** Superior aspect of **T12/L1 vertebra** (Diaphragm level).
*   **Inferior Boundary:** Inferior aspect of **L5 vertebra** (Pelvic inlet).
*   **Code:** `region="abdomen"`

### 4. Head & Neck
*   **Target:** H&N Cancer, Brain, Thyroid
*   **Superior Boundary:** Superior aspect of **Skull/Brain** + 2cm.
*   **Inferior Boundary:** Inferior aspect of **C7 vertebra** or **Clavicles** - 2cm.
*   **Code:** `region="head_neck"`

### 5. Brain
*   **Target:** GBM, Brain Mets
*   **Superior Boundary:** Superior aspect of **Brain** + 1cm.
*   **Inferior Boundary:** Inferior aspect of **Brain** - 1cm.
*   **Code:** `region="brain"`

---

## Configuration

Enable and configure cropping in your `config.yaml`:

```yaml
ct_cropping:
  enabled: true              # Master switch
  region: "pelvis"           # "pelvis", "thorax", "abdomen", "head_neck", "brain"
  
  # Margins extend the field of view beyond the strict landmark
  superior_margin_cm: 2.0    # e.g., 2cm above L1
  inferior_margin_cm: 10.0   # e.g., 10cm below femoral heads
  
  # Downstream integration
  use_cropped_for_dvh: true       # Calculate DVH on cropped volumes
  use_cropped_for_radiomics: true # Extract features from cropped volumes
  keep_original: true             # Keep original uncropped files
```

---

## How It Works (Technical)

The cropping process is implemented in `rtpipeline.anatomical_cropping` and runs as a dedicated pipeline stage (`crop_ct`) after segmentation.

1.  **Landmark Extraction:** The system scans TotalSegmentator outputs for key reference masks (e.g., `vertebrae_L1.nii.gz`, `femur_left.nii.gz`).
2.  **Boundary Calculation:** It computes the physical Z-coordinates (in mm) for the superior and inferior limits based on the selected `region` and configured margins.
    *   *Fallback Logic:* If primary landmarks (e.g., C7) are missing, it attempts to use secondary structures (e.g., Lung apex) and logs a warning.
3.  **Cropping:**
    *   **CT Image:** The primary planning CT NIfTI is cropped to the calculated Z-range.
    *   **Masks:** All segmentation masks (TotalSegmentator, Custom, etc.) are cropped to match the new CT geometry.
4.  **RTSTRUCT Generation:** A new DICOM RTSTRUCT file (`RS_auto_cropped.dcm`) is generated containing all cropped ROIs.
5.  **Metadata:** A `cropping_metadata.json` file is saved in the course directory, recording the exact boundaries used and the files modified.

### Output Files

*   **Cropped CT:** `NIFTI/ct_cropped.nii.gz`
*   **Cropped Masks:** `Segmentation_TotalSegmentator/*_cropped.nii.gz`
*   **Cropped RTSTRUCT:** `RS_auto_cropped.dcm`
*   **Metadata:** `cropping_metadata.json`

---

## Quality Control

Cropping is a destructive operation. The pipeline includes automatic QC to ensure no target structures are accidentally truncated.

1.  **Audit Report:** Run `rtpipeline audit-cropping` (or check the QC stage outputs) to see a summary of all cropped cases.
2.  **Warnings:** The logs will flag any case where the calculated boundaries fall outside the original scan extent (clamping).
3.  **Visual Check:** Use the "Visualize" stage to generate PNG snapshots of the cropped geometry.
