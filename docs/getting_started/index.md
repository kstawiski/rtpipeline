# Getting Started: The rtpipeline Workflow

**rtpipeline** is designed to minimize the "friction" between getting data from a hospital system and having it ready for statistical analysis.

This guide explains the workflow from the perspective of a researcher or data scientist.

## The "Big Data" Radiotherapy Challenge

In a typical research project, you might export 500 patients from your Treatment Planning System (TPS). You immediately face these problems:

1.  **Structure Chaos:** Patient A has a structure named `Heart`, Patient B has `hrt`, Patient C has `Heart_new`. You can't easily analyze "heart dose" across the cohort.
2.  **Missing Organs:** The physician only contoured the target and proximal organs. They didn't contour the spleen or individual lung lobes because they weren't clinically relevant *at the time*, but now you need them for your toxicity study.
3.  **Inconsistent FOV:** Some CT scans are full-body, others are cropped to the spine. Calculating $V_{20\%}$ (volume receiving 20% dose) is meaningless if the total body volume denominators are different.
4.  **Data Locking:** The dose-volume histogram (DVH) data is locked inside proprietary DICOM files. You need a script to extract it into a spreadsheet.

**rtpipeline solves these by standardizing the entire cohort.**

---

## 1. Input: What do I need?

You just need the **raw DICOM export** from your TPS (Eclipse, RayStation, Monaco, etc.).

*   **Format:** Standard DICOM (`.dcm`).
*   **Structure:** No specific folder structure is required. You can dump 500 patients into one folder, or have nested subdirectories. The pipeline's **Organizer** module will sort them out based on unique identifiers (UIDs).
*   **Required Modalities:**
    *   **CT Images:** (Required)
    *   **RTSTRUCT:** (Optional, but usually present)
    *   **RTDOSE:** (Optional, for dosimetric analysis)
    *   **RTPLAN:** (Optional, for beam metadata)
    *   **MR/PET:** (Optional, will be registered and processed if linked)

---

## 2. Processing: What happens inside?

Once you feed the data (via [Web UI](webui.md) or [CLI](docker_setup.md)), the pipeline executes these research-oriented steps:

### A. Intelligent Organization
It scans thousands of files and groups them into **Courses**. A "Course" is a coherent set of Images + Plans + Doses that belong together (e.g., "Prostate Treatment 2023"). It handles complex cases like re-plans and boosts automatically.

### B. "Digital Twin" Generation (Segmentation)
Regardless of what structures came from the hospital, the pipeline runs **TotalSegmentator** on every CT.
*   **Result:** You get ~100 standardized anatomical structures (Heart, Lungs, Liver, Bones, Muscles, Vessels) for *every* patient.
*   **Research Value:** You now have a "Heart" contour for every patient, named exactly `heart`, even if the physician never drew it.

### C. Standardization (Cropping)
If enabled, the pipeline identifies anatomical landmarks (e.g., L1 vertebra) and **crops the CT** to a standard field-of-view.
*   **Research Value:** Your "Volume" and "Percentage Volume" metrics ($V_{x}$) become statistically comparable across the cohort because the denominators are standardized.

### D. Feature Extraction
It calculates thousands of metrics for both the **Original** (physician) structures and the **Auto-Generated** structures:
*   **Dosimetry:** $D_{mean}$, $D_{max}$, $D_{95\%}$, $V_{5Gy}$, $V_{20Gy}$, etc.
*   **Radiomics:** Shape, Texture (GLCM, GLRLM), Intensity statistics.

---

## 3. Output: Your "Tidy Data"

The pipeline doesn't just give you more images; it gives you **Tables**.

In the `_RESULTS/` folder, you will find master spreadsheets ready for pandas/R:

| File | Content | Research Question Examples |
| :--- | :--- | :--- |
| `dvh_metrics.xlsx` | Dose metrics for every structure. | "Does mean heart dose predict cardiotoxicity?" |
| `radiomics_ct.xlsx` | Image features for every ROI. | "Does tumor texture (Entropy) predict survival?" |
| `case_metadata.xlsx` | Technical scan parameters. | "Did different CT kernels affect our model?" |
| `qc_reports.xlsx` | Quality Control flags. | "Which patients should be excluded due to errors?" |

### Example Analysis Workflow

1.  **Export** 200 patients from Eclipse to a folder.
2.  **Drag & Drop** the folder into rtpipeline Web UI.
3.  **Wait** (or run overnight).
4.  **Download** `dvh_metrics.xlsx`.
5.  **Open in Python/R:**

```python
import pandas as pd

# Load the data
df = pd.read_excel("dvh_metrics.xlsx")

# Filter for the standardized Heart structure
heart_data = df[df['structure_name'] == 'heart']

# Analyze
print(heart_data['mean_dose_gy'].describe())
```

**That's it.** No parsing DICOM, no manual contouring, no matching file paths.

---

## Next Steps

*   **[Install & Setup](docker_setup.md):** Get the pipeline running on your machine.
*   **[Web UI Guide](webui.md):** See the interface in action.
*   **[Output Reference](../user_guide/output_format.md):** See the exact columns available in the output files.