Continue working on our pipeline. Learn how pipeline works now, what was already implemented and what requires fixes and improvments. You can examine current outputs (Data_Snakemake/) and logs (Logs_Snakemake/).

# DICOM-RT Processing Pipeline Specification

## Goal
Pipeline runs using snakemake (with config.yaml) and axillary rtpipeline package. Read README.md for setup instructions.
Process DICOM-RT data, analyze it, and produce analysis-ready tabular outputs for research purposes.

## Expected Output Directory Layout
For each patient and radiotherapy course:
- <output_root>/<patient_id>/<YYYY-MM>/
  - DICOM/
    - Original DICOMs for this course only: CT, RTSTRUCT, RTDOSE, RTPLAN.
  - DICOM_related/
    - Registration DICOMs and all images referenced by those registrations (CT/CBCT, MRI, PET).
  - NIFTI/
    - NIfTI converted from DICOM/ and DICOM_related/.
    - Use informative names produced by the pipeline, e.g. MIEDNICA_5.0_...
  - Segmentation_Original/
    - NIfTI of original RTSTRUCTs, organized so each segmentation maps unambiguously to its source NIfTI in NIFTI/.
  - Segmentation_TotalSegmentator/
    - Outputs from TotalSegmentator:
      - <model>.dcm (DICOM-RT struct for that model)
      - <model>--<structure>.nii.gz (per-structure masks)
  - dvh_metrics.xlsx
  - radiomics_ct.xlsx
  - fractions.xlsx (if RT*.dcm present)
  - RD.dcm, RP.dcm, RS.dcm (merged Original/Manual and automatic segmentations on CT)
  - metadata/ (any auxiliary metadata produced by the pipeline)
  - qc_reports/ (QC outputs)
- <output_root>/_RESULTS/ (results merged across all patients/courses)
  - dvh_metrics.xlsx (merged across all patients/courses)
  - radiomics_ct.xlsx (merged across all patients/courses)
  - fractions.xlsx (merged across all patients/courses)
  - ct_images.xlsx and if avilables: mr_images.xlsx / pet_images.xlsx (merged metadata across all patients/courses)
  - dosimetrics.xlsx
  - metadata.xlsx
  - qc_summary.xlsx
  - registrations.xlsx
  - plans.xlsx
  - structures.xlsx
  - Any other merged tabular outputs

Notes:
- <YYYY-MM> is the course_start_date in year-month format (UTC or a clearly defined timezone; be consistent).
- <nifti_name> must match exactly the source NIfTI filename (without extension) used for segmentation.
- Files with DVH and radiomics results must include all structures from both original RTSTRUCT and TotalSegmentator outputs.

Output directory shouldn't contain any symlinks or hardlinks; all files should be real files.

## Processing Rules
- The pipeline is based on orginial code in Code/ directory, refactored into rtpipeline/ package. It is now extentended with multiple new features.
- Due to incompatibilities, TotalSegmentator and pyradiomics must run in separate conda environments. Use snakemake conda integration to manage this.
- Convert to NIfTI once per unique image; reuse results across steps.
- Do not duplicate images in output directory. Skip processing of duplicated images. Deduplicate by SOPInstanceUID series/stack identity and content hash when available.
- Maintain traceability:
  - For every NIfTI in NIFTI/, record source DICOM SeriesInstanceUID and instance list.
  - For every segmentation, record the linked NIfTI source.
- Sum plans for multiple stage treatments (into a single RP and RD) per course (Code/04 Organize part 1.py)
- The structures that are on the edge of the image are probably cropped and volume or % DVH metrics are not reliable. Flag these in QC reports and flag those in the dvh_metrics.xlsx and radiomics_ct.xlsx with a boolean column "structure_cropped".
- Due to incompatibilities, TotalSegmentator (conda env: rtpipeline.yaml) and pyradiomics (conda env: rtpipeline-radiomics.yaml) must run in separate conda environments. Use snakemake conda integration to manage this.
- fractions.xlsx should contain detailed informtion about fractions (as much as can be extracted) and include path to source file. See how it was handled in pipeline prototype in Code/ and expand it.
- Metadata extraction form CT images should include key CT reconstruction and acquisition parameters that affect radiomics: Reconstruction kernel/filter - This is perhaps the most impactful parameter. Sharp/bone kernels vs. soft tissue kernels dramatically affect texture features.  Reconstruction algorithm - Filtered back projection (FBP) vs. iterative reconstruction (IR) methods (e.g., ASIR, SAFIRE, iDose) and their strength levels. Slice thickness - Typically 1-5mm; thinner slices provide better resolution but increase noise. Slice increment/spacing - Affects whether slices overlap or have gaps. Tube voltage (kVp) - Usually 100-140 kVp; affects image contrast and noise.  Tube current (mAs) - Affects radiation dose and image noise. Pitch - Particularly for helical/spiral CT. Rotation time. Matrix size - e.g., 512×512. Field of view (FOV) - Affects in-plane pixel spacing.  Pixel spacing/voxel dimensions (mm). Contrast Protocol (if applicable;  contrast agent type and concentration, injection rate and volume, scan delay/phase [arterial, portal venous, delayed]). Learn more about this from `Knowledge/CT Acquisition Parameters for Radiomic Analysis_ A Comprehensive Guide.md`. 
- Unparallelized DVH calculations radiomics is very slow. We should parallelize calculations keeping in mind resource limits. The only step of pipeline that should be run carefull is segmentation with TotalSegmentator (due to GPU). I should be able to set how many parallel workers can be used for TotalSegmentator and how many for other steps (dvh, radiomics etc.). Default should be to use all available cores for non-segmentation steps and 4 workers for segmentation.
- CT images: run model "total".
- MR images: run model "total_mr".
- Outputs go to Segmentation_TotalSegmentator/ with naming:
  - <model>.dcm (RTSTRUCT)
  - <model>--<structure>.nii.gz
- Finally, in output directory create _RESULTS/ directory, merge all created tabular files into analysis-ready datasets (xlsx) combining all processed patients.

## Testing
- Run tests in conda base env. Envs "rtpipeline" and "rtpipeline-radiomics" are already created and funcitional! Test the pipeline on Example_data/ using test.sh. You can examine current outputs (Data_Snakemake/) and logs (Logs_Snakemake/).
- Current example inputs/outputs are illustrative only; do not rely on them as ground truth.
- Assess results if they make sense clinically and technically. Create/update PROBLEMS.md to track any issues found.
- Fix issues iteratively, re-run tests as needed. Kill all snakemake commands before running again. Commands pile up and you cannot proceed without checking.
- Clean Data_Snakemake/ and Logs_Snakemake/ before re-running tests only if deep changes were made and it is necessary.


If eveything works as expected, write/update README.md and docs/ to reflect any changes in usage or configuration. README can link to docs/ for detailed documentation.
