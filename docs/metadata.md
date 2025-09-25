# Metadata

## Global (under `outdir/Data/`)
- `plans.xlsx` (RP): plan label/date/intent/approval, CT series/study, patient ID/DOB/sex/PESEL (if present).
- `dosimetrics.xlsx` (RD): plan ref, CT series/study, patient ID.
- `structure_sets.xlsx` (RS): approval, CT series/study, patient ID, available_structures.
- `fractions.xlsx` (RT*): fraction id/number/date/time, plan ref, machine, verification/termination status, delivery time.
- `CT_images.xlsx` (CT): PatientID, Study/Series UIDs, Series/Instance numbers.
- `metadata.xlsx`: merged RP↔RD by core key from filename, with RS by patient ID.

## Per-case (per course)
Written to `case_metadata.json` and `case_metadata.xlsx`.

- Identification/paths: patient_id, course_key, course_dir, rp/rd/rs/rs_auto/seg paths, ct_dir, ct_study_uid.
- Plan summary: plan_name/date/time/intent, planned_fractions, approval_status (date/time), reviewer (date/time/name), machine.
- Clinicians: PhysiciansOfRecord, ReferringPhysicianName, PerformingPhysicianName, OperatorsName.
- Patient demographics: name, birth date, sex, weight (kg), height (m), BMI, age at plan.
- Prescriptions: DoseReferenceSequence entries and per‑ROI mapping (ReferencedROINumber → ROI_Name, TargetPrescriptionDose).
- Beam geometry: stats for gantry/collimator/couch; beam_info per beam (type, radiation, energy, control points, gantry span, meterset, IsArc); total_meterset; technique_inferred.
- Course timing: course_start_date, course_end_date, fractions_count (from RT treatment records matched to plan UID).
- Dose grid: DoseUnits, DoseType, Rows, Columns, NumberOfFrames, PixelSpacing, DoseSummationType.
- CT summary: manufacturer, model, institution, KVP, convolution kernel, slice thickness, pixel spacing.
- Structures: `structures` (manual RS ROI names), roi_count, ptv_count, ctv_count, oar_count.

## Merged case metadata
- `outdir/Data/case_metadata_all.{xlsx,json}` contains one row per course with the fields above.

