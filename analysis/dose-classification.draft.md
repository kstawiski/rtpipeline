# Dose classification and delivery accounting

## Purpose

The pipeline separates treatment intent from delivered treatment. Dose selection determines which referenced dose and plan objects belong to a course. Delivery accounting then uses treatment records to estimate what the patient received.

Free-text plan labels are not classification evidence. Dose-response analyses should use `delivered_dose_gy`. Treatment-intent analyses should use `total_prescription_gy`.

## Course dose selection

The classifier links RTDOSE to RTPLAN through `ReferencedRTPlanSequence`. It uses DICOM reference chains, prescription and fraction signatures, treatment-record support, dose summation type, frame of reference, and dose-grid geometry.

A `PLAN_SUM` object may represent a treatment planning system sum. Equivalent plan revisions are deduplicated by reference and prescription evidence. Distinct sequential phases remain separate and may be summed when the reference chain supports that interpretation.

Doses in incompatible frames or nonoverlapping grids are not silently combined. Ambiguous linkage fails closed or emits a warning for clinical review.

## Delivered dose

Treatment records are linked only to their referenced plan UIDs. A record referencing a plan absent from the export is counted and logged but is not assigned elsewhere.

Distinct fractions are counted by fraction number and treatment date when available. Treatment date is used when a fraction number is absent. This prevents several beam records from one treatment session being counted as several fractions.

Complete record-level delivered or calculated dose-reference values are preferred. When these are unavailable, the fallback multiplies the prescription by the smaller of 1.0 and delivered fractions divided by planned fractions.

`delivered_dose_gy` is null when the records cannot support an estimate. Unknown delivery is never converted to 0.0 Gy or to the prescription.

## Output fields

| Field | Meaning |
|---|---|
| `total_prescription_gy` | Selected treatment intent in Gy |
| `delivered_dose_gy` | Treatment-record estimate of delivered dose in Gy |
| `delivery_status` | `fully_delivered`, `partially_delivered`, `delivered_but_records_absent`, or `no_records_at_all` |
| `delivery_method` | Record dose-reference method, fraction-weighted fallback, mixed method, or unknown |
| `delivered_record_count` | Unique linked RTRECORD instances |
| `delivered_fraction_count` | Distinct treatment sessions inferred from the records |
| `planned_fraction_count` | Planned fractions across selected plans |
| `delivery_plan_details` | Plan-level prescription, delivery, method, and status |
| `unresolved_record_plan_uids` | Referenced plan UIDs absent from the indexed export |

## Plausibility warning

`max_total_dose_gy` configures the clinical plausibility threshold and defaults to 100.0 Gy. Separate warnings identify prescribed and delivered values above the threshold.

The warning does not replace reference-chain or delivery checks. It identifies a course for clinical review while preserving the value and the evidence behind it.

## Planning CT status

RT courses require a referenced planning CT. Course metadata records `planning_ct_status` and the RTSTRUCT-referenced CT series UIDs.

An absent referenced series is reported as `unresolved_reference`. A referenced series that resolves only to an excluded acquisition, such as a localizer, is reported as `classifier_excluded`. These courses are not emitted as complete planning-CT cases.

## Limits

Record-level dose values remain dependent on the exporting treatment system's DICOM semantics. The pipeline records the method and plan-level calculation so discordant values can be clinically adjudicated.

A null delivered dose must remain null in downstream analysis. Replacing it with prescription or 0.0 Gy would conflate unknown delivery with complete or absent treatment.
