# Clinical prescription evidence

The clinical treatment register is an optional cohort-owned evidence source. It never replaces a resolved DICOM prescription. It can resolve a DICOM-unknown course, corroborate DICOM, or record a disagreement for review.

## Configuration

Enable the source only for a cohort that owns the required workbook.

```yaml
clinical_prescription_records:
  enabled: true
  path: /absolute/path/to/rt_treatments.xlsx
```

Omitting the setting, or setting `enabled` to `false`, preserves DICOM-only behavior. This is the expected DFCI configuration because no equivalent DFCI treatment register is available in this project.

## Accepted grammar

The parser reads `Opis leczenia`. It accepts a named site followed by one of these forms.

```text
[do dawki] TOTAL Gy/p.ref po PER_FRACTION Gy
[do dawki] TOTAL Gy w N frakcjach po PER_FRACTION Gy
[do dawki] TOTAL Gy/p.ref po PER_1 Gy (N_1 frakcji) i PER_2 Gy (N_2 frakcji)
```

A site normally follows `na obszar` or `obszar`. The first site may follow a named technique such as IMRT, VMAT, or SBRT. Later sites may follow `oraz` or `i`.

Each site remains a separate evidence object. The parser stores the site name, total dose, fraction count, phase doses, phase counts, self-check result, and exact site clause.

## Arithmetic checks

A single unstated fraction count is inferred only when `total / per_fraction` is a positive integer. A stated count must satisfy `total = count * per_fraction`.

Every phase in a multiphase expression must state its fraction count. The sum of all phase products must equal the stated total. Any failed check refuses the complete record.

## Refusal cases

The parser refuses these states.

- Empty `Opis leczenia`.
- No total dose.
- No per-fraction dose.
- No named treatment site.
- Nonpositive dose or fraction values.
- A nonintegral implied fraction count.
- A stated fraction count that does not reproduce the total.
- A multiphase expression with any missing phase count.
- An unsupported extra dose expression.
- Dose-bearing clauses that cannot be separated safely.
- Duplicate site names in one record.
- Any partial parse that leaves dose-bearing text unexplained.

Distinct totals for different sites are valid per-site evidence. They do not become one course scalar. A DICOM-unknown course remains unresolved with `MULTISITE_DISTINCT_TOTALS`.

## Record matching

Matching first requires the same patient identifier and ICD-10 C67 record. A candidate must have a valid treatment start and end date.

The matcher accepts only direct temporal evidence. Full containment of DICOM treatment dates ranks first. Treatment-date overlap ranks next. Course-window overlap follows. An RTPLAN date inside the clinical window is the weakest accepted basis.

No nearest-date fallback exists. A tie at the strongest evidence level returns `AMBIGUOUS_RECORD_MATCH`. Missing dates, no overlap, and multiple equally supported records all remain unresolved.

A clinical record can provide a missing course scalar only when all DICOM treatment dates fall inside its window, or when a DICOM RTPLAN date falls inside that window and no stronger evidence exists. Partial treatment overlap and course-window overlap can support corroboration or disagreement, but return `INSUFFICIENT_TEMPORAL_EVIDENCE_FOR_RESOLUTION` when DICOM has no course total.

## Provenance and outcomes

Every configured course records the workbook path, workbook SHA-256, sheet, row count, patient, course, match decision, DICOM snapshot, parser version, and outcome.

A matched row also records its row number, record identifier, treatment dates, diagnosis fields, parsed field name, and exact `Opis leczenia` text. The course-contract validator reparses this text and repeats the arithmetic and DICOM-precedence checks.

The possible adjudication outcomes are `RESOLVED_FROM_CLINICAL_RECORD`, `CORROBORATED_DICOM`, `DISAGREES_WITH_DICOM`, and `UNRESOLVED`.

Clinically resolved courses use `COURSE_TOTAL_CLINICAL_RECORD`. Their delivery contract and DVH rows carry `prescription_source = CLINICAL_RECORD`. DICOM-resolved courses carry `prescription_source = DICOM`.

For a DICOM replacement-chain classification, a clinical multiphase record changes prescription semantics only when every clinical phase uniquely matches a delivered RTPLAN by fraction count and dose per fraction. The published classification becomes `TWO_FRACTIONATION_PHASES`. The original replacement-chain classification, per-plan planned counts, and zero-delivery plans remain visible as DICOM evidence. A completed one-to-one phase binding changes delivery status to fully delivered and records the clinical-plus-DICOM method.
