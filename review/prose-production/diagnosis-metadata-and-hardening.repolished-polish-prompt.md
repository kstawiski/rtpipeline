You are the polisher. A draft already exists. Improve the prose so a clinician
can follow it on one pass. Do not redo the science and do not restructure the
argument.

Return the polished Markdown document. No preamble and no explanation of what
you changed. Do not use tools. Do not read or write any file. The draft is
inline below.

If, and only if, an instruction here cannot be satisfied without breaking
another one, or the draft contains a defect you are forbidden to repair, append
after the document a final block beginning with the single line POLISH-NOTE and
one bullet per issue. That block is for the integrator, is not part of the
document, and is removed before the document goes out. Do not use it to describe
ordinary edits.

READER. A consultant oncologist or pathologist who treats patients and reads
trial papers. Not a bioinformatician. Knows hazard ratios, confidence intervals,
staging and standard therapy. Does not know the vocabulary of any particular
assay, and will not read a sentence twice.

## Fidelity, which outranks everything else here

When a mechanics rule below cannot be applied without altering a number, an
interval, a citation marker, a quoted passage, a table, or an identifier,
preservation wins. Leave the text as it is and record it in a POLISH-NOTE.
Fidelity outranks every rule in this file, without exception.

- Every number, percentage, sample size, hazard ratio, confidence interval,
  p value, denominator, count and citation marker such as [1] survives EXACTLY.
  Do not round, convert, recalculate, or spell a numeral out as a word. 16 stays
  16. This protects evidence, not ordinary prose. Where the draft writes a small
  count as a numeral in running text, such as "the 2 objectives", write "two".
  That is a wording fix and changes no evidence.
- The en dash is REQUIRED in numeric ranges and confidence intervals. Keep every
  en dash exactly as it is. Never turn it into a hyphen or an em dash.
- Add no finding, claim, citation, result or number that is not already in the
  draft. Do not compute anything, including an absolute effect the draft does
  not state.
- You may add words that explain something the draft already contains, and only
  that. The test is whether a reader could point at the source of your addition
  somewhere in the draft. Expanding an abbreviation, glossing a term the draft
  defines or plainly implies, and naming items the draft lists elsewhere all
  pass. Supplying a definition the draft never gives, or items it never lists,
  is a prohibited addition however obvious it seems. When a cross-reference
  points at content that is not in the draft at all, leave the reference and
  record it in a POLISH-NOTE, because only the drafter can supply what is
  missing.
- One exception, because it adds no knowledge. An abbreviation established in
  ordinary clinical use for this reader may be expanded to its accepted full
  form at first use even when the draft never expands it. ctDNA becomes
  circulating tumour DNA, and pCR becomes pathological complete response. This
  covers expansion only. It does not license explaining what the entity is, what
  it measures, or the role it plays here, which stay the drafter's. An
  abbreviation this project invented is never covered.
- Remove no limitation, caveat or uncertainty. They are load-bearing. Where one
  limitation is repeated across several sections, it must still appear, stated
  in full at least once, and you may shorten the later restatements only where
  the shortened form still carries the boundary for the claim beside it. Where a
  claim would read differently without the full statement, the full statement
  stays. Never let a
  limitation disappear entirely, and never move one out of a section where its
  absence would change how a claim reads.
- Keep the noun a qualifier attaches to. "The same structure" and "the same
  content" are different claims, and swapping them can contradict the next
  sentence.
- Preserve the distinction between observation, association, prediction,
  mechanism, and causation. Never turn a nonsignificant difference into
  similarity, or an association into an effect.
- Keep every heading verbatim, in the same order, with the same numbering. Do
  not restyle heading capitalisation.
- Keep every citation marker in the sentence it arrived in. A marker binds to
  the claim its sentence makes, and a managed manuscript records that binding as
  a hash of the claim's bytes. Moving a marker to a neighbouring sentence, or
  widening its scope by merging its sentence with another, breaks the binding
  while leaving every character in place. Where a sentence carrying a marker
  must be split, the marker stays with the half that makes the cited claim.
- Do not pad. Cut throat-clearing, hedging that carries no uncertainty, and
  repetition that serves no section's purpose, so that the glosses you add are
  paid for. Ending slightly longer is acceptable when a gloss required it.
  Ending longer because nothing was cut is not.
- Where plain language and technical precision conflict, keep the precise term
  and rebuild the sentence around it.

## What to fix

**Name the thing instead of pointing at the document.** "The specification
listed in section 1.1.1" makes the reader hunt for the content. Replace it with
the actual items when they are few enough to state in place. Where the
referenced material is long, give a concise specific summary and keep the
pointer. What must not survive is a pointer standing in place of any content at
all. In the documents this standard was built for, this is usually the edit that
helps the reader most.

**Introduce every concept before the text depends on it.** At first material use
of a dataset, cohort, score, biomarker, protocol or uncommon abbreviation, say
what it is and the role it plays here. A gloss of five words in place beats a
glossary. Do not open a section on a project-specific entity the reader cannot
resolve.

**Open on the problem in ordinary language,** then the entity with its gloss,
then the finding. Do not open on a method, model, assay, workflow, internal
label or checklist. Two or three sentences of genuinely concept-introducing
background are correct. Background that introduces nothing is throat-clearing
and gets cut.

**Never let the text apologise for existing.** A document that concedes its own
case before anyone attacks it has argued against itself. Fix every instance:

- An opening that leads with what the work is not, does not cover, or cannot
  establish.
- Diminishers about the work's own scope. "Merely", "only", "just a pilot", "a
  modest attempt", "this preliminary study simply". Preliminary, underpowered
  and exploratory data support real claims about feasibility, signal, direction
  and rationale. State what they establish for the next step. In a grant,
  preliminary data exists to argue the proposed work deserves funding, and
  presenting it as evidence of insufficiency argues against the applicant.
- Stacked hedges no evidence requires. "May potentially suggest that it could be
  argued" carries no more uncertainty than "may" and far less conviction. Keep
  the one qualification the evidence demands.
- A limitation promoted above the finding it qualifies, or repeated in every
  section until the finding is buried.
- Asking the reader's indulgence. "To the best of our limited knowledge", "we
  acknowledge this is only", "while specific details are limited".

Before applying any of this, separate a result from an apology. A null, a
negative trial, a failed validation, an absent safety signal, and a design
boundary that governs interpretation are evidence, and they stay wherever the
drafter put them, including an opening. "The regimens did not separate on
survival" and "no grade 4-5 events occurred" are findings, not apologies. What
this section removes is language about the authors and their work, not
information about the result. When in doubt, keep it and record the doubt in a
POLISH-NOTE.

Test the opening alone. If what a reader who stopped there would take away is an
apology or a hedge about the work itself, repair the wording within those
sentences. Do not move the finding, because the opening's order belongs to the
drafter.

**Claim strength is not yours.** The drafter set how strong each claim is, and a
separate verifier checks it. Do not raise a claim and do not lower one. If a
claim looks stronger or weaker than the evidence around it, leave the claim
exactly as written, polish the sentence containing it, and name it in a
POLISH-NOTE at the end. Changing a proposition while preserving its numbers is
the specific failure this division of labour exists to prevent.

**Do not reorder the argument.** Which result leads, what the opening asserts,
and the order of sections and paragraphs are the drafter's decisions. You may
repair a sentence, split one, merge two adjacent ones inside the same paragraph,
and reorder clauses inside a sentence. Do not move a finding into or out of an
opening.

**Never change the paragraph structure.** Return exactly the paragraphs you were
given, in the same order, with the same breaks. Do not merge two paragraphs, do
not split one, do not add a paragraph, and do not delete one, even when a
paragraph is a single sentence and merging would read better. In a managed
manuscript the paragraphs are addressable units that citations and claims are
bound to by byte range, so a merged or split paragraph breaks those bindings
even though every word survives. If a paragraph break genuinely damages the
prose, leave it and record it in a POLISH-NOTE.

**Do not end a document differently than it ends.** If the draft closes on
"further research is needed", that is a drafting defect. Leave it and record it
in a POLISH-NOTE.

## Why some things are frozen

Two of the rules above look arbitrary unless you know what happens around you,
and knowing the mechanism will help you judge a case this prompt does not name.

A managed manuscript stores the document as an ordered list of units whose byte
ranges tile the text exactly, one unit per paragraph. Citations and claims are
bound to those ranges, and the binding is recorded as a hash of the claim's
bytes. So a merged or split paragraph breaks the tiling, and a marker that moves
between sentences breaks its binding, in both cases without a single word being
lost. A validator downstream reports these as stale or non-tiling, and the fix
is manual.

That is also why the minimum effective edit is the right instinct on any
sentence carrying a citation marker. Rewriting such a sentence forces its
citation to be re-verified against the full text it cites, which is real work
for someone. Improve the sentence when it genuinely needs it, and leave it alone
when it does not.

Nothing here asks you to preserve bad prose. It asks you to make the smallest
change that fixes the problem, and to keep the shape of the document while you
do it.

## Mechanics

- No sentence longer than 44 words.
- Zero colons and zero semicolons inside a sentence. This does not apply to a
  structured abstract label such as "Results:", to an author-year citation, to a
  URL or DOI, to a time such as 14:30, or inside a table, code block or quoted
  passage. The prohibition is deliberate and absolute for running prose, and a
  colon introducing a list is not an exception to it. Where a sentence wants a
  colon before several items, either end the sentence and let the items follow
  as their own sentences, or set them as a displayed list, which is not running
  prose. Do not reach for a semicolon or an em dash instead.
- The em dash is forbidden. Use a comma or end the sentence. Do not substitute
  parentheses, en dashes or hyphens as disguised sentence breaks.
- Every sentence ends in a full stop, question mark or exclamation mark. A line
  that names something rather than asserting something is a label, not a
  sentence, and owes no terminal punctuation. That covers slide titles, axis
  labels, table headers, row labels, units, equations and mathematical displays,
  figure and table captions that are labels rather than statements, structured
  abstract fields, list fragments, and quoted material that ends as its source
  ends. Never add punctuation inside a quotation to satisfy this rule.
- Straight quotes, not curly. No decorative emoji. Headings are exempt from
  restyling and stay exactly as the draft has them.
- Prefer a verb to a nominalisation. Write "we measure", not "measurement of".
- Prefer active voice and name the actor. Passive is correct where the actor is
  unknown or does not matter, which covers most of Methods.
- Do not meet the sentence ceiling by splitting a clause. A severed "either ...
  or" reads as broken prose even though it passes every count. Where a
  scientific relationship cannot be divided without changing what it asserts,
  keep the sentence whole even if it exceeds the ceiling, and record it in a
  POLISH-NOTE.

## Register

Ask what each sentence tells the reader about this work. Interpretation,
rationale, synthesis, a stated uncertainty, a hypothesis and a transition all
pass, because each carries something specific. What fails is a sentence that
would appear unchanged in another project's document. Cut that one, and never
cut a sentence merely because it is not a fact or a number.

Judge the role a word plays in context. Never rewrite a word only because it
appears on a list. These are frequently empty and usually have a concrete
replacement: delve into, underscore, highlight, showcase, elucidate, unveil,
leverage, utilise, harness, facilitate, necessitate, robust, comprehensive,
seamless, meticulous, crucial, pivotal, vital, intricate, multifaceted, nuanced,
groundbreaking, transformative, unprecedented, remarkable, notable, compelling,
valuable insights, landscape, realm, tapestry, interplay, paradigm, uncharted,
pave the way for, align with, resonate with. Also cut fancy ways to say "is"
(serves as, stands as, boasts, features), "not just X but Y", forced groups of
three, synonym cycling, "in order to" for "to", and "it is important to note
that".

## Patterns to remove

Each is a defect when it performs the empty role described, not whenever the
words appear.

**Content.**
1. Name-dropping. Do not list outlets or institutions without context. Name the
   relevant source and say what it reported.
2. Superficial `-ing` tails. "Highlighting the importance of", "ensuring
   robustness", "reflecting the complexity", "showcasing", "fostering". Cut them
   or replace with the sourced fact.
3. Vague attributions. "Experts believe", "studies suggest", "it is widely
   accepted". Name the source, or delete the claim.
4. Formulaic challenge framing. "Despite challenges, the field continues to
   advance." Give the specific obstacle and what follows from it.

**Language.**
5. Fancy ways to say "is". "Serves as", "stands as", "boasts", "features".
   Write "is" or "has" when that is the meaning.
6. "Not just X, but Y." State the point directly.
7. The rule of three. Do not force ideas into groups of three. Use the number
   the subject requires.
8. Synonym cycling. Pick one accurate term for an entity and repeat it. Varying
   the noun for elegance makes the reader wonder whether the referent changed.
9. False ranges. Do not write "from X to Y" when X and Y form no scale.

**Formatting.**
10. Boldface overuse. Do not bold every proper noun, acronym or key term.
11. Inline-header lists. Remove a bold label and colon when the label only
    repeats the line that follows. A lead-in stays when it names the item and
    the next sentence adds something.

**Communication artifacts.** These do not belong in a scientific document at
all.
12. Chatbot phrases. "I hope this helps", "let me know if", "of course",
    "certainly".
13. Sycophancy. "Great question", "you are absolutely right".
14. Cutoff disclaimers. "While specific details are limited".
15. Generic conclusions. "The future looks bright", "this remains an exciting
    area". Replace with the specific plan, finding, or consequence.

These are ordinary scientific English and stay when precise: however, within,
across, during, including, while, despite, findings, outcomes, analysis,
approach, research, role, limitations, primary, assess, observed, conducted,
identified, revealed, demonstrated, examined, reported.

Sterile writing is as recognisable as florid writing. In Methods, Results and
captions, voice is precision and rhythm only, so vary sentence length and never
vary certainty. In a Discussion, significance statement, cover letter or grant,
react to the finding rather than listing considerations neutrally, and use "we"
where the authors' judgment is the point.

If a passage is already right, leave it alone. Do not rewrite for the sake of
change.

DRAFT FOLLOWS.


# Diagnosis of metadata export and cohort hardening

## Decision and consequence

RTpipeline now identifies CT, RTPLAN, RTSTRUCT, RTDOSE, and RTRECORD objects from the DICOM Modality tag. File names no longer decide which objects enter the cohort metadata tables. This removes the silent failure that hid every Kopernik plan, structure set, and dose.

Plan-to-dose association now follows the RTDOSE reference to the RTPLAN SOPInstanceUID. The old ARIA filename key remains only as a fallback when a dose has no DICOM plan reference. The exporter raises `MetadataExportError` when it discovers RTPLAN objects but cannot extract any plan rows.

DFCI's empty `metadata.xlsx` is a second defect in the plan-to-dose join. It is not caused by missing RT objects in organized course directories. The exporter reads the source DICOM root, where 1,219 dose references resolve to indexed plans, but no plan and dose share the legacy filename core key.

The completed cohort directories were inspected without modification. They were not rebuilt. The reported post-change behavior therefore comes from synthetic DICOM regression tests and direct source censuses, not from new production workbooks.

## Failure mechanism

The previous exporter enumerated files with prefix tests for `RP`, `RS`, `RD`, `RT`, and `CT`. It then required a `.dcm` extension. The plan-to-dose merge also required both filenames to match an ARIA-specific `R[PD].<digits>.<description>.dcm` pattern and to produce the same core key.

These predicates made filenames act as clinical metadata. They also failed silently. A valid `RTPLAN_1.dcm` object was absent from `plans.xlsx` because its name did not start with `RP`. A valid `RTDOSE_1.dcm` could not join to its referenced plan because neither filename supplied the required core key.

The `RT` prefix created a separate misclassification in Kopernik. Its 12,193 matching files comprised 375 RTPLAN, 2,231 RTSTRUCT, 871 RTDOSE, 3,247 RTIMAGE, and 5,469 RTRECORD objects. The completed `fractions.xlsx` contained 12,193 rows because the prefix selected every one of these modalities. The revised exporter selects the 5,469 RTRECORD objects by Modality.

## Cohort evidence

The census read every `.dcm` file under the configured source root. It used `pydicom.dcmread` with `stop_before_pixels=True` and `specific_tags=[Modality]` for classification. Plan, dose, and structure headers were then read for reference and target censuses. Patient identifiers and individual file paths were not written to the aggregate result files.

| Cohort | Source DICOM files | RTPLAN by Modality | RTSTRUCT by Modality | RTDOSE by Modality | Legacy RP | Legacy RS | Legacy RD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Kopernik | 340,607 | 375 | 2,231 | 871 | 0 | 0 | 0 |
| DFCI | 415,562 | 1,561 | 3,548 | 1,233 | 1,561 | 3,548 | 1,233 |

The completed Kopernik export has 293,222 CT rows and 12,193 fraction rows. Its plan, structure-set, dosimetry, and merged metadata workbooks are absent. The measured Modality inventory matches the filename failure prediction and provides a positive inventory behind each absence claim.

The completed DFCI export has 352,706 CT rows, 1,561 plan rows, 3,548 structure-set rows, 1,233 dose rows, 13,997 fraction rows, and 0 merged metadata rows. Modality indexing found 352,707 CT objects. One CT object was therefore outside the legacy CT prefix predicate.

## DFCI metadata verdict

The DFCI source contains 1,561 distinct plan SOPInstanceUIDs and 1,233 dose references to plans. Of those references, 1,219 resolve to an indexed source plan. All 1,561 plan files and all 1,233 dose files produce a legacy core key, but the two key sets have no shared value.

This establishes an independent course-level metadata join defect. The evidence does not establish why 14 dose references do not resolve to an indexed plan. The revised join retains only explicit RTDOSE references that resolve to an indexed plan. It does not infer those 14 associations from labels, order, or approximate identifiers.

The separate organizer linkage defect may still leave RT objects absent from emitted course directories. That does not explain the empty merged metadata workbook because `export_metadata` reads `config.dicom_root`, not the emitted course directories. This task did not use course attachment as a substitute for source DICOM references.

## Metadata implementation

The exporter now builds one modality index for the requested source scope. Every candidate file is read without pixel data and with only the Modality tag requested. Filename prefixes remain as hints for common names, but the DICOM tag verifies every hint before classification.

Plans and doses carry internal SOP reference columns during assembly. The merge first joins each dose's `ReferencedRTPlanSequence` to the plan `SOPInstanceUID`. It uses the legacy ARIA core key only for doses with no explicit plan reference. Internal association columns are removed before workbook output.

A cohort with detected RTPLAN objects cannot end with an empty extracted plan table without an exception. Synthetic generic filenames produce populated plan, structure-set, and dosimetry tables. Synthetic ARIA filenames preserve the previous exported row content. A synthetic plan and dose with unrelated filenames associate through the dose's DICOM reference.

## Measured classification cost

Both measurements used 16 workers. The filename baseline and the tag index each performed a full source walk. The filename baseline performed only prefix checks after enumeration. The tag index read the Modality element without pixel data. These are wall-clock measurements on the named hosts and filesystems, not portable throughput guarantees.

| Cohort | Files | Filename scan | Modality-tag index | Added wall time | Ratio | Tag time per 1,000 files |
|---|---:|---:|---:|---:|---:|---:|
| Kopernik | 340,607 | 7.211 s | 325.810 s, or 5.43 min | 318.599 s | 45.18 | 0.957 s |
| DFCI | 415,562 | 5.834 s | 182.264 s, or 3.04 min | 176.429 s | 31.24 | 0.439 s |

The added scan is material but bounded at 3.04–5.43 minutes in these measurements. It prevents a false empty export. The two timings should not be compared as host performance because storage and cache state differed.

## F5 target definition

Production and the cohort probe now call one target-name function. It requires a GTV, CTV, or PTV token to begin at the start of the ROI name or after a non-alphanumeric boundary. This left boundary excludes embedded matches while preserving compact clinical names such as `PTVbt` and `CTVn`.

The function rejects target tokens preceded by the boolean-crop separator ` - `. It rejects names beginning with `marg` and leading-`z` helper names. `Pecherz - PTV`, `marg PTV2`, and `zPtvOpt` therefore do not satisfy the plan-and-dose target gate.

The full census confirmed the expected ceilings. All 230 DFCI and all 122 Kopernik plan-referenced structure sets that had target status under the permissive rule retain target status under the shared rule. No set lost target status in either cohort.

## F6 structure-set path ambiguity

The organizer now groups structure-set source paths by RTSTRUCT SOPInstanceUID before deciding whether the course has one structure set. Two paths carrying the same SOPInstanceUID resolve to one authoritative source identity. The structure set is copied, and CT selection still requires its referenced series.

A course with genuinely distinct RTSTRUCT identities fails the course gate instead of falling through to the largest CT series. The CT selector also returns `missing_reference` when reference-based selection is required but no structure source path is available.

The full census found 0 duplicate RTSTRUCT SOPInstanceUID groups among 3,548 DFCI structure files and 0 among 2,231 Kopernik files. This snapshot does not exercise the duplicate-copy branch in production. The synthetic duplicate-path regression test does.

## F7 CT-only cohorts

CT-only output remains available behind an explicit configuration choice. The default is off so failed RT linkage cannot masquerade as a valid CT-only cohort. A CT-only input now raises `CTOnlyCohortError` unless `organize.allow_ct_only_courses` is true or the CLI receives `--allow-ct-only-courses`.

When enabled, eligible volumetric CT series reach the existing CT-only course writer. Synthetic tests confirm both the default error and successful output under the explicit option. This restores a reachable path for diagnostic CT radiomics without weakening RT cohort handling.

## F8 label-text classifier

The unused legacy dose classifier and its private label-text helpers were deleted. The active reference and delivery-evidence classifier remains. A repository-wide Python source search found no `_classify_doses_legacy`, `_is_replan_text`, `_is_boost_text`, `_bboxes_overlap`, or `_prescription_similarity` symbol after the change.

A regression test asserts that the label-text classifier and helpers are not callable from the organizer module. No dose-selection rule was added that uses a plan label as evidence.

## Verification and limits

The final local suite under `/home/konrad/micromamba/envs/rtpipeline/bin/python` recorded 850 passed, 1 skipped, 0 failures, and 0 errors in 31.64 seconds. The JUnit record is `analysis/results/pytest-full.xml`. Synthetic DICOM tests contain no real patient data.

The source measurements are recorded in `analysis/results/metadata_modality_cost_kopernik.json` and `analysis/results/metadata_modality_cost_dfci.json`. Claim classification, calculations, and inference boundaries are recorded in `analysis/evidence_ledger.json`.

No production cohort was rebuilt, and no production output was modified. A rebuild remains necessary to measure the resulting workbook row counts and to evaluate the 14 unresolved DFCI dose references. The separate organizer linkage repair must also pass its own checks before cohort results are interpreted.
