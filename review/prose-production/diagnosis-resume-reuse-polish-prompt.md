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


# Resume should reuse current content

The clinical processing question is whether a resumed course needs a new segmentation model run. The answer must depend on the current planning CT and the artefacts derived from it, not on whether a stage sentinel happens to exist. This distinction matters because a missing RS_auto can coexist with a complete, valid bank of TotalSegmentator masks.

After the automatic RTSTRUCT geometry fix, 121 DFCI courses and 1 Kopernik course reportedly contained all 117 TotalSegmentator masks but no RS_auto. The reported inventory was 16,424 DFCI mask files and 234 Kopernik mask files. The missing derived structure set did not show that the masks were missing or invalid [1].

Clearing `.segmentation_done` to rebuild RS_auto made TotalSegmentator run again. Recent courses on s1 produced 55 model-run log lines and took 322 to 494 seconds per course while 117 masks were already present. At roughly seven minutes per course, repeating the model across 230 DFCI courses would take about 27 hours. Repeating it across 122 Kopernik courses would take about 14 hours. These are reported cohort calculations, not a new campaign-time measurement [1].

The defect was a stage-level completion marker being used as an artefact-level decision. `.segmentation_done` records that a stage published completion. It does not prove that each output exists, is readable, is current, or corresponds to the inputs selected for the present course. The sentinel remains a workflow marker. It is not evidence that an expensive model run is necessary [1, 2].

The revised segmentation decision starts with the authoritative course contract. The contract identifies the planning CT series and its NIfTI. A reusable mask manifest must then identify that same series, match the current planning-NIfTI hash, and list a complete set of readable masks whose physical geometry matches the current planning NIfTI. If any identity, inventory, readability, geometry, or freshness check is uncertain, the model runs again. This is deliberately fail-closed because reusing a mask from another planning CT is more harmful than recomputing a valid mask [2].

Legacy mask banks receive only a narrow upgrade path. The sidecar series identity and NIfTI hash must match the current planning CT and NIfTI. When a sidecar contains the NIfTI content-generation timestamp, the mask manifest must be later than that timestamp. Older sidecars do not use their metadata write time as evidence. They require filesystem ordering that shows the mask manifest was written after the current NIfTI. An uncheckable ordering rejects reuse [2].

A refreshed metadata sidecar therefore does not make valid legacy masks stale merely because its own `generated_at` is newer than the mask manifest. When the NIfTI is reused, its sidecar preserves the existing generation timestamp. A newly converted NIfTI receives a separate `nifti_generated_at` value that describes content generation rather than a later metadata refresh [2].

The stage now separates model outputs from derived outputs. A complete current mask set is reused without a TotalSegmentator invocation. If the model RTSTRUCT or RS_auto is missing, it is rebuilt from the validated masks. Current and legacy binary-mask naming conventions are normalized in one pass. If both names identify one ROI, the current name wins, so a stale duplicate cannot create a second contour. A genuine duplicate receives a progressing deterministic suffix [2, 3].

An existing RS_auto that is invalid, stale, or bound to another planning CT is removed from the consumer-visible path before replacement begins. This happens before CT loading, segmentation selection, geometry checking, ROI addition, or saving. If rebuilding fails, failure cleanup also removes any incomplete replacement, so no RS_auto remains available to downstream stages [3].

`_standard_rtstruct_sources` no longer treats file presence as correspondence evidence. It offers RS_auto only when the file is a readable, non-empty DICOM RTSTRUCT, its referenced CT series set exactly matches the contracted planning CT series, and no contracted CT, NIfTI, or segmentation dependency is newer or unreadable. Missing identity, conflicting identity, unreadable content, stale dependencies, and validation errors all exclude the automatic source [3].

Removal is safer than retaining or quarantining doubtful clinical geometry in the course tree. Downstream stages already tolerate a missing best-effort automatic structure set, while a wrong structure set can corrupt every derived ROI without revealing the substitution. RS_custom retains its separate revocation rule. Production calls still require a loadable authoritative course contract [3].

Each course records the decision in `metadata/segmentation_resume.json`. When RS_auto is rejected, the record includes `rejected_artifact` with the `removed` action, the `RS_auto.dcm` path, and the exact rejection reason. The same decision records whether rebuilding then succeeded or failed. Model runs are recorded only after the attempt resolves. A QC-ineligible model is `skipped`, and a failed invocation is `failed`, rather than either being reported as rebuilt. A reuse-only run leaves the mask manifest generation time unchanged. Its audit update is separate from mask provenance. Earlier skipped-model information is retained when a manifest is rewritten [2, 3].

The direct measurement for this revision used the production mask-currentness predicate on Kopernik course 431057, course 2020-07. The course had 117 existing masks occupying 23,201,947 bytes. Five repeated checks had a median wall-clock time of 8.25 seconds in this run. Compared with the measured 322 to 494 seconds for a model rerun, validating this complete mask set saved 313.75 to 485.75 seconds per applicable course. The denominator was one course with a complete, current mask set and missing derived output. The measurement is recorded by `analysis/resume_reuse_evidence.py` in `analysis/resume-reuse-measurement.json` [4].

This revision applies the content boundary to TotalSegmentator masks and the RS_auto and RS_custom derived structure sets. It does not claim that every stage sentinel is now safe for content-based reuse. The campaign registry still lists `.organized`, `.segmentation_done`, `.custom_models_done`, `.crop_ct_done`, `.dvh_done`, `.radiomics_done`, `.radiomics_robustness_done`, and `.qc_done`. The remaining stages still use their sentinel workflow gates. Extending them requires stage-specific output inventories and dependency identities, so that work is deferred rather than represented as complete. In particular, the NTCV robustness chain remains outside this repair. Its reported scale is 81 perturbations across 5 target ROIs, or 405 perturbed extractions per course, at roughly 0.9 hours per course [1, 5].

Resume completeness is unchanged. A conversion that never completed remains ineligible for a resume skip. A checkpoint without a required adjudication remains rejected. The existing completeness tests and the new content-reuse tests provide the focused regression boundary. The operational next step is to rerun organize so each course receives its authoritative contract. Then inspect the per-course decisions before expensive stages proceed. Reuse only positively identified current outputs. Rebuild anything missing, stale, mismatched, incomplete, or uncertain [1, 6].

## Sources

[1] Version-3 repair packet at `.orchestrator/runs/resume-reuse-20260829T183929Z/packets/repair-material-findings.json` and the task packet in the same run directory. It supplies the measured cohort evidence and the updated course-contract context.

[2] `rtpipeline/segmentation.py`. It implements mask completeness, planning-CT correspondence, legacy provenance, selective derived-artefact rebuilding, and resume-decision records.

[3] `rtpipeline/auto_rtstruct.py` and `rtpipeline/custom_structures_rtstruct.py`. They implement RTSTRUCT correspondence, binary-mask normalization, stale-output revocation, and custom-structure currentness.

[4] `analysis/resume_reuse_evidence.py` and `analysis/resume-reuse-measurement.json`. They produce and record the read-only course measurement used for the per-course saving calculation.

[5] `workflow/scripts/campaign_ledger.py`. It is the campaign sentinel registry used to state which stages remain outside this content-based repair.

[6] `tests/test_resume_reuse_content.py` and `tests/test_resume_completeness.py`. They test selective reuse, planning-CT mismatch, incomplete masks, derived-output handling, audit integrity, and existing resume-completeness guarantees.
