#!/usr/bin/env python3
"""B3 regex case-test against the real P0 MR inventory (DESIGN_v3 acceptance criterion).

Reproduces the modality_classifier normalization + functional regex, compares the
current `_MR_FUNCTIONAL_RE` against the B3-extended version, and reports every series
the extension NEWLY captures plus any anatomic/weighted-image collision.

Run: python B3_artifacts/regex_case_test.py B3_artifacts/p0_mr_series_descriptions_20260620.txt
Inventory source: ~/miednice_inventory/dicom_inventory.sqlite (scan 2026-05-23),
  SELECT DISTINCT series_description FROM series WHERE modality='MR' AND series_description!='';
"""
import re, sys

OLD = re.compile(r"dwi|\bdiff|\bep2d|adc|\bperf|\btwist\b|\bdyn|\bdce\b|\bttp\b|\bpei\b|\bmipt\b", re.I)
# B3 extension (DESIGN_v3, rev2+: cross-cohort-safe `\bsub\b|\bsubtract`, not open `\bsub`):
# wash-out / wash-in / subtraction DCE-derived maps.
NEW = re.compile(
    r"dwi|\bdiff|\bep2d|adc|\bperf|\btwist\b|\bdyn|\bdce\b|\bttp\b|\bpei\b|\bmipt\b|"
    r"\bwo\b|\bwi\b|\bsub\b|\bsubtract",
    re.I,
)
WEIGHT = {"t1", "t2", "t1w", "t2w", "t1wi", "t2wi", "pd", "pdw"}
ANAT = re.compile(r"dixon|\b(tse|fse|frfse|frse|spc|space|tirm|stir|flair|blade|vibe|"
                  r"mpr|mprage|me2d|haste|trufi|truefisp|ffe)\b", re.I)

def norm(d):
    toks = re.findall(r"[a-z0-9]+", d.lower())
    return " ".join(toks), set(toks)

def main(path):
    descs = [l.rstrip("\n") for l in open(path) if l.strip()]
    newly = []
    for d in descs:
        n, ts = norm(d)
        if OLD.search(n):
            continue
        if NEW.search(n):
            newly.append((d, n, ts))
    print(f"distinct MR descriptions: {len(descs)}")
    print(f"newly captured by B3 extension: {len(newly)}")
    collisions = 0
    for d, n, ts in sorted(newly):
        flags = []
        if ts & WEIGHT:
            flags.append("WEIGHT=" + ",".join(sorted(ts & WEIGHT)))
        if ANAT.search(n):
            flags.append("ANATSEQ")
        # A collision that matters = newly-captured series that is NOT a DCE subtraction.
        # All current hits are DCE subtraction (`sub` token) -> legitimate functional.
        print(f"  {d!r}  {flags}")
    # Acceptance: wo/wi capture nothing (no T1WI/T2WI collision); every new hit carries a
    # `sub` TOKEN (token-membership, not substring — guards against e.g. sub-prefixed anatomy).
    assert all("sub" in ts for d, n, ts in newly), "non-sub newly-captured series -> investigate"
    # Cross-cohort safety (Claude-r2 #2 / Claude-r3): tightened regex must NOT capture
    # sub-prefixed anatomy present in other cohorts (thorax/GBM).
    for anat in ("subclavian artery", "AX subcutaneous fat", "submandibular T2", "subcarinal node"):
        an, _ = norm(anat)
        assert not NEW.search(an), f"regression: regex wrongly captured anatomy {anat!r}"
    print("ACCEPTANCE: all newly-captured series carry a `sub` token (DCE-subtraction); "
          "no wo/wi weighted-image collision; sub-prefixed anatomy rejected.")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "B3_artifacts/p0_mr_series_descriptions_20260620.txt")
