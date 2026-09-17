#!/usr/bin/env python3
"""Content-based pre-push scan for patient data in a git repo.

Replaces the inadequate filename-only check
    git ls-files | grep -iE "manuscript|abstract|\.bib$"
which cannot see identifiers embedded in prose inside legitimately named files.
That check passed on 2026-09-06 while docs/diagnosis-delivered-dose.md already
carried twelve DFCI patient identifiers with dose values.

Read-only. Exits 1 if anything is found, so it can gate a push.
"""
import re, subprocess, sys, collections

PATTERNS = {
    "dfci_patient_id":   re.compile(rb"\b10[0-9]{9}\b"),
    "kopernik_patient_id": re.compile(rb"\b(?:patient|pacjent|patient_id)[\s:=_-]{0,4}([0-9]{6})\b", re.I),
    "polish_clinical":   re.compile(rb"(?i)\b(rak p\xc4\x99cherza|cystektomi|usuni\xc4\x99ci\w* p\xc4\x99cherza|"
                                    rb"cystoprostatektomi|stan po TURBT|hist-?pat)\b"),
    # NOTE: 1.2.840.10008.* are STANDARD DICOM SOP Class UIDs defined by the
    # standard (RT Dose/Structure Set/Plan/Treatment Record storage). They are
    # constants, not patient data. Only non-10008 roots can be instance UIDs.
    "dicom_instance_uid": re.compile(rb"\b1\.2\.(?:246|392|826|840(?!\.10008))\.[0-9.]{20,}\b"),
    "explicit_phi_word": re.compile(rb"(?i)\b(date of birth|data urodzenia|\bDOB\b|PESEL|\bMRN\b)\b"),
}
# tokens that look like ids but are not
BENIGN = re.compile(rb"^(1000000000|1234567890)$")

def tracked(repo):
    out = subprocess.run(["git", "-C", repo, "ls-files", "-z"],
                         capture_output=True, check=True).stdout
    return [p for p in out.split(b"\0") if p]

def main(repo="."):
    findings = collections.defaultdict(lambda: collections.defaultdict(set))
    for rel in tracked(repo):
        path = rel.decode("utf-8", "replace")
        # binary third-party wheels produce coincidental byte matches
        if path.endswith((".png",".jpg",".jpeg",".gz",".zip",".pdf",".nii",".dcm",
                          ".parquet",".whl",".so",".pyd")):
            continue
        try:
            with open(f"{repo}/{path}", "rb") as fh:
                blob = fh.read(8_000_000)
        except OSError:
            continue
        for label, pat in PATTERNS.items():
            for m in pat.findall(blob):
                tok = m if isinstance(m, bytes) else m[0]
                if BENIGN.match(tok):
                    continue
                findings[label][path].add(tok.decode("utf-8", "replace"))
    if not findings:
        print("CLEAN: no patient identifiers or clinical text found in tracked files.")
        return 0
    print("FINDINGS — do not push until these are resolved.\n")
    for label in sorted(findings):
        files = findings[label]
        toks = set().union(*files.values())
        print(f"== {label}: {len(files)} file(s), {len(toks)} distinct value(s)")
        for path in sorted(files)[:12]:
            print(f"     {len(files[path]):4d} distinct  {path}")
        if len(files) > 12:
            print(f"     ... and {len(files)-12} more files")
        print()
    return 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
