#!/usr/bin/env python3
"""Create the evidence ledger for the resume-reuse diagnosis.

The measured inputs below are copied from the version-3 task packet. The script
keeps packet facts separate from arithmetic derived from those facts and from
interpretation of the repository change.
"""
from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from statistics import median
from time import perf_counter

import pydicom

from rtpipeline import segmentation

WORKSPACE = Path("/umed-projekty/rtpipeline")
OUTPUT = WORKSPACE / "analysis" / "resume-reuse-evidence-ledger.json"
PACKET = WORKSPACE / ".orchestrator" / "runs" / "resume-reuse-20260829T183929Z" / "packets" / "execute-hermes-sol-xhigh.json"


DEFAULT_COURSE = Path(
    "/home/konrad/rtpipeline_campaign/kopernik_bladder_v3/Output/431057/2020-07"
)


def _series_uid(ct_dir: Path) -> str:
    for path in sorted(ct_dir.rglob("*.dcm")):
        try:
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            uid = str(getattr(dataset, "SeriesInstanceUID", "") or "").strip()
            if uid:
                return uid
        except Exception:
            continue
    raise RuntimeError(f"No readable CT series identity in {ct_dir}")


def measure_reuse_path(course: Path, repeats: int = 5) -> dict:
    """Measure only the production mask-currentness predicate on one banked course."""
    metadata_path = course / "metadata" / "case_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    nifti = Path(str(metadata["primary_nifti"]))
    ct_dir = Path(str(metadata["ct_dir"]))
    seg_root = Path(str(metadata["seg_dir"]))
    manifests = sorted(seg_root.rglob("manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError(f"Expected one segmentation manifest, found {len(manifests)}")
    manifest_path = manifests[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = manifest.get("models")
    total_entry = next(
        item for item in models
        if isinstance(item, dict) and item.get("model") == "total"
    )
    masks = [manifest_path.parent / str(name) for name in total_entry.get("masks", [])]
    if not masks or not all(path.is_file() for path in masks):
        raise RuntimeError("The selected course does not contain a complete banked mask set")
    planning_uid = _series_uid(ct_dir)
    source_ct_sop_hash = str(manifest.get("source_ct_sop_hash") or "") or None
    samples = []
    decision = None
    for _ in range(max(1, repeats)):
        started = perf_counter()
        decision = segmentation._series_masks_current(
            manifest_path.parent,
            manifest_path.parent.name,
            "total",
            source_nifti=nifti,
            planning_ct_series_uid=planning_uid,
            source_ct_sop_hash=source_ct_sop_hash,
        )
        samples.append(perf_counter() - started)
    if decision is None or not decision[0]:
        raise RuntimeError(f"Production reuse predicate rejected the selected course: {decision}")
    validation_seconds = float(median(samples))
    return {
        "course": str(course),
        "mask_count": len(masks),
        "mask_bytes": sum(path.stat().st_size for path in masks),
        "planning_ct_series_instance_uid": planning_uid,
        "decision": decision[0],
        "reason": decision[1],
        "validation_seconds_median": validation_seconds,
        "validation_seconds_samples": samples,
        "measurement": "median wall-clock time for the production mask-currentness predicate",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course", type=Path, default=DEFAULT_COURSE)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    measurement = measure_reuse_path(args.course, repeats=args.repeats)
    measurement_path = WORKSPACE / "analysis" / "resume-reuse-measurement.json"
    measurement_path.write_text(
        json.dumps(measurement, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    dfci_courses = 230
    kopernik_courses = 122
    seconds_low = 322
    seconds_high = 494
    minutes_per_course = Decimal("7")
    perturbations_per_roi = 81
    target_rois = 5
    hours_per_course_ntcv = Decimal("0.9")

    ledger = {
        "ledger_version": 1,
        "task_packet": str(PACKET),
        "claim_kinds": {
            "fact": "reported directly in the task packet, repository, or test output",
            "calculation": "computed by this script from packet inputs",
            "inference": "interpretation that remains explicitly labeled",
        },
        "sources": [
            {
                "id": "S1",
                "kind": "task_packet",
                "path": str(PACKET),
                "scope": "measured cohort evidence, required behavior, and updated context",
            },
            {
                "id": "S2",
                "kind": "repository_source",
                "path": str(WORKSPACE / "rtpipeline" / "segmentation.py"),
                "scope": "segmentation resume predicate, mask materialization, provenance, and audit record",
            },
            {
                "id": "S3",
                "kind": "repository_source",
                "path": str(WORKSPACE / "rtpipeline" / "auto_rtstruct.py"),
                "scope": "planning-CT binding and RS_auto reuse or rebuild predicate",
            },
            {
                "id": "S4",
                "kind": "repository_source",
                "path": str(WORKSPACE / "rtpipeline" / "custom_structures_rtstruct.py"),
                "scope": "RS_custom staleness and planning-CT correspondence predicate",
            },
            {
                "id": "S5",
                "kind": "repository_tests",
                "path": str(WORKSPACE / "tests" / "test_resume_reuse_content.py"),
                "scope": "production-shaped regression tests for reuse, mismatch, incompleteness, and auditability",
            },
            {
                "id": "S6",
                "kind": "repository_tests",
                "path": str(WORKSPACE / "tests" / "test_resume_completeness.py"),
                "scope": "pre-existing resume-completeness guarantees",
            },
            {
                "id": "S7",
                "kind": "repository_measurement_script",
                "path": str(WORKSPACE / "analysis" / "resume_reuse_evidence.py"),
                "scope": "production mask-currentness timing and mask inventory on a read-only banked course",
            },
            {
                "id": "S8",
                "kind": "repository_measurement_output",
                "path": str(WORKSPACE / "analysis" / "resume-reuse-measurement.json"),
                "scope": "recorded validation samples and median used for the saving calculation",
            },
        ],
        "inputs": {
            "dfci_courses": dfci_courses,
            "kopernik_courses": kopernik_courses,
            "measured_model_seconds_per_course": [seconds_low, seconds_high],
            "reported_minutes_per_course": str(minutes_per_course),
            "ntcv_perturbations_per_roi": perturbations_per_roi,
            "ntcv_target_rois": target_rois,
            "ntcv_hours_per_course": str(hours_per_course_ntcv),
            "reuse_validation_seconds_median": measurement["validation_seconds_median"],
            "reuse_validation_seconds_samples": measurement["validation_seconds_samples"],
            "reuse_mask_count": measurement["mask_count"],
            "reuse_mask_bytes": measurement["mask_bytes"],
        },
        "claims": [
            {
                "id": "C1",
                "kind": "fact",
                "statement": "The packet reports 121 DFCI courses and 1 Kopernik course with 117 TotalSegmentator masks but no RS_auto after the geometry fix.",
                "support": ["S1"],
            },
            {
                "id": "C2",
                "kind": "fact",
                "statement": "The packet reports 16,424 mask files on DFCI and 234 on Kopernik in that measured state.",
                "support": ["S1"],
            },
            {
                "id": "C3",
                "kind": "fact",
                "statement": "The packet reports 55 model-run log lines per recent course and 322 to 494 seconds elapsed per course with 117 masks already present.",
                "support": ["S1"],
            },
            {
                "id": "C4",
                "kind": "calculation",
                "statement": "At the packet's roughly 7 minutes per course, re-running all 230 DFCI courses is approximately 26.8 hours, reported as about 27 hours, and re-running all 122 Kopernik courses is approximately 14.2 hours, reported as about 14 hours.",
                "calculation": {
                    "dfci_hours": float(Decimal(dfci_courses) * minutes_per_course / Decimal(60)),
                    "kopernik_hours": float(Decimal(kopernik_courses) * minutes_per_course / Decimal(60)),
                },
                "support": ["S1"],
            },
            {
                "id": "C5",
                "kind": "fact",
                "statement": "The packet reports that every existing course in both rebuilt cohorts predates the course-contract fields and that organize must re-run to write those contracts.",
                "support": ["S1"],
            },
            {
                "id": "C6",
                "kind": "calculation",
                "statement": "The NTCV robustness chain is 405 perturbed extractions per course, calculated as 81 perturbations across 5 target ROIs. At 0.9 hours per course, the packet's cohort estimates are approximately 110 hours for 122 Kopernik courses and 207 hours for 230 DFCI courses.",
                "calculation": {
                    "perturbed_extractions_per_course": perturbations_per_roi * target_rois,
                    "kopernik_hours": float(Decimal(kopernik_courses) * hours_per_course_ntcv),
                    "dfci_hours": float(Decimal(dfci_courses) * hours_per_course_ntcv),
                },
                "support": ["S1"],
            },
            {
                "id": "C7",
                "kind": "fact",
                "statement": "The implementation records source series identity and input fingerprints for mask reuse, validates mask completeness and geometry against the planning NIfTI, validates RS_auto and RS_custom against the contracted planning CT series, and records per-artifact reuse or rebuild decisions.",
                "support": ["S2", "S3", "S4"],
            },
            {
                "id": "C8",
                "kind": "fact",
                "statement": "The new regression tests cover complete current masks without a model run, a different planning CT, incomplete masks, current RS_auto and RS_custom reuse, and the recorded audit decision.",
                "support": ["S5"],
            },
            {
                "id": "C9",
                "kind": "fact",
                "statement": "The pre-existing resume-completeness test module remains in the verification scope.",
                "support": ["S6"],
            },
            {
                "id": "C10",
                "kind": "inference",
                "statement": "The sentinel was an insufficient decision boundary because it represented stage publication rather than the currentness of each expensive output and its input correspondence.",
                "uncertainty": "This is a software diagnosis inferred from the packet's measured rerun and the predicates inspected in the repository. It does not claim that every historical mask was stale.",
                "support": ["S1", "S2"],
            },
            {
                "id": "C11",
                "kind": "calculation",
                "statement": "On the selected banked course, the production mask-currentness check took a median of the recorded wall-clock samples. Relative to the measured 322 to 494 seconds for a model rerun, the corresponding net saving is the rerun interval minus that validation time.",
                "calculation": {
                    "validation_seconds_median": measurement["validation_seconds_median"],
                    "net_saving_seconds_range": [
                        seconds_low - measurement["validation_seconds_median"],
                        seconds_high - measurement["validation_seconds_median"],
                    ],
                },
                "support": ["S3", "S7", "S8"],
            },
        ],
    }
    OUTPUT.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
