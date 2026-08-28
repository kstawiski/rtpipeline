#!/usr/bin/env python3
"""Build the objective-check ledger from stored read-only measurements."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

WORKSPACE = Path("/umed-projekty/rtpipeline")
ANALYSIS = WORKSPACE / "analysis"


def load(name: str):
    return json.loads((ANALYSIS / name).read_text())


def target_group_counts(linkage: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for patient_id, patient in linkage["patients"].items():
        counts[patient_id] = len(
            {
                row["assigned_struct_uid"]
                for row in patient["linked_sets"]
                if row["assigned_struct_target_count"] > 0
            }
        )
    return counts


def plan_only_target_groups(linkage: dict) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for patient_id, patient in linkage["patients"].items():
        by_struct: dict[str, list[dict]] = defaultdict(list)
        for row in patient["linked_sets"]:
            if row["assigned_struct_target_count"] > 0:
                by_struct[row["assigned_struct_uid"]].append(row)
        plan_only = sorted(
            uid for uid, rows in by_struct.items() if all(not row["dose_uid"] for row in rows)
        )
        if plan_only:
            output[patient_id] = plan_only
    return output


def main() -> None:
    dfci_census = load("dfci-census-summary.json")
    dfci_output = load("dfci-output-snapshot.json")
    dfci_plan_struct = load("dfci-plan-struct-study-evidence.json")
    dfci_struct_ct = load("dfci-cross-study-evidence.json")
    dfci_log = load("dfci-organize-log-evidence.json")
    kopernik_census = load("kopernik-reference-course-census.json")
    kopernik_output = load("kopernik-output-summary.json")
    kopernik_struct_selection = load("kopernik-output-struct-selection.json")
    baseline = load("baseline-metadata-linkage.json")
    fixed = load("reference-driven-metadata-linkage.json")
    dose = load("reference-dose-classification.json")

    baseline_summary = {
        patient_id: {
            "linked_sets": len(patient["linked_sets"]),
            "correct_plan_struct_links": sum(
                bool(row["assigned_struct_uid"])
                and row["assigned_struct_uid"] == row["authoritative_struct_uid"]
                for row in patient["linked_sets"]
            ),
            "target_bearing_assigned_links": sum(
                row["assigned_struct_target_count"] > 0 for row in patient["linked_sets"]
            ),
            "grouped_courses": len(patient["grouped_courses"]),
        }
        for patient_id, patient in baseline["patients"].items()
    }
    dose_summary = {
        patient_id: [
            {
                "struct_uid": row["struct_uid"],
                "classification": row["classification"],
                "source_plans": row["source_plans"],
                "source_doses": row["source_doses"],
                "selected_plans": len(row["selected_plan_uids"]),
                "selected_doses": len(row["selected_dose_uids"]),
                "selected_total_rx_gy": row["selected_total_rx_gy"],
                "should_sum": row["should_sum"],
                "warnings": row["warnings"],
            }
            for row in rows
        ]
        for patient_id, rows in dose["patients"].items()
    }

    ledger = {
        "spec_digest": "4432fa965adc055d034201856188ed6532460a563e299358bf8624412a9cf165",
        "scope": "Objective execution evidence. No real cohort was rebuilt.",
        "provenance": {
            "dfci_source": {
                "path": "/local-data/ARIA_EXPORTS_DOWNLOADED/by_source/TMT",
                "host": "s1",
                "access": "read-only SSH",
            },
            "dfci_output": {
                "path": "/local-data/rtpipeline_campaign/dfci_tmt/Output",
                "host": "s1",
                "access": "read-only SSH",
            },
            "kopernik_source": {
                "path": "/home/konrad/rtpipeline_campaign/kopernik_bladder/Input",
                "access": "read-only local",
            },
            "kopernik_output": {
                "path": "/home/konrad/rtpipeline_campaign/kopernik_bladder/Output",
                "access": "read-only local",
            },
            "patient_data_storage": "Only derived DICOM metadata and aggregate counts are stored in analysis JSON. No DICOM instance was copied into the workspace.",
        },
        "facts": {
            "dfci_census": dfci_census,
            "dfci_current_output": dfci_output,
            "dfci_plan_to_struct_study_relationship": dfci_plan_struct,
            "dfci_struct_to_ct_measurement": dfci_struct_ct,
            "dfci_organize_log": dfci_log,
            "kopernik_reference_course_census": kopernik_census,
            "kopernik_current_output": kopernik_output,
            "kopernik_output_structure_selection": kopernik_struct_selection,
            "pre_fix_selected_patient_linkage": baseline_summary,
            "post_fix_selected_patient_target_group_counts": target_group_counts(fixed),
            "post_fix_selected_patient_plan_only_target_groups": plan_only_target_groups(fixed),
            "post_fix_selected_patient_dose_classification": dose_summary,
        },
        "calculations": [
            {
                "id": "dfci_rt_export_skip",
                "result": {
                    "courses": dfci_output["courses"],
                    "courses_with_rtstruct_files": dfci_output["courses_with_rtstruct_files"],
                    "courses_with_rtplan_files": dfci_output["courses_with_rtplan_files"],
                    "courses_with_rtdose_files": dfci_output["courses_with_rtdose_files"],
                },
                "inputs": ["dfci-output-snapshot.json", "dfci-organize-log-evidence.json"],
            },
            {
                "id": "dfci_cross_study_plan_struct_frequency",
                "result": {
                    "resolved_links": dfci_plan_struct["resolved_plan_struct_links"],
                    "cross_study_links": dfci_plan_struct["cross_study_plan_struct_links"],
                    "cross_study_fraction": (
                        dfci_plan_struct["cross_study_plan_struct_links"]
                        / dfci_plan_struct["resolved_plan_struct_links"]
                    ),
                },
                "inputs": ["dfci-plan-struct-study-evidence.json"],
            },
            {
                "id": "kopernik_reference_supported_courses",
                "result": {
                    "target_bearing_plan_referenced_structs": kopernik_census["target_bearing_referenced_structure_sets"],
                    "dose_linked_target_courses": kopernik_census["dose_linked_target_courses"],
                    "plan_only_target_courses": kopernik_census["target_refsets_without_dose_linked_course"],
                },
                "inputs": ["kopernik-reference-course-census.json"],
            },
        ],
        "inferences": [
            {
                "id": "dfci_zero_rt_primary_cause",
                "claim": "The DFCI zero-RT output was caused first by the patient/series branch that skipped every RT scan. Study equality could not fire because no RT metadata reached linkage.",
                "supports": [
                    "dfci-organize-log-evidence.json",
                    "dfci-output-snapshot.json",
                    "baseline code rtpipeline/organize.py at HEAD lines 2034-2039",
                ],
                "uncertainty": "The cohort was not rebuilt after the fix.",
            },
            {
                "id": "dfci_cross_study_scope_correction",
                "claim": "Cross-study linkage is real but not cohort-wide. The full RTPLAN-to-RTSTRUCT measurement found 15 of 1,554 resolved links crossing StudyInstanceUID. The resolvable RTSTRUCT-to-CT sample remained same-study.",
                "supports": [
                    "dfci-plan-struct-study-evidence.json",
                    "dfci-course-identity-sample.json",
                    "dfci-cross-study-evidence.json",
                ],
                "uncertainty": "The full RTSTRUCT-to-CT script indexed one representative CT header per patient and resolved only 156 references, so it cannot estimate the full-cohort RTSTRUCT-to-CT cross-study frequency.",
            },
            {
                "id": "kopernik_plan_only_exception",
                "claim": "Kopernik has 122 target-bearing plan-referenced structure sets. One has no RTDOSE that resolves to its plans, so retaining all 122 requires one target-bearing plan-only course rather than fabricating a dose.",
                "supports": ["kopernik-reference-course-census.json", "reference-driven-metadata-linkage.json"],
                "uncertainty": "Only an operator rebuild can verify emitted output counts.",
            },
        ],
        "limitations": [
            "No real cohort was reprocessed because the packet prohibited it.",
            "The selected-patient post-fix checks execute metadata linkage and dose classification, not image conversion or segmentation.",
            "Synthetic DICOM regressions establish code behavior but are not independent review.",
        ],
    }
    (ANALYSIS / "course-identity-objective-checks.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
