"""Frozen names-only and C1–C4 treatment-region classifiers.

These classes describe target anatomy and geometric overlap. They do not establish
cancer diagnosis, treatment delivery, clinical appropriateness or study eligibility.
See ``docs/features/treatment_scope.md`` for input contracts and interpretation.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

from .treatment_scope_rules import load_rules

CRITERIA = ("C1", "C2", "C3", "C4")
CLASSES = ("B", "P", "MP", "X", "MX", "U")
OVERLAP_FIELDS = (
    "target_volume_cm3",
    "overlap_cm3",
    "overlap_pct_target",
    "overlap_pct_bladder",
    "bladder_mask_volume_cm3",
)
__all__ = ["names_only_class", "geometric_classes", "load_rules", "CRITERIA", "CLASSES"]


def _names_input(value):
    """Reconstruct selection inputs, never cached tokens or class decisions."""
    if isinstance(value, (str, Path)):
        import pydicom

        value = pydicom.dcmread(value, stop_before_pixels=True)
    if not isinstance(value, dict):
        return value, []
    evidence = value.get("evidence", value)
    errors = evidence.get("resolution_errors", [])
    if errors:
        return None, errors
    frozen_selection = load_rules()["target_selection"]
    rois, observations, contours = [], [], []
    for row in evidence["rois"]:
        number = row["roi_number"]
        rois.append(SimpleNamespace(ROINumber=number, ROIName=row["roi_name"]))
        for interpreted_type in row["interpreted_types"]:
            observations.append(
                SimpleNamespace(
                    ReferencedROINumber=number, RTROIInterpretedType=interpreted_type
                )
            )
        contours.append(
            SimpleNamespace(
                ReferencedROINumber=number,
                ContourSequence=[
                    SimpleNamespace(
                        ContourGeometricType=frozen_selection["closed_contour_types"][
                            0
                        ],
                        NumberOfContourPoints=frozen_selection[
                            "minimum_contour_points"
                        ],
                    )
                ]
                * row["valid_contour_count"],
            )
        )
    return SimpleNamespace(
        StructureSetROISequence=rois,
        RTROIObservationsSequence=observations,
        ROIContourSequence=contours,
    ), []


def names_only_class(rtstruct, rule=None):
    """Return ``(class, evidence)`` for a Dataset, DICOM path or saved names evidence.

    ``rule`` is a full rule bundle, its names-rule mapping, or a JSON/YAML path.
    Evidence retains original ROI names and numbers; handle it as clinical data.
    Generic targets and organ-at-risk names never establish target anatomy.
    """
    from .treatment_scope_geometry import _is_target_name, norm

    bundle = load_rules(rule)
    names = bundle.get("names", bundle)
    selection = bundle.get("target_selection", load_rules()["target_selection"])
    evidence = {"rois": [], "resolution_errors": [], "reason": "", "n_targets": 0}
    try:
        ds, errors = _names_input(rtstruct)
        if errors:
            evidence.update(
                resolution_errors=list(errors), reason="source_resolution_error"
            )
            return "U", evidence
        observations = collections.defaultdict(set)
        for item in getattr(ds, "RTROIObservationsSequence", []):
            interpreted_type = (
                str(getattr(item, "RTROIInterpretedType", "")).strip().upper()
            )
            if interpreted_type:
                observations[int(item.ReferencedROINumber)].add(interpreted_type)
        contours = {
            int(item.ReferencedROINumber): sum(
                str(getattr(contour, "ContourGeometricType", ""))
                in selection["closed_contour_types"]
                and int(getattr(contour, "NumberOfContourPoints", 0))
                >= selection["minimum_contour_points"]
                for contour in getattr(item, "ContourSequence", [])
            )
            for item in getattr(ds, "ROIContourSequence", [])
        }
        seen = set()
        for roi in ds.StructureSetROISequence:
            number = int(roi.ROINumber)
            if number in seen:
                raise ValueError("duplicate_ROI_number")
            seen.add(number)
            name = str(roi.ROIName)
            types = observations[number]
            text = norm(name)
            selected = (
                bool(types & set(selection["target_types"]))
                if types
                else _is_target_name(name, selection)
            )
            exclusion = ""
            if selected:
                if types - set(selection["target_types"]):
                    exclusion = "contradictory_interpreted_types"
                elif any(re.search(pattern, text) for pattern in names["helper_regex"]):
                    exclusion = "documented_helper_pattern"
                elif not contours.get(number, 0):
                    exclusion = "no_valid_closed_contour"
            matches = (
                {
                    token: [match.group(0) for match in re.finditer(pattern, text)]
                    for token, pattern in names["anatomy_regex"].items()
                }
                if selected and not exclusion
                else {}
            )
            evidence["rois"].append(
                dict(
                    roi_number=number,
                    roi_name=name,
                    interpreted_types=sorted(types),
                    candidate=selected,
                    exclusion_reason=exclusion,
                    target=selected and not exclusion,
                    valid_contour_count=contours.get(number, 0),
                    tokens={
                        token: values for token, values in matches.items() if values
                    },
                )
            )
        targets = [row for row in evidence["rois"] if row["target"]]
        evidence["n_targets"] = len(targets)
        tokens = {token for row in targets for token in row["tokens"]}
        bladder = "bladder" in tokens
        pelvic = bool(tokens & {"other_pelvic_primary", "pelvic_region"})
        nonpelvic = "nonpelvic" in tokens
        if any(
            row["exclusion_reason"] == "contradictory_interpreted_types"
            for row in evidence["rois"]
        ):
            cls, evidence["reason"] = "U", "contradictory_interpreted_types"
        elif nonpelvic and (bladder or pelvic):
            cls = "MX"
        elif nonpelvic:
            cls = "X"
        elif bladder and "other_pelvic_primary" in tokens:
            cls = "MP"
        elif bladder:
            cls = "B"
        elif pelvic:
            cls = "P"
        else:
            cls, evidence["reason"] = "U", "insufficient_explicit_target_anatomy"
        return cls, evidence
    except Exception:
        evidence.update(
            reason="source_resolution_error",
            resolution_errors=["source_resolution_error"],
        )
        return "U", evidence


def _overlap_values(evidence, rule):
    from .treatment_scope_geometry import bladder_hit

    blank = dict.fromkeys(OVERLAP_FIELDS, "")
    if evidence.get("status") != "measured" or evidence.get("errors"):
        return blank, "measurement_or_identity_failure", [], None
    organ = evidence.get("organs", {}).get("urinary_bladder")
    if not organ:
        return blank, "empty_automatic_bladder_mask", [], None
    targets = []
    for target in evidence.get("targets", []):
        overlap = target["organ_overlaps"].get("urinary_bladder")
        if overlap is None:
            return blank, "bladder_overlap_evidence_absent", [], None
        targets.append(
            dict(
                roi_number=target["roi_number"],
                target_type=target["type"],
                target_volume_cm3=target["volume_ml"],
                overlap_cm3=overlap["volume_ml"],
                overlap_pct_target=100 * overlap["target_fraction"],
                overlap_pct_bladder=100 * overlap["organ_fraction"],
                bladder_mask_volume_cm3=organ["volume_ml"],
                **{
                    "meets_" + criterion: bladder_hit(target, criterion, rule)
                    for criterion in CRITERIA
                },
            )
        )
    if not targets:
        return blank, "selected_target_evidence_absent", [], None
    representative = min(
        targets, key=lambda row: (-row["overlap_cm3"], row["roi_number"])
    )
    return (
        {field: representative[field] for field in OVERLAP_FIELDS},
        "",
        targets,
        representative["roi_number"],
    )


def geometric_classes(course, rules=None):
    """Return C1–C4, consensus, reasons, raw hits and individual-target overlaps.

    ``course`` is an organized course directory, a saved geometry evidence mapping,
    or a saved record with an ``evidence`` mapping. Evidence is replayed without
    reopening source paths and is not independently authenticated. A directory
    triggers fresh CT-grid measurement with source identity and byte-closure checks.
    ``rules`` accepts a full bundle, geometric rule mapping or JSON/YAML path.
    """
    from . import treatment_scope_geometry as geometry

    bundle = load_rules(rules)
    rule = copy.deepcopy(bundle.get("geometry", bundle))
    rule.setdefault("measurement", load_rules()["geometry"]["measurement"])
    rule["target_selection_settings"] = (
        bundle.get("target_selection")
        if isinstance(bundle.get("target_selection"), dict)
        else load_rules()["target_selection"]
    )
    if isinstance(course, (str, Path)):
        evidence = geometry.measure_course(course, rule)
    else:
        evidence = course.get("evidence", course)
    classes, reasons = {}, {}
    for criterion in CRITERIA:
        try:
            if not evidence.get("targets") or not evidence.get("organs"):
                classes[criterion], reasons[criterion] = (
                    "U",
                    {
                        "reason": "measurement_or_identity_failure",
                        "errors": evidence.get("errors", []),
                    },
                )
            else:
                classes[criterion], reasons[criterion] = geometry.classify(
                    evidence, rule, criterion
                )
        except (KeyError, TypeError, ValueError, ZeroDivisionError, StopIteration):
            classes[criterion], reasons[criterion] = (
                "U",
                {
                    "reason": "measurement_or_identity_failure",
                    "errors": ["incomplete_geometry_evidence"],
                },
            )
    consensus = classes["C1"] if len(set(classes.values())) == 1 else "U"
    reason = (
        "cross_criterion_disagreement"
        if len(set(classes.values())) > 1
        else reasons["C1"].get("reason", "")
        if consensus == "U"
        else ""
    )
    if evidence.get("errors"):
        reason = ";".join(evidence["errors"])
    raw_hits = [
        dict(
            roi_number=target["roi_number"],
            **{
                criterion: geometry.bladder_hit(target, criterion, rule)
                for criterion in CRITERIA
            },
        )
        for target in evidence.get("targets", [])
    ]
    overlap, missing_reason, targets, representative = _overlap_values(evidence, rule)
    return dict(
        rule_version=bundle.get("version", rule.get("schema")),
        plan_selection_basis=evidence.get("plan_selection_basis", "not_recorded"),
        selected_plan_evidence=copy.deepcopy(
            evidence.get("selected_plan_evidence", [])
        ),
        classes=classes,
        consensus=consensus,
        u_reason=reason,
        reasons=reasons,
        u_reasons={
            criterion: (
                ";".join(evidence["errors"])
                if evidence.get("errors")
                else reasons[criterion].get("reason", "")
            )
            if classes[criterion] == "U"
            else ""
            for criterion in CRITERIA
        },
        raw_bladder_hits=raw_hits,
        representative_overlap=overlap,
        representative_target_roi_number=representative,
        selected_target_overlaps=targets,
        overlap_not_measured_reason=missing_reason,
        measurement_status=evidence["status"],
    )


def main(argv=None):
    """CLI with identifier-free stdout and optional private evidence output."""
    parser = argparse.ArgumentParser(prog="rtpipeline scope", description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--rtstruct", type=Path, help="Names-only classification of one DICOM RTSTRUCT"
    )
    source.add_argument(
        "--course", type=Path, help="Fresh geometry from an organized course directory"
    )
    source.add_argument(
        "--evidence", type=Path, help="Replay saved geometry evidence JSON"
    )
    parser.add_argument(
        "--names-evidence", type=Path, help="Also replay saved names evidence JSON"
    )
    parser.add_argument(
        "--rules", type=Path, help="Explicit versioned JSON/YAML rule bundle"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write detailed JSON with mode 0600; stdout remains a summary",
    )
    args = parser.parse_args(argv)
    try:
        rules = load_rules(args.rules)
        result, summary = {}, {"rule_version": rules["version"]}
        if args.rtstruct:
            cls, evidence = names_only_class(args.rtstruct, rules)
            result["names_only"] = {"class": cls, "evidence": evidence}
            summary["names_only"] = {
                "class": cls,
                "u_reason": evidence["reason"],
                "n_targets": evidence["n_targets"],
            }
        else:
            course = (
                args.course if args.course else json.loads(args.evidence.read_text())
            )
            result["geometry"] = geometric_classes(course, rules)
            summary["geometry"] = {
                key: result["geometry"][key]
                for key in (
                    "classes",
                    "consensus",
                    "u_reason",
                    "representative_overlap",
                    "overlap_not_measured_reason",
                )
            }
        if args.names_evidence:
            cls, evidence = names_only_class(
                json.loads(args.names_evidence.read_text()), rules
            )
            result["names_only"] = {"class": cls, "evidence": evidence}
            summary["names_only"] = {
                "class": cls,
                "u_reason": evidence["reason"],
                "n_targets": evidence["n_targets"],
            }
        if args.output:
            # New files only: avoid overwriting a clinical source passed in error.
            payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
            descriptor = os.open(
                args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            with os.fdopen(descriptor, "w") as stream:
                stream.write(payload)
        print(json.dumps(summary, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError):
        # Never echo source paths, names, DICOM values or exception details.
        print(
            "scope: invalid input, rules or output path", file=__import__("sys").stderr
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
