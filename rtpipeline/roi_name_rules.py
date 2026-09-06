"""Conservative, version-bound class lookup, never ROI identity normalization.

The shared target detector is necessary, not sufficient, for target admission.
It recognizes target tokens inside mixed expressions too. Only an unqualified
identifier can be admitted automatically; unknown qualifiers need adjudication.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

from .rt_details import DEFAULT_ROI_FAMILY_NAMES, is_target_volume_name

RULE_VERSION = "ct-roi-name-rules-v1"
_ASCII_FOLD = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
_DASHES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2013": "-", "\u2212": "-"})
_SPACE = re.compile(r"[ \t\r\n\f\v]+")
# Preserve the complete number, including leading zeros. Never join two numbers.
_INDEX_GAP = re.compile(r"((?:ctv|ptv|gtv)|gtv[npm]) ([0-9]+)\Z")
_FAMILIES = "|".join(name.lower() for name in DEFAULT_ROI_FAMILY_NAMES[:3])
_UNQUALIFIED = re.compile(r"(?:(?:" + _FAMILIES + r")[0-9]*|gtv[npm][0-9]*)\Z")
_DISTANCE = re.compile(r"(?:zz_)?[0-9]+(?:[.,][0-9]+)? *(?:mm|cm) od (.+)\Z")
_OPTIMIZATION = re.compile(r"(?:(?:z[ _]*)?marg[ _]*|z[ _]+min[ _]+|z[ _.]*)(.+)\Z")


def normalize_roi_class_name(name: str) -> str:
    """Normalize spelling for class lookup only, retaining semantic operators.

    Fold ASCII case, map hyphen/nonbreaking hyphen/en dash/minus to '-', collapse
    ASCII whitespace, and remove spaces adjacent to '-' or '+'. Remove one
    index gap only in a whole target-shaped identifier. Other punctuation,
    underscores, diacritics, prefixes, suffixes, and digit strings stay intact.
    """
    text = _SPACE.sub(" ", str(name).translate(_ASCII_FOLD).translate(_DASHES)).strip(" ")
    text = re.sub(r" *([-+]) *", r"\1", text)
    return _INDEX_GAP.sub(r"\1\2", text)


def _target_atom(text: str) -> bool:
    # Do not replace the shared left-boundary/helper detector with a second one.
    text = normalize_roi_class_name(text)
    return is_target_volume_name(text) and _UNQUALIFIED.fullmatch(text) is not None


def _result(roi_class: str, status: str, evidence: str) -> dict[str, str]:
    return {"roi_class": roi_class, "adjudication_status": status, "evidence_basis": evidence}


def _rule_result(policy: Mapping[str, Any], key: str) -> dict[str, str]:
    spec = policy.get("rules", {}).get(key)
    if not isinstance(spec, Mapping) or not all(
        spec.get(field) for field in ("roi_class", "adjudication_status", "evidence_basis")
    ):
        raise ValueError(f"ROI name rule lacks governance metadata: {key}")
    return dict(spec)


def lookup_manual_roi_class(
    name: str, data: Mapping[str, Any]
) -> tuple[Mapping[str, Any] | None, str]:
    """Return a governed entry and auditable lookup source, failing closed.

    Legacy maps without this rule contract retain exact-only behavior. Explicit
    pending entries veto automatic admission. Conflicting normalized entries
    fail closed, even if one spelling has an exact entry. Class aliases are not
    used to rename, combine, select, or identify contours.
    """
    entries = data.get("manual_custom_crosswalk", {})
    policy = data.get("name_rules")
    if not isinstance(policy, Mapping):
        return entries.get(name), "manual_custom_crosswalk:exact_name"
    if policy.get("version") != RULE_VERSION:
        raise ValueError("unsupported CT ROI name rule version")
    normalized = normalize_roi_class_name(name)
    matches = [
        entry for key, entry in entries.items()
        if normalize_roi_class_name(key) == normalized and isinstance(entry, Mapping)
    ]
    if matches:
        signatures = {
            (entry.get("roi_class"), entry.get("adjudication_status")) for entry in matches
        }
        if len(signatures) != 1:
            return _result(
                "unresolved_mixed", "operator_adjudication_required",
                "Normalized map entries disagree in class or adjudication status.",
            ), "manual_custom_crosswalk:normalization_conflict"
        entry = entries.get(name) or matches[0]
        # A pending human decision must never be promoted by a spelling rule.
        if str(entry.get("adjudication_status", "")).startswith(("pending", "operator_")):
            entry = {**entry, "roi_class": "unresolved_mixed", "adjudication_status": "operator_adjudication_required"}
        return entry, (
            "manual_custom_crosswalk:exact_name" if name in entries
            else "manual_custom_crosswalk:normalized_name"
        )

    if _target_atom(normalized):
        return _rule_result(policy, "unqualified_target"), "name_rule:unqualified_target"

    # An organ crop is identifiable without guessing the organ's tissue class.
    # Use only governed anatomy names/aliases, not arbitrary text before '-PTV'.
    anatomy = {
        normalize_roi_class_name(key) for key, entry in entries.items()
        if isinstance(entry, Mapping) and entry.get("roi_class") in {
            "hollow_pelvic_organ", "solid_soft_tissue_neural", "bone", "vessel"
        }
    }
    anatomy.update(normalize_roi_class_name(key) for key in policy.get("crop_anatomy_names", []))
    parts = normalized.split("-")
    if len(parts) == 2:
        left, right = parts
        if (_target_atom(left) and _target_atom(right)) or (
            left in anatomy and _target_atom(right)
        ):
            return _rule_result(policy, "boolean_subtraction"), "name_rule:boolean_subtraction"

    distance = _DISTANCE.fullmatch(normalized)
    if distance and _target_atom(normalize_roi_class_name(distance.group(1))):
        return _rule_result(policy, "distance_margin"), "name_rule:distance_margin"
    optimization = _OPTIMIZATION.fullmatch(normalized)
    if optimization and _target_atom(normalize_roi_class_name(optimization.group(1))):
        return _rule_result(policy, "optimization_helper"), "name_rule:optimization_helper"
    # Unrecognized and qualified names are not targets just because they contain
    # a target token. Preserve exact governance as the final fallback.
    return entries.get(name), "manual_custom_crosswalk:unlisted_exact_name"
