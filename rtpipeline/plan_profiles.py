"""One registry of the RTPLAN storage profiles this pipeline admits.

Three modules used to decide independently what counts as an RTPLAN:
``rt_details`` accepted any ``Modality=RTPLAN``, ``quality_control`` allowlisted
the standard and Varian classes, and ``course_contract`` allowed only the
standard class. A cohort containing Varian Halcyon/Ethos plans therefore passed
discovery and QC but aborted contract validation.

Source objects and derived objects are deliberately separated. ``_create_summed_plan``
clones its first source, SOP class included, so a single allowlist covering both
would let the pipeline mint a synthetic object claiming a vendor profile it does
not conform to.

References:
  Varian System Server 18.1 DICOM Conformance Statement, section 9.5.3.3
  Varian Ethos Therapy DICOM Conformance Statement (Sept 2023), section 8.5.3
"""
from __future__ import annotations

from typing import Final

# DICOM PS3.4 RT Plan Storage.
STANDARD_RT_PLAN_SOP_CLASS: Final = "1.2.840.10008.5.1.4.1.1.481.5"

# DICOM PS3.4 RT Brachy Treatment Record Storage. Brachytherapy plans still
# use RT Plan Storage and are identified from FractionGroupSequence evidence.
RT_BRACHY_TREATMENT_RECORD_SOP_CLASS: Final = "1.2.840.10008.5.1.4.1.1.481.6"

# "RT Plan Varian 1 Storage": a standard RT Plan extended for the dual-layer
# Halcyon MLC (RTBeamLimitingDeviceType MLCX1/MLCX2), plus private absolute-dose
# calibration (group 3249) and optional data-integrity information (group 3287).
# Every standard RT Plan module remains present.
VARIAN_RT_PLAN_1_SOP_CLASS: Final = "1.2.246.352.70.1.70"

#: Storage classes accepted for a plan that came from the treatment system.
SOURCE_RTPLAN_SOP_CLASSES: Final = frozenset(
    {
        STANDARD_RT_PLAN_SOP_CLASS,
        VARIAN_RT_PLAN_1_SOP_CLASS,
    }
)

#: Storage classes a pipeline-generated plan may claim. A synthesized summation
#: does not conform to the vendor profile, so it may not assert one.
DERIVED_RTPLAN_SOP_CLASSES: Final = frozenset({STANDARD_RT_PLAN_SOP_CLASS})

_PROFILE_NAMES: Final = {
    STANDARD_RT_PLAN_SOP_CLASS: "standard_rt_plan",
    RT_BRACHY_TREATMENT_RECORD_SOP_CLASS: "not_rt_plan",
    VARIAN_RT_PLAN_1_SOP_CLASS: "varian_rt_plan_1",
}


def plan_profile_name(sop_class_uid: str | None) -> str:
    """Return the governed profile name for a plan storage class."""
    return _PROFILE_NAMES.get(str(sop_class_uid or "").strip(), "unknown")


def is_private_plan_profile(sop_class_uid: str | None) -> bool:
    """True for a vendor-private plan profile that a derived object must not claim."""
    uid = str(sop_class_uid or "").strip()
    return uid in SOURCE_RTPLAN_SOP_CLASSES and uid not in DERIVED_RTPLAN_SOP_CLASSES


def fraction_count_semantics(sop_class_uid: str | None) -> str:
    """How NumberOfFractionsPlanned should be read for this profile.

    Ethos defines it as the fractions REMAINING in a treatment phase, not the
    whole-course denominator, so a fraction-weighted delivered dose computed from
    it would be plausible and wrong.
    """
    if is_private_plan_profile(sop_class_uid):
        return "phase_or_remaining_unverified"
    return "whole_course"
