from __future__ import annotations

import pytest

from rtpipeline.modality_classifier import classify_series


def _meta(**overrides):
    base = {
        "modality": "CT",
        "series_description": "",
        "manufacturer": "SIEMENS",
        "manufacturer_model": "Sensation Open",
        "frame_of_reference_uid": "1.2.3",
        "n_instances": 100,
        "image_types": ["ORIGINAL\\PRIMARY\\AXIAL"],
        "rt_linked": False,
        "has_pt_same_study_for": False,
    }
    base.update(overrides)
    return base


def test_advanced_reconstruction_blank_description_is_4dct_phase():
    image_class, reason = classify_series(
        _meta(manufacturer_model="Advanced Reconstruction", series_description="")
    )

    assert image_class == "fourdct_phase"
    assert reason is None


def test_halcyon_pva_is_cbct():
    image_class, reason = classify_series(
        _meta(
            manufacturer="Varian Medical Systems",
            manufacturer_model="Halcyon - PVA",
        )
    )

    assert image_class == "cbct"
    assert reason is None


def test_mip_projection_is_excluded():
    image_class, reason = classify_series(
        _meta(
            manufacturer_model="Advanced Reconstruction",
            series_description="MIP(10) 0%-90%",
        )
    )

    assert image_class == "exclude"
    assert reason in {"description_mip_projection", "fourdct_projection"}


def test_average_various_with_ave_image_type_is_4dct_ave():
    image_class, reason = classify_series(
        _meta(
            series_description="Average_Various_1",
            image_types=["DERIVED\\PRIMARY\\AXIAL\\CT_SOM5 AVE"],
        )
    )

    assert image_class == "fourdct_ave"
    assert reason is None


def test_resp_motion_maxip_projection_is_excluded():
    image_class, reason = classify_series(
        _meta(series_description="Respiratory Motion 3.0 t-MaxIP iMAR")
    )

    assert image_class == "exclude"
    assert reason == "fourdct_projection"


def test_varian_min_projection_is_excluded():
    image_class, reason = classify_series(
        _meta(
            manufacturer_model="Advanced Reconstruction",
            series_description="Min() 0%-90%",
        )
    )

    assert image_class == "exclude"
    assert reason == "fourdct_projection"


def test_resp_low_percent_is_4dct_phase():
    image_class, reason = classify_series(
        _meta(series_description="RespLow 1.5 B30s 100% In")
    )

    assert image_class == "fourdct_phase"
    assert reason is None


def test_localizer_image_type_is_excluded():
    image_class, reason = classify_series(
        _meta(image_types=["ORIGINAL\\PRIMARY\\LOCALIZER"])
    )

    assert image_class == "exclude"
    assert reason == "image_type_localizer"


def test_ep2d_diff_mr_is_functional():
    image_class, reason = classify_series(
        _meta(
            modality="MR",
            series_description="ep2d_diff_tra_0-800",
            manufacturer_model="MAGNETOM",
        )
    )

    assert image_class == "mr_functional"
    assert reason is None


def test_t2_tse_mr_is_anatomic():
    image_class, reason = classify_series(
        _meta(
            modality="MR",
            series_description="t2_tse_cor 3mm",
            manufacturer_model="MAGNETOM",
        )
    )

    assert image_class == "mr_anatomic"
    assert reason is None


def test_petct_ct_requires_pt_same_study_for():
    image_class, reason = classify_series(
        _meta(
            manufacturer_model="Biograph128",
            has_pt_same_study_for=True,
        )
    )

    assert image_class == "petct_ct"
    assert reason is None


def test_sub_volumetric_series_is_excluded():
    image_class, reason = classify_series(_meta(n_instances=9))

    assert image_class == "exclude"
    assert reason == "sub_volumetric_lt10"


def test_pet_dose_report_is_excluded():
    image_class, reason = classify_series(
        _meta(modality="PT", series_description="PET Dose Report")
    )

    assert image_class == "exclude"
    assert reason == "pt_report_or_derived"


def test_pet_sub_volumetric_series_is_excluded():
    image_class, reason = classify_series(_meta(modality="PT", n_instances=5))

    assert image_class == "exclude"
    assert reason == "pt_sub_volumetric_lt10"


def test_pet_secondary_capture_is_excluded():
    image_class, reason = classify_series(
        _meta(modality="PT", image_types=["DERIVED\\SECONDARY\\OTHER"])
    )

    assert image_class == "exclude"
    assert reason == "pt_secondary_capture"


def test_real_pet_reconstruction_is_kept():
    image_class, reason = classify_series(
        _meta(modality="PT", n_instances=200, image_types=["ORIGINAL\\PRIMARY"])
    )

    assert image_class == "pt"
    assert reason is None


def test_nms_ct_is_calibrated_for_planning():
    image_class, reason = classify_series(
        _meta(manufacturer="NMS", manufacturer_model="", rt_linked=True)
    )

    assert image_class == "planning_ct"
    assert reason is None


def test_hitachi_ct_is_calibrated_for_planning_and_diagnostic():
    planning_class, planning_reason = classify_series(
        _meta(manufacturer="Hitachi, Ltd.", manufacturer_model="", rt_linked=True)
    )
    diagnostic_class, diagnostic_reason = classify_series(
        _meta(manufacturer="Hitachi, Ltd.", manufacturer_model="", rt_linked=False)
    )

    assert planning_class == "planning_ct"
    assert planning_reason is None
    assert diagnostic_class == "diagnostic_ct"
    assert diagnostic_reason is None


def test_explicit_planning_flag_overrides_rt_link_fallback():
    image_class, reason = classify_series(_meta(is_planning_ct=False, rt_linked=True))

    assert image_class == "diagnostic_ct"
    assert reason is None


def test_explicit_planning_flag_marks_planning_ct():
    image_class, reason = classify_series(_meta(is_planning_ct=True, rt_linked=False))

    assert image_class == "planning_ct"
    assert reason is None


# --- MR sequence-family regression cases across representative vendor naming ---
# Each was silently dropped by the original \b-anchored / too-narrow MR regexes.
# These lock in vendor-robust normalized-token classification (Siemens/Philips/GE/UIH).

_MR_ANATOMIC_DESCS = [
    "T2W_TSE_sag",          # Philips capital-underscore TSE (trailing \b before _ bug)
    "T1W_TSE_Tra",          # Philips T1W TSE (no T1W-TSE alternative existed)
    "T2_AX_MV_HR_RT",       # generic T2 weighting, no explicit sequence token
    "Sag T2 frFSE",         # GE fast-recovery FSE naming
    "AX T1 FSE",            # GE FSE
    "+C Cor T1 FSE FS",     # GE post-contrast fat-sat FSE
    "COR T1WI",             # UIH/Orian T1-weighted
    "t2_blade_sag",         # Siemens BLADE motion-corrected T2
    "t2_me2d_tra",          # Siemens multi-echo 2D T2
    "t1_mpr_iso_1.5mm_cor",  # Siemens MPR isotropic
    "mDIXON XD",            # Philips modified Dixon (keyword embedded in token)
    "t1_tse_dixon_tra_largeFoV_W",  # Siemens Dixon water image
    "Odbytnica staging",      # synthetic Polish rectal MR-sim staging example
    "Odbytnica plan",         # synthetic Polish rectal MR-sim planning example
]

_MR_FUNCTIONAL_DESCS = [
    "DWI_b1500",            # Philips DWI (trailing \b before _ bug)
    "DWI_3b_dS_ZOOM",       # Philips multi-b DWI
    "IsoDWI",               # UIH isotropic DWI (keyword embedded in token)
    "cDWI.b=1000",          # computed DWI (keyword embedded in token)
    "dADC",                 # Philips derived ADC map (no separator before adc)
    "ep2d_diff_tra_0-800_ADC",  # Siemens ADC map
    "t1_twist_tra_dyn_TT=7.0s",  # Siemens TWIST DCE-perfusion
    "t1_vibe_tra_perf_CM_SUB",   # Siemens perfusion subtraction
]

_MR_EXCLUDE_DESCS = [
    ("C_localizer", "mr_nonanatomic_localizer"),      # underscore localizer (\b bug)
    ("L/S_localizer", "mr_nonanatomic_localizer"),
    ("3-Plane Loc", "mr_nonanatomic_loc"),
    ("SURVEY_BFFE", "mr_nonanatomic_survey"),          # Philips survey/scout
    ("Calibration", "mr_nonanatomic_calibration"),     # GE coil calibration
    ("", "mr_unrecognized_default_deny"),              # blank desc -> fail closed
]


@pytest.mark.parametrize("desc", _MR_ANATOMIC_DESCS)
def test_mr_anatomic_recovered(desc):
    image_class, reason = classify_series(_meta(modality="MR", series_description=desc))
    assert image_class == "mr_anatomic", f"{desc!r} -> {image_class}/{reason}"
    assert reason is None


@pytest.mark.parametrize("desc", _MR_FUNCTIONAL_DESCS)
def test_mr_functional_recovered(desc):
    image_class, reason = classify_series(_meta(modality="MR", series_description=desc))
    assert image_class == "mr_functional", f"{desc!r} -> {image_class}/{reason}"
    assert reason is None


@pytest.mark.parametrize("desc,expected_reason", _MR_EXCLUDE_DESCS)
def test_mr_genuine_exclusions(desc, expected_reason):
    image_class, reason = classify_series(_meta(modality="MR", series_description=desc))
    assert image_class == "exclude", f"{desc!r} -> {image_class}/{reason}"
    assert reason == expected_reason


def test_mr_anatomic_does_not_swallow_functional_with_t2_token():
    # A diffusion series carrying a t2 token must stay functional, not be routed to total_mr.
    image_class, _ = classify_series(
        _meta(modality="MR", series_description="ep2d_diff_tra t2 trace")
    )
    assert image_class == "mr_functional"


# --- RTSTRUCT-bound recovery plus projection/derived gates ---
# These cases exercise pixel reads and per-instance image-type sweeps.


def test_acuity_cbct_with_stray_localizer_is_kept_as_cbct():
    # FIX J: an AXIAL CBCT volume carrying ONE stray LOCALIZER instance (Varian Acuity:
    # 80 AXIAL + 1 LOCALIZER) must NOT be dropped by the series-level localizer gate.
    image_class, reason = classify_series(
        _meta(
            manufacturer="Varian Medical Systems",
            manufacturer_model="Acuity Cone-beam CT",
            image_types=["ORIGINAL\\PRIMARY\\AXIAL", "ORIGINAL\\PRIMARY\\LOCALIZER"],
            n_instances=81,
        )
    )
    assert image_class == "cbct"
    assert reason is None


def test_pure_localizer_series_still_excluded():
    # FIX J safety: a series with ONLY a LOCALIZER image type (no AXIAL/HELICAL) stays excluded.
    image_class, reason = classify_series(
        _meta(image_types=["ORIGINAL\\PRIMARY\\LOCALIZER"], n_instances=20)
    )
    assert image_class == "exclude"
    assert reason == "image_type_localizer"


def test_plastimatch_rtstruct_linked_ct_is_planning():
    # FIX I: a software-derived (Plastimatch/Velocity/…) CT directly contoured by an RTSTRUCT
    # is a real planning reference — recovered as planning_ct, not dropped as software_derived.
    image_class, reason = classify_series(
        _meta(
            manufacturer="Plastimatch",
            manufacturer_model="Plastimatch",
            is_planning_ct=True,
            rt_series_linked=True,
            image_types=["DERIVED\\SECONDARY\\AXIAL"],
            n_instances=195,
        )
    )
    assert image_class == "planning_ct"
    assert reason is None


def test_rtstruct_series_linked_unrecognized_ct_is_diagnostic():
    # FIX I (defensive fallback): if the _assign_planning_flags invariant ever breaks so a
    # series-linked CT lacks is_planning_ct, the rt_series_linked branch still RECOVERS it
    # (as diagnostic_ct) rather than silently dropping it. In production this state cannot
    # occur (series-linked CT always gets is_planning_ct=True), so this guards a latent regression.
    image_class, reason = classify_series(
        _meta(
            manufacturer="TheraPanacea",
            manufacturer_model="ART-Plan",
            is_planning_ct=False,
            rt_series_linked=True,
            n_instances=200,
        )
    )
    assert image_class == "diagnostic_ct"
    assert reason is None


def test_non_rtlinked_unrecognized_ct_default_denied():
    # FIX I keeps default-deny: an unrecognized-vendor CT with NO RTSTRUCT link is excluded.
    image_class, reason = classify_series(
        _meta(manufacturer="3D Slicer", manufacturer_model="", n_instances=200)
    )
    assert image_class == "exclude"
    assert reason == "ct_unrecognized_default_deny"


def test_non_rtlinked_derived_secondary_ct_gets_specific_reason():
    # A non-RT-linked DERIVED\SECONDARY CT gets an auditable reason.
    image_class, reason = classify_series(
        _meta(
            manufacturer="Plastimatch",
            manufacturer_model="Plastimatch",
            image_types=["DERIVED\\SECONDARY\\AXIAL"],
            n_instances=200,
        )
    )
    assert image_class == "exclude"
    assert reason == "ct_derived_secondary"


def test_syngo_via_report_excluded_subvolumetric():
    # FIX I: removing the software_derived_model gate does NOT leak syngo.via reports —
    # they are n<10 ("Results MM Oncology Reading") and still excluded by the prior gates.
    image_class, reason = classify_series(
        _meta(
            manufacturer="syngo.via",
            manufacturer_model="syngo.via",
            series_description="Results MM Oncology Reading",
            n_instances=3,
        )
    )
    assert image_class == "exclude"
    assert reason in {"sub_volumetric_lt10", "description_results_mm_oncology"}


def test_pet_mip_projection_excluded():
    # FIX K: a PET MIP rotating-projection (desc token) must not be kept as `pt`.
    image_class, reason = classify_series(
        _meta(
            modality="PT",
            series_description="MIP miednica",
            n_instances=12,
            image_types=["DERIVED\\PRIMARY\\REFORMATTED"],
        )
    )
    assert image_class == "exclude"
    assert reason == "pt_projection_mip"


def test_pet_maxip_image_type_excluded():
    # FIX K: image-type-driven projection gate (MAXIP) for PET.
    image_class, reason = classify_series(
        _meta(
            modality="PT",
            series_description="WB",
            n_instances=200,
            image_types=["DERIVED\\PRIMARY\\MAXIP"],
        )
    )
    assert image_class == "exclude"
    assert reason == "pt_projection_mip"


def test_real_pet_not_flagged_as_projection():
    # FIX K safety: a genuine PET reconstruction (no MIP token / image type) stays `pt`.
    image_class, reason = classify_series(
        _meta(
            modality="PT",
            series_description="PET WB Corrected",
            n_instances=200,
            image_types=["ORIGINAL\\PRIMARY"],
        )
    )
    assert image_class == "pt"
    assert reason is None


def test_vitesse_derived_secondary_mr_gets_specific_reason():
    # A DERIVED\SECONDARY MR with no anatomic/functional tokens is relabeled
    # `mr_derived_secondary` and remains excluded.
    image_class, reason = classify_series(
        _meta(
            modality="MR",
            manufacturer="Varian Medical Systems, Inc.",
            manufacturer_model="Vitesse",
            series_description="",
            n_instances=80,
            image_types=["DERIVED\\SECONDARY\\OTHER"],
        )
    )
    assert image_class == "exclude"
    assert reason == "mr_derived_secondary"


def test_ge_cal_mr_excluded_with_cal_token():
    # cal token: GE coil calibration ("Cal SpineArray") gets a specific non-anatomic reason.
    image_class, reason = classify_series(
        _meta(
            modality="MR",
            manufacturer="GE MEDICAL SYSTEMS",
            manufacturer_model="Optima MR360",
            series_description="Cal SpineArray_12",
            n_instances=128,
        )
    )
    assert image_class == "exclude"
    assert reason == "mr_nonanatomic_cal"


# --- Direct unit tests for _assign_planning_flags FoR-only branches ---

from rtpipeline.inventory import InventorySeries, _assign_planning_flags  # noqa: E402


def _ct_series(uid, *, study="ST1", for_uid="FOR1", series_linked=False, for_linked=False,
               model="", n=200):
    return InventorySeries(
        patient_id="P1", study_uid=study, study_description="", series_uid=uid,
        modality="CT", series_description="", manufacturer="", manufacturer_model=model,
        frame_of_reference_uid=for_uid, n_instances=n, instances=[],
        rt_linked=series_linked or for_linked,
        rt_series_linked=series_linked, rt_for_linked=for_linked,
    )


def test_assign_planning_flags_series_linked():
    s = _ct_series("u1", series_linked=True, for_linked=True)
    _assign_planning_flags([s])
    assert s.is_planning_ct is True
    assert s.rt_link_basis == "rtstruct_to_series"


def test_assign_planning_flags_for_unique():
    # Single FoR-only-linked candidate in its (study,FoR) -> treated as planning.
    s = _ct_series("u1", series_linked=False, for_linked=True)
    _assign_planning_flags([s])
    assert s.is_planning_ct is True
    assert s.rt_link_basis == "rtstruct_to_for_unique"


def test_assign_planning_flags_for_ambiguous():
    # Multiple FoR-only-linked candidates, none series-linked -> demote all to diagnostic.
    a = _ct_series("u1", series_linked=False, for_linked=True)
    b = _ct_series("u2", series_linked=False, for_linked=True)
    _assign_planning_flags([a, b])
    assert (a.is_planning_ct, a.rt_link_basis) == (False, "rtstruct_to_for_ambiguous")
    assert (b.is_planning_ct, b.rt_link_basis) == (False, "rtstruct_to_for_ambiguous")


def test_assign_planning_flags_for_superseded_by_series():
    # A direct series-linked CT supersedes its FoR-only siblings (which become diagnostic).
    direct = _ct_series("u1", series_linked=True, for_linked=True)
    sibling = _ct_series("u2", series_linked=False, for_linked=True)
    _assign_planning_flags([direct, sibling])
    assert (direct.is_planning_ct, direct.rt_link_basis) == (True, "rtstruct_to_series")
    assert (sibling.is_planning_ct, sibling.rt_link_basis) == (False, "for_superseded_by_series")


def test_classifier_meta_emits_rt_link_contract():
    # Production-contract test: classifier_meta() MUST emit the rt-link keys the classifier
    # reads (rt_series_linked / is_planning_ct / rt_linked / rt_link_basis), so the FIX-I
    # recovery cannot silently break if the meta builder changes.
    s = _ct_series("u1", series_linked=True, for_linked=True, model="Plastimatch")
    meta = s.classifier_meta()
    for key in ("rt_linked", "rt_series_linked", "is_planning_ct", "rt_link_basis"):
        assert key in meta, f"classifier_meta missing {key!r}"
    assert meta["rt_series_linked"] is True


def test_series_linked_unrecognized_ct_recovers_via_production_path():
    # End-to-end (not the dead branch): a series-linked unrecognized-vendor CT gets
    # is_planning_ct=True from _assign_planning_flags, then classifies as planning_ct.
    s = _ct_series("u1", series_linked=True, for_linked=True, model="Plastimatch", n=200)
    _assign_planning_flags([s])
    image_class, reason = classify_series(s.classifier_meta())
    assert image_class == "planning_ct"
    assert reason is None
