from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from rtpipeline import radiomics_ct_contract as contract
from rtpipeline.radiomics_conda import _ct_publication_key_text
from rtpipeline.radiomics_outcomes import resume_identity_pairs


class _FakeExtractor:
    def __init__(self) -> None:
        self.settings = {
            "minimumROISize": 64,
            "minimumROIDimensions": 2,
            "binWidth": 25,
            "resampledPixelSpacing": [1.0, 1.0, 1.0],
            "interpolator": "sitkBSpline",
        }
        self.enabledImagetypes = {"Original": {}}
        self.enabledFeatures = {"shape": [], "firstorder": [], "glcm": []}

    def disableAllImageTypes(self) -> None:
        self.enabledImagetypes = {}

    def enableImageTypeByName(self, name: str) -> None:
        self.enabledImagetypes[name] = {}

    def disableAllFeatures(self) -> None:
        self.enabledFeatures = {}

    def enableFeatureClassByName(self, name: str) -> None:
        self.enabledFeatures[name] = []

    def execute(self, _image, _mask):
        if set(self.enabledFeatures) == {"shape"}:
            return {
                "original_shape_VoxelVolume": 125.0,
                "original_shape_SurfaceArea": 150.0,
            }
        return {
            "original_firstorder_Mean": 20.0,
            "original_glcm_Contrast": 2.0,
        }


def _common_identity() -> dict[str, str]:
    return {
        "patient_id": "P1",
        "course_id": "C1",
        "series_uid": "1.2.3",
        "segmentation_source": "Manual",
        "mask_identity": "1.2.840.1",
        "roi_original_name": "PTV",
        "stable_roi_identifier": "rtstruct_roi_number:7",
        "roi_name": "PTV",
        "modality": "CT",
    }


def _qc(*_args, **_kwargs) -> dict[str, object]:
    return {
        "morphologic_resampled_voxel_count": 100,
        "resegment_after_count": 80,
        "resegment_below_lower_count": 10,
        "resegment_above_upper_count": 9,
        "resegment_nonfinite_count": 1,
        "components_26_before": 1,
        "components_26_after": 2,
        "largest_component_voxel_count_before": 100,
        "largest_component_voxel_count_after": 75,
        "resegment_retained_fraction": 0.8,
        "largest_component_retained_fraction": 0.75,
        "largest_component_fraction_after": 0.9375,
        "component_count_increased": True,
        "observed_roi_dimensions_after_resegmentation": 3,
    }


def _versions() -> dict[str, str]:
    return {
        "pyradiomics_version": "3.0.1",
        "simpleitk_version": sitk.Version_VersionString(),
        "numpy_version": np.__version__,
    }


def _dual_arm_rows(monkeypatch, *, decision=None, required: bool = True):
    monkeypatch.setattr(contract, "resampled_mask_qc", _qc)
    monkeypatch.setattr(contract, "_runtime_versions", _versions)
    decision = decision or contract.classify_ct_roi("Manual", "PTV")
    return contract.extract_ct_roi_arms(
        object(),
        object(),
        factory=_FakeExtractor,
        decision=decision,
        common_metadata=_common_identity(),
        run_identifier="run-1",
        code_revision="revision-1",
        native_voxel_count=120,
        required=required,
        configured_parameter_hashes={
            contract.PRIMARY_ARM: "configured-primary",
            contract.SENSITIVITY_ARM: "configured-sensitivity",
        },
    )


def test_dual_arm_shape_is_computed_once_without_resegmentation_and_is_identical(monkeypatch):
    rows = _dual_arm_rows(monkeypatch)

    assert {row["extraction_arm"] for row in rows} == set(contract.CT_EXTRACTION_ARMS)
    for feature in ("original_shape_VoxelVolume", "original_shape_SurfaceArea"):
        assert {row[feature] for row in rows} == {125.0 if feature.endswith("VoxelVolume") else 150.0}
    primary = next(row for row in rows if row["extraction_arm"] == contract.PRIMARY_ARM)
    sensitivity = next(row for row in rows if row["extraction_arm"] == contract.SENSITIVITY_ARM)
    assert primary["effective_resegment_lower_hu"] == -1000.0
    assert sensitivity["effective_resegment_lower_hu"] is None
    assert primary["original_shape_VoxelVolume"] == sensitivity["original_shape_VoxelVolume"]


def test_resampled_count_identity_uses_closed_window_and_26_connectivity(monkeypatch):
    image_array = np.array(
        [
            [[-1001.0, -1000.0], [400.0, 401.0]],
            [[np.nan, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    mask_array = np.ones_like(image_array, dtype=np.uint8)
    image = sitk.GetImageFromArray(image_array)
    mask = sitk.GetImageFromArray(mask_array)

    imageoperations = ModuleType("radiomics.imageoperations")
    imageoperations.checkMask = lambda loaded_image, loaded_mask, **_settings: (None, None)
    radiomics_module = ModuleType("radiomics")
    radiomics_module.imageoperations = imageoperations
    monkeypatch.setitem(sys.modules, "radiomics", radiomics_module)
    monkeypatch.setitem(sys.modules, "radiomics.imageoperations", imageoperations)

    extractor = _FakeExtractor()
    extractor.settings["label"] = 1
    extractor.loadImage = lambda *_args, **_kwargs: (image, mask)

    qc = contract.resampled_mask_qc(image, mask, extractor, (-1000.0, 400.0))

    assert qc["morphologic_resampled_voxel_count"] == 8
    assert qc["resegment_after_count"] == 5
    assert qc["resegment_below_lower_count"] == 1
    assert qc["resegment_above_upper_count"] == 1
    assert qc["resegment_nonfinite_count"] == 1
    assert 5 + 1 + 1 + 1 == 8
    assert qc["components_26_before"] == 1
    assert qc["components_26_after"] == 1


@pytest.mark.parametrize(
    ("source", "name", "roi_class", "window", "disposition"),
    [
        ("Manual", "PTV", "target", (-1000.0, 400.0), "success"),
        ("AutoRTS_total", "urinary_bladder", "hollow_pelvic_organ", (-1000.0, 400.0), "success"),
        ("AutoRTS_total", "brain", "solid_soft_tissue_neural", (-500.0, 400.0), "success"),
        ("AutoRTS_total", "femur_left", "bone", None, "not_applicable_bone"),
        ("AutoRTS_total", "aorta", "vessel", None, "not_applicable_pending_vessel_adjudication"),
        ("Manual", "positioning pad", "unresolved_mixed", None, "unclassified_roi"),
    ],
)
def test_class_governed_primary_windows(source, name, roi_class, window, disposition):
    decision = contract.classify_ct_roi(source, name)
    assert decision.roi_class == roi_class
    assert decision.primary_resegment_range_hu == window
    assert decision.primary_intensity_texture_disposition == disposition


def test_derived_roi_requires_recorded_operation_and_classifiable_bases():
    accepted = contract.classify_ct_roi(
        "Custom",
        "bowel_bag",
        custom_provenance={
            "bowel_bag": {
                "operation": "union",
                "source_structures": ["colon", "small_bowel", "duodenum"],
            }
        },
    )
    rejected = contract.classify_ct_roi("Custom", "bowel_bag", custom_provenance={})

    assert accepted.roi_class == "hollow_pelvic_organ"
    assert accepted.map_entry_source == "derived_crosswalk:recorded_operation_and_classified_bases"
    assert accepted.adjudication_status == "approved_by_binding_spec"
    assert accepted.feature_publication_policy == contract.FEATURE_POLICY_EXTRACT
    assert rejected.roi_class == "unresolved_mixed"
    assert rejected.adjudication_status == "operator_adjudication_required"


@pytest.mark.parametrize(
    (
        "name",
        "roi_class",
        "primary_disposition",
        "entry_source",
        "adjudication_status",
        "feature_policy",
    ),
    [
        (
            "iliac_vess",
            "vessel",
            "not_applicable_pending_vessel_adjudication",
            "derived_crosswalk:recorded_operation_and_classified_bases",
            "approved_by_binding_spec",
            contract.FEATURE_POLICY_EXTRACT,
        ),
        (
            "iliac_area",
            "planning_helper",
            "not_applicable_planning_helper",
            "derived_crosswalk:recorded_margin_operation",
            "approved_non_anatomic_derived_margin",
            contract.FEATURE_POLICY_INVENTORY_ONLY,
        ),
        (
            "pelvic_bones",
            "bone",
            "not_applicable_bone",
            "derived_crosswalk:recorded_operation_and_classified_bases",
            "approved_by_binding_spec",
            contract.FEATURE_POLICY_EXTRACT,
        ),
        (
            "pelvic_bones_3mm",
            "planning_helper",
            "not_applicable_planning_helper",
            "derived_crosswalk:recorded_margin_operation",
            "approved_non_anatomic_derived_margin",
            contract.FEATURE_POLICY_INVENTORY_ONLY,
        ),
        (
            "bowel_bag",
            "hollow_pelvic_organ",
            "success",
            "derived_crosswalk:recorded_operation_and_classified_bases",
            "approved_by_binding_spec",
            contract.FEATURE_POLICY_EXTRACT,
        ),
    ],
)
def test_configured_derived_rois_follow_operation_and_margin_provenance(
    name,
    roi_class,
    primary_disposition,
    entry_source,
    adjudication_status,
    feature_policy,
):
    config = Path(contract.__file__).parents[1] / "custom_structures_pelvic.yaml"
    provenance = contract.load_custom_structure_provenance(config)

    decision = contract.classify_ct_roi(
        "Custom", name, custom_provenance=provenance
    )

    assert decision.roi_class == roi_class
    assert decision.map_entry_source == entry_source
    assert decision.adjudication_status == adjudication_status
    assert decision.primary_intensity_texture_disposition == primary_disposition
    assert decision.feature_publication_policy == feature_policy


def test_partial_derived_roi_remains_inventory_only_pending_adjudication():
    config = Path(contract.__file__).parents[1] / "custom_structures_pelvic.yaml"
    decision = contract.classify_ct_roi(
        "Custom",
        "bowel_bag__partial",
        custom_provenance=contract.load_custom_structure_provenance(config),
    )

    assert decision.roi_class == "planning_helper"
    assert decision.map_entry_source == (
        "derived_crosswalk:recorded_partial_boolean_operation"
    )
    assert decision.adjudication_status == (
        "partial_derived_structure_pending_adjudication"
    )
    assert decision.feature_publication_policy == contract.FEATURE_POLICY_INVENTORY_ONLY


@pytest.mark.parametrize(
    ("name", "primary_disposition"),
    [
        ("iliac_vess", "not_applicable_pending_vessel_adjudication"),
        ("pelvic_bones", "not_applicable_bone"),
    ],
)
def test_derived_vessel_and_bone_publish_shape_without_primary_intensity_texture(
    monkeypatch, name, primary_disposition
):
    config = Path(contract.__file__).parents[1] / "custom_structures_pelvic.yaml"
    decision = contract.classify_ct_roi(
        "Custom",
        name,
        custom_provenance=contract.load_custom_structure_provenance(config),
    )

    rows = _dual_arm_rows(monkeypatch, decision=decision)
    primary = next(row for row in rows if row["extraction_arm"] == contract.PRIMARY_ARM)
    sensitivity = next(
        row for row in rows if row["extraction_arm"] == contract.SENSITIVITY_ARM
    )

    assert primary["extraction_status"] == "success"
    assert primary["shape_disposition"] == "success"
    assert primary["intensity_texture_disposition"] == primary_disposition
    assert primary["original_shape_VoxelVolume"] == 125.0
    assert not any(
        marker in key
        for key in primary
        for marker in contract.INTENSITY_TEXTURE_FEATURE_MARKERS
    )
    assert sensitivity["extraction_status"] == "success"
    assert sensitivity["original_firstorder_Mean"] == 20.0
    assert sensitivity["original_glcm_Contrast"] == 2.0


@pytest.mark.parametrize(
    ("name", "roi_class"),
    [
        ("Aorta", "vessel"),
        ("CTV1", "target"),
        ("CTV2", "target"),
        ("CTV3", "target"),
        ("Dwunastnica", "hollow_pelvic_organ"),
        ("Esica__partial", "hollow_pelvic_organ"),
        ("Jelito cienkie__partial", "hollow_pelvic_organ"),
        ("Jelito grube__partial", "hollow_pelvic_organ"),
        ("Kora Nerek Suma", "solid_soft_tissue_neural"),
        ("Kora Nerki Lewej", "solid_soft_tissue_neural"),
        ("Kora Nerki Prawej", "solid_soft_tissue_neural"),
        ("Kosci", "bone"),
        ("LAD", "vessel"),
        ("Mostek", "bone"),
        ("Nerka Lewa", "solid_soft_tissue_neural"),
        ("Nerka Prawa", "solid_soft_tissue_neural"),
        ("Nerki", "solid_soft_tissue_neural"),
        ("PTV1", "target"),
        ("PTV2", "target"),
        ("PTV3", "target"),
        ("Pluco Lewe", "unresolved_mixed"),
        ("Pluco Prawe", "unresolved_mixed"),
        ("Pluco Suma", "unresolved_mixed"),
        ("Przelyk", "unresolved_mixed"),
        ("Rdzen", "solid_soft_tissue_neural"),
        ("Serce", "unresolved_mixed"),
        ("Sledziona", "solid_soft_tissue_neural"),
        ("Splot Ramienny Lewy", "solid_soft_tissue_neural"),
        ("Splot Ramienny Prawy", "solid_soft_tissue_neural"),
        ("Tarczyca", "solid_soft_tissue_neural"),
        ("Tchawica", "unresolved_mixed"),
        ("Tetnica plucna", "vessel"),
        ("Watroba", "solid_soft_tissue_neural"),
        ("Zoladek", "unresolved_mixed"),
        ("Zyla Glowna Dolna", "vessel"),
        ("jelita", "hollow_pelvic_organ"),
        ("ogon konski", "solid_soft_tissue_neural"),
    ],
)
def test_kopernik_exact_crosswalk_uses_defensible_anatomic_classes(name, roi_class):
    decision = contract.classify_ct_roi("Manual", name)

    assert decision.roi_class == roi_class
    assert decision.adjudication_status == "approved_by_anatomic_equivalence"


@pytest.mark.parametrize(
    "name",
    [
        "Bronchus_L",
        "Bronchus_R",
        "Krtan",
        "Oskrzele_L",
        "Oskrzele_R",
        "PBT",
        "Pluco Suma M",
        "Pluco Suma M - PTV1",
        "Pluco Suma M - PTV3",
        "Rdzen Marg",
    ],
)
def test_kopernik_ambiguous_names_remain_unadjudicated_without_a_governed_class(name):
    decision = contract.classify_ct_roi("Manual", name)

    assert decision.roi_class == "unresolved_mixed"
    assert decision.adjudication_status == "operator_adjudication_required"


@pytest.mark.parametrize(
    ("name", "roi_class"),
    [
        ("2cm od PTV1", "planning_helper"),
        ("Bowel_Bag", "planning_helper"),
        ("Bowel_Small_PRV5mm", "planning_helper"),
        ("Fiz Body", "planning_helper"),
        ("Jelita - PTV", "planning_helper"),
        ("m3", "planning_helper"),
        ("m4", "planning_helper"),
        ("m5", "planning_helper"),
        ("m7", "planning_helper"),
        ("m8", "planning_helper"),
        ("obszar", "planning_helper"),
        ("podkladka", "positioning_support"),
        ("zzCivcoInterior", "positioning_support"),
        ("zz_2cm od PTV12", "planning_helper"),
    ],
)
def test_exact_non_anatomic_names_are_inventory_only(name, roi_class):
    decision = contract.classify_ct_roi("Manual", name)

    assert decision.roi_class == roi_class
    assert decision.feature_publication_policy == contract.FEATURE_POLICY_INVENTORY_ONLY


def test_inventory_only_roi_retains_identity_without_feature_values(monkeypatch):
    rows = _dual_arm_rows(
        monkeypatch,
        decision=contract.classify_ct_roi("Manual", "m3"),
        required=False,
    )

    assert {row["extraction_status"] for row in rows} == {
        "not_applicable_planning_helper"
    }
    assert {row["extraction_failure_kind"] for row in rows} == {"declared_ineligible"}
    assert {row["roi_class"] for row in rows} == {"planning_helper"}
    assert all(not any("_shape_" in key or "_firstorder_" in key for key in row) for row in rows)


class _InvalidShapeExtractor(_FakeExtractor):
    def execute(self, image, mask):
        result = super().execute(image, mask)
        if set(self.enabledFeatures) == {"shape"}:
            result.update(
                {
                    "original_shape_MeshVolume": -29.458333333333332,
                    "original_shape_SurfaceVolumeRatio": -11.745545713423589,
                    "original_shape_Sphericity": 0.13331372628121332,
                }
            )
        return result


def test_invalid_shape_physicality_is_recorded_without_feature_values(monkeypatch):
    monkeypatch.setattr(contract, "resampled_mask_qc", _qc)
    monkeypatch.setattr(contract, "_runtime_versions", _versions)
    rows = contract.extract_ct_roi_arms(
        object(),
        object(),
        factory=_InvalidShapeExtractor,
        decision=contract.classify_ct_roi("AutoRTS_total", "skull"),
        common_metadata={**_common_identity(), "roi_original_name": "skull", "roi_name": "skull"},
        run_identifier="run-1",
        code_revision="revision-1",
        native_voxel_count=150,
        required=False,
        configured_parameter_hashes={
            contract.PRIMARY_ARM: "configured-primary",
            contract.SENSITIVITY_ARM: "configured-sensitivity",
        },
    )

    assert {row["extraction_status"] for row in rows} == {
        "failed_shape_physical_validity"
    }
    assert {row["extraction_failure_kind"] for row in rows} == {
        "invalid_shape_physicality"
    }
    assert all("original_shape_MeshVolume" not in row for row in rows)
    assert all("original_shape_MeshVolume=-29.458333333333332" in row["extraction_status_detail"] for row in rows)


def test_disposition_rows_require_runtime_effective_parameter_hashes():
    decision = contract.classify_ct_roi("Manual", "PTV")

    with pytest.raises(ValueError, match="effective-parameter hashes"):
        contract.disposition_rows_for_arms(
            _common_identity(),
            decision=decision,
            disposition="below_minimum_voxels",
            detail="native mask is too small",
            failure_kind="degenerate_mask",
            run_identifier="run-1",
            code_revision="revision-1",
            native_voxel_count=3,
            required=True,
            configured_parameter_hashes={
                contract.PRIMARY_ARM: "configured-primary",
                contract.SENSITIVITY_ARM: "configured-sensitivity",
            },
        )


def test_required_unclassified_roi_blocks_publication(monkeypatch):
    decision = contract.classify_ct_roi("Manual", "operator must adjudicate this")
    rows = contract.disposition_rows_for_arms(
        {**_common_identity(), "roi_original_name": "operator must adjudicate this", "roi_name": "operator must adjudicate this"},
        decision=decision,
        disposition="below_minimum_voxels",
        detail="native mask is too small",
        failure_kind="degenerate_mask",
        run_identifier="run-1",
        code_revision="revision-1",
        native_voxel_count=3,
        required=True,
        configured_parameter_hashes={
            contract.PRIMARY_ARM: "configured-primary",
            contract.SENSITIVITY_ARM: "configured-sensitivity",
        },
        effective_hashes={
            contract.PRIMARY_ARM: "effective-primary",
            contract.SENSITIVITY_ARM: "effective-sensitivity",
        },
    )

    with pytest.raises(ValueError, match="operator adjudication is required"):
        contract.validate_ct_publication(pd.DataFrame(rows))


def test_every_resume_checkpoint_and_publication_identity_contains_arm(monkeypatch):
    rows = _dual_arm_rows(monkeypatch)
    frame = pd.DataFrame(rows)
    publication_keys = contract.validate_ct_publication(frame)
    resume_keys = resume_identity_pairs(frame)
    checkpoint_keys = {_ct_publication_key_text(row) for row in rows}

    assert publication_keys == resume_keys
    assert len(publication_keys) == 2
    assert {key[-1] for key in publication_keys} == set(contract.CT_EXTRACTION_ARMS)
    assert len(checkpoint_keys) == 2
    assert all(key.rsplit("\x1f", 1)[-1] in contract.CT_EXTRACTION_ARMS for key in checkpoint_keys)


def test_validation_rejects_count_identity_mismatch(monkeypatch):
    rows = _dual_arm_rows(monkeypatch)
    rows[0]["resegment_after_count"] = 79
    with pytest.raises(ValueError, match="resegmentation count identity failed"):
        contract.validate_ct_publication(pd.DataFrame(rows))


def test_completion_sentinel_binds_full_identity_and_current_hashes(tmp_path, monkeypatch):
    course = tmp_path / "P1" / "C1"
    rows = _dual_arm_rows(monkeypatch)
    contract.write_ct_publication_atomic(
        pd.DataFrame(rows), course / "radiomics_ct.xlsx"
    )
    sentinel = contract.write_completion_sentinel(course)

    payload = contract.validate_completion_sentinel(course, sentinel)
    assert payload["row_count"] == 2
    assert payload["configured_parameter_hashes"] == [
        "configured-primary",
        "configured-sensitivity",
    ]

    stale_rows = [dict(row) for row in rows]
    for row in stale_rows:
        row["configured_parameter_hash"] = "new-config-" + row["extraction_arm"]
    contract.write_ct_publication_atomic(
        pd.DataFrame(stale_rows), course / "radiomics_ct.xlsx"
    )
    with pytest.raises(ValueError, match="sentinel is stale"):
        contract.validate_completion_sentinel(course, sentinel)


def test_totalsegmentator_vocabulary_hash_is_enforced(tmp_path):
    source = Path(contract.__file__).with_name("roi_class_map_v1.yaml")
    text = source.read_text(encoding="utf-8").replace(
        "    liver:\n      roi_class: solid_soft_tissue_neural",
        "    liver:\n      roi_class: bone",
        1,
    )
    altered = tmp_path / "altered-map.yaml"
    altered.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="vocabulary hash is stale"):
        contract.load_roi_class_map(str(altered))
