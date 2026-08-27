"""A QA phantom must never be classified as a patient CT.

Delta4 and similar dosimetry volumes are exported with ``Modality=CT`` and DO
carry an RTSTRUCT, because the QA plan is computed on the phantom. The
RTSTRUCT-bound recovery in ``_classify_ct`` exists to rescue genuine TPS
exports whose manufacturer is not a scanner vendor, and it would otherwise
rescue phantoms too, sending phantom radiomics into a patient feature table
where nothing downstream could tell the difference.

Observed in a DFCI trimodality bladder export: 169 ScandiDos VirtualCT series
across 99 of 154 patients, alongside ~21 scanned films carrying Modality=CT at
Rows=1504, SliceThickness=820.
"""

from __future__ import annotations

import pytest

from rtpipeline.modality_classifier import classify_series


def _series(**overrides):
    meta = {
        "modality": "CT",
        "n_instances": 120,
        "manufacturer": "SIEMENS",
        "manufacturer_model": "SOMATOM Confidence",
        "series_description": "PELVIS",
        "image_type": ["ORIGINAL", "PRIMARY", "AXIAL"],
        "slice_thickness": 3.0,
    }
    meta.update(overrides)
    return meta


def test_baseline_planning_ct_still_classifies():
    image_class, reason = classify_series(_series(is_planning_ct=True))
    assert image_class == "planning_ct"
    assert reason is None


def test_scandidos_virtualct_is_excluded_even_when_rtstruct_linked():
    """The exact contamination path: phantom + RTSTRUCT would become planning_ct."""
    image_class, reason = classify_series(
        _series(
            manufacturer="ScandiDos",
            manufacturer_model="VirtualCT",
            series_description="Delta4 verification",
            is_planning_ct=True,
            rt_series_linked=True,
        )
    )
    assert image_class == "exclude"
    assert reason.startswith("qa_phantom_")


@pytest.mark.parametrize(
    "manufacturer",
    ["ScandiDos", "Sun Nuclear", "Standard Imaging", "PTW", "IBA Dosimetry", "CIRS"],
)
def test_phantom_manufacturers_are_excluded(manufacturer):
    image_class, reason = classify_series(
        _series(manufacturer=manufacturer, is_planning_ct=True)
    )
    assert image_class == "exclude"
    assert reason.startswith("qa_phantom_manufacturer_")


@pytest.mark.parametrize(
    "model", ["VirtualCT", "Virtual CT", "Delta4", "Delta 4", "ArcCHECK", "Catphan", "OCTAVIUS"]
)
def test_phantom_models_are_excluded(model):
    image_class, reason = classify_series(
        _series(manufacturer_model=model, is_planning_ct=True)
    )
    assert image_class == "exclude"
    assert reason.startswith("qa_phantom_model_")


@pytest.mark.parametrize(
    "description",
    ["QA phantom", "Delta4 QA", "ArcCHECK daily", "Quality Assurance CT", "CATPHAN 504"],
)
def test_phantom_descriptions_are_excluded(description):
    image_class, reason = classify_series(
        _series(series_description=description, is_planning_ct=True)
    )
    assert image_class == "exclude"
    assert reason.startswith("qa_phantom_")


def test_scanned_film_geometry_is_excluded():
    """Rows=1504 with SliceThickness=820 is a scanned film, not an acquisition."""
    image_class, reason = classify_series(
        _series(rows=1504, slice_thickness=820.0, series_description="", is_planning_ct=True)
    )
    assert image_class == "exclude"
    assert reason == "ct_implausible_slice_thickness"


@pytest.mark.parametrize("thickness", [0.6, 1.0, 1.5, 2.5, 3.0, 5.0, 10.0])
def test_plausible_slice_thicknesses_are_kept(thickness):
    image_class, _ = classify_series(_series(slice_thickness=thickness, is_planning_ct=True))
    assert image_class == "planning_ct"


@pytest.mark.parametrize("thickness", [None, "", "not-a-number"])
def test_unparseable_thickness_does_not_exclude(thickness):
    """A missing thickness is not evidence of a film; it must not drop real data."""
    image_class, _ = classify_series(_series(slice_thickness=thickness, is_planning_ct=True))
    assert image_class == "planning_ct"


def test_phantom_gate_precedes_the_cbct_path():
    image_class, reason = classify_series(
        _series(
            modality="CT",
            manufacturer="Varian Medical Systems",
            manufacturer_model="Patient Verification",
            series_description="Delta4 phantom check",
        )
    )
    assert image_class == "exclude"
    assert reason.startswith("qa_phantom_")


def test_anatomy_named_like_a_vendor_token_is_not_a_phantom():
    """Guard against over-exclusion: ordinary anatomy must survive the gate."""
    for description in ("PELVIS", "THORAX 1.5mm", "Bladder boost planning", "Miednica"):
        image_class, _ = classify_series(
            _series(series_description=description, is_planning_ct=True)
        )
        assert image_class == "planning_ct", description
