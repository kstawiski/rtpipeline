"""Regression tests for the 2026-07 hygiene batch A bug fixes:

1. cli._apply_dicom_copy_yaml_config: organize.use_hardlinks YAML default must be
   False (matching PipelineConfig.dicom_copy_use_hardlinks), not True.
2. meta._merge_plans_doses: RP/RD rows whose filename doesn't match the expected
   pattern get a null core_key and must be dropped before the merge, so they
   don't cross-join against each other.
3. anatomical_cropping.apply_systematic_cropping: superior/inferior margin
   sentinel-override bug -- explicit margins equal to a default value must be
   honored, not silently downgraded. Omitted (None) margins still get the
   region-specific default. Pelvis is unaffected either way.
4. anatomical_cropping.apply_systematic_cropping / PipelineConfig: the
   keep_original parameter (a documented-but-never-acted-on no-op) is kept as a
   DEPRECATED no-op for backward compatibility -- accepted and ignored, with a
   one-time DeprecationWarning when a caller passes a non-default value.
5. segmentation.segment_course: the resume fast-path must resolve dicom_seg to
   the modern "{base_name}--{model}.dcm" RTSTRUCT name (or the legacy
   "{model}.dcm" name as fallback), not the dead legacy-only "total.dcm" name.
6. pet_suv.CLINICAL_ROOT: the institution-specific hardcoded path has been
   replaced with an opt-in RTPIPELINE_CLINICAL_ROOT env var, default None.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import inspect
import json
from pathlib import Path

import pandas as pd

import rtpipeline.anatomical_cropping as anatomical_cropping
import rtpipeline.cli as cli
import rtpipeline.meta as meta
import rtpipeline.pet_suv as pet_suv
import rtpipeline.segmentation as segmentation
from rtpipeline.config import PipelineConfig
from rtpipeline.layout import build_course_dirs


def _cfg(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        dicom_root=tmp_path / "dicom",
        output_root=tmp_path / "out",
        logs_root=tmp_path / "logs",
    )


# ---------------------------------------------------------------------------
# 1. Hardlink default typo (cli.py)
# ---------------------------------------------------------------------------

def test_organize_yaml_use_hardlinks_defaults_false_when_omitted(tmp_path):
    cfg = _cfg(tmp_path)
    cli._apply_dicom_copy_yaml_config(cfg, {})  # 'organize' block omits use_hardlinks
    assert cfg.dicom_copy_use_hardlinks is False


def test_organize_yaml_use_hardlinks_explicit_true_is_honored(tmp_path):
    cfg = _cfg(tmp_path)
    cli._apply_dicom_copy_yaml_config(cfg, {"use_hardlinks": True})
    assert cfg.dicom_copy_use_hardlinks is True


def test_organize_yaml_default_matches_dataclass_default(tmp_path):
    # The YAML-omitted fallback must mirror config.py's own dataclass default,
    # otherwise a config file that simply omits the key silently changes behavior.
    cfg = _cfg(tmp_path)
    cli._apply_dicom_copy_yaml_config(cfg, {})
    assert cfg.dicom_copy_use_hardlinks == PipelineConfig(
        dicom_root=tmp_path, output_root=tmp_path, logs_root=tmp_path
    ).dicom_copy_use_hardlinks


# ---------------------------------------------------------------------------
# 2. meta.py cartesian join on unmatched (null core_key) RP/RD rows
# ---------------------------------------------------------------------------

def test_merge_plans_doses_drops_null_core_keys_before_merge():
    plans_df = pd.DataFrame(
        {
            "file_path": [
                "/x/RP.unmatched_name_1.dcm",  # doesn't match R[PD].<id>.<desc>.dcm -> None
                "/x/RP.unmatched_name_2.dcm",  # -> None
                "/x/RP.100.Prostate.dcm",      # -> core_key "100.Prostate" (real match)
            ],
            "plan_id": ["p1", "p2", "p3"],
        }
    )
    doses_df = pd.DataFrame(
        {
            "file_path": [
                "/y/RD.unmatched_name_1.dcm",  # -> None
                "/y/RD.unmatched_name_2.dcm",  # -> None
                "/y/RD.unmatched_name_3.dcm",  # -> None
                "/y/RD.100.Prostate.dcm",      # -> core_key "100.Prostate" (real match with p3)
            ],
            "dose_id": ["d1", "d2", "d3", "d4"],
        }
    )

    merged = meta._merge_plans_doses(plans_df, doses_df)

    # Pre-fix, the 2 null-keyed plan rows x 3 null-keyed dose rows would have
    # cross-joined into 6 spurious rows in addition to the real match.
    assert len(merged) == 1
    row = merged.iloc[0]
    assert row["core_key"] == "100.Prostate"
    assert row["plan_id"] == "p3"
    assert row["dose_id"] == "d4"


def test_merge_plans_doses_no_real_matches_yields_empty_frame():
    plans_df = pd.DataFrame({"file_path": ["/x/RP.bad1.dcm", "/x/RP.bad2.dcm"], "plan_id": ["p1", "p2"]})
    doses_df = pd.DataFrame({"file_path": ["/y/RD.bad1.dcm", "/y/RD.bad2.dcm"], "dose_id": ["d1", "d2"]})

    merged = meta._merge_plans_doses(plans_df, doses_df)

    assert len(merged) == 0


def test_core_key_from_filename():
    assert meta._core_key_from_filename("/a/b/RP.100.Prostate.dcm") == "100.Prostate"
    assert meta._core_key_from_filename("/a/b/RD.100.Prostate.dcm") == "100.Prostate"
    assert meta._core_key_from_filename("/a/b/not_a_plan_or_dose.dcm") is None


# ---------------------------------------------------------------------------
# 3. Sentinel margin override (anatomical_cropping.py)
# ---------------------------------------------------------------------------

def test_brain_explicit_margin_equal_to_default_is_not_downgraded(tmp_path, monkeypatch):
    captured: dict = {}

    def fake_brain_boundaries(course_dir, superior_margin_cm=1.0, inferior_margin_cm=1.0):
        captured["superior"] = superior_margin_cm
        captured["inferior"] = inferior_margin_cm
        return 100.0, -100.0

    monkeypatch.setattr(anatomical_cropping, "determine_brain_crop_boundaries", fake_brain_boundaries)

    # Explicitly pass 2.0 (equal to the OLD, unrelated pelvis/thorax sentinel value).
    # Pre-fix this was silently downgraded to the brain default of 1.0.
    anatomical_cropping.apply_systematic_cropping(
        tmp_path, region="brain", superior_margin_cm=2.0, inferior_margin_cm=None
    )
    assert captured["superior"] == 2.0
    assert captured["inferior"] == 1.0  # omitted -> brain region default


def test_brain_omitted_margins_use_region_defaults(tmp_path, monkeypatch):
    captured: dict = {}

    def fake_brain_boundaries(course_dir, superior_margin_cm=1.0, inferior_margin_cm=1.0):
        captured["superior"] = superior_margin_cm
        captured["inferior"] = inferior_margin_cm
        return 100.0, -100.0

    monkeypatch.setattr(anatomical_cropping, "determine_brain_crop_boundaries", fake_brain_boundaries)

    anatomical_cropping.apply_systematic_cropping(tmp_path, region="brain")
    assert captured["superior"] == 1.0
    assert captured["inferior"] == 1.0


def test_thorax_explicit_inferior_margin_equal_to_pelvis_default_is_not_downgraded(tmp_path, monkeypatch):
    captured: dict = {}

    def fake_thorax_boundaries(course_dir, superior_margin_cm=2.0, inferior_margin_cm=2.0):
        captured["superior"] = superior_margin_cm
        captured["inferior"] = inferior_margin_cm
        return 100.0, -100.0

    monkeypatch.setattr(anatomical_cropping, "determine_thorax_crop_boundaries", fake_thorax_boundaries)

    # 10.0 is the pelvis inferior-margin default, not thorax's (2.0). Pre-fix this
    # was silently downgraded to 2.0 because it matched the old cross-region sentinel.
    anatomical_cropping.apply_systematic_cropping(
        tmp_path, region="thorax", superior_margin_cm=2.0, inferior_margin_cm=10.0
    )
    assert captured["superior"] == 2.0
    assert captured["inferior"] == 10.0


def test_thorax_omitted_margins_use_region_defaults(tmp_path, monkeypatch):
    captured: dict = {}

    def fake_thorax_boundaries(course_dir, superior_margin_cm=2.0, inferior_margin_cm=2.0):
        captured["superior"] = superior_margin_cm
        captured["inferior"] = inferior_margin_cm
        return 100.0, -100.0

    monkeypatch.setattr(anatomical_cropping, "determine_thorax_crop_boundaries", fake_thorax_boundaries)

    anatomical_cropping.apply_systematic_cropping(tmp_path, region="thorax")
    assert captured["superior"] == 2.0
    assert captured["inferior"] == 2.0


def test_pelvis_margin_resolution_is_unaffected_by_the_fix(tmp_path, monkeypatch):
    captured: dict = {}

    def fake_pelvic_boundaries(course_dir, inferior_margin_cm=10.0):
        captured["inferior"] = inferior_margin_cm
        return 100.0, -100.0

    monkeypatch.setattr(anatomical_cropping, "determine_pelvic_crop_boundaries", fake_pelvic_boundaries)

    # Omitted -> pelvis default (10.0)
    anatomical_cropping.apply_systematic_cropping(tmp_path, region="pelvis")
    assert captured["inferior"] == 10.0

    # Explicit value equal to the default -> still honored (was already correct pre-fix)
    anatomical_cropping.apply_systematic_cropping(tmp_path, region="pelvis", inferior_margin_cm=10.0)
    assert captured["inferior"] == 10.0

    # Explicit different value -> passed through unchanged
    anatomical_cropping.apply_systematic_cropping(tmp_path, region="pelvis", inferior_margin_cm=5.0)
    assert captured["inferior"] == 5.0


def test_cropping_metadata_records_effective_margins_not_raw_none(tmp_path, monkeypatch):
    monkeypatch.setattr(
        anatomical_cropping,
        "determine_brain_crop_boundaries",
        lambda course_dir, superior_margin_cm=1.0, inferior_margin_cm=1.0: (100.0, -100.0),
    )

    anatomical_cropping.apply_systematic_cropping(tmp_path, region="brain")

    metadata = json.loads((tmp_path / "cropping_metadata.json").read_text(encoding="utf-8"))
    assert metadata["superior_margin_cm"] == 1.0
    assert metadata["inferior_margin_cm"] == 1.0


# ---------------------------------------------------------------------------
# 4. keep_original kept as a DEPRECATED no-op (public API compatibility)
# ---------------------------------------------------------------------------

def test_keep_original_parameter_accepted_by_apply_systematic_cropping():
    params = inspect.signature(anatomical_cropping.apply_systematic_cropping).parameters
    assert "keep_original" in params
    assert params["keep_original"].default is True


def test_keep_original_default_true_is_silent(tmp_path, monkeypatch, recwarn):
    monkeypatch.setattr(
        anatomical_cropping,
        "determine_pelvic_crop_boundaries",
        lambda course_dir, inferior_margin_cm=10.0: (100.0, -100.0),
    )
    anatomical_cropping.apply_systematic_cropping(tmp_path, region="pelvis")
    assert not any(issubclass(w.category, DeprecationWarning) for w in recwarn.list)


def test_keep_original_non_default_warns_and_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(
        anatomical_cropping,
        "determine_pelvic_crop_boundaries",
        lambda course_dir, inferior_margin_cm=10.0: (100.0, -100.0),
    )
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        result = anatomical_cropping.apply_systematic_cropping(
            tmp_path, region="pelvis", keep_original=False
        )
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    # Ignored: behavior (return value / files kept) is unaffected by the flag.
    assert result == {}


def test_keep_original_config_knob_kept_on_pipeline_config(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg.ct_cropping_keep_original is True


def test_keep_original_kept_on_crop_task(tmp_path):
    field_names = {f.name for f in dataclasses.fields(cli._CropTask)}
    assert "keep_original" in field_names


# ---------------------------------------------------------------------------
# 5. segmentation.py resume dead-filename bug
# ---------------------------------------------------------------------------

def _prepare_resumed_course(tmp_path, monkeypatch, *, dicom_name: str) -> tuple[Path, Path]:
    course_dir = tmp_path / "course"
    course_dirs = build_course_dirs(course_dir)
    course_dirs.ensure()

    nifti_path = course_dirs.nifti / "series1.nii.gz"
    nifti_path.write_bytes(b"fake nifti")
    monkeypatch.setattr(segmentation, "_ensure_ct_nifti", lambda *a, **k: nifti_path)

    base_name = "series1"
    base_dir = course_dirs.segmentation_totalseg / base_name
    base_dir.mkdir(parents=True, exist_ok=True)

    dicom_path = base_dir / dicom_name
    dicom_path.write_bytes(b"fake rtstruct")

    (base_dir / "total--liver.nii.gz").write_text("fake mask", encoding="utf-8")
    manifest = {"models": [{"model": "total", "masks": ["total--liver.nii.gz"]}]}
    (base_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    return course_dir, dicom_path


def test_segment_course_resume_uses_modern_named_dicom_seg(tmp_path, monkeypatch):
    course_dir, modern_dicom = _prepare_resumed_course(
        tmp_path, monkeypatch, dicom_name="series1--total.dcm"
    )
    cfg = _cfg(tmp_path)

    results = segmentation.segment_course(cfg, course_dir, force=False)

    assert results["dicom_seg"] == str(modern_dicom)


def test_segment_course_resume_falls_back_to_legacy_named_dicom_seg(tmp_path, monkeypatch):
    course_dir, legacy_dicom = _prepare_resumed_course(
        tmp_path, monkeypatch, dicom_name="total.dcm"
    )
    cfg = _cfg(tmp_path)

    results = segmentation.segment_course(cfg, course_dir, force=False)

    assert results["dicom_seg"] == str(legacy_dicom)


# ---------------------------------------------------------------------------
# 6. pet_suv.py hardcoded institution path
# ---------------------------------------------------------------------------

def test_resolve_clinical_root_env_honors_override(monkeypatch, tmp_path):
    monkeypatch.setenv("RTPIPELINE_CLINICAL_ROOT", str(tmp_path))
    assert pet_suv._resolve_clinical_root_env() == tmp_path


def test_resolve_clinical_root_env_defaults_to_none_when_unset(monkeypatch):
    monkeypatch.delenv("RTPIPELINE_CLINICAL_ROOT", raising=False)
    assert pet_suv._resolve_clinical_root_env() is None


def test_clinical_root_module_default_has_no_hardcoded_institution_path():
    source = inspect.getsource(pet_suv)
    assert "/umed-projekty" not in source
    assert "KOPERNIK" not in source


def test_clinical_measurement_degrades_when_clinical_root_unset(monkeypatch):
    monkeypatch.setattr(pet_suv, "CLINICAL_ROOT", None)
    result = pet_suv._clinical_measurement(
        "P1", dt.date(2024, 1, 1), key_predicate=lambda k, p: k == "weight_kg", window_days=30
    )
    assert result == (None, None, None)


def test_clinical_measurement_uses_configured_clinical_root(tmp_path, monkeypatch):
    clinical_root = tmp_path / "clinical"
    patient_dir = clinical_root / "P1"
    patient_dir.mkdir(parents=True)
    (patient_dir / "extraction_canonical.json").write_text(
        json.dumps({"weight_kg": {"value": 70.0, "source_date": "2024-01-02"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(pet_suv, "CLINICAL_ROOT", clinical_root)

    value, source_date, source_key = pet_suv._clinical_measurement(
        "P1", dt.date(2024, 1, 1), key_predicate=lambda k, p: k == "weight_kg", window_days=30
    )

    assert value == 70.0
    assert source_date == "2024-01-02"
