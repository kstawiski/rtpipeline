from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

import rtpipeline.cli as cli
import rtpipeline.pet_suv as pet_suv
from rtpipeline.config import PipelineConfig
from rtpipeline.layout import build_course_dirs
from rtpipeline.segmentation import run_dcm2niix


def _config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        dicom_root=tmp_path / "dicom",
        output_root=tmp_path / "out",
        logs_root=tmp_path / "logs",
    )


def _radiopharm_item(
    *,
    label: str = "FDG",
    isotope: str = "^18^Fluorine",
    dose_bq: float = 100_000_000.0,
    half_life_s: float = 3600.0,
    start_datetime: str = "20240101090000",
) -> Dataset:
    item = Dataset()
    item.Radiopharmaceutical = label
    item.RadiopharmaceuticalStartDateTime = start_datetime
    item.RadiopharmaceuticalStartTime = start_datetime[8:]
    item.RadionuclideTotalDose = dose_bq
    item.RadionuclideHalfLife = half_life_s
    code = Dataset()
    code.CodeMeaning = isotope
    item.RadionuclideCodeSequence = Sequence([code])
    return item


def _pet_dataset(
    *,
    pixel_value: int = 100,
    slope: float = 1.0,
    intercept: float = 0.0,
    instance_number: int = 1,
    z: float = 0.0,
    rows: int = 1,
    columns: int = 1,
    pixel_spacing: list[float] | None = None,
    image_orientation_patient: list[float] | None = None,
    image_position_patient: list[float] | None = None,
    series_uid: str | None = None,
    study_uid: str | None = None,
    units: str = "BQML",
    corrected: list[str] | None = None,
    decay_correction: str = "START",
    image_type: list[str] | None = None,
    manufacturer: str = "GE MEDICAL SYSTEMS",
    series_description: str = "VUE Point HD",
    reconstruction_method: str = "VUE Point HD",
    patient_weight: float | None = 70.0,
    radiopharm: Dataset | None = None,
    number_of_frames: int | None = None,
) -> FileDataset:
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "PT"
    ds.Units = units
    ds.CorrectedImage = corrected or ["ATTN", "SCAT", "DECY"]
    ds.DecayCorrection = decay_correction
    ds.ImageType = image_type or ["ORIGINAL", "PRIMARY", "WHOLEBODY", "AC"]
    ds.Manufacturer = manufacturer
    ds.ManufacturerModelName = "Discovery IQ"
    ds.StudyDescription = "PET study"
    ds.SeriesDescription = series_description
    ds.ReconstructionMethod = reconstruction_method
    ds.PatientID = "P1"
    ds.StudyInstanceUID = study_uid or generate_uid()
    ds.SeriesInstanceUID = series_uid or generate_uid()
    ds.FrameOfReferenceUID = generate_uid()
    ds.StudyDate = "20240101"
    ds.SeriesDate = "20240101"
    ds.SeriesTime = "093000"
    ds.AcquisitionTime = "093500"
    ds.FrameReferenceTime = 0.0
    ds.DecayFactor = 1.0
    ds.SeriesNumber = 1
    ds.InstanceNumber = instance_number
    ds.Rows = rows
    ds.Columns = columns
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelSpacing = pixel_spacing or [2.0, 2.0]
    ds.SliceThickness = 2.0
    ds.ImageOrientationPatient = image_orientation_patient or [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = image_position_patient or [0, 0, z]
    ds.RescaleSlope = slope
    ds.RescaleIntercept = intercept
    if number_of_frames is not None:
        ds.NumberOfFrames = number_of_frames
    if patient_weight is not None:
        ds.PatientWeight = patient_weight
    ds.PatientSize = 1.75
    ds.RadiopharmaceuticalInformationSequence = Sequence([radiopharm or _radiopharm_item()])
    ds.PixelData = np.full((rows, columns), pixel_value, dtype=np.uint16).tobytes()
    return ds


def _write(ds: FileDataset, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)
    return path


def test_suv_math_and_manifest_status(tmp_path):
    cfg = _config(tmp_path)
    patient_id = "P1"
    series_uid = generate_uid()
    study_uid = generate_uid()
    series_dir = cfg.output_root / patient_id / "all_series" / "DICOM" / "PT" / "series-1"
    _write(
        _pet_dataset(series_uid=series_uid, study_uid=study_uid, pixel_value=100, slope=2.0, instance_number=1, z=0.0),
        series_dir / "IM_0001.dcm",
    )
    _write(
        _pet_dataset(series_uid=series_uid, study_uid=study_uid, pixel_value=100, slope=3.0, instance_number=2, z=2.0),
        series_dir / "IM_0002.dcm",
    )
    manifest_path = build_course_dirs(cfg.output_root / patient_id / "all_series").metadata / "series_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "patient_id": patient_id,
                "series": [
                    {
                        "patient_id": patient_id,
                        "study_uid": study_uid,
                        "series_uid": series_uid,
                        "image_class": "pt",
                        "status": "materialized",
                        "output_dir": str(series_dir),
                        "manufacturer_model": "Discovery IQ",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = pet_suv.ingest_pet_suv_for_patient(cfg, patient_id)

    decayed = 100_000_000.0 * (2 ** (-(30 * 60) / 3600.0))
    scale = 70_000.0 / decayed
    expected = np.array([[[200.0 * scale, 300.0 * scale]]])
    import nibabel as nib

    suv_path = Path(json.loads(manifest_path.read_text(encoding="utf-8"))["series"][0]["pet_suv"]["suv_nifti"])
    actual = np.asarray(nib.load(str(suv_path)).get_fdata())
    assert summary["computed"] == 1
    assert np.allclose(actual, expected, rtol=0, atol=1e-6)
    assert cfg.output_root / patient_id / "all_series" / "NIFTI" / "SUV" in suv_path.parents

    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = persisted["series"][0]
    assert row["status"] == "suv_computed"
    assert row["pet_suv"]["is_primary_recon"] is True
    assert (manifest_path.parent / "pet_suv_manifest.json").exists()
    assert (manifest_path.parent / "pet_suv_exclusions.json").exists()


def test_suv_nifti_geometry_matches_actual_dcm2niix(tmp_path):
    if shutil.which("dcm2niix") is None:
        pytest.skip("dcm2niix executable unavailable for PET SUV geometry comparison")

    series_uid = generate_uid()
    study_uid = generate_uid()
    series_dir = tmp_path / "dicom"
    pet_storage_uid = "1.2.840.10008.5.1.4.1.1.128"
    paths = []
    for instance_number, z, pixel_value, slope in ((1, 30.0, 10, 2.0), (2, 35.0, 20, 3.0)):
        ds = _pet_dataset(
            series_uid=series_uid,
            study_uid=study_uid,
            pixel_value=pixel_value,
            slope=slope,
            instance_number=instance_number,
            rows=2,
            columns=3,
            pixel_spacing=[3.0, 4.0],
            image_orientation_patient=[1, 0, 0, 0, 1, 0],
            image_position_patient=[10, 20, z],
        )
        ds.SOPClassUID = pet_storage_uid
        ds.file_meta.MediaStorageSOPClassUID = pet_storage_uid
        paths.append(_write(ds, series_dir / f"IM_{instance_number:04d}.dcm"))

    built = pet_suv.build_activity_concentration_volume(paths)
    nifti_path = tmp_path / "suv.nii.gz"
    pet_suv._write_suv_nifti(built.activity_bqml, built.affine, nifti_path)

    import nibabel as nib

    dcm2niix_path = run_dcm2niix(_config(tmp_path), series_dir, tmp_path / "dcm2niix", recursive_depth=0)
    assert dcm2niix_path is not None, "dcm2niix did not produce a NIfTI for the synthetic PET series"
    suv_img = nib.load(str(nifti_path))
    dcm2niix_img = nib.load(str(dcm2niix_path))
    assert suv_img.shape == dcm2niix_img.shape
    assert np.allclose(suv_img.affine, dcm2niix_img.affine, atol=1e-2)


def test_per_instance_rescale_uses_each_slice(tmp_path):
    series_uid = generate_uid()
    study_uid = generate_uid()
    paths = [
        _write(
            _pet_dataset(series_uid=series_uid, study_uid=study_uid, pixel_value=10, slope=2.0, instance_number=1, z=0.0),
            tmp_path / "IM_0001.dcm",
        ),
        _write(
            _pet_dataset(series_uid=series_uid, study_uid=study_uid, pixel_value=10, slope=5.0, instance_number=2, z=2.0),
            tmp_path / "IM_0002.dcm",
        ),
    ]

    built = pet_suv.build_activity_concentration_volume(paths)

    assert built.activity_bqml.shape == (1, 1, 2)
    assert built.activity_bqml[0, 0, 0] == 20.0
    assert built.activity_bqml[0, 0, 1] == 50.0
    assert [item["rescale_slope"] for item in built.slice_scaling] == [2.0, 5.0]


def test_pet_suv_force_recomputes_existing_output(tmp_path):
    cfg = _config(tmp_path)
    patient_id = "P1"
    series_uid = generate_uid()
    study_uid = generate_uid()
    series_dir = cfg.output_root / patient_id / "all_series" / "DICOM" / "PT" / "series-1"
    _write(_pet_dataset(series_uid=series_uid, study_uid=study_uid), series_dir / "IM_0001.dcm")
    manifest_path = build_course_dirs(cfg.output_root / patient_id / "all_series").metadata / "series_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "patient_id": patient_id,
                "series": [
                    {
                        "patient_id": patient_id,
                        "study_uid": study_uid,
                        "series_uid": series_uid,
                        "image_class": "pt",
                        "status": "materialized",
                        "output_dir": str(series_dir),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    first = pet_suv.ingest_pet_suv_for_patient(cfg, patient_id)
    second = pet_suv.ingest_pet_suv_for_patient(cfg, patient_id)
    third = pet_suv.ingest_pet_suv_for_patient(cfg, patient_id, force=True)

    assert first["computed"] == 1
    assert second["skipped"] == 1
    assert third["computed"] == 1


def test_decay_factor_guard_converts_frame_reference_time_ms():
    result = pet_suv.compute_decay_factor_guard(583753.96, 4062.6, 1.10478, tolerance=0.02)
    unconverted_rel = abs(result.unconverted_expected_decay_factor - result.observed_decay_factor) / result.observed_decay_factor

    assert result.passed is True
    assert result.expected_decay_factor == pytest.approx(1.1047, rel=1e-3)
    assert unconverted_rel > 1e20


@pytest.mark.parametrize(
    ("label", "isotope", "expected", "manual"),
    [
        ("Choline", "^11^Carbon", "C11_choline", False),
        ("Fcholine", "^18^Fluorine", "F18_fluorocholine", False),
        ("[18F]Fluorocholine", "^18^Fluorine", "F18_fluorocholine", False),
        ("PSMA", "^68^Gallium", "Ga68_PSMA11", False),
        ("PSMA", "^18^Fluorine", "F18_PSMA_ligand", False),
        ("DOTATOC", "^68^Gallium", "Ga68_DOTATOC", False),
        ("DOTATATE", "^68^Gallium", "Ga68_DOTATATE", False),
    ],
)
def test_compound_isotope_resolution(label, isotope, expected, manual):
    resolved = pet_suv.resolve_compound(
        label,
        isotope,
        4062.6 if "68" in isotope else (1220.0 if "11" in isotope else 6586.0),
        study_description="",
        series_description="DOTATATE fallback",
    )

    assert resolved.compound == expected
    assert resolved.needs_manual_review is manual


def test_noninformative_solution_uses_description_fallback():
    resolved = pet_suv.resolve_compound(
        "Solution",
        "^68^Gallium",
        4062.6,
        study_description="PET study",
        series_description="DOTATATE fallback",
    )

    assert resolved.compound == "Ga68_DOTATATE"
    assert resolved.tracer_source == "description_fallback"
    assert resolved.source_text == "PET study DOTATATE fallback"


def test_compound_label_isotope_conflict_is_flagged_not_dropped():
    resolved = pet_suv.resolve_compound("[68Ga]PSMA", "^18^Fluorine", 6586.0)

    assert resolved.compound == "F18_PSMA_ligand"
    assert "tracer_label_isotope_conflict" in resolved.flags
    assert resolved.needs_manual_review is False


def test_unknown_compound_mapping_has_dedicated_flag():
    resolved = pet_suv.resolve_compound("FDG", "^68^Gallium", 4062.6)

    assert resolved.compound is None
    assert "unknown_compound_mapping" in resolved.flags
    assert "tracer_label_isotope_conflict" not in resolved.flags
    assert resolved.needs_manual_review is True


@pytest.mark.parametrize(
    ("reconstruction_method", "series_description", "expected"),
    [
        ("OSEM3D 2i24s", "", "OSEM-like"),
        ("", "PSF+TOF 2i21s", "PSF"),
        ("2D filteredbackprojection", "", "FBP"),
        ("not in dictionary", "", "Unknown"),
    ],
)
def test_recon_family_dictionary_round1_aliases(reconstruction_method, series_description, expected):
    result = pet_suv.classify_recon_family(reconstruction_method, series_description)

    assert result["family"] == expected
    assert result["needs_manual_review"] is (expected == "Unknown")


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"Units": "CNTS"}, "non_bqml_units"),
        ({"CorrectedImage": ["ATTN", "DECY"]}, "missing_required_corrections"),
        ({"ImageType": ["DERIVED", "SECONDARY"]}, "invalid_image_type"),
        ({"DecayCorrection": "ADMIN"}, "unsupported_decay_correction"),
        ({"Manufacturer": "Velocity", "ImageType": ["ORIGINAL", "PRIMARY"]}, "derived_rt_software_pet_export"),
        ({"ManufacturerModelName": "MIM Software", "ImageType": ["ORIGINAL", "PRIMARY"]}, "derived_rt_software_pet_export"),
        ({"NumberOfFrames": 2}, "unsupported_multiframe_pet"),
    ],
)
def test_validity_gate_exclusion_reasons(updates, reason):
    ds = _pet_dataset()
    for key, value in updates.items():
        setattr(ds, key, value)

    result = pet_suv.validate_pet_series([ds], {"patient_id": "P1"})

    assert result.valid is False
    assert reason in result.reasons
    assert reason in {entry["reason"] for entry in result.ledger}


def test_valid_image_type_with_extra_tokens_is_accepted():
    result = pet_suv.validate_pet_series([_pet_dataset(image_type=["ORIGINAL", "PRIMARY", "WHOLEBODY", "AC"])])

    assert result.valid is True


def test_zextent_gate_uses_physical_extent_not_slice_count():
    entries = [
        {
            "patient_id": "P1",
            "study_uid": "S",
            "series_uid": "419",
            "tracer_compound": "F18_FDG",
            "z_extent_mm": pet_suv.compute_physical_z_extent_mm(419, 2.027),
            "recon_family": "OSEM-like",
            "series_number": 2,
        },
        {
            "patient_id": "P1",
            "study_uid": "S",
            "series_uid": "567",
            "tracer_compound": "F18_FDG",
            "z_extent_mm": pet_suv.compute_physical_z_extent_mm(567, 1.5),
            "recon_family": "OSEM-like",
            "series_number": 1,
        },
    ]

    ledger = pet_suv.select_primary_recon(entries, zextent_fraction=0.90)

    assert not any(entry["reason"] == "partial_axial_extent_for_primary_selection" for entry in ledger)
    assert entries[1]["is_primary_recon"] is True


def test_zextent_ipp_fallback_projects_onto_slice_normal():
    normal = np.array([0.0, np.sqrt(0.5), np.sqrt(0.5)])
    positions = [[0.0, 0.0, 0.0], [0.0, 10.0 * normal[1], 10.0 * normal[2]]]

    extent = pet_suv.compute_physical_z_extent_mm(2, None, positions, normal)

    assert extent == pytest.approx(10.0)


def test_primary_selection_prefers_osem_and_routes_unknown_to_manual_review():
    entries = [
        {"patient_id": "P1", "study_uid": "S", "series_uid": "osem", "tracer_compound": "F18_FDG", "z_extent_mm": 800, "recon_family": "OSEM-like", "series_number": 3},
        {"patient_id": "P1", "study_uid": "S", "series_uid": "psf", "tracer_compound": "F18_FDG", "z_extent_mm": 800, "recon_family": "PSF", "series_number": 1},
        {"patient_id": "P1", "study_uid": "S", "series_uid": "bsrem", "tracer_compound": "F18_FDG", "z_extent_mm": 800, "recon_family": "BSREM", "series_number": 1},
        {"patient_id": "P1", "study_uid": "S", "series_uid": "unknown", "tracer_compound": "F18_FDG", "z_extent_mm": 800, "recon_family": "Unknown", "series_number": 0},
    ]

    ledger = pet_suv.select_primary_recon(entries)

    assert entries[0]["is_primary_recon"] is True
    assert entries[1]["is_primary_recon"] is False
    assert entries[2]["is_primary_recon"] is False
    assert entries[3]["is_primary_recon"] is False
    assert entries[3]["needs_manual_review"] is True
    assert "unknown_reconstruction_needing_manual_review" in {entry["reason"] for entry in ledger}


def test_weight_fallback_uses_clinical_json_within_window(tmp_path, monkeypatch):
    clinical_root = tmp_path / "clinical"
    clinical_patient = clinical_root / "P1"
    clinical_patient.mkdir(parents=True)
    (clinical_patient / "extraction_canonical.json").write_text(
        json.dumps({"visits": [{"weight_kg": {"value": 82.5, "source_date": "2024-01-20"}}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(pet_suv, "CLINICAL_ROOT", clinical_root)
    cfg = _config(tmp_path)
    ds = _pet_dataset(patient_weight=None)

    resolved = pet_suv.resolve_weight_and_height(ds, "P1", "20240101", cfg)
    outside = pet_suv.resolve_weight_and_height(ds, "P1", "20240501", cfg)
    missing = pet_suv.resolve_weight_and_height(ds, "P2", "20240101", cfg)

    assert resolved.weight_kg == 82.5
    assert resolved.weight_source.startswith("clinical:")
    assert outside.weight_kg is None
    assert missing.weight_kg is None


def test_weight_fallback_uses_pesel_from_other_patient_ids(tmp_path, monkeypatch):
    clinical_root = tmp_path / "clinical"
    clinical_patient = clinical_root / "46021708174"
    clinical_patient.mkdir(parents=True)
    (clinical_patient / "extraction_canonical.json").write_text(
        json.dumps({"visits": [{"weight_kg": {"value": 82.5, "source_date": "2024-01-20"}}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(pet_suv, "CLINICAL_ROOT", clinical_root)
    cfg = _config(tmp_path)
    ds = _pet_dataset(patient_weight=None)
    ds.OtherPatientIDs = "legacy-id\\46021708174"

    resolved = pet_suv.resolve_weight_and_height(ds, "107981", "20240101", cfg)

    assert resolved.weight_kg == 82.5
    assert resolved.weight_source.startswith("clinical:")


def test_fdg_short_uptake_time_is_flagged_not_excluded(tmp_path):
    cfg = _config(tmp_path)
    patient_id = "P1"
    series_uid = generate_uid()
    study_uid = generate_uid()
    series_dir = cfg.output_root / patient_id / "all_series" / "DICOM" / "PT" / "series-1"
    _write(
        _pet_dataset(
            series_uid=series_uid,
            study_uid=study_uid,
            radiopharm=_radiopharm_item(start_datetime="20240101091500", half_life_s=6586.0),
        ),
        series_dir / "IM_0001.dcm",
    )
    manifest_path = build_course_dirs(cfg.output_root / patient_id / "all_series").metadata / "series_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "patient_id": patient_id,
                "series": [
                    {
                        "patient_id": patient_id,
                        "study_uid": study_uid,
                        "series_uid": series_uid,
                        "image_class": "pt",
                        "status": "materialized",
                        "output_dir": str(series_dir),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = pet_suv.ingest_pet_suv_for_patient(cfg, patient_id)
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    prov_path = Path(persisted["series"][0]["pet_suv"]["provenance"])
    provenance = json.loads(prov_path.read_text(encoding="utf-8"))

    assert summary["computed"] == 1
    assert summary["excluded"] == 0
    assert "fdg_uptake_time_under_1800s" in provenance["timing_flags"]


def test_exclusion_ledger_and_manifest_status_persisted(tmp_path):
    cfg = _config(tmp_path)
    patient_id = "P1"
    series_uid = generate_uid()
    study_uid = generate_uid()
    series_dir = cfg.output_root / patient_id / "all_series" / "DICOM" / "PT" / "bad-series"
    _write(
        _pet_dataset(series_uid=series_uid, study_uid=study_uid, units="CNTS"),
        series_dir / "IM_0001.dcm",
    )
    manifest_path = build_course_dirs(cfg.output_root / patient_id / "all_series").metadata / "series_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "patient_id": patient_id,
                "series": [
                    {
                        "patient_id": patient_id,
                        "study_uid": study_uid,
                        "series_uid": series_uid,
                        "image_class": "pt",
                        "status": "materialized",
                        "output_dir": str(series_dir),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = pet_suv.ingest_pet_suv_for_patient(cfg, patient_id)

    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    ledger = json.loads((manifest_path.parent / "pet_suv_exclusions.json").read_text(encoding="utf-8"))["ledger"]
    assert summary["excluded"] == 1
    assert persisted["series"][0]["status"] == "suv_excluded"
    assert persisted["series"][0]["pet_suv"]["exclusion_reasons"] == ["non_bqml_units"]
    assert ledger[0]["reason"] == "non_bqml_units"


def test_cli_do_ingest_pet_suv_false_never_calls_pet_suv(tmp_path, monkeypatch):
    dicom_root = tmp_path / "dicom"
    dicom_root.mkdir()
    course_dirs = build_course_dirs(tmp_path / "out" / "P1" / "course")
    course_dirs.ensure()
    course = SimpleNamespace(patient_id="P1", course_id="course", dirs=course_dirs)
    runner_names: list[str] = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "organize_and_merge", lambda cfg: [course])
    monkeypatch.setattr(cli, "_detect_gpu_count", lambda: 0)

    def fake_runner(name, tasks, fn, **kwargs):
        runner_names.append(name)
        assert name != "PET SUV ingestion"
        return [True for _ in tasks]

    monkeypatch.setattr(cli, "run_tasks_with_adaptive_workers", fake_runner)

    assert cli.main(
        [
            "--dicom-root",
            str(dicom_root),
            "--outdir",
            str(tmp_path / "out"),
            "--logs",
            str(tmp_path / "logs"),
            "--stage",
            "segmentation",
            "--no-metadata",
        ]
    ) == 0
    assert runner_names == ["Segmentation"]


def test_cli_pet_suv_force_uses_force_redo_not_segmentation_force(tmp_path, monkeypatch):
    dicom_root = tmp_path / "dicom"
    dicom_root.mkdir()
    (tmp_path / "config.yaml").write_text("pet:\n  do_ingest_pet_suv: true\n", encoding="utf-8")
    course_dirs = build_course_dirs(tmp_path / "out" / "P1" / "course")
    course_dirs.ensure()
    course = SimpleNamespace(patient_id="P1", course_id="course", dirs=course_dirs)
    seen: dict[str, list[bool]] = {}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "organize_and_merge", lambda cfg: [course])
    monkeypatch.setattr(cli, "_detect_gpu_count", lambda: 0)

    def fake_runner(name, tasks, fn, **kwargs):
        if name == "Segmentation":
            seen["segmentation"] = [task.force_segmentation for task in tasks]
        if name == "PET SUV ingestion":
            seen["pet_suv"] = [task.force for task in tasks]
        return [True for _ in tasks]

    monkeypatch.setattr(cli, "run_tasks_with_adaptive_workers", fake_runner)

    assert cli.main(
        [
            "--dicom-root",
            str(dicom_root),
            "--outdir",
            str(tmp_path / "out"),
            "--logs",
            str(tmp_path / "logs"),
            "--stage",
            "segmentation",
            "--no-metadata",
            "--force-redo",
        ]
    ) == 0
    assert seen["segmentation"] == [False]
    assert seen["pet_suv"] == [True]
