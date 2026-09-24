"""Run the robustness course step and cohort aggregate from one package root.

Launched by ``test_robustness_equivalence.py`` once with an export of the base
revision and once with the working tree, on identical synthetic inputs at the
same absolute paths. Only seams present in both revisions are replaced: the
course contract, the main-radiomics identity catalog, the standard-source
resolver, the custom-model listing and the PyRadiomics execution (conda batch
and isolated worker). Mask reading, perturbation, flattening, validation,
publication, receipts and aggregation run as shipped.

Usage: python -B -s robustness_equivalence_driver.py <package-root> <output-root> <config>
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from types import SimpleNamespace

PACKAGE_ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(PACKAGE_ROOT))

import rtpipeline  # noqa: E402

if Path(rtpipeline.__file__).resolve().parent != PACKAGE_ROOT / "rtpipeline":
    raise SystemExit(f"imported {rtpipeline.__file__}, not the package under {PACKAGE_ROOT}")

from rtpipeline import cli, custom_models  # noqa: E402
from rtpipeline import radiomics as rm  # noqa: E402
from rtpipeline import radiomics_conda  # noqa: E402
from rtpipeline import radiomics_parallel as rp  # noqa: E402
from rtpipeline import radiomics_robustness as rr  # noqa: E402
from rtpipeline.course_manifest import CURRENT_COURSE_MANIFEST_SCHEMA  # noqa: E402
from rtpipeline.radiomics_ct_contract import CT_EXTRACTION_ARMS, PRIMARY_ARM  # noqa: E402

SHAPE = ("original_shape_VoxelVolume", "original_shape_Sphericity")
TEXTURE = ("original_firstorder_Mean", "original_glcm_Contrast", "original_glcm_MCC")


def _value(name, pid, arm, subject):
    base = {"original_shape_VoxelVolume": 700.0, "original_shape_Sphericity": 0.8,
            "original_firstorder_Mean": 40.0, "original_glcm_Contrast": 3.0,
            "original_glcm_MCC": 0.5}[name]
    return base * (1.0 + 0.3 * subject + (sum(map(ord, pid + arm + name)) % 17) / 400.0)


def _records(pid, identity, run_identifier, mask_identity):
    course = inputs(identity["patient_id"])
    subject = int(identity["patient_id"][1:])
    records = []
    for arm in CT_EXTRACTION_ARMS:
        record = {
            **identity, "roi_name": identity["roi_original_name"], "modality": "CT",
            "measurement_type": rr.ROBUSTNESS_MEASUREMENT_TYPE,
            "perturbed_mask_identity": mask_identity, "extraction_arm": arm,
            "roi_class": "target", "roi_class_map_version": "v1", "roi_class_map_hash": "h",
            "effective_parameter_hash": f"e-{arm}", "configured_parameter_hash": f"c-{arm}",
            "run_identifier": run_identifier, "extraction_status": "success",
            "shape_disposition": "success", "intensity_texture_disposition": "success",
            "radiomics_undefined_features_json": "[]", "resegment_after_count": 500,
        }
        record.update({name: _value(name, pid, arm, subject) for name in SHAPE + TEXTURE})
        if course.get("primary_below_minimum_everywhere") and arm == PRIMARY_ARM:
            # Every perturbation equally shape-only: no feature-set difference.
            for name in TEXTURE:
                record.pop(name)
            record["intensity_texture_disposition"] = "below_minimum_voxels"
        if course.get("mcc_undefined_everywhere") and arm != PRIMARY_ARM:
            record["original_glcm_MCC"] = None
            record["radiomics_undefined_features_json"] = '["original_glcm_MCC"]'
        records.append(record)
    return records


def fake_conda_batch(tasks, params_file, timeout_per_roi=120):
    results = []
    for index, task in enumerate(tasks):
        identity = {c: task["metadata"][c] for c in rr.ROBUSTNESS_SOURCE_IDENTITY_COLUMNS}
        results.append({"__task_index__": index, "__status__": "success",
                         "__records__": _records(task["robustness_perturbation_id"], identity,
                                                 task["run_identifier"], "sha256:" + "0" * 64)})
    return results


def fake_worker(task):
    params = task[1]
    pid = params["extra_metadata"]["perturbation_id"]
    identity = {c: params[c] for c in rr.ROBUSTNESS_SOURCE_IDENTITY_COLUMNS}
    return {"__records__": _records(pid, identity, params["run_identifier"],
                                    params["perturbed_mask_identity"]),
            "segmentation_source": params["segmentation_source"],
            "roi_name": params["roi_name"], "perturbation_id": pid}


OUTPUT_ROOT = Path(sys.argv[2]).resolve()
CONFIG = Path(sys.argv[3]).resolve()


def inputs(patient):
    return json.loads((OUTPUT_ROOT / patient / "C1" / "inputs.json").read_text())


def contract(course_dir):
    spec = inputs(Path(course_dir).parent.name)
    return SimpleNamespace(planning_ct_dir=Path(course_dir) / "CT",
                           planning_ct={"series_instance_uid": spec["series_uid"]},
                           planning_ct_nifti=None)


def catalog(course_dir, **kwargs):
    spec = inputs(Path(course_dir).parent.name)
    return ({("Manual", "GTV1"): rr.RobustnessRoiIdentity.from_mapping(dict(
        patient_id=Path(course_dir).parent.name, course_id="C1",
        series_uid=spec["series_uid"], segmentation_source="Manual",
        mask_identity="source-mask", roi_original_name="GTV1",
        stable_roi_identifier="roi-1",
    ))}, {})


def models(course_dir):
    model = Path(course_dir) / "Segmentation_CustomModels" / "ModelA"
    return [("ModelA", model)] if model.is_dir() else []


def main():
    rr.load_course_contract = contract
    rr._load_main_ct_identity_catalog = catalog
    rm._standard_rtstruct_sources = lambda c, course_dir: [("Manual", course_dir / "RS.dcm", None)]
    custom_models.list_custom_model_outputs = models
    radiomics_conda.check_radiomics_env = lambda *a, **k: True
    radiomics_conda.extract_radiomics_batch_with_conda = fake_conda_batch
    rp._isolated_radiomics_extraction_with_retry = fake_worker
    rr.get_context = lambda _: mp.get_context("fork")
    os.environ.update(RTPIPELINE_RADIOMICS_THREAD_LIMIT="1", RTPIPELINE_MAX_WORKERS="1")
    courses = sorted(p.parent.parent.name for p in OUTPUT_ROOT.glob("P*/C1/inputs.json"))
    codes = {}
    for patient in courses:
        course = OUTPUT_ROOT / patient / "C1"
        os.environ["RTPIPELINE_DISABLE_PARALLEL_RADIOMICS"] = (
            "0" if inputs(patient).get("parallel") else "1")
        codes[patient] = cli.main([
            "radiomics-robustness", "--course-dir", str(course), "--config", str(CONFIG),
            "--output", str(course / "radiomics_robustness_ct.parquet"),
            "--sentinel", str(course / ".radiomics_robustness_done"), "--campaign-mode",
        ])
    cohort = [p for p in courses if inputs(p).get("in_cohort")]
    manifest = OUTPUT_ROOT / "_COURSES" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "schema": CURRENT_COURSE_MANIFEST_SCHEMA,
        "intended_course_count": len(cohort), "attempted_course_count": len(cohort),
        "validated_course_count": len(cohort), "technical_quarantine_count": 0,
        "technical_quarantines": [],
        "courses": [{"patient": p, "course": "C1"} for p in cohort],
    }, indent=2))
    codes["cohort_aggregate"] = cli.main([
        "radiomics-robustness-aggregate", "--manifest", str(manifest),
        "--output-root", str(OUTPUT_ROOT),
        "--output", str(OUTPUT_ROOT / "_RESULTS" / "radiomics_robustness_summary.xlsx"),
        "--config", str(CONFIG),
    ])
    rob = rr.RobustnessConfig.from_dict(
        __import__("yaml").safe_load(CONFIG.read_text())["radiomics_robustness"])
    rr.aggregate_robustness_results(
        [OUTPUT_ROOT / p / "C1" / "radiomics_robustness_ct.parquet" for p in cohort],
        OUTPUT_ROOT / "_RESULTS" / "explicit_inputs.xlsx", rob)
    print(json.dumps(codes))


if __name__ == "__main__":
    main()
