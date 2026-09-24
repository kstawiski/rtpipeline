"""Run the robustness course step for synthetic courses from one package root.

Launched by ``test_robustness_prep_equivalence.py`` and
``benchmark_robustness_prep.py`` once with an export of the base revision and
once with the working tree, on identical inputs at the same absolute paths.
Only seams present in both revisions are replaced: the course contract, the
main-radiomics identity catalog (served from ``inputs.json``), the
standard-source resolver, the custom-model listing and PyRadiomics execution.
RTSTRUCT inspection, rasterization, selection, perturbation, task
preparation, validation and publication run as shipped.

Usage: python -B -s robustness_prep_driver.py <package-root> <output-root> <config> [timing.json]

Environment: ROBUSTNESS_DRIVER_WORKERS overrides each course's worker budget
and ROBUSTNESS_DRIVER_PREP_START its preparation start method.
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import resource
import sys
import time
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
from rtpipeline.radiomics_ct_contract import CT_EXTRACTION_ARMS, PRIMARY_ARM  # noqa: E402

SHAPE = ("original_shape_VoxelVolume", "original_shape_Sphericity")
TEXTURE = ("original_firstorder_Mean", "original_glcm_Contrast", "original_glcm_MCC")
OUTPUT_ROOT = Path(sys.argv[2]).resolve()
CONFIG = Path(sys.argv[3]).resolve()
TIMING = Path(sys.argv[4]).resolve() if len(sys.argv) > 4 else None


def inputs(patient):
    return json.loads((OUTPUT_ROOT / patient / "C1" / "inputs.json").read_text())


def _value(name, pid, arm, roi):
    base = {"original_shape_VoxelVolume": 700.0, "original_shape_Sphericity": 0.8,
            "original_firstorder_Mean": 40.0, "original_glcm_Contrast": 3.0,
            "original_glcm_MCC": 0.5}[name]
    return base * (1.0 + (sum(map(ord, pid + arm + name + roi)) % 17) / 400.0)


def _records(pid, identity, run_identifier, mask_identity):
    course = inputs(identity["patient_id"])
    roi = identity["roi_original_name"]
    records = []
    for arm in CT_EXTRACTION_ARMS:
        record = {
            **identity, "roi_name": roi, "modality": "CT",
            "measurement_type": rr.ROBUSTNESS_MEASUREMENT_TYPE,
            "perturbed_mask_identity": mask_identity, "extraction_arm": arm,
            "roi_class": "target", "roi_class_map_version": "v1", "roi_class_map_hash": "h",
            "effective_parameter_hash": f"e-{arm}", "configured_parameter_hash": f"c-{arm}",
            "run_identifier": run_identifier, "extraction_status": "success",
            "shape_disposition": "success", "intensity_texture_disposition": "success",
            "radiomics_undefined_features_json": "[]", "resegment_after_count": 500,
        }
        record.update({name: _value(name, pid, arm, roi) for name in SHAPE + TEXTURE})
        below = course.get("primary_below_minimum_when")
        if below and arm == PRIMARY_ARM and below in pid:
            for name in TEXTURE:
                record.pop(name)
            record["intensity_texture_disposition"] = "below_minimum_voxels"
        if course.get("identity_mismatch_when") and course["identity_mismatch_when"] in pid:
            record["stable_roi_identifier"] = "not-this-roi"
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
    # The mask file must exist and carry the identity the task claims.
    import SimpleITK as sitk
    mask = sitk.ReadImage(params["mask_path"])
    if rr._perturbed_mask_identity(mask) != params["perturbed_mask_identity"]:
        raise RuntimeError("mask file does not match its task identity")
    if not Path(params["image_path"]).is_file():
        raise RuntimeError("image file missing")
    identity = {c: params[c] for c in rr.ROBUSTNESS_SOURCE_IDENTITY_COLUMNS}
    return {"__records__": _records(pid, identity, params["run_identifier"],
                                    params["perturbed_mask_identity"]),
            "segmentation_source": params["segmentation_source"],
            "roi_name": params["roi_name"], "perturbation_id": pid}


def contract(course_dir):
    spec = inputs(Path(course_dir).parent.name)
    return SimpleNamespace(planning_ct_dir=Path(course_dir) / "CT",
                           planning_ct={"series_instance_uid": spec["series_uid"]},
                           planning_ct_nifti=None)


def catalog(course_dir, **kwargs):
    spec = inputs(Path(course_dir).parent.name)
    entries = {}
    for index, (source, roi) in enumerate(spec["catalog"]):
        entries[(source, roi)] = rr.RobustnessRoiIdentity.from_mapping(dict(
            patient_id=Path(course_dir).parent.name, course_id="C1",
            series_uid=spec["series_uid"], segmentation_source=source,
            mask_identity=f"source-mask-{index}", roi_original_name=roi,
            stable_roi_identifier=f"roi-{index}",
        ))
    return entries, {}


def standard_sources(contract_value, course_dir):
    sources = [("Manual", course_dir / "RS.dcm", None)]
    if (course_dir / "RS_auto.dcm").exists():
        sources.append(("AutoRTS_total", course_dir / "RS_auto.dcm", None))
    return sources


def models(course_dir):
    root = Path(course_dir) / "Segmentation_CustomModels"
    return sorted((p.name, p) for p in root.iterdir()) if root.is_dir() else []


class _Marks(logging.Handler):
    def __init__(self):
        super().__init__()
        self.marks = {}

    def emit(self, record):
        message = record.getMessage()
        for key, prefix in (("collected", "Total identity-matched masks collected"),
                            ("extraction_start", "Processing "),
                            ("selected", "Selected ")):
            if message.startswith(prefix) and key not in self.marks:
                self.marks[key] = time.monotonic()


def main():
    rr.load_course_contract = contract
    rr._load_main_ct_identity_catalog = catalog
    rm._standard_rtstruct_sources = standard_sources
    custom_models.list_custom_model_outputs = models
    radiomics_conda.check_radiomics_env = lambda *a, **k: True
    radiomics_conda.extract_radiomics_batch_with_conda = fake_conda_batch
    rp._isolated_radiomics_extraction_with_retry = fake_worker
    rr.get_context = lambda _: mp.get_context("fork")
    if hasattr(rr, "_prepare_robustness_rois"):
        # Preparation needs none of the seams above, so it can run under the
        # production start method; the course chooses it in inputs.json.
        prepare = rr._prepare_robustness_rois

        def prepare_with_course_context(requests, workers, context):
            method = os.environ.get("ROBUSTNESS_DRIVER_PREP_START") or inputs(
                Path(requests[0]["course_dir"]).parent.name).get("prep_start_method", "spawn")
            return prepare(requests, workers, mp.get_context(method))

        rr._prepare_robustness_rois = prepare_with_course_context
    os.environ.update(RTPIPELINE_RADIOMICS_THREAD_LIMIT="1")
    courses = sorted(p.parent.parent.name for p in OUTPUT_ROOT.glob("P*/C1/inputs.json"))
    codes, timing = {}, {}
    for patient in courses:
        course = OUTPUT_ROOT / patient / "C1"
        spec = inputs(patient)
        os.environ["RTPIPELINE_DISABLE_PARALLEL_RADIOMICS"] = "0" if spec.get("parallel") else "1"
        # ROBUSTNESS_DRIVER_WORKERS lets a caller rerun the same inputs with
        # another worker budget.
        os.environ["RTPIPELINE_MAX_WORKERS"] = str(
            os.environ.get("ROBUSTNESS_DRIVER_WORKERS") or spec.get("workers", 1))
        marks = _Marks()
        logging.getLogger("rtpipeline").addHandler(marks)
        start = time.monotonic()
        codes[patient] = cli.main([
            "radiomics-robustness", "--course-dir", str(course), "--config", str(CONFIG),
            "--output", str(course / "radiomics_robustness_ct.parquet"),
            "--sentinel", str(course / ".radiomics_robustness_done"), "--campaign-mode",
        ])
        end = time.monotonic()
        logging.getLogger("rtpipeline").removeHandler(marks)
        timing[patient] = {
            "total_s": end - start,
            **{f"{k}_s": v - start for k, v in marks.marks.items()},
            "parent_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "children_peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        }
    if TIMING is not None:
        TIMING.write_text(json.dumps(timing, indent=2))
    print(json.dumps(codes))


if __name__ == "__main__":
    main()
