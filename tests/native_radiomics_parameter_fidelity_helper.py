"""Pytest-free actual-PyRadiomics probe for native and isolated CT extractors.

Executed inside the NumPy 1.x ``rtpipeline-radiomics`` helper environment as
``python -c <this source> <spec.json>``. It never launches the pipeline, reads
clinical data, or touches the network. Inputs are tiny generated NRRD files
described by the spec; the JSON report is written to ``spec["out"]`` only.
"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace


def _state(extractor):
    return {
        "settings": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                     for k, v in sorted(dict(extractor.settings).items())
                     if k in ("resampledPixelSpacing", "binWidth", "interpolator", "label")},
        "image_types": sorted(dict(extractor.enabledImagetypes)),
        "features": {k: sorted(v) for k, v in sorted(dict(extractor.enabledFeatures).items())},
    }


def _features(row):
    out = {}
    for key, value in dict(row).items():
        if key.startswith(("original_", "square_")):
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                out[key] = None
    return out


def main(spec_path):
    spec = json.loads(open(spec_path, encoding="utf-8").read())
    import numpy
    import radiomics
    from rtpipeline import radiomics as native
    from rtpipeline import radiomics_parallel as parallel
    from rtpipeline.radiomics_ct_contract import (
        CT_EXTRACTION_ARMS, PRIMARY_ARM, RoiClassDecision, configured_parameter_hash,
        effective_parameter_hashes_for_arms,
    )

    config = SimpleNamespace(
        radiomics_params_file=spec["params"], logs_root=None,
        radiomics_min_voxels=None, radiomics_max_voxels=None,
    )
    decision = RoiClassDecision(
        roi_class="synthetic", map_version="fixture", map_hash="0" * 64,
        map_entry_source="fixture", adjudication_status="fixture",
        primary_resegment_range_hu=tuple(spec["window"]),
        primary_intensity_texture_disposition="extract",
        feature_publication_policy="extract",
    )
    report = {"versions": {"numpy": numpy.__version__, "pyradiomics": radiomics.__version__,
                           "python": sys.version.split()[0]}, "native": {}, "isolated": {}}

    # Native wrapper: the compatibility helper against the configured extractor.
    for label, factory in (("ordinary", lambda: native._extractor(config, "CT")),
                           ("large", lambda: native._extractor_large_roi(config, "CT"))):
        extractor = factory()
        if extractor is None:
            raise RuntimeError(f"{label} native extractor unavailable in helper environment")
        result = {k: (float(v) if hasattr(v, "__float__") else v)
                  for k, v in extractor.execute(spec["image"], spec["mask"]).items()}
        report["native"][label] = {
            "state": _state(factory()),
            "features": _features(result),
            "interpolated_spacing": list(result.get("diagnostics_Image-interpolated_Spacing", [])),
            "effective_hashes": effective_parameter_hashes_for_arms(factory, decision),
        }

    # Isolated robustness factory: the closure inside _isolated_radiomics_extraction.
    for roi_name, large_roi in (("small", False), ("small", True), ("BODY", False), ("BODY", True)):
        task_params = {
            "image_path": spec["image"], "mask_path": spec["mask"],
            "segmentation_source": "Synthetic", "roi_name": roi_name,
            "patient_id": "P0", "course_id": "C0", "series_uid": "1.2.3",
            "mask_identity": "sha256:0", "roi_original_name": roi_name,
            "stable_roi_identifier": "fixture", "large_roi": large_roi,
            "params_file": spec["params"], "dual_arm_ct": True,
            "roi_class_decision": decision.__dict__,
            "run_identifier": "run-fixture", "code_revision": "code-fixture",
            "native_voxel_count": int(spec["voxel_count"]),
            "configured_parameter_hashes": {
                arm: configured_parameter_hash(
                    spec["params"], arm=arm,
                    window=(decision.primary_resegment_range_hu if arm == PRIMARY_ARM else None),
                    large_roi=large_roi)
                for arm in CT_EXTRACTION_ARMS},
            "measurement_type": "robustness", "perturbed_mask_identity": "sha256:1",
            "extra_metadata": {"perturbation_id": "baseline"},
        }
        outcome = parallel._isolated_radiomics_extraction((spec["mask"], task_params))
        key = f"{roi_name}|large_roi={large_roi}"
        if "__records__" not in outcome:
            report["isolated"][key] = {"nonmeasurement": True,
                                       "keys": sorted(outcome)}
            continue
        report["isolated"][key] = {"nonmeasurement": False, "arms": {
            str(row["extraction_arm"]): {
                "extraction_status": row.get("extraction_status"),
                "intensity_texture_disposition": row.get("intensity_texture_disposition"),
                "effective_parameter_hash": row.get("effective_parameter_hash"),
                "configured_parameter_hash": row.get("configured_parameter_hash"),
                "features": _features(row),
            } for row in outcome["__records__"]}}
    with open(spec["out"], "w", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, indent=1)
    print("native_radiomics_parameter_fidelity_helper wrote", spec["out"])


if __name__ == "__main__":
    main(sys.argv[1])
