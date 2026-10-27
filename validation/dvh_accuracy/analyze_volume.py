"""Separate contour/slab discretization from subsequent workflow volume error.

For unsnapped dicompyler, measured minus polygon/slab volume isolates the
in-plane rasterization residual (apart from DICOM decimal rounding). For
RTpipeline the residual comes from its production path, which interpolates dose onto the original contour planes
(alignment recorded in dose_plane_alignment); contour planes are not moved.
"""
from pathlib import Path
import argparse
import json
import sys
import shlex

import numpy as np
import pandas as pd
from run_benchmark import ROOT, REPO, sha, utc


def analyze(out):
    started = utc()
    receipt_path = out / "run_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["status"] != "completed":
        raise ValueError("Run must be complete before volume analysis")
    metrics = pd.read_csv(out / "metrics_long.csv")
    geometry = pd.read_csv(out / "geometry.csv")
    volumes = metrics[(metrics.metric == "Volume_cc") & metrics.tool.isin(["rtpipeline", "dicompyler_native"])].copy()
    joined = volumes.merge(geometry[["case", "contour_slab_volume_cc", "original_planes", "helper_snap_planes",
                                     "helper_snap_max_mm", "helper_snap_min_thickness_mm", "dose_plane_alignment"]], on="case", validate="many_to_one")
    joined = joined.rename(columns={"value": "measured_cc", "analytic": "continuous_cc"})
    joined["contour_discretization_signed_cc"] = joined.contour_slab_volume_cc - joined.continuous_cc
    joined["post_contour_signed_cc"] = joined.measured_cc - joined.contour_slab_volume_cc
    joined["contour_discretization_abs_rel"] = joined.contour_discretization_signed_cc.abs() / joined.continuous_cc
    joined["post_contour_abs_rel"] = joined.post_contour_signed_cc.abs() / joined.contour_slab_volume_cc
    joined["post_contour_scope"] = np.where(joined.tool == "rtpipeline", "rasterization_original_contours_production", "rasterization_original_contours")
    assert np.allclose(joined.measured_cc - joined.continuous_cc,
                       joined.contour_discretization_signed_cc + joined.post_contour_signed_cc,
                       atol=1e-10, rtol=1e-12)
    joined.to_csv(out / "volume_error_components.csv", index=False, float_format="%.12g")
    summaries = []
    for (tool, dose, ct), group in joined.groupby(["tool", "dose_mm", "ct_mm"]):
        summaries.append(dict(tool=tool, dose_mm=dose, ct_mm=ct, n=len(group),
                              contour_discretization_median_abs_rel=group.contour_discretization_abs_rel.median(),
                              contour_discretization_max_abs_rel=group.contour_discretization_abs_rel.max(),
                              post_contour_median_abs_rel=group.post_contour_abs_rel.median(),
                              post_contour_max_abs_rel=group.post_contour_abs_rel.max(),
                              total_median_abs_rel=group.rel_err.median(), total_max_abs_rel=group.rel_err.max()))
    table = pd.DataFrame(summaries)
    table.to_csv(out / "volume_error_components_summary.csv", index=False, float_format="%.12g")
    lines = ["## Volume error components", "",
             "The polygon/slab reference integrates the finite contours with the original CT slice spacing. "
             "Its difference from continuous volume measures contour discretization. Measured volume minus this "
             "reference measures the original-contour rasterization residual for dicompyler. For RTpipeline it "
             "uses the original contour planes with dose interpolated onto them. Decimal contour rounding is negligible at the shown precision. "
             "The signed components sum to total volume error. Residual relative errors use polygon/slab volume as "
             "the denominator; contour and total errors use continuous volume. Native and harmonized dicompyler "
             "metrics share the same volume, so the comparator volume is shown once.", "",
             "| Workflow | Dose (mm) | CT (mm) | N | Contour error median/max (%) | Subsequent error median/max (%) | Total error median/max (%) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['tool']} | {r['dose_mm']:g} | {r['ct_mm']:g} | {r['n']} | "
                     f"{100*r['contour_discretization_median_abs_rel']:.6g} / {100*r['contour_discretization_max_abs_rel']:.6g} | "
                     f"{100*r['post_contour_median_abs_rel']:.6g} / {100*r['post_contour_max_abs_rel']:.6g} | "
                     f"{100*r['total_median_abs_rel']:.6g} / {100*r['total_max_abs_rel']:.6g} |")
    lines += ["", "Case-level components and geometry diagnostics are in `volume_error_components.csv`."]
    uniform = metrics[(metrics.field == "uniform") & (metrics.metric == "D95_Gy")]
    native_uniform = uniform[uniform.tool == "dicompyler_native"]
    primary_uniform = uniform[uniform.tool == "rtpipeline"]
    harmonized_uniform = uniform[uniform.tool == "dicompyler_harmonized"]
    uniform_mean = metrics[(metrics.field == "uniform") & (metrics.tool == "rtpipeline") & (metrics.metric == "Dmean_Gy")]
    lines += ["", "## Uniform-dose endpoint behavior", "",
              f"For the uniform 20 Gy field, dicompyler's native D95 API returned zero in "
              f"{int((native_uniform.value == 0).sum())}/{len(native_uniform)} cases. "
              f"RTpipeline D95 ranged from {primary_uniform.value.min():.8g} to {primary_uniform.value.max():.8g} Gy; "
              f"matched comparator extraction ranged from {harmonized_uniform.value.min():.8g} to "
              f"{harmonized_uniform.value.max():.8g} Gy. "
              f"RTpipeline Dmean ranged from {uniform_mean.value.min():.8g} to {uniform_mean.value.max():.8g} Gy. "
              "The native constraint API selects the first nearest-volume bin on a flat cumulative DVH. "
              "The 0.01 Gy bins also impose a distinct endpoint convention on the interpolated outputs. "
              "These results distinguish endpoint extraction from the accuracy of the shared histogram engine."]
    report = out / "summary.md"
    text = report.read_text()
    shape_counts = metrics[(metrics.tool == "rtpipeline") & (metrics.metric == "Volume_cc")]["shape"].value_counts()
    text = text.replace("Medians are descriptive across this balanced test design, not population estimates.",
                        "Medians describe the tested cases, not population estimates. Spheres and shells contribute "
                        f"{shape_counts['sphere']} cases each; cylinders and cubes contribute {shape_counts['cube']} each because radial fields apply only to spheres and shells.")
    marker = "\n## Volume error components\n"
    text = text.split(marker)[0]
    report.write_text(text.rstrip() + "\n\n" + "\n".join(lines) + "\n")
    receipt.setdefault("postprocessing", {})["volume_error_components"] = dict(
        started_utc=started, ended_utc=utc(), status="completed", code_sha256=sha(Path(__file__)),
        command='"${RTPIPELINE_PYTHON:-python}" validation/dvh_accuracy/analyze_volume.py --results ' + str(out.relative_to(REPO)),
        package_versions={name: receipt["pipeline_versions"].get(name) for name in ("numpy", "pandas")},
        outputs_sha256={p.name: sha(p) for p in (out / "volume_error_components.csv", out / "volume_error_components_summary.csv")})
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=ROOT / "results")
    a = p.parse_args()
    if not a.results.resolve().is_relative_to(ROOT):
        p.error("Results must be inside the benchmark directory")
    analyze(a.results.resolve())
