"""Run production RTpipeline and isolated dicompyler on identical synthetic DICOM.

RTpipeline outputs are read from its published parquet/curve files. No function
is replaced, and no pipeline source is changed. The isolated worker imports no
RTpipeline module. Its native API and harmonized interpolation are both retained.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import traceback
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
from analytic import Geometry, METRICS, REFERENCE_GY, metrics as truth


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def versions():
    packages = sorted((p.metadata["Name"], p.version) for p in metadata.distributions() if p.metadata["Name"])
    return dict(packages)


def engine_hashes():
    import dicompylercore
    package = Path(dicompylercore.__file__).resolve().parent
    return {name: sha(package / name) for name in
            ("dicomparser.py", "dvh.py", "dvhcalc.py", "config.py", "util.py")}


def public_value(value):
    """Represent retained paths relative to the checkout and omit host identity."""
    if isinstance(value, str):
        return value.replace(str(REPO) + "/", "").replace(
            str(Path(sys.executable).resolve().parent.parent), "${PYTHON_ENV}")
    if isinstance(value, dict):
        return {public_value(k): public_value(v) for k, v in value.items()
                if k not in {"host", "hostname"}}
    if isinstance(value, (list, tuple)):
        return [public_value(v) for v in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(public_value(value), indent=2, allow_nan=False, default=str) + "\n")


def isolated_worker(course, output):
    # This branch deliberately does not import generate_phantoms or RTpipeline.
    import numpy as np
    from dicompylercore import dvhcalc
    manifest = json.loads((course / "phantom.json").read_text())
    result = {"versions": versions(), "python": sys.version,
              "dicompyler_source_sha256": engine_hashes(), "rois": []}
    for item in manifest["rois"]:
        h = dvhcalc.get_dvh(str(course / "DICOM/RTSTRUCT/rs.dcm"),
                            str(course / "DICOM/RTDOSE/dose.dcm"), item["roi"])
        v = float(h.volume)
        native = dict(Dmean_Gy=float(h.mean), Dmin_Gy=float(h.min), Dmax_Gy=float(h.max), Volume_cc=v,
                      V95Rx_pct=100 * float(h.volume_constraint(.95 * REFERENCE_GY, "Gy").value) / v,
                      V20Gy_cc=float(h.volume_constraint(20., "Gy").value))
        statuses = {}
        for p in (98, 95, 50, 2):
            native[f"D{p}_Gy"] = float(h.dose_constraint(p).value)
        for cc in (2., .03):
            key = f"D{cc:g}cc_Gy"
            native[key] = float(h.dose_constraint(cc, "cc").value)
            if v < cc:
                statuses[key] = "native_zero_for_volume_below_request"
        # Independently written extraction with the same bins, linear crossings,
        # bounds and no extrapolation used by the production metric definitions.
        x, y = np.asarray(h.bincenters), np.asarray(h.counts)
        def quantile(fraction):
            j = np.flatnonzero(y <= fraction * y[0])
            if not len(j):
                answer = x[-1]
            elif j[0] == 0:
                answer = x[0]
            else:
                k = int(j[0])
                answer = np.interp(fraction * y[0], [y[k], y[k - 1]], [x[k], x[k - 1]])
            return float(np.clip(answer, h.min, h.max))
        harmonized = {k: native[k] for k in ("Dmean_Gy", "Dmin_Gy", "Dmax_Gy", "Volume_cc")}
        for p in (98, 95, 50, 2):
            harmonized[f"D{p}_Gy"] = quantile(p / 100)
        for cc in (2., .03):
            harmonized[f"D{cc:g}cc_Gy"] = quantile(cc / v) if v >= cc else None
        harmonized["V95Rx_pct"] = float(np.interp(.95 * REFERENCE_GY, x, y, left=v, right=0)) / v * 100
        harmonized["V20Gy_cc"] = float(np.interp(20., x, y, left=v, right=0))
        result["rois"].append(dict(roi=item["roi"], dicompyler_native=native,
                                    dicompyler_harmonized=harmonized, native_status=statuses))
    write_json(output, result)


def isolated_server():
    """Reuse imports only; every request loads its own serialized DICOM."""
    for line in sys.stdin:
        request = json.loads(line)
        try:
            isolated_worker(Path(request["course"]), Path(request["output"]))
        except Exception as e:
            print(json.dumps(dict(status="failed", error=str(e), traceback=traceback.format_exc())), flush=True)
        else:
            print(json.dumps(dict(status="completed")), flush=True)


PIPELINE_KEYS = dict(Dmean_Gy="DmeanGy", Dmin_Gy="DminGy", Dmax_Gy="DmaxGy",
                     D98_Gy="D98Gy", D95_Gy="D95Gy", D50_Gy="D50Gy", D2_Gy="D2Gy",
                     D2cc_Gy="D2ccGy", **{"D0.03cc_Gy": "D0.03ccGy"},
                     V95Rx_pct="V95%Rx (%)", V20Gy_cc="V20Gy (cm³)", Volume_cc="Volume (cm³)")


def numeric(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def add_rows(rows, spec, item, tool, values, statuses=None):
    g = Geometry(**item["geometry"])
    expected = truth(g, spec["field"])
    for metric, value in values.items():
        ref, value = expected[metric], numeric(value)
        err = abs(value - ref) if value is not None and ref is not None else None
        status = (statuses or {}).get(metric, "available" if value is not None else "unavailable")
        if ref is None:
            status = "analytic_volume_below_request"
        rows.append(dict(case=spec["group"] + "_" + g.shape, group=spec["group"], tool=tool,
                         metric=metric, value=value, analytic=ref, abs_err=err,
                         rel_err=err / abs(ref) if err is not None and ref != 0 else None,
                         signed_err=value - ref if value is not None and ref is not None else None,
                         unit="%" if metric == "V95Rx_pct" else "cm3" if metric.endswith("cc") else "Gy",
                         abs_err_unit="percentage_points" if metric == "V95Rx_pct" else "cm3" if metric.endswith("cc") else "Gy",
                         status=status, rel_err_status="undefined_zero_truth" if ref == 0 else "available" if err is not None else "unavailable",
                         shape=g.shape, volume_cc=g.volume_cc, phase=spec["phase"],
                         dose_mm=spec["dose_mm"], ct_mm=spec["ct_mm"], field=spec["field"],
                         center_x_mm=g.center[0], center_y_mm=g.center[1], center_z_mm=g.center[2]))


def source_hashes():
    paths = sorted((REPO / "rtpipeline").rglob("*.py"))
    paths += [REPO / "tests" / p for p in ("dvh_plan_target_fixtures.py", "course_contract_test_utils.py")]
    return {str(p.relative_to(REPO)): sha(p) for p in paths}


def summary_results(out, rows, geometry_rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics_long.csv", index=False, float_format="%.12g")
    pd.DataFrame(geometry_rows).to_csv(out / "geometry.csv", index=False, float_format="%.12g")
    records = []
    def median(series):
        defined = series.dropna()
        return defined.median() if len(defined) else float("nan")
    strata = [("overall", ["tool", "metric"]), ("field", ["tool", "metric", "field"]),
              ("shape_size_resolution", ["tool", "metric", "shape", "volume_cc", "dose_mm", "ct_mm"]),
              ("phase_resolution", ["tool", "metric", "phase", "dose_mm", "ct_mm"])]
    for label, columns in strata:
        for key, group in df.groupby(columns, sort=True):
            valid = group.dropna(subset=["abs_err"])
            record = dict(stratum=label, **dict(zip(columns, key)), n_total=len(group),
                          n_numeric=len(valid), n_missing=len(group) - len(valid),
                          n_rel_defined=int(valid.rel_err.notna().sum()),
                          median_abs_err=median(valid.abs_err), max_abs_err=valid.abs_err.max(),
                          median_rel_err=median(valid.rel_err), max_rel_err=valid.rel_err.max())
            record["worst_case"] = valid.loc[valid.abs_err.idxmax(), "case"] if len(valid) else ""
            records.append(record)
    summary = pd.DataFrame(records)
    summary.to_csv(out / "summary.csv", index=False, float_format="%.12g")
    worst = df.dropna(subset=["abs_err"]).sort_values("abs_err", ascending=False).groupby(["tool", "metric"]).head(5)
    worst.to_csv(out / "worst_cases.csv", index=False, float_format="%.12g")
    df[df.value.isna() | (df.status != "available")].to_csv(out / "metric_status.csv", index=False, float_format="%.12g")
    paired = df[df.tool.isin(["rtpipeline", "dicompyler_harmonized"])].pivot(index=["case", "metric"], columns="tool", values="value").reset_index()
    paired["difference_Gy_cc_or_pp"] = paired.rtpipeline - paired.dicompyler_harmonized
    paired.to_csv(out / "paired_difference.csv", index=False, float_format="%.12g")
    primary = df[df.tool == "rtpipeline"]
    volume_errors = primary[primary.metric == "Volume_cc"]
    d95_errors = primary[primary.metric == "D95_Gy"]
    worst_volume = volume_errors.loc[volume_errors.rel_err.idxmax()]
    finding = (f"RTpipeline's median absolute volume error was {volume_errors.abs_err.median():.6g} cm³ "
               f"(median relative error {100*volume_errors.rel_err.median():.6g}%). The largest relative volume error "
               f"was {100*worst_volume.rel_err:.6g}% in `{worst_volume.case}` "
               f"({worst_volume.value:.6g} versus {worst_volume.analytic:.6g} cm³). "
               f"Median absolute D95 error was {d95_errors.abs_err.median():.6g} Gy; "
               f"the maximum was {d95_errors.abs_err.max():.6g} Gy.")
    lines = ["# Synthetic DVH accuracy benchmark", "",
             f"{df.case.nunique()} cases; full production `rtpipeline.dvh.dvh_for_course`, one worker. "
             f"Dose grids: {', '.join(f'{x:g}' for x in sorted(df.dose_mm.unique()))} mm. "
             f"CT slices: {', '.join(f'{x:g}' for x in sorted(df.ct_mm.unique()))} mm; CT in-plane spacing: 2 mm. "
             "Shapes: sphere, cylinder (height = diameter), cube, and spherical shell (inner radius = 0.9 × outer radius). "
             "Volumes: 2, 50, 1,000 cm³. Centres: (0, 0, 0) and (0.37, −0.61, 0.43) mm.", "",
             "Uniform dose: 20 Gy; axial gradient: 20 + 0.1z Gy; radial gradient: 30 − 0.15r Gy. "
             "Radial dose is centred on the sphere/shell. Reference dose is 25 Gy. "
             "V95Rx is reported as percentage volume; its absolute error is in percentage points. "
             "V20Gy and volume are in cm³. Relative error is |value − truth|/|truth|, left undefined for zero truth.", "",
             "RTpipeline uses dicompyler-core internally. The isolated comparator uses unmodified dicompyler-core "
             "with pydicom 2.4.4. It provides a workflow comparison with shared rasterization, not an independent DVH engine. "
             "`dicompyler_native` uses its public metric API; `dicompyler_harmonized` uses the same raw DVH with independently "
             "written linear metric interpolation and bounds matching RTpipeline. Native API values, including zero outputs, are retained.", "",
             finding, "",
             "## Absolute and relative errors", "",
             "Each row pools all fields, sizes, phases and resolutions for that metric. "
             "N counts numeric comparisons; missing endpoints are excluded from errors and counted explicitly. "
             "Medians are descriptive across this balanced test design, not population estimates.", "",
             "| Tool | Metric | N/total | Median absolute | Maximum absolute | Median relative (%) | Maximum relative (%) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    def fmt(x, scale=1.):
        return "—" if pd.isna(x) else f"{x * scale:.6g}"
    for _, r in summary[summary.stratum == "overall"].iterrows():
        lines.append(f"| {r.tool} | {r.metric} | {r.n_numeric}/{r.n_total} | {fmt(r.median_abs_err)} | {fmt(r.max_abs_err)} | {fmt(r.median_rel_err, 100)} | {fmt(r.max_rel_err, 100)} |")
    lines += ["", "## Contour rasterization volume error", "",
              "These errors include finite contour spacing, polygon boundaries, rasterization on dose nodes, "
              "and each workflow's handling of contour planes. CT slice thickness changes the contour spacing; "
              "these are RTSTRUCT-based measurements, not CT-mask resampling. `geometry.csv` also reports the pre-rasterization "
              "polygon/slab volume, production dose-plane alignment and optional snapping-helper diagnostics. "
              "Columns prefixed `helper_snap_` describe the helper only; production retains the original contours.", "",
              "| Tool | Dose grid (mm) | CT slice (mm) | Median absolute volume error (cm³) | Maximum absolute (cm³) | Median relative (%) | Maximum relative (%) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    volumes = df[df.metric == "Volume_cc"]
    for (tool, dose, ct), sub in volumes.groupby(["tool", "dose_mm", "ct_mm"]):
        lines.append(f"| {tool} | {dose:g} | {ct:g} | {fmt(sub.abs_err.median())} | {fmt(sub.abs_err.max())} | {fmt(sub.rel_err.median(),100)} | {fmt(sub.rel_err.max(),100)} |")
    lines += ["", "## Largest absolute error per metric and workflow", "",
              "| Tool | Metric | Case | Value | Analytic | Absolute error |", "|---|---|---|---:|---:|---:|"]
    for (tool, metric), sub in df.dropna(subset=["abs_err"]).groupby(["tool", "metric"]):
        r = sub.loc[sub.abs_err.idxmax()]
        lines.append(f"| {tool} | {metric} | {r.case} | {fmt(r.value)} | {fmt(r.analytic)} | {fmt(r.abs_err)} |")
    lines += ["", "## Endpoint availability and interpretation", "",
              "The production axial path exports D2cc using linear cumulative-DVH interpolation, bounded by Dmin and Dmax. "
              "`rtpipeline_curve_supplement` reports only D2cc, extracted from the production curve with the production "
              "small-volume helper as a consistency check against the native endpoint. "
              "D2cc is undefined for a measured ROI volume below 2 cm³, with native status `roi_below_2cc`. The isolated native API returns zero "
              "in that situation; these outputs have an explicit status in the long table. "
              "Continuous truth includes D2cc at exactly 2 cm³, where it equals Dmin.", "",
              "Circular contours have 256 vertices (area deficit <0.0101% before slice discretization). "
              "Dose storage quantization is 0.0001 Gy; DVHs use the tools' default 0.01 Gy bins. "
              "Dmin and Dmax compare continuous extrema with histogram extrema and therefore include sampling and binning effects. "
              "The continuous formulas do not use the generated DICOM or either tool's voxels.", "",
              "This benchmark covers synthetic axial geometry and smooth dose fields. It does not assess oblique or rotated dose grids, "
              "patient anatomy, registration, heterogeneity corrections, delivered dose, or clinical or manuscript readiness. "
              "No acceptance threshold was prespecified, and no pass/fail accuracy claim is made.", "",
              "See `summary.csv` for errors by shape, volume, dose resolution and CT thickness; "
              "`worst_cases.csv` for five largest errors per metric; `paired_difference.csv` for production versus harmonized comparator; "
              "`run_receipt.json` for versions, source hashes, execution status and exact commands."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    return df


def run(args):
    os.environ["RTPIPELINE_CONTRACT_HEADER_THREADS"] = "4"
    os.environ.setdefault("MPLCONFIGDIR", str(args.results.resolve() / "tmp/matplotlib"))
    import numpy as np
    import pandas as pd
    import pydicom
    import SimpleITK as sitk
    from generate_phantoms import SEED, designs, generate
    from rtpipeline.dvh import dvh_for_course, _small_volume_dose, DVH_METRIC_VERSION
    from rtpipeline.utils import snap_rtstruct_to_dose_grid
    from dicompylercore import dicomparser
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    out = args.results.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("Results must be inside the authorized benchmark directory")
    out.mkdir(parents=True, exist_ok=True)
    temp_root = out / "tmp"
    temp_root.mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(temp_root)
    tempfile.tempdir = str(temp_root)
    child_env = dict(os.environ, PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1")
    receipt = dict(start_utc=utc(), seed=SEED, platform=platform.platform(),
                   python=sys.version, executable="RTPIPELINE_PYTHON", command=public_value(sys.argv),
                   reproduction_command='nice -n 10 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. TMPDIR=' + shlex.quote(public_value(str(temp_root))) + ' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 RTPIPELINE_CONTRACT_HEADER_THREADS=4 "${RTPIPELINE_PYTHON:-python}" ' + shlex.join(public_value(sys.argv)),
                   pipeline_versions=versions(), pipeline_metric_version=DVH_METRIC_VERSION,
                   pipeline_dicompyler_source_sha256=engine_hashes(),
                   pipeline_source_sha256_before=source_hashes(),
                   benchmark_code_sha256={p.name: sha(p) for p in sorted(ROOT.glob("*.py"))},
                   numerical_threads=1, dvh_workers=1, ct_header_threads=4,
                   requested_nice=10, effective_nice=os.getpriority(os.PRIO_PROCESS, 0),
                   thread_environment={name: os.environ.get(name) for name in
                                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                                        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "RTPIPELINE_CONTRACT_HEADER_THREADS")},
                   scope="synthetic axial DICOM only", groups=[], errors=[], status="running")
    test_xml = out / "analytic-tests.xml"
    if test_xml.exists():
        receipt["analytic_tests"] = dict(path=test_xml.name, sha256=sha(test_xml),
                                          suites=[s.attrib for s in ET.parse(test_xml).iter("testsuite")])
    receipt["requirements_sha256"] = {p.name: sha(p) for p in sorted(ROOT.glob("requirements*.txt"))}
    write_json(out / "run_receipt.json", receipt)
    rows, geometry_rows, inputs = [], [], []
    selected = [s for s in designs() if (s["group"] in args.group if args.group else s["dose_mm"] == 2.5)]
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        raise ValueError("No design groups selected")
    session_root = Path(tempfile.mkdtemp(prefix="run-", dir=temp_root))
    cache = session_root / "ct-cache"
    cached_key = None
    worker_log = (out / "isolated_worker.log").open("w")
    worker = subprocess.Popen([str(args.dicompyler_python), str(Path(__file__).resolve()), "--isolated-server"],
                              env=child_env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=worker_log, text=True, bufsize=1)
    try:
        for index, spec in enumerate(selected, 1):
            started = utc()
            work = Path(tempfile.mkdtemp(prefix="phantom-", dir=session_root))
            try:
                print(f"[{index}/{len(selected)}] {spec['group']} start {started}", flush=True)
                cache_key = (spec["volume_cc"], spec["phase"], spec["ct_mm"])
                if cache_key != cached_key and cache.exists():
                    shutil.rmtree(cache)
                cached_key = cache_key
                course, manifest = generate(work, spec, ct_cache=cache)
                raw = out / "raw" / spec["group"]
                raw.mkdir(parents=True, exist_ok=True)
                write_json(raw / "phantom.json", manifest)
                for path in sorted(course.rglob("*.dcm")):
                    inputs.append(dict(group=spec["group"], path=str(path.relative_to(course)), sha256=sha(path), bytes=path.stat().st_size))
                # Optional helper diagnostics on a separate copy. Production
                # retains the original contours and aligns dose samples instead.
                rs = pydicom.dcmread(course / "DICOM/RTSTRUCT/rs.dcm")
                dose = pydicom.dcmread(course / "DICOM/RTDOSE/dose.dcm")
                snapped = copy.deepcopy(rs)
                snap_rtstruct_to_dose_grid(snapped, dose)
                parser = dicomparser.DicomParser(snapped)
                group_geometry = {}
                for item, old_rc, new_rc in zip(manifest["rois"], rs.ROIContourSequence, snapped.ROIContourSequence):
                    old_z = [float(c.ContourData[2]) for c in old_rc.ContourSequence]
                    new_z = [float(c.ContourData[2]) for c in new_rc.ContourSequence]
                    planes = parser.GetStructureCoordinates(item["roi"])
                    geometry_rows.append(dict(group=spec["group"], case=spec["group"] + "_" + item["geometry"]["shape"],
                                              shape=item["geometry"]["shape"], volume_cc=spec["volume_cc"], ct_mm=spec["ct_mm"],
                                              dose_mm=spec["dose_mm"], phase=spec["phase"], field=spec["field"],
                                              contour_slab_volume_cc=item["contour_slab_volume_cc"],
                                              contour_slab_abs_err_cc=abs(item["contour_slab_volume_cc"]-spec["volume_cc"]),
                                              original_planes=len(set(old_z)), helper_snap_planes=len(set(new_z)),
                                              helper_snap_moved_contours=sum(a != b for a,b in zip(old_z,new_z)),
                                              helper_snap_max_mm=max(abs(a-b) for a,b in zip(old_z,new_z)),
                                              helper_snap_min_thickness_mm=parser.CalculatePlaneThickness(planes)))
                    group_geometry[item["roi"]] = geometry_rows[-1]
                # Isolated workflow reads identical, unmodified serialized DICOM.
                worker.stdin.write(json.dumps(dict(course=str(course), output=str(raw / "dicompyler.json"))) + "\n")
                worker.stdin.flush()
                response = worker.stdout.readline()
                (raw / "dicompyler.log").write_text(response)
                if not response or json.loads(response).get("status") != "completed":
                    raise RuntimeError(f"Isolated dicompyler failed: {response}; see isolated_worker.log")
                comparison = json.loads((raw / "dicompyler.json").read_text())
                receipt["dicompyler_versions"] = comparison["versions"]
                receipt["dicompyler_python"] = comparison["python"]
                receipt["isolated_dicompyler_source_sha256"] = comparison["dicompyler_source_sha256"]
                receipt["dicompyler_files_with_different_hashes"] = [name for name, digest in comparison["dicompyler_source_sha256"].items()
                                                                     if receipt["pipeline_dicompyler_source_sha256"][name] != digest]
                pipeline_output = dvh_for_course(course, parallel_workers=1, use_cropped=False,
                                                 rx_dose_gy=REFERENCE_GY, max_total_dose_gy=100.)
                if pipeline_output is None:
                    qc = (course / "metadata/dvh_qc.json").read_text() if (course / "metadata/dvh_qc.json").exists() else "No QC produced"
                    raise RuntimeError("Production DVH function could not process the synthetic DICOM: " + qc)
                table = pd.read_parquet(course / "dvh_metrics.parquet")
                table.to_csv(raw / "rtpipeline_metrics.csv", index=False, float_format="%.16g")
                for src, dst in [(course / "metadata/dvh_qc.json", "dvh_qc.json"),
                                 (course / "metadata/case_metadata.json", "case_metadata.json"),
                                 (course / "dvh_curves.json", "dvh_curves.json")]:
                    if src.exists():
                        shutil.copyfile(src, raw / dst)
                curves = {p["roi_name"]: p for p in json.loads((course / "dvh_curves.json").read_text())["points"]}
                for item, independent in zip(manifest["rois"], comparison["rois"]):
                    name = item["geometry"]["shape"]
                    matching = table[table.ROI_OriginalName == name]
                    if len(matching) != 1:
                        raise RuntimeError(f"Expected one published row for {name}, got {len(matching)}")
                    r = matching.iloc[0]
                    group_geometry[item["roi"]]["dose_plane_alignment"] = r.get("dose_plane_alignment")
                    values = {k: r.get(v) for k,v in PIPELINE_KEYS.items()}
                    statuses = {metric: r[status] for metric, status in (
                        ("D2cc_Gy", "D2cc_status"), ("D0.03cc_Gy", "D0.03cc_status"))}
                    for metric, requested in (("D2cc_Gy", 2.), ("D0.03cc_Gy", .03)):
                        computable = r['Volume (cm³)'] >= requested
                        expected_status = "available" if computable else f"roi_below_{requested:g}cc"
                        if statuses[metric] != expected_status or (numeric(values[metric]) is not None) != computable:
                            raise RuntimeError(f"Invalid native {metric} value/status for {name}")
                    if any(numeric(values[k]) is None for k in METRICS if k not in {"D2cc_Gy", "D0.03cc_Gy"}):
                        raise RuntimeError(f"Production returned incomplete metrics for {name}: {r.to_dict()}")
                    if r.get("dose_grid_coverage_status") != "fully_covered":
                        raise RuntimeError("Synthetic ROI was not fully covered")
                    add_rows(rows, spec, item, "rtpipeline", values, statuses)
                    for tool in ("dicompyler_native", "dicompyler_harmonized"):
                        add_rows(rows, spec, item, tool, independent[tool], independent["native_status"] if tool == "dicompyler_native" else None)
                    curve = curves[name]
                    bins = np.array([p["x"] for p in curve["data"]])
                    cumulative = np.array([p["y"] for p in curve["data"]]) / 100 * curve["volume_cm3"]
                    d2, status = _small_volume_dose(bins, cumulative, curve["volume_cm3"], r.DminGy, r.DmaxGy, 2.)
                    if d2 is None and curve["volume_cm3"] < 2:
                        status = "measured_roi_below_2cc"
                    add_rows(rows, spec, item, "rtpipeline_curve_supplement", {"D2cc_Gy": d2}, {"D2cc_Gy": status})
                receipt["groups"].append(dict(group=spec["group"], started_utc=started, ended_utc=utc(),
                                               case_count=len(manifest["rois"]), status="completed"))
                print(f"[{index}/{len(selected)}] {spec['group']} completed {utc()}", flush=True)
                pd.DataFrame(rows).to_csv(out / "metrics_long.csv", index=False, float_format="%.12g")
                write_json(out / "run_receipt.json", receipt)
            finally:
                shutil.rmtree(work)
    except (Exception, KeyboardInterrupt) as e:
        receipt["errors"].append(dict(type=type(e).__name__, message=str(e), traceback=traceback.format_exc()))
        receipt["status"] = "failed"
        raise
    else:
        receipt["status"] = "completed"
    finally:
        worker.stdin.close()
        try:
            worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker.terminate()
            worker.wait(timeout=5)
        worker_log.close()
        shutil.rmtree(session_root)
        receipt["ended_utc"] = utc()
        receipt["pipeline_source_sha256_after"] = source_hashes()
        receipt["pipeline_sources_unchanged"] = receipt["pipeline_source_sha256_before"] == receipt["pipeline_source_sha256_after"]
        receipt["completed_case_count"] = len({r["case"] for r in rows})
        receipt["metric_row_count"] = len(rows)
        write_json(out / "input_sha256.json", inputs)
        write_json(out / "run_receipt.json", receipt)
        if rows:
            summary_results(out, rows, geometry_rows)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--isolated-server":
        isolated_server()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--isolated-worker":
        isolated_worker(Path(sys.argv[2]), Path(sys.argv[3]))
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dicompyler-python", type=Path, default=ROOT / ".venv-dicompyler/bin/python")
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--group", action="append", help="Exact group ID; may be repeated")
    p.add_argument("--limit", type=int, help="Smoke test first N groups")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    run(args)


if __name__ == "__main__":
    main()
