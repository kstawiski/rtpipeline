from __future__ import annotations

import os
import sys
from pathlib import Path

configfile: "config.yaml"

ROOT_DIR = Path.cwd()


def _ensure_writable_dir(candidate: Path, fallback_name: str) -> Path:
    fallback = ROOT_DIR / fallback_name
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".write_test"
        probe.write_text("", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return candidate
    except OSError as exc:
        sys.stderr.write(
            f"[rtpipeline] Warning: unable to write to {candidate}: {exc}. "
            f"Using fallback {fallback}\n"
        )
        fallback.mkdir(parents=True, exist_ok=True)
        probe = fallback / ".write_test"
        probe.write_text("", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return fallback


DICOM_ROOT = (ROOT_DIR / config.get("dicom_root", "Example_data")).resolve()
OUTPUT_DIR = _ensure_writable_dir((ROOT_DIR / config.get("output_dir", "Data_Snakemake")).resolve(), "Data_Snakemake_fallback")
LOGS_DIR = _ensure_writable_dir((ROOT_DIR / config.get("logs_dir", "Logs_Snakemake")).resolve(), "Logs_Snakemake_fallback")
RESULTS_DIR = OUTPUT_DIR / "_RESULTS"

WORKERS = int(config.get("workers", os.cpu_count() or 4))

def _coerce_int(value, default=None):
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default

SEG_CONFIG = config.get("segmentation", {}) or {}
SEG_EXTRA_MODELS = SEG_CONFIG.get("extra_models") or []
if isinstance(SEG_EXTRA_MODELS, str):
    SEG_EXTRA_MODELS = [m.strip() for m in SEG_EXTRA_MODELS.replace(",", " ").split() if m.strip()]
else:
    SEG_EXTRA_MODELS = [str(m).strip() for m in SEG_EXTRA_MODELS if str(m).strip()]
SEG_FAST = bool(SEG_CONFIG.get("fast", False))
_seg_subset = SEG_CONFIG.get("roi_subset")
if isinstance(_seg_subset, str):
    SEG_ROI_SUBSET = _seg_subset
elif _seg_subset:
    SEG_ROI_SUBSET = ",".join(str(x) for x in _seg_subset)
else:
    SEG_ROI_SUBSET = None
SEG_MAX_WORKERS = _coerce_int(SEG_CONFIG.get("workers") or SEG_CONFIG.get("max_workers"), 1)
if SEG_MAX_WORKERS is not None and SEG_MAX_WORKERS < 1:
    SEG_MAX_WORKERS = 1

RADIOMICS_CONFIG = config.get("radiomics", {}) or {}
RADIOMICS_SEQUENTIAL = bool(RADIOMICS_CONFIG.get("sequential", False))
_radiomics_params = RADIOMICS_CONFIG.get("params_file")
if _radiomics_params:
    params_path = Path(_radiomics_params)
    if not params_path.is_absolute():
        params_path = ROOT_DIR / params_path
    RADIOMICS_PARAMS = str(params_path.resolve())
else:
    RADIOMICS_PARAMS = ""

_radiomics_skip_cfg = RADIOMICS_CONFIG.get("skip_rois") or []
if isinstance(_radiomics_skip_cfg, str):
    RADIOMICS_SKIP_ROIS = [item.strip() for item in _radiomics_skip_cfg.replace(";", ",").split(",") if item.strip()]
else:
    RADIOMICS_SKIP_ROIS = [str(item).strip() for item in _radiomics_skip_cfg if str(item).strip()]

RADIOMICS_MAX_VOXELS = _coerce_int(RADIOMICS_CONFIG.get("max_voxels"), 15_000_000)
if RADIOMICS_MAX_VOXELS is not None and RADIOMICS_MAX_VOXELS < 1:
    RADIOMICS_MAX_VOXELS = 15_000_000
RADIOMICS_MIN_VOXELS = _coerce_int(RADIOMICS_CONFIG.get("min_voxels"), 120)
if RADIOMICS_MIN_VOXELS is not None and RADIOMICS_MIN_VOXELS < 1:
    RADIOMICS_MIN_VOXELS = 1

_custom_structures_cfg = config.get("custom_structures")
if _custom_structures_cfg:
    cs_path = Path(_custom_structures_cfg)
    if not cs_path.is_absolute():
        cs_path = ROOT_DIR / cs_path
    CUSTOM_STRUCTURES_CONFIG = str(cs_path.resolve())
else:
    default_pelvic = ROOT_DIR / "custom_structures_pelvic.yaml"
    CUSTOM_STRUCTURES_CONFIG = str(default_pelvic.resolve()) if default_pelvic.exists() else ""

AGGREGATION_CONFIG = config.get("aggregation", {}) or {}
_agg_threads_raw = AGGREGATION_CONFIG.get("threads")
AGGREGATION_THREADS = _coerce_int(_agg_threads_raw, None)
if AGGREGATION_THREADS is not None and AGGREGATION_THREADS < 1:
    AGGREGATION_THREADS = 1

STAGE_SENTINELS = {
    "organize": OUTPUT_DIR / ".stage_organize",
    "segmentation": OUTPUT_DIR / ".stage_segmentation",
    "dvh": OUTPUT_DIR / ".stage_dvh",
    "radiomics": OUTPUT_DIR / ".stage_radiomics",
    "qc": OUTPUT_DIR / ".stage_qc",
}

AGG_OUTPUTS = {
    "dvh": RESULTS_DIR / "dvh_metrics.xlsx",
    "radiomics": RESULTS_DIR / "radiomics_ct.xlsx",
    "fractions": RESULTS_DIR / "fractions.xlsx",
    "metadata": RESULTS_DIR / "case_metadata.xlsx",
    "qc": RESULTS_DIR / "qc_reports.xlsx",
}


rule all:
    input:
        *(str(path) for path in AGG_OUTPUTS.values())


rule organize:
    output:
        sentinel=str(STAGE_SENTINELS["organize"])
    log:
        str(LOGS_DIR / "stage_organize.log")
    threads:
        max(1, WORKERS)
    conda:
        "envs/rtpipeline.yaml"
    run:
        import subprocess
        sentinel_path = Path(output.sentinel)
        sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        if sentinel_path.exists():
            sentinel_path.unlink()
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        cmd = [
            "rtpipeline",
            "--dicom-root", str(DICOM_ROOT),
            "--outdir", str(OUTPUT_DIR),
            "--logs", str(LOGS_DIR),
            "--workers", str(max(1, threads)),
            "--stage", "organize",
        ]
        if CUSTOM_STRUCTURES_CONFIG:
            cmd.extend(["--custom-structures", CUSTOM_STRUCTURES_CONFIG])
        with open(log[0], "w") as logf:
            subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT)
        sentinel_path.write_text("ok\n", encoding="utf-8")


rule segmentation:
    input:
        STAGE_SENTINELS["organize"]
    output:
        sentinel=str(STAGE_SENTINELS["segmentation"])
    log:
        str(LOGS_DIR / "stage_segmentation.log")
    threads:
        max(1, WORKERS)
    conda:
        "envs/rtpipeline.yaml"
    run:
        import subprocess
        sentinel_path = Path(output.sentinel)
        sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        if sentinel_path.exists():
            sentinel_path.unlink()
        cmd = [
            "rtpipeline",
            "--dicom-root", str(DICOM_ROOT),
            "--outdir", str(OUTPUT_DIR),
            "--logs", str(LOGS_DIR),
            "--workers", str(max(1, threads)),
            "--stage", "segmentation"
        ]
        if SEG_FAST:
            cmd.append("--totalseg-fast")
        for model in SEG_EXTRA_MODELS:
            cmd.extend(["--extra-seg-models", model])
        if SEG_ROI_SUBSET:
            cmd.extend(["--totalseg-roi-subset", SEG_ROI_SUBSET])
        if SEG_MAX_WORKERS:
            cmd.extend(["--seg-workers", str(SEG_MAX_WORKERS)])
        with open(log[0], "w") as logf:
            subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT)
        sentinel_path.write_text("ok\n", encoding="utf-8")


rule dvh:
    input:
        STAGE_SENTINELS["segmentation"]
    output:
        sentinel=str(STAGE_SENTINELS["dvh"])
    log:
        str(LOGS_DIR / "stage_dvh.log")
    threads:
        max(1, WORKERS)
    conda:
        "envs/rtpipeline.yaml"
    run:
        import subprocess
        sentinel_path = Path(output.sentinel)
        sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        if sentinel_path.exists():
            sentinel_path.unlink()
        cmd = [
            "rtpipeline",
            "--dicom-root", str(DICOM_ROOT),
            "--outdir", str(OUTPUT_DIR),
            "--logs", str(LOGS_DIR),
            "--workers", str(max(1, threads)),
            "--stage", "dvh",
        ]
        if CUSTOM_STRUCTURES_CONFIG:
            cmd.extend(["--custom-structures", CUSTOM_STRUCTURES_CONFIG])
        with open(log[0], "w") as logf:
            subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT)
        sentinel_path.write_text("ok\n", encoding="utf-8")


rule radiomics:
    input:
        STAGE_SENTINELS["segmentation"]
    output:
        sentinel=str(STAGE_SENTINELS["radiomics"])
    log:
        str(LOGS_DIR / "stage_radiomics.log")
    threads:
        max(1, WORKERS)
    conda:
        "envs/rtpipeline-radiomics.yaml"
    run:
        import os
        import subprocess
        sentinel_path = Path(output.sentinel)
        sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        if sentinel_path.exists():
            sentinel_path.unlink()
        if RADIOMICS_SEQUENTIAL:
            env_sequential = True
        else:
            env_sequential = False
        cmd = [
            "rtpipeline",
            "--dicom-root", str(DICOM_ROOT),
            "--outdir", str(OUTPUT_DIR),
            "--logs", str(LOGS_DIR),
            "--workers", str(max(1, threads)),
            "--stage", "radiomics",
        ]
        if env_sequential:
            cmd.append("--sequential-radiomics")
        if RADIOMICS_PARAMS:
            cmd.extend(["--radiomics-params", RADIOMICS_PARAMS])
        if RADIOMICS_MAX_VOXELS:
            cmd.extend(["--radiomics-max-voxels", str(RADIOMICS_MAX_VOXELS)])
        if RADIOMICS_MIN_VOXELS:
            cmd.extend(["--radiomics-min-voxels", str(RADIOMICS_MIN_VOXELS)])
        for roi in RADIOMICS_SKIP_ROIS:
            cmd.extend(["--radiomics-skip-roi", roi])
        if CUSTOM_STRUCTURES_CONFIG:
            cmd.extend(["--custom-structures", CUSTOM_STRUCTURES_CONFIG])
        with open(log[0], "w") as logf:
            subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT)
        sentinel_path.write_text("ok\n", encoding="utf-8")


rule qc:
    input:
        STAGE_SENTINELS["segmentation"]
    output:
        sentinel=str(STAGE_SENTINELS["qc"])
    log:
        str(LOGS_DIR / "stage_qc.log")
    threads:
        max(1, WORKERS)
    conda:
        "envs/rtpipeline.yaml"
    run:
        import subprocess
        sentinel_path = Path(output.sentinel)
        sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        if sentinel_path.exists():
            sentinel_path.unlink()
        cmd = [
            "rtpipeline",
            "--dicom-root", str(DICOM_ROOT),
            "--outdir", str(OUTPUT_DIR),
            "--logs", str(LOGS_DIR),
            "--workers", str(max(1, threads)),
            "--stage", "qc",
        ]
        with open(log[0], "w") as logf:
            subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT)
        sentinel_path.write_text("ok\n", encoding="utf-8")


rule aggregate_results:
    input:
        organize=STAGE_SENTINELS["organize"],
        dvh=STAGE_SENTINELS["dvh"],
        radiomics=STAGE_SENTINELS["radiomics"],
        qc=STAGE_SENTINELS["qc"]
    output:
        dvh=str(AGG_OUTPUTS["dvh"]),
        radiomics=str(AGG_OUTPUTS["radiomics"]),
        fractions=str(AGG_OUTPUTS["fractions"]),
        metadata=str(AGG_OUTPUTS["metadata"]),
        qc=str(AGG_OUTPUTS["qc"])
    conda:
        "envs/rtpipeline.yaml"
    run:
        import json
        import os
        import shutil
        from concurrent.futures import ThreadPoolExecutor
        import pandas as pd  # type: ignore

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        def iter_courses() -> list[tuple[str, str, Path]]:
            courses = []
            for patient_dir in sorted(OUTPUT_DIR.iterdir()):
                if not patient_dir.is_dir():
                    continue
                if patient_dir.name.startswith("_") or patient_dir.name.startswith("."):
                    continue
                if patient_dir.name in {"Data", "Data_Snakemake_fallback", "Logs_Snakemake_fallback", "_RESULTS"}:
                    continue
                for course_dir in sorted(patient_dir.iterdir()):
                    if not course_dir.is_dir():
                        continue
                    if course_dir.name.startswith("_"):
                        continue
                    courses.append((patient_dir.name, course_dir.name, course_dir))
            return courses

        courses = iter_courses()

        def _max_workers(default: int) -> int:
            if not courses:
                return 1
            if AGGREGATION_THREADS is not None:
                return min(len(courses), max(1, AGGREGATION_THREADS))
            return min(len(courses), max(1, default))

        worker_count = _max_workers(os.cpu_count() or 4)

        def _collect_frames(loader):
            frames = []
            if not courses:
                return frames
            with ThreadPoolExecutor(max_workers=max(1, worker_count)) as pool:
                for df in pool.map(loader, courses):
                    if df is not None and not df.empty:
                        frames.append(df)
            return frames

        def _load_dvh(course):
            pid, cid, cdir = course
            path = cdir / "dvh_metrics.xlsx"
            if not path.exists():
                return None
            try:
                df = pd.read_excel(path)
            except Exception:
                return None
            if "patient_id" in df.columns:
                df["patient_id"] = df["patient_id"].fillna(pid)
            else:
                df.insert(0, "patient_id", pid)
            if "course_id" in df.columns:
                df["course_id"] = df["course_id"].fillna(cid)
            else:
                df.insert(1, "course_id", cid)
            if "structure_cropped" not in df.columns:
                df["structure_cropped"] = False
            return df

        # DVH aggregation
        dvh_frames = _collect_frames(_load_dvh)
        if dvh_frames:
            dvh_all = pd.concat(dvh_frames, ignore_index=True)
            if "Segmentation_Source" not in dvh_all.columns:
                dvh_all["Segmentation_Source"] = "Unknown"
            if "ROI_Name" in dvh_all.columns:
                roi_series = dvh_all["ROI_Name"].astype(str)
            else:
                roi_series = pd.Series(["" for _ in range(len(dvh_all))], index=dvh_all.index)
                dvh_all.insert(len(dvh_all.columns), "ROI_Name", roi_series)
            dvh_all["_roi_key"] = roi_series.str.strip().str.lower()
            manual_keys = set(
                dvh_all.loc[
                    dvh_all["Segmentation_Source"].astype(str).str.lower() == "manual",
                    "_roi_key",
                ].dropna()
            )
            drop_mask = (
                dvh_all["Segmentation_Source"].astype(str).str.lower().isin({"custom", "merged"})
                & dvh_all["_roi_key"].isin(manual_keys)
            )
            if drop_mask.any():
                dvh_all = dvh_all.loc[~drop_mask].copy()
            dvh_all.drop(columns=["_roi_key"], errors="ignore", inplace=True)
            dvh_all.to_excel(output.dvh, index=False)
        else:
            pd.DataFrame(columns=["patient_id", "course_id", "ROI_Name", "structure_cropped"]).to_excel(output.dvh, index=False)

        # Radiomics aggregation
        def _load_radiomics(course):
            pid, cid, cdir = course
            path = cdir / "radiomics_ct.xlsx"
            if not path.exists():
                return None
            try:
                df = pd.read_excel(path)
            except Exception:
                return None
            if "patient_id" not in df.columns:
                df.insert(0, "patient_id", pid)
            else:
                df["patient_id"] = df["patient_id"].fillna(pid)
            if "course_id" not in df.columns:
                df.insert(1, "course_id", cid)
            else:
                df["course_id"] = df["course_id"].fillna(cid)
            if "structure_cropped" not in df.columns:
                df["structure_cropped"] = False
            return df

        rad_frames = _collect_frames(_load_radiomics)
        if rad_frames:
            pd.concat(rad_frames, ignore_index=True).to_excel(output.radiomics, index=False)
        else:
            pd.DataFrame(columns=["patient_id", "course_id", "roi_name", "structure_cropped"]).to_excel(output.radiomics, index=False)

        # Fractions aggregation
        def _load_fraction(course):
            pid, cid, cdir = course
            path = cdir / "fractions.xlsx"
            if not path.exists():
                return None
            try:
                df = pd.read_excel(path)
            except Exception:
                return None
            if "patient_id" in df.columns:
                df["patient_id"] = df["patient_id"].fillna(pid)
            else:
                df.insert(0, "patient_id", pid)
            if "course_id" in df.columns:
                df["course_id"] = df["course_id"].fillna(cid)
            else:
                df.insert(1, "course_id", cid)
            return df

        frac_frames = _collect_frames(_load_fraction)
        if frac_frames:
            pd.concat(frac_frames, ignore_index=True).to_excel(output.fractions, index=False)
        else:
            pd.DataFrame(columns=["patient_id", "course_id", "treatment_date", "source_path"]).to_excel(output.fractions, index=False)

        # Metadata aggregation
        def _load_metadata(course):
            pid, cid, cdir = course
            path = cdir / "metadata" / "case_metadata.xlsx"
            if not path.exists():
                return None
            try:
                df = pd.read_excel(path)
            except Exception:
                return None
            if "patient_id" in df.columns:
                df["patient_id"] = df["patient_id"].fillna(pid)
            else:
                df.insert(0, "patient_id", pid)
            if "course_id" in df.columns:
                df["course_id"] = df["course_id"].fillna(cid)
            else:
                df.insert(1, "course_id", cid)
            return df

        meta_frames = _collect_frames(_load_metadata)
        if meta_frames:
            pd.concat(meta_frames, ignore_index=True).to_excel(output.metadata, index=False)
        else:
            pd.DataFrame(columns=["patient_id", "course_id"]).to_excel(output.metadata, index=False)

        # Persist supplemental metadata exports from Data/ if present
        supplemental_sources = {
            "plans.xlsx": OUTPUT_DIR / "Data" / "plans.xlsx",
            "structure_sets.xlsx": OUTPUT_DIR / "Data" / "structure_sets.xlsx",
            "dosimetrics.xlsx": OUTPUT_DIR / "Data" / "dosimetrics.xlsx",
            "fractions.xlsx": OUTPUT_DIR / "Data" / "fractions.xlsx",
            "metadata.xlsx": OUTPUT_DIR / "Data" / "metadata.xlsx",
            "CT_images.xlsx": OUTPUT_DIR / "Data" / "CT_images.xlsx",
        }
        for fname, src_path in supplemental_sources.items():
            if src_path.exists():
                dst_path = RESULTS_DIR / fname
                try:
                    shutil.copy2(src_path, dst_path)
                except Exception as exc:
                    print(f"[aggregate_results] Warning: failed to copy {src_path} -> {dst_path}: {exc}")

        # QC aggregation
        qc_rows = []
        for pid, cid, cdir in courses:
            qc_dir = cdir / "qc_reports"
            if not qc_dir.exists():
                continue
            for report_path in qc_dir.glob("*.json"):
                try:
                    data = json.loads(report_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                qc_rows.append(
                    {
                        "patient_id": pid,
                        "course_id": cid,
                        "report_name": report_path.name,
                        "overall_status": data.get("overall_status"),
                        "structure_cropping": json.dumps(data.get("checks", {}).get("structure_cropping", {})),
                        "checks": json.dumps(data.get("checks", {})),
                    }
                )
        if qc_rows:
            pd.DataFrame(qc_rows).to_excel(output.qc, index=False)
        else:
            pd.DataFrame(columns=["patient_id", "course_id", "report_name", "overall_status"]).to_excel(output.qc, index=False)
