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

_custom_structures_cfg = config.get("custom_structures")
if _custom_structures_cfg:
    cs_path = Path(_custom_structures_cfg)
    if not cs_path.is_absolute():
        cs_path = ROOT_DIR / cs_path
    CUSTOM_STRUCTURES_CONFIG = str(cs_path.resolve())
else:
    default_pelvic = ROOT_DIR / "custom_structures_pelvic.yaml"
    CUSTOM_STRUCTURES_CONFIG = str(default_pelvic.resolve()) if default_pelvic.exists() else ""

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
    conda:
        "envs/rtpipeline.yaml"
    threads:
        max(1, WORKERS)
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
    conda:
        "envs/rtpipeline.yaml"
    threads:
        max(1, WORKERS)
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

        # DVH aggregation
        dvh_frames = []
        for pid, cid, cdir in courses:
            path = cdir / "dvh_metrics.xlsx"
            if not path.exists():
                continue
            try:
                df = pd.read_excel(path)
            except Exception:
                continue
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
            dvh_frames.append(df)
        if dvh_frames:
            pd.concat(dvh_frames, ignore_index=True).to_excel(output.dvh, index=False)
        else:
            pd.DataFrame(columns=["patient_id", "course_id", "ROI_Name", "structure_cropped"]).to_excel(output.dvh, index=False)

        # Radiomics aggregation
        rad_frames = []
        for pid, cid, cdir in courses:
            path = cdir / "radiomics_ct.xlsx"
            if not path.exists():
                continue
            try:
                df = pd.read_excel(path)
            except Exception:
                continue
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
            rad_frames.append(df)
        if rad_frames:
            pd.concat(rad_frames, ignore_index=True).to_excel(output.radiomics, index=False)
        else:
            pd.DataFrame(columns=["patient_id", "course_id", "roi_name", "structure_cropped"]).to_excel(output.radiomics, index=False)

        # Fractions aggregation
        frac_frames = []
        for pid, cid, cdir in courses:
            path = cdir / "fractions.xlsx"
            if not path.exists():
                continue
            try:
                df = pd.read_excel(path)
            except Exception:
                continue
            if "patient_id" in df.columns:
                df["patient_id"] = df["patient_id"].fillna(pid)
            else:
                df.insert(0, "patient_id", pid)
            if "course_id" in df.columns:
                df["course_id"] = df["course_id"].fillna(cid)
            else:
                df.insert(1, "course_id", cid)
            frac_frames.append(df)
        if frac_frames:
            pd.concat(frac_frames, ignore_index=True).to_excel(output.fractions, index=False)
        else:
            pd.DataFrame(columns=["patient_id", "course_id", "treatment_date", "source_path"]).to_excel(output.fractions, index=False)

        # Metadata aggregation
        meta_frames = []
        for pid, cid, cdir in courses:
            path = cdir / "metadata" / "case_metadata.xlsx"
            if not path.exists():
                continue
            try:
                df = pd.read_excel(path)
            except Exception:
                continue
            if "patient_id" in df.columns:
                df["patient_id"] = df["patient_id"].fillna(pid)
            else:
                df.insert(0, "patient_id", pid)
            if "course_id" in df.columns:
                df["course_id"] = df["course_id"].fillna(cid)
            else:
                df.insert(1, "course_id", cid)
            meta_frames.append(df)
        if meta_frames:
            pd.concat(meta_frames, ignore_index=True).to_excel(output.metadata, index=False)
        else:
            pd.DataFrame(columns=["patient_id", "course_id"]).to_excel(output.metadata, index=False)

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
