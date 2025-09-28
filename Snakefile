from __future__ import annotations

import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

configfile: "config.yaml"

ROOT_DIR = Path.cwd()
DICOM_ROOT = ROOT_DIR / config.get("dicom_root", "Example_data")
OUTPUT_DIR = ROOT_DIR / config.get("output_dir", "Data_Snakemake")
LOGS_DIR = ROOT_DIR / config.get("logs_dir", "Logs_Snakemake")
WORKERS = int(config.get("workers", 4))

SEG_CONFIG = config.get("segmentation", {}) or {}
SEG_FAST = bool(SEG_CONFIG.get("fast", False))
SEG_ROI_SUBSET = SEG_CONFIG.get("roi_subset") or []
if isinstance(SEG_ROI_SUBSET, str):
    SEG_ROI_SUBSET = [s.strip() for s in SEG_ROI_SUBSET.replace(",", " ").split() if s.strip()]
SEG_FLAGS: list[str] = []
if SEG_FAST:
    SEG_FLAGS.append("-f")
if SEG_ROI_SUBSET:
    SEG_FLAGS.extend(["-rs", *SEG_ROI_SUBSET])

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
    CUSTOM_STRUCTURES_CONFIG = ""

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

PATIENTS = sorted([p.name for p in DICOM_ROOT.iterdir() if p.is_dir()])


rule all:
    input:
        expand(str(OUTPUT_DIR / "{patient}" / "metadata.json"), patient=PATIENTS),
        expand(str(OUTPUT_DIR / "{patient}" / "CT_DICOM"), patient=PATIENTS),
        expand(str(OUTPUT_DIR / "{patient}" / "RS_auto.dcm"), patient=PATIENTS),
        expand(str(OUTPUT_DIR / "{patient}" / "RS_custom.dcm"), patient=PATIENTS),
        expand(str(OUTPUT_DIR / "{patient}" / "Radiomics_CT.xlsx"), patient=PATIENTS),
        expand(str(OUTPUT_DIR / "{patient}" / "dvh_metrics.xlsx"), patient=PATIENTS),
        expand(str(OUTPUT_DIR / "{patient}" / "Axial.html"), patient=PATIENTS),
        str(OUTPUT_DIR / "metadata_summary.xlsx"),
        str(OUTPUT_DIR / "radiomics_summary.xlsx"),
        str(OUTPUT_DIR / "dvh_summary.xlsx")


rule organize_data:
    input:
        dicom_dir=lambda wildcards: str(DICOM_ROOT / wildcards.patient)
    output:
        ct_dir=directory(str(OUTPUT_DIR / "{patient}" / "CT_DICOM")),
        metadata=str(OUTPUT_DIR / "{patient}" / "metadata.json"),
        rp=str(OUTPUT_DIR / "{patient}" / "RP.dcm"),
        rd=str(OUTPUT_DIR / "{patient}" / "RD.dcm")
    conda:
        "envs/rtpipeline.yaml"
    log:
        str(LOGS_DIR / "organize_{patient}.log")
    run:
        import json
        import shutil
        import subprocess
        import tempfile

        dicom_dir = Path(input.dicom_dir)
        if not dicom_dir.exists():
            raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")

        patient_dir = OUTPUT_DIR / wildcards.patient
        patient_dir.mkdir(parents=True, exist_ok=True)

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"{wildcards.patient}_", dir=str(OUTPUT_DIR)))
        try:
            cmd = [
                "rtpipeline",
                "--dicom-root", str(dicom_dir),
                "--outdir", str(tmp_dir),
                "--logs", str(LOGS_DIR),
                "--no-segmentation",
                "--no-dvh",
                "--no-visualize",
                "--no-radiomics",
                "--workers", "1",
            ]
            with open(log[0], "w") as logf, redirect_stdout(logf), redirect_stderr(logf):
                subprocess.run(cmd, check=True)

            course_dirs = sorted(p for p in tmp_dir.glob("**/course_*") if p.is_dir())
            if not course_dirs:
                raise RuntimeError(f"rtpipeline did not produce a course directory under {tmp_dir}")
            course_dir = course_dirs[0]

            src_ct = course_dir / "CT_DICOM"
            if not src_ct.exists():
                raise FileNotFoundError(f"CT_DICOM missing in {course_dir}")
            if Path(output.ct_dir).exists():
                shutil.rmtree(Path(output.ct_dir))
            shutil.copytree(src_ct, Path(output.ct_dir))

            for name, target in (("RP.dcm", Path(output.rp)), ("RD.dcm", Path(output.rd)), ("RS.dcm", patient_dir / "RS.dcm")):
                src = course_dir / name
                if src.exists():
                    shutil.copy2(src, target)
                else:
                    if name != "RS.dcm":
                        raise FileNotFoundError(f"Required file {name} missing in {course_dir}")

            meta_src = course_dir / "case_metadata.json"
            metadata_path = Path(output.metadata)
            if meta_src.exists():
                shutil.copy2(meta_src, metadata_path)
            else:
                metadata_path.write_text(json.dumps({
                    "patient_id": wildcards.patient,
                    "ct_dir": str(Path(output.ct_dir).resolve()),
                }, indent=2))

            data = json.loads(metadata_path.read_text())
            data.update({
                "patient_id": wildcards.patient,
                "ct_dir": str(Path(output.ct_dir).resolve()),
                "rp_path": str(Path(output.rp).resolve()),
                "rd_path": str(Path(output.rd).resolve()),
                "rs_path": str((patient_dir / "RS.dcm").resolve()),
                "rs_auto_path": str((patient_dir / "RS_auto.dcm").resolve()),
                "seg_nifti_dir": str((patient_dir / "TotalSegmentator_NIFTI").resolve()),
            })
            metadata_path.write_text(json.dumps(data, indent=2))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


rule segmentation:
    input:
        ct_dir=str(OUTPUT_DIR / "{patient}" / "CT_DICOM"),
        metadata=str(OUTPUT_DIR / "{patient}" / "metadata.json")
    output:
        nifti_dir=directory(str(OUTPUT_DIR / "{patient}" / "TotalSegmentator_NIFTI")),
        rs_auto=str(OUTPUT_DIR / "{patient}" / "RS_auto.dcm")
    conda:
        "envs/rtpipeline.yaml"
    log:
        str(LOGS_DIR / "segmentation_{patient}.log")
    run:
        import shutil
        import subprocess
        import sys

        course_dir = OUTPUT_DIR / wildcards.patient
        nifti_dir = Path(output.nifti_dir)
        rs_auto = Path(output.rs_auto)
        rs_auto.parent.mkdir(parents=True, exist_ok=True)

        if rs_auto.exists():
            rs_auto.unlink()
        if nifti_dir.exists():
            shutil.rmtree(nifti_dir)

        cmd = [
            "TotalSegmentator",
            "-i", str(Path(input.ct_dir)),
            "-o", str(nifti_dir),
            "-ot", "nifti",
        ]
        if SEG_FLAGS:
            cmd.extend(SEG_FLAGS)

        with open(log[0], "w") as logf, redirect_stdout(logf), redirect_stderr(logf):
            subprocess.run(cmd, check=True)

        sys.path.insert(0, str(ROOT_DIR))
        from rtpipeline.auto_rtstruct import build_auto_rtstruct

        rs_path = build_auto_rtstruct(course_dir)
        if not rs_path or not Path(rs_path).exists():
            raise RuntimeError(f"Failed to create RS_auto for {wildcards.patient}")
        if Path(rs_path) != rs_auto:
            shutil.copy2(rs_path, rs_auto)


rule custom_structures:
    input:
        rs_auto=str(OUTPUT_DIR / "{patient}" / "RS_auto.dcm"),
        ct_dir=str(OUTPUT_DIR / "{patient}" / "CT_DICOM")
    output:
        rs_custom=str(OUTPUT_DIR / "{patient}" / "RS_custom.dcm")
    conda:
        "envs/rtpipeline.yaml"
    log:
        str(LOGS_DIR / "custom_structures_{patient}.log")
    run:
        import shutil
        import sys

        course_dir = OUTPUT_DIR / wildcards.patient
        output_path = Path(output.rs_custom)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log[0], "w") as logf, redirect_stdout(logf), redirect_stderr(logf):
            sys.path.insert(0, str(ROOT_DIR))
            if CUSTOM_STRUCTURES_CONFIG:
                from rtpipeline.dvh import _create_custom_structures_rtstruct

                rs_manual = course_dir / "RS.dcm"
                rs_custom = _create_custom_structures_rtstruct(
                    course_dir,
                    Path(CUSTOM_STRUCTURES_CONFIG),
                    rs_manual if rs_manual.exists() else None,
                    Path(input.rs_auto)
                )
                if rs_custom and Path(rs_custom).exists():
                    shutil.copy2(rs_custom, output_path)
                else:
                    shutil.copy2(input.rs_auto, output_path)
                    print("Custom structure generation failed; falling back to RS_auto")
            else:
                shutil.copy2(input.rs_auto, output_path)


rule radiomics:
    input:
        ct_dir=str(OUTPUT_DIR / "{patient}" / "CT_DICOM"),
        rs_custom=str(OUTPUT_DIR / "{patient}" / "RS_custom.dcm")
    output:
        radiomics=str(OUTPUT_DIR / "{patient}" / "Radiomics_CT.xlsx")
    conda:
        "envs/rtpipeline-radiomics.yaml"
    threads: 4
    log:
        str(LOGS_DIR / "radiomics_{patient}.log")
    run:
        import contextlib
        import os
        import shutil
        import sys
        from pathlib import Path as _Path

        course_dir = OUTPUT_DIR / wildcards.patient
        output_path = Path(output.radiomics)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if RADIOMICS_SEQUENTIAL:
            os.environ["RTPIPELINE_RADIOMICS_SEQUENTIAL"] = "1"
        else:
            os.environ.pop("RTPIPELINE_RADIOMICS_SEQUENTIAL", None)

        params_path = _Path(RADIOMICS_PARAMS) if RADIOMICS_PARAMS else None
        custom_cfg = _Path(CUSTOM_STRUCTURES_CONFIG) if CUSTOM_STRUCTURES_CONFIG else None

        sys.path.insert(0, str(ROOT_DIR))
        from rtpipeline.config import PipelineConfig
        from rtpipeline.radiomics_conda import radiomics_for_course

        cfg = PipelineConfig(
            dicom_root=DICOM_ROOT.resolve(),
            output_root=OUTPUT_DIR.resolve(),
            logs_root=LOGS_DIR.resolve(),
            workers=WORKERS,
            radiomics_params_file=params_path,
            custom_structures_config=custom_cfg,
        )

        with open(log[0], "w") as logf, redirect_stdout(logf), redirect_stderr(logf):
            result = radiomics_for_course(course_dir, cfg, str(custom_cfg) if custom_cfg else None)
            if not result or not Path(result).exists():
                raise RuntimeError("Radiomics extraction failed")
            if Path(result) != output_path:
                shutil.copy2(result, output_path)


rule dvh:
    input:
        ct_dir=str(OUTPUT_DIR / "{patient}" / "CT_DICOM"),
        rp=str(OUTPUT_DIR / "{patient}" / "RP.dcm"),
        rd=str(OUTPUT_DIR / "{patient}" / "RD.dcm"),
        rs_custom=str(OUTPUT_DIR / "{patient}" / "RS_custom.dcm")
    output:
        dvh=str(OUTPUT_DIR / "{patient}" / "dvh_metrics.xlsx")
    conda:
        "envs/rtpipeline.yaml"
    log:
        str(LOGS_DIR / "dvh_{patient}.log")
    run:
        import shutil
        import sys

        course_dir = OUTPUT_DIR / wildcards.patient
        sys.path.insert(0, str(ROOT_DIR))
        from rtpipeline.dvh import dvh_for_course

        with open(log[0], "w") as logf, redirect_stdout(logf), redirect_stderr(logf):
            result = dvh_for_course(course_dir, CUSTOM_STRUCTURES_CONFIG or None)
            if not result or not Path(result).exists():
                raise RuntimeError("DVH computation failed")
            if Path(result) != Path(output.dvh):
                shutil.copy2(result, output.dvh)


rule visualization:
    input:
        ct_dir=str(OUTPUT_DIR / "{patient}" / "CT_DICOM"),
        dvh=str(OUTPUT_DIR / "{patient}" / "dvh_metrics.xlsx"),
        rs_custom=str(OUTPUT_DIR / "{patient}" / "RS_custom.dcm")
    output:
        viz=str(OUTPUT_DIR / "{patient}" / "Axial.html")
    conda:
        "envs/rtpipeline.yaml"
    log:
        str(LOGS_DIR / "visualization_{patient}.log")
    run:
        import shutil
        import sys

        course_dir = OUTPUT_DIR / wildcards.patient
        sys.path.insert(0, str(ROOT_DIR))
        from rtpipeline.visualize import visualize_course

        with open(log[0], "w") as logf, redirect_stdout(logf), redirect_stderr(logf):
            result = visualize_course(course_dir)
            if result:
                shutil.copy2(result, output.viz)
            else:
                Path(output.viz).write_text("<html><body><p>No DVH data available.</p></body></html>")


rule summarize:
    input:
        metadata=expand(str(OUTPUT_DIR / "{patient}" / "metadata.json"), patient=PATIENTS),
        radiomics=expand(str(OUTPUT_DIR / "{patient}" / "Radiomics_CT.xlsx"), patient=PATIENTS),
        dvh=expand(str(OUTPUT_DIR / "{patient}" / "dvh_metrics.xlsx"), patient=PATIENTS)
    output:
        metadata_summary=str(OUTPUT_DIR / "metadata_summary.xlsx"),
        radiomics_summary=str(OUTPUT_DIR / "radiomics_summary.xlsx"),
        dvh_summary=str(OUTPUT_DIR / "dvh_summary.xlsx")
    conda:
        "envs/rtpipeline.yaml"
    run:
        import json
        from pathlib import Path

        import pandas as pd

        meta_rows = []
        for meta_path in input.metadata:
            p = Path(meta_path)
            if not p.exists():
                continue
            data = json.loads(p.read_text())
            row = {
                "patient": p.parent.name,
                "plan_name": data.get("plan_name"),
                "plan_date": data.get("plan_date"),
                "course_start_date": data.get("course_start_date"),
                "course_end_date": data.get("course_end_date"),
                "total_prescription_gy": data.get("total_prescription_gy"),
                "fractions_count": data.get("fractions_count"),
                "ptv_count": data.get("ptv_count"),
                "ct_study_uid": data.get("ct_study_uid"),
            }
            meta_rows.append(row)
        pd.DataFrame(meta_rows).to_excel(output.metadata_summary, index=False)

        radiomics_files = [Path(p) for p in input.radiomics if Path(p).exists()]
        if radiomics_files:
            dfs = []
            for f in radiomics_files:
                df = pd.read_excel(f)
                df["patient"] = f.parent.name
                dfs.append(df)
            pd.concat(dfs, ignore_index=True).to_excel(output.radiomics_summary, index=False)
        else:
            pd.DataFrame().to_excel(output.radiomics_summary, index=False)

        dvh_files = [Path(p) for p in input.dvh if Path(p).exists()]
        if dvh_files:
            dfs = []
            for f in dvh_files:
                df = pd.read_excel(f)
                df["patient"] = f.parent.name
                dfs.append(df)
            pd.concat(dfs, ignore_index=True).to_excel(output.dvh_summary, index=False)
        else:
            pd.DataFrame().to_excel(output.dvh_summary, index=False)


rule clean:
    shell:
        """
        rm -rf {OUTPUT_DIR}/*/nifti
        rm -rf {OUTPUT_DIR}/*/TotalSegmentator_NIFTI
        echo "Cleaned intermediate files"
        """


rule clean_all:
    shell:
        """
        rm -rf {OUTPUT_DIR}
        rm -rf {LOGS_DIR}
        echo "Cleaned all outputs"
        """
