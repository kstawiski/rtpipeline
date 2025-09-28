from __future__ import annotations

import json
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

configfile: "config.yaml"

ROOT_DIR = Path.cwd()
DICOM_ROOT = ROOT_DIR / config.get("dicom_root", "Example_data")
OUTPUT_DIR = ROOT_DIR / config.get("output_dir", "Data_Snakemake")
LOGS_DIR = ROOT_DIR / config.get("logs_dir", "Logs_Snakemake")
WORKERS = int(config.get("workers", 4))

MAX_LOCAL_CORES = os.cpu_count() or 1
DEFAULT_RULE_THREADS = max(1, min(WORKERS, MAX_LOCAL_CORES))

# Dynamic thread allocation based on rule type
SEGMENTATION_THREADS = min(8, MAX_LOCAL_CORES)  # TotalSegmentator benefits from 4-8 cores
RADIOMICS_THREADS = min(4, MAX_LOCAL_CORES)     # PyRadiomics works well with 2-4 cores
IO_THREADS = min(2, MAX_LOCAL_CORES)           # I/O bound tasks need fewer cores

METADATA_EXPORT_DIR = OUTPUT_DIR / "Data"
METADATA_EXPORT_FILES = [
    METADATA_EXPORT_DIR / "metadata.xlsx",
    METADATA_EXPORT_DIR / "plans.xlsx",
    METADATA_EXPORT_DIR / "structure_sets.xlsx",
    METADATA_EXPORT_DIR / "dosimetrics.xlsx",
    METADATA_EXPORT_DIR / "fractions.xlsx",
    METADATA_EXPORT_DIR / "CT_images.xlsx",
    METADATA_EXPORT_DIR / "case_metadata_all.xlsx",
    METADATA_EXPORT_DIR / "case_metadata_all.json",
]

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
        expand(str(OUTPUT_DIR / "{patient}" / "TotalSegmentator_NIFTI"), patient=PATIENTS),
        expand(str(OUTPUT_DIR / "{patient}" / "RS_auto.dcm"), patient=PATIENTS),
        expand(str(OUTPUT_DIR / "{patient}" / "RS_custom.dcm"), patient=PATIENTS),
        expand(str(OUTPUT_DIR / "{patient}" / "structure_comparison_report.json"), patient=PATIENTS),
        expand(str(OUTPUT_DIR / "{patient}" / "Radiomics_CT.xlsx"), patient=PATIENTS),
        expand(str(OUTPUT_DIR / "{patient}" / "dvh_metrics.xlsx"), patient=PATIENTS),
        expand(str(OUTPUT_DIR / "{patient}" / "Axial.html"), patient=PATIENTS),
        expand(str(OUTPUT_DIR / "{patient}" / "qc_report.json"), patient=PATIENTS),
        str(OUTPUT_DIR / "metadata_summary.xlsx"),
        str(OUTPUT_DIR / "radiomics_summary.xlsx"),
        str(OUTPUT_DIR / "dvh_summary.xlsx"),
        str(OUTPUT_DIR / "qc_summary.xlsx"),
        *(str(p) for p in METADATA_EXPORT_FILES)


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
    threads:
        IO_THREADS
    run:
        import json
        import shutil
        import subprocess
        import tempfile

        job_threads = threads

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
                "--workers", str(max(1, job_threads)),
            ]

            # Set environment variables for numerical libraries
            env = os.environ.copy()
            thread_str = str(max(1, job_threads))
            env["OMP_NUM_THREADS"] = thread_str
            env["MKL_NUM_THREADS"] = thread_str
            env["NUMEXPR_NUM_THREADS"] = thread_str
            env["NUMEXPR_MAX_THREADS"] = str(MAX_LOCAL_CORES)
            env["OPENBLAS_NUM_THREADS"] = thread_str
            env["BLAS_NUM_THREADS"] = thread_str

            with open(log[0], "w") as logf, redirect_stdout(logf), redirect_stderr(logf):
                subprocess.run(cmd, check=True, env=env)

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


rule segment_nifti:
    cache: True
    input:
        ct_dir=str(OUTPUT_DIR / "{patient}" / "CT_DICOM"),
        metadata=str(OUTPUT_DIR / "{patient}" / "metadata.json")
    output:
        nifti_dir=directory(str(OUTPUT_DIR / "{patient}" / "TotalSegmentator_NIFTI"))
    conda:
        "envs/rtpipeline.yaml"
    threads:
        SEGMENTATION_THREADS
    log:
        str(LOGS_DIR / "segment_nifti_{patient}.log")
    run:
        import shutil
        import subprocess
        import sys
        import time

        nifti_dir = Path(output.nifti_dir)
        nifti_dir.parent.mkdir(parents=True, exist_ok=True)

        if nifti_dir.exists():
            shutil.rmtree(nifti_dir)

        job_threads = threads
        cmd = [
            "TotalSegmentator",
            "-i", str(Path(input.ct_dir)),
            "-o", str(nifti_dir),
            "-ot", "nifti",
            "-nr", str(job_threads),
            "-ns", str(job_threads),
        ]
        if SEG_FLAGS:
            cmd.extend(SEG_FLAGS)

        # Add GPU support if available
        import subprocess
        try:
            subprocess.run(["nvidia-smi"], check=True, capture_output=True, timeout=5)
            cmd.extend(["--device", "gpu"])
            print("GPU detected, using GPU acceleration for TotalSegmentator")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            cmd.extend(["--device", "cpu"])
            print("No GPU detected, using CPU for TotalSegmentator")

        lock_path = LOGS_DIR / ".totalsegmentator.lock"
        while True:
            try:
                lock_path.touch(exist_ok=False)
                break
            except FileExistsError:
                time.sleep(2)

        env = os.environ.copy()
        thread_str = str(max(1, job_threads))
        # Set appropriate thread limits for numerical libraries
        env["OMP_NUM_THREADS"] = thread_str
        env["MKL_NUM_THREADS"] = thread_str
        env["NUMEXPR_NUM_THREADS"] = thread_str
        env["NUMEXPR_MAX_THREADS"] = str(MAX_LOCAL_CORES)  # Use system's max cores for the upper limit
        env["OPENBLAS_NUM_THREADS"] = thread_str
        env["BLAS_NUM_THREADS"] = thread_str

        try:
            with open(log[0], "w") as logf, redirect_stdout(logf), redirect_stderr(logf):
                subprocess.run(cmd, check=True, env=env)
                print(f"TotalSegmentator NIFTI generation completed for {wildcards.patient}")
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


rule nifti_to_rtstruct:
    cache: True
    input:
        nifti_dir=str(OUTPUT_DIR / "{patient}" / "TotalSegmentator_NIFTI"),
        ct_dir=str(OUTPUT_DIR / "{patient}" / "CT_DICOM")
    output:
        rs_auto=str(OUTPUT_DIR / "{patient}" / "RS_auto.dcm")
    conda:
        "envs/rtpipeline.yaml"
    log:
        str(LOGS_DIR / "nifti_to_rtstruct_{patient}.log")
    run:
        import sys
        import shutil

        course_dir = OUTPUT_DIR / wildcards.patient
        rs_auto = Path(output.rs_auto)
        rs_auto.parent.mkdir(parents=True, exist_ok=True)

        with open(log[0], "w") as logf, redirect_stdout(logf), redirect_stderr(logf):
            sys.path.insert(0, str(ROOT_DIR))
            from rtpipeline.auto_rtstruct import build_auto_rtstruct

            rs_path = build_auto_rtstruct(course_dir)
            if not rs_path or not Path(rs_path).exists():
                raise RuntimeError(f"Failed to create RS_auto for {wildcards.patient}")
            if Path(rs_path) != rs_auto:
                shutil.copy2(rs_path, rs_auto)
            print(f"DICOM RT structure generation completed for {wildcards.patient}")


rule merge_structures:
    input:
        rs_auto=str(OUTPUT_DIR / "{patient}" / "RS_auto.dcm"),
        ct_dir=str(OUTPUT_DIR / "{patient}" / "CT_DICOM")
    output:
        rs_custom=str(OUTPUT_DIR / "{patient}" / "RS_custom.dcm"),
        structure_report=str(OUTPUT_DIR / "{patient}" / "structure_comparison_report.json")
    conda:
        "envs/rtpipeline.yaml"
    log:
        str(LOGS_DIR / "merge_structures_{patient}.log")
    run:
        import shutil
        import sys

        course_dir = OUTPUT_DIR / wildcards.patient
        output_path = Path(output.rs_custom)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log[0], "w") as logf, redirect_stdout(logf), redirect_stderr(logf):
            sys.path.insert(0, str(ROOT_DIR))

            try:
                # First try the new structure merger
                from rtpipeline.structure_merger import merge_patient_structures

                custom_config_path = Path(CUSTOM_STRUCTURES_CONFIG) if CUSTOM_STRUCTURES_CONFIG else None
                merged_file, report_file = merge_patient_structures(course_dir, custom_config_path)

                if merged_file != output_path:
                    shutil.copy2(merged_file, output_path)
                if report_file != Path(output.structure_report):
                    shutil.copy2(report_file, output.structure_report)

                print(f"Successfully merged structures using new merger for {wildcards.patient}")

            except Exception as e:
                print(f"New structure merger failed: {e}")
                print("Falling back to legacy method")

                # Fallback to legacy method
                if CUSTOM_STRUCTURES_CONFIG:
                    from rtpipeline.dvh import _create_custom_structures_rtstruct

                    rs_manual = course_dir / "RS.dcm"
                    rs_custom = _create_custom_structures_rtstruct(
                        course_dir,
                        Path(CUSTOM_STRUCTURES_CONFIG),
                        rs_manual if rs_manual.exists() else None,
                        Path(input.rs_auto)
                    )
                    src_path = Path(rs_custom) if rs_custom else None
                    if src_path and src_path.exists():
                        if src_path.resolve() != output_path.resolve():
                            shutil.copy2(src_path, output_path)
                    else:
                        shutil.copy2(input.rs_auto, output_path)
                        print("Custom structure generation failed; falling back to RS_auto")
                else:
                    shutil.copy2(input.rs_auto, output_path)

                # Create empty report for consistency
                import json
                empty_report = {"error": str(e), "fallback_used": True}
                with open(output.structure_report, 'w') as f:
                    json.dump(empty_report, f)


rule radiomics:
    input:
        ct_dir=str(OUTPUT_DIR / "{patient}" / "CT_DICOM"),
        rs_custom=str(OUTPUT_DIR / "{patient}" / "RS_custom.dcm")
    output:
        radiomics=str(OUTPUT_DIR / "{patient}" / "Radiomics_CT.xlsx")
    conda:
        "envs/rtpipeline-radiomics.yaml"
    threads:
        RADIOMICS_THREADS
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

        job_threads = RADIOMICS_THREADS

        cfg = PipelineConfig(
            dicom_root=DICOM_ROOT.resolve(),
            output_root=OUTPUT_DIR.resolve(),
            logs_root=LOGS_DIR.resolve(),
            workers=max(1, job_threads),
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
        import pandas as pd

        course_dir = OUTPUT_DIR / wildcards.patient
        output_path = Path(output.dvh)

        with open(log[0], "w") as logf, redirect_stdout(logf), redirect_stderr(logf):
            try:
                # Check if required dose files exist
                rp_path = Path(input.rp)
                rd_path = Path(input.rd)

                if not rp_path.exists() or not rd_path.exists():
                    print(f"Missing dose files for {wildcards.patient}: RP={rp_path.exists()}, RD={rd_path.exists()}")
                    print("Creating empty DVH metrics file")
                    # Create empty DataFrame with expected columns
                    empty_df = pd.DataFrame(columns=["ROI_Name", "Volume (cm³)", "Mean Dose (Gy)", "Max Dose (Gy)"])
                    empty_df.to_excel(output_path, index=False)
                    print(f"Empty DVH file created: {output_path}")
                else:
                    sys.path.insert(0, str(ROOT_DIR))
                    from rtpipeline.dvh import dvh_for_course

                    result = dvh_for_course(course_dir, CUSTOM_STRUCTURES_CONFIG or None)
                    if not result or not Path(result).exists():
                        print("DVH computation failed, creating empty file")
                        empty_df = pd.DataFrame(columns=["ROI_Name", "Volume (cm³)", "Mean Dose (Gy)", "Max Dose (Gy)"])
                        empty_df.to_excel(output_path, index=False)
                    else:
                        if Path(result) != output_path:
                            shutil.copy2(result, output_path)
                        print(f"DVH computation successful: {output_path}")

            except Exception as e:
                print(f"Error in DVH computation for {wildcards.patient}: {str(e)}")
                # Create empty file so pipeline can continue
                empty_df = pd.DataFrame(columns=["ROI_Name", "Volume (cm³)", "Mean Dose (Gy)", "Max Dose (Gy)"])
                empty_df.to_excel(output_path, index=False)
                print(f"Emergency empty DVH file created: {output_path}")


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
            try:
                # Try both visualization functions
                from rtpipeline.visualize import visualize_course, generate_axial_review

                # First try the full visualization with DVH
                result = visualize_course(course_dir)

                if result and Path(result).exists():
                    if Path(result) != Path(output.viz):
                        shutil.copy2(result, output.viz)
                    print(f"DVH visualization created: {output.viz}")
                else:
                    # Fall back to axial review without DVH
                    print("DVH visualization failed, trying axial review")
                    axial_result = generate_axial_review(course_dir)
                    if axial_result and Path(axial_result).exists():
                        if Path(axial_result) != Path(output.viz):
                            shutil.copy2(axial_result, output.viz)
                        print(f"Axial review created: {output.viz}")
                    else:
                        # Create minimal HTML
                        fallback_html = f"""
                        <html>
                        <head><title>Visualization - {wildcards.patient}</title></head>
                        <body>
                        <h1>Patient {wildcards.patient}</h1>
                        <p>No visualization data available.</p>
                        <p>This may be due to missing dose files or visualization errors.</p>
                        </body>
                        </html>
                        """
                        Path(output.viz).write_text(fallback_html)
                        print(f"Fallback HTML created: {output.viz}")

            except Exception as e:
                print(f"Error in visualization for {wildcards.patient}: {str(e)}")
                # Create error HTML
                error_html = f"""
                <html>
                <head><title>Visualization Error - {wildcards.patient}</title></head>
                <body>
                <h1>Patient {wildcards.patient}</h1>
                <p>Visualization failed with error:</p>
                <pre>{str(e)}</pre>
                </body>
                </html>
                """
                Path(output.viz).write_text(error_html)
                print(f"Error HTML created: {output.viz}")


rule metadata_exports:
    input:
        expand(str(OUTPUT_DIR / "{patient}" / "metadata.json"), patient=PATIENTS)
    output:
        [str(p) for p in METADATA_EXPORT_FILES]
    conda:
        "envs/rtpipeline.yaml"
    run:
        import json
        import sys
        from pathlib import Path

        import pandas as pd

        sys.path.insert(0, str(ROOT_DIR))
        from rtpipeline.config import PipelineConfig
        from rtpipeline.meta import export_metadata

        cfg = PipelineConfig(
            dicom_root=DICOM_ROOT.resolve(),
            output_root=OUTPUT_DIR.resolve(),
            logs_root=LOGS_DIR.resolve(),
            workers=WORKERS,
        )

        export_metadata(cfg)

        rows = []
        for meta_path in input:
            meta_file = Path(meta_path)
            if not meta_file.exists():
                continue
            data = json.loads(meta_file.read_text())
            flattened = {}
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    flattened[key] = json.dumps(value, ensure_ascii=False)
                else:
                    flattened[key] = value
            flattened.setdefault("patient_id", data.get("patient_id", meta_file.parent.name))
            flattened.setdefault("patient_dir", meta_file.parent.name)
            rows.append(flattened)

        METADATA_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        case_xlsx = METADATA_EXPORT_DIR / "case_metadata_all.xlsx"
        case_json = METADATA_EXPORT_DIR / "case_metadata_all.json"

        if rows:
            df = pd.DataFrame(rows)
            df.to_excel(case_xlsx, index=False)
            case_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        else:
            pd.DataFrame().to_excel(case_xlsx, index=False)
            case_json.write_text("[]\n")


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
            flattened = {}
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    flattened[key] = json.dumps(value, ensure_ascii=False)
                else:
                    flattened[key] = value
            flattened.setdefault("patient_id", data.get("patient_id", p.parent.name))
            flattened.setdefault("patient_dir", p.parent.name)
            meta_rows.append(flattened)

        if meta_rows:
            pd.DataFrame(meta_rows).to_excel(output.metadata_summary, index=False)
        else:
            pd.DataFrame().to_excel(output.metadata_summary, index=False)

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


rule quality_control:
    input:
        ct_dir=str(OUTPUT_DIR / "{patient}" / "CT_DICOM"),
        metadata=str(OUTPUT_DIR / "{patient}" / "metadata.json")
    output:
        qc_report=str(OUTPUT_DIR / "{patient}" / "qc_report.json")
    conda:
        "envs/rtpipeline.yaml"
    log:
        str(LOGS_DIR / "qc_{patient}.log")
    run:
        import sys
        import shutil
        sys.path.insert(0, str(ROOT_DIR))
        from rtpipeline.quality_control import generate_qc_report

        patient_dir = OUTPUT_DIR / wildcards.patient
        qc_dir = OUTPUT_DIR / "QC"
        qc_dir.mkdir(exist_ok=True)

        with open(log[0], "w") as logf, redirect_stdout(logf), redirect_stderr(logf):
            report_path = generate_qc_report(patient_dir, qc_dir)
            if report_path != Path(output.qc_report):
                shutil.copy2(report_path, output.qc_report)


rule qc_summary:
    input:
        qc_reports=expand(str(OUTPUT_DIR / "{patient}" / "qc_report.json"), patient=PATIENTS)
    output:
        qc_summary=str(OUTPUT_DIR / "qc_summary.xlsx")
    conda:
        "envs/rtpipeline.yaml"
    run:
        import json
        import pandas as pd

        qc_data = []
        for report_path in input.qc_reports:
            with open(report_path, 'r') as f:
                data = json.load(f)

            # Flatten the nested structure
            flat_data = {
                "patient_id": data.get("patient_id", ""),
                "timestamp": data.get("timestamp", ""),
                "overall_status": data.get("overall_status", ""),
            }

            # Add check results
            checks = data.get("checks", {})
            for check_name, check_data in checks.items():
                if isinstance(check_data, dict):
                    flat_data[f"{check_name}_status"] = check_data.get("status", "")
                    if "issues" in check_data:
                        flat_data[f"{check_name}_issues"] = "; ".join(check_data["issues"])

            qc_data.append(flat_data)

        df = pd.DataFrame(qc_data)
        df.to_excel(output.qc_summary, index=False)


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
