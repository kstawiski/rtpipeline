import sys
import subprocess
import os
from pathlib import Path
import pandas as pd

def main():
    log_path = Path(snakemake.log[0])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not snakemake.params.enabled:
        # Create empty file if disabled
        summary_path = Path(snakemake.output.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_excel(summary_path, index=False)
        return

    # Environment setup
    root_dir = snakemake.params.root_dir
    env = os.environ.copy()
    existing_py_path = env.get("PYTHONPATH")
    if existing_py_path:
        if root_dir not in existing_py_path.split(os.pathsep):
            env["PYTHONPATH"] = os.pathsep.join([root_dir, existing_py_path])
    else:
        env["PYTHONPATH"] = root_dir
    
    env.setdefault("PIP_NO_BUILD_ISOLATION", "1")

    # Collect parquet files (passed via input.parquet_files if we modify the rule, 
    # OR we can reconstruct the logic of finding them. 
    # The original rule iterated dirs.
    # But here in the script we don't have easy access to _iter_course_dirs helper unless we import it.
    # However, we can pass the list of expected input files via snakemake.input)
    
    # The original rule:
    # input: robustness=_per_course_sentinels(".radiomics_robustness_done")
    # run:
    #    ...
    #    parquet_files = []
    #    for ... in _iter_course_dirs():
    #       ...
    
    # If I change the rule to pass the parquet files as input?
    # The rule input uses sentinels, not the parquet files directly.
    # But the sentinel confirms the parquet file exists.
    # I can iterate the sentinels and derive the parquet path.
    
    # Or simpler: use the CLI tool's aggregation feature which might take a directory or list of files.
    # The CLI command used is: rtpipeline.cli radiomics-robustness-aggregate --inputs ...
    
    # Let's look at how to get the list of parquet files.
    # In the script, we can find them.
    
    output_dir = Path(snakemake.params.output_dir)
    parquet_files = []
    
    # We can walk the output dir or use glob
    # Ideally we only pick up the ones corresponding to the inputs we have?
    # The rule input ensures the sentinels are there.
    
    # Let's replicate the glob logic safely.
    # Assuming structure: OUTPUT_DIR / patient / course / radiomics_robustness_ct.parquet
    
    for patient_dir in output_dir.iterdir():
        if not patient_dir.is_dir() or patient_dir.name.startswith("_") or patient_dir.name.startswith("."):
            continue
        for course_dir in patient_dir.iterdir():
            if not course_dir.is_dir() or course_dir.name.startswith("_"):
                continue
            pfile = course_dir / "radiomics_robustness_ct.parquet"
            if pfile.exists():
                parquet_files.append(str(pfile))
    
    if not parquet_files:
        # Create empty output
        summary_path = Path(snakemake.output.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_excel(summary_path, index=False)
        with log_path.open("w") as logf:
            logf.write("No robustness parquet files found\n")
        return

    cmd = [
        sys.executable,
        "-m",
        "rtpipeline.cli",
        "radiomics-robustness-aggregate",
        "--inputs",
    ] + parquet_files + [
        "--output", str(snakemake.output.summary),
        "--config", str(snakemake.params.config_file),
    ]

    with log_path.open("w") as logf:
        subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT, env=env)

if __name__ == "__main__":
    main()
