import sys
import subprocess
import os
from pathlib import Path

def main():
    sentinel_path = Path(snakemake.output.sentinel)
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    
    log_path = Path(snakemake.log[0])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not snakemake.params.enabled:
        sentinel_path.write_text("disabled\n", encoding="utf-8")
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

    course_path = Path(snakemake.params.output_dir) / snakemake.wildcards.patient / snakemake.wildcards.course
    output_parquet = course_path / "radiomics_robustness_ct.parquet"

    cmd = [
        sys.executable,
        "-m",
        "rtpipeline.cli",
        "radiomics-robustness",
        "--course-dir", str(course_path),
        "--config", str(snakemake.params.config_file),
        "--output", str(output_parquet),
    ]
    
    cmd.extend(["--max-workers", str(snakemake.params.max_workers)])

    try:
        with log_path.open("w") as logf:
            subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT, env=env)
        sentinel_path.write_text("ok\n", encoding="utf-8")
    except subprocess.CalledProcessError as e:
        # Log error but don't fail the entire pipeline (as per original rule logic)
        with log_path.open("a") as logf:
            logf.write(f"\nRobustness analysis failed: {e}\n")
        sentinel_path.write_text(f"failed: {e}\n", encoding="utf-8")

if __name__ == "__main__":
    main()
