import sys
import subprocess
import os
from pathlib import Path

def main():
    # Snakemake injects 'snakemake' object
    
    sentinel_path = Path(snakemake.output.sentinel)
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    
    log_path = Path(snakemake.log[0])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Environment setup
    root_dir = snakemake.params.root_dir
    env = os.environ.copy()
    existing_py_path = env.get("PYTHONPATH")
    if existing_py_path:
        if root_dir not in existing_py_path.split(os.pathsep):
            env["PYTHONPATH"] = os.pathsep.join([root_dir, existing_py_path])
    else:
        env["PYTHONPATH"] = root_dir
        
    # Disable pip build isolation
    env.setdefault("PIP_NO_BUILD_ISOLATION", "1")

    cmd = [
        sys.executable,
        "-m",
        "rtpipeline.cli",
        "--dicom-root", snakemake.params.dicom_root,
        "--outdir", snakemake.params.outdir,
        "--logs", snakemake.params.logs_dir,
        "--stage", "radiomics",
        "--course-filter", f"{snakemake.wildcards.patient}/{snakemake.wildcards.course}",
    ]
    
    cmd.extend(["--max-workers", str(snakemake.params.max_workers)])
    
    if snakemake.params.sequential:
        cmd.append("--sequential-radiomics")
    
    if snakemake.params.params_file:
        cmd.extend(["--radiomics-params", snakemake.params.params_file])
        
    if snakemake.params.params_mr_file:
        cmd.extend(["--radiomics-params-mr", snakemake.params.params_mr_file])
        
    if snakemake.params.max_voxels:
        cmd.extend(["--radiomics-max-voxels", str(snakemake.params.max_voxels)])
        
    if snakemake.params.min_voxels:
        cmd.extend(["--radiomics-min-voxels", str(snakemake.params.min_voxels)])
        
    for roi in snakemake.params.skip_rois:
        cmd.extend(["--radiomics-skip-roi", roi])
        
    if snakemake.params.custom_structures:
        cmd.extend(["--custom-structures", snakemake.params.custom_structures])

    with log_path.open("w") as logf:
        subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT, env=env)
        
    sentinel_path.write_text("ok\n", encoding="utf-8")

if __name__ == "__main__":
    main()
