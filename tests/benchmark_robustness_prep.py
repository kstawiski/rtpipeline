"""Time robustness course preparation for a base revision and the working tree.

Builds (or reuses) one realistic synthetic course: a 512x512x150 CT with 19
Manual, 80 AutoRTS and 16 RS_custom ROIs, 10 of them selected by GTV*, CTV*,
PTV* and urinary_bladder. It then runs ``robustness_prep_driver.py`` for an
export of the base revision and for the working tree, each on a fresh copy,
and prints the seconds to "masks collected" and to "extraction start", the
parent peak RSS and the process-tree peak RSS (GNU time). Extraction itself is
replaced by the driver's synthetic worker. Everything is written below
``<scratch>``, which must be a new or reusable directory inside the worktree.

Usage: python -B -s tests/benchmark_robustness_prep.py <scratch> [base-rev] [workers]
Run with ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS set (e.g. 1) to bound CPU use.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

from robustness_prep_fixture import build_course, realistic_rois  # noqa: E402

CONFIG = """radiomics_robustness:
  enabled: true
  segmentation_perturbation:
    apply_to_structures: ["GTV*", "CTV*", "PTV*", "urinary_bladder"]
"""


def main() -> None:
    scratch = Path(sys.argv[1]).resolve()
    base_rev = sys.argv[2] if len(sys.argv) > 2 else "a027945"
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    if ROOT not in scratch.parents:
        raise SystemExit("scratch must be inside the worktree")
    course = scratch / "course" / "Output" / "P1" / "C1"
    if not (course / "inputs.json").exists():
        build_course(course, dict(rows=512, columns=512, slices=150, parallel=True,
                                  workers=workers, rois=realistic_rois(512, 512, 150)))
    (scratch / "config.yaml").write_text(CONFIG)
    base = scratch / "base"
    if not (base / "rtpipeline").is_dir():
        archive = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", base_rev,
                                  "rtpipeline"], check=True, capture_output=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(base)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               ROBUSTNESS_DRIVER_WORKERS=str(workers))
    report = {}
    for label, package in (("base", base), ("current", ROOT)):
        run = scratch / f"run-{label}"
        if run.exists():
            shutil.rmtree(run)
        shutil.copytree(scratch / "course", run)
        timing = scratch / f"{label}-timing.json"
        result = subprocess.run(
            ["/usr/bin/time", "-v", sys.executable, "-B", "-s",
             str(Path(__file__).with_name("robustness_prep_driver.py")), str(package),
             str(run / "Output"), str(scratch / "config.yaml"), str(timing)],
            env=env, capture_output=True, text=True,
        )
        (scratch / f"{label}.log").write_text(result.stderr)
        if result.returncode != 0:
            raise SystemExit(f"{label} driver failed; see {scratch / (label + '.log')}")
        entry = json.loads(timing.read_text())["P1"]
        tree_peak = next((line.split(":")[-1].strip() for line in result.stderr.splitlines()
                          if "Maximum resident set size" in line), None)
        report[label] = {
            "masks_collected_s": round(entry["collected_s"], 1),
            "extraction_start_s": round(entry["extraction_start_s"], 1),
            "preparation_after_collection_s": round(
                entry["extraction_start_s"] - entry["collected_s"], 1),
            "parent_peak_rss_mib": round(entry["parent_peak_rss_kib"] / 1024),
            "time_v_max_rss_mib": round(int(tree_peak) / 1024) if tree_peak else None,
            "exit_codes": json.loads(result.stdout.strip().splitlines()[-1]),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
