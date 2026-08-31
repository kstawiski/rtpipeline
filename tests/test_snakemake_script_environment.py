from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from course_contract_test_utils import write_minimal_course_contract
from rtpipeline.organize_ledger import write_organize_ledger


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_SCRIPTS = ROOT / "workflow" / "scripts"


def _snakefile_script_targets() -> set[Path]:
    snakefile = (ROOT / "Snakefile").read_text(encoding="utf-8")
    relative_paths = re.findall(
        r"^\s*script:\s*\n\s*[\"']([^\"']+\.py)[\"']",
        snakefile,
        flags=re.MULTILINE,
    )
    return {(ROOT / value).resolve() for value in relative_paths}


def test_workflow_scripts_import_without_pipeline_only_dependencies() -> None:
    """Snakemake script targets must load in a dependency-light interpreter."""

    script_paths = sorted(WORKFLOW_SCRIPTS.glob("*.py"))
    script_targets = _snakefile_script_targets()
    assert script_targets
    assert script_targets <= set(script_paths)

    blocked_roots = [
        "SimpleITK",
        "nibabel",
        "numpy",
        "pandas",
        "pydicom",
        "radiomics",
        "rt_utils",
        "scipy",
        "skimage",
        "torch",
        "rtpipeline.course_contract",
        "rtpipeline.dvh",
        "rtpipeline.organize",
        "rtpipeline.radiomics_ct_contract",
        "rtpipeline.segmentation",
    ]
    code = f"""
import importlib.abc
import importlib.util
import sys
import traceback
from pathlib import Path

root = Path({str(ROOT)!r})
scripts = {[str(path) for path in script_paths]!r}
blocked = {blocked_roots!r}
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'workflow' / 'scripts'))

class BlockPipelineOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise ModuleNotFoundError(
                f"pipeline-only dependency {{fullname}} is unavailable in Snakemake"
            )
        return None

sys.meta_path.insert(0, BlockPipelineOnly())
for index, script_text in enumerate(scripts):
    script = Path(script_text)
    module_name = f"_snakemake_import_check_{{index}}_{{script.stem}}"
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not create import spec for {{script}}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        print(f"FAILED_IMPORT={{script}}", file=sys.stderr)
        traceback.print_exc()
        raise
    finally:
        sys.modules.pop(module_name, None)

loaded_blocked = sorted(
    name for name in sys.modules
    if any(name == root_name or name.startswith(root_name + '.') for root_name in blocked)
)
if loaded_blocked:
    raise RuntimeError(f"workflow imports loaded blocked modules: {{loaded_blocked}}")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout


def test_manifest_resume_runs_with_dicom_dependencies_blocked_in_outer_interpreter(
    tmp_path: Path,
) -> None:
    """The outer script delegates real DICOM validation before accepting resume."""

    output_dir = tmp_path / "output"
    course_dir = output_dir / "P1" / "C1"
    write_minimal_course_contract(course_dir)
    ledger = write_organize_ledger(
        output_dir,
        [
            {
                "patient": "P1",
                "course": "C1",
                "course_key": "C1",
                "path": str(course_dir),
                "status": "validated",
                "reason": None,
                "quarantine_path": None,
            }
        ],
    )
    (course_dir / ".organized").write_text("ok\n", encoding="utf-8")
    manifest_path = output_dir / "_COURSES" / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "rtpipeline-organized-course-manifest-v2",
                "cohort_status": ledger["status"],
                "intended_course_count": 1,
                "attempted_course_count": 1,
                "validated_course_count": 1,
                "technical_quarantine_count": 0,
                "ledger_path": "_COURSES/organize_ledger.json",
                "courses": [
                    {
                        "patient": "P1",
                        "course": "C1",
                        "path": str(course_dir),
                        "complexity": 1,
                    }
                ],
                "technical_quarantines": [],
            }
        ),
        encoding="utf-8",
    )
    log_path = tmp_path / "logs" / "organize.log"
    config_path = tmp_path / "config.yaml"
    driver = f"""
import importlib.abc
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

root = Path({str(ROOT)!r})
sys.path.insert(0, str(root))
blocked = ['pydicom', 'SimpleITK', 'pandas', 'rtpipeline.course_contract']

class BlockPipelineOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise ModuleNotFoundError(f'blocked outer dependency: {{fullname}}')
        return None

sys.meta_path.insert(0, BlockPipelineOnly())
workflow = SimpleNamespace(
    output=SimpleNamespace(manifest={str(manifest_path)!r}),
    log=[{str(log_path)!r}],
    params=SimpleNamespace(
        output_dir={str(output_dir)!r},
        root_dir={str(ROOT)!r},
        configfile={str(config_path)!r},
        radiomics_env='rtpipeline-radiomics',
        python={sys.executable!r},
        python_bin={str(Path(sys.executable).parent)!r},
        dicom_root={str(tmp_path / 'input')!r},
        logs_dir={str(tmp_path / 'logs')!r},
        custom_structures='',
        prioritize_short_courses=False,
    ),
    threads=1,
)
runpy.run_path(
    str(root / 'workflow' / 'scripts' / 'organize_courses.py'),
    init_globals={{'snakemake': workflow}},
)
if any(name in sys.modules for name in blocked):
    raise RuntimeError('outer interpreter loaded a blocked dependency')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", driver],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout
    assert "course-contract validation" in log_path.read_text(encoding="utf-8")
    assert (course_dir / ".organized").read_text(encoding="utf-8") == "ok\n"


def test_script_targets_have_no_future_import() -> None:
    """Snakemake prepends a preamble to a ``script:`` target.

    A ``from __future__`` statement is then no longer the first statement in the
    generated file, so Python raises

        SyntaxError: from __future__ imports must occur at the beginning of the file

    and the rule fails before it can log anything. This aborted a 230-course
    cohort three times. A module imported normally may still use it; only the
    files Snakemake executes directly may not.
    """
    import re
    from pathlib import Path

    snakefile = Path(__file__).resolve().parents[1] / "Snakefile"
    targets = set(re.findall(r'"(workflow/scripts/[A-Za-z0-9_]+\.py)"', snakefile.read_text()))
    assert targets, "no script: targets discovered in the Snakefile"
    offenders = []
    for rel in sorted(targets):
        path = snakefile.parent / rel
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("from __future__"):
                offenders.append(rel)
                break
    assert not offenders, (
        "Snakemake script targets must not use a __future__ import "
        f"(preamble is prepended before them): {offenders}"
    )
