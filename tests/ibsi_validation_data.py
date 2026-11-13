"""Locate existing repository IBSI validation data, including Git worktrees."""
from pathlib import Path
import subprocess


def validation_root(repo):
    repo = Path(repo).resolve()
    candidates = [repo / "validation/ibsi"]
    # Ignored validation data belongs to the main checkout, not each worktree.
    common = Path(subprocess.check_output(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=repo, text=True).strip())
    candidates.append(common.parent / "validation/ibsi")
    for root in candidates:
        source = root / "data/public_sources_20261027"
        if (source / "theibsi/data_sets/ibsi_1_ct_radiomics_phantom/nifti/image/phantom.nii.gz").is_file():
            return root
    return None
