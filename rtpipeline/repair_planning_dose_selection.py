"""Counts-only DVH planning-dose repair. Dry run unless --apply is supplied."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile
import warnings

from .planning_dose_selection import (
    SIDECAR, PlanningDoseError, _local, load_planning_dose_sidecar, select_planning_dose,
)


def code_revision():
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                           stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                       cwd=root, stderr=subprocess.DEVNULL, text=True)
        return revision + ('+dirty' if dirty else '')
    except (OSError, subprocess.CalledProcessError):
        # Installed packages need a truthful source identity even without git.
        from .planning_dose_selection import _sha256
        return 'source-sha256:' + _sha256(Path(__file__))


def repair_course(course_dir, *, apply=False):
    """Atomically publish only the sidecar, with a fresh check under a lock."""
    course_dir = Path(course_dir)
    selection = select_planning_dose(course_dir)
    if not selection.accepted:
        return selection.reason_code
    root = course_dir.resolve()
    try:
        target = _local(root, root / SIDECAR)
        if target.exists():
            return 'unchanged' if load_planning_dose_sidecar(root).accepted else 'sidecar_conflict'
        if not apply:
            return 'accepted'
        # A directory lock serializes this CLI without publishing a lock file.
        descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        temporary = None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            selection = select_planning_dose(root)
            if not selection.accepted:
                return selection.reason_code
            if target.exists():
                return 'unchanged' if load_planning_dose_sidecar(root).accepted else 'sidecar_conflict'
            payload = selection.sidecar(code_revision())
            with tempfile.NamedTemporaryFile(mode='w', dir=target.parent,
                                             prefix='.planning-dose-', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            # Detect concurrent source updates before publication. Consumers
            # revalidate again; no producer lock can exclude external writers.
            fresh = select_planning_dose(root)
            if not fresh.accepted or fresh.evidence != selection.evidence:
                return 'inputs_changed'
            os.replace(temporary, target)
            os.fsync(descriptor)
            return 'applied'
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            os.close(descriptor)
    except PlanningDoseError as exc:
        return str(exc)
    except OSError:
        return 'publish_failed'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--course', type=Path, action='append', default=[])
    parser.add_argument('--courses-file', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings('ignore')
    courses = args.course
    if args.courses_file:
        try:
            courses += [Path(line.strip()) for line in args.courses_file.read_text().splitlines() if line.strip()]
        except OSError:
            print(json.dumps({'outcomes': {'course_list_unreadable': 1}}))
            return 1
    if not courses:
        parser.error('provide --course or --courses-file')
    counts = Counter(repair_course(course, apply=args.apply) for course in dict.fromkeys(courses))
    print(json.dumps({'mode': 'apply' if args.apply else 'dry_run',
                      'total': sum(counts.values()), 'outcomes': dict(sorted(counts.items()))}, sort_keys=True))
    return int(any(key not in {'accepted', 'applied', 'unchanged'} for key in counts))


if __name__ == '__main__':
    raise SystemExit(main())
