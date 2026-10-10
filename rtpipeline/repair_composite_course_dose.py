"""Counts-only RF19 composite course dose repair. Dry run unless --apply is supplied."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import logging
import os
from pathlib import Path
import tempfile
import warnings

from .composite_course_dose import (
    OUTPUT, SIDECAR, build_composite_dose, load_composite_sidecar, select_composite_course_dose,
)
from .planning_dose_selection import PlanningDoseError, _local, _sha256
from .repair_planning_dose_selection import code_revision


def repair_course(course_dir, *, apply=False):
    """Publish the composite RTDOSE, then its sidecar, under a directory lock."""
    course_dir = Path(course_dir)
    selection = select_composite_course_dose(course_dir)
    if not selection.accepted:
        return selection.reason_code
    root = course_dir.resolve()
    try:
        target = _local(root, root / SIDECAR)
        output = _local(root, root / OUTPUT)
        if target.exists():
            return 'unchanged' if load_composite_sidecar(root).accepted else 'sidecar_conflict'
        if output.exists():
            return 'output_conflict'
        if not apply:
            return 'accepted'
        descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        temporaries = []
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            selection = select_composite_course_dose(root)
            if not selection.accepted:
                return selection.reason_code
            if target.exists() or output.exists():
                return 'sidecar_conflict'
            dose, grid = build_composite_dose(selection)
            output.parent.mkdir(exist_ok=True)
            handle, name = tempfile.mkstemp(dir=output.parent, prefix='.rd-composite-')
            os.close(handle)
            temporaries.append(Path(name))
            dose.save_as(name, enforce_file_format=True)
            payload = {**selection.evidence, 'code_revision': code_revision(),
                       'output_dose_sha256': _sha256(Path(name)), 'output_grid': grid}
            with tempfile.NamedTemporaryFile(mode='w', dir=target.parent,
                                             prefix='.composite-dose-', delete=False) as stream:
                temporaries.append(Path(stream.name))
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            fresh = select_composite_course_dose(root)
            if not fresh.accepted or fresh.evidence != selection.evidence:
                return 'inputs_changed'
            # The output is published first; without its sidecar no consumer
            # uses it, and a later sidecar binds its exact bytes.
            os.replace(temporaries[0], output)
            os.replace(temporaries[1], target)
            os.fsync(descriptor)
            return 'applied'
        finally:
            for temporary in temporaries:
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
    courses = args.course
    if args.courses_file:
        try:
            courses += [Path(line.strip()) for line in args.courses_file.read_text().splitlines() if line.strip()]
        except OSError:
            print(json.dumps({'outcomes': {'course_list_unreadable': 1}}))
            return 1
    if not courses:
        parser.error('provide --course or --courses-file')
    previous_level = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            counts = Counter(repair_course(course, apply=args.apply) for course in dict.fromkeys(courses))
    finally:
        logging.disable(previous_level)
    print(json.dumps({'mode': 'apply' if args.apply else 'dry_run',
                      'total': sum(counts.values()), 'outcomes': dict(sorted(counts.items()))}, sort_keys=True))
    return int(any(key not in {'accepted', 'applied', 'unchanged'} for key in counts))


if __name__ == '__main__':
    raise SystemExit(main())
