"""Manager-run real-data equivalence harness. Does not launch a pipeline campaign.

Both implementations use the same input and output paths in separate Python
processes. Existing output or comparison destinations are refused. Input is
read-only; output is moved aside after each run, including failed runs.
"""
import argparse
import datetime
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    'organize_ledger.json:generated_at': 'wall-clock ledger creation time',
    'XLSX ZIP timestamps': 'workbook packaging time; member bytes are compared',
    'XLSX docProps/core.xml created/modified': 'workbook creation time; frozen in both workers to keep metadata-cache hashes comparable',
}


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def normalized(path):
    if path.suffix == '.xlsx':
        with zipfile.ZipFile(path) as archive:
            members = {}
            for name in archive.namelist():
                data = archive.read(name)
                if name == 'docProps/core.xml':
                    data = re.sub(rb'(<dcterms:(?:created|modified)[^>]*>)[^<]*(</dcterms:(?:created|modified)>)', rb'\1TIMESTAMP\2', data)
                members[name] = data
            return members
    if path.name == 'organize_ledger.json':
        return re.sub(rb'("generated_at": ")[^"]+"', rb'\1TIMESTAMP"', path.read_bytes())
    return digest(path)


def json_differences(a, b, prefix=''):
    if type(a) is not type(b):
        return [prefix or '/']
    if isinstance(a, dict):
        differences = []
        for key in sorted(a.keys() | b.keys()):
            name = prefix + '/' + str(key).replace('~', '~0').replace('/', '~1')
            if key not in a or key not in b:
                differences.append(name)
            else:
                differences.extend(json_differences(a[key], b[key], name))
        return differences
    if isinstance(a, list):
        if len(a) != len(b):
            return [prefix + '/length']
        return [key for i, (x, y) in enumerate(zip(a, b))
                for key in json_differences(x, y, prefix + '/' + str(i))]
    return [] if a == b else [prefix or '/']


def compare(before, after):
    left = {str(p.relative_to(before)): p for p in before.rglob('*')}
    right = {str(p.relative_to(after)): p for p in after.rglob('*')}
    result = dict(identical_files=0, identical_directories=0, different_files=[],
                  only_before=sorted(left.keys() - right.keys()),
                  only_after=sorted(right.keys() - left.keys()), json_key_differences={})
    for name in sorted(left.keys() & right.keys()):
        a, b = left[name], right[name]
        if a.is_dir() and b.is_dir():
            result['identical_directories'] += 1
        elif a.is_dir() != b.is_dir() or normalized(a) != normalized(b):
            result['different_files'].append(name)
            if a.suffix == '.json' and a.is_file() and b.is_file():
                try:
                    x, y = json.loads(a.read_text()), json.loads(b.read_text())
                    if a.name == 'organize_ledger.json':
                        x.pop('generated_at', None)
                        y.pop('generated_at', None)
                    result['json_key_differences'][name] = json_differences(x, y)
                except (ValueError, UnicodeError):
                    pass
        else:
            result['identical_files'] += 1
    result['identical'] = not any(result[k] for k in ('different_files', 'only_before', 'only_after'))
    return result


def worker(args):
    sys.path.insert(0, str(args.code))
    import xlsxwriter.core
    class FixedWorkbookTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2024, 1, 1, tzinfo=tz)
    xlsxwriter.core.datetime = FixedWorkbookTime
    from rtpipeline.config import PipelineConfig
    from rtpipeline import organize, meta
    assert Path(organize.__file__).resolve().parent.parent == args.code.resolve()
    worker_pids = {}
    if (args.code / 'rtpipeline/organize_scale.py').exists():
        from contextlib import contextmanager
        from rtpipeline import organize_scale
        original_results = organize_scale.ordered_process_results
        @contextmanager
        def observed_results(items, fn, workers, **kwargs):
            with original_results(items, fn, workers, **kwargs) as results:
                def observed():
                    for outcome in results:
                        if outcome.worker_pid is not None:
                            worker_pids.setdefault(fn.__name__, set()).add(outcome.worker_pid)
                        yield outcome
                yield observed()
        organize_scale.ordered_process_results = observed_results
    warnings = []
    class Capture(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                warnings.append([record.levelno, record.getMessage()])
    logging.getLogger().addHandler(Capture())
    config = PipelineConfig(args.input, args.output, args.output / '_logs',
                            max_workers_override=args.workers, dicom_copy_use_hardlinks=False)
    snapshot = {}
    try:
        courses = organize.organize_and_merge(config, metadata_snapshot=snapshot)
        meta.export_metadata(config, source_snapshot=snapshot)
        import runpy
        helpers = runpy.run_path(str(args.code / 'workflow/scripts/organize_courses.py'))
        ledger = json.loads((args.output / '_COURSES/organize_ledger.json').read_text())
        entries = sorted((dict(patient=co.patient_id, course=co.course_id,
                               path=str(co.dirs.root), complexity=helpers['_estimate_course_complexity'](co.dirs.root))
                          for co in courses), key=lambda r: (r['patient'], r['course']))
        path = args.output / 'manifests/courses.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        helpers['_write_text_atomic'](path, json.dumps(helpers['_manifest_payload'](ledger, entries), indent=2, sort_keys=True) + '\n')
    finally:
        args.warnings.write_text(json.dumps(warnings, indent=2))
        args.warnings.with_suffix('.processes.json').write_text(json.dumps(
            {name: sorted(pids) for name, pids in worker_pids.items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-code', type=Path)
    parser.add_argument('--candidate-code', type=Path, default=ROOT)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--index-processes', type=int, default=16)
    parser.add_argument('--mask-processes', type=int, default=4)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--code', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--warnings', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.baseline_code is None or args.report is None:
        parser.error('--baseline-code and --report are required')
    output = args.output.resolve()
    if output == args.input.resolve() or args.input.resolve() in output.parents:
        parser.error('output must be outside the input tree')
    destinations = {name: output.with_name(output.name + '.' + name) for name in ('before', 'after')}
    warning_paths = {name: path.with_name(path.name + '.warnings.json') for name, path in destinations.items()}
    process_paths = {name: path.with_suffix('.processes.json') for name, path in warning_paths.items()}
    logs = {name: path.with_name(path.name + '.log') for name, path in destinations.items()}
    for path in (output, args.report, *destinations.values(), *warning_paths.values(), *process_paths.values(), *logs.values()):
        if path.exists():
            parser.error(f'refusing existing destination: {path}')
    env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               RTPIPELINE_INDEX_PROCESSES=str(args.index_processes),
               RTPIPELINE_MASK_PROCESSES=str(args.mask_processes))
    statuses = {}
    for name, code in [('before', args.baseline_code), ('after', args.candidate_code)]:
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--code', str(code.resolve()),
                   '--input', str(args.input.resolve()), '--output', str(output),
                   '--workers', str(args.workers), '--warnings', str(warning_paths[name])]
        with logs[name].open('w') as log:
            statuses[name] = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        if output.exists():
            output.rename(destinations[name])
    report = dict(exit_codes=statuses, allowed_run_dependent_fields=ALLOWED,
                  workers=args.workers, index_processes=args.index_processes, mask_processes=args.mask_processes)
    report['observed_worker_pids'] = {
        name: json.loads(path.read_text()) if path.exists() else None
        for name, path in process_paths.items()}
    if all(p.exists() for p in destinations.values()):
        report.update(compare(destinations['before'], destinations['after']))
    report['warnings_identical_in_order'] = (all(p.exists() for p in warning_paths.values()) and
        warning_paths['before'].read_bytes() == warning_paths['after'].read_bytes())
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if any(statuses.values()) or not report.get('identical') or not report['warnings_identical_in_order']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
