"""Transactional, course-scoped repair of organize's planning CT copies.

All checks and derived writes happen in a private candidate. Linux renameat2
RENAME_EXCHANGE publishes the complete course atomically; unsupported filesystems
refuse the repair unless two-step publication is explicitly allowed (NFS clients
do not implement RENAME_EXCHANGE). Shared manifests, ledgers and configuration
inputs are read only.
"""
from __future__ import annotations

import argparse
import ctypes
from contextlib import contextmanager
import errno
import fcntl
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import warnings
from collections import Counter

from .course_contract import CourseContract, _ct_provenance, load_course_contract, validate_course_contract
from .nifti_provenance import annotate
from .organize_ledger import read_organize_ledger, STATUS_VALIDATED
from .planning_ct_localizers import (
    PlanningCTLocalizerError, select_localizers, verify_volume,
)
from .stage_completion import validate_stage_completion_sentinel, write_stage_completion_sentinel


def _refuse(code):
    raise PlanningCTLocalizerError(code)


def _safe_course(root: Path, patient: str, course: str) -> Path:
    for component in (patient, course):
        if (not component or component in ('.', '..') or '/' in component
                or '\\' in component or '\x00' in component):
            _refuse('ct_localizer_unsafe_course_path')
    path = root / patient / course
    if any(p.is_symlink() for p in (root, root / patient, path)):
        _refuse('ct_localizer_unsafe_course_path')
    if path.resolve().parent.parent != root.resolve():
        _refuse('ct_localizer_unsafe_course_path')
    return path


def _snapshot(course):
    """Detect concurrent changes without reading unrelated downstream content."""
    result = {}
    for path in sorted(course.rglob('*')):
        if path.is_symlink():
            _refuse('ct_localizer_symlink_in_course')
        stat = path.stat()
        if path.is_file():
            result[str(path.relative_to(course))] = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        elif not path.is_dir():
            _refuse('ct_localizer_unsupported_course_entry')
    return result


def _json(path, data):
    # Replacing a staging hardlink is essential: writing through it would alter
    # the original course (or an upstream source sharing that inode).
    from .organize_ledger import _write_json_atomic
    _write_json_atomic(path, data)


def _link_or_copy(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _exchange(left: Path, right: Path) -> None:
    """Single atomic publication; either both names swap or neither changes."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    if rename is None:
        _refuse('ct_localizer_atomic_publish_unavailable')
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(left), -100, os.fsencode(right), 2) != 0:
        code = ctypes.get_errno()
        if code in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EXDEV):
            _refuse('ct_localizer_atomic_publish_unavailable')
        _refuse('ct_localizer_atomic_publish_failed')


def _two_step_publish(course: Path, candidate: Path, work: Path) -> None:
    """Publish with two renames where RENAME_EXCHANGE is unavailable (e.g. NFS).

    Opt-in and only for idle courses: between the renames the course path does
    not exist. A journal in the private work directory names both trees for
    manual recovery if the process dies in that interval; a failed second rename
    is rolled back.
    """
    previous = work / 'previous_course'
    _json(work / 'publish_journal.json', {
        'method': 'two_step_rename', 'course': str(course),
        'previous_course': str(previous), 'candidate': str(candidate)})
    os.rename(course, previous)
    try:
        os.rename(candidate, course)
    except BaseException:
        os.rename(previous, course)
        raise


def _relative(contract, path):
    try:
        return path.relative_to(contract.course_dir)
    except ValueError:
        _refuse('ct_localizer_unsafe_course_path')


@contextmanager
def _quiet_converter_output():
    """dcm2niix inherits stdout/stderr; its filenames must not escape repair.

    The repair command is serial. Redirect descriptors as well as Python
    streams because the existing converter starts a native subprocess.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(fd) for fd in (1, 2)]
    try:
        with open(os.devnull, 'w') as sink:
            for fd in (1, 2):
                os.dup2(sink.fileno(), fd)
            try:
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
    finally:
        for fd, original in zip((1, 2), saved):
            os.dup2(original, fd)
            os.close(original)


def _repair_candidate(course, candidate, completion, selection, excluded):
    from .organize import _planning_ct_summary, _original_segmentation_provenance
    from .segmentation import _collect_series_metadata, _ensure_ct_nifti
    from .config import PipelineConfig

    original = load_course_contract(course)
    old_provenance = original.planning_ct['nifti_provenance']
    ct_relative = _relative(original, original.planning_ct_dir)
    nifti_relative = _relative(original, original.planning_ct_nifti)
    sidecar_relative = _relative(original, original.resolve_path(
        old_provenance['sidecar_path'], 'planning_ct.nifti_provenance.sidecar_path'))
    ct, nifti, sidecar = (candidate / p for p in (ct_relative, nifti_relative, sidecar_relative))
    for path in excluded:
        (candidate / path.relative_to(course)).unlink()
    existing = json.loads(sidecar.read_text(encoding='utf-8'))
    if existing.get('instance_selection') is not None:
        _refuse('ct_localizer_inconsistent_evidence')
    # A prior fallback selection has a different instance-set contract. Refuse
    # rather than silently overwrite a decision made by another repair method.
    if existing.get('nifti_conversion') is not None:
        _refuse('ct_localizer_existing_conversion_selection')
    rederived = False
    conversion = None
    try:
        verify_volume(ct, nifti)
    except PlanningCTLocalizerError:
        # Use precisely organize's converter and RF10/R10d fallback chain in a
        # fresh directory. Never write through the candidate's NIfTI hardlink.
        with tempfile.TemporaryDirectory(prefix='rederive-', dir=candidate.parent) as work:
            output = Path(work)
            config = PipelineConfig(dicom_root=ct, output_root=output, logs_root=output)
            if shutil.which(config.dcm2niix_cmd) is None:
                _refuse('ct_localizer_converter_unavailable')
            with _quiet_converter_output():
                generated = _ensure_ct_nifti(
                    config, ct, output, dcm2niix_depth=0,
                    authoritative_rtstruct=candidate / _relative(original, original.authoritative_rtstruct_path))
            if generated is None:
                _refuse('ct_localizer_volume_conversion_failed')
            verify_volume(ct, generated)
            generated_sidecar = generated.with_name(generated.name[:-7] + '.metadata.json')
            conversion = json.loads(generated_sidecar.read_text()).get('nifti_conversion')
            # Preserve the published NIfTI path (including uncompressed .nii).
            if nifti.name.endswith('.nii'):
                import gzip
                nifti.unlink()
                with gzip.open(generated, 'rb') as source, nifti.open('wb') as target:
                    shutil.copyfileobj(source, target)
            else:
                os.replace(generated, nifti)
            verify_volume(ct, nifti)
        rederived = True
    metadata = _collect_series_metadata(ct)
    metadata.update(_ct_provenance(ct))
    annotate(metadata, nifti, ct, regenerated=rederived, existing_sidecar=existing)
    metadata['nifti_path'] = str(course / nifti_relative)
    metadata['source_directory'] = str(course / ct_relative)
    if conversion is not None:
        metadata['nifti_conversion'] = conversion
    selection['nifti_repair'] = {
        'status': 'rederived' if rederived else 'preserved',
        'previous_nifti_sha256': old_provenance['nifti_sha256'],
        'segmentation_outputs': ('stale_left_in_place' if rederived
                                 else 'unchanged_grid_original_masks_rebound'),
    }
    metadata['instance_selection'] = selection
    _json(sidecar, metadata)
    case_path = candidate / 'metadata/case_metadata.json'
    case = json.loads(case_path.read_text(encoding='utf-8'))
    provenance = case['course_contract']['planning_ct']['nifti_provenance']
    for key in ('series_instance_uid', 'sop_hash', 'geometry', 'nifti_geometry', 'nifti_sha256', 'instance_selection'):
        provenance[key] = metadata[key]
    for key in ('nifti_path', 'source_directory', 'generated_at', 'nifti_generated_at', 'nifti_conversion'):
        if key in metadata:
            provenance[key] = metadata[key]
    summary = _planning_ct_summary(ct)
    if not summary:
        _refuse('ct_localizer_metadata_summary_failed')
    case.update(summary)
    _json(case_path, case)
    # Organize serializes the complete JSON metadata row into this workbook.
    workbook = candidate / 'metadata/case_metadata.xlsx'
    if workbook.exists():
        import pandas as pd
        workbook.unlink()
        pd.DataFrame([case]).to_excel(workbook, index=False)
    # Rebind original masks only when their NIfTI is unchanged. A re-derived
    # NIfTI makes every existing segmentation output stale; leave those bytes
    # and their old provenance intact for the workflow rerun via .organized.
    old_masks = _original_segmentation_provenance(original.authoritative_rtstruct_path,
                                                  original.planning_ct_nifti) if not rederived else None
    for path in ([] if rederived else candidate.glob('Segmentation_Original/**/provenance.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        if not old_masks or record != old_masks:
            _refuse('ct_localizer_original_mask_evidence_mismatch')
        record['source_ct_sop_hash'] = metadata['sop_hash']
        _json(path, record)
    # Some organize contracts contain absolute RTRECORD paths within the
    # course. Validate their candidate copies without changing those decisions
    # in the published JSON. External paths still fail the contract resolver.
    def staged_paths(value):
        if isinstance(value, dict):
            return {k: staged_paths(v) for k, v in value.items()}
        if isinstance(value, list):
            return [staged_paths(v) for v in value]
        if isinstance(value, str) and value.startswith(str(course) + os.sep):
            return str(candidate / Path(value).relative_to(course))
        return value
    validate_course_contract(CourseContract(candidate, case_path,
                                            staged_paths(case['course_contract'])))
    # Reuse the already validated embedded configuration record. No shared
    # configuration file is created, rewritten or timestamped.
    dependency = candidate.parent.parent / 'organize-config.json'
    _json(dependency, completion['configuration_dependency'])
    write_stage_completion_sentinel(candidate, candidate / '.organized', stage='organize',
                                    status='ok', configuration_dependency=dependency)
    return rederived


def repair_course(course: Path, *, dry_run=False, two_step=False, details=None) -> tuple[str, int]:
    """Repair one ledger-selected course; no mutation on refusal or dry-run."""
    leftover = None
    try:
        before = _snapshot(course)
        # Use the existing sentinel as the lock inode; no lock file is published.
        # NFS emulates flock with POSIX locks, which need a descriptor opened for
        # writing (EBADF otherwise); opening r+b changes neither bytes nor mtime.
        with (course / '.organized').open('r+b') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                _refuse('ct_localizer_course_busy')
            contract = load_course_contract(course)
            ct = contract.planning_ct_dir
            if ct is None:
                return 'unchanged', 0
            excluded, selection = select_localizers(ct, contract.authoritative_rtstruct_path)
            if selection is None:
                return 'unchanged', 0
            completion = validate_stage_completion_sentinel(
                course / '.organized', expected_stage='organize',
                expected_patient=course.parent.name, expected_course=course.name)
            if completion['status'] != 'ok':
                _refuse('ct_localizer_inconsistent_evidence')
            # _COURSES is outside the patient/course namespace and is not itself
            # a Snakemake input. Only temporary children are created here.
            scratch = course.parent.parent / '_COURSES'
            with tempfile.TemporaryDirectory(prefix='.localizer-repair-', dir=scratch,
                                             ignore_cleanup_errors=True) as work:
                candidate = Path(work) / course.parent.name / course.name
                candidate.parent.mkdir()
                shutil.copytree(course, candidate, copy_function=_link_or_copy)
                rederived = _repair_candidate(course, candidate, completion, selection, excluded)
                if details is not None:
                    details.update(
                        nifti_rederived=rederived,
                        localizers=sum(item['reason_code'] == 'localizer_image_type'
                                       for item in selection['excluded_instances']),
                        duplicates=sum(item['reason_code'] == 'identical_rescaled_duplicate_position'
                                       for item in selection['excluded_instances']))
                if _snapshot(course) != before:
                    _refuse('ct_localizer_course_changed_during_repair')
                if dry_run:
                    return 'would_repair', len(excluded)
                try:
                    _exchange(course, candidate)
                except PlanningCTLocalizerError as exc:
                    if not two_step or exc.reason_code != 'ct_localizer_atomic_publish_unavailable':
                        raise
                    _two_step_publish(course, candidate, Path(work))
                # The previous course now lives in the work directory. Temporary
                # cleanup only unlinks those names; hardlinked source files remain.
                # NFS keeps silly-renamed files while the lock is open, so the work
                # directory is removed again after the lock is released.
                leftover = Path(work)
                return 'repaired', len(excluded)
    except PlanningCTLocalizerError:
        raise
    except Exception as exc:
        raise PlanningCTLocalizerError('ct_localizer_inconsistent_evidence') from exc
    finally:
        if leftover is not None and leftover.exists():
            shutil.rmtree(leftover, ignore_errors=True)


def repair_output(output_dir: Path, *, courses: list[str] | None = None, dry_run=False,
                  two_step=False) -> dict:
    summary = {'validated_courses': 0, 'selected_courses': 0, 'repaired': 0,
               'would_repair': 0, 'unchanged': 0, 'refused': 0,
               'localizers_excluded': 0, 'localizers_would_exclude': 0,
               'duplicates_excluded': 0, 'duplicates_would_exclude': 0,
               'niftis_rederived': 0, 'niftis_would_rederive': 0,
               'segmentation_outputs_stale_courses': 0, 'reason_codes': {}}
    reasons = Counter()
    try:
        root = Path(output_dir).absolute()
        if any(p.is_symlink() for p in (root, root / '_COURSES', *root.parents)):
            _refuse('ct_localizer_unsafe_course_path')
        ledger = read_organize_ledger(root)
        entries = [e for e in ledger['courses'] if e['status'] == STATUS_VALIDATED]
        summary['validated_courses'] = len(entries)
        requested = set(courses) if courses else None
        known = {f"{e['patient']}/{e['course']}" for e in entries}
        if requested is not None and requested - known:
            _refuse('ct_localizer_course_not_validated')
        selected = [e for e in entries if requested is None or f"{e['patient']}/{e['course']}" in requested]
        summary['selected_courses'] = len(selected)
        for entry in selected:
            try:
                course = _safe_course(root, entry['patient'], entry['course'])
                details = {}
                status, count = repair_course(course, dry_run=dry_run, two_step=two_step, details=details)
                summary[status] += 1
                if status == 'repaired':
                    summary['localizers_excluded'] += details['localizers']
                    summary['duplicates_excluded'] += details['duplicates']
                    summary['niftis_rederived'] += int(details['nifti_rederived'])
                    summary['segmentation_outputs_stale_courses'] += int(details['nifti_rederived'])
                elif status == 'would_repair':
                    summary['localizers_would_exclude'] += details['localizers']
                    summary['duplicates_would_exclude'] += details['duplicates']
                    summary['niftis_would_rederive'] += int(details['nifti_rederived'])
            except PlanningCTLocalizerError as exc:
                summary['refused'] += 1
                reasons[exc.reason_code] += 1
    except PlanningCTLocalizerError as exc:
        summary['refused'] += 1
        reasons[exc.reason_code] += 1
    except Exception:
        summary['refused'] += 1
        reasons['ct_localizer_invalid_organize_ledger'] += 1
    summary['reason_codes'] = dict(sorted(reasons.items()))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Repair planning CT localizers and identical duplicate positions in validated courses')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--course', action='append', help='Select a validated patient/course; repeat as needed')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--allow-two-step-publish', action='store_true',
                        help='Where RENAME_EXCHANGE is unavailable (NFS), publish with two renames; '
                             'only for idle courses (the course path is briefly absent)')
    args = parser.parse_args(argv)
    # Reports expose counts and stable reason codes only. Library exceptions and
    # DICOM converter logging must never disclose identifiers through this CLI.
    import SimpleITK as sitk
    previous = logging.root.manager.disable
    previous_warnings = sitk.ProcessObject.GetGlobalWarningDisplay()
    logging.disable(logging.CRITICAL)
    sitk.ProcessObject.SetGlobalWarningDisplay(False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = repair_output(args.output_dir, courses=args.course, dry_run=args.dry_run,
                                   two_step=args.allow_two_step_publish)
    finally:
        logging.disable(previous)
        sitk.ProcessObject.SetGlobalWarningDisplay(previous_warnings)
    print(json.dumps(result, sort_keys=True))
    return 1 if result['refused'] else 0
