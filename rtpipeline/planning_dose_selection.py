"""Read-only selection of a single approved planning dose for descriptive DVH.

This does not revise organize membership or infer delivery. Archived external
paths in excluded_doses are never opened: selection uses the course's copies.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pydicom

from .course_contract import CourseContract, CourseContractError, load_course_contract, PLAN_LEVEL_DOSE_SUMMATION_TYPES
from .plan_approval import approval_status

BASIS = "single_plan_planning_dose_delivery_unlinked"
SIDECAR = "metadata/planning_dose_selection.json"


class PlanningDoseError(ValueError):
    """A counts-safe refusal; never includes an input path or DICOM identity."""


def _refuse(code):
    raise PlanningDoseError(code)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _local(root, path):
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    try:
        relative = path.relative_to(root)
    except ValueError:
        _refuse('unsafe_course_path')
    current = root
    for part in relative.parts:
        current = current / part
        if part == '..' or current.is_symlink():
            _refuse('unsafe_course_path')
    if not path.resolve().is_relative_to(root):
        _refuse('unsafe_course_path')
    return path


@dataclass(frozen=True)
class PlanningDoseSelection:
    reason_code: str
    dose_path: Path | None = None
    plan_path: Path | None = None
    evidence: dict | None = None

    @property
    def accepted(self):
        return self.reason_code == 'accepted'

    def sidecar(self, code_revision: str):
        if not self.accepted:
            _refuse(self.reason_code)
        return {**self.evidence, 'code_revision': code_revision}


def select_planning_dose(course_dir: Path | str) -> PlanningDoseSelection:
    """Return an evidence-bound selection or an explicit refusal; write nothing."""
    try:
        return _select(Path(course_dir))
    except PlanningDoseError as exc:
        return PlanningDoseSelection(str(exc))
    except (CourseContractError, OSError, ValueError, TypeError, KeyError, AttributeError):
        return PlanningDoseSelection('invalid_course_contract')


def _select(course_dir):
    if course_dir.is_symlink():
        _refuse('unsafe_course_path')
    root = course_dir.resolve()
    metadata = _local(root, root / 'metadata/case_metadata.json')
    try:
        data = json.loads(metadata.read_text())['course_contract']
        contract = CourseContract(root, metadata, data)
        if contract.selected_doses or data.get('dose_grid') is not None:
            _refuse('contract_selected_dose')
        classification = data['dose_classification']
        if classification.get('classification') != 'no_delivered_plan_dose':
            _refuse('classification_not_eligible')
        if len(contract.selected_plans) != 1:
            _refuse('selected_plan_count_not_one')
        plan_item = contract.selected_plans[0]
        plan_path = _local(root, plan_item['path'])
    except (KeyError, TypeError, AttributeError):
        _refuse('invalid_course_contract')
    try:
        plan = pydicom.dcmread(plan_path, stop_before_pixels=True)
    except Exception:
        _refuse('plan_unreadable')
    if approval_status(plan) != 'APPROVED':
        _refuse('plan_not_approved')
    plan_uid = str(getattr(plan, 'SOPInstanceUID', '') or '')
    if not plan_uid or plan_uid != plan_item.get('sop_instance_uid'):
        _refuse('plan_identity_mismatch')
    if contract.dose_qc.get('pass') is not True:
        _refuse('organize_dose_qc_failed')
    # Includes organize's identity, approval, prescription, CT provenance and
    # threshold/verdict consistency checks. No contract fields are changed.
    contract = load_course_contract(root)
    from .organize import _validated_dose_grid_geometry, _dose_plausibility
    qc = _dose_plausibility(contract.resolved_prescribed_dose_total_gy,
                           contract.delivered_dose_gy, float(contract.dose_qc['threshold_gy']))
    if not qc['dose_qc_pass']:
        _refuse('organize_dose_qc_failed')

    ct_dir = contract.planning_ct_dir
    if ct_dir is None:
        _refuse('planning_ct_unavailable')
    ct_dir = _local(root, ct_dir)
    ct_frames = set()
    try:
        for path in sorted(ct_dir.rglob('*')):
            _local(root, path)
            if not path.is_file():
                continue
            ct = pydicom.dcmread(path, stop_before_pixels=True)
            if str(getattr(ct, 'Modality', '')) != 'CT':
                _refuse('planning_ct_frame_unresolved')
            ct_frames.add(str(getattr(ct, 'FrameOfReferenceUID', '') or ''))
    except PlanningDoseError:
        raise
    except Exception:
        _refuse('planning_ct_frame_unresolved')
    if len(ct_frames) != 1 or '' in ct_frames:
        _refuse('planning_ct_frame_unresolved')

    dose_dir = _local(root, root / 'DICOM/RTDOSE')
    paths = set()
    if dose_dir.is_dir():
        for entry in dose_dir.rglob('*'):
            # Refuse directory and broken symlinks too: they can hide an
            # unreadable or additional candidate from the uniqueness check.
            entry = _local(root, entry)
            if entry.is_file():
                paths.add(entry)
    # Root dose artifacts and local excluded copies also participate in the
    # ambiguity check. External archive paths are historical evidence only.
    paths.update(root.glob('*.dcm'))
    excluded = classification.get('excluded_doses') or []
    if not isinstance(excluded, list):
        _refuse('invalid_course_contract')
    for entry in excluded:
        value = entry.get('path') if isinstance(entry, dict) else entry
        if not isinstance(value, str) or not value:
            _refuse('invalid_course_contract')
        path = Path(value)
        if not path.is_absolute() or path.is_relative_to(root):
            paths.add(_local(root, path))
    candidates = []
    matched_plan = False
    dose_seen = False
    for path in sorted(paths):
        path = _local(root, path)
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception:
            _refuse('dose_inventory_unreadable')
        if str(getattr(ds, 'Modality', '')) != 'RTDOSE':
            if path.is_relative_to(dose_dir):
                _refuse('dose_inventory_unreadable')
            continue
        dose_seen = True
        if str(getattr(ds, 'SOPClassUID', '')) != '1.2.840.10008.5.1.4.1.1.481.2':
            _refuse('dose_identity_invalid')
        refs = {str(getattr(r, 'ReferencedSOPInstanceUID', '') or '')
                for r in getattr(ds, 'ReferencedRTPlanSequence', [])}
        if refs != {plan_uid}:
            continue
        matched_plan = True
        if str(getattr(ds, 'DoseSummationType', '')).upper() in PLAN_LEVEL_DOSE_SUMMATION_TYPES:
            candidates.append((path, ds))
    if not dose_seen:
        _refuse('no_available_dose')
    if not matched_plan:
        _refuse('no_exact_plan_reference')
    if not candidates:
        _refuse('no_plan_level_dose')
    if len(candidates) != 1:
        _refuse('ambiguous_plan_level_doses')
    dose_path, dose = candidates[0]
    if not str(getattr(dose, 'SOPInstanceUID', '') or ''):
        _refuse('dose_identity_invalid')
    if str(getattr(dose, 'FrameOfReferenceUID', '') or '') not in ct_frames:
        _refuse('dose_ct_frame_mismatch')
    try:
        geometry = _validated_dose_grid_geometry(dose, dose_path)
    except (ValueError, TypeError, AttributeError):
        _refuse('dose_geometry_qc_failed')
    # Organize validates header geometry for a single grid. Also require
    # decodable, finite, nonnegative pixels before publishing a DVH selection.
    try:
        pixels = pydicom.dcmread(dose_path).pixel_array
        expected = (geometry['frames'], geometry['rows'], geometry['cols'])
        if pixels.shape not in (expected, expected[1:] if expected[0] == 1 else expected):
            _refuse('dose_pixels_invalid')
        values = pixels.astype(float) * float(dose.DoseGridScaling)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            _refuse('dose_pixels_invalid')
    except PlanningDoseError:
        raise
    except Exception:
        _refuse('dose_pixels_invalid')
    evidence = {
        'schema_version': 1,
        'selected_dose_path': dose_path.relative_to(root).as_posix(),
        'plan_uid_sha256': hashlib.sha256(plan_uid.encode()).hexdigest(),
        'basis': BASIS,
        'dose_response_eligible': False,
        'course_metadata_sha256': _sha256(metadata),
        'plan_sha256': _sha256(plan_path),
        'dose_sha256': _sha256(dose_path),
        'dose_summation_type': str(dose.DoseSummationType).upper(),
    }
    return PlanningDoseSelection('accepted', dose_path, plan_path, evidence)


def load_planning_dose_sidecar(course_dir: Path | str) -> PlanningDoseSelection:
    """Revalidate selection and byte bindings; an existing sidecar is no waiver."""
    root = Path(course_dir).resolve()
    try:
        path = _local(root, root / SIDECAR)
        if not path.exists():
            return PlanningDoseSelection('sidecar_missing')
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            _refuse('sidecar_invalid')
        if not isinstance(payload.get('code_revision'), str) or not payload['code_revision'].strip():
            _refuse('sidecar_invalid')
        selection = select_planning_dose(course_dir)
        if not selection.accepted:
            return selection
        # Exact JSON types matter: 0 must not impersonate false.
        expected = selection.sidecar(payload['code_revision'])
        if json.dumps(payload, sort_keys=True) != json.dumps(expected, sort_keys=True):
            _refuse('sidecar_stale_or_mismatched')
        return selection
    except PlanningDoseError as exc:
        return PlanningDoseSelection(str(exc))
    except (OSError, ValueError, TypeError):
        return PlanningDoseSelection('sidecar_invalid')
