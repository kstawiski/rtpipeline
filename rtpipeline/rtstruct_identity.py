"""Content-bound derived RTSTRUCT identity, separate from parent provenance."""
from __future__ import annotations

import copy
import hashlib
import io
from pathlib import Path

import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import PYDICOM_ROOT_UID

# This is the same root used by the project's existing generate_uid() calls.
UID_ROOT = PYDICOM_ROOT_UID
IDENTITY_SCHEME = 'rtpipeline-rtstruct-sha256-v1'
SCHEME_TAG = (0x0011, 0x1001)


class RTStructIdentityError(RuntimeError):
    """A requested identity does not identify the supplied RTSTRUCT content."""


def _content_uid(dataset) -> str:
    canonical = copy.deepcopy(dataset)
    canonical.SOPInstanceUID = '1.2.3'
    canonical.file_meta.MediaStorageSOPInstanceUID = '1.2.3'
    canonical.preamble = b'\0' * 128
    buffer = io.BytesIO()
    pydicom.dcmwrite(buffer, canonical, write_like_original=False)
    digest = hashlib.sha256(buffer.getvalue()).digest()
    return UID_ROOT + str(int.from_bytes(digest, 'big'))[:64-len(UID_ROOT)]


def assign_derived_identity(dataset, parent_uid: str, parent_class_uid: str) -> str:
    """Call after the final content edit, before atomic publication.

    Hash the deterministic DICOM serialization with only the self-referential
    instance UID normalized. Parent identity remains part of the hash.
    """
    if not parent_uid:
        raise RTStructIdentityError('RTSTRUCT_PARENT_IDENTITY_MISSING')
    if not getattr(dataset, 'file_meta', None):
        dataset.file_meta = FileMetaDataset()
    parent = Dataset()
    parent.ReferencedSOPClassUID = parent_class_uid
    parent.ReferencedSOPInstanceUID = parent_uid
    dataset.PredecessorStructureSetSequence = Sequence([parent])
    block = dataset.private_block(0x0011, 'RTPIPELINE_IDENTITY', create=True)
    block.add_new(1, 'LO', IDENTITY_SCHEME)
    uid = _content_uid(dataset)
    if uid == parent_uid:
        raise RTStructIdentityError('RTSTRUCT_PARENT_UID_COLLISION')
    dataset.SOPInstanceUID = uid
    dataset.file_meta.MediaStorageSOPInstanceUID = uid
    return uid


def validate_rtstruct_identity(dataset, expected_uid: str | None = None, *, require_derived=False) -> str:
    actual = str(getattr(dataset, 'SOPInstanceUID', '') or '')
    if expected_uid is not None and actual != expected_uid:
        raise RTStructIdentityError('RTSTRUCT_STALE_UID_REFERENCE')
    meta = getattr(dataset, 'file_meta', None)
    if meta is not None and str(getattr(meta, 'MediaStorageSOPInstanceUID', actual)) != actual:
        raise RTStructIdentityError('RTSTRUCT_FILE_META_UID_MISMATCH')
    try:
        block = dataset.private_block(0x0011, 'RTPIPELINE_IDENTITY', create=False)
        value = block[1].value
        scheme = value.decode('ascii').rstrip('\0 ') if isinstance(value, bytes) else str(value)
    except (KeyError, AttributeError):
        scheme = ''
    if require_derived and scheme != IDENTITY_SCHEME:
        raise RTStructIdentityError('RTSTRUCT_DERIVED_IDENTITY_MISSING')
    if scheme == IDENTITY_SCHEME:
        parents = getattr(dataset, 'PredecessorStructureSetSequence', [])
        if not parents or actual in {str(p.ReferencedSOPInstanceUID) for p in parents}:
            raise RTStructIdentityError('RTSTRUCT_PARENT_UID_COLLISION')
        if _content_uid(dataset) != actual:
            raise RTStructIdentityError('RTSTRUCT_CONTENT_UID_MISMATCH')
    return actual


def require_rtstruct_identity(path: Path, expected_uid: str | None = None, *, require_derived=False) -> str:
    dataset = pydicom.dcmread(path, stop_before_pixels=True)
    return validate_rtstruct_identity(dataset, expected_uid, require_derived=require_derived)
