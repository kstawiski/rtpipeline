"""
DICOM Copy Manager - Optimized copying with deduplication and verification.

Features:
- SOPInstanceUID-based deduplication (avoid copying same DICOM twice)
- DICOM header caching (faster re-indexing on reruns)
- Checksum verification (detect corrupted copies)
- Hardlink support (save disk space when source/dest on same filesystem)
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import pydicom

from .utils import ensure_dir, file_md5, read_dicom

logger = logging.getLogger(__name__)


@dataclass
class CopyStats:
    """Statistics from copy operations."""
    copied: int = 0
    skipped_duplicate: int = 0
    hardlinked: int = 0
    verified: int = 0
    verification_failed: int = 0

    def __str__(self) -> str:
        parts = [f"copied={self.copied}"]
        if self.skipped_duplicate:
            parts.append(f"skipped_dup={self.skipped_duplicate}")
        if self.hardlinked:
            parts.append(f"hardlinked={self.hardlinked}")
        if self.verified:
            parts.append(f"verified={self.verified}")
        if self.verification_failed:
            parts.append(f"verify_failed={self.verification_failed}")
        return ", ".join(parts)


@dataclass
class DicomCopyConfig:
    """Configuration for DICOM copy operations."""
    # SOPInstanceUID-based deduplication
    dedup_by_sop_uid: bool = True

    # Try hardlinks before copy (saves disk space)
    use_hardlinks: bool = True

    # Verify checksum after copy
    verify_checksum: bool = False

    # Cache DICOM headers for faster re-indexing
    cache_headers: bool = True

    # Path to store copy registry and header cache
    cache_dir: Optional[Path] = None


class DicomCopyManager:
    """
    Manages DICOM file copying with deduplication, verification, and caching.

    Thread-safe for concurrent copy operations.
    """

    def __init__(self, config: DicomCopyConfig, output_root: Path):
        self.config = config
        self.output_root = output_root
        self._lock = threading.Lock()

        # SOPInstanceUID -> destination path (for deduplication)
        self._sop_uid_registry: Dict[str, Path] = {}

        # Source path -> SOPInstanceUID cache
        self._header_cache: Dict[str, dict] = {}

        # Stats
        self.stats = CopyStats()

        # Cache file paths
        cache_dir = config.cache_dir or (output_root / "_CACHE")
        self._registry_path = cache_dir / "sop_uid_registry.json"
        self._header_cache_path = cache_dir / "dicom_headers.json"

        # Load existing caches if resuming
        self._load_caches()

    def _load_caches(self) -> None:
        """Load existing registry and header cache from disk."""
        if self._registry_path.exists():
            try:
                data = json.loads(self._registry_path.read_text(encoding="utf-8"))
                # Only keep entries where destination still exists
                for uid, path_str in data.items():
                    path = Path(path_str)
                    if path.exists():
                        self._sop_uid_registry[uid] = path
                logger.info(
                    "Loaded %d existing SOPInstanceUID entries from registry",
                    len(self._sop_uid_registry)
                )
            except Exception as e:
                logger.warning("Failed to load SOP UID registry: %s", e)

        if self.config.cache_headers and self._header_cache_path.exists():
            try:
                self._header_cache = json.loads(
                    self._header_cache_path.read_text(encoding="utf-8")
                )
                logger.info(
                    "Loaded %d cached DICOM headers",
                    len(self._header_cache)
                )
            except Exception as e:
                logger.warning("Failed to load header cache: %s", e)

    def save_caches(self) -> None:
        """Persist registry and header cache to disk."""
        ensure_dir(self._registry_path.parent)

        try:
            registry_data = {
                uid: str(path) for uid, path in self._sop_uid_registry.items()
            }
            self._registry_path.write_text(
                json.dumps(registry_data, indent=2),
                encoding="utf-8"
            )
        except Exception as e:
            logger.warning("Failed to save SOP UID registry: %s", e)

        if self.config.cache_headers:
            try:
                self._header_cache_path.write_text(
                    json.dumps(self._header_cache, indent=2),
                    encoding="utf-8"
                )
            except Exception as e:
                logger.warning("Failed to save header cache: %s", e)

    def get_cached_header(self, src: Path) -> Optional[dict]:
        """Get cached DICOM header for a source file."""
        key = str(src.resolve())
        return self._header_cache.get(key)

    def _cache_header(self, src: Path, ds: pydicom.Dataset) -> dict:
        """Cache essential DICOM header fields."""
        header = {
            "SOPInstanceUID": str(getattr(ds, "SOPInstanceUID", "")),
            "SOPClassUID": str(getattr(ds, "SOPClassUID", "")),
            "Modality": str(getattr(ds, "Modality", "")),
            "PatientID": str(getattr(ds, "PatientID", "")),
            "StudyInstanceUID": str(getattr(ds, "StudyInstanceUID", "")),
            "SeriesInstanceUID": str(getattr(ds, "SeriesInstanceUID", "")),
            "InstanceNumber": getattr(ds, "InstanceNumber", None),
            "SeriesNumber": getattr(ds, "SeriesNumber", None),
        }
        key = str(src.resolve())
        with self._lock:
            self._header_cache[key] = header
        return header

    def _get_sop_uid(self, src: Path) -> Optional[str]:
        """Get SOPInstanceUID from file, using cache if available."""
        key = str(src.resolve())

        # Check cache first
        cached = self._header_cache.get(key)
        if cached and cached.get("SOPInstanceUID"):
            return cached["SOPInstanceUID"]

        # Read from file
        ds = read_dicom(src)
        if ds is None:
            return None

        sop_uid = str(getattr(ds, "SOPInstanceUID", ""))
        if not sop_uid:
            return None

        # Cache the header
        if self.config.cache_headers:
            self._cache_header(src, ds)

        return sop_uid

    def _try_hardlink(self, src: Path, dst: Path) -> bool:
        """Try to create hardlink, return True if successful."""
        if not self.config.use_hardlinks:
            return False

        try:
            # Check if on same filesystem
            src_dev = src.stat().st_dev
            dst_dev = dst.parent.stat().st_dev
            if src_dev != dst_dev:
                return False

            # Remove existing destination if present
            if dst.exists():
                dst.unlink()

            os.link(src, dst)
            with self._lock:
                self.stats.hardlinked += 1
            return True
        except (OSError, PermissionError):
            return False

    def _verify_copy(self, src: Path, dst: Path) -> bool:
        """Verify destination matches source by checksum."""
        if not self.config.verify_checksum:
            return True

        try:
            src_hash = file_md5(src)
            dst_hash = file_md5(dst)
            match = src_hash == dst_hash

            with self._lock:
                if match:
                    self.stats.verified += 1
                else:
                    self.stats.verification_failed += 1
                    logger.error(
                        "Checksum mismatch: %s (%s) vs %s (%s)",
                        src, src_hash, dst, dst_hash
                    )
            return match
        except Exception as e:
            logger.warning("Checksum verification failed: %s", e)
            return False

    def copy_dicom(
        self,
        src: Path,
        dst: Path,
        skip_if_exists: bool = True,
    ) -> Tuple[Path, bool]:
        """
        Copy a DICOM file with deduplication and verification.

        Args:
            src: Source DICOM file path
            dst: Destination path
            skip_if_exists: Skip if destination already exists

        Returns:
            Tuple of (actual destination path, was_copied)
            If deduplicated, returns path to existing copy.
        """
        ensure_dir(dst.parent)

        # Check if destination already exists
        if skip_if_exists and dst.exists():
            return dst, False

        # SOPInstanceUID deduplication
        sop_uid = None  # Initialize to avoid UnboundLocalError when dedup disabled
        if self.config.dedup_by_sop_uid:
            sop_uid = self._get_sop_uid(src)
            if sop_uid:
                with self._lock:
                    existing = self._sop_uid_registry.get(sop_uid)
                    if existing and existing.exists() and existing != dst:
                        self.stats.skipped_duplicate += 1
                        logger.debug(
                            "Skipping duplicate SOP %s: %s already at %s",
                            sop_uid, src, existing
                        )
                        return existing, False

        # Try hardlink first
        if self._try_hardlink(src, dst):
            if self.config.dedup_by_sop_uid and sop_uid:
                with self._lock:
                    self._sop_uid_registry[sop_uid] = dst
            return dst, True

        # Fall back to regular copy
        if dst.exists() and not os.path.samefile(src, dst):
            dst.unlink()
        shutil.copy2(src, dst)

        with self._lock:
            self.stats.copied += 1
            if self.config.dedup_by_sop_uid and sop_uid:
                self._sop_uid_registry[sop_uid] = dst

        # Verify if enabled
        if self.config.verify_checksum:
            if not self._verify_copy(src, dst):
                # Re-copy on failure
                shutil.copy2(src, dst)
                self._verify_copy(src, dst)

        return dst, True

    def copy_dicom_into(
        self,
        src: Path,
        dst_dir: Path,
        prefix: Optional[str] = None,
    ) -> Tuple[Path, bool]:
        """
        Copy DICOM into directory, handling name clashes.

        Args:
            src: Source DICOM file
            dst_dir: Destination directory
            prefix: Optional filename prefix

        Returns:
            Tuple of (destination path, was_copied)
        """
        ensure_dir(dst_dir)

        name = src.name
        if prefix:
            name = f"{prefix}_{name}"

        dest = dst_dir / name

        # Handle name clashes
        if dest.exists():
            # Check if it's the same file by SOP UID
            if self.config.dedup_by_sop_uid:
                src_uid = self._get_sop_uid(src)
                dst_uid = self._get_sop_uid(dest)
                if src_uid and src_uid == dst_uid:
                    # Same DICOM, skip
                    with self._lock:
                        self.stats.skipped_duplicate += 1
                    return dest, False

            # Different file, need unique name
            stem = dest.stem
            suffix = dest.suffix
            counter = 1
            while dest.exists():
                dest = dst_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        return self.copy_dicom(src, dest)

    def is_duplicate(self, src: Path) -> bool:
        """Check if source DICOM was already copied (by SOP UID)."""
        if not self.config.dedup_by_sop_uid:
            return False

        sop_uid = self._get_sop_uid(src)
        if not sop_uid:
            return False

        with self._lock:
            existing = self._sop_uid_registry.get(sop_uid)
            return existing is not None and existing.exists()

    def get_existing_copy(self, src: Path) -> Optional[Path]:
        """Get path to existing copy of this DICOM (by SOP UID)."""
        if not self.config.dedup_by_sop_uid:
            return None

        sop_uid = self._get_sop_uid(src)
        if not sop_uid:
            return None

        with self._lock:
            existing = self._sop_uid_registry.get(sop_uid)
            if existing and existing.exists():
                return existing
        return None


# Module-level singleton for global access
_copy_manager: Optional[DicomCopyManager] = None
_copy_manager_lock = threading.Lock()


def get_copy_manager(
    output_root: Path,
    config: Optional[DicomCopyConfig] = None,
) -> DicomCopyManager:
    """Get or create the global DicomCopyManager instance."""
    global _copy_manager

    with _copy_manager_lock:
        if _copy_manager is None:
            if config is None:
                config = DicomCopyConfig()
            _copy_manager = DicomCopyManager(config, output_root)
        return _copy_manager


def reset_copy_manager() -> None:
    """Reset the global copy manager (for testing)."""
    global _copy_manager
    with _copy_manager_lock:
        if _copy_manager is not None:
            _copy_manager.save_caches()
        _copy_manager = None
