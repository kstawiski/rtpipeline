"""
Radiomics robustness analysis module.

Implements an **RTpipeline-adapted perturbation framework** inspired by
Zwanenburg et al. 2019 (Sci Rep) for feature stability assessment:

- NTCV perturbation chain (Noise + Translation + Contour + Volume)
- Rotation sensitivity analysis (optional, for Zwanenburg N/T/R/V/C parity)
- Mask perturbations (erosion/dilation with volume adaptation)
- Image noise injection (Gaussian noise in HU)
- Rigid translations (±2-4 mm shifts)
- Contour randomization (morphological boundary perturbation)
- ICC (Intraclass Correlation Coefficient) computation
- CoV (Coefficient of Variation) and QCD metrics
- Redundancy pruning (Spearman correlation clustering)
- Determinants-of-instability analysis (mixed-effects regression)
- Multi-axis robustness evaluation (segmentation perturbation, segmentation method, scan-rescan)

Note: The default NTCV chain omits **Rotation (R)** and uses **morphological**
contour perturbation rather than supervoxel-based randomization. These
differences from the original Zwanenburg 2019 N/T/R/V/C framework are
intentional simplifications documented in the package guide. Rotation
sensitivity analysis is available via
`generate_rotation_sensitivity_perturbations()`.

Based on 2023-2025 radiomics stability research:
- Zwanenburg et al. 2019 (Sci Rep): NTCV perturbation chains with ICC >0.75
- Lo Iacono et al. 2024 (SpringerLink): volume adaptation for stability
- Poirot et al. 2022 (Sci Rep): multi-method ICC with Pingouin
- Traverso et al. 2024: cross-extractor reproducibility
- OPC RobustDB 2025 (PMID: 41367878): feature stability atlas with pruning
- Perturbation count must be reported explicitly for the configured Cartesian grid
- Configurable conservative research thresholds: ICC >0.90 and CoV <10%
"""

from __future__ import annotations

import logging
import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from multiprocessing import get_context
from multiprocessing import TimeoutError as MPTimeoutError
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk

from .config import PipelineConfig
from .course_contract import load_course_contract
from .layout import build_course_dirs
from .radiomics_ct_contract import read_authoritative_ct_publication
from .rt_details import DEFAULT_ROI_FAMILY_NAMES

logger = logging.getLogger(__name__)


ROBUSTNESS_MEASUREMENT_TYPE = "segmentation_perturbation"
ROBUSTNESS_SOURCE_IDENTITY_COLUMNS = (
    "patient_id",
    "course_id",
    "series_uid",
    "segmentation_source",
    "mask_identity",
    "roi_original_name",
    "stable_roi_identifier",
)
ROBUSTNESS_PERTURBATION_IDENTITY_COLUMNS = (
    "measurement_type",
    "perturbation_id",
    "perturbed_mask_identity",
)


class RobustnessIdentityError(RuntimeError):
    """An individual robustness result cannot be traced to its source ROI."""


@dataclass(frozen=True)
class RobustnessRoiIdentity:
    """Original ROI identity copied from the authoritative CT publication."""

    patient_id: str
    course_id: str
    series_uid: str
    segmentation_source: str
    mask_identity: str
    roi_original_name: str
    stable_roi_identifier: str

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RobustnessRoiIdentity":
        normalized = {
            column: str(values.get(column) or "").strip()
            for column in ROBUSTNESS_SOURCE_IDENTITY_COLUMNS
        }
        missing = sorted(
            column
            for column, value in normalized.items()
            if not value or value.lower() in {"nan", "none", "<na>"}
        )
        if missing:
            raise ValueError(
                "Original CT ROI identity is incomplete: " + ", ".join(missing)
            )
        return cls(**normalized)

    def as_dict(self) -> Dict[str, str]:
        return {
            column: getattr(self, column)
            for column in ROBUSTNESS_SOURCE_IDENTITY_COLUMNS
        }

    @property
    def key(self) -> Tuple[str, str]:
        return self.segmentation_source, self.roi_original_name


def _perturbed_mask_identity(mask: sitk.Image) -> str:
    """Return a geometry-bound digest for one perturbed binary mask."""
    array = (sitk.GetArrayViewFromImage(mask) != 0).astype(np.uint8, copy=False)
    geometry = {
        "direction": [float(value) for value in mask.GetDirection()],
        "origin": [float(value) for value in mask.GetOrigin()],
        "size": [int(value) for value in mask.GetSize()],
        "spacing": [float(value) for value in mask.GetSpacing()],
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def _identity_catalog_from_main_frame(
    frame: pd.DataFrame,
    *,
    expected_patient_id: str,
    expected_course_id: str,
    expected_series_uid: str,
) -> Tuple[
    Dict[Tuple[str, str], RobustnessRoiIdentity],
    Dict[Tuple[str, str], Dict[str, str]],
]:
    """Build a unique original-ROI catalog from main CT radiomics output."""
    missing = sorted(set(ROBUSTNESS_SOURCE_IDENTITY_COLUMNS) - set(frame.columns))
    if missing:
        raise RuntimeError(
            "Authoritative CT radiomics publication lacks identity columns: "
            + ", ".join(missing)
        )

    catalog: Dict[Tuple[str, str], RobustnessRoiIdentity] = {}
    issues: Dict[Tuple[str, str], Dict[str, str]] = {}
    grouped = frame.groupby(
        ["segmentation_source", "roi_original_name"],
        sort=False,
        dropna=False,
    )
    for raw_key, group in grouped:
        key = tuple(str(value or "").strip() for value in raw_key)
        try:
            identities = {
                RobustnessRoiIdentity.from_mapping(row)
                for row in group[list(ROBUSTNESS_SOURCE_IDENTITY_COLUMNS)].to_dict(
                    "records"
                )
            }
        except ValueError as exc:
            issues[key] = {
                "reason_code": "main_radiomics_identity_incomplete",
                "reason_detail": str(exc),
            }
            continue
        if len(identities) != 1:
            issues[key] = {
                "reason_code": "main_radiomics_identity_ambiguous",
                "reason_detail": (
                    f"Main CT radiomics has {len(identities)} original identities "
                    f"for {key[0]}/{key[1]}"
                ),
            }
            continue
        identity = next(iter(identities))
        expected = {
            "patient_id": str(expected_patient_id),
            "course_id": str(expected_course_id),
            "series_uid": str(expected_series_uid),
        }
        mismatches = [
            f"{column}={getattr(identity, column)!r} expected {value!r}"
            for column, value in expected.items()
            if getattr(identity, column) != value
        ]
        if mismatches:
            issues[key] = {
                "reason_code": "main_radiomics_identity_course_mismatch",
                "reason_detail": "; ".join(mismatches),
            }
            continue
        catalog[key] = identity
    return catalog, issues


def _load_main_ct_identity_catalog(
    course_dir: Path,
    *,
    expected_series_uid: str,
) -> Tuple[
    Dict[Tuple[str, str], RobustnessRoiIdentity],
    Dict[Tuple[str, str], Dict[str, str]],
]:
    publication_path = course_dir / "radiomics_ct.parquet"
    frame = read_authoritative_ct_publication(publication_path)
    return _identity_catalog_from_main_frame(
        frame,
        expected_patient_id=course_dir.parent.name,
        expected_course_id=course_dir.name,
        expected_series_uid=expected_series_uid,
    )


def _write_robustness_identity_ledger(
    course_dir: Path,
    *,
    selected_count: int,
    rows: List[Dict[str, Any]],
) -> Path:
    """Atomically record resolved and excluded robustness ROI identities."""
    path = course_dir / "metadata" / "radiomics_robustness_identity.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "patient_id": course_dir.parent.name,
        "course_id": course_dir.name,
        "identity_source": "radiomics_ct.parquet",
        "selected_roi_count": int(selected_count),
        "identity_resolved_roi_count": sum(
            row.get("identity_status") == "resolved" for row in rows
        ),
        "identity_excluded_roi_count": sum(
            row.get("identity_status") == "excluded" for row in rows
        ),
        "rows": rows,
    }
    fd, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    return path


# ============================================================================
# Robustness source dispositions: terminal, non-technical RTSTRUCT source
# outcomes observed while the robustness pass collected masks.
# ============================================================================

ROBUSTNESS_SOURCE_DISPOSITIONS_FILENAME = "radiomics_robustness_source_dispositions.json"
ROBUSTNESS_SOURCE_DISPOSITIONS_KIND = "robustness_source_dispositions"
ROBUSTNESS_SOURCE_DISPOSITIONS_SCHEMA_VERSION = 1

# Only terminal, non-technical source dispositions may ever be published here.
# A technical read failure fails the course, so a row claiming a measurement
# ("success"/"measured") or an unresolved technical failure ("failed") is a
# corrupt artifact, not a weaker but acceptable record.
ROBUSTNESS_SOURCE_DISPOSITION_STATUSES = frozenset({"nonvolumetric_nonmeasurement"})
ROBUSTNESS_SOURCE_DISPOSITION_ROW_FIELDS = (
    "segmentation_source",
    "source_path",
    "rtstruct_sop_instance_uid",
    "roi_name",
    "roi_number",
    "status",
    "failure_kind",
    "structural_code",
    "reason",
)
# One published row per ROI identity within one RTSTRUCT source.
ROBUSTNESS_SOURCE_DISPOSITION_ROW_IDENTITY = (
    "segmentation_source",
    "source_path",
    "rtstruct_sop_instance_uid",
    "roi_number",
    "roi_name",
)
ROBUSTNESS_SOURCE_BINDING_FIELDS = (
    "segmentation_source",
    "source_path",
    "rtstruct_sop_instance_uid",
    "sha256",
)
ROBUSTNESS_MEASURED_OUTCOME = "measured"
# A source ROI that the run actually selected exists and is non-volumetric.
ROBUSTNESS_SOURCE_ONLY_OUTCOME = "source_only_nonvolumetric"
# Nothing in the sources matched the requested selection, or nothing was
# requested. That is an accounting fact about the request; it says nothing
# about the anatomy of a name that was never found, and it is not a clinical
# exclusion of anything.
ROBUSTNESS_UNMATCHED_SELECTION_OUTCOME = "selection_matched_no_source_structure"
ROBUSTNESS_NONMEASURED_OUTCOMES = frozenset(
    {ROBUSTNESS_SOURCE_ONLY_OUTCOME, ROBUSTNESS_UNMATCHED_SELECTION_OUTCOME}
)
ROBUSTNESS_SOURCE_DISPOSITION_OUTCOMES = (
    frozenset({ROBUSTNESS_MEASURED_OUTCOME}) | ROBUSTNESS_NONMEASURED_OUTCOMES
)

# Modules whose bytes decide which source ROIs become dispositions rather than
# measurements, plus the modules that directly execute the measurement of
# every admitted robustness row. Paths are relative to the package directory.
#
# This identity is the only code binding re-checked on every admission of a
# robustness artifact (course reuse, aggregation input, manifest cohort
# admission), so it must cover each module that computes or selects a measured
# value on the robustness path, not only the disposition deciders:
#
# - radiomics_parallel.py: the isolated per-condition extractor that produces
#   the perturbation rows in the default parallel mode.
# - radiomics_ct_contract.py: CT ROI arm classification and per-arm extraction,
#   executed in both the parallel and the sequential mode.
# - robustness_mcc.py: the MCC computation installed into PyRadiomics inside
#   the extraction worker.
# - radiomics_conda.py: the helper-environment batch extraction used by the
#   sequential path when PyRadiomics is not importable natively.
#
# Deliberately excluded: robustness_watchdog.py (progress instrumentation and
# supervision; it returns extractor results unchanged and its technical-row
# typing only describes runs that never publish a sidecar) and
# radiomics_resource_guard.py (admission/failure classification that produces
# no admitted value). This list binds on-disk content only; it does not prove
# which bytes the interpreter executed.
ROBUSTNESS_DISPOSITION_CODE_SOURCES = (
    "radiomics_robustness.py",
    "radiomics_robustness_outcomes.py",
    "radiomics.py",
    "rtstruct_geometry.py",
    "radiomics_parallel.py",
    "radiomics_ct_contract.py",
    "robustness_mcc.py",
    "radiomics_conda.py",
)


def _current_robustness_code_identity() -> Dict[str, Any]:
    """Content identity of the deciding modules *as they are on disk now*.

    A Git revision alone cannot certify a dirty worktree, so the binding is the
    actual file content. Nothing is memoized: a value cached on first use cannot
    notice a module that changed afterwards, and would let a stale artifact keep
    passing in an interpreter that has since seen the code change.
    """
    package_root = Path(__file__).resolve().parent
    sources = []
    for relative in sorted(set(ROBUSTNESS_DISPOSITION_CODE_SOURCES)):
        source = package_root / relative
        if not source.is_file():
            raise RuntimeError(
                f"robustness disposition code source is absent: {source}"
            )
        sources.append({"path": relative, "sha256": _file_sha256(source)})
    return {
        "pipeline_version": str(_pipeline_version()),
        "sources": sources,
        "sources_sha256": _content_sha256(sources),
    }


def _capture_robustness_code_identity() -> Dict[str, Any]:
    """Capture the deciding modules once, before a run processes anything.

    The captured value is this run's own record. It is re-verified against the
    current on-disk content before anything is published, so a module edited
    while the run was working can never be cited as the code that decided it.

    Assurance and its limit. This binds the *on-disk* content of the deciding
    modules from before processing to publication, and rechecks it on load. It
    does not prove which bytes the interpreter executed: modules already
    imported, reloaded or shadowed can differ from the files on disk. Exact
    executed-code proof needs a frozen execution environment, which is outside
    this module.
    """
    return _current_robustness_code_identity()


def _verify_robustness_code_identity(
    captured: Mapping[str, Any], *, context: str
) -> Dict[str, Any]:
    """Fail closed when the deciding code changed since it was captured."""
    current = _current_robustness_code_identity()
    recorded_digest = str(captured.get("sources_sha256") or "")
    if recorded_digest != current["sources_sha256"]:
        raise RuntimeError(
            f"{context}: the deciding code changed on disk during this run "
            f"(captured {recorded_digest!r}, current "
            f"{current['sources_sha256']!r}); nothing this run produced can "
            "cite code that was not on disk while it ran"
        )
    return current


def _pipeline_version() -> str:
    from . import __version__

    return __version__


def _file_sha256(path: Path) -> str:
    from .stage_completion import file_sha256

    return file_sha256(Path(path))


def _content_sha256(value: Any) -> str:
    from .stage_completion import content_sha256

    return content_sha256(value)


def _bind_configuration_values(value: Any) -> Any:
    """Bind a configuration value to whatever actually governs a disposition.

    ``RobustnessConfig`` declares no path-valued setting today (its fields are
    flags, counts, magnitudes, names and thresholds), but this snapshot also
    accepts foreign configuration objects. For those, a file path is not the
    setting: two runs can name the same file and be governed by different bytes.
    A path is therefore recorded together with the content digest of the file it
    currently names, and an absent or non-file path is recorded as such rather
    than passing as content. Any other non-JSON value keeps its qualified type
    so that two different objects cannot collapse into one string.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _bind_configuration_values(item)
            for key, item in value.items()
        }
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_bind_configuration_values(item) for item in value), key=repr
        )
    if isinstance(value, (list, tuple)):
        return [_bind_configuration_values(item) for item in value]
    if isinstance(value, os.PathLike):
        path = Path(value)
        if path.is_file():
            return {
                "__path__": str(path),
                "content_state": "file",
                "sha256": _file_sha256(path),
            }
        return {
            "__path__": str(path),
            "content_state": "directory" if path.is_dir() else "absent",
            "sha256": None,
        }
    return {
        "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def effective_robustness_configuration(
    rob_config: "RobustnessConfig",
    *,
    output_name: str,
) -> Dict[str, Any]:
    """Normalized snapshot of the configuration that governs one robustness run.

    The measured table and the disposition sidecar are only comparable across
    runs when the perturbation grid, thresholds and output identity are the
    same, so the snapshot is bound into the artifact and rechecked on load.
    """
    if is_dataclass(rob_config) and not isinstance(rob_config, type):
        declared = asdict(rob_config)
    else:
        declared = {
            key: value
            for key, value in vars(rob_config).items()
            if not key.startswith("_")
        }
    payload = {
        "measurement_type": ROBUSTNESS_MEASUREMENT_TYPE,
        "output_name": str(output_name),
        "robustness": _bind_configuration_values(declared),
    }
    # Normalize through JSON so the digest is stable for equal configurations
    # expressed with different (but equivalent) container types. No ``default``
    # coercion: a value that only survives as ``str(value)`` would let two
    # different settings share one fingerprint.
    return json.loads(json.dumps(payload, sort_keys=True))


def _rtstruct_source_binding(segmentation_source: str, rtstruct_path: Path) -> Dict[str, str]:
    """Capture RTSTRUCT identity and content *before* its masks are read."""
    from .rtstruct_identity import require_rtstruct_identity

    path = Path(rtstruct_path)
    return {
        "segmentation_source": str(segmentation_source),
        "source_path": str(path),
        "rtstruct_sop_instance_uid": str(require_rtstruct_identity(path)),
        "sha256": _file_sha256(path),
    }


def _verify_rtstruct_source_bindings(
    bindings: List[Dict[str, str]],
    *,
    error: type[Exception],
    context: str,
) -> None:
    """Fail closed when a captured RTSTRUCT source no longer has its bytes."""
    for binding in bindings:
        path = Path(str(binding.get("source_path") or ""))
        if not path.is_file():
            raise error(
                f"{context}: RTSTRUCT source {path} recorded for "
                f"{binding.get('segmentation_source')!r} is no longer readable"
            )
        current = _file_sha256(path)
        if current != str(binding.get("sha256") or ""):
            raise error(
                f"{context}: RTSTRUCT source {path} content changed "
                f"(recorded sha256 {binding.get('sha256')!r}, current {current!r}); "
                "an unchanged SOPInstanceUID does not certify unchanged content"
            )


def _present(value: Any) -> bool:
    """A required identity field is present when it carries readable text."""
    return value is not None and bool(str(value).strip())


def _validate_source_disposition_rows(
    rows: Any,
    *,
    bindings: List[Dict[str, str]],
    error: type[Exception],
) -> List[Dict[str, Any]]:
    """Validate row type, required identity, status and uniqueness."""
    if not isinstance(rows, list):
        raise error("robustness source disposition rows must be a list")
    bound_sources = {
        (
            str(binding.get("segmentation_source")),
            str(binding.get("source_path")),
            str(binding.get("rtstruct_sop_instance_uid")),
        )
        for binding in bindings
    }
    from .rtstruct_geometry import NONVOLUMETRIC_CODES

    seen: set[Tuple[str, ...]] = set()
    validated: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise error(
                f"robustness source disposition row {index} is "
                f"{type(row).__name__}, not a record"
            )
        missing = [
            field_name
            for field_name in ROBUSTNESS_SOURCE_DISPOSITION_ROW_FIELDS
            if not _present(row.get(field_name))
        ]
        if missing:
            raise error(
                f"robustness source disposition row {index} is missing required "
                f"identity: {', '.join(sorted(missing))}"
            )
        status = str(row["status"])
        if status not in ROBUSTNESS_SOURCE_DISPOSITION_STATUSES:
            raise error(
                f"robustness source disposition row {index} has status {status!r}, "
                f"which is not a terminal source disposition "
                f"({', '.join(sorted(ROBUSTNESS_SOURCE_DISPOSITION_STATUSES))})"
            )
        if str(row["structural_code"]) not in NONVOLUMETRIC_CODES:
            raise error(
                f"robustness source disposition row {index} has structural code "
                f"{row['structural_code']!r}, which is not a non-volumetric geometry"
            )
        origin = (
            str(row["segmentation_source"]),
            str(row["source_path"]),
            str(row["rtstruct_sop_instance_uid"]),
        )
        if origin not in bound_sources:
            raise error(
                f"robustness source disposition row {index} cites unbound source "
                f"{origin!r}"
            )
        identity = tuple(
            str(row[field_name])
            for field_name in ROBUSTNESS_SOURCE_DISPOSITION_ROW_IDENTITY
        )
        if identity in seen:
            raise error(
                "duplicate robustness source disposition for "
                f"{identity!r}; one ROI identity has exactly one disposition"
            )
        seen.add(identity)
        validated.append(row)
    return validated


def _write_robustness_source_dispositions(
    course_dir: Path,
    *,
    run_identifier: str,
    rows: List[Dict[str, Any]],
    source_bindings: List[Dict[str, str]],
    effective_configuration: Dict[str, Any],
    code_identity: Mapping[str, Any],
    output_path: Path,
    measured_output: Optional[Path],
    nonmeasured_outcome: str = ROBUSTNESS_SOURCE_ONLY_OUTCOME,
    source_only_basis: Optional[Dict[str, Any]] = None,
) -> Path:
    """Atomically publish the terminal RTSTRUCT source dispositions of one run.

    The sidecar preserves explicit source/ROI/RTSTRUCT identity for structural
    dispositions (e.g. non-volumetric ROIs) observed while robustness collected
    masks. It is distinct from the 81-condition measurement table: it carries no
    voxel or feature values and never claims a measurement. It is published only
    after the run reached a terminal, non-technical outcome, and it is bound to
    the run identifier, the source bytes, the deciding code, the effective
    configuration and the measured output (when one exists).
    """
    for binding in source_bindings:
        missing = [
            field_name
            for field_name in ROBUSTNESS_SOURCE_BINDING_FIELDS
            if not _present(binding.get(field_name))
        ]
        if missing:
            raise RuntimeError(
                "robustness source binding is incomplete: " + ", ".join(sorted(missing))
            )
    # Re-read the sources now: masks were read from these bytes, and a source
    # that changed during the run cannot certify the dispositions it produced.
    _verify_rtstruct_source_bindings(
        source_bindings,
        error=RuntimeError,
        context="refusing to publish robustness source dispositions",
    )
    _validate_source_disposition_rows(
        rows, bindings=source_bindings, error=RuntimeError
    )
    # The code captured before processing must still be the code on disk.
    _verify_robustness_code_identity(
        code_identity,
        context="refusing to publish robustness source dispositions",
    )

    if measured_output is not None:
        outcome = ROBUSTNESS_MEASURED_OUTCOME
    else:
        outcome = str(nonmeasured_outcome)
        if outcome not in ROBUSTNESS_NONMEASURED_OUTCOMES:
            raise RuntimeError(
                f"refusing to publish robustness outcome {outcome!r}, which is "
                "not a terminal non-measured outcome "
                f"({', '.join(sorted(ROBUSTNESS_NONMEASURED_OUTCOMES))})"
            )
        if outcome == ROBUSTNESS_SOURCE_ONLY_OUTCOME and not rows:
            raise RuntimeError(
                "refusing to publish a non-volumetric source-only robustness "
                "outcome with no source disposition to evidence it"
            )
    output_entry: Optional[Dict[str, Any]] = None
    if measured_output is not None:
        measured_output = Path(measured_output)
        if not measured_output.is_file():
            raise RuntimeError(
                "refusing to publish robustness source dispositions: measured "
                f"output {measured_output} is absent"
            )
        output_entry = {
            "path": measured_output.name,
            "sha256": _file_sha256(measured_output),
            "size_bytes": int(measured_output.stat().st_size),
        }
    elif Path(output_path).exists():
        # A source-only outcome measured nothing, so no table may stand for it.
        raise RuntimeError(
            "refusing to publish a source-only robustness disposition beside an "
            f"existing measurement table {output_path}"
        )

    path = Path(course_dir) / "metadata" / ROBUSTNESS_SOURCE_DISPOSITIONS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": ROBUSTNESS_SOURCE_DISPOSITIONS_SCHEMA_VERSION,
        "artifact_kind": ROBUSTNESS_SOURCE_DISPOSITIONS_KIND,
        "patient_id": course_dir.parent.name,
        "course_id": course_dir.name,
        "measurement_type": ROBUSTNESS_MEASUREMENT_TYPE,
        "robustness_run_identifier": str(run_identifier),
        "measurement_outcome": outcome,
        "row_count": len(rows),
        "rows": rows,
        "rows_sha256": _content_sha256(rows),
        "source_bindings": source_bindings,
        "code_identity": dict(code_identity),
        "effective_configuration": effective_configuration,
        "effective_configuration_sha256": _content_sha256(effective_configuration),
        "output_path": Path(output_path).name,
        "measured_output": output_entry,
        "source_only_basis": source_only_basis,
    }
    fd, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    return path


def robustness_source_dispositions_path(course_dir: Path) -> Path:
    return Path(course_dir) / "metadata" / ROBUSTNESS_SOURCE_DISPOSITIONS_FILENAME


def invalidate_robustness_source_dispositions(course_dir: Path) -> None:
    """Drop a previous run's sidecar so it can never describe a later run."""
    robustness_source_dispositions_path(course_dir).unlink(missing_ok=True)


@dataclass(frozen=True)
class RobustnessSourceDispositionInspection:
    """What one disposition sidecar says, and how far that was verified.

    This is an inspection result, not an admission decision. ``rows`` are the
    recorded dispositions after every run, source, code, digest, outcome and
    output check; ``configuration_verified`` says whether they were also
    checked against a caller-supplied current configuration. A result with
    ``configuration_verified`` false has only been shown to be internally
    consistent, and must not be used as if the current run's perturbation grid
    and thresholds had been confirmed.
    """

    course_dir: Path
    run_identifier: str
    measurement_outcome: str
    configuration_verified: bool
    rows: List[Dict[str, Any]]


def _read_robustness_source_dispositions(
    course_dir: Path,
    *,
    run_identifier: str,
    rob_config: Optional["RobustnessConfig"],
    output_name: Optional[str],
    snapshot: Optional[bytes] = None,
) -> RobustnessSourceDispositionInspection:
    """Read the run-bound disposition sidecar, rejecting stale or foreign bytes.

    Every binding except the current configuration is checked unconditionally.
    ``rob_config``, when supplied, additionally requires the recorded effective
    configuration to equal the one the caller is acting under, which detects a
    sidecar produced under a different perturbation grid or thresholds.
    ``output_name`` defaults to the recorded output name.
    """
    from .course_manifest import require_no_output_symlinks

    course_dir = Path(course_dir)
    path = robustness_source_dispositions_path(course_dir)
    require_no_output_symlinks(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no robustness source dispositions artifact in {course_dir}"
        )
    payload = json.loads(path.read_bytes() if snapshot is None else snapshot)
    if not isinstance(payload, dict):
        raise ValueError(
            "robustness source dispositions artifact is not a record: "
            f"{type(payload).__name__}"
        )
    if payload.get("artifact_kind") != ROBUSTNESS_SOURCE_DISPOSITIONS_KIND:
        raise ValueError(
            "unexpected robustness source dispositions artifact kind: "
            f"{payload.get('artifact_kind')!r}"
        )
    version = payload.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError(
            "robustness source dispositions schema version must be an integer, "
            f"not {version!r}"
        )
    if version != ROBUSTNESS_SOURCE_DISPOSITIONS_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported robustness source dispositions schema version {version!r}; "
            f"this build reads version {ROBUSTNESS_SOURCE_DISPOSITIONS_SCHEMA_VERSION}"
        )
    if (
        payload.get("patient_id") != course_dir.parent.name
        or payload.get("course_id") != course_dir.name
    ):
        raise ValueError(
            "robustness source dispositions artifact does not belong to this course"
        )
    if payload.get("measurement_type") != ROBUSTNESS_MEASUREMENT_TYPE:
        raise ValueError(
            "robustness source dispositions artifact has measurement type "
            f"{payload.get('measurement_type')!r}"
        )
    if str(payload.get("robustness_run_identifier")) != str(run_identifier):
        raise ValueError(
            "stale robustness source dispositions artifact: run identifier "
            f"{payload.get('robustness_run_identifier')!r} does not match "
            f"current run {run_identifier!r}"
        )

    bindings = payload.get("source_bindings")
    if not isinstance(bindings, list) or not all(
        isinstance(binding, dict) for binding in bindings
    ):
        raise ValueError(
            "robustness source dispositions artifact has no readable source bindings"
        )
    for binding in bindings:
        missing = [
            field_name
            for field_name in ROBUSTNESS_SOURCE_BINDING_FIELDS
            if not _present(binding.get(field_name))
        ]
        if missing:
            raise ValueError(
                "robustness source binding is incomplete: " + ", ".join(sorted(missing))
            )
    _verify_rtstruct_source_bindings(
        bindings,
        error=ValueError,
        context="robustness source dispositions do not describe the current sources",
    )

    rows = payload.get("rows")
    if not isinstance(rows, list) or payload.get("row_count") != len(rows):
        raise ValueError(
            "corrupt robustness source dispositions artifact: row_count "
            f"{payload.get('row_count')!r} does not match the recorded rows"
        )
    _validate_source_disposition_rows(rows, bindings=bindings, error=ValueError)
    recorded_rows_digest = str(payload.get("rows_sha256") or "")
    if recorded_rows_digest != _content_sha256(rows):
        raise ValueError(
            "robustness source disposition rows do not match their recorded digest"
        )

    recorded_code = payload.get("code_identity")
    current_code = _current_robustness_code_identity()
    if isinstance(recorded_code, dict):
        recorded_sources = recorded_code.get("sources")
        if not isinstance(recorded_sources, list) or _content_sha256(
            recorded_sources
        ) != str(recorded_code.get("sources_sha256") or ""):
            raise ValueError(
                "robustness code identity does not match its own per-file digests"
            )
    if not isinstance(recorded_code, dict) or recorded_code.get(
        "sources_sha256"
    ) != current_code["sources_sha256"]:
        raise ValueError(
            "robustness source dispositions were produced by different code "
            f"(recorded {(recorded_code or {}).get('sources_sha256')!r}, current "
            f"{current_code['sources_sha256']!r}); a matching revision does not "
            "certify a dirty worktree"
        )

    recorded_configuration = payload.get("effective_configuration")
    if not isinstance(recorded_configuration, dict):
        raise ValueError(
            "robustness source dispositions artifact records no effective configuration"
        )
    if str(payload.get("effective_configuration_sha256") or "") != _content_sha256(
        recorded_configuration
    ):
        raise ValueError(
            "robustness effective configuration does not match its recorded digest"
        )
    configuration_verified = False
    if rob_config is not None:
        expected = effective_robustness_configuration(
            rob_config,
            output_name=str(
                output_name
                or recorded_configuration.get("output_name")
                or payload.get("output_path")
                or ""
            ),
        )
        if _content_sha256(expected) != _content_sha256(recorded_configuration):
            raise ValueError(
                "robustness source dispositions were produced under a different "
                "effective configuration than the one requested"
            )
        configuration_verified = True

    outcome = payload.get("measurement_outcome")
    if outcome not in ROBUSTNESS_SOURCE_DISPOSITION_OUTCOMES:
        raise ValueError(
            f"unknown robustness measurement outcome {outcome!r}"
        )
    # A source-only completion is admissible only when this same canonical
    # sidecar carries non-empty, explicitly non-volumetric dispositions.  Do
    # not let an integrity-valid zero-row or unrelated-outcome sidecar certify
    # a course merely because its receipt names the source artifact.
    if outcome == ROBUSTNESS_SOURCE_ONLY_OUTCOME:
        if not rows:
            raise ValueError(
                "source-only robustness dispositions must contain at least one "
                "disposition row"
            )
    if outcome in ROBUSTNESS_NONMEASURED_OUTCOMES:
        from fnmatch import fnmatch

        selection = recorded_configuration.get("robustness", {}).get(
            "perturbation", {}
        ).get("apply_to_structures")
        if not isinstance(selection, list) or not all(isinstance(p, str) for p in selection):
            raise ValueError("robustness dispositions record no readable requested selection")
        matched = sum(
            any(fnmatch(str(row["roi_name"]).upper(), pattern.upper()) for pattern in selection)
            for row in rows
        )
        if outcome == ROBUSTNESS_SOURCE_ONLY_OUTCOME and not matched:
            raise ValueError(
                "source-only robustness selection matched no source disposition; "
                "an unmatched request is not non-volumetric anatomy"
            )
        basis = payload.get("source_only_basis")
        if basis is not None:
            if not isinstance(basis, dict):
                raise ValueError("robustness source-only basis must be a record")
            expected_basis = {
                "source_disposition_count": len(rows),
                "selection_matched_nonvolumetric_count": matched,
                "requested_selection": sorted(selection),
            }
            for key, expected_value in expected_basis.items():
                if key in basis and (
                    basis[key] != expected_value
                    or (isinstance(expected_value, int) and type(basis[key]) is not int)
                ):
                    raise ValueError(f"robustness source-only basis {key} does not reconcile")
    recorded_output_name = str(payload.get("output_path") or "").strip()
    if not recorded_output_name:
        raise ValueError(
            "robustness source dispositions artifact records no output identity"
        )
    if Path(recorded_output_name).name != recorded_output_name or recorded_output_name in {".", ".."}:
        raise ValueError("robustness output identity must be a plain filename")
    if output_name is not None and recorded_output_name != output_name:
        raise ValueError("robustness sidecar output identity differs from requested output")
    require_no_output_symlinks(course_dir / recorded_output_name)
    measured_output = payload.get("measured_output")
    if outcome == ROBUSTNESS_MEASURED_OUTCOME:
        if not isinstance(measured_output, dict):
            raise ValueError(
                "measured robustness dispositions record no measured output binding"
            )
        output_path = course_dir / str(measured_output.get("path") or "")
        if str(measured_output.get("path") or "") != recorded_output_name:
            raise ValueError(
                "robustness measured output binding does not match the recorded "
                f"output identity {recorded_output_name!r}"
            )
        if not output_path.is_file():
            raise ValueError(
                f"measured robustness output {output_path} is absent; its source "
                "dispositions no longer describe a published measurement"
            )
        current_digest = _file_sha256(output_path)
        if current_digest != str(measured_output.get("sha256") or ""):
            raise ValueError(
                f"measured robustness output {output_path} changed after publication "
                f"(recorded sha256 {measured_output.get('sha256')!r}, current "
                f"{current_digest!r})"
            )
    else:
        if measured_output is not None:
            raise ValueError(
                "source-only robustness dispositions must not bind a measured output"
            )
        if (course_dir / recorded_output_name).exists():
            raise ValueError(
                "source-only robustness dispositions stand beside an existing "
                f"measurement table {course_dir / recorded_output_name}"
            )
    return RobustnessSourceDispositionInspection(
        course_dir=course_dir,
        run_identifier=str(run_identifier),
        measurement_outcome=str(outcome),
        configuration_verified=configuration_verified,
        rows=list(rows),
    )


def inspect_robustness_source_dispositions(
    course_dir: Path,
    *,
    run_identifier: str,
    rob_config: Optional["RobustnessConfig"] = None,
    output_name: Optional[str] = None,
) -> RobustnessSourceDispositionInspection:
    """Inspect a disposition sidecar for integrity, without admitting its rows.

    Use this to look at an artifact (diagnostics, audits, reporting) when the
    caller has no current configuration to check it against. The typed result
    states how far verification went, so an integrity-only read can never be
    mistaken for the validated admission :func:`load_robustness_source_dispositions`
    performs.
    """
    return _read_robustness_source_dispositions(
        course_dir,
        run_identifier=run_identifier,
        rob_config=rob_config,
        output_name=output_name,
    )


def load_robustness_source_dispositions(
    course_dir: Path,
    *,
    run_identifier: str,
    rob_config: "RobustnessConfig",
    output_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Admit a run's source dispositions into the caller's current context.

    ``rob_config`` is required. Admission means "these dispositions describe the
    run I am acting under", and that cannot be established without the
    configuration the caller is acting under: identical rows read under a
    different perturbation grid or thresholds are not the same evidence. Callers
    that only want to check an artifact's integrity use
    :func:`inspect_robustness_source_dispositions`, whose typed result cannot be
    confused with an admitted row list.
    """
    if rob_config is None:
        raise TypeError(
            "load_robustness_source_dispositions() requires the current "
            "rob_config; use inspect_robustness_source_dispositions() for an "
            "explicitly integrity-only read"
        )
    inspection = _read_robustness_source_dispositions(
        course_dir,
        run_identifier=run_identifier,
        rob_config=rob_config,
        output_name=output_name,
    )
    if not inspection.configuration_verified:
        raise RuntimeError(
            "robustness source dispositions were not verified against the "
            "current configuration"
        )
    return inspection.rows


# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass
class PerturbationConfig:
    """Configuration for mask perturbation (RTpipeline-adapted NTCV chain: Noise + Translation + Contour + Volume).

    Note: The original Zwanenburg 2019 framework uses N/T/R/V/C including Rotation.
    Rotation is available as a separate sensitivity analysis via rotation_angles.
    """
    apply_to_structures: List[str] = field(
        default_factory=lambda: list(DEFAULT_ROI_FAMILY_NAMES)
    )
    small_volume_changes: List[float] = field(default_factory=lambda: [-0.15, 0.0, 0.15])
    large_volume_changes: List[float] = field(default_factory=lambda: [-0.30, 0.0, 0.30])
    n_random_contour_realizations: int = 2
    max_translation_mm: float = 4.0
    contour_randomization_mm: float = 0.0  # C8 fix: independent contour noise (0 = auto from max_translation_mm/2 for backwards compat)
    noise_levels: List[float] = field(default_factory=lambda: [0.0, 10.0, 20.0])  # Gaussian noise std dev in HU
    intensity: str = "standard"  # "mild", "standard", "aggressive" - controls perturbation count
    rotation_angles: List[float] = field(default_factory=list)  # Rotation sensitivity: e.g., [1, -1, 3, -3] degrees


@dataclass
class ICCConfig:
    """Configuration for ICC computation."""
    implementation: Literal["pingouin", "manual"] = "pingouin"
    icc_type: Literal["ICC1", "ICC2", "ICC3"] = "ICC3"
    ci: bool = True


@dataclass
class MetricsConfig:
    """Configuration for robustness metrics."""
    icc: ICCConfig = field(default_factory=ICCConfig)
    cov_enabled: bool = True
    qcd_enabled: bool = True


@dataclass
class RobustnessThresholds:
    """Thresholds for classifying feature robustness."""
    icc_robust: float = 0.90
    icc_acceptable: float = 0.75
    cov_robust_pct: float = 10.0
    cov_acceptable_pct: float = 20.0


@dataclass
class RobustnessConfig:
    """Main configuration for radiomics robustness analysis."""
    enabled: bool = True
    modes: List[str] = field(default_factory=lambda: ["segmentation_perturbation"])
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    thresholds: RobustnessThresholds = field(default_factory=RobustnessThresholds)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobustnessConfig":
        """Load configuration from dictionary (parsed YAML)."""
        if not data:
            return cls()

        enabled = data.get("enabled", False)
        modes = data.get("modes", ["segmentation_perturbation"])
        if isinstance(modes, str):
            modes = [modes]

        # Perturbation config
        pert_data = data.get("segmentation_perturbation", {})
        perturbation = PerturbationConfig(
            apply_to_structures=pert_data.get(
                "apply_to_structures", list(DEFAULT_ROI_FAMILY_NAMES)
            ),
            small_volume_changes=pert_data.get("small_volume_changes", [-0.15, 0.0, 0.15]),
            large_volume_changes=pert_data.get("large_volume_changes", [-0.30, 0.0, 0.30]),
            n_random_contour_realizations=pert_data.get("n_random_contour_realizations", 2),
            max_translation_mm=pert_data.get("max_translation_mm", 4.0),
            contour_randomization_mm=pert_data.get("contour_randomization_mm", 0.0),
            noise_levels=pert_data.get("noise_levels", [0.0, 10.0, 20.0]),
            intensity=pert_data.get("intensity", "standard"),
        )

        # Metrics config
        metrics_data = data.get("metrics", {})
        icc_data = metrics_data.get("icc", {})
        icc_config = ICCConfig(
            implementation=icc_data.get("implementation", "pingouin"),
            icc_type=icc_data.get("icc_type", "ICC3"),
            ci=icc_data.get("ci", True),
        )
        metrics = MetricsConfig(
            icc=icc_config,
            cov_enabled=metrics_data.get("cov", {}).get("enabled", True),
            qcd_enabled=metrics_data.get("qcd", {}).get("enabled", True),
        )

        # Thresholds
        thresh_data = data.get("thresholds", {})
        thresholds = RobustnessThresholds(
            icc_robust=thresh_data.get("icc", {}).get("robust", 0.90),
            icc_acceptable=thresh_data.get("icc", {}).get("acceptable", 0.75),
            cov_robust_pct=thresh_data.get("cov", {}).get("robust_pct", 10.0),
            cov_acceptable_pct=thresh_data.get("cov", {}).get("acceptable_pct", 20.0),
        )

        return cls(
            enabled=enabled,
            modes=modes,
            perturbation=perturbation,
            metrics=metrics,
            thresholds=thresholds,
        )


from .radiomics_robustness_outcomes import (
    GeometricNonmeasurement, GeometricNotExtractable, GEOMETRIC_REASON_CODES,
    volume_nonmeasurement, extraction_nonmeasurement, nonmeasurement_rows,
)


def _select_largest_scores_deterministically(
    candidates: np.ndarray,
    scores: np.ndarray,
    count: int,
) -> np.ndarray:
    """Select exact top-scoring indices with stable flat-index tie breaking."""
    if count <= 0:
        return np.empty(0, dtype=candidates.dtype)
    if count >= scores.size:
        return candidates.copy()
    threshold_index = scores.size - count
    threshold = np.partition(scores, threshold_index)[threshold_index]
    strict = candidates[scores > threshold]
    ties = np.sort(candidates[scores == threshold])
    needed = count - strict.size
    return np.concatenate((strict, ties[:needed]))


def volume_adapt_mask(mask: sitk.Image, tau: float, max_iterations: int = 20) -> Optional[sitk.Image]:
    """Adapt a binary mask to the requested voxel-count volume change.

    Voxels are ranked by a signed Maurer distance map computed in physical
    space. Erosion retains the deepest interior voxels; dilation adds the
    nearest exterior voxels. The returned mask therefore contains exactly the
    rounded target voxel count (and hence the closest representable physical
    volume) without forcing a one-voxel change through thick slices.

    Args:
        mask: Binary SimpleITK image
        tau: Target volume change ratio (e.g., 0.15 for +15%, -0.15 for -15%)
        max_iterations: Retained for API compatibility; no longer used

    Returns:
        Perturbed mask or None if unsuccessful
    """
    _ = max_iterations
    if tau <= -1.0:
        raise ValueError("tau must be greater than -1.0")
    if abs(tau) < 1e-6:
        return sitk.Image(mask)

    arr = sitk.GetArrayFromImage(mask).astype(bool)
    original_voxels = int(arr.sum())
    spacing = mask.GetSpacing()
    voxel_vol_mm3 = float(np.prod(spacing))
    min_voxels = max(10, int(np.ceil(10.0 / voxel_vol_mm3)))
    if original_voxels < min_voxels:
        logger.debug(
            "Mask too small for volume adaptation (volume=%d voxels, min=%d)",
            original_voxels,
            min_voxels,
        )
        return None

    target_voxels = int(round(original_voxels * (1.0 + tau)))
    if target_voxels < min_voxels or target_voxels > arr.size:
        logger.debug(
            "Target volume is outside the representable range (target=%d, range=%d..%d)",
            target_voxels,
            min_voxels,
            arr.size,
        )
        return None
    if target_voxels == original_voxels:
        logger.debug("Requested tau=%.6f rounds to the original voxel count", tau)
        return None

    signed_distance = sitk.SignedMaurerDistanceMap(
        sitk.Cast(mask > 0, sitk.sitkUInt8),
        insideIsPositive=True,
        squaredDistance=False,
        useImageSpacing=True,
    )
    distance_arr = sitk.GetArrayViewFromImage(signed_distance)
    result_arr = np.zeros_like(arr, dtype=bool)

    if target_voxels < original_voxels:
        candidates = np.flatnonzero(arr)
        scores = distance_arr.ravel()[candidates]
        selected = _select_largest_scores_deterministically(
            candidates, scores, target_voxels
        )
        result_arr.ravel()[selected] = True
    else:
        result_arr[...] = arr
        add_voxels = target_voxels - original_voxels
        candidates = np.flatnonzero(~arr)
        scores = distance_arr.ravel()[candidates]
        selected = _select_largest_scores_deterministically(
            candidates, scores, add_voxels
        )
        result_arr.ravel()[selected] = True

    achieved_voxels = int(result_arr.sum())
    if achieved_voxels != target_voxels:
        raise RuntimeError(
            f"volume adaptation produced {achieved_voxels} voxels; expected {target_voxels}"
        )

    result = sitk.GetImageFromArray(result_arr.astype(np.uint8))
    result.CopyInformation(mask)
    achieved_tau = (achieved_voxels - original_voxels) / original_voxels
    logger.debug(
        "Volume adaptation: target tau=%.3f, achieved tau=%.6f (voxels %d->%d)",
        tau,
        achieved_tau,
        original_voxels,
        achieved_voxels,
    )
    return result


def translate_mask(mask: sitk.Image, translation_mm: Tuple[float, float, float]) -> sitk.Image:
    """
    Apply rigid translation to mask.
    
    Args:
        mask: Binary SimpleITK image
        translation_mm: Translation vector in mm (x, y, z)
        
    Returns:
        Translated mask
    """
    # SimpleITK resampling transforms map output points back into the input
    # image. Negating the requested object displacement therefore moves the
    # foreground in the documented physical-space direction.
    transform = sitk.TranslationTransform(3, tuple(-float(v) for v in translation_mm))
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetReferenceImage(mask)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    
    return resampler.Execute(mask)


def _validate_translated_mask(
    original: sitk.Image,
    translated: sitk.Image,
    translation_mm: Tuple[float, float, float],
) -> None:
    """Fail closed when a requested translation is clipped or not realised."""

    geometry_fields = (
        ("size", original.GetSize(), translated.GetSize()),
        ("spacing", original.GetSpacing(), translated.GetSpacing()),
        ("origin", original.GetOrigin(), translated.GetOrigin()),
        ("direction", original.GetDirection(), translated.GetDirection()),
    )
    for field, expected, observed in geometry_fields:
        if tuple(observed) != tuple(expected):
            raise RuntimeError(
                f"translated mask changed {field}: expected={expected}, observed={observed}"
            )

    original_binary = sitk.Cast(original > 0, sitk.sitkUInt8)
    translated_binary = sitk.Cast(translated > 0, sitk.sitkUInt8)
    original_voxels = int(np.count_nonzero(sitk.GetArrayViewFromImage(original_binary)))
    translated_voxels = int(np.count_nonzero(sitk.GetArrayViewFromImage(translated_binary)))
    if original_voxels == 0:
        raise RuntimeError("cannot translate an empty mask")
    if translated_voxels < original_voxels:
        raise GeometricNotExtractable(GeometricNonmeasurement(
            "translation_outside_image", {
                "original_voxels": original_voxels, "translated_voxels": translated_voxels,
                "translation_mm": list(translation_mm), "image_size": list(original.GetSize()),
                "detail": "translation clipped foreground at image boundary",
            },
        ))
    if translated_voxels != original_voxels:
        raise RuntimeError(
            "translation clipped or duplicated foreground voxels at the image boundary: "
            f"{original_voxels}->{translated_voxels}"
        )

    component_filter = sitk.ConnectedComponentImageFilter()
    component_filter.Execute(original_binary)
    original_components = int(component_filter.GetObjectCount())
    component_filter.Execute(translated_binary)
    translated_components = int(component_filter.GetObjectCount())
    if translated_components != original_components:
        raise RuntimeError(
            "translation changed mask topology: "
            f"components {original_components}->{translated_components}"
        )

    shape = sitk.LabelShapeStatisticsImageFilter()
    shape.Execute(original_binary)
    original_centroid = np.asarray(shape.GetCentroid(1), dtype=float)
    shape.Execute(translated_binary)
    translated_centroid = np.asarray(shape.GetCentroid(1), dtype=float)
    requested = np.asarray(translation_mm, dtype=float)
    displacement_error = np.linalg.norm(
        (translated_centroid - original_centroid) - requested
    )
    # Nearest-neighbour resampling can quantise a physical shift by at most
    # half of one voxel diagonal. Anything larger indicates a sign, geometry,
    # or boundary failure rather than ordinary discretisation.
    tolerance_mm = 0.5 * np.linalg.norm(np.asarray(original.GetSpacing(), dtype=float))
    if displacement_error > tolerance_mm + 1e-6:
        raise RuntimeError(
            "requested mask translation was not realised within grid resolution: "
            f"requested={tuple(float(v) for v in requested)}, "
            f"observed={tuple(float(v) for v in translated_centroid - original_centroid)}, "
            f"tolerance_mm={tolerance_mm:.6g}"
        )


def randomize_contour(
    mask: sitk.Image,
    randomization_mm: float,
    rng: Optional[np.random.Generator] = None,
) -> sitk.Image:
    """
    Apply a reproducible random physical-space boundary offset.

    A signed distance map avoids the anisotropic-voxel error caused by forcing
    a one-voxel kernel in every dimension. Each realization randomly chooses an
    inward or outward displacement between 50% and 100% of the configured
    maximum.

    Args:
        mask: Binary SimpleITK image
        randomization_mm: Maximum boundary displacement in mm
        rng: Optional numpy Generator for reproducible randomization (avoids global state)

    Returns:
        Mask with randomized contour
    """
    if rng is None:
        rng = np.random.default_rng()

    if randomization_mm <= 0:
        raise ValueError("randomization_mm must be positive")

    binary = sitk.Cast(mask > 0, sitk.sitkUInt8)
    signed_distance = sitk.SignedMaurerDistanceMap(
        binary,
        insideIsPositive=True,
        squaredDistance=False,
        useImageSpacing=True,
    )
    magnitude_mm = float(rng.uniform(0.5 * randomization_mm, randomization_mm))
    if rng.random() < 0.5:
        result = signed_distance > magnitude_mm
    else:
        result = signed_distance >= -magnitude_mm
    return sitk.Cast(result, sitk.sitkUInt8)


def add_noise_to_image(
    image: sitk.Image,
    noise_std_hu: float,
    rng: Optional[np.random.Generator] = None,
) -> sitk.Image:
    """
    Add Gaussian noise to image for image-based perturbation testing.

    Args:
        image: CT/MR image
        noise_std_hu: Standard deviation of Gaussian noise in HU
        rng: Optional numpy Generator for reproducible noise (avoids global state)

    Returns:
        Noisy image
    """
    if noise_std_hu <= 0:
        return image

    if rng is None:
        rng = np.random.default_rng()

    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    noise = rng.normal(0, noise_std_hu, arr.shape).astype(np.float32)
    noisy_arr = arr + noise
    
    noisy_img = sitk.GetImageFromArray(noisy_arr)
    noisy_img.CopyInformation(image)
    
    return noisy_img


def generate_perturbed_masks(
    original_mask: sitk.Image,
    volume_changes: List[float],
    structure_name: str,
) -> Dict[str, sitk.Image]:
    """
    Generate multiple perturbed versions of a mask via volume adaptation.

    Args:
        original_mask: Original binary mask
        volume_changes: List of target volume change ratios (e.g., [-0.15, 0, 0.15])
        structure_name: Structure name for logging

    Returns:
        Dictionary mapping perturbation_id to perturbed mask
    """
    perturbed = {}

    for tau in volume_changes:
        pert_id = f"tau{tau:+.2f}".replace(".", "p").replace("+", "plus").replace("-", "minus")

        if abs(tau) < 1e-6:
            # Original mask
            perturbed[pert_id] = original_mask
        else:
            pert_mask = volume_adapt_mask(original_mask, tau)
            if pert_mask is not None:
                perturbed[pert_id] = pert_mask
            else:
                perturbed[pert_id] = volume_nonmeasurement(original_mask, tau)

    return perturbed


def expected_ntcv_perturbation_count(config: PerturbationConfig) -> int:
    """Return the exact Cartesian-grid size for an NTCV configuration."""
    if config.intensity == "mild":
        volume_count = len(
            config.small_volume_changes[:2]
            if len(config.small_volume_changes) > 2
            else config.small_volume_changes
        )
        translation_count = 3 if config.max_translation_mm > 0 else 1
        contour_count = 1 + min(1, config.n_random_contour_realizations)
        noise_count = len(
            config.noise_levels[:1]
            if len(config.noise_levels) > 1
            else config.noise_levels
        )
    elif config.intensity == "aggressive":
        volume_count = len(
            dict.fromkeys(config.large_volume_changes + config.small_volume_changes)
        )
        translation_count = 7 if config.max_translation_mm > 0 else 1
        contour_count = 1 + config.n_random_contour_realizations
        noise_count = len(config.noise_levels)
    else:
        volume_count = len(config.small_volume_changes)
        translation_count = 3 if config.max_translation_mm > 0 else 1
        contour_count = 1 + config.n_random_contour_realizations
        noise_count = len(config.noise_levels)
    return volume_count * translation_count * contour_count * noise_count


def generate_ntcv_perturbations(
    original_mask: sitk.Image,
    original_image: sitk.Image,
    config: PerturbationConfig,
    structure_name: str,
) -> Tuple[Dict[str, sitk.Image], Dict[str, sitk.Image]]:
    """
    Generate NTCV (Noise + Translation + Contour + Volume) perturbation chain.

    This is an RTpipeline-specific chain inspired by the perturbation framework
    described by Zwanenburg et al. (2019). It is not an exact reimplementation:
    rotation is separate and contour randomization is morphological. The fixed
    order below defines RTpipeline's comparable perturbation grid.

    Perturbation Order (implementation-defined):
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1. **N (Noise)**: Image intensity noise (Gaussian) - applied first to simulate
       acquisition variability. Applied to the IMAGE, not the mask.

    2. **T (Translation)**: Geometric shifts - simulates inter-observer ROI
       placement variability. Applied BEFORE contour randomization so that
       boundary perturbations are applied to the shifted ROI.

    3. **C (Contour)**: Boundary randomization - simulates segmentation
       uncertainty at ROI edges. Applied AFTER translation to perturb the
       already-shifted boundary.

    4. **V (Volume)**: Erosion/dilation - simulates systematic over/under-
       segmentation. Applied LAST as it represents the final morphological
       adjustment to the perturbed contour.

    Why This Order Matters:
    ~~~~~~~~~~~~~~~~~~~~~~~
    The NTCV order ensures proper propagation of uncertainty sources:
    - Noise affects feature values but not geometry
    - Translation affects geometric position before boundary uncertainty
    - Contour randomization adds edge uncertainty to translated position
    - Volume adaptation is the final morphological adjustment

    Reversing or randomizing the order would produce a different perturbation
    grid and prevent direct comparison with RTpipeline results generated under
    this contract.

    References:
        Zwanenburg et al. (2019). Assessing robustness of radiomic features by
        image perturbation. Scientific Reports, 9, 614.

    Args:
        original_mask: Original binary mask
        original_image: CT/MR image (for noise perturbations)
        config: Perturbation configuration
        structure_name: Structure name for logging

    Returns:
        Tuple of (perturbed_masks_dict, perturbed_images_dict)
        Both dictionaries map perturbation_id to SimpleITK images.
        The perturbation_id encodes the full chain: "ntcv_n{noise}_t{x}_{y}_{z}_c{idx}_v{pct}"
    """
    perturbed_masks = {}
    perturbed_images = {}
    
    # Determine perturbation count based on intensity level
    if config.intensity == "mild":
        # Minimal testing: 12 perturbations with the shipped NTCV defaults.
        volume_changes = config.small_volume_changes[:2] if len(config.small_volume_changes) > 2 else config.small_volume_changes
        translation_steps = 1 if config.max_translation_mm > 0 else 0
        contour_realizations = min(1, config.n_random_contour_realizations)
        noise_levels = config.noise_levels[:1] if len(config.noise_levels) > 1 else config.noise_levels
    elif config.intensity == "aggressive":
        # Comprehensive stress test: 315 unique perturbations with shipped defaults.
        # Preserve order while avoiding duplicate identifiers such as the zero
        # volume state present in both the large and small grids.
        volume_changes = list(dict.fromkeys(config.large_volume_changes + config.small_volume_changes))
        translation_steps = 2 if config.max_translation_mm > 0 else 0
        contour_realizations = config.n_random_contour_realizations
        noise_levels = config.noise_levels
    else:  # standard
        # Full standard chain: 81 perturbations with the shipped defaults.
        volume_changes = config.small_volume_changes
        translation_steps = 1 if config.max_translation_mm > 0 else 0
        contour_realizations = config.n_random_contour_realizations
        noise_levels = config.noise_levels
    
    # Construct each factor once and then take the strict Cartesian product.
    # In particular, contour draws are keyed only by translation and contour
    # realization, so changing the noise or volume factor cannot silently
    # change the sampled boundary.
    noise_images: List[Tuple[str, sitk.Image]] = []
    for noise_std in noise_levels:
        if noise_std > 0:
            noise_seed = np.random.SeedSequence(
                [42, 0, int(round(float(noise_std) * 1000.0))]
            )
            noise_rng = np.random.Generator(np.random.PCG64(noise_seed))
            noisy_image = add_noise_to_image(original_image, noise_std, rng=noise_rng)
            noise_suffix = f"_n{int(noise_std)}"
        else:
            noisy_image = original_image
            noise_suffix = ""
        noise_images.append((noise_suffix, noisy_image))

    translation_vectors = [(0.0, 0.0, 0.0)]
    if config.max_translation_mm > 0 and translation_steps > 0:
        max_t = float(config.max_translation_mm)
        if translation_steps == 1:
            translation_vectors.extend([(0.0, 0.0, max_t), (0.0, 0.0, -max_t)])
        else:
            translation_vectors.extend([
                (max_t, 0.0, 0.0), (-max_t, 0.0, 0.0),
                (0.0, max_t, 0.0), (0.0, -max_t, 0.0),
                (0.0, 0.0, max_t), (0.0, 0.0, -max_t),
            ])

    geometry_states: List[Tuple[str, sitk.Image]] = []
    contour_noise_mm = (
        config.contour_randomization_mm
        if config.contour_randomization_mm > 0
        else config.max_translation_mm / 2
    )
    for translation_index, trans_vec in enumerate(translation_vectors):
        if any(abs(t) > 1e-3 for t in trans_vec):
            translated_mask = translate_mask(original_mask, trans_vec)
            translation_outcome = None
            try:
                _validate_translated_mask(original_mask, translated_mask, trans_vec)
            except GeometricNotExtractable as exc:
                translation_outcome = exc.outcome
            trans_suffix = (
                f"_t{int(trans_vec[0])}_{int(trans_vec[1])}_{int(trans_vec[2])}"
            )
        else:
            translated_mask = original_mask
            trans_suffix = ""
            translation_outcome = None

        if translation_outcome is not None:
            for contour_index in range(contour_realizations + 1):
                contour_suffix = f"_c{contour_index}" if contour_index else ""
                for tau in volume_changes:
                    vol_suffix = "_v0" if abs(tau) < 1e-6 else f"_v{int(tau * 100):+03d}"
                    geometry_states.append((f"{trans_suffix}{contour_suffix}{vol_suffix}", translation_outcome))
            continue

        contour_variants = [translated_mask]
        for contour_index in range(1, contour_realizations + 1):
            contour_seed = np.random.SeedSequence(
                [42, 1, translation_index, contour_index]
            )
            contour_rng = np.random.Generator(np.random.PCG64(contour_seed))
            randomized = None
            for _attempt in range(16):
                candidate = randomize_contour(
                    translated_mask,
                    contour_noise_mm,
                    rng=contour_rng,
                )
                candidate_arr = sitk.GetArrayViewFromImage(candidate)
                if int(np.count_nonzero(candidate_arr)) < 5:
                    continue
                if any(
                    np.array_equal(
                        candidate_arr,
                        sitk.GetArrayViewFromImage(existing),
                    )
                    for existing in contour_variants
                ):
                    continue
                randomized = candidate
                break
            if randomized is None:
                raise RuntimeError(
                    f"could not generate unique contour realization {contour_index} "
                    f"for {structure_name} after 16 attempts"
                )
            contour_variants.append(randomized)

        for contour_index, contour_mask in enumerate(contour_variants):
            contour_suffix = f"_c{contour_index}" if contour_index > 0 else ""
            for tau in volume_changes:
                if abs(tau) < 1e-6:
                    final_mask = contour_mask
                    vol_suffix = "_v0"
                else:
                    final_mask = volume_adapt_mask(contour_mask, tau)
                    if final_mask is None:
                        final_mask = volume_nonmeasurement(contour_mask, tau)
                    vol_suffix = f"_v{int(tau * 100):+03d}"
                geometry_states.append(
                    (f"{trans_suffix}{contour_suffix}{vol_suffix}", final_mask)
                )

    for noise_suffix, noisy_image in noise_images:
        for geometry_suffix, final_mask in geometry_states:
            pert_id = f"ntcv{noise_suffix}{geometry_suffix}"
            if pert_id in perturbed_masks:
                raise RuntimeError(f"duplicate NTCV perturbation identifier: {pert_id}")
            perturbed_masks[pert_id] = final_mask
            perturbed_images[pert_id] = noisy_image

    pert_count = len(perturbed_masks)
    
    logger.info("Generated %d NTCV perturbations for %s (intensity=%s)",
                pert_count, structure_name, config.intensity)

    expected_count = expected_ntcv_perturbation_count(config)
    if len(perturbed_masks) != expected_count or len(perturbed_images) != expected_count:
        raise RuntimeError(
            f"incomplete NTCV grid for {structure_name}: generated "
            f"{len(perturbed_masks)} of {expected_count} perturbations"
        )

    return perturbed_masks, perturbed_images


# ============================================================================
# Radiomics Feature Extraction
# ============================================================================

def _coerce_scalar_feature_value(key: str, value: Any) -> Optional[float]:
    """Convert scalar PyRadiomics values, including zero-dimensional arrays."""

    if str(key).startswith("diagnostics_"):
        return None
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    if isinstance(value, np.ndarray) and value.size == 1 and np.issubdtype(
        value.dtype, np.number
    ):
        return float(value.item())
    return None


def _is_radiomics_feature_key(key: str) -> bool:
    return any(
        marker in str(key)
        for marker in (
            "_firstorder_", "_shape_", "_shape2D_", "_glcm_", "_glrlm_",
            "_glszm_", "_gldm_", "_ngtdm_",
        )
    )


def _feature_rows_from_worker_result(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Flatten one arm-aware worker result without dropping identity fields."""
    if "__nonmeasurement_rows__" in result:
        return [dict(row, robustness_attempts=result.get("robustness_attempts", 1),
                     robustness_retry_errors=json.dumps(result.get("robustness_retry_errors", [])))
                for row in result["__nonmeasurement_rows__"]]
    rows: List[Dict[str, Any]] = []
    for record in result.get("__records__", []):
        metadata = {
            "modality": record.get("modality", "CT"),
            "robustness_attempts": result.get("robustness_attempts", 1),
            "robustness_retry_errors": json.dumps(result.get("robustness_retry_errors", [])),
            "roi_name": result.get("roi_name", record.get("roi_name", "")),
            "perturbation_id": result.get("perturbation_id", ""),
            "extraction_arm": record.get("extraction_arm", ""),
            "measurement_type": record.get("measurement_type", ""),
            "perturbed_mask_identity": record.get(
                "perturbed_mask_identity", ""
            ),
            "roi_class": record.get("roi_class", ""),
            "roi_class_map_version": record.get("roi_class_map_version", ""),
            "roi_class_map_hash": record.get("roi_class_map_hash", ""),
            "effective_parameter_hash": record.get(
                "effective_parameter_hash", ""
            ),
            "configured_parameter_hash": record.get(
                "configured_parameter_hash", ""
            ),
            "run_identifier": record.get("run_identifier", ""),
        }
        metadata.update(
            {
                column: record.get(column, "")
                for column in ROBUSTNESS_SOURCE_IDENTITY_COLUMNS
            }
        )
        for key, value in record.items():
            if not _is_radiomics_feature_key(str(key)):
                continue
            scalar = _coerce_scalar_feature_value(str(key), value)
            if scalar is not None:
                rows.append(
                    {
                        **metadata,
                        "feature_name": str(key),
                        "value": scalar,
                    }
                )
    return rows


def _validate_extracted_feature_frame(
    frame, expected_perturbation_ids, context, *, expected_source_identity=None,
):
    from .robustness_watchdog import validate_technical_frame
    if "robustness_status" in frame and frame.robustness_status.eq("technical_failure").any():
        failed = frame.loc[frame.robustness_status.eq("technical_failure")]
        remaining = frame.loc[~frame.robustness_status.eq("technical_failure")]
        failed_ids = validate_technical_frame(failed, expected_perturbation_ids, context, expected_source_identity)
        if failed_ids & set(remaining.perturbation_id.astype(str)):
            raise RuntimeError(f"condition is both measured and technically failed for {context}")
        expected_perturbation_ids = set(expected_perturbation_ids) - failed_ids
        if not expected_perturbation_ids:
            if not remaining.empty:
                raise RuntimeError(f"unexpected results for {context}")
            return
        frame = remaining
    return _validate_nontechnical_feature_frame(
        frame, expected_perturbation_ids, context, expected_source_identity=expected_source_identity,
    )


def _validate_nontechnical_feature_frame(
    frame, expected_perturbation_ids, context, *, expected_source_identity=None,
):
    """Reconcile every requested condition with measured or proven impossible rows."""
    if "robustness_status" not in frame:
        return _validate_measured_feature_frame(
            frame, expected_perturbation_ids, context,
            expected_source_identity=expected_source_identity,
        )
    from .radiomics_ct_contract import CT_EXTRACTION_ARMS
    statuses = frame["robustness_status"].fillna("measured")
    if not set(statuses).issubset({"measured", "geometrically_impossible"}):
        raise RuntimeError(f"invalid robustness status for {context}")
    impossible = frame.loc[statuses == "geometrically_impossible"]
    measured = frame.loc[statuses == "measured"]
    impossible_ids = set(impossible["perturbation_id"].astype(str))
    if not impossible_ids <= expected_perturbation_ids:
        raise RuntimeError(f"unexpected impossible conditions for {context}")
    if impossible_ids & set(measured["perturbation_id"].astype(str)):
        raise RuntimeError(f"condition is both measured and impossible for {context}")
    for pid, group in impossible.groupby("perturbation_id"):
        if len(group) != len(CT_EXTRACTION_ARMS) or set(group["extraction_arm"]) != set(CT_EXTRACTION_ARMS):
            raise RuntimeError(f"incomplete non-measurement arms for {context}/{pid}")
        if group["value"].notna().any() or group["feature_name"].notna().any():
            raise RuntimeError(f"non-measurement contains feature values for {context}/{pid}")
        if group["reason_code"].nunique() != 1 or group["geometry_evidence"].nunique() != 1:
            raise RuntimeError(f"discordant non-measurement arms for {context}/{pid}")
        for row in group.to_dict("records"):
            GeometricNonmeasurement(row["reason_code"], json.loads(row["geometry_evidence"]))
            if row.get("measurement_type") != ROBUSTNESS_MEASUREMENT_TYPE:
                raise RuntimeError(f"invalid non-measurement type for {context}/{pid}")
            if expected_source_identity is not None:
                for col, value in expected_source_identity.as_dict().items():
                    if str(row.get(col, "")) != value:
                        raise RuntimeError(f"non-measurement identity mismatch for {context}/{pid}/{col}")
    possible_ids = set(expected_perturbation_ids) - impossible_ids
    if possible_ids:
        _validate_measured_feature_frame(
            measured, possible_ids, context, expected_source_identity=expected_source_identity,
        )
    elif not measured.empty:
        raise RuntimeError(f"unexpected measured conditions for {context}")


def _validate_measured_feature_frame(
    frame: pd.DataFrame,
    expected_perturbation_ids: set[str],
    context: str,
    *,
    expected_source_identity: Optional[RobustnessRoiIdentity] = None,
) -> None:
    """Fail closed when any perturbation or feature extraction is incomplete."""
    if frame.empty:
        raise RuntimeError(f"no radiomics features were extracted for {context}")
    if "extraction_arm" in frame.columns:
        from .radiomics_ct_contract import CT_EXTRACTION_ARMS

        identity_columns = list(ROBUSTNESS_SOURCE_IDENTITY_COLUMNS) + list(
            ROBUSTNESS_PERTURBATION_IDENTITY_COLUMNS
        ) + ["extraction_arm"]
        missing_identity_columns = sorted(set(identity_columns) - set(frame.columns))
        if missing_identity_columns:
            raise RobustnessIdentityError(
                "CT robustness extraction lacks identity columns: "
                + ", ".join(missing_identity_columns)
            )
        normalized_identity = frame[identity_columns].astype(str).apply(
            lambda column: column.str.strip()
        )
        blank_identity = normalized_identity.apply(
            lambda column: column.eq("")
            | column.str.lower().isin({"nan", "none", "<na>"})
        )
        if bool(blank_identity.to_numpy().any()):
            raise RobustnessIdentityError(
                "CT robustness extraction has a blank arm-aware identity field"
            )
        measurement_types = set(normalized_identity["measurement_type"])
        if measurement_types != {ROBUSTNESS_MEASUREMENT_TYPE}:
            raise RobustnessIdentityError(
                "CT robustness rows have invalid measurement_type values: "
                f"{sorted(measurement_types)}"
            )
        invalid_perturbed_identity = ~normalized_identity[
            "perturbed_mask_identity"
        ].str.fullmatch(r"sha256:[0-9a-f]{64}")
        if bool(invalid_perturbed_identity.any()):
            raise RobustnessIdentityError(
                "CT robustness extraction has an invalid perturbed_mask_identity"
            )
        mask_identities_per_perturbation = frame.groupby("perturbation_id")[
            "perturbed_mask_identity"
        ].nunique(dropna=False)
        if bool(mask_identities_per_perturbation.ne(1).any()):
            raise RobustnessIdentityError(
                "CT robustness extraction maps one perturbation to multiple perturbed masks"
            )
        if expected_source_identity is not None:
            for column, expected_value in expected_source_identity.as_dict().items():
                observed = set(normalized_identity[column])
                if observed != {expected_value}:
                    raise RobustnessIdentityError(
                        f"CT robustness {column} disagrees with main radiomics for "
                        f"{context}: observed={sorted(observed)!r}, "
                        f"expected={expected_value!r}"
                    )

        expected_ids: set[Any] = {
            (perturbation_id, arm)
            for perturbation_id in expected_perturbation_ids
            for arm in CT_EXTRACTION_ARMS
        }
        observed_ids: set[Any] = set(
            zip(frame["perturbation_id"].astype(str), frame["extraction_arm"].astype(str))
        )
    else:
        expected_ids = set(expected_perturbation_ids)
        observed_ids = set(frame["perturbation_id"].astype(str))
    missing_ids = sorted(expected_ids - observed_ids)
    unexpected_ids = sorted(observed_ids - expected_ids)
    if missing_ids or unexpected_ids:
        raise RuntimeError(
            f"incomplete radiomics extraction for {context}: "
            f"missing perturbations/arms={missing_ids}, unexpected perturbations/arms={unexpected_ids}"
        )
    if not np.isfinite(frame["value"].to_numpy(dtype=float)).all():
        raise RuntimeError(f"non-finite radiomics values were extracted for {context}")

    group_columns = ["perturbation_id"]
    if "extraction_arm" in frame.columns:
        group_columns.append("extraction_arm")
    feature_sets = {
        tuple(str(value) for value in (key if isinstance(key, tuple) else (key,))):
        frozenset(group["feature_name"].astype(str))
        for key, group in frame.groupby(group_columns)
    }
    if any(not features for features in feature_sets.values()):
        raise RuntimeError(f"no scalar radiomics features were extracted for {context}")
    mismatched = []
    arms = (
        frame["extraction_arm"].astype(str).unique()
        if "extraction_arm" in frame.columns
        else [None]
    )
    for arm in arms:
        arm_sets = {
            key: features
            for key, features in feature_sets.items()
            if arm is None or key[-1] == arm
        }
        reference_features = next(iter(arm_sets.values()))
        mismatched.extend(
            key for key, features in arm_sets.items() if features != reference_features
        )
    if mismatched:
        raise RuntimeError(
            f"feature columns differ across perturbations for {context}: {sorted(mismatched)}"
        )


def extract_features_for_masks(
    image: sitk.Image,
    masks: Dict[str, sitk.Image],
    config: PipelineConfig,
    modality: str = "CT",
    structure_name: str = "",
    patient_id: str = "",
    course_id: str = "",
    perturbed_images: Optional[Dict[str, sitk.Image]] = None,
    segmentation_source: str = "Manual",
    run_identifier: Optional[str] = None,
    source_identity: Optional[RobustnessRoiIdentity] = None,
) -> pd.DataFrame:
    """
    Extract radiomics features for multiple mask variants.

    Args:
        image: CT/MR image (base image)
        masks: Dictionary of {perturbation_id: mask}
        config: Pipeline configuration
        modality: "CT" or "MR"
        structure_name: ROI name
        patient_id: Patient identifier
        course_id: Course identifier
        perturbed_images: Optional dictionary of {perturbation_id: perturbed_image} for noise perturbations
        source_identity: Original CT ROI identity from ``radiomics_ct.parquet``

    Returns:
        Tidy DataFrame with columns [patient_id, course_id, structure, perturbation_id, feature_name, value]
    """
    from .radiomics import _extractor, _get_params_file

    rows = []

    if modality == "CT" and source_identity is None:
        raise RuntimeError(
            f"CT robustness identity is unavailable for "
            f"{segmentation_source}/{structure_name}"
        )

    # Check once whether we need the conda fallback
    ext_probe = _extractor(config, modality)
    use_conda = ext_probe is None

    if modality == "CT":
        identity_failures: Dict[str, str] = {}
        from .radiomics_ct_contract import (
            CT_EXTRACTION_ARMS,
            PRIMARY_ARM,
            classify_ct_roi,
            configured_parameter_hash,
            current_code_revision,
            extract_ct_roi_arms,
            load_custom_structure_provenance,
            new_run_identifier,
        )

        assert source_identity is not None
        expected_context = {
            "patient_id": patient_id,
            "course_id": course_id,
            "segmentation_source": segmentation_source,
            "roi_original_name": structure_name,
        }
        mismatches = [
            f"{column}={getattr(source_identity, column)!r} expected {value!r}"
            for column, value in expected_context.items()
            if getattr(source_identity, column) != str(value)
        ]
        if mismatches:
            raise RuntimeError(
                "CT robustness source identity does not match extraction context: "
                + "; ".join(mismatches)
            )
        perturbed_mask_identities = {
            perturbation_id: _perturbed_mask_identity(mask)
            for perturbation_id, mask in masks.items()
        }

        params_path = _get_params_file(config, "CT")
        custom_path = getattr(config, "custom_structures_config", None)
        decision = classify_ct_roi(
            segmentation_source,
            structure_name,
            custom_provenance=(
                load_custom_structure_provenance(Path(custom_path))
                if segmentation_source == "Custom" and custom_path
                else None
            ),
        )
        configured_hashes = {
            arm: configured_parameter_hash(
                params_path,
                arm=arm,
                window=(decision.primary_resegment_range_hu if arm == PRIMARY_ARM else None),
                large_roi=False,
            )
            for arm in CT_EXTRACTION_ARMS
        }
        shared_run_id = run_identifier or new_run_identifier()

        def _append_records(records: List[Dict[str, Any]], perturbation_id: str) -> None:
            for record in records:
                for column, expected_value in source_identity.as_dict().items():
                    observed_value = str(record.get(column) or "").strip()
                    if observed_value != expected_value:
                        raise RobustnessIdentityError(
                            f"CT robustness worker {column} disagrees with main "
                            f"radiomics for {segmentation_source}/{structure_name}: "
                            f"observed={observed_value!r}, expected={expected_value!r}"
                        )
                metadata = {
                    **source_identity.as_dict(),
                    "modality": "CT",
                    "structure": structure_name,
                    "perturbation_id": perturbation_id,
                    "measurement_type": ROBUSTNESS_MEASUREMENT_TYPE,
                    "perturbed_mask_identity": perturbed_mask_identities[
                        perturbation_id
                    ],
                    "extraction_arm": record.get("extraction_arm", ""),
                    "roi_class": record.get("roi_class", ""),
                    "roi_map_version": record.get("roi_map_version", ""),
                    "roi_map_hash": record.get("roi_map_hash", ""),
                    "effective_parameter_hash": record.get("effective_parameter_hash", ""),
                    "configured_parameter_hash": record.get("configured_parameter_hash", ""),
                    "run_identifier": record.get("run_identifier", ""),
                }
                for key, value in record.items():
                    if not _is_radiomics_feature_key(str(key)):
                        continue
                    scalar = _coerce_scalar_feature_value(str(key), value)
                    if scalar is not None:
                        rows.append({**metadata, "feature_name": str(key), "value": scalar})

        if use_conda:
            from .radiomics_conda import extract_radiomics_batch_with_conda, check_radiomics_env
            if not check_radiomics_env():
                raise RuntimeError(
                    "radiomics conda environment is unavailable; robustness extraction cannot continue"
                )
            tmp_dir = tempfile.mkdtemp(prefix="rtpipe_robust_batch_")
            try:
                batch_tasks = []
                for perturbation_id, mask in masks.items():
                    current_image = (
                        perturbed_images.get(perturbation_id, image)
                        if perturbed_images else image
                    )
                    image_path = Path(tmp_dir) / f"image_{len(batch_tasks)}.nrrd"
                    mask_path = Path(tmp_dir) / f"mask_{len(batch_tasks)}.nrrd"
                    sitk.WriteImage(current_image, str(image_path))
                    sitk.WriteImage(sitk.Cast(mask, sitk.sitkUInt8), str(mask_path))
                    batch_tasks.append({
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "roi_name": f"{structure_name}/{perturbation_id}",
                        "robustness_perturbation_id": perturbation_id,
                        "dual_arm_ct": True,
                        "roi_class_decision": {
                            "roi_class": decision.roi_class,
                            "map_version": decision.map_version,
                            "map_hash": decision.map_hash,
                            "map_entry_source": decision.map_entry_source,
                            "adjudication_status": decision.adjudication_status,
                            "primary_resegment_range_hu": decision.primary_resegment_range_hu,
                            "primary_intensity_texture_disposition": decision.primary_intensity_texture_disposition,
                            "feature_publication_policy": decision.feature_publication_policy,
                        },
                        "metadata": {
                            **source_identity.as_dict(),
                            "roi_name": structure_name,
                            "modality": "CT",
                        },
                        "run_identifier": shared_run_id,
                        "code_revision": current_code_revision(),
                        "native_voxel_count": int(np.count_nonzero(sitk.GetArrayViewFromImage(mask))),
                        "required": False,
                        "configured_parameter_hashes": configured_hashes,
                    })
                batch_results = extract_radiomics_batch_with_conda(
                    batch_tasks,
                    str(params_path) if params_path else None,
                    timeout_per_roi=120,
                )
                indexed_results = {}
                for result in batch_results:
                    if not isinstance(result, dict):
                        raise RuntimeError("robustness conda worker returned no result")
                    index = result.get("__task_index__")
                    if not isinstance(index, int) or index in indexed_results:
                        raise RuntimeError("robustness conda result has missing or duplicate task identity")
                    indexed_results[index] = result
                if set(indexed_results) != set(range(len(batch_tasks))):
                    raise RuntimeError("incomplete robustness conda task results")
                for index, task in enumerate(batch_tasks):
                    result = indexed_results[index]
                    if result.get("__status__") != "success":
                        raise RuntimeError(
                            f"robustness extraction failed for {task['roi_name']}: "
                            f"{result.get('__error__', result.get('__reason__', 'unknown error'))}"
                        )
                    perturbation_id = task["robustness_perturbation_id"]
                    if "__nonmeasurement__" in result:
                        outcome = GeometricNonmeasurement(**result["__nonmeasurement__"])
                        rows.extend(nonmeasurement_rows(
                            outcome, source_identity.as_dict(), perturbation_id,
                            shared_run_id, mask_identity=perturbed_mask_identities[perturbation_id],
                        ))
                        continue
                    try:
                        _append_records(result["__records__"], perturbation_id)
                    except RobustnessIdentityError as exc:
                        identity_failures[perturbation_id] = str(exc)
                        logger.warning(
                            "Excluding robustness perturbation %s/%s for identity failure: %s",
                            structure_name,
                            perturbation_id,
                            exc,
                        )
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            from radiomics import featureextractor

            def _factory():
                return (
                    featureextractor.RadiomicsFeatureExtractor(str(params_path))
                    if params_path else featureextractor.RadiomicsFeatureExtractor()
                )

            for perturbation_id, mask in masks.items():
                current_image = (
                    perturbed_images.get(perturbation_id, image)
                    if perturbed_images else image
                )
                try:
                    records = extract_ct_roi_arms(
                        current_image, mask, factory=_factory, decision=decision,
                        common_metadata={**source_identity.as_dict(), "roi_name": structure_name, "modality": "CT"},
                        run_identifier=shared_run_id, code_revision=current_code_revision(),
                        native_voxel_count=int(np.count_nonzero(sitk.GetArrayViewFromImage(mask))),
                        required=False, configured_parameter_hashes=configured_hashes,
                    )
                except (TimeoutError, MemoryError):
                    raise
                except Exception:
                    outcome = extraction_nonmeasurement(current_image, mask, _factory)
                    if outcome is None:
                        raise
                    rows.extend(nonmeasurement_rows(
                        outcome, source_identity.as_dict(), perturbation_id,
                        shared_run_id, mask_identity=perturbed_mask_identities[perturbation_id],
                    ))
                    continue
                from .radiomics_robustness_outcomes import returned_geometry_nonmeasurement
                outcome = returned_geometry_nonmeasurement(records, current_image, mask, _factory)
                if outcome is not None:
                    rows.extend(nonmeasurement_rows(
                        outcome, source_identity.as_dict(), perturbation_id,
                        shared_run_id, mask_identity=perturbed_mask_identities[perturbation_id],
                    ))
                    continue
                try:
                    _append_records(records, perturbation_id)
                except RobustnessIdentityError as exc:
                    identity_failures[perturbation_id] = str(exc)
                    logger.warning(
                        "Excluding robustness perturbation %s/%s for identity failure: %s",
                        structure_name,
                        perturbation_id,
                        exc,
                    )
        if identity_failures:
            rows = [
                row
                for row in rows
                if str(row.get("perturbation_id", "")) not in identity_failures
            ]
        frame = pd.DataFrame(rows)
        valid_perturbations = set(masks) - set(identity_failures)
        if valid_perturbations:
            _validate_extracted_feature_frame(
                frame,
                valid_perturbations,
                structure_name,
                expected_source_identity=source_identity,
            )
        frame.attrs["identity_failures"] = dict(identity_failures)
        return frame

    if use_conda:
        from .radiomics_conda import extract_radiomics_batch_with_conda, check_radiomics_env
        if not check_radiomics_env():
            raise RuntimeError(
                "radiomics conda environment is unavailable; "
                "robustness extraction cannot continue"
            )
        params_file = _get_params_file(config, modality)
        params_file_str = str(params_file) if params_file else None
        logger.info(
            "Using conda-based batch radiomics for robustness (%d perturbations for %s)",
            len(masks), structure_name,
        )

        # Batch approach: save all images/masks to temp files, extract in single subprocess
        tmp_dir = tempfile.mkdtemp(prefix="rtpipe_robust_batch_")
        try:
            batch_tasks = []

            # Cache unique images to avoid writing duplicates
            image_cache: Dict[int, str] = {}  # id(sitk_image) -> file path

            for pert_id, mask in masks.items():
                current_image = perturbed_images.get(pert_id, image) if perturbed_images else image
                img_id = id(current_image)

                if img_id not in image_cache:
                    img_path = os.path.join(tmp_dir, f"img_{len(image_cache)}.nrrd")
                    sitk.WriteImage(current_image, img_path)
                    image_cache[img_id] = img_path

                mask_path = os.path.join(tmp_dir, f"mask_{len(batch_tasks)}.nrrd")
                sitk.WriteImage(sitk.Cast(mask, sitk.sitkUInt8), mask_path)

                batch_tasks.append({
                    "image_path": image_cache[img_id],
                    "mask_path": mask_path,
                    "roi_name": f"{structure_name}/{pert_id}",
                })

            logger.info(
                "Saved %d perturbation files (%d unique images) for %s; "
                "starting batch conda extraction",
                len(batch_tasks), len(image_cache), structure_name,
            )

            batch_results = extract_radiomics_batch_with_conda(
                tasks=batch_tasks,
                params_file=params_file_str,
                timeout_per_roi=120,
            )

            for result in batch_results:
                if result is None:
                    continue
                status = result.get("__status__", "")
                # Extract pert_id from __roi_name__ (format: "structure/pert_id")
                # instead of positional indexing to avoid misattribution
                # if batch output lines are dropped or unparseable.
                roi_name = result.get("__roi_name__", "")
                if "/" in roi_name:
                    pert_id = roi_name.split("/", 1)[1]
                else:
                    logger.debug(
                        "Batch result missing valid __roi_name__: %r", roi_name,
                    )
                    continue
                if status != "success":
                    logger.debug(
                        "Batch extraction %s for %s/%s: %s",
                        status, structure_name, pert_id,
                        result.get("__error__", result.get("__reason__", "")),
                    )
                    continue
                for key, value in result.items():
                    if key.startswith("__"):
                        continue
                    if isinstance(value, (int, float)):
                        rows.append({
                            "patient_id": patient_id,
                            "course_id": course_id,
                            "modality": modality,
                            "structure": structure_name,
                            "perturbation_id": pert_id,
                            "feature_name": str(key),
                            "value": float(value),
                        })
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        frame = pd.DataFrame(rows)
        _validate_extracted_feature_frame(frame, set(masks), structure_name)
        return frame

    # Direct PyRadiomics path (NumPy 1.x) — reuse single extractor instance
    ext = _extractor(config, modality)
    if ext is None:
        raise RuntimeError(f"no radiomics extractor is available for {structure_name}")

    for pert_id, mask in masks.items():
        try:
            current_image = perturbed_images.get(pert_id, image) if perturbed_images else image
            result = ext.execute(current_image, mask)

            for key, value in result.items():
                scalar_value = _coerce_scalar_feature_value(str(key), value)
                if scalar_value is not None:
                    rows.append({
                        "patient_id": patient_id,
                        "course_id": course_id,
                        "modality": modality,
                        "structure": structure_name,
                        "perturbation_id": pert_id,
                        "feature_name": str(key),
                        "value": scalar_value,
                    })
        except Exception as e:
            raise RuntimeError(
                f"feature extraction failed for {structure_name}/{pert_id}: {e}"
            ) from e

    frame = pd.DataFrame(rows)
    _validate_extracted_feature_frame(frame, set(masks), structure_name)
    return frame


# ============================================================================
# Robustness Metrics Computation
# ============================================================================

def compute_icc_pingouin(
    df: pd.DataFrame,
    icc_config: ICCConfig,
) -> Dict[str, float]:
    """
    Compute ICC using Pingouin library.

    Args:
        df: DataFrame with columns ['subject', 'rater', 'value']
        icc_config: ICC configuration

    Returns:
        Dictionary with keys: icc, icc_ci95_low, icc_ci95_high
    """
    try:
        import pingouin as pg
    except ImportError:
        logger.warning("Pingouin not available; install with: pip install pingouin")
        return {"icc": np.nan, "icc_ci95_low": np.nan, "icc_ci95_high": np.nan}

    try:
        # --- C1 fix: detect unbalanced data before ICC computation ---
        # Pingouin's nan_policy="omit" performs listwise deletion, dropping entire
        # subjects when ANY rater value is missing. This silently biases ICC upward
        # (surviving subjects are easier cases). We detect and report this explicitly.
        n_raters_expected = df["rater"].nunique()
        subject_counts = df.groupby("subject")["rater"].nunique()
        complete_subjects = (subject_counts == n_raters_expected).sum()
        total_subjects = len(subject_counts)
        incomplete_subjects = total_subjects - complete_subjects
        drop_pct = (incomplete_subjects / total_subjects * 100) if total_subjects > 0 else 0

        if incomplete_subjects > 0:
            logger.warning(
                "ICC: %d/%d subjects (%.1f%%) have missing perturbations and will be "
                "dropped by listwise deletion. ICC may be biased upward.",
                incomplete_subjects, total_subjects, drop_pct,
            )
            if drop_pct > 10:
                logger.warning(
                    "ICC: >10%% subjects dropped — consider using ICC2 (two-way random) "
                    "or mixed-effects models for unbalanced designs."
                )

        # Use only complete cases to make the listwise deletion explicit
        # (rather than relying on Pingouin's silent nan_policy="omit")
        if incomplete_subjects > 0:
            complete_subject_ids = subject_counts[subject_counts == n_raters_expected].index
            df_balanced = df[df["subject"].isin(complete_subject_ids)].copy()
            if df_balanced["subject"].nunique() < 2:
                logger.warning("ICC: fewer than 2 complete subjects remain after dropping incomplete cases")
                return {
                    "icc": np.nan, "icc_ci95_low": np.nan, "icc_ci95_high": np.nan,
                    "n_subjects_dropped": int(incomplete_subjects),
                    "n_subjects_complete": int(complete_subjects),
                }
        else:
            df_balanced = df

        icc_res = pg.intraclass_corr(
            data=df_balanced,
            targets="subject",
            raters="rater",
            ratings="value",
            nan_policy="raise"  # Fail loudly if data is still unbalanced
        )

        # Pingouin 0.6 renamed the historical ICC1/2/3 labels while retaining
        # the same Shrout-Fleiss/McGraw-Wong estimands.
        type_aliases = {
            "ICC1": "ICC(1,1)",
            "ICC2": "ICC(A,1)",
            "ICC3": "ICC(C,1)",
            "ICC1k": "ICC(1,k)",
            "ICC2k": "ICC(A,k)",
            "ICC3k": "ICC(C,k)",
        }
        requested_types = [icc_config.icc_type]
        if icc_config.icc_type in type_aliases:
            requested_types.append(type_aliases[icc_config.icc_type])
        filtered = icc_res.loc[icc_res["Type"].isin(requested_types)]
        if filtered.empty:
            logger.debug("ICC type %s not found in results", icc_config.icc_type)
            return {"icc": np.nan, "icc_ci95_low": np.nan, "icc_ci95_high": np.nan}
        row = filtered.iloc[0]

        result = {
            "icc": float(row["ICC"]),
            "n_subjects_dropped": int(incomplete_subjects),
            "n_subjects_complete": int(complete_subjects),
        }

        if icc_config.ci:
            ci_column = "CI95%" if "CI95%" in row.index else "CI95"
            ci = row[ci_column]
            result["icc_ci95_low"] = float(ci[0])
            result["icc_ci95_high"] = float(ci[1])
        else:
            result["icc_ci95_low"] = np.nan
            result["icc_ci95_high"] = np.nan

        return result
    except Exception as e:
        logger.debug("ICC computation failed: %s", e)
        return {"icc": np.nan, "icc_ci95_low": np.nan, "icc_ci95_high": np.nan}


def compute_cov(values: np.ndarray) -> float:
    """Compute CoV after the caller establishes relative-scale suitability.

    The numeric guard cannot establish measurement-scale semantics. It rejects
    nonfinite, zero, or negative perturbation values, which cannot support the
    implemented relative-dispersion calculation, instead of converting them
    into a deceptively finite absolute-mean CoV.
    """
    if len(values) < 2:
        return np.nan

    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        return np.nan

    mean_val = np.mean(values)
    if abs(mean_val) < 1e-10:
        return np.nan

    std_val = np.std(values, ddof=1)
    return float((std_val / abs(mean_val)) * 100.0)


def compute_qcd(values: np.ndarray) -> float:
    """Compute QCD after the caller establishes relative-scale suitability."""
    if len(values) < 4:
        return np.nan

    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        return np.nan

    q1, q3 = np.percentile(values, [25, 75])

    if abs(q1 + q3) < 1e-10:
        return np.nan

    return float((q3 - q1) / (q3 + q1))


def summarize_feature_stability(
    df_long: pd.DataFrame,
    config: RobustnessConfig,
    group_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute per-feature robustness metrics (ICC, CoV, QCD) and classify robustness.

    Args:
        df_long: Tidy DataFrame with columns [patient_id, course_id, structure, segmentation_source,
                 perturbation_id, feature_name, value]
        config: Robustness configuration

    Returns:
        DataFrame with one row per (structure, segmentation_source, feature_name) containing metrics and robustness label
    """
    if "robustness_status" in df_long and df_long.robustness_status.eq("technical_failure").any():
        raise RuntimeError("technical robustness failures require recovery before aggregation")
    if "robustness_status" in df_long and df_long["robustness_status"].eq("geometrically_impossible").any():
        raise ValueError(
            "geometric non-measurements require an explicit comparable-condition analysis; "
            "do not drop subjects, impute values, or pool varying condition sets into fixed-grid ICC"
        )
    rows = []

    if group_columns is None:
        group_columns = ["structure"]
        if "segmentation_source" in df_long.columns:
            group_columns.append("segmentation_source")
        if "extraction_arm" in df_long.columns:
            group_columns.append("extraction_arm")
        group_columns.append("feature_name")

    for group_key, group in df_long.groupby(group_columns):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row_data = {col: val for col, val in zip(group_columns, key_values)}

        # Build unique subject identifier including course_id to avoid collapsing
        # different courses of the same patient into one subject for ICC computation.
        # Subject = patient_id + course_id + structure + segmentation_source
        # This ensures each course/timepoint is treated as a distinct subject instance.
        icc_df = group[["patient_id", "perturbation_id", "value"]].copy()
        icc_df.rename(columns={"perturbation_id": "rater"}, inplace=True)

        # Start with patient_id
        subject_parts = [group["patient_id"].astype(str)]

        # Include course_id if available (critical for multi-course/longitudinal data)
        if "course_id" in group.columns:
            subject_parts.append(group["course_id"].astype(str))

        # Include structure (already fixed by grouping, but ensures uniqueness across groups)
        if "structure" in df_long.columns:
            subject_parts.append(group["structure"].astype(str))

        # Include segmentation_source
        if "segmentation_source" in df_long.columns:
            subject_parts.append(group.get("segmentation_source", "unknown").astype(str))

        # Tuples preserve field boundaries (unlike underscore concatenation,
        # which can collide for legitimate identifiers containing underscores).
        icc_df["subject"] = list(zip(*(part.tolist() for part in subject_parts)))

        if icc_df.duplicated(["subject", "rater"]).any():
            raise ValueError(
                f"duplicate subject/perturbation values in robustness group {row_data}"
            )

        rater_sets = icc_df.groupby("subject")["rater"].agg(
            lambda values: frozenset(values.astype(str))
        )
        all_raters = frozenset().union(*rater_sets.tolist()) if len(rater_sets) else frozenset()
        incomplete_subjects = int((rater_sets != all_raters).sum())
        if incomplete_subjects:
            raise ValueError(
                f"incomplete perturbation grid for {incomplete_subjects} subject(s) "
                f"in robustness group {row_data}"
            )

        # CoV and QCD quantify within-subject perturbation dispersion. Computing
        # them over pooled raw values would confound perturbation instability
        # with between-patient anatomy/biology. Report the cohort median and IQR
        # of the subject-level metrics.
        subject_metric_frame = icc_df[["subject", "value"]]
        subject_cov = subject_metric_frame.groupby("subject")["value"].apply(
            lambda values: compute_cov(values.to_numpy(dtype=float))
        )
        subject_qcd = subject_metric_frame.groupby("subject")["value"].apply(
            lambda values: compute_qcd(values.to_numpy(dtype=float))
        )
        finite_cov = subject_cov[np.isfinite(subject_cov.to_numpy(dtype=float))]
        finite_qcd = subject_qcd[np.isfinite(subject_qcd.to_numpy(dtype=float))]
        cov = (
            float(np.median(finite_cov))
            if config.metrics.cov_enabled and not finite_cov.empty
            else np.nan
        )
        qcd = (
            float(np.median(finite_qcd))
            if config.metrics.qcd_enabled and not finite_qcd.empty
            else np.nan
        )
        cov_q1, cov_q3 = (
            tuple(float(value) for value in np.percentile(finite_cov, [25, 75]))
            if not finite_cov.empty
            else (np.nan, np.nan)
        )
        qcd_q1, qcd_q3 = (
            tuple(float(value) for value in np.percentile(finite_qcd, [25, 75]))
            if not finite_qcd.empty
            else (np.nan, np.nan)
        )

        n_subjects = icc_df["subject"].nunique()
        n_raters = icc_df["rater"].nunique()

        if n_subjects < 2 or n_raters < 2:
            icc_info = {"icc": np.nan, "icc_ci95_low": np.nan, "icc_ci95_high": np.nan}
        else:
            icc_info = compute_icc_pingouin(icc_df, config.metrics.icc)

        icc = icc_info["icc"]
        icc_ci_low = icc_info.get("icc_ci95_low", np.nan)
        icc_ci_high = icc_info.get("icc_ci95_high", np.nan)

        icc_for_threshold = icc_ci_low if not np.isnan(icc_ci_low) else icc

        robust_icc = icc_for_threshold >= config.thresholds.icc_robust
        acceptable_icc = icc_for_threshold >= config.thresholds.icc_acceptable

        robust_cov = cov <= config.thresholds.cov_robust_pct if not np.isnan(cov) else False
        acceptable_cov = cov <= config.thresholds.cov_acceptable_pct if not np.isnan(cov) else False
        cov_complete = bool(
            config.metrics.cov_enabled and len(finite_cov) == n_subjects
        )

        if not cov_complete:
            robustness_label = "not_evaluable"
        elif robust_icc and robust_cov:
            robustness_label = "robust"
        elif acceptable_icc and acceptable_cov:
            robustness_label = "acceptable"
        else:
            robustness_label = "poor"

        row_data.update({
            "feature_name": row_data.get("feature_name", group["feature_name"].iloc[0]),
            "n_subjects": n_subjects,
            "n_subjects_complete": n_subjects,
            "n_subjects_dropped": 0,
            # Course labels repeat across patients. Count the full identity,
            # consistent with the subject key used for the statistics above.
            "n_courses": (
                len(group[["patient_id", "course_id"]].astype(str).drop_duplicates())
                if "course_id" in group.columns else np.nan
            ),
            "n_perturbations": n_raters,
            "icc": icc,
            "icc_ci95_low": icc_ci_low,
            "icc_ci95_high": icc_ci_high,
            "cov_pct": cov,
            "cov_pct_q1": cov_q1,
            "cov_pct_q3": cov_q3,
            "n_subjects_cov": int(len(finite_cov)),
            "cov_status": (
                "complete"
                if len(finite_cov) == n_subjects
                else "not_evaluable"
                if finite_cov.empty
                else "partial"
            ),
            "qcd": qcd,
            "qcd_q1": qcd_q1,
            "qcd_q3": qcd_q3,
            "n_subjects_qcd": int(len(finite_qcd)),
            "qcd_status": (
                "complete"
                if len(finite_qcd) == n_subjects
                else "not_evaluable"
                if finite_qcd.empty
                else "partial"
            ),
            "robustness_label": robustness_label,
            "pass_seg_perturb": robustness_label in ["robust", "acceptable"],
        })

        rows.append(row_data)

    return pd.DataFrame(rows)


def _validate_cohort_feature_sets(df_long: pd.DataFrame) -> None:
    """Require a common feature inventory for comparable subjects in each ROI/source.

    Per-course validation proves that a feature is present for every configured
    perturbation *within* that course. Cohort aggregation must additionally prove
    that the same feature inventory exists for every patient/course contributing
    a given structure and segmentation source; otherwise an entirely absent
    feature would disappear from that subject before feature-wise grouping and
    silently bias the cohort summary.
    """
    required = {"patient_id", "structure", "feature_name"}
    missing = sorted(required - set(df_long.columns))
    if missing:
        raise ValueError(
            "robustness input is missing required cohort columns: " + ", ".join(missing)
        )

    subject_columns = ["patient_id"]
    if "course_id" in df_long.columns:
        subject_columns.append("course_id")
    comparison_columns = ["structure"]
    if "segmentation_source" in df_long.columns:
        comparison_columns.append("segmentation_source")
    if "extraction_arm" in df_long.columns:
        comparison_columns.append("extraction_arm")

    for group_key, group in df_long.groupby(comparison_columns, dropna=False):
        feature_sets = group.groupby(subject_columns, dropna=False)["feature_name"].agg(
            lambda values: frozenset(values.astype(str))
        )
        if feature_sets.empty:
            continue
        union = frozenset().union(*feature_sets.tolist())
        inconsistent = feature_sets[feature_sets != union]
        if not inconsistent.empty:
            group_values = group_key if isinstance(group_key, tuple) else (group_key,)
            group_label = dict(zip(comparison_columns, group_values))
            missing_counts = [len(union - features) for features in inconsistent.tolist()]
            raise ValueError(
                "inconsistent feature sets across subjects for "
                f"{group_label}: {len(inconsistent)} subject(s) missing "
                f"{sum(missing_counts)} feature assignment(s)"
            )


# ============================================================================
# Rotation Perturbation (v1.2 — sensitivity analysis)
# ============================================================================

def rotate_image_and_mask(
    image: sitk.Image,
    mask: sitk.Image,
    angle_degrees: float,
    axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Apply rigid rotation to both image and mask around the volume center.

    This implements the **R** (Rotation) perturbation absent from the default
    RTpipeline NTCV chain. The original Zwanenburg 2019 framework (N/T/R/V/C)
    included rotation; RTpipeline's adapted framework omits it by default.
    This function enables rotation sensitivity analyses to quantify the impact
    of that omission.

    Args:
        image: CT/MR SimpleITK image
        mask: Binary SimpleITK mask (same geometry as image)
        angle_degrees: Rotation angle in degrees
        axis: Rotation axis as (x, y, z) unit vector.
              Default (0,0,1) = axial rotation (around superior-inferior axis).

    Returns:
        Tuple of (rotated_image, rotated_mask)
    """
    angle_rad = np.deg2rad(angle_degrees)

    # Compute center of rotation (volume center in physical coordinates)
    size = image.GetSize()
    center_index = [s / 2.0 for s in size]
    center_physical = image.TransformContinuousIndexToPhysicalPoint(center_index)

    # Create Euler3D transform (rotation around center)
    transform = sitk.Euler3DTransform()
    transform.SetCenter(center_physical)

    # Set rotation angles based on axis
    ax = np.array(axis, dtype=float)
    ax = ax / (np.linalg.norm(ax) + 1e-12)
    # Euler3D uses rotations around x, y, z axes
    transform.SetRotation(
        float(ax[0] * angle_rad),
        float(ax[1] * angle_rad),
        float(ax[2] * angle_rad),
    )

    # Resample image with linear interpolation
    rotated_image = sitk.Resample(
        image,
        image,  # reference
        transform,
        sitk.sitkLinear,
        float(sitk.GetArrayViewFromImage(image).min()),  # default pixel value
    )

    # Resample mask with nearest neighbor (preserve binary)
    rotated_mask = sitk.Resample(
        mask,
        mask,
        transform,
        sitk.sitkNearestNeighbor,
        0,
    )

    return rotated_image, rotated_mask


def generate_rotation_sensitivity_perturbations(
    original_mask: sitk.Image,
    original_image: sitk.Image,
    rotation_angles: Optional[List[float]] = None,
    base_config: Optional[PerturbationConfig] = None,
    structure_name: str = "",
) -> Tuple[Dict[str, sitk.Image], Dict[str, sitk.Image]]:
    """
    Generate rotation-augmented perturbations for sensitivity analysis.

    Produces perturbation sets with and without rotation, enabling direct
    comparison of ICC values to quantify the impact of omitting rotation
    from the default NTCV chain.

    Default rotation angles: [+1, -1, +3, -3] degrees (axial plane).

    Args:
        original_mask: Original binary mask
        original_image: CT/MR image
        rotation_angles: List of rotation angles in degrees. Default: [1, -1, 3, -3]
        base_config: Optional base perturbation config (for combining with NTCV)
        structure_name: Structure name for logging

    Returns:
        Tuple of (perturbed_masks_dict, perturbed_images_dict)
    """
    if rotation_angles is None:
        rotation_angles = [1.0, -1.0, 3.0, -3.0]

    perturbed_masks = {"rot_original": original_mask}
    perturbed_images = {"rot_original": original_image}

    for angle in rotation_angles:
        try:
            rot_img, rot_mask = rotate_image_and_mask(
                original_image, original_mask, angle
            )
            pert_id = f"rot_{angle:+.1f}deg".replace(".", "p").replace("+", "plus").replace("-", "minus")
            perturbed_masks[pert_id] = rot_mask
            perturbed_images[pert_id] = rot_img
        except Exception as e:
            logger.warning(
                "Rotation perturbation failed for %s at %.1f°: %s",
                structure_name, angle, e,
            )

    logger.info(
        "Generated %d rotation perturbations for %s (angles: %s)",
        len(perturbed_masks) - 1, structure_name, rotation_angles,
    )

    return perturbed_masks, perturbed_images


# ============================================================================
# Redundancy Pruning (v1.2 — non-redundant robust feature panels)
# ============================================================================

def prune_redundant_features(
    stability_df: pd.DataFrame,
    feature_values_df: pd.DataFrame,
    correlation_threshold: float = 0.90,
    robustness_label_col: str = "robustness_label",
    icc_col: str = "icc",
    feature_col: str = "feature_name",
    structure_col: str = "structure",
) -> pd.DataFrame:
    """
    Prune redundant features using Spearman correlation clustering.

    For each structure, clusters robust features by Spearman |r| > threshold,
    then selects the representative with the highest mean ICC from each cluster.
    Reports both the full robust set and the non-redundant subset.

    This addresses the concern that wavelet/filter-expanded feature families
    can inflate robustness counts without adding independent information
    (METRICS, PMID: 38228979).

    Args:
        stability_df: Output from summarize_feature_stability() with ICC/robustness labels.
        feature_values_df: Wide-format DataFrame with features as columns, patients as rows.
            Must contain 'structure' and 'patient_id' columns plus feature value columns.
        correlation_threshold: Spearman |r| above which features are considered redundant.
            Default: 0.90 (a conservative project convention).
        robustness_label_col: Column name for robustness classification.
        icc_col: Column name for ICC values.
        feature_col: Column name for feature names.
        structure_col: Column name for structure names.

    Returns:
        stability_df with additional columns:
        - 'redundancy_cluster': cluster ID (integer) within each structure
        - 'is_cluster_representative': True if this feature is the cluster representative
        - 'cluster_size': number of features in the cluster
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    result_rows = []

    for structure, struct_stability in stability_df.groupby(structure_col):
        # Filter to robust/acceptable features only
        robust_mask = struct_stability[robustness_label_col].isin(["robust", "acceptable"])
        robust_features = struct_stability.loc[robust_mask, feature_col].tolist()

        if len(robust_features) < 2:
            # No clustering needed
            for _, row in struct_stability.iterrows():
                row_copy = row.to_dict()
                row_copy["redundancy_cluster"] = 0
                row_copy["is_cluster_representative"] = row[feature_col] in robust_features
                row_copy["cluster_size"] = 1
                result_rows.append(row_copy)
            continue

        # Get feature values for this structure
        struct_values = feature_values_df[
            feature_values_df[structure_col] == structure
        ] if structure_col in feature_values_df.columns else feature_values_df

        # Build correlation matrix for robust features
        available_features = [f for f in robust_features if f in struct_values.columns]
        if len(available_features) < 2:
            for _, row in struct_stability.iterrows():
                row_copy = row.to_dict()
                row_copy["redundancy_cluster"] = 0
                row_copy["is_cluster_representative"] = row[feature_col] in robust_features
                row_copy["cluster_size"] = 1
                result_rows.append(row_copy)
            continue

        corr_matrix = struct_values[available_features].corr(method="spearman").abs()

        # Convert to distance matrix and cluster
        distance_matrix = 1.0 - corr_matrix.values
        np.fill_diagonal(distance_matrix, 0)
        # Ensure symmetry and non-negativity
        distance_matrix = np.maximum(distance_matrix, 0)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2

        try:
            condensed = squareform(distance_matrix)
            Z = linkage(condensed, method="complete")
            clusters = fcluster(Z, t=1.0 - correlation_threshold, criterion="distance")
        except Exception as e:
            logger.warning("Clustering failed for %s: %s", structure, e)
            clusters = np.arange(len(available_features))

        # Map features to clusters
        feature_to_cluster = dict(zip(available_features, clusters))

        # Select representatives (highest ICC per cluster)
        icc_lookup = dict(
            zip(
                struct_stability[feature_col],
                struct_stability[icc_col],
            )
        )

        cluster_representatives = {}
        cluster_sizes = {}
        for cluster_id in set(clusters):
            cluster_features = [f for f, c in feature_to_cluster.items() if c == cluster_id]
            cluster_sizes[cluster_id] = len(cluster_features)
            # Pick feature with highest ICC
            best_feature = max(cluster_features, key=lambda f: icc_lookup.get(f, -1))
            cluster_representatives[cluster_id] = best_feature

        # Build output
        for _, row in struct_stability.iterrows():
            row_copy = row.to_dict()
            fname = row[feature_col]
            if fname in feature_to_cluster:
                cid = feature_to_cluster[fname]
                row_copy["redundancy_cluster"] = int(cid)
                row_copy["is_cluster_representative"] = (cluster_representatives.get(cid) == fname)
                row_copy["cluster_size"] = cluster_sizes.get(cid, 1)
            else:
                # Non-robust feature
                row_copy["redundancy_cluster"] = -1
                row_copy["is_cluster_representative"] = False
                row_copy["cluster_size"] = 0
            result_rows.append(row_copy)

    return pd.DataFrame(result_rows)


# ============================================================================
# Determinants of Feature Instability (v1.2 — exploratory mixed-effects)
# ============================================================================

def model_instability_determinants(
    stability_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    icc_col: str = "icc",
    structure_col: str = "structure",
    feature_col: str = "feature_name",
) -> Dict[str, Any]:
    """
    Fit exploratory mixed-effects model to identify determinants of feature instability.

    Model: ICC ~ ROI_volume + surface_volume_ratio + manufacturer + slice_thickness + feature_family + (1|patient)

    This answers: "which ROIs are radiomics-ready on planning CT, and which are not?"
    Bounded to pre-specified predictors only (no data-driven variable selection).

    Args:
        stability_df: Output from summarize_feature_stability() with per-(structure, feature) ICC.
            Must contain 'icc', 'structure', 'feature_name' columns.
        metadata_df: Patient/acquisition metadata with columns:
            - 'patient_id': patient identifier
            - 'manufacturer' (optional): CT manufacturer
            - 'slice_thickness' (optional): slice thickness in mm
            - 'pixel_spacing' (optional): pixel spacing in mm
            - 'roi_volume_cc' (optional): ROI volume in cc
            - 'surface_volume_ratio' (optional): surface/volume ratio
        icc_col: Column name for ICC values.
        structure_col: Column name for structure names.
        feature_col: Column name for feature names.

    Returns:
        Dictionary with:
        - 'model_summary': string summary of the mixed-effects model
        - 'fixed_effects': DataFrame of fixed effect coefficients
        - 'significant_predictors': list of significant predictors (p < 0.05)
        - 'radiomics_ready_rois': list of ROIs with mean ICC > 0.75
        - 'high_risk_rois': list of ROIs with mean ICC < 0.50
        - 'feature_family_effects': DataFrame of per-family mean ICC
    """
    results: Dict[str, Any] = {}

    # --- Feature family extraction ---
    def _extract_family(feature_name: str) -> str:
        """Extract feature family from PyRadiomics feature name."""
        parts = feature_name.split("_")
        if len(parts) >= 2:
            # e.g., "original_glcm_Autocorrelation" → "glcm"
            # or "wavelet-LLH_firstorder_Mean" → "firstorder"
            for i, part in enumerate(parts):
                if part.lower() in (
                    "shape", "shape2d", "firstorder",
                    "glcm", "glrlm", "glszm", "gldm", "ngtdm",
                ):
                    return part.lower()
        return "unknown"

    df = stability_df.copy()
    df["feature_family"] = df[feature_col].apply(_extract_family)

    # --- Per-structure summary ---
    structure_icc = df.groupby(structure_col)[icc_col].agg(["mean", "median", "std", "count"])
    structure_icc.columns = ["mean_icc", "median_icc", "std_icc", "n_features"]
    results["structure_summary"] = structure_icc.reset_index()

    results["radiomics_ready_rois"] = structure_icc[
        structure_icc["mean_icc"] >= 0.75
    ].index.tolist()

    results["high_risk_rois"] = structure_icc[
        structure_icc["mean_icc"] < 0.50
    ].index.tolist()

    # --- Per-feature-family summary ---
    family_icc = df.groupby("feature_family")[icc_col].agg(["mean", "median", "std", "count"])
    family_icc.columns = ["mean_icc", "median_icc", "std_icc", "n_features"]
    results["feature_family_effects"] = family_icc.reset_index()

    # --- Mixed-effects model (if statsmodels available) ---
    try:
        import statsmodels.formula.api as smf

        # Merge with metadata if available
        if metadata_df is not None and not metadata_df.empty:
            # Try to merge on patient_id if both have it
            if "patient_id" in df.columns and "patient_id" in metadata_df.columns:
                df = df.merge(metadata_df, on="patient_id", how="left", suffixes=("", "_meta"))

        # Build formula from available predictors
        predictors = []
        for col in ["feature_family", structure_col]:
            if col in df.columns and df[col].nunique() > 1:
                predictors.append(f"C({col})")
        for col in ["manufacturer"]:
            if col in df.columns and df[col].nunique() > 1:
                predictors.append(f"C({col})")
        for col in ["slice_thickness", "pixel_spacing", "roi_volume_cc", "surface_volume_ratio"]:
            if col in df.columns and df[col].notna().sum() > 10:
                predictors.append(col)

        if predictors and "patient_id" in df.columns:
            formula = f"{icc_col} ~ " + " + ".join(predictors)
            # Drop NaN ICC values
            df_model = df.dropna(subset=[icc_col])

            if df_model["patient_id"].nunique() > 2:
                try:
                    model = smf.mixedlm(
                        formula, df_model, groups=df_model["patient_id"]
                    )
                    fit = model.fit(reml=True)
                    results["model_summary"] = str(fit.summary())
                    results["fixed_effects"] = fit.summary().tables[1] if hasattr(fit.summary(), "tables") else str(fit.params)
                    results["significant_predictors"] = [
                        name for name, pval in fit.pvalues.items()
                        if pval < 0.05 and name != "Intercept"
                    ]
                except Exception as e:
                    logger.warning("Mixed-effects model fitting failed: %s", e)
                    results["model_summary"] = f"Model fitting failed: {e}"
                    results["significant_predictors"] = []
            else:
                results["model_summary"] = "Insufficient patient groups for mixed-effects model"
                results["significant_predictors"] = []
        else:
            results["model_summary"] = "Insufficient predictors for mixed-effects model"
            results["significant_predictors"] = []

    except ImportError:
        logger.info("statsmodels not available; skipping mixed-effects model")
        results["model_summary"] = "statsmodels not installed"
        results["significant_predictors"] = []

    return results


# ============================================================================
# Main Workflow Functions
# ============================================================================

def robustness_for_course(
    config: PipelineConfig,
    rob_config: RobustnessConfig,
    course_dir: Path,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Run radiomics robustness analysis for a single course.

    Collects masks from multiple sources:
    - Contracted manual RTSTRUCT
    - TotalSegmentator (RS_auto.dcm)
    - Custom structures (RS_custom.dcm)
    - Custom models (Segmentation_{model_name}/rtstruct.dcm)

    Args:
        config: Pipeline configuration
        rob_config: Robustness configuration
        course_dir: Course directory path

    Returns:
        Path to output parquet file or None
    """
    if not rob_config.enabled:
        logger.info("Radiomics robustness disabled; skipping")
        return None

    if "segmentation_perturbation" not in rob_config.modes:
        logger.info("Segmentation perturbation mode not enabled; skipping")
        return None

    logger.info("Running radiomics robustness analysis for %s", course_dir.name)

    if output_path is None:
        output_path = course_dir / "radiomics_robustness_ct.parquet"
    # Invalidate the previous run's published artifacts before anything the
    # current run can fail on, including course-contract validation. A stale
    # parquet must never survive a failed rerun, and a stale sidecar must never
    # be left claiming source dispositions for a course whose current contract
    # no longer validates.
    output_path.unlink(missing_ok=True)
    invalidate_robustness_source_dispositions(course_dir)

    # Capture the deciding code before any source is read, so the binding
    # describes the code this run started under rather than whatever the files
    # happen to contain by the time it publishes.
    run_code_identity = _capture_robustness_code_identity()

    contract = load_course_contract(course_dir)
    course_dirs = build_course_dirs(course_dir)

    # Load CT image
    from .radiomics import _load_series_image
    ct_dir = contract.planning_ct_dir
    ct_image = _load_series_image(ct_dir) if ct_dir is not None else None

    # NIfTI fallback when DICOM/CT is not available (e.g. remote staging)
    _use_nifti_masks = False
    if ct_image is None:
        ct_nifti = contract.planning_ct_nifti
        if ct_nifti is not None:
            try:
                ct_image = sitk.ReadImage(str(ct_nifti))
                _use_nifti_masks = True
                logger.info("Loaded contracted planning CT NIfTI: %s", ct_nifti.name)
            except Exception as e:
                logger.warning("Failed to load contracted CT NIfTI: %s", e)

    if ct_image is None:
        logger.warning("No CT image found for robustness analysis in %s", course_dir)
        return None

    series_uid = str(
        contract.planning_ct.get("series_instance_uid") or ""
    ).strip()
    if not series_uid:
        raise RuntimeError(
            f"Planning CT contract has no SeriesInstanceUID for robustness in {course_dir}"
        )
    identity_catalog, identity_catalog_issues = _load_main_ct_identity_catalog(
        course_dir,
        expected_series_uid=series_uid,
    )

    # ========================================================================
    # Collect masks from all segmentation sources
    # ========================================================================
    from .radiomics import _rtstruct_masks, _standard_rtstruct_sources
    from .custom_models import list_custom_model_outputs
    from .radiomics_ct_contract import new_run_identifier
    from fnmatch import fnmatch

    # One run identifier for the whole course robustness run: the published
    # table and the source-disposition sidecar are bound to the same value, so
    # a sidecar from a previous (or failed) run is rejected as stale.
    robustness_run_identifier = new_run_identifier()

    # Dictionary: {(roi_name, source): mask_array}
    all_masks: Dict[Tuple[str, str], np.ndarray] = {}
    unresolved_selected_identities: Dict[Tuple[str, str], Dict[str, str]] = {}
    # Terminal RTSTRUCT source dispositions recorded while collecting masks
    # (e.g. non-volumetric source ROIs), identity-bound and persisted to the
    # run-bound sidecar only once the run reaches a terminal, non-technical
    # outcome. Technical extraction failures raise before any publication.
    source_disposition_rows: List[Dict[str, Any]] = []
    rtstruct_source_bindings: List[Dict[str, str]] = []

    def _bind_rtstruct_source(
        source: str, rtstruct_path: Path
    ) -> Optional[Dict[str, str]]:
        """Capture source identity and content before its masks are read.

        A path that is not a readable file cannot certify anything; the mask
        reader remains responsible for failing closed on it, and an unbound
        source is not allowed to contribute published dispositions.
        """
        if not Path(rtstruct_path).is_file():
            return None
        binding = _rtstruct_source_binding(source, rtstruct_path)
        rtstruct_source_bindings.append(binding)
        return binding

    def _record_source_dispositions(
        binding: Optional[Dict[str, str]], sink: List[Dict[str, Any]]
    ) -> None:
        """Bind reader-recorded source dispositions to the source that produced them."""
        if sink and binding is None:
            raise RuntimeError(
                f"robustness read source dispositions from an unbound RTSTRUCT in "
                f"{course_dir}; refusing to publish uncertified dispositions"
            )
        for outcome in sink:
            outcome["segmentation_source"] = binding["segmentation_source"]
            # The reader records the source it read; the robustness pass binds
            # the exact bytes it read them from. A disagreement means the two
            # did not observe the same source, which nothing may paper over.
            for column in ("source_path", "rtstruct_sop_instance_uid"):
                recorded = outcome.get(column)
                if recorded is not None and str(recorded) != binding[column]:
                    raise RuntimeError(
                        f"robustness source disposition for "
                        f"{outcome.get('roi_name')!r} reports {column}="
                        f"{recorded!r}, but the bound source is "
                        f"{binding[column]!r}"
                    )
                outcome[column] = binding[column]
            source_disposition_rows.append(outcome)

    def _matches_robustness_pattern(roi_name: str) -> bool:
        return any(
            fnmatch(roi_name.upper(), pattern.upper())
            for pattern in rob_config.perturbation.apply_to_structures
        )

    def _register_mask(
        roi_name: str,
        source: str,
        mask_array: np.ndarray,
        *,
        report_missing_identity: bool = True,
    ) -> bool:
        """Register only masks with a unique main-radiomics identity."""
        key = (source, roi_name)
        if key in identity_catalog:
            all_masks[(roi_name, source)] = mask_array
            return True
        if report_missing_identity and _matches_robustness_pattern(roi_name):
            unresolved_selected_identities[key] = identity_catalog_issues.get(
                key,
                {
                    "reason_code": "identity_absent_from_main_radiomics",
                    "reason_detail": (
                        f"No authoritative CT radiomics identity for "
                        f"{source}/{roi_name}"
                    ),
                },
            )
        return False

    if _use_nifti_masks:
        # Load masks directly from NIfTI segmentation files (no DICOM needed)
        seg_dir = course_dir / "Segmentation_TotalSegmentator"
        if seg_dir.exists():
            loaded = 0
            # Search both flat and nested structures (some cohorts use scan subdirs)
            nii_files = list(seg_dir.glob("*.nii.gz"))
            if not nii_files:
                nii_files = list(seg_dir.glob("*/*.nii.gz"))
            for seg_file in sorted(nii_files):
                if "_cropped" in seg_file.name:
                    continue
                roi_name = seg_file.name.replace(".nii.gz", "")
                if roi_name.startswith("total--"):
                    roi_name = roi_name[len("total--"):]
                try:
                    mask_img = sitk.ReadImage(str(seg_file))
                    if mask_img.GetSize() != ct_image.GetSize():
                        mask_img = sitk.Resample(
                            mask_img, ct_image, sitk.Transform(),
                            sitk.sitkNearestNeighbor, 0)
                    mask_array = sitk.GetArrayFromImage(mask_img).astype(bool)
                    if mask_array.any() and _register_mask(
                        roi_name, "AutoRTS_total", mask_array
                    ):
                        loaded += 1
                except Exception as e:
                    logger.debug("Failed to load seg mask %s: %s", seg_file.name, e)
            logger.info("Loaded %d structures from NIfTI segmentation files", loaded)
        else:
            logger.info("Segmentation_TotalSegmentator not found")

        # Custom models NIfTI loading
        seg_custom = course_dir / "Segmentation_CustomModels"
        if seg_custom.exists():
            loaded = 0
            for model_dir in sorted(seg_custom.iterdir()):
                if not model_dir.is_dir():
                    continue
                for seg_file in sorted(model_dir.glob("*.nii.gz")):
                    roi_name = seg_file.name.replace(".nii.gz", "")
                    try:
                        mask_img = sitk.ReadImage(str(seg_file))
                        if mask_img.GetSize() != ct_image.GetSize():
                            mask_img = sitk.Resample(
                                mask_img, ct_image, sitk.Transform(),
                                sitk.sitkNearestNeighbor, 0)
                        mask_array = sitk.GetArrayFromImage(mask_img).astype(bool)
                        if mask_array.any() and _register_mask(
                            roi_name,
                            f"CustomModel:{model_dir.name}",
                            mask_array,
                        ):
                            loaded += 1
                    except Exception as e:
                        logger.debug("Failed to load custom mask %s: %s", seg_file.name, e)
            if loaded:
                logger.info("Loaded %d structures from custom model NIfTI files", loaded)
    else:
        # Use the same contracted standard-source resolver as main CT radiomics.
        # This includes Manual and a provenance-current AutoRTS source.
        assert ct_dir is not None
        for source, rtstruct_path, expected_rois in _standard_rtstruct_sources(
            contract, course_dir
        ):
            source_sink: List[Dict[str, Any]] = []
            source_binding = _bind_rtstruct_source(source, rtstruct_path)
            source_masks = _rtstruct_masks(
                ct_dir, rtstruct_path, failure_outcomes=source_sink
            )
            _record_source_dispositions(source_binding, source_sink)
            allowed_names = set(expected_rois) if expected_rois else None
            loaded = 0
            for roi_name, mask_array in source_masks.items():
                if allowed_names is not None and roi_name not in allowed_names:
                    continue
                if _register_mask(roi_name, source, mask_array):
                    loaded += 1
            logger.info(
                "Loaded %d identity-matched structures from %s",
                loaded,
                source,
            )

        # RS_custom contains copied Manual/Auto support ROIs. Only identities
        # published as Custom by the main path are independent Custom ROIs.
        rs_custom = course_dir / "RS_custom.dcm"
        if rs_custom.exists():
            custom_sink: List[Dict[str, Any]] = []
            custom_binding = _bind_rtstruct_source("Custom", rs_custom)
            custom_masks = _rtstruct_masks(
                ct_dir, rs_custom, failure_outcomes=custom_sink
            )
            _record_source_dispositions(custom_binding, custom_sink)
            loaded = 0
            for roi_name, mask_array in custom_masks.items():
                if _register_mask(
                    roi_name,
                    "Custom",
                    mask_array,
                    report_missing_identity=True,
                ):
                    loaded += 1
            logger.info(
                "Loaded %d identity-matched custom structures from %d RS_custom ROIs",
                loaded,
                len(custom_masks),
            )
        else:
            logger.debug("RS_custom.dcm not found")

    # Custom models (DICOM path, skipped in NIfTI mode and handled above).
    # Technical extraction failures propagate (fail-closed); non-volumetric
    # source ROIs are recorded as geometric dispositions in the sink.
    if not _use_nifti_masks:
        for model_name, model_dir in list_custom_model_outputs(course_dir):
            rs_model = model_dir / "rtstruct.dcm"
            if rs_model.exists():
                assert ct_dir is not None
                model_sink: List[Dict[str, Any]] = []
                source_label = f"CustomModel:{model_name}"
                model_binding = _bind_rtstruct_source(source_label, rs_model)
                model_masks = _rtstruct_masks(
                    ct_dir, rs_model, failure_outcomes=model_sink
                )
                _record_source_dispositions(model_binding, model_sink)
                loaded = 0
                for roi_name, mask_array in model_masks.items():
                    if _register_mask(roi_name, source_label, mask_array):
                        loaded += 1
                logger.info(
                    "Loaded %d identity-matched structures from custom model '%s'",
                    loaded,
                    model_name,
                )

    # NIfTI fallback: if DICOM-based mask loading returned 0 masks, try NIfTI
    if not all_masks and not _use_nifti_masks:
        logger.info("DICOM mask loading returned 0 masks; trying NIfTI fallback")
        seg_dir = course_dir / "Segmentation_TotalSegmentator"
        if seg_dir.exists():
            loaded = 0
            # Search both flat and nested structures (some cohorts use scan subdirs)
            nii_files = list(seg_dir.glob("*.nii.gz"))
            if not nii_files:
                nii_files = list(seg_dir.glob("*/*.nii.gz"))
            for seg_file in sorted(nii_files):
                # Skip cropped variants
                if "_cropped" in seg_file.name:
                    continue
                roi_name = seg_file.name.replace(".nii.gz", "")
                # Strip common prefixes (e.g. "total--")
                if roi_name.startswith("total--"):
                    roi_name = roi_name[len("total--"):]
                try:
                    mask_img = sitk.ReadImage(str(seg_file))
                    if mask_img.GetSize() != ct_image.GetSize():
                        mask_img = sitk.Resample(
                            mask_img, ct_image, sitk.Transform(),
                            sitk.sitkNearestNeighbor, 0)
                    mask_array = sitk.GetArrayFromImage(mask_img).astype(bool)
                    if mask_array.any() and _register_mask(
                        roi_name, "AutoRTS_total", mask_array
                    ):
                        loaded += 1
                except Exception as e:
                    logger.debug("Failed to load seg mask %s: %s", seg_file.name, e)
            logger.info("NIfTI fallback: loaded %d structures from segmentation files", loaded)

    logger.info(
        "Total identity-matched masks collected: %d from %d unique sources",
        len(all_masks),
        len(set(source for _, source in all_masks.keys())),
    )

    # ========================================================================
    # Filter structures and resolve identities before perturbation compute
    # ========================================================================
    selected_structures: List[Tuple[str, str]] = []
    identity_ledger_rows: List[Dict[str, Any]] = []

    def _record_identity_failure(
        source: str,
        roi_name: str,
        perturbation_id: str,
        reason_detail: str,
    ) -> None:
        identity_failed_perturbations[(roi_name, source, perturbation_id)] = reason_detail
        identity_ledger_rows.append(
            {
                "segmentation_source": source,
                "roi_original_name": roi_name,
                "identity_status": "excluded",
                "measurement_type": ROBUSTNESS_MEASUREMENT_TYPE,
                "perturbation_id": perturbation_id,
                "reason_code": "perturbation_identity_validation_failed",
                "reason_detail": reason_detail,
            }
        )
        logger.warning(
            "Excluding robustness perturbation %s/%s/%s for identity failure: %s",
            source,
            roi_name,
            perturbation_id,
            reason_detail,
        )

    for roi_name, source in all_masks:
        if not _matches_robustness_pattern(roi_name):
            continue
        selected_structures.append((roi_name, source))
        identity = identity_catalog[(source, roi_name)]
        identity_ledger_rows.append(
            {
                "segmentation_source": source,
                "roi_original_name": roi_name,
                "identity_status": "resolved",
                "measurement_type": ROBUSTNESS_MEASUREMENT_TYPE,
                **identity.as_dict(),
            }
        )

    for (source, roi_name), issue in sorted(
        unresolved_selected_identities.items()
    ):
        identity_ledger_rows.append(
            {
                "segmentation_source": source,
                "roi_original_name": roi_name,
                "identity_status": "excluded",
                "measurement_type": ROBUSTNESS_MEASUREMENT_TYPE,
                **issue,
            }
        )
        logger.warning(
            "Excluding robustness ROI %s/%s before perturbation: %s (%s)",
            source,
            roi_name,
            issue["reason_code"],
            issue["reason_detail"],
        )

    if identity_ledger_rows:
        _write_robustness_identity_ledger(
            course_dir,
            selected_count=len(identity_ledger_rows),
            rows=identity_ledger_rows,
        )

    def _publish_source_dispositions(
        measured_output: Optional[Path],
        *,
        nonmeasured_outcome: str = ROBUSTNESS_SOURCE_ONLY_OUTCOME,
        source_only_basis: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Publish this run's terminal source dispositions, or nothing at all.

        The sidecar is written only once the run has reached a terminal,
        non-technical outcome: either a published measurement table, or a
        source-only course that measured nothing and therefore has no table.
        A technical failure raises before reaching either call site, so a
        completed sidecar is never exposed for a failed run.
        """
        _write_robustness_source_dispositions(
            course_dir,
            run_identifier=robustness_run_identifier,
            rows=source_disposition_rows,
            source_bindings=rtstruct_source_bindings,
            effective_configuration=effective_robustness_configuration(
                rob_config, output_name=output_path.name
            ),
            code_identity=run_code_identity,
            output_path=output_path,
            measured_output=measured_output,
            nonmeasured_outcome=nonmeasured_outcome,
            source_only_basis=source_only_basis,
        )
        logger.info(
            "Recorded %d RTSTRUCT source disposition(s) for robustness in %s (%s)",
            len(source_disposition_rows),
            course_dir,
            "measured" if measured_output is not None else "source-only",
        )

    if not selected_structures:
        if unresolved_selected_identities:
            logger.warning(
                "No robustness ROI has publishable identity for %s",
                course_dir,
            )
            # An unresolved identity is not a terminal source disposition: the
            # source picture is incomplete, so nothing is published for it.
            logger.warning(
                "Not publishing robustness source dispositions for %s: %d selected "
                "ROI identities are unresolved",
                course_dir,
                len(unresolved_selected_identities),
            )
        else:
            logger.info(
                "No structures matched robustness patterns; skipping %s",
                course_dir,
            )
            # Two different terminal outcomes hide behind "nothing was
            # selected". Only a source ROI that the selection actually matched
            # and that turned out to be non-volumetric is evidence about
            # anatomy; a requested name that no source carries (or no request
            # at all) is a fact about the request, and an unrelated
            # non-volumetric ROI elsewhere in the source does not upgrade it.
            selection_matched_dispositions = [
                row
                for row in source_disposition_rows
                if _matches_robustness_pattern(str(row.get("roi_name") or ""))
            ]
            if selection_matched_dispositions:
                nonmeasured_outcome = ROBUSTNESS_SOURCE_ONLY_OUTCOME
                reason = (
                    "every source ROI matching the robustness selection is "
                    "non-volumetric"
                )
            else:
                nonmeasured_outcome = ROBUSTNESS_UNMATCHED_SELECTION_OUTCOME
                reason = (
                    "no source ROI matched the requested robustness selection; "
                    "an unmatched selection is not evidence of non-volumetric "
                    "anatomy and excludes nothing"
                )
            _publish_source_dispositions(
                None,
                nonmeasured_outcome=nonmeasured_outcome,
                source_only_basis={
                    "selected_structure_count": 0,
                    "unresolved_identity_count": 0,
                    "collected_mask_count": len(all_masks),
                    "source_disposition_count": len(source_disposition_rows),
                    "selection_matched_nonvolumetric_count": len(
                        selection_matched_dispositions
                    ),
                    "requested_selection": sorted(
                        str(pattern)
                        for pattern in rob_config.perturbation.apply_to_structures
                    ),
                    "reason": reason,
                },
            )
        return None

    logger.info("Selected %d structure(s) for robustness analysis:", len(selected_structures))
    for roi_name, source in selected_structures:
        logger.info("  - %s (from %s)", roi_name, source)

    # ========================================================================
    # Extract features for each structure with perturbations
    # ========================================================================
    from .radiomics import _mask_from_array_like
    
    # Import parallel processing helpers
    try:
        from .radiomics_parallel import (
            _prepare_radiomics_task,
            _isolated_radiomics_extraction_with_retry,
            _calculate_optimal_workers,
            _apply_thread_limit,
            _resolve_thread_limit
        )
        has_parallel = True
    except ImportError:
        logger.warning("Parallel radiomics helpers not available; falling back to sequential processing")
        has_parallel = False

    disable_parallel = os.environ.get("RTPIPELINE_DISABLE_PARALLEL_RADIOMICS", "0").strip().lower()
    if disable_parallel not in {"", "0", "false", "no"}:
        logger.info("Parallel radiomics disabled via RTPIPELINE_DISABLE_PARALLEL_RADIOMICS=%s", disable_parallel)
        has_parallel = False
    all_features = []
    generated_nonmeasurement_keys = set()
    from .radiomics_ct_contract import CT_EXTRACTION_ARMS
    # robustness_run_identifier was generated before mask collection so the
    # source-disposition sidecar and the published table share one identity.
    expected_perturbations: Dict[Tuple[str, str], set[str]] = {}
    identity_failed_perturbations: Dict[Tuple[str, str, str], str] = {}
    
    # Prepare tasks for parallel execution
    tasks = []
    temp_dir = None
    
    if has_parallel:
        temp_dir = Path(tempfile.mkdtemp(prefix='rob_radiomics_'))
        # Apply thread limit to main process
        _apply_thread_limit(_resolve_thread_limit(getattr(config, 'radiomics_thread_limit', None)))
    
    try:
        for roi_name, source in selected_structures:
            mask_array = all_masks[(roi_name, source)]
            mask_img = _mask_from_array_like(ct_image, mask_array)

            # Check if NTCV mode is enabled (any perturbation beyond volume is configured)
            use_ntcv = (
                rob_config.perturbation.max_translation_mm > 0 or
                rob_config.perturbation.n_random_contour_realizations > 0 or
                any(n > 0 for n in rob_config.perturbation.noise_levels)
            )

            if use_ntcv:
                # Generate NTCV perturbation chain
                perturbed_masks, perturbed_images = generate_ntcv_perturbations(
                    mask_img,
                    ct_image,
                    rob_config.perturbation,
                    roi_name,
                )
            else:
                # Legacy mode: volume-only perturbations
                # C19 fix: honor intensity config — use large_volume_changes for aggressive
                intensity = rob_config.perturbation.intensity
                if intensity == "aggressive":
                    volume_changes = rob_config.perturbation.large_volume_changes
                else:
                    volume_changes = rob_config.perturbation.small_volume_changes
                perturbed_masks = generate_perturbed_masks(
                    mask_img,
                    volume_changes,
                    roi_name,
                )
                perturbed_images = None

            if len(perturbed_masks) < 2:
                raise RuntimeError(
                    f"insufficient perturbations for {roi_name} ({source}): "
                    f"generated {len(perturbed_masks)}"
                )
            expected_perturbations[(roi_name, source)] = set(perturbed_masks)
            for pert_id, outcome in list(perturbed_masks.items()):
                if isinstance(outcome, GeometricNonmeasurement):
                    all_features.append(pd.DataFrame(nonmeasurement_rows(
                        outcome, identity_catalog[(source, roi_name)].as_dict(),
                        pert_id, robustness_run_identifier,
                    )))
                    generated_nonmeasurement_keys.update(
                        (roi_name, source, pert_id, arm) for arm in CT_EXTRACTION_ARMS
                    )
                    del perturbed_masks[pert_id]
            if not perturbed_masks:
                continue

            if has_parallel:
                # Prepare parallel tasks
                assert temp_dir is not None
                for pert_id, mask in perturbed_masks.items():
                    # Use perturbed image if available
                    current_image = perturbed_images.get(pert_id, ct_image) if perturbed_images else ct_image
                    
                    try:
                        # We treat each perturbation as a "structure" for the parallel worker
                        task_file, task_params = _prepare_radiomics_task(
                            current_image,
                            mask,
                            config,
                            source,
                            roi_name,
                            course_dir,
                            temp_dir,
                            False,
                            robustness_run_identifier,
                            identity_catalog[(source, roi_name)].as_dict(),
                        )
                        # Add perturbation-specific metadata to extra_metadata
                        task_params['extra_metadata'] = {'perturbation_id': pert_id}
                        tasks.append((task_file, task_params))
                    except Exception as e:
                        raise RuntimeError(
                            f"failed to prepare robustness task for "
                            f"{source}/{roi_name}/{pert_id}: {e}"
                        ) from e
            else:
                # Sequential fallback
                features_df = extract_features_for_masks(
                    ct_image,
                    perturbed_masks,
                    config,
                    modality="CT",
                    structure_name=roi_name,
                    patient_id=course_dir.parent.name,
                    course_id=course_dir.name,
                    perturbed_images=perturbed_images,
                    segmentation_source=source,
                    run_identifier=robustness_run_identifier,
                    source_identity=identity_catalog[(source, roi_name)],
                )
                for perturbation_id, reason_detail in features_df.attrs.get(
                    "identity_failures", {}
                ).items():
                    _record_identity_failure(
                        source, roi_name, perturbation_id, str(reason_detail)
                    )
                    expected_perturbations[(roi_name, source)].discard(
                        perturbation_id
                    )
                if not features_df.empty:
                    features_df["segmentation_source"] = source
                    all_features.append(features_df)

        # Execute parallel tasks
        if has_parallel and tasks:
            max_workers = _calculate_optimal_workers()
            # Respect global worker limit if set
            try:
                config_workers = int(getattr(config, 'effective_workers')())
                max_workers = min(max_workers, config_workers)
            except Exception:
                pass
            
            max_workers = max(1, min(max_workers, len(tasks)))
            logger.info("Processing %d robustness perturbations with %d workers", len(tasks), max_workers)

            ctx = get_context('spawn')

            # Timeout configuration for watchdog
            course_timeout = int(os.environ.get("RTPIPELINE_ROBUSTNESS_COURSE_TIMEOUT", "14400"))  # 4 hour default
            progress_timeout = int(os.environ.get("RTPIPELINE_ROBUSTNESS_PROGRESS_TIMEOUT", "300"))  # 5 min default

            from .robustness_watchdog import SupervisedResults, technical_rows
            with SupervisedResults(
                ctx, _isolated_radiomics_extraction_with_retry, tasks, max_workers,
                course_timeout=course_timeout, progress_timeout=progress_timeout,
            ) as results_iter:
                completed_count = 0
                total_count = len(tasks)
                successful_task_keys: set[Tuple[str, str, str, str]] = set(generated_nonmeasurement_keys)
                start_time = time.monotonic()
                timed_out = False
                returned_indices = set()
                while completed_count < total_count:
                    try:
                        result = results_iter.next(timeout=10)
                    except MPTimeoutError:
                        continue
                    except StopIteration:
                        break
                    index = result.get("__task_index__")
                    if not isinstance(index, int) or index in returned_indices or not 0 <= index < total_count:
                        raise RuntimeError("duplicate or invalid robustness task result identity")
                    returned_indices.add(index)
                    task_params = tasks[index][1]
                    if any(str(result.get(key, "")) != str(task_params.get(key, ""))
                           for key in ("roi_name", "segmentation_source")) or (
                        result.get("perturbation_id") != task_params.get("extra_metadata", {}).get("perturbation_id")
                    ):
                        raise RuntimeError("robustness result does not match submitted condition")
                    completed_count += 1
                    if completed_count % 10 == 0 or completed_count == total_count:
                        elapsed = time.monotonic() - start_time
                        eta = (total_count-completed_count)*elapsed/completed_count
                        logger.info("Robustness progress: %d/%d (%.1f%%), ETA: %.1fs",
                                    completed_count, total_count, 100*completed_count/total_count, eta)
                    if "__technical_failure__" in result:
                        roi_name, source, perturbation_id = (
                            result["roi_name"], result["segmentation_source"], result["perturbation_id"])
                        failed_rows = technical_rows(
                            result, identity_catalog[(source, roi_name)].as_dict(),
                            robustness_run_identifier, task_params["perturbed_mask_identity"], task_params,
                        )
                        all_features.append(pd.DataFrame(failed_rows))
                        successful_task_keys.update((roi_name, source, perturbation_id, arm) for arm in CT_EXTRACTION_ARMS)
                        continue

                    # Process result if we got one

                    if result:
                        rows = _feature_rows_from_worker_result(result)
                        if rows:
                            result_frame = pd.DataFrame(rows)
                            roi_name = str(result.get("roi_name", ""))
                            source = str(result.get("segmentation_source", ""))
                            perturbation_id = str(result.get("perturbation_id", ""))
                            if (roi_name, source, perturbation_id) in identity_failed_perturbations:
                                continue
                            source_identity = identity_catalog.get((source, roi_name))
                            try:
                                if source_identity is None:
                                    raise RobustnessIdentityError(
                                        "Robustness worker result has no authoritative "
                                        f"identity for {source}/{roi_name}"
                                    )
                                _validate_extracted_feature_frame(
                                    result_frame,
                                    {perturbation_id},
                                    f"{source}/{roi_name}/{perturbation_id}",
                                    expected_source_identity=source_identity,
                                )
                            except RobustnessIdentityError as exc:
                                _record_identity_failure(
                                    source, roi_name, perturbation_id, str(exc)
                                )
                                task_key = (roi_name, source, perturbation_id)
                                expected_perturbations.get(
                                    (roi_name, source), set()
                                ).discard(perturbation_id)
                                for arm in CT_EXTRACTION_ARMS:
                                    successful_task_keys.discard((*task_key, arm))
                                continue
                            for extraction_arm in result_frame[
                                "extraction_arm"
                            ].astype(str).unique():
                                successful_task_keys.add(
                                    (
                                        roi_name,
                                        source,
                                        perturbation_id,
                                        extraction_arm,
                                    )
                                )
                            all_features.append(result_frame)
                        else:
                            logger.error(
                                "Robustness worker returned no scalar features for %s/%s "
                                "(perturbation %s)",
                                result.get("segmentation_source"),
                                result.get("roi_name"),
                                result.get("perturbation_id"),
                            )

                expected_task_keys = {
                    (roi_name, source, perturbation_id, arm)
                    for (roi_name, source), perturbation_ids in expected_perturbations.items()
                    for perturbation_id in perturbation_ids
                    for arm in CT_EXTRACTION_ARMS
                }
                missing_task_keys = sorted(expected_task_keys - successful_task_keys)
                unexpected_task_keys = sorted(successful_task_keys - expected_task_keys)
                if (
                    timed_out
                    or completed_count != total_count
                    or missing_task_keys
                    or unexpected_task_keys
                ):
                    raise RuntimeError(
                        "incomplete robustness extraction: "
                        f"completed={completed_count}/{total_count}, "
                        f"successful={len(successful_task_keys)}, "
                        f"missing={missing_task_keys}, unexpected={unexpected_task_keys}, "
                        f"timed_out={timed_out}"
                    )

    except Exception as e:
        logger.error("Robustness analysis failed: %s", e)
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.debug("Failed to clean up temp dir %s: %s", temp_dir, e)

    if identity_failed_perturbations:
        _write_robustness_identity_ledger(
            course_dir,
            selected_count=len(identity_ledger_rows),
            rows=identity_ledger_rows,
        )

    if not all_features:
        logger.warning("No features extracted for robustness analysis in %s", course_dir)
        return None

    # Combine all features
    combined_df = pd.concat(all_features, ignore_index=True)
    if "robustness_status" not in combined_df:
        combined_df["robustness_status"] = "measured"
    else:
        combined_df["robustness_status"] = combined_df["robustness_status"].fillna("measured")
    if "roi_name" in combined_df.columns and "structure" not in combined_df.columns:
        combined_df.rename(columns={"roi_name": "structure"}, inplace=True)
    if "roi_name" in combined_df and "structure" in combined_df:
        combined_df["structure"] = combined_df["structure"].fillna(combined_df["roi_name"])
    if identity_failed_perturbations:
        failed_keys = set(identity_failed_perturbations)
        combined_df = combined_df.loc[
            ~combined_df.apply(
                lambda row: (
                    str(row.get("structure", "")),
                    str(row.get("segmentation_source", "")),
                    str(row.get("perturbation_id", "")),
                ) in failed_keys,
                axis=1,
            )
        ].reset_index(drop=True)
    for (roi_name, source), expected_ids in expected_perturbations.items():
        if not expected_ids:
            continue
        group = combined_df[
            (combined_df["structure"].astype(str) == str(roi_name))
            & (combined_df["segmentation_source"].astype(str) == str(source))
        ]
        _validate_extracted_feature_frame(
            group,
            expected_ids,
            f"{source}/{roi_name}",
            expected_source_identity=identity_catalog[(source, roi_name)],
        )

    for (roi_name, source), requested_ids in expected_perturbations.items():
        selected = ((combined_df["structure"] == roi_name)
                    & (combined_df["segmentation_source"] == source))
        impossible_ids = sorted(set(combined_df.loc[
            selected & combined_df["robustness_status"].eq("geometrically_impossible"),
            "perturbation_id",
        ]))
        combined_df.loc[selected, "requested_condition_ids"] = json.dumps(sorted(requested_ids))
        combined_df.loc[selected, "impossible_condition_ids"] = json.dumps(impossible_ids)
        combined_df.loc[selected, "possible_condition_count"] = len(requested_ids) - len(impossible_ids)

    # Bind the published rows to the run that produced them: a table carrying a
    # foreign run identifier cannot be certified by this run's dispositions.
    if "run_identifier" in combined_df.columns:
        published_runs = {
            str(value)
            for value in combined_df["run_identifier"].dropna().unique()
        }
        foreign_runs = sorted(published_runs - {str(robustness_run_identifier)})
        if foreign_runs:
            raise RuntimeError(
                f"robustness rows for {course_dir} carry run identifiers "
                f"{foreign_runs} that are not this run "
                f"({robustness_run_identifier})"
            )

    # Preserve validated measurements and the exact failed-condition inventory.
    # A technical failure remains a failed stage, so the CLI never creates a
    # success sentinel and aggregation cannot silently use an incomplete grid.
    technical_failure_count = int(combined_df["robustness_status"].eq("technical_failure").sum())

    # Verify provenance *before* publishing, not after: a table that nothing
    # can certify must never exist, not even briefly. The masks were read from
    # these RTSTRUCT bytes and decided by this code, so a source or a deciding
    # module that changed while the run worked invalidates the run itself.
    _verify_rtstruct_source_bindings(
        rtstruct_source_bindings,
        error=RuntimeError,
        context="refusing to publish robustness measurements",
    )
    _verify_robustness_code_identity(
        run_code_identity,
        context="refusing to publish robustness measurements",
    )

    # Save raw feature values for aggregation stage
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output_path.with_name(output_path.name + ".tmp")
        try:
            combined_df.to_parquet(temporary_output, index=False)
            os.replace(temporary_output, output_path)
        finally:
            temporary_output.unlink(missing_ok=True)
        if technical_failure_count:
            # The partial table stays published as failed-stage evidence, but a
            # failed run never gets a completed source-disposition sidecar.
            raise RuntimeError(
                f"incomplete robustness extraction: published partial results to {output_path}; "
                f"technical_failure_arm_rows={technical_failure_count}; see exact conditions and evidence in parquet"
            )
    except Exception as e:
        if technical_failure_count and output_path.exists() and "published partial results" in str(e):
            raise
        raise RuntimeError(f"failed to save robustness results to {output_path}: {e}") from e

    try:
        _publish_source_dispositions(output_path)
    except BaseException:
        # The nominal completed table was never certified, so it is withdrawn
        # rather than left looking like a measured result. The deliberately
        # partial technical_failure table is a different case: it raises above,
        # before this point, and its failed-condition evidence stays published.
        output_path.unlink(missing_ok=True)
        raise
    logger.info(
        "Saved robustness feature values to %s (%d rows, %d unique perturbations)",
        output_path,
        len(combined_df),
        combined_df["perturbation_id"].nunique(),
    )
    return output_path


# ============================================================================
# Typed course outcome: what one robustness course step actually established
# ============================================================================


class RobustnessNotRequestedError(RuntimeError):
    """Robustness (or its perturbation mode) is switched off for this course.

    A disabled stage is a configuration state, not a course outcome. It is its
    own error type so a caller must branch on it explicitly instead of reading
    a ``None`` as either success or failure, and so a disabled run can never
    reach the admission path and adopt a previous run's sidecar.
    """


class RobustnessSelectionUnmatchedError(RuntimeError):
    """No source ROI matched the requested robustness selection.

    This is a blocking selection/configuration outcome. Nothing was measured
    and nothing was found to be non-volumetric, so the course has no anatomical
    finding to publish: an unfound name excludes no anatomy, and no technical
    geometry may be fabricated to stand in for it. The run's sidecar evidence
    is preserved, but the course completes nothing and enters no cohort.
    """


# A measured course and a course whose selected source ROIs are all
# non-volumetric both reached a terminal, non-technical answer. An unmatched
# selection did not: it stays blocking.
ROBUSTNESS_STEP_COMPLETING_OUTCOMES = frozenset(
    {ROBUSTNESS_MEASURED_OUTCOME, ROBUSTNESS_SOURCE_ONLY_OUTCOME}
)


@dataclass(frozen=True)
class RobustnessCourseOutcome:
    """One course's terminal, admitted robustness outcome.

    Produced only by :func:`run_robustness_course`, and only from evidence the
    current run published: the outcome, the run identifier, the exact output
    name, the disposition sidecar and its digest, and (for a measured course)
    the exact table bytes. ``None`` alone never produces one of these.
    """

    course_dir: Path
    patient_id: str
    course_id: str
    run_identifier: str
    measurement_outcome: str
    output_name: str
    dispositions_path: Path
    dispositions_sha256: str
    source_disposition_count: int
    effective_configuration_sha256: str
    measured_output: Optional[Path] = None
    measured_output_sha256: Optional[str] = None

    @property
    def measured(self) -> bool:
        return self.measurement_outcome == ROBUSTNESS_MEASURED_OUTCOME

    @property
    def completes_step(self) -> bool:
        """Measured and source-only are terminal completions; nothing else is."""
        return self.measurement_outcome in ROBUSTNESS_STEP_COMPLETING_OUTCOMES


def _require_course_output_path(course_dir: Path, output_path: Path) -> Path:
    """The output must be one named file inside the course being processed."""
    from .course_manifest import require_no_output_symlinks

    course_dir = Path(course_dir)
    output_path = Path(output_path)
    require_no_output_symlinks(course_dir)
    require_no_output_symlinks(output_path)
    require_no_output_symlinks(robustness_source_dispositions_path(course_dir))
    if output_path.name != str(output_path.name).strip() or not output_path.name:
        raise ValueError(f"robustness output name {output_path.name!r} is unusable")
    if output_path.parent.resolve(strict=False) != course_dir.resolve(strict=False):
        raise ValueError(
            f"robustness output {output_path} is not inside course {course_dir}"
        )
    return output_path


def _admit_source_only_course_outcome(
    course_dir: Path,
    *,
    rob_config: "RobustnessConfig",
    output_name: str,
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Read this run's sidecar for a course that published no table.

    The run identifier comes from the sidecar because no table exists to
    declare one; every other binding (sources, code, configuration, digests,
    outcome, and the absence of a table) is re-checked by the loader, and the
    caller has already established that the sidecar is this run's own.
    """
    path = robustness_source_dispositions_path(course_dir)
    if not path.is_file():
        raise RuntimeError(
            f"robustness for {course_dir} produced no measurement table and no "
            "source-disposition evidence; an absent answer is a failure, not a "
            "completed course"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"robustness source dispositions {path} are not a record: "
            f"{type(payload).__name__}"
        )
    run_identifier = str(payload.get("robustness_run_identifier") or "").strip()
    if not run_identifier:
        raise ValueError(
            f"robustness source dispositions {path} record no run identity"
        )
    outcome = str(payload.get("measurement_outcome") or "")
    rows = load_robustness_source_dispositions(
        course_dir,
        run_identifier=run_identifier,
        rob_config=rob_config,
        output_name=output_name,
    )
    return run_identifier, outcome, rows


def run_robustness_course(
    config: PipelineConfig,
    rob_config: "RobustnessConfig",
    course_dir: Path,
    *,
    output_path: Path,
) -> RobustnessCourseOutcome:
    """Run one course's robustness pass and admit exactly what it published.

    This is the consumer adapter for :func:`robustness_for_course`, whose
    ``Optional[Path]`` return is deliberately unchanged. That return value
    cannot distinguish a course that legitimately measured nothing from one
    that found no CT or extracted nothing, so this function never reads it as
    an outcome on its own: it requires evidence this call produced, under the
    caller's current configuration and for the exact output name requested.

    Raises:
        RobustnessNotRequestedError: robustness or its mode is disabled.
        RobustnessSelectionUnmatchedError: the selection matched no source ROI.
        Exception: any technical failure of the run, unchanged and unswallowed.
    """
    course_dir = Path(course_dir)
    output_path = _require_course_output_path(course_dir, output_path)
    output_name = output_path.name

    if not rob_config.enabled:
        raise RobustnessNotRequestedError(
            "radiomics robustness is disabled in the effective configuration"
        )
    if ROBUSTNESS_MEASUREMENT_TYPE not in rob_config.modes:
        raise RobustnessNotRequestedError(
            f"robustness mode {ROBUSTNESS_MEASUREMENT_TYPE!r} is not enabled in "
            f"the effective configuration (modes={list(rob_config.modes)})"
        )

    # Drop any previous run's sidecar before this run starts. The producer does
    # the same, but doing it here as well is what lets the admission below say
    # that whatever it finds afterwards was written by *this* call, so an
    # unexpected ``None`` can never be certified by an older run's evidence.
    invalidate_robustness_source_dispositions(course_dir)

    produced = robustness_for_course(
        config, rob_config, course_dir, output_path=output_path
    )

    if produced is not None:
        produced = Path(produced)
        if produced.resolve(strict=False) != output_path.resolve(strict=False):
            raise RuntimeError(
                f"robustness published {produced}, not the requested output "
                f"{output_path}"
            )
        admitted = _admit_robustness_aggregation_input(output_path, rob_config)
        dispositions_path = robustness_source_dispositions_path(course_dir)
        return RobustnessCourseOutcome(
            course_dir=course_dir,
            patient_id=admitted.patient_id,
            course_id=admitted.course_id,
            run_identifier=admitted.run_identifier,
            measurement_outcome=ROBUSTNESS_MEASURED_OUTCOME,
            output_name=output_name,
            dispositions_path=dispositions_path,
            dispositions_sha256=_file_sha256(dispositions_path),
            source_disposition_count=len(admitted.source_dispositions),
            effective_configuration_sha256=_content_sha256(
                effective_robustness_configuration(
                    rob_config, output_name=output_name
                )
            ),
            measured_output=output_path,
            measured_output_sha256=admitted.table_sha256,
        )

    run_identifier, outcome, rows = _admit_source_only_course_outcome(
        course_dir, rob_config=rob_config, output_name=output_name
    )
    if outcome == ROBUSTNESS_MEASURED_OUTCOME:
        raise RuntimeError(
            f"robustness for {course_dir} published no table but its source "
            "dispositions claim a measured outcome"
        )
    if outcome == ROBUSTNESS_UNMATCHED_SELECTION_OUTCOME:
        raise RobustnessSelectionUnmatchedError(
            f"robustness for {course_dir} matched no source structure for the "
            f"requested selection "
            f"{sorted(str(p) for p in rob_config.perturbation.apply_to_structures)}; "
            "this is a selection/configuration outcome, not an anatomical "
            "finding, and it completes no course"
        )
    if outcome != ROBUSTNESS_SOURCE_ONLY_OUTCOME:
        raise RuntimeError(
            f"robustness for {course_dir} recorded outcome {outcome!r}, which is "
            "not a terminal course outcome"
        )
    dispositions_path = robustness_source_dispositions_path(course_dir)
    return RobustnessCourseOutcome(
        course_dir=course_dir,
        patient_id=course_dir.parent.name,
        course_id=course_dir.name,
        run_identifier=run_identifier,
        measurement_outcome=outcome,
        output_name=output_name,
        dispositions_path=dispositions_path,
        dispositions_sha256=_file_sha256(dispositions_path),
        source_disposition_count=len(rows),
        effective_configuration_sha256=_content_sha256(
            effective_robustness_configuration(rob_config, output_name=output_name)
        ),
    )


@dataclass(frozen=True)
class _AdmittedRobustnessInput:
    """One per-course table that the current context is allowed to aggregate.

    ``frame`` was parsed from ``table_sha256``'s exact bytes, so everything the
    cohort summary says is bound to the bytes admission certified. Re-reading
    the file later would reopen the gap this closes.
    """

    path: Path
    course_dir: Path
    patient_id: str
    course_id: str
    run_identifier: str
    table_sha256: str
    frame: pd.DataFrame
    source_dispositions: List[Dict[str, Any]]


def _read_robustness_input_bytes(path: Path) -> Tuple[str, pd.DataFrame]:
    """Read one aggregation input exactly once and hash what was read."""
    from io import BytesIO

    if not path.exists():
        raise FileNotFoundError(f"robustness input parquet does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(
            f"robustness input parquet does not exist as a readable file: {path}"
        )
    data = path.read_bytes()
    try:
        frame = pd.read_parquet(BytesIO(data))
    except Exception as e:
        raise RuntimeError(f"failed to read robustness input {path}: {e}") from e
    return hashlib.sha256(data).hexdigest(), frame


def _admitted_table_identity(
    frame: pd.DataFrame, path: Path
) -> Tuple[str, str, str]:
    """The patient, course and run identity the table itself declares.

    A table with no readable run identity cannot be matched to the run-bound
    disposition sidecar at all, and a table carrying several runs is not one
    run's product; neither may be admitted on the strength of its filename.
    """

    def _readable(column: str) -> set[str]:
        if column not in frame.columns:
            raise RuntimeError(
                f"robustness input {path} has no {column} column; a table that "
                "does not declare its own identity cannot be admitted"
            )
        return {
            text
            for text in frame[column].dropna().astype(str).str.strip().unique()
            if text and text.lower() not in {"nan", "none", "<na>"}
        }

    runs = _readable("run_identifier")
    if not runs:
        raise RuntimeError(
            f"robustness input {path} carries no readable robustness run identity"
        )
    if len(runs) > 1:
        raise RuntimeError(
            f"robustness input {path} carries {len(runs)} robustness run "
            f"identities {sorted(runs)}; one course table is the product of "
            "exactly one run"
        )
    patients = _readable("patient_id")
    courses = _readable("course_id")
    if len(patients) != 1 or len(courses) != 1:
        raise RuntimeError(
            f"robustness input {path} carries {len(patients)} patient and "
            f"{len(courses)} course identities; one course table describes "
            "exactly one course"
        )
    return patients.pop(), courses.pop(), runs.pop()


def _verify_admitted_table_bytes(
    course_dir: Path, path: Path, table_sha256: str
) -> None:
    """Bind the bytes this aggregation parsed to the certified measured output.

    The disposition loader re-hashes whatever is on disk when it runs. This
    additionally requires the certified digest to equal the digest of the bytes
    that were actually read for the summary, so an input replaced between the
    two reads cannot be summarised as if it had been certified.
    """
    payload = json.loads(
        robustness_source_dispositions_path(course_dir).read_text(encoding="utf-8")
    )
    outcome = payload.get("measurement_outcome")
    if outcome != ROBUSTNESS_MEASURED_OUTCOME:
        raise ValueError(
            f"robustness source dispositions for {course_dir} record outcome "
            f"{outcome!r}, which certifies no measurement table"
        )
    measured_output = payload.get("measured_output")
    if not isinstance(measured_output, dict):
        raise ValueError(
            f"robustness source dispositions for {course_dir} bind no measured output"
        )
    if str(measured_output.get("path") or "") != path.name:
        raise ValueError(
            f"robustness source dispositions for {course_dir} certify "
            f"{measured_output.get('path')!r}, not the aggregated {path.name!r}"
        )
    recorded = str(measured_output.get("sha256") or "")
    if recorded != table_sha256:
        raise ValueError(
            f"robustness input {path} changed after publication (recorded sha256 "
            f"{recorded!r}, aggregated {table_sha256!r})"
        )


def _admit_robustness_aggregation_input(
    parquet_path: Path, rob_config: RobustnessConfig
) -> _AdmittedRobustnessInput:
    """Admit one per-course table into this aggregation's current context.

    Admission is the same run/source/code/configuration binding the course
    published under, re-checked against the code and configuration now on
    disk, plus the filename and byte identity of this specific input. A course
    whose sidecar is missing, unreadable, stale, foreign or bound to a
    different table is rejected; it is not silently summarised.

    Assurance and its limit. This certifies which bytes were aggregated and
    which recorded code and configuration decided them. It does not certify the
    code the interpreter executed, nor the DICOM to NIfTI derivation upstream
    of the table; those remain open provenance gaps outside this module.
    """
    path = Path(parquet_path)
    table_sha256, frame = _read_robustness_input_bytes(path)

    # A table that declares its own technical failures is rejected on its own
    # evidence first, so the failed course is named rather than the sidecar it
    # was never allowed to publish.
    if (
        "robustness_status" in frame
        and frame["robustness_status"].eq("technical_failure").any()
    ):
        raise RuntimeError(
            "technical robustness failures require recovery before aggregation: "
            f"{path}"
        )

    patient_id, course_id, run_identifier = _admitted_table_identity(frame, path)
    course_dir = path.parent
    if (patient_id, course_id) != (course_dir.parent.name, course_dir.name):
        raise RuntimeError(
            f"robustness input {path} declares {patient_id}/{course_id} and "
            f"does not belong to course {course_dir.parent.name}/{course_dir.name}"
        )

    source_dispositions = load_robustness_source_dispositions(
        course_dir,
        run_identifier=run_identifier,
        rob_config=rob_config,
        output_name=path.name,
    )
    _verify_admitted_table_bytes(course_dir, path, table_sha256)

    return _AdmittedRobustnessInput(
        path=path,
        course_dir=course_dir,
        patient_id=patient_id,
        course_id=course_id,
        run_identifier=run_identifier,
        table_sha256=table_sha256,
        frame=frame,
        source_dispositions=source_dispositions,
    )


def _admit_robustness_aggregation_inputs(
    input_parquets: List[Path], rob_config: RobustnessConfig
) -> List[_AdmittedRobustnessInput]:
    """Admit every input, refusing to count one course twice."""
    admitted: List[_AdmittedRobustnessInput] = []
    seen_paths: Dict[str, Path] = {}
    seen_courses: Dict[Tuple[str, str], Path] = {}
    seen_tables: Dict[str, Path] = {}
    for parquet_path in input_parquets:
        entry = _admit_robustness_aggregation_input(parquet_path, rob_config)
        for registry, key, repeated in (
            (seen_paths, str(entry.path.resolve()), "input path"),
            (seen_courses, (entry.patient_id, entry.course_id), "course identity"),
            (seen_tables, entry.table_sha256, "table content"),
        ):
            previous = registry.get(key)
            if previous is not None:
                raise RuntimeError(
                    f"duplicate robustness aggregation input: {entry.path} repeats "
                    f"the {repeated} of {previous}; one course contributes to the "
                    "cohort exactly once"
                )
            registry[key] = entry.path
        admitted.append(entry)

    if not admitted:
        raise RuntimeError("no robustness results were supplied for aggregation")
    return admitted


def _publish_robustness_cohort_outputs(
    *,
    output_excel: Path,
    raw_parquet_path: Path,
    raw_frame: pd.DataFrame,
    sheets: List[Tuple[str, pd.DataFrame]],
    verify_evidence=None,
) -> None:
    """Publish the workbook and the raw values as one cohort result.

    Both files are built beside their destinations and only then moved into
    place, so an aggregation that fails in this process leaves neither a
    nominal-looking workbook nor a nominal raw Parquet that no summary
    certifies. Any output that cannot be withdrawn is named in the log rather
    than passed over.

    Limit, stated rather than implied: two files cannot be moved into place by
    one atomic operation. A process killed between the two ``os.replace`` calls
    can leave a current raw Parquet beside an absent or previous workbook.
    Aggregation deletes both destinations before it starts, so the next run
    cannot inherit that state silently, but a crash in this window is real
    residue and not something this function can prevent.
    """
    temporary_raw = raw_parquet_path.with_name(raw_parquet_path.name + ".tmp")
    temporary_excel = output_excel.with_name(output_excel.name + ".tmp")
    from .course_manifest import require_no_output_symlinks
    for path in (output_excel, raw_parquet_path, temporary_raw, temporary_excel):
        require_no_output_symlinks(path)
    try:
        # Raw values go to parquet (no row limit, unlike Excel's 1,048,576).
        raw_frame.to_parquet(temporary_raw, index=False)

        with pd.ExcelWriter(temporary_excel, engine="openpyxl") as writer:
            for sheet_name, frame in sheets:
                frame.to_excel(writer, sheet_name=sheet_name, index=False)
            # Note: Raw values saved to parquet file (see caller's log)
            # Excel has 1,048,576 row limit; large datasets exceed this

        if verify_evidence is not None:
            verify_evidence()
        for path in (output_excel, raw_parquet_path, temporary_raw, temporary_excel):
            require_no_output_symlinks(path)
        os.replace(temporary_raw, raw_parquet_path)
        os.replace(temporary_excel, output_excel)
    except BaseException as e:
        residual = []
        for withdrawn in (
            temporary_raw,
            temporary_excel,
            raw_parquet_path,
            output_excel,
        ):
            try:
                withdrawn.unlink(missing_ok=True)
            except OSError as unlink_error:
                residual.append(f"{withdrawn} ({unlink_error})")
        logger.error("Failed to write aggregated results: %s", e)
        if residual:
            logger.error(
                "Residual cohort output(s) could not be withdrawn: %s",
                ", ".join(residual),
            )
        raise


def aggregate_robustness_results(
    input_parquets: List[Path],
    output_excel: Path,
    rob_config: RobustnessConfig,
) -> None:
    """
    Aggregate per-course robustness results into cohort-level summary.

    Only tables admitted by :func:`_admit_robustness_aggregation_input` are
    summarised: each input must still be the run-bound, code-bound,
    configuration-bound and byte-bound measurement its course published. A
    course that cannot be admitted fails the cohort; it is never turned into a
    clinical exclusion and never shrinks the denominator silently.

    Args:
        input_parquets: List of per-course parquet files
        output_excel: Output Excel file path
        rob_config: Robustness configuration
    """
    logger.info("Aggregating robustness results from %d courses", len(input_parquets))
    output_excel = Path(output_excel)
    withdraw_robustness_cohort_outputs(output_excel)
    raw_parquet_path = output_excel.parent / (output_excel.stem + "_raw_values.parquet")

    admitted = _admit_robustness_aggregation_inputs(input_parquets, rob_config)
    logger.info(
        "Admitted %d certified robustness course table(s): %s",
        len(admitted),
        ", ".join(
            f"{entry.patient_id}/{entry.course_id}@{entry.run_identifier}"
            for entry in admitted
        ),
    )

    combined_raw = pd.concat(
        [entry.frame for entry in admitted], ignore_index=True
    )

    # Normalize column names
    if "roi_name" in combined_raw.columns and "structure" not in combined_raw.columns:
        combined_raw.rename(columns={"roi_name": "structure"}, inplace=True)

    if "robustness_status" in combined_raw and combined_raw.robustness_status.eq("technical_failure").any():
        raise RuntimeError("technical robustness failures require recovery before aggregation")
    _validate_cohort_feature_sets(combined_raw)
    per_structure_summary = summarize_feature_stability(combined_raw, rob_config)
    # Keep the historical sheet name for compatibility, but never pool raw
    # values across heterogeneous structures or segmentation sources. Each row
    # remains a structure/source/feature estimate.
    global_summary = per_structure_summary.copy()

    per_source_summary: Optional[pd.DataFrame] = None
    if "segmentation_source" in combined_raw.columns:
        per_source_columns = ["segmentation_source", "structure"]
        if "extraction_arm" in combined_raw.columns:
            per_source_columns.append("extraction_arm")
        per_source_columns.append("feature_name")
        per_source_summary = summarize_feature_stability(
            combined_raw,
            rob_config,
            group_columns=per_source_columns,
        )

    if global_summary.empty:
        robust_features = pd.DataFrame(columns=global_summary.columns)
        acceptable_features = pd.DataFrame(columns=global_summary.columns)
    else:
        robust_features = global_summary[global_summary["robustness_label"] == "robust"]
        acceptable_features = global_summary[global_summary["pass_seg_perturb"]]

    if per_source_summary is not None and not per_source_summary.empty:
        robust_per_source = per_source_summary[per_source_summary["robustness_label"] == "robust"]
    else:
        robust_per_source = None

    output_excel.parent.mkdir(parents=True, exist_ok=True)

    sheets: List[Tuple[str, pd.DataFrame]] = [("global_summary", global_summary)]
    if per_source_summary is not None:
        sheets.append(("per_source_summary", per_source_summary))
    sheets.append(("per_structure_source", per_structure_summary))
    sheets.append(("robust_features", robust_features))
    sheets.append(("acceptable_features", acceptable_features))
    if robust_per_source is not None:
        sheets.append(("robust_features_per_source", robust_per_source))
    _publish_robustness_cohort_outputs(
        output_excel=output_excel,
        raw_parquet_path=raw_parquet_path,
        raw_frame=combined_raw,
        sheets=sheets,
    )

    logger.info(
        "Saved raw robustness values to %s (%d rows)",
        raw_parquet_path,
        len(combined_raw),
    )
    logger.info(
        "Saved aggregated robustness results to %s (features=%d, structures=%d)",
        output_excel,
        len(global_summary),
        len(per_structure_summary),
    )
    if not global_summary.empty:
        logger.info(
            "Global summary: %d robust, %d acceptable",
            len(robust_features),
            len(acceptable_features),
        )
    if per_source_summary is not None:
        logger.info(
            "Per-source summary: %d combinations",
            len(per_source_summary),
        )


# ============================================================================
# Manifest-driven cohort consumption
#
# The measured-table entry point above answers "summarise these tables". This
# section answers the question an all-patient cohort actually asks: "account
# for every course organize validated". The manifest is the authority for that
# denominator, so every course it names must produce a revalidated receipt, and
# a course that measured nothing is carried as an explicit accounting record
# rather than disappearing from the cohort.
# ============================================================================

# The canonical raw robustness value schema. It is written even when a cohort
# measured nothing, so a consumer of the raw Parquet always meets the same
# columns and can tell "no measurements" from "a different artifact".
ROBUSTNESS_RAW_VALUE_COLUMNS: Tuple[str, ...] = (
    *ROBUSTNESS_SOURCE_IDENTITY_COLUMNS,
    "structure",
    "modality",
    *ROBUSTNESS_PERTURBATION_IDENTITY_COLUMNS,
    "extraction_arm",
    "run_identifier",
    "feature_name",
    "value",
    "robustness_status",
)
_ROBUSTNESS_RAW_FLOAT_COLUMNS = frozenset({"value"})

# The columns :func:`summarize_feature_stability` produces for a non-empty
# cohort, used to keep an empty summary sheet readable instead of blank.
ROBUSTNESS_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "structure",
    "segmentation_source",
    "extraction_arm",
    "feature_name",
    "n_subjects",
    "n_subjects_complete",
    "n_subjects_dropped",
    "n_courses",
    "n_perturbations",
    "icc",
    "icc_ci95_low",
    "icc_ci95_high",
    "cov_pct",
    "cov_pct_q1",
    "cov_pct_q3",
    "n_subjects_cov",
    "cov_status",
    "qcd",
    "qcd_q1",
    "qcd_q3",
    "n_subjects_qcd",
    "qcd_status",
    "robustness_label",
    "pass_seg_perturb",
)

ROBUSTNESS_COURSE_OUTCOME_COLUMNS: Tuple[str, ...] = (
    "patient_id",
    "course_id",
    "measurement_outcome",
    "robustness_run_identifier",
    "output_path",
    "table_present",
    "table_sha256",
    "table_row_count",
    "measured_value_row_count",
    "geometric_nonmeasurement_row_count",
    "other_status_row_count",
    "contributes_measurements",
    "source_disposition_count",
    "source_dispositions_sha256",
    "effective_configuration_sha256",
)


@dataclass(frozen=True)
class RobustnessCohortCourse:
    """One manifest course, admitted with everything its receipt binds.

    ``frame`` is present only for a measured course, and holds the exact bytes
    admission certified. A source-only course carries its dispositions and no
    frame: it measured nothing, and nothing here may invent perturbations,
    voxels or features for it.
    """

    patient_id: str
    course_id: str
    course_dir: Path
    run_identifier: str
    measurement_outcome: str
    output_name: str
    source_dispositions: List[Dict[str, Any]]
    dispositions_sha256: str
    effective_configuration_sha256: str
    measured_output: Optional[Path] = None
    measured_output_sha256: Optional[str] = None
    frame: Optional[pd.DataFrame] = None
    sidecar_snapshot: bytes = b""
    receipt_snapshot: bytes = b""
    table_snapshot: Optional[bytes] = None

    @property
    def measured(self) -> bool:
        return self.measurement_outcome == ROBUSTNESS_MEASURED_OUTCOME


def admit_robustness_cohort_course(
    course_dir: Path,
    *,
    patient_id: str,
    course_id: str,
    rob_config: RobustnessConfig,
) -> RobustnessCohortCourse:
    """Admit one manifest course from its completion receipt.

    Every binding the course published is re-checked here against the current
    configuration and the artifacts on disk: the receipt's own identity and
    digests, the disposition sidecar under the current perturbation grid and
    thresholds, and — for a measured course — the table bytes, its declared
    run/patient/course identity and the sidecar's certification of exactly
    that file. A legacy ``ok`` sentinel is rejected, not migrated.
    """
    from .robustness_completion import (
        read_robustness_completion_sentinel,
        robustness_completion_sentinel_path,
        RobustnessCompletionError,
    )

    from .course_manifest import require_no_output_symlinks

    course_dir = Path(course_dir)
    require_no_output_symlinks(course_dir)
    require_no_output_symlinks(robustness_source_dispositions_path(course_dir))
    require_no_output_symlinks(robustness_completion_sentinel_path(course_dir))
    try:
        receipt_snapshot = robustness_completion_sentinel_path(course_dir).read_bytes()
    except FileNotFoundError as exc:
        raise RobustnessCompletionError(
            f"no robustness completion receipt at {robustness_completion_sentinel_path(course_dir)}"
        ) from exc
    receipt = read_robustness_completion_sentinel(
        robustness_completion_sentinel_path(course_dir), course_dir=course_dir
    )
    if (receipt.patient_id, receipt.course_id) != (str(patient_id), str(course_id)):
        raise RuntimeError(
            f"robustness completion receipt in {course_dir} certifies "
            f"{receipt.patient_id}/{receipt.course_id}, not the manifest course "
            f"{patient_id}/{course_id}"
        )
    expected_configuration = _content_sha256(
        effective_robustness_configuration(
            rob_config, output_name=receipt.output_name
        )
    )
    if receipt.effective_configuration_sha256 != expected_configuration:
        raise RuntimeError(
            f"robustness completion receipt in {course_dir} was recorded under "
            f"effective configuration {receipt.effective_configuration_sha256!r}, "
            f"not the current {expected_configuration!r}"
        )
    if rob_config is None:
        raise TypeError("cohort admission requires the current rob_config")
    canonical_path = robustness_source_dispositions_path(course_dir)
    if receipt.dispositions_path != canonical_path:
        raise ValueError("completion receipt must bind the canonical disposition sidecar")
    snapshot = canonical_path.read_bytes()
    if hashlib.sha256(snapshot).hexdigest() != receipt.dispositions_sha256:
        raise ValueError("canonical disposition sidecar changed after receipt validation")
    inspection = _read_robustness_source_dispositions(
        course_dir,
        run_identifier=receipt.run_identifier,
        rob_config=rob_config,
        output_name=receipt.output_name,
        snapshot=snapshot,
    )
    if inspection.measurement_outcome != receipt.measurement_outcome:
        raise ValueError("canonical disposition sidecar outcome differs from completion receipt")
    rows = inspection.rows
    if len(rows) != receipt.source_disposition_count:
        raise RuntimeError(
            f"robustness completion receipt in {course_dir} counts "
            f"{receipt.source_disposition_count} source disposition(s), but the "
            f"sidecar carries {len(rows)}"
        )

    if not receipt.measured:
        return RobustnessCohortCourse(
            patient_id=receipt.patient_id,
            course_id=receipt.course_id,
            course_dir=course_dir,
            run_identifier=receipt.run_identifier,
            measurement_outcome=receipt.measurement_outcome,
            output_name=receipt.output_name,
            source_dispositions=rows,
            dispositions_sha256=receipt.dispositions_sha256,
            effective_configuration_sha256=expected_configuration,
            sidecar_snapshot=snapshot,
            receipt_snapshot=receipt_snapshot,
        )

    table_path = course_dir / receipt.output_name
    require_no_output_symlinks(table_path)
    admitted = _admit_robustness_aggregation_input(table_path, rob_config)
    if admitted.run_identifier != receipt.run_identifier:
        raise RuntimeError(
            f"robustness table {table_path} declares run "
            f"{admitted.run_identifier!r}, but its completion receipt certifies "
            f"{receipt.run_identifier!r}"
        )
    if admitted.table_sha256 != receipt.measured_output_sha256:
        raise RuntimeError(
            f"robustness table {table_path} does not carry the bytes its "
            f"completion receipt certified (receipt "
            f"{receipt.measured_output_sha256!r}, aggregated "
            f"{admitted.table_sha256!r})"
        )
    if (admitted.patient_id, admitted.course_id) != (
        receipt.patient_id,
        receipt.course_id,
    ):
        raise RuntimeError(
            f"robustness table {table_path} declares "
            f"{admitted.patient_id}/{admitted.course_id}, not the certified "
            f"{receipt.patient_id}/{receipt.course_id}"
        )
    table_snapshot = table_path.read_bytes()
    if hashlib.sha256(table_snapshot).hexdigest() != admitted.table_sha256:
        raise ValueError("measurement table changed during admission")
    if admitted.source_dispositions != rows:
        raise ValueError("disposition rows changed during admission")
    return RobustnessCohortCourse(
        patient_id=receipt.patient_id,
        course_id=receipt.course_id,
        course_dir=course_dir,
        run_identifier=receipt.run_identifier,
        measurement_outcome=receipt.measurement_outcome,
        output_name=receipt.output_name,
        source_dispositions=admitted.source_dispositions,
        dispositions_sha256=receipt.dispositions_sha256,
        effective_configuration_sha256=expected_configuration,
        measured_output=table_path,
        measured_output_sha256=admitted.table_sha256,
        frame=admitted.frame,
        sidecar_snapshot=snapshot,
        receipt_snapshot=receipt_snapshot,
        table_snapshot=table_snapshot,
    )


def _verify_cohort_snapshot(courses, cohort, rob_config):
    """Revalidate source evidence and public views without replacing certified data."""
    from io import BytesIO
    from .course_manifest import parse_course_manifest, require_no_output_symlinks
    from .robustness_completion import robustness_completion_sentinel_path

    snapshot = cohort.get("manifest_snapshot")
    if not isinstance(snapshot, bytes):
        raise ValueError("cohort requires a current-schema manifest snapshot")
    manifest = Path(cohort["manifest_path"])
    require_no_output_symlinks(manifest)
    if manifest.read_bytes() != snapshot:
        raise ValueError("course manifest changed after admission")
    entries, current = parse_course_manifest(
        json.loads(snapshot), output_dir=Path(cohort["output_root"]),
        manifest_path=manifest, require_current_schema=True,
    )
    if current != dict(cohort):
        raise ValueError("cohort manifest metadata was mutated")
    actual = {(c.patient_id, c.course_id, c.course_dir.absolute()) for c in courses}
    expected = {(p, c, d.absolute()) for p, c, d in entries}
    if actual != expected or len(courses) != len(entries):
        raise ValueError("admitted courses differ from exact manifest membership")
    for course in courses:
        for path in (course.course_dir, robustness_source_dispositions_path(course.course_dir),
                     robustness_completion_sentinel_path(course.course_dir),
                     course.course_dir / course.output_name):
            require_no_output_symlinks(path)
        if robustness_completion_sentinel_path(course.course_dir).read_bytes() != course.receipt_snapshot:
            raise ValueError("completion receipt changed after admission")
        if robustness_source_dispositions_path(course.course_dir).read_bytes() != course.sidecar_snapshot:
            raise ValueError("canonical sidecar changed after admission")
        fresh = admit_robustness_cohort_course(
            course.course_dir, patient_id=course.patient_id, course_id=course.course_id,
            rob_config=rob_config,
        )
        for field in ("run_identifier", "measurement_outcome", "output_name",
                      "dispositions_sha256", "effective_configuration_sha256",
                      "measured_output", "measured_output_sha256", "table_snapshot"):
            if getattr(fresh, field) != getattr(course, field):
                raise ValueError(f"admitted course snapshot changed: {field}")
        if fresh.source_dispositions != course.source_dispositions:
            raise ValueError("admitted disposition view was mutated")
        if course.table_snapshot is None:
            if course.frame is not None:
                raise ValueError("source-only course acquired a measurement frame")
        elif course.frame is None or not course.frame.equals(pd.read_parquet(BytesIO(course.table_snapshot))):
            raise ValueError("admitted measurement frame was mutated")


def _empty_robustness_raw_frame() -> pd.DataFrame:
    """The raw value schema with no rows: an explicit zero, not a blank file."""
    return pd.DataFrame(
        {
            column: pd.Series(
                dtype="float64" if column in _ROBUSTNESS_RAW_FLOAT_COLUMNS else "object"
            )
            for column in ROBUSTNESS_RAW_VALUE_COLUMNS
        }
    )


def _robustness_course_outcome_rows(
    courses: List[RobustnessCohortCourse],
) -> pd.DataFrame:
    """One row per admitted manifest course: the cohort's fixed denominator.

    A table-bearing course and a measured value are counted separately, because
    a published table can also carry geometric non-measurement rows. Collapsing
    the two would let a course that measured nothing be reported as measured.
    """
    rows: List[Dict[str, Any]] = []
    for course in courses:
        table_rows = 0
        measured_rows = 0
        geometric_rows = 0
        other_rows = 0
        if course.frame is not None:
            frame = course.frame
            table_rows = int(len(frame))
            if "robustness_status" in frame.columns:
                status = frame["robustness_status"].astype("string").fillna("measured")
            else:
                status = pd.Series(
                    ["measured"] * table_rows, index=frame.index, dtype="string"
                )
            is_measured = status.eq(ROBUSTNESS_MEASURED_OUTCOME)
            if "value" in frame.columns:
                is_measured &= frame["value"].notna()
            measured_rows = int(is_measured.sum())
            geometric_rows = int(status.eq("geometrically_impossible").sum())
            other_rows = int(table_rows - measured_rows - geometric_rows)
        rows.append(
            {
                "patient_id": course.patient_id,
                "course_id": course.course_id,
                "measurement_outcome": course.measurement_outcome,
                "robustness_run_identifier": course.run_identifier,
                "output_path": course.output_name,
                "table_present": course.frame is not None,
                "table_sha256": course.measured_output_sha256,
                "table_row_count": table_rows,
                "measured_value_row_count": measured_rows,
                "geometric_nonmeasurement_row_count": geometric_rows,
                "other_status_row_count": other_rows,
                "contributes_measurements": bool(measured_rows),
                "source_disposition_count": len(course.source_dispositions),
                "source_dispositions_sha256": course.dispositions_sha256,
                "effective_configuration_sha256": course.effective_configuration_sha256,
            }
        )
    return pd.DataFrame(rows, columns=list(ROBUSTNESS_COURSE_OUTCOME_COLUMNS))


def _robustness_source_disposition_rows(
    courses: List[RobustnessCohortCourse],
) -> pd.DataFrame:
    """Every admitted course's source dispositions, including measured courses.

    A non-volumetric source ROI in a course that also measured something is the
    same kind of fact as one in a course that measured nothing. Publishing only
    the second would make a mixed course look like it had no such ROI.
    """
    rows: List[Dict[str, Any]] = []
    for course in courses:
        for disposition in course.source_dispositions:
            row = {
                "patient_id": course.patient_id,
                "course_id": course.course_id,
                "robustness_run_identifier": course.run_identifier,
                "course_measurement_outcome": course.measurement_outcome,
            }
            row.update({key: disposition.get(key) for key in disposition})
            rows.append(row)
    columns = [
        "patient_id",
        "course_id",
        "robustness_run_identifier",
        "course_measurement_outcome",
        *ROBUSTNESS_SOURCE_DISPOSITION_ROW_FIELDS,
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    ordered = [column for column in columns if column in frame.columns]
    remaining = [column for column in frame.columns if column not in ordered]
    return frame[ordered + remaining]


def robustness_cohort_output_paths(output_excel: Path) -> Tuple[Path, Path]:
    """The workbook and the raw-value Parquet that form one cohort result."""
    output_excel = Path(output_excel)
    return output_excel, output_excel.parent / (
        output_excel.stem + "_raw_values.parquet"
    )


def withdraw_robustness_cohort_outputs(output_excel: Path) -> None:
    """Remove any nominal cohort outputs, whatever produced them.

    A cohort attempt that cannot be completed must not leave the previous
    attempt's workbook standing where a reader would take it for this one.
    """
    from .course_manifest import require_no_output_symlinks

    paths = robustness_cohort_output_paths(output_excel)
    for path in paths:
        require_no_output_symlinks(path)
    for path in paths:
        path.unlink(missing_ok=True)


def _require_no_unresolved_quarantine(
    cohort: Mapping[str, Any], output_excel: Path
) -> None:
    """An unresolved technical quarantine blocks the nominal cohort output.

    This mode reports every course organize validated. A course organize could
    not validate is a technical failure awaiting repair, not a smaller cohort,
    so the nominal workbook is withheld while any remain. The manifest keeps
    the quarantine evidence either way; nothing here rewrites or clears it.
    """
    quarantines = cohort.get("technical_quarantines") or []
    count = int(cohort.get("technical_quarantine_count") or 0)
    if not count and not quarantines:
        return
    named = ", ".join(
        f"{entry.get('patient')}/{entry.get('course')}: {entry.get('reason')}"
        for entry in quarantines
        if isinstance(entry, Mapping)
    )
    raise RuntimeError(
        f"refusing to publish {output_excel}: the course manifest still records "
        f"{count} unresolved technical quarantine(s) [{named}]; a technically "
        "quarantined course is unfinished work, not a cohort exclusion"
    )


def aggregate_robustness_cohort(
    courses: List[RobustnessCohortCourse],
    output_excel: Path,
    rob_config: RobustnessConfig,
    *,
    cohort: Mapping[str, Any],
) -> None:
    """Publish the cohort robustness result for every admitted manifest course.

    Measured courses go through the same summary path as the explicit-input
    aggregation: identical admission, identical ICC/CoV/QCD, identical fixed
    per-course denominator and identity linkage. Source-only courses are
    accounted for in ``course_outcomes`` and ``source_dispositions`` and never
    enter the measurement path, so no perturbation, voxel or feature is ever
    fabricated for a course that measured nothing.

    A cohort in which every course is source-only is a completed accounting
    artifact with an explicit zero: empty, schema-preserving raw values and no
    ICC estimates. That is a statement about what the sources contained; it is
    not analysis-ready stability evidence, and nothing here converts it into
    one.
    """
    output_excel, raw_parquet_path = robustness_cohort_output_paths(output_excel)
    withdraw_robustness_cohort_outputs(output_excel)

    if not courses:
        raise RuntimeError(
            "the course manifest names no course; a cohort robustness summary "
            "requires at least one accounted course"
        )
    _require_no_unresolved_quarantine(cohort, output_excel)
    _verify_cohort_snapshot(courses, cohort, rob_config)

    validated = cohort.get("validated_course_count")
    if validated is not None and int(validated) != len(courses):
        raise RuntimeError(
            f"admitted {len(courses)} course(s) for a manifest that validated "
            f"{int(validated)}; the cohort denominator is fixed by the manifest"
        )

    seen_courses: Dict[Tuple[str, str], Path] = {}
    seen_tables: Dict[str, Path] = {}
    for course in courses:
        key = (course.patient_id, course.course_id)
        previous = seen_courses.get(key)
        if previous is not None:
            raise RuntimeError(
                f"duplicate robustness cohort course {course.patient_id}/"
                f"{course.course_id}: {course.course_dir} repeats {previous}"
            )
        seen_courses[key] = course.course_dir
        if course.measured_output_sha256 is not None:
            previous_table = seen_tables.get(course.measured_output_sha256)
            if previous_table is not None:
                raise RuntimeError(
                    f"duplicate robustness measurement table content: "
                    f"{course.measured_output} repeats {previous_table}; one "
                    "course contributes to the cohort exactly once"
                )
            seen_tables[course.measured_output_sha256] = course.measured_output

    measured = [course for course in courses if course.frame is not None]
    course_outcomes = _robustness_course_outcome_rows(courses)
    source_dispositions = _robustness_source_disposition_rows(courses)

    logger.info(
        "Accounting for %d manifest course(s): %d measured, %d source-only",
        len(courses),
        len(measured),
        len(courses) - len(measured),
    )

    per_source_summary: Optional[pd.DataFrame] = None
    robust_per_source: Optional[pd.DataFrame] = None
    if measured:
        combined_raw = pd.concat(
            [course.frame for course in measured], ignore_index=True
        )
        if "roi_name" in combined_raw.columns and "structure" not in combined_raw.columns:
            combined_raw.rename(columns={"roi_name": "structure"}, inplace=True)
        if (
            "robustness_status" in combined_raw
            and combined_raw.robustness_status.eq("technical_failure").any()
        ):
            raise RuntimeError(
                "technical robustness failures require recovery before aggregation"
            )
        _validate_cohort_feature_sets(combined_raw)
        per_structure_summary = summarize_feature_stability(combined_raw, rob_config)
        global_summary = per_structure_summary.copy()
        if "segmentation_source" in combined_raw.columns:
            per_source_columns = ["segmentation_source", "structure"]
            if "extraction_arm" in combined_raw.columns:
                per_source_columns.append("extraction_arm")
            per_source_columns.append("feature_name")
            per_source_summary = summarize_feature_stability(
                combined_raw, rob_config, group_columns=per_source_columns
            )
        if global_summary.empty:
            robust_features = pd.DataFrame(columns=global_summary.columns)
            acceptable_features = pd.DataFrame(columns=global_summary.columns)
        else:
            robust_features = global_summary[
                global_summary["robustness_label"] == "robust"
            ]
            acceptable_features = global_summary[global_summary["pass_seg_perturb"]]
        if per_source_summary is not None and not per_source_summary.empty:
            robust_per_source = per_source_summary[
                per_source_summary["robustness_label"] == "robust"
            ]
    else:
        # Nothing measured anywhere. The raw values keep their schema and stay
        # empty, and no summary row is invented for an estimate nobody made.
        combined_raw = _empty_robustness_raw_frame()
        empty_summary = pd.DataFrame(columns=list(ROBUSTNESS_SUMMARY_COLUMNS))
        per_structure_summary = empty_summary
        global_summary = empty_summary.copy()
        robust_features = empty_summary.copy()
        acceptable_features = empty_summary.copy()

    sheets: List[Tuple[str, pd.DataFrame]] = [("global_summary", global_summary)]
    if per_source_summary is not None:
        sheets.append(("per_source_summary", per_source_summary))
    sheets.append(("per_structure_source", per_structure_summary))
    sheets.append(("robust_features", robust_features))
    sheets.append(("acceptable_features", acceptable_features))
    if robust_per_source is not None:
        sheets.append(("robust_features_per_source", robust_per_source))
    sheets.append(("course_outcomes", course_outcomes))
    sheets.append(("source_dispositions", source_dispositions))

    # Recheck after summarization, immediately before publication. Filesystem
    # changes after this check and multi-file crash atomicity remain unavoidable
    # without external locking/transactional storage.
    _verify_cohort_snapshot(courses, cohort, rob_config)
    from .course_manifest import require_no_output_symlinks
    require_no_output_symlinks(output_excel)
    require_no_output_symlinks(raw_parquet_path)
    output_excel.parent.mkdir(parents=True, exist_ok=True)
    _publish_robustness_cohort_outputs(
        output_excel=output_excel,
        raw_parquet_path=raw_parquet_path,
        raw_frame=combined_raw,
        sheets=sheets,
        verify_evidence=lambda: _verify_cohort_snapshot(courses, cohort, rob_config),
    )

    logger.info(
        "Saved raw robustness values to %s (%d rows)",
        raw_parquet_path,
        len(combined_raw),
    )
    logger.info(
        "Saved cohort robustness accounting to %s (courses=%d, measured courses=%d, "
        "measured value rows=%d, source dispositions=%d, summary rows=%d)",
        output_excel,
        len(course_outcomes),
        len(measured),
        int(course_outcomes["measured_value_row_count"].sum()),
        len(source_dispositions),
        len(global_summary),
    )
    if not measured:
        logger.warning(
            "No manifest course published a robustness measurement: %s is a "
            "completed accounting artifact with zero measurements and no ICC "
            "estimates, not analysis-ready stability evidence",
            output_excel,
        )
