"""Memory-bounded cohort aggregation, outside the producer's code identity.

Admission and statistics remain in radiomics_robustness. Certified frames are
staged one course at a time; only dictionary-encoded metric columns survive
between courses. Publication uses the existing paired-output transaction.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import radiomics_robustness as rr
from .course_manifest import parse_course_manifest, require_no_output_symlinks


@dataclass(frozen=True)
class RobustnessCourseReference:
    """Manifest membership only; admission happens during aggregation."""

    course_dir: Path
    patient_id: str
    course_id: str


def robustness_course_reference(course_dir, *, patient_id, course_id):
    return RobustnessCourseReference(Path(course_dir), patient_id, course_id)


def _verify_manifest(courses, cohort):
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


def _check_snapshot(expected, fresh, *, check_frame=False):
    for field in (
        "run_identifier", "measurement_outcome", "output_name",
        "dispositions_sha256", "effective_configuration_sha256",
        "measured_output", "measured_output_sha256", "sidecar_snapshot",
        "receipt_snapshot",
    ):
        if getattr(expected, field) != getattr(fresh, field):
            raise ValueError(f"admitted course snapshot changed: {field}")
    if expected.source_dispositions != fresh.source_dispositions:
        raise ValueError("admitted disposition view was mutated")
    if check_frame:
        if expected.table_snapshot != fresh.table_snapshot:
            raise ValueError("admitted course snapshot changed: table_snapshot")
        if expected.table_snapshot is None:
            if expected.frame is not None:
                raise ValueError("source-only course acquired a measurement frame")
        elif expected.frame is None or not expected.frame.equals(
            pd.read_parquet(BytesIO(expected.table_snapshot))
        ):
            raise ValueError("admitted measurement frame was mutated")


def _prototype(frame):
    """A dtype/all-null representative for pandas' cross-course concat rules."""
    if frame.empty:
        return frame.iloc[:0].copy()
    columns = []
    for name in frame:
        series = frame[name]
        valid = series.first_valid_index()
        columns.append(series.loc[[valid]].reset_index(drop=True) if valid is not None
                       else series.iloc[:1].reset_index(drop=True))
    return pd.concat(columns, axis=1)


def _dictionary_table(frame, schema):
    table = pa.Table.from_pandas(frame, preserve_index=False,
                                schema=pa.schema([schema.field(c) for c in frame]))
    return pa.table([
        column.dictionary_encode() if pa.types.is_string(column.type) else column
        for column in table.columns
    ], names=table.column_names)


class _StagedValues:
    """Disk-backed raw frames and compact per-feature Arrow accumulators."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.paths = []
        self.prototypes = []
        self.row_count = 0

    def add(self, frame):
        path = self.directory / f"{len(self.paths):06d}.parquet"
        frame.to_parquet(path, index=False)
        self.paths.append(path)
        self.prototypes.append(_prototype(frame))
        self.row_count += len(frame)

    def prepare(self):
        prototype = (pd.concat(self.prototypes, ignore_index=True) if self.paths
                     else rr._empty_robustness_raw_frame())
        self.rename_roi = "roi_name" in prototype and "structure" not in prototype
        if self.rename_roi:
            prototype = prototype.rename(columns={"roi_name": "structure"})
        self.columns = list(prototype.columns)
        self.schema = pa.Schema.from_pandas(prototype, preserve_index=False)
        self.groups = ["structure"]
        self.groups += [c for c in ("segmentation_source", "extraction_arm")
                        if c in self.columns]
        self.groups.append("feature_name")
        self.accumulators = {}
        inventories = []
        needed = list(dict.fromkeys([
            *self.groups, "patient_id", "course_id", "perturbation_id", "value",
            "robustness_status", "reason_code",
        ]))
        needed = [c for c in needed if c in self.columns]
        inventory_columns = [c for c in needed if c not in
                             {"perturbation_id", "value", "reason_code"}]
        for path in self.paths:
            available = pq.read_schema(path).names
            read_columns = ["roi_name" if self.rename_roi and c == "structure" else c
                            for c in needed]
            frame = pd.read_parquet(path, columns=[c for c in read_columns if c in available])
            if self.rename_roi:
                frame = frame.rename(columns={"roi_name": "structure"})
            frame = frame.reindex(columns=needed)
            # Match the legacy global guards, including rows with null group keys.
            if "robustness_status" in frame:
                if frame.robustness_status.eq("technical_failure").any():
                    raise RuntimeError("technical robustness failures require recovery before aggregation")
                if frame.robustness_status.eq("geometrically_impossible").any():
                    raise ValueError(
                        "geometric non-measurements require an explicit comparable-condition analysis; "
                        "do not drop subjects, impute values, or pool varying condition sets into fixed-grid ICC"
                    )
            inventories.append(_dictionary_table(frame[inventory_columns].drop_duplicates(), self.schema))
            table = _dictionary_table(frame, self.schema)
            for key, indices in frame.groupby(self.groups).indices.items():
                self.accumulators.setdefault(key, []).append(table.take(pa.array(indices)))
            del frame, table
        if inventories:
            inventory = pa.concat_tables(inventories, promote_options="permissive").to_pandas()
            # Dictionary arrays become categoricals; the shared validator uses
            # pandas' object-column grouping semantics (including null keys).
            for column in inventory.select_dtypes("category"):
                inventory[column] = inventory[column].astype(object)
            rr._validate_cohort_feature_sets(inventory)

    def summaries(self, config):
        summaries = []
        for key in sorted(self.accumulators):
            tables = self.accumulators.pop(key)
            group = pa.concat_tables(tables, promote_options="permissive").to_pandas()
            for column in group.select_dtypes("category"):
                group[column] = group[column].astype(object)
            summaries.append(rr.summarize_feature_stability(group, config))
        if summaries:
            summary = pd.concat(summaries, ignore_index=True)
        else:
            summary = (pd.DataFrame() if self.paths else
                       pd.DataFrame(columns=list(rr.ROBUSTNESS_SUMMARY_COLUMNS)))
        per_source = None
        if self.paths and "segmentation_source" in self.columns:
            columns = ["segmentation_source", "structure"]
            columns += [c for c in ("extraction_arm",) if c in self.columns]
            columns.append("feature_name")
            if summary.empty:
                per_source = summary.copy()
            else:
                per_source = summary.sort_values(columns, kind="stable").reset_index(drop=True)
                per_source = per_source[columns + [c for c in summary if c not in columns]]
        return summary, per_source

    def to_parquet(self, path, *, index=False):
        """Publication adapter for the unchanged paired-output helper.

        Row order and the pandas/Arrow schema match concatenation. Row groups
        end at course boundaries, so compressed bytes and file metadata sizes
        may differ from the legacy single-DataFrame writer.
        """
        with pq.ParquetWriter(path, self.schema) as writer:
            if not self.paths:
                writer.write_table(pa.Table.from_pandas(rr._empty_robustness_raw_frame(),
                                                       preserve_index=False))
            for source in self.paths:
                table = pq.read_table(source)
                if self.rename_roi:
                    table = table.rename_columns(["structure" if c == "roi_name" else c
                                                  for c in table.column_names])
                arrays = [table[field.name] if field.name in table.column_names else
                          pa.nulls(table.num_rows, type=field.type) for field in self.schema]
                table = pa.table(arrays, names=self.columns).cast(self.schema)
                writer.write_table(table)


def _sheets(summary, per_source):
    if summary.empty:
        robust = pd.DataFrame(columns=summary.columns)
        acceptable = pd.DataFrame(columns=summary.columns)
    else:
        robust = summary[summary.robustness_label == "robust"]
        acceptable = summary[summary.pass_seg_perturb]
    sheets = [("global_summary", summary.copy())]
    if per_source is not None:
        sheets.append(("per_source_summary", per_source))
    sheets.extend([("per_structure_source", summary), ("robust_features", robust),
                   ("acceptable_features", acceptable)])
    if per_source is not None and not per_source.empty:
        sheets.append(("robust_features_per_source",
                       per_source[per_source.robustness_label == "robust"]))
    return sheets


def _publish(staged, output, config, *, extra_sheets=(), verify_evidence=None):
    staged.prepare()
    summary, per_source = staged.summaries(config)
    output, raw = rr.robustness_cohort_output_paths(output)
    rr._publish_robustness_cohort_outputs(
        output_excel=output, raw_parquet_path=raw, raw_frame=staged,
        sheets=[*_sheets(summary, per_source), *extra_sheets],
        verify_evidence=verify_evidence,
    )
    rr.logger.info("Saved raw robustness values (%d rows) and %d summary groups",
                   staged.row_count, len(summary))


def _duplicate(registry, key, path, repeated):
    previous = registry.get(key)
    if previous is not None:
        raise RuntimeError(
            f"duplicate robustness aggregation input: {path} repeats the {repeated} "
            f"of {previous}; one course contributes to the cohort exactly once"
        )
    registry[key] = path


def aggregate_robustness_results(input_parquets, output_excel, rob_config):
    output_excel = Path(output_excel)
    rr.withdraw_robustness_cohort_outputs(output_excel)
    if not input_parquets:
        raise RuntimeError("no robustness results were supplied for aggregation")
    output_excel.parent.mkdir(parents=True, exist_ok=True)
    seen_paths, seen_courses, seen_tables = {}, {}, {}
    with TemporaryDirectory(prefix=".robustness-aggregate-", dir=output_excel.parent) as directory:
        staged = _StagedValues(directory)
        for path in input_parquets:
            entry = rr._admit_robustness_aggregation_input(path, rob_config)
            for registry, key, repeated in (
                (seen_paths, str(entry.path.resolve()), "input path"),
                (seen_courses, (entry.patient_id, entry.course_id), "course identity"),
                (seen_tables, entry.table_sha256, "table content"),
            ):
                _duplicate(registry, key, entry.path, repeated)
            staged.add(entry.frame)
            del entry
        _publish(staged, output_excel, rob_config)


def aggregate_robustness_cohort(
    courses: Sequence[RobustnessCourseReference | rr.RobustnessCohortCourse],
    output_excel: Path, rob_config: rr.RobustnessConfig, *, cohort: Mapping,
):
    """Account for exact manifest membership with one live admitted frame.

    Callers should pass references, as the CLI does. Already-admitted legacy
    objects remain supported, including rejection of mutated public frames.
    """
    output_excel = Path(output_excel)
    rr.withdraw_robustness_cohort_outputs(output_excel)
    if not courses:
        raise RuntimeError("the course manifest names no course; a cohort robustness summary requires at least one accounted course")
    rr._require_no_unresolved_quarantine(cohort, output_excel)
    _verify_manifest(courses, cohort)
    output_excel.parent.mkdir(parents=True, exist_ok=True)
    snapshots, outcomes = [], []
    seen_courses, seen_tables = {}, {}
    with TemporaryDirectory(prefix=".robustness-aggregate-", dir=output_excel.parent) as directory:
        staged = _StagedValues(directory)
        for course in courses:
            fresh = rr.admit_robustness_cohort_course(
                course.course_dir, patient_id=course.patient_id,
                course_id=course.course_id, rob_config=rob_config,
            )
            if isinstance(course, rr.RobustnessCohortCourse):
                _check_snapshot(course, fresh, check_frame=True)
            _duplicate(seen_courses, (fresh.patient_id, fresh.course_id),
                       fresh.course_dir, "course identity")
            if fresh.measured_output_sha256 is not None:
                _duplicate(seen_tables, fresh.measured_output_sha256,
                           fresh.measured_output, "table content")
            outcomes.append(rr._robustness_course_outcome_rows([fresh]))
            if fresh.frame is not None:
                staged.add(fresh.frame)
            snapshots.append(replace(fresh, frame=None, table_snapshot=None))
            del fresh
        rr.logger.info("Accounting for %d manifest course(s): %d measured, %d non-measured",
                       len(courses), len(staged.paths), len(courses) - len(staged.paths))

        def verify():
            _verify_manifest(courses, cohort)
            for snapshot in snapshots:
                fresh = rr.admit_robustness_cohort_course(
                    snapshot.course_dir, patient_id=snapshot.patient_id,
                    course_id=snapshot.course_id, rob_config=rob_config,
                )
                _check_snapshot(snapshot, fresh)
                del fresh
            # Legacy callers can mutate their public view while summaries run.
            for original, snapshot in zip(courses, snapshots):
                if isinstance(original, rr.RobustnessCohortCourse):
                    _check_snapshot(snapshot, original)
                    if original.table_snapshot is not None and (
                        original.frame is None or not original.frame.equals(
                            pd.read_parquet(BytesIO(original.table_snapshot)))):
                        raise ValueError("admitted measurement frame was mutated")

        _publish(staged, output_excel, rob_config,
                 extra_sheets=[("course_outcomes", pd.concat(outcomes, ignore_index=True)),
                               ("source_dispositions", rr._robustness_source_disposition_rows(snapshots))],
                 verify_evidence=verify)
