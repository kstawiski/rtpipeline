#!/usr/bin/env python3
"""Read-only scale benchmark for Task 16 combined discovery."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import threading
import time
from collections import Counter
from pathlib import Path


WORKTREE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKTREE_ROOT))


def _rss_kb() -> int:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    raise RuntimeError("VmRSS is unavailable")


def _file_count(root: Path, *, followlinks: bool) -> int:
    count = 0
    for _base, _dirs, files in os.walk(root, followlinks=followlinks):
        count += len(files)
    return count


def _select_scope(
    root: Path,
    *,
    minimum_files: int,
    followlinks: bool,
) -> tuple[list[str], int]:
    selected: list[str] = []
    count = 0
    for candidate in sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    ):
        selected.append(candidate.name)
        count += _file_count(candidate, followlinks=followlinks)
        if count >= minimum_files:
            return selected, count
    raise RuntimeError(
        f"Only {count} files were visible under {root}; need at least {minimum_files}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--min-files", type=int, default=50_000)
    parser.add_argument("--max-workers", type=int, required=True)
    parser.add_argument("--follow-symlinks", action="store_true")
    parser.add_argument("--source-cache-state", default="unknown_not_controlled")
    args = parser.parse_args()

    root = args.root.resolve(strict=True)
    if args.follow_symlinks:
        os.environ["RTPIPELINE_FOLLOW_INPUT_SYMLINKS"] = "1"
    else:
        os.environ.pop("RTPIPELINE_FOLLOW_INPUT_SYMLINKS", None)

    patient_ids, selected_file_count = _select_scope(
        root,
        minimum_files=max(1, args.min_files),
        followlinks=args.follow_symlinks,
    )

    import pydicom

    from rtpipeline.organize import _index_series_dataset
    from rtpipeline.rt_details import extract_rt_with_records

    dcmread_calls = 0
    counter_lock = threading.Lock()
    original_dcmread = pydicom.dcmread

    def counted_dcmread(*call_args, **call_kwargs):
        nonlocal dcmread_calls
        with counter_lock:
            dcmread_calls += 1
        return original_dcmread(*call_args, **call_kwargs)

    pydicom.dcmread = counted_dcmread
    series_index = {}
    registrations = {}
    series_meta = {}
    snapshot = {}

    def collect(path, dataset):
        _index_series_dataset(
            path,
            dataset,
            series_index,
            registrations,
            series_meta,
        )

    rss_before_kb = _rss_kb()
    started = time.perf_counter()
    try:
        plans, doses, structs, records, ct_index = extract_rt_with_records(
            root,
            patient_ids,
            max_workers=max(1, args.max_workers),
            metadata_snapshot=snapshot,
            include_ct_index=True,
            dataset_callback=collect,
        )
    finally:
        pydicom.dcmread = original_dcmread
    wall_seconds = time.perf_counter() - started

    snapshot_count = snapshot["identity"].file_count
    if snapshot_count != selected_file_count:
        raise RuntimeError(
            "Benchmark scope changed while being measured or discovery did not cover "
            f"the selected scope: selected={selected_file_count}, snapshot={snapshot_count}"
        )

    modality_counts = Counter(
        result.modality or "<missing>" for result in snapshot["results"]
    )
    ct_instances = sum(
        len(instances)
        for patient in ct_index.values()
        for study in patient.values()
        for instances in study.values()
    )
    indexed_series_paths = sum(len(paths) for paths in series_index.values())
    peak_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    payload = {
        "shape": "symlinked" if args.follow_symlinks else "real_directory",
        "root_files_selected": selected_file_count,
        "scope_directory_count": len(patient_ids),
        "workers": max(1, args.max_workers),
        "wall_seconds": wall_seconds,
        "rss_before_kb": rss_before_kb,
        "rss_after_kb": _rss_kb(),
        "peak_rss_kb": peak_rss_kb,
        "peak_minus_baseline_kb": peak_rss_kb - rss_before_kb,
        "dcmread_calls": dcmread_calls,
        "snapshot_results": len(snapshot["results"]),
        "inventory_sha256": snapshot["identity"].digest,
        "modality_counts": dict(sorted(modality_counts.items())),
        "plan_count": len(plans),
        "dose_count": len(doses),
        "struct_count": len(structs),
        "record_count": sum(len(paths) for paths in records.values()),
        "ct_instance_count": ct_instances,
        "series_index_path_count": indexed_series_paths,
        "registration_count": sum(len(items) for items in registrations.values()),
        "metadata_export_cache": "not_consulted_direct_discovery_benchmark",
        "source_page_cache": args.source_cache_state,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
