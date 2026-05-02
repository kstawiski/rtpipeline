from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pydicom

logger = logging.getLogger(__name__)

RIDER_COLLECTION = "RIDER Lung CT"
RIDER_EXPECTED_PATIENTS = 32


def _bool_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def _load_nbia():
    try:
        from tcia_utils import nbia
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "tcia_utils is required for TCIA acquisition. Install it in the rtpipeline env."
        ) from exc
    return nbia


def query_rider_lung_ct_original_series(collection: str = RIDER_COLLECTION) -> pd.DataFrame:
    """Return original RIDER Lung CT image series, excluding third-party analyses."""

    nbia = _load_nbia()
    data = nbia.getSeries(collection=collection, format="df")
    if data is None or data.empty:
        raise RuntimeError(f"No TCIA series returned for collection: {collection}")
    if "Modality" not in data.columns or "SeriesInstanceUID" not in data.columns:
        raise RuntimeError("TCIA series response lacks required Modality/SeriesInstanceUID columns")

    filtered = data[data["Modality"].astype(str).str.upper().eq("CT")].copy()
    if "ThirdPartyAnalysis" in filtered.columns:
        filtered = filtered[filtered["ThirdPartyAnalysis"].map(_bool_missing)].copy()

    filtered.sort_values(
        by=[col for col in ["PatientID", "StudyDate", "StudyInstanceUID", "SeriesNumber", "SeriesInstanceUID"] if col in filtered.columns],
        inplace=True,
        kind="stable",
    )
    filtered.reset_index(drop=True, inplace=True)
    return filtered


def summarize_series(series_df: pd.DataFrame) -> dict[str, Any]:
    if series_df.empty:
        return {
            "n_patients": 0,
            "n_studies": 0,
            "n_series": 0,
            "n_images": 0,
            "total_bytes": 0,
            "paired_patients": 0,
            "single_scan_patients": [],
        }

    by_patient = series_df.groupby("PatientID", dropna=False).agg(
        n_series=("SeriesInstanceUID", "nunique"),
        n_studies=("StudyInstanceUID", "nunique"),
        n_images=("ImageCount", "sum"),
        total_bytes=("FileSize", "sum"),
    )
    single_scan = sorted(str(pid) for pid, row in by_patient.iterrows() if int(row["n_series"]) < 2)
    return {
        "n_patients": int(series_df["PatientID"].nunique()),
        "n_studies": int(series_df["StudyInstanceUID"].nunique()),
        "n_series": int(series_df["SeriesInstanceUID"].nunique()),
        "n_images": int(series_df.get("ImageCount", pd.Series(dtype=int)).fillna(0).astype(int).sum()),
        "total_bytes": int(series_df.get("FileSize", pd.Series(dtype=int)).fillna(0).astype(int).sum()),
        "paired_patients": int((by_patient["n_series"] >= 2).sum()),
        "single_scan_patients": single_scan,
        "patient_series_counts": {
            str(pid): int(row["n_series"])
            for pid, row in by_patient.sort_index().iterrows()
        },
    }


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file(follow_symlinks=False):
                count += 1
    return count


def _download_patient_series(patient_id: str, patient_series: pd.DataFrame, input_dir: Path, max_workers: int) -> None:
    nbia = _load_nbia()
    patient_dir = input_dir / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)
    nbia.downloadSeries(
        patient_series,
        path=str(patient_dir),
        input_type="df",
        max_workers=max(1, int(max_workers)),
    )


def download_series_by_patient(series_df: pd.DataFrame, input_dir: Path, max_workers: int = 4) -> pd.DataFrame:
    """Download original CT series into input_dir/PatientID/SeriesInstanceUID."""

    input_dir.mkdir(parents=True, exist_ok=True)
    for patient_id, patient_series in series_df.groupby("PatientID", sort=True):
        logger.info("Downloading %s: %d series", patient_id, len(patient_series))
        _download_patient_series(str(patient_id), patient_series, input_dir, max_workers=max_workers)
    return verify_download(series_df, input_dir)


def verify_download(series_df: pd.DataFrame, input_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in series_df.iterrows():
        patient_id = str(row["PatientID"])
        series_uid = str(row["SeriesInstanceUID"])
        series_dir = input_dir / patient_id / series_uid
        expected = int(row.get("ImageCount") or 0)
        actual = _count_files(series_dir)
        rows.append(
            {
                "patient_id": patient_id,
                "study_uid": str(row.get("StudyInstanceUID") or ""),
                "series_uid": series_uid,
                "expected_images": expected,
                "downloaded_files": actual,
                "complete": bool(expected and actual >= expected),
                "series_dir": str(series_dir),
            }
        )
    return pd.DataFrame(rows)


def _first_ct_dataset(ct_dir: Path):
    for path in ct_dir.iterdir():
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception:
            continue
        if str(getattr(ds, "Modality", "")).upper() == "CT":
            return ds
    return None


def _nifti_metadata(course_dir: Path) -> dict[str, Any]:
    nifti_dir = course_dir / "NIFTI"
    if not nifti_dir.exists():
        return {}
    for meta_path in sorted(nifti_dir.glob("*.metadata.json")):
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return {}


def _has_lung_masks(course_dir: Path) -> bool:
    seg_root = course_dir / "Segmentation_TotalSegmentator"
    if not seg_root.exists():
        return False
    for child in seg_root.iterdir():
        if child.is_dir() and any(child.glob("total--lung_*.nii.gz")):
            return True
    return False


def _course_dirs(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    courses: list[Path] = []
    for patient_dir in sorted(p for p in output_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for course_dir in sorted(p for p in patient_dir.iterdir() if p.is_dir()):
            if (course_dir / "DICOM" / "CT").exists():
                courses.append(course_dir)
    return courses


def build_organized_courses_manifest(output_dir: Path, manifest_json: Path) -> dict[str, Any]:
    courses = [
        {
            "patient": course_dir.parent.name,
            "course": course_dir.name,
            "path": str(course_dir),
        }
        for course_dir in _course_dirs(output_dir)
    ]
    manifest = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "courses": courses,
    }
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_rider_manifest(output_dir: Path, manifest_csv: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for course_dir in _course_dirs(output_dir):
        ct_dir = course_dir / "DICOM" / "CT"
        nifti_meta = _nifti_metadata(course_dir)
        ds = _first_ct_dataset(ct_dir)
        scan_uid = str(nifti_meta.get("series_instance_uid") or "")
        if not scan_uid and ds is not None:
            scan_uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
        n_slices = int(nifti_meta.get("instance_count") or 0)
        if not n_slices:
            n_slices = _count_files(ct_dir)
        slice_thickness = None
        scanner_model = ""
        if ds is not None:
            scanner_model = str(getattr(ds, "ManufacturerModelName", "") or "")
            try:
                slice_thickness = float(getattr(ds, "SliceThickness", ""))
            except Exception:
                slice_thickness = None
        lung_masks_present = _has_lung_masks(course_dir)
        rows.append(
            {
                "patient_id": course_dir.parent.name,
                "course_id": course_dir.name,
                "scan_uid": scan_uid,
                "n_slices": n_slices,
                "slice_thickness_mm": slice_thickness,
                "scanner_model": scanner_model,
                "ts_lung_mask_present": lung_masks_present,
                "ready_for_phase2b": bool(scan_uid and n_slices and lung_masks_present),
            }
        )
    manifest = pd.DataFrame(
        rows,
        columns=[
            "patient_id",
            "course_id",
            "scan_uid",
            "n_slices",
            "slice_thickness_mm",
            "scanner_model",
            "ts_lung_mask_present",
            "ready_for_phase2b",
        ],
    )
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_csv, index=False)
    build_organized_courses_manifest(output_dir, output_dir / "_organized_courses_manifest.json")
    return manifest


def _disk_usage_bytes(path: Path, seen: set[tuple[int, int]] | None = None) -> int:
    total = 0
    seen = seen if seen is not None else set()
    if path.exists():
        stack = [path]
        while stack:
            current = stack.pop()
            try:
                with os.scandir(current) as entries:
                    for entry in entries:
                        try:
                            stat_result = entry.stat(follow_symlinks=False)
                        except OSError:
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        inode = (stat_result.st_dev, stat_result.st_ino)
                        if inode in seen:
                            continue
                        seen.add(inode)
                        total += stat_result.st_size
            except OSError:
                continue
    return total


def _combined_disk_usage_bytes(paths: list[Path]) -> int:
    seen: set[tuple[int, int]] = set()
    return sum(_disk_usage_bytes(path, seen=seen) for path in paths)


def _logical_disk_usage_bytes(path: Path) -> int:
    total = 0
    if path.exists():
        stack = [path]
        while stack:
            current = stack.pop()
            try:
                with os.scandir(current) as entries:
                    for entry in entries:
                        try:
                            stat_result = entry.stat(follow_symlinks=False)
                        except OSError:
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += stat_result.st_size
            except OSError:
                continue
    return total


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def _derived_output_bytes(output_dir: Path) -> int:
    """Estimate derived output footprint without traversing hardlinked CT DICOM trees."""

    total = 0
    for course_dir in _course_dirs(output_dir):
        total += _file_size(course_dir / "RS_auto.dcm")
        for subdir_name in ("NIFTI", "Segmentation_TotalSegmentator"):
            subdir = course_dir / subdir_name
            if subdir.exists():
                total += _logical_disk_usage_bytes(subdir)

    for top_name in ("Data", "_CACHE"):
        top = output_dir / top_name
        if top.exists():
            total += _logical_disk_usage_bytes(top)
    for meta_path in output_dir.glob("_*.csv"):
        total += _file_size(meta_path)
    for meta_path in output_dir.glob("_*.json"):
        total += _file_size(meta_path)
    return total


def _fmt_gib(num_bytes: int) -> str:
    return f"{num_bytes / 1024 ** 3:.2f} GiB"


def write_rider_report(
    report_path: Path,
    input_dir: Path,
    output_dir: Path,
    series_df: pd.DataFrame,
    download_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    failures: list[str] | None = None,
    open_questions: list[str] | None = None,
) -> None:
    failures = failures or []
    open_questions = open_questions or []
    expected = summarize_series(series_df)

    downloaded_complete = int(download_df["complete"].sum()) if "complete" in download_df else 0
    downloaded_patients = int(download_df.loc[download_df.get("complete", False), "patient_id"].nunique()) if not download_df.empty else 0
    patient_counts = download_df.groupby("patient_id")["complete"].sum() if not download_df.empty else pd.Series(dtype=int)
    paired_downloaded = int((patient_counts >= 2).sum()) if not patient_counts.empty else 0
    ts_success = int(manifest_df["ts_lung_mask_present"].sum()) if "ts_lung_mask_present" in manifest_df else 0
    ready = int(manifest_df["ready_for_phase2b"].sum()) if "ready_for_phase2b" in manifest_df else 0
    n_courses = int(len(manifest_df))
    complete = (
        expected["n_patients"] == RIDER_EXPECTED_PATIENTS
        and downloaded_complete == expected["n_series"]
        and n_courses == expected["n_series"]
        and ts_success == n_courses
        and ready == n_courses
    )
    status = "COMPLETE" if complete else "PARTIAL"
    missing_downloads = download_df[~download_df["complete"]].copy() if "complete" in download_df else pd.DataFrame()
    single_scan = expected.get("single_scan_patients", [])
    input_source = int(expected["total_bytes"])
    derived_output = _derived_output_bytes(output_dir)
    estimated_unique = input_source + derived_output

    lines = [
        f"# RIDER Lung CT Acquisition Report",
        "",
        f"Status marker: {status}",
        f"Generated: {_dt.datetime.now(_dt.timezone.utc).isoformat()}",
        "",
        f"- TCIA collection: {RIDER_COLLECTION}",
        f"- Expected original CT cohort: {expected['n_patients']} patients, {expected['n_series']} CT series, {expected['n_studies']} studies, {expected['n_images']} images, {_fmt_gib(expected['total_bytes'])}",
        f"- Patients downloaded: {downloaded_patients} / {RIDER_EXPECTED_PATIENTS}",
        f"- Complete CT series downloads: {downloaded_complete} / {expected['n_series']}",
        f"- Patients with paired test-retest CT series: {paired_downloaded} / {RIDER_EXPECTED_PATIENTS}",
        f"- Single-scan patients retained: {', '.join(single_scan) if single_scan else 'none'}",
        f"- Organized courses: {n_courses}",
        f"- Total disk usage estimate: {_fmt_gib(estimated_unique)} unique bytes (TCIA source {_fmt_gib(input_source)} + derived output {_fmt_gib(derived_output)}; organized CT DICOM are hardlinks)",
        f"- TS lung mask success rate: {ts_success} / {n_courses}",
        f"- Cases ready for Phase 2b: {ready} / {n_courses}",
        "",
        "## Failures + Reasons",
    ]
    if failures:
        lines.extend([f"- {item}" for item in failures])
    if not missing_downloads.empty:
        for _, row in missing_downloads.iterrows():
            lines.append(
                f"- Incomplete download: {row['patient_id']} {row['series_uid']} "
                f"({row['downloaded_files']} / {row['expected_images']} files)"
            )
    if not failures and missing_downloads.empty:
        lines.append("- none")

    lines.extend(["", "## Open Questions"])
    if open_questions:
        lines.extend([f"- {item}" for item in open_questions])
    else:
        lines.append("- none")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rider_acquisition_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rtpipeline tcia-rider-lung-ct",
        description="Query and download original RIDER Lung CT CT series from TCIA.",
    )
    parser.add_argument("--input-dir", required=True, help="Destination input directory")
    parser.add_argument("--metadata-dir", default=None, help="Directory for TCIA query/verification CSVs")
    parser.add_argument("--series-csv", default=None, help="Reuse an existing TCIA series CSV instead of querying NBIA")
    parser.add_argument("--download", action="store_true", help="Download series after querying")
    parser.add_argument("--max-workers", type=int, default=4, help="TCIA download workers per patient")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    input_dir = Path(args.input_dir).resolve()
    metadata_dir = Path(args.metadata_dir).resolve() if args.metadata_dir else input_dir.parent / "output"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    series_csv = metadata_dir / "_tcia_original_ct_series.csv"
    if args.series_csv:
        series = pd.read_csv(Path(args.series_csv).expanduser())
    else:
        series = query_rider_lung_ct_original_series()
    summary_json = metadata_dir / "_tcia_original_ct_summary.json"
    series.to_csv(series_csv, index=False)
    summary_json.write_text(json.dumps(summarize_series(series), indent=2), encoding="utf-8")

    if args.download:
        verification = download_series_by_patient(series, input_dir=input_dir, max_workers=args.max_workers)
    else:
        verification = verify_download(series, input_dir=input_dir)
    verification.to_csv(metadata_dir / "_tcia_download_verification.csv", index=False)

    if not verification.empty and not bool(verification["complete"].all()):
        logger.warning("RIDER download verification is incomplete")
        return 1 if args.download else 0
    return 0


def rider_manifest_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rtpipeline rider-manifest",
        description="Build RIDER Lung CT Phase 2b manifest and concise acquisition report.",
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--series-csv", default=None)
    parser.add_argument("--download-verification-csv", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    series_csv = Path(args.series_csv).resolve() if args.series_csv else output_dir / "_tcia_original_ct_series.csv"
    verification_csv = (
        Path(args.download_verification_csv).resolve()
        if args.download_verification_csv
        else output_dir / "_tcia_download_verification.csv"
    )

    if series_csv.exists():
        series = pd.read_csv(series_csv)
    else:
        series = query_rider_lung_ct_original_series()
        series.to_csv(series_csv, index=False)
    if verification_csv.exists():
        verification = pd.read_csv(verification_csv)
    else:
        verification = verify_download(series, input_dir)
        verification.to_csv(verification_csv, index=False)

    manifest = build_rider_manifest(output_dir, output_dir / "_manifest.csv")
    failures: list[str] = []
    if len(manifest) != int(series["SeriesInstanceUID"].nunique()):
        failures.append(
            f"organized course count mismatch: {len(manifest)} courses vs {series['SeriesInstanceUID'].nunique()} expected original CT series"
        )
    if "complete" in verification and not bool(verification["complete"].all()):
        failures.append("one or more TCIA series downloads are incomplete")
    if not manifest.empty and not bool(manifest["ts_lung_mask_present"].all()):
        failures.append("one or more organized courses lack TotalSegmentator lung masks")

    write_rider_report(
        Path(args.report).resolve(),
        input_dir=input_dir,
        output_dir=output_dir,
        series_df=series,
        download_df=verification,
        manifest_df=manifest,
        failures=failures,
        open_questions=[
            "The TCIA v4 getCollectionValues endpoint timed out from this host; the collection and original CT series were confirmed through TCIA/NBIA public API access used by tcia_utils.",
            "TCIA wiki reports CC BY 3.0 while current NBIA API series metadata reports CC BY 4.0; use TCIA landing page DOI/license citation trail in any external package."
        ],
    )
    return 0 if not failures else 1
