from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import PipelineConfig
from time import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from .organize import organize_and_merge


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rtpipeline", description="End-to-end DICOM-RT pipeline")
    p.add_argument("--dicom-root", required=True, help="Path to root with DICOM files")
    p.add_argument("--outdir", default="./Data_Organized", help="Output directory for organized data")
    p.add_argument("--logs", default="./Logs", help="Logs directory")
    p.add_argument(
        "--merge-criteria",
        choices=["same_ct_study", "frame_of_reference"],
        default="same_ct_study",
        help="Course grouping criterion. Default: same_ct_study",
    )
    p.add_argument("--max-days", type=int, default=None, help="Optional max days within a course")
    p.add_argument("--no-segmentation", action="store_true", help="Skip TotalSegmentator")
    p.add_argument("--force-segmentation", action="store_true", help="Re-run TotalSegmentator even if outputs exist")
    p.add_argument("--no-dvh", action="store_true", help="Skip DVH computation")
    p.add_argument("--no-visualize", action="store_true", help="Skip HTML visualization")
    p.add_argument("--no-metadata", action="store_true", help="Skip XLSX metadata extraction")
    p.add_argument("--conda-activate", default=None, help="Prefix shell with conda activate (segmentation)")
    p.add_argument("--dcm2niix", default="dcm2niix", help="dcm2niix command name")
    p.add_argument("--totalseg", default="TotalSegmentator", help="TotalSegmentator command name")
    p.add_argument("--totalseg-license", default=None, help="TotalSegmentator license key (if required)")
    p.add_argument(
        "--extra-seg-models",
        action="append",
        default=[],
        help="Extra TotalSegmentator tasks to run in addition to 'total' (comma-separated or repeat)"
    )
    p.add_argument("--totalseg-fast", action="store_true", help="Add --fast for CPU runs to improve runtime")
    p.add_argument("--totalseg-roi-subset", default=None, help="Restrict to subset of ROIs (comma-separated)")
    p.add_argument("--workers", type=int, default=None, help="Parallel workers for non-segmentation phases (default: auto)")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity")
    return p


def _doctor(argv: list[str]) -> int:
    import platform
    import shutil
    from importlib import metadata as importlib_metadata
    p = argparse.ArgumentParser(prog="rtpipeline doctor", description="Check environment for rtpipeline")
    p.add_argument("--logs", default="./Logs", help="Logs directory (used for fallback extraction)")
    p.add_argument("--conda-activate", default=None, help="Conda activation prefix to consider")
    p.add_argument("--dcm2niix", default="dcm2niix", help="dcm2niix command name to check")
    p.add_argument("--totalseg", default="TotalSegmentator", help="TotalSegmentator command name to check")
    args = p.parse_args(argv)

    print("rtpipeline doctor")
    print(f"- Python: {platform.python_version()} on {platform.system()} {platform.release()}")
    # Core Python packages and versions
    def ver(name: str) -> str:
        try:
            return importlib_metadata.version(name)
        except Exception:
            return "not installed"
    print(f"- pydicom: {ver('pydicom')}")
    print(f"- SimpleITK: {ver('SimpleITK')}")
    print(f"- dicompyler-core: {ver('dicompyler-core')}")
    print(f"- pydicom-seg: {ver('pydicom-seg')}")
    print(f"- rt-utils: {ver('rt-utils')}")
    print(f"- TotalSegmentator (pkg): {ver('TotalSegmentator')}")

    # CLI tools
    conda_prefix = args.conda_activate
    dcm2 = shutil.which(args.dcm2niix) if not conda_prefix else None
    totseg = shutil.which(args.totalseg) if not conda_prefix else None
    print(f"- dcm2niix in PATH: {dcm2 or ('requires conda env' if conda_prefix else 'not found')}")
    print(f"- TotalSegmentator in PATH: {totseg or ('requires conda env' if conda_prefix else 'not found')}")

    # Bundled zips
    from importlib import resources as importlib_resources
    bundled = []
    for nm in ("dcm2niix_lnx.zip", "dcm2niix_mac.zip", "dcm2niix_win.zip"):
        try:
            res = importlib_resources.files('rtpipeline').joinpath('ext', nm)
            if res.is_file():
                bundled.append(nm)
        except Exception:
            pass
    print(f"- Bundled dcm2niix zips in package: {', '.join(bundled) if bundled else 'none'}")

    # Fallback decision
    from .segmentation import _ensure_local_dcm2niix
    cfg = PipelineConfig(
        dicom_root=Path('.'),
        output_root=Path('.'),
        logs_root=Path(args.logs).resolve(),
        conda_activate=conda_prefix,
    )
    # Do not actually run any conversion; just see if fallback can be prepared
    fallback_possible = False
    if not conda_prefix and dcm2 is None:
        # Try dry-run: if bundle exists, we can extract when needed
        for nm in ("dcm2niix_lnx.zip", "dcm2niix_mac.zip", "dcm2niix_win.zip"):
            try:
                res = importlib_resources.files('rtpipeline').joinpath('ext', nm)
                if res.is_file():
                    fallback_possible = True
                    break
            except Exception:
                continue
    print(f"- dcm2niix fallback available: {'yes' if fallback_possible else 'no'}")
    if not conda_prefix and dcm2 is None and not fallback_possible:
        print("  -> NIfTI conversion will be skipped; DICOM-mode segmentation still runs.")
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    # Lightweight subcommand dispatch to preserve backward compatibility
    if argv and argv[0] == "doctor":
        return _doctor(argv[1:])
    args = build_parser().parse_args(argv)
    level = logging.INFO if args.verbose == 0 else logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = PipelineConfig(
        dicom_root=Path(args.dicom_root).resolve(),
        output_root=Path(args.outdir).resolve(),
        logs_root=Path(args.logs).resolve(),
        merge_criteria=args.merge_criteria,
        max_days_between_plans=args.max_days,
        do_segmentation=not args.no_segmentation,
        do_dvh=not args.no_dvh,
        do_visualize=not args.no_visualize,
        conda_activate=args.conda_activate,
        dcm2niix_cmd=args.dcm2niix,
        totalseg_cmd=args.totalseg,
        totalseg_license_key=args.totalseg_license,
        extra_seg_models=[m.strip() for part in (args.extra_seg_models or []) for m in part.split(",") if m.strip()],
        totalseg_fast=args.totalseg_fast,
        totalseg_roi_subset=args.totalseg_roi_subset,
        workers=args.workers,
    )
    # Ensure directories and also route logs to a file for traceability
    try:
        cfg.ensure_dirs()
        fh = logging.FileHandler(cfg.logs_root / "rtpipeline.log", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logging.getLogger().addHandler(fh)
    except Exception:
        # Non-fatal; continue with console-only logging
        pass

    # 0) Metadata extraction (XLSX) if desired
    if not args.no_metadata:
        from .meta import export_metadata
        export_metadata(cfg)

    # 1) Organize courses (primary+boost only, by same CT study by default)
    courses = organize_and_merge(cfg)

    # 2) Optional segmentation (TotalSegmentator)
    if cfg.do_segmentation:
        from .segmentation import segment_course, segment_extra_models_mr  # lazy import
        # Keep segmentation sequential (resource heavy) with progress/ETA
        total_seg = len(courses)
        if total_seg:
            t0 = time()
            for i, c in enumerate(courses, start=1):
                segment_course(cfg, c.dir, force=args.force_segmentation)
                elapsed = time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total_seg - i) / rate if rate > 0 else float('inf')
                logging.info("Segmentation: %d/%d (%.0f%%) elapsed %.0fs ETA %.0fs", i, total_seg, 100*i/total_seg, elapsed, eta)
        # Build auto RTSTRUCTs so DVH can include TotalSegmentator output
        from .auto_rtstruct import build_auto_rtstruct
        # Parallelize RTSTRUCT builds with progress/ETA
        def _run_pool(label: str, items, func):
            total = len(items)
            if total == 0:
                return
            t0 = time()
            done = 0
            with ThreadPoolExecutor(max_workers=cfg.effective_workers()) as ex:
                futs = {ex.submit(func, it): it for it in items}
                for _ in as_completed(futs):
                    done += 1
                    elapsed = time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else float('inf')
                    logging.info("%s: %d/%d (%.0f%%) elapsed %.0fs ETA %.0fs", label, done, total, 100*done/total, elapsed, eta)
        _run_pool("Build RS_auto", courses, lambda c: build_auto_rtstruct(c.dir))
        # If extra models were requested, also run them for MR series found in dicom_root
        if cfg.extra_seg_models:
            segment_extra_models_mr(cfg, force=args.force_segmentation)

    # 3) DVH per course
    if cfg.do_dvh:
        from .dvh import dvh_for_course  # lazy import
        def _dvh(c):
            return dvh_for_course(c.dir)
        # Parallel with progress
        def _run_pool(label: str, items, func):
            total = len(items)
            if total == 0:
                return []
            results = []
            t0 = time()
            done = 0
            with ThreadPoolExecutor(max_workers=cfg.effective_workers()) as ex:
                futs = {ex.submit(func, it): it for it in items}
                for f in as_completed(futs):
                    done += 1
                    try:
                        results.append(f.result())
                    except Exception:
                        results.append(None)
                    elapsed = time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else float('inf')
                    logging.info("%s: %d/%d (%.0f%%) elapsed %.0fs ETA %.0fs", label, done, total, 100*done/total, elapsed, eta)
            return results
        _run_pool("DVH", courses, _dvh)

    # 4) Visualization
    if cfg.do_visualize:
        from .visualize import visualize_course, generate_axial_review  # lazy import
        def _both(c):
            try:
                visualize_course(c.dir)
            finally:
                generate_axial_review(c.dir)
        # Parallel with progress
        def _run_pool(label: str, items, func):
            total = len(items)
            if total == 0:
                return
            t0 = time()
            done = 0
            with ThreadPoolExecutor(max_workers=cfg.effective_workers()) as ex:
                futs = {ex.submit(func, it): it for it in items}
                for _ in as_completed(futs):
                    done += 1
                    elapsed = time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else float('inf')
                    logging.info("%s: %d/%d (%.0f%%) elapsed %.0fs ETA %.0fs", label, done, total, 100*done/total, elapsed, eta)
        _run_pool("Visualization", courses, _both)

    # 5) Merge DVH metrics across all courses (if any)
    try:
        import pandas as _pd
        merged = []
        for c in courses:
            dvh_path = c.dir / "dvh_metrics.xlsx"
            if dvh_path.exists():
                try:
                    df = _pd.read_excel(dvh_path)
                    df.insert(0, "patient_id", c.patient_id)
                    df.insert(1, "course_key", c.course_key)
                    df.insert(2, "course_dir", str(c.dir))
                    merged.append(df)
                except Exception:
                    continue
        if merged:
            all_df = _pd.concat(merged, ignore_index=True)
            out_all = cfg.output_root / "DVH_metrics_all.xlsx"
            all_df.to_excel(out_all, index=False)
    except Exception as e:
        logging.getLogger(__name__).warning("Failed to write merged DVH metrics: %s", e)

    # 6) Merge per-case metadata across all courses (if any)
    try:
        import json as _json
        import pandas as _pd
        data_dir = cfg.output_root / "Data"
        data_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for c in courses:
            meta_json = c.dir / "case_metadata.json"
            if meta_json.exists():
                try:
                    with open(meta_json, "r", encoding="utf-8") as f:
                        d = _json.load(f)
                    # Ensure identifiers are present
                    d.setdefault('patient_id', c.patient_id)
                    d.setdefault('course_key', c.course_key)
                    d.setdefault('course_dir', str(c.dir))
                    rows.append(d)
                except Exception:
                    continue
        if rows:
            df = _pd.DataFrame(rows)
            (data_dir / "case_metadata_all.xlsx").parent.mkdir(parents=True, exist_ok=True)
            df.to_excel(data_dir / "case_metadata_all.xlsx", index=False)
            # Also write JSON for machine consumption
            with open(data_dir / "case_metadata_all.json", "w", encoding="utf-8") as f:
                _json.dump(rows, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.getLogger(__name__).warning("Failed to write merged case metadata: %s", e)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
