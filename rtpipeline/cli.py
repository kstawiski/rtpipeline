from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import PipelineConfig
from .organize import organize_and_merge


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rtpipeline", description="End-to-end DICOM-RT pipeline")
    p.add_argument("run", nargs="?", help="Run the pipeline", default="run")
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
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity")
    return p


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
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
    )

    # 0) Metadata extraction (XLSX) if desired
    if not args.no_metadata:
        from .meta import export_metadata
        export_metadata(cfg)

    # 1) Organize courses (primary+boost only, by same CT study by default)
    courses = organize_and_merge(cfg)

    # 2) Optional segmentation (TotalSegmentator)
    if cfg.do_segmentation:
        from .segmentation import segment_course  # lazy import
        for c in courses:
            segment_course(cfg, c.dir, force=args.force_segmentation)
        # Build auto RTSTRUCTs so DVH can include TotalSegmentator output
        from .auto_rtstruct import build_auto_rtstruct
        for c in courses:
            build_auto_rtstruct(c.dir)

    # 3) DVH per course
    if cfg.do_dvh:
        from .dvh import dvh_for_course  # lazy import
        for c in courses:
            dvh_for_course(c.dir)

    # 4) Visualization
    if cfg.do_visualize:
        from .visualize import visualize_course, generate_axial_review  # lazy import
        for c in courses:
            visualize_course(c.dir)
            generate_axial_review(c.dir)

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
