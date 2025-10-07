from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


from .config import PipelineConfig
from .organize import organize_and_merge
from .utils import run_tasks_with_adaptive_workers

logger = logging.getLogger(__name__)


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
    p.add_argument("--no-radiomics", action="store_true", help="Skip pyradiomics extraction")
    p.add_argument("--radiomics-params", default=None, help="Path to custom pyradiomics YAML parameter file")
    p.add_argument("--radiomics-params-mr", default=None, help="Path to pyradiomics YAML parameters for MR segmentation")
    p.add_argument("--sequential-radiomics", action="store_true", help="Use sequential radiomics processing (parallel is default)")
    p.add_argument(
        "--radiomics-skip-roi",
        action="append",
        default=[],
        help="ROI name(s) to exclude from radiomics. Accepts comma-separated values; provide multiple times to add more.",
    )
    p.add_argument(
        "--radiomics-max-voxels",
        type=int,
        default=None,
        help="Skip radiomics for ROIs exceeding this voxel count (default: 15,000,000)",
    )
    p.add_argument(
        "--radiomics-min-voxels",
        type=int,
        default=None,
        help="Skip radiomics for ROIs smaller than this voxel count (default: 120)",
    )
    p.add_argument("--custom-structures", default=None, help="Path to YAML configuration file for custom structures (uses pelvic template by default)")
    p.add_argument("--no-metadata", action="store_true", help="Skip XLSX metadata extraction")
    p.add_argument("--conda-activate", default=None, help="Prefix shell with conda activate (segmentation)")
    p.add_argument("--dcm2niix", default="dcm2niix", help="dcm2niix command name")
    p.add_argument("--totalseg", default="TotalSegmentator", help="TotalSegmentator command name")
    p.add_argument("--totalseg-license", default=None, help="TotalSegmentator license key (if required)")
    p.add_argument("--totalseg-weights", default=None, help="Path to pretrained weights (nnUNet_pretrained_models) for TotalSegmentator (offline)")
    p.add_argument(
        "--extra-seg-models",
        action="append",
        default=[],
        help="Extra TotalSegmentator tasks to run in addition to 'total' (comma-separated or repeat)"
    )
    p.add_argument("--totalseg-fast", action="store_true", help="Add --fast for CPU runs to improve runtime")
    p.add_argument("--totalseg-roi-subset", default=None, help="Restrict to subset of ROIs (comma-separated)")
    p.add_argument("--custom-models-root", default=None, help="Directory containing custom segmentation model definitions (default: ./custom_models)")
    p.add_argument(
        "--custom-model",
        action="append",
        default=[],
        help="Restrict custom segmentation to selected model names (comma-separated or repeat)",
    )
    p.add_argument("--custom-model-workers", type=int, default=None, help="Maximum concurrent courses for custom segmentation models")
    p.add_argument("--force-custom-models", action="store_true", help="Force re-run custom segmentation models even if outputs exist")
    p.add_argument("--custom-model-conda-activate", default=None, help="Override conda activation prefix for custom segmentation models")
    p.add_argument("--nnunet-predict", default="nnUNetv2_predict", help="nnUNetv2 prediction command (default: nnUNetv2_predict)")
    p.add_argument("--purge-custom-model-weights", action="store_true", help="Delete extracted nnUNet caches after each custom model run")
    p.add_argument("--workers", type=int, default=None, help="Parallel workers for non-segmentation phases (default: auto)")
    p.add_argument("--seg-workers", type=int, default=None, help="Maximum concurrent courses for TotalSegmentator (default: 1)")
    p.add_argument(
        "--seg-proc-threads",
        type=int,
        default=None,
        help="CPU threads per TotalSegmentator invocation (<=0 to disable limit)",
    )
    p.add_argument(
        "--radiomics-proc-threads",
        type=int,
        default=None,
        help="CPU threads per radiomics worker (<=0 to disable limit)",
    )
    p.add_argument(
        "--course-filter",
        action="append",
        default=[],
        help="Restrict stages to specific courses. Accepts PATIENT or PATIENT/COURSE; may be passed multiple times.",
    )
    p.add_argument("--force-redo", action="store_true", help="Force redo all steps, even if outputs exist (resume is default)")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity")
    p.add_argument(
        "--stage",
        action="append",
        choices=["organize", "segmentation", "segmentation_custom", "dvh", "visualize", "radiomics", "qc"],
        help="Execute only the selected pipeline stage(s); may be provided multiple times. Default: full pipeline.",
    )
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

    # GPU diagnostics
    try:
        import subprocess as _sp
        out = _sp.check_output(["python","-c","import torch;print('torch_cuda',getattr(torch,'cuda',None) is not None);print('cuda_available',getattr(torch,'cuda',None) and torch.cuda.is_available());print('cuda_count',getattr(torch,'cuda',None) and torch.cuda.device_count())"], stderr=_sp.STDOUT)
        print(out.decode().strip())
    except Exception:
        print("- PyTorch CUDA diagnostics: unavailable")
    # Weights path
    try:
        from pathlib import Path as _P
        weights = _P(args.logs) / 'nnunet'
        print(f"- Expected TS weights dir: {weights} (exists={weights.exists()})")
    except Exception:
        pass
    try:
        import shutil as _sh
        if _sh.which('nvidia-smi'):
            out = _sp.check_output(['nvidia-smi','-L'])
            print('- nvidia-smi:', out.decode().strip().splitlines()[0])
        else:
            print('- nvidia-smi not found in PATH')
    except Exception:
        print('- nvidia-smi check failed')

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
        segmentation_workers=None,
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

    skip_rois: list[str] = []
    for item in args.radiomics_skip_roi or []:
        if not item:
            continue
        parts = [seg.strip() for seg in str(item).replace(";", ",").split(",") if seg.strip()]
        skip_rois.extend(parts)

    seg_proc_threads = args.seg_proc_threads
    if seg_proc_threads is not None and seg_proc_threads < 1:
        seg_proc_threads = None

    rad_proc_threads = args.radiomics_proc_threads
    if rad_proc_threads is not None and rad_proc_threads < 1:
        rad_proc_threads = None

    raw_course_filters = args.course_filter or []
    patient_filter_ids: set[str] = set()
    course_filter_pairs: set[tuple[str, str]] = set()
    for raw_entry in raw_course_filters:
        if not raw_entry:
            continue
        token = str(raw_entry).strip()
        if not token:
            continue
        if "/" in token:
            patient_part, course_part = token.split("/", 1)
            patient_part = patient_part.strip()
            course_part = course_part.strip()
            if patient_part and course_part:
                course_filter_pairs.add((patient_part, course_part))
        else:
            patient_filter_ids.add(token)

    filters_active = bool(course_filter_pairs or patient_filter_ids)

    def _filter_courses(seq):
        selected = []
        for course in seq:
            pid = str(getattr(course, "patient_id", ""))
            cid = str(getattr(course, "course_id", ""))
            if not filters_active:
                selected.append(course)
            elif (pid, cid) in course_filter_pairs or pid in patient_filter_ids:
                selected.append(course)
        return selected

    def _log_skip(stage_name: str) -> None:
        if filters_active:
            logger.info("%s: no courses matched course filter; skipping stage", stage_name)

    cfg = PipelineConfig(
        dicom_root=Path(args.dicom_root).resolve(),
        output_root=Path(args.outdir).resolve(),
        logs_root=Path(args.logs).resolve(),
        merge_criteria=args.merge_criteria,
        max_days_between_plans=args.max_days,
        do_segmentation=not args.no_segmentation,
        do_dvh=not args.no_dvh,
        do_visualize=not args.no_visualize,
        do_radiomics=not args.no_radiomics,
        conda_activate=args.conda_activate,
        dcm2niix_cmd=args.dcm2niix,
        totalseg_cmd=args.totalseg,
        totalseg_license_key=args.totalseg_license,
        totalseg_weights_dir=Path(args.totalseg_weights).resolve() if args.totalseg_weights else None,
        extra_seg_models=[m.strip() for part in (args.extra_seg_models or []) for m in part.split(",") if m.strip()],
        segmentation_workers=args.seg_workers,
        segmentation_thread_limit=seg_proc_threads,
        totalseg_fast=args.totalseg_fast,
        totalseg_roi_subset=args.totalseg_roi_subset,
        workers=args.workers,
        radiomics_params_file=Path(args.radiomics_params).resolve() if args.radiomics_params else None,
        radiomics_params_file_mr=Path(args.radiomics_params_mr).resolve() if args.radiomics_params_mr else None,
        radiomics_skip_rois=skip_rois,
        radiomics_max_voxels=args.radiomics_max_voxels,
        radiomics_min_voxels=args.radiomics_min_voxels,
        radiomics_thread_limit=rad_proc_threads,
        custom_structures_config=None,  # Will be set below
        resume=not args.force_redo,  # Resume is default, disable only with --force-redo
    )

    # Parse custom structures configuration
    custom_structures_config = None
    if args.custom_structures:
        # User provided custom config
        custom_structures_config = Path(args.custom_structures).resolve()
        if not custom_structures_config.exists():
            logger.warning("Custom structures config file not found: %s", custom_structures_config)
            custom_structures_config = None
    else:
        # Use default pelvic template if it exists
        default_pelvic_config = Path(__file__).parent.parent / "custom_structures_pelvic.yaml"
        if default_pelvic_config.exists():
            custom_structures_config = default_pelvic_config
            logger.info("Using default pelvic custom structures template: %s", custom_structures_config)
        else:
            logger.info("No custom structures configuration provided")

    # Update config object
    cfg.custom_structures_config = custom_structures_config

    # Configure custom segmentation models
    custom_models_root: Path | None = None
    if args.custom_models_root:
        custom_models_root = Path(args.custom_models_root).resolve()
    else:
        default_models_root = Path.cwd() / "custom_models"
        if default_models_root.exists():
            custom_models_root = default_models_root
    if custom_models_root and not custom_models_root.exists():
        logger.warning("Custom models root not found: %s", custom_models_root)
        custom_models_root = None

    selected_models: list[str] = []
    for entry in args.custom_model or []:
        if not entry:
            continue
        parts = [seg.strip() for seg in str(entry).replace(";", ",").split(",") if seg.strip()]
        selected_models.extend(parts)

    cfg.custom_models_root = custom_models_root
    cfg.custom_model_names = selected_models
    cfg.custom_models_force = bool(args.force_custom_models)
    cfg.custom_models_workers = args.custom_model_workers
    if args.nnunet_predict:
        cfg.nnunet_predict_cmd = args.nnunet_predict
    cfg.custom_models_conda_activate = args.custom_model_conda_activate or cfg.conda_activate
    cfg.custom_models_retain_weights = not bool(args.purge_custom_model_weights)

    # Log resume behavior
    if cfg.resume:
        logging.getLogger(__name__).info("Resume mode enabled (default): skipping existing outputs")
    else:
        logging.getLogger(__name__).info("Force redo mode enabled: regenerating all outputs")

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

    default_order = ["organize", "segmentation", "segmentation_custom", "dvh", "visualize", "radiomics", "qc"]
    requested = [stage.lower() for stage in (args.stage or default_order)]
    stages = [stage for stage in default_order if stage in requested]
    if not stages:
        stages = default_order

    courses: list | None = None

    def ensure_courses() -> list:
        nonlocal courses
        if courses is None:
            courses = organize_and_merge(cfg)
        return courses

    if "organize" in stages:
        courses = organize_and_merge(cfg)
        if not args.no_metadata:
            from .meta import export_metadata
            export_metadata(cfg)

    if "segmentation" in stages:
        from .segmentation import segment_course  # lazy import
        from .auto_rtstruct import build_auto_rtstruct

        courses = ensure_courses()
        selected_courses = _filter_courses(courses)
        if not selected_courses:
            _log_skip("Segmentation")
        else:

            def _segment(course):
                try:
                    segment_course(cfg, course.dirs.root, force=args.force_segmentation)
                    build_auto_rtstruct(course.dirs.root)
                except Exception as exc:
                    logger.warning("Segmentation failed for %s: %s", course.dirs.root, exc)
                return None

            seg_worker_limit = cfg.segmentation_workers if cfg.segmentation_workers is not None else 1
            try:
                seg_worker_limit = int(seg_worker_limit)
            except Exception:
                seg_worker_limit = 1
            if seg_worker_limit < 1:
                seg_worker_limit = 1
            seg_worker_limit = min(seg_worker_limit, max(1, cfg.effective_workers()))

            run_tasks_with_adaptive_workers(
                "Segmentation",
                selected_courses,
                _segment,
                max_workers=seg_worker_limit,
                logger=logging.getLogger(__name__),
                show_progress=True,
            )

    if "segmentation_custom" in stages:
        from .custom_models import discover_custom_models, run_custom_models_for_course  # lazy import

        models_root = cfg.custom_models_root
        available_models = []
        if models_root is None:
            logger.info("Custom segmentation stage skipped: no custom models root configured")
        else:
            try:
                available_models = discover_custom_models(models_root, cfg.custom_model_names, cfg.nnunet_predict_cmd)
            except Exception as exc:
                logger.warning("Failed to discover custom models in %s: %s", models_root, exc)
                available_models = []

        if not available_models:
            logger.info("No custom segmentation models found; skipping custom stage")
        else:
            courses = ensure_courses()
            selected_courses = _filter_courses(courses)
            if not selected_courses:
                _log_skip("Custom Segmentation")
            else:
                force_custom = cfg.custom_models_force

                def _custom(course):
                    try:
                        run_custom_models_for_course(cfg, course, available_models, force=force_custom)
                    except Exception as exc:
                        logger.warning("Custom segmentation failed for %s: %s", course.dirs.root, exc)
                    return None

                custom_worker_limit = cfg.custom_models_workers or 1
                try:
                    custom_worker_limit = int(custom_worker_limit)
                except Exception:
                    custom_worker_limit = 1
                if custom_worker_limit < 1:
                    custom_worker_limit = 1
                custom_worker_limit = min(custom_worker_limit, max(1, cfg.effective_workers()))

                run_tasks_with_adaptive_workers(
                    "CustomSegmentation",
                    selected_courses,
                    _custom,
                    max_workers=custom_worker_limit,
                    logger=logging.getLogger(__name__),
                    show_progress=True,
                )

    if "dvh" in stages:
        from .dvh import dvh_for_course  # lazy import

        courses = ensure_courses()
        selected_courses = _filter_courses(courses)

        if not selected_courses:
            _log_skip("DVH")
        else:

            def _dvh(course):
                try:
                    return dvh_for_course(
                        course.dirs.root,
                        cfg.custom_structures_config,
                        parallel_workers=cfg.effective_workers(),
                    )
                except Exception as exc:
                    logger.warning("DVH failed for %s: %s", course.dirs.root, exc)
                    return None

            run_tasks_with_adaptive_workers(
                "DVH",
                selected_courses,
                _dvh,
                max_workers=cfg.effective_workers(),
                logger=logging.getLogger(__name__),
                show_progress=True,
            )

    if "visualize" in stages:
        from .visualize import generate_axial_review, visualize_course  # lazy import

        courses = ensure_courses()
        selected_courses = _filter_courses(courses)

        if not selected_courses:
            _log_skip("Visualization")
        else:

            def _visualize(course):
                try:
                    visualize_course(course.dirs.root)
                finally:
                    try:
                        generate_axial_review(course.dirs.root)
                    except Exception as exc:
                        logger.debug("Axial review failed for %s: %s", course.dirs.root, exc)
                return None

            run_tasks_with_adaptive_workers(
                "Visualization",
                selected_courses,
                _visualize,
                max_workers=cfg.effective_workers(),
                logger=logging.getLogger(__name__),
                show_progress=True,
            )

    if "radiomics" in stages:
        import os

        courses = ensure_courses()
        selected_courses = _filter_courses(courses)

        if args.sequential_radiomics:
            os.environ['RTPIPELINE_RADIOMICS_SEQUENTIAL'] = '1'
            logger.info("Using sequential radiomics processing (--sequential-radiomics specified)")
        else:
            try:
                from .radiomics_parallel import enable_parallel_radiomics_processing
                enable_parallel_radiomics_processing(cfg.radiomics_thread_limit)
                logger.info("Enabled parallel radiomics processing")
            except ImportError:
                logger.debug("Parallel radiomics helpers unavailable; proceeding with default extractor")

        can_use_radiomics = False
        try:
            from .radiomics import _have_pyradiomics
            can_use_radiomics = _have_pyradiomics()
        except ImportError:
            can_use_radiomics = False

        if not can_use_radiomics:
            try:
                from .radiomics_conda import check_radiomics_env
                can_use_radiomics = check_radiomics_env()
                if can_use_radiomics:
                    logger.info("PyRadiomics will run via dedicated conda environment")
            except ImportError:
                can_use_radiomics = False

        if not can_use_radiomics:
            logger.warning("Radiomics dependencies unavailable; skipping radiomics stage")
        else:
            if not selected_courses:
                _log_skip("Radiomics")
            else:
                from .radiomics import run_radiomics
                try:
                    run_radiomics(cfg, selected_courses, cfg.custom_structures_config)
                except Exception as exc:
                    logger.warning("Radiomics stage failed: %s", exc)

    if "qc" in stages:
        from .quality_control import generate_qc_report

        courses = ensure_courses()
        selected_courses = _filter_courses(courses)

        if not selected_courses:
            _log_skip("QC")
        else:
            for course in selected_courses:
                try:
                    qc_dir = course.dirs.qc_reports
                    qc_dir.mkdir(parents=True, exist_ok=True)
                    generate_qc_report(course.dirs.root, qc_dir)
                except Exception as exc:
                    logger.warning("QC stage failed for %s: %s", course.dirs.root, exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
