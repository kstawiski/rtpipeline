import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    # Inputs/outputs
    dicom_root: Path
    output_root: Path
    logs_root: Path
    max_workers_override: int | None = None

    # Course merge policy
    merge_criteria: str = "same_ct_study"  # one of: same_ct_study, frame_of_reference
    max_days_between_plans: int | None = None  # optional time window filter (days)

    # Steps
    do_segmentation: bool = True
    do_dvh: bool = True
    do_visualize: bool = True
    do_radiomics: bool = True
    do_segment_all_series: bool = False
    do_ingest_pet_suv: bool = False
    # Resume mode
    resume: bool = False

    # External tools (segmentation)
    conda_activate: str | None = None  # e.g. "source ~/miniconda3/etc/profile.d/conda.sh && conda activate rt"
    dcm2niix_cmd: str = "dcm2niix"
    totalseg_cmd: str = "TotalSegmentator"
    totalseg_license_key: str | None = None
    totalseg_weights_dir: Path | None = None
    totalseg_device: str = "gpu"
    totalseg_force_split: bool = True
    totalseg_nr_thr_resamp: int | None = None  # Auto-detect (25-50% of cores)
    totalseg_nr_thr_saving: int | None = None  # Auto-detect (25-50% of cores)
    totalseg_num_proc_pre: int | None = 1      # Keep at 1 for Docker stability
    totalseg_num_proc_export: int | None = 1   # Keep at 1 for Docker stability
    totalseg_allow_fallback: bool = False
    cbct_totalseg_extra_args: list[str] = field(default_factory=lambda: ["--body_seg"])
    # All-series segmentation scope. None => segment every eligible image_class (legacy behavior).
    # When set, only series whose image_class is in this allow-list are segmented in the all-series stage;
    # excluded series stay materialized but unsegmented.
    all_series_segment_classes: list[str] | None = None
    # When True, segment at most ONE representative 4DCT volume per patient: the first fourdct_ave
    # (averaged reconstruction) if any exists, else the first fourdct_phase. All other 4DCT series are
    # left materialized but unsegmented. No effect when 4DCT classes are not in all_series_segment_classes.
    all_series_fourdct_single_representative: bool = False
    # Optional allow-list of image_class names to MATERIALIZE (byte-copy) in the all-series inventory
    # path. None (default) => materialize every non-excluded series (legacy behavior). An explicit list
    # ([] included) is an allow-list: series whose image_class is not in it are recorded in the manifest
    # with status "materialize_skipped_out_of_scope" and NOT copied (matches the all_series_segment_classes
    # contract: [] => none, None => all). This avoids the heavy per-file NFS copy of classes nothing
    # downstream consumes (e.g. CBCT, 70-90% of per-patient DICOM files, never segmented/extracted when not
    # in all_series_segment_classes). Fail-closed: the materialization stage unions the EFFECTIVE
    # segmentation scope into this set (the explicit all_series_segment_classes, or every
    # TotalSegmentator-eligible class when segmentation runs over all classes) so a class that will be
    # segmented is never skip-materialized.
    all_series_materialize_classes: list[str] | None = None

    # Additional segmentation models (in addition to default 'total')
    extra_seg_models: list[str] = field(default_factory=list)
    segmentation_workers: int | None = None
    segmentation_thread_limit: int | None = None
    segmentation_temp_root: Path | None = None

    # PET SUV ingestion. Disabled by default.
    suv_decay_guard_tol: float = 0.02
    suv_zextent_primary_fraction: float = 0.90
    pet_clinical_weight_window_days: int = 30

    # Performance/CPU options
    totalseg_fast: bool = False
    totalseg_roi_subset: str | None = None
    nnunet_predict_cmd: str = "nnUNetv2_predict"

    # Radiomics
    radiomics_params_file: Path | None = None
    radiomics_skip_rois: list[str] = field(default_factory=list)
    radiomics_max_voxels: int | None = None
    radiomics_min_voxels: int | None = None
    radiomics_thread_limit: int | None = None
    radiomics_params_file_mr: Path | None = None

    # Custom structures
    custom_structures_config: Path | None = None

    # Systematic CT cropping for consistent volume analysis
    ct_cropping_enabled: bool = False
    ct_cropping_region: str = "pelvis"  # "pelvis", "thorax", "abdomen", "head_neck", "brain"
    # None means "use the region-specific default" (see anatomical_cropping.apply_systematic_cropping)
    ct_cropping_superior_margin_cm: float | None = None
    ct_cropping_inferior_margin_cm: float | None = None
    ct_cropping_use_for_dvh: bool = True
    # NOTE: cropped RTSTRUCTs (e.g., RS_auto_cropped.dcm) have shown geometric
    # misregistration issues when paired with the original DICOM CT series.
    # Until a robust cropped-geometry path exists for CT radiomics, default to
    # using the original (uncropped) RTSTRUCTs for radiomics extraction.
    ct_cropping_use_for_radiomics: bool = False
    # Deprecated, ignored: uncropped files are always kept (see
    # anatomical_cropping.apply_systematic_cropping). Retained for backward
    # compatibility with existing config files/callers.
    ct_cropping_keep_original: bool = True

    # Custom segmentation models
    custom_models_root: Path | None = None
    custom_model_names: list[str] = field(default_factory=list)
    custom_models_force: bool = False
    custom_models_workers: int | None = None
    custom_models_conda_activate: str | None = None
    custom_models_retain_weights: bool = True

    # Body region QC and model gating
    # Maps model names to their required body regions and confidence thresholds
    # Example: {"cardiac_STOPSTORM": {"required_regions": ["THORAX"], "min_confidence": 0.6}}
    model_region_requirements: dict[str, dict] = field(default_factory=lambda: {
        # Custom models (cardiac)
        "cardiac_STOPSTORM": {"required_regions": ["THORAX"], "min_confidence": 0.6},
        # Thoracic tumor comparator models
        "lung_tumor_totalseg_lung_nodules": {"required_regions": ["THORAX"], "min_confidence": 0.5},
        "lung_tumor_pancancer_lung": {"required_regions": ["THORAX"], "min_confidence": 0.5},
        "lung_tumor_medsam_boxprompt": {"required_regions": ["THORAX"], "min_confidence": 0.5},
        # TotalSegmentator extra models (thorax-specific)
        "heartchambers_highres": {"required_regions": ["THORAX"], "min_confidence": 0.5},
        "coronary_arteries": {"required_regions": ["THORAX"], "min_confidence": 0.6},
        # Head/neck models
        "head_neck_oar": {"required_regions": ["HEAD_NECK"], "min_confidence": 0.5},
        "head_glands_cavities": {"required_regions": ["HEAD_NECK"], "min_confidence": 0.5},
        # Generic models (no region requirements)
        "total": {"required_regions": []},
        "total_mr": {"required_regions": []},
    })
    # Whether to block models when required regions are missing (True) or just warn (False)
    body_region_qc_block_missing: bool = True

    # Radiomics robustness analysis
    radiomics_robustness_enabled: bool = True
    radiomics_robustness_config: dict = field(default_factory=dict)

    # Safety timeouts
    task_timeout: int | None = None

    # DICOM copy optimization (organize step)
    dicom_copy_dedup_by_sop_uid: bool = True    # Skip duplicate SOPInstanceUIDs
    dicom_copy_use_hardlinks: bool = False      # Avoid hardlinks in outputs by default
    dicom_copy_verify_checksum: bool = False    # Verify MD5 after copy
    dicom_copy_cache_headers: bool = True       # Cache DICOM headers for re-runs

    # Inventory-driven all-series discovery.
    inventory_db_path: Path | None = None
    inventory_scan_run_id: int | None = None
    inventory_patient_ids: list[str] = field(default_factory=list)
    # Optional cohort scope for organize-stage discovery walks (RT/CT/series).
    # When non-empty, RT/series/CT indexing only traverses these top-level
    # patient directories instead of the entire dicom_root. Empty => walk
    # everything (unchanged behaviour). Populated by the CLI from active
    # course/patient filters so a filtered run does not stat the whole DICOM tree.
    discover_patient_ids: list[str] = field(default_factory=list)
    cbct_manufacturer_models: list[str] = field(default_factory=lambda: [
        "Patient Verification",
        "Halcyon - PVA",
        "OBI Cone-beam CT",
        "Acuity Cone-beam CT",
        "RDS - PVA",
    ])
    fourdct_models: list[str] = field(default_factory=lambda: [
        "Advanced Reconstruction",
        "ARIA RTM",
    ])

    def ensure_dirs(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.logs_root.mkdir(parents=True, exist_ok=True)

    def effective_workers(self) -> int:
        def _coerce(value: int | str | None) -> int | None:
            if value is None:
                return None
            try:
                ivalue = int(value)
            except (TypeError, ValueError):
                return None
            return ivalue if ivalue > 0 else None

        base = max(1, (os.cpu_count() or 2) - 1)
        override = _coerce(self.max_workers_override)
        if override is None:
            override = _coerce(os.environ.get("RTPIPELINE_MAX_WORKERS"))
        if override is not None:
            base = max(1, min(base, override))
        return base
