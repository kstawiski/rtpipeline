from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    # Inputs/outputs
    dicom_root: Path
    output_root: Path
    logs_root: Path

    # Course merge policy
    merge_criteria: str = "same_ct_study"  # one of: same_ct_study, frame_of_reference
    max_days_between_plans: int | None = None  # optional time window filter (days)

    # Steps
    do_segmentation: bool = True
    do_dvh: bool = True
    do_visualize: bool = True
    do_radiomics: bool = True
    # Resume mode
    resume: bool = False

    # External tools (segmentation)
    conda_activate: str | None = None  # e.g. "source ~/miniconda3/etc/profile.d/conda.sh && conda activate rt"
    dcm2niix_cmd: str = "dcm2niix"
    totalseg_cmd: str = "TotalSegmentator"
    totalseg_license_key: str | None = None
    totalseg_weights_dir: Path | None = None

    # Additional segmentation models (in addition to default 'total')
    extra_seg_models: list[str] = field(default_factory=list)
    segmentation_workers: int | None = None
    segmentation_thread_limit: int | None = None

    # Performance/CPU options
    totalseg_fast: bool = False
    totalseg_roi_subset: str | None = None
    nnunet_predict_cmd: str = "nnUNetv2_predict"

    # Concurrency
    workers: int | None = None  # None => auto (cpu_count - 1)

    # Radiomics
    radiomics_params_file: Path | None = None
    radiomics_skip_rois: list[str] = field(default_factory=list)
    radiomics_max_voxels: int | None = None
    radiomics_min_voxels: int | None = None
    radiomics_thread_limit: int | None = None
    radiomics_params_file_mr: Path | None = None

    # Custom structures
    custom_structures_config: Path | None = None

    # Custom segmentation models
    custom_models_root: Path | None = None
    custom_model_names: list[str] = field(default_factory=list)
    custom_models_force: bool = False
    custom_models_workers: int | None = None
    custom_models_conda_activate: str | None = None
    custom_models_retain_weights: bool = True

    def ensure_dirs(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.logs_root.mkdir(parents=True, exist_ok=True)

    def effective_workers(self) -> int:
        import os as _os
        if self.workers and self.workers > 0:
            return int(self.workers)
        cpu = _os.cpu_count() or 2
        return max(1, cpu - 1)
