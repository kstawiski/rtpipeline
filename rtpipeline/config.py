from dataclasses import dataclass
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

    # External tools (segmentation)
    conda_activate: str | None = None  # e.g. "source ~/miniconda3/etc/profile.d/conda.sh && conda activate rt"
    dcm2niix_cmd: str = "dcm2niix"
    totalseg_cmd: str = "TotalSegmentator"

    def ensure_dirs(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.logs_root.mkdir(parents=True, exist_ok=True)
