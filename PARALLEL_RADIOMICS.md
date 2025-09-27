# Parallel Radiomics Processing

This document describes the parallel radiomics functionality in rtpipeline, which can help avoid segmentation faults and improve performance.

## Overview

**Parallel radiomics processing is now the default behavior** in rtpipeline. The parallel radiomics module (`rtpipeline.radiomics_parallel`) provides process-based parallel processing for radiomics feature extraction. This addresses two key issues:

1. **Segmentation Faults**: The sequential version can experience segmentation faults due to pyradiomics/OpenMP interactions
2. **Performance**: Parallel processing can significantly speed up radiomics extraction for large datasets

Since parallel processing is more stable and faster, it's enabled by default for all users.

## Usage

### Command Line

**Parallel radiomics processing is enabled by default** for improved performance and stability. To use sequential processing instead:

```bash
# Default behavior (parallel radiomics)
rtpipeline --dicom-root /path/to/dicom --outdir /path/to/output

# Force sequential processing (if needed)
rtpipeline --dicom-root /path/to/dicom --outdir /path/to/output --sequential-radiomics
```

### Programmatic Usage

```python
import os
from pathlib import Path
from rtpipeline.config import PipelineConfig
from rtpipeline.radiomics_parallel import enable_parallel_radiomics_processing
from rtpipeline.radiomics import radiomics_for_course

# Enable parallel processing
enable_parallel_radiomics_processing()

# Or set environment variable directly
os.environ['RTPIPELINE_USE_PARALLEL_RADIOMICS'] = '1'

# Configure pipeline
config = PipelineConfig(
    dicom_root=Path('/path/to/dicom'),
    output_root=Path('/path/to/output'),
    logs_root=Path('/path/to/logs')
)

# Run radiomics (will automatically use parallel processing)
course_dir = Path('/path/to/course')
custom_config = Path('/path/to/custom_structures.yaml')
result = radiomics_for_course(config, course_dir, custom_config)
```

## Configuration

### Environment Variables

- `RTPIPELINE_USE_PARALLEL_RADIOMICS`: Set to `1`, `true`, or `yes` to enable parallel processing
- `RTPIPELINE_RADIOMICS_WORKERS`: Number of parallel workers (default: auto-detect, max 4)

### Threading Control

The parallel implementation automatically sets these environment variables in worker processes:
- `OMP_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `NUMEXPR_NUM_THREADS=1`
- `NUMBA_NUM_THREADS=1`

## Technical Details

### Process-Based Parallelism

The implementation uses `multiprocessing` with the `spawn` context to avoid fork-related issues. Each radiomics extraction runs in a separate process with isolated memory space.

### Task Serialization

Images and masks are serialized via temporary pickle files to enable cross-process communication while maintaining SimpleITK image metadata.

### Error Handling

- Falls back to sequential processing if parallel module import fails
- Individual task failures don't stop the entire batch
- Comprehensive logging for debugging

### Performance Considerations

- Default worker count: `min(4, max(1, num_tasks // 2))`
- Process spawn overhead vs. parallelization benefit
- Memory usage scales with number of workers

## Troubleshooting

### Segmentation Faults
If you experience segmentation faults with the sequential version:
1. Enable parallel processing with `--parallel-radiomics`
2. Reduce worker count with `RTPIPELINE_RADIOMICS_WORKERS=2`

### Import Errors
If parallel radiomics module fails to import:
- Check that all dependencies are available
- Falls back to sequential processing automatically

### Performance Issues
If parallel processing is slower than expected:
- Check that you have sufficient CPU cores
- Verify that tasks aren't I/O bound
- Consider reducing worker count for memory-constrained systems

## Custom Structures Support

The parallel implementation fully supports custom structures created via Boolean operations:

```bash
rtpipeline --dicom-root /path/to/dicom --parallel-radiomics --custom-structures /path/to/pelvic_config.yaml
```

Custom structures are processed alongside manual and automatic segmentations, with all results combined in the final radiomics output file.