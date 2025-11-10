# rtpipeline Web UI Documentation

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Output Structure](#output-structure)
- [Advanced Configuration](#advanced-configuration)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)
- [Security Considerations](#security-considerations)
- [Performance Tips](#performance-tips)
- [Examples](#examples)
- [Support](#support)

## Overview

The rtpipeline Web UI provides a browser-based interface for uploading DICOM files and processing them through the rtpipeline automated radiotherapy data processing workflow. This eliminates the need for command-line interaction and provides a user-friendly way to manage multiple processing jobs.

**Access the Web UI:** After starting the container with `docker-compose up -d`, open your browser to **http://localhost:8080**

## Features

- **🎯 Drag-and-Drop Upload**: Easily upload DICOM files by dragging them into the browser
- **📁 Multiple Format Support**: Accepts .dcm files, .zip archives, tar archives, DICOMDIR, and entire directories
- **✅ Automatic Validation**: Validates uploaded DICOM files and provides feedback on data quality
- **⚙️ Configurable Processing**: Customize pipeline settings (segmentation, radiomics, CT cropping)
- **📊 Progress Monitoring**: Real-time progress tracking for all processing jobs
- **📥 Easy Results Download**: Download all results as a single ZIP archive
- **🔍 Log Viewer**: View detailed processing logs for debugging

## Quick Start

### 1. Start the Docker Container

**GPU-enabled (default):**
```bash
docker-compose up -d
```

**CPU-only:**
```bash
docker-compose --profile cpu-only up -d
```

### 2. Access the Web UI

Open your browser and navigate to:
```
http://localhost:8080
```

### 3. Upload DICOM Files

You can upload DICOM files in several ways:

#### Method A: Drag and Drop
- Drag files or folders directly into the upload area
- The interface will accept multiple files and folders

#### Method B: Click to Browse
- Click the "Browse Files" button
- Select files or folders from your computer

#### Supported Formats
- `.dcm` - Individual DICOM files
- `.zip` - ZIP archives containing DICOM files
- `.tar`, `.tar.gz`, `.tgz` - TAR archives
- `DICOMDIR` - DICOM directory files
- Entire directories with subdirectories

### 4. Review Validation Results

After upload, the system will automatically:
- Extract files from archives
- Scan for DICOM files
- Validate DICOM structure
- Display summary statistics:
  - Number of DICOM files found
  - Patients, studies, and series counts
  - Modalities detected (CT, MR, RTPLAN, RTDOSE, RTSTRUCT)
- Show warnings and suggestions

**Example Warnings:**
- ⚠️ No CT or MR imaging data found
- ⚠️ No RT DOSE or RT STRUCT found
- ⚠️ Multiple patients found

**Example Suggestions:**
- 💡 rtpipeline requires CT or MR images for processing
- 💡 For DVH analysis, you need RTDOSE and RTSTRUCT files
- 💡 Segmentation and radiomics can run without RT data

### 5. Configure Processing Options

Before starting processing, you can customize:

#### Fast Mode (CPU-friendly segmentation)
- ✅ **Enabled (default)**: Uses faster, CPU-optimized segmentation
- ❌ **Disabled**: Uses standard GPU-accelerated mode (requires GPU)

#### Radiomics Extraction
- ✅ **Enabled (default)**: Extracts radiomic features from CT and MR
- ❌ **Disabled**: Skips radiomics (faster processing)

#### Custom Models
- ❌ **Disabled (default)**: Only uses TotalSegmentator
- ✅ **Enabled**: Runs custom nnUNet models (if available in `/data/models`)

#### CT Anatomical Cropping
- ❌ **Disabled (default)**: Processes full CT volumes
- ✅ **Enabled**: Crops CT to anatomical regions (pelvis, thorax, etc.)

### 6. Start Processing

Click **🚀 Start Processing** to begin pipeline execution.

The job will:
1. **Organize** DICOM files into courses
2. **Segment** CT/MR images (TotalSegmentator)
3. **Run custom models** (if enabled)
4. **Compute DVH metrics** (if RTDOSE/RTSTRUCT available)
5. **Extract radiomics** (if enabled)
6. **Generate QC reports**
7. **Aggregate results**

### 7. Monitor Progress

The Jobs section shows all processing jobs with:
- **Status badges**: uploaded, running, completed, failed, cancelled
- **Progress bar**: Real-time percentage and current stage
- **Timestamps**: Creation and update times

**Progress Stages:**
- Organizing DICOM files... (20%)
- Running segmentation... (40%)
- Computing DVH metrics... (60%)
- Extracting radiomics features... (80%)
- Aggregating results... (90%)
- Pipeline completed successfully (100%)

### 8. View Logs

Click **View Logs** on any job to see detailed processing logs in a modal window.

Logs include:
- Snakemake workflow output
- TotalSegmentator progress
- DVH computation details
- Radiomics extraction status
- Error messages (if any)

### 9. Download Results

Once a job is completed:
1. Click **Download Results**
2. A ZIP archive will be downloaded containing:
   - `_RESULTS/` - Aggregated Excel files
     - `dvh_metrics.xlsx` - DVH metrics for all structures
     - `radiomics_ct.xlsx` - CT radiomic features
     - `radiomics_mr.xlsx` - MR radiomic features
     - `case_metadata.xlsx` - Patient and course metadata
   - Per-patient/course directories with:
     - NIFTI images
     - Segmentation masks
     - DICOM RTSTRUCT files
     - QC reports

### 10. Manage Jobs

#### Cancel Running Job
- Click **Cancel** to stop a running job
- The process will be terminated gracefully

#### Delete Job
- Click **Delete** to remove a completed/failed job
- All input, output, and log files will be deleted
- **Warning**: This action cannot be undone

## Output Structure

Results are organized as follows:

```
/data/output/
└── <job-id>/
    ├── _RESULTS/                    # Aggregated results
    │   ├── dvh_metrics.xlsx         # All DVH metrics
    │   ├── radiomics_ct.xlsx        # All CT radiomics
    │   ├── radiomics_mr.xlsx        # All MR radiomics
    │   ├── case_metadata.xlsx       # Metadata
    │   ├── fractions.xlsx           # Fraction data
    │   └── qc_reports.xlsx          # Quality control
    └── <patient_id>/
        └── <course_id>/
            ├── DICOM/               # Original DICOM files
            ├── NIFTI/               # NIfTI images
            ├── Segmentation_TotalSegmentator/
            ├── Segmentation_CustomModels/
            ├── dvh_metrics.xlsx
            ├── radiomics_ct.xlsx
            └── qc_reports/
```

## Advanced Configuration

### Custom Configuration Files

For advanced users, you can mount a custom config file:

1. Create `custom_config.yaml` based on `/app/config.container.yaml`
2. Mount it in `docker-compose.yml`:
   ```yaml
   volumes:
     - ./custom_config.yaml:/app/custom_config.yaml
   ```
3. Jobs will use default config unless you implement custom config selection in UI

### Environment Variables

Available environment variables in `docker-compose.yml`:

- `PORT`: Web UI port (default: 8080)
- `DEBUG`: Enable Flask debug mode (default: false)
- `PYTHONPATH`: Python path (default: /app)
- `NVIDIA_VISIBLE_DEVICES`: GPU visibility (default: all)

### Resource Limits

Default resource allocation (in `docker-compose.yml`):

**GPU mode:**
- CPUs: 4-24 cores
- RAM: 8-64 GB
- Shared memory: 4 GB

**CPU-only mode:**
- CPUs: 4-16 cores
- RAM: 8-32 GB
- Shared memory: 4 GB

Adjust in `docker-compose.yml` under `deploy.resources` if needed.

## Troubleshooting

### Upload Issues

**Problem**: Files won't upload
- **Check**: Browser console for errors (F12)
- **Check**: File size limits (default: 50 GB per upload)
- **Solution**: Try smaller batches or increase `MAX_CONTENT_LENGTH` in `webui/app.py`

**Problem**: No DICOM files found
- **Check**: Files are actually DICOM format
- **Check**: Archives extract correctly
- **Solution**: Use `pydicom` to verify DICOM files locally

### Processing Failures

**Problem**: Job stuck at 0%
- **Check**: View logs for error messages
- **Check**: Docker container logs: `docker logs rtpipeline`
- **Solution**: Ensure Snakemake can run: `docker exec rtpipeline snakemake --version`

**Problem**: Segmentation fails
- **GPU mode**: Check GPU availability: `docker exec rtpipeline nvidia-smi`
- **CPU mode**: Enable fast mode in configuration
- **Solution**: Check TotalSegmentator installation

**Problem**: Out of memory errors
- **Solution**: Increase memory limits in `docker-compose.yml`
- **Solution**: Process fewer cases at once
- **Solution**: Enable CT cropping to reduce volume size

### Results Issues

**Problem**: Download fails
- **Check**: Job completed successfully (status: completed)
- **Check**: Output directory has files: `docker exec rtpipeline ls /data/output/<job-id>`
- **Solution**: Manually copy from `./Output/<job-id>/`

**Problem**: Missing results files
- **Check**: Job logs for stage failures
- **Check**: Specific stages completed in Snakemake output
- **Solution**: Re-run with `keep_going: true` in config

## API Reference

The Web UI provides a REST API for programmatic access:

### Upload Files
```http
POST /api/upload
Content-Type: multipart/form-data

files[]: <binary>
```

### Start Job
```http
POST /api/jobs/{job_id}/start
Content-Type: application/json

{
  "segmentation": {"fast": true},
  "radiomics": {"enabled": true},
  "custom_models_enabled": false,
  "ct_cropping": {"enabled": false}
}
```

### Get Job Status
```http
GET /api/jobs/{job_id}/status
```

### Get Job Logs
```http
GET /api/jobs/{job_id}/logs
```

### Download Results
```http
GET /api/jobs/{job_id}/download
```

### Cancel Job
```http
POST /api/jobs/{job_id}/cancel
```

### Delete Job
```http
DELETE /api/jobs/{job_id}
```

### List All Jobs
```http
GET /api/jobs
```

### Health Check
```http
GET /health
```

## Security Considerations

### File Upload Security

- Files are validated before processing
- Secure filename sanitization prevents path traversal
- File type validation prevents malicious uploads
- Upload size limits prevent DoS attacks

### Custom Models

- Custom models are **disabled by default**
- Enable only if you trust the model source
- Models run with full system access (container isolated)

### Network Security

- Web UI runs on port 8080 (not exposed to internet by default)
- For production: Use reverse proxy (nginx) with HTTPS
- For production: Implement authentication (OAuth, basic auth)

## Performance Tips

### Upload Performance
- Use ZIP/TAR archives for many small files (faster than individual uploads)
- Local network transfers are faster than remote
- Compress archives with pigz/gzip for faster extraction

### Processing Performance
- **GPU mode**: Much faster for segmentation (5-10x speedup)
- **Fast mode**: Good balance for CPU-only systems
- **Parallel processing**: Pipeline runs stages in parallel automatically
- **Resource allocation**: Increase CPU/RAM for faster processing

### Storage Management
- Delete old jobs regularly to free space
- Results can be large (1-10 GB per patient)
- Use compression for long-term storage
- Archive to external storage periodically

## Examples

### Example 1: Single Patient CT+RTDOSE+RTSTRUCT

1. Upload ZIP file: `patient_001.zip`
   - Contains: CT slices, RTPLAN, RTDOSE, RTSTRUCT
2. Validation shows:
   - ✅ 150 DICOM files
   - ✅ Modalities: CT, RTPLAN, RTDOSE, RTSTRUCT
3. Configure: Fast mode ✅, Radiomics ✅
4. Start processing
5. Results include:
   - Automatic segmentation (100+ structures)
   - DVH metrics for all structures
   - Radiomic features
   - QC report

### Example 2: Multiple Patients, CT-only

1. Upload directory with subdirectories
   - `patient_001/` → CT series
   - `patient_002/` → CT series
   - `patient_003/` → CT series
2. Validation shows:
   - ⚠️ No RT data found
   - 💡 Segmentation and radiomics can run without RT data
3. Configure: Fast mode ✅, Radiomics ✅
4. Start processing
5. Results include:
   - Segmentation for all patients
   - Radiomic features
   - No DVH metrics (no RTDOSE)

### Example 3: CT+MR with Registered Series

1. Upload DICOMDIR
   - Contains: CT, MR, REG objects
2. Validation shows:
   - ✅ Modalities: CT, MR, REG
3. Configure: Fast mode ✅, Radiomics ✅
4. Start processing
5. Results include:
   - CT segmentation
   - MR segmentation (via total_mr)
   - CT radiomics
   - MR radiomics
   - Registered series preservation

## Support

For issues, questions, or feature requests:
- **GitHub Issues**: https://github.com/kstawiski/rtpipeline/issues
- **Documentation**: See `docs/` directory
- **Docker Logs**: `docker logs rtpipeline`
- **Web UI Logs**: Check `/data/logs/` directory

## License

Same as rtpipeline main project.
