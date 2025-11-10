# rtpipeline Web UI

A Flask-based web interface for rtpipeline DICOM processing.

## Components

- **app.py**: Main Flask application with REST API
- **dicom_validator.py**: DICOM file validation and analysis
- **job_manager.py**: Job queue and processing management
- **templates/index.html**: Single-page web interface
- **requirements.txt**: Python dependencies

## Running Locally (Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export PORT=8080
export DEBUG=true

# Run the app
python app.py
```

## Running in Docker

The Web UI is integrated into the main rtpipeline Docker container:

```bash
# Start with GPU support (default)
docker-compose up -d

# Start CPU-only
docker-compose --profile cpu-only up -d
```

Access at: http://localhost:8080

## API Endpoints

### Upload
- `POST /api/upload` - Upload DICOM files

### Job Management
- `POST /api/jobs/{job_id}/start` - Start processing
- `GET /api/jobs/{job_id}/status` - Get status
- `GET /api/jobs/{job_id}/logs` - Get logs
- `GET /api/jobs/{job_id}/results` - List results
- `GET /api/jobs/{job_id}/download` - Download results ZIP
- `POST /api/jobs/{job_id}/cancel` - Cancel job
- `DELETE /api/jobs/{job_id}` - Delete job
- `GET /api/jobs` - List all jobs

### Health
- `GET /health` - Health check

## Directory Structure

```
/data/
├── input/        # Extracted DICOM files (per job)
├── output/       # Pipeline results (per job)
├── logs/         # Processing logs
└── uploads/      # Temporary upload storage
```

## Configuration

Environment variables:
- `PORT`: Server port (default: 8080)
- `DEBUG`: Debug mode (default: false)

## Architecture

1. **Upload Handler**: Receives files, extracts archives, stores in `/data/uploads/{job_id}`
2. **DICOM Validator**: Scans for DICOM files, validates structure, provides suggestions
3. **Job Manager**:
   - Creates job with unique ID
   - Generates job-specific config YAML
   - Launches Snakemake in background thread
   - Monitors progress via log parsing
   - Manages job lifecycle (running, completed, failed, cancelled)
4. **Results Handler**: Creates ZIP archives, serves downloads

## Security

- File upload validation and sanitization
- Custom models disabled by default
- Process isolation via containerization
- Resource limits via Docker

## Development

### Adding New Features

1. Update API endpoints in `app.py`
2. Update frontend in `templates/index.html`
3. Update validation logic in `dicom_validator.py`
4. Update job processing in `job_manager.py`

### Testing

```bash
# Upload test
curl -X POST http://localhost:8080/api/upload \
  -F "files[]=@test.dcm"

# Start job
curl -X POST http://localhost:8080/api/jobs/{job_id}/start \
  -H "Content-Type: application/json" \
  -d '{"segmentation": {"fast": true}}'

# Check status
curl http://localhost:8080/api/jobs/{job_id}/status
```
