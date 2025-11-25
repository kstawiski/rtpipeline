"""
rtpipeline Web UI - Flask Application
Provides a web interface for uploading DICOM files and processing them through rtpipeline
"""

import os
import uuid
import shutil
import zipfile
import psutil
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import logging
import pydicom

from dicom_validator import DICOMValidator
from job_manager import JobManager
import rtpipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', str(uuid.uuid4()))
# 50GB max upload - large limit to handle complete patient imaging datasets
# Note: For very large uploads, consider splitting into smaller batches to avoid
# browser timeouts and reduce memory consumption
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 * 1024  # 50GB max upload

# Directories configuration
BASE_DIR = Path('/data')
UPLOAD_DIR = BASE_DIR / 'uploads'
INPUT_DIR = BASE_DIR / 'input'
OUTPUT_DIR = BASE_DIR / 'output'
LOGS_DIR = BASE_DIR / 'logs'

# Create necessary directories
for directory in [UPLOAD_DIR, INPUT_DIR, OUTPUT_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Initialize validators and job manager
dicom_validator = DICOMValidator()
job_manager = JobManager(INPUT_DIR, OUTPUT_DIR, LOGS_DIR)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.dcm', '.zip', '.tar', '.gz', '.tgz', '.tar.gz', '.dicomdir'}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)


@app.route('/')
def index():
    """Main page with upload interface"""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_files():
    """
    Handle file uploads
    Accepts: DICOM files, ZIP archives, directories
    """
    try:
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files[]')
        if not files:
            return jsonify({'error': 'No files selected'}), 400

        # Create unique job ID
        job_id = str(uuid.uuid4())
        job_upload_dir = UPLOAD_DIR / job_id
        job_upload_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing upload for job {job_id}")

        uploaded_files = []
        for file in files:
            if file.filename == '':
                continue

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = job_upload_dir / filename
                file.save(str(file_path))
                uploaded_files.append({
                    'name': filename,
                    'path': str(file_path),
                    'size': file_path.stat().st_size
                })
                logger.info(f"Saved file: {filename}")

        if not uploaded_files:
            shutil.rmtree(job_upload_dir, ignore_errors=True)
            return jsonify({'error': 'No valid files uploaded'}), 400

        # Extract and organize files
        extraction_result = extract_and_organize(job_id, uploaded_files)

        if not extraction_result['success']:
            return jsonify({
                'error': 'Failed to extract files',
                'details': extraction_result.get('error')
            }), 400

        # Validate DICOM files
        validation_result = dicom_validator.validate_directory(
            extraction_result['dicom_dir']
        )

        # Create job
        job_manager.create_job(
            job_id=job_id,
            dicom_dir=extraction_result['dicom_dir'],
            validation_result=validation_result,
            uploaded_files=uploaded_files
        )

        return jsonify({
            'job_id': job_id,
            'status': 'uploaded',
            'validation': validation_result,
            'uploaded_files': len(uploaded_files),
            'dicom_files_found': validation_result.get('dicom_count', 0),
            'message': 'Files uploaded successfully'
        }), 200

    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


def extract_and_organize(job_id, uploaded_files):
    """
    Extract uploaded files and organize DICOM files
    Handles: ZIP files, direct DICOM files, nested directories
    """
    try:
        job_dicom_dir = INPUT_DIR / job_id
        job_dicom_dir.mkdir(parents=True, exist_ok=True)

        for file_info in uploaded_files:
            file_path = Path(file_info['path'])

            # Handle ZIP files
            if file_path.suffix.lower() == '.zip':
                logger.info(f"Extracting ZIP: {file_path.name}")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(job_dicom_dir)

            # Handle TAR files
            elif file_path.suffix.lower() in ['.tar', '.gz', '.tgz'] or '.tar.gz' in file_path.name.lower():
                logger.info(f"Extracting TAR: {file_path.name}")
                import tarfile
                with tarfile.open(file_path, 'r:*') as tar_ref:
                    tar_ref.extractall(job_dicom_dir)

            # Handle direct DICOM files or DICOMDIR
            else:
                dest_path = job_dicom_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                logger.info(f"Copied file: {file_path.name}")

        return {
            'success': True,
            'dicom_dir': str(job_dicom_dir)
        }

    except Exception as e:
        logger.error(f"Extraction error: {str(e)}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


@app.route('/api/jobs/<job_id>/start', methods=['POST'])
def start_job(job_id):
    """Start processing a job"""
    try:
        config = request.get_json() or {}
        result = job_manager.start_job(job_id, config)

        if result['success']:
            return jsonify({
                'job_id': job_id,
                'status': 'running',
                'message': 'Job started successfully'
            }), 200
        else:
            return jsonify({'error': result.get('error')}), 400

    except Exception as e:
        logger.error(f"Start job error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs/<job_id>/status', methods=['GET'])
def get_job_status(job_id):
    """Get job status and progress"""
    try:
        status = job_manager.get_job_status(job_id)

        if status:
            return jsonify(status), 200
        else:
            return jsonify({'error': 'Job not found'}), 404

    except Exception as e:
        logger.error(f"Get status error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs/<job_id>/logs', methods=['GET'])
def get_job_logs(job_id):
    """Get job logs"""
    try:
        logs = job_manager.get_job_logs(job_id)
        return jsonify({'logs': logs}), 200

    except Exception as e:
        logger.error(f"Get logs error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id):
    """Cancel a running job"""
    try:
        result = job_manager.cancel_job(job_id)

        if result['success']:
            return jsonify({
                'job_id': job_id,
                'status': 'cancelled',
                'message': 'Job cancelled successfully'
            }), 200
        else:
            return jsonify({'error': result.get('error')}), 400

    except Exception as e:
        logger.error(f"Cancel job error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs/<job_id>/results', methods=['GET'])
def get_job_results(job_id):
    """Get list of result files"""
    try:
        results = job_manager.get_job_results(job_id)
        return jsonify({'results': results}), 200

    except Exception as e:
        logger.error(f"Get results error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs/<job_id>/download', methods=['GET'])
def download_results(job_id):
    """Download all results as ZIP"""
    try:
        zip_path = job_manager.create_results_archive(job_id)

        if zip_path and Path(zip_path).exists():
            return send_file(
                zip_path,
                as_attachment=True,
                download_name=f'rtpipeline_results_{job_id}.zip',
                mimetype='application/zip'
            )
        else:
            return jsonify({'error': 'Results not available'}), 404

    except Exception as e:
        logger.error(f"Download error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """List all jobs"""
    try:
        jobs = job_manager.list_jobs()
        return jsonify({'jobs': jobs}), 200

    except Exception as e:
        logger.error(f"List jobs error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/config/custom_models', methods=['GET'])
def list_custom_models():
    """List available custom models"""
    try:
        models_dir = BASE_DIR / 'models' # Mapped to /data/models in container
        models = []
        
        if models_dir.exists():
            for item in models_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    # Check if it looks like a valid model (e.g. has weights or config)
                    # For now, just listing directories is sufficient
                    models.append(item.name)
        
        return jsonify({'models': sorted(models)}), 200
    except Exception as e:
        logger.error(f"List models error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs/<job_id>', methods=['DELETE'])
def delete_job(job_id):
    """Delete a job and its data"""
    try:
        result = job_manager.delete_job(job_id)

        if result['success']:
            return jsonify({
                'job_id': job_id,
                'message': 'Job deleted successfully'
            }), 200
        else:
            return jsonify({'error': result.get('error')}), 400

    except Exception as e:
        logger.error(f"Delete job error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs/<job_id>/report/<path:filename>', methods=['GET'])
def view_report(job_id, filename):
    """Serve HTML reports and assets"""
    try:
        status = job_manager.get_job_status(job_id)
        if not status:
            return jsonify({'error': 'Job not found'}), 404

        output_dir = Path(status['output_dir']).resolve()
        file_path = (output_dir / filename).resolve()

        # Security check: ensure file is within output directory
        if not str(file_path).startswith(str(output_dir)):
            return jsonify({'error': 'Access denied'}), 403

        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404

        return send_file(file_path)

    except Exception as e:
        logger.error(f"Report view error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': rtpipeline.__version__
    }), 200


@app.route('/api/system/status')
def system_status():
    """Get system resource usage"""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        gpu_stats = []
        try:
            # Basic NVIDIA GPU check if nvidia-smi is available
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    idx, name, util, mem_used, mem_total = line.split(', ')
                    gpu_stats.append({
                        'index': idx,
                        'name': name,
                        'utilization': float(util),
                        'memory_used': float(mem_used),
                        'memory_total': float(mem_total),
                        'memory_percent': round((float(mem_used) / float(mem_total)) * 100, 1)
                    })
        except Exception as exc:
            logger.warning("Failed to collect GPU stats: %s", exc)

        return jsonify({
            'cpu': {
                'percent': cpu_percent,
                'count': psutil.cpu_count()
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent
            },
            'disk': {
                'total': disk.total,
                'free': disk.free,
                'percent': disk.percent
            },
            'gpu': gpu_stats
        })
    except Exception as e:
        logger.error(f"System status error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs/<job_id>/preview')
def preview_results(job_id):
    """Get preview data from result files"""
    try:
        status = job_manager.get_job_status(job_id)
        if not status:
            return jsonify({'error': 'Job not found'}), 404

        output_dir = Path(status['output_dir'])
        results_dir = output_dir / '_RESULTS'
        
        preview_data = {
            'dvh': [],
            'radiomics': [],
            'qc': []
        }

        # Load DVH metrics
        dvh_path = results_dir / 'dvh_metrics.xlsx'
        if dvh_path.exists():
            try:
                df = pd.read_excel(dvh_path)
                # Replace NaN with None for JSON serialization
                preview_data['dvh'] = df.where(pd.notnull(df), None).to_dict(orient='records')
            except Exception as e:
                logger.error(f"Failed to load DVH metrics: {e}")

        # Load Radiomics
        rad_path = results_dir / 'radiomics_ct.xlsx'
        if rad_path.exists():
            try:
                df = pd.read_excel(rad_path)
                # Limit rows to avoid huge payload
                df_preview = df.head(100)
                preview_data['radiomics'] = df_preview.where(pd.notnull(df_preview), None).to_dict(orient='records')
            except Exception as e:
                logger.error(f"Failed to load Radiomics: {e}")
        
        # Load QC
        qc_path = results_dir / 'qc_reports.xlsx'
        if qc_path.exists():
             try:
                df = pd.read_excel(qc_path)
                preview_data['qc'] = df.where(pd.notnull(df), None).to_dict(orient='records')
             except Exception as e:
                logger.error(f"Failed to load QC: {e}")

        return jsonify(preview_data)

    except Exception as e:
        logger.error(f"Preview error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs/<job_id>/dvh-curves')
def get_dvh_curves(job_id):
    """Get raw DVH curve data for plotting"""
    try:
        status = job_manager.get_job_status(job_id)
        if not status:
            return jsonify({'error': 'Job not found'}), 404

        output_dir = Path(status['output_dir'])
        results_dir = output_dir / '_RESULTS'
        
        # The JSON file might be in a course subdirectory if multiple courses were processed
        # For simplicity in the WebUI context (often 1 patient upload), we search for it.
        # Note: The pipeline outputs to OUTPUT_DIR/{PatientID}/{CourseID}/dvh_curves.json
        # The WebUI job structure is: /data/output/{job_id}/{PatientID}/{CourseID}/...
        
        curve_files = list(output_dir.rglob('dvh_curves.json'))
        
        if not curve_files:
            return jsonify({'error': 'No DVH curves found'}), 404
            
        # If multiple, merge or just take the first one for now.
        # ideally we'd structure the UI to select patient/course
        combined_data = []
        for f in curve_files:
            try:
                with open(f) as json_f:
                    data = json.load(json_f)
                    # Add course info if possible
                    course_name = f.parent.name
                    for point_set in data.get('points', []):
                        point_set['course'] = course_name
                        combined_data.append(point_set)
            except Exception as exc:
                logger.warning("Failed to parse DVH curve file %s: %s", f, exc, exc_info=True)

        return jsonify({'points': combined_data})

    except Exception as e:
        logger.error(f"DVH curve error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs/<job_id>/files/<path:filepath>')
def serve_job_file(job_id, filepath):
    """Serve any file from the job directory (input or output) for visualization"""
    try:
        status = job_manager.get_job_status(job_id)
        if not status:
            return jsonify({'error': 'Job not found'}), 404

        # Allow access to Input (DICOMs) and Output (Results)
        job_input_dir = Path(status['dicom_dir']).resolve()
        job_output_dir = Path(status['output_dir']).resolve()
        
        # We need to determine where the file is.
        # The frontend might request "input/image.dcm" or "output/results.json"
        # Or we map paths.
        
        # Strategy: The viewer needs DICOM series. 
        # Let's assume the request is relative to the job root, but we don't have a single root.
        # We'll use a prefix.
        
        requested_path = Path(filepath)
        parts = requested_path.parts
        if not parts:
            return jsonify({'error': 'Invalid path'}), 400

        if parts[0] == 'input':
            target_file = job_input_dir.joinpath(*parts[1:]).resolve()
            try:
                target_file.relative_to(job_input_dir)
            except ValueError:
                return jsonify({'error': 'Access denied'}), 403
        elif parts[0] == 'output':
            target_file = job_output_dir.joinpath(*parts[1:]).resolve()
            try:
                target_file.relative_to(job_output_dir)
            except ValueError:
                return jsonify({'error': 'Access denied'}), 403
        else:
             return jsonify({'error': 'Invalid path prefix. Use input/ or output/'}), 400

        if not target_file.exists():
            return jsonify({'error': 'File not found'}), 404

        return send_file(target_file)

    except Exception as e:
        logger.error(f"File serve error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs/<job_id>/dicom-series')
def list_dicom_series(job_id):
    """List DICOM series for the viewer"""
    try:
        status = job_manager.get_job_status(job_id)
        if not status:
             return jsonify({'error': 'Job not found'}), 404
             
        input_dir = Path(status['dicom_dir'])
        
        # Find series
        series_map = {}
        
        for root, dirs, files in os.walk(input_dir):
            dicom_files = [f for f in files if f.endswith('.dcm')]
            if dicom_files:
                # Try to read one to get SeriesUID
                try:
                    ds = pydicom.dcmread(Path(root) / dicom_files[0], stop_before_pixels=True)
                    uid = getattr(ds, 'SeriesInstanceUID', 'unknown')
                    desc = getattr(ds, 'SeriesDescription', 'No Description')
                    modality = getattr(ds, 'Modality', 'Unknown')
                    
                    if uid not in series_map:
                        # Construct relative paths for serving
                        rel_root = Path(root).relative_to(input_dir)
                        # Fix paths to match the serve_job_file endpoint
                        file_urls = [f"api/jobs/{job_id}/files/input/{str(rel_root / f)}" for f in dicom_files]
                        
                        series_map[uid] = {
                            'uid': uid,
                            'description': desc,
                            'modality': modality,
                            'num_instances': len(dicom_files),
                            'files': sorted(file_urls)
                        }
                except Exception as exc:
                    logger.warning(
                        "Failed to parse DICOM header in %s: %s",
                        root,
                        exc,
                        exc_info=True,
                    )
        
        return jsonify({'series': list(series_map.values())})
        
    except Exception as e:
        logger.error(f"Series list error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs/<job_id>/segmentations')
def list_segmentations(job_id):
    """List available RTSTRUCT files"""
    try:
        status = job_manager.get_job_status(job_id)
        if not status:
             return jsonify({'error': 'Job not found'}), 404
             
        output_dir = Path(status['output_dir'])
        
        segmentations = []
        
        # Common RTSTRUCT names in root output
        for f in output_dir.glob('RS*.dcm'):
            segmentations.append({
                'name': f.name,
                'path': f"api/jobs/{job_id}/files/output/{f.name}",
                'size': f.stat().st_size
            })
            
        # Custom models subdirectories
        for f in output_dir.glob('**/rtstruct.dcm'):
             # Get parent folder name as label (e.g. Segmentation_Prostate)
             label = f.parent.name
             rel_path = f.relative_to(output_dir)
             segmentations.append({
                'name': label,
                'path': f"api/jobs/{job_id}/files/output/{str(rel_path)}",
                'size': f.stat().st_size
            })

        return jsonify({'segmentations': segmentations})

    except Exception as e:
        logger.error(f"Seg list error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Run Flask app
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'

    logger.info(f"Starting rtpipeline Web UI on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)
