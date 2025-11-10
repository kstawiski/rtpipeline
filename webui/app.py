"""
rtpipeline Web UI - Flask Application
Provides a web interface for uploading DICOM files and processing them through rtpipeline
"""

import os
import json
import uuid
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
import threading
import logging

from dicom_validator import DICOMValidator
from job_manager import JobManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', str(uuid.uuid4()))
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
        job_info = job_manager.create_job(
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


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    }), 200


if __name__ == '__main__':
    # Run Flask app
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'

    logger.info(f"Starting rtpipeline Web UI on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)
