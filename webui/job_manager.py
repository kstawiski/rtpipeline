"""
Job Manager Module
Manages job lifecycle, processing queue, and results
"""

import os
import json
import shutil
import subprocess
import threading
import time
import zipfile
import signal
import re
from datetime import datetime
from pathlib import Path
from pathlib import PurePosixPath
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class JobManager:
    """Manages rtpipeline processing jobs"""

    def __init__(self, input_dir: Path, output_dir: Path, logs_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.logs_dir = logs_dir
        self.jobs_file = logs_dir / 'jobs.json'
        self.jobs = self._load_jobs()
        self.active_processes = {}
        self.lock = threading.Lock()

    def _load_jobs(self) -> Dict[str, Dict[str, Any]]:
        """Load jobs from persistent storage"""
        try:
            if self.jobs_file.exists():
                with open(self.jobs_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load jobs: {str(e)}")

        return {}

    def _save_jobs(self):
        """Save jobs to persistent storage"""
        try:
            with self.lock:
                with open(self.jobs_file, 'w') as f:
                    json.dump(self.jobs, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save jobs: {str(e)}")

    def create_job(
        self,
        job_id: str,
        dicom_dir: str,
        validation_result: Dict[str, Any],
        uploaded_files: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a new job

        Args:
            job_id: Unique job identifier
            dicom_dir: Path to DICOM directory
            validation_result: DICOM validation results
            uploaded_files: List of uploaded files

        Returns:
            Job information dictionary
        """
        job_info = {
            'job_id': job_id,
            'status': 'uploaded',
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat(),
            'dicom_dir': dicom_dir,
            'output_dir': str(self.output_dir / job_id),
            'log_file': str(self.logs_dir / f'{job_id}.log'),
            'validation': validation_result,
            'uploaded_files': uploaded_files,
            'config': {},
            'progress': {
                'current_stage': 'uploaded',
                'percent': 0,
                'message': 'Files uploaded, ready to process'
            }
        }

        with self.lock:
            self.jobs[job_id] = job_info
            self._save_jobs()

        logger.info(f"Created job {job_id}")
        return job_info

    def start_job(self, job_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start processing a job

        Args:
            job_id: Job identifier
            config: Pipeline configuration options

        Returns:
            Result dictionary with success status
        """
        try:
            with self.lock:
                if job_id not in self.jobs:
                    return {'success': False, 'error': 'Job not found'}

                job = self.jobs[job_id]

                if job['status'] == 'running':
                    return {'success': False, 'error': 'Job already running'}

                # Update job status
                job['status'] = 'running'
                job['started_at'] = datetime.utcnow().isoformat()
                job['updated_at'] = datetime.utcnow().isoformat()
                job['config'] = config
                job['progress'] = {
                    'current_stage': 'starting',
                    'percent': 0,
                    'message': 'Initializing pipeline...'
                }
                self._save_jobs()

            # Start processing in background thread
            thread = threading.Thread(
                target=self._process_job,
                args=(job_id,),
                daemon=True
            )
            thread.start()

            logger.info(f"Started job {job_id}")
            return {'success': True}

        except Exception as e:
            logger.error(f"Failed to start job {job_id}: {str(e)}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def _process_job(self, job_id: str):
        """
        Process a job using Snakemake pipeline

        Args:
            job_id: Job identifier
        """
        try:
            job = self.jobs[job_id]
            log_file = Path(job['log_file'])
            output_dir = Path(job['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create job-specific config
            config_file = self.logs_dir / f'{job_id}_config.yaml'
            self._create_job_config(job, config_file)

            # Prepare Snakemake command
            cmd = self._build_snakemake_command(job, config_file)

            logger.info(f"Executing pipeline for job {job_id}: {' '.join(cmd)}")

            # Update progress
            self._update_job_progress(
                job_id,
                stage='organizing',
                percent=10,
                message='Organizing DICOM files...'
            )

            # Execute pipeline
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd='/app',
                    start_new_session=True  # Create new process group for cleanup
                )

                # Store process for potential cancellation
                with self.lock:
                    self.active_processes[job_id] = process

                # Monitor progress
                self._monitor_progress(job_id, log_file)

                # Wait for completion
                return_code = process.wait()

                # Remove from active processes
                with self.lock:
                    self.active_processes.pop(job_id, None)

                # Update job status based on return code
                if return_code == 0:
                    self._complete_job(job_id)
                else:
                    self._fail_job(job_id, f'Pipeline failed with return code {return_code}')

        except Exception as e:
            logger.error(f"Job {job_id} processing error: {str(e)}", exc_info=True)
            self._fail_job(job_id, str(e))

    def _build_snakemake_command(self, job: Dict[str, Any], config_file: Path) -> List[str]:
        """
        Build Snakemake command with appropriate options

        Args:
            job: Job information
            config_file: Path to config file

        Returns:
            Command list
        """
        config = job.get('config', {})

        # Base command
        cmd = [
            'snakemake',
            '--cores', str(config.get('cores', 'all')),
            '--use-conda',
            '--configfile', str(config_file)
        ]

        # Add optional flags
        if config.get('keep_going', False):
            cmd.append('--keep-going')

        if config.get('verbose', False):
            cmd.extend(['--verbose', '--printshellcmds'])

        # Add specific targets if requested
        targets = config.get('targets', [])
        if targets:
            cmd.extend(self._validate_snakemake_targets(targets))

        return cmd

    @staticmethod
    def _validate_snakemake_targets(targets: Any) -> List[str]:
        """Validate user-provided Snakemake targets.

        Targets are appended to the Snakemake argv list. Without validation, a client
        could smuggle additional Snakemake options (e.g. '--snakefile', '--configfile')
        by passing them as "targets".
        """
        if not isinstance(targets, list):
            raise ValueError("Invalid targets: expected a list")

        cleaned: List[str] = []
        for raw in targets:
            if not isinstance(raw, str):
                raise ValueError("Invalid target: must be a string")
            token = raw.strip()
            if not token:
                continue
            if token.startswith("-"):
                raise ValueError(f"Invalid target '{raw}': options are not allowed")
            if any(ch.isspace() for ch in token):
                raise ValueError(f"Invalid target '{raw}': whitespace is not allowed")

            normalized = token.replace("\\", "/")
            path = PurePosixPath(normalized)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Invalid target '{raw}': absolute/parent paths not allowed")

            if re.fullmatch(r"[A-Za-z0-9_./-]+", normalized) is None:
                raise ValueError(f"Invalid target '{raw}': contains forbidden characters")

            cleaned.append(normalized)

        return cleaned

    def _create_job_config(self, job: Dict[str, Any], config_file: Path):
        """
        Create job-specific configuration file

        Args:
            job: Job information
            config_file: Path to write config
        """
        config = {
            'dicom_root': job['dicom_dir'],
            'output_dir': job['output_dir'],
            'logs_dir': str(self.logs_dir),
        }

        # Merge with user config
        user_config = job.get('config', {})

        # Segmentation settings
        if 'segmentation' in user_config:
            config['segmentation'] = user_config['segmentation']
        else:
            config['segmentation'] = {
                'fast': True
            }

        # Custom models (disabled by default for security)
        custom_models_config = user_config.get('custom_models', {})
        # Handle both old (boolean) and new (dict) formats
        if isinstance(custom_models_config, bool):
             is_enabled = custom_models_config
             models_list = []
        else:
             is_enabled = custom_models_config.get('enabled', False)
             models_list = custom_models_config.get('models', [])

        config['custom_models'] = {
            'enabled': is_enabled,
            'models': models_list,
            'workers': 1
        }

        # Radiomics settings
        if 'radiomics' in user_config:
            radiomics_config = user_config['radiomics']
            config['radiomics'] = {
                'sequential': radiomics_config.get('sequential', False)
            }
            # If radiomics is explicitly disabled, set it in config
            if not radiomics_config.get('enabled', True):
                config['radiomics']['enabled'] = False

        # Radiomics robustness settings
        if 'radiomics_robustness' in user_config:
            robustness_config = user_config['radiomics_robustness']
            if robustness_config.get('enabled', False):
                config['radiomics_robustness'] = {
                    'enabled': True,
                    'modes': ['segmentation_perturbation'],
                    'segmentation_perturbation': {
                        'intensity': robustness_config.get('intensity', 'standard'),
                        # Defaults that match setup_new_project.sh logic
                        'apply_to_structures': ["GTV*", "CTV*", "PTV*", "BLADDER", "RECTUM", "PROSTATE"],
                        'small_volume_changes': [-0.15, 0.0, 0.15],
                        'max_translation_mm': 3.0,
                        'n_random_contour_realizations': 2
                    }
                }

        # CT cropping
        if 'ct_cropping' in user_config:
            ct_cropping_config = user_config['ct_cropping']
            if ct_cropping_config.get('enabled', False):
                region = ct_cropping_config.get('region', 'pelvis')
                # Set sensible defaults based on region (matching setup script logic)
                if region == 'brain':
                    sup, inf = 1.0, 1.0
                else:
                    sup, inf = 2.0, 10.0 if region == 'pelvis' else 2.0

                config['ct_cropping'] = {
                    'enabled': True,
                    'region': region,
                    'superior_margin_cm': sup,
                    'inferior_margin_cm': inf,
                    'use_cropped_for_dvh': True,
                    'use_cropped_for_radiomics': True,
                    'keep_original': True
                }

        # Write config file
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"Created config file: {config_file}")

    def _monitor_progress(self, job_id: str, log_file: Path):
        """
        Monitor job progress by parsing log file

        Args:
            job_id: Job identifier
            log_file: Path to log file
        """
        # Map Snakemake stages to progress
        stage_mapping = {
            'organize_courses': ('organizing', 20, 'Organizing DICOM files...'),
            'segmentation_course': ('segmentation', 40, 'Running segmentation...'),
            'crop_ct_course': ('cropping', 50, 'Cropping CT volumes...'),
            'dvh_course': ('dvh', 60, 'Computing DVH metrics...'),
            'radiomics_course': ('radiomics', 80, 'Extracting radiomics features...'),
            'aggregate_results': ('aggregating', 90, 'Aggregating results...')
        }

        last_stage = None
        last_size = 0

        while job_id in self.active_processes:
            try:
                if log_file.exists():
                    # Read new log content
                    current_size = log_file.stat().st_size
                    if current_size > last_size:
                        with open(log_file, 'r') as f:
                            f.seek(last_size)
                            new_content = f.read()
                            last_size = current_size

                            # Check for stage indicators
                            for stage_key, (stage, percent, message) in stage_mapping.items():
                                if stage_key in new_content and stage != last_stage:
                                    self._update_job_progress(job_id, stage, percent, message)
                                    last_stage = stage

                time.sleep(2)  # Check every 2 seconds

            except Exception as e:
                logger.error(f"Progress monitoring error: {str(e)}")
                time.sleep(5)

    def _update_job_progress(
        self,
        job_id: str,
        stage: str,
        percent: int,
        message: str
    ):
        """Update job progress"""
        try:
            with self.lock:
                if job_id in self.jobs:
                    self.jobs[job_id]['progress'] = {
                        'current_stage': stage,
                        'percent': percent,
                        'message': message
                    }
                    self.jobs[job_id]['updated_at'] = datetime.utcnow().isoformat()
                    self._save_jobs()

            logger.info(f"Job {job_id} progress: {percent}% - {message}")

        except Exception as e:
            logger.error(f"Failed to update progress: {str(e)}")

    def _complete_job(self, job_id: str):
        """Mark job as completed"""
        try:
            with self.lock:
                if job_id in self.jobs:
                    self.jobs[job_id]['status'] = 'completed'
                    self.jobs[job_id]['completed_at'] = datetime.utcnow().isoformat()
                    self.jobs[job_id]['updated_at'] = datetime.utcnow().isoformat()
                    self.jobs[job_id]['progress'] = {
                        'current_stage': 'completed',
                        'percent': 100,
                        'message': 'Pipeline completed successfully'
                    }
                    self._save_jobs()

            logger.info(f"Job {job_id} completed successfully")

        except Exception as e:
            logger.error(f"Failed to complete job: {str(e)}")

    def _fail_job(self, job_id: str, error: str):
        """Mark job as failed"""
        try:
            with self.lock:
                if job_id in self.jobs:
                    self.jobs[job_id]['status'] = 'failed'
                    self.jobs[job_id]['error'] = error
                    self.jobs[job_id]['failed_at'] = datetime.utcnow().isoformat()
                    self.jobs[job_id]['updated_at'] = datetime.utcnow().isoformat()
                    self._save_jobs()

            logger.error(f"Job {job_id} failed: {error}")

        except Exception as e:
            logger.error(f"Failed to mark job as failed: {str(e)}")

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status"""
        with self.lock:
            return self.jobs.get(job_id)

    def get_job_logs(self, job_id: str) -> str:
        """Get job logs"""
        try:
            job = self.jobs.get(job_id)
            if not job:
                return ''

            log_file = Path(job['log_file'])
            if log_file.exists():
                with open(log_file, 'r') as f:
                    # Return last 1000 lines
                    lines = f.readlines()
                    return ''.join(lines[-1000:])

            return ''

        except Exception as e:
            logger.error(f"Failed to get logs: {str(e)}")
            return f"Error reading logs: {str(e)}"

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a running job"""
        try:
            with self.lock:
                if job_id not in self.jobs:
                    return {'success': False, 'error': 'Job not found'}

                job = self.jobs[job_id]

                if job['status'] != 'running':
                    return {'success': False, 'error': 'Job is not running'}

                # Kill process
                if job_id in self.active_processes:
                    process = self.active_processes[job_id]
                    try:
                        # Kill process group
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    except Exception as e:
                        logger.error(f"Failed to kill process: {str(e)}")

                    self.active_processes.pop(job_id, None)

                # Update job status
                job['status'] = 'cancelled'
                job['cancelled_at'] = datetime.utcnow().isoformat()
                job['updated_at'] = datetime.utcnow().isoformat()
                self._save_jobs()

            logger.info(f"Cancelled job {job_id}")
            return {'success': True}

        except Exception as e:
            logger.error(f"Failed to cancel job: {str(e)}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def get_job_results(self, job_id: str) -> List[Dict[str, Any]]:
        """Get list of result files"""
        try:
            job = self.jobs.get(job_id)
            if not job:
                return []

            output_dir = Path(job['output_dir'])
            if not output_dir.exists():
                return []

            results = []
            results_dir = output_dir / '_RESULTS'

            if results_dir.exists():
                for file_path in results_dir.rglob('*'):
                    if file_path.is_file():
                        results.append({
                            'name': file_path.name,
                            'path': str(file_path.relative_to(output_dir)),
                            'size': file_path.stat().st_size,
                            'type': file_path.suffix
                        })

            return results

        except Exception as e:
            logger.error(f"Failed to get results: {str(e)}")
            return []

    def create_results_archive(self, job_id: str) -> Optional[str]:
        """Create a ZIP archive of job results"""
        try:
            job = self.jobs.get(job_id)
            if not job:
                return None

            output_dir = Path(job['output_dir'])
            if not output_dir.exists():
                return None

            # Create ZIP file
            zip_path = self.logs_dir / f'{job_id}_results.zip'

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all files from output directory
                for file_path in output_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(output_dir)
                        zipf.write(file_path, arcname)

            logger.info(f"Created results archive: {zip_path}")
            return str(zip_path)

        except Exception as e:
            logger.error(f"Failed to create results archive: {str(e)}")
            return None

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs"""
        with self.lock:
            return [
                {
                    'job_id': job_id,
                    'status': job['status'],
                    'created_at': job['created_at'],
                    'updated_at': job['updated_at'],
                    'progress': job.get('progress', {})
                }
                for job_id, job in self.jobs.items()
            ]

    def delete_job(self, job_id: str) -> Dict[str, Any]:
        """Delete a job and its data"""
        try:
            with self.lock:
                if job_id not in self.jobs:
                    return {'success': False, 'error': 'Job not found'}

                job = self.jobs[job_id]

                # Cannot delete running job
                if job['status'] == 'running':
                    return {'success': False, 'error': 'Cannot delete running job'}

                # Delete files
                try:
                    # Delete input
                    input_dir = Path(job['dicom_dir'])
                    if input_dir.exists():
                        shutil.rmtree(input_dir)

                    # Delete output
                    output_dir = Path(job['output_dir'])
                    if output_dir.exists():
                        shutil.rmtree(output_dir)

                    # Delete log
                    log_file = Path(job['log_file'])
                    if log_file.exists():
                        log_file.unlink()

                except Exception as e:
                    logger.error(f"Failed to delete files for job {job_id}: {str(e)}")

                # Remove from jobs
                del self.jobs[job_id]
                self._save_jobs()

            logger.info(f"Deleted job {job_id}")
            return {'success': True}

        except Exception as e:
            logger.error(f"Failed to delete job: {str(e)}", exc_info=True)
            return {'success': False, 'error': str(e)}
