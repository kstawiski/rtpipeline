#!/bin/bash
#$ -S /bin/bash
#$ -N step33_sigma_calib
#$ -pe smp 2
#$ -l h_vmem=32G
#$ -l h=!argos10
#$ -l h_rt=10800
#$ -cwd
#$ -o /home/kgs24/rtpipeline_manuscript/analysis/logs/$JOB_NAME.$JOB_ID.out
#$ -e /home/kgs24/rtpipeline_manuscript/analysis/logs/$JOB_NAME.$JOB_ID.err

# -----------------------------------------------------------------------------
# Step 33: Human-vs-NTCV-V σ calibration on NSCLC_Interobserver.
#
# Submit from argos-worker with:
#   qsub /home/kgs24/rtpipeline_manuscript/analysis/run_script33_sigma_calibration.sh
#
# Resources: pe smp 2, h_vmem=32G, h=!argos10 (avoid argos10 per local policy),
# h_rt=10800 (3 h wall — generous for 21 patients × 1223 features × 10 OARs).
# -----------------------------------------------------------------------------

SCRIPT=/home/kgs24/rtpipeline_manuscript/analysis/33_human_vs_ntcv_magnitude_calibration.py
PYBIN=/home/kgs24/miniforge3/envs/radiomics_sge/bin/python
OUT_DIR=/home/kgs24/rtpipeline_manuscript/analysis

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ARROW_IO_THREADS=2

echo "[$(date '+%F %H:%M:%S')] step33 on $(hostname) (pe smp 2, h_vmem=32G)"
"$PYBIN" "$SCRIPT" \
    --output-dir "$OUT_DIR" \
    --bootstrap-iterations 1000 \
    --seed 12345 \
    --workers 2
EC=$?
echo "[$(date '+%F %H:%M:%S')] step33 done: exit=$EC"
exit $EC
