#!/bin/bash
#$ -S /bin/bash
#$ -N sc50_bridge
#$ -pe smp 2
#$ -l h_vmem=32G
#$ -l h=!argos10
#$ -l h_rt=10800
#$ -cwd
#$ -o /umed-projekty/rtpipeline/manuscript/analysis/logs/$JOB_NAME.$JOB_ID.out
#$ -e /umed-projekty/rtpipeline/manuscript/analysis/logs/$JOB_NAME.$JOB_ID.err

# Step 50: 107-feature bridge sensitivity on the section-30 within-patient CoV analysis.

SCRIPT=/umed-projekty/rtpipeline/manuscript/scripts/50_shared_feature_bridge.py
PYBIN=/home/kgs24/miniforge3/envs/radiomics_sge/bin/python

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ARROW_IO_THREADS=2

echo "[$(date '+%F %H:%M:%S')] step50 on $(hostname) jobid=${JOB_ID:-manual}"
"$PYBIN" "$SCRIPT" \
    --bootstrap-iterations 1000 \
    --seed 2604 \
    --job-id "${JOB_ID:-manual}"
EC=$?
echo "[$(date '+%F %H:%M:%S')] step50 done: exit=$EC"
exit $EC
