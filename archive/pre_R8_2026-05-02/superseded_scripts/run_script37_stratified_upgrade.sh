#!/bin/bash
#$ -S /bin/bash
#$ -N sc37_cov_up
#$ -pe smp 4
#$ -l h_vmem=128G
#$ -l h=!argos10
#$ -l h_rt=21600
#$ -cwd
#$ -o /home/kgs24/rtpipeline_manuscript/analysis/logs/$JOB_NAME.$JOB_ID.out
#$ -e /home/kgs24/rtpipeline_manuscript/analysis/logs/$JOB_NAME.$JOB_ID.err

# Step 37: feature-class upgrade stratified by baseline human CoV.

SCRIPT=/home/kgs24/rtpipeline_manuscript/analysis/37_cov_upgrade_stratified.py
PYBIN=/home/kgs24/miniforge3/envs/radiomics_sge/bin/python

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ARROW_IO_THREADS=2

echo "[$(date '+%F %H:%M:%S')] step37 on $(hostname) jobid=${JOB_ID:-manual}"
"$PYBIN" "$SCRIPT" \
    --bootstrap-iterations 1000 \
    --cohorts GBM,Hipokampy,Immunodozymetria,LCTSC,NSCLC_Interobserver,NSCLC_Radiomics,Odbytnice,PlucaRCHT,Prostata,RIDER \
    --job-id "${JOB_ID:-manual}"
EC=$?
echo "[$(date '+%F %H:%M:%S')] step37 done: exit=$EC"
exit $EC
