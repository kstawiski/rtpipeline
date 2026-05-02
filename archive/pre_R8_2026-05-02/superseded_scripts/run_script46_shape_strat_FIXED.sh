#!/bin/bash
#$ -S /bin/bash
#$ -N sc46_shape_fx
#$ -pe smp 4
#$ -l h_vmem=64G
#$ -l h=!argos10
#$ -l h_rt=43200
#$ -cwd
#$ -o /home/kgs24/rtpipeline_manuscript/analysis/logs/$JOB_NAME.$JOB_ID.out
#$ -e /home/kgs24/rtpipeline_manuscript/analysis/logs/$JOB_NAME.$JOB_ID.err

# Step 46 fixed: stratified shape ICC with a genuine patient bootstrap.
# The Python script enforces the post-2026-04-18 rebuild wait gate for
# analysis/data/icc_results.parquet before reading point estimates.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${SCRIPT_DIR}/46_stratified_shape_icc_FIXED.py"
PYBIN=/home/kgs24/miniforge3/envs/radiomics_sge/bin/python

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ARROW_IO_THREADS=2

echo "[$(date '+%F %H:%M:%S')] step46 fixed on $(hostname) jobid=${JOB_ID:-manual}"
"$PYBIN" "$SCRIPT" \
    --bootstrap-iterations 1000 \
    --seed 2604 \
    --cohorts GBM,Hipokampy,Immunodozymetria,LCTSC,NSCLC_Interobserver,NSCLC_Radiomics,Odbytnice,PlucaRCHT,Prostata,RIDER \
    --min-icc-mtime 2026-04-18T00:00:00+00:00 \
    --poll-seconds 300 \
    --max-wait-seconds 21600 \
    --job-id "${JOB_ID:-manual}"
EC=$?
echo "[$(date '+%F %H:%M:%S')] step46 fixed done: exit=$EC"
exit $EC
