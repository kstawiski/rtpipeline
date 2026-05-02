#!/bin/bash
#$ -S /bin/bash
#$ -N sc26_qcpass
#$ -pe smp 4
#$ -l h_vmem=96G
#$ -l h=!argos10
#$ -l h_rt=86400
#$ -cwd
#$ -o /home/kgs24/rtpipeline_manuscript/analysis/logs/$JOB_NAME.$JOB_ID.out
#$ -e /home/kgs24/rtpipeline_manuscript/analysis/logs/$JOB_NAME.$JOB_ID.err

PYBIN=/home/kgs24/miniforge3/envs/radiomics_sge/bin/python

SCRIPT=""
for candidate in \
    /umed-projekty/rtpipeline/manuscript/scripts/26_qc_pass_sensitivity.py \
    /home/kgs24/rtpipeline_manuscript/analysis/26_qc_pass_sensitivity.py
do
    if [ -f "$candidate" ]; then
        SCRIPT="$candidate"
        break
    fi
done

if [ -z "$SCRIPT" ]; then
    echo "Missing 26_qc_pass_sensitivity.py in all expected locations." >&2
    exit 2
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ARROW_IO_THREADS=2

echo "[$(date '+%F %H:%M:%S')] step26 QC-pass sensitivity on $(hostname) jobid=${JOB_ID:-manual}"
"$PYBIN" "$SCRIPT" \
    --cohorts GBM,Hipokampy,Immunodozymetria,LCTSC,NSCLC_Interobserver,NSCLC_Radiomics,Odbytnice,PlucaRCHT,Prostata,RIDER \
    --batch-size 32 \
    --job-id "${JOB_ID:-manual}"
EC=$?
echo "[$(date '+%F %H:%M:%S')] step26 QC-pass sensitivity done: exit=$EC"
exit $EC
