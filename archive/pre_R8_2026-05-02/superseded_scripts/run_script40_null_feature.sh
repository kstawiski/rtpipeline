#!/bin/bash
#$ -S /bin/bash
#$ -N sc40_null_ctl
#$ -pe smp 4
#$ -l h_vmem=128G
#$ -l h=!argos10
#$ -l h_rt=21600
#$ -cwd
#$ -o /home/kgs24/rtpipeline_manuscript/analysis/logs/$JOB_NAME.$JOB_ID.out
#$ -e /home/kgs24/rtpipeline_manuscript/analysis/logs/$JOB_NAME.$JOB_ID.err

# Step 40: null-feature control for contour-insensitive first-order features.

SCRIPT=""
for candidate in \
  /umed-projekty/rtpipeline/manuscript/scripts/40_null_feature_control.py \
  /home/kgs24/rtpipeline_manuscript/analysis/40_null_feature_control.py
do
  if [[ -f "$candidate" ]]; then
    SCRIPT="$candidate"
    break
  fi
done

if [[ -z "$SCRIPT" ]]; then
  echo "No staged script found. Expected one of:" >&2
  echo "  /umed-projekty/rtpipeline/manuscript/scripts/40_null_feature_control.py" >&2
  echo "  /home/kgs24/rtpipeline_manuscript/analysis/40_null_feature_control.py" >&2
  exit 2
fi

PYBIN=/home/kgs24/miniforge3/envs/radiomics_sge/bin/python

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ARROW_IO_THREADS=2

echo "[$(date '+%F %H:%M:%S')] step40 on $(hostname) jobid=${JOB_ID:-manual}"
echo "[$(date '+%H:%M:%S')] script=${SCRIPT}"
"$PYBIN" "$SCRIPT" \
    --bootstrap-iterations 1000 \
    --cohorts NSCLC_Interobserver \
    --job-id "${JOB_ID:-manual}"
EC=$?
echo "[$(date '+%F %H:%M:%S')] step40 done: exit=$EC"
exit $EC
