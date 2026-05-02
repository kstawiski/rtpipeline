#!/bin/bash
#$ -S /bin/bash
#$ -N sc51_white
#$ -pe smp 2
#$ -l h_vmem=32G
#$ -l h=!argos10
#$ -l h_rt=10800
#$ -cwd
#$ -o /home/kgs24/rtpipeline_manuscript/analysis/logs/$JOB_NAME.$JOB_ID.out
#$ -e /home/kgs24/rtpipeline_manuscript/analysis/logs/$JOB_NAME.$JOB_ID.err

PYBIN=/home/kgs24/miniforge3/envs/radiomics_sge/bin/python

SCRIPT=""
for candidate in \
  /umed-projekty/rtpipeline/manuscript/scripts/51_outcome_anchored_robust_whitelist.py \
  /home/kgs24/rtpipeline_manuscript/analysis/51_outcome_anchored_robust_whitelist.py
do
  if [[ -f "$candidate" ]]; then
    SCRIPT="$candidate"
    break
  fi
done

if [[ -z "$SCRIPT" ]]; then
  echo "No staged script found. Expected one of:" >&2
  echo "  /umed-projekty/rtpipeline/manuscript/scripts/51_outcome_anchored_robust_whitelist.py" >&2
  echo "  /home/kgs24/rtpipeline_manuscript/analysis/51_outcome_anchored_robust_whitelist.py" >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ARROW_IO_THREADS=2

echo "[$(date '+%F %H:%M:%S')] step51 on $(hostname) jobid=${JOB_ID:-manual}"
echo "[$(date '+%H:%M:%S')] script=${SCRIPT}"
"$PYBIN" "$SCRIPT" \
  --job-id "${JOB_ID:-manual}"
EC=$?
echo "[$(date '+%F %H:%M:%S')] step51 done: exit=$EC"
exit $EC
