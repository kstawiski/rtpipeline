#!/bin/bash
#$ -S /bin/bash
#$ -N sc41_prog_fix
#$ -pe smp 4
#$ -l h_vmem=128G
#$ -l h=!argos10
#$ -l h_rt=21600
#$ -cwd
#$ -o /home/kgs24/rtpipeline_manuscript/analysis/logs/$JOB_NAME.$JOB_ID.out
#$ -e /home/kgs24/rtpipeline_manuscript/analysis/logs/$JOB_NAME.$JOB_ID.err

# Step 41 FIXED: prognostic retention, production-safe scope.
# Production dispatch intentionally excludes:
#   - Prostata OS (no observed last-follow-up in audited accessible sources)
#   - Odbytnice legs (no production-safe GTV/RTSTRUCT merged radiomics export)

PYBIN=/home/kgs24/miniforge3/envs/radiomics_sge/bin/python

SCRIPT=""
for candidate in \
    /umed-projekty/rtpipeline/manuscript/scripts/41_prognostic_retention_FIXED.py \
    /home/kgs24/rtpipeline_manuscript/analysis/41_prognostic_retention_FIXED.py \
    /home/kgs24/rtpipeline_manuscript/scripts/41_prognostic_retention_FIXED.py
do
    if [ -f "$candidate" ]; then
        SCRIPT="$candidate"
        break
    fi
done

if [ -z "$SCRIPT" ]; then
    echo "Missing 41_prognostic_retention_FIXED.py in all expected locations." >&2
    exit 2
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ARROW_IO_THREADS=2

echo "[$(date '+%F %H:%M:%S')] step41 fixed on $(hostname) jobid=${JOB_ID:-manual}"
"$PYBIN" "$SCRIPT" \
    --outer-folds 5 \
    --prescreen-k 25 \
    --job-id "${JOB_ID:-manual}"
EC=$?
echo "[$(date '+%F %H:%M:%S')] step41 fixed done: exit=$EC"
exit $EC
