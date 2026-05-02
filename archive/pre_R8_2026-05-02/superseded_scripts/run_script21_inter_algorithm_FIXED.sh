#!/bin/bash
#$ -S /bin/bash
#$ -N inter_algo_ai_fix
#$ -pe smp 4
#$ -l h_vmem=160G
#$ -l h=!argos10
#$ -cwd
#$ -o /home/kgs24/rtpipeline_manuscript/analysis/logs/inter_algo_ai_fix.$JOB_NAME.$JOB_ID.out
#$ -e /home/kgs24/rtpipeline_manuscript/analysis/logs/inter_algo_ai_fix.$JOB_NAME.$JOB_ID.err

COHORT="${1:-}"
BOOT_ITERS="${2:-1000}"

PYBIN=/home/kgs24/miniforge3/envs/radiomics_sge/bin/python
SCRIPT=""
for candidate in \
  /umed-projekty/rtpipeline/manuscript/scripts/21_inter_algorithm_ai_comparison_FIXED.py \
  /home/kgs24/rtpipeline_manuscript/analysis/21_inter_algorithm_ai_comparison_FIXED.py
do
  if [[ -f "$candidate" ]]; then
    SCRIPT="$candidate"
    break
  fi
done

if [[ -z "$SCRIPT" ]]; then
  echo "No staged script found. Expected one of:" >&2
  echo "  /umed-projekty/rtpipeline/manuscript/scripts/21_inter_algorithm_ai_comparison_FIXED.py" >&2
  echo "  /home/kgs24/rtpipeline_manuscript/analysis/21_inter_algorithm_ai_comparison_FIXED.py" >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 21 fixed inter-algorithm AI comparison"
echo "[$(date '+%H:%M:%S')] host=$(hostname) JOB_ID=${JOB_ID:-local} NSLOTS=${NSLOTS:-?}"
echo "[$(date '+%H:%M:%S')] cohort='${COHORT}' bootstrap_iterations=${BOOT_ITERS}"
echo "[$(date '+%H:%M:%S')] python=${PYBIN}"
echo "[$(date '+%H:%M:%S')] script=${SCRIPT}"

CMD=(
  "$PYBIN" "$SCRIPT"
  --workers 4
  --bootstrap-iterations "$BOOT_ITERS"
  --job-id "${JOB_ID:-local}"
)

if [[ -n "$COHORT" ]]; then
  CMD+=( --cohort "$COHORT" )
fi

echo "[$(date '+%H:%M:%S')] Running: ${CMD[*]}"
"${CMD[@]}"
rc=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Exit code: $rc"
exit $rc
