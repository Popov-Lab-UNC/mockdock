#!/bin/bash
#SBATCH --job-name=md_aizynth_libinv
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --array=0-1
#SBATCH --output=/hickory/users/s/h/shuhang/devel/mockdock/scripts/scoring/logs/aizynth_libinv_%A_%a.log
#SBATCH --error=/hickory/users/s/h/shuhang/devel/mockdock/scripts/scoring/logs/aizynth_libinv_%A_%a.err

set -euo pipefail

RUNS=("r04" "r05")
RUN_FILTER="${RUNS[$SLURM_ARRAY_TASK_ID]}"

echo "=================================================="
echo "Resuming AIZynthFinder scoring for libinvent (${RUN_FILTER})"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=================================================="

export AIZYNTH_PYTHON="/hickory/users/s/h/shuhang/utils/aizynthfinder/.venv/bin/python"
export AIZYNTH_MAX_WORKERS=8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export CUDA_VISIBLE_DEVICES=""

EXPS_DIR="/hickory/users/s/h/shuhang/devel/mockdock_data/exps_capped_no_rmsd"
SCRIPT="/hickory/users/s/h/shuhang/devel/mockdock/scripts/scoring/score_molecules.py"

PYTHONUNBUFFERED=1 ${AIZYNTH_PYTHON} -u "${SCRIPT}" \
    --scorer aizynthfinder \
    --exps-dir "${EXPS_DIR}" \
    --filter-model libinvent \
    --filter-run "${RUN_FILTER}" \
    --skip-reference-set

echo "=================================================="
echo "Finished AIZynthFinder scoring for libinvent (${RUN_FILTER})"
echo "Date: $(date)"
echo "=================================================="
