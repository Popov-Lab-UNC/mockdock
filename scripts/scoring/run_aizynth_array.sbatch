#!/bin/bash
#SBATCH --job-name=md_aizynth
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-8
#SBATCH --output=/hickory/users/s/h/shuhang/devel/mockdock/scripts/scoring/logs/aizynth_%A_%a.log
#SBATCH --error=/hickory/users/s/h/shuhang/devel/mockdock/scripts/scoring/logs/aizynth_%A_%a.err

set -euo pipefail

MODELS=(
    "acegen-a2c"
    "acegen-ahc"
    "acegen-ppo"
    "acegen-ppod"
    "acegen-reinforce"
    "acegen-reinvent"
    "genmol"
    "invirtuogen"
    "libinvent"
)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "=================================================="
echo "Starting AIZynthFinder scoring for model: ${MODEL}"
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

${AIZYNTH_PYTHON} "${SCRIPT}" \
    --scorer aizynthfinder \
    --exps-dir "${EXPS_DIR}" \
    --filter-model "${MODEL}" \
    --skip-reference-set

echo "=================================================="
echo "Finished AIZynthFinder scoring for model: ${MODEL}"
echo "Date: $(date)"
echo "=================================================="
