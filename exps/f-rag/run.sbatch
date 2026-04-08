#!/bin/bash
#SBATCH --job-name=frag_mockdock
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=frag_%j.log
#SBATCH --error=frag_%j.err

module load autodock-gpu
module load autogrid

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate py312
source /work/users/s/h/shuhang/f-RAG/.venv/bin/activate

SCRIPT_DIR=/work/users/s/h/shuhang/benchmark/exps/f-rag
FRAG_ROOT=/work/users/s/h/shuhang/f-RAG
OUT_DIR=${SCRIPT_DIR}/outputs
BUDGET=1000
SEED=0
NUM_RUNS=1

cd "${SCRIPT_DIR}"

echo "========================================================"
echo " f-RAG x mockdock Evaluation"
echo "========================================================"
echo " Budget      : ${BUDGET}"
echo " Seed        : ${SEED}"
echo " Num runs    : ${NUM_RUNS}"
echo " Output dir  : ${OUT_DIR}"
echo " SLURM job   : ${SLURM_JOB_ID}"
echo "========================================================"
echo ""

BASE_TS="$(date +%Y%m%d_%H%M%S)"
for RUN_IDX in $(seq 1 "${NUM_RUNS}"); do
    RUN_TAG=$(printf "r%02d" "${RUN_IDX}")
    RUN_SEED=$((SEED + RUN_IDX - 1))
    RUN_PARENT="${SCRIPT_DIR}/run_${BASE_TS}_${RUN_TAG}"
    mkdir -p "${RUN_PARENT}"

    echo "──────────────────────────────────────────────────────"
    echo " Suite run    : ${RUN_IDX}/${NUM_RUNS}"
    echo " Seed         : ${RUN_SEED}"
    echo " Run parent   : ${RUN_PARENT}"
    echo " $(date)"
    echo "──────────────────────────────────────────────────────"

    python -u run.py \
        --budget   "${BUDGET}" \
        --seed     "${RUN_SEED}" \
        --out      "${OUT_DIR}" \
        --run-dir  "${RUN_PARENT}" \
        --num-runs 1

    echo ""
done
