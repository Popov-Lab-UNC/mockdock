#!/bin/bash
#SBATCH --job-name=invgen_mockdock
#SBATCH --partition=l40-gpu,a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=invgen_%j.log
#SBATCH --error=invgen_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shuhang@unc.edu

set -euo pipefail
module load autodock-gpu

WORKDIR=/work/users/s/h/shuhang/benchmark
INVGEN_DIR=/work/users/s/h/shuhang/invirtuogen
INVGEN_PYTHON="$INVGEN_DIR/.venv/bin/python"
SCRIPT_DIR="$WORKDIR/exps/invirtuogen"
CKPT="$INVGEN_DIR/checkpoints/invirtuo_gen_big.ckpt"
OUT_DIR="$SCRIPT_DIR/outputs"

BUDGET=1000
SEED=0
NUM_RUNS=5
CLIP_REWARD_UPPER_BOUND=false

if [ "${CLIP_REWARD_UPPER_BOUND}" = true ]; then
    CLIP_REWARD_FLAG="--clip-reward-upper-bound"
else
    CLIP_REWARD_FLAG="--no-clip-reward-upper-bound"
fi

source /work/users/s/h/shuhang/benchmark/.venv/bin/activate

echo "========================================================"
# Print execution info
echo " InVirtuoGen x mockdock Evaluation (Uncapped)"
echo "========================================================"
echo " Budget          : ${BUDGET}"
echo " Seed            : ${SEED}"
echo " Num runs        : ${NUM_RUNS}"
echo " Clip reward     : ${CLIP_REWARD_UPPER_BOUND}"
echo " Output dir      : ${OUT_DIR}"
echo " SLURM job ID    : ${SLURM_JOB_ID}"
echo " Host Python     : $(which python) ($(python --version))"
echo " InVirtuoGen Py  : ${INVGEN_PYTHON}"
echo "========================================================"
echo ""

cd "$WORKDIR"

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

    python "$SCRIPT_DIR/run.py" \
        --budget "${BUDGET}" \
        --seed "${RUN_SEED}" \
        --out "${OUT_DIR}" \
        --run-dir "${RUN_PARENT}" \
        ${CLIP_REWARD_FLAG} \
        --batch-size 64 \
        --ckpt "${CKPT}" \
        --invgen-python "${INVGEN_PYTHON}" \
        --invgen-dir "${INVGEN_DIR}"

    echo ""
done

echo "========================================================"
echo " All benchmarks complete."
echo " Results in: ${OUT_DIR}"
echo "========================================================"
