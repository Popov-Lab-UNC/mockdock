#!/bin/bash
#SBATCH --job-name=ahc_fcgmb
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=ahc_%j.log
#SBATCH --error=ahc_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shuhang@unc.edu

# ── Environment ───────────────────────────────────────────────────────────────
module load autodock-gpu
module load autogrid

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate py312
source /work/users/s/h/shuhang/acegen-open/.venv/bin/activate

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR=/work/users/s/h/shuhang/benchmark/exps/acegen-ahc
ACEGEN_ROOT=/work/users/s/h/shuhang/acegen-open
OUT_DIR=${SCRIPT_DIR}/outputs

# ── Parameters ────────────────────────────────────────────────────────────────
BUDGET=1000
SEED=0
N_WARMUP=25
NUM_RUNS=5

echo "========================================================"
echo " AceGen-AHC x FCGMB Evaluation"
echo "========================================================"
echo " Algorithm   : AHC (Augmented Hill-Climbing)"
echo " acegen-open : ${ACEGEN_ROOT}"
echo " Budget      : ${BUDGET}"
echo " Seed        : ${SEED}"
echo " Warmup      : ${N_WARMUP} initial compounds"
echo " Num runs    : ${NUM_RUNS}"
echo " Output dir  : ${OUT_DIR}"
echo " SLURM job   : ${SLURM_JOB_ID}"
echo "========================================================"
echo ""

cd "${SCRIPT_DIR}"

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
        --budget      "${BUDGET}"     \
        --seed        "${RUN_SEED}"   \
        --out         "${OUT_DIR}"    \
        --acegen-root "${ACEGEN_ROOT}" \
        --n-warmup    "${N_WARMUP}"   \
        --run-dir     "${RUN_PARENT}"

    echo ""
done

echo ""
echo "========================================================"
echo " All benchmarks complete."
echo " Results in: ${OUT_DIR}"
echo "========================================================"
