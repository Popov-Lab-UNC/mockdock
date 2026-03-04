#!/bin/bash
#SBATCH --job-name=reinforce_fcgmb
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=reinforce_%j.log
#SBATCH --error=reinforce_%j.err

# ── Environment ───────────────────────────────────────────────────────────────
module load autodock-gpu
module load autogrid

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate py312
source /work/users/s/h/shuhang/acegen-open/.venv/bin/activate

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR=/work/users/s/h/shuhang/benchmark/exps/acegen-reinforce
ACEGEN_ROOT=/work/users/s/h/shuhang/acegen-open
OUT_DIR=${SCRIPT_DIR}/outputs

# ── Parameters ────────────────────────────────────────────────────────────────
BUDGET=1000
SEED=0
N_WARMUP=25

echo "========================================================"
echo " AceGen-REINFORCE x FCGMB Evaluation"
echo "========================================================"
echo " Algorithm   : REINFORCE"
echo " acegen-open : ${ACEGEN_ROOT}"
echo " Budget      : ${BUDGET}"
echo " Seed        : ${SEED}"
echo " Warmup      : ${N_WARMUP} initial compounds"
echo " Output dir  : ${OUT_DIR}"
echo " SLURM job   : ${SLURM_JOB_ID}"
echo "========================================================"
echo ""

cd "${SCRIPT_DIR}"
python -u run.py \
    --budget    "${BUDGET}"     \
    --seed      "${SEED}"       \
    --out       "${OUT_DIR}"    \
    --acegen-root "${ACEGEN_ROOT}" \
    --n-warmup  "${N_WARMUP}"

echo ""
echo "========================================================"
echo " All benchmarks complete."
echo " Results in: ${OUT_DIR}"
echo "========================================================"
