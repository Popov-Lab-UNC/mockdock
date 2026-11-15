#!/bin/bash
#SBATCH --job-name=libinv_cap_no_rmsd
#SBATCH --partition=volta-gpu,gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=libinv_capped_no_rmsd_%j.log
#SBATCH --error=libinv_capped_no_rmsd_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shuhang@unc.edu

set -euo pipefail

module load autodock-gpu
module load autogrid

source /hickory/users/s/h/shuhang/utils/REINVENT4/.venv/bin/activate
export PATH="/hickory/users/s/h/shuhang/utils/REINVENT4/.venv/bin:$PATH"

SCRIPT_DIR=/hickory/users/s/h/shuhang/devel/mockdock/exps/libinvent
OUT_DIR="${SCRIPT_DIR}/outputs_capped_no_rmsd"
RUNS_BASE="${SCRIPT_DIR}/runs_capped_no_rmsd"
mkdir -p "${OUT_DIR}" "${RUNS_BASE}"

# Path to Lib-INVENT prior model.
LIBINVENT_PRIOR=/hickory/users/s/h/shuhang/utils/REINVENT4/priors/libinvent.prior

BUDGET=1000
SEED=1
N_WARMUP=25
NUM_RUNS=5
DEVICE="cuda:0"
DOCKING_BACKEND="auto"
CLIP_REWARD_UPPER_BOUND=true
REQUIRE_POSE_RMSD=false
BENCHMARKS=(DPP4 CHK1 ITK PEPCK TTK VEGFR2 PptT)

CLIP_REWARD_FLAG="--clip-reward-upper-bound"
RMSD_FLAG="--no-require-pose-rmsd"

echo "========================================================"
echo " REINVENT4 Lib-INVENT Clean Suite: Capped + No-RMSD"
echo "========================================================"
echo " Benchmarks      : ${BENCHMARKS[*]}"
echo " Budget          : ${BUDGET}"
echo " Seed            : ${SEED}"
echo " Warmup          : ${N_WARMUP} initial compounds"
echo " Num runs        : ${NUM_RUNS}"
echo " Device          : ${DEVICE}"
echo " Docking backend : ${DOCKING_BACKEND}"
echo " Clip reward     : ${CLIP_REWARD_UPPER_BOUND}"
echo " Require RMSD    : ${REQUIRE_POSE_RMSD}"
echo " Prior           : ${LIBINVENT_PRIOR}"
echo " Output dir      : ${OUT_DIR}"
echo " Runs base       : ${RUNS_BASE}"
echo " SLURM job ID    : ${SLURM_JOB_ID:-local}"
echo "========================================================"
echo ""

cd "${SCRIPT_DIR}"
BASE_TS="$(date +%Y%m%d_%H%M%S)"

for RUN_IDX in $(seq 1 "${NUM_RUNS}"); do
    RUN_TAG=$(printf "r%02d" "${RUN_IDX}")
    RUN_SEED=$((SEED + RUN_IDX - 1))
    RUN_PARENT="${RUNS_BASE}/run_${BASE_TS}_${RUN_TAG}"
    mkdir -p "${RUN_PARENT}"

    echo "──────────────────────────────────────────────────────"
    echo " Suite run    : ${RUN_IDX}/${NUM_RUNS}"
    echo " Seed         : ${RUN_SEED}"
    echo " Run parent   : ${RUN_PARENT}"
    echo " $(date)"
    echo "──────────────────────────────────────────────────────"

    python -u run.py \
        --budget "${BUDGET}" \
        --seed "${RUN_SEED}" \
        --out "${OUT_DIR}" \
        --run-dir "${RUN_PARENT}" \
        --prior-file "${LIBINVENT_PRIOR}" \
        --n-warmup "${N_WARMUP}" \
        --device "${DEVICE}" \
        --docking-backend "${DOCKING_BACKEND}" \
        ${CLIP_REWARD_FLAG} \
        ${RMSD_FLAG} \
        $(for b in "${BENCHMARKS[@]}"; do printf -- "--benchmark %q " "$b"; done)

    echo ""
done

echo "========================================================"
echo " All benchmarks complete for Capped + No-RMSD suite."
echo " Results in: ${RUNS_BASE}"
echo "========================================================"
