#!/bin/bash
#SBATCH --job-name=libinvent_mockdock
#SBATCH --partition=l40-gpu,volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=libinvent_%j.log
#SBATCH --error=libinvent_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shuhang@unc.edu

module load autodock-gpu
module load autogrid

source /work/users/s/h/shuhang/REINVENT4/.venv/bin/activate

SCRIPT_DIR=/work/users/s/h/shuhang/benchmark/exps/libinvent
OUT_DIR=${SCRIPT_DIR}/outputs

# Path to Lib-INVENT prior model.
LIBINVENT_PRIOR=/work/users/s/h/shuhang/REINVENT4/priors/libinvent.prior

BUDGET=1000
SEED=1
N_WARMUP=25
NUM_RUNS=5
DEVICE="cuda:0"
DOCKING_BACKEND="auto"
CLIP_REWARD_UPPER_BOUND=false
BENCHMARKS=(DPP4 CHK1 ITK PEPCK TTK VEGFR2)

if [ "${CLIP_REWARD_UPPER_BOUND}" = true ]; then
    CLIP_REWARD_FLAG="--clip-reward-upper-bound"
else
    CLIP_REWARD_FLAG="--no-clip-reward-upper-bound"
fi

echo "========================================================"
echo " REINVENT4 Lib-INVENT x mockdock Evaluation"
echo "========================================================"
echo " Benchmarks      : ${BENCHMARKS[*]}"
echo " Budget          : ${BUDGET}"
echo " Seed            : ${SEED}"
echo " Warmup          : ${N_WARMUP} initial compounds"
echo " Num runs        : ${NUM_RUNS}"
echo " Device          : ${DEVICE}"
echo " Docking backend : ${DOCKING_BACKEND}"
echo " Clip reward     : ${CLIP_REWARD_UPPER_BOUND}"
echo " Prior           : ${LIBINVENT_PRIOR}"
echo " Output dir      : ${OUT_DIR}"
echo " SLURM job ID    : ${SLURM_JOB_ID}"
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
        --budget "${BUDGET}" \
        --seed "${RUN_SEED}" \
        --out "${OUT_DIR}" \
        --run-dir "${RUN_PARENT}" \
        --prior-file "${LIBINVENT_PRIOR}" \
        --n-warmup "${N_WARMUP}" \
        --device "${DEVICE}" \
        --docking-backend "${DOCKING_BACKEND}" \
        ${CLIP_REWARD_FLAG} \
        $(for b in "${BENCHMARKS[@]}"; do printf -- "--benchmark %q " "$b"; done)

    echo ""
done

echo "========================================================"
echo " All benchmarks complete."
echo " Results in: ${OUT_DIR}"
echo "========================================================"
