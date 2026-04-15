#!/bin/bash
#SBATCH --job-name=prexsyn_mockdock
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=prexsyn_%j.log
#SBATCH --error=prexsyn_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shuhang@unc.edu

# ── Environment ───────────────────────────────────────────────────────────────
module load autodock-gpu
module load autogrid

# Canonical Longleaf path — $HOME is not always the same on compute vs login.
MINICONDA_ROOT=/nas/longleaf/home/shuhang/miniconda3
CONDA_PATH="${MINICONDA_ROOT}/bin/conda"
eval "$($CONDA_PATH shell.bash hook)"
conda activate prexsyn

# ── Paths ─────────────────────────────────────────────────────────────────────
export PYTHONPATH=$PYTHONPATH:/work/users/s/h/shuhang/prexsyn

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR=/work/users/s/h/shuhang/benchmark/exps/prexsyn
PREXSYN_MODEL=/work/users/s/h/shuhang/prexsyn/data/trained_models/v1_converted.yaml
OUT_DIR=${SCRIPT_DIR}/outputs

# ── Parameters ────────────────────────────────────────────────────────────────
# budget        : max oracle (docking) calls per benchmark
# num_runs      : independent optimization runs per benchmark (increase for stats)
# num_init      : molecules sampled during initialisation
# bottleneck    : elite population size fed to FingerprintGenetic each step
# time_limit    : per-run wall-clock cap in seconds (empty = no limit)
BUDGET=1000
NUM_RUNS=5
NUM_INIT=25
BOTTLENECK=50
TIME_LIMIT=""

# Set to "--fragment-condition" to enable soft BRICS fragment bias in PrexSyn.
# Leave empty ("") for unconditioned generation (oracle still enforces fragment).
FRAGMENT_CONDITION="--fragment-condition"

# ── Benchmarks ────────────────────────────────────────────────────────────────
BENCHMARKS=(DPP4 CHK1 ITK PEPCK TTK VEGFR2)

echo "========================================================"
echo " PrexSyn × mockdock Evaluation"
echo "========================================================"
echo " Model          : ${PREXSYN_MODEL}"
echo " Benchmarks     : ${BENCHMARKS[*]}"
echo " Budget         : ${BUDGET}"
echo " Num runs       : ${NUM_RUNS}"
echo " Init samples   : ${NUM_INIT}"
echo " Bottleneck     : ${BOTTLENECK}"
echo " Time limit     : ${TIME_LIMIT:-none}"
echo " Fragment cond. : ${FRAGMENT_CONDITION:-off}"
echo " Output dir     : ${OUT_DIR}"
echo " SLURM job ID   : ${SLURM_JOB_ID}"
echo "========================================================"
echo ""

cd "${SCRIPT_DIR}"

# When NUM_RUNS>1, we want *independent* full-suite runs to land in
# different top-level run directories:
#   run_<timestamp>_r01/<BENCHMARK>/...
#   run_<timestamp>_r02/<BENCHMARK>/...
#   ...
#
# Each suite run calls run.py with --num-runs 1 (one optimization run per
# benchmark) so the directory split is clean and consistent.
BASE_TS="$(date +%Y%m%d_%H%M%S)"

# Build optional time-limit flag
TIME_LIMIT_FLAG=""
[ -n "${TIME_LIMIT}" ] && TIME_LIMIT_FLAG="--time-limit ${TIME_LIMIT}"

for RUN_IDX in $(seq 1 "${NUM_RUNS}"); do
    RUN_TAG=$(printf "r%02d" "${RUN_IDX}")
    RUN_PARENT="${SCRIPT_DIR}/run_${BASE_TS}_${RUN_TAG}"
    mkdir -p "${RUN_PARENT}"

    echo "──────────────────────────────────────────────────────"
    echo " Suite run      : ${RUN_IDX}/${NUM_RUNS}"
    echo " Run parent     : ${RUN_PARENT}"
    echo " $(date)"
    echo "──────────────────────────────────────────────────────"

    "${CONDA_PREFIX}/bin/python" -u run.py \
        --model             "${PREXSYN_MODEL}" \
        --budget            "${BUDGET}" \
        --num-runs          1 \
        --num-init-samples  "${NUM_INIT}" \
        --bottleneck-size   "${BOTTLENECK}" \
        --out               "${OUT_DIR}" \
        --run-dir           "${RUN_PARENT}" \
        ${TIME_LIMIT_FLAG} \
        ${FRAGMENT_CONDITION} \
        $(for b in "${BENCHMARKS[@]}"; do printf -- "--benchmark %q " "$b"; done)

    echo ""
done

echo "========================================================"
echo " All benchmarks complete."
echo " Results in: ${OUT_DIR}"
echo "========================================================"
