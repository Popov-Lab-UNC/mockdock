#!/bin/bash
#SBATCH --job-name=prexsyn_fcgmb
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=prexsyn_%j.log
#SBATCH --error=prexsyn_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shuhang@unc.edu

# ── Environment ───────────────────────────────────────────────────────────────
module load autodock-gpu
module load autogrid

CONDA_PATH=$HOME/miniconda3/bin/conda
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
NUM_RUNS=1
NUM_INIT=500
BOTTLENECK=50
TIME_LIMIT=""

# Set to "--fragment-condition" to enable soft BRICS fragment bias in PrexSyn.
# Leave empty ("") for unconditioned generation (oracle still enforces fragment).
FRAGMENT_CONDITION="--fragment-condition"

# ── Benchmarks ────────────────────────────────────────────────────────────────
BENCHMARKS=(AKT1 CHK1 ITK PCK1 TTK VEGFR2)

echo "========================================================"
echo " PrexSyn × FCGMB Evaluation"
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

for BENCHMARK in "${BENCHMARKS[@]}"; do
    echo "──────────────────────────────────────────────────────"
    echo " Starting benchmark: ${BENCHMARK}"
    echo " $(date)"
    echo "──────────────────────────────────────────────────────"

    # Build optional time-limit flag
    TIME_LIMIT_FLAG=""
    [ -n "${TIME_LIMIT}" ] && TIME_LIMIT_FLAG="--time-limit ${TIME_LIMIT}"

    python -u run.py \
        --benchmark         "${BENCHMARK}" \
        --model             "${PREXSYN_MODEL}" \
        --budget            "${BUDGET}" \
        --num-runs          "${NUM_RUNS}" \
        --num-init-samples  "${NUM_INIT}" \
        --bottleneck-size   "${BOTTLENECK}" \
        --out               "${OUT_DIR}" \
        ${TIME_LIMIT_FLAG} \
        ${FRAGMENT_CONDITION}

    echo ""
    echo " Finished: ${BENCHMARK}  ($(date))"
    echo ""
done

echo "========================================================"
echo " All benchmarks complete."
echo " Results in: ${OUT_DIR}"
echo "========================================================"
