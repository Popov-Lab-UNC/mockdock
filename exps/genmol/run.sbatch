#!/bin/bash
#SBATCH --job-name=genmol_mockdock
#SBATCH --partition=l40-gpu,volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=genmol_%j.log
#SBATCH --error=genmol_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shuhang@unc.edu

set -euo pipefail
module load apptainer
module load autodock-gpu

WORKDIR=/work/users/s/h/shuhang/benchmark
GENMOL_DIR=/work/users/s/h/shuhang/genmol
IMAGE="$GENMOL_DIR/genmol.sif"
SCRIPT_DIR="$WORKDIR/exps/genmol"
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

# ── Host Python environment (mockdock + newer RDKit/Meeko) ────────────────────
# GenMol/rdkit-pypi stays inside the container; only mockdock runs on the host.
# Create the venv once with:
#   python3 -m venv /work/users/s/h/shuhang/benchmark/.venv
#   source /work/users/s/h/shuhang/benchmark/.venv/bin/activate
#   pip install -e /work/users/s/h/shuhang/benchmark
source /work/users/s/h/shuhang/benchmark/.venv/bin/activate

echo "========================================================"
echo " GenMol x mockdock Evaluation"
echo "========================================================"
echo " Budget          : ${BUDGET}"
echo " Seed            : ${SEED}"
echo " Num runs        : ${NUM_RUNS}"
echo " Clip reward     : ${CLIP_REWARD_UPPER_BOUND}"
echo " Output dir      : ${OUT_DIR}"
echo " SLURM job ID    : ${SLURM_JOB_ID}"
echo " Host Python     : $(which python) ($(python --version))"
echo "========================================================"
echo ""

# ── Preflight: verify the host env has the right APIs ────────────────────────
python - <<'PREFLIGHT'
from rdkit import Chem
from rdkit.Chem import rdDistGeom
import meeko, molscrub
m = Chem.MolFromSmiles("CC")
p = rdDistGeom.ETKDGv3()
assert hasattr(m, "HasQuery"), "FAIL: Mol.HasQuery missing — wrong rdkit version"
assert hasattr(p, "GetFailureCounts"), "FAIL: ETKDGv3.GetFailureCounts missing — wrong rdkit version"
print("Preflight OK: rdkit API compatible with meeko/molscrub")
PREFLIGHT

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

    # run.py now runs on the host; it calls the container internally for generation only
    python "$SCRIPT_DIR/run.py" \
        --budget "${BUDGET}" \
        --seed "${RUN_SEED}" \
        --out "${OUT_DIR}" \
        --run-dir "${RUN_PARENT}" \
        ${CLIP_REWARD_FLAG} \
        --batch-size 64 \
        --container-image "${IMAGE}" \
        --genmol-dir "${GENMOL_DIR}" \
        --workdir "${WORKDIR}"

    echo ""
done

echo "========================================================"
echo " All benchmarks complete."
echo " Results in: ${OUT_DIR}"
echo "========================================================"
