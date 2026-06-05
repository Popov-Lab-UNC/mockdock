#!/bin/bash
#SBATCH --job-name=mockdock_molskill
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01-00:00:00
#SBATCH --output=logs/molskill_%j.log
#SBATCH --error=logs/molskill_%j.err

BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${BENCHMARK_DIR}" || exit 1

# Ensure log directory exists
mkdir -p logs

echo "Starting MolSkill Scoring Job at $(date)"
echo "Running on node: $SLURM_NODENAME"
echo "Assigned GPU: $CUDA_VISIBLE_DEVICES"

echo "Working directory: $(pwd)"

# Activate molskill conda environment
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate molskill
else
    echo "Could not find conda at ${HOME}/miniconda3. Activate molskill manually."
    exit 1
fi

echo "Using Python: $(which python)"
python -c "from molskill.scorer import MolSkillScorer; print('MolSkill import OK')"

# Set PYTHONPATH to include src (for any shared benchmark utilities)
export PYTHONPATH="$(pwd)/src"


MOLSKILL_ARGS="--batch-size 64 --quiet"

# Run batch scoring on exps
echo "Scoring main experiments..."
python scripts/score_molecules.py --scorer molskill --exps-dir exps ${MOLSKILL_ARGS} --force

# Run batch scoring on exps_upperbound
echo "Scoring upper-bound experiments..."
python scripts/score_molecules.py --scorer molskill --exps-dir exps_upperbound --skip-reference-set ${MOLSKILL_ARGS} --force

echo "MolSkill Job finished at $(date)"
