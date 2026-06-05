#!/bin/bash
#SBATCH --job-name=mockdock_aizynth
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/aizynth_%j.log
#SBATCH --error=logs/aizynth_%j.err

# Ensure log directory exists
mkdir -p logs

echo "Starting AIZynthFinder Scoring Job at $(date)"
echo "Running on node: $SLURM_NODENAME"
echo "Assigned GPU: $CUDA_VISIBLE_DEVICES"

# Move to workspace directory if not already there
BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${BENCHMARK_DIR}" || exit 1

# Activate virtual environment
if [ -d "/work/users/s/h/shuhang/aizynthfinder/.venv" ]; then
    echo "Activating AIZynthFinder virtual environment..."
    source /work/users/s/h/shuhang/aizynthfinder/.venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating local virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment (venv)..."
    source venv/bin/activate
fi

# Set PYTHONPATH to include src
export PYTHONPATH="$(pwd)/src"


# Run batch scoring on exps
echo "Scoring main experiments..."
python scripts/score_molecules.py --scorer aizynthfinder --exps-dir exps --force

# Run batch scoring on exps_upperbound
echo "Scoring upper-bound experiments..."
python scripts/score_molecules.py --scorer aizynthfinder --exps-dir exps_upperbound --skip-reference-set --force

echo "AIZynthFinder Job finished at $(date)"
