#!/bin/bash
#SBATCH --job-name=mockdock_stoplight
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/stoplight_%j.log
#SBATCH --error=logs/stoplight_%j.err

# Ensure log directory exists
mkdir -p logs

echo "Starting Stoplight Scoring Job at $(date)"
echo "Running on node: $SLURM_NODENAME"

# Move to workspace directory if not already there
BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${BENCHMARK_DIR}" || exit 1

# Activate virtual environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment (venv)..."
    source venv/bin/activate
fi

# Set PYTHONPATH to include src
export PYTHONPATH="$(pwd)/src"


# Run batch scoring on exps
echo "Scoring main experiments..."
python scripts/score_molecules.py --scorer stoplight --exps-dir exps --force

# Run batch scoring on exps_upperbound
echo "Scoring upper-bound experiments..."
python scripts/score_molecules.py --scorer stoplight --exps-dir exps_upperbound --skip-reference-set --force

echo "Stoplight Job finished at $(date)"
