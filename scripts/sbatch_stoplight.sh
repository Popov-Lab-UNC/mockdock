#!/bin/bash
#SBATCH --job-name=mockdock_stoplight
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/stoplight_%j.log
#SBATCH --error=logs/stoplight_%j.err

# Ensure log directory exists
mkdir -p logs

echo "Starting Stoplight Scoring Job at $(date)"
echo "Running on node: $SLURM_NODENAME"

# Move to workspace directory if not already there
cd "$(dirname "$0")/.." || exit 1

# Activate virtual environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment (venv)..."
    source venv/bin/activate
fi

# Set PYTHONPATH to include src
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run batch scoring on exps
echo "Scoring main experiments..."
python scripts/score_molecules.py --scorer stoplight --exps-dir exps

# Run batch scoring on exps_upperbound
echo "Scoring upper-bound experiments..."
python scripts/score_molecules.py --scorer stoplight --exps-dir exps_upperbound

echo "Stoplight Job finished at $(date)"
