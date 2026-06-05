#!/bin/bash
#SBATCH --job-name=mockdock_analyze
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/analyze_%j.log
#SBATCH --error=logs/analyze_%j.err

# Ensure log directory exists
mkdir -p logs

echo "Starting Mockdock Analysis Job at $(date)"
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


# Run main analysis on exps
echo "Analyzing main experiments..."
python scripts/analyze_experiments.py --exps-dir exps --output-dir analysis_exps --force

# Run main analysis on exps_upperbound
echo "Analyzing upper-bound experiments..."
python scripts/analyze_experiments.py --exps-dir exps_upperbound --output-dir analysis_exps_upperbound --force

echo "Job finished at $(date)"
