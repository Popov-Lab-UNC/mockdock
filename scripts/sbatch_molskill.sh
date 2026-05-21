#!/bin/bash
#SBATCH --job-name=mockdock_molskill
#SBATCH --partition=volta-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/molskill_%j.log
#SBATCH --error=logs/molskill_%j.err

# Ensure log directory exists
mkdir -p logs

echo "Starting MolSkill Scoring Job at $(date)"
echo "Running on node: $SLURM_NODENAME"
echo "Assigned GPU: $CUDA_VISIBLE_DEVICES"

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
python scripts/run_molskill.py --exps-dir exps

# Run batch scoring on exps_upperbound
echo "Scoring upper-bound experiments..."
python scripts/run_molskill.py --exps-dir exps_upperbound

echo "MolSkill Job finished at $(date)"
