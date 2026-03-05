#!/bin/bash
#SBATCH --job-name=chembl_targets
#SBATCH --output=logs/01_targets_%j.log
#SBATCH --error=logs/01_targets_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=general

# Step 1: Fetch ChEMBL targets (quick, ~5 min)

# Ensure we are in the project root
if [[ -z "${PROJECT_ROOT}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
fi
cd "$PROJECT_ROOT"
source .venv/bin/activate

mkdir -p data logs

python scripts/pipeline/fetch_chembl_targets.py \
    --output data/chembl_targets.csv \
    --chembl-sqlite /work/users/s/h/shuhang/chembl_36/chembl_36_sqlite/chembl_36.db

echo "Done! Check data/chembl_targets.csv"
