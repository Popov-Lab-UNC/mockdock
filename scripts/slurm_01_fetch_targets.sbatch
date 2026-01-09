#!/bin/bash
#SBATCH --job-name=chembl_targets
#SBATCH --output=logs/01_targets_%j.log
#SBATCH --error=logs/01_targets_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=general

# Step 1: Fetch ChEMBL targets (quick, ~5 min)

cd /work/users/s/h/shuhang/benchmark
source .venv/bin/activate

mkdir -p data logs

python scripts/01_fetch_chembl_targets.py --output data/chembl_targets.csv

echo "Done! Check data/chembl_targets.csv"
