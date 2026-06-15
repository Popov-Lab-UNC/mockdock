#!/bin/bash
#SBATCH --job-name=chembl_targets
#SBATCH --output=logs/01_targets_%j.log
#SBATCH --error=logs/01_targets_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=general

# Step 1: Fetch ChEMBL targets (quick, ~5 min)

BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${BENCHMARK_DIR}/scripts/common/env.sh"
mockdock_setup_root
mockdock_activate_venv
mkdir -p data logs

python scripts/dataset/fetch_chembl_targets.py \
    --output data/chembl_targets.csv \
    --chembl-sqlite "${CHEMBL_SQLITE_PATH:-/work/users/s/h/shuhang/chembl_36/chembl_36_sqlite/chembl_36.db}"

echo "Done! Check data/chembl_targets.csv"
