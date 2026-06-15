#!/bin/bash
#SBATCH --job-name=filter_druglike
#SBATCH --output=logs/03_filter_%j.log
#SBATCH --error=logs/03_filter_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

# Step 3: Merge chunks and filter to drug-like molecules

BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${BENCHMARK_DIR}/scripts/common/env.sh"
mockdock_setup_root
mockdock_activate_venv
mkdir -p data logs

# First merge all chunks
echo "Merging PDB map chunks..."
python scripts/dataset/merge_results.py \
    --pattern "data/intermediate/chembl_pdb_map/chunk_*.csv" \
    --output data/chembl_pdb_map.csv \
    --dedup-cols pdb_id ligand_resname

# Then filter to drug-like
echo "Filtering to drug-like molecules..."
python scripts/dataset/filter_druglike.py \
    --input data/chembl_pdb_map.csv \
    --output data/chembl_pdb_druglike.csv

echo "Done! Check data/chembl_pdb_druglike.csv"
