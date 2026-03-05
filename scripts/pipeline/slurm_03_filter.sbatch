#!/bin/bash
#SBATCH --job-name=filter_druglike
#SBATCH --output=logs/03_filter_%j.log
#SBATCH --error=logs/03_filter_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

# Step 3: Merge chunks and filter to drug-like molecules

# Ensure we are in the project root
if [[ -z "${PROJECT_ROOT}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
fi
cd "$PROJECT_ROOT"
source .venv/bin/activate

mkdir -p data logs

# First merge all chunks
echo "Merging PDB map chunks..."
python scripts/pipeline/merge_results.py \
    --pattern "data/intermediate/chembl_pdb_map/chunk_*.csv" \
    --output data/chembl_pdb_map.csv \
    --dedup-cols pdb_id ligand_resname

# Then filter to drug-like
echo "Filtering to drug-like molecules..."
python scripts/pipeline/filter_druglike.py \
    --input data/chembl_pdb_map.csv \
    --output data/chembl_pdb_druglike.csv

echo "Done! Check data/chembl_pdb_druglike.csv"
