#!/bin/bash
#SBATCH --job-name=filter_druglike
#SBATCH --output=logs/03_filter_%j.log
#SBATCH --error=logs/03_filter_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

# Step 3: Merge chunks and filter to drug-like molecules

cd /work/users/s/h/shuhang/benchmark
source .venv/bin/activate

mkdir -p data logs

# First merge all chunks
echo "Merging PDB map chunks..."
python fcgmb/pipeline/merge_results.py \
    --pattern "data/intermediate/chembl_pdb_map/chunk_*.csv" \
    --output data/chembl_pdb_map.csv \
    --dedup-cols pdb_id ligand_resname

# Then filter to drug-like
echo "Filtering to drug-like molecules..."
python fcgmb/pipeline/filter_druglike.py \
    --input data/chembl_pdb_map.csv \
    --output data/chembl_pdb_druglike.csv

echo "Done! Check data/chembl_pdb_druglike.csv"
