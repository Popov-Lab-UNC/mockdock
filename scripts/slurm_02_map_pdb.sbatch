#!/bin/bash
#SBATCH --job-name=pdb_map
#SBATCH --output=logs/02_pdb_map_%A_%a.log
#SBATCH --error=logs/02_pdb_map_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=general
#SBATCH --array=0-11  # 12 parallel jobs, adjust based on total targets

# Step 2: Map UniProt to PDB (slow, ~1-2 hours per 500 targets)
# Uses SLURM array jobs to parallelize

cd /work/users/s/h/shuhang/benchmark
source .venv/bin/activate

mkdir -p data logs

# Calculate chunk boundaries
TOTAL_TARGETS=$(wc -l < data/chembl_targets.csv)
TOTAL_TARGETS=$((TOTAL_TARGETS - 1))  # Subtract header
CHUNK_SIZE=$(( (TOTAL_TARGETS + 12 - 1) / 12 ))  # Ceiling division

START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END=$((START + CHUNK_SIZE))

echo "Processing targets $START to $END (of $TOTAL_TARGETS)"

python scripts/02_map_pdb_ligands.py \
    --input data/chembl_targets.csv \
    --output data/chembl_pdb_map.csv \
    --start $START \
    --end $END \
    --delay 0.1

echo "Chunk $SLURM_ARRAY_TASK_ID done!"
