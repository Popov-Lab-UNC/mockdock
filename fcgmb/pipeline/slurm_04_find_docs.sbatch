#!/bin/bash
#SBATCH --job-name=find_docs
#SBATCH --output=logs/04_find_docs_%A_%a.log
#SBATCH --error=logs/04_find_docs_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=general
#SBATCH --array=0-9  # 10 parallel jobs

# Step 4: Find ChEMBL documents matching crystal ligands (slow, ~2-4 hours per chunk)
# Uses SLURM array jobs to parallelize

cd /work/users/s/h/shuhang/benchmark
source .venv/bin/activate

mkdir -p data logs

# Calculate chunk boundaries based on unique targets
TOTAL_TARGETS=$(python -c "import pandas as pd; print(pd.read_csv('data/chembl_pdb_druglike.csv')['chembl_target_id'].nunique())")
CHUNK_SIZE=$(( (TOTAL_TARGETS + 10 - 1) / 10 ))  # Ceiling division

START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END=$((START + CHUNK_SIZE))

echo "Processing targets $START to $END (of $TOTAL_TARGETS unique)"

# Configure: Include both patents and publications by default
# Add --no-patents or --no-publications to exclude
python fcgmb/pipeline/find_matching_documents.py \
    --input data/chembl_pdb_druglike.csv \
    --output data/chembl_docking_benchmark.csv \
    --include-patents \
    --include-publications \
    --similarity-threshold 0.7 \
    --min-compounds 20 \
    --start $START \
    --end $END \
    --delay 0.5

echo "Chunk $SLURM_ARRAY_TASK_ID done!"
