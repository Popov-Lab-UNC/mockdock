#!/bin/bash
#SBATCH --job-name=find_docs
#SBATCH --output=logs/04_find_docs_%A_%a.log
#SBATCH --error=logs/04_find_docs_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=general
#SBATCH --array=0-9  # 10 parallel jobs

# Step 4: Find ChEMBL documents matching crystal ligands (slow, ~2-4 hours per chunk)
# Uses SLURM array jobs to parallelize

# Ensure we are in the project root
if [[ -z "${PROJECT_ROOT}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
fi
cd "$PROJECT_ROOT"
source .venv/bin/activate

mkdir -p data logs

# Calculate chunk boundaries based on unique targets
TOTAL_TARGETS=$(python -c "import pandas as pd; print(pd.read_csv('data/chembl_pdb_druglike.csv')['target_chembl_id'].nunique())")
CHUNK_SIZE=$(( (TOTAL_TARGETS + 10 - 1) / 10 ))  # Ceiling division

START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END=$((START + CHUNK_SIZE))

echo "Processing targets $START to $END (of $TOTAL_TARGETS unique)"

# Configure: Include both patents and publications by default
# Add --no-patents or --no-publications to exclude
python scripts/pipeline/find_matching_documents.py \
    --input data/chembl_pdb_druglike.csv \
    --output data/chembl_docking_benchmark.csv \
    --min-compounds 20 \
    --min-pchembl-range 1.0 \
    --start $START \
    --end $END \
    --delay 0.5 \
    --chembl-sqlite /work/users/s/h/shuhang/chembl_36/chembl_36_sqlite/chembl_36.db

echo "Chunk $SLURM_ARRAY_TASK_ID done!"
