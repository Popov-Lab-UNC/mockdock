#!/bin/bash
#SBATCH --job-name=compute_mcs
#SBATCH --output=logs/04b_mcs_%A_%a.log
#SBATCH --error=logs/04b_mcs_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=general
#SBATCH --array=0-9  # 10 parallel jobs

# Step 4b: Compute Maximum Common Substructure (MCS) for documents
# Uses SLURM array jobs to parallelize
# After all array jobs complete, run merge:
#   python scripts/pipeline/compute_mcs.py --merge --input data/mcs_results.csv --output data/mcs_results.csv

# Ensure we are in the project root
if [[ -z "${PROJECT_ROOT}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
fi
cd "$PROJECT_ROOT"
source .venv/bin/activate

mkdir -p data logs

# Step 4 runs as chunks and writes to data/intermediate/chembl_docking_benchmark/chunk_*.csv.
# Ensure the merged input exists for step 4b.
if [ ! -f data/chembl_docking_benchmark.csv ] || [ ! -s data/chembl_docking_benchmark.csv ]; then
    echo "Merged benchmark file missing; merging step 4 chunks..."
    python scripts/pipeline/merge_results.py \
        --pattern "data/intermediate/chembl_docking_benchmark/chunk_*.csv" \
        --output data/chembl_docking_benchmark.csv \
        --dedup-cols target_chembl_id pdb_id document_chembl_id
fi

# Check again after merge attempt
if [ ! -f data/chembl_docking_benchmark.csv ] || [ ! -s data/chembl_docking_benchmark.csv ]; then
    echo "ERROR: data/chembl_docking_benchmark.csv does not exist or is empty!"
    echo "Step 4 may not have produced any chunk outputs under data/intermediate/chembl_docking_benchmark/."
    exit 1
fi

# Calculate chunk boundaries based on unique documents
TOTAL_DOCS=$(python -c "
import pandas as pd
import sys
try:
    df = pd.read_csv('data/chembl_docking_benchmark.csv')
    if len(df) == 0:
        print('0', file=sys.stderr)
        sys.exit(1)
    print(df['document_chembl_id'].nunique())
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
")

if [ -z "$TOTAL_DOCS" ] || [ "$TOTAL_DOCS" -eq 0 ]; then
    echo "ERROR: Could not determine number of documents or file is empty!"
    exit 1
fi

CHUNK_SIZE=$(( (TOTAL_DOCS + 10 - 1) / 10 ))  # Ceiling division

START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END=$((START + CHUNK_SIZE))

echo "Processing documents $START to $END (of $TOTAL_DOCS unique)"

python scripts/pipeline/compute_mcs.py \
    --input data/chembl_docking_benchmark.csv \
    --output data/mcs_results.csv \
    --max-compounds 100 \
    --timeout 10 \
    --start $START \
    --end $END \
    --delay 0.5

echo "Chunk $SLURM_ARRAY_TASK_ID done!"
