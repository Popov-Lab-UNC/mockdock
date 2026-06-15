#!/bin/bash
#SBATCH --job-name=gen_configs
#SBATCH --output=logs/05_configs_%j.log
#SBATCH --error=logs/05_configs_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=general

# Step 5: Merge benchmark results and generate YAML configs

BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${BENCHMARK_DIR}/scripts/common/env.sh"
mockdock_setup_root
mockdock_activate_venv
mkdir -p data logs generated_configs

# Merge benchmark chunks (from step 4)
echo "Merging benchmark chunks..."
python scripts/dataset/merge_results.py \
    --pattern "data/intermediate/chembl_docking_benchmark/chunk_*.csv" \
    --output data/chembl_docking_benchmark.csv \
    --dedup-cols target_chembl_id pdb_id document_chembl_id

# Merge MCS results (from step 4b)
echo "Merging MCS results..."
python scripts/dataset/compute_mcs.py \
    --merge \
    --input data/mcs_results.csv \
    --output data/mcs_results.csv

# Generate configs (uses MCS from mcs_results.csv)
echo "Generating YAML configs..."
python scripts/dataset/generate_benchmark_configs.py \
    --input data/chembl_docking_benchmark.csv \
    --output-dir generated_configs \
    --top-n 0 \
    --require-crystal-in-assay \
    --max-resolution 3.0 \
    --min-compounds 20

echo "Done! Check generated_configs/"
