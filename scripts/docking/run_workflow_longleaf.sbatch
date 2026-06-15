#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --mem=32G
#SBATCH --time=11-00:00:00
#SBATCH --output=benchmark_%j.log
#SBATCH --error=benchmark_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shuhang@unc.edu

module load autodock-gpu
module load autogrid
BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${BENCHMARK_DIR}/scripts/common/env.sh"
mockdock_setup_root
mockdock_activate_venv

# Skip configs already completed in previous runs.
SKIP_STATUSES="${SKIP_STATUSES:-SUCCESS,FAILED_REF_MATCH,FAILED_GRID_PREP}"
export SKIP_STATUSES

# Build a lookup table of (target_id|pdb_id|doc_id|assay_id) keys to skip
declare -A SKIP_KEYS
echo "Scanning prior runs for completed workflows (SKIP_STATUSES=${SKIP_STATUSES})..."
while IFS= read -r key; do
    [ -n "$key" ] && SKIP_KEYS["$key"]=1
done < <(
python - <<'PY'
import csv
import glob
import os

skip_statuses = {s.strip() for s in os.environ.get("SKIP_STATUSES", "SUCCESS").split(",") if s.strip()}
paths = sorted(glob.glob("benchmark_runs/run_*/benchmark_summary.csv"))

keys = set()
for p in paths:
    try:
        with open(p, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                if (row.get("status") or "").strip() in skip_statuses:
                    target_id = (row.get("target_id") or "").strip()
                    pdb_id = (row.get("pdb_id") or "").strip()
                    doc_id = (row.get("doc_id") or "").strip()
                    assay_id = (row.get("assay_id") or "").strip()
                    if target_id and pdb_id and doc_id and assay_id:
                        keys.add(f"{target_id}|{pdb_id}|{doc_id}|{assay_id}")
    except Exception:
        continue

for k in sorted(keys):
    print(k)
PY
)
echo "Will skip ${#SKIP_KEYS[@]} workflows already present in prior run summaries."
echo ""

# Create a timestamped run directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="benchmark_runs/run_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

echo "Centralized Run Directory: $RUN_DIR"
echo ""

# Loop through all YAML configs in generated_configs/ as:
# <target_id>_<pdb_id>_<doc_id>_<assay_id>.yaml
shopt -s nullglob
for config_file in generated_configs/*.yaml; do
        base="$(basename "$config_file" .yaml)"
        IFS='_' read -r target_id pdb_id doc_id assay_id extra <<< "$base"
        if [ -n "${extra:-}" ] || [ -z "${target_id:-}" ] || [ -z "${pdb_id:-}" ] || [ -z "${doc_id:-}" ] || [ -z "${assay_id:-}" ]; then
            echo "Skipping malformed config name (expected 4 parts): $config_file"
            continue
        fi
        key="${target_id}|${pdb_id}|${doc_id}|${assay_id}"

        if [ -n "${SKIP_KEYS[$key]:-}" ]; then
            echo "Skipping (already done): $config_file"
            echo "Key: $key"
            echo ""
            continue
        fi

        echo "Running: $config_file"
        echo "===================="
        python -u scripts/docking/run_workflow.py --config "$config_file" --run-dir "$RUN_DIR" --cpus "${SLURM_CPUS_PER_TASK:-32}" --gpus "${SLURM_GPUS:-1}" --backend autodock_gpu --cache-dir "data/chembl_cache"
        echo ""
done

echo "All workflows completed! Results summary at $RUN_DIR/benchmark_summary.csv"