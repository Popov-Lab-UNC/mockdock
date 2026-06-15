#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --partition=general
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=benchmark_%j.log
#SBATCH --error=benchmark_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shuhang@unc.edu

module load autogrid
BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${BENCHMARK_DIR}/scripts/common/env.sh"
mockdock_setup_root
mockdock_activate_venv

# Skip configs already completed in previous runs.
# Default: skip previously SUCCESSful workflows + common non-recoverable failures.
# Override to re-try failures, e.g.:
#   export SKIP_STATUSES="SUCCESS"
# or to customize:
#   export SKIP_STATUSES="SUCCESS,FAILED_REF_MATCH,FAILED_PDB_404,FAILED_LIGAND_MISSING"
#SKIP_STATUSES="${SKIP_STATUSES:-SUCCESS,FAILED_REF_MATCH,FAILED_LIGAND_MISSING}"
SKIP_STATUSES="${SKIP_STATUSES:-SUCCESS}"
# Make sure the Python skip-scanner can see it.
export SKIP_STATUSES

# Build a lookup table of (target_id|pdb_id|doc_id) keys to skip
declare -A SKIP_KEYS
echo "Scanning prior runs for completed workflows (SKIP_STATUSES=${SKIP_STATUSES})..."
while IFS= read -r key; do
    if [ -n "$key" ]; then
        SKIP_KEYS["$key"]=1
    fi
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
                    if target_id and pdb_id and doc_id:
                        keys.add(f"{target_id}|{pdb_id}|{doc_id}")
    except Exception:
        # Ignore unreadable/partial files
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

# Loop through all YAML configs in generated_configs/
for config_file in generated_configs/*.yaml; do
    if [ -f "$config_file" ]; then
        # Configs are generated as: <target_id>_<pdb_id>_<doc_id>.yaml
        base="$(basename "$config_file" .yaml)"
        target_id="${base%%_*}"
        rest="${base#*_}"
        pdb_id="${rest%%_*}"
        doc_id="${rest#*_}"
        key="${target_id}|${pdb_id}|${doc_id}"

        if [ -n "${SKIP_KEYS[$key]}" ]; then
            echo "===================="
            echo "Skipping (already done): $config_file"
            echo "Key: $key"
            echo "===================="
            echo ""
            continue
        fi

        echo "===================="
        echo "Running: $config_file"
        echo "===================="
        CPUS_ARG=()
        CPUS_ARG=()
        if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
            CPUS_ARG=(--cpus "$SLURM_CPUS_PER_TASK")
        fi
        python -u scripts/docking/run_workflow.py --config "$config_file" --run-dir "$RUN_DIR" "${CPUS_ARG[@]}"
        echo ""
    fi
done

echo "All workflows completed! Results summary at $RUN_DIR/benchmark_summary.csv"