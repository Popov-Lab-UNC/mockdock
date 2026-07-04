#!/bin/bash
#SBATCH --job-name=stoplight
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/stoplight_%j.log
#SBATCH --error=logs/stoplight_%j.err

set -euo pipefail
BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${BENCHMARK_DIR}/scripts/common/env.sh"
source "${BENCHMARK_DIR}/scripts/common/slurm.sh"
mockdock_setup_root
mockdock_activate_venv
mockdock_slurm_header "Stoplight scoring"

# Sycamore nodes expose 192 CPUs, but Stoplight worker processes each load large
# model bundles once. Cap concurrent workers to avoid filesystem/memory storms.
export STOPLIGHT_MAX_WORKERS="${STOPLIGHT_MAX_WORKERS:-64}"
export STOPLIGHT_TIMEOUT_STARTUP_SEC="${STOPLIGHT_TIMEOUT_STARTUP_SEC:-900}"
export STOPLIGHT_TIMEOUT_PER_MOL_SEC="${STOPLIGHT_TIMEOUT_PER_MOL_SEC:-15}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "STOPLIGHT_MAX_WORKERS=${STOPLIGHT_MAX_WORKERS}"
echo "STOPLIGHT_TIMEOUT_STARTUP_SEC=${STOPLIGHT_TIMEOUT_STARTUP_SEC}"
echo "STOPLIGHT_TIMEOUT_PER_MOL_SEC=${STOPLIGHT_TIMEOUT_PER_MOL_SEC}"

python scripts/scoring/score_molecules.py --scorer stoplight --exps-dir exps --skip-reference-set
python scripts/scoring/score_molecules.py --scorer stoplight --exps-dir exps_upperbound --skip-reference-set
mockdock_slurm_footer "Stoplight scoring"
