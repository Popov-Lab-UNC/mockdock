#!/bin/bash
#SBATCH --job-name=ref_aizynth
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/ref_aizynth_%j.log
#SBATCH --error=logs/ref_aizynth_%j.err

set -euo pipefail
BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${BENCHMARK_DIR}/scripts/common/env.sh"
source "${BENCHMARK_DIR}/scripts/common/slurm.sh"
mockdock_setup_root
mockdock_activate_venv /work/users/s/h/shuhang/aizynthfinder/.venv
mockdock_slurm_header "AIZynthFinder reference scoring"

# Sycamore nodes expose 192 CPUs, but AIZynthFinder worker processes each load large
# policy and stock files. Cap concurrent workers to avoid filesystem/memory storms.
export AIZYNTH_MAX_WORKERS="${AIZYNTH_MAX_WORKERS:-64}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export CUDA_VISIBLE_DEVICES=""

echo "AIZYNTH_MAX_WORKERS=${AIZYNTH_MAX_WORKERS}"

python scripts/scoring/score_molecules.py --scorer aizynthfinder --reference-only --reference-targets PptT --force
mockdock_slurm_footer "AIZynthFinder reference scoring"
