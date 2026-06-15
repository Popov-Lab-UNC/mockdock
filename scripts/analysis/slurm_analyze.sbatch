#!/bin/bash
#SBATCH --job-name=analyze
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/analyze_%j.log
#SBATCH --error=logs/analyze_%j.err

set -euo pipefail
BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${BENCHMARK_DIR}/scripts/common/env.sh"
source "${BENCHMARK_DIR}/scripts/common/slurm.sh"
mockdock_setup_root
mockdock_activate_venv
mockdock_slurm_header "MockDock analysis"

python scripts/analysis/analyze_experiments.py --exps-dir exps --output-dir analysis_exps --force
python scripts/analysis/analyze_experiments.py --exps-dir exps_upperbound --output-dir analysis_exps_upperbound --force

mockdock_slurm_footer "MockDock analysis"
