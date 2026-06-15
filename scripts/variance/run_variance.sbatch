#!/bin/bash
#SBATCH --job-name=dock_variance
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01-00:00:00
#SBATCH --output=logs/variance_%j.log
#SBATCH --error=logs/variance_%j.err
#SBATCH --mail-user=shuhang@unc.edu
#SBATCH --mail-type=ALL

module load autodock-gpu
module load autogrid
set -euo pipefail
BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${BENCHMARK_DIR}/scripts/common/env.sh"
source "${BENCHMARK_DIR}/scripts/common/slurm.sh"
mockdock_setup_root
mockdock_activate_venv
mockdock_slurm_header "Docking variance"
python scripts/variance/run_variance.py --config-dir top_30_configs --iterations 5 --output-dir variance_runs
mockdock_slurm_footer "Docking variance"
