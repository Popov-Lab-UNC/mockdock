#!/bin/bash
#SBATCH --job-name=mockdock_val
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/validation_%j.log
#SBATCH --error=logs/validation_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shuhang@unc.edu

module load autodock-gpu
module load autogrid
set -euo pipefail
BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${BENCHMARK_DIR}/scripts/common/env.sh"
source "${BENCHMARK_DIR}/scripts/common/slurm.sh"
mockdock_setup_root
mockdock_activate_venv
mockdock_slurm_header "MockDock validation"
export ADGPU_EXE="${ADGPU_EXE:-adgpu}"
python -u scripts/validation/validation_test.py
mockdock_slurm_footer "MockDock validation"
