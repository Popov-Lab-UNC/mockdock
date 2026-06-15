#!/bin/bash
#SBATCH --job-name=pptt_variance
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/pptt_variance_%j.log
#SBATCH --error=logs/pptt_variance_%j.err

module load autogrid
module load autodock-gpu
set -euo pipefail
BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${BENCHMARK_DIR}/scripts/common/env.sh"
source "${BENCHMARK_DIR}/scripts/common/slurm.sh"
mockdock_setup_root
CONDA_SH=/nas/longleaf/home/shuhang/miniconda3/etc/profile.d/conda.sh mockdock_activate_conda cctbx
mockdock_activate_venv
mockdock_slurm_header "PptT variance"
CONFIG_PATH="${BENCHMARK_DIR}/src/mockdock/configs/PptT.toml"
RUN_DIR="${BENCHMARK_DIR}/variance_runs/PptT"
OUT_DIR="${BENCHMARK_DIR}/variance_analysis/PptT"
python scripts/variance/run_variance.py --config "${CONFIG_PATH}" --run-dir "${RUN_DIR}" --output-dir "${OUT_DIR}" --n-iters 5
mockdock_slurm_footer "PptT variance"
