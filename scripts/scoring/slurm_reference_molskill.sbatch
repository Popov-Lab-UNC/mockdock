#!/bin/bash
#SBATCH --job-name=ref_molskill
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ref_molskill_%j.log
#SBATCH --error=logs/ref_molskill_%j.err

set -euo pipefail
BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${BENCHMARK_DIR}/scripts/common/env.sh"
source "${BENCHMARK_DIR}/scripts/common/slurm.sh"
mockdock_setup_root
mockdock_slurm_header "MolSkill reference scoring"
mockdock_activate_conda molskill

python -c "from molskill.scorer import MolSkillScorer; print('MolSkill import OK')"
MOLSKILL_ARGS="--batch-size 64 --quiet"
python scripts/scoring/score_molecules.py --scorer molskill --reference-only --reference-targets PptT ${MOLSKILL_ARGS}

mockdock_slurm_footer "MolSkill reference scoring"
