#!/bin/bash
#SBATCH --job-name=molskill
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01-00:00:00
#SBATCH --output=logs/molskill_%j.log
#SBATCH --error=logs/molskill_%j.err

set -euo pipefail
BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${BENCHMARK_DIR}/scripts/common/env.sh"
source "${BENCHMARK_DIR}/scripts/common/slurm.sh"
mockdock_setup_root
mockdock_slurm_header "MolSkill scoring"
mockdock_activate_conda molskill
python -c "from molskill.scorer import MolSkillScorer; print('MolSkill import OK')"
MOLSKILL_ARGS="--batch-size 64 --quiet"
python scripts/scoring/score_molecules.py --scorer molskill --exps-dir exps --skip-reference-set ${MOLSKILL_ARGS}
python scripts/scoring/score_molecules.py --scorer molskill --exps-dir exps_upperbound --skip-reference-set ${MOLSKILL_ARGS}
mockdock_slurm_footer "MolSkill scoring"
