#!/bin/bash
# Submit the MockDock analysis and scoring pipeline with SLURM dependencies.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/env.sh"
mockdock_setup_root

echo "Submitting MockDock analysis and scoring pipeline..."
JOB1_OUT=$(sbatch scripts/analysis/slurm_analyze.sbatch)
JOB1_ID=$(echo "${JOB1_OUT}" | awk '{print $NF}')
echo "Submitted initial analysis: ${JOB1_ID}"
JOB2_OUT=$(sbatch --dependency=afterok:${JOB1_ID} scripts/scoring/slurm_molskill.sbatch)
JOB2_ID=$(echo "${JOB2_OUT}" | awk '{print $NF}')
echo "Submitted MolSkill scoring: ${JOB2_ID}"
JOB3_OUT=$(sbatch --dependency=afterok:${JOB1_ID} scripts/scoring/slurm_aizynth.sbatch)
JOB3_ID=$(echo "${JOB3_OUT}" | awk '{print $NF}')
echo "Submitted AIZynthFinder scoring: ${JOB3_ID}"
JOB4_OUT=$(sbatch --dependency=afterok:${JOB1_ID} scripts/scoring/slurm_stoplight.sbatch)
JOB4_ID=$(echo "${JOB4_OUT}" | awk '{print $NF}')
echo "Submitted Stoplight scoring: ${JOB4_ID}"
JOB5_OUT=$(sbatch --dependency=afterok:${JOB2_ID}:${JOB3_ID}:${JOB4_ID} scripts/analysis/slurm_analyze.sbatch)
JOB5_ID=$(echo "${JOB5_OUT}" | awk '{print $NF}')
echo "Submitted final analysis: ${JOB5_ID}"
