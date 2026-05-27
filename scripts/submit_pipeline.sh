#!/bin/bash
# scripts/submit_pipeline.sh
# Automates the entire Mockdock analysis and scoring pipeline using SLURM job dependencies.

echo "================================================================="
echo "Submitting Mockdock Analysis & Scoring Pipeline..."
echo "================================================================="

# 1. Submit initial analyze job to generate cache files
JOB1_OUT=$(sbatch scripts/sbatch_analyze.sh)
JOB1_ID=$(echo "${JOB1_OUT}" | awk '{print $NF}')
echo "Submitted Initial Analysis (Job 1 ID: ${JOB1_ID})"

# 2. Submit MolSkill scoring job (dependent on Job 1 succeeding)
JOB2_OUT=$(sbatch --dependency=afterok:${JOB1_ID} scripts/sbatch_molskill.sh)
JOB2_ID=$(echo "${JOB2_OUT}" | awk '{print $NF}')
echo "Submitted MolSkill Scoring (Job 2 ID: ${JOB2_ID}, dependent on ${JOB1_ID})"

# 3. Submit AIZynthFinder scoring job (dependent on Job 1 succeeding)
JOB3_OUT=$(sbatch --dependency=afterok:${JOB1_ID} scripts/sbatch_aizynth.sh)
JOB3_ID=$(echo "${JOB3_OUT}" | awk '{print $NF}')
echo "Submitted AIZynthFinder Scoring (Job 3 ID: ${JOB3_ID}, dependent on ${JOB1_ID})"

# 4. Submit Stoplight scoring job (dependent on Job 1 succeeding)
JOB4_OUT=$(sbatch --dependency=afterok:${JOB1_ID} scripts/sbatch_stoplight.sh)
JOB4_ID=$(echo "${JOB4_OUT}" | awk '{print $NF}')
echo "Submitted Stoplight ADMET Scoring (Job 4 ID: ${JOB4_ID}, dependent on ${JOB1_ID})"

# 5. Submit final analyze job (dependent on all scoring jobs succeeding)
JOB5_OUT=$(sbatch --dependency=afterok:${JOB2_ID}:${JOB3_ID}:${JOB4_ID} scripts/sbatch_analyze.sh)
JOB5_ID=$(echo "${JOB5_OUT}" | awk '{print $NF}')
echo "Submitted Final Analysis & Plotting (Job 5 ID: ${JOB5_ID}, dependent on ${JOB2_ID}, ${JOB3_ID}, and ${JOB4_ID})"

echo "================================================================="
echo "All jobs submitted! The pipeline will run automatically in order."
echo "You can check status with: squeue -u $(whoami)"
echo "================================================================="
