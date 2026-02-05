#!/bin/bash
# =============================================================================
# ChEMBL → PDB → Docking Benchmark Pipeline
# =============================================================================
#
# This pipeline creates a comprehensive benchmark dataset for molecular docking:
#
# OUTPUTS:
#   1. data/chembl_pdb_map.csv          - Full ChEMBL → PDB mapping
#   2. data/chembl_pdb_druglike.csv     - Filtered to drug-like ligands only
#   3. data/chembl_docking_benchmark.csv - Target + Doc + PDB with matching ligands
#   4. data/chembl_docking_benchmark.csv - (updated with mcs_smiles from step 4b)
#   5. generated_configs/*.yaml          - Ready-to-use docking workflow configs
#
# USAGE:
#   Option 1: Run sequentially (slow, ~24+ hours)
#       ./run_pipeline.sh sequential
#
#   Option 2: Submit to SLURM (parallel, ~4-6 hours)
#       ./run_pipeline.sh slurm
#
#   Option 3: Run individual steps
#       ./run_pipeline.sh step1  # Fetch ChEMBL targets
#       ./run_pipeline.sh step2  # Map to PDB (submit array job)
#       ./run_pipeline.sh step3  # Filter drug-like
#       ./run_pipeline.sh step4  # Find matching documents (submit array job)
#       ./run_pipeline.sh step4b # Compute MCS for documents (submit array job)
#       ./run_pipeline.sh step5  # Generate configs
#
# =============================================================================

set -e

cd /work/users/s/h/shuhang/benchmark
source .venv/bin/activate
mkdir -p data logs generated_configs

SBATCH_EXPORT="ALL"
if [[ -n "${CHEMBL_SQLITE_PATH}" ]]; then
    SBATCH_EXPORT="ALL,CHEMBL_SQLITE_PATH"
fi

case "$1" in
    sequential)
        echo "Running full pipeline sequentially (this will take many hours)..."
        if [[ -n "${CHEMBL_SQLITE_PATH}" ]]; then
            python pipeline/fetch_chembl_targets.py --chembl-sqlite "${CHEMBL_SQLITE_PATH}"
        else
            python pipeline/fetch_chembl_targets.py
        fi
        python pipeline/map_pdb_ligands.py
        python pipeline/filter_druglike.py
        if [[ -n "${CHEMBL_SQLITE_PATH}" ]]; then
            python pipeline/find_matching_documents.py --chembl-sqlite "${CHEMBL_SQLITE_PATH}"
        else
            python pipeline/find_matching_documents.py
        fi
        python pipeline/compute_mcs.py
        python pipeline/generate_benchmark_configs.py
        echo "Done!"
        ;;
    
    slurm)
        echo "Submitting pipeline to SLURM..."
        
        # Step 1: Fetch targets
        JOB1=$(sbatch --parsable --export=${SBATCH_EXPORT} pipeline/slurm_01_fetch_targets.sbatch)
        echo "Step 1 submitted: Job $JOB1"
        
        # Step 2: Map to PDB (depends on step 1)
        JOB2=$(sbatch --parsable --export=${SBATCH_EXPORT} --dependency=afterok:$JOB1 pipeline/slurm_02_map_pdb.sbatch)
        echo "Step 2 submitted: Job $JOB2 (array)"
        
        # Step 3: Filter (depends on all of step 2)
        JOB3=$(sbatch --parsable --export=${SBATCH_EXPORT} --dependency=afterok:$JOB2 pipeline/slurm_03_filter.sbatch)
        echo "Step 3 submitted: Job $JOB3"
        
        # Step 4: Find documents (depends on step 3)
        JOB4=$(sbatch --parsable --export=${SBATCH_EXPORT} --dependency=afterok:$JOB3 pipeline/slurm_04_find_docs.sbatch)
        echo "Step 4 submitted: Job $JOB4 (array)"
        
        # Step 4b: Compute MCS (depends on all of step 4)
        JOB4B=$(sbatch --parsable --export=${SBATCH_EXPORT} --dependency=afterok:$JOB4 pipeline/slurm_04b_mcs.sbatch)
        echo "Step 4b submitted: Job $JOB4B (array)"
        
        # Step 5: Generate configs (depends on all of step 4b)
        JOB5=$(sbatch --parsable --export=${SBATCH_EXPORT} --dependency=afterok:$JOB4B pipeline/slurm_05_generate_configs.sbatch)
        echo "Step 5 submitted: Job $JOB5"
        
        echo ""
        echo "Pipeline submitted! Monitor with: squeue -u $USER"
        ;;
    
    step1)
        echo "Running Step 1: Fetch ChEMBL targets..."
        if [[ -n "${CHEMBL_SQLITE_PATH}" ]]; then
            python pipeline/fetch_chembl_targets.py --chembl-sqlite "${CHEMBL_SQLITE_PATH}"
        else
            python pipeline/fetch_chembl_targets.py
        fi
        ;;
    
    step2)
        echo "Submitting Step 2: Map to PDB (SLURM array)..."
        sbatch --export=${SBATCH_EXPORT} pipeline/slurm_02_map_pdb.sbatch
        ;;
    
    step3)
        echo "Running Step 3: Filter drug-like..."
        sbatch --export=${SBATCH_EXPORT} pipeline/slurm_03_filter.sbatch
        ;;
    
    step4)
        echo "Submitting Step 4: Find matching documents (SLURM array)..."
        sbatch --export=${SBATCH_EXPORT} pipeline/slurm_04_find_docs.sbatch
        ;;
    
    step4b)
        echo "Submitting Step 4b: Compute MCS for documents (SLURM array)..."
        sbatch --export=${SBATCH_EXPORT} pipeline/slurm_04b_mcs.sbatch
        ;;
    
    step5)
        echo "Running Step 5: Generate configs..."
        sbatch --export=${SBATCH_EXPORT} pipeline/slurm_05_generate_configs.sbatch
        ;;
    
    *)
        echo "Usage: $0 {sequential|slurm|step1|step2|step3|step4|step4b|step5}"
        echo ""
        echo "  sequential  - Run all steps sequentially (slow)"
        echo "  slurm       - Submit all steps to SLURM with dependencies"
        echo "  step1-5     - Run/submit individual steps"
        echo ""
        echo "Environment:"
        echo "  CHEMBL_SQLITE_PATH=/path/to/chembl_36.db (optional; enables SQLite instead of API)"
        exit 1
        ;;
esac
