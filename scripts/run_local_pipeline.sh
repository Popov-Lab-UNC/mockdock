#!/bin/bash
# scripts/run_local_pipeline.sh
# Runs the entire Mockdock analysis and scoring pipeline locally in a single command.

# Move to workspace directory if not already there
cd "$(dirname "$0")/.." || exit 1

echo "================================================================="
echo "Starting local Mockdock Analysis & Scoring Pipeline..."
echo "================================================================="

# Helper to activate virtualenv
activate_venv() {
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    elif [ -d "venv" ]; then
        source venv/bin/activate
    else
        echo "Error: Virtual env not found!"
        exit 1
    fi
}

# Set PYTHONPATH to include src
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Step 1: Initial analyze_experiments.py to generate clean cache files
echo "--> [Step 1/5] Generating initial caches..."
activate_venv
python scripts/analyze_experiments.py --exps-dir exps --output-dir analysis_exps --force
python scripts/analyze_experiments.py --exps-dir exps_upperbound --output-dir analysis_exps_upperbound --force

# Step 2: score MolSkill
echo "--> [Step 2/5] Scoring with MolSkill..."
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate molskill
    python scripts/score_molecules.py --scorer molskill --exps-dir exps --batch-size 64 --quiet --force
    python scripts/score_molecules.py --scorer molskill --exps-dir exps_upperbound --skip-reference-set --batch-size 64 --quiet --force
else
    echo "Warning: Conda not found at ${HOME}/miniconda3. Skipping MolSkill."
fi

# Step 3: score AIZynthFinder
echo "--> [Step 3/5] Scoring with AIZynthFinder..."
activate_venv
python scripts/score_molecules.py --scorer aizynthfinder --exps-dir exps --force
python scripts/score_molecules.py --scorer aizynthfinder --exps-dir exps_upperbound --skip-reference-set --force

# Step 4: score Stoplight
echo "--> [Step 4/5] Scoring with Stoplight..."
activate_venv
python scripts/score_molecules.py --scorer stoplight --exps-dir exps --force
python scripts/score_molecules.py --scorer stoplight --exps-dir exps_upperbound --skip-reference-set --force

# Step 5: Final analyze_experiments.py to regenerate figures with all scores
echo "--> [Step 5/5] Regenerating final figures..."
python scripts/analyze_experiments.py --exps-dir exps --output-dir analysis_exps
python scripts/analyze_experiments.py --exps-dir exps_upperbound --output-dir analysis_exps_upperbound

echo "================================================================="
echo "Pipeline finished successfully!"
echo "Outputs and plots saved in analysis_exps and analysis_exps_upperbound"
echo "================================================================="
