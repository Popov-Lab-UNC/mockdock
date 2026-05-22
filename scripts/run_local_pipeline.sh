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

# Step 1: Initial analyze_experiments.py to generate cache files
echo "--> [Step 1/4] Generating initial caches..."
activate_venv
python scripts/analyze_experiments.py --exps-dir exps --output-dir analysis_exps

# Step 2: run_molskill.py
echo "--> [Step 2/4] Scoring with MolSkill..."
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate molskill
    python scripts/run_molskill.py --exps-dir exps --batch-size 64 --quiet
else
    echo "Warning: Conda not found at ${HOME}/miniconda3. Skipping MolSkill."
fi

# Step 3: run_aizynthfinder.py
echo "--> [Step 3/4] Scoring with AIZynthFinder..."
activate_venv
python scripts/run_aizynthfinder.py --exps-dir exps

# Step 4: Final analyze_experiments.py to regenerate figures with all scores
echo "--> [Step 4/4] Regenerating final figures..."
python scripts/analyze_experiments.py --exps-dir exps --output-dir analysis_exps

echo "================================================================="
echo "Pipeline finished successfully!"
echo "Outputs and plots saved in analysis_exps"
echo "================================================================="
