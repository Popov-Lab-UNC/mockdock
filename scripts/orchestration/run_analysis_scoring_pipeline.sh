#!/bin/bash
# Run the MockDock analysis and scoring pipeline locally.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/env.sh"
mockdock_setup_root

echo "Starting local MockDock analysis and scoring pipeline..."
mockdock_activate_venv
python scripts/analysis/analyze_experiments.py --exps-dir exps --output-dir analysis_exps --force
python scripts/analysis/analyze_experiments.py --exps-dir exps_upperbound --output-dir analysis_exps_upperbound --force
if mockdock_activate_conda molskill; then
    python scripts/scoring/score_molecules.py --scorer molskill --exps-dir exps --batch-size 64 --quiet --force
    python scripts/scoring/score_molecules.py --scorer molskill --exps-dir exps_upperbound --skip-reference-set --batch-size 64 --quiet --force
else
    echo "Skipping MolSkill because the molskill conda environment is unavailable."
fi
mockdock_activate_venv
python scripts/scoring/score_molecules.py --scorer aizynthfinder --exps-dir exps --force
python scripts/scoring/score_molecules.py --scorer aizynthfinder --exps-dir exps_upperbound --skip-reference-set --force
python scripts/scoring/score_molecules.py --scorer stoplight --exps-dir exps --force
python scripts/scoring/score_molecules.py --scorer stoplight --exps-dir exps_upperbound --skip-reference-set --force
python scripts/analysis/analyze_experiments.py --exps-dir exps --output-dir analysis_exps
python scripts/analysis/analyze_experiments.py --exps-dir exps_upperbound --output-dir analysis_exps_upperbound
echo "Pipeline finished. Outputs are in analysis_exps and analysis_exps_upperbound."
