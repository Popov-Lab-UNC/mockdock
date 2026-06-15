# Benchmark Scripts

Scripts are organized by benchmark stage. Run commands from the repository root unless a script says otherwise.

| Stage | Directory | Purpose |
| --- | --- | --- |
| Dataset generation | `scripts/dataset/` | Build ChEMBL/PDB benchmark candidate datasets and generated configs. |
| Docking workflows | `scripts/docking/` | Run the core docking workflow locally or through SLURM. |
| Variance calibration | `scripts/variance/` | Run repeated docking, analyze score variance, and prepare variance support data. |
| Experiments | `scripts/experiments/` | Launch model suites, including the parameterized PptT suite. |
| Scoring | `scripts/scoring/` | Run MolSkill, Stoplight, and AIZynthFinder scoring jobs. |
| Analysis | `scripts/analysis/` | Aggregate experiment metrics, compare runs, and generate figures. |
| Validation | `scripts/validation/` | Validate benchmark setup and docking behavior. |
| Orchestration | `scripts/orchestration/` | Submit or run the analysis/scoring pipeline end to end. |
| Common helpers | `scripts/common/` | Shared shell setup and Python constants used by other stages. |
| Archive | `scripts/archive/` | Deprecated launchers retained for reference with replacements documented. |

## Environments

Most scripts expect the benchmark virtual environment at `.venv` and source `scripts/common/env.sh`, which sets `BENCHMARK_DIR`, `PYTHONPATH`, and prints the active Python. Scoring jobs may activate specialized environments: MolSkill uses the `molskill` conda environment, AIZynthFinder uses `/work/users/s/h/shuhang/aizynthfinder/.venv`, and PptT model launches activate each model environment as needed.

Dataset scripts accept `CHEMBL_SQLITE_PATH=/path/to/chembl_36.db`; the SLURM scripts fall back to the Longleaf ChEMBL SQLite path when the variable is unset.

## Common Commands

```bash
# Dataset pipeline
bash scripts/dataset/run_dataset_pipeline.sh sequential
bash scripts/dataset/run_dataset_pipeline.sh slurm

# Variance calibration
python scripts/variance/run_variance.py --config src/mockdock/configs/PptT.toml --run-dir variance_runs/PptT --output-dir variance_analysis/PptT --n-iters 5
sbatch scripts/variance/slurm_pptt_variance.sbatch

# PptT model suite
sbatch --export=ALL,CLIP_REWARD_UPPER_BOUND=false scripts/experiments/slurm_pptt_model_suite.sh
sbatch --export=ALL,CLIP_REWARD_UPPER_BOUND=true scripts/experiments/slurm_pptt_model_suite.sh

# Analysis and scoring orchestration
bash scripts/orchestration/run_analysis_scoring_pipeline.sh
bash scripts/orchestration/submit_analysis_scoring_pipeline.sh
```
